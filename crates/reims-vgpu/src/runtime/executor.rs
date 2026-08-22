//! Device-owned backend submission port.
//!
//! Render and compute commands, surrounding identities, resource lists, and
//! segment boundaries are backend-independent before they cross this port.

use crate::model::TargetIdentity;
use reims_vgpu_protocol::StorageImageFormat;
pub use reims_vgpu_vulkan::engine::{
    gather_phase::GatherPhaseWindow, gpu_span::GpuSpanWindow, stage_phase::StagePhaseWindow,
};
pub use reims_vgpu_vulkan::engine::{
    CounterSnapshot, DrawError, DrawPhaseWindow, EngineFacadeDecline,
};
pub use reims_vgpu_vulkan::m2v_cache::M2vCacheDecline;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub use reims_vgpu_core::{
    CapabilityService, ComputeOutput, ComputeRequest, ComputeResidencyService, DrawOutput,
    DrawRequest, ExecutionPort, ExecutorCapabilities, GuestWriteReach, GuestWriteService,
    PresentDecline, PresentationService, ReadbackLease, ReadbackService, ResidentContent,
    ResidentContentBacking, ResidentReadPlan, ResidentService, ResolvedCommand,
    ResolvedCommandBuffer, ResourceLifetimeRef, SubmissionContext, TargetReadback,
};

struct DeviceTelemetry;

impl reims_vgpu_vulkan::telemetry::BackendTelemetry for DeviceTelemetry {
    fn route(&self, name: &'static str) {
        crate::runtime::drain::note_store_route(name);
    }

    fn route_n(&self, name: &'static str, count: u64) {
        crate::runtime::drain::note_store_route_n(name, count);
    }

    fn route_us(&self, name: &'static str, micros: u64) {
        crate::runtime::drain::note_store_route_us(name, micros);
    }

    fn readback_phase(&self, phase: reims_vgpu_vulkan::telemetry::ReadbackPhase, micros: u64) {
        use crate::runtime::drain::ReadbackPhase as DevicePhase;
        use reims_vgpu_vulkan::telemetry::ReadbackPhase as BackendPhase;
        let phase = match phase {
            BackendPhase::Submit => DevicePhase::Submit,
            BackendPhase::Fence => DevicePhase::Fence,
            BackendPhase::Map => DevicePhase::Map,
            BackendPhase::Write => DevicePhase::Write,
            BackendPhase::Vouch => DevicePhase::Vouch,
            BackendPhase::Resolve => DevicePhase::Resolve,
        };
        crate::runtime::drain::note_readback_phase(phase, micros);
    }

    fn readback_gpu_us(&self, barrier: u64, copy: u64) {
        crate::runtime::drain::note_readback_gpu_us(barrier, copy);
    }

    fn guest_imports_invalidated(&self) {
        crate::runtime::guest_ram_map::reset();
    }
}

static DEVICE_TELEMETRY: DeviceTelemetry = DeviceTelemetry;

fn install_telemetry() {
    reims_vgpu_vulkan::telemetry::install(&DEVICE_TELEMETRY);
}

/// Attribute backend lock observations to the device drain worker before it
/// acquires the device-state lock.
pub(crate) fn mark_drain_thread() {
    reims_vgpu_vulkan::engine::mark_drain_thread();
}

/// Dynamic executor-session scope for one device operation.
pub struct ExecutionScope {
    _engine: Option<reims_vgpu_vulkan::engine::SessionScope>,
    #[cfg(test)]
    _test: Option<Box<dyn std::any::Any>>,
}

impl ExecutionScope {
    fn none() -> Self {
        Self {
            _engine: None,
            #[cfg(test)]
            _test: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn test(guard: impl std::any::Any) -> Self {
        Self {
            _engine: None,
            _test: Some(Box::new(guard)),
        }
    }
}

/// Snapshot the active protocol context before entering a backend call.
pub fn context_for(state: &crate::runtime::Device, task_id: u32) -> SubmissionContext {
    state.submissions.context_or_standalone(task_id)
}

pub type ResolvedSubmission =
    reims_vgpu_core::ResolvedSubmission<Box<DrawRequest>, Box<ComputeRequest>>;
pub type ExecutionOutput = reims_vgpu_core::ExecutionOutput<DrawOutput, ComputeOutput>;
pub type ExecutionCompletion = reims_vgpu_core::ExecutionCompletion<Box<[ExecutionOutput]>>;
pub type ExecutionReceipt<T> = reims_vgpu_core::ExecutionReceipt<T>;
pub type StampAnnounce = std::sync::Arc<dyn Fn(u32) + Send + Sync>;

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug)]
pub struct WindowPresentationFrame<'a> {
    pub width: u32,
    pub height: u32,
    pub seq: u64,
    pub payload: WindowPresentationPayload<'a>,
}

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug)]
pub enum WindowPresentationPayload<'a> {
    CpuBgra(&'a [u8]),
    Resident(&'a reims_vgpu_core::PreparedPresentation),
}

#[cfg(feature = "host-window")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WindowPresentOutcome {
    Busy,
    Presented {
        route: reims_vgpu_core::PresentationRoute,
        width: u32,
        height: u32,
        swapchain_images: usize,
        suboptimal: bool,
    },
}

/// A presentation-port failure with backend diagnostics preserved at the
/// composition boundary.
#[cfg(feature = "host-window")]
#[derive(Debug)]
pub struct WindowPresentationError {
    diagnostic: ExecutorDiagnostic,
    presenter_detached: bool,
}

/// Exact diagnostic projected across an executor capability boundary without
/// exporting the implementation's error type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutorDiagnostic {
    reason: &'static str,
    fields: Vec<(&'static str, String)>,
    detail: String,
}

impl ExecutorDiagnostic {
    pub(crate) fn from_decline(
        error: &(impl reims_vgpu_observe::Decline + std::fmt::Display),
    ) -> Self {
        Self {
            reason: error.slug(),
            fields: error.fields(),
            detail: error.to_string(),
        }
    }

    #[cfg(feature = "host-window")]
    fn named(reason: &'static str, fields: Vec<(&'static str, String)>, detail: String) -> Self {
        Self {
            reason,
            fields,
            detail,
        }
    }
}

impl std::fmt::Display for ExecutorDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ExecutorDiagnostic {}

impl reims_vgpu_observe::Decline for ExecutorDiagnostic {
    fn slug(&self) -> &'static str {
        self.reason
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        self.fields.clone()
    }
}

#[cfg(feature = "host-window")]
impl WindowPresentationError {
    fn service_unavailable(service: &'static str) -> Self {
        Self {
            diagnostic: ExecutorDiagnostic::named(
                "window_presentation_service_unavailable",
                vec![("service", service.to_string())],
                format!("window presentation service unavailable: {service}"),
            ),
            presenter_detached: false,
        }
    }

    fn from_draw(error: DrawError) -> Self {
        let presenter_detached = matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::WindowPresenterNotAttached)
        );
        Self {
            diagnostic: ExecutorDiagnostic::from_decline(&error),
            presenter_detached,
        }
    }

    pub fn presenter_detached(&self) -> bool {
        self.presenter_detached
    }
}

#[cfg(feature = "host-window")]
impl std::fmt::Display for WindowPresentationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.diagnostic.fmt(f)
    }
}

#[cfg(feature = "host-window")]
impl std::error::Error for WindowPresentationError {}

#[cfg(feature = "host-window")]
impl reims_vgpu_observe::Decline for WindowPresentationError {
    fn slug(&self) -> &'static str {
        self.diagnostic.reason
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        self.diagnostic.fields.clone()
    }
}

/// Native-window lifecycle and presentation owned by one executor session.
pub trait WindowPresentationService: std::fmt::Debug + Send + Sync {
    #[cfg(feature = "host-window")]
    fn attach_window_presenter(
        &self,
        _display: raw_window_handle::RawDisplayHandle,
        _window: raw_window_handle::RawWindowHandle,
        _width: u32,
        _height: u32,
    ) -> Result<(), WindowPresentationError> {
        Err(WindowPresentationError::service_unavailable(
            "window_present_attach",
        ))
    }

    #[cfg(feature = "host-window")]
    fn resize_window_presenter(&self, _width: u32, _height: u32) {}

    #[cfg(feature = "host-window")]
    fn present_window_frame(
        &self,
        _frame: Option<WindowPresentationFrame<'_>>,
    ) -> Result<WindowPresentOutcome, WindowPresentationError> {
        Err(WindowPresentationError::service_unavailable(
            "window_present_frame",
        ))
    }

    #[cfg(feature = "host-window")]
    fn detach_window_presenter(&self) {}
}

#[cfg(feature = "host-window")]
pub const MAX_WINDOW_REATTACHES: u32 = reims_vgpu_vulkan::engine::MAX_DEVICE_RECREATES;
pub const IDLE_MAINTENANCE_START_MS: u64 = reims_vgpu_vulkan::engine::IDLE_MAINTENANCE_START_MS;

/// Executor service that lands a semantic resident in bounded guest pages.
pub trait GuestPageTransferService: std::fmt::Debug + Send + Sync {
    fn copy_target_to_guest_pages(
        &self,
        _identity: &TargetIdentity,
        _target: &reims_vgpu_memory::GuestPageTarget,
        _pages: &[u64],
    ) -> Result<(), DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "target_to_guest_pages",
            },
        ))
    }
}

/// Completion-word ordering against outstanding executor access to guest RAM.
pub trait CompletionService: std::fmt::Debug + Send + Sync {
    fn install_stamp_announce(&self, _hook: StampAnnounce) {}

    /// Whether accepted executor work preceding a completion word still needs
    /// a GPU ordering point.
    fn completion_work_outstanding(&self) -> bool {
        false
    }

    fn completion_stamp_pending(&self, _index: u32) -> bool {
        false
    }

    fn write_completion_stamp(
        &self,
        _guest_ref: &reims_vgpu_memory::GuestRef,
        _index: u32,
        _value: u32,
    ) -> Result<(), DrawError> {
        Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "completion_stamp",
            },
        ))
    }

    fn quiesce_completion_stamps(&self, _index: u32) {}

    fn quiesce_guest_reads(&self) {}
}

/// Ownership of the executor session's one deferred submission batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionWaitBatch {
    /// The awaited completion point was parked in the recording batch, which
    /// this transition submitted.
    Submitted,
    /// No recording batch owned the point; it was already on the GPU rail.
    AlreadyInFlight,
}

/// Submission-boundary transitions over the executor's open batch.
pub trait SubmissionBatchService: std::fmt::Debug + Send + Sync {
    fn submit_for_completion_wait(&self, _index: u32) -> CompletionWaitBatch {
        CompletionWaitBatch::AlreadyInFlight
    }

    fn flush_submission_tail(&self) {}
}

/// Backend materializations of guest allocation lifetimes.
pub trait GuestImportService: std::fmt::Debug + Send + Sync {
    fn warm_guest_ram_imports(
        &self,
        _imports: &[std::sync::Arc<reims_vgpu_memory::GuestRamImport>],
    ) -> (usize, u64) {
        (0, 0)
    }

    /// End backend access to an allocation. `true` means no asynchronous
    /// backend owner remains; otherwise completion is reported by
    /// [`Self::take_completed_guest_imports`].
    fn retire_guest_import(&self, _import: reims_vgpu_memory::ImportId) -> bool {
        true
    }

    fn take_completed_guest_imports(&self) -> Vec<reims_vgpu_memory::ImportId> {
        Vec::new()
    }
}

/// Backend image-layout planning performed before a resource-shaped page alias
/// is finalized.
pub trait GuestImagePlanningService: std::fmt::Debug + Send + Sync {
    fn sampled_image_binding_requirement(
        &self,
        _request: reims_vgpu_memory::GuestImageBindingRequest,
    ) -> Option<reims_vgpu_memory::GuestImageBindingDisposition> {
        None
    }
}

/// Backend housekeeping which does not itself execute a guest command.
pub trait MaintenanceService: std::fmt::Debug + Send + Sync {
    fn maintain_resources(&self, _now_ms: u64) {}

    fn idle_reclaim_start_ms(&self) -> Option<u64> {
        None
    }
}

/// Per-vGPU backend-session selection and teardown.
pub trait SessionService: std::fmt::Debug + Send + Sync {
    /// End one guest lifetime while preserving shareable physical-GPU state.
    fn reset(&self) {}

    /// Select this executor's device-local backend session for a product call.
    fn enter(&self) -> ExecutionScope {
        ExecutionScope::none()
    }
}

/// Observation-only backend snapshots. Semantic planning must never read this port.
pub trait ObservationService: std::fmt::Debug + Send + Sync {
    fn sampled_working_set_census(&self) -> Option<String> {
        None
    }

    fn buffer_gather_working_set_census(&self) -> Option<String> {
        None
    }

    fn guest_import_census(&self) -> (u64, usize, usize, usize, usize, usize) {
        (0, 0, 0, 0, 0, 0)
    }

    fn object_cache_levels(&self) -> [usize; 6] {
        [0; 6]
    }

    fn shader_translation_cache_level(&self) -> usize {
        0
    }

    fn counter_snapshot(&self) -> CounterSnapshot {
        Default::default()
    }

    fn draw_phase_window(&self) -> Option<DrawPhaseWindow> {
        None
    }

    fn gpu_span_window(&self) -> Option<GpuSpanWindow> {
        None
    }

    fn gather_phase_window(&self) -> Option<GatherPhaseWindow> {
        None
    }

    fn stage_phase_window(&self) -> Option<StagePhaseWindow> {
        None
    }

    fn take_engine_lock_census(&self, _win_ms: u64) -> Option<String> {
        None
    }

    fn note_draw_hang_candidate(&self, _note: DrawHangCandidate) {}

    fn draw_hang_trail(&self) -> Option<String> {
        None
    }

    fn draw_hang_outstanding(&self) -> Option<String> {
        None
    }

    fn recent_pipeline_firsts(&self) -> Option<String> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawHangIndexSource {
    CpuBytes,
    GuestRuns,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawHangIndexedNote {
    pub index_count: u32,
    pub index_width: u8,
    pub vertex_offset: i32,
    pub base_instance: u32,
    pub byte_len: u64,
    pub source: DrawHangIndexSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawHangSampledNote {
    pub binding: u32,
    pub kind: u8,
    pub format: reims_vgpu_protocol::ImageFormat,
    pub width: u32,
    pub height: u32,
    pub texel0: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawHangSamplerNote {
    pub binding: u32,
    pub min_filter: u8,
    pub mag_filter: u8,
    pub mip_filter: u8,
    pub address_u: u8,
    pub address_v: u8,
    pub provenance: u8,
    pub unnormalized: bool,
    pub lod_min: u32,
    pub lod_max: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DrawHangCandidate {
    pub pipeline_ref: u32,
    pub vert_words: u32,
    pub frag_words: u32,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub indexed: Option<DrawHangIndexedNote>,
    pub fragment_declared_bindings: Arc<[u32]>,
    pub fragment_provided_bindings: Vec<u32>,
    pub sampled: Vec<DrawHangSampledNote>,
    pub samplers: Vec<DrawHangSamplerNote>,
}

/// Shader translation and publication owned by the backend session.
///
/// Runtime supplies extracted AIR and semantic stage/local-size facts. Native
/// modules and translator reflection remain behind this port; successful
/// render preparation returns only the core executable projection.
pub trait ShaderTranslationService: std::fmt::Debug + Send + Sync {
    fn ensure_render_translation(
        &self,
        _air: &[u8],
        _stage: reims_vgpu_core::ShaderStage,
        _raster_sample_count: u32,
        _pipeline_ref: u32,
    ) -> bool {
        true
    }

    fn prepare_render_translation(
        &self,
        _air: &[u8],
        _stage: reims_vgpu_core::ShaderStage,
        _raster_sample_count: u32,
        _pipeline_ref: u32,
    ) -> Result<reims_vgpu_core::PreparedShaderFamily, M2vCacheDecline> {
        panic!("executor does not provide shader translation")
    }

    fn specialize_render_samplers(
        &self,
        variant: &reims_vgpu_core::PreparedShaderVariant,
        _samplers: &[reims_vgpu_core::SamplerResource],
    ) -> Result<reims_vgpu_core::PreparedShaderVariant, M2vCacheDecline> {
        Ok(variant.clone())
    }

    fn ensure_compute_translation(
        &self,
        _air: &[u8],
        _local_size: [u32; 3],
        _pipeline_ref: u32,
    ) -> bool {
        true
    }

    fn translate_compute(
        &self,
        _air: &[u8],
        _local_size: [u32; 3],
        _pipeline_ref: u32,
    ) -> Result<Arc<dyn ComputeTranslation>, M2vCacheDecline> {
        panic!("executor does not provide compute translation")
    }
}

/// Backend policy for bounding render buffer arguments.
///
/// The semantic shader interface and draw geometry cross this port. Reflection
/// implementation details remain owned by the executor adapter.
pub trait RenderBufferPlanningService: std::fmt::Debug + Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn render_buffer_extent(
        &self,
        _interface: &reims_vgpu_core::ShaderInterface,
        _metal_index: u32,
        _feeds_stage_in: bool,
        _first_vertex: u32,
        _vertex_count: u32,
        _base_instance: u32,
        _instance_count: u32,
        _indexed: bool,
    ) -> Option<u64> {
        None
    }
}

/// Backend-owned translated compute module exposed only through semantic facts.
pub trait ComputeTranslation: std::fmt::Debug + Send + Sync {
    fn interface(&self) -> &reims_vgpu_core::ShaderInterface;

    fn buffer_extent(
        &self,
        metal_index: u32,
        workgroups: [u32; 3],
        local_size: [u32; 3],
    ) -> Option<u64>;

    fn storage_image_access(&self, binding: u32) -> Option<reims_vgpu_core::StorageImageAccess>;

    fn null_sampled_image_bindings(&self, bound: &[u32]) -> Vec<u32>;

    fn samplers(&self) -> Arc<[reims_vgpu_core::ReflectedSamplerDescriptor]>;

    fn prepare_program(
        &self,
        requests: &[(u32, StorageImageFormat)],
    ) -> Result<PreparedComputeProgram, ComputeProgramDecline>;
}

#[derive(Debug)]
pub struct PreparedComputeProgram {
    pub stage: reims_vgpu_core::PreparedShaderStage,
    pub storage_image_formats: Vec<(u32, Option<StorageImageFormat>)>,
    _native_lifetime: Box<dyn std::any::Any + Send + Sync>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeProgramDecline {
    Specialization(M2vCacheDecline),
}

impl crate::observe::Decline for ComputeProgramDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Specialization(decline) => crate::observe::Decline::slug(decline),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Specialization(decline) => crate::observe::Decline::fields(decline),
        }
    }
}

crate::observe::decline_display!(ComputeProgramDecline);

impl std::error::Error for ComputeProgramDecline {}

/// Backend execution contract implemented per device.
pub trait Executor:
    ExecutionPort<
        Submission = ResolvedSubmission,
        Completion = ExecutionCompletion,
        Error = DrawError,
    > + ResidentService
    + GuestWriteService
    + ComputeResidencyService
    + CapabilityService
    + PresentationService
    + ReadbackService<Error = DrawError>
    + GuestPageTransferService
    + CompletionService
    + SubmissionBatchService
    + GuestImportService
    + GuestImagePlanningService
    + MaintenanceService
    + SessionService
    + ObservationService
    + ShaderTranslationService
    + RenderBufferPlanningService
    + WindowPresentationService
{
}

/// Compatibility adapter over the current Vulkan engine facade.
#[derive(Debug)]
pub struct VulkanExecutor {
    session: reims_vgpu_vulkan::engine::SessionHandle,
    resident_leases: Mutex<ResidentLeaseStore<reims_vgpu_vulkan::engine::ResidentResourceLease>>,
}

trait ExecutorResidentLease: std::fmt::Debug + Send {
    fn matches(&self, identity: &TargetIdentity) -> bool;
    fn backing(&self) -> ResidentContentBacking;
}

impl ExecutorResidentLease for reims_vgpu_vulkan::engine::ResidentResourceLease {
    fn matches(&self, identity: &TargetIdentity) -> bool {
        self.matches(identity)
    }

    fn backing(&self) -> ResidentContentBacking {
        self.backing()
    }
}

#[derive(Debug)]
struct HeldResident<L> {
    owner: ResourceLifetimeRef,
    lease: L,
}

#[derive(Debug)]
struct ResidentLeaseStore<L> {
    entries: HashMap<(u64, TargetIdentity), HeldResident<L>>,
}

impl<L> Default for ResidentLeaseStore<L> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<L: ExecutorResidentLease> ResidentLeaseStore<L> {
    fn retain_with(
        &mut self,
        owner: ResourceLifetimeRef,
        identity: &TargetIdentity,
        acquire: impl FnOnce(&TargetIdentity) -> Option<L>,
    ) -> (ResidentContentBacking, bool) {
        self.reap_dead();
        let key = (owner.id(), identity.clone());
        if let Some(held) = self
            .entries
            .get(&key)
            .filter(|held| held.lease.matches(identity))
        {
            return (held.lease.backing(), false);
        }
        self.entries.remove(&key);
        let Some(lease) = acquire(identity) else {
            return (ResidentContentBacking::NotReady, false);
        };
        let backing = lease.backing();
        self.entries.insert(key, HeldResident { owner, lease });
        (backing, true)
    }

    fn reap_dead(&mut self) {
        self.entries.retain(|_, held| held.owner.is_live());
    }

    fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for VulkanExecutor {
    fn default() -> Self {
        install_telemetry();
        Self {
            session: reims_vgpu_vulkan::engine::SessionHandle::allocate(),
            resident_leases: Mutex::new(ResidentLeaseStore::default()),
        }
    }
}

impl Drop for VulkanExecutor {
    fn drop(&mut self) {
        self.resident_leases
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        reims_vgpu_vulkan::engine::release_session(&self.session);
    }
}

impl GuestPageTransferService for VulkanExecutor {
    fn copy_target_to_guest_pages(
        &self,
        identity: &TargetIdentity,
        target: &reims_vgpu_memory::GuestPageTarget,
        pages: &[u64],
    ) -> Result<(), DrawError> {
        reims_vgpu_vulkan::engine::copy_target_to_guest_pages(identity, target, pages)
    }
}

impl CompletionService for VulkanExecutor {
    fn install_stamp_announce(&self, hook: StampAnnounce) {
        let _scope = reims_vgpu_vulkan::engine::enter_session(&self.session);
        reims_vgpu_vulkan::engine::install_stamp_announce(hook);
    }

    fn completion_work_outstanding(&self) -> bool {
        reims_vgpu_vulkan::engine::completion_work_outstanding()
    }

    fn completion_stamp_pending(&self, index: u32) -> bool {
        reims_vgpu_vulkan::engine::completion_stamp_pending(index)
    }

    fn write_completion_stamp(
        &self,
        guest_ref: &reims_vgpu_memory::GuestRef,
        index: u32,
        value: u32,
    ) -> Result<(), DrawError> {
        reims_vgpu_vulkan::engine::write_completion_stamp(guest_ref, index, value)
    }

    fn quiesce_completion_stamps(&self, index: u32) {
        reims_vgpu_vulkan::engine::quiesce_completion_stamps(index);
    }

    fn quiesce_guest_reads(&self) {
        reims_vgpu_vulkan::engine::quiesce_guest_reads();
    }
}

impl SubmissionBatchService for VulkanExecutor {
    fn submit_for_completion_wait(&self, index: u32) -> CompletionWaitBatch {
        if reims_vgpu_vulkan::engine::submit_batch_for_waiting_stamp(index) {
            CompletionWaitBatch::Submitted
        } else {
            CompletionWaitBatch::AlreadyInFlight
        }
    }

    fn flush_submission_tail(&self) {
        reims_vgpu_vulkan::engine::flush_batched_draws();
    }
}

impl GuestImportService for VulkanExecutor {
    fn warm_guest_ram_imports(
        &self,
        imports: &[std::sync::Arc<reims_vgpu_memory::GuestRamImport>],
    ) -> (usize, u64) {
        reims_vgpu_vulkan::engine::warm_guest_ram_imports(imports)
    }

    fn retire_guest_import(&self, import: reims_vgpu_memory::ImportId) -> bool {
        reims_vgpu_vulkan::engine::retire_guest_import(import)
    }

    fn take_completed_guest_imports(&self) -> Vec<reims_vgpu_memory::ImportId> {
        reims_vgpu_vulkan::engine::take_completed_guest_imports()
    }
}

impl ObservationService for VulkanExecutor {
    fn sampled_working_set_census(&self) -> Option<String> {
        reims_vgpu_vulkan::engine::sampled_working_set_census()
    }

    fn buffer_gather_working_set_census(&self) -> Option<String> {
        reims_vgpu_vulkan::engine::buffer_gather_working_set_census()
    }

    fn guest_import_census(&self) -> (u64, usize, usize, usize, usize, usize) {
        reims_vgpu_vulkan::engine::guest_import_census()
    }

    fn object_cache_levels(&self) -> [usize; 6] {
        reims_vgpu_vulkan::engine::object_cache_levels()
    }

    fn shader_translation_cache_level(&self) -> usize {
        reims_vgpu_vulkan::m2v_cache::stats().2
    }

    fn counter_snapshot(&self) -> CounterSnapshot {
        reims_vgpu_vulkan::engine::counter_snapshot()
    }

    fn draw_phase_window(&self) -> Option<DrawPhaseWindow> {
        reims_vgpu_vulkan::engine::draw_phase_window()
    }

    fn gpu_span_window(&self) -> Option<GpuSpanWindow> {
        reims_vgpu_vulkan::engine::gpu_span::take_window()
    }

    fn gather_phase_window(&self) -> Option<GatherPhaseWindow> {
        reims_vgpu_vulkan::engine::gather_phase::take_window()
    }

    fn stage_phase_window(&self) -> Option<StagePhaseWindow> {
        reims_vgpu_vulkan::engine::stage_phase::take_window()
    }

    fn take_engine_lock_census(&self, win_ms: u64) -> Option<String> {
        reims_vgpu_vulkan::engine::take_engine_lock_census(win_ms)
    }

    fn note_draw_hang_candidate(&self, note: DrawHangCandidate) {
        use reims_vgpu_vulkan::gpu_hang_trail as trail;

        let gap = trail::gap(
            &note.fragment_declared_bindings,
            &note.fragment_provided_bindings,
        );
        let mut sampled = [trail::SampledNote::default(); trail::SAMPLED_KEPT];
        for (slot, source) in sampled.iter_mut().zip(note.sampled.iter().copied()) {
            *slot = trail::SampledNote {
                binding: source.binding,
                kind: source.kind,
                format: reims_vgpu_vulkan::format::vk_image_format(source.format).as_raw() as u32,
                width: source.width,
                height: source.height,
                texel0: source.texel0,
            };
        }
        let mut samplers = [trail::SamplerNote::default(); trail::SAMPLER_KEPT];
        for (slot, source) in samplers.iter_mut().zip(note.samplers.iter().copied()) {
            *slot = trail::SamplerNote {
                binding: source.binding,
                min_filter: source.min_filter,
                mag_filter: source.mag_filter,
                mip_filter: source.mip_filter,
                address_u: source.address_u,
                address_v: source.address_v,
                provenance: source.provenance,
                unnormalized: source.unnormalized,
                lod_min: source.lod_min,
                lod_max: source.lod_max,
            };
        }
        trail::note_draw(trail::DrawNote {
            pipeline_ref: note.pipeline_ref,
            vert_words: note.vert_words,
            frag_words: note.frag_words,
            width: note.width,
            height: note.height,
            vertex_count: note.vertex_count,
            instance_count: note.instance_count,
            indexed: note.indexed.map(|indexed| trail::IndexedNote {
                index_count: indexed.index_count,
                index_width: indexed.index_width,
                vertex_offset: indexed.vertex_offset,
                base_instance: indexed.base_instance,
                byte_len: indexed.byte_len,
                source: match indexed.source {
                    DrawHangIndexSource::CpuBytes => trail::IndexSource::CpuBytes,
                    DrawHangIndexSource::GuestRuns => trail::IndexSource::GuestRuns,
                },
            }),
            frag_declared: note.fragment_declared_bindings.len() as u32,
            frag_provided: note.fragment_provided_bindings.len() as u32,
            frag_gap: gap.0,
            frag_gap_lo: gap.1,
            sampled,
            sampled_count: note.sampled.len() as u32,
            samplers,
            sampler_count: note.samplers.len() as u32,
        });
    }

    fn draw_hang_trail(&self) -> Option<String> {
        reims_vgpu_vulkan::gpu_hang_trail::trail()
    }

    fn draw_hang_outstanding(&self) -> Option<String> {
        reims_vgpu_vulkan::gpu_hang_trail::outstanding()
    }

    fn recent_pipeline_firsts(&self) -> Option<String> {
        reims_vgpu_vulkan::gpu_hang_trail::recent_pipeline_firsts()
    }
}

impl Executor for VulkanExecutor {}

impl GuestImagePlanningService for VulkanExecutor {
    fn sampled_image_binding_requirement(
        &self,
        request: reims_vgpu_memory::GuestImageBindingRequest,
    ) -> Option<reims_vgpu_memory::GuestImageBindingDisposition> {
        let _scope = self.enter();
        reims_vgpu_vulkan::engine::sampled_guest_image_binding_requirement(request)
    }
}

impl RenderBufferPlanningService for VulkanExecutor {
    fn render_buffer_extent(
        &self,
        interface: &reims_vgpu_core::ShaderInterface,
        metal_index: u32,
        feeds_stage_in: bool,
        first_vertex: u32,
        vertex_count: u32,
        base_instance: u32,
        instance_count: u32,
        indexed: bool,
    ) -> Option<u64> {
        let bounds = reims_vgpu_vulkan::spirv_bind::RenderBufferIndexBounds::new(
            first_vertex,
            vertex_count,
            base_instance,
            instance_count,
            indexed,
        );
        reims_vgpu_vulkan::spirv_bind::vertex_buffer_extent_interface(
            interface,
            metal_index,
            feeds_stage_in,
            bounds,
        )
    }
}

struct VulkanComputeTranslation {
    shader: Arc<reims_vgpu_vulkan::m2v_cache::CachedShader>,
}

impl std::fmt::Debug for VulkanComputeTranslation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanComputeTranslation")
            .field("interface", &self.shader.interface)
            .finish_non_exhaustive()
    }
}

impl ComputeTranslation for VulkanComputeTranslation {
    fn interface(&self) -> &reims_vgpu_core::ShaderInterface {
        &self.shader.interface
    }

    fn buffer_extent(
        &self,
        metal_index: u32,
        workgroups: [u32; 3],
        local_size: [u32; 3],
    ) -> Option<u64> {
        reims_vgpu_vulkan::spirv_bind::reflected_compute_buffer_extent_interface(
            &self.shader.interface,
            metal_index,
            workgroups,
            local_size,
        )
    }

    fn storage_image_access(&self, binding: u32) -> Option<reims_vgpu_core::StorageImageAccess> {
        self.shader.storage_image_access(binding)
    }

    fn null_sampled_image_bindings(&self, bound: &[u32]) -> Vec<u32> {
        self.shader.null_sampled_image_bindings(bound)
    }

    fn samplers(&self) -> Arc<[reims_vgpu_core::ReflectedSamplerDescriptor]> {
        self.shader.kernel_samplers()
    }

    fn prepare_program(
        &self,
        requests: &[(u32, StorageImageFormat)],
    ) -> Result<PreparedComputeProgram, ComputeProgramDecline> {
        let native = requests
            .iter()
            .map(|(binding, format)| {
                let metal_index = self
                    .shader
                    .interface
                    .bindings
                    .iter()
                    .find(|resource| {
                        resource.descriptor.map(|descriptor| descriptor.binding) == Some(*binding)
                            && matches!(
                                resource.kind,
                                reims_vgpu_core::ShaderResourceKind::StorageImage
                                    | reims_vgpu_core::ShaderResourceKind::TextureArray
                                    | reims_vgpu_core::ShaderResourceKind::EmbeddedArgBufferTexture
                            )
                    })
                    .map(|resource| resource.metal_index)
                    .ok_or_else(|| {
                        ComputeProgramDecline::Specialization(
                            M2vCacheDecline::RuntimeStorageImageSpecialize {
                                detail: format!(
                                    "no reflected storage image at descriptor binding {binding}"
                                ),
                            },
                        )
                    })?;
                let caps = reims_vgpu_vulkan::engine::runtime_storage_image_capabilities(*format);
                Ok(reims_vgpu_vulkan::m2v_cache::RuntimeStorageImageRequest {
                    binding: *binding,
                    metal_index,
                    format: *format,
                    storage_image: caps.storage_image,
                    storage_image_atomic: caps.storage_image_atomic,
                    read_without_format: caps.read_without_format,
                    write_without_format: caps.write_without_format,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prepared = self
            .shader
            .prepare_kernel(&native)
            .map_err(ComputeProgramDecline::Specialization)?;
        Ok(PreparedComputeProgram {
            stage: reims_vgpu_vulkan::m2v_cache::prepared_stage(&prepared.variant),
            storage_image_formats: prepared.storage_formats.clone(),
            _native_lifetime: Box::new(prepared),
        })
    }
}

impl ShaderTranslationService for VulkanExecutor {
    fn ensure_render_translation(
        &self,
        air: &[u8],
        stage: reims_vgpu_core::ShaderStage,
        raster_sample_count: u32,
        pipeline_ref: u32,
    ) -> bool {
        let stage = match stage {
            reims_vgpu_core::ShaderStage::Vertex => {
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Vertex
            }
            reims_vgpu_core::ShaderStage::Fragment => {
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Fragment
            }
            reims_vgpu_core::ShaderStage::Unknown => return true,
        };
        reims_vgpu_vulkan::m2v_cache::ensure_render_cached_async(
            air,
            stage,
            raster_sample_count,
            pipeline_ref,
        )
    }

    fn prepare_render_translation(
        &self,
        air: &[u8],
        stage: reims_vgpu_core::ShaderStage,
        raster_sample_count: u32,
        pipeline_ref: u32,
    ) -> Result<reims_vgpu_core::PreparedShaderFamily, M2vCacheDecline> {
        let stage = match stage {
            reims_vgpu_core::ShaderStage::Vertex => {
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Vertex
            }
            reims_vgpu_core::ShaderStage::Fragment => {
                reims_vgpu_vulkan::m2v_cache::RenderTranslationStage::Fragment
            }
            reims_vgpu_core::ShaderStage::Unknown => {
                unreachable!("render translation requires a concrete stage")
            }
        };
        let shader = reims_vgpu_vulkan::m2v_cache::translate_render_cached_reflected(
            air,
            stage,
            raster_sample_count,
            pipeline_ref,
        )?;
        Ok(reims_vgpu_vulkan::m2v_cache::prepare_render_shader(
            &shader, stage,
        ))
    }

    fn specialize_render_samplers(
        &self,
        variant: &reims_vgpu_core::PreparedShaderVariant,
        samplers: &[reims_vgpu_core::SamplerResource],
    ) -> Result<reims_vgpu_core::PreparedShaderVariant, M2vCacheDecline> {
        reims_vgpu_vulkan::m2v_cache::specialize_render_samplers(variant, samplers)
    }

    fn ensure_compute_translation(
        &self,
        air: &[u8],
        local_size: [u32; 3],
        pipeline_ref: u32,
    ) -> bool {
        reims_vgpu_vulkan::m2v_cache::ensure_cached_kernel_async(air, local_size, pipeline_ref)
    }

    fn translate_compute(
        &self,
        air: &[u8],
        local_size: [u32; 3],
        pipeline_ref: u32,
    ) -> Result<Arc<dyn ComputeTranslation>, M2vCacheDecline> {
        let shader = reims_vgpu_vulkan::m2v_cache::translate_cached_kernel_reflected(
            air,
            local_size,
            pipeline_ref,
        )?;
        Ok(Arc::new(VulkanComputeTranslation { shader }))
    }
}

impl MaintenanceService for VulkanExecutor {
    fn maintain_resources(&self, now_ms: u64) {
        self.resident_leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .reap_dead();
        reims_vgpu_vulkan::engine::maintain_resources(now_ms);
    }

    fn idle_reclaim_start_ms(&self) -> Option<u64> {
        Some(reims_vgpu_vulkan::engine::IDLE_MAINTENANCE_START_MS)
    }
}

impl SessionService for VulkanExecutor {
    fn reset(&self) {
        let _scope = self.enter();
        self.resident_leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        reims_vgpu_vulkan::engine::reset_guest_state();
    }

    fn enter(&self) -> ExecutionScope {
        ExecutionScope {
            _engine: Some(reims_vgpu_vulkan::engine::enter_session(&self.session)),
            #[cfg(test)]
            _test: None,
        }
    }
}

impl CapabilityService for VulkanExecutor {
    fn capabilities(&self) -> ExecutorCapabilities {
        let (max_compute_workgroup_invocations, thread_execution_width) =
            reims_vgpu_vulkan::engine::compute_threadgroup_limits();
        ExecutorCapabilities {
            device_info: reims_vgpu_vulkan::engine::device_info_limits(),
            max_compute_workgroup_invocations,
            thread_execution_width,
            max_render_target_dimension: reims_vgpu_vulkan::engine::max_render_target_dimension(),
            deferred_gpu_only_content: reims_vgpu_vulkan::engine::deferred_gpu_only_content_allowed(
            ),
            storage_image_write_without_format:
                reims_vgpu_vulkan::engine::supports_storage_image_write_without_format(),
        }
    }

    fn render_target_layout_supported(
        &self,
        layout: reims_vgpu_core::pixel_format::TexelLayout,
    ) -> bool {
        reims_vgpu_vulkan::engine::render_target_layout_supported(layout)
    }

    fn sampled_layout_linear_filter_supported(
        &self,
        layout: reims_vgpu_core::pixel_format::TexelLayout,
    ) -> bool {
        reims_vgpu_vulkan::engine::supports_sampled_layout_linear_filter(layout)
    }
}

impl ReadbackService for VulkanExecutor {
    type Error = DrawError;

    fn read_target(&self, identity: &TargetIdentity) -> Result<TargetReadback, Self::Error> {
        reims_vgpu_vulkan::engine::read_target(identity)
    }

    fn read_target_leased(
        &self,
        identity: &TargetIdentity,
    ) -> Result<Option<Box<dyn ReadbackLease>>, Self::Error> {
        reims_vgpu_vulkan::engine::read_target_leased(identity)
            .map(|lease| lease.map(|lease| Box::new(lease) as Box<dyn ReadbackLease>))
    }

    fn read_resident_bgra(&self, identity: &TargetIdentity, need: usize) -> Option<Vec<u8>> {
        reims_vgpu_vulkan::engine::read_resident_bgra(identity, need)
    }
}

impl PresentationService for VulkanExecutor {
    fn resident_presentable(&self, identity: &TargetIdentity, width: u32, height: u32) -> bool {
        reims_vgpu_vulkan::engine::resident_presentable(identity, width, height)
    }

    fn prepare_window_resident_present(
        &self,
        source: &reims_vgpu_core::PresentationSource,
    ) -> Result<reims_vgpu_core::PreparedPresentation, PresentDecline> {
        #[cfg(feature = "host-window")]
        return reims_vgpu_vulkan::engine::prepare_window_resident_present(
            source.identity(),
            source.width(),
            source.height(),
        )
        .map(|()| reims_vgpu_core::PreparedPresentation::accepted(source.clone()));
        #[cfg(not(feature = "host-window"))]
        {
            let _ = source;
            Err(PresentDecline::WindowNotAttached)
        }
    }
}

impl WindowPresentationService for VulkanExecutor {
    #[cfg(feature = "host-window")]
    fn attach_window_presenter(
        &self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<(), WindowPresentationError> {
        reims_vgpu_vulkan::engine::window_present_attach(display, window, width, height)
            .map_err(WindowPresentationError::from_draw)
    }

    #[cfg(feature = "host-window")]
    fn resize_window_presenter(&self, width: u32, height: u32) {
        reims_vgpu_vulkan::engine::window_present_resize(width, height);
    }

    #[cfg(feature = "host-window")]
    fn present_window_frame(
        &self,
        frame: Option<WindowPresentationFrame<'_>>,
    ) -> Result<WindowPresentOutcome, WindowPresentationError> {
        let source = frame.and_then(|frame| match frame.payload {
            WindowPresentationPayload::Resident(source) => Some(source),
            WindowPresentationPayload::CpuBgra(_) => None,
        });
        let cpu = frame.and_then(|frame| match frame.payload {
            WindowPresentationPayload::CpuBgra(bgra) => {
                Some(reims_vgpu_vulkan::engine::WindowCpuFrame {
                    bgra,
                    width: frame.width,
                    height: frame.height,
                    seq: frame.seq,
                })
            }
            WindowPresentationPayload::Resident(_) => None,
        });
        reims_vgpu_vulkan::engine::window_present_frame(frame.map(|frame| frame.seq), source, cpu)
            .map(|outcome| match outcome {
                reims_vgpu_vulkan::engine::WindowPresentOutcome::Busy => WindowPresentOutcome::Busy,
                reims_vgpu_vulkan::engine::WindowPresentOutcome::Presented {
                    route,
                    width,
                    height,
                    swapchain_images,
                    suboptimal,
                } => WindowPresentOutcome::Presented {
                    route,
                    width,
                    height,
                    swapchain_images,
                    suboptimal,
                },
            })
            .map_err(WindowPresentationError::from_draw)
    }

    #[cfg(feature = "host-window")]
    fn detach_window_presenter(&self) {
        reims_vgpu_vulkan::engine::window_present_detach();
    }
}

impl ResidentService for VulkanExecutor {
    fn resident_read_plan(&self, identity: &TargetIdentity) -> ResidentReadPlan {
        reims_vgpu_vulkan::engine::resident_read_plan(identity)
    }

    fn resident_content_state(&self, identity: &TargetIdentity) -> ResidentContent {
        reims_vgpu_vulkan::engine::resident_content_state(identity)
    }

    fn stamp_resident_content_epoch(&self, identity: &TargetIdentity, epoch: u32) -> bool {
        reims_vgpu_vulkan::engine::stamp_resident_content_epoch(identity, epoch)
    }

    fn note_resident_guest_write(&self, identity: &TargetIdentity, epoch: u32) -> bool {
        reims_vgpu_vulkan::engine::note_resident_guest_write(identity, epoch)
    }

    fn note_resident_content_copied_out(&self, identity: &TargetIdentity) -> bool {
        reims_vgpu_vulkan::engine::note_resident_content_copied_out(identity)
    }

    fn retain_resident_resource(
        &self,
        owner: ResourceLifetimeRef,
        identity: &TargetIdentity,
    ) -> ResidentContentBacking {
        let (backing, acquired) = self
            .resident_leases
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain_with(owner, identity, |identity| {
                reims_vgpu_vulkan::engine::retain_resident_resource(identity)
            });
        crate::runtime::drain::note_store_route(if acquired {
            "resident_resource_acquired"
        } else if backing == ResidentContentBacking::NotReady {
            "resident_resource_unavailable"
        } else {
            return backing;
        });
        backing
    }
}

impl GuestWriteService for VulkanExecutor {
    fn guest_writes_outstanding(&self) -> bool {
        reims_vgpu_vulkan::engine::guest_writes_outstanding()
    }

    fn guest_writes_reaching(&self, pages: &[u64]) -> GuestWriteReach {
        reims_vgpu_vulkan::engine::guest_writes_reaching(pages)
    }

    fn quiesce_guest_writes(&self) {
        reims_vgpu_vulkan::engine::quiesce_guest_writes();
    }
}

impl ComputeResidencyService for VulkanExecutor {
    fn compute_resident_storage_generation(
        &self,
        identity: &reims_vgpu_core::ComputeStorageResidencyKey,
    ) -> Option<u32> {
        reims_vgpu_vulkan::engine::compute_resident_storage_generation(identity)
    }

    fn compute_resident_sample_source(
        &self,
        identity: &reims_vgpu_core::ComputeStorageResidencyKey,
    ) -> Option<(u32, StorageImageFormat)> {
        reims_vgpu_vulkan::engine::compute_resident_sample_source(identity)
    }

    fn unpin_resident_storage(&self, identity: &reims_vgpu_core::ComputeStorageResidencyKey) {
        reims_vgpu_vulkan::engine::unpin_resident_storage(identity);
    }

    fn retire_resident_storage_content(
        &self,
        identity: &reims_vgpu_core::ComputeStorageResidencyKey,
    ) {
        reims_vgpu_vulkan::engine::retire_resident_storage_content(identity);
    }

    fn note_resident_storage_copied_out(
        &self,
        identity: &reims_vgpu_core::ComputeStorageResidencyKey,
    ) {
        reims_vgpu_vulkan::engine::note_resident_storage_copied_out(identity);
    }
}

impl ExecutionPort for VulkanExecutor {
    type Submission = ResolvedSubmission;
    type Completion = ExecutionCompletion;
    type Error = DrawError;

    fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
        let _scope = self.enter();
        reims_vgpu_core::execute_resolved_submission(
            submission,
            |context, request| {
                let materialized = request
                    .sampled_images
                    .iter()
                    .filter(|image| {
                        matches!(
                            &image.source,
                            reims_vgpu_core::SampledSource::Bytes(_)
                                | reims_vgpu_core::SampledSource::GuestRuns(..)
                        )
                    })
                    .filter_map(|image| image.content)
                    .collect::<Vec<_>>();
                let output = reims_vgpu_vulkan::engine::execute_draw_request_in_submission(
                    context, &request,
                )?;
                Ok(reims_vgpu_core::CommandExecution::new(output, materialized))
            },
            |_, request| {
                let materialized = request
                    .sampled_images
                    .iter()
                    .filter(|image| {
                        matches!(
                            &image.source,
                            reims_vgpu_core::ComputeSampledImageSource::Bytes(_)
                                | reims_vgpu_core::ComputeSampledImageSource::GuestPages(_)
                        )
                    })
                    .filter_map(|image| image.content)
                    .collect::<Vec<_>>();
                let output = reims_vgpu_vulkan::engine::execute_compute_request(&request)?;
                Ok(reims_vgpu_core::CommandExecution::new(output, materialized))
            },
            |_, _| {
                Err(DrawError::Facade(
                    EngineFacadeDecline::ExecutorServiceUnavailable {
                        service: "host_memory_blit",
                    },
                ))
            },
            |_, _| {
                Err(DrawError::Facade(
                    EngineFacadeDecline::ExecutorServiceUnavailable {
                        service: "core_resource_state",
                    },
                ))
            },
        )
    }
}

/// Execute a draw and enforce that the executor returns the matching completion.
pub fn execute_draw(
    executor: &dyn Executor,
    context: SubmissionContext,
    request: DrawRequest,
) -> Result<ExecutionReceipt<DrawOutput>, DrawError> {
    let expected_identity = context.identity;
    let expected_kind = reims_vgpu_core::ExecutionKind::Draw.as_str();
    let expected = ResolvedSubmission::single(context, ResolvedCommand::Draw(Box::new(request)));
    let completion = executor.execute(expected)?;
    if completion.submission != expected_identity {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected: expected_identity,
                actual: completion.submission,
            },
        ));
    }
    let mut outputs = completion.output.into_vec();
    if outputs.len() != 1 {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionCountMismatch {
                expected: 1,
                actual: outputs.len(),
            },
        ));
    }
    match outputs.pop().expect("one checked completion") {
        ExecutionOutput::Draw(output) => Ok(ExecutionReceipt {
            submission: completion.submission,
            output,
            gpu_materialized: completion.gpu_materialized,
        }),
        other => Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: expected_kind,
                actual: other.kind().as_str(),
            },
        )),
    }
}

/// Execute a compute dispatch and enforce the matching completion kind.
pub fn execute_compute(
    executor: &dyn Executor,
    context: SubmissionContext,
    request: ComputeRequest,
) -> Result<ExecutionReceipt<ComputeOutput>, DrawError> {
    let expected_identity = context.identity;
    let expected_kind = reims_vgpu_core::ExecutionKind::Compute.as_str();
    let expected = ResolvedSubmission::single(context, ResolvedCommand::Compute(Box::new(request)));
    let completion = executor.execute(expected)?;
    if completion.submission != expected_identity {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected: expected_identity,
                actual: completion.submission,
            },
        ));
    }
    let mut outputs = completion.output.into_vec();
    if outputs.len() != 1 {
        return Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionCountMismatch {
                expected: 1,
                actual: outputs.len(),
            },
        ));
    }
    match outputs.pop().expect("one checked completion") {
        ExecutionOutput::Compute(output) => Ok(ExecutionReceipt {
            submission: completion.submission,
            output,
            gpu_materialized: completion.gpu_materialized,
        }),
        other => Err(DrawError::Facade(
            EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: expected_kind,
                actual: other.kind().as_str(),
            },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DeviceId;
    use crate::runtime::Device;
    use reims_vgpu_protocol::{
        ObjectTableRef, ResourceValidity, SegmentBoundary, SegmentKind, SubmissionId,
        SubmissionIdentity, SubmissionResourceUse, TaskId,
    };
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    };

    #[derive(Debug)]
    struct TestResidentLease {
        identity: TargetIdentity,
        live: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }

    impl ExecutorResidentLease for TestResidentLease {
        fn matches(&self, identity: &TargetIdentity) -> bool {
            self.identity == *identity && self.live.load(Ordering::Acquire)
        }

        fn backing(&self) -> ResidentContentBacking {
            ResidentContentBacking::DeviceAllocation
        }
    }

    impl Drop for TestResidentLease {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[cfg(feature = "host-window")]
    #[test]
    fn presentation_errors_preserve_diagnostics_and_expose_only_recovery_semantics() {
        let detached = WindowPresentationError::from_draw(DrawError::Facade(
            EngineFacadeDecline::WindowPresenterNotAttached,
        ));
        assert!(detached.presenter_detached());
        assert_eq!(
            reims_vgpu_observe::Decline::slug(&detached),
            "vk_engine_window_presenter_not_attached"
        );

        let unavailable = WindowPresentationError::from_draw(DrawError::Facade(
            EngineFacadeDecline::ExecutorServiceUnavailable {
                service: "test_service",
            },
        ));
        assert!(!unavailable.presenter_detached());
        assert_eq!(
            reims_vgpu_observe::Decline::fields(&unavailable),
            vec![("service", "test_service".to_string())]
        );
    }

    fn test_target(generation: u64) -> TargetIdentity {
        TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 32,
            generation,
            format: reims_vgpu_core::pixel_format::TexelLayout::Bgra8,
        }
    }

    #[test]
    fn executor_retains_children_until_the_semantic_owner_ends() {
        let owner = reims_vgpu_core::ResourceLifetime::new();
        let first = test_target(1);
        let second = test_target(2);
        let live = Arc::new(AtomicBool::new(true));
        let drops = Arc::new(AtomicUsize::new(0));
        let acquisitions = AtomicUsize::new(0);
        let mut store = ResidentLeaseStore::default();

        for identity in [&first, &first, &second] {
            let (backing, _) = store.retain_with(owner.reference(), identity, |identity| {
                acquisitions.fetch_add(1, Ordering::Relaxed);
                Some(TestResidentLease {
                    identity: identity.clone(),
                    live: Arc::clone(&live),
                    drops: Arc::clone(&drops),
                })
            });
            assert_eq!(backing, ResidentContentBacking::DeviceAllocation);
        }
        assert_eq!(acquisitions.load(Ordering::Relaxed), 2);
        assert_eq!(drops.load(Ordering::Relaxed), 0);

        drop(owner);
        store.reap_dead();
        assert_eq!(drops.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn stale_executor_lease_is_reacquired_under_the_same_identity() {
        let owner = reims_vgpu_core::ResourceLifetime::new();
        let identity = test_target(1);
        let first_live = Arc::new(AtomicBool::new(true));
        let drops = Arc::new(AtomicUsize::new(0));
        let mut store = ResidentLeaseStore::default();

        store.retain_with(owner.reference(), &identity, |identity| {
            Some(TestResidentLease {
                identity: identity.clone(),
                live: Arc::clone(&first_live),
                drops: Arc::clone(&drops),
            })
        });
        first_live.store(false, Ordering::Release);
        let (backing, acquired) = store.retain_with(owner.reference(), &identity, |identity| {
            Some(TestResidentLease {
                identity: identity.clone(),
                live: Arc::new(AtomicBool::new(true)),
                drops: Arc::clone(&drops),
            })
        });

        assert_eq!(backing, ResidentContentBacking::DeviceAllocation);
        assert!(acquired);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[derive(Clone, Copy, Debug)]
    enum ScriptedCompletion {
        Draw,
        Compute,
    }

    #[derive(Debug)]
    struct ScriptedExecutor {
        completion: ScriptedCompletion,
        capabilities: ExecutorCapabilities,
        resident_generation: Option<u32>,
        guest_writes: GuestWriteReach,
        seen: Mutex<Vec<SubmissionContext>>,
        resets: AtomicUsize,
        write_quiesces: AtomicUsize,
    }

    impl ScriptedExecutor {
        fn new(completion: ScriptedCompletion) -> Self {
            Self {
                completion,
                capabilities: ExecutorCapabilities::default(),
                resident_generation: None,
                guest_writes: GuestWriteReach::Disjoint,
                seen: Mutex::new(Vec::new()),
                resets: AtomicUsize::new(0),
                write_quiesces: AtomicUsize::new(0),
            }
        }

        fn with_max_render_target_dimension(mut self, dimension: u32) -> Self {
            self.capabilities.max_render_target_dimension = dimension;
            self
        }

        fn with_resident_generation(mut self, generation: u32) -> Self {
            self.resident_generation = Some(generation);
            self
        }

        fn with_guest_writes(mut self, reach: GuestWriteReach) -> Self {
            self.guest_writes = reach;
            self
        }
    }

    impl CapabilityService for ScriptedExecutor {
        fn capabilities(&self) -> ExecutorCapabilities {
            self.capabilities
        }
    }

    impl PresentationService for ScriptedExecutor {}
    impl WindowPresentationService for ScriptedExecutor {}
    impl GuestPageTransferService for ScriptedExecutor {}
    impl CompletionService for ScriptedExecutor {}
    impl SubmissionBatchService for ScriptedExecutor {}
    impl GuestImportService for ScriptedExecutor {}
    impl GuestImagePlanningService for ScriptedExecutor {}
    impl MaintenanceService for ScriptedExecutor {}
    impl ObservationService for ScriptedExecutor {}
    impl ShaderTranslationService for ScriptedExecutor {}
    impl RenderBufferPlanningService for ScriptedExecutor {}

    impl ReadbackService for ScriptedExecutor {
        type Error = DrawError;

        fn read_target(&self, _identity: &TargetIdentity) -> Result<TargetReadback, Self::Error> {
            Err(DrawError::Facade(
                EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "target_readback",
                },
            ))
        }
    }

    impl SessionService for ScriptedExecutor {
        fn reset(&self) {
            self.resets.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl Executor for ScriptedExecutor {}

    impl ResidentService for ScriptedExecutor {}

    impl ComputeResidencyService for ScriptedExecutor {
        fn compute_resident_storage_generation(
            &self,
            _identity: &reims_vgpu_core::ComputeStorageResidencyKey,
        ) -> Option<u32> {
            self.resident_generation
        }
    }

    impl GuestWriteService for ScriptedExecutor {
        fn guest_writes_outstanding(&self) -> bool {
            self.guest_writes != GuestWriteReach::Disjoint
        }

        fn guest_writes_reaching(&self, _pages: &[u64]) -> GuestWriteReach {
            self.guest_writes
        }

        fn quiesce_guest_writes(&self) {
            self.write_quiesces.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl ExecutionPort for ScriptedExecutor {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            let context = submission.context;
            let identity = context.identity;
            self.seen.lock().unwrap().push(context.clone());
            Ok(ExecutionCompletion {
                submission: identity,
                output: vec![match self.completion {
                    ScriptedCompletion::Draw => ExecutionOutput::Draw(DrawOutput::default()),
                    ScriptedCompletion::Compute => {
                        ExecutionOutput::Compute(ComputeOutput::default())
                    }
                }]
                .into_boxed_slice(),
                gpu_materialized: Arc::from([]),
            })
        }
    }

    fn context() -> SubmissionContext {
        SubmissionContext {
            identity: SubmissionIdentity {
                id: SubmissionId::new(19),
                task: TaskId::new(7),
            },
            resources: Arc::from([SubmissionResourceUse {
                object: ObjectTableRef::new(31),
                resource: None,
                expected_content: None,
                validity: ResourceValidity {
                    clear_host: true,
                    set_host: false,
                    clear_guest: false,
                    set_guest: true,
                },
            }]),
            segments: Arc::from([SegmentBoundary {
                stream_index: 2,
                index: 3,
                kind: SegmentKind::Render,
                continues_previous: false,
                continues_next: true,
            }]),
            segment: Some(SegmentBoundary {
                stream_index: 2,
                index: 3,
                kind: SegmentKind::Render,
                continues_previous: true,
                continues_next: false,
            }),
        }
    }

    #[test]
    fn device_injected_executor_receives_the_complete_submission_context() {
        let scripted = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let state = Device::new_with_executor(DeviceId(1), 12, scripted.clone());
        let context = context();

        execute_draw(
            state.executor.as_ref(),
            context.clone(),
            DrawRequest::default(),
        )
        .unwrap();

        let seen = scripted.seen.lock().unwrap();
        assert_eq!(seen.as_slice(), &[context]);
    }

    #[test]
    fn device_injected_executor_owns_residency_queries() {
        let scripted = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Compute).with_resident_generation(41),
        );
        let state = Device::new_with_executor(DeviceId(1), 12, scripted);
        let key = crate::model::ComputeStorageResidencyKey::linear(
            reims_vgpu_protocol::ResourceId::new(3, 1),
            0x4000,
            256,
            4096,
            64,
            16,
            reims_vgpu_core::pixel_format::MTL_FORMAT_RGBA8_UNORM,
        );

        assert_eq!(
            state.executor.compute_resident_storage_generation(&key),
            Some(41)
        );
    }

    #[test]
    fn an_executor_without_readback_refuses_by_service_name() {
        let state = Device::new_with_executor(
            DeviceId(1),
            12,
            Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw)),
        );
        let identity = TargetIdentity::Gva {
            gva: 0x8000,
            width: 16,
            height: 16,
            generation: 3,
            format: reims_vgpu_core::pixel_format::TexelLayout::Rgba8,
        };

        assert!(matches!(
            state.executor.read_target(&identity),
            Err(DrawError::Facade(
                EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "target_readback"
                }
            ))
        ));
    }

    #[test]
    fn guest_write_settlement_uses_the_injected_executor() {
        let scripted = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw)
                .with_guest_writes(GuestWriteReach::Overlap),
        );

        crate::runtime::render_writeback::settle_guest_writes(
            scripted.as_ref(),
            crate::runtime::render_writeback::SettleSite::CompletionStamp,
        );

        assert_eq!(scripted.write_quiesces.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn executor_capabilities_are_device_owned() {
        let first = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw).with_max_render_target_dimension(4096),
        );
        let second = Arc::new(
            ScriptedExecutor::new(ScriptedCompletion::Draw)
                .with_max_render_target_dimension(16_384),
        );
        let first_state = Device::new_with_executor(DeviceId(1), 12, first);
        let second_state = Device::new_with_executor(DeviceId(2), 12, second);

        assert_eq!(
            first_state
                .executor
                .capabilities()
                .max_render_target_dimension,
            4096
        );
        assert_eq!(
            second_state
                .executor
                .capabilities()
                .max_render_target_dimension,
            16_384
        );
    }

    #[test]
    fn executor_cannot_return_a_completion_for_another_operation_kind() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        let error = execute_draw(&scripted, context(), DrawRequest::default()).unwrap_err();

        assert!(matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::ExecutorCompletionKindMismatch {
                expected: "draw",
                actual: "compute",
            })
        ));
    }

    #[derive(Debug)]
    struct WrongIdentityExecutor;

    impl CapabilityService for WrongIdentityExecutor {}
    impl PresentationService for WrongIdentityExecutor {}
    impl WindowPresentationService for WrongIdentityExecutor {}
    impl GuestPageTransferService for WrongIdentityExecutor {}
    impl CompletionService for WrongIdentityExecutor {}
    impl SubmissionBatchService for WrongIdentityExecutor {}
    impl GuestImportService for WrongIdentityExecutor {}
    impl GuestImagePlanningService for WrongIdentityExecutor {}
    impl MaintenanceService for WrongIdentityExecutor {}
    impl ObservationService for WrongIdentityExecutor {}
    impl ShaderTranslationService for WrongIdentityExecutor {}
    impl RenderBufferPlanningService for WrongIdentityExecutor {}
    impl SessionService for WrongIdentityExecutor {}
    impl ReadbackService for WrongIdentityExecutor {
        type Error = DrawError;

        fn read_target(&self, _identity: &TargetIdentity) -> Result<TargetReadback, Self::Error> {
            Err(DrawError::Facade(
                EngineFacadeDecline::ExecutorServiceUnavailable {
                    service: "target_readback",
                },
            ))
        }
    }
    impl Executor for WrongIdentityExecutor {}
    impl ResidentService for WrongIdentityExecutor {}
    impl GuestWriteService for WrongIdentityExecutor {}
    impl ComputeResidencyService for WrongIdentityExecutor {}

    impl ExecutionPort for WrongIdentityExecutor {
        type Submission = ResolvedSubmission;
        type Completion = ExecutionCompletion;
        type Error = DrawError;

        fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error> {
            let task = submission.context.identity.task;
            Ok(ExecutionCompletion {
                submission: SubmissionIdentity {
                    id: SubmissionId::new(20),
                    task,
                },
                output: vec![ExecutionOutput::Draw(DrawOutput::default())].into_boxed_slice(),
                gpu_materialized: Arc::from([]),
            })
        }
    }

    #[test]
    fn completion_identity_must_match_the_owned_submission() {
        let error =
            execute_draw(&WrongIdentityExecutor, context(), DrawRequest::default()).unwrap_err();
        assert!(matches!(
            error,
            DrawError::Facade(EngineFacadeDecline::ExecutorCompletionIdentityMismatch {
                expected,
                actual,
            }) if expected == context().identity && actual == SubmissionIdentity {
                id: SubmissionId::new(20),
                task: TaskId::new(7),
            }
        ));
    }

    #[test]
    fn compute_uses_the_same_execution_port() {
        let scripted = ScriptedExecutor::new(ScriptedCompletion::Compute);
        execute_compute(&scripted, context(), ComputeRequest::default()).unwrap();

        assert_eq!(scripted.seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn resetting_one_device_preserves_its_executor_and_does_not_reset_another() {
        let first_executor = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let second_executor = Arc::new(ScriptedExecutor::new(ScriptedCompletion::Draw));
        let mut first =
            crate::runtime::Device::new_with_executor(DeviceId(1), 12, first_executor.clone());
        let mut second =
            crate::runtime::Device::new_with_executor(DeviceId(2), 12, second_executor.clone());

        first.reset();
        execute_draw(first.executor.as_ref(), context(), DrawRequest::default()).unwrap();
        execute_draw(second.executor.as_ref(), context(), DrawRequest::default()).unwrap();

        assert_eq!(first_executor.resets.load(Ordering::Relaxed), 1);
        assert_eq!(second_executor.resets.load(Ordering::Relaxed), 0);
        assert_eq!(first_executor.seen.lock().unwrap().len(), 1);
        assert_eq!(second_executor.seen.lock().unwrap().len(), 1);
        second.reset();
    }
}
