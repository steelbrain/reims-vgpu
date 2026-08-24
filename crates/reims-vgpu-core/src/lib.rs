//! Backend-independent state and transitions for the virtual GPU.
//!
//! This crate deliberately has no QEMU, Vulkan, windowing, or host-OS types.

pub mod blit;
pub mod capabilities;
pub mod compute;
pub mod content_tracking;
pub mod display;
pub mod draw;
pub mod draw_preparation;
pub mod endian;
pub mod execution;
pub mod fnv;
pub mod gather;
pub mod icb;
pub mod map_audit;
pub mod mapper;
pub mod mapping;
pub mod materialization;
pub mod namespace;
pub mod node_guard;
pub mod observation;
pub mod pixel_format;
pub mod preparation;
pub mod registers;
pub mod released_pages;
pub mod render;
pub mod residency;
pub mod resource;
pub mod resource_state;
pub mod scheduler;
pub mod service;
pub mod shader_interface;
pub mod stamp;
pub mod submission;
pub mod synchronization;
pub mod target;
pub mod task;
pub mod texel;
pub mod viewport;
pub mod visibility;

pub use blit::{
    BufferFillPattern, ResolvedBlit, ResolvedBufferRange, ResolvedBufferToTextureBlit,
    ResolvedLinearTextureLevel, ResolvedSurfaceTextureBacking, ResolvedTextureBacking,
    ResolvedTextureCopyBatch, ResolvedTextureEndpoint, ResolvedTextureLevelCopy,
    ResolvedTextureToBufferBlit, ResolvedTextureToTextureBlit, TextureExtent, TextureOrigin,
};
pub use capabilities::{CapabilityService, DeviceInfoLimits, ExecutorCapabilities, MAX_CHANNELS};
pub use compute::{
    ComputeBarrier, ComputeBufferBacking, ComputeBufferOutput, ComputeBufferResource,
    ComputeBufferResult, ComputeImageDestination, ComputeImageResult, ComputeOutput,
    ComputeRequest, ComputeResidentSampleBind, ComputeSampledImageResource,
    ComputeSampledImageSource, ComputeStorageImageResource, ComputeStorageImageSeed,
    ComputeStorageResidency, SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction,
    SamplerFilter, SamplerMipFilter, SamplerResource, SamplerSource,
};
pub use content_tracking::{
    BufferWriteGens, BufferWriteStamp, GatherKey, GvaPlaneKey, GvaResourceKey, GvaStoreWitness,
    GvaTargetKey, GvaWriteReach, GvaWritebackDebt, HostWriteVerdict, HostWrites, LinearColorTarget,
    PendingWritebacks, ResourceWriteStamp, StatedGeneration,
};
pub use display::{
    CursorGlyph, CursorPosition, CursorState, DisplayHandshake, DisplayOnlinePoll,
    DisplaySharedPage,
};
pub use draw_preparation::{
    AttachmentPlanDecline, AttachmentTargetRole, DrawPreparationDecline, SamplerBindingSource,
};
pub use execution::{
    execute_resolved_submission, execute_resolved_submission_progress, BlitCompletion,
    CommandExecution, ExecutionCompletion, ExecutionKind, ExecutionOutput, ExecutionPort,
    ExecutionReceipt, ResolvedCommand, ResolvedCommandBuffer, ResolvedExecutionCompletion,
    ResolvedSubmission, ResourceStateCompletion, SubmissionExecutionProgress,
};
pub use gather::{
    fold_runs, AuditDensity, ContentAudit, GatherObservation, GatherOutcome, GatherPolicies,
    GatherReadings, GatherVerdict, GatherWindow, GatherWitness, GatheredIdentity, StatedGuestWrite,
    VouchPolicy, AUDIT_REBASELINE_LIMIT,
};
pub use icb::{IcbRecord, IcbRegistry};
pub use map_audit::{MapAudit, MapIntervals, PageSize};
pub use mapper::{MapperCapture, MapperService};
pub use mapping::{MappingContentState, ResourceValidity};
pub use materialization::{
    BoundWindowKey, GuestAddressSpan, MaterializationOwner, MaterializationRegistry,
    MaterializationRetirement, MaterializationShape,
};
pub use namespace::{NamespaceError, ReferenceNamespace, TaskReferenceStates};
pub use node_guard::{NodeVerdict, NodeWatch};
pub use observation::DeviceObservations;
pub use preparation::{
    sampled_image_shape, BindTableClass, IndexLoadReason, MrtDrop, MtlbDecline, PastTableBind,
    ResolvedRenderPipeline, SampledImageShape, SecondaryMrtRefusal, ShaderStage, VertexBindPlan,
    MAX_ANY_BIND_SLOTS, MAX_BUFFER_BIND_SLOTS, MAX_SAMPLER_BIND_SLOTS, MAX_TEXTURE_BIND_SLOTS,
};
pub use registers::{DeviceRegisters, GfxRegisters, IosfcRegisters, GFX_MMIO_SIZE};
pub use released_pages::{ReleasedPages, ReleasedVerdict, RELEASED_PAGE_WATCH_CAP};
pub use render::{
    viewport_slot_count, AttachmentInitial, AttachmentSlot, BlendFactor, BlendOp,
    BlendStateResource, BufferContent, ColorLoadAction, CullMode, DepthAspectAttachment,
    DepthAttachment, DepthClipMode, DepthState, DrawOutput, DrawRequest, FillMode, IndexType,
    IndexedDrawResource, LineWidth, PreparedRenderProgram, PreparedShaderStage, PrimitiveTopology,
    RenderBarrier, RenderBarrierStages, RenderEncoderDelta, RenderTargetExtent, SampledByteOrigin,
    SampledContentIdentity, SampledImageResource, SampledSource, ScissorResource,
    SecondaryColorTarget, SeedOrder, StencilAttachment, StencilFaceOps, StencilOp, StencilState,
    StorageBufferResource, VertexAttributeFormat, VertexAttributeResource, VertexStepFunction,
    ViewportResource, VisibilityResultMode,
};
pub use residency::{
    ComputeResidencyLedger, ComputeResidencyService, ComputeStorageOrigin,
    ComputeStorageResidencyKey, GatherVouch, ResidentContentBacking,
};
pub use resource::{
    ContentAuthority, ContentError, ContentStamp, ContentState, GraphError, LifecycleState,
    MappingNode, PendingContentWrite, ReplicaVersions, ResourceGraph, ResourceLifetime,
    ResourceLifetimeRef, ResourceNode, StorageBacking, StorageNode,
};
pub use resource_state::ResolvedResourceState;
pub use scheduler::{
    ChannelRing, ChildDrainNestingError, ChildDrainStack, PendingWork, PresentTranslationBarrier,
    TranslationOrderHold, TranslationScheduling, UnreleasedTranslationHold, WorkSchedulingState,
};
pub use service::{
    GuestWriteReach, GuestWriteService, HeapTextureImagePlan, HeapTextureRequirements,
    PreparedPresentation, PresentDecline, PresentationRoute, PresentationService,
    PresentationSource, ReadbackLease, ReadbackService, ResidentContent, ResidentReadPlan,
    ResidentReclaim, ResidentService, TargetReadback,
};
pub use shader_interface::*;
pub use stamp::{CompletionPublications, PendingStamp, StampLedger, StampWait, UnmetSource};
pub use submission::{SubmissionContext, SubmissionTracker};
pub use synchronization::{
    plan_event, plan_fence, BarrierResource, Decision as SynchronizationDecision,
    Domain as SynchronizationDomain, EventKind, FenceAction, FenceSignal, MemoryBarrierScope,
    Plan as SynchronizationPlan, Reason as SynchronizationReason, TaskEventStates, TaskFenceStates,
    TaskGenerationStates, FENCE_INITIAL_GENERATION,
};
pub use target::{TargetIdentity, TargetKeyDivergence};
pub use task::{TaskEntry, TaskTable};
pub use texel::{
    expand_rgba8_to_texel, f16_to_f32, f16_to_unorm8, narrow_texel_to_rgba8, unorm8_to_f16,
};
pub use viewport::{aspect_fit_viewport, pointer_to_guest, PresentationViewport};
