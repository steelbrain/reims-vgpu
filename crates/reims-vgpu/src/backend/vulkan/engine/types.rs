//! Draw request surface for the internal Vulkan engine (v1 §1.2 surface).
//!
//! Field meanings match the historical Metal→Vulkan product draw seam
//! (blend, Load seed, stage-in attributes, SSBOs, sampled images).

use ash::vk;

use crate::backend::vulkan::translate;
pub use crate::contract::pass_action::LoadAction;
pub use crate::runtime::decode::resource::ColorWriteMask;

/// Named engine failure. Stable prefixes for observe greps (`vk_engine_*`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    /// Init / ICD / device selection failed. Latched by `ContextOwner`, except
    /// when it is out of memory — see `ContextOwner::note_init_failure`.
    Init(super::init_decline::InitDecline),
    /// Understood but declined — a capability this device or this engine does
    /// not have. Typed so each distinct check carries its own `reason=` slug;
    /// see [`super::reason::DrawReason`].
    Unsupported(super::reason::DrawReason),
    /// Engine façade or host-window presenter state changed under a valid
    /// request, or a façade input cannot describe a scanout.
    Facade(super::facade_decline::EngineFacadeDecline),
    /// Runtime pipeline/MTLB/AIR preparation failed before an engine request
    /// could be validated.
    DrawPreparation(super::draw_preparation::DrawPreparationDecline),
    /// Draw request rejected before context creation or GPU work.
    DrawValidation(super::draw_validation::DrawValidationDecline),
    /// A validated draw request failed while materializing execution state.
    DrawExecution(super::draw_execution::DrawExecutionDecline),
    /// Compute request rejected before context creation or GPU work.
    ComputeValidation(super::compute_validation::ComputeValidationDecline),
    /// A validated compute request lost or mismatched resident execution state.
    ComputeExecution(super::compute_execution::ComputeExecutionDecline),
    /// A resident-target readback could not find its content.
    /// See [`super::reason::TargetReadDecline`].
    TargetRead(super::reason::TargetReadDecline),
    /// A resident's frame could not be copied straight into the guest's pages,
    /// so the flush owes the CPU route instead.
    /// See [`super::host_ram::GuestWriteDecline`].
    GuestPageWrite(super::host_ram::GuestWriteDecline),
    /// A specific Vulkan call that returned an error, typed by *(rail,
    /// operation)*. Former `Vulkan(String)` sites move here so the log names
    /// which call refused.
    /// See [`super::vk_call::VkCall`].
    VkCall(super::vk_call::VkCall),
    /// The image-memory slab rejected an impossible allocation/invariant
    /// without pretending the driver returned OOM.
    Slab(super::slab::SlabDecline),
    /// Fence wait timed out.
    FenceTimeout,
    /// Device lost and recreate budget exhausted (or mid-draw loss).
    DeviceLost(super::device_lost::DeviceLostDecline),
}

impl DrawError {
    /// Whether this refusal is the device saying it has no memory left, as
    /// opposed to refusing for any other reason.
    ///
    /// The one class worth retrying: it is a statement about how much memory is
    /// in use at this instant rather than about the request, so giving memory
    /// back can change the answer. Every other `DrawError` describes something
    /// about the request or the driver that a second identical attempt would
    /// meet again.
    ///
    /// Both Vulkan out-of-memory results count. `ERROR_OUT_OF_HOST_MEMORY` is
    /// included because this device's pools hold host allocations too — the
    /// HOST_VISIBLE staging and readback rings — so the same reclaim is the
    /// right response to either. `ERROR_DEVICE_LOST` deliberately is not: it has
    /// its own variant and is answered by recreating the context, and retrying
    /// an allocation against a lost device would only fail again.
    ///
    /// [`Self::Init`] answers here too, and it is the arm with the widest blast
    /// radius. `vkCreateInstance` and `vkCreateDevice` both refuse with
    /// `ERROR_OUT_OF_HOST_MEMORY`, and bring-up is latched by
    /// `ContextOwner::init_error` — so a host that was momentarily short of RAM
    /// at the first draw would otherwise take the whole Vulkan engine down for
    /// the life of the process. The bring-up checks this device decides itself
    /// (no loader, no device, no graphics queue, below the API floor) carry no
    /// result and are correctly permanent.
    pub fn out_of_memory(&self) -> bool {
        let result = match self {
            Self::VkCall(c) => Some(c.result),
            Self::Init(d) => d.vk_result(),
            _ => None,
        };
        matches!(
            result,
            Some(ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
                | Some(ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        )
    }
}

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(d) => write!(f, "vk_engine_init: {d}"),
            Self::Unsupported(r) => write!(f, "vk_engine_unsupported: {r}"),
            Self::Facade(d) => write!(f, "vk_engine_facade: {d}"),
            Self::DrawPreparation(d) => write!(f, "vk_engine_draw_preparation: {d}"),
            Self::DrawValidation(d) => write!(f, "vk_engine_draw_validation: {d}"),
            Self::DrawExecution(d) => write!(f, "vk_engine_draw_execution: {d}"),
            Self::ComputeValidation(d) => write!(f, "vk_engine_compute_validation: {d}"),
            Self::ComputeExecution(d) => write!(f, "vk_engine_compute_execution: {d}"),
            Self::TargetRead(d) => write!(f, "vk_engine_target_read: {d}"),
            Self::GuestPageWrite(d) => write!(f, "vk_engine_guest_page_write: {d}"),
            Self::VkCall(c) => write!(f, "vk_engine_vk: {c}"),
            Self::Slab(d) => write!(f, "vk_engine_slab: {d}"),
            Self::FenceTimeout => write!(f, "vk_engine_fence_timeout"),
            Self::DeviceLost(d) => write!(f, "vk_engine_device_lost: {d}"),
        }
    }
}

impl std::error::Error for DrawError {}

impl crate::observe::Decline for DrawError {
    /// Every variant delegates to the typed decline that names its check, so
    /// one event has one reason at every layer.
    fn slug(&self) -> &'static str {
        match self {
            Self::TargetRead(d) => d.slug(),
            Self::GuestPageWrite(d) => d.slug(),
            Self::Unsupported(r) => r.slug(),
            // Delegates like the two typed variants above: the call names itself,
            // so one event has one name whether it is read here or on `VkCall`.
            Self::VkCall(c) => c.slug(),
            Self::Slab(d) => d.slug(),
            Self::FenceTimeout => "vk_engine_fence_timeout",
            Self::Init(d) => d.slug(),
            Self::Facade(d) => d.slug(),
            Self::DrawPreparation(d) => d.slug(),
            Self::DrawValidation(d) => d.slug(),
            Self::DrawExecution(d) => d.slug(),
            Self::ComputeValidation(d) => d.slug(),
            Self::ComputeExecution(d) => d.slug(),
            Self::DeviceLost(d) => d.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::TargetRead(d) => d.fields(),
            Self::GuestPageWrite(d) => d.fields(),
            Self::Unsupported(r) => r.fields(),
            Self::VkCall(c) => c.fields(),
            Self::Slab(d) => d.fields(),
            Self::Init(d) => d.fields(),
            Self::DrawValidation(d) => d.fields(),
            Self::DrawExecution(d) => d.fields(),
            Self::ComputeValidation(d) => d.fields(),
            Self::ComputeExecution(d) => d.fields(),
            Self::DeviceLost(d) => d.fields(),
            Self::FenceTimeout => Vec::new(),
            Self::Facade(d) => d.fields(),
            Self::DrawPreparation(d) => d.fields(),
        }
    }
}

impl From<DrawError> for String {
    fn from(e: DrawError) -> Self {
        e.to_string()
    }
}

/// What an armed occlusion query counts (Metal `MTLVisibilityResultMode`).
///
/// `MTLVisibilityResultModeDisabled` is deliberately **not** a variant. It means
/// "no query", which is what the `Option` in [`DrawRequest::occlusion_query`]
/// already says, and a second spelling of the same fact is a state two readers
/// can disagree about. [`crate::backend::vulkan::translate::raster::visibility_result_mode`]
/// is where the guest's `0` becomes that `None`.
///
/// The two arms are not equally cheap. Vulkan's occlusion query is imprecise by
/// default — it promises only "non-zero if any sample passed", which is exactly
/// [`Self::Boolean`] — and an exact count needs `VK_QUERY_CONTROL_PRECISE_BIT`,
/// which is gated on the `occlusionQueryPrecise` device feature. So a host that
/// lacks the feature can still serve `Boolean` and must **refuse** `Counting`:
/// an imprecise query answering a counting guest is a plausible wrong number,
/// which is the one outcome worse than a named refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VisibilityResultMode {
    /// `MTLVisibilityResultModeBoolean` — did anything pass.
    Boolean,
    /// `MTLVisibilityResultModeCounting` — how many samples passed.
    Counting,
}

/// Face-culling mode (Metal `MTLCullMode`). The macOS 2D compositor issues no
/// draw that binds a cull mode, so `None` (the default) keeps the whole UI path
/// byte-identical to the pre-cull engine — the raster state stays `CULL_NONE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum CullMode {
    #[default]
    None,
    Front,
    Back,
}

/// Triangle rasterization mode (Metal `MTLTriangleFillMode`).
///
/// Metal has two: fill the interior, or rasterize the edges as lines. Vulkan
/// spells the second as `VK_POLYGON_MODE_LINE`, which is gated on the
/// `fillModeNonSolid` device feature — so unlike [`CullMode`] the non-default
/// arm can be refused by the host, and `engine::caches` declines the pipeline
/// rather than filling a wireframe.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum FillMode {
    #[default]
    Fill,
    Lines,
}

/// What happens to a fragment outside the depth range (Metal
/// `MTLDepthClipMode`).
///
/// `Clip` discards it, which is Metal's default and Vulkan's unconditional
/// behaviour with `depthClampEnable` clear. `Clamp` pins its depth to the near
/// or far plane and keeps it — Vulkan's `depthClampEnable`, gated on the
/// `depthClamp` device feature. A shadow-map or skybox pass that asked for
/// `Clamp` and got `Clip` loses the geometry nearest the camera, so the absent
/// feature is a refusal rather than a fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub enum DepthClipMode {
    #[default]
    Clip,
    Clamp,
}

/// Per-draw depth-test state (Metal `MTLDepthStencilState` + depth attachment).
/// When a `DrawRequest` carries `Some`, the engine attaches a depth buffer to
/// the pass and enables the depth test; `None` (the default) means no depth
/// attachment at all — byte-identical to the pre-depth engine, which is the
/// whole macOS 2D UI path (it binds no depth-stencil).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthState {
    /// The guest texture this depth attachment names, when the render pass
    /// descriptor named one.
    ///
    /// The depth buffer is **the guest's resource**, not this device's scratch:
    /// the guest allocated a depth texture and bound it, and
    /// [`crate::runtime::decode::render::DepthAttachment::texture_ref`] is its
    /// ref. Carrying it lets the engine resolve one resident per guest texture
    /// out of the registry, which is what makes the depth allocation live as
    /// long as the guest's texture does instead of as long as one draw.
    ///
    /// `None` is a draw that bound a non-trivial `MTLDepthStencilState` with no
    /// depth attachment in its pass descriptor. There is no guest resource to
    /// key on, so the engine falls back to a per-draw transient buffer. The two
    /// rails are counted apart in the `vk_alloc_sites` census — `depth_resident`
    /// against `transient_depth` — so the fallback's share is a reading rather
    /// than an assumption.
    pub identity: Option<TargetIdentity>,
    /// `false` disables the test (draw always passes) — used only when a bound
    /// depth-stencil is non-trivial in some *other* way (e.g. a write with
    /// compare Always); the plain trivial state never reaches here.
    pub test_enable: bool,
    pub write_enable: bool,
    /// Metal `MTLCompareFunction` — the same enum Metal uses for sampler
    /// compare, hence the shared type (values Never=0 .. Always=7).
    pub compare: SamplerCompareFunction,
    /// Depth clear value (Metal `MTLRenderPassDepthAttachment.clearDepth`).
    pub clear_value: f32,
    /// `true` ⇒ LOAD the existing depth resident (multi-pass); `false` ⇒ CLEAR
    /// to `clear_value`. The transient-depth increment only supports CLEAR.
    pub load: bool,
    /// `Some` when the bound `MTLDepthStencilState` enables the stencil test on
    /// either face. Engages the combined depth-stencil attachment (D32_SFLOAT_S8
    /// with a STENCIL aspect) and the pipeline's front/back stencil op state.
    /// `None` (the default) keeps the depth-only D32_SFLOAT path byte-identical.
    pub stencil: Option<StencilState>,
}

/// Metal `MTLStencilOperation` → Vulkan `VkStencilOp`. The two enums share the
/// same ordering (Keep=0 .. DecrementWrap=7), but the mapping is spelled out
/// explicitly so a contract drift is caught by the compiler rather than aliased
/// through a numeric cast.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
}

impl StencilOp {
    pub(crate) fn vk(self) -> vk::StencilOp {
        translate::raster::vk_stencil_op(self)
    }
}

/// The pipeline-relevant half of one Metal `MTLStencilDescriptor` face: the
/// compare function, the three stencil ops, and the read/write masks. Excludes
/// the reference value, which Metal sets via a *separate* dynamic command
/// (`SetStencilReferenceValue`) — mirrored on Vulkan with dynamic stencil
/// reference so distinct reference values do not multiply pipelines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StencilFaceOps {
    pub compare: SamplerCompareFunction,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// Per-draw stencil-test state (Metal `MTLDepthStencilState` front/back faces +
/// `SetStencilReferenceValue`). Present in [`DepthState::stencil`] only when the
/// bound state enables stencil on a face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StencilState {
    pub front: StencilFaceOps,
    pub back: StencilFaceOps,
    /// Reference values (Metal `setStencilFrontReferenceValue:backReferenceValue:`),
    /// applied as Vulkan dynamic stencil reference per face — not baked into the
    /// pipeline key.
    pub reference_front: u32,
    pub reference_back: u32,
    /// Stencil clear value (Metal `MTLRenderPassStencilAttachment.clearStencil`).
    /// The transient stencil buffer only supports CLEAR.
    pub clear_value: u32,
}

/// Identity of one retained guest render-pipeline state.
///
/// The token carries no Vulkan object and exposes no guest reference. It lets
/// the engine remember the last exact Vulkan variant used by this immutable
/// pipeline object without globally hashing the object's complete content key
/// on every draw. A weak copy in the engine index follows this token's
/// lifetime, so deleting the guest object does not leave an immortal identity
/// entry behind.
#[derive(Clone, Debug)]
pub struct PipelineObjectIdentity {
    id: std::num::NonZeroU64,
    life: std::sync::Arc<PipelineObjectLife>,
}

#[derive(Debug)]
pub(crate) struct PipelineObjectLife;

impl PipelineObjectIdentity {
    pub(crate) fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1);
        let raw = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .expect("retained pipeline identity space exhausted");
        Self {
            id: std::num::NonZeroU64::new(raw)
                .expect("pipeline identity allocator never publishes zero"),
            life: std::sync::Arc::new(PipelineObjectLife),
        }
    }

    pub(crate) fn id(&self) -> std::num::NonZeroU64 {
        self.id
    }

    pub(crate) fn downgrade(&self) -> std::sync::Weak<PipelineObjectLife> {
        std::sync::Arc::downgrade(&self.life)
    }
}

/// Inputs for one offscreen draw. Engine receives resolved bytes + post-reloc SPIR-V only.
#[derive(Debug, Default)]
pub struct DrawRequest {
    /// The retained guest pipeline object this draw resolved, when the retained
    /// lifecycle is enabled. Vulkan still compares the complete variant key;
    /// this identity only chooses the object's exact front entry.
    pub pipeline_object: Option<PipelineObjectIdentity>,
    /// Shared from the runtime translation cache — the engine never mutates
    /// module words; `Arc` avoids a full-module copy per draw.
    pub vert_spirv: std::sync::Arc<Vec<u32>>,
    pub frag_spirv: std::sync::Arc<Vec<u32>>,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub first_vertex: u32,
    pub instance_count: Option<u32>,
    /// Metal baseInstance / Vulkan firstInstance. Constant step-function shift uses this.
    pub base_instance: u32,
    pub primitive_topology: PrimitiveTopology,
    /// Pipeline rasterization sample count.
    pub raster_sample_count: u32,
    /// Sample count of the colour attachment the fragment pipeline writes.
    /// An explicit resolve still names its multisample source here; the
    /// single-sample destination is represented by [`Self::multisample_resolve`].
    pub color_sample_count: u32,
    /// Rasterize into an N-sample attachment and resolve into the ordinary
    /// primary target at render-pass end.
    pub multisample_resolve: bool,
    /// Every viewport the guest bound, in its order. Empty takes the
    /// full-target default, and so does any slot past the end of this list when
    /// [`Self::scissors`] is longer — the two counts are independent in Metal
    /// and must be one number in a Vulkan pipeline, so the shorter list is
    /// defaulted per slot rather than the longer one truncated.
    pub viewports: Vec<ViewportResource>,
    /// Every scissor rect the guest bound, in its order, on the same terms as
    /// [`Self::viewports`]. Slot `i` clips viewport `i`.
    pub scissors: Vec<ScissorResource>,
    // `viewport_slot_count` below is the one reader that turns the two lists
    // above into the single number Vulkan wants.
    /// The occlusion query this draw is armed with, or `None` for a draw the
    /// guest left unarmed — either because the pass bound no visibility result
    /// buffer or because the encoder state is `MTLVisibilityResultModeDisabled`.
    ///
    /// Where the guest's *offset* into that buffer went is deliberately not
    /// here. This engine begins and ends one render pass per request and Vulkan
    /// requires a query to begin and end inside one subpass, so a Metal pass
    /// whose counter spans several draws becomes several queries whose results
    /// the caller sums into one offset. The engine answers "how many samples
    /// did *this* draw pass"; which guest word that accumulates into is the
    /// caller's question, and splitting it that way is what keeps the sum in
    /// one place instead of once per backend.
    pub occlusion_query: Option<VisibilityResultMode>,
    pub indexed: Option<IndexedDrawResource>,
    pub vertex_attributes: Vec<VertexAttributeResource>,
    pub storage_buffers: Vec<StorageBufferResource>,
    pub sampled_images: Vec<SampledImageResource>,
    pub samplers: Vec<SamplerResource>,
    /// CPU Load seed for the color target, in the order
    /// [`DrawRequest::target_seed_order`] names.
    ///
    /// Shared rather than owned so a caller holding the frame behind an `Arc` —
    /// `surface_cache` does — can seed a draw with a refcount instead of a
    /// whole-framebuffer copy.
    pub target_rgba8: Option<std::sync::Arc<Vec<u8>>>,
    /// Guest-page form of the same LOAD seed. Mutually exclusive with
    /// [`Self::target_rgba8`], [`Self::load_from_target`] and
    /// [`Self::seed_from_target`]. The engine imports/gathers these bytes in
    /// the draw command buffer and falls back to their host aliases if import
    /// is unavailable, so the runtime never needs to allocate a framebuffer to
    /// express the surface's own contents.
    pub target_guest_seed: Option<GuestTargetSeed>,
    /// Byte order of the CPU seed above, relative to the attachment it seeds.
    ///
    /// The attachment's order is [`TargetIdentity::is_bgra`] and nothing else.
    /// When the two disagree the exchange is folded into the copy
    /// into the mapped staging span, which has to happen regardless — so a
    /// caller whose pixels are already in guest scanout order never has to
    /// materialize a converted frame to seed a draw with them.
    pub target_seed_order: SeedOrder,
    pub blend: Option<BlendStateResource>,
    /// Which channels the primary colour attachment writes.
    ///
    /// Separate from `blend` because `MTLColorWriteMask` is independent of
    /// `blendingEnabled`: an unblended attachment with a mask still leaves the
    /// unwritten channels alone, so folding it into `Option<BlendStateResource>`
    /// would drop it on every unblended draw.
    pub color_write_mask: ColorWriteMask,
    /// Protocol-derived target identity for GPU residency (workstream D).
    pub target_identity: Option<TargetIdentity>,
    /// Format of colour attachment zero's texture view.
    ///
    /// This can differ from [`TargetIdentity::resident_format`] without naming
    /// another allocation. Metal texture views over one surface commonly use
    /// the linear and sRGB members of one format-compatibility class; Vulkan
    /// represents that distinction on the image view and render pass.
    pub color_attachment_format: Option<vk::Format>,
    /// Stable shared allocation that may back the primary resident image
    /// directly.
    ///
    /// This is the retained backing named by the guest surface, not a staging
    /// source. The runtime only constructs it after revalidating the mapping's
    /// page ownership and obtaining a host alias whose lifetime covers the
    /// device. The Vulkan engine still verifies the complete image-binding
    /// equation (layout offset, row pitch, allocation extent and memory type)
    /// before using it; any mismatch keeps the ordinary resident image.
    pub guest_target_memory: Option<GuestTargetMemory>,
    /// Load the primary attachment's prior contents from
    /// [`Self::guest_target_memory`] when that backing is admitted.
    ///
    /// Separate from carrying the backing because CLEAR and DontCare Stores
    /// should still render directly into guest memory while discarding its old
    /// texels. This is true only when the guest's load source is that same
    /// surface allocation, never for an explicit texture-derived seed.
    pub load_guest_target_backing: bool,
    /// Load the live GPU image for [`DrawRequest::target_identity`] instead of
    /// seeding the attachment from the CPU. Requires that resident to exist.
    ///
    /// This, [`Self::target_rgba8`], [`Self::target_guest_seed`] and
    /// [`Self::seed_from_target`] are the four ways slot 0's prior contents can
    /// arrive, and they are ordered: `load_from_target` wins, else exactly one
    /// seed is copied.
    ///
    /// They used to be described here as "the whole load action", and the engine
    /// read them that way -- `PassKey::load_seed` was `any of these four is
    /// set`. They are not the action; they are what a `Load` action was able to
    /// find. The action itself is [`Self::color0_load`], and the two questions
    /// are answered separately because they have different answers: a guest can
    /// declare `Load` and this device can arrive with nothing, which is a loss
    /// (`load_seed_lost`) rather than a `Clear`.
    pub load_from_target: bool,
    /// The guest's declared `MTLLoadAction` for colour slot 0.
    ///
    /// Exported by the runtime rather than reconstructed here from the seed
    /// fields above. That reconstruction is what made `MTLLoadActionDontCare`
    /// unreachable: no seed and no clear is indistinguishable from a `Clear`
    /// once the only thing crossing the boundary is "did bytes arrive", and the
    /// engine duly cleared -- to `target_clear`, which for the DontCare arm of
    /// the producer was left at its `[0.0; 4]` initializer, i.e. **transparent
    /// black**. A menu whose backing the guest declared undefined therefore
    /// arrived with alpha 0 across the whole attachment.
    ///
    /// `Default` is [`LoadAction::Clear`]; see that type for why an unstated
    /// action must not be the one that discards.
    pub color0_load: LoadAction,
    /// Clear value for the primary colour attachment, in semantic float
    /// channels — the same shape [`SecondaryColorTarget::clear`] has carried all
    /// along, and consulted only when the pass resolves to `loadOp = CLEAR`.
    ///
    /// This used to not exist. The primary's `VkClearValue` was `[0, 0, 0, 0]`
    /// unconditionally, so a `MTLLoadActionClear` with a colour could not be
    /// expressed — and the runtime met the contract by allocating a
    /// whole-attachment RGBA8 bitmap of that solid colour on the CPU, handing it
    /// over as `target_rgba8`, and paying a channel exchange and a staged upload
    /// to put a constant into every texel. That also forced the pass key to a
    /// LOAD pass, because a present seed is what `load_seed` means, so a draw
    /// that asked to discard its attachment loaded it instead.
    ///
    /// Floats rather than the unorm8 the seed quantised to, which is what the
    /// contract says: an sRGB attachment takes its clear in linear space and the
    /// driver encodes it, where the byte path wrote pre-quantised values past
    /// the encode entirely.
    pub target_clear: [f32; 4],
    /// When true, skip full-frame readback (non-Store / ticket path). Content
    /// remains on the GPU under `target_identity` when provided.
    pub skip_readback: bool,
    /// Publish a Store into an admitted guest-backed primary attachment to the
    /// guest-write completion ledger in this draw's engine transaction.
    ///
    /// This is meaningful only when the resolved target is actually backed by
    /// [`Self::guest_target_memory`]. An ordinary resident ignores it and keeps
    /// the copied-resource writeback path.
    pub record_guest_store: bool,
    /// Present-boundary GPU seed: copy this READY resident target's content
    /// into the draw target before the pass (which then runs with LOAD),
    /// eliding the CPU front-frame read + full-frame seed upload. Requires
    /// `target_identity`, identical geometry, and the same bgra format;
    /// mutually exclusive with a CPU seed / `LoadFromTarget`, and the source
    /// must not also be bound as a sampled image in the same draw.
    pub seed_from_target: Option<TargetIdentity>,
    /// Secondary color attachments (MRT slot >= 1). Empty ⇒ the classic
    /// single-attachment path, byte-identical to the pre-MRT engine. Slot 0 is
    /// the primary target (`target_identity` / pooled). Each secondary persists
    /// as its own resident so a later draw can bind it via
    /// [`SampledSource::Target`] — this is how a fragment shader's secondary
    /// output (e.g. the vibrancy coverage mask that a subsequent draw samples)
    /// is produced instead of silently discarded. Requires `target_identity`
    /// (the resident path); the pooled single-RT path never carries secondaries.
    pub secondary_targets: Vec<SecondaryColorTarget>,
    /// Face culling (Metal `MTLCullMode`). `None` (default) draws both faces —
    /// the 2D UI path. `Front`/`Back` reproduce Metal culling; which winding is
    /// "front" is `front_face_ccw`, mapped to a Vulkan winding by
    /// [`crate::backend::vulkan::translate::raster::vk_front_face`].
    pub cull_mode: CullMode,
    /// Metal front-facing winding: `true` = counter-clockwise (`MTLWinding`
    /// CounterClockwise), `false` = the Metal default clockwise. Only affects
    /// rasterization when `cull_mode` culls a face.
    pub front_face_ccw: bool,
    /// Triangle fill mode (Metal `setTriangleFillMode:`). `Fill` (the default)
    /// is Metal's own and needs no device feature; `Lines` names
    /// `VK_POLYGON_MODE_LINE` and is refused where `fillModeNonSolid` is not
    /// advertised.
    pub fill_mode: FillMode,
    /// Depth clip mode (Metal `setDepthClipMode:`). `Clip` (the default) is
    /// Metal's own; `Clamp` sets `depthClampEnable` and is refused where the
    /// `depthClamp` feature is not advertised.
    pub depth_clip: DepthClipMode,
    /// Depth test + transient depth attachment. `None` (default) = no depth
    /// buffer, byte-identical to the pre-depth 2D path. Set only for a draw that
    /// bound a non-trivial `MTLDepthStencilState` (see `runtime::draw`).
    pub depth: Option<DepthState>,
    /// Fragment shader reads its destination pixel (Metal framebuffer fetch:
    /// an `air.render_target` INPUT param, translated as a `SubpassData` image
    /// at [`COLOR_INPUT_BINDING`]). The engine then references attachment 0 as
    /// a subpass input (GENERAL layout, BY_REGION self-dependency) and writes
    /// an INPUT_ATTACHMENT descriptor pointing at the color target's view.
    /// `false` (default) keeps the pass byte-identical to the pre-fetch engine.
    pub color_input: bool,
    /// The preceding engine request belongs to this draw's Metal render
    /// encoder. Used only when its Vulkan pass is still open and identical.
    pub continues_render_pass: bool,
    /// The decoded Metal render encoder contains another draw after this one.
    /// Allows the Vulkan pass to remain open across the engine-call boundary.
    pub render_pass_continues: bool,
}

impl DrawRequest {
    /// Whether this draw binds `identity` as one of its own attachments.
    ///
    /// Sampling an attachment the same draw renders into is an attachment
    /// feedback loop. Vulkan requires that relationship to be declared through
    /// an optional extension; `exec` binds the resident under that contract
    /// where available and snapshots it on every other host or view shape.
    ///
    /// **Every attachment, not just the primary.** The test used to be
    /// `req.target_identity == Some(identity)` written at the one call site,
    /// which is slot 0 alone: a draw sampling one of its own MRT secondaries or
    /// its own depth target compared unequal and took the bind-it-directly arm.
    /// `SecondaryColorTarget::identity` exists precisely so a later draw can
    /// sample that attachment, so the same-draw case is reachable by
    /// construction rather than hypothetically.
    ///
    /// Widening this can only select an attachment-safe disposition: the native
    /// extension rail when its narrower view contract also holds, or the
    /// snapshot fallback otherwise.
    pub fn writes_attachment(&self, identity: &TargetIdentity) -> bool {
        self.attachment_slot(identity).is_some()
    }

    /// The colour-attachment index occupied by `identity`, with primary at
    /// zero and MRT secondaries following in framebuffer order.
    ///
    /// Vulkan's feedback-loop layout is selected per attachment, so the exact
    /// index is part of the answer. Keeping that ordering here beside the
    /// request fields prevents the render-pass builder and sampled resolver
    /// from each reconstructing it differently.
    pub fn color_attachment_index(&self, identity: &TargetIdentity) -> Option<usize> {
        if self.target_identity.as_ref() == Some(identity) {
            Some(0)
        } else {
            self.secondary_targets
                .iter()
                .position(|target| &target.identity == identity)
                .map(|index| index + 1)
        }
    }

    /// Which of this draw's attachments `identity` is, when it is one.
    ///
    /// The slot is carried rather than a bare `bool` so the census can say which
    /// of the three matched, and two of the three answers are alarms: `Primary`
    /// is the long-handled case, while a `Secondary` or `Depth` firing is a draw
    /// the primary-only test used to hand the driver as a live feedback loop.
    /// Zero on those two is the healthy reading.
    pub fn attachment_slot(&self, identity: &TargetIdentity) -> Option<AttachmentSlot> {
        if let Some(index) = self.color_attachment_index(identity) {
            Some(if index == 0 {
                AttachmentSlot::Primary
            } else {
                AttachmentSlot::Secondary
            })
        } else if self.depth.as_ref().and_then(|d| d.identity.as_ref()) == Some(identity) {
            Some(AttachmentSlot::Depth)
        } else {
            None
        }
    }
}

/// Which attachment of a draw a sampled identity turned out to be.
///
/// A type rather than the slot number it could have been: these three are this
/// crate's own vocabulary rather than a guest value, and the consumers are a
/// census name and a snapshot decision, so an integer would cost the
/// exhaustiveness check at both and buy nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentSlot {
    Primary,
    Secondary,
    Depth,
}

impl AttachmentSlot {
    /// The census route for a draw that samples this attachment of its own.
    pub fn sampled_self_route(self) -> &'static str {
        match self {
            Self::Primary => "sampled_self_primary",
            Self::Secondary => "sampled_self_secondary",
            Self::Depth => "sampled_self_depth",
        }
    }
}

/// How many viewport/scissor slots one draw rasterizes into.
///
/// The single number a Vulkan pipeline declares and `vkCmdSetViewport` /
/// `vkCmdSetScissor` must then bind exactly. It exists as a function rather
/// than as two `len()` calls at two sites because those two sites are the
/// pipeline key and the dynamic bind: if they ever disagree the draw is
/// invalid, and the disagreement would be a validation-layer message rather
/// than a compile error.
///
/// The maximum, not either count alone. Metal lets a guest set three viewports
/// and one scissor rect; Vulkan requires `scissorCount == viewportCount`, so
/// the shorter list is defaulted per slot in the bind rather than the longer
/// one truncated — truncating would drop a viewport the guest set, which is the
/// thing this list exists to stop doing.
///
/// Never zero: a pipeline with no viewport rasterizes nothing, and an empty
/// list means "the guest bound none", which takes the full-target default.
pub fn viewport_slot_count(req: &DrawRequest) -> usize {
    req.viewports.len().max(req.scissors.len()).max(1)
}

/// Descriptor binding of the attachment-0 framebuffer-fetch input attachment.
///
/// This is the *device's* ColorInput band base, not the translator's: the band
/// moved up when the texture band was widened to Metal's 128 entries
/// (`runtime::spirv_bind::widen_sampled_bands` rewrites `dest_N` from the
/// translator's `96+N` to `192+N`). Only `dest_0` is supported. Kept equal to
/// `runtime::spirv_bind::COLOR_INPUT_BINDING_BASE` by a unit test there, because
/// the two constants live on opposite sides of the runtime/engine layering.
/// Both fragment relocations preserve it.
pub const COLOR_INPUT_BINDING: u32 = 192;

/// One MRT color attachment beyond the primary (slot 0). Persisted as its own
/// registry resident so a later draw can sample it.
#[derive(Debug, Clone)]
pub struct SecondaryColorTarget {
    /// Residency identity — the key a later draw uses to bind this attachment
    /// as a sampled `SampledSource::Target`.
    pub identity: TargetIdentity,
    pub width: u32,
    pub height: u32,
    /// Attachment format, already resolved from the guest's `MTLPixelFormat` by
    /// `translate::pixel::color_attachment`. A real `VkFormat` rather than a
    /// three-way enum, so the render pass, the pipeline key and the image agree
    /// by construction and an sRGB attachment is expressible the day the rail
    /// flips.
    pub format: vk::Format,
    /// Clear value, consulted only when [`Self::load`] is
    /// [`LoadAction::Clear`] (semantic float channels).
    ///
    /// "Only when Clear" is Metal's rule, not a local one:
    /// `MTLRenderPassAttachmentDescriptor.clearColor` is documented as read if
    /// and only if `loadAction == MTLLoadActionClear`. This field used to be
    /// consulted for DontCare as well, because `load` was a bool and DontCare
    /// fell into its false arm -- so a guest that declared these contents
    /// undefined got them filled with the descriptor's clear colour, which under
    /// Metal's own `MTLClearColorMake(0, 0, 0, 1)` default is *opaque black*.
    pub clear: [f32; 4],
    /// What this pass does with the attachment's prior contents.
    ///
    /// The guest's declared `MTLLoadAction`, folded through
    /// [`LoadAction::from_declared`] and otherwise unchanged. A `bool` here
    /// could not express DontCare and the runtime narrowed it away at the
    /// producer, which is the far side of the same collapse
    /// [`DrawRequest::color0_load`] describes -- except that the two arms
    /// fabricated *different colours* out of it, so one wire value produced
    /// transparent black on slot 0 and opaque black here.
    pub load: LoadAction,
    /// This slot's own blend state, from the pipeline's per-attachment blend
    /// descriptor. `None` ⇒ the slot writes unblended.
    ///
    /// This used to not exist, and the builder forced every secondary
    /// attachment unblended with a comment claiming "the decode side does not
    /// (yet) carry per-attachment blend state". It did:
    /// `decode::resource::RenderPipelineDescriptor::color_attachments` is a
    /// `Vec<PipelineColorAttachment>` and each entry has carried its own six
    /// blend fields all along — the Metal arm has read them per slot for as
    /// long as MRT has existed. Only the Vulkan `PipelineKey` collapsed them to
    /// one, so a guest MRT pipeline that blended slot 1 got a raw store.
    pub blend: Option<BlendStateResource>,
    /// This slot's own `MTLColorWriteMask`, for the same reason the primary
    /// carries one: it is not part of the blend state.
    pub color_write_mask: ColorWriteMask,
}

#[derive(Debug, Default)]
pub struct DrawOutput {
    pub pixels: Vec<u8>,
    /// Whether color attachment zero was rendered through the retained guest
    /// allocation supplied on this request.
    ///
    /// Reported by the engine rather than inferred by the runtime: capability,
    /// layout, memory-type and creation checks can all send one request to the
    /// ordinary resident fallback, and only the engine knows which image the
    /// draw actually encoded against.
    pub target_guest_backed: bool,
    /// Whether this draw recorded its guest-backed Store in the completion
    /// ledger before releasing the engine transaction.
    pub guest_store_recorded: bool,
    /// Exact physical pages retained by the guest-backed target whose Store was
    /// recorded. The runtime publishes this same admitted footprint to its
    /// coherence ledgers instead of reconstructing it from mutable mapping
    /// state after the engine transaction.
    pub guest_store_footprint: Option<crate::runtime::guest_ram::GuestPageFootprint>,
    /// Physical channel order of `pixels`: BGRA8 when true, semantic RGBA8
    /// otherwise. Empty when `skip_readback`, in which case this states the
    /// order the attachment *would* have read back in.
    ///
    /// Reported rather than re-derived. The order follows the resolved
    /// attachment — [`TargetIdentity::is_bgra`] for a resident target, RGBA for
    /// the pooled path — and a caller that recomputes the predicate is a caller
    /// that can disagree with the image the readback actually came out of. This
    /// is the same rule the typed-decline work applies to a `reason=`: the side
    /// that performed the operation says what it did.
    pub pixels_bgra: bool,
    /// Samples this draw passed, for a draw that armed an occlusion query.
    ///
    /// `None` where no query was armed, and never `Some(0)` standing in for it:
    /// a draw that armed a query and passed nothing is a real, useful answer —
    /// it is the whole point of an occlusion test — and folding it into the
    /// unarmed case would make "fully occluded" indistinguishable from "never
    /// asked". Same rule as [`Self::pixels_bgra`] above: the side that performed
    /// the operation says what it did.
    ///
    /// For `Boolean` this is still a count rather than a 0/1, because Vulkan
    /// reports one either way and narrowing it here would throw away
    /// information the caller may want; the guest sees whatever its own mode
    /// asked for once the caller writes it back.
    pub occlusion_samples: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub struct ViewportResource {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct ScissorResource {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PrimitiveTopology {
    Point,
    Line,
    LineStrip,
    #[default]
    Triangle,
    TriangleStrip,
}

impl PrimitiveTopology {
    pub(crate) fn vk(self) -> vk::PrimitiveTopology {
        translate::raster::vk_topology(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub(crate) fn vk(self) -> vk::IndexType {
        translate::raster::vk_index_type(self)
    }

    pub fn byte_size(self) -> usize {
        match self {
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }
}

#[derive(Debug)]
pub struct IndexedDrawResource {
    pub index_type: IndexType,
    pub index_count: u32,
    pub vertex_offset: i32,
    /// Exact resource window consumed by the fixed-function index fetch.
    pub content: BufferContent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum VertexAttributeFormat {
    UChar2,
    UChar3,
    UChar4,
    Char2,
    Char3,
    Char4,
    UChar2Normalized,
    UChar3Normalized,
    UChar4Normalized,
    Char2Normalized,
    Char3Normalized,
    Char4Normalized,
    UShort2,
    UShort3,
    UShort4,
    Short2,
    Short3,
    Short4,
    UShort2Normalized,
    UShort3Normalized,
    UShort4Normalized,
    Short2Normalized,
    Short3Normalized,
    Short4Normalized,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    Int1010102Normalized,
    UInt1010102Normalized,
    UChar4NormalizedBgra,
    UChar,
    Char,
    UCharNormalized,
    CharNormalized,
    UShort,
    Short,
    UShortNormalized,
    ShortNormalized,
    Half,
    FloatRg11B10,
    FloatRgb9E5,
}

impl VertexAttributeFormat {
    /// Deliberately no `vk_format()` here. An attribute's Vulkan format is not
    /// a property of the attribute alone: Vulkan makes the three-component
    /// 8/16-bit formats optional, so the bindable format depends on the device
    /// and on whether the attribute's stride leaves room for a wider
    /// substitute. Ask `translate::support::VertexFormatSupport::resolve`,
    /// which answers both at once; `translate::vertex::vk_format` gives the
    /// device-independent spelling for tables and tests.
    ///
    /// Bytes this attribute occupies in the guest's vertex buffer.
    ///
    /// Stated beside the Vulkan format in one table so the two cannot drift —
    /// they are the same fact twice, and held apart they diverge into a stride
    /// bug nobody is looking for.
    pub fn byte_size(self) -> u32 {
        translate::vertex::byte_size(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum VertexStepFunction {
    Constant,
    #[default]
    PerVertex,
    PerInstance,
}

impl VertexStepFunction {
    /// The `MTLVertexStepFunction` ordinal this engine step came from.
    ///
    /// The inverse of [`translate::vertex::step_function`], which is where the
    /// three accepted ordinals are chosen and where the round trip is pinned.
    /// It exists so a rule stated over the *wire* value — the step/rate pair in
    /// [`crate::contract::vertex_step`] — can be asked on this side without a
    /// second copy of the mapping.
    pub fn mtl_ordinal(self) -> u32 {
        use crate::contract::vertex_step as step;
        match self {
            Self::Constant => step::MTL_VERTEX_STEP_FUNCTION_CONSTANT,
            Self::PerVertex => step::MTL_VERTEX_STEP_FUNCTION_PER_VERTEX,
            Self::PerInstance => step::MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE,
        }
    }
}

#[derive(Debug)]
pub struct VertexAttributeResource {
    pub location: u32,
    pub binding: u32,
    pub format: VertexAttributeFormat,
    pub offset: u32,
    pub stride: u32,
    pub step_function: VertexStepFunction,
    pub step_rate: u32,
    pub content: BufferContent,
}

#[derive(Debug)]
pub struct StorageBufferResource {
    pub binding: u32,
    pub content: BufferContent,
}

/// Where a draw-time buffer's bytes come from (vertex streams, index input and
/// storage/SSBO binds).
///
/// `Bytes` is the CPU staging origin: the runtime read the guest span at
/// encode time and the engine memcpys it into a pooled host-visible staging
/// buffer. The `Arc` makes intra-draw sharing free — several attributes on
/// one interleaved stream, or a stage-in buffer doubling as a storage bind,
/// reference the same allocation instead of cloning it.
///
/// `GuestRuns` is the zero-copy origin: the GPU gathers the span straight
/// from imported guest RAM inside the draw's own command buffer (per-run
/// `cmd_copy_buffer` into the pooled staging slot the bind then uses). No
/// CPU read, no CPU memcpy — guest CPU writes are observed at execute time,
/// at least as fresh as the CPU path's encode-time read (the same in-flight
/// window contract the sampled `SampledSource::GuestRuns` rail relies on).
/// `row_length_texels` MUST be 0 (buffers have no row stride semantics).
#[derive(Clone, Debug)]
pub enum BufferContent {
    Bytes(std::sync::Arc<Vec<u8>>),
    GuestRuns(GuestRunSource),
}

impl BufferContent {
    /// Total byte length of the content (the staged/gathered span).
    pub fn len(&self) -> usize {
        match self {
            Self::Bytes(b) => b.len(),
            Self::GuestRuns(src) => src.total_len as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// CPU view of the content. `Bytes` borrows; `GuestRuns` copies the runs
    /// out of guest RAM (same freshness as the CPU staging path's encode-time
    /// read).
    ///
    /// **Nothing in the product calls this.** Both call sites are `#[cfg(test)]`,
    /// and that is the whole story of the method: it materializes a fragmented
    /// gather into one contiguous `Vec` so a test can compare it against what
    /// the guest laid out. It is not a rail, and the heap `Vec` it builds is
    /// not a cost the device pays.
    ///
    /// It claimed the opposite until the host-pointer import landed — "every
    /// `GuestRuns` bind is a CPU gather now, the GPU has no way to reach guest
    /// pages". That was true when written and is now contradicted by the
    /// `GuestRuns` doc a few lines above, on this same type: a draw-time buffer
    /// bind is gathered by `vkCmdCopyBuffer` inside the draw's own command
    /// buffer and never crosses the CPU. `write_staging_from_runs` does still
    /// exist, but on the sampled rail, where `stage_phase` records it as zero
    /// on a host that can import.
    ///
    /// Two doc comments on one type disagreeing is the divergence class
    /// `AGENTS.md` warns about. This one earns a paragraph rather than a
    /// deletion because the false half was the one a reader met first on
    /// arriving at the method, and what it told them was that the gather does
    /// not exist.
    pub fn cpu_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Self::Bytes(b) => std::borrow::Cow::Borrowed(b.as_slice()),
            Self::GuestRuns(src) => {
                let mut out = Vec::with_capacity(src.total_len as usize);
                let mut skip = src.source_offset;
                for run in src.runs.iter() {
                    let take = (src.total_len as usize).saturating_sub(out.len());
                    if take == 0 {
                        break;
                    }
                    if skip >= run.len {
                        skip -= run.len;
                        continue;
                    }
                    let within = skip as usize;
                    skip = 0;
                    let n = (run.len as usize).saturating_sub(within).min(take);
                    // SAFETY: `host_ptr` is a stable RAMBlock alias from
                    // `HostOps::map_pages`, valid for the VM lifetime; the
                    // read races guest CPU writes exactly like the staging
                    // path's `read_task_gva_by_id` copy does.
                    unsafe {
                        let slice = std::slice::from_raw_parts(
                            (run.host_ptr as *const u8).add(within),
                            n,
                        );
                        out.extend_from_slice(slice);
                    }
                }
                out.resize(src.total_len as usize, 0);
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

impl From<Vec<u8>> for BufferContent {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(std::sync::Arc::new(bytes))
    }
}

#[derive(Debug)]
pub struct SampledImageResource {
    pub binding: u32,
    /// Element within the Vulkan descriptor array at [`Self::binding`].
    pub array_element: u32,
    /// Declared descriptor-array cardinality. Scalar Metal textures carry one.
    pub descriptor_count: u32,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub arrayed: bool,
    pub volume: bool,
    pub cube: bool,
    /// Metal `texture1d` / `texture1d_array` (color-transfer LUTs): the image is
    /// created as a Vulkan 1D image so the sampled descriptor type matches the
    /// shader's declared 1D image. `height` is 1; `arrayed` selects
    /// `TYPE_1D_ARRAY`. Mutually exclusive with `volume` and `cube`.
    pub one_dim: bool,
    /// The shader declares a multisampled 2D image at this binding. Such an
    /// image can only come from a retained multisample target; linear bytes
    /// cannot be uploaded into one with a buffer-to-image copy.
    pub multisampled: bool,
    pub source: SampledSource,
    /// API resource family that produced [`SampledSource::Bytes`].
    ///
    /// This is accounting metadata, not an execution selector: it cannot change
    /// how a texture is validated, cached, uploaded, or sampled. Keeping the
    /// family on the resource lets the upload site attribute only bytes that
    /// actually missed the sampled-image cache; counting at the runtime
    /// resolver would charge cache hits as copies that never happened.
    pub byte_origin: SampledByteOrigin,
    /// Format the image and its view are created with, and the layout
    /// [`SampledSource::Bytes`] / [`SampledSource::GuestRuns`] content is read
    /// as (ignored for [`SampledSource::Target`], which carries its own
    /// resident format).
    ///
    /// Resolved by `translate::pixel::vk_texel_layout` from the contract
    /// `TexelLayout` the decode rails speak. Storing the Vulkan format rather
    /// than the layout keeps one spelling on this side of the boundary and
    /// leaves room for formats no byte-layout enum can name — an sRGB view
    /// first among them.
    pub format: vk::Format,
    /// Optional identity fast path for [`SampledSource::Bytes`] (see
    /// [`SampledContentIdentity`]); `None` keeps the content-addressed path.
    pub identity: Option<SampledContentIdentity>,
    /// Decoded type-8 view swizzle, applied as the image view's component
    /// mapping so the GPU performs it at sample time. Identity (the default)
    /// creates the same view as before. Doing this on the view rather than by
    /// rewriting texels is what lets a swizzled texture stay on whatever
    /// content rail it was already on, including the zero-copy one — a CPU
    /// remap would force every swizzled bind onto the upload path.
    pub swizzle: crate::contract::pixel_format::SwizzlePlan,
}

/// Contract-level source of a CPU-materialized sampled image.
///
/// The variants follow the decoded resource families rather than call sites so
/// the census can identify an API rail that should expose stronger backing
/// guarantees. [`Self::Synthetic`] covers tests and the fail-visible neutral
/// texture used after an unbound guest resource.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SampledByteOrigin {
    #[default]
    Synthetic,
    AttachmentAlias,
    BufferBackedTexture,
    SerializedSurfaceView,
    SurfaceHostCache,
    SurfaceGuestFallback,
    LinearTexture,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerFilter {
    #[default]
    Nearest,
    Linear,
}

impl SamplerFilter {
    pub(crate) fn vk(self) -> vk::Filter {
        translate::sampler::vk_filter(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerMipFilter {
    #[default]
    NotMipmapped,
    Nearest,
    Linear,
}

impl SamplerMipFilter {
    pub(crate) fn vk(self) -> vk::SamplerMipmapMode {
        translate::sampler::vk_mipmap_mode(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerAddressMode {
    #[default]
    ClampToEdge,
    MirrorClampToEdge,
    Repeat,
    MirrorRepeat,
    ClampToZero,
    ClampToBorderColor,
}

impl SamplerAddressMode {
    pub(crate) fn vk(self) -> vk::SamplerAddressMode {
        translate::sampler::vk_address_mode(self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerBorderColor {
    #[default]
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

// Deliberately no `vk()` here. A sampler's border colour is not a property of
// the declared colour alone: Metal's `ClampToZero` address mode forces
// transparent black whatever the descriptor says, so the two must be decided
// together — see `translate::sampler::vk_border_color_with_clamp_to_zero`.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SamplerCompareFunction {
    #[default]
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl SamplerCompareFunction {
    pub(crate) fn vk(self) -> vk::CompareOp {
        translate::raster::vk_compare_op(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SamplerResource {
    pub binding: u32,
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32, // f32 bits for Hash
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

impl SamplerResource {
    pub fn normalized_default(binding: u32) -> Self {
        Self {
            binding,
            min_filter: SamplerFilter::Linear,
            mag_filter: SamplerFilter::Linear,
            mip_filter: SamplerMipFilter::NotMipmapped,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            border_color: SamplerBorderColor::TransparentBlack,
            compare_function: SamplerCompareFunction::Never,
            lod_min: 0.0f32.to_bits(),
            lod_max: f32::MAX.to_bits(),
            max_anisotropy: 1,
            unnormalized_coordinates: false,
        }
    }

    pub fn lod_min_f32(&self) -> f32 {
        f32::from_bits(self.lod_min)
    }

    pub fn lod_max_f32(&self) -> f32 {
        f32::from_bits(self.lod_max)
    }

    /// State without binding (for L6 cache key).
    pub(crate) fn state_key(&self) -> SamplerStateKey {
        SamplerStateKey {
            min_filter: self.min_filter,
            mag_filter: self.mag_filter,
            mip_filter: self.mip_filter,
            address_mode_u: self.address_mode_u,
            address_mode_v: self.address_mode_v,
            address_mode_w: self.address_mode_w,
            border_color: self.border_color,
            compare_function: self.compare_function,
            lod_min: self.lod_min,
            lod_max: self.lod_max,
            max_anisotropy: self.max_anisotropy,
            unnormalized_coordinates: self.unnormalized_coordinates,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct SamplerStateKey {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub border_color: SamplerBorderColor,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32,
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstColor,
    OneMinusDstColor,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    /// `MTLBlendFactorSource1Color` and its three siblings — the dual-source
    /// factors, which read the fragment shader's *second* colour output.
    ///
    /// Separated from the fifteen above by [`BlendFactor::is_dual_source`]
    /// because Vulkan gates exactly these four behind the `dualSrcBlend` device
    /// feature, and a pipeline naming one without it is invalid.
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

impl BlendFactor {
    pub(crate) fn vk(self) -> vk::BlendFactor {
        translate::blend::vk_factor(self)
    }

    /// Whether this factor reads the second fragment output, and so needs
    /// `VkPhysicalDeviceFeatures::dualSrcBlend`.
    pub(crate) fn is_dual_source(self) -> bool {
        matches!(
            self,
            Self::Src1Color | Self::OneMinusSrc1Color | Self::Src1Alpha | Self::OneMinusSrc1Alpha
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOp {
    pub(crate) fn vk(self) -> vk::BlendOp {
        translate::blend::vk_operation(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BlendStateResource {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
    pub constants: [f32; 4],
}

impl BlendStateResource {
    pub(crate) fn key(&self) -> BlendKey {
        BlendKey {
            src_color: self.src_color,
            dst_color: self.dst_color,
            color_op: self.color_op,
            src_alpha: self.src_alpha,
            dst_alpha: self.dst_alpha,
            alpha_op: self.alpha_op,
            constants: self.constants.map(|c| c.to_bits()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BlendKey {
    pub src_color: BlendFactor,
    pub dst_color: BlendFactor,
    pub color_op: BlendOp,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub alpha_op: BlendOp,
    pub constants: [u32; 4],
}

// ---------------------------------------------------------------------------
// Compute request surface
// ---------------------------------------------------------------------------

/// Named compute failure. Same `vk_engine_*` prefix family as draw.
pub type ComputeError = DrawError;

/// Inputs for one compute dispatch. Engine receives resolved bytes + SPIR-V only.
#[derive(Debug, Default)]
pub struct ComputeRequest {
    /// Vulkan-dialect compute SPIR-V (LocalSize baked in by metal2vulkan).
    pub spirv: Vec<u32>,
    /// Entry point name (m2v kernel entry is `"main"`).
    pub entry: String,
    /// Workgroup counts in (x, y, z). Runtime converts threads→groups when needed.
    pub grid: [u32; 3],
    /// Storage-buffer descriptors with reflected shader write access.
    pub storage_buffers: Vec<ComputeBufferResource>,
    /// Sampled images (binding, format, geometry, immutable input bytes).
    pub sampled_images: Vec<ComputeSampledImageResource>,
    /// Separate sampler descriptors used by sampled-image operands.
    pub samplers: Vec<SamplerResource>,
    /// Storage images (binding, format, geometry, seed bytes); always read back.
    pub storage_images: Vec<ComputeStorageImageResource>,
}

#[derive(Debug, Default)]
pub struct ComputeOutput {
    /// Writable-buffer readbacks only. Read-only descriptors never cross the
    /// device→host boundary after dispatch.
    pub buffers: Vec<ComputeBufferOutput>,
    /// Image readbacks in request order (same length as `storage_images`).
    ///
    /// A third case used to leave this empty too: the dispatch copied into an
    /// imported view of the caller's guest window and `images_direct` said so,
    /// so the caller skipped its own writeback. That window is gone, and with
    /// it the flag — every non-deferred image now comes back through here.
    pub images: Vec<Vec<u8>>,
}

#[derive(Debug)]
pub struct ComputeBufferResource {
    pub binding: u32,
    pub bytes: Vec<u8>,
    /// Structurally proven write access in the SPIR-V pointer-use graph.
    pub writable: bool,
}

#[derive(Debug)]
pub struct ComputeBufferOutput {
    pub binding: u32,
    pub bytes: Vec<u8>,
}

/// Storage image for compute. Formats mirror the live `simg_u32_to_vk_storage` map.
///
/// Single-layer 2D only: a compute texture binding is staged from one type-11
/// plane window or one linear GVA level, both of which are a flat `width ×
/// height` rectangle. There is no decoded slice or depth axis on this rail, so
/// the engine builds `TYPE_2D` unconditionally.
#[derive(Debug)]
pub struct ComputeStorageImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub format: StorageImageFormat,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    /// Exact type-11 resource lifetime/view contract for persistent GPU
    /// storage. `None` keeps the conservative transient upload path.
    pub residency: Option<ComputeStorageResidency>,
    /// The caller skipped reading guest pages into `bytes` because the
    /// resident generation matched at stage time. The engine must fail
    /// visibly (never seed the zero placeholder) if the resident image is
    /// gone by acquire time.
    pub seed_skipped: bool,
}

/// Bind request for a sampled input whose window content the engine already
/// holds GPU-resident (a prior dispatch's storage output). The engine copies
/// the resident image into the transient sampled image device-locally instead
/// of uploading `bytes` (which is a zero placeholder and must never reach the
/// GPU): the copy never aliases the live resident, so the same dispatch may
/// also storage-write that identity. A missing/mismatched resident fails
/// visibly with a `vk_compute_exec_resident_sample_*` decline naming the check
/// that refused.
#[derive(Clone, Copy, Debug)]
pub struct ComputeResidentSampleBind {
    pub identity: crate::model::ComputeStorageResidencyKey,
    /// Generation the caller verified against the registry at stage time.
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeStorageResidency {
    pub identity: crate::model::ComputeStorageResidencyKey,
    /// Generation represented by `bytes` before this dispatch.
    pub seed_generation: u32,
    /// Generation guest memory will represent after successful writeback.
    pub output_generation: u32,
}

/// Read-only sampled image for compute. The format set is shared with storage
/// images because both are derived from the same Metal pixel-format contract;
/// descriptor access is carried separately by the request field.
///
/// Single-layer 2D only, for the same reason as
/// [`ComputeStorageImageResource`].
#[derive(Debug)]
pub struct ComputeSampledImageResource {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub format: StorageImageFormat,
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    /// When set, `bytes` is a zero placeholder: the engine seeds the sampled
    /// image with a device-local copy of the named resident storage image
    /// instead of uploading from the host (see [`ComputeResidentSampleBind`]).
    pub resident_bind: Option<ComputeResidentSampleBind>,
}

/// Pixel formats the product compute path maps. Storage and sampled images
/// share this type; access is carried separately by the request resource.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum StorageImageFormat {
    #[default]
    Rgba32Float,
    Rgba16Float,
    R16Float,
    Rgba16Uint,
    Rgba8Uint,
    Rgba8Sint,
    Rgba8Unorm,
    Bgra8Unorm,
    Rg16Float,
    R8Unorm,
    Rg8Unorm,
    Rgba32Uint,
    R32Uint,
    R32Sint,
    R32Float,
    /// Packed three-channel shared-exponent float; sampled-image only on the
    /// product path (`MTLPixelFormatRGB9E5Float`).
    Rgb9e5Ufloat,
    /// Single-channel sixteen-bit normalized; **sampled-image only**, for
    /// `Rgb9e5Ufloat`'s reason and one more.
    ///
    /// This is the ten-bit biplanar video luma plane
    /// (`MTLPixelFormatR16Unorm`). macOS 14 and macOS 15 each bind one to a
    /// `DispatchThreadgroups` and lost the whole dispatch to
    /// `sampled_format_unsupported` until it was named here.
    ///
    /// It must **not** reach a storage bind. Vulkan mandates `R16_UNORM` for
    /// `SAMPLED_IMAGE` and `SAMPLED_IMAGE_FILTER_LINEAR` and does *not* mandate
    /// it for `STORAGE_IMAGE`, so admitting it to a storage image would claim a
    /// capability the host may not have — which is why it is reachable through
    /// [`translate::pixel::sampled_image`] and not through
    /// `translate::pixel::storage_image`.
    R16Unorm,
    /// Two-channel sixteen-bit normalized; **sampled-image only**, and
    /// [`Self::R16Unorm`]'s other half.
    ///
    /// A ten-bit biplanar video texture is two planes, and this is the chroma
    /// one (`MTLPixelFormatRG16Unorm`) to that one's luma. A shader sampling such
    /// a frame binds both planes, so admitting only the luma one still loses the
    /// whole dispatch — the refusal moves to the other binding rather than going
    /// away.
    ///
    /// `STORAGE_IMAGE` is no more mandatory for `R16G16_UNORM` than for
    /// `R16_UNORM`, so it is reachable by the same single route for the same
    /// reason.
    Rg16Unorm,
    /// Four-channel sixteen-bit normalized; **sampled-image only**, the widest
    /// member of the same family.
    ///
    /// `SAMPLED_IMAGE` with `SAMPLED_IMAGE_FILTER_LINEAR` is mandatory for
    /// `R16G16B16A16_UNORM` and `STORAGE_IMAGE` is not, which is the whole of why
    /// it sits here rather than in the storage selector.
    Rgba16Unorm,
    /// Ten bits per colour channel and two of alpha in one packed word, red in
    /// the low bits (`MTLPixelFormatRGB10A2Unorm`); **sampled-image only**.
    ///
    /// Here for [`Self::R16Unorm`]'s reason: Vulkan mandates
    /// `A2B10G10R10_UNORM_PACK32` for `SAMPLED_IMAGE` and
    /// `SAMPLED_IMAGE_FILTER_LINEAR` and mandates nothing for `STORAGE_IMAGE`,
    /// so it is reachable through `translate::pixel::sampled_image` and not
    /// through `translate::pixel::storage_image`.
    Rgb10a2Unorm,
    /// [`Self::Rgb10a2Unorm`] with the colour channels the other way round in
    /// the word (`MTLPixelFormatBGR10A2Unorm`); **sampled-image only**.
    ///
    /// One caveat its two neighbours do not carry:
    /// `A2R10G10B10_UNORM_PACK32` is **not** in Vulkan's mandatory format
    /// table at all, where `A2B10G10R10_UNORM_PACK32` and
    /// `B10G11R11_UFLOAT_PACK32` are. A host without it fails image creation
    /// and declines by name, which is the same work the guest lost when the
    /// format was refused at decode — but a capability gate would say so before
    /// the allocation rather than after it, and that gate is not written.
    Bgr10a2Unorm,
    /// Eleven bits of red and green, ten of blue, no alpha, in one packed word
    /// (`MTLPixelFormatRG11B10Float`); **sampled-image only**, for
    /// [`Self::Rgb10a2Unorm`]'s reason.
    Rg11b10Float,
}

impl StorageImageFormat {
    pub(crate) fn vk_format(self) -> vk::Format {
        translate::pixel::vk_storage_image(self)
    }

    pub fn bytes_per_texel(self) -> usize {
        match self {
            Self::Rgba32Float | Self::Rgba32Uint => 16,
            Self::Rgba16Float | Self::Rgba16Uint => 8,
            Self::Rg16Float => 4,
            Self::Rgba16Unorm => 8,
            Self::Rg16Unorm => 4,
            Self::R16Float | Self::Rg8Unorm | Self::R16Unorm => 2,
            Self::R8Unorm => 1,
            Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba8Unorm
            | Self::Bgra8Unorm
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rgb9e5Ufloat
            | Self::Rgb10a2Unorm
            | Self::Bgr10a2Unorm
            | Self::Rg11b10Float => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw residency (workstream D)
// ---------------------------------------------------------------------------

/// Protocol-derived render-target identity (resource state, not content hash).
///
/// Every field of every variant is a scalar the protocol handed over, so an
/// identity is a *value* and never a handle. That is what lets
/// [`crate::runtime::writeback_debt::WritebackDebt`] hold one without breaking
/// the rule its module doc states — the rail this replaces held resolved host
/// pointers and corrupted the guest's page tables with them. It is `Clone` and
/// not `Copy` only because several hundred call sites spell the clone, and
/// rewriting them would bury whatever change asked for it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TargetIdentity {
    /// Type-4 mapping / surface id namespace.
    Surface {
        id: u32,
        width: u32,
        height: u32,
        generation: u64,
        /// This target's resident image format, from the pixel format the
        /// mapping declares for its own plane.
        ///
        /// A type-11 mapping is not BGRA8 by its contract, which is what this
        /// namespace assumed for as long as it held no format: it declares a
        /// format, `mapping_write` reads that declaration to lay out the
        /// writeback, and macOS 26 declares `MTLPixelFormatRGBA16Float` for
        /// some of its compositing surfaces. Rendering those into a BGRA8
        /// resident quantized the guest's half-float compositing to eight bits
        /// with nothing to say so — the same loss the `Gva` namespace had, for
        /// the same reason, found the same way.
        ///
        /// [`crate::runtime::present_identity::surface_identity`] is the only
        /// producer, and it resolves this through
        /// [`crate::runtime::mapping_write::mapping_store_format`] — the same
        /// function the writeback lays its rows out from, so the resident and
        /// its destination cannot disagree about what the guest asked for.
        format: vk::Format,
    },
    /// Type-2/3 texture ref namespace.
    Texture {
        ref_: u32,
        width: u32,
        height: u32,
        generation: u64,
        /// Whether this resident carries a stencil aspect beside its depth one.
        ///
        /// **Part of the key because it selects the image's format**, and the
        /// registry's reuse test compares formats: a depth texture drawn into
        /// with the stencil test on and then off would otherwise retire and
        /// recreate its resident on every alternation — one allocation per draw
        /// again, and arrived at by a path that looks like reuse. The two are
        /// genuinely different images, so they are two residents, each stable.
        ///
        /// Always `false` for a colour target, which is what every non-depth
        /// constructor of this variant passes.
        stencil: bool,
    },
    /// Guest-VA surface namespace.
    Gva {
        gva: u64,
        width: u32,
        height: u32,
        generation: u64,
        /// This target's resident image format, from the pixel format the guest
        /// declared for the attachment.
        ///
        /// See [`TargetIdentity::is_bgra`] for why it has to be part of the key
        /// rather than a per-draw argument, and why this namespace is the one
        /// that carries it: a surface is BGRA by its own contract and a pooled
        /// target has no declaration to follow, but a GVA render target's
        /// declaration is the whole answer.
        ///
        /// It is a format and not a `bgra: bool` because the guest declares
        /// more than a channel order. A flag can only ever reconstruct
        /// `B8G8R8A8_UNORM` or `R8G8B8A8_UNORM`, so every render target was
        /// eight bits per channel whatever was asked for — the twin, on the
        /// Store side, of the sampled-half-float bug. It also made this key
        /// disagree with the image for a secondary MRT attachment, whose
        /// resident is created from the guest's real format while its identity
        /// claimed `bgra = false`.
        format: vk::Format,
    },
    /// Anonymous / no protocol identity (oracle / one-shot draws).
    Anonymous { slot: u64 },
}

/// What this device last did with a resident it no longer holds.
///
/// A draw that samples a missing resident cannot say, on its own, whether the
/// pixels were taken from under it or never existed: both read as an absent
/// registry entry. Those are different defects with different repairs — one is a
/// reclaim policy that counted an actively-read resident as idle, the other is a
/// target the guest never rendered into — and telling them apart is the whole
/// value of recording this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentReclaim {
    /// An allocation was refused and the reclaim retry gave it back, because it
    /// was neither pinned nor the only copy of its pixels. A terminal destroy of
    /// the image, but not of the pixels — the guest's own pages still hold them,
    /// which is the predicate `ResourcePools::recoverable_residents` selects on.
    AllocationReclaimed,
    /// `registry_ensure` replaced it for the same identity at a new geometry,
    /// generation or format.
    Recreated,
    /// The serialized resource that owned this resident was explicitly deleted
    /// or replaced. The guest ended the resource lifetime, so the host object no
    /// longer participates in allocation-pressure recovery.
    ResourceReleased,
}

impl ResidentReclaim {
    pub fn slug(self) -> &'static str {
        match self {
            Self::AllocationReclaimed => "allocation_reclaimed",
            Self::Recreated => "recreated",
            Self::ResourceReleased => "resource_released",
        }
    }
}

pub type PresentRect = (u32, u32, u32, u32);

/// The resident a host-window present should blit from.
///
/// One identity, not a list. The display transaction names exactly one surface,
/// and `present_identity::surface_identity` turns that name into exactly one
/// identity — so there was never a second candidate to rank against the first.
/// It stays a request rather than a resolved slot because only the engine, under
/// its own lock, can say whether that identity is resident and presentable at
/// `width`x`height`.
#[derive(Clone, Debug)]
pub struct WindowPresentSource {
    pub width: u32,
    pub height: u32,
    pub identity: TargetIdentity,
}

impl Default for TargetIdentity {
    fn default() -> Self {
        Self::Anonymous { slot: 0 }
    }
}

/// Why a registry lookup missed, given the closest key the registry does hold.
///
/// A miss is not one finding. The registry is keyed by whole
/// [`TargetIdentity`], so every field of it can be the reason, and each has a
/// different repair: a namespace difference is two producers disagreeing about
/// which object this is, a geometry difference is a surface that resized under
/// a caller holding the old extent, a generation difference is a key that moved
/// between the draw and the reader, and `Other` is a format. Reporting the miss
/// without saying which sent one session hunting a stale generation that was
/// the minority case.
///
/// Ordered from coarsest to finest, and answered as the *first* difference
/// rather than the only one — the same rule
/// [`super::pools::PassEchoField`] states for its own ladder, and for the same
/// reason: two identities in different namespaces are not about one object, so
/// nothing finer about them is worth reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKeyDivergence {
    /// Nothing in the registry names this object at all.
    Absent,
    /// The registry holds this id in a different namespace — a mapping id
    /// against a texture ref, a GVA against a surface.
    Namespace,
    /// Same object, different extent.
    Geometry,
    /// Same object and extent, and the key moved.
    Generation,
    /// Same object and extent, and re-generating still does not match. The only
    /// field left is the format, and a new field would land here too rather
    /// than be misreported as one of the above.
    Other,
}

impl TargetKeyDivergence {
    /// The name this goes on the fail line as.
    pub fn label(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Namespace => "namespace",
            Self::Geometry => "geometry",
            Self::Generation => "generation",
            Self::Other => "other",
        }
    }
}

impl TargetIdentity {
    pub fn width(&self) -> u32 {
        match self {
            Self::Surface { width, .. } | Self::Texture { width, .. } | Self::Gva { width, .. } => {
                *width
            }
            Self::Anonymous { .. } => 0,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Surface { height, .. }
            | Self::Texture { height, .. }
            | Self::Gva { height, .. } => *height,
            Self::Anonymous { .. } => 0,
        }
    }

    pub fn generation(&self) -> u64 {
        match self {
            Self::Surface { generation, .. }
            | Self::Texture { generation, .. }
            | Self::Gva { generation, .. } => *generation,
            Self::Anonymous { .. } => 0,
        }
    }

    /// Which namespace this identity is in, and what names it there.
    ///
    /// Two identities with the same answer are about the same guest object; two
    /// with different answers cannot be, whatever else they agree on. That is
    /// what splits "the registry holds nothing for this key" into "it holds
    /// nothing for this object" and "it holds this object under a key differing
    /// in geometry, format or generation" — see [`TargetKeyDivergence`].
    ///
    /// The discriminant is folded in rather than returned beside the value: a
    /// mapping id 7 and a texture ref 7 are different objects, and a bare `u64`
    /// would call them one.
    pub fn namespaced_id(&self) -> (u8, u64) {
        match self {
            Self::Surface { id, .. } => (0, u64::from(*id)),
            Self::Texture { ref_, .. } => (1, u64::from(*ref_)),
            Self::Gva { gva, .. } => (2, *gva),
            Self::Anonymous { slot } => (3, *slot),
        }
    }

    /// How `held` differs from this identity, for a registry lookup that missed.
    pub fn diverges_from(&self, held: &Self) -> TargetKeyDivergence {
        if self.namespaced_id() != held.namespaced_id() {
            return TargetKeyDivergence::Namespace;
        }
        if (self.width(), self.height()) != (held.width(), held.height()) {
            return TargetKeyDivergence::Geometry;
        }
        // Whatever is left is spared by re-generation or it is not. Asked with
        // `PartialEq` so a field this enum gains lands in `Other` rather than
        // being reported as a generation difference it is not.
        if self.with_generation(held.generation()) == *held {
            return TargetKeyDivergence::Generation;
        }
        TargetKeyDivergence::Other
    }

    /// The same target named at a different generation.
    ///
    /// Exists so that "is this the same surface under a newer key?" can be asked
    /// with `PartialEq` rather than by a hand-written field-by-field comparison:
    /// `a.with_generation(b.generation()) == *b` is total over every field this
    /// enum has now and every one it gains, where a comparison spelling out the
    /// fields it cares about goes stale the moment one is added. `Anonymous`
    /// carries no generation, so it is returned unchanged and compares as
    /// itself.
    pub fn with_generation(&self, generation: u64) -> Self {
        let mut next = self.clone();
        match &mut next {
            Self::Surface { generation: g, .. }
            | Self::Texture { generation: g, .. }
            | Self::Gva { generation: g, .. } => *g = generation,
            Self::Anonymous { .. } => {}
        }
        next
    }

    /// Physical channel order of the resident image behind this identity.
    ///
    /// The rule is one sentence: **a resident holds the bytes its destination
    /// stores.** Rendering it that way makes a raw image→buffer copy land the
    /// frame in guest memory unchanged, which is what deletes the whole-frame
    /// CPU swizzle and the blocking readback in front of it.
    ///
    /// Each namespace answers it from what it knows:
    ///
    /// * `Surface` backs a type-11 guest IOSurface, whose plane carries a
    ///   declared pixel format exactly as a GVA target does — usually guest
    ///   scanout order, and not always.
    /// * `Gva` is a render target the guest declared a pixel format for, and
    ///   that declaration is the answer — carried in the key as a whole
    ///   [`vk::Format`], not just its order. Two allocations at one address
    ///   declaring different formats are two keys and therefore two slots,
    ///   which is what stops them recreating one image between them.
    /// * `Texture` and `Anonymous` have no destination to follow — nothing
    ///   copies them out to guest memory byte-for-byte — so they stay RGBA.
    ///
    /// This is a property of the *identity*, not of the draw, and that is the
    /// whole point: `ResourcePools::registry` is keyed by identity and
    /// `registry_ensure` destroys and recreates the image whenever a draw's
    /// requested order disagrees with the slot's. Several runtime paths render
    /// into one identity in a frame — a composite Store, a chain intermediate,
    /// an MRT primary — and deriving the order from the key they already agree
    /// on is what makes them agree here too. A per-path predicate would let one
    /// of them recreate the image every frame, which reads as `target_evicts`
    /// climbing and costs a fresh allocation plus a lost `content_ready` per
    /// composite.
    ///
    /// Nothing downstream of here assumes either order: the seed upload folds
    /// an exchange into the staging copy when the seed and the attachment
    /// disagree, and every readback reports the order it copied. The identity
    /// is the only place the answer was pinned to a namespace.
    pub fn is_bgra(&self) -> bool {
        translate::pixel::has_bgra_order(self.resident_format())
    }

    /// Whether these two identities name the same destination, whatever format
    /// each declares for it.
    ///
    /// Not `==`. Equality is the *registry* question — do these share one image
    /// — and the format belongs in it, because two formats at one address are
    /// two images. This is the *conflict* question, asked of two attachments of
    /// one render pass, and there the answer must ignore the format: a pass with
    /// two colour attachments over one guest span writes that span twice, and
    /// which of the two lands is whichever Store runs last.
    ///
    /// The distinction only appeared once the key could hold a format. While it
    /// held a `bgra: bool`, `==` answered this by accident for every pair that
    /// shared an order, and the two questions were indistinguishable.
    pub fn aliases(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Surface { id: a, .. }, Self::Surface { id: b, .. }) => a == b,
            (Self::Gva { gva: a, .. }, Self::Gva { gva: b, .. }) => a == b,
            (Self::Texture { ref_: a, .. }, Self::Texture { ref_: b, .. }) => a == b,
            (Self::Anonymous { slot: a }, Self::Anonymous { slot: b }) => a == b,
            _ => false,
        }
    }

    /// The format of the resident image behind this identity — the answer
    /// `registry_ensure` creates the image with and the render pass is built
    /// against.
    ///
    /// [`Self::is_bgra`] is now a question *about* this rather than the thing
    /// the key stores, because a channel order cannot express how wide a
    /// channel is. The two namespaces the guest declares a format for —
    /// `Surface` and `Gva` — answer with that declaration; `Texture` and
    /// `Anonymous` have none to follow and answer with the constant they
    /// always did.
    ///
    /// Whoever reads this to size a buffer must go through
    /// [`translate::pixel::bytes_per_texel`] rather than assuming four. That
    /// assumption is exactly what made a wider format unrepresentable.
    pub fn resident_format(&self) -> vk::Format {
        match self {
            Self::Surface { format, .. } => *format,
            Self::Gva { format, .. } => *format,
            Self::Texture { .. } | Self::Anonymous { .. } => translate::pixel::RESIDENT_RGBA_FORMAT,
        }
    }
}

/// Byte order of a CPU load seed, relative to the attachment it seeds.
///
/// Vulkan buffer→image copies perform no format conversion, so the staged bytes
/// must already be in the attachment's physical order. Stating the seed's own
/// order — rather than assuming one — lets the exchange fold into the copy into
/// the mapped staging span instead of being paid as a separate converted frame:
/// `surface_cache` holds guest scanout order and the pooled target is RGBA, so
/// the runtime used to allocate, copy and swizzle a whole framebuffer per seeded
/// draw purely to restate the pixels it already had.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum SeedOrder {
    /// Semantic RGBA8 — R, G, B, A in memory.
    #[default]
    Rgba8,
    /// Guest scanout order — B, G, R, A in memory.
    Bgra8,
}

/// Where a sampled image's content comes from.
#[derive(Debug)]
pub enum SampledSource {
    /// CPU origin (bytes re-staged each draw unless warm-path caches geometry only).
    Bytes(std::sync::Arc<Vec<u8>>),
    /// Bind a prior GPU-resident target directly (no CPU round-trip).
    Target(TargetIdentity),
    /// Guest-memory origin. A resource-owned packed allocation binds as a
    /// linear sampled image directly; hosts or layouts that decline it retain
    /// the copy-backed route from imported buffers into an optimal image. No
    /// CPU read or hash is required on either GPU route.
    ///
    /// A copy is elided where a retained image already answers to the bind's
    /// identity, which is what [`crate::runtime::gather_witness::GatherVouch`]
    /// says is possible. Resource-owned direct images carry no copied-content
    /// identity. If the backend declines one, the copy fallback therefore runs
    /// conservatively instead of reusing content that was never witnessed.
    GuestRuns(GuestRunSource, crate::runtime::gather_witness::GatherVouch),
}

/// One packed-contiguous guest-RAM span (a direct RAMBlock alias from
/// `HostOps::map_pages`; stable for the VM lifetime, unmap is a no-op).
#[derive(Clone, Copy, Debug)]
pub struct GuestRun {
    /// Host VA of the span start (page-aligned base + in-page offset).
    pub host_ptr: usize,
    /// Byte length of the span.
    pub len: u64,
}

/// Guest-RAM texel source: the requested window is
/// `source_offset..source_offset + total_len` inside `runs`. With
/// `row_length_texels == 0` the window is
/// tight (`total_len == tight_row_bytes * height`); a nonzero value gives
/// the guest row stride in texels for padded layouts, and the window then
/// spans `(height-1) * stride_bytes + tight_row_bytes` (the final row needs
/// only its texels — padding past the last row may not be mapped). Every run's
/// [`GuestRun::host_ptr`]`..+`[`len`](GuestRun::len) must already be a live
/// `HostOps::map_pages` alias when the source is built: the gather reads it
/// directly and has nothing to check it against.
#[derive(Clone, Debug)]
pub struct GuestRunSource {
    pub runs: std::sync::Arc<Vec<GuestRun>>,
    /// First byte of the requested window inside `runs` and `pages`.
    ///
    /// Normally zero. Task buffers reconstructed as one stable allocation keep
    /// one source per resource and vary this offset at bind time, just as the
    /// guest command carries one buffer reference plus an offset.
    pub source_offset: u64,
    pub total_len: u64,
    /// Guest row stride in texels for the buffer→image copy
    /// (`bufferRowLength`); 0 = tight rows.
    pub row_length_texels: u32,
    /// The same bytes [`Self::runs`] cover, as bounded references into this
    /// process's import of the RAMBlock behind them — one per maximal
    /// GPA-contiguous stretch, ascending, tiling the window exactly.
    ///
    /// Separate from [`GuestRun`] because a run is a *host-pointer* span the CPU
    /// gather walks, while these are offsets the GPU binds or copies from.
    /// Keeping both lets one source feed either without reconstructing the
    /// other's view.
    ///
    /// # Why a list and not one reference
    ///
    /// It was one, and a driven boot found the consequence: the guest backs a
    /// surface in 16 KiB physically-contiguous granules, so a draw-time buffer
    /// window is 9-32 stretches 98.5 % of the time and **never** one. A single
    /// reference could therefore only ever be `None`, and every bind on a host
    /// whose `vk_caps` said `host_pointer_import=supported` still fell to the
    /// CPU gather — 371 422 of them against 0 imports. A one-element list is
    /// still the direct bind, and a longer one is a GPU copy per stretch, which
    /// is what [`crate::backend::vulkan::engine::exec`] does with it.
    ///
    /// `None` is the honest answer for a synthetic source — a test fixture over
    /// a host `Vec` has no guest pages — and for a host that cannot import at
    /// all. The CPU gather path needs only [`GuestRun::host_ptr`] and is
    /// unaffected either way.
    ///
    /// `Arc` because a source is cloned per bind and these are shared, immutable
    /// and never rebuilt.
    pub pages: Option<std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>>,
    /// One resource-owned packed allocation that can back a linear sampled
    /// image directly. When the host declines that image layout, `runs` and
    /// `pages` remain the complete copy-backed fallback for the same texels.
    pub direct_image: Option<GuestSampledBacking>,
}

/// One stretch of a [`GuestRunSource`]'s window, already clipped to it.
///
/// `skip` is the distance from the stretch's own first requested byte to the
/// first byte of the window that lands in it, and `window_offset` is where those
/// bytes belong in the assembled window. Neither is the number nearest to hand:
/// a [`crate::runtime::guest_ram_map::GuestWindowRun`] is positioned against the
/// whole allocation its `pages` list describes, while the window is
/// `source_offset..source_offset + total_len` inside that.
#[derive(Debug)]
pub struct WindowStretch<'a> {
    pub guest: &'a crate::runtime::guest_ram::GuestRef,
    pub skip: u64,
    pub window_offset: u64,
    pub len: u64,
}

impl GuestRunSource {
    /// This source's window as the single guest stretch holding it, when it is
    /// one — the arm that binds the import in place with nothing copied.
    ///
    /// A single run starting at allocation byte zero *is* the whole allocation:
    /// [`crate::runtime::guest_ram_map::references_for_runs`] guarantees the runs
    /// ascend and tile it exactly, so one of them covering byte zero leaves
    /// nothing else to name. Anything longer has to be gathered, because a
    /// vertex, index, storage or copy source names one contiguous range.
    ///
    /// The window still need not start at that stretch's first byte: a mapped
    /// sampled plane names the whole allocation as its one stretch and puts the
    /// plane's own offset in `source_offset`, which is what [`WindowStretch::skip`]
    /// carries. `None` when the window is scattered, or when it does not fit
    /// inside the one stretch named, which is a malformed source rather than a
    /// slow one.
    pub fn single_stretch(&self) -> Option<WindowStretch<'_>> {
        let [only] = self.pages.as_ref()?.as_slice() else {
            return None;
        };
        if only.window_offset != 0 {
            return None;
        }
        let end = self.source_offset.checked_add(self.total_len)?;
        if end > only.guest.requested() {
            return None;
        }
        Some(WindowStretch {
            guest: &only.guest,
            skip: self.source_offset,
            window_offset: 0,
            len: self.total_len,
        })
    }

    /// Every stretch this source's window touches, in window order, each
    /// clipped to the window. Stretches the window does not reach are absent
    /// rather than empty, so the lengths sum to [`Self::total_len`] exactly.
    pub fn window_stretches(&self) -> Option<impl Iterator<Item = WindowStretch<'_>> + '_> {
        let pages = self.pages.as_ref()?;
        let wanted_end = self.source_offset.checked_add(self.total_len)?;
        Some(pages.iter().filter_map(move |run| {
            let run_end = run.window_offset.checked_add(run.guest.requested())?;
            let start = run.window_offset.max(self.source_offset);
            let end = run_end.min(wanted_end);
            if start >= end {
                return None;
            }
            Some(WindowStretch {
                guest: &run.guest,
                skip: start - run.window_offset,
                window_offset: start - self.source_offset,
                len: end - start,
            })
        }))
    }
}

/// A render attachment's prior contents, read from the surface's own guest
/// pages rather than materialized as a host framebuffer.
///
/// `source` carries both representations of the same window: bounded RAMBlock
/// references for the native import rail and stable host aliases for its exact
/// CPU fallback. `format` is the guest plane's physical texel layout; a raw
/// buffer→image copy performs no conversion, so validation requires it to equal
/// the attachment format before either representation may be used.
#[derive(Clone, Debug)]
pub struct GuestTargetSeed {
    pub source: GuestRunSource,
    pub format: ash::vk::Format,
}

/// One guest surface plane within a stable shared host allocation.
///
/// `allocation_host_ptr..allocation_len` is the object imported into Vulkan.
/// `plane_offset` identifies the attachment's first texel within it, while
/// `row_pitch` is the plane's declared physical stride. Keeping the whole
/// allocation and the plane coordinates together is what lets the engine
/// derive `vkBindImageMemory`'s offset without manufacturing a pointer before
/// the plane or extending the import past its real bound.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestTargetBacking {
    pub allocation_host_ptr: usize,
    pub allocation_len: u64,
    pub plane_offset: u64,
    pub row_pitch: u64,
}

/// A sampled plane within the packed allocation retained for its guest
/// resource. The import owns the checked allocation bound; `backing` carries
/// only the image-layout coordinates derived inside that bound.
#[derive(Clone, Debug)]
pub struct GuestSampledBacking {
    pub backing: GuestTargetBacking,
    pub import: std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>,
    /// The serialized resource that owns this image. The engine keeps this
    /// weak so its cache cannot extend the guest-visible resource lifetime.
    pub owner: crate::model::TaskResourceLifetimeRef,
    /// Resource family for accounting only; never an execution selector.
    pub origin: SampledByteOrigin,
}

/// An importable guest allocation and the physical pages it owns.
///
/// Keeping these together makes the retained resource its own synchronization
/// authority: once admitted, the engine can publish the exact footprint that
/// was validated with the allocation instead of reconstructing it at Store.
#[derive(Clone, Debug)]
pub struct GuestTargetMemory {
    pub backing: GuestTargetBacking,
    /// The parent allocation whose one backend import all child views share.
    pub import: std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>,
    pub footprint: crate::runtime::guest_ram::GuestPageFootprint,
}

/// Producer-assigned identity + generation for CPU-sourced sampled content.
///
/// When two draws bind [`SampledSource::Bytes`] with the same identity, the
/// content is byte-identical by the producer's coherence model (the runtime
/// bumps `generation` whenever its authoritative cache entry is rewritten),
/// so the sampled cache may bind the retained GPU image without re-hashing
/// or comparing the bytes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SampledContentIdentity {
    /// Stable key of the guest resource (runtime-chosen keyspace).
    pub key: u64,
    /// Content generation of the producer's authoritative cache entry.
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A draw that samples one of its own attachments must reach the snapshot
    /// arm, and "its own" is every attachment it binds rather than slot 0.
    ///
    /// The secondary and depth cases below are the ones that fail against the
    /// primary-only test this replaced, which is what makes them worth writing:
    /// each was a live attachment feedback loop handed to the driver.
    #[test]
    fn a_draw_samples_its_own_attachment_on_every_slot_that_can_carry_one() {
        let surface = |id: u32| TargetIdentity::Surface {
            id,
            width: 64,
            height: 64,
            generation: 0,
            format: vk::Format::B8G8R8A8_UNORM,
        };

        let mut req = DrawRequest {
            target_identity: Some(surface(1)),
            ..DrawRequest::default()
        };
        assert!(req.writes_attachment(&surface(1)), "primary colour");
        assert!(
            !req.writes_attachment(&surface(9)),
            "a target this draw does not bind is not a feedback loop, and \
             routing it through the snapshot would cost a copy per draw"
        );

        req.secondary_targets.push(SecondaryColorTarget {
            identity: surface(2),
            width: 64,
            height: 64,
            format: vk::Format::B8G8R8A8_UNORM,
            clear: [0.0; 4],
            load: crate::contract::pass_action::LoadAction::Clear,
            blend: None,
            color_write_mask: ColorWriteMask::default(),
        });
        assert!(req.writes_attachment(&surface(2)), "MRT secondary");
        assert_eq!(
            req.attachment_slot(&surface(2)),
            Some(AttachmentSlot::Secondary),
            "the census has to be able to say which slot matched"
        );
        assert_eq!(req.color_attachment_index(&surface(1)), Some(0));
        assert_eq!(req.color_attachment_index(&surface(2)), Some(1));
        assert_eq!(
            req.attachment_slot(&surface(1)),
            Some(AttachmentSlot::Primary)
        );

        req.depth = Some(DepthState {
            identity: Some(surface(3)),
            test_enable: true,
            write_enable: true,
            compare: SamplerCompareFunction::Less,
            clear_value: 1.0,
            load: false,
            stencil: None,
        });
        assert!(req.writes_attachment(&surface(3)), "depth");
        assert_eq!(
            req.attachment_slot(&surface(3)),
            Some(AttachmentSlot::Depth)
        );
        assert_eq!(req.attachment_slot(&surface(9)), None);
        // Three distinct routes, so a census reading one of them cannot be a
        // different slot's population.
        let routes = [
            AttachmentSlot::Primary,
            AttachmentSlot::Secondary,
            AttachmentSlot::Depth,
        ]
        .map(AttachmentSlot::sampled_self_route);
        assert_eq!(
            routes
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );

        // The generation is part of the identity, so a resident the guest has
        // since rewritten is a different target and not this draw's attachment.
        assert!(!req.writes_attachment(&TargetIdentity::Surface {
            id: 1,
            width: 64,
            height: 64,
            generation: 1,
            format: vk::Format::B8G8R8A8_UNORM,
        }));
    }

    #[test]
    fn indexed_draw_widths_are_the_fixed_function_element_widths() {
        assert_eq!(IndexType::U16.byte_size(), 2);
        assert_eq!(IndexType::U32.byte_size(), 4);
    }

    #[test]
    fn sampler_cache_state_excludes_binding_but_preserves_sampler_state() {
        let first = SamplerResource::normalized_default(3);
        let mut rebound = SamplerResource::normalized_default(27);
        assert_eq!(first.state_key(), rebound.state_key());
        assert_eq!(first.lod_min_f32(), 0.0);
        assert_eq!(first.lod_max_f32(), f32::MAX);

        rebound.address_mode_v = SamplerAddressMode::Repeat;
        assert_ne!(first.state_key(), rebound.state_key());
    }

    #[test]
    fn target_identity_accessors_never_infer_anonymous_geometry() {
        let surface = TargetIdentity::Surface {
            id: 7,
            width: 1920,
            height: 1080,
            generation: 4,
            format: translate::pixel::SCANOUT_FORMAT,
        };
        assert_eq!(
            (surface.width(), surface.height(), surface.generation()),
            (1920, 1080, 4)
        );
        let anonymous = TargetIdentity::Anonymous { slot: 99 };
        assert_eq!(
            (
                anonymous.width(),
                anonymous.height(),
                anonymous.generation()
            ),
            (0, 0, 0)
        );
        assert_eq!(
            TargetIdentity::default(),
            TargetIdentity::Anonymous { slot: 0 }
        );
    }

    /// Re-generation changes the generation and nothing else, on every variant
    /// that has one — which is what lets "is this the same target under a newer
    /// key?" be asked with `PartialEq` instead of a field-by-field comparison
    /// that a new field would silently fall out of.
    #[test]
    fn re_generation_moves_only_the_generation() {
        let all = [
            TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 1080,
                generation: 4,
                format: translate::pixel::SCANOUT_FORMAT,
            },
            TargetIdentity::Texture {
                ref_: 12,
                width: 64,
                height: 64,
                generation: 4,
                stencil: true,
            },
            TargetIdentity::Gva {
                gva: 0xdead_0000,
                width: 8,
                height: 8,
                generation: 4,
                format: translate::pixel::SCANOUT_FORMAT,
            },
        ];
        for identity in &all {
            let moved = identity.with_generation(9);
            assert_eq!(moved.generation(), 9, "{identity:?}");
            assert_ne!(&moved, identity, "{identity:?}");
            // The round trip is the whole claim: everything but the generation
            // survived, so equality after restoring it is field-complete.
            assert_eq!(&moved.with_generation(identity.generation()), identity);
        }
        // `Anonymous` carries no generation, so it is returned as itself rather
        // than being given one it has nowhere to keep.
        let anonymous = TargetIdentity::Anonymous { slot: 99 };
        assert_eq!(anonymous.with_generation(9), anonymous);
    }

    /// The four ways a registry key can miss are told apart, and the ladder is
    /// answered coarsest-first: two identities in different namespaces are not
    /// about one object, so nothing finer about them is reported. A miss that
    /// named none of these sent one session hunting the generation case, which
    /// turned out to be the minority.
    #[test]
    fn a_registry_miss_names_which_field_moved() {
        let asked = TargetIdentity::Surface {
            id: 7,
            width: 1920,
            height: 1080,
            generation: 2,
            format: translate::pixel::SCANOUT_FORMAT,
        };
        assert_eq!(
            asked.diverges_from(&asked.with_generation(1)),
            TargetKeyDivergence::Generation
        );
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 900,
                generation: 2,
                format: translate::pixel::SCANOUT_FORMAT,
            }),
            TargetKeyDivergence::Geometry
        );
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Texture {
                ref_: 7,
                width: 1920,
                height: 1080,
                generation: 2,
                stencil: false,
            }),
            TargetKeyDivergence::Namespace
        );
        // A format change is what is left once the object, the extent and the
        // generation all agree — and so is any field this enum gains, which is
        // the point of asking the last question with `PartialEq`.
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Surface {
                id: 7,
                width: 1920,
                height: 1080,
                generation: 2,
                format: vk::Format::R16G16B16A16_SFLOAT,
            }),
            TargetKeyDivergence::Other
        );
        // Namespace outranks everything: a texture ref that happens to equal a
        // mapping id must not be reported as a resize of it.
        assert_eq!(
            asked.diverges_from(&TargetIdentity::Texture {
                ref_: 7,
                width: 8,
                height: 8,
                generation: 99,
                stencil: false,
            }),
            TargetKeyDivergence::Namespace
        );
    }

    #[test]
    fn storage_format_texel_sizes_cover_every_format_variant() {
        let cases = [
            (StorageImageFormat::Rgba32Float, 16),
            (StorageImageFormat::Rgba16Float, 8),
            (StorageImageFormat::R16Float, 2),
            (StorageImageFormat::Rgba16Uint, 8),
            (StorageImageFormat::Rgba8Uint, 4),
            (StorageImageFormat::Rgba8Sint, 4),
            (StorageImageFormat::Rgba8Unorm, 4),
            (StorageImageFormat::Bgra8Unorm, 4),
            (StorageImageFormat::Rg16Float, 4),
            (StorageImageFormat::R8Unorm, 1),
            (StorageImageFormat::Rg8Unorm, 2),
            (StorageImageFormat::Rgba32Uint, 16),
            (StorageImageFormat::R32Uint, 4),
            (StorageImageFormat::R32Sint, 4),
            (StorageImageFormat::R32Float, 4),
            (StorageImageFormat::Rgb9e5Ufloat, 4),
        ];
        for (format, expected) in cases {
            assert_eq!(format.bytes_per_texel(), expected);
        }
    }

    #[test]
    fn byte_buffer_content_reports_and_borrows_its_exact_payload() {
        let content = BufferContent::from(vec![1, 2, 3, 4]);
        assert_eq!(content.len(), 4);
        assert!(!content.is_empty());
        assert_eq!(content.cpu_bytes().as_ref(), &[1, 2, 3, 4]);
        assert!(BufferContent::from(Vec::new()).is_empty());
    }

    #[test]
    fn default_requests_keep_optional_product_paths_disabled() {
        let draw = DrawRequest::default();
        assert_eq!((draw.width, draw.height, draw.vertex_count), (0, 0, 0));
        assert_eq!(draw.primitive_topology, PrimitiveTopology::Triangle);
        assert_eq!(draw.cull_mode, CullMode::None);
        assert!(draw.target_identity.is_none());
        assert!(draw.depth.is_none());
        assert!(!draw.skip_readback);
        assert!(!draw.color_input);

        let compute = ComputeRequest::default();
        assert_eq!(compute.grid, [0, 0, 0]);
        assert!(compute.storage_buffers.is_empty());
        assert!(compute.storage_images.is_empty());
    }

    /// The order is a property of the identity, and the three answers matter for
    /// different reasons.
    ///
    /// `Surface` answers from the format its mapping declared, and one
    /// constructed at the scanout format reports BGRA: every CPU consumer of a
    /// type-11 composite Store is declared in guest scanout order, so an RGBA
    /// resident under a scanout-declared mapping costs a whole-frame exchange
    /// per Store.
    ///
    /// `Gva` must answer from its own field and from nothing else. That is the
    /// half a future edit is likely to get wrong in either direction — pinning
    /// it to `false` sends every BGRA-declared render target back through the
    /// blocking readback, and pinning it to `true` silently exchanges R and B
    /// on every RGBA-declared one.
    ///
    /// `Texture` and `Anonymous` must not be, and `Anonymous` in particular is
    /// the pooled path the parity suite uses as its semantic control.
    #[test]
    fn a_targets_order_follows_its_own_namespace() {
        assert!(TargetIdentity::Surface {
            id: 1,
            width: 8,
            height: 8,
            generation: 0,
            format: translate::pixel::SCANOUT_FORMAT,
        }
        .is_bgra());
        for (format, bgra) in [
            (translate::pixel::RESIDENT_RGBA_FORMAT, false),
            (translate::pixel::SCANOUT_FORMAT, true),
            (ash::vk::Format::R8G8B8A8_SRGB, false),
            (ash::vk::Format::B8G8R8A8_SRGB, true),
        ] {
            let gva = TargetIdentity::Gva {
                gva: 0x1000,
                width: 8,
                height: 8,
                generation: 0,
                format,
            };
            assert_eq!(gva.resident_format(), format, "{gva:?} must answer its key");
            assert_eq!(gva.is_bgra(), bgra, "{gva:?} must answer from its key");
        }
        for other in [
            TargetIdentity::Texture {
                ref_: 2,
                width: 8,
                height: 8,
                generation: 0,
                stencil: false,
            },
            TargetIdentity::Anonymous { slot: 0 },
        ] {
            assert!(!other.is_bgra(), "{other:?} must stay semantic RGBA");
        }
    }

    /// Two allocations at one address declaring different formats are two keys.
    ///
    /// The format has to be *in* the key, not beside it. If it were not, both
    /// would hash to one registry slot whose image can only be built one way,
    /// and `registry_ensure` answers a requested format that disagrees with the
    /// slot's by destroying and recreating the image — every frame, for as long
    /// as both keep drawing.
    ///
    /// The third format here is the point. `R16G16B16A16_SFLOAT` and
    /// `R8G8B8A8_UNORM` are the **same channel order** and different images, so
    /// while this key held a `bgra: bool` they were one entry — and the wider
    /// one could not be asked for at all, which is why nothing noticed. A key
    /// that separates the two orders but not those two formats passes the first
    /// assertion here and fails the second.
    #[test]
    fn a_gva_targets_format_separates_it_from_the_same_address_in_another_format() {
        let at = |format| TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format,
        };
        let rgba8 = at(translate::pixel::RESIDENT_RGBA_FORMAT);
        let bgra8 = at(translate::pixel::SCANOUT_FORMAT);
        let rgba16f = at(vk::Format::R16G16B16A16_SFLOAT);
        assert_ne!(rgba8, bgra8);
        assert_ne!(
            rgba8, rgba16f,
            "two widths of one channel order are two residents"
        );
        let mut seen = std::collections::HashSet::new();
        for (id, what) in [(bgra8, "bgra8"), (rgba8, "rgba8"), (rgba16f, "rgba16f")] {
            assert!(
                seen.insert(id),
                "{what} must not collide in the registry's key space"
            );
        }
    }

    /// The registry question and the conflict question must answer differently,
    /// and only one of them may look at the format.
    ///
    /// Two colour attachments of one pass over one guest span write that span
    /// twice whatever format each declares, so the MRT alias check has to refuse
    /// the pair — while the registry has to keep them apart, because they are
    /// two images. `==` cannot serve both: it either has the format and misses
    /// the conflict, or lacks it and merges two images into one slot.
    ///
    /// This is the pair the old `bgra: bool` key could not express. Both of
    /// these are RGBA-ordered, so it answered `==` for them and the alias check
    /// fired by accident.
    #[test]
    fn one_span_at_two_formats_is_two_registry_keys_and_still_one_conflict() {
        let at = |format| TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format,
        };
        let rgba8 = at(translate::pixel::RESIDENT_RGBA_FORMAT);
        let rgba16f = at(vk::Format::R16G16B16A16_SFLOAT);
        assert_ne!(rgba8, rgba16f, "two images, so two registry slots");
        assert!(
            rgba8.aliases(&rgba16f),
            "one guest span, so one destination and a refused pass"
        );

        // A different span is neither, and the two namespaces never alias each
        // other however their numbers line up.
        let elsewhere = TargetIdentity::Gva {
            gva: 0x5000,
            width: 64,
            height: 64,
            generation: 7,
            format: translate::pixel::RESIDENT_RGBA_FORMAT,
        };
        assert!(!rgba8.aliases(&elsewhere));
        assert!(!rgba8.aliases(&TargetIdentity::Surface {
            id: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            format: translate::pixel::SCANOUT_FORMAT,
        }));
    }
}
