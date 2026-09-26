//! Draw request surface for the internal Vulkan engine (v1 §1.2 surface).
//!
//! Field meanings match the historical Metal→Vulkan product draw seam
//! (blend, Load seed, stage-in attributes, SSBOs, sampled images).

use ash::vk;

use crate::backend::vulkan::translate;
pub use crate::runtime::decode::resource::ColorWriteMask;
use reims_vgpu_core::sampler;

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

    /// Delegated arm for arm with `slug`.
    ///
    /// A delegating decline that kept the default owner would claim every
    /// inner type's slug under its own name, and
    /// [`crate::observe::slugs`] would report a collision on a wrapper that
    /// shares no check with anything. The claim has to name the type that
    /// spelled the slug, not the one that passed it on.
    fn owner(&self) -> &'static str {
        match self {
            Self::TargetRead(d) => d.owner(),
            Self::GuestPageWrite(d) => d.owner(),
            Self::Unsupported(r) => r.owner(),
            Self::VkCall(c) => c.owner(),
            Self::Slab(d) => d.owner(),
            Self::FenceTimeout => std::any::type_name::<Self>(),
            Self::Init(d) => d.owner(),
            Self::Facade(d) => d.owner(),
            Self::DrawPreparation(d) => d.owner(),
            Self::DrawValidation(d) => d.owner(),
            Self::DrawExecution(d) => d.owner(),
            Self::ComputeValidation(d) => d.owner(),
            Self::ComputeExecution(d) => d.owner(),
            Self::DeviceLost(d) => d.owner(),
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
    /// [`crate::runtime::render_pass::DepthAttachment::texture_ref`] is its
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
    ///
    /// **Not a pipeline dimension.** Metal has no depth-test enable, so a draw
    /// that reaches the engine with this clear is exactly `compare = Always,
    /// write = false`, which tests nothing and writes nothing under either
    /// spelling — see `super::exec::depth_stencil_state`. It used to be a term
    /// of the pipeline key, where it produced two byte-identical pipelines for
    /// one state.
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
    /// The `MTLStencilOperation` ordinal this was decoded from.
    ///
    /// Declaration order is the ABI order, asserted against the wire in
    /// `translate::raster`'s own tests, so the discriminant is the ordinal.
    /// The Vulkan spelling is not here: `reims_vgpu_vulkan::depth_stencil`
    /// owns it, and a second copy of a total mapping is a second thing that
    /// can come to disagree with the first.
    pub(crate) fn mtl_ordinal(self) -> u32 {
        self as u32
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

/// A colour-attachment clear after the attachment's numeric type is known.
///
/// Vulkan represents these three cases as members of a union. Keeping a bare
/// `[f32; 4]` in [`DrawRequest`] lets an integer attachment reach the float
/// member and turns the float's bits into an integer value. This enum makes
/// that mismatch unrepresentable at the engine boundary: translation chooses
/// the variant from the decoded pixel format, and execution only lowers it to
/// the corresponding Vulkan union member.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorClearValue {
    Float([f32; 4]),
    Uint([u32; 4]),
    Sint([i32; 4]),
}

impl ColorClearValue {
    /// Convert Metal's double-precision component carrier according to the
    /// attachment's declared numeric type.
    pub(crate) fn from_components(
        numeric: crate::protocol::pixel_format::ColorNumericType,
        components: [f64; 4],
    ) -> Self {
        use crate::protocol::pixel_format::ClearComponents;
        match numeric.clear_components(components) {
            ClearComponents::Float(values) => Self::Float(values),
            ClearComponents::Uint(values) => Self::Uint(values),
            ClearComponents::Sint(values) => Self::Sint(values),
        }
    }

    pub(crate) fn vk(self) -> vk::ClearColorValue {
        match self {
            Self::Float(float32) => vk::ClearColorValue { float32 },
            Self::Uint(uint32) => vk::ClearColorValue { uint32 },
            Self::Sint(int32) => vk::ClearColorValue { int32 },
        }
    }
}

impl Default for ColorClearValue {
    fn default() -> Self {
        Self::Float([0.0; 4])
    }
}

/// One colour attachment's format and clear value as a single engine input.
///
/// Fields are private because the pair is constructed by pixel-format
/// translation. This prevents a request from pairing an integer image format
/// with a float clear (or the reverse) after translation chose them together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorAttachmentState {
    format: vk::Format,
    clear: ColorClearValue,
}

impl ColorAttachmentState {
    pub(crate) fn new(format: vk::Format, clear: ColorClearValue) -> Self {
        Self { format, clear }
    }

    pub fn format(self) -> vk::Format {
        self.format
    }

    pub(crate) fn clear(self) -> ColorClearValue {
        self.clear
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
    /// Statically used uniform-constant bindings derived once from the exact
    /// executable variants above. The engine compares these with each draw's
    /// layout before it lets the host compile the pipeline.
    pub vert_used_descriptor_bindings: std::sync::Arc<[u32]>,
    pub frag_used_descriptor_bindings: std::sync::Arc<[u32]>,
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
    /// `setBlendColorRed:green:blue:alpha:`, which the four
    /// `MTLBlendFactorBlendColor` factors read.
    ///
    /// Encoder state, not pipeline state: the guest changes it without
    /// changing the pipeline, and one encoder has one of it however many
    /// attachments name a constant factor. So it is here rather than inside
    /// `blend`, it is not part of the pipeline key, and the rail supplies it
    /// through `VK_DYNAMIC_STATE_BLEND_CONSTANTS` per draw. A guest animating
    /// a fade used to compile a pipeline per frame.
    ///
    /// Unconditional rather than `Option`, because there is no state in which
    /// an encoder does not have one. A guest that issued no
    /// `setBlendColorRed:` reaches here as the all-zero value the runtime
    /// substitutes, which is what this device has always used for it.
    pub blend_color: [f32; 4],
    /// Which channels the primary colour attachment writes.
    ///
    /// Separate from `blend` because `MTLColorWriteMask` is independent of
    /// `blendingEnabled`: an unblended attachment with a mask still leaves the
    /// unwritten channels alone, so folding it into `Option<BlendStateResource>`
    /// would drop it on every unblended draw.
    pub color_write_mask: ColorWriteMask,
    /// Protocol-derived target identity for GPU residency (workstream D).
    pub target_identity: Option<TargetIdentity>,
    /// Format and clear value of colour attachment zero's texture view.
    ///
    /// This can differ from [`TargetIdentity::resident_format`] without naming
    /// another allocation. Metal texture views over one surface commonly use
    /// the linear and sRGB members of one format-compatibility class; Vulkan
    /// represents that distinction on the image view and render pass. The
    /// paired clear keeps continuous and normalized formats as semantic floats,
    /// while integer formats carry integer values rather than float bit
    /// patterns. `None` means the ordinary format fallback and a zero clear.
    pub color_attachment: Option<ColorAttachmentState>,
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
    /// This, [`Self::target_rgba8`], [`Self::target_guest_seed`],
    /// [`Self::color0_declared`] and [`DrawRequest::color_attachment`] are the
    /// whole load action, and they are ordered: `load_from_target` wins, else
    /// exactly one seed is copied, else `color0_declared` decides between
    /// clearing to [`Self::color_attachment`]'s value and beginning the pass
    /// undefined.
    pub load_from_target: bool,
    /// The load action the **guest** declared for slot 0, independent of
    /// whether this device found any prior contents to honour it with.
    ///
    /// The seed fields above say what this device *can* offer; this says what
    /// the guest *asked for*, and they are different questions. Without it the
    /// engine cannot tell a pass that asked to clear from a pass that promised
    /// to keep its contents and arrived with none — and it then has to invent a
    /// colour for both. See [`super::caches::Color0Load`].
    ///
    /// `None` means the caller did not state one, which is a different fact
    /// from "the guest asked for a clear"; the engine keeps its historical
    /// answer for it and clears. Only a caller that has decoded a real load
    /// action fills this in, so an unfilled request cannot silently acquire a
    /// preserving reading it was never given evidence for.
    pub color0_declared: Option<reims_vgpu_protocol::pass_action::LoadAction>,
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
    /// The four fixed-function states a guest sets on the render command
    /// encoder — `setCullMode:`, `setFrontFacingWinding:`,
    /// `setTriangleFillMode:` and `setDepthClipMode: `— as the ordinals it
    /// wrote, defaulting to Metal's own where it set nothing.
    ///
    /// One aggregate rather than four fields, because four raw states in a row
    /// is a struct literal that compiles with two of them transposed. The
    /// ordinals travel unparsed for the reason
    /// [`BlendStateResource`]'s do; `reims_vgpu_vulkan::raster::plan` is the
    /// one place that decides what they mean and whether this device can
    /// serve them.
    pub raster: reims_vgpu_vulkan::raster::GuestRasterState,
    /// `setLineWidth:` — the width the stream last set, `None` where it set
    /// none and Metal's own default stands.
    ///
    /// Beside [`Self::raster`] rather than inside it, because it is not the
    /// same kind of state: those four are closed ordinals that decide how a
    /// pipeline is built, and this is a float that decides nothing about the
    /// pipeline at all — `VK_DYNAMIC_STATE_LINE_WIDTH` is 1.0 core, so it is
    /// dynamic on every host and never a cache dimension. Folding it into that
    /// aggregate would also cost it `Ord` and `Hash`, which the pipeline key
    /// needs and a float cannot give.
    ///
    /// Unparsed, like the ordinals beside it:
    /// `reims_vgpu_vulkan::raster::line_width` is the one place that decides
    /// whether this device can serve it, and it needs the draw's topology to
    /// decide — which only exists here.
    pub line_width: Option<f32>,
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
    /// Attachment format and clear, resolved together from the guest's
    /// `MTLPixelFormat` by `translate::pixel::color_attachment`. Keeping the
    /// pair opaque makes the render pass, image and Vulkan clear union member
    /// agree by construction.
    pub attachment: ColorAttachmentState,
    /// true ⇒ LOAD the existing resident content; false ⇒ CLEAR to this
    /// attachment's paired clear value.
    pub load: bool,
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

/// The `MTLPrimitiveType` a draw names, with this engine's answer for a request
/// that names none.
///
/// A newtype over [`reims_vgpu_core::topology::PrimitiveType`] rather than a
/// second enum beside it, and it exists only for the `Default`. Metal has no
/// default primitive type — `drawPrimitives:` states one on every call — so
/// there is no wire fact for the protocol layer to record, and the value a
/// zero-built [`DrawRequest`] carries is this crate's convention rather than
/// anything the guest declared. Keeping the convention here is what lets the
/// enum itself stay a statement about `MTLPrimitiveType` and nothing else.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PrimitiveTopology(pub reims_vgpu_core::topology::PrimitiveType);

impl Default for PrimitiveTopology {
    fn default() -> Self {
        Self(reims_vgpu_core::topology::PrimitiveType::Triangle)
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

/// `MTLVertexFormat`, parsed.
///
/// [`reims_vgpu_core::vertex_format::VertexFormat`] under this crate's name for
/// it. It used to be a second fifty-three-arm enumeration beside that one, with
/// the byte size and the Vulkan spelling in a table of its own — and those two
/// are the same fact stated twice, because a `Short3` occupies six bytes
/// *because* it is three 16-bit components, which is also why it is
/// `R16G16B16_UINT`. The owning layer states the size and the component count;
/// [`reims_vgpu_vulkan::vertex::format`] states the spelling.
pub use reims_vgpu_core::vertex_format::VertexFormat as VertexAttributeFormat;

/// `MTLVertexStepFunction`, parsed.
///
/// All five of them, where this crate's own enumeration held three and the
/// tessellation pair were refusals in the translation table. They are
/// recognised values that this rail declines — see
/// [`reims_vgpu_vulkan::vertex::input_rate`], which is where the decline
/// happens now, because whether a step function has a `VkVertexInputRate` is a
/// fact about Vulkan and not about `MTLVertexStepFunction`.
pub use reims_vgpu_core::vertex_step::StepFunction as VertexStepFunction;

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
                    if skip >= run.len() {
                        skip -= run.len();
                        continue;
                    }
                    let within = skip as usize;
                    skip = 0;
                    let n = (run.len() as usize).saturating_sub(within).min(take);
                    // SAFETY: `host_ptr` is a stable RAMBlock alias from
                    // `HostOps::map_pages`, valid for the VM lifetime; the
                    // read races guest CPU writes exactly like the staging
                    // path's `read_task_gva_by_id` copy does.
                    unsafe {
                        let slice = std::slice::from_raw_parts(
                            (run.host_ptr() as *const u8).add(within),
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
    /// The image and view shape, as **one** total answer.
    ///
    /// This was four independent booleans — `arrayed`, `volume`, `cube`,
    /// `one_dim` — with the exclusions between them stated in a doc comment
    /// ("mutually exclusive with `volume` and `cube`") rather than in the type.
    /// Twelve of their sixteen combinations name no Vulkan image, the three
    /// structs that carried them set them in two different field orders, and
    /// the only thing standing between a permuted assignment and the wrong view
    /// type was an ordered `if` cascade at the creation site and a test that
    /// existed to catch the permutation. A kind cannot be permuted.
    ///
    /// Multisampling is deliberately **not** folded in here even though
    /// [`reims_vgpu_core::texture_shape::TextureKind`] can spell it: the
    /// neutral 1×1 substitute below binds a shape without being multisampled,
    /// so the two are independent facts and [`Self::multisampled`] stays its
    /// own field. Metal `texture1d` / `texture1d_array` (colour-transfer LUTs)
    /// arrive as `D1` / `D1Array`, which is what makes the image a Vulkan 1D
    /// image so the sampled descriptor type matches the shader's declared 1D
    /// image; `height` is 1 for those.
    pub kind: reims_vgpu_core::texture_shape::TextureKind,
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
    /// Decoded texture-view swizzle, applied as the image view's component
    /// mapping so the GPU performs it at sample time. Identity (the default)
    /// creates the same view as before. Doing this on the view rather than by
    /// rewriting texels is what lets a swizzled texture stay on whatever
    /// content rail it was already on, including the zero-copy one — a CPU
    /// remap would force every swizzled bind onto the upload path.
    pub swizzle: crate::protocol::pixel_format::SwizzlePlan,
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

// A sampler's filters, mip filter, address modes and border colour are not
// spelled as engine enums. They travel as the guest's own `MTLSampler*`
// ordinals to `reims_vgpu_core::sampler::SamplerShape`, whose `checked()` is
// the one parse and the one place a declaration is admitted or refused. A
// second set of names here would be a second table to keep in step with it,
// and the one thing the two could disagree about is which ordinal means what.
//
// `SamplerCompareFunction` below is the exception, and not because samplers
// are special: `MTLCompareFunction` is also the depth test and the stencil
// test, which are decoded on a different path and have their own owner.

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
    /// The `MTLCompareFunction` ordinal this was decoded from.
    ///
    /// Declaration order is the ABI order, asserted against the wire in
    /// `translate::raster`'s own tests, so the discriminant is the ordinal and
    /// a second table would be a second thing to keep in step with it.
    pub(crate) fn mtl_ordinal(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SamplerResource {
    pub binding: u32,
    /// `MTLSamplerMinMagFilter`, as the guest wrote it. See the note above
    /// [`SamplerCompareFunction`] for why these are ordinals.
    pub min_filter: u32,
    pub mag_filter: u32,
    /// `MTLSamplerMipFilter`.
    pub mip_filter: u32,
    /// `MTLSamplerAddressMode`, one per axis.
    pub address_mode_u: u32,
    pub address_mode_v: u32,
    pub address_mode_w: u32,
    /// `MTLSamplerBorderColor`.
    pub border_color: u32,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32, // f32 bits for Hash
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

impl SamplerResource {
    /// The sampler metal2vulkan's `SynthesizedReadSampler` placeholder must be
    /// bound with: nearest filtering and clamp-to-edge, per the translator's
    /// own statement of what its harness binds there. Nearest is also the only
    /// filter Vulkan permits when the same placeholder pairs with an integer
    /// sampled image, which the translator does for LOD queries.
    pub fn read_placeholder(binding: u32) -> Self {
        Self {
            min_filter: sampler::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            mag_filter: sampler::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
            ..Self::normalized_default(binding)
        }
    }

    pub fn normalized_default(binding: u32) -> Self {
        Self {
            binding,
            min_filter: sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mag_filter: sampler::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
            mip_filter: sampler::MTL_SAMPLER_MIP_FILTER_NOT_MIPMAPPED,
            address_mode_u: sampler::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_v: sampler::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            address_mode_w: sampler::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            border_color: sampler::MTL_SAMPLER_BORDER_COLOR_TRANSPARENT_BLACK,
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
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub address_mode_u: u32,
    pub address_mode_v: u32,
    pub address_mode_w: u32,
    pub border_color: u32,
    pub compare_function: SamplerCompareFunction,
    pub lod_min: u32,
    pub lod_max: u32,
    pub max_anisotropy: u32,
    pub unnormalized_coordinates: bool,
}

/// One colour attachment's blend declaration, as the six `MTLBlendFactor` and
/// `MTLBlendOperation` ordinals the guest serialized.
///
/// The blend colour is **not** here. `setBlendColorRed:green:blue:alpha:` is a
/// command on the render command encoder, not a property of the pipeline
/// descriptor, so it is one value per draw rather than one per attachment —
/// see [`DrawRequest::blend_color`]. Carrying it per attachment made four
/// floats that could disagree with each other and could not, and put a value
/// the guest animates into the pipeline cache key.
///
/// The ordinals travel **unparsed**. Which nineteen values are factors, which
/// five are operations, and which four of the factors need the second fragment
/// output are all statements about `MTLRenderPipeline.h`, and this crate is
/// not where that is decided: `reims_vgpu_core::blend::ColorAttachmentShape`
/// owns the parse and `reims_vgpu_vulkan::blend` owns the spelling. Carrying a
/// pair of local enums between the two would be a third table that has to
/// agree with both, and the way it disagreed last time was by stopping at
/// `MTLBlendFactorOneMinusBlendAlpha = 14` while Apple's enum ran to 18.
///
/// A consequence worth stating: an ordinal outside the guest API no longer
/// refuses the *slot* at translation time, it refuses the *pipeline* at build
/// time, which is where the same declaration is also weighed against what this
/// device can blend. There is one answer per pipeline rather than one per
/// attachment reached by two different paths.
#[derive(Clone, Copy, Debug)]
pub struct BlendStateResource {
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
}

impl BlendStateResource {
    pub(crate) fn key(&self) -> BlendKey {
        BlendKey {
            src_rgb: self.src_rgb,
            dst_rgb: self.dst_rgb,
            op_rgb: self.op_rgb,
            src_alpha: self.src_alpha,
            dst_alpha: self.dst_alpha,
            op_alpha: self.op_alpha,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct BlendKey {
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
}

// ---------------------------------------------------------------------------
// Compute request surface
// ---------------------------------------------------------------------------

/// Named compute failure. Same `vk_engine_*` prefix family as draw.
pub type ComputeError = DrawError;

/// The translator's per-region dispatch payload: logical thread grid, thread
/// base, threadgroup base, and logical threadgroup grid, three `u32` each.
///
/// Named once so the payload the runtime copies, the field that carries it, and
/// the byte range the pipeline layout declares are the same width by
/// construction rather than by three matching literals.
pub type ComputeDispatchPayload = [u32; 12];

/// One Vulkan dispatch of a Metal exact-thread launch.
///
/// `dispatchThreads` may end a dimension in a partial threadgroup, and Vulkan
/// fixes a workgroup size per pipeline. The translator answers that by cutting
/// the launch into at most eight rectangular regions — the interior plus the
/// boundary slab of each axis — and giving each its own local size and its own
/// logical origin. A region is a whole dispatch, not a correction to one, so a
/// dropped region is dropped guest work rather than a rounding error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeDispatchRegion {
    /// The workgroup size this region's pipeline is specialized to, written to
    /// the translator's three local-size specialization constants.
    pub local_size: [u32; 3],
    /// Workgroup counts for this region's `vkCmdDispatch`.
    pub group_count: [u32; 3],
    /// The translator's complete dispatch payload for this region: logical
    /// thread grid, thread base, threadgroup base, logical threadgroup grid.
    pub push_constants: ComputeDispatchPayload,
}

/// How one compute request reaches the device.
///
/// The two forms are the translated kernel's own dispatch contract, not a
/// device-side choice: a module built for whole workgroups bakes its local size
/// as a constant and cannot serve a partial threadgroup, and a module built for
/// exact threads leaves its local size specializable and reads its logical grid
/// from push constants. Carrying them as one enum is what keeps an exact-thread
/// launch from being issued as a single rounded-up dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeDispatch {
    /// Whole-workgroup launch: one dispatch of these workgroup counts, no push
    /// constants. Only for a module whose dispatch contract proved every
    /// workgroup complete.
    Workgroups([u32; 3]),
    /// Exact-thread launch. Every region is issued, each against a pipeline
    /// specialized to its own local size, each preceded by its own payload
    /// written at `push_offset`.
    Regions {
        /// Reflected byte offset of the 48-byte dispatch payload.
        push_offset: u32,
        /// The logical threadgroup grid the regions tile. Census and
        /// zero-work validation read this, so both stay answerable without
        /// walking the regions.
        threadgroups_per_grid: [u32; 3],
        regions: Vec<ComputeDispatchRegion>,
    },
}

impl Default for ComputeDispatch {
    fn default() -> Self {
        Self::Workgroups([0; 3])
    }
}

impl ComputeDispatch {
    /// The whole launch's workgroup counts, whichever form issues them.
    pub fn threadgroups_per_grid(&self) -> [u32; 3] {
        match self {
            Self::Workgroups(grid) => *grid,
            Self::Regions {
                threadgroups_per_grid,
                ..
            } => *threadgroups_per_grid,
        }
    }

    /// The push-constant range a pipeline layout for this launch must declare.
    pub fn push_constant_range(&self) -> Option<(u32, u32)> {
        match self {
            Self::Workgroups(_) => None,
            Self::Regions { push_offset, .. } => {
                Some((*push_offset, COMPUTE_DISPATCH_PUSH_CONSTANT_SIZE))
            }
        }
    }
}

/// Byte size of one region's dispatch payload.
///
/// Derived from the payload itself rather than restated, so the range a layout
/// declares cannot drift from the bytes `vkCmdPushConstants` writes. The
/// assertion below holds it against the translator's own published size; the
/// two are independently derived — ours from the field, the translator's from
/// its ABI — and a layout that declares fewer bytes than the shader reads is a
/// validation error the driver reports, not a wrong answer we could see.
pub const COMPUTE_DISPATCH_PUSH_CONSTANT_SIZE: u32 =
    core::mem::size_of::<ComputeDispatchPayload>() as u32;

const _: () = assert!(
    COMPUTE_DISPATCH_PUSH_CONSTANT_SIZE
        == metal2vulkan::reflect::KERNEL_DISPATCH_PUSH_CONSTANT_SIZE,
    "the region payload must be exactly the translator's reflected dispatch range"
);

/// Inputs for one compute dispatch. Engine receives resolved bytes + SPIR-V only.
#[derive(Debug, Default)]
pub struct ComputeRequest {
    /// Vulkan-dialect compute SPIR-V. A whole-workgroup module bakes its local
    /// size; an exact-thread module leaves it specializable per region.
    pub spirv: Vec<u32>,
    /// Entry point name (m2v kernel entry is `"main"`).
    pub entry: String,
    /// The launch form the translated kernel's dispatch contract requires.
    pub dispatch: ComputeDispatch,
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
    /// One result per requested storage image, in request order (same length as
    /// `storage_images`).
    ///
    /// Which variant comes back is decided entirely by the matching request's
    /// [`ComputeStorageImageResource::destination`] — the engine does not choose
    /// a rail here, it honours the one it was given, so the caller can pair
    /// result `i` with its own destination without asking the engine what it
    /// did.
    pub images: Vec<ComputeImageResult>,
}

/// Where one storage image's post-dispatch pixels land.
///
/// The two variants are the two ways of naming the same landing site, which is
/// why this is one field and not a `Vec<u8>` with a flag beside it: a request
/// carrying both a host buffer and a guest window could name two destinations,
/// and nothing downstream could say which one the guest would read.
///
/// This replaces a deleted `images_direct` boolean. The flag went when the
/// deferred-flush rail was retired and the direct arm went with it; the render
/// side got [`crate::runtime::render_writeback`] as its replacement and this
/// rail got nothing, so every image came back through a host readback. See
/// [`ComputeImageResult`].
#[derive(Default)]
pub enum ComputeImageDestination {
    /// The engine reads the pixels back into host memory and the caller writes
    /// guest memory itself.
    ///
    /// The default, and the *only* form available on a host whose
    /// `VK_EXT_external_memory_host` capability is not
    /// [`crate::backend::vulkan::caps::host_pointer::HostPointerImport::Supported`].
    /// It is not a fallback bolted in front of a general path: on such a host it
    /// is the general path, and it is what a discrete GPU takes whenever staging
    /// is the correct answer.
    #[default]
    Host,
    /// The dispatch's own image→buffer copy lands directly in these guest pages
    /// and no pixels cross device→host.
    ///
    /// Carries the same [`super::GuestPageTarget`] the render rail writes through, so
    /// the guest-window geometry has exactly one spelling in this crate.
    ///
    /// # Why the guest cannot see these pages before the copy lands
    ///
    /// The copy is only *submitted*, so the ordering argument is the whole
    /// licence for this variant, and it is inherited rather than built. Arming
    /// `record_guest_write_debt` sets the process-global `GUEST_WRITE_DEBT`,
    /// which makes `guest_access_outstanding()` true, which removes
    /// `StampOrder::CpuReady` from the answers
    /// `runtime::drain::stamp_word_order_on_fifo` may give. The stamp is then
    /// handed to `write_completion_stamp`, and the completion thread waits the
    /// queue's monotonic timeline before it release-stores the word the guest
    /// polls. This dispatch submitted through the same `ResourcePools` ring and
    /// the same `submit_guest_work` the render rail uses, so its timeline value
    /// is below the awaited one and the bytes are in RAM before the guest is
    /// told anything.
    ///
    /// Nothing in that chain names a rail — it is the contract the render rail
    /// happened to be the only caller of. Two things it is easy to misread:
    /// the `settle_guest_writes` in `write_stamp` is on the *declined* path
    /// only and is not what carries a healthy boot, and `write_completion_stamp`
    /// takes only the *read* debt, deliberately leaving this write debt set so
    /// later stamps stay ordered until a host reader settles it.
    GuestPages {
        target: Box<super::GuestPageTarget>,
        /// The pages the copy is licensed over, in guest-virtual order.
        ///
        /// Carried beside the target and not derived from it, for the same
        /// reason `copy_target_to_guest_pages` takes the two separately: the
        /// target holds *host* references into those pages and the write debt
        /// has to be armed on their guest-physical addresses, which a reference
        /// cannot be turned back into. One walk produced both.
        pages: Vec<u64>,
    },
}

impl std::fmt::Debug for ComputeImageDestination {
    /// Names the arm and the window's size, never the window's addresses. A
    /// [`super::GuestPageTarget`] holds live host pointers into guest RAM, and a
    /// derived `Debug` would put them in any log line that formats a
    /// `ComputeRequest`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host => f.write_str("Host"),
            Self::GuestPages { target, pages } => write!(
                f,
                "GuestPages({}x{}, {} runs, {} pages)",
                target.width,
                target.height,
                target.runs.len(),
                pages.len()
            ),
        }
    }
}

/// What one storage image's dispatch produced, paired with the destination that
/// was asked for.
#[derive(Debug)]
pub enum ComputeImageResult {
    /// Readback pixels, for [`ComputeImageDestination::Host`]. Tight rows: the
    /// caller re-pitches them into the guest's window.
    Bytes(Vec<u8>),
    /// The copy into the guest's own pages is on the queue, for
    /// [`ComputeImageDestination::GuestPages`]. There are no bytes to hand back
    /// because none were read.
    ///
    /// The caller owes a settle before anything reads those pages, not a
    /// writeback — the same debt the render rail arms through
    /// `record_guest_write_debt`. `bytes` is what the copy will land, for the
    /// census, and is not a length of anything the caller holds.
    Landed { bytes: u64 },
}

impl ComputeImageResult {
    /// The readback pixels, or `None` where the engine wrote guest pages
    /// directly and never read any.
    ///
    /// `None` is not an error and not an empty frame — it means the bytes are
    /// already where they were going. A caller that treats it as "no output"
    /// would silently drop a landed frame, so match the variant where the two
    /// cases need different work and use this only where they do not.
    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            Self::Landed { .. } => None,
        }
    }
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
/// Single-layer 2D only: a compute texture binding is staged from one mapper-ref-texture
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
    /// Seed content, read *into* the image before the dispatch.
    ///
    /// Deliberately still a host allocation, and not paired with
    /// [`Self::destination`]. The seed direction is the GPU *reading* guest
    /// pages, which this device refuses on its own grounds: it acks a command
    /// before the work runs, so the guest may repaint those pages first. Only
    /// the output direction is routed.
    pub bytes: Vec<u8>,
    /// Where the post-dispatch pixels go. See [`ComputeImageDestination`].
    pub destination: ComputeImageDestination,
    /// Exact mapper-ref-texture resource lifetime/view contract for persistent GPU
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
    /// Levels `bytes` carries, base first, tightly packed by
    /// [`reims_vgpu_protocol::extent::tight_pyramid_spans`] — `1` for every
    /// binding but a guest mip chain sampled by an explicit LOD.
    pub mip_levels: u32,
    /// Where this binding's texels come from.
    ///
    /// One field rather than a `Vec<u8>` beside an `Option<ComputeResidentSampleBind>`,
    /// for the reason [`ComputeSampledImageResource`]'s sibling
    /// `StagedTexture::serve` already gives: those were the tag and the payload
    /// of an enum stored apart, so every producer had to build both halves and
    /// nothing made a producer that set one without the other fail to compile.
    /// The third source below could not be expressed at all in that shape — it
    /// has no bytes, valid or placeholder.
    pub source: ComputeSampledSource,
}

/// Where a compute sampled binding's texels come from.
#[derive(Clone, Debug)]
pub enum ComputeSampledSource {
    /// Host bytes uploaded into a pooled transient: every level the binding
    /// declares, base first, tightly packed by
    /// [`reims_vgpu_protocol::extent::tight_pyramid_spans`].
    Bytes(Vec<u8>),
    /// A device-local copy from the named resident storage image into a pooled
    /// transient (copy-on-sample: the transient never aliases the live
    /// resident, so the same dispatch may storage-write it).
    ///
    /// Only ever a single-level binding: a resident is one window at one level,
    /// so seeding a pyramid from it would leave levels 1.. empty.
    ///
    /// The copy's byte weight is *derived* from the binding's own geometry
    /// rather than carried. It used to travel as a zero-filled `Vec<u8>` of
    /// exactly that length, which the request validation then checked against
    /// the same derivation — a value that cannot disagree, checked as if it
    /// could.
    ResidentCopy(ComputeResidentSampleBind),
    /// A retained multisample render target, bound through its own registry
    /// view.
    ///
    /// Nothing is allocated, uploaded, or copied for this source, and that is
    /// the contract rather than an optimisation: [`SampledResource::multisampled`]
    /// states that such an image "can only come from a retained multisample
    /// target; linear bytes cannot be uploaded into one with a buffer-to-image
    /// copy". A Metal kernel reaches it by declaring
    /// `texture2d_ms<T, access::read>` and calling `read(coord, sample)`.
    MultisampleTarget(TargetIdentity),
}

/// Pixel formats the product compute path maps, re-exported from the rail that
/// owns them.
///
/// The enum names Vulkan formats and nothing else — no engine object, no pool,
/// no request — so it belongs beside the table that resolves it. It is named
/// here because the engine's request types are built out of it.
pub use reims_vgpu_vulkan::pixel::StorageImageFormat;

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
///
/// # Why it is still this rail's type, although neutral modules hold one
///
/// `writeback_debt`, `model::state` and `gva_store_witness` all name this type
/// while owning nothing about Vulkan, which reads like a leak and has been
/// attempted as one. It is not fixed by moving the type below the rail, because
/// `format` is **not** a protocol scalar the way every other field is. It is
/// what *this host* resolved the guest's declaration to:
///
/// * `Gva` takes `translate::pixel::color_attachment(guest_u16).vk`, which
///   keeps the sRGB spelling the guest declared — a distinction
///   `protocol::pixel_format::TexelLayout` deliberately folds away, so a
///   `TexelLayout` field would silently drop the transfer function.
/// * `runtime::draw::vulkan::gva_resident_format` narrows that again by asking
///   the *host* whether it renders to and blends the layout, falling back to
///   [`translate::pixel::RESIDENT_RGBA_FORMAT`] when it does not.
///
/// Because this is the registry key, either substitution changes which draws
/// share one `VkImage`: keying on the guest's declaration instead forks a
/// resident per spelling, and keying on a folded layout fuses two that the host
/// resolved apart. Both are the damage-history classes `storage_format` and
/// `surface_identity` document, arrived at from opposite directions.
///
/// So a neutral spelling has to be lossless *for the resolved set*, which means
/// a layout **and** its transfer function together — a contract type that does
/// not exist yet. Until it does, the honest statement is that this key carries
/// one rail-resolved component, and the neutral holders keep it opaque: none of
/// them reads `format`. `gva_store_witness` reads [`Self::is_bgra`], which is a
/// derived `bool` and stays derived for the reason its own field documents; a
/// neutral holder that started matching on `format` itself would be spelling
/// this rail's resolution outside this rail.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TargetIdentity {
    /// Backing mapping / surface id namespace.
    Surface {
        id: u32,
        width: u32,
        height: u32,
        generation: u64,
        /// This target's resident image format, from the pixel format the
        /// mapping declares for its own plane.
        ///
        /// A mapper-ref-texture mapping is not BGRA8 by its contract, which is what this
        /// namespace assumed for as long as it held no format: it declares a
        /// format, `mapping_write` reads that declaration to lay out the
        /// writeback, and macOS 26 declares `MTLPixelFormatRGBA16Float` for
        /// some of its compositing surfaces. Rendering those into a BGRA8
        /// resident quantized the guest's half-float compositing to eight bits
        /// with nothing to say so — the same loss the `Gva` namespace had, for
        /// the same reason, found the same way.
        ///
        /// [`crate::backend::vulkan::present_identity::surface_identity`] is the only
        /// producer, and it resolves this through
        /// [`crate::runtime::mapping_write::mapping_store_format`] — the same
        /// function the writeback lays its rows out from, so the resident and
        /// its destination cannot disagree about what the guest asked for.
        format: vk::Format,
    },
    /// normal texture ref namespace.
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
    /// [`super::pools::window_source_epoch`] as it stood when the drain resolved
    /// this identity and recorded it as the window's published resident.
    ///
    /// The window thread's check that the resident it was promised is still the
    /// one the registry holds — answerable with a single atomic load and no
    /// registry access, which is the whole point. The registry is guest-derived
    /// state on its way to the device the guest declared it against, and the
    /// window thread holds no device and cannot be given one: the lock a device
    /// would come behind is the one the drain holds for a whole render tranche,
    /// measured at 935-979 ms per exec packet, and a present waiting on that is
    /// not a present.
    ///
    /// It is the authority, not a check beside one. The window re-resolved the
    /// identity under the engine lock for three commits while a divergence
    /// census compared the two on every present and reported no disagreement
    /// across three driven boots; the re-resolve is gone.
    pub epoch: u64,
    /// What the publish resolved this identity to — the image, its layout, its
    /// extent and its storage class.
    ///
    /// The stamp above says the *promise* still stands; this says what was
    /// promised. Carried because the window's registry access is two
    /// dependencies and the stamp answers only one: the blit also reads and
    /// writes `slot.access`, and the barrier it builds from that layout is
    /// invalid — not merely stale — if the image has since moved. See
    /// [`super::pools::ResolvedResident`].
    ///
    /// This is what the blit reads. The window resolves nothing: `epoch` above
    /// says the publish's decision still stands, and this is that decision.
    // Only the host-window arm reads this: it is the window presenter's
    // comparison against its own re-resolve, and a build with no window has no
    // presenter. A `cfg` here answers "what did this build compile", which is
    // the only question one may answer.
    #[cfg_attr(not(feature = "host-window"), allow(dead_code))]
    pub(crate) resolved: super::pools::ResolvedResident,
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
    /// * `Surface` backs a mapper-ref-texture guest IOSurface, whose plane carries a
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
    /// The attachment's own texel, already in its physical layout.
    ///
    /// The eight-bit orders above are a *statement about colour*, and a
    /// half-float attachment has none: `f16_to_unorm8_lut` clamps it to
    /// `[0, 1]` at 256 levels, which for a coverage or HDR layer is the frame
    /// and not a rounding of it. A seed that arrives this way was read from the
    /// guest's pages in the attachment's layout and stages verbatim, so nothing
    /// narrows and nothing is expanded back.
    Native(crate::protocol::pixel_format::TexelLayout),
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
///
/// Owned by [`crate::runtime::guest_ram`] and re-exported here, because a host
/// span over guest memory is the memory layer's vocabulary and not the
/// engine's. It was a pair of public fields on this side of the boundary, and
/// what that cost is written up on the type itself.
pub use crate::runtime::guest_ram::GuestRun;

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
            attachment: ColorAttachmentState::new(
                vk::Format::B8G8R8A8_UNORM,
                ColorClearValue::default(),
            ),
            load: false,
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

        rebound.address_mode_v = sampler::MTL_SAMPLER_ADDRESS_MODE_REPEAT;
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
        assert_eq!(
            draw.primitive_topology,
            PrimitiveTopology(reims_vgpu_core::topology::PrimitiveType::Triangle)
        );
        assert_eq!(
            draw.raster,
            reims_vgpu_vulkan::raster::GuestRasterState::DEFAULT
        );
        assert!(draw.target_identity.is_none());
        assert!(draw.depth.is_none());
        assert!(!draw.skip_readback);
        assert!(!draw.color_input);

        let compute = ComputeRequest::default();
        assert_eq!(compute.dispatch, ComputeDispatch::Workgroups([0, 0, 0]));
        assert!(compute.storage_buffers.is_empty());
        assert!(compute.storage_images.is_empty());
    }

    /// The order is a property of the identity, and the three answers matter for
    /// different reasons.
    ///
    /// `Surface` answers from the format its mapping declared, and one
    /// constructed at the scanout format reports BGRA: every CPU consumer of a
    /// mapper-ref-texture composite Store is declared in guest scanout order, so an RGBA
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
