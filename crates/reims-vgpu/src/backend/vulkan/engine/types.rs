//! Draw request surface for the internal Vulkan engine (v1 §1.2 surface).
//!
//! Field meanings match the historical Metal→Vulkan product draw seam
//! (blend, Load seed, stage-in attributes, SSBOs, sampled images).

use ash::vk;

use crate::backend::vulkan::translate;
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

/// Inputs for one offscreen draw. Engine receives resolved bytes + post-reloc SPIR-V only.
#[derive(Debug, Default)]
pub struct DrawRequest {
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
    /// Load the live GPU image for [`DrawRequest::target_identity`] instead of
    /// seeding the attachment from the CPU. Requires that resident to exist.
    ///
    /// This, `target_rgba8` and [`DrawRequest::target_clear`] are the whole
    /// load action, and they are ordered: `load_from_target` wins, else a seed
    /// is uploaded, else the attachment clears to `target_clear`.
    pub load_from_target: bool,
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
    /// Clear value used when `load` is false (semantic float channels).
    pub clear: [f32; 4],
    /// true ⇒ LOAD the existing resident content; false ⇒ CLEAR to `clear`.
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
    pub indices: Vec<u8>,
}

impl IndexedDrawResource {
    pub(crate) fn index_range(&self) -> (u32, u32) {
        let mut min = u32::MAX;
        let mut max = 0u32;
        for i in 0..self.index_count as usize {
            let v = match self.index_type {
                IndexType::U16 => {
                    u16::from_le_bytes([self.indices[i * 2], self.indices[i * 2 + 1]]) as u32
                }
                IndexType::U32 => u32::from_le_bytes([
                    self.indices[i * 4],
                    self.indices[i * 4 + 1],
                    self.indices[i * 4 + 2],
                    self.indices[i * 4 + 3],
                ]),
            };
            min = min.min(v);
            max = max.max(v);
        }
        (min, max)
    }
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

/// Where a draw-time buffer's bytes come from (vertex attribute streams and
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
                for run in src.runs.iter() {
                    let take = (src.total_len as usize).saturating_sub(out.len());
                    if take == 0 {
                        break;
                    }
                    let n = (run.len as usize).min(take);
                    // SAFETY: `host_ptr` is a stable RAMBlock alias from
                    // `HostOps::map_pages`, valid for the VM lifetime; the
                    // read races guest CPU writes exactly like the staging
                    // path's `read_task_gva_by_id` copy does.
                    unsafe {
                        let slice = std::slice::from_raw_parts(run.host_ptr as *const u8, n);
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
    pub source: SampledSource,
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
            Self::R16Float | Self::Rg8Unorm => 2,
            Self::R8Unorm => 1,
            Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::Rgba8Unorm
            | Self::Bgra8Unorm
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float
            | Self::Rgb9e5Ufloat => 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Draw residency (workstream D)
// ---------------------------------------------------------------------------

/// Protocol-derived render-target identity (resource state, not content hash).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum TargetIdentity {
    /// Type-4 mapping / surface id namespace.
    Surface {
        id: u32,
        width: u32,
        height: u32,
        generation: u64,
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
        /// Channel order of this target's resident, from the pixel format the
        /// guest declared for the attachment. See [`TargetIdentity::is_bgra`]
        /// for why an order has to be part of the key rather than a per-draw
        /// argument, and why this namespace is the one that carries it: a
        /// surface is BGRA by its own contract and a pooled target has no
        /// declaration to follow, but a GVA render target's declaration is the
        /// whole answer.
        bgra: bool,
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
    /// The idle drain aged it out. A terminal destroy, not a recycle.
    IdleDrained,
    /// An allocation was refused and the reclaim retry gave it back, because it
    /// was neither pinned nor the only copy of its pixels. A terminal destroy of
    /// the image, but not of the pixels — the guest's own pages still hold them,
    /// which is the predicate `ResourcePools::recoverable_residents` selects on.
    AllocationReclaimed,
    /// `registry_ensure` replaced it for the same identity at a new geometry,
    /// generation or format.
    Recreated,
}

impl ResidentReclaim {
    pub fn slug(self) -> &'static str {
        match self {
            Self::IdleDrained => "idle_drained",
            Self::AllocationReclaimed => "allocation_reclaimed",
            Self::Recreated => "recreated",
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

    /// Physical channel order of the resident image behind this identity.
    ///
    /// The rule is one sentence: **a resident holds the bytes its destination
    /// stores.** Rendering it that way makes a raw image→buffer copy land the
    /// frame in guest memory unchanged, which is what deletes the whole-frame
    /// CPU swizzle and the blocking readback in front of it.
    ///
    /// Each namespace answers it from what it knows:
    ///
    /// * `Surface` backs a type-11 guest IOSurface, whose pages are BGRA8 by
    ///   that resource's own contract. Always BGRA.
    /// * `Gva` is a render target the guest declared a pixel format for, and
    ///   that declaration is the answer — carried in the key, from
    ///   `pixel_format::store_texel_order`. Two allocations at one address
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
        match self {
            Self::Surface { .. } => true,
            Self::Gva { bgra, .. } => *bgra,
            Self::Texture { .. } | Self::Anonymous { .. } => false,
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
    /// Zero-copy guest origin: the GPU gathers the texel bytes from imported
    /// guest RAM inside the draw's own command buffer (two-hop: imported
    /// buffer → pooled scratch → image). No CPU read and no hash — guest CPU
    /// writes are observed at execute time (at least as fresh as the CPU path's
    /// encode-time read).
    ///
    /// The gather is elided where a retained image already answers to the bind's
    /// identity, which is what [`crate::runtime::gather_witness::GatherVouch`]
    /// says is possible. `Fresh` means the identity was minted this bind and no
    /// retained image can match it, so the copy runs and the result is retained
    /// for the next bind to hit; carrying it lets the engine report *why* a
    /// gather happened instead of inferring it from the identity being present,
    /// which it always is.
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

/// Zero-copy sampled source: `runs` cover the linear texel window in order
/// (`sum(len) == total_len`). With `row_length_texels == 0` the window is
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
}

/// Producer-assigned identity + generation for CPU-sourced sampled content.
///
/// When two draws bind [`SampledSource::Bytes`] with the same identity, the
/// content is byte-identical by the producer's coherence model (the runtime
/// bumps `generation` whenever its authoritative cache entry is rewritten),
/// so the sampled cache may bind the retained GPU image without re-hashing
/// or comparing the bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledContentIdentity {
    /// Stable key of the guest resource (runtime-chosen keyspace).
    pub key: u64,
    /// Content generation of the producer's authoritative cache entry.
    pub generation: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_draw_range_decodes_both_wire_index_widths() {
        let u16_draw = IndexedDrawResource {
            index_type: IndexType::U16,
            index_count: 4,
            vertex_offset: 0,
            indices: [9u16, 2, 17, 4]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        };
        assert_eq!(u16_draw.index_range(), (2, 17));

        let u32_draw = IndexedDrawResource {
            index_type: IndexType::U32,
            index_count: 3,
            vertex_offset: 0,
            indices: [u32::MAX, 7, 99]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        };
        assert_eq!(u32_draw.index_range(), (7, u32::MAX));
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
    /// `Surface` must be BGRA whatever else is true: every CPU consumer of a
    /// type-11 composite Store is declared in guest scanout order, so an RGBA
    /// resident costs a whole-frame exchange per Store.
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
        }
        .is_bgra());
        for bgra in [false, true] {
            let gva = TargetIdentity::Gva {
                gva: 0x1000,
                width: 8,
                height: 8,
                generation: 0,
                bgra,
            };
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
    /// The order has to be *in* the key, not beside it. If it were not, both
    /// would hash to one registry slot whose image can only be built one way,
    /// and `registry_ensure` answers a requested order that disagrees with the
    /// slot's by destroying and recreating the image — every frame, for as long
    /// as both keep drawing.
    #[test]
    fn a_gva_targets_order_separates_it_from_the_same_address_in_the_other_order() {
        let at = |bgra| TargetIdentity::Gva {
            gva: 0x4000,
            width: 64,
            height: 64,
            generation: 7,
            bgra,
        };
        assert_ne!(at(true), at(false));
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(at(true)));
        assert!(
            seen.insert(at(false)),
            "the two orders must not collide in the registry's key space"
        );
    }
}
