//! Process-global metal2vulkan SPIR-V cache (AIR bytes → SPIR-V).
//!
//! Product Linux draws call m2v on the **doorbell MMIO vCPU** (sync drain under
//! BQL — see `runtime/mmio.rs` CONTROL_FIFO / child doorbell). Live OFF logs
//! showed the same pipelines re-translated dozens of times per boot (e.g.
//! pipe=39 fragment SPIR-V ~400 KiB × 28). That holds guest CPUs long enough
//! for `pmap_flush_tlbs` **IPI timeout** panics (WindowServer in
//! `processExecIndirect` / `submitOnChannel`).
//!
//! The cache key is the AIR blob itself plus its stage — not the pipeline object
//! id, which recycles, and not a hash of the AIR, which can collide. A content
//! hash narrows the bucket and `Slot::is` decides the hit; `ShaderId` carries
//! the argument for why the digest is not allowed to.
//! Measure-only hit/miss counters for fail-log census.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use metal2vulkan::passes::Stage;
use metal2vulkan::reflect::ShaderReflection;

use reims_vgpu_observe::Decline;

/// Render-stage translation requested by semantic pipeline preparation.
///
/// The translator's native stage enum remains an implementation detail of this
/// crate; composition code names only the two render stages it can request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderTranslationStage {
    Vertex,
    Fragment,
}

impl From<RenderTranslationStage> for Stage {
    fn from(stage: RenderTranslationStage) -> Self {
        match stage {
            RenderTranslationStage::Vertex => Self::Vertex,
            RenderTranslationStage::Fragment => Self::Fragment,
        }
    }
}

type M2vResult<T> = Result<T, M2vCacheDecline>;

/// A specific failure while caching, translating, or reconciling AIR layout.
///
/// Raw tool/IO/layout text stays payload; the reason itself is stable and
/// registered so render, compute, and the async worker name the same check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum M2vCacheDecline {
    VertexScratchWrite {
        detail: String,
    },
    FragmentScratchWrite {
        detail: String,
    },
    KernelScratchWrite {
        detail: String,
    },
    VertexTranslate {
        detail: String,
    },
    FragmentTranslate {
        detail: String,
    },
    KernelTranslate {
        detail: String,
    },
    ReflectionMalformed {
        stage: &'static str,
        violations: usize,
    },
    TranslationPending {
        stage: &'static str,
    },
    KernelLocalSizeZero {
        local_size: [u32; 3],
    },
    KernelLocalSizeMismatch {
        requested: [u32; 3],
        reflected: Option<[u32; 3]>,
    },
    RuntimeSamplerSpecialize {
        stage: &'static str,
        detail: String,
    },
    RuntimeStorageImageSpecialize {
        detail: String,
    },
}

impl M2vCacheDecline {
    /// Whether a second attempt at the same AIR could answer differently.
    ///
    /// Every entry in this cache is keyed by the AIR blob's content, so a
    /// refusal that is *about the AIR and its complete translation options* is
    /// reached again by every later ask and is worth remembering: malformed
    /// reflection or a degenerate requested threadgroup refuses the same way
    /// forever.
    ///
    /// The scratch writes are not about the AIR. `translate_air` and
    /// `translate_kernel_air` each begin by writing the blob to a fixed path
    /// under [`tmp_dir`], and that write fails for reasons belonging to the host
    /// filesystem at that instant — no space, no descriptors, a transient I/O
    /// error. Remembering one turns "the host could not spare a scratch file
    /// just then" into "this shader never renders again", and because the cache
    /// is unbounded and nothing evicts, "again" means for the life of the
    /// process. It is the same rule
    /// [`crate::engine::types::DrawError::out_of_memory`]
    /// states for the object caches.
    ///
    /// The translate declines are deliberately *not* here even though
    /// metal2vulkan also touches the filesystem. Their detail is an opaque
    /// string from the tool, so telling a disk failure from a malformed module
    /// would mean matching on its prose — and re-running a full translation on
    /// every draw of a genuinely untranslatable shader costs far more than the
    /// scratch write does. The split is by what the variant *names*, not by what
    /// its message happens to say.
    fn is_transient(&self) -> bool {
        match self {
            Self::VertexScratchWrite { .. }
            | Self::FragmentScratchWrite { .. }
            | Self::KernelScratchWrite { .. } => true,
            Self::VertexTranslate { .. }
            | Self::FragmentTranslate { .. }
            | Self::KernelTranslate { .. }
            | Self::ReflectionMalformed { .. }
            | Self::TranslationPending { .. }
            | Self::KernelLocalSizeZero { .. }
            | Self::KernelLocalSizeMismatch { .. }
            | Self::RuntimeSamplerSpecialize { .. }
            | Self::RuntimeStorageImageSpecialize { .. } => false,
        }
    }
}

/// Hand a stored failure to its caller, and drop it if a later ask could get a
/// different answer.
///
/// This is where the retry is armed, and it is the only point in the cache's
/// life cycle where arming one is safe. The async admission
/// ([`ensure_cached_async_keyed`]) cannot do it: its caller re-polls the same
/// guest packet until it answers `true`, so an arm that removed the entry and
/// re-queued would translate, fail, remove, re-queue — and the packet at the
/// channel head would never advance. Here the error is already on its way to
/// the caller, so this draw fails whatever we do; all that changes is that the
/// *next* draw finds no entry and translates again.
///
/// The cost while the host stays unable is one re-translation per draw of that
/// shader. The alarm is fail-visible and `fail_once`-deduped by key, so a
/// persistent one names itself once rather than per attempt.
fn forget_if_transient(cache: &mut Cache, id: ShaderId<'_>, error: &M2vCacheDecline) {
    if !error.is_transient() {
        return;
    }
    cache.forget(id);
    reims_vgpu_observe::Emit::decline("m2v_transient_failure_forgotten", error)
        .field("key", id.digest)
        .fail_once(id.digest);
}

fn log_token(detail: &str) -> String {
    detail.replace(char::is_whitespace, "_")
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Kernel => "kernel",
    }
}

fn translate_decline(stage: Stage, detail: String) -> M2vCacheDecline {
    match stage {
        Stage::Vertex => M2vCacheDecline::VertexTranslate { detail },
        Stage::Fragment => M2vCacheDecline::FragmentTranslate { detail },
        Stage::Kernel => M2vCacheDecline::KernelTranslate { detail },
    }
}

fn scratch_write_decline(stage: Stage, detail: String) -> M2vCacheDecline {
    match stage {
        Stage::Vertex => M2vCacheDecline::VertexScratchWrite { detail },
        Stage::Fragment => M2vCacheDecline::FragmentScratchWrite { detail },
        Stage::Kernel => M2vCacheDecline::KernelScratchWrite { detail },
    }
}

impl Decline for M2vCacheDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::VertexScratchWrite { .. } => "m2v_vertex_scratch_write",
            Self::FragmentScratchWrite { .. } => "m2v_fragment_scratch_write",
            Self::KernelScratchWrite { .. } => "m2v_kernel_scratch_write",
            Self::VertexTranslate { .. } => "m2v_vertex_translate",
            Self::FragmentTranslate { .. } => "m2v_fragment_translate",
            Self::KernelTranslate { .. } => "m2v_kernel_translate",
            Self::ReflectionMalformed { .. } => "m2v_reflection_malformed",
            Self::TranslationPending { .. } => "m2v_translation_pending_at_sync_boundary",
            Self::KernelLocalSizeZero { .. } => "m2v_kernel_local_size_zero",
            Self::KernelLocalSizeMismatch { .. } => "m2v_kernel_local_size_mismatch",
            Self::RuntimeSamplerSpecialize { .. } => "m2v_runtime_sampler_specialize",
            Self::RuntimeStorageImageSpecialize { .. } => "m2v_runtime_storage_image_specialize",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::VertexScratchWrite { detail } => vec![
                ("stage", "vertex".to_string()),
                ("file", "v.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::FragmentScratchWrite { detail } => vec![
                ("stage", "fragment".to_string()),
                ("file", "f.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::KernelScratchWrite { detail } => vec![
                ("stage", "kernel".to_string()),
                ("file", "k.air".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::VertexTranslate { detail } => vec![
                ("stage", "vertex".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::FragmentTranslate { detail } => vec![
                ("stage", "fragment".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::KernelTranslate { detail } => vec![
                ("stage", "kernel".to_string()),
                ("detail", log_token(detail)),
            ],
            Self::TranslationPending { stage } => {
                vec![("stage", (*stage).to_string())]
            }
            Self::ReflectionMalformed { stage, violations } => vec![
                ("stage", (*stage).to_string()),
                ("violations", violations.to_string()),
            ],
            Self::KernelLocalSizeZero { local_size } => vec![
                ("tg_x", local_size[0].to_string()),
                ("tg_y", local_size[1].to_string()),
                ("tg_z", local_size[2].to_string()),
            ],
            Self::KernelLocalSizeMismatch {
                requested,
                reflected,
            } => vec![
                (
                    "requested",
                    format!("{},{},{}", requested[0], requested[1], requested[2]),
                ),
                (
                    "reflected",
                    reflected.map_or_else(
                        || "none".to_string(),
                        |size| format!("{},{},{}", size[0], size[1], size[2]),
                    ),
                ),
            ],
            Self::RuntimeSamplerSpecialize { stage, detail } => vec![
                ("stage", (*stage).to_string()),
                ("detail", log_token(detail)),
            ],
            Self::RuntimeStorageImageSpecialize { detail } => {
                vec![("detail", log_token(detail))]
            }
        }
    }
}

reims_vgpu_observe::decline_display!(M2vCacheDecline);

impl std::error::Error for M2vCacheDecline {}

/// A translated shader: the SPIR-V bytes we hand to the Vulkan engine, plus the
/// `metal2vulkan` reflection facade derived from the same parsed AIR metadata
/// (descriptor bindings, texture shapes, vertex builtins, datalayout, …). The
/// reflection is the single source of truth for stage-interface facts so the
/// consumer never re-parses AIR. `spirv` is the translator-validated module
/// that Vulkan executes.
type RuntimeSamplerKey = Vec<(u32, reims_vgpu_core::SamplerResource)>;
type RenderSpecializationCache = Mutex<HashMap<RuntimeSamplerKey, Arc<CachedShader>>>;

pub struct CachedShader {
    pub(crate) spirv: Vec<u8>,
    reflection: Arc<ShaderReflection>,
    /// Backend-neutral resource interface consumed by device preparation.
    pub interface: Arc<reims_vgpu_core::ShaderInterface>,
    /// The same module as u32 words, materialized once — draw paths clone the
    /// `Arc`, never re-collect per draw (was a full-module copy ×2 per draw).
    pub words: Arc<Vec<u32>>,
    /// The module in the reflected effective descriptor layout.
    base: OnceLock<Arc<ShaderVariant>>,
    render_source: Option<RenderTranslationSource>,
    render_specializations: RenderSpecializationCache,
    kernel_source: Option<KernelTranslationSource>,
    kernel_specializations: Mutex<HashMap<Vec<RuntimeStorageImageRequest>, Arc<CachedShader>>>,
}

#[derive(Clone)]
struct RenderTranslationSource {
    air: Arc<[u8]>,
    stage: Stage,
    raster_sample_count: u32,
}

#[derive(Clone)]
struct KernelTranslationSource {
    air: Arc<[u8]>,
    local_size: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RuntimeStorageImageRequest {
    pub binding: u32,
    pub metal_index: u32,
    pub format: reims_vgpu_protocol::StorageImageFormat,
    pub storage_image: bool,
    pub storage_image_atomic: bool,
    pub read_without_format: bool,
    pub write_without_format: bool,
}

/// One binding numbering of a translated module, beside the reflected
/// interface transformed into *that* numbering.
///
/// The pair is one struct because the second is only true of the first.
/// The translator selects descriptor bindings before emission. Keeping the
/// executable and projected interface together prevents a caller from pairing
/// words from one reflected layout with resources from another.
/// The push-constant bytes a translated kernel reads its exact Metal thread
/// grid from, in this device's own hashable spelling.
///
/// metal2vulkan reflects this as `KernelGridPushConstantRange`, which is not a
/// hash key; the pipeline layout it forces *is* cached by one. The `size` is
/// taken from the translator's constant rather than restated, so the two
/// spellings cannot drift apart — only the presence and the offset are this
/// device's to carry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct KernelGridRange {
    pub offset: u32,
    pub size: u32,
}

impl From<metal2vulkan::reflect::KernelGridPushConstantRange> for KernelGridRange {
    fn from(range: metal2vulkan::reflect::KernelGridPushConstantRange) -> Self {
        Self {
            offset: range.offset,
            size: range.size,
        }
    }
}

impl KernelGridRange {
    /// Three tightly packed `u32` dimensions.
    pub const GRID_BYTES: usize = 3 * size_of::<u32>();

    /// The three `u32` dimensions, in the order the translated guard reads them
    /// at `offset`, `offset + 4` and `offset + 8`.
    #[must_use]
    pub fn bytes(threads_per_grid: [u32; 3]) -> [u8; Self::GRID_BYTES] {
        let mut out = [0u8; Self::GRID_BYTES];
        for (axis, value) in threads_per_grid.iter().enumerate() {
            out[axis * 4..axis * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

/// The byte count `bytes` produces is the translator's, not a number chosen
/// here: a widened grid ABI fails this rather than pushing a short range.
const _: () = assert!(
    KernelGridRange::GRID_BYTES as u32 == metal2vulkan::reflect::KERNEL_GRID_PUSH_CONSTANT_SIZE
);

pub struct ShaderVariant {
    /// Process-local identity used by semantic execution requests.
    ///
    /// The registry behind this identity retains only a weak reference: the
    /// translated shader cache remains the owner, so prepared modules die with
    /// the shader content that produced them rather than with an invented
    /// executor cache bound.
    pub id: reims_vgpu_protocol::PreparedShaderId,
    /// Where this kernel reads its exact Metal thread grid, when the translated
    /// entry point culls its surplus invocations against one.
    ///
    /// `None` for every render stage, and for a kernel the translator proved
    /// needs no cull. Derived from reflection here, beside the other
    /// reflection-derived facts, because the answer depends only on the
    /// translated shader and must not be rediscovered per dispatch.
    pub kernel_grid: Option<KernelGridRange>,
    /// The module, in this variant's numbering.
    pub words: Arc<Vec<u32>>,
    /// The typed sampler descriptors reflection declares, transformed into
    /// this variant's numbering. Constexpr state stays attached to its binding.
    ///
    /// Derived here rather than at the draw because the answer depends only on
    /// the translated shader. Reflection is the authority; the
    /// former implementation rediscovered the same interface by walking every
    /// SPIR-V instruction, twice per draw before this value was cached.
    ///
    /// # What it removed
    ///
    /// Twelve interleaved driven macos-13 sustained-animation boots, two pinned
    /// binaries differing only by this and by
    /// [`reims_vgpu_core::VertexBindPlan`], scored over the
    /// fast population. Per draw, median over busy census windows, then mean and
    /// range across boots:
    ///
    /// ```text
    ///                before (n=4)              after (n=6)
    /// reflect_us     0.340 [0.332..0.345]      0.021 [0.020..0.021]
    /// sampled_us     1.911 [1.861..1.972]      1.640 [1.574..1.691]
    /// engine_us      6.235 [6.065..6.338]      6.208 [6.033..6.283]
    /// ```
    ///
    /// `reflect_us` is **−94 %** and `sampled_us` **−14 %**, both fully
    /// disjoint. `engine_us` — a phase neither change touches — does not move,
    /// which is the control that says the two arms are comparable at all.
    ///
    /// `draw_us/draw` overall reads 13.33 [13.05..13.61] against 13.03
    /// [12.82..13.44]: −2.3 %, ranges overlapping. `present_hz` is 114.25
    /// against 114.16 — no movement, which is exactly what a 2 % per-draw change
    /// is supposed to look like here. **This bought a phase, not a frame**, and
    /// the elasticity beside `crate::runtime::drain::census::VBL_REPORT_EARLY`
    /// says it could not have bought a frame at that size.
    ///
    /// A first attempt at this reading ran the two pins in sequence rather than
    /// interleaved, and reported `engine_us` **+18 %** with `gpu occupancy` up
    /// and `present_hz` down — the GPU clocking down between the two groups,
    /// with nothing in the log to say so. Interleaving is not optional for a pin
    /// comparison on this host.
    pub samplers: Arc<[crate::spirv_bind::ReflectedSamplerDescriptor]>,
    /// Complete reflected descriptor population for these exact executable
    /// words. Reflection owns descriptor identity; SPIR-V is consulted only to
    /// distinguish executable static use within this population.
    pub declared_bindings: Arc<[u32]>,
    /// Descriptor bindings the executable module statically uses, in this
    /// variant's binding numbering.
    ///
    /// Vulkan requires each of these to appear in the pipeline layout. The
    /// engine checks that relation for every draw because a missing binding can
    /// crash a host driver during pipeline creation, but the left-hand set is
    /// immutable shader state: deriving it from the full module at every draw
    /// reconstructed state already retained here. Keeping it beside `words`
    /// also keeps executable use and reflected descriptor locations together.
    pub used_descriptor_bindings: Arc<[u32]>,
    /// Vertex stage-in widths projected from the translator's reflection.
    /// Unknown reflected types remain unreadable and cannot authorize a Vulkan
    /// vertex-format widening.
    pub vertex_inputs: crate::spirv_vertex_input::VertexInputWidths,
}

impl ShaderVariant {
    fn of(
        words: Arc<Vec<u32>>,
        samplers: Arc<[crate::spirv_bind::ReflectedSamplerDescriptor]>,
        vertex_inputs: crate::spirv_vertex_input::VertexInputWidths,
        declared_bindings: Arc<[u32]>,
        kernel_grid: Option<KernelGridRange>,
    ) -> Arc<Self> {
        let used_descriptor_bindings = declared_bindings
            .iter()
            .copied()
            .filter(|binding| {
                crate::spirv_bind::descriptor_static_use(&words, *binding).is_violation()
            })
            .collect::<Vec<_>>()
            .into();
        let variant = Arc::new(Self {
            id: allocate_prepared_shader_id(),
            words,
            kernel_grid,
            samplers,
            declared_bindings,
            used_descriptor_bindings,
            vertex_inputs,
        });
        prepared_shader_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(variant.id, Arc::downgrade(&variant));
        variant
    }
}

static NEXT_PREPARED_SHADER_ID: AtomicU64 = AtomicU64::new(1);
static PREPARED_SHADER_REGISTRY: OnceLock<
    Mutex<HashMap<reims_vgpu_protocol::PreparedShaderId, Weak<ShaderVariant>>>,
> = OnceLock::new();
static PREPARED_RENDER_SOURCE_REGISTRY: OnceLock<
    Mutex<HashMap<reims_vgpu_protocol::PreparedShaderId, Weak<CachedShader>>>,
> = OnceLock::new();

fn prepared_shader_registry(
) -> &'static Mutex<HashMap<reims_vgpu_protocol::PreparedShaderId, Weak<ShaderVariant>>> {
    PREPARED_SHADER_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn prepared_render_source_registry(
) -> &'static Mutex<HashMap<reims_vgpu_protocol::PreparedShaderId, Weak<CachedShader>>> {
    PREPARED_RENDER_SOURCE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn allocate_prepared_shader_id() -> reims_vgpu_protocol::PreparedShaderId {
    let id = NEXT_PREPARED_SHADER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .expect("prepared shader identity space exhausted");
    reims_vgpu_protocol::PreparedShaderId::new(id)
}

/// Convert a backend-owned translated module into the semantic stage carried
/// by a resolved command. Native words stay behind the prepared identity.
pub fn prepared_stage(variant: &Arc<ShaderVariant>) -> reims_vgpu_core::PreparedShaderStage {
    reims_vgpu_core::PreparedShaderStage {
        id: variant.id,
        used_descriptor_bindings: variant.used_descriptor_bindings.clone(),
    }
}

fn project_prepared_variant(
    variant: Arc<ShaderVariant>,
    interface: &reims_vgpu_core::ShaderInterface,
    layout: metal2vulkan::reflect::DescriptorLayout,
) -> reims_vgpu_core::PreparedShaderVariant {
    let declared_bindings = variant.declared_bindings.clone();
    let descriptor_uses: Arc<[(u32, reims_vgpu_core::DescriptorUse)]> = declared_bindings
        .iter()
        .filter_map(|binding| {
            let use_ = crate::spirv_bind::descriptor_static_use(&variant.words, *binding);
            (use_ != reims_vgpu_core::DescriptorUse::NotDeclared).then_some((*binding, use_))
        })
        .collect::<Vec<_>>()
        .into();
    let mut texture_uses = interface
        .bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                reims_vgpu_core::ShaderResourceKind::Texture
                    | reims_vgpu_core::ShaderResourceKind::TextureArray
            )
        })
        .map(|binding| {
            let executable_binding = binding
                .descriptor
                .map(|descriptor| descriptor.binding)
                .unwrap_or(layout.sampled_textures.start + binding.metal_index);
            (
                binding.metal_index,
                crate::spirv_bind::descriptor_static_use(&variant.words, executable_binding),
            )
        })
        .collect::<Vec<_>>();
    texture_uses.sort_unstable_by_key(|(metal_index, _)| *metal_index);
    texture_uses.dedup_by_key(|(metal_index, _)| *metal_index);
    let texture_uses = texture_uses.into();
    reims_vgpu_core::PreparedShaderVariant {
        program: prepared_stage(&variant),
        samplers: variant.samplers.clone(),
        declared_bindings,
        descriptor_uses,
        texture_uses,
        buffer_binding_base: layout.buffers.start,
        texture_binding_base: layout.sampled_textures.start,
        sampler_binding_base: layout.samplers.start,
        word_count: u32::try_from(variant.words.len()).expect("SPIR-V module length fits u32"),
    }
}

/// Project a translated render shader into backend-neutral executable facts.
/// Native words remain owned by this crate's content cache and prepared-ID
/// registry; retained guest pipeline state stores only this projection.
pub fn prepare_render_shader(
    shader: &Arc<CachedShader>,
    _stage: RenderTranslationStage,
) -> reims_vgpu_core::PreparedShaderFamily {
    let variant = project_prepared_variant(
        shader.variant(),
        &shader.interface,
        shader.reflection.descriptor_layout,
    );
    prepared_render_source_registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(variant.program.id, Arc::downgrade(shader));
    reims_vgpu_core::PreparedShaderFamily::new(shader.interface.clone(), variant)
}

fn runtime_sampler_state(
    sampler: &reims_vgpu_core::SamplerResource,
) -> Result<metal2vulkan::reflect::RuntimeSamplerState, String> {
    use metal2vulkan::reflect as native;
    use reims_vgpu_core::{
        SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerFilter,
        SamplerMipFilter,
    };

    let sampler = crate::engine::types::effective_sampler_state(sampler)
        .map_err(|reason| reason.to_string())?;
    let filter = |value| match value {
        SamplerFilter::Nearest => native::SamplerFilter::Nearest,
        SamplerFilter::Linear => native::SamplerFilter::Linear,
    };
    let address = |value| match value {
        SamplerAddressMode::ClampToEdge => Ok(native::SamplerAddressMode::ClampToEdge),
        SamplerAddressMode::Repeat => Ok(native::SamplerAddressMode::Repeat),
        SamplerAddressMode::MirrorRepeat => Ok(native::SamplerAddressMode::MirroredRepeat),
        SamplerAddressMode::ClampToZero => Ok(native::SamplerAddressMode::ClampToZero),
        SamplerAddressMode::ClampToBorderColor => Ok(native::SamplerAddressMode::ClampToBorder),
        SamplerAddressMode::MirrorClampToEdge => {
            Err("pixel-coordinate mirror-clamp-to-edge is not representable".to_string())
        }
    };
    let compare_function = match sampler.compare_function {
        // Core uses `Never` as the no-comparison sentinel and Vulkan creates
        // these samplers with compareEnable=false. Preserve that descriptor
        // state in the translator contract instead of asking it to model an
        // enabled comparison which always fails.
        SamplerCompareFunction::Never => native::SamplerCompareFunction::None,
        SamplerCompareFunction::Less => native::SamplerCompareFunction::Less,
        SamplerCompareFunction::Equal => native::SamplerCompareFunction::Equal,
        SamplerCompareFunction::LessEqual => native::SamplerCompareFunction::LessEqual,
        SamplerCompareFunction::Greater => native::SamplerCompareFunction::Greater,
        SamplerCompareFunction::NotEqual => native::SamplerCompareFunction::NotEqual,
        SamplerCompareFunction::GreaterEqual => native::SamplerCompareFunction::GreaterEqual,
        SamplerCompareFunction::Always => native::SamplerCompareFunction::Always,
    };
    Ok(native::RuntimeSamplerState {
        min_filter: filter(sampler.min_filter),
        mag_filter: filter(sampler.mag_filter),
        mip_filter: match sampler.mip_filter {
            SamplerMipFilter::NotMipmapped => native::SamplerMipFilter::None,
            SamplerMipFilter::Nearest => native::SamplerMipFilter::Nearest,
            SamplerMipFilter::Linear => native::SamplerMipFilter::Linear,
        },
        address_mode_s: address(sampler.address_mode_u)?,
        address_mode_t: address(sampler.address_mode_v)?,
        address_mode_r: address(sampler.address_mode_w)?,
        coordinates: if sampler.unnormalized_coordinates {
            native::SamplerCoordinates::Pixel
        } else {
            native::SamplerCoordinates::Normalized
        },
        compare_function,
        max_anisotropy: sampler.max_anisotropy,
        lod_min_clamp: f32::from_bits(sampler.lod_min),
        lod_max_clamp: f32::from_bits(sampler.lod_max),
        border_color: match sampler.border_color {
            SamplerBorderColor::TransparentBlack => native::SamplerBorderColor::TransparentBlack,
            SamplerBorderColor::OpaqueBlack => native::SamplerBorderColor::OpaqueBlack,
            SamplerBorderColor::OpaqueWhite => native::SamplerBorderColor::OpaqueWhite,
        },
        reduction: native::SamplerReduction::WeightedAverage,
        lod_bias: 0.0,
    })
}

fn translate_render_with_samplers(
    source: &RenderTranslationSource,
    states: &[(u32, reims_vgpu_core::SamplerResource)],
) -> M2vResult<CachedShader> {
    let _guard = translation_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let tmp = tmp_dir();
    let name = match source.stage {
        Stage::Vertex => "v.air",
        Stage::Fragment => "f.air",
        Stage::Kernel => unreachable!("render specialization cannot be a kernel"),
    };
    let path = tmp.join(name);
    std::fs::write(&path, source.air.as_ref())
        .map_err(|error| scratch_write_decline(source.stage, error.to_string()))?;
    let mut options = metal2vulkan::passes::TransformOptions::default()
        .with_descriptor_layout(render_descriptor_layout(source.stage))
        .map_err(|error| M2vCacheDecline::RuntimeSamplerSpecialize {
            stage: stage_name(source.stage),
            detail: error.to_string(),
        })?;
    if source.stage == Stage::Fragment {
        options.raster_sample_count = Some(source.raster_sample_count);
    }
    for (metal_index, sampler) in states {
        options = options
            .with_runtime_sampler(
                *metal_index,
                runtime_sampler_state(sampler).map_err(|detail| {
                    M2vCacheDecline::RuntimeSamplerSpecialize {
                        stage: stage_name(source.stage),
                        detail,
                    }
                })?,
            )
            .map_err(|detail| M2vCacheDecline::RuntimeSamplerSpecialize {
                stage: stage_name(source.stage),
                detail,
            })?;
    }
    let (spirv, reflection) = metal2vulkan::translate_reflected_with_options(
        path.to_str().unwrap_or(name),
        source.stage,
        &tmp,
        options,
    )
    .map_err(|detail| M2vCacheDecline::RuntimeSamplerSpecialize {
        stage: stage_name(source.stage),
        detail,
    })?;
    if reflection.runtime_sampler_specializations.len() != states.len() {
        return Err(M2vCacheDecline::RuntimeSamplerSpecialize {
            stage: stage_name(source.stage),
            detail: format!(
                "translator applied {} of {} runtime sampler states",
                reflection.runtime_sampler_specializations.len(),
                states.len()
            ),
        });
    }
    for (metal_index, sampler) in states {
        let expected = runtime_sampler_state(sampler).map_err(|detail| {
            M2vCacheDecline::RuntimeSamplerSpecialize {
                stage: stage_name(source.stage),
                detail,
            }
        })?;
        let reflected = reflection
            .runtime_sampler_specializations
            .iter()
            .find(|specialization| specialization.metal_index == *metal_index);
        if reflected.map(|specialization| specialization.state) != Some(expected) {
            return Err(M2vCacheDecline::RuntimeSamplerSpecialize {
                stage: stage_name(source.stage),
                detail: format!(
                    "translator reflected a different runtime state for Metal sampler {metal_index}"
                ),
            });
        }
    }
    finish_translated(spirv, reflection, source.stage, None, None)
}

/// Specialize pixel-coordinate sampler operations through metal2vulkan and
/// return a prepared module in the same reflected descriptor layout.
pub fn specialize_render_samplers(
    base: &reims_vgpu_core::PreparedShaderVariant,
    samplers: &[reims_vgpu_core::SamplerResource],
) -> M2vResult<reims_vgpu_core::PreparedShaderVariant> {
    let shader = prepared_render_source_registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&base.program.id)
        .and_then(Weak::upgrade)
        .ok_or_else(|| M2vCacheDecline::RuntimeSamplerSpecialize {
            stage: "render",
            detail: "prepared shader source lifetime ended".to_string(),
        })?;
    let Some(source) = shader.render_source.as_ref() else {
        return Ok(base.clone());
    };
    let mut states = base
        .samplers
        .iter()
        .filter(|reflected| reflected.static_state.is_none())
        .filter_map(|reflected| {
            samplers
                .iter()
                .find(|sampler| {
                    sampler.binding == reflected.binding
                        && sampler.source == reims_vgpu_core::SamplerSource::State
                        && sampler.unnormalized_coordinates
                })
                .cloned()
                .map(|sampler| (reflected.metal_index, sampler))
        })
        .collect::<Vec<_>>();
    states.sort_by_key(|(metal_index, _)| *metal_index);
    let mut canonical = Vec::with_capacity(states.len());
    for (metal_index, mut sampler) in states {
        // Multiple AIR parameters may intentionally alias one Metal sampler
        // index. The translator accepts one runtime state per Metal index, so
        // aliases must agree on that state; the Vulkan binding is not part of
        // the Metal sampler contract supplied to translation.
        sampler.binding = 0;
        if let Some((previous_index, previous)) = canonical.last() {
            if *previous_index == metal_index {
                if *previous != sampler {
                    return Err(M2vCacheDecline::RuntimeSamplerSpecialize {
                        stage: stage_name(source.stage),
                        detail: format!(
                            "aliased Metal sampler {metal_index} has conflicting runtime states"
                        ),
                    });
                }
                continue;
            }
        }
        canonical.push((metal_index, sampler));
    }
    let states = canonical;
    if states.is_empty() {
        return Ok(base.clone());
    }
    if let Some(cached) = shader
        .render_specializations
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&states)
        .cloned()
    {
        return Ok(project_prepared_variant(
            cached.variant(),
            &cached.interface,
            cached.reflection.descriptor_layout,
        ));
    }
    let specialized = Arc::new(translate_render_with_samplers(source, &states)?);
    let projected = project_prepared_variant(
        specialized.variant(),
        &specialized.interface,
        specialized.reflection.descriptor_layout,
    );
    shader
        .render_specializations
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(states, specialized);
    Ok(projected)
}

/// Prepare an executor-native module produced by a backend specialization
/// pass. The returned variant owns the native words; callers keep it alive
/// until the semantic request carrying [`ShaderVariant::id`] has executed.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn prepare_shader_words(words: Vec<u32>) -> Arc<ShaderVariant> {
    let declared_bindings = crate::spirv_bind::declared_binding_numbers(&words).into();
    // A fixture hands over raw module words with no reflection beside them, so
    // the kernel-grid range has to come from the module itself. Every module
    // metal2vulkan emits for a kernel places that block at the contract's
    // default offset, which is what this reconstructs — and a module with no
    // push-constant variable gets `None`, which is also what a render stage
    // gets. Real shaders never come through here: they carry reflection, and
    // reflection stays the authority for them.
    let kernel_grid =
        crate::spirv_bind::declares_push_constants(&words).then_some(KernelGridRange {
            offset: metal2vulkan::reflect::DEFAULT_KERNEL_GRID_PUSH_CONSTANT_OFFSET,
            size: metal2vulkan::reflect::KERNEL_GRID_PUSH_CONSTANT_SIZE,
        });
    ShaderVariant::of(
        Arc::new(words),
        Arc::from([]),
        crate::spirv_vertex_input::VertexInputWidths::unknown(),
        declared_bindings,
        kernel_grid,
    )
}

pub(crate) fn resolve_prepared_shader(
    id: reims_vgpu_protocol::PreparedShaderId,
) -> Option<Arc<ShaderVariant>> {
    prepared_shader_registry()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&id)
        .and_then(Weak::upgrade)
}

/// Register native test words with the same prepared-program boundary used by
/// product execution. Test ownership is process-long because integration tests
/// do not have a translated-shader cache whose lifetime can own the variant.
#[cfg(feature = "test-fixtures")]
pub fn prepare_test_shader(words: Vec<u32>) -> reims_vgpu_core::PreparedShaderStage {
    static OWNERS: OnceLock<Mutex<HashMap<Vec<u32>, Arc<ShaderVariant>>>> = OnceLock::new();
    let mut owners = OWNERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let variant = owners
        .entry(words.clone())
        .or_insert_with(|| prepare_shader_words(words));
    prepared_stage(variant)
}

/// Empty translated render shader for product-crate lifecycle fixtures.
#[cfg(feature = "test-fixtures")]
pub fn empty_test_shader(stage: RenderTranslationStage) -> Arc<CachedShader> {
    use metal2vulkan::reflect::{ShaderReflection, ShaderStage, REFLECTION_VERSION};
    let stage = match stage {
        RenderTranslationStage::Vertex => ShaderStage::Vertex,
        RenderTranslationStage::Fragment => ShaderStage::Fragment,
    };
    Arc::new(CachedShader::new(
        Vec::new(),
        Arc::new(ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            descriptor_layout: Default::default(),
            stage,
            // A render fixture, so there is no dispatch grid to cull against.
            kernel_dispatch: None,
            entry_point: None,
            bindings: vec![],
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            datalayout: None,
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }),
    ))
}

impl Drop for ShaderVariant {
    fn drop(&mut self) {
        prepared_shader_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
        prepared_render_source_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .remove(&self.id);
    }
}

pub(crate) fn project_shader_interface(
    reflection: &ShaderReflection,
) -> reims_vgpu_core::ShaderInterface {
    use metal2vulkan::meta::{TextureComponent, TextureDimension, TextureFormat};
    use metal2vulkan::reflect::{
        BufferExtent, BufferIndexSource, ResourceAccess, ResourceKind, ShaderStage,
    };
    use reims_vgpu_core::{
        ReflectedShaderStage, ShaderBufferByteRange, ShaderBufferExtent, ShaderBufferFootprint,
        ShaderBufferIndexSource, ShaderBufferStrideTerm, ShaderBufferStridedAccess,
        ShaderDescriptorLocation, ShaderInterface, ShaderResourceAccess, ShaderResourceBinding,
        ShaderResourceKind, ShaderTextureComponent, ShaderTextureDimension, ShaderTextureShape,
        UnsupportedShaderInterface,
    };
    use reims_vgpu_protocol::StorageImageFormat;

    let stage = match reflection.stage {
        ShaderStage::Vertex => ReflectedShaderStage::Vertex,
        ShaderStage::TessellationEvaluation => ReflectedShaderStage::TessellationEvaluation,
        ShaderStage::Fragment => ReflectedShaderStage::Fragment,
        ShaderStage::Kernel => ReflectedShaderStage::Kernel,
    };
    let unsupported = if reflection.tessellation.is_some() {
        Some(UnsupportedShaderInterface {
            feature: "tessellation",
            count: 1,
        })
    } else if !reflection.imageblock_layouts.is_empty() {
        Some(UnsupportedShaderInterface {
            feature: "kernel_imageblock",
            count: reflection.imageblock_layouts.len(),
        })
    } else if !reflection.implicit_imageblock_attachments.is_empty() {
        Some(UnsupportedShaderInterface {
            feature: "implicit_imageblock_attachments",
            count: reflection.implicit_imageblock_attachments.len(),
        })
    } else {
        reflection
            .fragment_imageblock
            .as_ref()
            .map(|imageblock| UnsupportedShaderInterface {
                feature: "fragment_imageblock",
                count: imageblock.members.len(),
            })
    };
    let bindings = reflection
        .bindings
        .iter()
        .map(|binding| {
            let kind = match binding.kind {
                ResourceKind::Buffer => ShaderResourceKind::Buffer,
                ResourceKind::ThreadgroupBuffer => ShaderResourceKind::ThreadgroupBuffer,
                ResourceKind::KernelStageInput => ShaderResourceKind::KernelStageInput,
                ResourceKind::Texture => ShaderResourceKind::Texture,
                ResourceKind::TextureArray => ShaderResourceKind::TextureArray,
                ResourceKind::StorageImage => ShaderResourceKind::StorageImage,
                ResourceKind::Sampler => ShaderResourceKind::Sampler,
                ResourceKind::StaticSampler => ShaderResourceKind::StaticSampler,
                ResourceKind::ColorInput => ShaderResourceKind::ColorInput,
                ResourceKind::AccelerationStructureShadow => {
                    ShaderResourceKind::AccelerationStructureShadow
                }
                ResourceKind::PrimitiveAccelerationStructure => {
                    ShaderResourceKind::PrimitiveAccelerationStructure
                }
                ResourceKind::VisibleFunctionTable => ShaderResourceKind::VisibleFunctionTable,
                ResourceKind::IntersectionFunctionTable => {
                    ShaderResourceKind::IntersectionFunctionTable
                }
                ResourceKind::EmbeddedArgBufferTexture => {
                    ShaderResourceKind::EmbeddedArgBufferTexture
                }
                ResourceKind::EmbeddedArgBufferBuffer => {
                    ShaderResourceKind::EmbeddedArgBufferBuffer
                }
                ResourceKind::BufferAddressTable => ShaderResourceKind::BufferAddressTable,
            };
            let descriptor = binding
                .descriptor
                .map(|descriptor| ShaderDescriptorLocation {
                    set: descriptor.set,
                    binding: descriptor.binding,
                    count: descriptor.count,
                });
            let extent = binding.extent.map(|extent| match extent {
                BufferExtent::Object { bytes } => ShaderBufferExtent::Object { bytes },
                BufferExtent::Unbounded => ShaderBufferExtent::Unbounded,
                BufferExtent::Unknown => ShaderBufferExtent::Unknown,
            });
            let footprint = binding
                .footprint
                .as_ref()
                .map(|footprint| ShaderBufferFootprint {
                    static_ranges: footprint
                        .static_ranges
                        .iter()
                        .map(|range| ShaderBufferByteRange {
                            offset: range.offset,
                            size: range.size,
                        })
                        .collect(),
                    strided_accesses: footprint
                        .strided_accesses
                        .iter()
                        .map(|access| ShaderBufferStridedAccess {
                            base_offset: access.base_offset,
                            access_size: access.access_size,
                            terms: access
                                .terms
                                .iter()
                                .map(|term| ShaderBufferStrideTerm {
                                    source: match term.source {
                                        BufferIndexSource::VertexIndex => {
                                            ShaderBufferIndexSource::VertexIndex
                                        }
                                        BufferIndexSource::InstanceIndex => {
                                            ShaderBufferIndexSource::InstanceIndex
                                        }
                                        BufferIndexSource::GlobalInvocationIdX => {
                                            ShaderBufferIndexSource::GlobalInvocationIdX
                                        }
                                        BufferIndexSource::GlobalInvocationIdY => {
                                            ShaderBufferIndexSource::GlobalInvocationIdY
                                        }
                                        BufferIndexSource::GlobalInvocationIdZ => {
                                            ShaderBufferIndexSource::GlobalInvocationIdZ
                                        }
                                        BufferIndexSource::LocalInvocationIdX => {
                                            ShaderBufferIndexSource::LocalInvocationIdX
                                        }
                                        BufferIndexSource::LocalInvocationIdY => {
                                            ShaderBufferIndexSource::LocalInvocationIdY
                                        }
                                        BufferIndexSource::LocalInvocationIdZ => {
                                            ShaderBufferIndexSource::LocalInvocationIdZ
                                        }
                                        BufferIndexSource::WorkgroupIdX => {
                                            ShaderBufferIndexSource::WorkgroupIdX
                                        }
                                        BufferIndexSource::WorkgroupIdY => {
                                            ShaderBufferIndexSource::WorkgroupIdY
                                        }
                                        BufferIndexSource::WorkgroupIdZ => {
                                            ShaderBufferIndexSource::WorkgroupIdZ
                                        }
                                        BufferIndexSource::LocalInvocationIndex => {
                                            ShaderBufferIndexSource::LocalInvocationIndex
                                        }
                                    },
                                    stride: term.stride,
                                })
                                .collect(),
                        })
                        .collect(),
                    has_unbounded_access: footprint.has_unbounded_access,
                });
            let texture_shape = binding.texture_shape.map(|shape| ShaderTextureShape {
                dimension: match shape.dimension {
                    TextureDimension::D1 => ShaderTextureDimension::D1,
                    TextureDimension::D2 => ShaderTextureDimension::D2,
                    TextureDimension::D3 => ShaderTextureDimension::D3,
                    TextureDimension::Cube => ShaderTextureDimension::Cube,
                    TextureDimension::Buffer => ShaderTextureDimension::Buffer,
                },
                arrayed: shape.arrayed,
                multisampled: shape.multisampled,
                component: match shape.component {
                    TextureComponent::Float => ShaderTextureComponent::Float,
                    TextureComponent::Sint => ShaderTextureComponent::Sint,
                    TextureComponent::Uint => ShaderTextureComponent::Uint,
                },
                writable: shape.writable,
                array_ref: shape.array_ref,
                array_length: shape.array_length,
                storage_format: shape.storage_format.map(|format| match format {
                    TextureFormat::R8 => StorageImageFormat::R8Unorm,
                    TextureFormat::Rgba8 => StorageImageFormat::Rgba8Unorm,
                    TextureFormat::R16f => StorageImageFormat::R16Float,
                    TextureFormat::R16ui => StorageImageFormat::R16Uint,
                    TextureFormat::Rg16f => StorageImageFormat::Rg16Float,
                    TextureFormat::R32f => StorageImageFormat::R32Float,
                    TextureFormat::R32i => StorageImageFormat::R32Sint,
                    TextureFormat::R32ui => StorageImageFormat::R32Uint,
                    TextureFormat::Rgba32i => StorageImageFormat::Rgba32Sint,
                    TextureFormat::Rgba32ui => StorageImageFormat::Rgba32Uint,
                    TextureFormat::Rgba32f => StorageImageFormat::Rgba32Float,
                    TextureFormat::Rgba16f => StorageImageFormat::Rgba16Float,
                    TextureFormat::Rgba8ui => StorageImageFormat::Rgba8Uint,
                    TextureFormat::Rgba16ui => StorageImageFormat::Rgba16Uint,
                    TextureFormat::Rgba8i => StorageImageFormat::Rgba8Sint,
                }),
            });
            let access = binding.access.map(|access| match access {
                ResourceAccess::Unused => ShaderResourceAccess::Unused,
                ResourceAccess::ReadOnly => ShaderResourceAccess::ReadOnly,
                ResourceAccess::WriteOnly => ShaderResourceAccess::WriteOnly,
                ResourceAccess::ReadWrite => ShaderResourceAccess::ReadWrite,
                ResourceAccess::Sampled => ShaderResourceAccess::Sampled,
                ResourceAccess::Storage => ShaderResourceAccess::Storage,
            });
            ShaderResourceBinding {
                kind,
                metal_index: binding.metal_index,
                descriptor,
                extent,
                footprint,
                texture_shape,
                access,
            }
        })
        .collect();
    ShaderInterface {
        stage,
        bindings,
        local_size: reflection.local_size,
        unsupported,
    }
}

pub struct PreparedKernelVariant {
    pub variant: Arc<ShaderVariant>,
    pub storage_formats: Vec<(u32, Option<reims_vgpu_protocol::StorageImageFormat>)>,
    _owner: Arc<CachedShader>,
}

fn runtime_storage_format(
    format: reims_vgpu_protocol::StorageImageFormat,
) -> Result<metal2vulkan::reflect::RuntimeStorageImageFormat, String> {
    use metal2vulkan::reflect::RuntimeStorageImageFormat as Native;
    use reims_vgpu_protocol::StorageImageFormat as Semantic;
    Ok(match format {
        Semantic::R8Unorm => Native::R8Unorm,
        Semantic::Rgba8Unorm => Native::Rgba8Unorm,
        Semantic::Bgra8Unorm => Native::Bgra8Unorm,
        Semantic::R16Float => Native::R16Float,
        Semantic::Rg16Float => Native::Rg16Float,
        Semantic::Rgba16Float => Native::Rgba16Float,
        Semantic::R32Float => Native::R32Float,
        Semantic::Rgba32Float => Native::Rgba32Float,
        Semantic::R16Uint => Native::R16Uint,
        Semantic::R32Uint => Native::R32Uint,
        Semantic::Rgba8Uint => Native::Rgba8Uint,
        Semantic::Rgba16Uint => Native::Rgba16Uint,
        Semantic::Rgba32Uint => Native::Rgba32Uint,
        Semantic::R32Sint => Native::R32Sint,
        Semantic::Rgba8Sint => Native::Rgba8Sint,
        Semantic::Rgba32Sint => Native::Rgba32Sint,
        unsupported => {
            return Err(format!(
                "runtime storage format {unsupported:?} unsupported"
            ))
        }
    })
}

fn reflected_storage_format(
    format: metal2vulkan::meta::TextureFormat,
) -> reims_vgpu_protocol::StorageImageFormat {
    use metal2vulkan::meta::TextureFormat as Native;
    use reims_vgpu_protocol::StorageImageFormat as Semantic;
    match format {
        Native::R8 => Semantic::R8Unorm,
        Native::Rgba8 => Semantic::Rgba8Unorm,
        Native::R16f => Semantic::R16Float,
        Native::R16ui => Semantic::R16Uint,
        Native::Rg16f => Semantic::Rg16Float,
        Native::R32f => Semantic::R32Float,
        Native::R32i => Semantic::R32Sint,
        Native::R32ui => Semantic::R32Uint,
        Native::Rgba32i => Semantic::Rgba32Sint,
        Native::Rgba32ui => Semantic::Rgba32Uint,
        Native::Rgba32f => Semantic::Rgba32Float,
        Native::Rgba16f => Semantic::Rgba16Float,
        Native::Rgba8ui => Semantic::Rgba8Uint,
        Native::Rgba16ui => Semantic::Rgba16Uint,
        Native::Rgba8i => Semantic::Rgba8Sint,
    }
}

fn translate_kernel_with_storage(
    source: &KernelTranslationSource,
    requests: &[RuntimeStorageImageRequest],
) -> M2vResult<CachedShader> {
    let _guard = translation_lock()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let tmp = tmp_dir();
    let path = tmp.join("k.air");
    std::fs::write(&path, source.air.as_ref()).map_err(|error| {
        M2vCacheDecline::KernelScratchWrite {
            detail: error.to_string(),
        }
    })?;
    let mut canonical = requests.to_vec();
    canonical.sort_by_key(|request| request.metal_index);
    for request in &mut canonical {
        request.binding = 0;
    }
    let mut unique: Vec<RuntimeStorageImageRequest> = Vec::with_capacity(canonical.len());
    for request in canonical {
        if let Some(previous) = unique.last() {
            if previous.metal_index == request.metal_index {
                if *previous != request {
                    return Err(M2vCacheDecline::RuntimeStorageImageSpecialize {
                        detail: format!(
                            "aliased Metal storage image {} has conflicting runtime states",
                            request.metal_index
                        ),
                    });
                }
                continue;
            }
        }
        unique.push(request);
    }
    let mut options = metal2vulkan::passes::TransformOptions {
        kernel_local_size: source.local_size,
        ..Default::default()
    };
    for request in &unique {
        options = options
            .with_runtime_storage_image(
                request.metal_index,
                metal2vulkan::reflect::RuntimeStorageImageState {
                    format: runtime_storage_format(request.format).map_err(|detail| {
                        M2vCacheDecline::RuntimeStorageImageSpecialize { detail }
                    })?,
                    capabilities: metal2vulkan::reflect::RuntimeStorageImageCapabilities {
                        storage_image: request.storage_image,
                        storage_image_atomic: request.storage_image_atomic,
                        read_without_format: request.read_without_format,
                        write_without_format: request.write_without_format,
                    },
                },
            )
            .map_err(|detail| M2vCacheDecline::RuntimeStorageImageSpecialize { detail })?;
    }
    let (spirv, reflection) = metal2vulkan::translate_reflected_with_options(
        path.to_str().unwrap_or("k.air"),
        Stage::Kernel,
        &tmp,
        options,
    )
    .map_err(|detail| M2vCacheDecline::RuntimeStorageImageSpecialize { detail })?;
    if reflection.local_size != Some(source.local_size)
        || reflection.runtime_storage_image_specializations.len() != unique.len()
    {
        return Err(M2vCacheDecline::RuntimeStorageImageSpecialize {
            detail: format!(
                "translator reflected local_size={:?} and {} of {} storage states",
                reflection.local_size,
                reflection.runtime_storage_image_specializations.len(),
                unique.len()
            ),
        });
    }
    for request in &unique {
        let expected = metal2vulkan::reflect::RuntimeStorageImageState {
            format: runtime_storage_format(request.format)
                .map_err(|detail| M2vCacheDecline::RuntimeStorageImageSpecialize { detail })?,
            capabilities: metal2vulkan::reflect::RuntimeStorageImageCapabilities {
                storage_image: request.storage_image,
                storage_image_atomic: request.storage_image_atomic,
                read_without_format: request.read_without_format,
                write_without_format: request.write_without_format,
            },
        };
        let reflected = reflection
            .runtime_storage_image_specializations
            .iter()
            .find(|specialization| specialization.metal_index == request.metal_index);
        if reflected.map(|specialization| specialization.state) != Some(expected) {
            return Err(M2vCacheDecline::RuntimeStorageImageSpecialize {
                detail: format!(
                    "translator reflected a different runtime state for Metal texture {}",
                    request.metal_index
                ),
            });
        }
    }
    finish_translated(spirv, reflection, Stage::Kernel, None, None)
}

impl CachedShader {
    /// Size of the translated executable module for diagnostics.
    pub fn module_byte_len(&self) -> usize {
        self.spirv.len()
    }

    /// Analyze one storage binding in the canonical translated module.
    pub fn storage_image_access(
        &self,
        binding: u32,
    ) -> Option<reims_vgpu_core::StorageImageAccess> {
        crate::spirv_bind::storage_image_access(&self.words, binding)
    }

    /// Statically-used sampled bindings absent from `bound`.
    pub fn null_sampled_image_bindings(&self, bound: &[u32]) -> Vec<u32> {
        crate::spirv_bind::reflected_null_sampled_image_bindings(
            &self.reflection,
            &self.words,
            bound,
        )
    }

    /// Sampler interface of a kernel module in its reflected layout.
    pub fn kernel_samplers(&self) -> Arc<[reims_vgpu_core::ReflectedSamplerDescriptor]> {
        self.variant().samplers.clone()
    }

    /// Translate runtime storage-image state through metal2vulkan and publish
    /// the exact reflected module behind its semantic prepared identity.
    pub fn prepare_kernel(
        self: &Arc<Self>,
        requests: &[RuntimeStorageImageRequest],
    ) -> M2vResult<PreparedKernelVariant> {
        let cached = (!requests.is_empty())
            .then(|| {
                self.kernel_specializations
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .get(requests)
                    .cloned()
            })
            .flatten();
        let owner = if requests.is_empty() {
            Arc::clone(self)
        } else if let Some(cached) = cached {
            cached
        } else {
            let source = self.kernel_source.as_ref().ok_or_else(|| {
                M2vCacheDecline::RuntimeStorageImageSpecialize {
                    detail: "kernel translation source lifetime ended".to_string(),
                }
            })?;
            let specialized = Arc::new(translate_kernel_with_storage(source, requests)?);
            self.kernel_specializations
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(requests.to_vec(), Arc::clone(&specialized));
            specialized
        };
        let storage_formats = requests
            .iter()
            .map(|request| {
                let specialization = owner
                    .reflection
                    .runtime_storage_image_specializations
                    .iter()
                    .find(|state| state.metal_index == request.metal_index);
                (
                    request.binding,
                    specialization
                        .and_then(|state| state.spirv_format)
                        .map(reflected_storage_format),
                )
            })
            .collect();
        Ok(PreparedKernelVariant {
            variant: owner.variant(),
            storage_formats,
            _owner: owner,
        })
    }

    /// Whether translation retained the source module's data-layout contract.
    pub fn source_datalayout_present(&self) -> bool {
        self.reflection.datalayout.is_some()
    }

    /// Materialize the translator-validated module and its reflection without
    /// changing either. Descriptor numbering is selected before translation
    /// and the effective layout is carried by the reflection.
    pub fn new(spirv: Vec<u8>, reflection: Arc<ShaderReflection>) -> Self {
        Self::new_with_source(spirv, reflection, None, None)
    }

    fn new_with_source(
        spirv: Vec<u8>,
        reflection: Arc<ShaderReflection>,
        render_source: Option<RenderTranslationSource>,
        kernel_source: Option<KernelTranslationSource>,
    ) -> Self {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let interface = Arc::new(project_shader_interface(&reflection));
        Self {
            spirv,
            reflection,
            interface,
            words: Arc::new(words),
            base: OnceLock::new(),
            render_source,
            render_specializations: Mutex::new(HashMap::new()),
            kernel_source,
            kernel_specializations: Mutex::new(HashMap::new()),
        }
    }

    /// Executable module and reflected resources in the selected layout.
    pub fn variant(&self) -> Arc<ShaderVariant> {
        self.base
            .get_or_init(|| {
                ShaderVariant::of(
                    self.words.clone(),
                    crate::spirv_bind::reflected_sampler_descriptors(&self.reflection).into(),
                    crate::spirv_vertex_input::VertexInputWidths::from_reflection(
                        &self.reflection.vertex_attributes,
                    ),
                    crate::spirv_bind::reflected_descriptor_bindings(&self.reflection).into(),
                    self.reflection
                        .kernel_dispatch
                        .and_then(metal2vulkan::reflect::KernelDispatch::push_constant_range)
                        .map(KernelGridRange::from),
                )
            })
            .clone()
    }
}

/// Translated shaders, keyed by the content hash of the AIR the guest supplied.
///
/// **Deliberately unbounded.** The key is content, so the live entry count is the
/// number of *distinct* shaders the guest has ever compiled — a property of the
/// guest's own program set, not of how long the device has run. A driven x86
/// boot with a window-drag probe against Safari settles at 75 and stays there
/// for the rest of the run, read off the `object_cache_levels` census; an idle
/// boot reaches the same. That is the same bound a real driver's pipeline cache
/// has, and it is why no reclaim rule is needed here.
///
/// It used to hold 256 entries and evict in insertion order. Both halves were
/// wrong for this workload. Insertion order makes the *first* shader compiled —
/// the compositor's, drawn every frame for the life of the boot — the first
/// victim, so crossing the cap evicts the hot set and nothing else. And a miss is
/// not free: the module header above records why re-translating on the doorbell
/// vCPU holds guest CPUs long enough to trip `pmap_flush_tlbs` IPI-timeout
/// panics. A cap whose crossing degrades the guest is a failure mode bought for
/// a bound the workload never approached.
#[derive(Default)]
struct Cache {
    /// `hash(stage_tag || air_bytes)` → every shader filed under that digest.
    ///
    /// A bucket holds more than one entry only on a digest collision, and
    /// [`Slot::is`] is what decides a hit — see [`ShaderId`] for why the digest
    /// is not allowed to.
    entries: HashMap<u64, Vec<Slot>>,
    hits: u64,
    misses: u64,
    async_queue: VecDeque<TranslationTask>,
    async_worker_running: bool,
}

#[derive(Clone)]
enum Entry {
    Loading,
    Ready(Arc<CachedShader>),
    Failed(M2vCacheDecline),
}

/// One cached translation and the identity it was filed under.
///
/// The AIR is retained. That is the whole cost of the identity compare, and it
/// is a good deal smaller than [`Cache`]'s own doc once implied: a
/// [`CachedShader`] already holds the module as `spirv` bytes *and* as `words`,
/// so one copy of the AIR that produced them is a fraction of the entry rather
/// than a doubling of it.
struct Slot {
    stage: u8,
    /// `Some` for a kernel, whose LocalSize is baked into the SPIR-V, so one
    /// AIR blob dispatched at two geometries is two shaders. `None` for a render
    /// stage, which has no such parameter.
    local_size: Option<[u32; 3]>,
    raster_sample_count: Option<u32>,
    air: Arc<[u8]>,
    entry: Entry,
}

impl Slot {
    /// The full identity compare. This alone decides a hit.
    fn is(&self, id: ShaderId<'_>) -> bool {
        self.stage == id.stage
            && self.local_size == id.local_size
            && self.raster_sample_count == id.raster_sample_count
            && *self.air == *id.air
    }
}

/// What makes two translations the same translation: the whole of what the
/// guest supplied, borrowed for a lookup.
///
/// # The digest narrows the bucket and decides nothing
///
/// [`Cache::entries`] used to be keyed on `digest` alone and hold no copy of the
/// AIR, so a lookup that landed on an entry returned it — there was no
/// confirmation step. Two distinct AIR blobs colliding would hand the guest a
/// shader it never compiled, silently and with no failure line, which is a worse
/// outcome than any eviction this device can make.
///
/// That was argued as acceptable against the birthday bound — `n² / 2^65` for
/// the 75 distinct shaders a driven boot settles at, about `2e-16`. The argument
/// is arithmetically right and it is the wrong shape: it prices a failure mode
/// instead of removing one, and the price it quotes is not one this device is
/// able to observe if it is ever wrong. Retaining the AIR removes the class
/// outright, which is what the old doc itself named as the answer, and it is
/// cheaper than a wider hash because a wider hash only moves the exponent.
///
/// It is also the shape used by the Vulkan resident caches: a digest selects a
/// candidate and retained content decides the hit.
///
/// This doc used to close by calling itself "the one digest-keyed cache in the
/// crate that trusted its key", on the strength of a sweep run when it was
/// fixed. **The sweep was wrong by three, and how it went wrong is the useful
/// *length*, and its own doc argued that carrying the length made it an
/// identity — so a reader auditing for "digest alone" read the length as the
/// confirming compare and moved on. It is not one. It makes a collision need
/// equal lengths, which narrows the population by a factor and removes nothing.
/// Three caches keyed on it. When auditing this class, the question is not
/// whether the key has a second field; it is whether **anything retained the
/// bytes**.
///
/// Borrowed rather than owned because a lookup happens per pipeline build and an
/// owned key would allocate a copy of the AIR to throw away on every hit. Only
/// [`Cache::put`] takes ownership, once per distinct shader.
#[derive(Clone, Copy)]
struct ShaderId<'a> {
    digest: u64,
    stage: u8,
    local_size: Option<[u32; 3]>,
    raster_sample_count: Option<u32>,
    air: &'a [u8],
}

impl<'a> ShaderId<'a> {
    fn render(stage: Stage, air: &'a [u8], raster_sample_count: u32) -> Self {
        let raster_sample_count = effective_raster_sample_count(stage, raster_sample_count);
        Self {
            digest: air_key(stage, air, raster_sample_count),
            stage: stage_tag(stage),
            local_size: None,
            raster_sample_count: Some(raster_sample_count),
            air,
        }
    }

    fn kernel(air: &'a [u8], local_size: [u32; 3]) -> Self {
        Self {
            digest: air_key_kernel(air, local_size),
            stage: stage_tag(Stage::Kernel),
            local_size: Some(local_size),
            raster_sample_count: None,
            air,
        }
    }
}

impl Cache {
    /// The entry filed under `id`, if this cache holds one.
    fn find(&self, id: ShaderId<'_>) -> Option<&Entry> {
        self.entries
            .get(&id.digest)?
            .iter()
            .find(|s| s.is(id))
            .map(|s| &s.entry)
    }

    /// File `entry` under `id`, replacing whatever that identity held.
    ///
    /// Replacing rather than pushing is the `Loading` -> `Ready`/`Failed`
    /// transition, which is the common case: the admission puts `Loading` and
    /// the worker puts the result under the same identity. Pushing there would
    /// leave the `Loading` slot in front of the `Ready` one, where `find`
    /// reaches it first and every later ask reports the translation still
    /// pending.
    fn put(&mut self, id: ShaderId<'_>, air: &Arc<[u8]>, entry: Entry) {
        let bucket = self.entries.entry(id.digest).or_default();
        match bucket.iter_mut().find(|s| s.is(id)) {
            Some(slot) => slot.entry = entry,
            None => bucket.push(Slot {
                stage: id.stage,
                local_size: id.local_size,
                raster_sample_count: id.raster_sample_count,
                air: Arc::clone(air),
                entry,
            }),
        }
    }

    /// Drop the entry filed under `id`, and the bucket with it if it was the
    /// last — so a forgotten transient failure leaves nothing behind to walk.
    fn forget(&mut self, id: ShaderId<'_>) {
        let std::collections::hash_map::Entry::Occupied(mut bucket) = self.entries.entry(id.digest)
        else {
            return;
        };
        bucket.get_mut().retain(|s| !s.is(id));
        if bucket.get().is_empty() {
            bucket.remove();
        }
    }

    /// Live entries across every bucket.
    ///
    /// The level `object_cache_levels` publishes. It sums rather than counting
    /// buckets because those differ exactly when a digest collides, which is the
    /// event this cache now survives and must not under-report.
    fn len(&self) -> usize {
        self.entries.values().map(Vec::len).sum()
    }
}

struct TranslationTask {
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    raster_sample_count: Option<u32>,
    /// Shared with the [`Slot`] the admission filed, so the identity the worker
    /// completes under is the same bytes the admission was asked about and not a
    /// second copy that could differ.
    air: Arc<[u8]>,
    pipeline_ref: u32,
}

impl TranslationTask {
    fn id(&self) -> ShaderId<'_> {
        match self.kernel_local_size {
            Some(local_size) => ShaderId::kernel(&self.air, local_size),
            None => ShaderId::render(
                self.stage,
                &self.air,
                self.raster_sample_count
                    .expect("render translation has sample count"),
            ),
        }
    }
}

fn global() -> &'static Mutex<Cache> {
    static C: std::sync::OnceLock<Mutex<Cache>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(Cache::default()))
}

fn stage_tag(stage: Stage) -> u8 {
    match stage {
        Stage::Vertex => 1,
        Stage::Fragment => 2,
        Stage::Kernel => 3,
    }
}

fn effective_raster_sample_count(stage: Stage, raster_sample_count: u32) -> u32 {
    if stage == Stage::Fragment {
        raster_sample_count
    } else {
        1
    }
}

/// The bucket digest for a non-kernel shader: 64 bits of SipHash over stage +
/// AIR.
///
/// # This narrows the search and decides nothing
///
/// It is not the cache key. [`ShaderId`] is, and [`Slot::is`] compares the AIR
/// itself on every hit, so a collision here costs a two-entry walk and cannot
/// cost correctness. Read [`ShaderId`] before changing this function: what it
/// must be is *stable within one process*, and what it must not be is *trusted*.
///
/// This doc used to argue the opposite — that the digest was the identity, and
/// that the birthday bound (`n² / 2^65`, about `2e-16` at the 75 distinct
/// shaders a driven boot settles at) made the resulting wrong-shader hazard
/// acceptable. The arithmetic was right. Pricing a silent failure this device
/// cannot observe was not, and the same doc already named retaining the AIR as
/// the answer.
///
/// `air.hash()` writes a length prefix before the bytes, per `Hash for [u8]`, so
/// two blobs of different lengths are not merely unlikely to share a bucket —
/// the length is in the digest too. That is now a statement about walk length
/// rather than about correctness.
fn air_key(stage: Stage, air: &[u8], raster_sample_count: u32) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(stage).hash(&mut h);
    air.hash(&mut h);
    effective_raster_sample_count(stage, raster_sample_count).hash(&mut h);
    h.finish()
}

/// The bucket digest for a kernel, which includes LocalSize because the SPIR-V
/// workgroup size is baked in at translate.
///
/// `local_size` is in the digest *and* in [`ShaderId`], and it has to be in both
/// for different reasons: in the identity because one AIR blob dispatched at two
/// geometries is genuinely two shaders, and in the digest so the two do not
/// share a bucket and pay a walk on every hit. Leaving it out of the identity
/// would be the wrong-shader hazard [`air_key`] describes, except reachable by a
/// guest doing something entirely ordinary rather than by chance.
fn air_key_kernel(air: &[u8], local_size: [u32; 3]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(Stage::Kernel).hash(&mut h);
    air.hash(&mut h);
    local_size.hash(&mut h);
    h.finish()
}

/// Process-local temp dir for m2v path-based translate (reused across draws).
fn tmp_dir() -> PathBuf {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let p = std::env::temp_dir().join(format!("reims-vgpu-m2v-cache-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&p);
        p
    })
    .clone()
}

fn translation_lock() -> &'static Mutex<()> {
    static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Independently translated graphics stages share set zero, but never share a
/// binding range. The translator validates this complete layout and reflects
/// the exact result beside the module.
fn render_descriptor_layout(stage: Stage) -> metal2vulkan::reflect::DescriptorLayout {
    use metal2vulkan::reflect::{DescriptorBindingRange, DescriptorLayout};

    let layout = DescriptorLayout::default();
    if stage != Stage::Fragment {
        return layout;
    }
    let fragment_base = [
        layout.buffers.end,
        layout.sampled_textures.end,
        layout.samplers.end,
        layout.color_inputs.end,
        layout.imageblocks.end,
        layout.fragment_imageblocks.end,
        layout.storage_textures.end,
        layout.synthetic.end,
    ]
    .into_iter()
    .max()
    .expect("descriptor layout has fixed resource classes");
    let shifted = |range: DescriptorBindingRange| DescriptorBindingRange {
        start: range
            .start
            .checked_add(fragment_base)
            .expect("validated default descriptor layout fits a second stage"),
        end: range
            .end
            .checked_add(fragment_base)
            .expect("validated default descriptor layout fits a second stage"),
    };
    DescriptorLayout {
        buffers: shifted(layout.buffers),
        sampled_textures: shifted(layout.sampled_textures),
        samplers: shifted(layout.samplers),
        // Color inputs exist only in fragment shaders, so they cannot collide
        // with the independently translated vertex stage. Keeping their
        // default range also keeps the engine's attachment write at the exact
        // translator-owned ABI binding.
        color_inputs: layout.color_inputs,
        imageblocks: shifted(layout.imageblocks),
        fragment_imageblocks: shifted(layout.fragment_imageblocks),
        storage_textures: shifted(layout.storage_textures),
        synthetic: shifted(layout.synthetic),
        ..layout
    }
}

/// Complete translator-owned descriptor layout for one independently
/// translated graphics stage.
pub fn descriptor_layout_for_render_stage(
    stage: RenderTranslationStage,
) -> metal2vulkan::reflect::DescriptorLayout {
    render_descriptor_layout(stage.into())
}

fn translate_air(air: &[u8], stage: Stage, raster_sample_count: u32) -> M2vResult<CachedShader> {
    // metal2vulkan's tool boundary uses fixed scratch names inside `tmp_dir`.
    // Serialize sync and background calls so their AIR/LLVM/SPIR-V files never
    // alias one another.
    let _guard = translation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tmp_dir();
    let name = match stage {
        Stage::Vertex => "v.air",
        Stage::Fragment => "f.air",
        Stage::Kernel => "k.air",
    };
    let path = tmp.join(name);
    std::fs::write(&path, air).map_err(|e| scratch_write_decline(stage, e.to_string()))?;
    // Reflected translate: translator-validated SPIR-V plus the stage-interface
    // facade derived from the same AIR parse.
    let mut options = metal2vulkan::passes::TransformOptions::default()
        .with_descriptor_layout(render_descriptor_layout(stage))
        .map_err(|error| translate_decline(stage, error.to_string()))?;
    if stage == Stage::Fragment {
        options.raster_sample_count = Some(raster_sample_count);
    }
    let (spirv, reflection) = metal2vulkan::translate_reflected_with_options(
        path.to_str().unwrap_or(name),
        stage,
        &tmp,
        options,
    )
    .map_err(|e| translate_decline(stage, e.to_string()))?;
    finish_translated(
        spirv,
        reflection,
        stage,
        Some(RenderTranslationSource {
            air: Arc::from(air),
            stage,
            raster_sample_count,
        }),
        None,
    )
}

fn translate_kernel_air(air: &[u8], local_size: [u32; 3]) -> M2vResult<CachedShader> {
    let _guard = translation_lock().lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tmp_dir();
    let path = tmp.join("k.air");
    std::fs::write(&path, air).map_err(|e| M2vCacheDecline::KernelScratchWrite {
        detail: e.to_string(),
    })?;
    let opts = metal2vulkan::passes::TransformOptions {
        kernel_local_size: local_size,
        ..Default::default()
    };
    let (spirv, reflection) = metal2vulkan::translate_reflected_with_options(
        path.to_str().unwrap_or("k.air"),
        Stage::Kernel,
        &tmp,
        opts,
    )
    .map_err(|e| M2vCacheDecline::KernelTranslate {
        detail: e.to_string(),
    })?;
    if reflection.local_size != Some(local_size) {
        return Err(M2vCacheDecline::KernelLocalSizeMismatch {
            requested: local_size,
            reflected: reflection.local_size,
        });
    }
    finish_translated(
        spirv,
        reflection,
        Stage::Kernel,
        None,
        Some(KernelTranslationSource {
            air: Arc::from(air),
            local_size,
        }),
    )
}

/// Package the translator-validated module with reflection produced by the
/// same translation and options.
fn finish_translated(
    spirv: Vec<u8>,
    reflection: ShaderReflection,
    stage: Stage,
    render_source: Option<RenderTranslationSource>,
    kernel_source: Option<KernelTranslationSource>,
) -> M2vResult<CachedShader> {
    validate_reflection(&reflection, stage)?;
    // Once per translate, surface runtime function constants for which the
    // paravirt stream supplied no values. The translator exposes exact byte
    // specialization, but inventing those bytes here would violate the guest
    // contract. Silent for shaders without function constants.
    crate::spirv_bind::log_unavailable_function_constants(&reflection);
    Ok(CachedShader::new_with_source(
        spirv,
        Arc::new(reflection),
        render_source,
        kernel_source,
    ))
}

/// Once-per-translate (miss path) well-formedness guard on the AIR-derived
/// reflection. This is the always-on regression proxy for the reflection-fed hot
/// path: it validates the reflection's ABI version and its internal
/// sampled-vs-storage consistency without a second walk of the SPIR-V. Must read
/// zero on a healthy boot.
fn validate_reflection(reflection: &ShaderReflection, stage: Stage) -> M2vResult<()> {
    // pipeline_ref is telemetry only here; the stage tag localizes the shader.
    let pipe = stage_tag(stage) as u32;
    let violations = crate::spirv_bind::census_reflection_wellformed(reflection, pipe);
    if violations != 0 {
        return Err(M2vCacheDecline::ReflectionMalformed {
            stage: stage_name(stage),
            violations,
        });
    }
    Ok(())
}

/// Start translating a render stage without holding protocol state or the
/// sole FIFO scheduler. Returns true when the content is already resolved
/// (success or deterministic failure), false while the background worker owns
/// it. Callers keep the guest packet at the channel head and retry on poll.
pub fn ensure_cached_async(
    air: &[u8],
    stage: Stage,
    raster_sample_count: u32,
    pipeline_ref: u32,
) -> bool {
    ensure_cached_async_keyed(
        ShaderId::render(stage, air, raster_sample_count),
        stage,
        None,
        Some(raster_sample_count),
        pipeline_ref,
    )
}

/// Start translating a render stage without exposing translator-native types
/// to the device composition layer.
pub fn ensure_render_cached_async(
    air: &[u8],
    stage: RenderTranslationStage,
    raster_sample_count: u32,
    pipeline_ref: u32,
) -> bool {
    ensure_cached_async(air, stage.into(), raster_sample_count, pipeline_ref)
}

/// Kernel counterpart to [`ensure_cached_async`]. LocalSize is part of both
/// the translation options and cache key, so two dispatch geometries can never
/// alias one another.
pub fn ensure_cached_kernel_async(air: &[u8], local_size: [u32; 3], pipeline_ref: u32) -> bool {
    if local_size.contains(&0) {
        return true;
    }
    ensure_cached_async_keyed(
        ShaderId::kernel(air, local_size),
        Stage::Kernel,
        Some(local_size),
        None,
        pipeline_ref,
    )
}

fn ensure_cached_async_keyed(
    id: ShaderId<'_>,
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    raster_sample_count: Option<u32>,
    pipeline_ref: u32,
) -> bool {
    let mut start_worker = false;
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id) {
            // A stored failure resolves the ask, transient or not, and that is
            // load-bearing: the caller holds the guest packet at the channel
            // head and re-polls until this answers `true`, so an arm that
            // re-queued instead would never let the packet advance. The retry
            // is armed where the error is *consumed* — see the `Entry::Failed`
            // arm of [`translate_cached_reflected`] — which costs the failing
            // draw its failure and leaves the next draw a clean cache.
            Some(Entry::Ready(_)) | Some(Entry::Failed(_)) => return true,
            Some(Entry::Loading) => return false,
            None => {}
        }
        let air: Arc<[u8]> = Arc::from(id.air);
        c.put(id, &air, Entry::Loading);
        c.misses = c.misses.saturating_add(1);
        c.async_queue.push_back(TranslationTask {
            stage,
            kernel_local_size,
            raster_sample_count,
            air,
            pipeline_ref,
        });
        if !c.async_worker_running {
            c.async_worker_running = true;
            start_worker = true;
        }
    }
    let dims = kernel_local_size
        .map(|d| format!(" tg=[{},{},{}]", d[0], d[1], d[2]))
        .unwrap_or_default();
    // Queue census: fires once per cold shader on the normal path — OFF, not a
    // curated failure. The `done` line below stays fail-visible on translation
    // failure (a shader that will not render).
    reims_vgpu_observe::off(format!(
        "linux_m2v_async queued pipe={pipeline_ref} stage={stage:?}{dims} air={}",
        id.air.len()
    ));
    if start_worker {
        std::thread::spawn(async_worker);
    }
    false
}

fn async_worker() {
    loop {
        let task = {
            let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
            match c.async_queue.pop_front() {
                Some(task) => task,
                None => {
                    c.async_worker_running = false;
                    return;
                }
            }
        };
        let result = match task.kernel_local_size {
            Some(local_size) => translate_kernel_air(&task.air, local_size),
            None => translate_air(
                &task.air,
                task.stage,
                task.raster_sample_count
                    .expect("render translation has sample count"),
            ),
        };
        let (hits, misses, detail, failure) = {
            let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
            let (detail, failure) = match result {
                Ok(shader) => {
                    let len = shader.spirv.len();
                    // Report the vertex builtins the translated shader uses.
                    // instance_index presence proves an instanced draw's
                    // per-instance indexing survived AIR->SPIR-V; its absence on a
                    // shader driven by an instanced draw would localize a dropped
                    // [[instance_id]]. Load-bearing for instanced-draw diagnosis.
                    // The reflection facade reports it from the SAME parsed AIR
                    // roles the emitter decorated from — single source of truth.
                    if task.stage == Stage::Vertex {
                        let vb = shader.reflection.vertex_builtins.unwrap_or_default();
                        // Decoded vertex-builtin usage census (not an error);
                        // route off() so it leaves the curated real-error view.
                        // The genuine reflect-vs-emit divergence above stays
                        // fail()-visible.
                        reims_vgpu_observe::off(format!(
                            "m2v_builtins pipe={} stage=Vertex instance_index={} vertex_index={}",
                            task.pipeline_ref,
                            vb.uses_instance_index as u8,
                            vb.uses_vertex_index as u8
                        ));
                    }
                    c.put(task.id(), &task.air, Entry::Ready(Arc::new(shader)));
                    (format!("ok spv={len}"), None)
                }
                Err(e) => {
                    c.put(task.id(), &task.air, Entry::Failed(e.clone()));
                    ("fail".to_string(), Some(e))
                }
            };
            (c.hits, c.misses, detail, failure)
        };
        let dims = task
            .kernel_local_size
            .map(|d| format!(" tg=[{},{},{}]", d[0], d[1], d[2]))
            .unwrap_or_default();
        // A successful translate is census (OFF); a failed one means the shader
        // will not render — stay fail-visible.
        if let Some(error) = failure {
            let mut emit = reims_vgpu_observe::Emit::decline("linux_m2v_async_done", &error)
                .field("pipe", task.pipeline_ref)
                .field("hits", hits)
                .field("misses", misses);
            if let Some([x, y, z]) = task.kernel_local_size {
                emit = emit.field("tg_x", x).field("tg_y", y).field("tg_z", z);
            }
            emit.fail_once(task.id().digest);
        } else {
            let done = format!(
                "linux_m2v_async done pipe={} stage={:?}{dims} {detail} hits={hits} misses={misses}",
                task.pipeline_ref, task.stage
            );
            reims_vgpu_observe::off(done);
        }
    }
}

/// Translate `air` for `stage`, returning cached SPIR-V when AIR matches a prior
/// translate. Logs `linux_m2v_translate` on miss and `linux_m2v_translate_hit`
/// on hit (pipe id is telemetry only).
/// Translate `air` for `stage`, returning the whole [`CachedShader`] (SPIR-V +
/// reflection) as a shared handle, so a consumer reads stage-interface facts
/// (texture shapes, vertex builtins, descriptor bindings) from the reflection
/// instead of re-walking the emitted SPIR-V. The returned `Arc` is a clone of the
/// cached entry — a warm hit performs no allocation beyond the refcount bump.
pub fn translate_cached_reflected(
    air: &[u8],
    stage: Stage,
    raster_sample_count: u32,
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    let id = ShaderId::render(stage, air, raster_sample_count);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id).cloned() {
            Some(Entry::Ready(shader)) => {
                c.hits = c.hits.saturating_add(1);
                let hits = c.hits;
                let misses = c.misses;
                drop(c);
                // A cache HIT is the per-draw hot path (≈2×/draw) and carries only a
                // cumulative counter — logging it always-on flooded /tmp/reims-vgpu-fail.log
                // (~1.1M lines / boot, drowning real failures) and paid a file write
                // per draw on the render path. Verbose-gated only; the miss/`ok` line
                // below stays always-on so cache lifecycle is still visible.
                if reims_vgpu_observe::draw_log_enabled() {
                    reims_vgpu_observe::line(format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage={stage:?} spv={} hits={hits} misses={misses}",
                        shader.spirv.len()
                    ));
                }
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => {
                forget_if_transient(&mut c, id, &e);
                return Err(e);
            }
            Some(Entry::Loading) => {
                return Err(M2vCacheDecline::TranslationPending {
                    stage: stage_name(stage),
                })
            }
            None => {}
        }
    }

    let shader = Arc::new(translate_air(air, stage, raster_sample_count)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        c.put(id, &Arc::from(air), Entry::Ready(Arc::clone(&shader)));
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        reims_vgpu_observe::fail(format!(
            "linux_m2v_translate ok pipe={pipeline_ref} stage={stage:?} v_spv_or_f={} hits={hits} misses={misses}",
            shader.spirv.len()
        ));
    }
    Ok(shader)
}

/// Resolve a translated render shader through the backend-owned stage type.
pub fn translate_render_cached_reflected(
    air: &[u8],
    stage: RenderTranslationStage,
    raster_sample_count: u32,
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    translate_cached_reflected(air, stage.into(), raster_sample_count, pipeline_ref)
}

/// Translate a **compute** AIR kernel with explicit LocalSize (threadgroup dims),
/// returning the whole [`CachedShader`] (SPIR-V + reflection). See
/// [`translate_cached_reflected`].
///
/// Wallpaper CI dispatches use non-default tg sizes (e.g. 16×16×1). Default
/// m2v LocalSize is 64×1×1 — wrong for those grids. Cache key includes LocalSize.
pub fn translate_cached_kernel_reflected(
    air: &[u8],
    local_size: [u32; 3],
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    if local_size.contains(&0) {
        return Err(M2vCacheDecline::KernelLocalSizeZero { local_size });
    }
    let id = ShaderId::kernel(air, local_size);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.find(id).cloned() {
            Some(Entry::Ready(shader)) => {
                c.hits = c.hits.saturating_add(1);
                let hits = c.hits;
                let misses = c.misses;
                drop(c);
                // Verbose-gated (see `translate_cached_reflected`): a compute-kernel cache hit
                // is the hot path and only carries a cumulative counter.
                if reims_vgpu_observe::draw_log_enabled() {
                    reims_vgpu_observe::line(format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage=Kernel tg=[{},{},{}] spv={} hits={hits} misses={misses}",
                        local_size[0], local_size[1], local_size[2], shader.spirv.len()
                    ));
                }
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => {
                forget_if_transient(&mut c, id, &e);
                return Err(e);
            }
            Some(Entry::Loading) => {
                return Err(M2vCacheDecline::TranslationPending { stage: "kernel" })
            }
            None => {}
        }
    }

    let shader = Arc::new(translate_kernel_air(air, local_size)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        c.put(id, &Arc::from(air), Entry::Ready(Arc::clone(&shader)));
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        reims_vgpu_observe::fail(format!(
            "linux_m2v_translate ok pipe={pipeline_ref} stage=Kernel tg=[{},{},{}] spv={} hits={hits} misses={misses}",
            local_size[0], local_size[1], local_size[2], shader.spirv.len()
        ));
    }
    Ok(shader)
}

/// Snapshot counters for tests.
///
/// The census does not come through here — the `linux_m2v_translate ok` line
/// above reads `hits`/`misses` off the lock it already holds. This is only how a
/// test asks the same question from outside, which is why it cannot be
/// `#[cfg(test)]`: `tests/reflection_adoption.rs` is a separate crate.
pub fn stats() -> (u64, u64, usize) {
    let c = global().lock().unwrap_or_else(|e| e.into_inner());
    (c.hits, c.misses, c.len())
}

/// Test isolation.
#[cfg(test)]
pub fn reset_for_test() {
    let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
    c.entries.clear();
    c.hits = 0;
    c.misses = 0;
    c.async_queue.clear();
    c.async_worker_running = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn empty_reflection(
        stage: metal2vulkan::reflect::ShaderStage,
        datalayout: Option<&str>,
    ) -> metal2vulkan::reflect::ShaderReflection {
        use metal2vulkan::reflect::{ShaderReflection, REFLECTION_VERSION};
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage,
            entry_point: None,
            kernel_dispatch: None,
            bindings: vec![],
            argument_buffer_fields: vec![],
            vertex_attributes: vec![],
            varyings: vec![],
            render_targets: vec![],
            depth_members: vec![],
            depth_qualifier: None,
            stencil_members: vec![],
            local_size: (stage == metal2vulkan::reflect::ShaderStage::Kernel).then_some([1, 1, 1]),
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: vec![],
            implicit_imageblock_attachments: vec![],
            fragment_imageblock: None,
            descriptor_layout: metal2vulkan::reflect::DescriptorLayout::default(),
            datalayout: datalayout.map(str::to_owned),
            runtime_sampler_specializations: vec![],
            runtime_storage_image_specializations: vec![],
            function_constants: vec![],
        }
    }

    #[test]
    fn reflected_sampler_bindings_are_the_executable_interface() {
        let translator_base = metal2vulkan::reflect::SAMPLER_BINDING_BASE;
        let words = crate::spirv_bind::test_module_with_samplers(&[
            translator_base + 3,
            translator_base + 7,
            translator_base + 11,
        ]);
        let spirv: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader_with_samplers(Stage::Fragment, spirv, &[3, 7, 11]);

        let variant = shader.variant();
        assert_eq!(variant.samplers.len(), 3);
        assert_eq!(
            variant
                .samplers
                .iter()
                .map(|sampler| sampler.binding)
                .collect::<Vec<_>>(),
            vec![
                translator_base + 3,
                translator_base + 7,
                translator_base + 11
            ]
        );
    }

    /// A variant is computed once and handed out by pointer after that, which is
    /// what makes the walk a per-shader cost rather than a per-draw one.
    #[test]
    fn a_variant_is_memoized_rather_than_rewalked() {
        let words = crate::spirv_bind::test_module_with_samplers(&[5]);
        let spirv: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader(Stage::Fragment, spirv);
        let first = shader.variant();
        let again = shader.variant();
        assert!(Arc::ptr_eq(&first, &again));
    }

    #[test]
    fn prepared_identity_resolves_only_for_the_variant_lifetime() {
        let shader = synth_shader(Stage::Vertex, Vec::new());
        let variant = shader.variant();
        let stage = prepared_stage(&variant);

        let resolved = resolve_prepared_shader(stage.id).expect("live variant must resolve");
        assert!(Arc::ptr_eq(&variant, &resolved));
        assert_eq!(
            stage.used_descriptor_bindings,
            variant.used_descriptor_bindings
        );

        drop(resolved);
        drop(variant);
        drop(shader);
        assert!(
            resolve_prepared_shader(stage.id).is_none(),
            "prepared identity must not extend native module ownership"
        );
    }

    /// Static use belongs to the executable variant, not to each draw that
    /// binds it. A declaration alone is legal to omit from a Vulkan layout;
    /// only an instruction reference enters the retained guard set.
    #[test]
    fn a_variant_retains_only_statically_used_descriptor_bindings() {
        let unused = crate::spirv_bind::test_support::module_with_descriptor(33, false);
        let used = crate::spirv_bind::test_support::module_with_descriptor(34, true);

        let unused_shader = synth_shader_with_resources(
            Stage::Fragment,
            unused.iter().flat_map(|word| word.to_le_bytes()).collect(),
            &[1],
            metal2vulkan::reflect::ResourceKind::Texture,
            metal2vulkan::reflect::TEXTURE_BINDING_BASE,
        );
        let used_shader = synth_shader_with_resources(
            Stage::Fragment,
            used.iter().flat_map(|word| word.to_le_bytes()).collect(),
            &[2],
            metal2vulkan::reflect::ResourceKind::Texture,
            metal2vulkan::reflect::TEXTURE_BINDING_BASE,
        );

        assert!(unused_shader.variant().used_descriptor_bindings.is_empty());
        assert_eq!(
            used_shader.variant().used_descriptor_bindings.as_ref(),
            &[34]
        );
    }

    #[test]
    fn a_missing_sampled_descriptor_uses_reflected_class_and_executable_use() {
        let words = crate::spirv_bind::test_module_with_two_sampled_images(33, 34);
        let shader = synth_shader_with_resources(
            Stage::Kernel,
            words.iter().flat_map(|word| word.to_le_bytes()).collect(),
            &[1, 2],
            metal2vulkan::reflect::ResourceKind::Texture,
            metal2vulkan::reflect::TEXTURE_BINDING_BASE,
        );

        assert_eq!(shader.null_sampled_image_bindings(&[]), vec![33]);
        assert!(shader.null_sampled_image_bindings(&[33]).is_empty());
    }

    #[test]
    fn retained_pipeline_projection_keeps_semantics_and_hides_native_words() {
        let words = crate::spirv_bind::test_support::module_with_descriptor(34, true);
        let shader = synth_shader_with_resources(
            Stage::Fragment,
            words.iter().flat_map(|word| word.to_le_bytes()).collect(),
            &[2],
            metal2vulkan::reflect::ResourceKind::Texture,
            metal2vulkan::reflect::TEXTURE_BINDING_BASE,
        );
        let family = prepare_render_shader(&shader, RenderTranslationStage::Fragment);

        let variant = family.variant();
        assert_eq!(variant.word_count as usize, words.len());
        assert_eq!(variant.declared_bindings.len(), 1);
        let binding = variant.declared_bindings[0];
        assert_eq!(
            variant.descriptor_use(binding),
            reims_vgpu_core::DescriptorUse::Used
        );
        assert!(resolve_prepared_shader(variant.program.id).is_some());
    }

    #[test]
    fn retained_buffer_projection_uses_the_executable_storage_buffer_variable() {
        let unused = crate::spirv_bind::test_support::module_with_buffer_descriptor(1, false);
        let used = crate::spirv_bind::test_support::module_with_buffer_descriptor(1, true);
        let shader = |words: &[u32]| {
            synth_shader_with_resources(
                Stage::Fragment,
                words.iter().flat_map(|word| word.to_le_bytes()).collect(),
                &[1],
                metal2vulkan::reflect::ResourceKind::Buffer,
                0,
            )
        };

        let unused = prepare_render_shader(&shader(&unused), RenderTranslationStage::Fragment);
        let used = prepare_render_shader(&shader(&used), RenderTranslationStage::Fragment);

        assert_eq!(
            unused.variant().buffer_use(1),
            reims_vgpu_core::DescriptorUse::DeclaredUnused
        );
        assert_eq!(
            used.variant().buffer_use(1),
            reims_vgpu_core::DescriptorUse::Used
        );
        assert_eq!(
            used.variant().program.used_descriptor_bindings.as_ref(),
            &[1]
        );
    }

    /// A minimal `CachedShader` wrapping raw bytes with an empty reflection —
    /// enough to prime the cache in unit tests that never call metal2vulkan.
    fn synth_shader(stage: Stage, spirv: Vec<u8>) -> Arc<CachedShader> {
        use metal2vulkan::reflect::ShaderStage;
        let stage = match stage {
            Stage::Vertex => ShaderStage::Vertex,
            Stage::Fragment => ShaderStage::Fragment,
            Stage::Kernel => ShaderStage::Kernel,
        };
        Arc::new(CachedShader::new(
            spirv,
            Arc::new(empty_reflection(stage, None)),
        ))
    }

    fn synth_shader_with_samplers(
        stage: Stage,
        spirv: Vec<u8>,
        metal_indices: &[u32],
    ) -> Arc<CachedShader> {
        synth_shader_with_resources(
            stage,
            spirv,
            metal_indices,
            metal2vulkan::reflect::ResourceKind::Sampler,
            metal2vulkan::reflect::SAMPLER_BINDING_BASE,
        )
    }

    fn synth_shader_with_resources(
        stage: Stage,
        spirv: Vec<u8>,
        metal_indices: &[u32],
        resource_kind: metal2vulkan::reflect::ResourceKind,
        binding_base: u32,
    ) -> Arc<CachedShader> {
        use metal2vulkan::reflect::{
            DescriptorLocation, ResourceBinding, ShaderReflection, ShaderStage, REFLECTION_VERSION,
            RESOURCE_DESCRIPTOR_SET,
        };
        let reflected_stage = match stage {
            Stage::Vertex => ShaderStage::Vertex,
            Stage::Fragment => ShaderStage::Fragment,
            Stage::Kernel => ShaderStage::Kernel,
        };
        let bindings = metal_indices
            .iter()
            .copied()
            .map(|metal_index| ResourceBinding {
                kind: resource_kind,
                metal_index,
                descriptor: Some(DescriptorLocation {
                    set: RESOURCE_DESCRIPTOR_SET,
                    binding: binding_base + metal_index,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: None,
            })
            .collect();
        Arc::new(CachedShader::new(
            spirv,
            Arc::new(ShaderReflection {
                reflection_version: REFLECTION_VERSION,
                stage: reflected_stage,
                entry_point: None,
                kernel_dispatch: None,
                bindings,
                argument_buffer_fields: vec![],
                vertex_attributes: vec![],
                varyings: vec![],
                render_targets: vec![],
                depth_members: vec![],
                depth_qualifier: None,
                stencil_members: vec![],
                local_size: (reflected_stage == ShaderStage::Kernel).then_some([1, 1, 1]),
                vertex_builtins: None,
                tessellation: None,
                imageblock_layouts: vec![],
                implicit_imageblock_attachments: vec![],
                fragment_imageblock: None,
                descriptor_layout: metal2vulkan::reflect::DescriptorLayout::default(),
                datalayout: None,
                runtime_sampler_specializations: vec![],
                runtime_storage_image_specializations: vec![],
                function_constants: vec![],
            }),
        ))
    }

    #[test]
    fn prepared_texture_use_is_keyed_by_metal_index() {
        const METAL_INDEX: u32 = 2;
        let binding = crate::spirv_bind::TEXTURE_BINDING_BASE + METAL_INDEX;
        let words = crate::spirv_bind::test_support::module_with_descriptor(binding, true);
        let shader = synth_shader_with_resources(
            Stage::Fragment,
            words.iter().flat_map(|word| word.to_le_bytes()).collect(),
            &[METAL_INDEX],
            metal2vulkan::reflect::ResourceKind::Texture,
            crate::spirv_bind::TEXTURE_BINDING_BASE,
        );
        let family = prepare_render_shader(&shader, RenderTranslationStage::Fragment);

        let variant = family.variant();
        assert_eq!(
            variant.texture_use(METAL_INDEX),
            reims_vgpu_core::DescriptorUse::Used
        );
        assert_eq!(
            variant.texture_use(METAL_INDEX + 1),
            reims_vgpu_core::DescriptorUse::NotDeclared
        );
        assert_eq!(variant.texture_binding(METAL_INDEX, Some(binding)), binding);
        assert_eq!(
            variant.sampler_binding(METAL_INDEX),
            metal2vulkan::reflect::SAMPLER_BINDING_BASE + METAL_INDEX
        );
        assert_eq!(variant.buffer_binding(METAL_INDEX), METAL_INDEX);
    }

    #[test]
    fn malformed_reflection_is_refused_before_it_reaches_a_binding_path() {
        let shader = synth_shader(Stage::Fragment, Vec::new());
        let mut reflection = (*shader.reflection).clone();
        reflection.reflection_version = reflection.reflection_version.wrapping_add(1);
        assert!(matches!(
            validate_reflection(&reflection, Stage::Fragment),
            Err(M2vCacheDecline::ReflectionMalformed {
                stage: "fragment",
                violations: 1,
            })
        ));
    }

    #[test]
    fn air_key_differs_by_stage_and_bytes() {
        let a = b"same-air-bytes";
        assert_ne!(air_key(Stage::Vertex, a, 1), air_key(Stage::Fragment, a, 1));
        assert_ne!(
            air_key(Stage::Vertex, a, 1),
            air_key(Stage::Vertex, b"other", 1)
        );
        assert_eq!(air_key(Stage::Vertex, a, 1), air_key(Stage::Vertex, a, 1));
        assert_eq!(
            air_key(Stage::Vertex, a, 1),
            air_key(Stage::Vertex, a, 8),
            "raster sample count is not a vertex translation option"
        );
        assert_ne!(
            air_key(Stage::Fragment, a, 1),
            air_key(Stage::Fragment, a, 8),
            "fragment sample count changes the translated interface"
        );
        assert_ne!(air_key_kernel(a, [16, 16, 1]), air_key_kernel(a, [8, 8, 1]));
    }

    #[test]
    fn render_stage_layouts_are_disjoint_and_self_describing() {
        let vertex = descriptor_layout_for_render_stage(RenderTranslationStage::Vertex);
        let fragment = descriptor_layout_for_render_stage(RenderTranslationStage::Fragment);
        assert_eq!(vertex, metal2vulkan::reflect::DescriptorLayout::default());
        assert_eq!(fragment.set, vertex.set);
        assert_eq!(fragment.color_inputs, vertex.color_inputs);
        for fragment_range in [
            fragment.buffers,
            fragment.sampled_textures,
            fragment.samplers,
            fragment.storage_textures,
            fragment.synthetic,
        ] {
            for vertex_range in [
                vertex.buffers,
                vertex.sampled_textures,
                vertex.samplers,
                vertex.storage_textures,
                vertex.synthetic,
            ] {
                assert!(
                    fragment_range.start >= vertex_range.end
                        || vertex_range.start >= fragment_range.end,
                    "fragment {fragment_range:?} overlaps vertex {vertex_range:?}"
                );
            }
        }
        fragment
            .validate()
            .expect("fragment layout is translator-valid");
    }

    #[test]
    fn runtime_sampler_state_matches_the_descriptor_contract() {
        use metal2vulkan::reflect::{SamplerCompareFunction, SamplerCoordinates};

        let mut sampler = reims_vgpu_core::SamplerResource::normalized_default(17);
        let normalized = runtime_sampler_state(&sampler).expect("default sampler is representable");
        assert_eq!(normalized.coordinates, SamplerCoordinates::Normalized);
        assert_eq!(normalized.compare_function, SamplerCompareFunction::None);

        sampler.unnormalized_coordinates = true;
        let pixel = runtime_sampler_state(&sampler).expect("pixel sampler is representable");
        assert_eq!(pixel.coordinates, SamplerCoordinates::Pixel);
        assert_eq!(pixel.compare_function, SamplerCompareFunction::None);
        assert_eq!(pixel.lod_min_clamp, 0.0);
        assert_eq!(pixel.lod_max_clamp, 0.0);
    }

    #[test]
    fn runtime_storage_format_covers_the_upstream_specialization_contract() {
        use reims_vgpu_protocol::StorageImageFormat as F;

        for format in [
            F::R8Unorm,
            F::Rgba8Unorm,
            F::Bgra8Unorm,
            F::R16Float,
            F::Rg16Float,
            F::Rgba16Float,
            F::R32Float,
            F::Rgba32Float,
            F::R16Uint,
            F::R32Uint,
            F::Rgba8Uint,
            F::Rgba16Uint,
            F::Rgba32Uint,
            F::R32Sint,
            F::Rgba8Sint,
            F::Rgba32Sint,
        ] {
            runtime_storage_format(format)
                .unwrap_or_else(|error| panic!("{format:?} lost upstream specialization: {error}"));
        }
        for sampled_only in [
            F::Rg8Unorm,
            F::Rgb9e5Ufloat,
            F::R16Unorm,
            F::Rg16Unorm,
            F::Rgba16Unorm,
            F::Rgb10a2Unorm,
            F::Bgr10a2Unorm,
            F::Rg11b10Float,
        ] {
            assert!(
                runtime_storage_format(sampled_only).is_err(),
                "{sampled_only:?} is sampled-only"
            );
        }
    }

    /// Each `local_size` component alone changes the kernel key.
    ///
    /// The assertion above varies two components at once, so a digest that fed
    /// only `local_size[0]` would satisfy it. That matters here more than the
    /// usual thoroughness argument: the translator bakes the workgroup size into
    /// the SPIR-V, so a key blind to one axis returns a shader compiled for a
    /// different one — the guest's dispatch runs, produces wrong results, and
    /// nothing anywhere reports a miss, because it was a *hit*.
    ///
    /// Also pinned: the same triple keys the same, so this cannot pass by
    /// accident on a key that simply changes every call.
    #[test]
    fn every_local_size_axis_is_in_the_kernel_key() {
        let air = b"kernel-air";
        let base = [8u32, 8, 8];
        assert_eq!(air_key_kernel(air, base), air_key_kernel(air, base));
        for axis in 0..3 {
            let mut moved = base;
            moved[axis] += 1;
            assert_ne!(
                air_key_kernel(air, base),
                air_key_kernel(air, moved),
                "local_size[{axis}] does not reach the key: {base:?} vs {moved:?}"
            );
        }
    }

    /// Every decline is classified, and the split is by what the variant names.
    ///
    /// The scratch writes are the host filesystem at one instant; everything
    /// else is a property of the AIR blob this cache is keyed by, and is reached
    /// again by every later ask.
    #[test]
    fn only_the_scratch_writes_are_transient() {
        let detail = || "ENOSPC".to_string();
        for transient in [
            M2vCacheDecline::VertexScratchWrite { detail: detail() },
            M2vCacheDecline::FragmentScratchWrite { detail: detail() },
            M2vCacheDecline::KernelScratchWrite { detail: detail() },
        ] {
            assert!(transient.is_transient(), "{transient:?} is about the host");
        }
        for permanent in [
            M2vCacheDecline::VertexTranslate { detail: detail() },
            M2vCacheDecline::FragmentTranslate { detail: detail() },
            M2vCacheDecline::KernelTranslate { detail: detail() },
            M2vCacheDecline::ReflectionMalformed {
                stage: "vertex",
                violations: 1,
            },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 1, 1],
            },
            M2vCacheDecline::RuntimeSamplerSpecialize {
                stage: "fragment",
                detail: detail(),
            },
            M2vCacheDecline::RuntimeStorageImageSpecialize { detail: detail() },
        ] {
            assert!(!permanent.is_transient(), "{permanent:?} is about the AIR");
        }
    }

    /// A scratch-write failure fails the draw that met it and then gets out of
    /// the way. The cache is unbounded and nothing evicts, so before this the
    /// entry answered every later draw of that shader for the life of the
    /// process — a full stop for one moment when the host had no room for a
    /// temp file.
    #[test]
    fn a_transient_failure_is_forgotten_once_it_has_been_answered() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"air-whose-scratch-write-failed";
        let id = ShaderId::render(Stage::Vertex, air, 1);
        let stored = M2vCacheDecline::VertexScratchWrite {
            detail: "No space left on device".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        // The admission still resolves, so the guest packet advances rather
        // than re-polling a translation that keeps failing.
        assert!(
            ensure_cached_async(air, Stage::Vertex, 1, 7),
            "a stored failure must resolve the ask, or the packet never advances"
        );

        // This draw gets the real reason, unchanged.
        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 1, 7) else {
            panic!("a stored failure still fails its draw");
        };
        assert_eq!(err, stored);

        // And the next draw finds a clean cache and translates again.
        assert!(
            global().lock().unwrap().find(id).is_none(),
            "the transient failure outlived the instant it described"
        );
        reset_for_test();
    }

    /// Two shaders that land in one bucket each resolve to their own entry.
    ///
    /// The bucket is forced rather than found: this cache keys on a 64-bit
    /// digest and nobody has a pair of AIR blobs that collide under it, so the
    /// only way to drive the collision path is to file both under one digest and
    /// then ask through the real lookup. That is exactly the state a natural
    /// collision would produce.
    ///
    /// Before the AIR was retained, `find` returned on digest equality alone, so
    /// the second ask here would have been answered with the first shader's
    /// SPIR-V — a shader the guest never compiled, handed over silently and with
    /// no failure line. The probability was small; the failure was
    /// unobservable, which is the part no birthday bound prices.
    #[test]
    fn two_shaders_in_one_bucket_each_resolve_to_their_own() {
        let _guard = test_lock();
        reset_for_test();

        let first = b"the-air-that-got-there-first";
        let second = b"a-different-shader-entirely";
        let id = ShaderId::render(Stage::Vertex, first, 1);
        // The second identity, forced into the first one's bucket. Everything
        // but `digest` is the second shader's own.
        let collided = ShaderId {
            digest: id.digest,
            ..ShaderId::render(Stage::Vertex, second, 1)
        };
        assert_ne!(
            ShaderId::render(Stage::Vertex, second, 1).digest,
            id.digest,
            "the two blobs do not collide naturally; the bucket below is forced"
        );

        {
            let mut c = global().lock().unwrap();
            c.put(
                id,
                &Arc::from(&first[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![1, 1, 1, 1])),
            );
            c.put(
                collided,
                &Arc::from(&second[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![2, 2, 2, 2])),
            );
            assert_eq!(
                c.entries.get(&id.digest).map(Vec::len),
                Some(2),
                "both are filed under one digest, which is the whole setup"
            );
            assert_eq!(c.len(), 2, "and the level counts entries, not buckets");
        }

        let got = |i| match global().lock().unwrap().find(i) {
            Some(Entry::Ready(s)) => s.spirv.clone(),
            other => panic!("expected a ready entry, got {}", other.is_some()),
        };
        assert_eq!(got(id), vec![1, 1, 1, 1], "the first blob got its own");
        assert_eq!(
            got(collided),
            vec![2, 2, 2, 2],
            "the second blob got its own, not the one sharing its digest"
        );

        // And forgetting one leaves the other where it is, rather than taking
        // the whole bucket with it.
        global().lock().unwrap().forget(id);
        assert!(global().lock().unwrap().find(id).is_none());
        assert_eq!(got(collided), vec![2, 2, 2, 2]);
        reset_for_test();
    }

    /// The converse, so the test above cannot pass by forgetting everything. A
    /// module that cannot be translated is reached again by every later ask, and
    /// re-running metal2vulkan per draw to rediscover that would cost far more
    /// than the answer is worth.
    #[test]
    fn a_failure_about_the_module_is_kept() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"air-that-cannot-be-translated";
        let id = ShaderId::render(Stage::Vertex, air, 1);
        let stored = M2vCacheDecline::VertexTranslate {
            detail: "unsupported instruction".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 1, 7) else {
            panic!("an untranslatable module still fails its draw");
        };
        assert_eq!(err, stored);
        assert!(
            matches!(
                global().lock().unwrap().find(id),
                Some(Entry::Failed(kept)) if *kept == stored
            ),
            "a verdict about the AIR is reached again, so it is kept"
        );
        reset_for_test();
    }

    /// The kernel lookup is a second copy of the same `Entry::Failed` arm, so it
    /// gets the same question asked of it.
    #[test]
    fn the_kernel_lookup_forgets_a_transient_failure_too() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"kernel-air-whose-scratch-write-failed";
        let local_size = [16u32, 16, 1];
        let id = ShaderId::kernel(air, local_size);
        let stored = M2vCacheDecline::KernelScratchWrite {
            detail: "No space left on device".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        let Err(err) = translate_cached_kernel_reflected(air, local_size, 7) else {
            panic!("a stored failure still fails its dispatch");
        };
        assert_eq!(err, stored);
        assert!(global().lock().unwrap().find(id).is_none());
        reset_for_test();
    }

    #[test]
    fn cache_hit_skips_second_lookup_path() {
        let _guard = test_lock();
        reset_for_test();
        // Inject without metal2vulkan: put entry then get_or via public API by
        // priming the map through the same key logic.
        let air = b"synthetic-air-for-cache-unit";
        {
            let mut c = global().lock().unwrap();
            // fake SPIR-V magic-ish
            c.put(
                ShaderId::render(Stage::Vertex, air, 1),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![0x03, 0x02, 0x23, 0x07])),
            );
        }
        // One hit total, carrying the bytes plus the (empty) reflection.
        let shader = translate_cached_reflected(air, Stage::Vertex, 1, 99).expect("hit");
        assert_eq!(shader.spirv, vec![0x03, 0x02, 0x23, 0x07]);
        assert!(shader.reflection.vertex_builtins.is_none());
        let (hits, misses, n) = stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 0);
        assert_eq!(n, 1);
        reset_for_test();
    }

    #[test]
    fn async_cache_state_distinguishes_pending_from_resolved_failure() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-async-cache-state";
        let id = ShaderId::render(Stage::Fragment, air, 1);
        {
            let mut c = global().lock().unwrap();
            c.put(id, &Arc::from(&air[..]), Entry::Loading);
        }
        assert!(!ensure_cached_async(air, Stage::Fragment, 1, 7));
        assert!(global().lock().unwrap().async_queue.is_empty());

        global().lock().unwrap().put(
            id,
            &Arc::from(&air[..]),
            Entry::Failed(M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into(),
            }),
        );
        assert!(ensure_cached_async(air, Stage::Fragment, 1, 7));
        assert_eq!(
            // `.err()` rather than `unwrap_err()`: the success arm is an
            // `Arc<CachedShader>`, which carries no `Debug`.
            translate_cached_reflected(air, Stage::Fragment, 1, 7)
                .err()
                .expect("a Failed entry translates to its decline"),
            M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into()
            }
        );
        reset_for_test();
    }

    /// The shader a boot compiles first is the compositor's, and it is drawn
    /// every frame for the life of the boot. Under the entry cap this cache used
    /// to carry it was also the first eviction victim, because the order was
    /// insertion rather than use — so crossing the bound threw away exactly the
    /// entry that was still hot. Drive far past the old 256-entry bound and
    /// assert the first key is still resolved.
    #[test]
    fn a_hot_first_shader_survives_far_past_the_old_entry_cap() {
        let _guard = test_lock();
        reset_for_test();
        let hot = b"the-compositor-shader-compiled-first".to_vec();
        {
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::render(Stage::Fragment, &hot, 1),
                &Arc::from(&hot[..]),
                Entry::Ready(synth_shader(Stage::Fragment, vec![1, 2, 3, 4])),
            );
        }
        // Four times the retired bound, so a cap of any nearby size would have
        // reached the first key several times over.
        for i in 0..1024u32 {
            let air = format!("cold-shader-{i}").into_bytes();
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::render(Stage::Fragment, &air, 1),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Fragment, vec![0, 0, 0, 0])),
            );
        }
        assert!(
            ensure_cached_async(&hot, Stage::Fragment, 1, 1),
            "the first-compiled shader is still resolved after 1024 later ones"
        );
        assert_eq!(
            global().lock().unwrap().len(),
            1025,
            "every distinct shader is retained; nothing is evicted for count"
        );
        reset_for_test();
    }

    #[test]
    fn async_kernel_cache_uses_local_size_key() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-kernel-async-state";
        {
            let mut c = global().lock().unwrap();
            c.put(
                ShaderId::kernel(air, [16, 16, 1]),
                &Arc::from(&air[..]),
                Entry::Loading,
            );
        }
        assert!(!ensure_cached_kernel_async(air, [16, 16, 1], 20));
        assert!(global().lock().unwrap().async_queue.is_empty());
        assert!(ensure_cached_kernel_async(air, [0, 16, 1], 20));
        reset_for_test();
    }

    #[test]
    fn every_cache_decline_has_a_unique_log_safe_reason_and_fields() {
        let all = [
            M2vCacheDecline::VertexScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::FragmentScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::KernelScratchWrite {
                detail: "permission denied".into(),
            },
            M2vCacheDecline::VertexTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::FragmentTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::KernelTranslate {
                detail: "tool failed".into(),
            },
            M2vCacheDecline::ReflectionMalformed {
                stage: "fragment",
                violations: 1,
            },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 16, 1],
            },
            M2vCacheDecline::KernelLocalSizeMismatch {
                requested: [16, 16, 1],
                reflected: Some([8, 8, 1]),
            },
            M2vCacheDecline::RuntimeSamplerSpecialize {
                stage: "fragment",
                detail: "unsupported state".into(),
            },
            M2vCacheDecline::RuntimeStorageImageSpecialize {
                detail: "unsupported format".into(),
            },
        ];
        let mut slugs = Vec::new();
        for decline in all {
            assert!(decline
                .slug()
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
            for (key, value) in decline.fields() {
                assert!(!key.contains(char::is_whitespace));
                assert!(!value.contains(char::is_whitespace), "{key}={value}");
            }
            slugs.push(decline.slug());
        }
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, 12, "the m2v cache decline census moved");
        assert_eq!(before, slugs.len(), "duplicate m2v cache decline");
    }
}
