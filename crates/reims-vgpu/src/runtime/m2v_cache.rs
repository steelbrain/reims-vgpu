//! Process-global metal2vulkan SPIR-V cache (AIR bytes → SPIR-V).
//!
//! Product Linux draws call m2v on the **doorbell MMIO vCPU** (sync drain under
//! BQL — see `runtime/mmio.rs` CONTROL_FIFO / child doorbell). Live OFF logs
//! showed the same pipelines re-translated dozens of times per boot (e.g.
//! pipe=39 fragment SPIR-V ~400 KiB × 28). That holds guest CPUs long enough
//! for `pmap_flush_tlbs` **IPI timeout** panics (WindowServer in
//! `processExecIndirect` / `submitOnChannel`).
//!
//! Cache key is a content hash of the AIR blob + stage — not pipeline object
//! id (ids recycle; AIR content is the stable unit of work). Measure-only
//! hit/miss counters for fail-log census.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use metal2vulkan::passes::Stage;
use metal2vulkan::reflect::ShaderReflection;

use crate::observe::Decline;

type FragmentRelocationCache = HashMap<(bool, bool), Arc<Vec<u32>>>;
type M2vResult<T> = Result<T, M2vCacheDecline>;

/// A specific failure while caching, translating, or post-processing AIR.
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
    ReflectionDatalayoutMissing {
        stage: &'static str,
    },
    LayoutRepair {
        stage: &'static str,
        reason: crate::runtime::spirv_layout::SpirvLayoutDecline,
    },
    TranslationPending {
        stage: &'static str,
    },
    KernelLocalSizeZero {
        local_size: [u32; 3],
    },
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
            Self::ReflectionDatalayoutMissing { .. } => "m2v_reflection_datalayout_missing",
            Self::LayoutRepair { reason, .. } => reason.slug(),
            Self::TranslationPending { .. } => "m2v_translation_pending_at_sync_boundary",
            Self::KernelLocalSizeZero { .. } => "m2v_kernel_local_size_zero",
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
            Self::LayoutRepair { stage, reason } => {
                let mut fields = vec![("stage", (*stage).to_string())];
                fields.extend(reason.fields());
                fields
            }
            Self::ReflectionDatalayoutMissing { stage } | Self::TranslationPending { stage } => {
                vec![("stage", (*stage).to_string())]
            }
            Self::KernelLocalSizeZero { local_size } => vec![
                ("tg_x", local_size[0].to_string()),
                ("tg_y", local_size[1].to_string()),
                ("tg_z", local_size[2].to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(M2vCacheDecline);

impl std::error::Error for M2vCacheDecline {}

/// A translated shader: the SPIR-V bytes we hand to the Vulkan engine, plus the
/// `metal2vulkan` reflection facade derived from the SAME parsed AIR metadata
/// (descriptor bindings, texture shapes, vertex builtins, datalayout, …). The
/// reflection is the single source of truth for stage-interface facts so the
/// consumer never re-parses AIR or re-walks the emitted SPIR-V. `spirv` is the
/// post-`repair_layout` module (byte-identical to what the plain cache returned).
/// A module's basic-block count and the case count of its largest `OpSwitch`.
///
/// The pair identifies a relooper state machine, which is what metal2vulkan
/// emits when it cannot structure a function's control flow: one loop, one
/// switch, and one case per basic block, with the next block index written to a
/// variable each iteration. In that shape the two numbers are equal; in
/// structured code a switch has a handful of cases and many blocks belong to no
/// switch at all. Measured on one boot's modules: the vertex shader was
/// 13 blocks / 4 cases, the compositor fragment 2 731 / 2 725.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModuleShape {
    pub blocks: u32,
    pub max_switch_cases: u32,
    /// An `OpCompositeInsert` or `OpCompositeExtract` whose object is an opaque
    /// handle — an image, a sampler or a sampled image.
    ///
    /// This is invalid SPIR-V, not a style question. Under the Logical
    /// addressing model an opaque handle has no representation inside a
    /// composite, so the type at the indexed path can never be the handle's own
    /// type. `spirv-val` puts it plainly:
    ///
    /// ```text
    /// The Object type (OpTypeImage) does not match the type that results from
    /// indexing into the Composite (OpTypePointer).
    ///   %193 = OpCompositeInsert %_struct_51 %84 %55 0 0
    /// ```
    ///
    /// Creating a shader module from an invalid one is licence for the driver to
    /// do anything, and here it did: three consecutive boots of a macOS desktop
    /// stopped being served at the compute pipeline carrying exactly this
    /// instruction, with no panic, no Vulkan error, no device loss and no host
    /// crash record.
    pub opaque_in_composite: bool,
}

impl ModuleShape {
    /// Does this look like a relooper state machine rather than structured code?
    ///
    /// Two conditions, both needed. The switch must dispatch essentially every
    /// block — that is the state machine's defining shape, and no structured
    /// shader approaches it. And the module must be past the translator's own
    /// relooper block cap, which is the size at which metal2vulkan itself
    /// declines to structure; below it a large switch is a real guest switch and
    /// compiles fine.
    ///
    /// Using the translator's cap rather than a number chosen here keeps the
    /// threshold tied to the thing that produces the shape.
    pub fn is_relooper_state_machine(&self) -> bool {
        const RELOOPER_MAX_BLOCKS: u32 = 1024;
        self.blocks > RELOOPER_MAX_BLOCKS
            && self.max_switch_cases.saturating_mul(10) >= self.blocks.saturating_mul(9)
    }
}

/// Count basic blocks and the largest `OpSwitch`'s case count.
///
/// Bounds-checked word walk over the little-endian SPIR-V stream, header
/// skipped, in the same shape as [`spirv_uses_builtin`]. `OpLabel` is 248 and
/// starts a basic block; `OpSwitch` is 251 and carries
/// `[selector, default, (literal, label)...]`, so its case count is
/// `(word_count - 3) / 2`.
fn spirv_module_shape(words: &[u32]) -> ModuleShape {
    let mut shape = ModuleShape::default();
    // <id>s whose type is an opaque handle. Seeded from the type declarations
    // and propagated through the few instructions that can carry one.
    let mut opaque_types: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut i = 5; // skip header
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xffff;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        match opcode {
            // OpTypeImage / OpTypeSampler / OpTypeSampledImage: record the
            // result <id> so the composite check below can recognise a handle
            // of that type without a full type walk.
            OP_TYPE_IMAGE..=OP_TYPE_SAMPLED_IMAGE if word_count >= 2 => {
                opaque_types.insert(words[i + 1]);
            }
            // Every other type-declaring or value-producing instruction whose
            // result type is opaque makes its result opaque too. Only the ones
            // that can name an image are worth following: OpLoad (61),
            // OpSampledImage (86), OpImage (100), OpCopyObject (83) and
            // OpFunctionParameter (55) all carry `result-type, result-id`.
            55 | 61 | 83 | 86 | 100
                if word_count >= 3 && opaque_types.contains(&words[i + 1]) =>
            {
                opaque_types.insert(words[i + 2]);
            }
            248 => shape.blocks = shape.blocks.saturating_add(1),
            251 if word_count >= 3 => {
                let cases = ((word_count - 3) / 2) as u32;
                shape.max_switch_cases = shape.max_switch_cases.max(cases);
            }
            // OpCompositeInsert: result-type, result-id, object, composite, ...
            // The object is word 3.
            OP_COMPOSITE_INSERT
                if word_count >= 5 && opaque_types.contains(&words[i + 3]) =>
            {
                shape.opaque_in_composite = true;
            }
            // OpCompositeExtract: result-type, result-id, composite, ... — an
            // opaque *result type* means a handle is being read back out of a
            // composite it could never have been put into.
            OP_COMPOSITE_EXTRACT
                if word_count >= 4 && opaque_types.contains(&words[i + 1]) =>
            {
                shape.opaque_in_composite = true;
            }
            _ => {}
        }
        i += word_count;
    }
    shape
}

/// The three opaque-handle type declarations, contiguous in the SPIR-V core
/// grammar: `OpTypeImage`, `OpTypeSampler`, `OpTypeSampledImage`.
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
/// `OpCompositeInsert`, from the SPIR-V core grammar.
const OP_COMPOSITE_INSERT: u32 = 82;
/// `OpCompositeExtract`, likewise.
const OP_COMPOSITE_EXTRACT: u32 = 81;

pub struct CachedShader {
    pub spirv: Vec<u8>,
    pub reflection: Arc<ShaderReflection>,
    /// Block/switch shape, computed once when the module is cached.
    pub shape: ModuleShape,
    /// The same module as u32 words, materialized once — draw paths clone the
    /// `Arc`, never re-collect per draw (was a full-module copy ×2 per draw).
    pub words: Arc<Vec<u32>>,
    /// Fragment binding-relocation variants keyed by
    /// `(separate_sampled, buf_collide)` — the relocation is a pure function of
    /// the module and those two flags, so each variant is computed once per
    /// shader lifetime instead of mutating a fresh copy per draw.
    frag_reloc: Mutex<FragmentRelocationCache>,
}

impl CachedShader {
    pub fn new(spirv: Vec<u8>, reflection: Arc<ShaderReflection>) -> Self {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let shape = spirv_module_shape(&words);
        Self {
            spirv,
            reflection,
            shape,
            words: Arc::new(words),
            frag_reloc: Mutex::new(HashMap::new()),
        }
    }

    /// Fragment words with stage-collision binding relocations applied.
    /// `separate_sampled` relocates sampled-resource bindings first (archive
    /// order), then `buf_collide` relocates the buffer band — matching the
    /// historical per-draw mutation order exactly. Cached per variant.
    pub fn fragment_words(&self, separate_sampled: bool, buf_collide: bool) -> Arc<Vec<u32>> {
        if !separate_sampled && !buf_collide {
            return self.words.clone();
        }
        let mut cache = self.frag_reloc.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .entry((separate_sampled, buf_collide))
            .or_insert_with(|| {
                let mut w: Vec<u32> = (*self.words).clone();
                if separate_sampled {
                    let n = crate::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(
                        &mut w,
                    );
                    crate::observe::line(format!("linux_m2v frag_sampled_reloc n={n}"));
                }
                if buf_collide {
                    let n = crate::runtime::spirv_bind::offset_fragment_buffer_bindings(&mut w);
                    crate::observe::line(format!("linux_m2v frag_buf_reloc n={n}"));
                }
                Arc::new(w)
            })
            .clone()
    }
}

/// Cap entries so a long session cannot grow without bound (research host).
const MAX_ENTRIES: usize = 256;

#[derive(Default)]
struct Cache {
    /// key = hash(stage_tag || air_bytes) → SPIR-V bytes
    entries: HashMap<u64, Entry>,
    /// Insertion order for crude eviction (FIFO).
    order: Vec<u64>,
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

struct TranslationTask {
    key: u64,
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    air: Vec<u8>,
    pipeline_ref: u32,
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

fn air_key(stage: Stage, air: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(stage).hash(&mut h);
    air.hash(&mut h);
    h.finish()
}

/// Kernel cache key includes LocalSize (SPIR-V workgroup size is baked at translate).
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

fn translate_air(air: &[u8], stage: Stage) -> M2vResult<CachedShader> {
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
    // Reflected translate: byte-identical SPIR-V PLUS the stage-interface facade.
    // `reflection.datalayout` carries the source `target datalayout` the sanitizer
    // strips, so the post-emit ABI reconciliation below no longer re-reads `k.ll`.
    // Validated: `Ok` from the plain entry can carry bytes no tier validated, and
    // those go straight to `vkCreate*Pipelines`. An NVIDIA driver segfaults on
    // one and takes the process with it, with nothing in the validation log
    // because the module never reaches a layer that would reject it.
    let (spirv, reflection) = metal2vulkan::translate_reflected_validated(
        path.to_str().unwrap_or(name),
        stage,
        &tmp,
        metal2vulkan::passes::TransformOptions::default(),
    )
    .map_err(|e| translate_decline(stage, e.to_string()))?;
    finish_translated(spirv, reflection, stage)
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
    let (spirv, reflection) = metal2vulkan::translate_reflected_validated(
        path.to_str().unwrap_or("k.air"),
        Stage::Kernel,
        &tmp,
        opts,
    )
    .map_err(|e| M2vCacheDecline::KernelTranslate {
        detail: e.to_string(),
    })?;
    finish_translated(spirv, reflection, Stage::Kernel)
}

/// Post-emit ABI reconciliation + package the translated shader with its reflection.
fn finish_translated(
    spirv: Vec<u8>,
    reflection: ShaderReflection,
    stage: Stage,
) -> M2vResult<CachedShader> {
    let spirv = repair_layout(&reflection, spirv, stage)?;
    census_reflection(&reflection, stage);
    // Once-per-translate: surface any runtime function constants this shader
    // declares. metal2vulkan folds them to their disabled default and the paravirt
    // stream carries no MTLFunctionConstantValues, so the FC-disabled variant is
    // what we run; this makes that reliance measurable (which shaders use FCs)
    // without touching translation or rendering. Silent for FC-free shaders.
    crate::runtime::spirv_bind::log_folded_function_constants(&reflection);
    Ok(CachedShader::new(spirv, Arc::new(reflection)))
}

/// Once-per-translate (miss path) well-formedness guard on the AIR-derived
/// reflection. This is the always-on regression proxy for the reflection-fed hot
/// path: it validates the reflection's ABI version and its internal
/// sampled-vs-storage consistency without a second walk of the SPIR-V. Must read
/// zero on a healthy boot.
fn census_reflection(reflection: &ShaderReflection, stage: Stage) {
    // pipeline_ref is telemetry only here; the stage tag localizes the shader.
    let pipe = stage_tag(stage) as u32;
    let _ = crate::runtime::spirv_bind::census_reflection_wellformed(reflection, pipe);
}

fn repair_layout(
    reflection: &ShaderReflection,
    spirv: Vec<u8>,
    stage: Stage,
) -> M2vResult<Vec<u8>> {
    // The datalayout is the reflection's — single source of truth, no `k.ll` re-read.
    // A reflected translate from unsanitized AIR always populates it; its absence
    // is a genuine gap (fail-visible), matching the prior `air_datalayout_missing`.
    let datalayout =
        reflection
            .datalayout
            .as_deref()
            .ok_or(M2vCacheDecline::ReflectionDatalayoutMissing {
                stage: stage_name(stage),
            })?;
    let (spirv, stats) =
        crate::runtime::spirv_layout::repair_llvm_vector_alloc_offsets_from_datalayout(
            datalayout, &spirv,
        )
        .map_err(|reason| M2vCacheDecline::LayoutRepair {
            stage: stage_name(stage),
            reason,
        })?;
    if stats.members != 0 {
        crate::observe::fail(format!(
            "linux_m2v_layout_repair stage={stage:?} reason=llvm_vector_alloc_stride structs={} members={}",
            stats.structs, stats.members
        ));
    }
    Ok(spirv)
}

fn evict_one(c: &mut Cache) {
    if c.entries.len() < MAX_ENTRIES {
        return;
    }
    if let Some(pos) = c
        .order
        .iter()
        .position(|key| !matches!(c.entries.get(key), Some(Entry::Loading)))
    {
        let old = c.order.remove(pos);
        c.entries.remove(&old);
    }
}

/// Start translating a render stage without holding protocol state or the
/// sole FIFO scheduler. Returns true when the content is already resolved
/// (success or deterministic failure), false while the background worker owns
/// it. Callers keep the guest packet at the channel head and retry on poll.
pub fn ensure_cached_async(air: &[u8], stage: Stage, pipeline_ref: u32) -> bool {
    let key = air_key(stage, air);
    ensure_cached_async_keyed(air, stage, None, pipeline_ref, key)
}

/// Kernel counterpart to [`ensure_cached_async`]. LocalSize is part of both
/// the translation options and cache key, so two dispatch geometries can never
/// alias one another.
pub fn ensure_cached_kernel_async(air: &[u8], local_size: [u32; 3], pipeline_ref: u32) -> bool {
    if local_size.contains(&0) {
        return true;
    }
    let key = air_key_kernel(air, local_size);
    ensure_cached_async_keyed(air, Stage::Kernel, Some(local_size), pipeline_ref, key)
}

fn ensure_cached_async_keyed(
    air: &[u8],
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
    pipeline_ref: u32,
    key: u64,
) -> bool {
    let mut start_worker = false;
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.entries.get(&key) {
            Some(Entry::Ready(_)) | Some(Entry::Failed(_)) => return true,
            Some(Entry::Loading) => return false,
            None => {}
        }
        evict_one(&mut c);
        c.entries.insert(key, Entry::Loading);
        c.order.push(key);
        c.misses = c.misses.saturating_add(1);
        c.async_queue.push_back(TranslationTask {
            key,
            stage,
            kernel_local_size,
            air: air.to_vec(),
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
    crate::observe::off(format!(
        "linux_m2v_async queued pipe={pipeline_ref} stage={stage:?}{dims} air={}",
        air.len()
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
            None => translate_air(&task.air, task.stage),
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
                        // Cross-check the AIR-derived reflection against the emitted
                        // SPIR-V decorations, once per translate (miss path only), so a
                        // translator regression that drops a builtin between AIR meta and
                        // emit is fail-visible instead of silent. Quiet on a healthy boot.
                        let emit_instance = spirv_uses_builtin(&shader.spirv, 43);
                        let emit_vertex = spirv_uses_builtin(&shader.spirv, 42);
                        if emit_instance != vb.uses_instance_index
                            || emit_vertex != vb.uses_vertex_index
                        {
                            crate::observe::fail(format!(
                                "m2v_reflect_divergence pipe={} stage=Vertex kind=vertex_builtins reflect_instance={} emit_instance={} reflect_vertex={} emit_vertex={}",
                                task.pipeline_ref,
                                vb.uses_instance_index as u8, emit_instance as u8,
                                vb.uses_vertex_index as u8, emit_vertex as u8
                            ));
                        }
                        // Decoded vertex-builtin usage census (not an error);
                        // route off() so it leaves the curated real-error view.
                        // The genuine reflect-vs-emit divergence above stays
                        // fail()-visible.
                        crate::observe::off(format!(
                            "m2v_builtins pipe={} stage=Vertex instance_index={} vertex_index={}",
                            task.pipeline_ref,
                            vb.uses_instance_index as u8,
                            vb.uses_vertex_index as u8
                        ));
                    }
                    c.entries.insert(task.key, Entry::Ready(Arc::new(shader)));
                    (format!("ok spv={len}"), None)
                }
                Err(e) => {
                    c.entries.insert(task.key, Entry::Failed(e.clone()));
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
            let mut emit = crate::observe::Emit::decline("linux_m2v_async_done", &error)
                .field("pipe", task.pipeline_ref)
                .field("hits", hits)
                .field("misses", misses);
            if let Some([x, y, z]) = task.kernel_local_size {
                emit = emit.field("tg_x", x).field("tg_y", y).field("tg_z", z);
            }
            emit.fail_once(task.key);
        } else {
            let done = format!(
                "linux_m2v_async done pipe={} stage={:?}{dims} {detail} hits={hits} misses={misses}",
                task.pipeline_ref, task.stage
            );
            crate::observe::off(done);
        }
    }
}

/// Translate `air` for `stage`, returning cached SPIR-V when AIR matches a prior
/// translate. Logs `linux_m2v_translate` on miss and `linux_m2v_translate_hit`
/// on hit (pipe id is telemetry only).
/// Does the SPIR-V module decorate any id with `BuiltIn <builtin>`? Scans
/// `OpDecorate` (opcode 71) instructions for a `BuiltIn` (decoration 11) whose
/// value equals `builtin` (e.g. 42 = VertexIndex, 43 = InstanceIndex).
/// Bounds-checked word walk over the little-endian SPIR-V stream (header skipped).
fn spirv_uses_builtin(spv: &[u8], builtin: u32) -> bool {
    if spv.len() < 20 || !spv.len().is_multiple_of(4) {
        return false;
    }
    let words: Vec<u32> = spv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut i = 5; // skip header
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xffff;
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        // OpDecorate = 71: [target, decoration, operands...]; BuiltIn = 11.
        if opcode == 71 && word_count >= 4 && words[i + 2] == 11 && words[i + 3] == builtin {
            return true;
        }
        i += word_count;
    }
    false
}

/// Translate `air` for `stage`, returning the whole [`CachedShader`] (SPIR-V +
/// reflection) as a shared handle, so a consumer reads stage-interface facts
/// (texture shapes, vertex builtins, descriptor bindings) from the reflection
/// instead of re-walking the emitted SPIR-V. The returned `Arc` is a clone of the
/// cached entry — a warm hit performs no allocation beyond the refcount bump.
pub fn translate_cached_reflected(
    air: &[u8],
    stage: Stage,
    pipeline_ref: u32,
) -> M2vResult<Arc<CachedShader>> {
    let key = air_key(stage, air);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.entries.get(&key).cloned() {
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
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage={stage:?} spv={} hits={hits} misses={misses}",
                        shader.spirv.len()
                    ));
                }
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => return Err(e),
            Some(Entry::Loading) => {
                return Err(M2vCacheDecline::TranslationPending {
                    stage: stage_name(stage),
                })
            }
            None => {}
        }
    }

    let shader = Arc::new(translate_air(air, stage)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        evict_one(&mut c);
        c.entries.insert(key, Entry::Ready(Arc::clone(&shader)));
        c.order.push(key);
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        crate::observe::fail(format!(
            "linux_m2v_translate ok pipe={pipeline_ref} stage={stage:?} v_spv_or_f={} hits={hits} misses={misses}",
            shader.spirv.len()
        ));
    }
    Ok(shader)
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
    let key = air_key_kernel(air, local_size);
    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        match c.entries.get(&key).cloned() {
            Some(Entry::Ready(shader)) => {
                c.hits = c.hits.saturating_add(1);
                let hits = c.hits;
                let misses = c.misses;
                drop(c);
                // Verbose-gated (see `translate_cached_reflected`): a compute-kernel cache hit
                // is the hot path and only carries a cumulative counter.
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
                        "linux_m2v_translate_hit pipe={pipeline_ref} stage=Kernel tg=[{},{},{}] spv={} hits={hits} misses={misses}",
                        local_size[0], local_size[1], local_size[2], shader.spirv.len()
                    ));
                }
                return Ok(shader);
            }
            Some(Entry::Failed(e)) => return Err(e),
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
        evict_one(&mut c);
        c.entries.insert(key, Entry::Ready(Arc::clone(&shader)));
        c.order.push(key);
        let hits = c.hits;
        let misses = c.misses;
        drop(c);
        crate::observe::fail(format!(
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
    (c.hits, c.misses, c.entries.len())
}

/// Test isolation.
#[cfg(test)]
pub fn reset_for_test() {
    let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
    c.entries.clear();
    c.order.clear();
    c.hits = 0;
    c.misses = 0;
    c.async_queue.clear();
    c.async_worker_running = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a SPIR-V word stream carrying `blocks` `OpLabel`s and one
    /// `OpSwitch` with `cases` cases. Only the two opcodes the shape scanner
    /// reads are emitted — this is a probe for the scanner, not a valid module.
    fn shaped_module(blocks: u32, cases: u32) -> Vec<u32> {
        const OP_LABEL: u32 = 248;
        const OP_SWITCH: u32 = 251;
        let mut w = vec![0x0723_0203, 0x0001_0500, 0, 1, 0]; // 5-word header
        for _ in 0..blocks {
            w.push((2 << 16) | OP_LABEL);
            w.push(1);
        }
        if cases > 0 {
            let word_count = 3 + cases * 2;
            w.push((word_count << 16) | OP_SWITCH);
            w.push(1); // selector
            w.push(2); // default label
            for c in 0..cases {
                w.push(c);
                w.push(3);
            }
        }
        w
    }

    /// The scanner must read the two numbers the discriminator is built on.
    #[test]
    fn the_module_shape_scanner_counts_blocks_and_the_widest_switch() {
        let shape = spirv_module_shape(&shaped_module(2731, 2725));
        assert_eq!(shape.blocks, 2731);
        assert_eq!(shape.max_switch_cases, 2725);

        // A module with no switch at all still reports its blocks.
        let plain = spirv_module_shape(&shaped_module(13, 0));
        assert_eq!(plain.blocks, 13);
        assert_eq!(plain.max_switch_cases, 0);

        // Truncated stream: stop, do not read past the end.
        let mut short = shaped_module(4, 0);
        short.truncate(6);
        assert!(spirv_module_shape(&short).blocks <= 4);
    }

    /// The discriminator separates the two real modules that motivated it, and
    /// does not fire on a large *structured* switch.
    ///
    /// Both shapes are measured, from one boot of the x86 guest: the vertex
    /// shader of pipe 48 and the compositor fragment shader that held
    /// `vkCreateGraphicsPipelines` past 22 minutes.
    /// An image handle inserted into a composite is spotted; ordinary composite
    /// work is not.
    ///
    /// The positive case is the shape of the real instruction, taken from the
    /// module a live macOS desktop was translating when the host process stopped
    /// being served three boots in a row:
    ///
    /// ```text
    /// %193 = OpCompositeInsert %_struct_51 %84 %55 0 0
    /// ```
    ///
    /// where `%84` is an `OpLoad` of an `OpTypeImage` variable. `spirv-val`
    /// rejects it — under the Logical addressing model an opaque handle has no
    /// representation inside a composite — and a shader module built from an
    /// invalid one lets the driver do anything.
    ///
    /// The negative case matters just as much: `OpCompositeInsert` is ordinary
    /// and frequent, so a check that fired on all of it would decline nearly
    /// every shader.
    #[test]
    fn an_opaque_handle_inside_a_composite_is_detected() {
        // (opcode | word_count << 16), then operands.
        fn op(opcode: u32, operands: &[u32]) -> Vec<u32> {
            let mut out = vec![((operands.len() as u32 + 1) << 16) | opcode];
            out.extend_from_slice(operands);
            out
        }
        fn module(body: &[Vec<u32>]) -> Vec<u32> {
            let mut words = vec![0x0723_0203, 0x0001_0300, 0, 1, 0];
            for instruction in body {
                words.extend_from_slice(instruction);
            }
            words
        }

        // %10 = OpTypeImage, %11 = OpTypePointer to it, %12 = OpVariable,
        // %13 = OpLoad %10 %12  -> %13 is an image handle,
        // %14 = OpCompositeInsert %20 %13 %21 0 0  -> the invalid instruction.
        let with_image = module(&[
            op(25, &[10, 1, 1, 0, 0, 0, 0, 1, 0]), // OpTypeImage
            op(61, &[10, 13, 12]),                 // OpLoad, result type %10
            op(82, &[20, 14, 13, 21, 0, 0]),       // OpCompositeInsert, object %13
        ]);
        assert!(spirv_module_shape(&with_image).opaque_in_composite);

        // A sampler handle counts too, and so does reading one back out.
        let extract_sampler = module(&[
            op(26, &[10]),                   // OpTypeSampler
            op(81, &[10, 14, 21, 0]),        // OpCompositeExtract, result type %10
        ]);
        assert!(spirv_module_shape(&extract_sampler).opaque_in_composite);

        // Ordinary composite work on non-opaque types: must stay silent. Same
        // instructions, but the object and result types are a plain float.
        let ordinary = module(&[
            op(25, &[10, 1, 1, 0, 0, 0, 0, 1, 0]), // OpTypeImage %10 (declared, unused)
            op(22, &[30]),                         // OpTypeFloat %30
            op(61, &[30, 31, 32]),                 // OpLoad %30 %31
            op(82, &[33, 34, 31, 35, 0]),          // OpCompositeInsert of the float
            op(81, &[30, 36, 35, 0]),              // OpCompositeExtract of a float
        ]);
        assert!(!spirv_module_shape(&ordinary).opaque_in_composite);

        // An empty module declares nothing and inserts nothing.
        assert!(!spirv_module_shape(&module(&[])).opaque_in_composite);
    }

    #[test]
    fn only_a_relooper_state_machine_is_declined() {
        // Measured: pipe 48 vertex, structured, compiled in milliseconds.
        let vertex = ModuleShape {
            blocks: 13,
            max_switch_cases: 4,
            ..ModuleShape::default()
        };
        assert!(!vertex.is_relooper_state_machine());

        // Measured: pipe 48 fragment, the state machine that never compiled.
        let compositor = ModuleShape {
            blocks: 2731,
            max_switch_cases: 2725,
            ..ModuleShape::default()
        };
        assert!(compositor.is_relooper_state_machine());

        // A genuinely large guest switch, but structured: most blocks are not
        // cases. Must not be declined.
        let big_structured = ModuleShape {
            blocks: 4000,
            max_switch_cases: 300,
            ..ModuleShape::default()
        };
        assert!(!big_structured.is_relooper_state_machine());

        // A small function that happens to be almost all switch: below the
        // translator's relooper cap, so it was structured and compiles.
        let small_dispatch = ModuleShape {
            blocks: 200,
            max_switch_cases: 199,
            ..ModuleShape::default()
        };
        assert!(!small_dispatch.is_relooper_state_machine());
    }

    fn test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A minimal `CachedShader` wrapping raw bytes with an empty reflection —
    /// enough to prime the cache in unit tests that never call metal2vulkan.
    fn synth_shader(stage: Stage, spirv: Vec<u8>) -> Arc<CachedShader> {
        use metal2vulkan::reflect::{ShaderReflection, ShaderStage, REFLECTION_VERSION};
        let stage = match stage {
            Stage::Vertex => ShaderStage::Vertex,
            Stage::Fragment => ShaderStage::Fragment,
            Stage::Kernel => ShaderStage::Kernel,
        };
        Arc::new(CachedShader::new(
            spirv,
            Arc::new(ShaderReflection {
                reflection_version: REFLECTION_VERSION,
                stage,
                entry_point: None,
                bindings: vec![],
                vertex_attributes: vec![],
                varyings: vec![],
                render_targets: vec![],
                depth_members: vec![],
                stencil_members: vec![],
                local_size: None,
                vertex_builtins: None,
                imageblock_layouts: vec![],
                datalayout: None,
                function_constants: vec![],
            }),
        ))
    }

    #[test]
    fn fragment_words_variants_match_direct_relocation_and_cache() {
        // Minimal module: 5-word header + three OpDecorate Binding instructions,
        // one in the buffer band and the low/high edges exercised in the
        // sampled band. Binding 95 is the maximum relocated source and proves
        // the now-infallible addition tops out at 223.
        let decorate = |id: u32, binding: u32| vec![(4u32 << 16) | 71, id, 33, binding];
        let mut words: Vec<u32> = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
        words.extend(decorate(7, 3));
        words.extend(decorate(8, 40));
        words.extend(decorate(9, 95));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader(Stage::Fragment, bytes);
        assert_eq!(*shader.words, words);

        // No flags → the base Arc, unrelocated.
        let base = shader.fragment_words(false, false);
        assert!(Arc::ptr_eq(&base, &shader.words));

        // Both flags → sampled reloc first, then buffer band, matching the
        // historical per-draw mutation order.
        let mut expect = words.clone();
        let n = crate::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(&mut expect);
        assert_eq!(n, 2);
        let n = crate::runtime::spirv_bind::offset_fragment_buffer_bindings(&mut expect);
        assert_eq!(n, 1);
        let both = shader.fragment_words(true, true);
        assert_eq!(*both, expect);
        assert_eq!(
            both[8],
            3 + crate::runtime::spirv_bind::FRAG_BUFFER_BINDING_OFFSET
        );
        assert_eq!(
            both[12],
            40 + crate::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            both[16],
            95 + crate::runtime::spirv_bind::FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(both[16], 223);

        // Second call returns the cached variant (same allocation), and the
        // base module is never mutated.
        let again = shader.fragment_words(true, true);
        assert!(Arc::ptr_eq(&both, &again));
        assert_eq!(*shader.words, words);
    }

    #[test]
    fn air_key_differs_by_stage_and_bytes() {
        let a = b"same-air-bytes";
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Fragment, a));
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, b"other"));
        assert_eq!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, a));
        assert_ne!(air_key_kernel(a, [16, 16, 1]), air_key_kernel(a, [8, 8, 1]));
    }

    #[test]
    fn cache_hit_skips_second_lookup_path() {
        let _guard = test_lock();
        reset_for_test();
        // Inject without metal2vulkan: put entry then get_or via public API by
        // priming the map through the same key logic.
        let air = b"synthetic-air-for-cache-unit";
        let key = air_key(Stage::Vertex, air);
        {
            let mut c = global().lock().unwrap();
            // fake SPIR-V magic-ish
            c.entries.insert(
                key,
                Entry::Ready(synth_shader(Stage::Vertex, vec![0x03, 0x02, 0x23, 0x07])),
            );
            c.order.push(key);
        }
        // One hit total, carrying the bytes plus the (empty) reflection.
        let shader = translate_cached_reflected(air, Stage::Vertex, 99).expect("hit");
        assert_eq!(shader.spirv, vec![0x03, 0x02, 0x23, 0x07]);
        assert!(shader.reflection.vertex_builtins.is_none());
        let (hits, misses, n) = stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 0);
        assert_eq!(n, 1);
        reset_for_test();
    }

    #[test]
    fn spirv_builtin_scan_finds_instance_index() {
        // Minimal module: 5-word header, then OpDecorate(71) %id BuiltIn(11) InstanceIndex(43).
        // word0 = (wordCount<<16)|opcode = (4<<16)|71.
        let words: [u32; 9] = [0x07230203, 0x00010000, 0, 1, 0, (4 << 16) | 71, 5, 11, 43];
        let mut spv = Vec::new();
        for w in words {
            spv.extend_from_slice(&w.to_le_bytes());
        }
        assert!(spirv_uses_builtin(&spv, 43)); // InstanceIndex present
        assert!(!spirv_uses_builtin(&spv, 42)); // VertexIndex absent
        assert!(!spirv_uses_builtin(&[], 43)); // empty is safe
    }

    #[test]
    fn async_cache_state_distinguishes_pending_from_resolved_failure() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-async-cache-state";
        let key = air_key(Stage::Fragment, air);
        {
            let mut c = global().lock().unwrap();
            c.entries.insert(key, Entry::Loading);
            c.order.push(key);
        }
        assert!(!ensure_cached_async(air, Stage::Fragment, 7));
        assert!(global().lock().unwrap().async_queue.is_empty());

        global().lock().unwrap().entries.insert(
            key,
            Entry::Failed(M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into(),
            }),
        );
        assert!(ensure_cached_async(air, Stage::Fragment, 7));
        assert_eq!(
            // `.err()` rather than `unwrap_err()`: the success arm is an
            // `Arc<CachedShader>`, which carries no `Debug`.
            translate_cached_reflected(air, Stage::Fragment, 7)
                .err()
                .expect("a Failed entry translates to its decline"),
            M2vCacheDecline::FragmentTranslate {
                detail: "synthetic failure".into()
            }
        );
        reset_for_test();
    }

    #[test]
    fn async_kernel_cache_uses_local_size_key() {
        let _guard = test_lock();
        reset_for_test();
        let air = b"synthetic-kernel-async-state";
        let key = air_key_kernel(air, [16, 16, 1]);
        {
            let mut c = global().lock().unwrap();
            c.entries.insert(key, Entry::Loading);
            c.order.push(key);
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
            M2vCacheDecline::ReflectionDatalayoutMissing { stage: "fragment" },
            M2vCacheDecline::LayoutRepair {
                stage: "kernel",
                reason:
                    crate::runtime::spirv_layout::SpirvLayoutDecline::DataLayoutVectorAlignmentMissing,
            },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 16, 1],
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
        assert_eq!(before, 10, "the m2v cache decline census moved");
        assert_eq!(before, slugs.len(), "duplicate m2v cache decline");
    }
}
