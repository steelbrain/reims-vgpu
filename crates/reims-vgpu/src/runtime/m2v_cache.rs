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

impl M2vCacheDecline {
    /// Whether a second attempt at the same AIR could answer differently.
    ///
    /// Every entry in this cache is keyed by the AIR blob's content, so a
    /// refusal that is *about the AIR* is reached again by every later ask and
    /// is worth remembering: a module whose datalayout is missing, whose layout
    /// repair fails, or whose requested threadgroup is degenerate refuses the
    /// same way forever.
    ///
    /// The scratch writes are not about the AIR. `translate_air` and
    /// `translate_kernel_air` each begin by writing the blob to a fixed path
    /// under [`tmp_dir`], and that write fails for reasons belonging to the host
    /// filesystem at that instant — no space, no descriptors, a transient I/O
    /// error. Remembering one turns "the host could not spare a scratch file
    /// just then" into "this shader never renders again", and because the cache
    /// is unbounded and nothing evicts, "again" means for the life of the
    /// process. It is the same rule
    /// [`crate::backend::vulkan::engine::types::DrawError::out_of_memory`]
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
            | Self::ReflectionDatalayoutMissing { .. }
            | Self::LayoutRepair { .. }
            | Self::TranslationPending { .. }
            | Self::KernelLocalSizeZero { .. } => false,
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
    crate::observe::Emit::decline("m2v_transient_failure_forgotten", error)
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
pub struct CachedShader {
    pub spirv: Vec<u8>,
    pub reflection: Arc<ShaderReflection>,
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
    /// Materialize a freshly translated module, in the device's binding
    /// numbering rather than the translator's.
    ///
    /// [`crate::runtime::spirv_bind::widen_sampled_bands`] runs exactly here,
    /// once per shader on the translate miss path, because this is the one point
    /// every consumer of a module goes through — both `spirv` and `words`, and
    /// therefore every fragment relocation variant derived from `words`. Doing it
    /// per draw would be a module rewrite on the hot path; doing it in only one
    /// of the two representations would let the compute rail (which reads
    /// `spirv`) and the render rail (which reads `words`) disagree about what
    /// binding a texture has.
    ///
    /// `spirv` is rebuilt from the widened words for that reason: it is no longer
    /// byte-identical to what the translator returned, and the bytes and the
    /// words must not be allowed to drift apart.
    pub fn new(spirv: Vec<u8>, reflection: Arc<ShaderReflection>) -> Self {
        let mut words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let widened = crate::runtime::spirv_bind::widen_sampled_bands(&mut words);
        let spirv = if widened == 0 {
            // Nothing moved, so the translator's bytes already are the device's.
            spirv
        } else {
            words.iter().flat_map(|w| w.to_le_bytes()).collect()
        };
        Self {
            spirv,
            reflection,
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
/// plus a relocated word copy per fragment variant, so one copy of the AIR that
/// produced them is a fraction of the entry rather than a doubling of it.
struct Slot {
    stage: u8,
    /// `Some` for a kernel, whose LocalSize is baked into the SPIR-V, so one
    /// AIR blob dispatched at two geometries is two shaders. `None` for a render
    /// stage, which has no such parameter.
    local_size: Option<[u32; 3]>,
    air: Arc<[u8]>,
    entry: Entry,
}

impl Slot {
    /// The full identity compare. This alone decides a hit.
    fn is(&self, id: ShaderId<'_>) -> bool {
        self.stage == id.stage && self.local_size == id.local_size && *self.air == *id.air
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
/// It is also the shape the rest of this crate uses:
/// [`crate::model::content_cache`] buckets by a `u64` prefilter and decides on
/// `CacheEntry::matches`, and [`crate::backend::blob`] buckets a shader by its
/// digest and decides on the retained bytes.
///
/// This doc used to close by calling itself "the one digest-keyed cache in the
/// crate that trusted its key", on the strength of a sweep run when it was
/// fixed. **The sweep was wrong by three, and how it went wrong is the useful
/// part.** `backend::metal::cache`'s `BlobKey` was a digest beside the blob's
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
    air: &'a [u8],
}

impl<'a> ShaderId<'a> {
    fn render(stage: Stage, air: &'a [u8]) -> Self {
        Self {
            digest: air_key(stage, air),
            stage: stage_tag(stage),
            local_size: None,
            air,
        }
    }

    fn kernel(air: &'a [u8], local_size: [u32; 3]) -> Self {
        Self {
            digest: air_key_kernel(air, local_size),
            stage: stage_tag(Stage::Kernel),
            local_size: Some(local_size),
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
            None => ShaderId::render(self.stage, &self.air),
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
fn air_key(stage: Stage, air: &[u8]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    stage_tag(stage).hash(&mut h);
    air.hash(&mut h);
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
    let (spirv, reflection) =
        metal2vulkan::translate_reflected(path.to_str().unwrap_or(name), stage, &tmp)
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
    let (spirv, reflection) = metal2vulkan::translate_reflected_with_options(
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

/// Start translating a render stage without holding protocol state or the
/// sole FIFO scheduler. Returns true when the content is already resolved
/// (success or deterministic failure), false while the background worker owns
/// it. Callers keep the guest packet at the channel head and retry on poll.
pub fn ensure_cached_async(air: &[u8], stage: Stage, pipeline_ref: u32) -> bool {
    ensure_cached_async_keyed(ShaderId::render(stage, air), stage, None, pipeline_ref)
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
        pipeline_ref,
    )
}

fn ensure_cached_async_keyed(
    id: ShaderId<'_>,
    stage: Stage,
    kernel_local_size: Option<[u32; 3]>,
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
    crate::observe::off(format!(
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
            let mut emit = crate::observe::Emit::decline("linux_m2v_async_done", &error)
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
    let id = ShaderId::render(stage, air);
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
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
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

    let shader = Arc::new(translate_air(air, stage)?);

    {
        let mut c = global().lock().unwrap_or_else(|e| e.into_inner());
        c.misses = c.misses.saturating_add(1);
        c.put(id, &Arc::from(air), Entry::Ready(Arc::clone(&shader)));
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
                if crate::observe::draw_log_enabled() {
                    crate::observe::line(format!(
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

    /// `CachedShader::new` widens the translator's bands, and `fragment_words`
    /// relocates on top of that — in that order, for every representation.
    #[test]
    fn fragment_words_variants_match_direct_relocation_and_cache() {
        use crate::runtime::spirv_bind::{
            FRAG_BUFFER_BINDING_OFFSET, FRAG_SAMPLED_RESOURCE_BINDING_OFFSET,
            M2V_SAMPLER_BINDING_BASE, M2V_TEXTURE_BINDING_BASE, SAMPLED_TAIL_WIDEN_OFFSET,
        };

        // Minimal module: 5-word header + three OpDecorate Binding instructions
        // in the TRANSLATOR's numbering, which is what a fresh translate hands
        // over. One buffer, one texture, and the top of the translator's sampler
        // band — the maximum source the widen has to carry.
        const TOP_SAMPLER: u32 = M2V_SAMPLER_BINDING_BASE + 31;
        let decorate = |id: u32, binding: u32| vec![(4u32 << 16) | 71, id, 33, binding];
        let mut words: Vec<u32> = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
        words.extend(decorate(7, 3));
        words.extend(decorate(8, M2V_TEXTURE_BINDING_BASE + 8));
        words.extend(decorate(9, TOP_SAMPLER));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let shader = synth_shader(Stage::Fragment, bytes);

        // The stored module is the widened one: the sampler moved out of the
        // texture band's way, the buffer and texture did not move at all.
        let mut widened = words.clone();
        let moved = crate::runtime::spirv_bind::widen_sampled_bands(&mut widened);
        assert_eq!(moved, 1, "only the sampler band moves");
        assert_eq!(*shader.words, widened);
        assert_eq!(shader.words[8], 3);
        assert_eq!(shader.words[12], M2V_TEXTURE_BINDING_BASE + 8);
        assert_eq!(shader.words[16], TOP_SAMPLER + SAMPLED_TAIL_WIDEN_OFFSET);
        // The bytes must not be allowed to disagree with the words.
        let from_bytes: Vec<u32> = shader
            .spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(from_bytes, widened);

        // No flags → the base Arc, un-relocated.
        let base = shader.fragment_words(false, false);
        assert!(Arc::ptr_eq(&base, &shader.words));

        // Both flags → sampled reloc first, then buffer band, matching the
        // historical per-draw mutation order.
        let mut expect = widened.clone();
        let n = crate::runtime::spirv_bind::offset_fragment_sampled_resource_bindings(&mut expect);
        assert_eq!(n, 2);
        let n = crate::runtime::spirv_bind::offset_fragment_buffer_bindings(&mut expect);
        assert_eq!(n, 1);
        let both = shader.fragment_words(true, true);
        assert_eq!(*both, expect);
        assert_eq!(both[8], 3 + FRAG_BUFFER_BINDING_OFFSET);
        assert_eq!(
            both[12],
            M2V_TEXTURE_BINDING_BASE + 8 + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        assert_eq!(
            both[16],
            TOP_SAMPLER + SAMPLED_TAIL_WIDEN_OFFSET + FRAG_SAMPLED_RESOURCE_BINDING_OFFSET
        );
        // Every relocated band stays clear of every un-relocated one.
        assert!(both[8] > crate::runtime::spirv_bind::COLOR_INPUT_BINDING_BASE);
        assert!(both[12] > both[8] && both[16] > both[12]);

        // Second call returns the cached variant (same allocation), and the
        // stored (widened) module is never mutated by a relocation.
        let again = shader.fragment_words(true, true);
        assert!(Arc::ptr_eq(&both, &again));
        assert_eq!(*shader.words, widened);
    }

    #[test]
    fn air_key_differs_by_stage_and_bytes() {
        let a = b"same-air-bytes";
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Fragment, a));
        assert_ne!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, b"other"));
        assert_eq!(air_key(Stage::Vertex, a), air_key(Stage::Vertex, a));
        assert_ne!(air_key_kernel(a, [16, 16, 1]), air_key_kernel(a, [8, 8, 1]));
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
            M2vCacheDecline::ReflectionDatalayoutMissing { stage: "vertex" },
            M2vCacheDecline::TranslationPending { stage: "vertex" },
            M2vCacheDecline::KernelLocalSizeZero {
                local_size: [0, 1, 1],
            },
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
        let id = ShaderId::render(Stage::Vertex, air);
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
            ensure_cached_async(air, Stage::Vertex, 7),
            "a stored failure must resolve the ask, or the packet never advances"
        );

        // This draw gets the real reason, unchanged.
        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 7) else {
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
        let id = ShaderId::render(Stage::Vertex, first);
        // The second identity, forced into the first one's bucket. Everything
        // but `digest` is the second shader's own.
        let collided = ShaderId {
            digest: id.digest,
            ..ShaderId::render(Stage::Vertex, second)
        };
        assert_ne!(
            ShaderId::render(Stage::Vertex, second).digest,
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
        let id = ShaderId::render(Stage::Vertex, air);
        let stored = M2vCacheDecline::VertexTranslate {
            detail: "unsupported instruction".to_string(),
        };
        global()
            .lock()
            .unwrap()
            .put(id, &Arc::from(&air[..]), Entry::Failed(stored.clone()));

        let Err(err) = translate_cached_reflected(air, Stage::Vertex, 7) else {
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
                ShaderId::render(Stage::Vertex, air),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Vertex, vec![0x03, 0x02, 0x23, 0x07])),
            );
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
        let id = ShaderId::render(Stage::Fragment, air);
        {
            let mut c = global().lock().unwrap();
            c.put(id, &Arc::from(&air[..]), Entry::Loading);
        }
        assert!(!ensure_cached_async(air, Stage::Fragment, 7));
        assert!(global().lock().unwrap().async_queue.is_empty());

        global().lock().unwrap().put(
            id,
            &Arc::from(&air[..]),
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
                ShaderId::render(Stage::Fragment, &hot),
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
                ShaderId::render(Stage::Fragment, &air),
                &Arc::from(&air[..]),
                Entry::Ready(synth_shader(Stage::Fragment, vec![0, 0, 0, 0])),
            );
        }
        assert!(
            ensure_cached_async(&hot, Stage::Fragment, 1),
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
