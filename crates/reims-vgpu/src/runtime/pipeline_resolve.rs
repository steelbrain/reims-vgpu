//! Resolving a draw's pipeline and both its shaders, once per pipeline object
//! rather than once per draw.
//!
//! # What this replaces, and what it cost
//!
//! Every draw chain used to walk the whole path from `pipeline_ref` to two
//! translated SPIR-V modules: an object-list read and a descriptor read for the
//! pipeline, a full TLV decode of that descriptor, then for each of the two
//! functions another object-list read, another descriptor read, a decode, and a
//! read of the whole MTLB container out of guest memory — followed by a linear
//! scan of each container for the wrapper magic and a content hash of each AIR
//! blob to key the translate cache.
//!
//! That is **eight guest page-table walks and five allocations per draw**, and
//! the answer is the same one every time: on a driven macos-13
//! sustained-animation boot, `pipeline_misses` and `shader_misses` are both
//! **zero** across 27 000 chains a second. `chain_phase`'s split of the span
//! (see [`crate::runtime::chain_phase::Phase::PipelineMtlb`]) priced the four
//! parts, per chain of 26.5 us:
//!
//! ```text
//! pl_desc_us    2.03 us   load_render_pipeline: 2 walks, 1 alloc, TLV decode
//! pl_mtlb_us    1.26 us   both load_mtlb: 6 walks, 4 allocs, 2 decodes
//! pl_xlate_us   1.19 us   both content hashes and the translate cache mutex
//! pl_air_us     0.20 us   both wrapper-magic scans
//! ```
//!
//! 4.68 us, **17.7 % of a draw chain**, spent arriving somewhere this device had
//! already been. A hit that costs what a compile costs is the thing this module
//! deletes.
//!
//! # What it removed, and why that is only worth 1.7 % of the frames
//!
//! Ten driven macos-13 sustained-animation boots, interleaved arms of
//! `REIMS_VGPU_PIPELINE_MEMO`. The span it targets collapses, and the four bars
//! are disjoint across every pair of boots:
//!
//! ```text
//!                 on (n=4)     off (n=4)
//! pl_desc_us        20 000        52 400     the three-entry identity check
//! pl_mtlb_us             0        34 900     gone
//! pl_air_us              0         5 300     gone
//! pl_xlate_us            0        33 100     gone
//! span per chain   0.89 us       4.86 us     -3.97 us, 15 % of a 26.5 us chain
//! ```
//!
//! **And the frames barely moved: +1.70 %, disjoint but at sep 0.9x.** What the
//! same boots say about why is not ambiguous, because it is also disjoint —
//! every on boot blocked longer than every off boot:
//!
//! ```text
//! slot_us     on  129 484  132 023  121 964  121 124     (min 121 124)
//!            off   90 427   78 494   68 489   83 515     (max  90 427)
//! ```
//!
//! `slot_us` is the drain worker blocked waiting for a ring slot, which is
//! waiting for the GPU. It rose 1.74 us a chain of the 3.97 us this saved, and
//! `drain_duty`'s `busy_us` is unchanged at ~840 ms a second across both arms.
//! So the worker did not go idle and did not draw more; it spent what it saved
//! waiting.
//!
//! **This rail is GPU-bound on this host, and that is the finding, not this
//! memo.** A CPU saving here converts at roughly the residual rate until the GPU
//! work comes down — the largest piece of which is the guest buffer gather, at
//! 5.8 GB/s over 427 000 transfer regions a second.
//!
//! The memo stays anyway, and not out of sunk cost. It is a strict reduction in
//! work with no measured regression on any metric; the balance it is measured
//! against is one host's (a discrete NVIDIA GPU with a fast CPU beside it) and
//! the unified-memory cells of the support matrix have the opposite one, where
//! guest page-table walks contend with the GPU for the same memory; and a probe
//! heavy enough to make the drain worker the bottleneck again would convert it.
//! What may **not** be said is that it bought 15 % of anything.
//!
//! # The identity a memo entry is checked against
//!
//! A guest object's identity is its **object-list entry**, and this module's
//! whole correctness argument is that sentence. A 12-byte entry is the guest's
//! own authoritative statement of what a ref means — its type tag, its
//! descriptor's address and its descriptor's length — and the guest writes it
//! into shared memory with no doorbell, which is exactly why every rail here
//! re-reads it instead of caching what it once said.
//!
//! So this memo re-reads it too. [`resolve`] reads the three entries a draw
//! depends on — the pipeline object and both function objects — and serves the
//! cached resolution only when all three are byte-identical to the ones the
//! entry was built from. Three 12-byte reads is three page-table walks, ~0.6 us,
//! against the 4.68 us of work they authorise skipping.
//!
//! ## What that check does not cover, stated exactly
//!
//! An entry that has not changed permits two things this memo will not notice:
//!
//! - the guest **rewriting a descriptor in place**, at the same address and the
//!   same length, to mean a different object;
//! - the guest **rewriting the MTLB bytes in place**, at the `blob_gva` and
//!   `blob_size` the unchanged descriptor names, to hold a different shader.
//!
//! Neither is a shape Metal produces. A `MTLRenderPipelineState` and a
//! `MTLFunction` are immutable once created; a recompile is a new object, and a
//! new object gets a new descriptor allocation and therefore a new entry. But
//! "Metal does not do this" is a claim about a guest and not about the contract,
//! which is why it is written here rather than assumed, and why the memo is
//! switchable: `REIMS_VGPU_PIPELINE_MEMO=off` takes every chain back down the
//! full path, so a guest that ever contradicts the paragraph above can be
//! confirmed against a binary that cannot be wrong about it.
//!
//! The hazard this is **not** is page recycling. A memo entry holds an
//! `Arc<CachedShader>` and an `Arc<RenderPipelineDescriptor>` — host-side owned
//! copies — so it never reads through a stale guest pointer and holds no
//! reference over a guest page. The failure mode of a wrong entry is a stale
//! *shader*, which is a visual defect, not a memory-safety one.
//!
//! # Counters
//!
//! On `store_routes`, so a boot says which path it took rather than leaving it
//! inferred from a frame rate:
//!
//! | route | meaning |
//! |---|---|
//! | `pipe_memo_hit` | all three entries matched; the resolution was reused |
//! | `pipe_memo_miss` | no entry for this `(task, pipeline_ref)` |
//! | `pipe_memo_stale` | an entry existed and one of the three had changed |
//! | `pipe_memo_evict` | [`MEMO_CAPACITY`] pushed a resolution out |
//! | `pipe_memo_forget_all` | a device reset invalidated every key at once |
//! | `pipe_memo_off` | the memo is switched off; one per chain |
//! | `preflight_memo_ready` | [`translations_ready`] served the exec preflight |
//! | `preflight_memo_absent` | no entry; the preflight loaded the AIR itself |
//! | `preflight_memo_stale` | an entry existed and the identity had changed |
//!
//! The three `preflight_*` routes are the same memo asked a **different
//! question** by a different caller — "are these shaders already translated?"
//! rather than "give me the resolution" — so they are counted apart from the
//! `pipe_memo_*` set. Summing the two would double-count a pipeline that both
//! the preflight and the draw asked about, which is every pipeline on a healthy
//! packet.
//!
//! `pipe_memo_stale` is the one to read. It is the population the paragraph
//! above says should be near zero on a steady desktop and non-zero only when a
//! guest genuinely replaces a pipeline; a boot where it tracks the hit count is
//! a boot where this memo is buying nothing and the check should be reconsidered
//! rather than the cap raised.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use crate::backend::vulkan::engine::DrawPreparationDecline;
use crate::model::DeviceState;
use crate::runtime::decode::resource::{ListObjectEntry, RenderPipelineDescriptor};
use crate::runtime::drain::note_store_route;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::m2v_cache::CachedShader;
use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
use crate::runtime::objects;

/// How many `(task_id, pipeline_ref)` resolutions are held at once.
///
/// Sized against what a guest asks for rather than picked: a macOS desktop
/// compositing session drives a few dozen distinct pipeline objects per task
/// across a handful of live tasks, so this is roughly an order of magnitude of
/// headroom. `pipe_memo_evict` is the reading that says whether that is true on
/// a rail — a non-zero count means the working set exceeded this and the cap is
/// costing hits, which is the only thing that would justify raising it.
///
/// An entry is three `Arc` clones and a `[ListObjectEntry; 3]`; the shaders it
/// names are the same allocations the translate cache already holds, so the cap
/// bounds pointers rather than shader bytes.
pub const MEMO_CAPACITY: usize = 1024;

/// The two buffer-index sets every draw of one pipeline used to rebuild from
/// that pipeline's attribute list.
///
/// Both are functions of `RenderPipelineDescriptor::vertex_attributes` alone —
/// no field of the draw request reaches either — so they belong to the
/// resolution and not to the draw, and building them per draw was two heap
/// allocations and two tree builds on the path `chain_phase` reports as
/// `binds_us`, this draw path's largest column.
///
/// Sorted slices rather than `BTreeSet`s. The population is the attribute list's
/// distinct buffer indices, which a real pipeline runs in the single digits, and
/// at that size a sorted `binary_search` is a cache line and a compare where a
/// tree is a pointer chase per level. The sort also makes the two sets
/// canonical, so an equality between two resolutions means what it reads as.
/// # The measurement did not confirm it, and this is what it said
///
/// The twelve interleaved boots quoted on
/// [`crate::runtime::m2v_cache::ShaderVariant::sampler_bindings`] carried this
/// change too, and `binds_us` — the column this targets — moved the **wrong
/// way**: 2.477 [2.418..2.525] before against 2.636 [2.510..2.695] after, per
/// draw. The ranges touch rather than separate, and the sub-column where the two
/// set builds used to sit (`binds_us` less `bind_phase`'s three parts) rose 0.04
/// us, which removing work cannot cause. So the honest reading is that
/// `binds_us`'s boot-to-boot spread is wider than what this change is worth, not
/// that the change costs anything.
///
/// It stays because it is strictly less work — two heap allocations and two tree
/// builds a draw become two `binary_search`es over data the pipeline resolution
/// already holds — and because per-draw allocation churn is a jitter source as
/// well as a mean one. **No claim is made that it bought time.** If a future
/// session wants one, `bind_phase` would need a fourth `Part` bracketing the
/// lookups themselves; the three it has today do not reach them.
pub struct VertexBindPlan {
    /// Buffer indices feeding at least one Constant-step attribute. A bind of
    /// one of these may not take the zero-copy rail: the engine prepends a CPU
    /// base-instance prefix to those bytes at prepare time.
    constant_step: Box<[u32]>,
    /// Every buffer index the attribute list names, whatever the attribute's
    /// format or stride turns out to be.
    ///
    /// Unfiltered on purpose, and this is the one place that reasoning now
    /// lives. An attribute with `format == 0` or a zero stride is skipped by the
    /// draw's attribute walk and reads no bytes, but excluding those here would
    /// make this set depend on the same two fields the walk re-derives through
    /// `bind_attribute_stride`, and the two would drift apart the first time
    /// that derivation changed. Listing an index the walk turns out to skip
    /// costs one gather and never correctness, which is the direction this set
    /// is allowed to be wrong in.
    attribute: Box<[u32]>,
}

impl VertexBindPlan {
    fn build(desc: &RenderPipelineDescriptor) -> Self {
        let mut constant_step: Vec<u32> = desc
            .vertex_attributes
            .iter()
            .filter(|a| {
                a.format != 0
                    && a.stride != 0
                    && crate::backend::vulkan::translate::vertex::step_function(a.declared_step_function)
                        == Ok(crate::backend::vulkan::engine::VertexStepFunction::Constant)
            })
            .map(|a| a.buffer_index)
            .collect();
        constant_step.sort_unstable();
        constant_step.dedup();
        let mut attribute: Vec<u32> = desc
            .vertex_attributes
            .iter()
            .map(|a| a.buffer_index)
            .collect();
        attribute.sort_unstable();
        attribute.dedup();
        Self {
            constant_step: constant_step.into_boxed_slice(),
            attribute: attribute.into_boxed_slice(),
        }
    }

    /// Whether a bind of this buffer index feeds a Constant-step attribute, and
    /// so must stay on the CPU staging read.
    pub fn is_constant_step(&self, buffer_index: u32) -> bool {
        self.constant_step.binary_search(&buffer_index).is_ok()
    }

    /// Whether the pipeline's attribute list names this buffer index at all.
    pub fn feeds_stage_in(&self, buffer_index: u32) -> bool {
        self.attribute.binary_search(&buffer_index).is_ok()
    }
}

/// Everything a draw chain needs from its pipeline ref, resolved once.
///
/// Every field is an `Arc` because the whole point is that a hit copies nothing:
/// `RenderPipelineDescriptor` owns two `Vec`s (its vertex attributes and its
/// colour attachments) and cloning it per draw would put back a fraction of the
/// allocation traffic this module exists to remove.
#[derive(Clone)]
pub struct ResolvedRenderPipeline {
    pub desc: Arc<RenderPipelineDescriptor>,
    pub vertex: Arc<CachedShader>,
    pub fragment: Arc<CachedShader>,
    /// Derived from `desc` and memoized with it — see [`VertexBindPlan`].
    pub bind_plan: Arc<VertexBindPlan>,
}

/// The three object-list entries a resolution depends on, in the order they are
/// read: the pipeline object, then the vertex and fragment function objects.
///
/// A fixed-size array rather than three named fields because the only operation
/// on it is equality against a freshly-read one, and a named-field struct
/// invites a comparison that forgets a member. See the module doc for why the
/// entry is the identity.
type EntryTriple = [ListObjectEntry; 3];

struct Entry {
    identity: EntryTriple,
    resolved: ResolvedRenderPipeline,
}

/// A map that holds at most `CAP` entries, dropping the oldest *insertion* to
/// stay there.
///
/// The capacity is a const parameter rather than a field so it cannot be passed
/// wrong at a second construction site, and the map is private with `insert` as
/// its only mutator so a caller cannot reach `entries` and grow it past the
/// bound — `AGENTS.md`'s "make the invariant unrepresentable" rather than a scan
/// looking for places that forgot to check.
///
/// Oldest-insertion and not least-recently-used: a resolution's value does not
/// decay with time, and the population this bounds is pipeline objects a guest
/// creates at app launch and then keeps. LRU would buy a different eviction
/// order for a working set that does not exceed the cap at all —
/// `pipe_memo_evict` is what says whether that assumption holds on a rail.
struct BoundedByInsertion<K: Copy + Eq + Hash, V, const CAP: usize> {
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Copy + Eq + Hash, V, const CAP: usize> BoundedByInsertion<K, V, CAP> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key)
    }

    /// File `value` under `key`, returning whatever the cap pushed out.
    ///
    /// Re-filing a live key replaces its value and does **not** queue a second
    /// slot in the order — otherwise the deque outgrows the map and starts
    /// evicting keys that are the newest thing in it.
    fn insert(&mut self, key: K, value: V) -> Option<K> {
        if self.entries.insert(key, value).is_some() {
            return None;
        }
        self.order.push_back(key);
        if self.order.len() <= CAP {
            return None;
        }
        let old = self.order.pop_front()?;
        self.entries.remove(&old);
        Some(old)
    }
}

type Memo = BoundedByInsertion<(u32, u32), Entry, MEMO_CAPACITY>;

fn memo() -> &'static Mutex<Memo> {
    static MEMO: OnceLock<Mutex<Memo>> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(Memo::new()))
}

/// Drop every resolution. Called from `device_reset`, which is where the keys
/// stop meaning anything — see the comment at that call site.
pub fn forget_all() {
    let mut m = memo().lock().unwrap_or_else(|e| e.into_inner());
    *m = Memo::new();
    note_store_route("pipe_memo_forget_all");
}

/// Whether the memo is on. See [`crate::env::PIPELINE_MEMO`].
fn memo_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            crate::env::read(crate::env::PIPELINE_MEMO).0,
            crate::env::Switch::Off
        )
    })
}

/// Read the object-list entries for a pipeline and the two functions it names.
///
/// `None` from any of the three is "the guest has not told us", which the full
/// path reports with its own rung — so a miss here does not refuse, it declines
/// to *serve from the memo* and lets [`resolve_uncached`] produce the named
/// failure. There is exactly one place a draw's pipeline failure is described
/// and it is not this function.
///
/// Neither func ref can be zero here. `ref == 0` is "no function bound" and
/// would read object-list slot 0 rather than refusing, but every triple this is
/// called with comes from a descriptor [`resolve_uncached`] already accepted,
/// and `load_render_pipeline` refuses a zero in either stage before returning
/// one.
fn read_identity<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
    vertex_ref: u32,
    fragment_ref: u32,
) -> Option<EntryTriple> {
    Some([
        objects::lookup_list_entry(state, host, task_id, pipeline_ref)?,
        objects::lookup_list_entry(state, host, task_id, vertex_ref)?,
        objects::lookup_list_entry(state, host, task_id, fragment_ref)?,
    ])
}

/// Whether `pipeline_ref`'s two shaders are **already translated**, answered
/// from this memo alone and without resolving, translating or reading any AIR.
///
/// # Why the exec preflight can ask this instead of loading the AIR
///
/// `ExecPhase::Preflight` exists to answer one question before any record of a
/// packet runs: will executing this stream have to wait for a translation? It
/// answered it by resolving each pipeline's AIR out of guest memory and offering
/// it to `m2v_cache::ensure_cached_async` — three guest resolves at **4.3 us a
/// pipeline ref, 12 700 refs a second, ~54 ms of every second**.
///
/// A memo hit answers the same question for ~0.6 us, and it is not a weaker
/// answer:
///
/// - an entry is only ever filed after a successful [`resolve_uncached`], and it
///   holds the two `Arc<CachedShader>` that resolution produced — so **an entry
///   existing means those shaders were translated**;
/// - the m2v translate cache is **unbounded and nothing evicts it** (its only
///   removal is `forget_if_transient`, dropping a transient failure so it can be
///   retried), so a shader translated once is translated for the life of the
///   process;
/// - therefore even if this memo's own [`MEMO_CAPACITY`] evicts the entry before
///   the draw reaches it, the draw's `translate_cached_reflected` finds the
///   shader ready and does not translate synchronously. The eviction costs the
///   resolve, never the translation.
///
/// The identity check is the same three object-list entry reads [`resolve`]
/// makes, with the same coverage and the same two documented gaps — a
/// descriptor or an MTLB rewritten in place. Those gaps are not widened by
/// asking here: the draw that follows performs the identical check, so a guest
/// that could fool this one already fools that one.
///
/// Returns `false` whenever the memo is switched off, so
/// `REIMS_VGPU_PIPELINE_MEMO=off` takes the preflight back down its full path
/// along with everything else this module short-circuits.
///
/// # What it measured, two driven macos-13 boots
///
/// ```text
/// preflight_memo_ready    716 427 / 708 872
/// preflight_memo_absent     1 207 /   1 428      99.83 % hit
/// preflight_memo_stale          0 /       0
///
///                    before        after
/// preflight ms/s     ~76           15.19 / 14.99      -80 %
/// air ms/s           53.90/54.23    0.00 /  0.00
/// cache ms/s         16.30/16.60    0.00 /  0.00
/// refs ms/s           6.24/6.23     6.21 /  6.10      control, held
/// refs_us/call        0.41/0.42     0.39 /  0.40      control, held
/// op0x37 us/packet  101.2          94.34 / 94.43
/// ```
///
/// **~60 ms/s off the drain worker.** `cache` falling is not a second win and
/// not a regression — it only ever ran on a miss, so it tracks the hit rate by
/// construction. `refs` still runs for every packet and is the control that says
/// the two boots are comparable.
///
/// `preflight_memo_stale` being **0** across 1.4 million asks is the reading
/// that matters for the identity check: on a steady desktop the three entries
/// never move, which is what the module doc's argument predicts.
///
/// # And on the other five drivers, where it is *not* zero
///
/// One undriven boot of each x86 rail, which catches the cold memo and the
/// pipeline churn of app launch rather than a settled desktop:
///
/// ```text
/// macos-11  ready=2353  absent= 617  stale= 0     79 % hit
/// macos-12  ready=1613  absent= 951  stale= 0     63 %
/// macos-13  ready= 930  absent= 919  stale= 0     50 %
/// macos-14  ready= 906  absent=1137  stale=19     44 %
/// macos-15  ready= 838  absent=1610  stale= 0     34 %
/// macos-26  ready=5415  absent=2146  stale=23     72 %
/// ```
///
/// Two things worth keeping. First, **the 99.83 % above is a steady-state
/// number, not a universal one** — a cold memo during boot hits 34-79 %, which
/// is the population this optimisation does not serve and does not need to.
///
/// Second, and more useful: **`stale` is not always zero.** macos-14 and
/// macos-26 replace pipelines during boot and the identity check catches it,
/// falling back to the full preflight exactly as designed. That is the reading
/// that says the check is **not vacuous** — a comparison that returned zero on
/// every rail under every condition would be indistinguishable from one that
/// could never fire, and this one demonstrably fires on real guest behaviour and
/// then declines the entry.
///
/// One artifact to know: `sum vs preflight_us` drops to ~0.41, because the three
/// timed sub-spans now cover almost nothing while this check — which is inside
/// `preflight_us` and outside all three — covers the rest. That is the
/// instrument no longer tiling its subject, not time going missing.
pub fn translations_ready<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> bool {
    if !memo_enabled() {
        return false;
    }
    let cached = {
        let m = memo().lock().unwrap_or_else(|e| e.into_inner());
        m.get(&(task_id, pipeline_ref)).map(|e| {
            (
                e.identity,
                e.resolved.desc.vertex_func_ref,
                e.resolved.desc.fragment_func_ref,
            )
        })
    };
    let Some((identity, vertex_ref, fragment_ref)) = cached else {
        note_store_route("preflight_memo_absent");
        return false;
    };
    if read_identity(state, host, task_id, pipeline_ref, vertex_ref, fragment_ref)
        == Some(identity)
    {
        note_store_route("preflight_memo_ready");
        return true;
    }
    note_store_route("preflight_memo_stale");
    false
}

/// Resolve `pipeline_ref` to its descriptor and both translated shaders.
///
/// Serves a memoized resolution when the three object-list entries it was built
/// from still read identically; see the module doc for what that check is and
/// what it does not cover.
pub fn resolve<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<ResolvedRenderPipeline, DrawPreparationDecline> {
    if !memo_enabled() {
        note_store_route("pipe_memo_off");
        return resolve_uncached(state, host, task_id, pipeline_ref);
    }

    // The func refs come from the cached entry rather than from a fresh
    // descriptor read: reading the descriptor to learn which functions to check
    // would pay most of what the memo is here to skip. The pipeline object's own
    // entry is the first of the three compared, so a pipeline that has been
    // replaced fails the check before its stale func refs are believed for
    // anything but the two reads that then also fail it.
    let cached = {
        let m = memo().lock().unwrap_or_else(|e| e.into_inner());
        m.get(&(task_id, pipeline_ref))
            .map(|e| (e.identity, e.resolved.clone()))
    };
    if let Some((identity, resolved)) = cached {
        let fresh = read_identity(
            state,
            host,
            task_id,
            pipeline_ref,
            resolved.desc.vertex_func_ref,
            resolved.desc.fragment_func_ref,
        );
        if fresh == Some(identity) {
            note_store_route("pipe_memo_hit");
            return Ok(resolved);
        }
        note_store_route("pipe_memo_stale");
    } else {
        note_store_route("pipe_memo_miss");
    }

    let resolved = resolve_uncached(state, host, task_id, pipeline_ref)?;
    // Built after the resolution and from the same refs it used, so an entry can
    // only be filed under an identity that was readable at the moment the
    // resolution was taken. An unreadable one files nothing rather than filing a
    // resolution no later read can invalidate.
    if let Some(identity) = read_identity(
        state,
        host,
        task_id,
        pipeline_ref,
        resolved.desc.vertex_func_ref,
        resolved.desc.fragment_func_ref,
    ) {
        let evicted = memo().lock().unwrap_or_else(|e| e.into_inner()).insert(
            (task_id, pipeline_ref),
            Entry {
                identity,
                resolved: resolved.clone(),
            },
        );
        if evicted.is_some() {
            note_store_route("pipe_memo_evict");
        }
    }
    Ok(resolved)
}

/// The full path: object list → descriptor → decode → MTLB → AIR → SPIR-V, for
/// the pipeline and both of its functions.
///
/// This is the only place a draw's pipeline resolution can fail, and each of its
/// seven refusals keeps the `DrawPreparationDecline` variant it always had — the
/// memo in front of it neither adds a failure nor renames one.
fn resolve_uncached<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    pipeline_ref: u32,
) -> Result<ResolvedRenderPipeline, DrawPreparationDecline> {
    let desc = crate::runtime::draw::load_render_pipeline(state, host, task_id, pipeline_ref)
        .ok_or(DrawPreparationDecline::PipelineMissing {
            task_id,
            pipeline_ref,
        })?;
    // The same three sub-phases the call site used to open around this work,
    // moved in with it. They are inert outside a live `ChainTimer`, so the two
    // non-draw callers of the loaders below are unaffected — and on the draw
    // rail `pl_desc_us` now brackets the memo's own identity check, which is
    // what makes the hit path's cost readable against the miss path's.
    use crate::runtime::chain_phase::{enter, Phase};
    enter(Phase::PipelineMtlb);
    let v_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.vertex_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::VertexMtlbMissing {
        task_id,
        function_ref: desc.vertex_func_ref,
    })?;
    let f_mtlb = load_mtlb(
        state,
        host,
        task_id,
        desc.fragment_func_ref,
        AirLoadRail::Draw,
    )
    .ok_or(DrawPreparationDecline::FragmentMtlbMissing {
        task_id,
        function_ref: desc.fragment_func_ref,
    })?;
    enter(Phase::PipelineAir);
    let v_air = crate::runtime::mtlb::extract_air(&v_mtlb).map_err(|reason| {
        DrawPreparationDecline::VertexAirExtract {
            function_ref: desc.vertex_func_ref,
            reason,
        }
    })?;
    let f_air = crate::runtime::mtlb::extract_air(&f_mtlb).map_err(|reason| {
        DrawPreparationDecline::FragmentAirExtract {
            function_ref: desc.fragment_func_ref,
            reason,
        }
    })?;
    enter(Phase::PipelineXlate);
    let vertex = crate::runtime::m2v_cache::translate_cached_reflected(
        v_air,
        metal2vulkan::passes::Stage::Vertex,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::VertexTranslate {
        pipeline_ref,
        reason,
    })?;
    let fragment = crate::runtime::m2v_cache::translate_cached_reflected(
        f_air,
        metal2vulkan::passes::Stage::Fragment,
        pipeline_ref,
    )
    .map_err(|reason| DrawPreparationDecline::FragmentTranslate {
        pipeline_ref,
        reason,
    })?;
    let bind_plan = Arc::new(VertexBindPlan::build(&desc));
    Ok(ResolvedRenderPipeline {
        desc: Arc::new(desc),
        vertex,
        fragment,
        bind_plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::decode::resource::VertexAttribute;

    fn entry(gva: u64, len: u32, ot: u8) -> ListObjectEntry {
        ListObjectEntry {
            object_type: ot,
            descriptor_length: len,
            descriptor_gva: gva,
        }
    }

    /// An empty memo must answer **not ready**, and that direction is the whole
    /// safety of asking it.
    ///
    /// `translations_ready` gates whether the exec preflight skips loading a
    /// pipeline's AIR. A wrong `false` costs the resolve it was trying to save;
    /// a wrong `true` tells the packet its shaders are translated when nothing
    /// has translated them, and the draw then meets an untranslated pipeline
    /// with the packet already committed. So the absent case is pinned
    /// explicitly rather than left to follow from the `Option` being `None`.
    #[test]
    fn an_absent_memo_entry_is_never_reported_ready() {
        use crate::model::DeviceId;
        use crate::runtime::host::FakeHost;

        forget_all();
        let state = DeviceState::new(DeviceId(1), 12);
        let host = FakeHost::new();
        assert!(
            !translations_ready(&state, &host, 7, 9),
            "a pipeline this memo has never resolved must send the preflight \
             down its own path, not be waved through as translated"
        );
    }

    /// The cap has to evict, or a guest that cycles pipeline refs grows this map
    /// without bound for the life of the VM.
    #[test]
    fn the_capacity_evicts_the_oldest_insertion() {
        let mut m: BoundedByInsertion<u32, u32, 4> = BoundedByInsertion::new();
        assert_eq!(m.insert(1, 10), None, "under the cap evicts nothing");
        for k in 2..=4 {
            assert_eq!(m.insert(k, k * 10), None);
        }
        assert_eq!(m.insert(5, 50), Some(1), "the oldest insertion is named");
        assert_eq!(m.entries.len(), 4, "the cap holds");
        assert_eq!(m.get(&1), None, "and it is gone");
        assert_eq!(m.get(&5), Some(&50));
    }

    /// Re-inserting a live key must not queue a second eviction slot for it, or
    /// the order deque outgrows the map and evicts entries that are still the
    /// newest thing in it.
    #[test]
    fn re_inserting_a_key_does_not_grow_the_order() {
        let mut m: BoundedByInsertion<u32, u32, 4> = BoundedByInsertion::new();
        for v in 0..64 {
            assert_eq!(m.insert(5, v), None, "a replacement evicts nothing");
        }
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.order.len(), 1, "one key, one slot in the order");
        assert_eq!(m.get(&5), Some(&63), "and the newest value won");
    }

    /// The identity is compared as a whole. A change to any of the three
    /// entries, in any of their three fields, has to read as different — this is
    /// the check the module's whole correctness argument rests on.
    #[test]
    fn every_field_of_every_entry_is_part_of_the_identity() {
        let base: EntryTriple = [
            entry(0x1000, 64, 7),
            entry(0x2000, 32, 6),
            entry(0x3000, 32, 6),
        ];
        for slot in 0..3 {
            let mut gva = base;
            gva[slot].descriptor_gva += 1;
            assert_ne!(base, gva, "slot {slot} descriptor_gva");
            let mut len = base;
            len[slot].descriptor_length += 1;
            assert_ne!(base, len, "slot {slot} descriptor_length");
            let mut ot = base;
            ot[slot].object_type += 1;
            assert_ne!(base, ot, "slot {slot} object_type");
        }
    }

    /// The two sets [`VertexBindPlan`] carries used to be rebuilt inside the
    /// draw path from the same attribute list, and this pins the classification
    /// they replaced rather than the shape of the code that does it.
    ///
    /// The interesting rows are the ones the old inline filter got right by
    /// construction and a rewrite can get wrong: a Constant-step attribute whose
    /// `format` is zero, and one whose `stride` is zero, are **not** constant
    /// step for this purpose — the draw's attribute walk skips them, so a bind
    /// of their buffer must stay eligible for the zero-copy rail — while both
    /// still count as named by the attribute list, because that set is
    /// deliberately unfiltered.
    #[test]
    fn the_bind_plan_separates_constant_step_from_merely_named() {
        const CONSTANT: Option<u32> = Some(0);
        const PER_INSTANCE: Option<u32> = Some(2);
        let attr = |buffer_index, format, stride, declared_step_function| VertexAttribute {
            location: 0,
            format,
            offset: 0,
            buffer_index,
            stride,
            declared_step_function,
            declared_step_rate: None,
        };
        let desc = RenderPipelineDescriptor {
            vertex_attributes: vec![
                attr(1, 0x21, 16, CONSTANT),      // constant, and it counts
                attr(2, 0x21, 16, PER_INSTANCE),  // named, not constant
                attr(3, 0, 16, CONSTANT),         // format 0: the walk skips it
                attr(4, 0x21, 0, CONSTANT),       // stride 0: the walk skips it
                attr(1, 0x21, 32, PER_INSTANCE),  // a second attribute on buffer 1
                attr(5, 0x21, 16, None),          // undeclared step is per-vertex
            ],
            ..Default::default()
        };
        let plan = VertexBindPlan::build(&desc);

        assert!(plan.is_constant_step(1), "declared Constant with real bytes");
        for index in [2, 3, 4, 5] {
            assert!(
                !plan.is_constant_step(index),
                "buffer {index} must keep the zero-copy rail"
            );
        }
        // Unfiltered: every index the list mentions, skipped by the walk or not.
        for index in 1..=5 {
            assert!(plan.feeds_stage_in(index), "buffer {index} is named");
        }
        assert!(!plan.feeds_stage_in(0), "an index the list never names");
        assert!(!plan.feeds_stage_in(6));
    }

    /// A pipeline with no vertex block answers "no" to both questions rather
    /// than panicking on an empty search, which is the shape every fullscreen
    /// pass that builds its vertices in the shader takes.
    #[test]
    fn an_empty_attribute_list_names_nothing() {
        let plan = VertexBindPlan::build(&RenderPipelineDescriptor::default());
        assert!(!plan.is_constant_step(0));
        assert!(!plan.feeds_stage_in(0));
    }
}
