//! Where a draw chain's wall clock goes on the *runtime* side of the engine
//! boundary, which is where most of it goes and where nothing was looking.
//!
//! # The hole this fills
//!
//! Two censuses already divide a draw and they do not meet. `drain_duty`'s
//! `draw_us` brackets [`crate::runtime::draw::encode_draw_chain`] from
//! the drain worker; `draw_phase` brackets the Vulkan engine's own
//! `execute_draw_request` from inside. On a driven x86/PCI boot, one second:
//!
//! ```text
//! drain_duty  draw_us=499184 draws=275              1.82 ms per draw
//! draw_phase  draws=275  (its phases summed)        0.34 ms per draw
//! ```
//!
//! So **82% of draw time — 407 ms of that second, 41% of the wall clock — is
//! spent between the two brackets and no phase claims it.** That is the largest
//! unattributed cost in the device, larger than the deferred flush's fence wait
//! in the same second (464 ms), and unlike the fence wait it is CPU work held
//! under the device lock, which is guest stall rather than GPU latency.
//!
//! The gap is real work, not bracket error: `encode_draw_chain` resolves the
//! pipeline, extracts and translates both AIR blobs, materializes every buffer
//! bind, resolves every sampled image, resolves the colour Load seed and
//! assembles the engine's `DrawRequest` — all before the engine is called — and
//! then routes the Store afterwards. `draw_phase` starts at the engine and ends
//! at the engine, so none of that is inside it.
//!
//! # Why these split points
//!
//! Same rule `draw_phase` uses: split where the fix changes, not where the code
//! happens to be indented. Each phase below has a different lever, and a single
//! `setup_us` bar could not choose between them — which is exactly the mistake
//! `draw_phase`'s own doc records having made once.
//!
//! | phase | from | to | what would fix it |
//! |---|---|---|---|
//! | `prep` | `encode_draw_chain` entry | the metal2vulkan call | the CLEAR-only fast path, which never leaves this phase |
//! | `pipeline` | there | both shaders translated | `runtime::pipeline_resolve` (backend-vulkan only, so named not linked), which is what the five bars below it were measured to justify |
//! | `binds` | there | vertex/fragment buffer content materialized | zero-copy buffer binds |
//! | `sampled` | there | every sampled image and sampler resolved | the gather witness and the sampled cache |
//! | `seed` | there | the colour Load seed resolved | resident Load elision |
//! | `assemble` | there | the engine `DrawRequest` is built | allocation churn in request assembly |
//! | `engine` | there | `execute_draw_request` returns | whatever `draw_phase` says |
//! | `store` | there | `encode_draw_chain` returns | the deferred Store rails |
//!
//! # It divides against two lines, and that is the point
//!
//! The eight numbers are charged one at a time and committed on `Drop`, so they
//! sum to the chain. Two identities make the reading self-checking rather than
//! merely plausible, and a reader should check both before believing any single
//! bar:
//!
//! - The eight sum to `drain_duty`'s `draw_us`, and `chain_phase draws` equals
//!   its `draws`. A shortfall means a draw path that does not pass through
//!   [`ChainTimer`] at all.
//! - `engine_us` equals `draw_phase`'s phases summed — all of them, including
//!   the five the pipeline span is divided into, which are carved out of
//!   `pipeline_us` rather than added beside it. A shortfall there
//!   means the engine is being entered by some route this bracket does not see.
//!
//! This line's own `pipeline_us` is divided the same way, by the five
//! `pipeline_*_us` bars — see [`Phase::PipelineMtlb`] for why. They are carved
//! out of it, so it is the six that sum to the chain and not `pipeline_us`
//! alone.
//!
//! Neither identity is asserted in code, because a census that panics on its own
//! arithmetic is worse than one a reader can divide. They are stated here so the
//! division is the first thing done with the line.
//!
//! # What it does not do
//!
//! It reports no loss. Every phase here is a draw that drew; a slow draw is not
//! a declined one, and `AGENTS.md`'s rule that a census must not be the only
//! record of lost guest work is satisfied by there being no loss to record. The
//! decline paths keep their own typed reasons and emit them as they always did.
//!
//! A chain that returns early — a decline, a resident-chain intermediate, a
//! deferred Store that returns from the middle of its own block — charges its
//! remainder to whichever phase was open, because the commit is in `Drop`. That
//! is deliberate for the same reason `draw_phase` gives: an exit is not a phase,
//! and threading a commit through every `?` is the one thing guaranteed to go
//! stale.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};

/// Phase slots, in the order a draw chain passes through them.
///
/// The five `Pipeline*` sub-phases are appended after `Store` rather than
/// inserted next to `Pipeline`, so every existing ordinal keeps its value and
/// [`PHASES`] stays derived from [`Phase::LAST`] rather than hand-counted. Same
/// arrangement, and the same reason, as `draw_phase`'s pipeline split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Prep = 0,
    /// What is left of the pipeline span once the five below are taken out of
    /// it: the render-target extent check and the phase's own bookkeeping.
    Pipeline = 1,
    Binds = 2,
    Sampled = 3,
    Seed = 4,
    Assemble = 5,
    Engine = 6,
    Store = 7,
    /// `gva_alloc_generation` — the colour0 target's allocation walk, taken once
    /// per chain because two walks of one address across a submit are two
    /// answers.
    PipelineGen = 8,
    /// `load_render_pipeline`: one object-list lookup, one descriptor read out
    /// of guest memory, and the decode of the bytes it read.
    PipelineDesc = 9,
    /// Both `load_mtlb` calls.
    ///
    /// # Why the pipeline span is split at all
    ///
    /// It is the third-largest column on the line — 17.8 % of a chain on a
    /// driven macos-13 sustained-animation boot, 4.66 us of 26.2 — and the whole
    /// of it is spent arriving at a shader this device already holds:
    /// `shader_misses` and `pipeline_misses` are both **zero** over a boot that
    /// runs 27 000 chains a second.
    ///
    /// A hit that costs that much is a lookup priced like a compile, and six
    /// candidates share the span. This bar is the one with a structural reason to
    /// be large: each `load_mtlb` resolves an object descriptor (an object-list
    /// read plus a descriptor read) and then reads the whole MTLB container out
    /// of guest memory into a fresh `Vec`, so the pair is six guest-memory walks
    /// and three allocations per chain for bytes that have not changed since the
    /// guest compiled them.
    PipelineMtlb = 10,
    /// Both `extract_air` calls — a linear scan of each MTLB container for the
    /// wrapper magic, so it is charged separately from the read that produced
    /// the container and from the hash that consumes the blob.
    PipelineAir = 11,
    /// Both `translate_cached_reflected` calls: hashing each AIR blob for its
    /// content key, and the global cache mutex. The translate itself is a miss
    /// path and this boot has none.
    PipelineXlate = 12,
    /// The CLEAR-seed Store loop at the head of `draw_vk`: one full-surface
    /// solid buffer per colour attachment, a channel swap for a type-11 target,
    /// and the write of it into the guest's pages.
    ///
    /// # Why the prep span is split at all
    ///
    /// `prep_us` is 0.16 µs a chain on a compositing load and **9.27 µs** on the
    /// `blur=40` load that runs the drain worker at duty 0.90 — 22.8 % of that
    /// chain, second only to the engine. A bar that moves by a factor of 58
    /// between two workloads is measuring something one of them does and the
    /// other does not, and the span holds exactly two candidates.
    PrepSeed = 13,
    /// `sync_store_allowed_pages` — the page-table walk that bounds the
    /// synchronous GVA Store, taken before any GPU work so the set predates the
    /// submit.
    PrepPages = 14,
}

impl Phase {
    /// Highest ordinal, so [`PHASES`] is derived from the enum rather than
    /// hand-counted beside it.
    const LAST: Phase = Phase::PrepPages;
}

const PHASES: usize = Phase::LAST as usize + 1;

/// Nanoseconds, per [`crate::observe::phase_clock`]. `prep_us` and
/// `pipeline_us` are a couple of microseconds across a whole chain, so their
/// constituent spans sit under the microsecond a truncating accumulator can
/// see.
static ACC: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
static CHAINS: AtomicU64 = AtomicU64::new(0);
static MAX_NS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The phase currently being charged on this thread, and when it opened.
    /// `None` when no [`ChainTimer`] is live, which makes a stray [`enter`]
    /// inert rather than mis-attributing to whatever ran last.
    static OPEN: Cell<Option<(Phase, Instant)>> = const { Cell::new(None) };
}

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChainPhaseWindow {
    pub prep_us: u64,
    /// The residue of the pipeline span; the five below are carved out of it and
    /// the six together are what `pipeline_us` used to be alone.
    pub pipeline_us: u64,
    pub pipeline_gen_us: u64,
    pub pipeline_desc_us: u64,
    pub pipeline_mtlb_us: u64,
    pub pipeline_air_us: u64,
    pub pipeline_xlate_us: u64,
    pub binds_us: u64,
    pub sampled_us: u64,
    pub seed_us: u64,
    pub assemble_us: u64,
    pub engine_us: u64,
    pub store_us: u64,
    /// The two spans carved out of `prep_us`; the three together are what
    /// `prep_us` used to be alone.
    pub prep_seed_us: u64,
    pub prep_pages_us: u64,
    pub chains: u64,
    pub max_us: u64,
}

/// Take and clear the window. `None` when no chain ran, so an idle second costs
/// no line.
pub fn take_window() -> Option<ChainPhaseWindow> {
    let chains = CHAINS.swap(0, Ordering::Relaxed);
    let w = ChainPhaseWindow {
        prep_us: to_us(ACC[Phase::Prep as usize].swap(0, Ordering::Relaxed)),
        pipeline_us: to_us(ACC[Phase::Pipeline as usize].swap(0, Ordering::Relaxed)),
        pipeline_gen_us: to_us(ACC[Phase::PipelineGen as usize].swap(0, Ordering::Relaxed)),
        pipeline_desc_us: to_us(ACC[Phase::PipelineDesc as usize].swap(0, Ordering::Relaxed)),
        pipeline_mtlb_us: to_us(ACC[Phase::PipelineMtlb as usize].swap(0, Ordering::Relaxed)),
        pipeline_air_us: to_us(ACC[Phase::PipelineAir as usize].swap(0, Ordering::Relaxed)),
        pipeline_xlate_us: to_us(ACC[Phase::PipelineXlate as usize].swap(0, Ordering::Relaxed)),
        binds_us: to_us(ACC[Phase::Binds as usize].swap(0, Ordering::Relaxed)),
        sampled_us: to_us(ACC[Phase::Sampled as usize].swap(0, Ordering::Relaxed)),
        seed_us: to_us(ACC[Phase::Seed as usize].swap(0, Ordering::Relaxed)),
        assemble_us: to_us(ACC[Phase::Assemble as usize].swap(0, Ordering::Relaxed)),
        engine_us: to_us(ACC[Phase::Engine as usize].swap(0, Ordering::Relaxed)),
        store_us: to_us(ACC[Phase::Store as usize].swap(0, Ordering::Relaxed)),
        prep_seed_us: to_us(ACC[Phase::PrepSeed as usize].swap(0, Ordering::Relaxed)),
        prep_pages_us: to_us(ACC[Phase::PrepPages as usize].swap(0, Ordering::Relaxed)),
        chains,
        max_us: to_us(MAX_NS.swap(0, Ordering::Relaxed)),
    };
    (chains > 0).then_some(w)
}

/// Close the open phase and open `next`. Inert when no [`ChainTimer`] is live.
pub fn enter(next: Phase) {
    OPEN.with(|open| {
        let now = Instant::now();
        if let Some((phase, since)) = open.get() {
            ACC[phase as usize].fetch_add(
                charge_ns(now.saturating_duration_since(since)),
                Ordering::Relaxed,
            );
            open.set(Some((next, now)));
        }
    });
}

/// Charges one draw chain's wall clock to one phase at a time.
///
/// Held by value in `encode_draw_chain`; [`enter`] closes the open phase and
/// opens the next from anywhere below it, including inside the metal2vulkan
/// call, without threading a `&mut` through a 1500-line function.
///
/// A live timer is saved and restored across nesting, and the restored outer
/// phase reopens at the *current* instant rather than its original one, so an
/// inner chain's time is charged to the inner chain's phases exactly once
/// instead of to both. No caller nests today; getting it wrong silently would
/// double the largest number on the line, so it is handled rather than assumed.
pub struct ChainTimer {
    started: Instant,
    outer: Option<(Phase, Instant)>,
}

impl ChainTimer {
    /// Open [`Phase::Prep`] and start the chain's total.
    pub fn start() -> Self {
        let now = Instant::now();
        let outer = OPEN.with(|open| open.replace(Some((Phase::Prep, now))));
        Self {
            started: now,
            outer,
        }
    }
}

impl Drop for ChainTimer {
    fn drop(&mut self) {
        let now = Instant::now();
        OPEN.with(|open| {
            if let Some((phase, since)) = open.get() {
                ACC[phase as usize].fetch_add(
                    charge_ns(now.saturating_duration_since(since)),
                    Ordering::Relaxed,
                );
            }
            open.set(self.outer.map(|(phase, _)| (phase, now)));
        });
        let total = charge_ns(now.saturating_duration_since(self.started));
        CHAINS.fetch_add(1, Ordering::Relaxed);
        MAX_NS.fetch_max(total, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phases must sum to the chain, because the whole reading is that sum
    /// divided against `drain_duty`'s `draw_us`.
    #[test]
    fn the_phases_sum_to_the_chain() {
        let _ = take_window();
        {
            let _t = ChainTimer::start();
            std::thread::sleep(std::time::Duration::from_millis(2));
            enter(Phase::Engine);
            std::thread::sleep(std::time::Duration::from_millis(2));
            enter(Phase::Store);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let w = take_window().expect("one chain ran");
        assert_eq!(w.chains, 1);
        let sum = w.prep_us
            + w.pipeline_us
            + w.pipeline_gen_us
            + w.pipeline_desc_us
            + w.pipeline_mtlb_us
            + w.pipeline_air_us
            + w.pipeline_xlate_us
            + w.binds_us
            + w.sampled_us
            + w.seed_us
            + w.assemble_us
            + w.engine_us
            + w.store_us;
        // Each `enter` reads the clock twice at the same instant, so the sum can
        // trail the total by at most the rounding of three microsecond
        // truncations.
        assert!(
            sum + 3 >= w.max_us && sum <= w.max_us + 3,
            "phases {sum} must sum to the chain {}",
            w.max_us
        );
        assert!(w.prep_us >= 1_000, "prep held the first sleep: {w:?}");
        assert!(w.engine_us >= 1_000, "engine held the second sleep: {w:?}");
        assert!(w.store_us >= 1_000, "store held the third sleep: {w:?}");
    }

    /// The pipeline sub-phases are **carved out of** `pipeline_us`, not added
    /// beside it. A reader who sums the six against the chain has to get the
    /// chain, so a sub-phase that also charged the residue would double its own
    /// time and read as the largest bar on the line.
    #[test]
    fn a_pipeline_sub_phase_is_carved_out_of_the_residue() {
        let _ = take_window();
        {
            let _t = ChainTimer::start();
            enter(Phase::PipelineMtlb);
            std::thread::sleep(std::time::Duration::from_millis(3));
            enter(Phase::Store);
        }
        let w = take_window().expect("one chain ran");
        assert!(
            w.pipeline_mtlb_us >= 2_000,
            "the sleep charged the sub-phase: {w:?}"
        );
        assert!(w.pipeline_us < 1_000, "and not the residue as well: {w:?}");
    }

    /// A phase change with no timer live must not charge anything, or a stray
    /// call would attribute unrelated work to whichever phase ran last.
    #[test]
    fn a_phase_change_outside_a_chain_is_inert() {
        let _ = take_window();
        enter(Phase::Sampled);
        std::thread::sleep(std::time::Duration::from_millis(2));
        enter(Phase::Store);
        assert!(
            take_window().is_none(),
            "no chain ran, so there is no window"
        );
    }

    /// An early return charges its remainder to the open phase rather than
    /// losing it, which is what makes the sum identity hold on decline paths.
    #[test]
    fn an_early_return_lands_its_remainder_on_the_open_phase() {
        let _ = take_window();
        fn declines() {
            let _t = ChainTimer::start();
            enter(Phase::Sampled);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        declines();
        let w = take_window().expect("the declining chain still counted");
        assert_eq!(w.chains, 1);
        assert!(w.sampled_us >= 1_000, "the open phase took it: {w:?}");
        assert_eq!(w.store_us, 0, "a phase never entered stays at zero");
    }
}
