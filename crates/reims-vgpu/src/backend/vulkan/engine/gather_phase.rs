//! Where a compute-gather dispatch's CPU cost goes, split by the mechanism that
//! would remove it.
//!
//! `draw_phase`'s `record_us` is where all of it lands, and one bar cannot
//! choose between four fixes. Ten interleaved driven macos-13 boots put the
//! whole of the gather's remaining regression there and nowhere else — the
//! matched pair `on3` / `off5`, ~27 000 draws each:
//!
//! ```text
//!                on3       off5
//! slot_us      47 468   111 275     -57 %   the GPU saving, which is real
//! record_us    79 682    48 055     +66 %   what pays for it
//! descriptors_us 8 119     6 333     +28 %   the draw's own, not the gather's
//! stage_us     32 198    28 902     +11 %
//! ```
//!
//! +31.6 ms a second over ~36 700 dispatches is **0.86 µs each**, and that is
//! the number that keeps [`crate::env::COMPUTE_GATHER`] switched off. A
//! command-buffer run-table arena and a recycled descriptor set already took it
//! down from ~1.05 µs; guessing which of what is left is the next ~0.8 is how a
//! session spends a day on `vkCmdBindPipeline` and finds it was never the cost.
//!
//! So the four candidates are timed apart:
//!
//! | part | what it is | what would remove it |
//! |---|---|---|
//! | `plan` | the `ScatterRun` vector and [`super::guest_scatter::build_gather_run_tables`] | building the table in place, from the copy regions, with no intermediate allocation |
//! | `stage` | the shared run-table arena — one `acquire_staging` and one `write_staging` per draw | nothing; it is already amortised over the draw's dispatches |
//! | `dset` | `alloc_scatter_descriptor_set` (a free-list pop) and `vkUpdateDescriptorSets` | a destination arena, which makes all three bindings constant so a draw needs one set instead of one per window |
//! | `record` | `vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdPushConstants`, `vkCmdDispatch` | hoisting the pipeline bind out of the loop, and the same destination arena, which merges a draw's dispatches into one |
//!
//! Read them against the dispatch count and not against the draw count: a draw
//! gathers ~1.4 windows, so a per-draw reading understates each part by that
//! factor and a reader comparing one to `record_us` per draw would conclude the
//! parts do not sum.
//!
//! # What it said, and why it was worth building
//!
//! Two driven macos-13 sustained-animation boots, ~21 000 dispatches a census
//! second each, agreeing to a hundredth of a microsecond:
//!
//! ```text
//!            us/dispatch    share
//! record        0.376        39 %
//! plan          0.350        37 %
//! stage         0.150        16 %
//! dset          0.082         9 %
//!               0.959
//! ```
//!
//! **The descriptor set is already nearly free, and the fix this crate had
//! written down was aimed at it.** `env::COMPUTE_GATHER` named the destination
//! arena — the change that makes all three bindings constant so a draw needs one
//! set instead of 1.4 — as "the candidate that survives the arithmetic without a
//! reading". It attacks the 9 % column. Recycling the sets, which cost four
//! lines, had already taken `vkAllocateDescriptorSets` and its matching free out
//! of the steady state, and what is left of `dset` is one
//! `vkUpdateDescriptorSets` of three buffers.
//!
//! The two that matter are:
//!
//! * **`plan`, and it is entirely this crate's own code.** No driver call is in
//!   it: a `Vec<ScatterRun>` built from the copy regions, a second `Vec` inside
//!   [`super::guest_scatter::build_gather_run_tables`] holding those runs with
//!   their two indices exchanged, and the table's own `Vec<u32>` — three heap
//!   allocations and three passes over ~13 runs, to produce ~200 bytes.
//! * **`record`, which is four driver calls** — bind pipeline, bind set, push,
//!   dispatch. The pipeline bind is the same handle every time and only the
//!   first of a draw's dispatches needs it. Beyond that this column falls only
//!   with the dispatch *count*, which is what the destination arena would buy
//!   after all — 1.4 down to 1.0, so ~30 % of it, and not the 9 % column it was
//!   proposed for.
//!
//! `stage` at 0.150 is the shared run-table arena, already amortised over a
//! draw's 1.4 dispatches, and there is nothing left in it to remove.
//!
//! # And then the two repairs it pointed at moved `plan` and not the total
//!
//! Hoisting `vkCmdBindPipeline` out of the dispatch loop and building the run
//! table at its exact capacity, measured the same way on two more driven boots,
//! all four in the collapsed regime (draws/frame 256.8-257.5) so they are one
//! population:
//!
//! ```text
//!             plan   stage    dset  record   total   dispatches/s
//! before     0.350   0.150   0.082   0.376   0.959      21 949
//! before     0.333   0.145   0.082   0.368   0.928      20 120
//! after      0.223   0.128   0.140   0.404   0.895      24 606
//! after      0.225   0.142   0.146   0.419   0.932      23 085
//! ```
//!
//! **`plan` fell 34 % with the arms disjoint, and the total did not move.** The
//! ~0.12 µs the reallocations were costing reappeared in `dset` and `record` —
//! `dset` by 74 %, in a path neither change touched at all. That is the signal
//! this split cannot resolve: at 80-150 ns a part, two `Instant::now()` calls
//! are a large share of what is being timed, and an untouched column moving 74 %
//! between two runs is the instrument's own floor rather than a mechanism.
//!
//! Do not read the `after` rows as a regression, and do not read the `plan`
//! column as a win in frames. What they jointly say is that **the dispatch's
//! cost is not reachable by shaving this crate's own work**: what is left is
//! four driver calls, at a release build with `lto = "fat"`, and the only lever
//! on four driver calls is issuing fewer of them.
//!
//! # Which is why the next change is not in this file
//!
//! Fewer dispatches has a floor of one per draw — the destination arena, ~35 %
//! of `record`. It does **not** have a ceiling of none: the content cache that
//! would have removed the dispatch, the run table, the descriptor set and the
//! GPU copy together is closed, because
//! [`crate::runtime::buffer_gather_freshness`]'s audit found only ~27 % of this
//! rail's repeats unchanged. Three quarters of it is the guest genuinely
//! changing its vertex and constant data, and moving those bytes more cheaply is
//! what this rail is for.
//!
//! So the destination arena is the remaining change here, and it is worth ~35 %
//! of one of four columns. Weigh that against the writeback rail, whose own
//! ablation is worth ~30 Hz, before spending a day on it.
//!
//! # This measures the planning, not the copy
//!
//! Every part here is CPU time spent *arranging* a copy the GPU makes later, in
//! the draw's own command buffer. None of it moves a byte, which is the whole
//! point of the rail — [`super::stage_phase`]'s `Gather` part states the same
//! caveat one layer up. A reading of zero here on a boot with a non-zero
//! `buffer_gather_dispatches` would mean the timer is not on the path, never
//! that the path is free.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The steps of planning and recording one draw's gather dispatches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    /// Turning copy regions into run tables. Per dispatch.
    Plan = 0,
    /// The shared run-table staging arena. Per draw, not per dispatch.
    Stage = 1,
    /// Taking a descriptor set and writing its three bindings. Per dispatch.
    Dset = 2,
    /// The command-buffer calls themselves. Per dispatch.
    Record = 3,
}

const PARTS: usize = 4;

/// Nanoseconds, per [`crate::observe::phase_clock`]. Tens of thousands of spans
/// a second is exactly the population a microsecond accumulator reports as
/// free.
static NS: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static N: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GatherPhaseWindow {
    pub plan_us: u64,
    pub plan_n: u64,
    pub stage_us: u64,
    pub stage_n: u64,
    pub dset_us: u64,
    pub dset_n: u64,
    pub record_us: u64,
    pub record_n: u64,
}

/// Take and clear the window. `None` when no gather dispatched, so a boot with
/// the rail switched off costs no line — and a line's *presence* is what says
/// which arm a boot ran.
pub fn take_window() -> Option<GatherPhaseWindow> {
    let us =
        |p: Part| crate::observe::phase_clock::to_us(NS[p as usize].swap(0, Ordering::Relaxed));
    let n = |p: Part| N[p as usize].swap(0, Ordering::Relaxed);
    let w = GatherPhaseWindow {
        plan_us: us(Part::Plan),
        plan_n: n(Part::Plan),
        stage_us: us(Part::Stage),
        stage_n: n(Part::Stage),
        dset_us: us(Part::Dset),
        dset_n: n(Part::Dset),
        record_us: us(Part::Record),
        record_n: n(Part::Record),
    };
    (w.plan_n + w.stage_n + w.dset_n + w.record_n > 0).then_some(w)
}

/// Charges one step to one part, from `open` to `Drop`.
pub(crate) struct Span {
    part: Part,
    started: Instant,
}

impl Span {
    pub(crate) fn open(part: Part) -> Self {
        Self {
            part,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let slot = self.part as usize;
        NS[slot].fetch_add(
            crate::observe::phase_clock::charge_ns(self.started.elapsed()),
            Ordering::Relaxed,
        );
        N[slot].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window reports every part it was given and clears itself, so the next
    /// census second starts from zero rather than from the boot's total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        let _ = take_window();
        drop(Span::open(Part::Plan));
        drop(Span::open(Part::Plan));
        drop(Span::open(Part::Dset));
        let w = take_window().expect("three spans were charged");
        assert_eq!(w.plan_n, 2);
        assert_eq!(w.dset_n, 1);
        assert_eq!(w.stage_n, 0);
        assert_eq!(w.record_n, 0);
        assert_eq!(take_window(), None, "the window cleared itself");
    }

    /// A boot that never dispatches a gather publishes no line at all, which is
    /// how the census says which arm of [`crate::env::COMPUTE_GATHER`] ran
    /// without a second counter to disagree with.
    #[test]
    fn no_gather_publishes_no_window() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }
}
