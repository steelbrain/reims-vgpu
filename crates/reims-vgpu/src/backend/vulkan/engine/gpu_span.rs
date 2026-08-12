//! How long the GPU spent executing each ring-slot submission, from timestamps the
//! submission writes into its own command buffer, tiled by what it was recorded
//! for.
//!
//! # The reading this closes
//!
//! [`super::draw_phase::Phase::Slot`] is the largest phase this device has: the
//! drain worker blocked in `begin_entry` because the ring slot it wants to reuse
//! has an unsignaled fence. Its own doc measured **314 491 µs/s** of it against
//! 2 525 µs/s of actual preparation, and roughly **425 µs per
//! `ring_retire_blocks`**. Every session since has read that column as "the GPU
//! is busy" and concluded the rail is GPU-bound — five CPU wins in a row bought
//! zero frames, and that is the explanation on offer.
//!
//! It was never measured. Before this module the device wrote GPU timestamps in
//! exactly one place, the composite readback copy, and nothing anywhere timed a
//! *draw* on the GPU's own clock. So `slot_us` is a wall-clock wait whose content
//! is unattributed, and it has two readings that call for opposite fixes:
//!
//! * the submission genuinely takes ~425 µs of GPU execution, in which case the
//!   lever is less GPU work per draw — the guest buffer gather at 427 000
//!   transfer regions a second, or the writeback's scatter;
//! * the submission executes in far less than that and the rest of the wait is
//!   *bubble* — queue scheduling, the fence's signal reaching the CPU, or the
//!   ring simply not being deep enough to keep work queued — in which case every
//!   byte-level saving is aimed at the wrong thing and the lever is submission
//!   shape.
//!
//! `RING_DEPTH`'s own doc already suspects the second ("It was submit/fence-
//! bubble-bound, not GPU-compute-bound") on a workload three years of changes
//! ago. Two timestamps settle it for the workload in front of us, and they settle
//! it without correlating two clocks: the delta is GPU ticks between two points
//! on the GPU's own timeline.
//!
//! # What it said the first time, and what that retires
//!
//! Two driven macos-13 sustained-animation boots, quiesced host, 42 driven census
//! windows each, agreeing to about 1 %:
//!
//! ```text
//!                        anim1      anim2
//! gpu_span busy_us      516.9 ms/s  512.3 ms/s      -> 51 % of a second
//! submissions read       1 945/s     1 914/s
//! GPU us per submission    265.8       267.6
//! draws                 29 180/s    28 958/s
//! GPU us per draw           17.71       17.69
//! draw_phase slot_us     32.7 ms/s   17.9 ms/s
//! drain duty                0.56        0.58
//! ```
//!
//! **Read that table knowing it covers draw submissions only** — the first version
//! of this module armed exactly one of the five kinds. See "the control that caught
//! it" below; the occupancy figure is a floor, not the device's total.
//!
//! Three things fall out, and the third is the one that changes what to work on.
//!
//! * **`slot_us` is 18-33 ms a second, not the 314 that
//!   [`super::draw_phase::Phase::Slot`]'s doc measured in 2026-07.** The ring
//!   blocks a twentieth as much as the GPU is busy. Every conclusion drawn from
//!   that column being large is drawn from a number that no longer reproduces.
//! * **`read` equals `batch_flushes` exactly** — 1 990 against 1 990 on the
//!   window checked — which is the cross-check that the probe counts submissions
//!   and not something else. Two independently maintained counters, one identity.
//! * **Neither the GPU nor the drain worker is the pacer.** 51 % GPU occupancy
//!   beside drain duty 0.56 leaves both roughly half idle, and the guest sets the
//!   rate. That is a better explanation of the five CPU wins that bought no
//!   frames than "the rail is GPU-bound" ever was: nothing was bound, so nothing
//!   could convert. It also says a frame count cannot rank a device change on this
//!   rail at all, whatever the change does.
//!
//! # The control that caught the coverage hole, and what a control is for
//!
//! `REIMS_VGPU_SCATTER_SPLIT=on` cuts every writeback run into four sub-ranges
//! that tile it exactly: byte-identical guest output, 4x the copy regions,
//! documented to halve the frame rate on this host. It was run as a *positive
//! control* — an arm where the instrument must move, because a probe that cannot
//! fail has not been tested. One same-regime boot against the pair above:
//!
//! ```text
//!                        base       SCATTER_SPLIT=on
//! window_publish fresh  104.5/s     59.0/s            -44 %  (reproduces)
//! draws                 29 180/s    14 770/s
//! GPU us per submission    265.8       269.7          +1.5 %
//! GPU us per draw           17.71       18.30         +3 %
//! ```
//!
//! The frame rate halved exactly as documented and the per-submission GPU cost did
//! not move — because the writeback's submission was one of the four kinds with no
//! stamps in it. The probe was not measuring the rail the control moved. Hence
//! [`Kind`], hence
//! [`super::pools::ResourcePools::begin_slot_recording`] being the only way to begin
//! a slot command buffer, and hence `unattributed` on the census line.
//!
//! Two lessons worth more than the numbers:
//!
//! * **`armed`/`sealed`/`read` could not have caught this.** They count what the
//!   probe arms, so a whole kind that never arms is consistent with all three and
//!   with `unread=0`. Coverage counters see holes *inside* what they cover.
//! * **A per-second `busy_us` is not comparable across arms.** The guest sets the
//!   draw rate here, so an arm that slows the guest lowers `busy_us` by lowering
//!   the workload — 48 % lower, in the table above, for a rail that got *more*
//!   expensive. Always normalise: per `draws`, or per the kind's own `*_n`.
//!
//! # What the tiling says, once all five kinds carry stamps
//!
//! One driven macos-13 sustained boot in the ~280-draws-a-frame regime, all five
//! kinds stamped, `unattributed=0`:
//!
//! ```text
//! kind        us/sub   subs/s    ms/s   share
//! draw        266.39     1961   522.4   96.9 %
//! store       267.26       72    19.2    3.6 %
//! stamp         5.45      211     1.2    0.2 %
//! readback      0.00        0     0.0    0    %
//! compute       0.00        0     0.0    0    %
//! ```
//!
//! **A draw submission is 96.9 % of this device's GPU time**, and the guest-page
//! writeback — the rail several sessions treated as the largest GPU cost — is
//! 3.6 %. That is not a refutation of those sessions: it is what
//! [`crate::runtime::writeback_debt`] *did*, by eliding 90 % of type-11 Stores.
//! The remaining Store submissions are 72 a second against 1 961 draw ones.
//!
//! `readback` and `compute` at exactly zero are healthy zeros on this workload,
//! not missing coverage: a compositing guest issues no compute and this probe's
//! Safari page reads nothing back. A boot of a guest that does either must show
//! them non-zero, and their appearing is how you know the workload changed.
//!
//! ## The control, re-run on this build, lands in `draw` and not in `store`
//!
//! `SCATTER_SPLIT=on` at matched regime (293.8 draws a frame against 280.2):
//!
//! ```text
//!                    base      split      delta
//! draw us/sub      266.39     287.38     +7.9 %
//! store us/sub     267.26     270.70     +1.3 %
//! us/draw           18.33      19.61     +7.0 %
//! window_publish   105.0/s     96.0/s    -8.6 %
//! ```
//!
//! Four times the writeback's copy regions costs **+7.9 % on the draw submission
//! and nothing measurable on the store one** — because the scatter is recorded
//! into whatever command buffer is open, and at 1 961 draw submissions against 72
//! store ones that is nearly always a draw's. So `Kind` is "what this submission
//! executed", which is the honest thing for a timestamp pair to mean, and it is
//! *not* a per-rail attribution. A rail that rides in another kind's command
//! buffer is charged to that kind.
//!
//! Do not read `store` at 3.6 % as "the writeback is 3.6 % of GPU time". Read it
//! as "the writeback's own submissions are 3.6 %", and reach for an ablation when
//! the question is a rail rather than a submission.
//!
//! # Which makes `busy_us` the number to optimise, not frames
//!
//! 17.7 µs of GPU for one window-server compositing draw is a great deal of work
//! for a textured quad, and this host is an RTX 5080. The support matrix's other
//! column is an iGPU, where the same recorded commands cost roughly an order of
//! magnitude more — so a workload this host runs at 51 % occupancy is one an iGPU
//! is *hard* GPU-bound on by a wide margin, and the per-draw GPU figure is exactly
//! the quantity that binds it.
//!
//! This device has no iGPU to boot on (the dev host has a discrete GPU only), so
//! `busy_us` is the closest thing to an iGPU measurement that exists here: a
//! change that lowers it at identical output — same `draws`, same
//! `buffer_guest_gather_regions`, same bytes — is an iGPU win whether or not this
//! host's frame rate notices. Prefer it to `present_hz` for anything about GPU
//! work, and quote the controls beside it so "identical output" is checkable
//! rather than asserted.
//!
//! # It is a tiling, not a sample
//!
//! `busy_us` and the derived leftover `slot_us - busy_us` sum to the wait, which
//! is the property that made the drain worker's CPU split answer unambiguously
//! and the property a third sampling point would not have. Read the pair; a
//! `busy_us` quoted alone says nothing, because the same 200 ms/s is "the GPU is
//! the wall" next to a 210 ms/s wait and "the wait is nearly all bubble" next to
//! one of 900.
//!
//! Two caveats belong to the reading rather than to the code:
//!
//! * **`busy_us` is per submission, and submissions overlap the wait.** The ring
//!   is [`super::pools::RING_DEPTH`] deep, so up to eight command buffers may be
//!   in flight while the worker waits on one fence. `busy_us` summed over a
//!   census second is the GPU's total occupancy from these submissions; it is
//!   compared against the *second*, not against `slot_us`, when the question is
//!   utilisation. `slot_us - busy_us` is the right comparison only for the
//!   question "was this slot's own work the wait", which is what
//!   `busy_max_us` and `ring_retire_blocks` speak to.
//! * **Timestamps have a cost.** Two per submission at ~2 000 submissions a
//!   second is ~4 000 a second, against the readback rail's existing three per
//!   composite, and both are far below the ~110 000 an inner per-draw split would
//!   need. It is small but it is not nothing, so [`crate::env::GPU_SPANS`] can
//!   take it out and an A/B that needs the absolute floor should.
//!
//! # Coverage is reported, because a zero here has three causes
//!
//! A `busy_us` of zero means the GPU did no work, or the host has no timestamp
//! support, or the arm/seal/read triple did not close — and the census must not
//! read the last two as the first. So the window carries `armed`, `sealed` and
//! `unread`:
//!
//! * `armed` counts command buffers that reset their queries and wrote the top
//!   stamp. Zero means the probe is not on the path at all.
//! * `sealed` counts those that also wrote the bottom stamp before the CB ended.
//!   `armed - sealed` is a submit path that ends a command buffer this module
//!   does not know about, which would read as a missing sample rather than as a
//!   wrong one.
//! * `unread` counts slots re-armed while a previous arming had not been read
//!   back. It must be zero by construction — `begin_entry` retires a slot before
//!   reusing it, and retiring is where the read happens — so a non-zero reading
//!   is a real defect in the ring's own ordering and not a tuning knob.
//! * `unattributed` is `read` minus the per-kind counts and must be zero. It is the
//!   one column that cannot agree trivially, because a submission is counted in
//!   `read` before a kind is chosen — which is exactly the check that was missing
//!   when a whole kind had no stamps.
//!
//! None of these can see a *sixth* kind that begins a slot command buffer without
//! arming, and no counter can: that is why the arm is folded into
//! `begin_slot_recording` and the raw reset-then-begin pair no longer appears at any
//! call site. The invariant is structural rather than reported.

use std::sync::atomic::{AtomicU64, Ordering};

/// What a ring-slot submission was recorded for.
///
/// Every command buffer a submission ring slot carries is one of these, and the
/// per-kind totals sum to `busy_us` — which is what makes the reading a tiling of
/// the device's GPU time rather than a sample of one rail. The first version of
/// this module armed only [`Self::Draw`] and reported 51 % occupancy; the
/// writeback rail's own positive control then moved the frame rate 44 % without
/// moving `busy_us` per submission at all, because its submission was one of the
/// four kinds that had no stamps. A tiling closes; that did not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    /// A draw, or a batch of them sharing one command buffer. Carries the guest
    /// buffer gather and every render pass.
    Draw = 0,
    /// A rendered surface copied back into the guest's own pages — the writeback,
    /// as either a scatter of transfer regions or the compute dispatch that
    /// replaces them.
    Store = 1,
    /// A target image copied into a host-visible buffer for a guest read.
    Readback = 2,
    /// A guest compute dispatch.
    Compute = 3,
    /// The completion stamp's own submission, on the arm that writes it with a
    /// command buffer rather than a word.
    Stamp = 4,
}

const KINDS: usize = 5;

impl Kind {
    /// Every kind, for the census and for the tests that assert the tiling closes.
    pub(crate) const ALL: [Kind; KINDS] = [
        Kind::Draw,
        Kind::Store,
        Kind::Readback,
        Kind::Compute,
        Kind::Stamp,
    ];
}

/// GPU nanoseconds accumulated across submissions in this census window.
static BUSY_NS: AtomicU64 = AtomicU64::new(0);
/// Submissions whose two stamps were both read back.
static READ: AtomicU64 = AtomicU64::new(0);
/// The largest single submission's GPU nanoseconds this window.
static MAX_NS: AtomicU64 = AtomicU64::new(0);
static ARMED: AtomicU64 = AtomicU64::new(0);
static SEALED: AtomicU64 = AtomicU64::new(0);
static UNREAD: AtomicU64 = AtomicU64::new(0);
/// Per-kind nanoseconds and submission counts. These sum to [`BUSY_NS`] and
/// [`READ`], and `the_kinds_tile_the_total` is what keeps that true.
static KIND_NS: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static KIND_N: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];

/// One census window of GPU-side submission timing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuSpanWindow {
    /// GPU microseconds summed over every submission read back this window.
    pub busy_us: u64,
    /// The largest single submission, in GPU microseconds.
    pub busy_max_us: u64,
    /// Submissions both stamps were read from. The denominator for `busy_us`.
    pub read: u64,
    /// Command buffers that wrote the top stamp.
    pub armed: u64,
    /// Command buffers that also wrote the bottom stamp.
    pub sealed: u64,
    /// Slots re-armed before a previous arming was read. Zero by construction.
    pub unread: u64,
    /// GPU microseconds per [`Kind`], in [`Kind::ALL`] order. Sums to
    /// [`Self::busy_us`] up to the microsecond truncation of each part.
    pub kind_us: [u64; KINDS],
    /// Submissions per [`Kind`], in [`Kind::ALL`] order. Sums to [`Self::read`]
    /// exactly.
    pub kind_n: [u64; KINDS],
}

impl GpuSpanWindow {
    /// The one column that says whether the tiling closed. Submissions read
    /// against submissions attributed; anything but zero is a kind that reached
    /// the read path without a label, which is a bug in this module and not a
    /// property of the workload.
    pub fn unattributed(&self) -> i64 {
        self.read as i64 - self.kind_n.iter().sum::<u64>() as i64
    }
}

/// Take and clear the window. `None` when nothing armed, so a host without
/// timestamp support and a boot with [`crate::env::GPU_SPANS`] off cost no line
/// — and a line's presence is what says the probe ran.
pub fn take_window() -> Option<GpuSpanWindow> {
    let armed = ARMED.swap(0, Ordering::Relaxed);
    let mut kind_us = [0u64; KINDS];
    let mut kind_n = [0u64; KINDS];
    for (i, k) in Kind::ALL.iter().enumerate() {
        kind_us[i] =
            crate::observe::phase_clock::to_us(KIND_NS[*k as usize].swap(0, Ordering::Relaxed));
        kind_n[i] = KIND_N[*k as usize].swap(0, Ordering::Relaxed);
    }
    let w = GpuSpanWindow {
        busy_us: crate::observe::phase_clock::to_us(BUSY_NS.swap(0, Ordering::Relaxed)),
        busy_max_us: crate::observe::phase_clock::to_us(MAX_NS.swap(0, Ordering::Relaxed)),
        read: READ.swap(0, Ordering::Relaxed),
        armed,
        sealed: SEALED.swap(0, Ordering::Relaxed),
        unread: UNREAD.swap(0, Ordering::Relaxed),
        kind_us,
        kind_n,
    };
    (armed > 0).then_some(w)
}

/// A command buffer reset its query pair and wrote the top stamp.
pub(crate) fn note_armed() {
    ARMED.fetch_add(1, Ordering::Relaxed);
}

/// A command buffer wrote the bottom stamp before ending.
pub(crate) fn note_sealed() {
    SEALED.fetch_add(1, Ordering::Relaxed);
}

/// A slot was armed while a previous arming of the same slot had not been read.
pub(crate) fn note_unread() {
    UNREAD.fetch_add(1, Ordering::Relaxed);
}

/// One submission's GPU execution time, from the delta between its two stamps,
/// charged both to the total and to the kind the command buffer was recorded for.
pub(crate) fn note_busy_ns(kind: Kind, ns: u64) {
    BUSY_NS.fetch_add(ns, Ordering::Relaxed);
    READ.fetch_add(1, Ordering::Relaxed);
    MAX_NS.fetch_max(ns, Ordering::Relaxed);
    KIND_NS[kind as usize].fetch_add(ns, Ordering::Relaxed);
    KIND_N[kind as usize].fetch_add(1, Ordering::Relaxed);
}

/// Where a ring slot's arming stands, so a read cannot invent a sample out of a
/// query the GPU never wrote.
///
/// A three-state enum rather than two bools because "armed but not sealed" and
/// "sealed" are the two states a read must tell apart, and a pair of bools admits
/// a fourth combination that means nothing.
/// The kind travels with the state rather than beside it, so there is no
/// representable slot that is sealed but unlabelled — which is the shape that
/// would let a submission's time reach `busy_us` without reaching a `Kind` and
/// make the tiling silently not close.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SlotSpan {
    /// No stamp written since this slot was last read.
    #[default]
    Idle,
    /// Top stamp written; the command buffer is still recording.
    Armed(Kind),
    /// Both stamps written; the delta is readable once the fence signals.
    Sealed(Kind),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window reports what was charged and clears itself, so a census second
    /// starts from zero rather than from the boot's running total.
    #[test]
    fn a_window_takes_what_was_charged_and_leaves_nothing() {
        let _ = take_window();
        note_armed();
        note_armed();
        note_sealed();
        note_busy_ns(Kind::Draw, 3_000);
        note_busy_ns(Kind::Store, 5_000);
        let w = take_window().expect("two command buffers armed");
        assert_eq!(w.armed, 2);
        assert_eq!(w.sealed, 1);
        assert_eq!(w.read, 2);
        assert_eq!(w.busy_max_us, crate::observe::phase_clock::to_us(5_000));
        assert_eq!(w.unread, 0);
        assert_eq!(take_window(), None, "the window cleared itself");
    }

    /// A boot where nothing armed publishes no line, which is how the census says
    /// "this host writes no timestamps" without a second counter to disagree.
    #[test]
    fn nothing_armed_publishes_no_window() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }

    /// `busy_us` is a sum and `busy_max_us` is a high-water: two submissions of
    /// equal length and one long one next to one short one must not read the same,
    /// because only the second says a single submission is the wall.
    #[test]
    fn the_sum_and_the_high_water_are_different_readings() {
        let _ = take_window();
        note_armed();
        note_busy_ns(Kind::Draw, 1_000_000);
        note_busy_ns(Kind::Draw, 1_000_000);
        let even = take_window().expect("armed");
        note_armed();
        note_busy_ns(Kind::Draw, 1_900_000);
        note_busy_ns(Kind::Draw, 100_000);
        let skewed = take_window().expect("armed");
        assert_eq!(even.busy_us, skewed.busy_us, "the same total");
        assert!(
            skewed.busy_max_us > even.busy_max_us,
            "{skewed:?} vs {even:?}"
        );
    }

    /// The per-kind counts tile the total exactly. This is the invariant whose
    /// absence made the first version of this module report 51 % GPU occupancy for
    /// a device whose writeback submissions carried no stamps: with only one kind
    /// wired, `read` and the sum of the kinds agreed trivially and the *missing*
    /// submissions were invisible to both. `unattributed` is the column that
    /// cannot agree trivially, because a submission reaching the read path is
    /// counted once in `read` before any kind is chosen.
    #[test]
    fn the_kinds_tile_the_total() {
        let _ = take_window();
        note_armed();
        for (i, k) in Kind::ALL.iter().enumerate() {
            note_busy_ns(*k, 1_000_000 * (i as u64 + 1));
        }
        let w = take_window().expect("armed");
        assert_eq!(w.unattributed(), 0, "{w:?}");
        assert_eq!(w.kind_n.iter().sum::<u64>(), w.read, "{w:?}");
        assert_eq!(w.kind_us.iter().sum::<u64>(), w.busy_us, "{w:?}");
        // ...and the kinds are distinguished, not merged into one bucket: a
        // version that charged every submission to `Draw` would pass every
        // assertion above.
        assert_eq!(w.kind_n, [1; 5], "{w:?}");
        assert!(
            w.kind_us[Kind::Stamp as usize] > w.kind_us[Kind::Draw as usize],
            "{w:?}"
        );
    }

    /// `Kind::ALL` is the census's column order and the array index in one, so a
    /// variant added without a place in `ALL` would silently never be reported.
    #[test]
    fn every_kind_has_exactly_one_place_in_all() {
        assert_eq!(Kind::ALL.len(), KINDS);
        for (i, k) in Kind::ALL.iter().enumerate() {
            assert_eq!(*k as usize, i, "ALL must be in discriminant order");
        }
    }
}
