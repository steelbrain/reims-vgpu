//! What the drain measured, and the vocabulary it measured it in.
//!
//! Same membership rule as `runtime::exec::report`: **nothing here decides
//! anything.** Every
//! item is either a reading (`DoorbellCensus`, `VcpuLockCensus`,
//! `DrainDutyCensus`), the phase vocabulary a reading is filed under
//! (`ReadbackPhase`, `SurfaceWritePhase`, `WindowPublish`, `FlushRail`), or the
//! `note_*` / `emit_*` entry point that files it.
//!
//! It is named `census` rather than `report` because that is what the rest of
//! the crate already calls this surface — `note_store_route` and
//! `take_store_routes` live here and are read from a dozen modules — and
//! because a large part of it is per-second windows rather than one-shot lines.
//!
//! The whole of it is re-exported from `super`, so every existing
//! `crate::runtime::drain::note_*` path still resolves; this module is where
//! they are written, not a new place to reach them from.

use super::DISPLAY_VBL_MIN_INTERVAL_US;
/// Delivered-VBL rate, reported from the branch that decides it.
///
/// VBL is what paces the guest's compositor: WindowServer produces a frame off
/// its display-link callback, so whatever rate we deliver here is a ceiling on
/// guest frame rate no matter how fast the present path runs. Nothing measured
/// it. A driven boot emitted **zero** lines matching `vbl` anywhere in the
/// always-on channel, so "are we starving the display link" could not be
/// answered from a log, only guessed at from the constants.
///
/// The arms are counted separately because a single "delivered" tally cannot
/// tell the silences apart, and they have opposite meanings: `not_online` is
/// the display never having come up (no VBL is owed at all), `not_claimed` is
/// the limiter doing its job at the advertised rate, and `not_enabled` is the
/// guest having declined this class in the shared page's enable mask. Reading a
/// low delivered count without them would license all three conclusions.
///
/// `not_enabled` is the arm whose absence cost a reader most, and it was
/// absent: both x86 rails measured here run with VBL disabled in that mask, and
/// this line reported `delivered=13312 ... window_hz=120.0` for a guest that
/// had asked for no VBL at all. That reads as "the compositor is being paced at
/// the grid rate" when nothing is owed and nothing is being consumed.
///
/// One line per 1024 deliveries — about 8 s at the grid rate, and it costs one
/// relaxed increment per poll otherwise.
/// Which way the VBL path went. Indices into [`VblCensus`].
pub(crate) const VBL_NOT_ONLINE: usize = 0;
pub(crate) const VBL_NOT_CLAIMED: usize = 1;
pub(crate) const VBL_DELIVERED: usize = 2;
pub(crate) const VBL_NOT_ENABLED: usize = 3;

/// One report per this many deliveries — about 8 s at the grid rate.
const VBL_REPORT_EVERY: u64 = 1024;

/// One report per this many deliveries, for an arm's first
/// [`VBL_EARLY_UNTIL`] — about half a second at the grid rate.
///
/// # The window this exists to make visible
///
/// A macOS 13 guest latches its display link at either ~60 Hz or ~120 Hz within
/// the first seconds of the display coming up, and then holds it for the life of
/// the boot. Which one it picks varies boot to boot on a byte-identical device,
/// and it is worth a **factor of two** in presented frames, draws and every
/// per-second reading taken off them — far more than any rail this device has
/// been tuned on.
///
/// At [`VBL_REPORT_EVERY`] the first line of a boot lands after about eleven
/// seconds, which is long after the guest has decided. So the cadence during the
/// window that decides it was simply not observable, and no amount of reading
/// later lines could recover it. Sixteen finer lines at the head of each arm cost
/// nothing and cover the first ~9 s.
///
/// # What it found, and the false positive it found first
///
/// Boots on this rail are sharply bimodal: over 40 interleaved driven macos-13
/// boots, 28 presented 94.8-117.0 frames a second and 12 presented 59.8-60.5,
/// with nothing in between. Roughly three in ten lose half their frame rate.
///
/// The first reading off this instrument was that the **first sustained
/// `arm=delivered` window** — the guest's first stretch of holding VBL armed,
/// landing near `delivered=128` — predicted which population a boot fell into,
/// 8 times out of 8. **It does not.** That run contained exactly one slow boot,
/// so every feature of that one boot "predicted" the outcome perfectly. Twenty
/// instrumented boots later:
///
/// ```text
/// boots   first sustained window   latched
///  B8,B9,B12,B16   119.6 - 120.3     slow
///  B7                    44.1        fast
///  the other 15     119.4 - 120.5    fast
/// ```
///
/// Four slow boots were served a clean ~120 Hz across that window and latched 60
/// anyway, and one boot served 44 Hz latched fast. The window is ~120 Hz in 18
/// of 20 boots whatever happens afterwards, so it carries no signal at all.
///
/// **Treat a single-slow-boot sample as no sample.** The population that matters
/// here shows up in 30 % of boots, so a run of eight contains one or two of them
/// and cannot separate a cause from a coincidence. Forty boots is the order of
/// sample size this question needs, and the same trap applies to any other
/// feature someone thinks distinguishes the two.
///
/// What every boot does share is the shape: two windows at ~120 Hz, a dip, a
/// quiet stretch of ten seconds or more while the desktop comes up, then a ramp.
/// The populations diverge only in where that ramp settles — fast boots at
/// 110-120 Hz delivered, slow boots at 75-90 — and that is the outcome rather
/// than a cause, because a 60 Hz compositor arms VBL less often by construction.
/// A slow boot is not being starved of VBL when it settles: it receives ~85 a
/// second and presents 60.
///
/// # It is not a frame-time cliff either
///
/// A guest holding a 120 Hz link that presents either ~120 or exactly ~60 is the
/// signature of vsync-locked halving: miss the 8.33 ms budget and you fall to
/// every other vblank. If that were it, this device would be sitting on the edge
/// of the threshold, and every microsecond saved anywhere would buy a
/// *probability* of doubling the frame rate — which would be the most important
/// fact in this crate.
///
/// It is not it. Twenty-four interleaved boots, with the device deliberately
/// pushed the wrong way by `REIMS_VGPU_COMPUTE_GATHER=off` (about 20 % more GPU
/// work per draw, and the arm's positive control confirmed
/// `buffer_gather_dispatches=0`):
///
/// ```text
/// arm                       fast   slow
/// ~20 % slower device         10      2
/// shipping default             5      7
/// ```
///
/// Making the device slower did not push boots over a cliff; the slower arm
/// latched fast *more* often. Whatever direction that effect is (p≈0.09, and one
/// interleaved run of twelve an arm is not enough to claim it), it rules out the
/// cliff, which predicted the opposite and predicted it strongly.
///
/// # Nor is it the host GPU's clock
///
/// The direction above pointed at the governor: this part idles at 180 MHz
/// against a 3090 MHz cap, so "more GPU work" plausibly means "higher clock"
/// means "lower latency for everything the guest waits on". Sixteen boots with
/// `nvidia-smi` sampled at 2 Hz throughout, split at the 25 s mark so the window
/// *before* the guest has latched is read separately from the driven one:
///
/// ```text
///        early (0-25 s)          driven (25 s-end)
///        med MHz   p90   util    med MHz   util
/// fast     1040   1590    24 %     1547    31 %
/// slow     1044   1594    26 %     1266    24 %
/// ```
///
/// The early window is **identical**. The driven window differs, and that is the
/// causality running backwards rather than a cause: a guest presenting 60 frames
/// a second asks for half the work and so clocks lower by construction. Reading
/// the driven column alone would have produced a confident wrong answer, which
/// is why the run split the windows before scoring.
///
/// # The cause: the compositor is paced by a constant the guest cannot correct
///
/// It is not the VBL rate at all, which is why six device-side hypotheses in a
/// row came back null. The contract, and then the live confirmation:
///
/// 1. The guest's compositor schedules each frame on a timer at
///    `lastVBL + period`. Nothing in it measures VBL arrivals, divides a rate or
///    counts missed frames; `period` is taken verbatim from the kernel.
/// 2. The kernel display-pipe layer ships `(lastVBL, lastVBL + fRefreshPeriod)`
///    on every VBL notification, so `period` **is** `fRefreshPeriod`.
/// 3. `fRefreshPeriod` is written in one place, on display-mode change. It is
///    initialised to a synthesised **1/60 s = 16 666 666 ns** and only then
///    replaced by the true value, `IOFBCurrentPixelCount * 1e9 /
///    IOFBCurrentPixelClock`. Five early returns leave the 60 Hz default standing.
/// 4. Those two `IOFramebuffer` properties are published only when the mode's
///    detailed timing is marked valid, and the paravirtual framebuffer driver
///    clears that valid bit and fills the timing with nothing — so the
///    framebuffer layer *removes* both properties.
///
/// Confirmed live rather than argued: over eleven driven macos-13 boots `ioreg`
/// finds **neither property on any boot** — not on the 59.5-60.1 Hz ones and not
/// on the 97.5-114.3 Hz ones. The numbers are not even hard to come by; the same
/// dump carries the detailed timing the guest built out of our table (pixel clock
/// 15 848 840 000 Hz, 1920 + 10 000 by 1080 + 10 000), which works out to
/// 8 333 332 ns — 120.00 Hz to five figures. The guest holds the inputs and the
/// path that would use them is closed.
///
/// # Which of two values a boot latches is the whole split
///
/// `fRefreshPeriod` is never recomputed once set, so a boot ends on one of two:
///
/// - **16 666 666 ns** — the compositor paces on it and produces **exactly 60 Hz**.
/// - **0**, when the first notification is sent before the mode-change path has
///   run. A zero period puts the next wake time in the past, so the compositor
///   **free-runs** and produces whatever it can, work-limited.
///
/// The measured populations are that fingerprint and nothing else: every slow
/// boot on record sits in 59.5-60.5 Hz, a *constant*, and every fast one in
/// 94.8-117.0 Hz, a *spread of twenty-two Hz*. A paced compositor cannot vary
/// that much and a free-running one cannot be that flat.
///
/// So the fast boots are the ones where the guest never learned a period, and the
/// correctly-configured outcome on this pathway is the *slow* one. State that
/// before trying to fix it: forcing a second mode change — the obvious lever, and
/// one this device owns through an offline/online cycle — would set
/// `fRefreshPeriod` on every boot and take the good 70 % down to 60 Hz.
///
/// # "Work-limited" is measured, and it makes the frame rate this device's score
///
/// The free-running branch above is the useful one to optimise against, because
/// *work-limited* there is a measured claim rather than a reading of the code.
/// The twenty-four-boot run whose latch result appears above also moved this
/// device's per-draw cost by a known amount, so scoring its **fast boots only**
/// says what a per-draw saving is worth in frames:
///
/// ```text
/// arm                          n fast   present_hz mean (range)   us/draw mean (range)
/// shipping                          5   113.2  (109.8-116.3)      13.21  (12.66-14.02)
/// 20.6 % more GPU work per draw     9   105.5  (101.4-107.7)      15.07  (13.92-16.16)
/// ```
///
/// The frame rates are **disjoint** — the slowest shipping boot beats the fastest
/// slowed one — while `us/draw` overlaps across the same fourteen boots. Two
/// things follow, and the second is the one that changes what to do:
///
/// - a free-running guest converts device work into frames at roughly **0.35
///   frames per unit of per-draw GPU work**, so a candidate worth under ~5 % per
///   draw is not worth a boot chain here;
/// - `present_hz` over the fast population is a *sharper* instrument than
///   `us/draw`, not a noisier one, and it is the number to rank a change by.
///
/// That also settles the standing puzzle of per-draw wins that "bought no
/// frames": they were measured through a presenter that clamped at ~41 Hz. It
/// does not clamp now, and the frames are there.
///
/// # What this closes
///
/// Not the limiter (~118 Hz on every boot, fast or slow), not the claim ordering
/// (40 interleaved boots, p≈0.7, see [`super::signal_display_refresh_classes`]),
/// not the display mode (see [`super::fill_display_descriptor`]; both populations
/// report 120.00 Hz), not the early delivery window, not a frame-time threshold,
/// and not the host GPU clock. All six were null for one reason: each was about
/// how fast this device delivers VBL, and the guest's frame period is not derived
/// from that.
///
/// One methodological result came out of the same runs and belongs to anyone
/// testing the next theory: the base rate **drifts**. It was 12 slow in 40 early
/// in a session and 7 in 12 twice, hours later, on the same binary. Compare arms
/// only within one interleaved run.
///
/// This is an instrument, not a rail: nothing branches on it.
const VBL_REPORT_EARLY: u64 = 64;

/// How far into an arm's count the finer [`VBL_REPORT_EARLY`] cadence runs.
///
/// Equal to [`VBL_REPORT_EVERY`] so the two cadences meet exactly at the
/// boundary: the last early report and the first ordinary one are the same
/// event, and no window is ever measured across a change of step.
const VBL_EARLY_UNTIL: u64 = VBL_REPORT_EVERY;
const _: () = assert!(VBL_EARLY_UNTIL.is_multiple_of(VBL_REPORT_EARLY));

/// Width of [`VblCensus::arms`], derived from the last arm index so a new arm
/// cannot be added without the array growing with it.
const VBL_ARMS: usize = VBL_NOT_ENABLED + 1;

/// # `window_hz` is per reporting arm, and used not to be
///
/// Two arms report — `delivered` and `not_enabled` — and they shared one
/// `last_report_ms`/`last_report_n` pair. Whenever both were live in the same
/// boot the shared counter made every window wrong: an arm reporting at its own
/// `n = 1024` subtracted the *other* arm's 1024 and printed `window_hz=0.0`, and
/// the next report measured its count against a timestamp the other arm had
/// moved. A boot alternating the two produced a column of zeroes and a column of
/// rates over windows that never happened.
///
/// That is not a cosmetic log defect. `window_hz` is the only per-window reading
/// of the rate the guest's compositor is paced at, and it read as broken exactly
/// on the boots worth reading — a guest that arms VBL one shot at a time is the
/// guest that makes both arms report. The delivered rate had to be recovered by
/// differencing consecutive `delivered=` fields across lines instead.
///
/// So the window is per arm. `since_n` went with the fix rather than being
/// repaired: an arm reports on every multiple of [`VBL_REPORT_EVERY`] of its own
/// count and then stores that count, so the gap is always exactly
/// `VBL_REPORT_EVERY` and computing it was the thing that could be wrong.
#[derive(Default)]
pub(crate) struct VblCensus {
    arms: [std::sync::atomic::AtomicU64; VBL_ARMS],
    last_report_ms: [std::sync::atomic::AtomicU64; VBL_ARMS],
}

impl VblCensus {
    /// Count one traversal and return the line to emit when a report is due.
    ///
    /// Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: the interesting properties are "only the
    /// post-limiter arms report", "the rate is measured over the window and not
    /// the process lifetime", and "the silent arms stay separable", and all
    /// three are assertions about this return value.
    ///
    /// **Two arms report, not one.** `delivered` and `not_enabled` are the two
    /// outcomes of a tick that got past the online check and the limiter, and
    /// exactly one of them can be live on a given boot — the guest either has
    /// the class enabled or it does not. Reporting only on `delivered` is why a
    /// guest that declines VBL produced no `display_vbl` line at all, which
    /// reads identically to a device whose VBL path is not running.
    ///
    /// `hz` is the reporting arm's own rate over the window, so it stays a rate
    /// of the thing that triggered the line; `arm=` names which one, because
    /// 120 Hz of delivery and 120 Hz of declining are the same number and
    /// opposite facts.
    pub(crate) fn note(&self, arm: usize, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.arms[arm].fetch_add(1, Relaxed) + 1;
        let reports = arm == VBL_DELIVERED || arm == VBL_NOT_ENABLED;
        // The head of each arm reports finely, because that is the window in
        // which the guest picks the display-link rate it then keeps; see
        // `VBL_REPORT_EARLY`. The two cadences meet at `VBL_EARLY_UNTIL`, so
        // `step` is also exactly how many events this window covers.
        let step = if n <= VBL_EARLY_UNTIL {
            VBL_REPORT_EARLY
        } else {
            VBL_REPORT_EVERY
        };
        if !reports || !n.is_multiple_of(step) {
            return None;
        }
        let since_ms = now_ms.saturating_sub(self.last_report_ms[arm].swap(now_ms, Relaxed));
        // Window rate, not a lifetime average: the lifetime figure carries the
        // pre-online stretch forever and would read low long after the display
        // came up. The count in the window is `step` by construction — this line
        // exists because this arm's own count just reached a multiple of it — so
        // the only variable is how long that took.
        let hz = if since_ms > 0 {
            (step * 1000) as f64 / since_ms as f64
        } else {
            0.0
        };
        let name = if arm == VBL_DELIVERED {
            "delivered"
        } else {
            "not_enabled"
        };
        Some(format!(
            "display_vbl delivered={} not_claimed={} not_online={} not_enabled={} \
             arm={name} window_hz={hz:.1} grid_hz={:.1}",
            self.arms[VBL_DELIVERED].load(Relaxed),
            self.arms[VBL_NOT_CLAIMED].load(Relaxed),
            self.arms[VBL_NOT_ONLINE].load(Relaxed),
            self.arms[VBL_NOT_ENABLED].load(Relaxed),
            1_000_000.0 / DISPLAY_VBL_MIN_INTERVAL_US as f64,
        ))
    }
}

pub(crate) fn note_vbl(arm: usize, now_ms: u64) {
    static VBL: std::sync::LazyLock<VblCensus> = std::sync::LazyLock::new(VblCensus::default);
    if let Some(line) = VBL.note(arm, now_ms) {
        crate::observe::off(line);
    }
}

/// One interrupt pulse dropped because an undelivered pulse of the same kind
/// was still queued.
///
/// # Why a coalesced pulse is not the same as a delivered one
///
/// The prompt queue in `qemu::host_ops` collapses a second pulse of a kind into
/// the first while the first is still waiting for QEMU's bottom half. For a
/// status the guest reads back that is free — the status bits accumulate, so one
/// interrupt carries both. **For VBL it is not free**, because the guest does not
/// read a count, it reads a clock: it timestamps the vblank it is told about, and
/// the interval it measures between two of them is what its compositor uses as
/// its frame period. A vblank folded into another one is a vblank that never
/// happened as far as the guest is concerned, and the interval it measures is
/// then two grid periods instead of one.
///
/// So this counter and `display_vbl`'s `delivered` count different things and
/// must not be read as one. `delivered` counts what **this device wrote**; the
/// guest receives `delivered` minus the pulses counted here. A boot can report a
/// healthy ~118 Hz delivered rate while the guest is being told about half of it,
/// and every census in this crate would still read clean — which is why the two
/// boot populations looked identical from the device side for as long as they
/// did.
///
/// Counted, not refused: coalescing is still the right behaviour for the IOSFC
/// pulse and for a backlog this device cannot help. The reading is what says
/// whether it is happening on the VBL rail often enough to matter, and it is per
/// kind so the two rails cannot be confused for one another.
pub(crate) fn note_irq_coalesced(kind: crate::runtime::host::HostActionKind) {
    let name = match kind {
        crate::runtime::host::HostActionKind::IrqGfxPulse => "irq_coalesced_gfx",
        _ => "irq_coalesced_iosfc",
    };
    crate::runtime::drain::note_store_route(name);
}

/// Which way the display present/transaction signal went. Indices into the
/// counter set behind [`note_display_present_signal`].
pub(crate) const DISPLAY_PRESENT_NO_GPA: usize = 0;
pub(crate) const DISPLAY_PRESENT_NOT_ENABLED: usize = 1;
pub(crate) const DISPLAY_PRESENT_DELIVERED: usize = 2;
/// Raised by the refresh tick rather than by a present.
///
/// Separate from `DELIVERED` because the two answer different questions and
/// summing them hides both: `delivered` is "a frame finished", `refresh` is "the
/// pipe was told its live frame is done with". A guest that drives its display
/// from this class shows a high `refresh` and a low `delivered`; one that never
/// arms it shows zero of both, and that is not the same reading.
pub(crate) const DISPLAY_PRESENT_REFRESH: usize = 3;

/// Width of the counter set, derived from the last arm so a new arm cannot be
/// added without the array growing with it.
const DISPLAY_PRESENT_ARMS: usize = DISPLAY_PRESENT_REFRESH + 1;

/// One report per this many signals. Presents are far rarer than VBL ticks, so
/// this is a much smaller stride than [`VBL_REPORT_EVERY`] — a rail that
/// presents a handful of times a second should still produce a line.
const DISPLAY_PRESENT_REPORT_EVERY: u64 = 64;

/// Count one traversal of the display present/transaction signal.
///
/// **VBL had a census and this edge had none, and they fail differently.** A
/// starved VBL costs the compositor its pacing, which reads as slowness. A
/// withheld transaction interrupt can cost liveness outright: the guest's
/// queue-idle wait has no deadline, so "how many times did this device raise
/// bit 1, and how many times did it decline to" is the difference between a
/// device that is merely behind and one the guest will wait on forever. Neither
/// question had an answer in any log.
///
/// `not_enabled` is the arm to read on a rail that is not advancing. It is not a
/// fault by itself — a guest that has not armed the class is not owed the
/// event — but paired with `delivered=0` it says the device never had the
/// opportunity, which is a different bug from having missed it.
///
/// **Every arm reports its first traversal, not just its 64th.** A stride alone
/// is the wrong instrument for a rail that is *stuck*: the interesting readings
/// here are single digits, and a wedged guest that took an arm three times would
/// produce no line at all — indistinguishable in the log from a device on which
/// this edge never runs, which is the exact confusion this census exists to
/// remove. The stride bounds a busy rail; the first-sight line bounds a dead one.
pub(crate) fn note_display_present_signal(arm: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    static ARMS: [std::sync::atomic::AtomicU64; DISPLAY_PRESENT_ARMS] =
        [const { std::sync::atomic::AtomicU64::new(0) }; DISPLAY_PRESENT_ARMS];
    let n = ARMS[arm].fetch_add(1, Relaxed) + 1;
    if n != 1 && !n.is_multiple_of(DISPLAY_PRESENT_REPORT_EVERY) {
        return;
    }
    crate::observe::off(format!(
        "display_present_signal delivered={} refresh={} not_enabled={} no_gpa={}",
        ARMS[DISPLAY_PRESENT_DELIVERED].load(Relaxed),
        ARMS[DISPLAY_PRESENT_REFRESH].load(Relaxed),
        ARMS[DISPLAY_PRESENT_NOT_ENABLED].load(Relaxed),
        ARMS[DISPLAY_PRESENT_NO_GPA].load(Relaxed),
    ));
}

/// Sentinel for "no enable word has been read yet".
///
/// The mask is four meaningful bits, so any value with a high bit set is
/// unreachable as a real reading and a first read of `0` still reports.
const DISPLAY_ENABLE_UNREAD: u32 = u32::MAX;

/// Report the guest's display event-enable word, once per distinct value.
///
/// **Which classes a guest arms is a per-generation decision, and it is the
/// first thing worth knowing about a display pipe that is not advancing.** The
/// word lives in guest RAM and is never trapped, so the only way to see it used
/// to be to stop the world and read the page by hand over QMP — which is how it
/// was read, per rail, one sample at a time. A sample is also the wrong shape:
/// the guest arms VBL while compositing and disarms when idle, so a single
/// reading cannot distinguish "never armed" from "not armed just now", and those
/// are the two answers a stalled rail is being triaged between.
///
/// Edge-triggered rather than sampled for that reason: one line per transition
/// costs nothing on a steady guest and yields the arm/disarm history on a busy
/// one. `first_sight` cannot serve here — it keys on the formatted line, so a
/// mask that flips between two values would report each exactly once and then go
/// quiet, losing the very history this exists to show.
pub(crate) fn note_display_enable_mask(mask: u32) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU32 =
        std::sync::atomic::AtomicU32::new(DISPLAY_ENABLE_UNREAD);
    let prev = LAST.swap(mask, Relaxed);
    if prev == mask {
        return;
    }
    // Name every bit the guest's dispatch can claim, and say plainly when it has
    // armed one this device never signals. A reader looking at `0xe` should not
    // have to go and find out what bit 3 is; that question cost a session.
    let names = |m: u32| -> String {
        use crate::model::{
            DISPLAY_EVENT_MASK_ALL, DISPLAY_OFFLINE_EVENT_MASK, DISPLAY_ONLINE_EVENT_MASK,
            DISPLAY_PRESENT_EVENT_MASK, DISPLAY_VBL_EVENT_MASK,
        };
        let mut out = Vec::new();
        for (bit, name) in [
            (DISPLAY_VBL_EVENT_MASK, "vbl"),
            (DISPLAY_PRESENT_EVENT_MASK, "transaction"),
            (DISPLAY_ONLINE_EVENT_MASK, "online"),
            (DISPLAY_OFFLINE_EVENT_MASK, "offline"),
        ] {
            if m & bit != 0 {
                out.push(name);
            }
        }
        // A bit outside the guest's own dispatch would sit in the pending word
        // forever, so an unknown one is worth naming loudly rather than dropping.
        if m & !DISPLAY_EVENT_MASK_ALL != 0 {
            out.push("unknown");
        }
        if out.is_empty() {
            "none".to_string()
        } else {
            out.join("+")
        }
    };
    if prev == DISPLAY_ENABLE_UNREAD {
        crate::observe::off(format!(
            "display_enable_mask first mask=0x{mask:x} armed={}",
            names(mask)
        ));
    } else {
        crate::observe::off(format!(
            "display_enable_mask 0x{prev:x} -> 0x{mask:x} armed={} was={}",
            names(mask),
            names(prev)
        ));
    }
}

/// Report at most this often. One line per second is bounded enough to leave on
/// for the life of the device and dense enough to see a stall move.
const DRAIN_DUTY_REPORT_MS: u64 = 1000;

/// Where the drain worker's wall clock goes.
///
/// The worker is the device's only executor: `device_drain` holds the device
/// lock for a whole tranche, so every guest FIFO packet, every GPU encode and
/// the host-window export are serialised behind it, and the guest's composite
/// rate cannot exceed the rate at which this thread finishes tranches.
///
/// Nothing else measures that. `sync_exec_lock_hold` is a per-packet threshold
/// line that only fires above `SYNC_EXEC_STALL_US`, so a worker pinned at 100%
/// by a steady stream of 200 ms tranches is completely silent — which is the
/// "an event count is not a state" trap, applied to a cost. This reads the
/// state: what fraction of wall clock the worker spends holding the lock, split
/// by the two phases that can own it.
///
/// The split is the point. `drain_us` is guest work (FIFO decode, draws, compute,
/// guest writeback); `publish_us` is our host-window export, which quiesces the
/// whole GPU twice per present. A duty near 1 says the ~2 Hz composite rate is
/// ours and names which half to attack; a duty near 0 says the worker is idle
/// and the guest is blocked on something upstream of us. No other line separates
/// those two readings.
///
/// `skipped` counts tranches that returned before taking the lock at all
/// (`present_action_pending`): a worker that keeps bailing looks identical to an
/// idle one in the duty figure alone, and it is not the same fault.
/// Which phase of guest work a slice of `drain_us` belongs to.
///
/// These are attributions inside `drain_us`, not a partition of it: a flush
/// reached from inside a draw is counted by both. That is deliberate and it is
/// self-checking — if the three sum to more than `drain_us` the phases nest, and
/// if they sum to much less the time is somewhere none of them names. Either
/// reading is useful and a single fused figure gives neither.
#[derive(Clone, Copy)]
pub enum DrainPhase {
    /// `encode_draw_chain`: metal2vulkan translate, encode, submit, readback.
    Draw,
    /// One compute record applied: bind bookkeeping for most kinds, encode +
    /// execute for a dispatch. Timed as a whole because "the binds are the cost"
    /// is exactly as interesting an answer as "the dispatch is".
    Compute,
    /// Deferred window flush: resident readback + guest writeback.
    Flush(FlushRail),
}

/// Which deferred-writeback rail a [`DrainPhase::Flush`] was spent on.
///
/// The aggregate `flush_us` is three quarters of the drain worker's wall clock
/// on a driven boot and had no owner, so every fix aimed at it was aimed by
/// guess. It is not one mechanism: four independent rails report as `Flush`, and
/// their counts are nowhere near proportional. One measured second read
/// `flushes=103` beside `surface_flush=15`, so the render rail the cost had been
/// attributed to is under a sixth of the *count* — and nothing said whose
/// microseconds those were.
///
/// Count and cost answer different questions here. A rail that flushes 71 times
/// at 50 µs and a rail that flushes 15 times at 7 ms are indistinguishable in
/// `flushes` and are opposite problems, so both are reported per rail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlushRail {
    /// `flush_render_one`: pinned resident readback, then guest writeback.
    Render,
    /// `flush_gva_one`: deferred GVA-addressed surface writeback.
    Gva,
    /// `flush_linear_one`: deferred linear-texture writeback.
    Linear,
    /// `flush_storage_one`: deferred compute-storage writeback.
    Storage,
}

/// The inside of a [`FlushRail::Render`] flush, which is 100% of the drain
/// worker's flush cost on a driven boot and is four very different things.
///
/// Live: `render_us=688003 render=100`, i.e. 6.9 ms per flush, ~69% of the
/// worker's entire second, with the other three rails at zero. Knowing that is
/// not yet enough to fix it, because the four parts below have opposite fixes.
/// A cost in [`Fence`](Self::Fence) is a GPU round trip and shrinking the copy
/// would not touch it; a cost in [`Map`](Self::Map) or [`Write`](Self::Write)
/// is bytes and a dirty rect would. Guessing between them is how the last
/// attempt picked its target.
///
/// Splitting it paid, and the record of what it bought belongs here rather than
/// only in a commit body. [`Write`](Self::Write) turned out to be the largest
/// phase and to be three whole-frame passes sharing one counter — see
/// [`SurfaceWritePhase`], which divides it again. Removing the two that were not
/// the guest's bytes took the render flush from **7.98 ms to 3.95 ms**, with the
/// drain worker's duty falling from 0.915 to ~0.72 and its worst tranche from
/// 46.5 ms to 18.5 ms. Those are device-side numbers and they reproduce: this
/// rail now measures 3.86 ms per flush.
///
/// **A Safari `requestAnimationFrame` figure was also attributed to that change
/// — "59.1 fps to 119.2 fps" — and that attribution does not hold.** rAF on this
/// pathway is bimodal at ~59 and ~118 with nothing in between, and *both* states
/// occur on one build within one boot: probing the same unchanged binary four
/// times in six minutes read 59.5, 117.3, 119.0 and 120.0, the low one being the
/// first probe after login. A single rAF number therefore cannot attribute
/// anything to a code change, in either direction — it nearly caused this rail's
/// BGRA8 upload change to be reverted as a regression when re-probing the same
/// build returned 117.3. Pair rAF with the device-side counters above, and see
/// `AGENTS.md` for the probe rule.
///
/// The phase left holding the flush is [`Fence`](Self::Fence) at ~45%, with
/// [`Write`](Self::Write) ~28% and [`Map`](Self::Map) ~22% — 94% of `flush_us`
/// accounted. What the rail moves is the headline: 116 flushes a second, each a
/// whole 1920x1080 frame, is **962 MB/s** read back from the GPU and landed in
/// guest pages, for ~62 presented frames. Every phase here is proportional to
/// that volume, so the next lever is reading back less than the whole
/// attachment, not making any one phase faster.
///
/// The obvious form of that lever does not pay, and the number is recorded here
/// so it is not re-derived. The guest already supplies a damage rect —
/// `OPCODE_SET_SCISSOR`, decoded verbatim into `req.scissor` — so a writeback could
/// land only the scissored region. A 30 s driven Safari probe on the
/// x86/PCI/Vulkan pathway bucketed every window-arming Store by the fraction of
/// its attachment the scissor covered: **99.34% of the texels a Store arms are
/// texels it covers**. Half the Stores carry no scissor at all and the other
/// half carry one spanning the whole attachment; the small ones were 0.8% of the
/// population and 0.66% of the area. The 35% of *all* draws that are scissored
/// are the small draws *inside* a pass — an icon, a glyph run, a window's own
/// layer — while the Store that ends a full-screen composite declares the full
/// screen. Reading back less has to find its evidence somewhere other than the
/// guest's scissor.
///
/// This paragraph used to end "[`Fence`](Self::Fence) is the GPU rendering the
/// frame rather than latency to reschedule — that is measured, not assumed",
/// and the measurement it pointed at does not say that. What
/// [`ResidentArmCensus`] measured is that submitting the copy *earlier* cannot
/// help, which rules out one explanation without establishing another. Holding
/// the host GPU at its top clock then moved the same wait from 2.55-2.83 ms to
/// **0.40 ms** with no code change, so roughly six sevenths of it is the
/// governor and only the last seventh is work. Read [`ResidentArmCensus`] for
/// the table, and record the host GPU's power state beside any number taken
/// from this phase.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadbackPhase {
    /// Record the copy command buffer and submit it. No GPU wait.
    Submit,
    /// Block on the readback fence: pure GPU round-trip latency, and the part
    /// no smaller copy can reduce.
    ///
    /// **It is no longer inside `flush_us`, and that is the trap.** The
    /// GPU-direct writeback rail submits without waiting and settles at the
    /// completion stamp (`engine::quiesce_guest_writes`), which is outside
    /// `DrainPhase::Flush`. So `flush_us` fell 1370 → 123 us per flush across
    /// that change while the total moved −0.7%: the wait relocated. **Never
    /// compare `flush_us` across it** — add `fence_us` back first.
    ///
    /// `fence` counts settles rather than windows in principle, but no boot has
    /// yet made those differ: it equals `submit` on both sides of the change,
    /// because this workload arms one window per stamp. When they do diverge,
    /// `fence_us / fence` is the mean cost of a pass and `fence_us / submit` is
    /// the mean per frame.
    ///
    /// What the split is for is `fence_us` minus `gpu_us` — submit-to-start plus
    /// signal-to-wake, the part no smaller copy reduces. It measured 734 us
    /// before that change and 740 us after, on ~410 settles a second. That is
    /// the largest single cost left on this rail and nothing has touched it.
    ///
    /// The copying rail below still reports one `Fence` per readback, so a
    /// window in which both ran mixes the two counts.
    Fence,
    /// Make the staging buffer readable. On the leased arm that is the
    /// invalidate alone, because the mapping already exists for the slot's
    /// lifetime; on the fallback arm it is map, invalidate and a whole-frame
    /// memcpy into a host `Vec`. The two differ by ~8 MB, so this phase reads
    /// near zero exactly when every readback in the window was leased and
    /// climbs in proportion to the ones that were not.
    Map,
    /// Write the frame into the guest's pages (`write_bgra8`).
    ///
    /// Reads zero on a window the GPU rail landed, because that rail's
    /// destination *is* the guest's pages and there is no second pass to time.
    /// See `mapping_write::write_bgra8_from_resident_gpu`.
    Write,
    /// Re-walk the mapping's page list against the guest's page table
    /// (`mapper::vouch_mapping_pages_verdict`), which is what licenses any write
    /// to those pages at all.
    ///
    /// Split out because it is `O(pages)` with a guest page-table walk each, it
    /// runs once per flush on every rail, and until the host copies were removed
    /// it was hidden inside a millisecond of memcpy. A rail that has stopped
    /// moving bytes is measured by what it does instead.
    Vouch,
    /// Everything else a flush does before the copy: resolve the sample window,
    /// turn the page list into a slice of the imported RAMBlock, and mark the
    /// write footprint. Also `O(pages)`, and separate from `Vouch` because the
    /// two walks answer different questions and would be shortened differently —
    /// this one turns page entries into addresses, that one licenses writing to
    /// them at all.
    Resolve,
}

impl ReadbackPhase {
    /// How many phases there are. The census arrays are sized from this, so a
    /// new variant that forgets to bump it fails to build [`Self::ALL`] rather
    /// than overflowing an array at report time.
    pub(crate) const COUNT: usize = 6;

    const ALL: [ReadbackPhase; Self::COUNT] = [
        ReadbackPhase::Submit,
        ReadbackPhase::Fence,
        ReadbackPhase::Map,
        ReadbackPhase::Write,
        ReadbackPhase::Vouch,
        ReadbackPhase::Resolve,
    ];

    const fn index(self) -> usize {
        match self {
            ReadbackPhase::Submit => 0,
            ReadbackPhase::Fence => 1,
            ReadbackPhase::Map => 2,
            ReadbackPhase::Write => 3,
            ReadbackPhase::Vouch => 4,
            ReadbackPhase::Resolve => 5,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            ReadbackPhase::Submit => "submit",
            ReadbackPhase::Fence => "fence",
            ReadbackPhase::Map => "map",
            ReadbackPhase::Write => "write",
            ReadbackPhase::Vouch => "vouch",
            ReadbackPhase::Resolve => "resolve",
        }
    }
}

/// Which part of `process_exec_indirect2` a span was spent in.
///
/// One opcode carries the whole dispatch: the per-opcode split of `proc_us` puts
/// `CHILD_OP_EXEC_INDIRECT2` at **754-775 ms/s** against **under 3.3 ms/s for
/// the eleven other opcodes combined**. `draw_us` names 585 ms/s of that, and it
/// names it narrowly — `DrainPhase::Draw` wraps `encode_draw_chain` alone,
/// inside the per-draw loop. **The remaining ~197 ms/s is the whole of the drain
/// residue that is left**, and no span reaches it.
///
/// These tile the function rather than nominate a part of it, which is the
/// method that worked on the child-FIFO loop after nominating one twice did not.
/// [`Self::Header`] is deliberately the leftover: it is timed as the function's
/// total minus the other four, so a cost in a corner nobody listed still lands
/// somewhere and the sum still equals `op0x37_us`.
///
/// # What it measured, driven macos-13, 74 windows
///
/// `sum` against `op0x37_us` is **0.999**, so the tiling closes and the split is
/// arithmetic:
///
/// ```text
/// finish     639.6 ms/s   (contains draw_us)
/// preflight   74.6 ms/s   <- the largest span outside the encode
/// walk        45.7 ms/s
/// header       6.7 ms/s
/// load         3.7 ms/s
/// finish - draw  ~26 ms/s
/// ```
///
/// **The largest non-draw cost in this device is the speculative preflight**, at
/// ~10 % of the whole drain worker. `Load` is 2 % — a structural read had
/// nominated the per-command-buffer allocation and GVA copy as the likely
/// dominant cost, and it is not, which is the third nomination this tiling has
/// retired. `Header` being small is the reassuring reading: the cost is in spans
/// that were named rather than in a corner nobody listed.
///
/// On a host with no host-pointer import (`REIMS_VGPU_GUEST_IMPORT=off`, two
/// boots) the shape changes and only in the expected place: `finish` rises to
/// 811-821 ms/s while `preflight` *falls* to 39-40, because that arm runs at
/// less than half the packet rate. The whole difference is the writeback copies,
/// which is the copying rail working rather than a regression.
#[derive(Clone, Copy)]
pub enum ExecPhase {
    /// The command-buffer load loop: one `vec![0u8; len]` and one
    /// `read_task_gva_by_id` per command buffer, each a GVA resolve and a copy
    /// out of guest memory. Sized by the guest, so a multi-MiB stream is a
    /// multi-MiB allocation and copy on the drain thread.
    Load,
    /// The speculative translation preflight: scans each stream for RENDER and
    /// COMPUTE pipeline refs and asks whether metal2vulkan has them, so a packet
    /// can be deferred whole rather than half-executed.
    Preflight,
    /// `walk_stream`: decoding every record of every segment and applying the
    /// bind bookkeeping, which is where `render::decode` and `apply_binds` run.
    /// The inner loop of the whole device.
    Walk,
    /// `finish_stream`: clears, ICB executes, the draw list, and the per-draw
    /// loop. **Contains `draw_us`**, which names itself, so `Finish` minus
    /// `draw_us` is the per-draw setup and result handling around the encode.
    Finish,
    /// `semantic_submission_segments`: a second pass over every loaded stream,
    /// before the walk that decodes them, to cut the submission into segments.
    ///
    /// Carved out of [`Self::Header`] because it is a whole extra traversal of
    /// the same bytes [`Self::Walk`] then traverses, and an aggregate that
    /// merely says "the leftover is large" cannot say whether the cost is a
    /// traversal or the bookkeeping beside it. Those want opposite repairs.
    Segments,
    /// Opening the submission: `consume_resource_table`, `begin_submission`,
    /// materializing one `SubmissionResourceUse` per resource descriptor, and
    /// `submissions.begin`.
    ///
    /// Scales with the resource table's length rather than with the stream's,
    /// which is why it is separated from [`Self::Segments`] beside it — one is
    /// paid per byte of command stream and the other per resource the guest
    /// named, and a boot cannot tell them apart while they share a field.
    Open,
    /// Closing it: `submissions.finish` and `complete_submission`.
    Close,
    /// Everything else the function does — header and payload validation, the
    /// resource-table decode, and any path that returns early. Derived rather
    /// than measured directly, so the phases sum to the function's own total by
    /// construction.
    ///
    /// **A large reading here is a finding about this census, not about the
    /// device**: it means real cost is sitting in a corner no span names, and
    /// the response is to tile that corner rather than to reason about what
    /// might be in it. That has now happened twice — the first time this was
    /// documented at 6.7 ms/s with the note that "Header being small is the
    /// reassuring reading", and a later driven Maps boot read it at 176 ms/s,
    /// second only to the encode. [`Self::Segments`], [`Self::Open`] and
    /// [`Self::Close`] are what that reading was carved into.
    Header,
}

impl ExecPhase {
    /// How many phases there are. The census arrays are sized from this, so a
    /// new variant that forgets to bump it fails to build [`Self::ALL`] rather
    /// than overflowing an array at report time.
    pub(crate) const COUNT: usize = 8;

    const ALL: [ExecPhase; Self::COUNT] = [
        ExecPhase::Load,
        ExecPhase::Preflight,
        ExecPhase::Walk,
        ExecPhase::Finish,
        ExecPhase::Segments,
        ExecPhase::Open,
        ExecPhase::Close,
        ExecPhase::Header,
    ];

    const fn index(self) -> usize {
        match self {
            ExecPhase::Load => 0,
            ExecPhase::Preflight => 1,
            ExecPhase::Walk => 2,
            ExecPhase::Finish => 3,
            ExecPhase::Segments => 4,
            ExecPhase::Open => 5,
            ExecPhase::Close => 6,
            ExecPhase::Header => 7,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            ExecPhase::Load => "load",
            ExecPhase::Preflight => "preflight",
            ExecPhase::Walk => "walk",
            ExecPhase::Finish => "finish",
            ExecPhase::Segments => "segments",
            ExecPhase::Open => "open",
            ExecPhase::Close => "close",
            ExecPhase::Header => "header",
        }
    }
}

/// Which part of the translation preflight a span was spent in.
///
/// [`ExecPhase::Preflight`] is **74.6 ms/s, the largest cost in this device
/// outside the draw encode**, and it is speculative work: it re-derives, per
/// exec packet, whether metal2vulkan already holds every pipeline the streams
/// reference. Three unlike things happen in there and the aggregate cannot say
/// which one costs, which is exactly the shape `RegsOp` was added for after the
/// same mistake.
///
/// The three sum to `preflight_us`, so the identity is checkable on the line.
/// It reads ~0.95 rather than 1.00 because the `extract_air` calls that sit
/// between the `Air` and `Cache` spans are outside both.
///
/// # What it measured, two driven macos-13 boots
///
/// ```text
/// air      53.90 / 54.23 ms/s    4.34 / 4.30 us per pipeline ref   <- 71 %
/// cache    16.30 / 16.60 ms/s    1.30 / 1.31 us per pipeline ref
/// refs      6.24 / 6.23 ms/s     0.41 / 0.40 us per call
/// pipes/s  12 650 / 12 786
/// ```
///
/// `Refs` — the second full decode of the stream, the part most obviously
/// redundant since `walk_stream` decodes the same records straight afterwards —
/// is **8 %**. The cost is `Air`, and *within* `Air` it is the three
/// guest-memory resolves rather than the AIR copies: removing both copies moved
/// `air_us/pipe` by only 4.7 %.
///
/// # The lever that is left, and why it is soundable
///
/// The remaining ~50 ms/s comes off only by **not resolving at all** — a memo of
/// `(task_id, pipeline_ref)` already confirmed translated.
///
/// What makes that keepable-sound is that **the m2v cache is unbounded and
/// nothing evicts it**: its sole removal is `forget_if_transient`, which drops a
/// transient *failure* so it can be retried. An `Entry::Ready` stays Ready for
/// the life of the process, so the only staleness is the guest repointing a ref
/// at different AIR — which the object-deletion paths already hook.
///
/// The failure mode is bounded but **not free**, which is why it has not been
/// rushed in: `translate_cached_reflected` falls through to a *synchronous*
/// translate on a miss, so a stale memo does not lose guest work — it runs an
/// AIR-to-SPIR-V translation inline on the drain thread while the device lock is
/// held, which is exactly what the asynchronous preflight exists to avoid.
/// Design the invalidation against the deletion paths before taking the memo.
#[derive(Clone, Copy)]
pub enum PreflightPart {
    /// Collecting the distinct pipeline refs: `iter_segments` and a full
    /// `render::decode` / `compute::decode` of every record in the stream — the
    /// *same* walk `walk_stream` is about to make, done a second time because
    /// the answer has to be complete before any record runs.
    Refs,
    /// `load_render_air_pair` and its compute counterpart: resolving each
    /// pipeline's AIR out of guest memory.
    Air,
    /// `m2v_cache::ensure_cached_async`, which digests the whole AIR blob to
    /// build the key and then takes the cache's global lock. Twice per render
    /// pipeline, once per kernel.
    Cache,
}

impl PreflightPart {
    /// How many parts there are. The census arrays are sized from this, so a new
    /// variant that forgets to bump it fails to build [`Self::ALL`] rather than
    /// overflowing an array at report time.
    pub(crate) const COUNT: usize = 3;

    const ALL: [PreflightPart; Self::COUNT] = [
        PreflightPart::Refs,
        PreflightPart::Air,
        PreflightPart::Cache,
    ];

    const fn index(self) -> usize {
        match self {
            PreflightPart::Refs => 0,
            PreflightPart::Air => 1,
            PreflightPart::Cache => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            PreflightPart::Refs => "refs",
            PreflightPart::Air => "air",
            PreflightPart::Cache => "cache",
        }
    }
}

/// Which part of opening a submission a span was spent in.
///
/// [`ExecPhase::Open`] is **147 ms/s of a drain worker at 0.95 duty** on driven
/// fullscreen Maps — 260 us of CPU to open one submission, and the second
/// largest cost in this device after the draw encode. Three unlike things
/// happen in there over the same descriptor slice, and the aggregate cannot say
/// which one costs.
///
/// It has already misdirected one repair. `begin_submission` took four
/// `BTreeMap` descents per descriptor where two would do, which is real
/// duplicated work and looked like the answer; collapsing it to two moved
/// `open_us` by nothing measurable (4.158/4.225 against 4.182 us a draw). So
/// the map descents are not where the time is, and the next guess would have
/// been another guess. These three tile it instead.
///
/// `descs` is here because the per-packet figure alone cannot be reasoned
/// about: 260 us is a hundred entries at 2.6 us each or ten thousand at 26 ns,
/// and those are opposite problems. The count is the denominator every other
/// field in this line needs.
///
/// The two passes are **not** redundant and must not be merged. Every validity
/// record has to be applied before any expected content is snapshotted, because
/// a guest-write declaration creates the version the following commands expect
/// — one fused loop would snapshot descriptor `i` before descriptor `j`'s
/// declaration had landed.
#[derive(Clone, Copy)]
pub enum OpenPart {
    /// `consume_resource_table`: the guest's own statement of who owns each
    /// resource's authoritative bytes, applied one descriptor at a time.
    Table,
    /// `begin_submission`: resolving each descriptor and entering the resource
    /// it names into this submission.
    Begin,
    /// Materializing one `SubmissionResourceUse` per descriptor and handing the
    /// slice to `submissions.begin`.
    Use,
}

impl OpenPart {
    /// How many parts there are. The census arrays are sized from this, so a new
    /// variant that forgets to bump it fails to build [`Self::ALL`] rather than
    /// overflowing an array at report time.
    pub(crate) const COUNT: usize = 3;

    const ALL: [OpenPart; Self::COUNT] = [OpenPart::Table, OpenPart::Begin, OpenPart::Use];

    const fn index(self) -> usize {
        match self {
            OpenPart::Table => 0,
            OpenPart::Begin => 1,
            OpenPart::Use => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            OpenPart::Table => "table",
            OpenPart::Begin => "begin",
            OpenPart::Use => "use",
        }
    }
}

/// Which part of `finish_stream` a span was spent in.
///
/// [`ExecPhase::Finish`] is the largest phase in this device — 9.61 µs a draw of
/// an 11.72 µs `proc_us` on a driven Maps boot — and it *contains* `draw_us`,
/// which names itself at 8.36. So 1.25 µs a draw, 15 % of the whole draw path
/// and larger than any single bar in `draw_phase`, sits between the two with no
/// field naming it. This divides that residue, in the same shape and for the
/// same reason as the `Record` and `Stage` splits inside the engine.
///
/// The parts tile `finish_us` by construction: every one of them is a lexical
/// span in `finish_stream` and [`Prelude`](Self::Prelude) is what is left, so
/// the sum is checkable on the line rather than assumed.
///
/// Read [`Encode`](Self::Encode) against `drain_duty`'s own `draw_us`: they
/// measure the same call from two places, so a divergence between them is a
/// census bug and not a finding.
#[derive(Clone, Copy)]
pub enum FinishPhase {
    /// Everything outside the other parts: the clear-only prelude, the ICB
    /// executes, the `draw_list` collect, `mrt_draw_request` for the first
    /// record, and the attachment template it is turned into. Derived rather
    /// than measured directly, so the parts sum to the whole by construction.
    Prelude,
    /// `retarget_render_pass_draw` — rebuilding record N's `DrawRequest` from
    /// the pass template.
    ///
    /// Entered once per record so `fin_retarget_n` is the loop's own trip count
    /// and every other part has a denominator; record 0 only moves the request
    /// the prelude already built, and charges near nothing.
    ///
    /// A `MTLRenderCommandEncoder`'s attachment set is fixed for its life and
    /// the guest never re-states it, so anything this costs is this device
    /// re-materializing state the guest sent once.
    Retarget,
    /// `fill_draw_binds_from_pending` and the per-record request fixups around
    /// it: the chain position, the records-2+ load-action rewrite, and choosing
    /// the chain source.
    Binds,
    /// `encode_draw_chain`, which is also what `drain_duty`'s `draw_us` spans.
    Encode,
    /// Reading the encode's result: the visibility-count bookkeeping and the
    /// status match, including the abandon paths.
    Result,
    /// After the per-draw loop: `write_visibility_results` and the
    /// draw-failed clear fallback.
    Tail,
}

impl FinishPhase {
    /// How many parts there are. The census arrays are sized from this, so a new
    /// variant that forgets to bump it fails to build [`Self::ALL`] rather than
    /// overflowing an array at report time.
    pub(crate) const COUNT: usize = 6;

    pub(crate) const ALL: [FinishPhase; Self::COUNT] = [
        FinishPhase::Prelude,
        FinishPhase::Retarget,
        FinishPhase::Binds,
        FinishPhase::Encode,
        FinishPhase::Result,
        FinishPhase::Tail,
    ];

    pub(crate) const fn index(self) -> usize {
        match self {
            FinishPhase::Prelude => 0,
            FinishPhase::Retarget => 1,
            FinishPhase::Binds => 2,
            FinishPhase::Encode => 3,
            FinishPhase::Result => 4,
            FinishPhase::Tail => 5,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            FinishPhase::Prelude => "fin_prelude",
            FinishPhase::Retarget => "fin_retarget",
            FinishPhase::Binds => "fin_binds",
            FinishPhase::Encode => "fin_encode",
            FinishPhase::Result => "fin_result",
            FinishPhase::Tail => "fin_tail",
        }
    }
}

/// The per-opcode split of `proc_ns`, indexed by the opcode itself.
///
/// Sized from the contract rather than from a count of the arms
/// `process_child_packet` happens to have: [`CHILD_OP_MAX`] is the largest
/// opcode the child FIFO defines, so a table one wider than it can hold every
/// opcode the guest can legally send and there is no bound left to overflow.
/// That is the whole reason this is direct-indexed rather than an associative
/// table with a probe and a dropped-entry counter — a capacity derived from the
/// wire format cannot be one short the way a hand-picked slot count can.
///
/// An opcode above the maximum is counted in `above_max` rather than indexed;
/// decode is expected to have refused it long before here, so a non-zero reading
/// is a decoder result, not a table result.
///
/// Its own type because `#[derive(Default)]` stops at 32-element arrays and
/// this is 65 wide. The manual impl is the whole cost of getting the bound from
/// the contract.
struct ProcOpTable {
    ns: [std::sync::atomic::AtomicU64; PROC_OP_SLOTS],
    count: [std::sync::atomic::AtomicU64; PROC_OP_SLOTS],
    above_max: std::sync::atomic::AtomicU64,
}

/// One slot per legal child opcode, `0..=CHILD_OP_MAX`. Derived from the
/// contract's own maximum, so a new opcode widens the table by widening that.
const PROC_OP_SLOTS: usize = crate::model::CHILD_OP_MAX as usize + 1;

impl Default for ProcOpTable {
    fn default() -> Self {
        Self {
            ns: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)),
            count: std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0)),
            above_max: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Which of the three accesses `drain_child_fifo` makes around a packet.
///
/// They were one counter first, and the aggregate misled: it read 5055 ns an
/// access, which is not a plausible cost for four bytes and is only explicable
/// if the three are not alike. They are not — see the `regs_op_ns` field doc.
#[derive(Clone, Copy)]
pub enum RegsOp {
    /// The `CHILD_REG_TAIL` read at the top of each loop iteration. A four-byte
    /// `address_space_read`, and the one op here that really is one.
    TailRead,
    /// The `CHILD_REG_HEAD` writeback after a packet is processed. Four bytes
    /// again, but through `gpa_map::write_u32` rather than the raw callback, so
    /// it carries whatever page bookkeeping that path does.
    HeadWrite,
    /// The completion stamp. **Not a word write**: on the Vulkan arm this tries
    /// the GPU rail first — resolving a guest RAM reference and submitting a
    /// command buffer — and falls through to a blocking settle when that
    /// declines. If the 97 ms/s is anywhere, the prior is that it is here, which
    /// is exactly why it is measured rather than assumed.
    Stamp,
}

impl RegsOp {
    /// How many accesses there are. The census arrays are sized from this, so a
    /// new variant that forgets to bump it fails to build [`Self::ALL`] rather
    /// than overflowing an array at report time.
    pub(crate) const COUNT: usize = 3;

    const ALL: [RegsOp; Self::COUNT] = [RegsOp::TailRead, RegsOp::HeadWrite, RegsOp::Stamp];

    const fn index(self) -> usize {
        match self {
            RegsOp::TailRead => 0,
            RegsOp::HeadWrite => 1,
            RegsOp::Stamp => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            RegsOp::TailRead => "tailrd",
            RegsOp::HeadWrite => "headwr",
            RegsOp::Stamp => "stamp",
        }
    }
}

/// One of the per-tranche sweeps that run after `publish_us` is banked.
///
/// They are the whole of `gap_post_us`, which is **~21 ms of every driven second**
/// on the x86/Vulkan iGPU — the only device-side item left in the drain worker's
/// missing third once `gap_lock_us` (0.02 %) and the interrupt hop (6 % of the
/// idle, ~10 µs a pulse) were measured and excluded.
///
/// Every one of them documents itself as returning immediately when there is
/// nothing to do, and collectively they cost this, so which one it is cannot be
/// reasoned out from those docs. Hence a split rather than a guess.
#[derive(Clone, Copy)]
pub enum PostSweep {
    /// `surface_cache::note_cache_levels` — self-gated to a one-second cadence.
    CacheLevels,
    /// `objects::slot_recheck::sweep` — deliberately per tranche, because the
    /// sampling interval is the resolution of the answer it gives. Watches
    /// nothing on every rail but macos-26.
    SlotRecheck,
    /// `released_pages::sweep` + `note_levels`, timed as one because they are
    /// the two halves of the same question and neither has a caller elsewhere.
    ReleasedPages,
    /// `bound_buffers::note_registry_levels`, Vulkan only.
    BindLevels,
}

impl PostSweep {
    /// How many sweeps there are. The census array is sized from this, so a new
    /// variant that forgets to bump it fails to build [`Self::ALL`] rather than
    /// overflowing an array at report time.
    pub(crate) const COUNT: usize = 4;

    const ALL: [PostSweep; Self::COUNT] = [
        PostSweep::CacheLevels,
        PostSweep::SlotRecheck,
        PostSweep::ReleasedPages,
        PostSweep::BindLevels,
    ];

    const fn index(self) -> usize {
        match self {
            PostSweep::CacheLevels => 0,
            PostSweep::SlotRecheck => 1,
            PostSweep::ReleasedPages => 2,
            PostSweep::BindLevels => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            PostSweep::CacheLevels => "cachelv",
            PostSweep::SlotRecheck => "slotre",
            PostSweep::ReleasedPages => "relpg",
            PostSweep::BindLevels => "bindlv",
        }
    }
}

impl FlushRail {
    const ALL: [FlushRail; 4] = [
        FlushRail::Render,
        FlushRail::Gva,
        FlushRail::Linear,
        FlushRail::Storage,
    ];

    const fn index(self) -> usize {
        match self {
            FlushRail::Render => 0,
            FlushRail::Gva => 1,
            FlushRail::Linear => 2,
            FlushRail::Storage => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            FlushRail::Render => "render",
            FlushRail::Gva => "gva",
            FlushRail::Linear => "linear",
            FlushRail::Storage => "storage",
        }
    }
}

/// Why a drain tranche did or did not hand the host window a new frame.
///
/// With the swapchain fixed, `host_window_cadence` reads `presents == offered`
/// with `busy_acquire=0` — the window shows every frame it is offered and drops
/// none. So the remaining deficit is entirely in the offer rate, which was 58/s
/// on a host panel at 120 Hz while the drain worker completed 110–132 render
/// flushes a second and the guest sustained ~117 fps. Something between the
/// composite and the window is halving it, and `publish_window_frame` has four
/// separate ways to return without publishing that the cadence census cannot see
/// from the other side.
///
/// The one that matters is [`SameKey`](Self::SameKey) against
/// [`Fresh`](Self::Fresh). A large `same_key` means the guest is presenting at
/// the offer rate and the window is being given everything there is — the
/// deficit would then be the guest's own present cadence, not ours. `fresh` near
/// the tranche rate with `same_key` small would mean the opposite. Those have
/// nothing in common as fixes, which is why this is measured before either is
/// attempted.
///
/// `fresh` counts a new key **reaching** the publish, not a frame landing in the
/// window's slot: the four ways the publish itself can still fail after that
/// point already have their own census in
/// [`crate::runtime::census::present_proxy::window_publish`], and duplicating
/// them here would give two counters that could disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowPublish {
    /// A frame key not yet published reached the publish.
    Fresh,
    /// No window is attached to consume a frame.
    NoWindow,
    /// The device holds no valid captured frame yet.
    NoFrame,
    /// The captured frame is the one already published — same mapping,
    /// generation and present epoch.
    SameKey,
}

impl WindowPublish {
    const ALL: [WindowPublish; 4] = [
        WindowPublish::Fresh,
        WindowPublish::NoWindow,
        WindowPublish::NoFrame,
        WindowPublish::SameKey,
    ];

    const fn index(self) -> usize {
        match self {
            WindowPublish::Fresh => 0,
            WindowPublish::NoWindow => 1,
            WindowPublish::NoFrame => 2,
            WindowPublish::SameKey => 3,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            WindowPublish::Fresh => "fresh",
            WindowPublish::NoWindow => "no_window",
            WindowPublish::NoFrame => "no_frame",
            WindowPublish::SameKey => "same_key",
        }
    }
}

#[derive(Default)]
pub(crate) struct WindowPublishCensus {
    arms: [std::sync::atomic::AtomicU64; 4],
}

impl WindowPublishCensus {
    pub(super) fn note(&self, arm: WindowPublish) {
        self.arms[arm.index()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(super) fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let counts: Vec<u64> = WindowPublish::ALL
            .iter()
            .map(|arm| self.arms[arm.index()].swap(0, Relaxed))
            .collect();
        if counts.iter().all(|&n| n == 0) {
            return None;
        }
        let body: String = WindowPublish::ALL
            .iter()
            .zip(&counts)
            .map(|(arm, n)| format!(" {}={n}", arm.label()))
            .collect();
        Some(format!("window_publish win_ms={win_ms}{body}"))
    }
}

/// The inside of [`ReadbackPhase::Write`], which is now the largest phase of the
/// largest rail and is three full-frame passes wearing one name.
///
/// `write_us=377356 write=95` is 3.97 ms per flush and 40% of the drain worker's
/// busy second, on an 8.29 MB frame — an effective 2.1 GB/s against ~9 GB/s for
/// the readback's own memcpy of the identical bytes. A previous attempt read
/// that gap as "cache-cold scattered writes into guest RAM, so only fewer bytes
/// help", removed a staging hop on that basis and measured no change.
///
/// The gap is a factor of four, which is the shape of doing the work four times,
/// not of doing it once badly. `write_bgra8` makes up to three
/// whole-frame passes and the name covers all of them, so none of them can be
/// ruled in or out:
///
/// - [`Stage`](Self::Stage) — the fragmented path's `frame` buffer: an 8 MB
///   allocation plus every row copied into it, before a single guest byte moves.
///   The contiguous path skips this entirely, so which path the composite takes
///   decides whether it exists at all.
/// - [`Land`](Self::Land) — the bytes actually reaching guest pages. This is the
///   only pass the guest needs and the only one a dirty rect would shrink.
/// - [`Cache`](Self::Cache) — a second 8 MB allocation holding a host-side
///   duplicate of the same frame for [`crate::runtime::surface_cache`], built
///   unconditionally on every non-skipping write.
///
/// Two of the three are freshly allocated multi-megabyte buffers per flush, ~95
/// times a second. A `vec![0u8; 8_290_000]` is not free even zeroed by the
/// allocator: the pages come back untouched and the fill faults every one of
/// them in. Whether that is the missing factor is exactly what this measures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfaceWritePhase {
    /// Build the staged whole-frame buffer (fragmented path only).
    Stage,
    /// Move the bytes into the guest's pages.
    Land,
    /// Build and store the host-side [`crate::runtime::surface_cache`] copy.
    Cache,
}

impl SurfaceWritePhase {
    const ALL: [SurfaceWritePhase; 3] = [
        SurfaceWritePhase::Stage,
        SurfaceWritePhase::Land,
        SurfaceWritePhase::Cache,
    ];

    const fn index(self) -> usize {
        match self {
            SurfaceWritePhase::Stage => 0,
            SurfaceWritePhase::Land => 1,
            SurfaceWritePhase::Cache => 2,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            SurfaceWritePhase::Stage => "stage",
            SurfaceWritePhase::Land => "land",
            SurfaceWritePhase::Cache => "cache",
        }
    }
}

/// [`SurfaceWritePhase`] totals over the census window, plus which of the two
/// landing paths the writes took.
///
/// `contig` and `frag` are counted because the split is not readable from the
/// phase totals alone: a `stage_us` of zero means the contiguous path, and a
/// reader with no path count cannot tell that from "the staging is free".
#[derive(Default)]
pub(crate) struct SurfaceWriteCensus {
    us: [std::sync::atomic::AtomicU64; 3],
    count: [std::sync::atomic::AtomicU64; 3],
    max_us: [std::sync::atomic::AtomicU64; 3],
    contig: std::sync::atomic::AtomicU64,
    frag: std::sync::atomic::AtomicU64,
    bytes: std::sync::atomic::AtomicU64,
}

impl SurfaceWriteCensus {
    pub(super) fn note(&self, phase: SurfaceWritePhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.us[i].fetch_add(us, Relaxed);
        self.count[i].fetch_add(1, Relaxed);
        self.max_us[i].fetch_max(us, Relaxed);
    }

    pub(super) fn note_path(&self, contiguous: bool, bytes: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if contiguous {
            self.contig.fetch_add(1, Relaxed);
        } else {
            self.frag.fetch_add(1, Relaxed);
        }
        self.bytes.fetch_add(bytes, Relaxed);
    }

    pub(super) fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let contig = self.contig.swap(0, Relaxed);
        let frag = self.frag.swap(0, Relaxed);
        if contig == 0 && frag == 0 {
            return None;
        }
        let bytes = self.bytes.swap(0, Relaxed);
        let mut body = String::new();
        for phase in SurfaceWritePhase::ALL {
            let i = phase.index();
            let us = self.us[i].swap(0, Relaxed);
            let n = self.count[i].swap(0, Relaxed);
            let max = self.max_us[i].swap(0, Relaxed);
            let label = phase.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        Some(format!(
            "write_split win_ms={win_ms} contig={contig} frag={frag} bytes={bytes}{body}"
        ))
    }
}

/// How long a resident-backed render window sits armed before its flush reads
/// it, which is the only interval the readback's GPU round trip could hide in.
///
/// [`ReadbackPhase::Fence`] is 46% of the render rail and is paid at the flush,
/// because that is where the copy is submitted. Submitting it at the arm instead
/// — the guest's Store, where the window is created — would only shorten the
/// wait by however much wall clock separates the two: if the flush follows the
/// arm by less than the round trip, the fence still blocks and the move buys
/// nothing but complexity in the path that publishes composited pixels.
///
/// So this measures that separation before anything is built on it, rather than
/// assuming the GPU has idle time in between. **It refuted the proposal.** On a
/// driven boot: `arms=95 flushes=95 aged=95 age_us=33341 max_age_us=372`, i.e.
/// a mean arm→flush interval of **351 µs** with a **372 µs** worst case, beside
/// `fence_us=248748 fence=95` — a **2.6 ms** mean fence wait in the same second.
/// The interval is seven times shorter than the wait it would have to hide, and
/// the tight max says that is the whole distribution and not a mean concealing
/// a long tail. Submitting at the arm would leave ~2.2 ms of the 2.6 ms still to
/// wait, for a deferred readback slot and a second fence lifetime in the path
/// that publishes composited pixels.
///
/// What that also settles is *what the fence wait is not*. It is not scheduling
/// latency with slack to reclaim: the arm and the flush are 351 µs apart inside
/// one tranche, so the draws that produce the composite are still executing when
/// the copy is submitted, and the copy queues behind them however early it is
/// sent. Submitting earlier cannot help.
///
/// This paragraph used to continue "the 2.6 ms is the GPU rendering the frame.
/// Only cheaper draws can move it", and that inference does not follow from the
/// premise. Waiting on the GPU says the GPU is slow; it does not say the work is
/// large. **Measured, and it is not the work.** Same boot, same build, same
/// driven probe, with the only difference a synthetic load holding the host GPU
/// at its top clock instead of letting it choose:
///
/// | | host GPU at its own clock (P5, 800-1450 MHz) | held at P0, 2820 MHz |
/// |---|---|---|
/// | `fence_us`/`fence` | 2.55 - 2.83 ms | **0.40 ms** |
/// | total fence time per second | 265 - 341 ms | **35 ms** |
/// | `flush_us`/`flushes` | 4.0 ms | **0.83 - 1.75 ms** |
/// | Safari rAF long frames | 7 (0.39 %) | **0** |
/// | Safari rAF worst frame | 42 ms | **21 ms** |
///
/// So roughly six sevenths of the wait was the host GPU running at a third of
/// its clock or less, and the device's actual GPU cost per composited frame is
/// about **0.40 ms**. The governor is not misbehaving: this workload submits
/// ~0.4 ms of work per frame and then blocks, which reads as a few per cent
/// occupancy, and a few per cent occupancy is what a low clock is for.
///
/// Two consequences, and the second is the one that changes what to build:
///
/// - Any measurement of GPU-side latency here must record the host GPU's clock
///   and power state beside it, or it is a measurement of the governor. A number
///   taken at P5 against one taken at P0 is a 6x artefact with no code in it.
/// - The second consequence used to read "**this device is latency-bound on a
///   GPU that is usually downclocked, not throughput-bound**", and concluded
///   that removing a whole GPU round trip is worth about six times the flat GPU
///   cost while removing bytes is worth what it always was. **That does not
///   follow from the table above, and it is now measured false.** The premise is
///   that the wait shrinks with clock — and a copy moving 8 MB shrinks with
///   clock just as much as a latency does. The reading could not tell them
///   apart, and it picked one.
///
/// `readback_split`'s `bar_us` and `gpu_us` are the device's own timestamps
/// either side of the copy, and they settle it. Driven one-second windows on an
/// x86/PCI boot with the host GPU at P5:
///
/// ```text
/// fence 2.549 ms   copy 2.286 ms (89.7%)   draw-wait 0.0010 ms   ask 0.262 ms
/// fence 1.906 ms   copy 1.710 ms (89.7%)   draw-wait 0.0010 ms   ask 0.195 ms
/// fence 1.474 ms   copy 1.296 ms (87.9%)   draw-wait 0.0010 ms   ask 0.177 ms
/// ```
///
/// **87-91% of the fence wait is the copy executing.** 0.05% is the draw batch
/// it waits on — so the composite render is effectively free and the readback is
/// the device's whole GPU cost — and ~0.19 ms is the cost of asking. The copy
/// moves 8.29 MB at 3.6-6.4 GB/s in that power state. Two things follow, and
/// both are the opposite of what the old wording argued:
///
/// - **Removing bytes is worth ~1:1 against 90% of the largest cost in the
///   device.** The four levers the deferred-flush ledger priced in bytes are the
///   ones that would pay, and they were not being weighed against the right
///   number.
/// - **Removing the second submission is worth the other ~11%** — a stable
///   0.18-0.26 ms per readback, and no more. That prices a step left queued as a
///   top item on the grounds that "round trips *are* the cost". They are not.
///
/// `multi` is not noise to be averaged away. The age of "the arm" is a single
/// number only when exactly one window was armed since the last flush; a window
/// that drifted out through one of `flush_render_one`'s refusals never reaches
/// the flush site at all, so the count self-heals on the next arm rather than
/// sticking at a wrong live population forever.
#[derive(Default)]
pub(crate) struct ResidentArmCensus {
    /// Arms since the last flush read the counter. Reset to 0 on every read.
    arms_since_flush: std::sync::atomic::AtomicU64,
    /// [`crate::observe::elapsed_us`] at the most recent arm.
    last_arm_us: std::sync::atomic::AtomicU64,
    arms: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
    aged: std::sync::atomic::AtomicU64,
    age_us: std::sync::atomic::AtomicU64,
    max_age_us: std::sync::atomic::AtomicU64,
    /// Flushes reached with a count other than exactly one arm outstanding.
    multi: std::sync::atomic::AtomicU64,
}

impl ResidentArmCensus {
    pub(super) fn note_arm(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.arms.fetch_add(1, Relaxed);
        self.arms_since_flush.fetch_add(1, Relaxed);
        self.last_arm_us.store(now_us, Relaxed);
    }

    pub(super) fn note_flush(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.flushes.fetch_add(1, Relaxed);
        if self.arms_since_flush.swap(0, Relaxed) != 1 {
            self.multi.fetch_add(1, Relaxed);
            return;
        }
        let age = now_us.saturating_sub(self.last_arm_us.load(Relaxed));
        self.aged.fetch_add(1, Relaxed);
        self.age_us.fetch_add(age, Relaxed);
        self.max_age_us.fetch_max(age, Relaxed);
    }

    /// The line for the window that just closed, or `None` when no resident
    /// window was armed or flushed in it.
    pub(super) fn take(&self, win_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let arms = self.arms.swap(0, Relaxed);
        let flushes = self.flushes.swap(0, Relaxed);
        if arms == 0 && flushes == 0 {
            return None;
        }
        let aged = self.aged.swap(0, Relaxed);
        let total = self.age_us.swap(0, Relaxed);
        let max = self.max_age_us.swap(0, Relaxed);
        let multi = self.multi.swap(0, Relaxed);
        Some(format!(
            "resident_arm_age win_ms={win_ms} arms={arms} flushes={flushes} aged={aged} \
             age_us={total} max_age_us={max} multi={multi}"
        ))
    }
}

#[derive(Default)]
pub(crate) struct DrainDutyCensus {
    tranches: std::sync::atomic::AtomicU64,
    skipped: std::sync::atomic::AtomicU64,
    /// The wall clock this worker spent **not** in a tranche, split four ways so
    /// that `idle + lock + skip + post + busy` tiles the census window.
    ///
    /// `duty` says how much of the window the worker was busy. What the other
    /// `1 - duty` was has never been named, and on a driven Maps boot it is
    /// **31 %** of the one thread every guest packet serializes through — worth
    /// more than another microsecond off any phase, because at `duty` 1.0 and
    /// today's per-draw cost this device would run ~65 fps where it runs ~51.
    ///
    /// The four are the only things the worker can be doing. It waits on a
    /// condvar for the doorbell (`idle`); it takes the device lock, which the
    /// vCPU thread also holds (`lock`); it bails before the lock when the action
    /// BH owes a present (`skip`); and it runs the per-tranche sweeps after
    /// `publish_us` is taken (`post`). A reading concentrated in `idle` says the
    /// guest is upstream of us and no device change moves it; one in `lock` says
    /// the vCPU is the contender; one in `post` says a sweep on the worker's own
    /// path is the cost.
    ///
    /// Kept as separate accumulators rather than derived from a residue: a
    /// residue absorbs every mistake in the other three and always tiles.
    gap_idle_us: std::sync::atomic::AtomicU64,
    gap_lock_us: std::sync::atomic::AtomicU64,
    gap_skip_us: std::sync::atomic::AtomicU64,
    gap_post_us: std::sync::atomic::AtomicU64,
    /// The per-sweep division of [`Self::gap_post_us`], indexed by
    /// [`PostSweep::index`]. Emitted beside the total it divides, so
    /// `sum(post_*_us) == gap_post_us` is checkable on the line — a sweep that
    /// gains a call site and no timer shows up as a shortfall there.
    post_sweep_ns: [std::sync::atomic::AtomicU64; PostSweep::COUNT],
    /// `observe::elapsed_us()` when this worker last returned from a drain
    /// entry point, skipped or not. Zero before the first, which is the one
    /// entry whose `idle` cannot be measured and is dropped rather than
    /// attributed to a zero origin.
    gap_last_exit_us: std::sync::atomic::AtomicU64,
    /// How long a prompt `HostAction` — an IRQ pulse or a cursor move — sits in
    /// the slot queue between being enqueued here and the QEMU action BH popping
    /// it.
    ///
    /// This is the one candidate for `gap_idle_us` that is **ours**. The drain
    /// worker parks on a condvar until the guest doorbells, and the guest cannot
    /// doorbell until it has been interrupted; the interrupt is enqueued on the
    /// prompt queue and raised on the QEMU main loop, one `qemu_bh_schedule`
    /// later. If that hop costs hundreds of microseconds under load, the
    /// worker's idle is a main-loop round trip this device causes rather than
    /// the guest thinking — and the two have opposite repairs.
    ///
    /// `max` beside the total because a mean over a thousand pulses hides the
    /// tail, and the tail is what a frame waits on.
    irq_wait_us: std::sync::atomic::AtomicU64,
    irq_waits: std::sync::atomic::AtomicU64,
    irq_wait_max_us: std::sync::atomic::AtomicU64,
    /// `observe::elapsed_us()` at which the prompt queue stopped being empty, or
    /// zero while it is empty. The *oldest* undelivered action is the one whose
    /// wait matters, so arming over a non-empty queue leaves this alone.
    irq_armed_us: std::sync::atomic::AtomicU64,
    drain_us: std::sync::atomic::AtomicU64,
    publish_us: std::sync::atomic::AtomicU64,
    draw_us: std::sync::atomic::AtomicU64,
    draws: std::sync::atomic::AtomicU64,
    compute_us: std::sync::atomic::AtomicU64,
    computes: std::sync::atomic::AtomicU64,
    flush_us: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
    /// The two things a tranche does after `Device::drain` returns and before
    /// `drain_us` is taken: submit the deferred draw batch, and publish the
    /// present boundary.
    ///
    /// They are inside `drain_us` and outside every `DrainPhase`, which is the
    /// gap this pair closes. On a driven `blur=40` boot `drain_us` was 933 ms a
    /// second against `draw_us` 604 ms — **a third of the drain worker's wall
    /// clock named by nothing**, on the one thread every guest packet is
    /// serialized through. The same gap is 37 % on the sustained-animation
    /// probe, so it is not a property of one workload.
    tail_us: std::sync::atomic::AtomicU64,
    boundary_us: std::sync::atomic::AtomicU64,
    /// The per-packet halves of that same residue: the ring snapshot reads and
    /// the decode. Both run for every packet either FIFO drains, both sit
    /// inside `drain_us`, and neither is inside any [`DrainPhase`] — so if the
    /// third of the drain worker named by nothing is per-packet rather than
    /// per-opcode, it is here.
    ///
    /// **Nanoseconds, not microseconds, and that is not a style choice.** These
    /// fire tens of thousands of times a second and a single one costs well
    /// under a microsecond, so `as_micros()` would truncate most samples to
    /// zero and report a rail that runs constantly as free. `tail_us` above can
    /// afford microseconds because it is sampled once per tranche.
    ring_ns: std::sync::atomic::AtomicU64,
    ring_reads: std::sync::atomic::AtomicU64,
    decode_ns: std::sync::atomic::AtomicU64,
    packets: std::sync::atomic::AtomicU64,
    /// The rest of `drain_child_fifo`, so that the loop is covered end to end
    /// rather than sampled at two points.
    ///
    /// Two instruments in a row named a candidate from reading the code and
    /// were worth 0.3 % between them, so this does not name a third: `proc_ns`
    /// spans the whole opcode dispatch, `regs_ns` the guest register traffic
    /// every packet pays around it, and `setup_ns` the per-call prologue. With
    /// `ring_ns` and `decode_ns` those five tile one iteration of the loop, so
    /// whatever is left after subtracting them from the residue is *outside*
    /// the child FIFO entirely and the search moves rather than guesses again.
    ///
    /// `proc_ns` contains `draw_us` and `compute_us`, which name themselves on
    /// the same line — subtract them and what remains is the dispatch's own
    /// per-opcode cost. The other two contain nothing that names itself.
    ///
    /// Nanoseconds for the same reason as the pair above: `regs_ns` fires three
    /// times a packet and each one is a handful of guest word accesses.
    proc_ns: std::sync::atomic::AtomicU64,
    regs_ns: std::sync::atomic::AtomicU64,
    regs_ops: std::sync::atomic::AtomicU64,
    setup_ns: std::sync::atomic::AtomicU64,
    setup_calls: std::sync::atomic::AtomicU64,
    /// `regs_ns` split by which of the three accesses it was, because the
    /// aggregate turned out to be averaging three unlike things.
    ///
    /// The span was added expecting three cheap guest word accesses a packet and
    /// measured **5055 ns each**, 25x what an `address_space_read` of four bytes
    /// costs. `write_stamp` is why it cannot be read as register traffic: on the
    /// Vulkan arm it tries `stamp_word_ordered_on_gpu` first, which resolves a
    /// guest RAM reference and *submits a GPU command buffer*, and falls through
    /// to a blocking settle when that declines. So one of the three is a
    /// submission or a quiesce and the other two are word accesses, and a single
    /// bucket cannot say which carries the 97 ms/s.
    ///
    /// Indexed by [`RegsOp::index`]. The three **must** sum to `regs_ns` and the
    /// counts to `regs_ops`; that identity is the cheapest way to catch a
    /// mis-attributed site, so both the total and the split are emitted.
    regs_op_ns: [std::sync::atomic::AtomicU64; RegsOp::COUNT],
    regs_op_count: [std::sync::atomic::AtomicU64; RegsOp::COUNT],
    /// `proc_ns` split by the opcode that was dispatched, as an associative
    /// table keyed by the guest opcode itself.
    ///
    /// `proc - draw - compute` is 155-165 ms/s over two driven boots — the
    /// larger half of the residue, and 24 us a packet against a decode that
    /// costs 117 ns. `process_child_packet` is a match over ~24 arms and only
    /// one of them has ever been timed, so nothing says whether that 155 ms is
    /// one arm or spread across all of them.
    ///
    /// Keyed by opcode rather than by a hand-written class enum because the
    /// point is to find out which arms cost, and a class enum written now would
    /// encode the guess this instrument exists to avoid. Slots are claimed on
    /// first sight and never released, so the key scan is over the handful of
    /// opcodes a boot actually issues.
    ///
    /// See [`ProcOpTable`] for why this is indexed by the opcode rather than
    /// keyed by a class this device would have had to invent.
    proc_ops: ProcOpTable,
    /// The inside of `CHILD_OP_EXEC_INDIRECT2`, indexed by [`ExecPhase::index`].
    /// Sums to that opcode's own `op0x37_us` on the `drain_ops` line.
    exec_ns: [std::sync::atomic::AtomicU64; ExecPhase::COUNT],
    exec_count: [std::sync::atomic::AtomicU64; ExecPhase::COUNT],
    /// The inside of [`ExecPhase::Preflight`], indexed by
    /// [`PreflightPart::index`]. Sums to `preflight_us` on the `exec_phase`
    /// line. `pre_pipes` is the distinct pipeline refs the scan resolved, which
    /// is the denominator every per-pipeline figure needs.
    open_ns: [std::sync::atomic::AtomicU64; OpenPart::COUNT],
    open_count: [std::sync::atomic::AtomicU64; OpenPart::COUNT],
    open_descs: std::sync::atomic::AtomicU64,
    pre_ns: [std::sync::atomic::AtomicU64; PreflightPart::COUNT],
    pre_count: [std::sync::atomic::AtomicU64; PreflightPart::COUNT],
    pre_pipes: std::sync::atomic::AtomicU64,
    /// The inside of [`ExecPhase::Finish`], indexed by [`FinishPhase::index`].
    /// Sums to `finish_us` on the `exec_phase` line, and its `fin_encode` is
    /// `drain_duty`'s own `draw_us` measured a second time.
    fin_ns: [std::sync::atomic::AtomicU64; FinishPhase::COUNT],
    fin_count: [std::sync::atomic::AtomicU64; FinishPhase::COUNT],
    max_tranche_us: std::sync::atomic::AtomicU64,
    /// Longest single Flush in the window. `flush_us/flushes` is a mean, and a
    /// mean cannot tell "every flush costs 7.7 ms" from "most are free and one
    /// blocked 30 ms" — which are different defects with different fixes.
    max_flush_us: std::sync::atomic::AtomicU64,
    /// `flush_us`, `flushes` and `max_flush_us` again, split by [`FlushRail`]
    /// and indexed by [`FlushRail::index`].
    rail_us: [std::sync::atomic::AtomicU64; 4],
    rail_count: [std::sync::atomic::AtomicU64; 4],
    rail_max_us: [std::sync::atomic::AtomicU64; 4],
    /// The inside of the render rail, indexed by [`ReadbackPhase::index`].
    ///
    /// Sized from [`ReadbackPhase::COUNT`] and not from a literal. These were
    /// `[_; 4]` while the enum grew to six, and the only thing that noticed was
    /// an index-out-of-bounds panic in the census emitter — on the reporting
    /// path, so a build could pass every compile check and die the first time it
    /// printed a line.
    rb_us: [std::sync::atomic::AtomicU64; ReadbackPhase::COUNT],
    rb_count: [std::sync::atomic::AtomicU64; ReadbackPhase::COUNT],
    rb_max_us: [std::sync::atomic::AtomicU64; ReadbackPhase::COUNT],
    /// GPU-side execution of the readback command buffer, from the device's own
    /// timestamp queries, split at the barrier. `rb_bar_us` is the copy command
    /// buffer waiting for the draw batch ahead of it to finish; `rb_gpu_us` is
    /// the copy itself. Together they divide [`ReadbackPhase::Fence`], which is
    /// CPU wall clock and cannot tell either from the cost of asking.
    rb_bar_us: std::sync::atomic::AtomicU64,
    rb_gpu_us: std::sync::atomic::AtomicU64,
    rb_gpu_count: std::sync::atomic::AtomicU64,
    rb_gpu_max_us: std::sync::atomic::AtomicU64,
    /// The window length `note` last reported, so `take_flush_rails` states the
    /// same denominator instead of deriving a second one.
    last_win_ms: std::sync::atomic::AtomicU64,
    /// Tranches that held the device lock for at least one whole guest frame.
    /// `max_tranche_us` is a max with no count, so it cannot distinguish one
    /// 38 ms tranche from three 20 ms ones; this is that count.
    slow_tranches: std::sync::atomic::AtomicU64,
    last_report_ms: std::sync::atomic::AtomicU64,
}

/// A tranche at or above this held the device lock for a whole guest frame.
///
/// Derived from the VBL cadence we actually deliver, because that *is* the
/// budget: the vCPU blocks on the same mutex this tranche holds, so a tranche
/// longer than one frame interval is one the guest cannot have serviced in
/// time. Deriving it also means it tracks the refresh rate instead of becoming
/// a stale constant beside it.
const DRAIN_TRANCHE_SLOW_US: u64 = DISPLAY_VBL_MIN_INTERVAL_US;

impl DrainDutyCensus {
    /// Count one skipped tranche (lock never taken).
    pub(crate) fn note_skipped(&self) {
        self.skipped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Open a drain entry at `entry_us`, banking the condvar wait that preceded
    /// it, and hand back the same instant for the caller to time its lock from.
    ///
    /// The wait is measured from the *previous* exit rather than by bracketing
    /// `qemu_cond_wait`, because that wait is in the C shim and this crate does
    /// not get to hold a timer across it. The difference is the shim's own few
    /// instructions, which is below this line's resolution.
    pub(crate) fn note_gap_entry(&self, entry_us: u64) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.gap_last_exit_us.load(Relaxed);
        if last != 0 {
            self.gap_idle_us
                .fetch_add(entry_us.saturating_sub(last), Relaxed);
        }
        entry_us
    }

    /// The prompt queue has gone from empty to holding something at `now_us`.
    ///
    /// Idempotent while the queue stays non-empty: the first arm is the oldest
    /// undelivered action and is the one whose wait the BH hop costs.
    pub(crate) fn note_irq_armed(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let _ = self
            .irq_armed_us
            .compare_exchange(0, now_us.max(1), Relaxed, Relaxed);
    }

    /// The BH has emptied the prompt queue at `now_us`; bank the hop.
    pub(crate) fn note_irq_delivered(&self, now_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let armed = self.irq_armed_us.swap(0, Relaxed);
        if armed == 0 {
            return;
        }
        let waited = now_us.saturating_sub(armed);
        self.irq_wait_us.fetch_add(waited, Relaxed);
        self.irq_waits.fetch_add(1, Relaxed);
        self.irq_wait_max_us.fetch_max(waited, Relaxed);
    }

    /// Attribute `ns` of this entry's `gap_post_us` to one sweep.
    ///
    /// Nanoseconds because a single sweep call is a few hundred of them; the
    /// report divides back to microseconds like every other span on the line.
    pub(crate) fn note_post_sweep(&self, sweep: PostSweep, ns: u64) {
        self.post_sweep_ns[sweep.index()].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }

    /// Bank the device-lock wait of an entry that went on to run a tranche.
    pub(crate) fn note_gap_lock(&self, us: u64) {
        self.gap_lock_us
            .fetch_add(us, std::sync::atomic::Ordering::Relaxed);
    }

    /// Close a drain entry at `exit_us`, banking everything since `busy_end_us`
    /// as post-tranche work — or as a skip, when the entry never ran one.
    pub(crate) fn note_gap_exit(&self, exit_us: u64, busy_end_us: u64, skipped: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        let bucket = match skipped {
            true => &self.gap_skip_us,
            false => &self.gap_post_us,
        };
        bucket.fetch_add(exit_us.saturating_sub(busy_end_us), Relaxed);
        self.gap_last_exit_us.store(exit_us, Relaxed);
    }

    /// Attribute `us` of the current tranche's `drain_us` to one phase.
    pub(crate) fn note_phase(&self, phase: DrainPhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let (total, count) = match phase {
            DrainPhase::Draw => (&self.draw_us, &self.draws),
            DrainPhase::Compute => (&self.compute_us, &self.computes),
            DrainPhase::Flush(_) => (&self.flush_us, &self.flushes),
        };
        total.fetch_add(us, Relaxed);
        count.fetch_add(1, Relaxed);
        if let DrainPhase::Flush(rail) = phase {
            self.max_flush_us.fetch_max(us, Relaxed);
            let i = rail.index();
            self.rail_us[i].fetch_add(us, Relaxed);
            self.rail_count[i].fetch_add(1, Relaxed);
            self.rail_max_us[i].fetch_max(us, Relaxed);
        }
    }

    /// The per-rail split of the window `drain_duty` just reported, or `None`
    /// when nothing flushed in it.
    ///
    /// A separate line rather than twelve more columns on `drain_duty`, and
    /// driven by that line's emitter rather than a cadence of its own, so the
    /// two divide against each other: the rails must sum to `flush_us` and their
    /// counts to `flushes`. Valid only immediately after `note` returns `Some`,
    /// which is the only place it is called.
    pub(crate) fn take_flush_rails(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for rail in FlushRail::ALL {
            let i = rail.index();
            let us = self.rail_us[i].swap(0, Relaxed);
            let n = self.rail_count[i].swap(0, Relaxed);
            let max = self.rail_max_us[i].swap(0, Relaxed);
            any |= n != 0;
            let label = rail.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        any.then(|| format!("flush_rails win_ms={win_ms}{body}"))
    }

    /// The inside of the render rail over the window `drain_duty` just
    /// reported, or `None` when nothing was read back in it.
    ///
    /// Sits under `flush_rails`'s `render_us` and divides it. Read `gpu_us` and
    /// `bar_us` before concluding anything from `fence_us`: they are the GPU's
    /// own timestamps taken from inside that wait, so `fence_us` owning the line
    /// means latency only when `gpu_us` is a small part of it. When `gpu_us`
    /// owns `fence_us` the wait is the readback command buffer copying, which is
    /// bytes and a smaller copy does touch it; `bar_us` is the draw batch queued
    /// ahead of it, and only that part is a scheduling cost rather than a size
    /// one. `map_us`/`write_us` are host-side bytes either way.
    pub(crate) fn take_readback_split(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for phase in ReadbackPhase::ALL {
            let i = phase.index();
            let us = self.rb_us[i].swap(0, Relaxed);
            let n = self.rb_count[i].swap(0, Relaxed);
            let max = self.rb_max_us[i].swap(0, Relaxed);
            any |= n != 0;
            let label = phase.label();
            body.push_str(&format!(
                " {label}_us={us} {label}={n} {label}_max_us={max}"
            ));
        }
        let bar_us = self.rb_bar_us.swap(0, Relaxed);
        let gpu_us = self.rb_gpu_us.swap(0, Relaxed);
        let gpu = self.rb_gpu_count.swap(0, Relaxed);
        let gpu_max_us = self.rb_gpu_max_us.swap(0, Relaxed);
        body.push_str(&format!(
            " bar_us={bar_us} gpu_us={gpu_us} gpu={gpu} gpu_max_us={gpu_max_us}"
        ));
        any.then(|| format!("readback_split win_ms={win_ms}{body}"))
    }

    /// Record one readback command buffer's two GPU-side spans: `barrier_us`
    /// waiting for the draws, then `copy_us` moving the frame.
    fn note_readback_gpu(&self, barrier_us: u64, copy_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.rb_bar_us.fetch_add(barrier_us, Relaxed);
        self.rb_gpu_us.fetch_add(copy_us, Relaxed);
        self.rb_gpu_count.fetch_add(1, Relaxed);
        self.rb_gpu_max_us.fetch_max(copy_us, Relaxed);
    }

    /// The window length [`Self::note`] last reported over, so a census emitted
    /// beside `drain_duty` states the same denominator rather than deriving a
    /// second one from a clock that has moved since.
    pub(crate) fn last_window_ms(&self) -> u64 {
        self.last_win_ms.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn note_readback(&self, phase: ReadbackPhase, us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.rb_us[i].fetch_add(us, Relaxed);
        self.rb_count[i].fetch_add(1, Relaxed);
        self.rb_max_us[i].fetch_max(us, Relaxed);
    }

    /// Attribute the tranche tail — the deferred-batch submit and the present
    /// boundary — which sit inside `drain_us` and inside no [`DrainPhase`].
    pub(crate) fn note_tail(&self, tail_us: u64, boundary_us: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.tail_us.fetch_add(tail_us, Relaxed);
        self.boundary_us.fetch_add(boundary_us, Relaxed);
    }

    /// One ring snapshot read, in nanoseconds. Twice per packet — the header,
    /// then the packet the header sized.
    pub(crate) fn note_ring(&self, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.ring_ns.fetch_add(ns, Relaxed);
        self.ring_reads.fetch_add(1, Relaxed);
    }

    /// One packet decode, in nanoseconds. The count is the packet count for
    /// both FIFOs, which is what every per-packet figure is normalized by.
    pub(crate) fn note_decode(&self, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.decode_ns.fetch_add(ns, Relaxed);
        self.packets.fetch_add(1, Relaxed);
    }

    /// One `process_child_packet` dispatch, in nanoseconds, into the total and
    /// into the slot for its opcode. Contains the draw and compute phases, which
    /// name themselves on the same line.
    pub(crate) fn note_proc(&self, opcode: u16, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.proc_ns.fetch_add(ns, Relaxed);
        let Some(slot) = self.proc_ops.ns.get(opcode as usize) else {
            self.proc_ops.above_max.fetch_add(1, Relaxed);
            return;
        };
        slot.fetch_add(ns, Relaxed);
        self.proc_ops.count[opcode as usize].fetch_add(1, Relaxed);
    }

    /// The per-opcode split of the window [`Self::note`] just reported, or
    /// `None` when no packet was dispatched in it.
    ///
    /// Sits under `drain_duty`'s `proc_us` and divides it: the `_us` fields sum
    /// to `proc_us` and the `_n` fields to `packets`. Only opcodes the window
    /// actually saw are printed, so a line names the arms this workload takes
    /// rather than all 65 the contract allows.
    pub(crate) fn take_proc_ops(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for opcode in 0..PROC_OP_SLOTS {
            let n = self.proc_ops.count[opcode].swap(0, Relaxed);
            let us = self.proc_ops.ns[opcode].swap(0, Relaxed) / 1000;
            if n == 0 {
                continue;
            }
            any = true;
            body.push_str(&format!(" op{opcode:#04x}_us={us} op{opcode:#04x}_n={n}"));
        }
        let above = self.proc_ops.above_max.swap(0, Relaxed);
        any.then(|| format!("drain_ops win_ms={win_ms} above_max={above}{body}"))
    }

    /// One span inside the translation preflight, in nanoseconds.
    /// One span inside opening a submission, in nanoseconds, plus how many
    /// resource descriptors that submission carried.
    pub(crate) fn note_open(&self, part: OpenPart, ns: u64, descs: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = part.index();
        self.open_ns[i].fetch_add(ns, Relaxed);
        self.open_count[i].fetch_add(1, Relaxed);
        if matches!(part, OpenPart::Table) {
            self.open_descs.fetch_add(descs, Relaxed);
        }
    }

    /// The inside of [`ExecPhase::Open`] over the window just reported, or
    /// `None` when no submission opened in it. The `_us` fields sum to
    /// `exec_phase`'s `open_us`, so the identity is checkable on the line.
    pub(crate) fn take_open_parts(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for part in OpenPart::ALL {
            let i = part.index();
            let us = self.open_ns[i].swap(0, Relaxed) / 1000;
            let n = self.open_count[i].swap(0, Relaxed);
            any |= n != 0;
            let label = part.label();
            body.push_str(&format!(" {label}_us={us} {label}_n={n}"));
        }
        let descs = self.open_descs.swap(0, Relaxed);
        any.then(|| format!("open_split win_ms={win_ms} descs={descs}{body}"))
    }

    pub(crate) fn note_preflight(&self, part: PreflightPart, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = part.index();
        self.pre_ns[i].fetch_add(ns, Relaxed);
        self.pre_count[i].fetch_add(1, Relaxed);
    }

    /// One distinct pipeline ref the preflight scan resolved.
    pub(crate) fn note_preflight_pipe(&self) {
        self.pre_pipes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The inside of the preflight over the window [`Self::note`] just reported.
    ///
    /// Read against `exec_phase`: these three sum to its `preflight_us`.
    /// `pipes` is the distinct pipeline refs resolved, so `air_us / pipes` is
    /// what one AIR resolve costs and `pipes / preflight_n` is how many the
    /// average packet re-derives.
    pub(crate) fn take_preflight_parts(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for part in PreflightPart::ALL {
            let i = part.index();
            let us = self.pre_ns[i].swap(0, Relaxed) / 1000;
            let n = self.pre_count[i].swap(0, Relaxed);
            any |= n != 0;
            let label = part.label();
            body.push_str(&format!(" {label}_us={us} {label}_n={n}"));
        }
        let pipes = self.pre_pipes.swap(0, Relaxed);
        any.then(|| format!("preflight_split win_ms={win_ms} pipes={pipes}{body}"))
    }

    /// One span inside `process_exec_indirect2`, in nanoseconds.
    pub(crate) fn note_exec(&self, phase: ExecPhase, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.exec_ns[i].fetch_add(ns, Relaxed);
        self.exec_count[i].fetch_add(1, Relaxed);
    }

    /// The inside of `CHILD_OP_EXEC_INDIRECT2` over the window [`Self::note`]
    /// just reported, or `None` when no exec packet ran in it.
    ///
    /// Read against `drain_ops`: these sum to its `op0x37_us`. `finish_us`
    /// contains `drain_duty`'s `draw_us`, so `finish_us - draw_us` is the
    /// per-draw setup and result handling that sits around the encode and is
    /// named by nothing else.
    pub(crate) fn take_exec_phases(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for phase in ExecPhase::ALL {
            let i = phase.index();
            let us = self.exec_ns[i].swap(0, Relaxed) / 1000;
            let n = self.exec_count[i].swap(0, Relaxed);
            any |= n != 0;
            let label = phase.label();
            body.push_str(&format!(" {label}_us={us} {label}_n={n}"));
        }
        any.then(|| format!("exec_phase win_ms={win_ms}{body}"))
    }

    /// One stream's spans in one part of `finish_stream`: the nanoseconds it
    /// spent there and how many times it entered.
    ///
    /// Both at once because the caller accumulates a whole stream locally — a
    /// packet of ninety draws enters `Encode` ninety times and pays two atomics
    /// for all of them.
    pub(crate) fn note_finish(&self, phase: FinishPhase, ns: u64, entries: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let i = phase.index();
        self.fin_ns[i].fetch_add(ns, Relaxed);
        self.fin_count[i].fetch_add(entries, Relaxed);
    }

    /// The inside of [`ExecPhase::Finish`] over the window [`Self::note`] just
    /// reported.
    ///
    /// Read against `exec_phase`: these six sum to its `finish_us`, and
    /// `fin_encode_us` is `drain_duty`'s `draw_us` measured from the other side
    /// of the same call.
    pub(crate) fn take_finish_phases(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let win_ms = self.last_win_ms.load(Relaxed);
        let mut body = String::new();
        let mut any = false;
        for phase in FinishPhase::ALL {
            let i = phase.index();
            let us = self.fin_ns[i].swap(0, Relaxed) / 1000;
            let n = self.fin_count[i].swap(0, Relaxed);
            any |= n != 0;
            let label = phase.label();
            body.push_str(&format!(" {label}_us={us} {label}_n={n}"));
        }
        any.then(|| format!("finish_phase win_ms={win_ms}{body}"))
    }

    /// One access around a packet, in nanoseconds, into both the total and the
    /// per-op split. Recording both is what makes the sum identity checkable.
    pub(crate) fn note_regs(&self, op: RegsOp, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.regs_ns.fetch_add(ns, Relaxed);
        self.regs_ops.fetch_add(1, Relaxed);
        let i = op.index();
        self.regs_op_ns[i].fetch_add(ns, Relaxed);
        self.regs_op_count[i].fetch_add(1, Relaxed);
    }

    /// One `drain_child_fifo` prologue, in nanoseconds: the three register
    /// reads, the ring resolve, and the page-GPA copy the loop reads from.
    pub(crate) fn note_setup(&self, ns: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.setup_ns.fetch_add(ns, Relaxed);
        self.setup_calls.fetch_add(1, Relaxed);
    }

    /// Accumulate one completed tranche and return the line when a report is
    /// due. Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: that the window resets on report (so the
    /// figure is a rate over the window, not a lifetime average), and that duty
    /// is busy time over elapsed time.
    pub(crate) fn note(&self, drain_us: u64, publish_us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.tranches.fetch_add(1, Relaxed);
        self.drain_us.fetch_add(drain_us, Relaxed);
        self.publish_us.fetch_add(publish_us, Relaxed);
        let tranche_us = drain_us.saturating_add(publish_us);
        self.max_tranche_us.fetch_max(tranche_us, Relaxed);
        if tranche_us >= DRAIN_TRANCHE_SLOW_US {
            self.slow_tranches.fetch_add(1, Relaxed);
        }
        let last = self.last_report_ms.load(Relaxed);
        // First call arms the window; it does not report a duty against a zero
        // origin, which would divide the whole boot's idle time into one tranche.
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        self.last_win_ms.store(win_ms, Relaxed);
        let tranches = self.tranches.swap(0, Relaxed);
        let skipped = self.skipped.swap(0, Relaxed);
        let drain = self.drain_us.swap(0, Relaxed);
        let publish = self.publish_us.swap(0, Relaxed);
        let max = self.max_tranche_us.swap(0, Relaxed);
        let draw = self.draw_us.swap(0, Relaxed);
        let draws = self.draws.swap(0, Relaxed);
        let compute = self.compute_us.swap(0, Relaxed);
        let computes = self.computes.swap(0, Relaxed);
        let flush = self.flush_us.swap(0, Relaxed);
        let flushes = self.flushes.swap(0, Relaxed);
        let max_flush = self.max_flush_us.swap(0, Relaxed);
        let tail = self.tail_us.swap(0, Relaxed);
        let boundary = self.boundary_us.swap(0, Relaxed);
        // Reported in microseconds like every other span on this line, but
        // accumulated in nanoseconds — see the field docs.
        let ring = self.ring_ns.swap(0, Relaxed) / 1000;
        let ring_reads = self.ring_reads.swap(0, Relaxed);
        let decode = self.decode_ns.swap(0, Relaxed) / 1000;
        let packets = self.packets.swap(0, Relaxed);
        let proc_us = self.proc_ns.swap(0, Relaxed) / 1000;
        let regs = self.regs_ns.swap(0, Relaxed) / 1000;
        let regs_ops = self.regs_ops.swap(0, Relaxed);
        let setup = self.setup_ns.swap(0, Relaxed) / 1000;
        let setup_calls = self.setup_calls.swap(0, Relaxed);
        // Emitted beside the total it divides, so `tailrd_us + headwr_us +
        // stamp_us == regs_us` is checkable on the line itself.
        let mut regs_split = String::new();
        for op in RegsOp::ALL {
            let i = op.index();
            let us = self.regs_op_ns[i].swap(0, Relaxed) / 1000;
            let n = self.regs_op_count[i].swap(0, Relaxed);
            let label = op.label();
            regs_split.push_str(&format!(" {label}_us={us} {label}_n={n}"));
        }
        let slow = self.slow_tranches.swap(0, Relaxed);
        // The four buckets that tile `1 - duty`. Emitted beside the total they
        // divide, so `idle + lock + skip + post + busy == win_ms * 1000` is
        // checkable on the line itself — the same rule `regs_split` follows.
        // They will not tile exactly: each is banked at a different instant of
        // the window and the swap below races the worker, so a few hundred
        // microseconds either way is sampling, not a lost bucket.
        let gap_idle = self.gap_idle_us.swap(0, Relaxed);
        let gap_lock = self.gap_lock_us.swap(0, Relaxed);
        let gap_skip = self.gap_skip_us.swap(0, Relaxed);
        let gap_post = self.gap_post_us.swap(0, Relaxed);
        // Emitted beside the total it divides, so `sum(post_*_us) == gap_post_us`
        // is checkable on the line itself.
        let mut post_split = String::new();
        for sweep in PostSweep::ALL {
            let us = self.post_sweep_ns[sweep.index()].swap(0, Relaxed) / 1000;
            post_split.push_str(&format!(" post_{}_us={us}", sweep.label()));
        }
        // Beside the gap they are a candidate cause of, not on a line of their
        // own: the question is only ever "is `gap_idle_us` this?".
        let irq_wait = self.irq_wait_us.swap(0, Relaxed);
        let irq_waits = self.irq_waits.swap(0, Relaxed);
        let irq_wait_max = self.irq_wait_max_us.swap(0, Relaxed);
        let busy = drain.saturating_add(publish);
        let duty = busy as f64 / (win_ms as f64 * 1000.0);
        Some(format!(
            "drain_duty win_ms={win_ms} tranches={tranches} skipped={skipped} busy_us={busy} \
             duty={duty:.3} drain_us={drain} publish_us={publish} max_tranche_us={max} \
             gap_idle_us={gap_idle} gap_lock_us={gap_lock} gap_skip_us={gap_skip} \
             gap_post_us={gap_post}{post_split} \
             irq_wait_us={irq_wait} irq_waits={irq_waits} irq_wait_max_us={irq_wait_max} \
             draw_us={draw} draws={draws} compute_us={compute} computes={computes} \
             flush_us={flush} flushes={flushes} max_flush_us={max_flush} \
             tail_us={tail} boundary_us={boundary} \
             ring_us={ring} ring_reads={ring_reads} decode_us={decode} packets={packets} \
             proc_us={proc_us} regs_us={regs} regs_ops={regs_ops}{regs_split} \
             setup_us={setup} setup_calls={setup_calls} \
             slow_tranches={slow}/{tranches} slow_us={DRAIN_TRANCHE_SLOW_US}"
        ))
    }
}

/// How long the vCPU thread waited for the device lock, measured where it waits.
///
/// Every other figure about tranche length is taken from the side that *holds*
/// the lock, so the step from "the drain held it 38 ms" to "the guest missed a
/// frame" was an inference. This measures the stall from the side that suffers
/// it: the guest's MMIO access is stopped for exactly this long, on the vCPU
/// thread, inside `device_iosfc_read`/`device_iosfc_write`.
///
/// Those two are reached only from `reims_vgpu_qemu_iosfc_read`/`_write`, which
/// only `reims-vgpu-mmio` calls: the PCI device exposes no IOSFC region, so on
/// x86 this census is silent because the path does not exist, not because the
/// guest was never stalled. x86's own mechanism is [`DoorbellCensus`].
///
/// Only the contended path is timed: the uncontended path takes `try_lock` and
/// costs an atomic increment, so a fast access pays nothing for the measurement
/// itself. It does still drive the report, once per [`UNCONTENDED_POLL`]
/// acquisitions — without that, a window with zero waits emits nothing and
/// silence means both "the guest was never blocked" and "no IOSFC traffic
/// arrived". Reading the second as the first is how an instrument talks someone
/// out of a real stall.
#[derive(Default)]
pub(crate) struct VcpuLockCensus {
    waits: std::sync::atomic::AtomicU64,
    wait_us: std::sync::atomic::AtomicU64,
    max_wait_us: std::sync::atomic::AtomicU64,
    /// Waits that cost the guest at least a whole frame interval.
    frame_waits: std::sync::atomic::AtomicU64,
    uncontended: std::sync::atomic::AtomicU64,
    last_report_ms: std::sync::atomic::AtomicU64,
}

/// One in this many uncontended acquisitions reads the clock.
///
/// The uncontended path is the guest's hot MMIO path — hundreds of thousands of
/// acquisitions a second on a driven boot — so it cannot afford an
/// `Instant::now()` each time. It still has to reach the report, or a window
/// with no waits at all stays silent and "the guest was never blocked" is
/// indistinguishable from "no IOSFC traffic reached this device". Those are
/// opposite conclusions and the whole point of the census is to tell them apart.
pub(crate) const UNCONTENDED_POLL: u64 = 1024;

impl VcpuLockCensus {
    /// Count one free acquisition, returning the line when a report is due.
    ///
    /// The clock is read once per [`UNCONTENDED_POLL`] acquisitions, which puts
    /// the report's granularity at that many MMIO accesses rather than at the
    /// exact second boundary. `win_ms` is measured, not assumed, so a window
    /// that closes late reports its true length.
    pub(crate) fn note_uncontended(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .uncontended
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Record one contended wait and return the line when a report is due.
    pub(crate) fn note_wait(&self, us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.waits.fetch_add(1, Relaxed);
        self.wait_us.fetch_add(us, Relaxed);
        self.max_wait_us.fetch_max(us, Relaxed);
        if us >= DRAIN_TRANCHE_SLOW_US {
            self.frame_waits.fetch_add(1, Relaxed);
        }
        self.maybe_report(now_ms)
    }

    /// The window logic both paths share.
    fn maybe_report(&self, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.last_report_ms.load(Relaxed);
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        let waits = self.waits.swap(0, Relaxed);
        let total = self.wait_us.swap(0, Relaxed);
        let max = self.max_wait_us.swap(0, Relaxed);
        let frames = self.frame_waits.swap(0, Relaxed);
        let free = self.uncontended.swap(0, Relaxed);
        Some(format!(
            "vcpu_lock_wait win_ms={win_ms} waits={waits} uncontended={free} \
             wait_us={total} max_wait_us={max} frame_waits={frames} slow_us={DRAIN_TRANCHE_SLOW_US}"
        ))
    }
}

/// How long a guest MMIO doorbell sat queued before the host applied it.
///
/// This is the *other* half of the stall, and on the PCI pathway it is the only
/// half there is. `reims-vgpu-pci` exposes no IOSFC region — only
/// `reims-vgpu-mmio` calls `reims_vgpu_qemu_iosfc_read`/`_write` — so
/// [`VcpuLockCensus`], which instruments `lock_device_for_vcpu`, measures a code
/// path x86 does not have and is silent there by construction rather than by
/// result. Reading that silence as "the drain never stalled the guest" is
/// exactly the mistake it was rebuilt to prevent, so the x86 mechanism gets its
/// own census.
///
/// x86's vCPU never blocks: `device_gfx_write` takes `inner` with `try_lock` and
/// on failure pushes to `gfx_ingress` and returns, so the guest's store retires
/// immediately. The write is then applied by `lock_for_drain`, which takes
/// `inner` with a **blocking** lock and therefore cannot run until the drain
/// worker's current tranche ends. The cost is not a stopped vCPU, it is a
/// doorbell that the guest believes was accepted and whose work does not start
/// for up to a whole tranche — measured at `max_tranche_us` up to 43 ms while
/// `drain_duty` sat at 0.92.
///
/// `direct` counts writes that found the lock free and skipped the queue, and it
/// is load-bearing for the same reason `uncontended` is next door: `queued=0`
/// with a large `direct` is a working doorbell path, while both at zero is no
/// traffic at all.
///
/// # The delay is the tranche, and that is measured rather than inferred
///
/// The paragraph above said the cost was "up to a whole tranche". It is the
/// tranche, to within the measurement's own noise. Three windows of one driven
/// x86/PCI boot, each pairing this census against `drain_duty` at the same `t=`:
///
/// ```text
/// max_tranche_us  42563   42117   105308
/// max_age_us      41711   40627   103619
/// ```
///
/// Two consequences, and the second is the one that redirects the search.
///
/// **The rate is not marginal.** The same windows read `queued=71 direct=69`
/// and `queued=67 direct=74` — about half of the guest's register writes miss
/// the lock — with `age_us/queued` at 28.9 ms mean and `frame_late` 63 of 71.
/// Nine in ten deferred doorbells start their work more than a frame after the
/// guest was told the store retired.
///
/// **Lowering `duty` does not fix it.** The fourth window of the same run read
/// `duty=0.147` with `max_age_us=103619`: the worker was idle for 85 % of that
/// second and still held the guest's next submission for a tenth of it. A
/// doorbell does not wait for the device to be *busy*, it waits for the device
/// to be *holding the lock*, and one long tranche in an otherwise empty second
/// costs exactly as much as a full one. So the flush rail's cost and this stall
/// are separate problems that happen to share a cause, and the fix for one is
/// not the fix for the other.
///
/// What that leaves is the observation that a queued write does not need the
/// drain to stop — it needs its register applied, which costs microseconds and
/// adds work rather than interrupting any. `queued_offsets` is here to say which
/// registers those actually are, because "apply it sooner" is only safe for
/// registers whose effect is to publish more work.
///
/// # It is one register, and it is the one that only publishes work
///
/// `offsets=1` on every window of a driven boot that queued anything, with no
/// exceptions:
///
/// ```text
/// queued=106 direct=434  off_0x1020=106/25246
/// queued=92  direct=426  off_0x1020=92/20589
/// queued=110 direct=436  off_0x1020=110/25296
/// queued=40  direct=291  off_0x1020=40/45487
/// ```
///
/// `0x1020` is [`crate::model::GFX_REG_CHILD_DOORBELL`]. Every other register
/// the guest writes finds the lock free; the entire stall is one doorbell, rung
/// about a hundred times a second and applied up to 45 ms later.
///
/// That is the best case the paragraph above could have hoped for. A doorbell
/// carries no state the decode depends on — its whole effect is to say a child
/// channel has work — so there is nothing about it that has to be ordered
/// against a tranche in flight, and picking it up mid-tranche only lengthens the
/// work list. The two registers already served lock-free
/// (`GFX_REG_INTR_STATUS_DISP` / `_GPU`) are lock-free for exactly this reason.
///
/// Note the shape of the remaining risk, which is not this census's to answer:
/// recording the doorbell sooner is not the same as *acting* on it sooner. A bit
/// set in an atomic while `drain_pending` is midway through its channel loop
/// still waits for the next tranche unless that loop re-reads the mask. Making
/// it re-read is the un-refuted half of the budget experiment — that one
/// returned early and left `child_mask` set with nothing to re-arm it, and froze
/// a boot for 29 s; adding to the mask and continuing has no such gap.
///
/// # The residual is one other register, and it is *not* the same case
///
/// With `0x1020` served lock-free, two driven boots read 25 and 30 deferred
/// writes across their entire runs, against 348 in four windows before. All but
/// one are `0x1008` (`GFX_REG_FIFO_WRITTEN`), at roughly one a window with ages
/// of 3-28 ms; the odd one is `0x1220` once at boot.
///
/// `0x1008` looks like the obvious next application of the same trick, and
/// superficially it is even tidier: `fifo_read` is already an `Arc<AtomicU32>`
/// read lock-free in the other direction, so the producer counter would just be
/// its mirror. It is not tidier, and the difference is not about the register.
///
/// A doorbell only names work. `fifo_written` **bounds a loop**:
/// `drain_main_fifo` runs until `fifo_read` catches it, and today that comparand
/// cannot move mid-tranche because the guest's write needs the device lock. Make
/// it live and the loop follows a producer free to keep writing, so a guest
/// submitting steadily holds the device lock indefinitely — a hang, traded for a
/// delay that is now about one write per ten seconds. The refill above is
/// bounded precisely because it has the same hazard; a live comparand has
/// nowhere to put a bound without snapshotting, and a snapshot is what the
/// current code already is.
///
/// So the remaining 1 % is left where it is, deliberately. If it is ever worth
/// taking, the shape is a snapshot re-taken at a bounded number of points, not a
/// live read.
#[derive(Default)]
pub(crate) struct DoorbellCensus {
    queued: std::sync::atomic::AtomicU64,
    age_us: std::sync::atomic::AtomicU64,
    max_age_us: std::sync::atomic::AtomicU64,
    /// Queued writes whose apply was late by at least a whole frame interval.
    frame_late: std::sync::atomic::AtomicU64,
    direct: std::sync::atomic::AtomicU64,
    /// Rings served without asking for the device lock at all.
    lock_free: std::sync::atomic::AtomicU64,
    /// Which registers are actually being deferred: offset -> (count, max age).
    ///
    /// A lock on a census, which the atomics next door exist to avoid — and it
    /// is only reached on the *queued* path, which this census measures at ~70 a
    /// second against a `direct` path that never touches it. The alternative was
    /// a fixed offset table, which would have to be kept in step with the
    /// register map by hand and would silently drop whatever it did not list.
    queued_offsets: parking_lot::Mutex<std::collections::BTreeMap<u64, (u64, u64)>>,
    last_report_ms: std::sync::atomic::AtomicU64,
}

impl DoorbellCensus {
    /// Count one write applied straight from the vCPU thread, lock uncontended.
    ///
    /// Polled on the same one-in-[`UNCONTENDED_POLL`] rule as the free
    /// acquisitions next door, and for the same reason: this is the hot MMIO
    /// path, and a window with no queueing still has to be able to say so.
    pub(crate) fn note_direct(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .direct
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Count one ring taken with no device lock asked for.
    ///
    /// Polled on the same one-in-[`UNCONTENDED_POLL`] rule as `note_direct`,
    /// and for the same reason: this is the hot MMIO path.
    pub(crate) fn note_lock_free(&self, now_ms: impl FnOnce() -> u64) -> Option<String> {
        let prior = self
            .lock_free
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !prior.is_multiple_of(UNCONTENDED_POLL) {
            return None;
        }
        self.maybe_report(now_ms())
    }

    /// Record the queue age of one applied doorbell, and which register it was.
    pub(crate) fn note_queued(&self, offset: u64, age_us: u64, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        self.queued.fetch_add(1, Relaxed);
        self.age_us.fetch_add(age_us, Relaxed);
        self.max_age_us.fetch_max(age_us, Relaxed);
        if age_us >= DRAIN_TRANCHE_SLOW_US {
            self.frame_late.fetch_add(1, Relaxed);
        }
        {
            let mut by_offset = self.queued_offsets.lock();
            let slot = by_offset.entry(offset).or_insert((0, 0));
            slot.0 += 1;
            slot.1 = slot.1.max(age_us);
        }
        self.maybe_report(now_ms)
    }

    fn maybe_report(&self, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let last = self.last_report_ms.load(Relaxed);
        if last == 0 {
            self.last_report_ms.store(now_ms, Relaxed);
            return None;
        }
        let win_ms = now_ms.saturating_sub(last);
        if win_ms < DRAIN_DUTY_REPORT_MS {
            return None;
        }
        self.last_report_ms.store(now_ms, Relaxed);
        let queued = self.queued.swap(0, Relaxed);
        let total = self.age_us.swap(0, Relaxed);
        let max = self.max_age_us.swap(0, Relaxed);
        let late = self.frame_late.swap(0, Relaxed);
        let direct = self.direct.swap(0, Relaxed);
        let lockfree = self.lock_free.swap(0, Relaxed);
        // Descending by count, capped, and the cap is reported rather than
        // silently applied: a register that misses the list because three others
        // out-counted it must not read as a register that never deferred.
        let mut offsets: Vec<(u64, (u64, u64))> = std::mem::take(&mut *self.queued_offsets.lock())
            .into_iter()
            .collect();
        offsets.sort_by_key(|(off, (count, _))| (std::cmp::Reverse(*count), *off));
        let distinct = offsets.len();
        let mut body = String::new();
        for (off, (count, max_us)) in offsets.iter().take(DOORBELL_OFFSETS_REPORTED_MAX) {
            body.push_str(&format!(" off_{off:#x}={count}/{max_us}"));
        }
        Some(format!(
            "gfx_doorbell_delay win_ms={win_ms} queued={queued} direct={direct} \
             lockfree={lockfree} age_us={total} max_age_us={max} frame_late={late} \
             slow_us={DRAIN_TRANCHE_SLOW_US} offsets={distinct} shown={}{body}",
            distinct.min(DOORBELL_OFFSETS_REPORTED_MAX)
        ))
    }
}

/// How many deferred register offsets `gfx_doorbell_delay` names per window.
///
/// The line has to stay one line, and the question it answers — "which
/// registers are being held back" — is answered by the head of the
/// distribution: a register deferring twice a second is not what costs a frame.
/// `offsets=` states how many distinct ones there were, so a truncated tail is
/// visible rather than implied.
///
/// The `_MAX` says this is a cut and not a size: the walk stops here whether or
/// not the guest's data has run out, so a reader who wants the true count reads
/// `offsets=` and not the length of what was printed.
const DOORBELL_OFFSETS_REPORTED_MAX: usize = 4;

static DOORBELL: std::sync::LazyLock<DoorbellCensus> =
    std::sync::LazyLock::new(DoorbellCensus::default);

/// Count one child doorbell taken on the vCPU thread with no device lock at all.
///
/// Distinct from [`note_doorbell_direct`], which counts a write that *took* the
/// lock and found it free. This one never asks, so it can neither queue nor
/// contend — and the pair is what says so: `lockfree` rising while `queued`
/// falls to zero is the register leaving the contended path, whereas `queued`
/// staying up would mean something is still routing it through `gfx_ingress`.
pub fn note_doorbell_lock_free() {
    if let Some(line) = DOORBELL.note_lock_free(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Count one doorbell applied on the vCPU thread without queueing.
pub fn note_doorbell_direct() {
    if let Some(line) = DOORBELL.note_direct(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Record how long one doorbell sat in `gfx_ingress` before being applied.
pub fn note_doorbell_queued(offset: u64, age_us: u64) {
    if let Some(line) = DOORBELL.note_queued(offset, age_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

static VCPU_LOCK: std::sync::LazyLock<VcpuLockCensus> =
    std::sync::LazyLock::new(VcpuLockCensus::default);

/// Count one uncontended device-lock acquisition from the vCPU thread.
///
/// Emits the same one-line census as the wait path, so a window that saw
/// traffic but never blocked still says so.
pub fn note_vcpu_lock_free() {
    if let Some(line) = VCPU_LOCK.note_uncontended(|| crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

/// Record one contended device-lock wait from the vCPU thread; emits at most
/// once per second.
pub fn note_vcpu_lock_wait(us: u64) {
    if let Some(line) = VCPU_LOCK.note_wait(us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
    }
}

static DRAIN_DUTY: std::sync::LazyLock<DrainDutyCensus> =
    std::sync::LazyLock::new(DrainDutyCensus::default);

static RESIDENT_ARM: std::sync::LazyLock<ResidentArmCensus> =
    std::sync::LazyLock::new(ResidentArmCensus::default);

pub(crate) static SURFACE_WRITE: std::sync::LazyLock<SurfaceWriteCensus> =
    std::sync::LazyLock::new(SurfaceWriteCensus::default);

static WINDOW_PUBLISH: std::sync::LazyLock<WindowPublishCensus> =
    std::sync::LazyLock::new(WindowPublishCensus::default);

/// Record how one tranche's host-window publish attempt ended.
pub fn note_window_publish(arm: WindowPublish) {
    WINDOW_PUBLISH.note(arm);
}

/// Attribute `us` of one surface writeback to one of its whole-frame passes.
pub fn note_surface_write_phase(phase: SurfaceWritePhase, us: u64) {
    SURFACE_WRITE.note(phase, us);
}

/// Record which landing path one surface writeback took, and how many bytes of
/// frame it carried.
pub fn note_surface_write_path(contiguous: bool, bytes: u64) {
    SURFACE_WRITE.note_path(contiguous, bytes);
}

/// Stamp one resident-backed render window as armed.
pub fn note_resident_window_armed() {
    RESIDENT_ARM.note_arm(crate::observe::elapsed_us());
}

/// Record that a flush reached a resident-backed window's readback.
pub fn note_resident_window_flushed() {
    RESIDENT_ARM.note_flush(crate::observe::elapsed_us());
}

/// When the drain tranche now running started, in [`crate::observe::elapsed_us`].
///
/// One word, written once per tranche by the worker that owns it. A device runs
/// one drain worker, so there is no interleaving to lose; a second device would
/// share this and the two would blend, which is why nothing here claims to be
/// per device.
static TRANCHE_START_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Mark the start of a drain tranche, for [`tranche_elapsed_us`].
pub fn note_tranche_started(now_us: u64) {
    TRANCHE_START_US.store(now_us, std::sync::atomic::Ordering::Relaxed);
}

/// How long the tranche now running has been running.
///
/// The question this exists for: a guest clears an object-list slot by writing
/// its own memory, which nothing orders against the ring except how fast this
/// device reads it — so a lookup that finds a cleared slot should be one that
/// happened *late* in a long tranche. Reading zero before the first tranche is
/// harmless; nothing consults it outside one.
pub fn tranche_elapsed_us() -> u64 {
    crate::observe::elapsed_us()
        .saturating_sub(TRANCHE_START_US.load(std::sync::atomic::Ordering::Relaxed))
}

/// Band one object-list lookup by how late in its tranche it happened.
///
/// Bands rather than a mean, because the claim being tested is about a **tail**:
/// "the losing lookups sit behind the long tranches" is false if they are spread
/// like every other lookup, and two means that differ by a little cannot tell
/// those apart.
///
/// `hit` picks the family, and the hit family is the point. A miss banding that
/// skewed late would prove nothing on its own if *every* lookup skews late — the
/// control is the whole reason this is worth recording, and the session that
/// added it had already been caught once reading an instrument without one.
pub fn note_list_lookup_age(hit: bool, us: u64) {
    note_store_route(list_lookup_age_route(hit, us));
}

/// The counter name for one banded lookup, total over both families and every
/// age.
fn list_lookup_age_route(hit: bool, us: u64) -> &'static str {
    match (hit, us) {
        (true, 0..=99) => "list_hit_age_under_100us",
        (true, 100..=999) => "list_hit_age_under_1ms",
        (true, 1_000..=9_999) => "list_hit_age_under_10ms",
        (true, 10_000..=99_999) => "list_hit_age_under_100ms",
        (true, _) => "list_hit_age_over_100ms",
        (false, 0..=99) => "list_miss_age_under_100us",
        (false, 100..=999) => "list_miss_age_under_1ms",
        (false, 1_000..=9_999) => "list_miss_age_under_10ms",
        (false, 10_000..=99_999) => "list_miss_age_under_100ms",
        (false, _) => "list_miss_age_over_100ms",
    }
}

/// Attribute the tranche tail: the deferred-batch submit and the present
/// boundary, both inside `drain_us` and inside no [`DrainPhase`].
pub fn note_drain_tail(tail_us: u64, boundary_us: u64) {
    DRAIN_DUTY.note_tail(tail_us, boundary_us);
}

/// Attribute one ring snapshot read, in nanoseconds.
pub fn note_drain_ring(ns: u64) {
    DRAIN_DUTY.note_ring(ns);
}

/// Attribute one packet decode, in nanoseconds.
pub fn note_drain_decode(ns: u64) {
    DRAIN_DUTY.note_decode(ns);
}

/// Attribute one `process_child_packet` dispatch, in nanoseconds, by opcode.
pub fn note_drain_proc(opcode: u16, ns: u64) {
    DRAIN_DUTY.note_proc(opcode, ns);
}

/// Attribute one span inside `process_exec_indirect2`, in nanoseconds.
pub fn note_exec_phase(phase: ExecPhase, ns: u64) {
    DRAIN_DUTY.note_exec(phase, ns);
}

/// Attribute one span inside `finish_stream`, in nanoseconds.
///
/// The caller is [`crate::runtime::exec::finish_phase::FinishTimer`], which
/// accumulates a whole stream's spans locally and flushes them here once, so a
/// packet of a hundred draws costs twelve atomics rather than twelve hundred.
pub fn note_finish_phase(phase: FinishPhase, ns: u64, entries: u64) {
    DRAIN_DUTY.note_finish(phase, ns, entries);
}

/// Attribute one span inside opening a submission, in nanoseconds.
///
/// `descs` is the submission's resource-descriptor count and is banked once,
/// from [`OpenPart::Table`], so the three parts do not triple it.
pub fn note_open_part(part: OpenPart, ns: u64, descs: u64) {
    DRAIN_DUTY.note_open(part, ns, descs);
}

/// Attribute one span inside the translation preflight, in nanoseconds.
pub fn note_preflight_part(part: PreflightPart, ns: u64) {
    DRAIN_DUTY.note_preflight(part, ns);
}

/// Count one distinct pipeline ref the preflight scan resolved.
pub fn note_preflight_pipe() {
    DRAIN_DUTY.note_preflight_pipe();
}

/// Attribute one access around a packet, in nanoseconds, by which one it was.
pub fn note_drain_regs(op: RegsOp, ns: u64) {
    DRAIN_DUTY.note_regs(op, ns);
}

/// Attribute one `drain_child_fifo` prologue, in nanoseconds.
pub fn note_drain_setup(ns: u64) {
    DRAIN_DUTY.note_setup(ns);
}

/// Accumulate one completed drain tranche; emits at most once per second.
pub fn note_drain_tranche(
    executor: &dyn crate::runtime::executor::Executor,
    drain_us: u64,
    publish_us: u64,
) {
    if let Some(line) = DRAIN_DUTY.note(drain_us, publish_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
        // Immediately after `drain_duty`, so the two read as one record: the
        // rails must sum to its `flush_us` and their counts to its `flushes`.
        if let Some(rails) = DRAIN_DUTY.take_flush_rails() {
            crate::observe::off(rails);
        }
        // Also immediately after `drain_duty`, and read the same way: the
        // per-opcode `_us` fields sum to its `proc_us` and the `_n` fields to
        // its `packets`, less `op_overflow`.
        if let Some(ops) = DRAIN_DUTY.take_proc_ops() {
            crate::observe::off(ops);
        }
        // Under `drain_ops`, dividing its `op0x37_us` the way `chain_phase`
        // divides `draw_us`.
        if let Some(exec) = DRAIN_DUTY.take_exec_phases() {
            crate::observe::off(exec);
        }
        // Under `exec_phase`, dividing its `preflight_us`.
        if let Some(open) = DRAIN_DUTY.take_open_parts() {
            crate::observe::off(open);
        }
        // The alias walk's own totals, owned by the semantic core because the
        // walk is core's. `iters_per_walk` is the reading that matters: how many
        // storage nodes one guest write examines to find the ranges overlapping
        // it. A number near the device's whole storage population says the
        // overlap search is a linear scan being paid per write.
        {
            let (walks, visited, scan_iters) = reims_vgpu_core::resource::alias_walk_census::take();
            if walks != 0 {
                crate::observe::off(format!(
                    "alias_walk walks={walks} visited={visited} scan_iters={scan_iters} \
                     visited_per_walk={:.2} iters_per_walk={:.1}",
                    visited as f64 / walks as f64,
                    scan_iters as f64 / walks as f64,
                ));
            }
        }
        // What the guest's own stream looks like: decoded render records
        // against the draws among them. The stream is a delta and the draw path
        // resolves the whole accumulated state per draw, so this ratio is the
        // size of what a resolve-on-write design would stop redoing.
        {
            let (records, draws) = crate::runtime::exec::stream_shape_census::take();
            if draws != 0 {
                crate::observe::off(format!(
                    "stream_shape records={records} draws={draws} records_per_draw={:.2}",
                    records as f64 / draws as f64,
                ));
            }
        }
        // The per-draw visibility merge's own totals, owned by the Vulkan crate
        // because the ledger is its. The reading that matters is
        // `skipped_per_ask` against `walked_per_ask`: the merge is linear in
        // pages *plus* the ledger runs its cursor has to reach past, and every
        // set restarts that cursor, so the second term grows with what the
        // guest has outstanding rather than with what this draw reads. It is a
        // distance and not a cost — the seek bisects it — but it is the number
        // that says how much reach the seek needs, and a merge that went back
        // to stepping would pay all of it.
        {
            let (
                asks,
                sets,
                given,
                walked,
                runs,
                span_misses,
                runs_skipped,
                rebuilds,
                rebuild_pages,
                rebuild_ns,
            ) = reims_vgpu_vulkan::engine::vis_walk_census::take();
            if asks != 0 && sets != 0 {
                crate::observe::off(format!(
                    "vis_walk asks={asks} sets={sets} given={given} walked={walked} \
                     runs_skipped={runs_skipped} runs_per_ask={:.1} sets_per_ask={:.2} \
                     given_per_set={:.1} walked_per_ask={:.1} skipped_per_ask={:.1} \
                     span_miss_frac={:.3} rebuilds={rebuilds} rebuild_pages={rebuild_pages} \
                     rebuild_us={} rebuild_us_per_ask={:.3}",
                    runs as f64 / asks as f64,
                    sets as f64 / asks as f64,
                    given as f64 / sets as f64,
                    walked as f64 / asks as f64,
                    runs_skipped as f64 / asks as f64,
                    span_misses as f64 / sets as f64,
                    rebuild_ns / 1000,
                    rebuild_ns as f64 / 1000.0 / asks as f64,
                ));
            }
        }
        if let Some(pre) = DRAIN_DUTY.take_preflight_parts() {
            crate::observe::off(pre);
        }
        // Also under `exec_phase`, dividing its `finish_us` — the phase that
        // holds `draw_us` and 1.25 µs a draw of unnamed work around it.
        if let Some(fin) = DRAIN_DUTY.take_finish_phases() {
            crate::observe::off(fin);
        }
        // Under `flush_rails`, dividing its `render_us`.
        if let Some(split) = DRAIN_DUTY.take_readback_split() {
            crate::observe::off(split);
        }
        // Beside `readback_split`, because it is only readable against it: the
        // question is whether `age_us/aged` leaves room for `fence_us/fence`.
        if let Some(age) = RESIDENT_ARM.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(age);
        }
        // Under `readback_split`, dividing its `write_us` the same way it
        // divides `flush_rails`'s `render_us`.
        if let Some(write) = SURFACE_WRITE.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(write);
        }
        // The offer side of `host_window_cadence`, which can only see the
        // frames that reached it.
        if let Some(publish) = WINDOW_PUBLISH.take(DRAIN_DUTY.last_window_ms()) {
            crate::observe::off(publish);
        }
        // Under `window_publish`, which says how many frames were offered but
        // not why fewer reached the screen.
        emit_engine_lock(executor, DRAIN_DUTY.last_window_ms());
        if let Some(routes) = take_store_routes() {
            crate::observe::off(routes);
        }
        // The width any packet-level fan-out could use. Joined by `t=` so it
        // is read against the same window's draws and duty.
        for tranche in take_drain_tranche(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(tranche);
        }
        // The one genuine per-draw write on the resolve side, and therefore
        // where a packet-parallel encoder's threads would meet.
        if let Some(ledger) = reims_vgpu_core::content_tracking::host_write_census::take(
            crate::observe::elapsed_ms() as u64,
        ) {
            crate::observe::off(ledger);
        }
        // What canonical page-set construction costs on the draw path. Joined
        // by `t=` like the rest, so `builds` divides by this window's draws.
        if let Some(sets) =
            reims_vgpu_memory::page_set_census::take(crate::observe::elapsed_ms() as u64)
        {
            crate::observe::off(sets);
        }
        // Beside `store_routes` deliberately: the two are read against each
        // other. `surface_backing_fail` lines equal `surface_backing_recovered +
        // surface_backing_superseded` from that line plus this one's `n`, and a
        // refusal that never recovered is only visible as the residue.
        if let Some(outstanding) = crate::runtime::objects::surface_backing_outstanding_census() {
            crate::observe::off(outstanding);
        }
        // The same reason and the same place: `store_routes` counts the watches
        // that *ended*, and a slot still waiting is skipped by every sweep it
        // survives, so without this line the misses and the verdicts do not
        // reconcile and the difference reads as lost records.
        if let Some(watching) = crate::runtime::objects::slot_recheck::outstanding_census() {
            crate::observe::off(watching);
        }
        // Onto the census cadence rather than a timer of its own, so a reader
        // pairing the footprint against `store_routes` is reading one clock.
        // The run dump rate-limits itself; this is the only caller.
        for line in crate::observe::footprint::census_lines(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
        // Beside the engine counters it has to be read against: the eviction
        // routes say which cap fired and this says how much the workload wanted,
        // and neither is interpretable without the other.
        if let Some(wanted) = executor.sampled_working_set_census() {
            crate::observe::off(wanted);
        }
        // The same question one rail over, and the one with no cache behind it
        // yet: `buffer_guest_gathers` says how many gathers ran and this says
        // how few distinct windows they were.
        if let Some(wanted) = executor.buffer_gather_working_set_census() {
            crate::observe::off(wanted);
        }
        emit_engine_delta(executor);
        // After `emit_engine_delta`, which emits `draw_phase`: the two divide
        // against each other and reading them in the other order invites
        // treating the engine's phases as the whole draw, which is the
        // misreading this line exists to correct. Not gated on the backend —
        // second census.
        emit_chain_phase();
        emit_object_cache_levels(executor);
        emit_guest_import_levels(executor);
    }
}

/// How many RAMBlocks this device has imported, and how many bytes they cover,
/// as **levels** rather than per-window deltas.
///
/// This is the reading that says whether the one-import-per-RAMBlock model held.
/// The count should be one or two for a whole boot and flat across every window;
/// a count that tracks the workload is the per-resource import the model exists
/// to avoid, which `VK_EXT_external_memory_host` does not guarantee works twice
/// over one allocation and which would pay the driver's page pinning thousands
/// of times a second for an answer that never changes.
///
/// Flat is therefore the healthy reading and a rise is the alarm — the opposite
/// polarity to most lines here, which is why the count is emitted every window
/// rather than once at import time. A single line at import time could not
/// distinguish "imported once" from "imported once per window".
///
/// # Both terms, because the numerator alone is ambiguous
///
/// A backend imports a span at its **first reference**, not at device init, so
/// the imported count is bounded above by the number of spans the shim reported
/// and starts below it. `ramblocks=1` alone cannot distinguish "this machine has
/// one RAMBlock and it is imported" from "this machine has two and the workload
/// has only ever touched one" — and the second is a workload fact, not a defect.
/// The denominator comes from [`crate::runtime::guest_ram_map::span_census`],
/// which is the shim's answer, so the pair reads `imported/reported`.
///
/// This is not hypothetical on the x86 pathway. `vm/boot-x86.sh` boots `-m 16G`
/// and a driven Safari boot measures `ramblocks=1/4 mib=14336/16399`: the shim
/// reports four writable spans and the workload imports one, the 14 GiB half of
/// `-m` above the PCI hole. The numerator alone reads as 2 GiB of guest RAM
/// having gone missing against `-m 16G`, which is how it was first misread.
///
/// The reported set is larger than `-m` — 16399 MiB against 16384 — because the
/// shim walks the flat view rather than `-m`. `guest_ram_span` names each span
/// at build time; on this boot they are:
///
/// ```text
/// n=0/4 gpa=0x0          len=786432      (768 KiB, below the legacy VGA hole)
/// n=1/4 gpa=0x100000     len=2146435072  (2047 MiB, 1 MiB up to the PCI hole)
/// n=2/4 gpa=0x80000000   len=16777216    (16 MiB — this device's own BAR1)
/// n=3/4 gpa=0x100000000  len=15032385536 (14336 MiB, above 4 GiB — imported)
/// ```
///
/// Spans 0, 1 and 3 are the two halves of `-m 16G` either side of the PCI hole,
/// with the low half split again by the legacy hole at `0xA0000`. Span 2 is the
/// 15 MiB of "extra": it is `REIMS_VGPU_PCI_FB_SIZE`, the linear GOP framebuffer
/// `reims-vgpu-pci.c` registers as BAR1 with `memory_region_init_ram`, assigned
/// into the PCI hole at 2 GiB. A plain RAM BAR is not ROM, not ROMD, not a
/// `ram_device` and not readonly, so it passes the shim's filter — that filter
/// screens out memory the guest cannot store into, and the guest *can* store
/// into a GOP framebuffer, which is what a GOP framebuffer is for.
///
/// It is reported and never imported: only span 3 has ever been referenced. The
/// consequence worth knowing is that a GPA landing inside BAR1 would resolve
/// rather than earning `GpaNotInAnyImport`, so it is bounded to this device's
/// own framebuffer rather than refused. That is the host console's bytes, not
/// another RAMBlock's and not this process's private state, and the guest
/// already writes them through the BAR. Narrowing the filter is **not** an
/// obvious improvement: the EFI console path exists precisely because the guest
/// points at BAR1, so excluding it would need evidence that no legitimate
/// reference lands there. Nothing has measured that.
///
/// `mib` is the same level and is not a rate: it is guest RAM the device can
/// currently reach, against what the machine reported.
fn emit_guest_import_levels(executor: &dyn crate::runtime::executor::Executor) {
    let (bytes, count, aliases, guest_images, live_guest_images, recycled_images) =
        executor.guest_import_census();
    let (spans, span_bytes) = crate::runtime::guest_ram_map::span_census();
    // An engine that never imported emits nothing, so a host on a negative
    // `host_pointer` rung — or a boot before the first guest window — costs no
    // line, and a zero here always means the copying rails rather than silence.
    if count == 0 && aliases == 0 {
        return;
    }
    crate::observe::off(format!(
        "guest_import_levels (levels, not per-interval) ramblocks={count}/{spans} aliases={aliases} \
         guest_images={guest_images}/{live_guest_images}_live recycled_images={recycled_images} \
         imported_mib={} ramblock_reported_mib={} (RAMBlock spans import lazily; \
         packed aliases add to imported_mib without changing the reported RAM size)",
        bytes / (1024 * 1024),
        span_bytes / (1024 * 1024),
    ));
}

/// Live entry counts of the caches that hold one entry per distinct guest
/// object, as **levels** rather than per-window deltas.
///
/// These caches carry no capacity and no replacement rule. The argument for that
/// is that each key is a content digest or a complete descriptor of guest state,
/// so the count is the guest's own distinct object set and settles once the
/// guest has finished compiling. That is a claim about a running guest, and this
/// line is what can falsify it: a level still climbing minutes into a boot means
/// some key is carrying per-frame state, and the argument is wrong for that
/// cache. A settling level is the argument holding.
///
/// `m2v` counts translated shaders (`reims_vgpu_vulkan::m2v_cache`); the rest are the
/// Vulkan engine's immutable-object caches.
fn emit_object_cache_levels(executor: &dyn crate::runtime::executor::Executor) {
    let [shaders, layouts, passes, pipelines, samplers, compute_pipelines] =
        executor.object_cache_levels();
    let m2v = executor.shader_translation_cache_level();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) m2v={m2v} shaders={shaders} \
         layouts={layouts} passes={passes} pipelines={pipelines} samplers={samplers} \
         compute_pipelines={compute_pipelines}"
    ));
}

fn emit_engine_delta(executor: &dyn crate::runtime::executor::Executor) {
    use crate::runtime::executor::CounterSnapshot;
    static PREV: std::sync::Mutex<Option<CounterSnapshot>> = std::sync::Mutex::new(None);
    let now = executor.counter_snapshot();
    let Ok(mut prev) = PREV.lock() else {
        return;
    };
    let d = now.delta_since(&prev.unwrap_or_default());
    *prev = Some(now);
    // Generated from the counter vocabulary rather than named here, so this line
    // cannot fall behind it again; see `CounterSnapshot::delta_fields`.
    let mut line = String::from("engine_delta");
    for (name, value) in d.delta_fields() {
        use std::fmt::Write as _;
        let _ = write!(line, " {name}={value}");
    }
    crate::observe::off(line);
    emit_registry_pressure(&now);
    emit_draw_phase(executor);
}

/// How far the resident registries reached, and what the populations that
/// cannot be given back cost.
///
/// Separate from `engine_delta` because these fields are read **absolute**, and
/// that line reports differences. A high-water mark deltas to nonsense — the
/// difference between two peaks is not a peak, and reads as zero for the rest of
/// the boot once the true maximum is behind the window — so it is taken from the
/// snapshot rather than from `delta_since`.
///
/// `peak` has no `cap` beside it any more, and that is the point: the
/// resident-target population is bounded by the allocator refusing rather than by
/// a slot count (see `ResourcePools::recoverable_residents`). Read `peak` against
/// `peak_mib` — the pair is what says whether a count was ever a proxy for VRAM,
/// and it answered no: 194 slots against 211 MiB on one workload and 41 against
/// 74 MiB on another, a 1.65x spread in MiB per slot between two ordinary
/// desktop workloads.
///
/// `sole_copy` is the half of that population the allocation-failure retry
/// cannot hand back. Its ratio against `peak` is the reading that matters now:
/// near 1 means a retry would find nothing to give, and the copy-out sites are
/// what needs work.
///
/// # Why `resident_samples` is on this line
///
/// It is the *denominator* of `sampled_resident_missing`, which is raised from
/// one place — the `SampledSource::Target` arm of the engine's sampled loop,
/// also the sole increment of `sampled_gpu_binds`. When it is zero, no draw
/// bound a resident as a texture, nothing could have observed a destroyed one,
/// and a zero missing-count is a null instrument rather than a pass.
///
/// This field exists because that denominator was once argued about from two
/// boots that had never been compared. The since-retired slot cap was driven six
/// times over its bound and reported `evicts=1591` against
/// `sampled_resident_missing=0`; a later reading of `sampled_gpu_binds=0` —
/// taken on a *different* workload — was used to call that pair a null
/// instrument. Printing the denominator beside the pair settled it in one boot:
/// `web-content-probe --churn 1` reports `resident_samples=11742`, so the arm
/// does run, the zero was a real measurement, and the null-instrument objection
/// was itself the unfounded claim. The reading still matters: it is what says a
/// draw would have noticed had anything gone missing.
///
/// `cs_sole_copy` is the same protected-population reading over the
/// compute-storage registry. It is worth reading separately rather than summed:
/// that registry holds standalone `VkDeviceMemory` where the target registry
/// holds slab suballocations, so the two say different things about what an
/// allocation failure would have found to give back.
///
/// Neither registry publishes an eviction count any more, because neither has a
/// slot count to evict for. `vram_reclaim_retry` and
/// `vram_compute_storage_reclaim_retry` on the fail channel are what report a
/// reclaim now, and they fire only when an allocation was actually refused.
fn emit_registry_pressure(now: &crate::runtime::executor::CounterSnapshot) {
    crate::observe::off(format!(
        "registry_pressure (levels, not per-interval) peak={} peak_mib={} \
         resident_samples={} resample_peak_ms={}/{} \
         slab_mib={}/{} sole_copy={}/{}mib cs_sole_copy={}/{}mib",
        now.registry_non_pinned_peak,
        now.registry_non_pinned_peak_bytes >> 20,
        now.sampled_gpu_binds,
        now.resident_resample_peak_ms,
        crate::runtime::executor::IDLE_MAINTENANCE_START_MS,
        now.slab_carved_bytes >> 20,
        now.slab_held_bytes >> 20,
        now.registry_sole_copy_peak,
        now.registry_sole_copy_peak_bytes >> 20,
        now.compute_storage_sole_copy_peak,
        now.compute_storage_sole_copy_peak_bytes >> 20,
    ));
}

/// The split of `drain_duty`'s `draw_us` that actually covers it, over the same
/// window.
///
/// `draw_phase` divides the engine and `chain_phase` divides everything around
/// it, so this line is emitted immediately after that one and the two are read
/// together: `chain_phase`'s `engine_us` must equal `draw_phase`'s phases
/// summed, and `chain_phase`'s eight must equal `drain_duty`'s `draw_us`.
/// Whatever `draw_phase` does not account for is the other seven bars here, and
/// on the boot that motivated this line that was 82% of the draw.
///
/// Silent when no chain ran, so an idle desktop costs nothing.
///
/// The split of `chain_phase`'s `binds_us`, over the same window.
///
/// Emitted immediately after it, in the same relationship `draw_phase` has to
/// `engine_us`: divide the three against the column above. They are not claimed
/// to sum to it — see [`crate::runtime::bind_phase`] for why a computed
/// remainder was left out.
fn emit_bind_phase() {
    let Some(w) = crate::runtime::bind_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "bind_phase binds={} vertex_us={} fragment_us={} attrs_us={} \
         acc_unused={} acc_deref={} acc_undecl={} acc_n={} acc_unused_staged={}",
        w.binds,
        w.vertex_us,
        w.fragment_us,
        w.attrs_us,
        w.access_unused,
        w.access_dereferenced,
        w.access_undeclared,
        // The three classes partition the buffer binds resolved in the window,
        // so this is their sum and not a separately-counted total: a reader who
        // divides gets an identity that holds or a bug that shows.
        w.access_total(),
        w.access_unused_staged,
    ));
}

fn emit_chain_phase() {
    let Some(w) = crate::runtime::chain_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "chain_phase chains={} prep_us={} pipeline_us={} pl_gen_us={} pl_desc_us={} \
         pl_mtlb_us={} pl_air_us={} pl_xlate_us={} binds_us={} sampled_us={} \
         seed_us={} assemble_us={} engine_us={} store_us={} prep_seed_us={} \
         prep_pages_us={} asm_target_us={} asm_depth_us={} asm_trail_us={} max_us={}",
        w.chains,
        w.prep_us,
        w.pipeline_us,
        w.pipeline_gen_us,
        w.pipeline_desc_us,
        w.pipeline_mtlb_us,
        w.pipeline_air_us,
        w.pipeline_xlate_us,
        w.binds_us,
        w.sampled_us,
        w.seed_us,
        w.assemble_us,
        w.engine_us,
        w.store_us,
        w.prep_seed_us,
        w.prep_pages_us,
        w.assemble_target_us,
        w.assemble_depth_us,
        w.assemble_trail_us,
        w.max_us,
    ));
    // Under `chain_phase`, dividing its largest column the same way
    // `draw_phase` divides its `engine_us`.
    emit_bind_phase();
    emit_sampled_phase();
}

/// The split of `chain_phase`'s `sampled_us`, over the same window.
///
/// Emitted immediately after `bind_phase`, in the same relationship both have to
/// the column above them. `sampled_us` is what was left once `binds_us` had
/// `bind_phase` and `engine_us` had `draw_phase`. The five are not claimed to
/// sum to it — see [`crate::runtime::sampled_phase`] for what they deliberately
/// leave out and why a computed remainder is worse than none.
fn emit_sampled_phase() {
    let Some(w) = crate::runtime::sampled_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "sampled_phase sampled={} lookup_us={} alias_us={} resolve_us={} samplers_us={} \
         reflect_us={} linear_packed_us={} linear_admission_us={} gather_witness_us={}",
        w.sampled,
        w.lookup_us,
        w.alias_us,
        w.resolve_us,
        w.samplers_us,
        w.reflect_us,
        w.linear_packed_us,
        w.linear_admission_us,
        w.gather_witness_us,
    ));
}

/// The split of `drain_duty`'s `draw_us`, over the same window.
///
/// `drain_duty` says a saturated second is 93-99% `draw_us` and `engine_delta`
/// says ~450 MB/s crosses the bus each way. Those two are consistent with
/// opposite fixes — moving fewer bytes, or stopping the per-draw GPU round trip
/// — and neither line can tell them apart. This one can: `readback_us` and the
/// staging half of `setup_us` scale with bytes, `wait_us` does not.
///
/// Silent when no draw ran, so an idle desktop costs nothing.
fn emit_draw_phase(executor: &dyn crate::runtime::executor::Executor) {
    let Some(w) = executor.draw_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "draw_phase draws={} prep_us={} slot_us={} pipeline_us={} \
         pl_depth_us={} pl_shader_us={} pl_layoutpass_us={} pl_compile_us={} pl_sampler_us={} \
         stage_us={} sg_roles_us={} sg_vertex_us={} sg_index_us={} sg_storage_us={} \
         sg_seed_us={} stage_pass_us={} \
         acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
         descriptors_us={} \
         record_us={} rec_begin_us={} rec_barrier_us={} \
         rb_imported_test_us={} rb_read_set_us={} rb_visibility_us={} rb_pass_break_us={} \
         rb_snapshot_us={} rb_seed_us={} rb_materialize_us={} \
         rb_resident_us={} rb_upload_us={} rb_attachment_us={} \
         rec_pass_us={} rec_state_us={} \
         rec_draw_us={} submit_us={} post_target_us={} post_store_us={} post_sampled_us={} \
         post_park_us={} wait_us={} readback_us={} max_us={} stalls={}",
        w.draws,
        w.prep_us,
        w.slot_us,
        w.pipeline_us,
        w.pipeline_depth_us,
        w.pipeline_shader_us,
        w.pipeline_layout_pass_us,
        w.pipeline_compile_us,
        w.pipeline_sampler_us,
        w.stage_us,
        w.stage_roles_us,
        w.stage_vertex_us,
        w.stage_index_us,
        w.stage_storage_us,
        w.stage_seed_us,
        w.stage_pass_us,
        w.acquire_us,
        w.acquire_sampled_us,
        w.sampled_upload_us,
        w.acquire_readback_us,
        w.descriptors_us,
        w.record_us,
        w.rec_begin_us,
        w.rec_barrier_us,
        w.rb_imported_test_us,
        w.rb_read_set_us,
        w.rb_visibility_us,
        w.rb_pass_break_us,
        w.rb_snapshot_us,
        w.rb_seed_us,
        w.rb_materialize_us,
        w.rb_resident_us,
        w.rb_upload_us,
        w.rb_attachment_us,
        w.rec_pass_us,
        w.rec_state_us,
        w.rec_draw_us,
        w.submit_us,
        w.post_target_us,
        w.post_store_us,
        w.post_sampled_us,
        w.post_park_us,
        w.wait_us,
        w.readback_us,
        w.max_us,
        w.stalls,
    ));
    emit_stage_phase(executor);
    emit_gather_phase(executor);
    emit_gpu_span(executor);
}

/// Beside `draw_phase`, because it is the one column in it the GPU wrote.
///
/// `slot_us` above is the drain worker blocked on a ring fence, and every session
/// before this one read that as "the GPU is busy" without a GPU-side number
/// existing. `busy_us` is that number: GPU microseconds summed over the
/// submissions retired this window, from timestamps each command buffer wrote
/// into itself.
///
/// Read the pair and never `busy_us` alone. Against a census second it is
/// utilisation; against `slot_us` it is how much of the worker's wait was this
/// device's own recorded work rather than queue latency, and those are two
/// different questions with two different fixes. `armed`/`sealed`/`read` say
/// whether a low reading is a quiet GPU or a probe that did not close.
///
/// The five `*_us`/`*_n` pairs tile `busy_us`/`read` by what the submission was
/// recorded for, so the shares say which rail owns the device's GPU time without
/// an ablation. `unattributed` is the identity that keeps that honest: it is
/// `read` minus the per-kind counts and must be zero.
///
/// **A per-second `busy_us` is not comparable across boots that delivered
/// different amounts of work.** The guest sets the draw rate on this rail, so a
/// change that slows the guest lowers `busy_us` by lowering the workload. Divide
/// by `retired_draws` or by the kind's own `*_n` before comparing two arms —
/// the writeback's own positive control halved the frame rate and lowered
/// `busy_us` by 48 % while per-submission GPU cost moved 1.5 %.
/// # `pass_us` splits the GPU second where the kinds cannot
///
/// The per-[`Kind`] columns tile `busy_us`, but they stop at the submission, and
/// a draw submission on this rail carries tens of draws across several render
/// pass instances. So the tiling can say the GPU second is all `draw_us` — it
/// reads exactly that on a driven Maps boot — while saying nothing about
/// whether that second goes on drawing or on beginning and ending passes.
///
/// `pass_us` is the time stamped *inside* pass instances, so it is a part of
/// `busy_us` and not a peer of it. Read the remainder:
///
/// ```text
/// pass_us              inside pass instances: the draws themselves
/// busy_us - pass_us    outside them: pass boundaries and non-pass work
/// ```
///
/// That remainder is the number that says whether fewer pass boundaries is a
/// lever worth building. `pass_n` is its denominator, and it counts instances
/// whose slot retired with both stamps readable — not `passbegin_*`, which
/// counts every instance begun.
fn emit_gpu_span(executor: &dyn crate::runtime::executor::Executor) {
    let Some(w) = executor.gpu_span_window() else {
        return;
    };
    crate::observe::off(format!(
        "gpu_span busy_us={} busy_max_us={} read={} armed={} sealed={} unread={} \
         unattributed={} draw_us={} draw_n={} retired_draws={} store_us={} store_n={} \
         readback_us={} readback_n={} compute_us={} compute_n={} stamp_us={} stamp_n={} \
         pass_us={} pass_n={}",
        w.busy_us,
        w.busy_max_us,
        w.read,
        w.armed,
        w.sealed,
        w.unread,
        w.unattributed(),
        w.kind_us[0],
        w.kind_n[0],
        w.retired_draws,
        w.kind_us[1],
        w.kind_n[1],
        w.kind_us[2],
        w.kind_n[2],
        w.kind_us[3],
        w.kind_n[3],
        w.kind_us[4],
        w.kind_n[4],
        w.pass_us,
        w.pass_n,
    ));
}

/// Where a compute-gather dispatch's CPU cost goes, four ways.
///
/// Emitted only when a gather dispatched, so the line's presence is itself the
/// statement that this boot ran the dispatch arm — see
/// [`reims_vgpu_vulkan::engine::gather_phase`] for what each part is and
/// what would remove it.
fn emit_gather_phase(executor: &dyn crate::runtime::executor::Executor) {
    let Some(w) = executor.gather_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "gather_phase plan_us={} plan_n={} stage_us={} stage_n={} \
         dset_us={} dset_n={} record_us={} record_n={}",
        w.plan_us, w.plan_n, w.stage_us, w.stage_n, w.dset_us, w.dset_n, w.record_us, w.record_n,
    ));
}

/// Under `draw_phase`, dividing its largest column — `stage_us` is 83 % of that
/// phase's second on a driven drag, and the five parts want opposite fixes.
fn emit_stage_phase(executor: &dyn crate::runtime::executor::Executor) {
    let Some(w) = executor.stage_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "stage_phase acquire_us={} acquires={} bytes_us={} bytes_n={} bytes_b={} \
         runs_us={} runs_n={} runs_b={} swap_us={} swap_n={} swap_b={} \
         shift_us={} shift_n={} shift_b={} \
         gather_us={} gather_n={} gather_b={}",
        w.acquire_us,
        w.acquires,
        w.bytes_us,
        w.bytes_n,
        w.bytes_b,
        w.runs_us,
        w.runs_n,
        w.runs_b,
        w.swap_us,
        w.swap_n,
        w.swap_b,
        w.shift_us,
        w.shift_n,
        w.shift_b,
        w.gather_us,
        w.gather_n,
        w.gather_b,
    ));
}

/// The engine mutex's wait and hold time over the same window, split by which
/// thread class asked for it.
///
/// Emitted beside `window_publish` because it divides the gap that line opens:
/// `window_publish fresh` is what the device offered the window and
/// `host_window_cadence presents` is what reached the screen, and when the two
/// disagree the first candidate is that the window thread could not have the
/// engine while the worker held it.
fn emit_engine_lock(executor: &dyn crate::runtime::executor::Executor, win_ms: u64) {
    if let Some(line) = executor.take_engine_lock_census(win_ms) {
        crate::observe::off(line);
    }
}

/// Count a drain wake-up that returned before taking the device lock.
pub fn note_drain_skipped() {
    DRAIN_DUTY.note_skipped();
}

/// Open a drain entry, banking the condvar wait that preceded it. Returns the
/// entry instant, which the caller times its device-lock wait from.
pub fn note_drain_entry() -> u64 {
    DRAIN_DUTY.note_gap_entry(crate::observe::elapsed_us())
}

/// Bank the device-lock wait of an entry that went on to run a tranche.
pub fn note_drain_lock_wait(us: u64) {
    DRAIN_DUTY.note_gap_lock(us);
}

/// Time one post-tranche sweep and attribute it, returning what it returned.
///
/// A wrapper rather than a `started`/`note` pair at each call site: these four
/// sit in one straight run of statements, and a pair spelled four times is four
/// chances to time the wrong one.
pub fn post_sweep<T>(sweep: PostSweep, run: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = run();
    DRAIN_DUTY.note_post_sweep(sweep, started.elapsed().as_nanos() as u64);
    out
}

/// A prompt `HostAction` has just been pushed onto an empty prompt queue.
///
/// Process-global, like every counter here: this device is instantiated once per
/// QEMU process, and a second slot would fold its pulses into the first's mean
/// rather than corrupt it.
pub fn note_irq_armed() {
    DRAIN_DUTY.note_irq_armed(crate::observe::elapsed_us());
}

/// The QEMU action BH has just emptied the prompt queue.
pub fn note_irq_delivered() {
    DRAIN_DUTY.note_irq_delivered(crate::observe::elapsed_us());
}

/// Close a drain entry: everything since `busy_end_us` is post-tranche work, or
/// the whole entry is a skip when it never took the lock.
///
/// Must be called on **every** return path out of the entry point, including the
/// ones that bail after taking the lock — an unclosed entry leaves
/// `gap_last_exit_us` stale and folds this entry's whole duration into the next
/// one's `gap_idle_us`, which reads as an idle worker rather than as a bug here.
pub fn note_drain_exit(busy_end_us: u64, skipped: bool) {
    DRAIN_DUTY.note_gap_exit(crate::observe::elapsed_us(), busy_end_us, skipped);
}

/// Attribute elapsed time since `started` to one phase of the current tranche.
pub fn note_drain_phase(phase: DrainPhase, started: std::time::Instant) {
    DRAIN_DUTY.note_phase(phase, started.elapsed().as_micros() as u64);
}

/// Attribute one slice of a render-rail flush to the part of it that was spent.
pub fn note_readback_phase(phase: ReadbackPhase, us: u64) {
    DRAIN_DUTY.note_readback(phase, us);
}

/// Record the two GPU-side spans of one readback command buffer, read from its
/// own timestamp queries after the fence signalled.
///
/// Reported as `readback_split bar_us`/`gpu_us` beside `fence_us`, which they
/// divide. `fence_us` is CPU wall clock across `vkWaitForFences` and therefore
/// contains three different things with three different fixes: the draw batch
/// still executing (`bar_us`, the copy's barrier waiting on it), the copy
/// itself (`gpu_us`), and the cost of asking (the remainder). Both spans are
/// deltas between two points on the GPU's own timeline, so no clock correlation
/// is involved and the subtraction is exact.
pub fn note_readback_gpu_us(barrier_us: u64, copy_us: u64) {
    DRAIN_DUTY.note_readback_gpu(barrier_us, copy_us);
}

/// Count one guest-Store routing decision, by route name.
///
/// The routes are the attribution for `engine_delta`'s readback bytes: only
/// `cpu_portability` reads a full frame back and CPU-copies it into the guest's
/// pages, and only it is forced to — `gva_store_defer_eligible` refuses any
/// target with a nonzero `mapping_id`, so an IOSurface texture composite Store has no
/// deferred rail to take. Whether that is 2 Stores a second or 20 decides
/// whether building one is worth it, and the route's own first-appearance line
/// is deduplicated per process and cannot say.
type StoreRouteCounter = std::sync::Arc<std::sync::atomic::AtomicU64>;

static STORE_ROUTES: std::sync::Mutex<std::collections::BTreeMap<&'static str, StoreRouteCounter>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

thread_local! {
    /// Routes repeat at render-command frequency. Keep their registry lookup
    /// off the hot path; the registry remains sorted solely for census output.
    static STORE_ROUTE_CACHE: std::cell::RefCell<
        std::collections::HashMap<&'static str, StoreRouteCounter>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// # This census cannot find the Finder icon defect, and that is now measured
///
/// Several sessions have concluded "no counter separates a corrupt icon round
/// from a clean one" by printing six or eight hand-picked columns. That is a
/// statement about the columns someone thought to print, not about the census.
/// It has now been asked of the whole census at once.
///
/// Three 14-round `icon-composite.sh` boots, x86 / Vulkan, pooled: **42 scored
/// rounds, 9 corrupt, 33 clean**. Every counter in this map present in at least
/// 80% of rounds was normalised per 1000 `draw_scissor_full` — round length
/// varies ~40% on this rig and almost every draw-path counter is proportional
/// to it — and ranked by AUC, the probability that a random corrupt round
/// scores above a random clean one. The best column in the entire census:
///
/// ```text
/// AUC 0.75  surface_flush             permutation p = 0.021 raw
/// AUC 0.73  load_seed_ok                            p = 0.914 Bonferroni
/// AUC 0.72  iosurface_texture_seed_uploaded      (43 columns tested)
/// AUC 0.72  iosurface_texture_seed_guest_wrote
/// AUC 0.71  iosurface_gw_ref_moved
/// ```
///
/// Corrected for having looked at 43 columns, nothing is distinguishable from
/// noise. The leaders are also largely one quantity wearing different names — a
/// IOSurface texture seed upload is a `load_seed_ok_mapping` — so they are one weak
/// signal, not five.
///
/// The reason is structural rather than a gap to be filled by adding counters.
/// A round runs ~11 000 draws and the defect is **one** icon: a single
/// operation going wrong is a ~1e-4 perturbation of any population, which no
/// aggregate can resolve. Adding a counter to this map cannot change that, and
/// a session that adds one and reads it per round is repeating a measurement
/// that has now been shown to have no power.
///
/// What would have power is a *screen-to-resource join*: name the 64x64 target
/// backing the cell that is blank in the capture, then dump that one target's
/// history. A distinct-texel content summary would be one half of it — a
/// correct icon carries hundreds of distinct texels and a blank one collapses
/// to one — and the other half is the mapping from a screen rectangle to a
/// target identity. Neither exists today; `observe::bgra_present_stats` is the
/// nearest thing and it summarises a whole frame, not one target.
///
/// Settled by the same three boots, so nobody re-runs it: the Vulkan
/// synchronization repairs are not the producer either. Corruption rates were
/// 3/14 before them, 4/14 after the first, 2/14 after all five. The hazards
/// they closed were real undefined behaviour and those fixes stand on that
/// ground alone — see
/// `engine::exec::barrier_resident_for_transfer_read` — but
/// they do not move this class.
///
/// # A scoring flaw that inverts verdicts, recorded here because the harness is not tracked
///
/// The repro scripts live under `.agents/`, which is gitignored, so a fix made
/// there does not survive to the next session and this warning would vanish
/// with it.
///
/// `iconscore.py` scores a capture by counting blue blobs in a horizontal band
/// and comparing the count to `--expect`. Its own description defines the
/// population as "blue blobs of near-identical area", but it only ever policed
/// the *small* side of that (a `shrunk` class). On 2026-07-31 an unrelated blue
/// object of area 3247, against an icon median of 1235, entered the band and
/// was counted toward `expect`. That **inverted the verdict of all fourteen
/// rounds of a probe boot**: a round showing all seven icons counted 8 and read
/// CORRUPT, and a round genuinely missing one counted 7 and read CLEAN.
///
/// It was caught only by re-deriving each round's verdict from the *positions*
/// of the blobs rather than their number. Any conclusion of the form "n corrupt
/// rounds out of m" is worth exactly as much as the assumption that nothing
/// else blue and icon-sized was on screen, and that assumption is not
/// self-checking. A symmetric `outsized` exclusion, reported on the output line
/// rather than applied silently, is the fix; if the harness in front of you
/// does not print `outsized=` when something is excluded, it predates this and
/// its verdicts should be re-derived positionally before they are believed.
pub fn note_store_route(route: &'static str) {
    note_store_route_n(route, 1);
}

/// Add `n` to a named count in the same per-second window as [`note_store_route`].
///
/// For events that arrive in batches — one notify marking many cache entries —
/// where the number that matters is the entries, not the notifies. The common
/// path is a thread-local lookup and a relaxed atomic addition; the registry
/// lock is taken only the first time one thread sees one route.
pub fn note_store_route_n(route: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    STORE_ROUTE_CACHE.with(|cache| {
        let hit = {
            let cache = cache.borrow();
            if let Some(counter) = cache.get(route) {
                counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                true
            } else {
                false
            }
        };
        if hit {
            return;
        }

        let counter = {
            let Ok(mut routes) = STORE_ROUTES.lock() else {
                return;
            };
            std::sync::Arc::clone(
                routes
                    .entry(route)
                    .or_insert_with(|| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0))),
            )
        };
        counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        cache.borrow_mut().insert(route, counter);
    });
}

/// Accumulate microseconds against a named cost, into the same per-second window
/// as the route counts above.
///
/// The same map on purpose. `store_routes` is already drained once a second
/// beside `drain_duty`, so a cost reported here divides into that window's
/// `draw_us` with no join and no cross-boot comparison. `draw_phase` cannot
/// carry these: it brackets the *engine's* internals, and this is the runtime
/// work on either side of them — which is where **28 % of `draw_us`** was
/// going unattributed (~245 ms per second, stable across 200 windows of the
/// 2026-07-30 boot, larger than `stage_us` and `readback_us` and second only to
/// `wait_us`). A phase table that sums to 72 % of the thing it decomposes
/// cannot be used to choose what to fix.
pub fn note_store_route_us(name: &'static str, us: u64) {
    note_store_route_n(name, us);
}

/// Read one route's count out of the live window, for tests that assert a
/// census fired rather than trusting that it was wired up.
///
/// A counter nobody reads back is a counter that can be deleted, mistyped, or
/// placed on the wrong side of an early return without any test noticing — and
/// several of this crate's readings have turned on exactly which side of a
/// branch a `note_store_route` sat on.
#[cfg(test)]
pub(crate) fn store_route_count(route: &str) -> u64 {
    STORE_ROUTES
        .lock()
        .ok()
        .and_then(|routes| routes.get(route).cloned())
        .map(|counter| counter.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(0)
}

/// Drain and format the window's route counts, or `None` if none were taken.
fn take_store_routes() -> Option<String> {
    let routes = STORE_ROUTES.lock().ok()?;
    let mut out = String::from("store_routes");
    let mut any = false;
    for (route, counter) in routes.iter() {
        let n = counter.swap(0, std::sync::atomic::Ordering::Relaxed);
        if n != 0 {
            out.push_str(&format!(" {route}={n}"));
            any = true;
        }
    }
    any.then_some(out)
}

#[cfg(test)]
mod drain_gap_tests {
    use super::{DrainDutyCensus, PostSweep, DRAIN_DUTY_REPORT_MS};

    /// The four gap buckets plus `busy_us` account for the whole window.
    ///
    /// `duty` has always said what fraction of a second the worker was busy and
    /// nothing has ever said what the rest was — 31 % of the one thread every
    /// guest packet serializes through, on a driven Maps boot. These four are
    /// the only things it can be doing, so the check that they are the right
    /// four is that they tile: a bucket that is really two, or a span banked
    /// into no bucket at all, shows up here as a shortfall.
    #[test]
    fn the_gap_buckets_and_the_busy_time_tile_the_window() {
        let c = DrainDutyCensus::default();
        // Arms the window at t=1ms. The first `note` never reports — it would
        // divide the whole process lifetime into one tranche — so the window
        // under test has to be opened before anything is accumulated into it.
        assert!(c.note(0, 0, 1).is_none(), "the first call arms the window");
        // Six entries on a fixed stride, each: 40 idle, 10 lock, 30 drain, 5
        // publish, 16 post — 16 so the four-way sweep split divides it whole and
        // the shortfall check below reads truncation as nothing. Times are microseconds on the crate clock. The
        // first entry's idle is the one span that cannot be measured — there is
        // no previous exit to measure it from — so it lies before `t0` and is
        // not part of what has to tile.
        let (idle, lock, drain, publish, post) = (40u64, 10u64, 30u64, 5u64, 16u64);
        let stride = idle + lock + drain + publish + post;
        let entries = 6u64;
        let t0 = 1_000u64;
        for i in 0..entries {
            let entry = t0 + i * stride;
            c.note_gap_entry(entry);
            c.note_gap_lock(lock);
            let busy_end = entry + lock + drain + publish;
            assert!(c.note(drain, publish, 1).is_none(), "the window is not due");
            // The four sweeps that make up `post`, in nanoseconds. They tile it
            // exactly here, which is what the shortfall check below is for: on a
            // real tranche the wrapper's own `Instant` reads sit outside them.
            for sweep in PostSweep::ALL {
                c.note_post_sweep(sweep, post * 1_000 / PostSweep::COUNT as u64);
            }
            c.note_gap_exit(busy_end + post, busy_end, false);
        }
        // One skipped entry on the same stride: no lock, no tranche, the whole
        // call is a skip.
        let skip_entry = t0 + entries * stride;
        let skip_us = 7u64;
        c.note_gap_entry(skip_entry);
        c.note_skipped();
        c.note_gap_exit(skip_entry + skip_us, skip_entry, true);
        let last_exit = skip_entry + skip_us;

        let line = c
            .note(0, 0, 1 + DRAIN_DUTY_REPORT_MS)
            .expect("the window is due");
        let field = |name: &str| -> u64 {
            line.split_whitespace()
                .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
                .unwrap_or_else(|| panic!("{name} is on the line: {line}"))
                .parse()
                .expect("a microsecond count")
        };
        // `entries` idle gaps, not `entries + 1`: the first entry contributes
        // none and the skipped one contributes the last.
        assert_eq!(field("gap_idle_us"), idle * entries);
        assert_eq!(field("gap_lock_us"), lock * entries);
        assert_eq!(field("gap_post_us"), post * entries);
        // The split has to add up to the total it divides, so a sweep that gains
        // a call site and no timer reads as a shortfall rather than as noise.
        let post_split: u64 = ["cachelv", "slotre", "relpg", "bindlv"]
            .into_iter()
            .map(|s| field(&format!("post_{s}_us")))
            .sum();
        assert_eq!(post_split, post * entries);
        assert_eq!(field("gap_skip_us"), skip_us);
        assert_eq!(field("busy_us"), (drain + publish) * entries);
        // The whole point: nothing the worker did between `t0` and its last
        // exit is outside these five.
        assert_eq!(
            field("gap_idle_us")
                + field("gap_lock_us")
                + field("gap_post_us")
                + field("gap_skip_us")
                + field("busy_us"),
            last_exit - t0,
            "a span banked into no bucket reads as a shortfall here: {line}"
        );
    }
}

#[cfg(test)]
mod irq_wait_tests {
    use super::DrainDutyCensus;

    /// The delivery clock measures the **oldest** undelivered prompt action, and
    /// banks one wait per emptying of the queue.
    ///
    /// Both halves matter. Re-arming over a non-empty queue would restart the
    /// clock and report the newest pulse's wait, which is not the one a frame is
    /// blocked behind; banking per action instead of per emptying would count
    /// the same BH hop once for every action that rode it.
    #[test]
    fn the_delivery_clock_times_the_oldest_action_and_banks_once_per_hop() {
        let c = DrainDutyCensus::default();
        c.note_irq_armed(1_000);
        // A second pulse joining a queue that is already waiting does not
        // restart the clock.
        c.note_irq_armed(1_040);
        c.note_irq_delivered(1_300);
        // A delivery with nothing armed banks nothing — the BH runs on its own
        // cadence and finds the queue empty far more often than not.
        c.note_irq_delivered(1_400);
        c.note_irq_armed(2_000);
        c.note_irq_delivered(2_050);

        assert!(c.note(0, 0, 1).is_none(), "the first call arms the window");
        let line = c
            .note(0, 0, 1 + super::DRAIN_DUTY_REPORT_MS)
            .expect("the window is due");
        let field = |name: &str| -> u64 {
            line.split_whitespace()
                .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
                .unwrap_or_else(|| panic!("{name} is on the line: {line}"))
                .parse()
                .expect("a microsecond count")
        };
        assert_eq!(
            field("irq_waits"),
            2,
            "one per emptying, not one per action"
        );
        assert_eq!(field("irq_wait_us"), 300 + 50);
        assert_eq!(
            field("irq_wait_max_us"),
            300,
            "the tail is what a frame waits on, so it is reported and not averaged away"
        );
    }
}

#[cfg(test)]
mod store_route_tests {
    use super::{
        note_store_route, note_store_route_n, note_store_route_us, store_route_count,
        take_store_routes,
    };

    fn clear_window() {
        let _ = take_store_routes();
    }

    #[test]
    fn counts_and_costs_share_the_window() {
        clear_window();
        note_store_route("test_route_count");
        note_store_route_n("test_route_count", 4);
        note_store_route_us("test_route_cost_us", 17);

        assert_eq!(store_route_count("test_route_count"), 5);
        assert_eq!(store_route_count("test_route_cost_us"), 17);
    }

    #[test]
    fn counters_coalesce_across_threads() {
        clear_window();
        let threads: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..1_000 {
                        note_store_route("test_route_threaded");
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(store_route_count("test_route_threaded"), 4_000);
    }

    #[test]
    fn drain_is_sorted_and_does_not_repeat_values() {
        clear_window();
        note_store_route("test_route_zulu");
        note_store_route("test_route_alpha");

        let line = take_store_routes().unwrap();
        let alpha = line.find(" test_route_alpha=1").unwrap();
        let zulu = line.find(" test_route_zulu=1").unwrap();
        assert!(alpha < zulu);
        assert!(take_store_routes().is_none());
    }
}

#[cfg(test)]
mod lookup_age_tests {
    use super::list_lookup_age_route;

    /// Every decade boundary lands in the band below it, and the two families
    /// never share a counter.
    ///
    /// The boundaries are the whole content of this function, and an off-by-one
    /// at one of them would move a population between decades without any
    /// reading looking wrong — which is exactly the shape of error a banding is
    /// built to avoid making.
    #[test]
    fn the_bands_are_decades_and_the_two_families_are_disjoint() {
        const EDGES: [(u64, &str, &str); 6] = [
            (0, "list_hit_age_under_100us", "list_miss_age_under_100us"),
            (99, "list_hit_age_under_100us", "list_miss_age_under_100us"),
            (100, "list_hit_age_under_1ms", "list_miss_age_under_1ms"),
            (999, "list_hit_age_under_1ms", "list_miss_age_under_1ms"),
            (1_000, "list_hit_age_under_10ms", "list_miss_age_under_10ms"),
            (
                10_000,
                "list_hit_age_under_100ms",
                "list_miss_age_under_100ms",
            ),
        ];
        for (us, hit, miss) in EDGES {
            assert_eq!(list_lookup_age_route(true, us), hit, "hit at {us}");
            assert_eq!(list_lookup_age_route(false, us), miss, "miss at {us}");
        }
        for us in [100_000u64, 264_000, u64::MAX] {
            assert_eq!(list_lookup_age_route(true, us), "list_hit_age_over_100ms");
            assert_eq!(list_lookup_age_route(false, us), "list_miss_age_over_100ms");
        }
    }

    /// No age reaches the same counter from both families, so a hit can never be
    /// summed into the miss banding it is the control for.
    #[test]
    fn a_hit_and_a_miss_never_share_a_counter() {
        for us in [0u64, 50, 100, 500, 1_000, 5_000, 10_000, 99_999, 100_000] {
            assert_ne!(
                list_lookup_age_route(true, us),
                list_lookup_age_route(false, us),
                "at {us}"
            );
        }
    }
}

/// How many packets one drain call consumes before it stops.
///
/// # The question this answers
///
/// The x86/Vulkan rail is drain-CPU bound and the route to 60 fps is encoding
/// command buffers concurrently — `reims_vgpu_core::render`'s `thread_seam`
/// doc establishes that the seam is the *packet* and not the draw, because a
/// resolver and a recorder cut apart at draw granularity must synchronise 6.4
/// times a draw at 7.6 µs a time.
///
/// A packet seam only pays if packets are available together. The guest may
/// publish more packets while this device is draining, so the child population
/// is a throughput upper bound rather than an initial-tail snapshot. The
/// separate `exec_run` population removes non-EXEC boundaries; a scheduler
/// still has to measure readiness and resource dependencies before treating
/// that run length as executable width.
///
/// This is a count and a distribution, not a timing: it is read to size a
/// design, and it must survive both host contention and the perturbation of
/// measuring it.
///
/// The buckets are powers of two because the reading wanted is an order of
/// magnitude — "usually one", "usually a handful", "usually dozens" — and a
/// mean would hide a bimodal ring behind a comfortable-looking average.
#[derive(Default)]
struct TrancheCensus {
    wakes: std::sync::atomic::AtomicU64,
    packets: std::sync::atomic::AtomicU64,
    max: std::sync::atomic::AtomicU64,
    /// `[1, 2-3, 4-7, 8-15, 16-31, 32-63, 64+]`.
    buckets: [std::sync::atomic::AtomicU64; 7],
}

/// Which ring a wake drained. They are two populations and blending them
/// would hide the one that matters: the render command stream arrives on the
/// child FIFOs, and the root ring carries control traffic whose width says
/// nothing about encoder fan-out.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrancheRing {
    Root = 0,
    Child = 1,
}

static TRANCHE: [TrancheCensus; 2] = [
    TrancheCensus {
        wakes: std::sync::atomic::AtomicU64::new(0),
        packets: std::sync::atomic::AtomicU64::new(0),
        max: std::sync::atomic::AtomicU64::new(0),
        buckets: [const { std::sync::atomic::AtomicU64::new(0) }; 7],
    },
    TrancheCensus {
        wakes: std::sync::atomic::AtomicU64::new(0),
        packets: std::sync::atomic::AtomicU64::new(0),
        max: std::sync::atomic::AtomicU64::new(0),
        buckets: [const { std::sync::atomic::AtomicU64::new(0) }; 7],
    },
];

/// Consecutive successfully-consumed `EXEC_INDIRECT2` packets. Unlike the
/// child tranche population, this excludes resource/control packets and splits
/// at each non-EXEC opcode. It is the width a whole-submission encoder could
/// actually consume without crossing another guest command class.
static EXEC_RUN: TrancheCensus = TrancheCensus {
    wakes: std::sync::atomic::AtomicU64::new(0),
    packets: std::sync::atomic::AtomicU64::new(0),
    max: std::sync::atomic::AtomicU64::new(0),
    buckets: [const { std::sync::atomic::AtomicU64::new(0) }; 7],
};

fn note_width(c: &TrancheCensus, packets: u64) {
    use std::sync::atomic::Ordering;
    if packets == 0 {
        return;
    }
    c.wakes.fetch_add(1, Ordering::Relaxed);
    c.packets.fetch_add(packets, Ordering::Relaxed);
    c.max.fetch_max(packets, Ordering::Relaxed);
    let bucket = match packets {
        1 => 0,
        2..=3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        16..=31 => 4,
        32..=63 => 5,
        _ => 6,
    };
    c.buckets[bucket].fetch_add(1, Ordering::Relaxed);
}

/// Record that one drain wake consumed `packets` packets before the ring ran
/// dry. A wake that found nothing is not a wake for this purpose — it says
/// nothing about available width — so zero is dropped.
pub fn note_tranche_width(ring: TrancheRing, packets: u64) {
    note_width(&TRANCHE[ring as usize], packets);
}

pub fn note_exec_run_width(packets: u64) {
    note_width(&EXEC_RUN, packets);
}

/// [`take_drain_tranche`] for the census's own test.
#[cfg(test)]
pub(super) fn take_drain_tranche_for_test(t_ms: u64) -> Vec<String> {
    take_drain_tranche(t_ms)
}

/// One window of [`note_tranche_width`], taken and reset, one line per ring.
fn take_drain_tranche(t_ms: u64) -> Vec<String> {
    use std::sync::atomic::Ordering;
    let mut lines = Vec::new();
    for (ring, c) in [
        ("root", &TRANCHE[0]),
        ("child", &TRANCHE[1]),
        ("exec_run", &EXEC_RUN),
    ] {
        let wakes = c.wakes.swap(0, Ordering::Relaxed);
        if wakes == 0 {
            continue;
        }
        let packets = c.packets.swap(0, Ordering::Relaxed);
        let max = c.max.swap(0, Ordering::Relaxed);
        let b: Vec<String> = c
            .buckets
            .iter()
            .map(|x| x.swap(0, Ordering::Relaxed).to_string())
            .collect();
        let populations = if ring == "exec_run" { "runs" } else { "wakes" };
        lines.push(format!(
            "drain_tranche t={t_ms} ring={ring} {populations}={wakes} packets={packets} max={max} b1={} b2={} b4={} b8={} b16={} b32={} b64={}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6]
        ));
    }
    lines
}
