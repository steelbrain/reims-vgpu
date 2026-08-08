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
/// The three arms are counted separately because a single "delivered" tally
/// cannot tell the two silences apart, and they have opposite meanings:
/// `not_online` is the display never having come up (no VBL is owed at all),
/// while `not_claimed` is the limiter doing its job at the advertised rate.
/// Reading a low delivered count without them would license both conclusions.
///
/// One line per 1024 deliveries — about 8 s at the grid rate, and it costs three
/// relaxed increments per poll otherwise.
/// Which way the VBL path went. Indices into [`VblCensus`].
pub(crate) const VBL_NOT_ONLINE: usize = 0;
pub(crate) const VBL_NOT_CLAIMED: usize = 1;
pub(crate) const VBL_DELIVERED: usize = 2;

/// One report per this many deliveries — about 8 s at the grid rate.
const VBL_REPORT_EVERY: u64 = 1024;

#[derive(Default)]
pub(crate) struct VblCensus {
    arms: [std::sync::atomic::AtomicU64; 3],
    last_report_ms: std::sync::atomic::AtomicU64,
    last_report_n: std::sync::atomic::AtomicU64,
}

impl VblCensus {
    /// Count one traversal and return the line to emit when a report is due.
    ///
    /// Returns the line rather than emitting it so the reporting rule is
    /// testable without a log sink: the interesting properties are "only
    /// deliveries report", "the rate is measured over the window and not the
    /// process lifetime", and "the two silent arms stay separable", and all
    /// three are assertions about this return value.
    pub(crate) fn note(&self, arm: usize, now_ms: u64) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let n = self.arms[arm].fetch_add(1, Relaxed) + 1;
        if arm != VBL_DELIVERED || !n.is_multiple_of(VBL_REPORT_EVERY) {
            return None;
        }
        let since_ms = now_ms.saturating_sub(self.last_report_ms.swap(now_ms, Relaxed));
        let since_n = n.saturating_sub(self.last_report_n.swap(n, Relaxed));
        // Window rate, not a lifetime average: the lifetime figure carries the
        // pre-online stretch forever and would read low long after the display
        // came up.
        let hz = if since_ms > 0 {
            (since_n * 1000) as f64 / since_ms as f64
        } else {
            0.0
        };
        Some(format!(
            "display_vbl delivered={n} not_claimed={} not_online={} window_hz={hz:.1} \
             grid_hz={:.1}",
            self.arms[VBL_NOT_CLAIMED].load(Relaxed),
            self.arms[VBL_NOT_ONLINE].load(Relaxed),
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
    /// Write the frame into the guest's pages (`write_bgra8_skipping`).
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
/// not of doing it once badly. `write_bgra8_skipping` makes up to three
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
    drain_us: std::sync::atomic::AtomicU64,
    publish_us: std::sync::atomic::AtomicU64,
    draw_us: std::sync::atomic::AtomicU64,
    draws: std::sync::atomic::AtomicU64,
    compute_us: std::sync::atomic::AtomicU64,
    computes: std::sync::atomic::AtomicU64,
    flush_us: std::sync::atomic::AtomicU64,
    flushes: std::sync::atomic::AtomicU64,
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
        let slow = self.slow_tranches.swap(0, Relaxed);
        let busy = drain.saturating_add(publish);
        let duty = busy as f64 / (win_ms as f64 * 1000.0);
        Some(format!(
            "drain_duty win_ms={win_ms} tranches={tranches} skipped={skipped} busy_us={busy} \
             duty={duty:.3} drain_us={drain} publish_us={publish} max_tranche_us={max} \
             draw_us={draw} draws={draws} compute_us={compute} computes={computes} \
             flush_us={flush} flushes={flushes} max_flush_us={max_flush} \
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

/// Accumulate one completed drain tranche; emits at most once per second.
pub fn note_drain_tranche(drain_us: u64, publish_us: u64) {
    if let Some(line) = DRAIN_DUTY.note(drain_us, publish_us, crate::observe::elapsed_ms() as u64) {
        crate::observe::off(line);
        // Immediately after `drain_duty`, so the two read as one record: the
        // rails must sum to its `flush_us` and their counts to its `flushes`.
        if let Some(rails) = DRAIN_DUTY.take_flush_rails() {
            crate::observe::off(rails);
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
        emit_engine_lock(DRAIN_DUTY.last_window_ms());
        if let Some(routes) = take_store_routes() {
            crate::observe::off(routes);
        }
        // Beside `store_routes` deliberately: the two are read against each
        // other. `type4_backing_fail` lines equal `type4_backing_recovered +
        // type4_backing_superseded` from that line plus this one's `n`, and a
        // refusal that never recovered is only visible as the residue.
        if let Some(outstanding) = crate::runtime::objects::type4_backing_outstanding_census() {
            crate::observe::off(outstanding);
        }
        // Onto the census cadence rather than a timer of its own, so a reader
        // pairing the footprint against `store_routes` is reading one clock.
        // The run dump rate-limits itself; this is the only caller.
        for line in crate::observe::footprint::census_lines(crate::observe::elapsed_ms() as u64) {
            crate::observe::off(line);
        }
        emit_engine_delta();
        // After `emit_engine_delta`, which emits `draw_phase`: the two divide
        // against each other and reading them in the other order invites
        // treating the engine's twelve phases as the whole draw, which is the
        // misreading this line exists to correct. Not gated on the backend —
        // the timer is runtime-side and the Metal arm can adopt it without a
        // second census.
        emit_chain_phase();
        emit_object_cache_levels();
        emit_guest_import_levels();
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
#[cfg(feature = "backend-vulkan")]
fn emit_guest_import_levels() {
    let (bytes, count) = crate::backend::vulkan::engine::guest_import_census();
    let (spans, span_bytes) = crate::runtime::guest_ram_map::span_census();
    // An engine that never imported emits nothing, so a host on a negative
    // `host_pointer` rung — or a boot before the first guest window — costs no
    // line, and a zero here always means the copying rails rather than silence.
    if count == 0 {
        return;
    }
    crate::observe::off(format!(
        "guest_import_levels (levels, not per-interval) ramblocks={count}/{spans} \
         mib={}/{} (imported/reported; a span is imported at first reference, \
         so below is lazy and above is impossible)",
        bytes / (1024 * 1024),
        span_bytes / (1024 * 1024),
    ));
}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_guest_import_levels() {}

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
/// `m2v` counts translated shaders (`runtime::m2v_cache`); the rest are the
/// Vulkan engine's immutable-object caches.
#[cfg(feature = "backend-vulkan")]
fn emit_object_cache_levels() {
    let [shaders, layouts, passes, pipelines, samplers, compute_pipelines] =
        crate::backend::vulkan::engine::object_cache_levels();
    let (_, _, m2v) = crate::runtime::m2v_cache::stats();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) m2v={m2v} shaders={shaders} \
         layouts={layouts} passes={passes} pipelines={pipelines} samplers={samplers} \
         compute_pipelines={compute_pipelines}"
    ));
}

#[cfg(feature = "backend-vulkan")]
fn emit_engine_delta() {
    use crate::backend::vulkan::engine::CounterSnapshot;
    static PREV: std::sync::Mutex<Option<CounterSnapshot>> = std::sync::Mutex::new(None);
    let now = crate::backend::vulkan::engine::counter_snapshot();
    let Ok(mut prev) = PREV.lock() else {
        return;
    };
    let d = now.delta_since(&prev.unwrap_or_default());
    *prev = Some(now);
    crate::observe::off(format!(
        "engine_delta creates={} allocs={} batch_opens={} batch_joins={} batch_flushes={} \
         batch_flush_draws={} batch_readback_joins={} readbacks={} readback_bytes={} render_post_wait_skips={} \
         target_reads={} target_read_bytes={} gpu_stamps={} pipeline_misses={} \
         shader_misses={} pass_misses={} layout_misses={} sampler_misses={} \
         sampled_cache_hits={} sampled_identity_hits={} sampled_cache_hit_bytes={} \
         sampled_cache_misses={} sampled_reuploads={} \
         sampled_reupload_bytes={} sampled_gathers={} sampled_gather_bytes={} \
         sampled_gather_skips={} sampled_gather_skip_bytes={} \
         sampled_guest_imports={} sampled_guest_import_bytes={} \
         sampled_gather_unvouched={} sampled_gather_unretained={} \
         draw_cover_full={} draw_cover_loaded_full_scissor={} \
         draw_cover_loaded_partial_scissor={} \
         buffer_guest_imports={} buffer_guest_import_bytes={} \
         buffer_guest_gathers={} buffer_guest_gather_bytes={} \
         buffer_guest_gather_regions={} \
         buffer_bind_reuses={} \
         buffer_snapshot_binds={} \
         guest_write_linear={} guest_write_rects={} guest_write_regions={} \
         seed_uploads={} seed_upload_bytes={} \
         ring_retire_blocks={} target_evicts={} desc_pool_grow={} gen_mismatch={}",
        d.creates,
        d.allocs,
        d.batch_opens,
        d.batch_joins,
        d.batch_flushes,
        d.batch_flush_draws,
        d.batch_readback_joins,
        d.readbacks,
        d.readback_bytes,
        d.render_post_wait_skips,
        d.target_reads,
        d.target_read_bytes,
        d.gpu_stamps,
        d.pipeline_misses,
        d.shader_misses,
        d.pass_misses,
        d.layout_misses,
        d.sampler_misses,
        d.sampled_cache_hits,
        d.sampled_identity_hits,
        d.sampled_cache_hit_bytes,
        d.sampled_cache_misses,
        d.sampled_reuploads,
        d.sampled_reupload_bytes,
        d.sampled_gathers,
        d.sampled_gather_bytes,
        d.sampled_gather_skips,
        d.sampled_gather_skip_bytes,
        d.sampled_guest_imports,
        d.sampled_guest_import_bytes,
        d.sampled_gather_unvouched,
        d.sampled_gather_unretained,
        d.draw_cover_full,
        d.draw_cover_loaded_full_scissor,
        d.draw_cover_loaded_partial_scissor,
        d.buffer_guest_imports,
        d.buffer_guest_import_bytes,
        d.buffer_guest_gathers,
        d.buffer_guest_gather_bytes,
        d.buffer_guest_gather_regions,
        d.buffer_bind_reuses,
        d.buffer_snapshot_binds,
        d.guest_write_linear,
        d.guest_write_rects,
        d.guest_write_regions,
        d.seed_uploads,
        d.seed_upload_bytes,
        d.ring_retire_blocks,
        d.target_evicts,
        d.desc_pool_grow,
        d.gen_mismatch,
    ));
    emit_registry_pressure(&now);
    emit_draw_phase();
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
#[cfg(feature = "backend-vulkan")]
fn emit_registry_pressure(now: &crate::backend::vulkan::engine::CounterSnapshot) {
    crate::observe::off(format!(
        "registry_pressure (levels, not per-interval) peak={} peak_mib={} \
         resident_samples={} resample_peak_ms={}/{} \
         slab_mib={}/{} sole_copy={}/{}mib cs_sole_copy={}/{}mib",
        now.registry_non_pinned_peak,
        now.registry_non_pinned_peak_bytes >> 20,
        now.sampled_gpu_binds,
        now.resident_resample_peak_ms,
        crate::backend::vulkan::engine::IDLE_TARGET_AGE_MS,
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
/// together: `chain_phase`'s `engine_us` must equal `draw_phase`'s twelve
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
        "bind_phase binds={} vertex_us={} fragment_us={} attrs_us={}",
        w.binds, w.vertex_us, w.fragment_us, w.attrs_us,
    ));
}

fn emit_chain_phase() {
    let Some(w) = crate::runtime::chain_phase::take_window() else {
        return;
    };
    crate::observe::off(format!(
        "chain_phase chains={} prep_us={} pipeline_us={} binds_us={} sampled_us={} \
         seed_us={} assemble_us={} engine_us={} store_us={} max_us={}",
        w.chains,
        w.prep_us,
        w.pipeline_us,
        w.binds_us,
        w.sampled_us,
        w.seed_us,
        w.assemble_us,
        w.engine_us,
        w.store_us,
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
         reflect_us={}",
        w.sampled, w.lookup_us, w.alias_us, w.resolve_us, w.samplers_us, w.reflect_us,
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
#[cfg(feature = "backend-vulkan")]
fn emit_draw_phase() {
    let Some(w) = crate::backend::vulkan::engine::draw_phase_window() else {
        return;
    };
    crate::observe::off(format!(
        "draw_phase draws={} prep_us={} slot_us={} pipeline_us={} stage_us={} stage_pass_us={} \
         acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
         descriptors_us={} \
         record_us={} submit_us={} wait_us={} readback_us={} max_us={} stalls={}",
        w.draws,
        w.prep_us,
        w.slot_us,
        w.pipeline_us,
        w.stage_us,
        w.stage_pass_us,
        w.acquire_us,
        w.acquire_sampled_us,
        w.sampled_upload_us,
        w.acquire_readback_us,
        w.descriptors_us,
        w.record_us,
        w.submit_us,
        w.wait_us,
        w.readback_us,
        w.max_us,
        w.stalls,
    ));
    emit_stage_phase();
}

/// Under `draw_phase`, dividing its largest column — `stage_us` is 83 % of that
/// phase's second on a driven drag, and the five parts want opposite fixes.
#[cfg(feature = "backend-vulkan")]
fn emit_stage_phase() {
    let Some(w) = crate::backend::vulkan::engine::stage_phase::take_window() else {
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

#[cfg(not(feature = "backend-vulkan"))]
fn emit_engine_delta() {}

/// The Metal arm's counterpart. Same question, same cadence, different tables:
/// this arm builds `MTLFunction` / `MTLRenderPipelineState` /
/// `MTLComputePipelineState` / `MTLSamplerState` / `MTLDepthStencilState` and
/// compute reflections, and holds them in `backend::metal::cache`.
///
/// No `m2v` field: AIR reaches Metal directly on this arm, so
/// `runtime::m2v_cache` is never populated and a zero there would read as an
/// empty cache rather than an absent rail.
#[cfg(all(
    not(feature = "backend-vulkan"),
    feature = "backend-metal",
    target_os = "macos"
))]
fn emit_object_cache_levels() {
    let [functions, render_pso, compute_pso, samplers, depth_stencil, reflections] =
        crate::backend::metal::cache_levels();
    crate::observe::off(format!(
        "object_cache_levels (levels, not per-interval) functions={functions} \
         render_pso={render_pso} compute_pso={compute_pso} samplers={samplers} \
         depth_stencil={depth_stencil} reflections={reflections}"
    ));
}

/// No compiled-object caches on this build: either no backend, or the Metal
/// feature without the Apple target that carries `backend::metal::cache`.
#[cfg(not(any(
    feature = "backend-vulkan",
    all(feature = "backend-metal", target_os = "macos")
)))]
fn emit_object_cache_levels() {}

/// The engine mutex's wait and hold time over the same window, split by which
/// thread class asked for it.
///
/// Emitted beside `window_publish` because it divides the gap that line opens:
/// `window_publish fresh` is what the device offered the window and
/// `host_window_cadence presents` is what reached the screen, and when the two
/// disagree the first candidate is that the window thread could not have the
/// engine while the worker held it.
#[cfg(feature = "backend-vulkan")]
fn emit_engine_lock(win_ms: u64) {
    if let Some(line) = crate::backend::vulkan::engine::take_engine_lock_census(win_ms) {
        crate::observe::off(line);
    }
}

#[cfg(not(feature = "backend-vulkan"))]
fn emit_engine_lock(_win_ms: u64) {}

/// Count a drain wake-up that returned before taking the device lock.
pub fn note_drain_skipped() {
    DRAIN_DUTY.note_skipped();
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
/// target with a nonzero `mapping_id`, so a type-11 composite Store has no
/// deferred rail to take. Whether that is 2 Stores a second or 20 decides
/// whether building one is worth it, and the route's own first-appearance line
/// is deduplicated per process and cannot say.
static STORE_ROUTES: std::sync::Mutex<Option<std::collections::BTreeMap<&'static str, u64>>> =
    std::sync::Mutex::new(None);

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
/// AUC 0.72  type11_seed_uploaded      (43 columns tested)
/// AUC 0.72  type11_seed_guest_wrote
/// AUC 0.71  t11_gw_ref_moved
/// ```
///
/// Corrected for having looked at 43 columns, nothing is distinguishable from
/// noise. The leaders are also largely one quantity wearing different names — a
/// type-11 seed upload is a `load_seed_ok_mapping` — so they are one weak
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
/// where the number that matters is the entries, not the notifies, and taking
/// the lock once per entry would cost more than the census is worth.
pub fn note_store_route_n(route: &'static str, n: u64) {
    if n == 0 {
        return;
    }
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(route)
            .or_default() += n;
    }
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
    if let Ok(mut g) = STORE_ROUTES.lock() {
        *g.get_or_insert_with(Default::default)
            .entry(name)
            .or_default() += us;
    }
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
        .and_then(|g| g.as_ref().and_then(|m| m.get(route).copied()))
        .unwrap_or(0)
}

/// Drain and format the window's route counts, or `None` if none were taken.
fn take_store_routes() -> Option<String> {
    let mut g = STORE_ROUTES.lock().ok()?;
    let routes = g.as_mut()?;
    if routes.is_empty() {
        return None;
    }
    let mut out = String::from("store_routes");
    for (route, n) in routes.iter() {
        out.push_str(&format!(" {route}={n}"));
    }
    routes.clear();
    Some(out)
}
