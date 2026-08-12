//! How many of the draw-time buffer gathers land on bytes the guest has not
//! said it changed.
//!
//! # The one number the cache design turns on
//!
//! `backend::vulkan::engine::pools::buffer_gather_working_set` measured ~20 800
//! gathers a second over ~1 900 distinct windows on a driven macos-13
//! sustained-animation boot, so **91 % of them re-assemble a window this device
//! already assembled**. That is a statement about *keys*. A content cache turns
//! on whether the *bytes* moved in between, and nothing has measured that.
//!
//! This crosses the two. For each bind that takes the gather rail it compares
//! the owning buffer object's [`crate::runtime::buffer_write_gen`] stamp against the one
//! this window carried the last time it was gathered, and reports the split:
//!
//! | route | meaning |
//! |---|---|
//! | `bgf_quiet` / `bgf_quiet_kb` | the guest declared no write to this object since the last gather — a hit a declaration-invalidated cache would have served |
//! | `bgf_wrote` / `bgf_wrote_kb` | the guest declared a write, so the copy was owed |
//! | `bgf_first` | no previous gather of this window to compare against |
//! | `bgf_dropped` | the tracking map was full, so this bind is not in the split |
//!
//! **Read `quiet_rate` beside `buffer_write_gen_bump`, always.** A reader here
//! compares a stamp taken on a `(task, reference)` pair at a draw-time bind
//! against a generation the decoder recorded under a `(task, object)` pair from
//! a validity record. If those two turn out to be different namespaces then no
//! comparison ever moves and this reports ~100 % quiet — a false positive in the
//! direction that licenses a cache serving stale bytes. A boot reading
//! `quiet_rate=1.000` beside `buffer_write_gen_bump=0` has measured a wiring
//! fault and not a workload.
//!
//! Read `bgf_quiet` against the two together and not against the gather count:
//! `bgf_first` is a compulsory miss no cache size removes, and folding it in
//! understates the achievable rate by the working set's own turnover.
//!
//! # It measures a ceiling, not a licence
//!
//! Nothing here decides a skip, and a high reading is not on its own permission
//! to build the cache. A cache invalidated this way would be trusting that the
//! guest's `writeInvalidates` and exec-table quads are a **complete** account of
//! CPU writes to a buffer's bytes. A surface's equivalent claim is not complete
//! — which is exactly why `runtime::gather_witness` carries a hypervisor half as
//! well — and the buffer case has not been tested either way.
//!
//! What the split does settle is whether it is worth testing. No cache
//! invalidated by declarations can beat `bgf_quiet`, so a low reading closes the
//! design outright and a high one says go and establish the soundness.
//!
//! # What it read, and the number in it that is a warning
//!
//! Two driven macos-13 sustained-animation boots, one census second each:
//!
//! ```text
//! quiet=57123  wrote=9396  first=178  dropped=0  tracked=6968  quiet_rate=0.859
//! quiet=38990  wrote=4951  first=352  dropped=0  tracked=5707  quiet_rate=0.887
//! ```
//!
//! The wiring check passes — `buffer_write_gen_bump` is non-zero on both — so
//! the two id namespaces do overlap and `wrote` is a real population rather than
//! a comparison that never moves.
//!
//! **But read the bump rate itself, because it is the finding.** Across the same
//! two boots:
//!
//! ```text
//!          validity_no_surface/s      buffer_write_gen_bump/s
//! boot 1          4 847 (median)             520 (median), 99-626
//! boot 2          4 151 (median)              31 (median), 29-60
//! ```
//!
//! The guest issues ~4 800 validity records a second naming objects this device
//! holds no mapping for, and on one of the two boots **31 of them a second**
//! carried `clear_host_valid`. That is the whole of the guest's declared account
//! of writing its own buffers, against ~20 800 gathers a second of ~1 900
//! windows whose contents are animating at 68 frames a second.
//!
//! It is not credible that the guest rewrote its vertex and constant data 31
//! times in a second it drew 15 000 frames' worth of moving geometry. The
//! likeliest reading is that **it does not have to declare** — a Metal buffer in
//! a shared storage mode needs no `didModifyRange`, because the host reads the
//! same memory — and if that is so the declaration is not a complete account and
//! `quiet_rate` is not an achievable hit rate but an upper bound on a rule that
//! would serve stale bytes.
//!
//! So the 86-89 % above is **not** a licence, and the two orders of magnitude
//! between the boots' bump rates is the reason to say so out loud rather than
//! quote the higher one.
//!
//! # The audit ran, and it closes the design — twice over
//!
//! Two driven macos-13 boots, 25 census seconds each, **21 204 real
//! comparisons** (a fold of a sampled window against the previous fold of the
//! same window, with no declared write in between):
//!
//! ```text
//!         audit_ok   audit_moved   seed   restart   unchanged share
//! boot 1     2 827         8 771      7       436             24.4 %
//! boot 2     2 915         6 691      8     1 100             30.3 %
//! ```
//!
//! Unlike [`crate::runtime::gather_witness`]'s audit — whose comparison ran **zero**
//! times across three consecutive boots, so its `unsound=0` was never a
//! measurement — this one compared twenty-one thousand times, and `restart` is
//! 4-10 % rather than the dominant column. The reading is real.
//!
//! **First conclusion: the declaration is not a complete account.** Three
//! quarters of the windows the guest declared nothing about had their bytes
//! change between two binds. One boot read `quiet_rate=1.000` with `wrote=17` in
//! a census second whose audit found 379 moved windows against 138 still ones —
//! which is exactly the false positive this module's wiring check was written to
//! catch, arriving through the other door. A cache invalidated by
//! `writeInvalidates` would have served stale vertex data on most of its hits.
//!
//! **Second conclusion, and the larger one: there was never 91 % of work to
//! remove.** `buffer_gather_working_set` measures 91 % of gathers as repeats of
//! a window already assembled, and this crate's notes have read that as the
//! size of the prize. It is not: a repeat whose bytes moved has to be re-copied
//! by any cache, sound or not. **~27 % of repeats are unchanged**, so the
//! ceiling on a content cache with a *perfect* oracle is about
//! `0.91 x 0.27 = 25 %` of this rail's gathers — ~5 100 a second of ~20 800, and
//! ~105 000 of its ~427 000 transfer regions.
//!
//! Against that, a sound cache needs the hypervisor half (a harvest cost over
//! ~1 900 windows, not a resize) and ~288 MiB of device-local memory held across
//! submissions. **Do not build it.** Three quarters of this rail's traffic is
//! the guest genuinely changing its vertex and constant data, and the way to
//! make that cheaper is to move it more cheaply — which is what the compute
//! gather does — rather than to try not to move it.
//!
//! # Why the hypervisor witness is not the instrument here
//!
//! [`crate::runtime::gather_witness`] answers the same question soundly for the sampled
//! rails, and its `MAX_TRACKED_WINDOWS` of 256 is a **harvest** bound rather
//! than a memory one: `reims_vgpu_dirty_harvest` walks every page of every armed
//! set on the BQL thread at each register write that hands the device work. The
//! buffer working set is ~1 900 windows of ~38 pages, so arming it there would
//! put ~72 000 pages into a walk the whole VM waits on. Measuring with it would
//! change what is being measured.

use std::collections::HashMap;

use super::buffer_write_gen::BufferWriteStamp;

/// Which window a bind names, at the granularity `bound_buffers` resolves.
///
/// The same four fields that key a held resolution — a reference bound at two
/// offsets is two windows and a cache would hold two buffers, so counting them
/// as one would report a hit rate for a cache nobody could build. See
/// [`super::bound_buffers`] on why the offset is the dominant axis rather than
/// an inert field.
type WindowKey = (u32, u32, u64, Option<u64>);

/// What one tracked window carries between binds.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// The stamp this window carried when it was last bound.
    stamp: BufferWriteStamp,
    /// The window's bytes as of the last audited bind, for the sampled subset
    /// only. `None` on every window the audit does not sample, and on a sampled
    /// one until its first fold.
    fold: Option<u128>,
    /// Whether the guest declared a write to this object since `fold` was taken.
    ///
    /// A fold taken before a declared write proves nothing about one taken
    /// after it — the bytes were free to move and the rule under test would have
    /// re-copied them. Comparing across it is how an audit reports a fault the
    /// design never claimed to avoid.
    wrote_since_fold: bool,
}

#[derive(Default)]
struct Window {
    /// What each window carried when it was last bound.
    ///
    /// Survives across census seconds, unlike the counters: a window bound once
    /// a second must still be comparable, and clearing this each second would
    /// report every one of them as `bgf_first` forever.
    last: HashMap<WindowKey, Entry>,
    quiet: u64,
    quiet_kb: u64,
    wrote: u64,
    wrote_kb: u64,
    first: u64,
    dropped: u64,
    audit_ok: u64,
    audit_moved: u64,
    audit_seed: u64,
    audit_restart: u64,
    audit_kb: u64,
}

/// The verdict one audited bind reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Audit {
    /// Not a sampled window, so nothing was folded.
    Skipped,
    /// Folded with no trustworthy predecessor — first sight of this window.
    Seeded,
    /// Folded, but a declared write sits between this fold and the last, so the
    /// bytes were entitled to move and the comparison says nothing.
    Restarted,
    /// Folded under an unbroken run of `quiet` binds, and the bytes agreed.
    Agreed,
    /// Folded under an unbroken run of `quiet` binds, and **the bytes moved**.
    Disagreed,
}

impl Window {
    /// The most windows tracked at once.
    ///
    /// Above the ~1 900 the working-set census measured, so the reading is not
    /// censored by its own instrument, and `dropped` says if that stops holding.
    /// This map costs one stamp per window and arms nothing on the host, which
    /// is what lets it sit an order of magnitude above `gather_witness`'s cap.
    const CAPACITY: usize = 16384;

    /// One window in this many is folded, on every bind of it.
    ///
    /// Sampling **windows** and not binds, because a stride over binds would
    /// need a run of consecutive `quiet` ones to reach a comparison — which is
    /// the trap `gather_witness`'s own audit fell into, where the comparison
    /// never once ran and its zero was read as agreement. Every bind of a
    /// sampled window folds, so the first repeat of it produces a verdict.
    ///
    /// The cost is the rate the audit reads guest bytes with the CPU, reported
    /// as `audit_kb`. At ~66 700 binds a second averaging ~180 KiB this is ~190
    /// MB/s against the rail's own ~3.5 GB/s — real, and the price of the only
    /// question that decides the design.
    const AUDIT_ONE_IN: u64 = 64;

    /// Whether this window is in the audited sample.
    ///
    /// A hash over the whole key rather than a low bit of it, **finalised**. The
    /// finaliser is not decoration: FNV alone leaves the low bits of the result
    /// a function of the low bits of the last few inputs, and an `offset` is
    /// page-aligned — its low twelve bits are zero — so a bare FNV over a family
    /// of offsets returns the *same* verdict for every one of them. That is
    /// precisely the structured slice this is supposed to avoid, and it sampled
    /// either all of such a family or none. `the_sampler_picks_about_one_window
    /// _in_its_stride` is the test that caught it, by not terminating.
    fn audited(key: WindowKey) -> bool {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in [key.0 as u64, key.1 as u64, key.2, key.3.unwrap_or(u64::MAX)] {
            h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        }
        // splitmix64's finaliser: two shift-xors around a multiply, which is
        // what carries the high bits down into the ones this then tests.
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h.is_multiple_of(Self::AUDIT_ONE_IN)
    }

    /// # Safety
    ///
    /// `runs` must satisfy [`super::gather_witness::fold_runs`]'s precondition:
    /// every `host_ptr` a live mapping of at least `len` bytes. The caller reads
    /// them at the same point in the draw the gather itself does.
    unsafe fn note(
        &mut self,
        key: WindowKey,
        stamp: BufferWriteStamp,
        bytes: u64,
        runs: &[crate::backend::vulkan::engine::GuestRun],
    ) {
        let kb = bytes / 1024;
        let (quiet, mut entry) = match self.last.get(&key) {
            Some(&earlier) if stamp.quiet_since(earlier.stamp) => {
                self.quiet += 1;
                self.quiet_kb = self.quiet_kb.saturating_add(kb);
                (true, earlier)
            }
            Some(&earlier) => {
                self.wrote += 1;
                self.wrote_kb = self.wrote_kb.saturating_add(kb);
                (false, earlier)
            }
            None if self.last.len() >= Self::CAPACITY => {
                self.dropped += 1;
                return;
            }
            None => {
                self.first += 1;
                (
                    true,
                    Entry {
                        stamp,
                        fold: None,
                        wrote_since_fold: false,
                    },
                )
            }
        };
        entry.stamp = stamp;
        entry.wrote_since_fold |= !quiet;
        // SAFETY: forwarded from this function's own precondition.
        match unsafe { self.audit(key, &mut entry, bytes, runs) } {
            Audit::Skipped => {}
            Audit::Seeded => self.audit_seed += 1,
            Audit::Restarted => self.audit_restart += 1,
            Audit::Agreed => self.audit_ok += 1,
            Audit::Disagreed => self.audit_moved += 1,
        }
        self.last.insert(key, entry);
    }

    /// Fold a sampled window and say what the fold proves.
    ///
    /// # Safety
    ///
    /// As [`Self::note`].
    unsafe fn audit(
        &mut self,
        key: WindowKey,
        entry: &mut Entry,
        bytes: u64,
        runs: &[crate::backend::vulkan::engine::GuestRun],
    ) -> Audit {
        if !Self::audited(key) {
            return Audit::Skipped;
        }
        // SAFETY: forwarded from the caller's precondition.
        let fold = unsafe { super::gather_witness::fold_runs(runs, bytes) };
        self.audit_kb = self.audit_kb.saturating_add(bytes / 1024);
        let verdict = match entry.fold {
            None => Audit::Seeded,
            Some(_) if entry.wrote_since_fold => Audit::Restarted,
            Some(previous) if previous == fold => Audit::Agreed,
            Some(_) => Audit::Disagreed,
        };
        // Re-seeded on every audited bind, including the ones a declared write
        // made unusable. That is what keeps a baseline live: an audit that only
        // re-seeded on agreement would lose its baseline at the first declared
        // write and never regain it, which is how `gather_witness`'s audit
        // reached zero comparisons in three consecutive boots.
        entry.fold = Some(fold);
        entry.wrote_since_fold = false;
        verdict
    }

    /// The line, or `None` when nothing gathered this second.
    ///
    /// Clears the counters and **keeps** `last`: the counters are a per-window
    /// rate and the stamps are the state the next second compares against.
    fn take(&mut self) -> Option<String> {
        let asked = self.quiet + self.wrote + self.first + self.dropped;
        if asked == 0 {
            return None;
        }
        let comparable = self.quiet + self.wrote;
        let rate = if comparable == 0 {
            0.0
        } else {
            self.quiet as f64 / comparable as f64
        };
        let line = format!(
            "buffer_gather_freshness quiet={} quiet_kb={} wrote={} wrote_kb={} first={} \
             dropped={} tracked={} quiet_rate={rate:.3} \
             audit_ok={} audit_moved={} audit_seed={} audit_restart={} audit_kb={} \
             (of the gathers with a previous gather of the same window to compare against, the \
              share the guest declared no write to; the ceiling on any cache invalidated by the \
              guest's own declarations, and not a licence. audit_moved is the number that closes \
              the design: bytes that moved under an unbroken run of quiet binds. Read audit_ok \
              beside it — while audit_restart dominates, the alarm is not running.)",
            self.quiet,
            self.quiet_kb,
            self.wrote,
            self.wrote_kb,
            self.first,
            self.dropped,
            self.last.len(),
            self.audit_ok,
            self.audit_moved,
            self.audit_seed,
            self.audit_restart,
            self.audit_kb,
        );
        self.quiet = 0;
        self.quiet_kb = 0;
        self.wrote = 0;
        self.wrote_kb = 0;
        self.first = 0;
        self.dropped = 0;
        self.audit_ok = 0;
        self.audit_moved = 0;
        self.audit_seed = 0;
        self.audit_restart = 0;
        self.audit_kb = 0;
        Some(line)
    }
}

fn window() -> &'static std::sync::Mutex<Window> {
    use std::sync::{Mutex, OnceLock};
    static WINDOW: OnceLock<Mutex<Window>> = OnceLock::new();
    WINDOW.get_or_init(|| Mutex::new(Window::default()))
}

/// Record one draw-time buffer bind that took the zero-copy rail.
///
/// # Safety
///
/// `runs` must satisfy [`super::gather_witness::fold_runs`]'s precondition:
/// every `host_ptr` a live mapping of at least `len` bytes. That is the same
/// precondition the gather itself relies on, and this is called at the same
/// point in the draw.
pub unsafe fn note_bind(
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    extent_cap: Option<u64>,
    stamp: BufferWriteStamp,
    bytes: u64,
    runs: &[crate::backend::vulkan::engine::GuestRun],
) {
    // SAFETY: forwarded from this function's own precondition.
    unsafe {
        window().lock().unwrap_or_else(|e| e.into_inner()).note(
            (task_id, buffer_ref, offset, extent_cap),
            stamp,
            bytes,
            runs,
        )
    };
}

/// Drain the second's split into a census line.
pub fn census() -> Option<String> {
    window().lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[cfg(test)]
mod tests {
    use super::super::buffer_write_gen::BufferWriteGens;
    use super::*;
    use crate::backend::vulkan::engine::GuestRun;

    /// A key the audit sampler does not pick, so a test about the split alone is
    /// not also a test about folding.
    fn unaudited() -> WindowKey {
        (0..4096u64)
            .map(|i| (1u32, 2u32, i * 4096, None))
            .find(|&k| !Window::audited(k))
            .expect("a sampler blind to page-aligned offsets would find none")
    }

    /// A key the audit sampler does pick.
    fn audited() -> WindowKey {
        (0..4096u64)
            .map(|i| (1u32, 2u32, i * 4096, None))
            .find(|&k| Window::audited(k))
            .expect("a sampler blind to page-aligned offsets would find none")
    }

    /// One run over `bytes`, which the fold reads in place.
    fn runs_over(bytes: &[u8]) -> Vec<GuestRun> {
        vec![GuestRun {
            host_ptr: bytes.as_ptr() as usize,
            len: bytes.len() as u64,
        }]
    }

    /// `note` with no bytes to fold — for the tests that are about the split.
    fn note(w: &mut Window, key: WindowKey, stamp: BufferWriteStamp, bytes: u64) {
        // SAFETY: an empty run slice reads nothing.
        unsafe { w.note(key, stamp, bytes, &[]) };
    }

    /// The first sight of a window is a compulsory miss and must not be counted
    /// as either side of the split — folding it in understates the achievable
    /// rate by the working set's turnover.
    #[test]
    fn a_windows_first_gather_is_neither_quiet_nor_written() {
        let mut w = Window::default();
        note(&mut w, unaudited(), BufferWriteStamp::default(), 4096);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("first=1"), "{line}");
        assert!(line.contains("quiet=0"), "{line}");
        assert!(line.contains("wrote=0"), "{line}");
    }

    /// A second gather with no declared write in between is the hit a
    /// declaration-invalidated cache would have served.
    #[test]
    fn a_repeat_with_no_declared_write_is_quiet() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = unaudited();
        note(&mut w, k, g.stamp(1, 2), 8192);
        note(&mut w, k, g.stamp(1, 2), 8192);
        let line = w.take().expect("binds happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("quiet_kb=8"), "{line}");
        assert!(line.contains("quiet_rate=1.000"), "{line}");
    }

    /// A declared write between two gathers is a copy that was owed, and it
    /// must land on the other side of the split.
    #[test]
    fn a_declared_write_between_two_gathers_is_not_quiet() {
        let mut g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = unaudited();
        note(&mut w, k, g.stamp(1, 2), 1024);
        g.note_write(1, 2);
        note(&mut w, k, g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("wrote=1"), "{line}");
        assert!(line.contains("quiet=0"), "{line}");
        assert!(line.contains("quiet_rate=0.000"), "{line}");
    }

    /// The stamps outlive the census second. A window bound once a second would
    /// otherwise read as `first` forever and the rate would be undefined for
    /// exactly the population a cache has to hold longest.
    #[test]
    fn the_stamps_survive_a_census_second_even_though_the_counters_do_not() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = unaudited();
        note(&mut w, k, g.stamp(1, 2), 1024);
        w.take().expect("a bind happened");
        note(&mut w, k, g.stamp(1, 2), 1024);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("first=0"), "{line}");
    }

    /// One reference bound at two offsets is two windows: a cache would hold two
    /// buffers, so counting them as one would report a rate for a cache nobody
    /// could build.
    #[test]
    fn one_reference_at_two_offsets_is_two_windows() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        note(&mut w, (1, 2, 0, None), g.stamp(1, 2), 1024);
        note(&mut w, (1, 2, 4096, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("first=2"), "{line}");
    }

    /// Past the capacity a new window is dropped rather than evicting one whose
    /// stamp is still wanted, and the line says how many.
    #[test]
    fn a_new_window_past_the_capacity_is_dropped_and_named() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64) {
            note(&mut w, (1, 2, i, None), g.stamp(1, 2), 1024);
        }
        note(&mut w, (9, 9, 9, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("binds happened");
        assert!(line.contains("dropped=1"), "{line}");
        assert!(
            line.contains(&format!("tracked={}", Window::CAPACITY)),
            "{line}"
        );
    }

    /// A window already tracked still counts past the capacity, so the split
    /// stays exact for the population it can see.
    #[test]
    fn a_tracked_window_still_counts_when_the_map_is_full() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64) {
            note(&mut w, (1, 2, i, None), g.stamp(1, 2), 1024);
        }
        w.take().expect("binds happened");
        note(&mut w, (1, 2, 0, None), g.stamp(1, 2), 1024);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("quiet=1"), "{line}");
        assert!(line.contains("dropped=0"), "{line}");
    }

    /// Nothing gathered is no line, so an idle second does not publish a zero
    /// that reads like a measured rate.
    #[test]
    fn an_idle_second_publishes_nothing() {
        let mut w = Window::default();
        assert!(w.take().is_none());
    }

    /// **The alarm.** Bytes that move under an unbroken run of `quiet` binds are
    /// the guest writing a buffer it did not declare, which is exactly what
    /// would make a declaration-invalidated cache serve a stale frame.
    #[test]
    fn bytes_that_move_under_a_quiet_run_are_reported_as_moved() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = audited();
        let mut bytes = vec![7u8; 4096];
        // SAFETY: `bytes` outlives both calls and the run covers exactly it.
        unsafe {
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
            bytes[0] = 9;
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
        }
        let line = w.take().expect("binds happened");
        assert!(line.contains("audit_moved=1"), "{line}");
        assert!(line.contains("audit_seed=1"), "{line}");
        assert!(line.contains("quiet=1"), "{line}");
    }

    /// Bytes that do not move under a quiet run agree, which is the reading
    /// `audit_moved`'s zero is only meaningful beside.
    #[test]
    fn bytes_that_hold_still_under_a_quiet_run_agree() {
        let g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = audited();
        let bytes = vec![7u8; 4096];
        // SAFETY: `bytes` outlives both calls and the run covers exactly it.
        unsafe {
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
        }
        let line = w.take().expect("binds happened");
        assert!(line.contains("audit_ok=1"), "{line}");
        assert!(line.contains("audit_moved=0"), "{line}");
    }

    /// A declared write between two folds entitles the bytes to move, so the
    /// comparison must be refused rather than reported as a fault the rule never
    /// claimed to avoid.
    #[test]
    fn a_declared_write_between_two_folds_refuses_the_comparison() {
        let mut g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = audited();
        let mut bytes = vec![7u8; 4096];
        // SAFETY: `bytes` outlives every call and the run covers exactly it.
        unsafe {
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
            g.note_write(1, 2);
            bytes[0] = 9;
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
        }
        let line = w.take().expect("binds happened");
        assert!(line.contains("audit_restart=1"), "{line}");
        assert!(line.contains("audit_moved=0"), "{line}");
    }

    /// The baseline is re-seeded on the refused bind too, so a single declared
    /// write does not cost the window its alarm forever — the failure that left
    /// `gather_witness`'s own audit at zero comparisons for three boots.
    #[test]
    fn a_refused_comparison_leaves_a_live_baseline_behind_it() {
        let mut g = BufferWriteGens::default();
        let mut w = Window::default();
        let k = audited();
        let mut bytes = vec![7u8; 4096];
        // SAFETY: `bytes` outlives every call and the run covers exactly it.
        unsafe {
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
            g.note_write(1, 2);
            bytes[0] = 9;
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
            bytes[1] = 11;
            w.note(k, g.stamp(1, 2), 4096, &runs_over(&bytes));
        }
        let line = w.take().expect("binds happened");
        assert!(line.contains("audit_restart=1"), "{line}");
        assert!(
            line.contains("audit_moved=1"),
            "the bind after the refusal must compare again: {line}"
        );
    }

    /// The sampler picks a spread rather than a structured slice, and it picks
    /// about one key in `AUDIT_ONE_IN`.
    #[test]
    fn the_sampler_picks_about_one_window_in_its_stride() {
        let picked = (0..6400u64)
            .filter(|&i| Window::audited((1, 2, i * 4096, None)))
            .count();
        let want = 6400 / Window::AUDIT_ONE_IN as usize;
        assert!(
            picked > want / 2 && picked < want * 2,
            "picked {picked} of 6400, wanted about {want}"
        );
    }
}
