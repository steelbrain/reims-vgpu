//! Guest pages the guest has taken back, and whether this device wrote to one
//! afterwards.
//!
//! # Why this exists, and why [`crate::runtime::node_guard`] is not enough
//!
//! `node_guard` asks whether a host write landed on a page that **was already**
//! a page-table node. On a boot that panicked it read zero, and that result has
//! a blind spot which is now the leading reading of the whole defect:
//!
//! 1. Page `P` is an ordinary data page of task `T`, mapped for a surface.
//! 2. The guest sends `UnmapMemory`, and this device stamps it complete.
//! 3. The guest frees `P` and the allocator hands it back as a **page-table
//!    node** for some other tree.
//! 4. Work this device had in flight over step 1's mapping lands on `P`.
//!
//! `node_guard` first sees `P` as a node at step 3 or later, so the write at
//! step 4 is before its first sighting and reads as `FirstSight` rather than as
//! a finding.
//!
//! **The ordering claim this paragraph used to make is wrong and is worth not
//! repeating.** It said the guest submits the unmap, blocks on this device's
//! reply, and only then unwires — so that our stamp opened step 4. It does not
//! block: the submit and the unwire are adjacent, and measured, the guest
//! finishes unwiring before this device even reads the packet about nineteen
//! times in twenty. Nothing this device replies opens that window; the guest
//! opens it itself.
//!
//! So this watches the other end. A page released by the guest is a page this
//! device has been told to stop writing, whatever it later becomes, and a write
//! to one is a defect on its own terms — it does not need the page to have
//! become a page table for the answer to be "we wrote where we were told not
//! to". That the corrupting value must be a **zero** word for the guest's
//! assertion to fire is a property of the panic, not of this check.
//!
//! # Terminals, not a horizon
//!
//! A watched page stops being watched when **any** task maps it again — at which
//! point writing to it is legitimate, and keeping it would report ordinary work
//! as a defect — or when it reports. Neither is a duration chosen in advance.
//! That matters here for the same reason it did for
//! [`crate::runtime::objects::slot_recheck`]: a horizon would have to come from
//! somewhere, and the number would end up deciding the answer.
//!
//! Note "any task", not "the task that released it". See [`ReleasedPages`] for
//! why the watch is not keyed by task, which is the difference between a finding
//! worth acting on and one that is just a shared surface.
//!
//! # What it costs
//!
//! The page list of a released range, resolved through the run walker that
//! already batches its deepest level, and one lookup per watched page per census
//! sweep. The resolve happens **before** the unmap is applied, which is the only
//! moment those addresses still translate.
//!
//! # What a finding means, and what it does not
//!
//! `released_write` means: between the guest releasing this page and now, this
//! device wrote to it, and the write named its pages. Only
//! [`HostWriteVerdict::Overlap`] counts, for the reason `node_guard` states —
//! this is an alarm, and an alarm that fires on what it cannot decide is worse
//! than no alarm.
//!
//! It does **not** mean the write reached a page table. It means the ordering
//! this device relies on did not hold, which is the precondition for that.

use std::collections::BTreeMap;

use crate::runtime::host_writes::{HostWriteVerdict, HostWrites};

/// How many released pages the watch will hold.
///
/// A single release can cover tens of megabytes, so this is a bound on the
/// watch and not on the guest. A page that does not fit is **refused and
/// counted** — see [`ReleasedPages::refused`] — because a quiet drop would
/// shrink the watched population while the readings kept their shape, which is
/// the failure that reads as a clean sweep.
const WATCH_CAP: usize = 4096;

/// What a sweep found for one released page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasedVerdict {
    /// Nothing this device recorded has touched the page since it was released.
    Quiet,
    /// **This device wrote to a page the guest had taken back.**
    Wrote { since_us: u64 },
    /// A write in the window named no pages, so this page cannot be judged.
    Undecidable,
}

impl ReleasedVerdict {
    /// Whether this is the reading the module exists to find.
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Wrote { .. })
    }

    /// The counter name, one per variant and exhaustive.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "released_quiet",
            Self::Wrote { .. } => "released_write_after_release",
            Self::Undecidable => "released_undecidable",
        }
    }
}

/// When a page was released, and by which task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Released {
    /// The [`HostWrites`] epoch current at the release. Any write carrying a
    /// higher epoch landed after the guest took the page back.
    epoch: u64,
    /// `crate::observe::elapsed_us` at the release.
    at_us: u64,
    /// The task whose unmap armed it, carried for the finding's own line rather
    /// than for keying — see why the watch is not per task.
    task_id: u32,
}

/// The released pages, across every task.
///
/// **Global on purpose, and this is the correction that makes a finding
/// trustworthy.** A guest page is guest-physical and more than one task can map
/// it — a shared IOSurface is exactly that — so a per-task watch arms a page
/// when task A unmaps it and then reports the perfectly legitimate write that
/// arrives through task B's live mapping. Keyed globally, any task mapping the
/// page disarms it, and that whole class of false finding cannot arise.
///
/// The residue, which no keying fixes: if A and B both map a page and only A
/// unmaps, the page is armed while B still holds it, and a write through B reads
/// as a finding. Counting live mappings per page would answer it, and it is not
/// done here because the count would have to be complete to be worth anything —
/// a page mapped by a route this watch does not see would make every later
/// answer for it wrong in the quiet direction. So the residue is stated, and the
/// discriminator for a specific finding is whether the page is still reachable
/// by any task at the time it fires.
#[derive(Default, Debug)]
pub struct ReleasedPages {
    pages: BTreeMap<u64, Released>,
    refused: u64,
}

impl ReleasedPages {
    /// Record that the guest has taken `gpa` back.
    ///
    /// A page released twice without an intervening map keeps its **first**
    /// release epoch: the question is whether anything was written since the
    /// guest stopped wanting us there, and re-stamping it would forgive a write
    /// that had already happened.
    pub fn release(&mut self, writes: &HostWrites, task_id: u32, gpa: u64, now_us: u64) {
        if self.pages.contains_key(&gpa) {
            return;
        }
        if self.pages.len() >= WATCH_CAP {
            self.refused += 1;
            return;
        }
        self.pages.insert(
            gpa,
            Released {
                epoch: writes.epoch(),
                at_us: now_us,
                task_id,
            },
        );
    }

    /// The guest has mapped `gpa` again, so writing to it is legitimate.
    pub fn remapped(&mut self, gpa: u64) {
        self.pages.remove(&gpa);
    }

    /// Judge every watched page and report only the ones that answered.
    ///
    /// A page that reports is removed so that one late write is one finding
    /// rather than one per sweep for the rest of the boot. A `Quiet` page stays,
    /// because the write it is waiting for has not happened *yet* — and it is
    /// **not** reported: a quiet page is the normal state of every watched page
    /// on every sweep, so returning them makes the output the product of the
    /// watch size and the tranche count. That is not a hypothetical. Returning
    /// them cost one boot **104 million** counter increments, each taking the
    /// census mutex, from an instrument whose entire job is to not perturb a
    /// race. The watch size is reported as a level instead.
    pub fn sweep(
        &mut self,
        writes: &HostWrites,
        now_us: u64,
    ) -> Vec<(u64, u32, ReleasedVerdict)> {
        let mut out = Vec::new();
        self.pages.retain(|&gpa, rel| {
            let verdict = match writes.wrote_any_since(rel.epoch, &[gpa]) {
                HostWriteVerdict::Quiet => return true,
                HostWriteVerdict::Overlap => ReleasedVerdict::Wrote {
                    since_us: now_us.saturating_sub(rel.at_us),
                },
                _ => ReleasedVerdict::Undecidable,
            };
            out.push((gpa, rel.task_id, verdict));
            false
        });
        out
    }

    /// How many pages are being watched.
    pub fn watched(&self) -> usize {
        self.pages.len()
    }

    /// How many releases this watch turned away because it was full.
    pub fn refused(&self) -> u64 {
        self.refused
    }
}

/// Judge every task's released pages and report the writes that landed after
/// the guest took a page back.
///
/// Runs on the drain tranche, beside
/// [`crate::runtime::objects::slot_recheck::sweep`] and for the same reason: it
/// returns immediately when nothing is watched, which is every rail on which the
/// guest does not release pages under load.
pub fn sweep(state: &mut crate::model::DeviceState) {
    let now_us = crate::observe::elapsed_us();
    let crate::model::DeviceState {
        host_writes,
        released_pages: watch,
        ..
    } = state;
    if watch.watched() == 0 {
        return;
    }
    for (gpa, task_id, verdict) in watch.sweep(host_writes, now_us) {
        crate::runtime::drain::note_store_route(verdict.route());
        if let ReleasedVerdict::Wrote { since_us } = verdict {
            if crate::observe::first_sight("released_write_after_release", gpa) {
                crate::observe::fail(format!(
                    "released_pages reason={} task={task_id} gpa={gpa:#x} \
                     since_us={since_us} watched={} refused={} (this device wrote to a guest \
                     page after the guest released it; the guest is entitled to have given \
                     that page to something else, including its own page table)",
                    verdict.route(),
                    watch.watched(),
                    watch.refused(),
                ));
            }
        }
    }
}

/// The watch's own size, at most once per census interval.
///
/// A level rather than a per-page count, because the population is the thing
/// worth knowing and counting each quiet page every sweep is what made this
/// instrument expensive. `refused` non-zero means the readings describe a
/// smaller set than the guest released.
///
/// **The cadence is enforced here and cannot be left to the call site.** This is
/// called from the drain tranche, beside [`sweep`], which genuinely wants to run
/// every tranche — so a levels line with no gate of its own inherits the tranche
/// rate. It did: a driven macos-13 window ran 363 tranches a second and this
/// emitted 8 297 lines over 25 s, **62 % of every line in the log**, while its
/// own doc said "on the census cadence". A level that is sampled 363 times a
/// second says nothing a once-a-second sample does not, and it is a `format!`
/// and a sink write per tranche on the drain worker's critical path.
///
/// The gate is [`crate::runtime::surface_cache::note_cache_levels`]'s, deliberately:
/// sharing the one-second interval is what lets a boot read this row-for-row
/// against `store_routes` and `drain_duty`. Unlike that one, the values here are
/// two loads rather than a cache walk, so the gate goes *before* them and the
/// quiet tranche costs one clock read.
pub fn note_levels(state: &crate::model::DeviceState) {
    static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let watch = &state.released_pages;
    if watch.watched() == 0 && watch.refused() == 0 {
        return;
    }
    if !claim_census_interval(&LAST_MS, crate::observe::elapsed_ms() as u64) {
        return;
    }
    crate::observe::off(format!(
        "released_pages_levels watching={} refused={} capacity={WATCH_CAP}",
        watch.watched(),
        watch.refused(),
    ));
}

/// The census interval every levels line in this device shares.
///
/// One second, so `released_pages_levels`, `cache_levels`, `store_routes` and
/// `drain_duty` all describe the same window and can be read as one row.
const CENSUS_INTERVAL_MS: u64 = 1_000;

/// Atomically claim the next census interval, or refuse.
///
/// Split out and given the clock as an argument for the same reason
/// `drain::claim_display_vbl` is: a rate gate written inline against
/// `elapsed_ms()` can only be checked by a boot, and this one was wrong for as
/// long as it was inline. Here it is a pure function of `(last, now)` and the
/// tests below pin both edges.
///
/// The claimed timestamp is set to `now` rather than advanced by one interval —
/// the opposite of the VBL grid, and deliberately. A VBL is a cadence the guest
/// latches onto, so its phase must not drift; this is a sample of a level, where
/// landing exactly on a grid buys nothing and back-dating would let a burst of
/// tranches after a long stall each emit a line.
fn claim_census_interval(last_ms: &std::sync::atomic::AtomicU64, now_ms: u64) -> bool {
    use std::sync::atomic::Ordering;
    let last = last_ms.load(Ordering::Relaxed);
    if now_ms.saturating_sub(last) < CENSUS_INTERVAL_MS {
        return false;
    }
    // Losing the race only costs a skipped interval, never a double line.
    last_ms
        .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    /// A released page nobody writes to stays watched and stays quiet — it is
    /// waiting for a write that may still come.
    #[test]
    fn a_released_page_nobody_wrote_to_stays_quiet_and_stays_watched() {
        let mut r = ReleasedPages::default();
        let writes = HostWrites::default();
        r.release(&writes, 7, 9 * P, 0);
        for t in 1..4 {
            assert!(
                r.sweep(&writes, t).is_empty(),
                "a quiet page is not reported — it is the normal state of every watched page"
            );
            assert_eq!(r.watched(), 1, "and it stays watched");
        }
    }

    /// A write after the release is the finding, it carries how long after, and
    /// it is reported exactly once.
    #[test]
    fn a_write_after_the_release_is_reported_once() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 7, 9 * P, 100);
        writes.note_pages(vec![9 * P]);

        let found = r.sweep(&writes, 700);
        assert_eq!(found, vec![(9 * P, 7, ReleasedVerdict::Wrote { since_us: 600 })]);
        assert!(found[0].2.is_finding());
        assert_eq!(r.watched(), 0, "a page that reported is not watched again");
        assert!(r.sweep(&writes, 800).is_empty());
    }

    /// A write *before* the release says nothing — that is ordinary work on a
    /// page the guest still wanted us in.
    #[test]
    fn a_write_before_the_release_is_not_a_finding() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        writes.note_pages(vec![9 * P]);
        r.release(&writes, 7, 9 * P, 0);
        assert!(r.sweep(&writes, 1).is_empty());
        assert_eq!(r.watched(), 1, "still watched, still waiting");
    }

    /// A page the guest maps again leaves the watch, so the writes that follow
    /// are ordinary work and not findings. Without this every recycled page
    /// would report.
    #[test]
    fn a_remapped_page_leaves_the_watch() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 7, 9 * P, 0);
        r.remapped(9 * P);
        assert_eq!(r.watched(), 0);
        writes.note_pages(vec![9 * P]);
        assert!(r.sweep(&writes, 1).is_empty());
    }

    /// Releasing a page twice keeps the first epoch, so a write that already
    /// happened is not forgiven by the second release.
    #[test]
    fn a_second_release_does_not_forgive_a_write_that_already_landed() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 7, 9 * P, 0);
        writes.note_pages(vec![9 * P]);
        r.release(&writes, 7, 9 * P, 10);
        assert_eq!(
            r.sweep(&writes, 20),
            vec![(9 * P, 7, ReleasedVerdict::Wrote { since_us: 20 })],
            "the gap is measured from the first release, not the second"
        );
    }

    /// A write that named no pages cannot judge this one, and is not a finding.
    #[test]
    fn an_unnamed_write_is_undecidable() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 7, 9 * P, 0);
        writes.note_unknown();
        let found = r.sweep(&writes, 1);
        assert_eq!(found, vec![(9 * P, 7, ReleasedVerdict::Undecidable)]);
        assert!(!found[0].2.is_finding());
    }

    /// The watch stops at its capacity and counts what it turned away.
    #[test]
    fn a_full_watch_refuses_and_says_how_often() {
        let mut r = ReleasedPages::default();
        let writes = HostWrites::default();
        for i in 0..WATCH_CAP as u64 {
            r.release(&writes, 7, i * P, 0);
        }
        assert_eq!(r.watched(), WATCH_CAP);
        for i in 0..5u64 {
            r.release(&writes, 7, (WATCH_CAP as u64 + i) * P, 0);
        }
        assert_eq!(r.refused(), 5);
        assert_eq!(r.watched(), WATCH_CAP, "a refusal does not evict");
    }

    /// A page released by one task and mapped by **another** is disarmed.
    ///
    /// This is the whole reason the watch is not keyed by task. A shared surface
    /// is mapped by more than one task, so keying by task would arm the page on
    /// the first unmap and then report every legitimate write arriving through
    /// the other task's live mapping — a finding per shared page per boot, all
    /// of them wrong, in an instrument whose only value is that a hit is a
    /// proof.
    #[test]
    fn a_page_mapped_again_by_a_different_task_is_disarmed() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 3, 9 * P, 0);
        // Task 8, not task 3, maps it.
        r.remapped(9 * P);
        assert_eq!(r.watched(), 0);
        writes.note_pages(vec![9 * P]);
        assert!(
            r.sweep(&writes, 1).is_empty(),
            "a write through the other task's live mapping is ordinary work"
        );
    }

    /// The finding carries the task whose unmap armed the page, so a reading
    /// names something even though the watch is global.
    #[test]
    fn a_finding_names_the_task_that_released_the_page() {
        let mut r = ReleasedPages::default();
        let mut writes = HostWrites::default();
        r.release(&writes, 21, 4 * P, 0);
        writes.note_pages(vec![4 * P]);
        let found = r.sweep(&writes, 5);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].1, 21);
    }

    /// Every verdict names itself, and exactly one of them is the finding.
    #[test]
    fn every_verdict_names_itself() {
        let all = [
            ReleasedVerdict::Quiet,
            ReleasedVerdict::Wrote { since_us: 0 },
            ReleasedVerdict::Undecidable,
        ];
        let mut names: Vec<&str> = all.iter().map(|v| v.route()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "two verdicts share a route name");
        assert_eq!(all.iter().filter(|v| v.is_finding()).count(), 1);
    }

    /// The gate that makes this a census line and not a per-tranche one.
    ///
    /// A driven macos-13 window runs ~363 drain tranches a second and
    /// `note_levels` is called from every one of them, so without a claim here
    /// the level is emitted 363 times a second. This asserts the tranche rate
    /// cannot get through: 363 calls spread across one second yield one line.
    #[test]
    fn a_seconds_worth_of_tranches_claims_one_census_interval() {
        let last = std::sync::atomic::AtomicU64::new(0);
        // Start at a nonzero clock so the first call is not a special case of
        // the zero initialiser.
        let base = 5_000;
        assert!(claim_census_interval(&last, base), "the first sample emits");

        // Up to 362, not 363: the 363rd tranche lands exactly on `base + 1000`,
        // which is a full interval later and is *supposed* to claim. It is the
        // next assertion, not part of this one.
        let claims = (1..363)
            .filter(|i| claim_census_interval(&last, base + (1000 * i) / 363))
            .count();
        assert_eq!(claims, 0, "a tranche inside the interval must not emit");

        assert!(
            claim_census_interval(&last, base + CENSUS_INTERVAL_MS),
            "the tranche that reaches the next interval emits"
        );
    }

    /// A stall does not bank intervals it slept through.
    ///
    /// The claim moves to `now`, not forward by one interval, so the tranches
    /// that arrive in a burst after a long drain stall produce one line between
    /// them rather than one per interval the stall covered.
    #[test]
    fn a_long_stall_does_not_release_a_burst_of_lines() {
        let last = std::sync::atomic::AtomicU64::new(0);
        assert!(claim_census_interval(&last, 1_000));
        // Ten intervals pass inside one tranche, then the burst arrives.
        assert!(claim_census_interval(&last, 11_000));
        let banked = (1..=10)
            .filter(|i| claim_census_interval(&last, 11_000 + i))
            .count();
        assert_eq!(banked, 0, "the stall must not bank a line per interval");
    }
}
