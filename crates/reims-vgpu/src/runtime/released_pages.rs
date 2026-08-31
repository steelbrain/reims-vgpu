//! Guest pages the guest has taken back, and whether this device wrote to one
//! afterwards.
//!
//! # Why [`crate::runtime::node_guard`] is not enough
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
//! # The guard is armed at the page, not swept from a watch
//!
//! This was a `BTreeMap` of released pages, judged against the write census
//! once per drain tranche. That shape cost the watched population per tranche,
//! which is why it carried a capacity — and the capacity is what decided what
//! it could see. Every macos-13 boot logged `watching=4071 refused=58595`: the
//! instrument held 6 % of its own subject and had never once reported.
//!
//! Inverted, the release marker lives in the [`crate::runtime::host_writes`]
//! cell for that page — the same cell every writer that names its pages already
//! touches. Arming, disarming and detection are each one cell access, no sweep
//! exists, and the armed population is bounded only by the guest's own
//! releases. This module is what the readings *mean*; `host_writes` is where
//! they are kept.
//!
//! # Terminals, not a horizon
//!
//! A page stops being armed when **any** task maps it again — at which point
//! writing to it is legitimate, and keeping it would report ordinary work as a
//! defect — or when it reports. Neither is a duration chosen in advance.
//!
//! Note "any task". A guest page is guest-physical and more than one task can
//! map it — a shared IOSurface is exactly that — so a per-task watch arms a page
//! when task A unmaps it and then reports the perfectly legitimate write that
//! arrives through task B's live mapping. Armed globally, any task mapping the
//! page disarms it, and that whole class of false finding cannot arise.
//!
//! The residue, which no keying fixes: if A and B both map a page and only A
//! unmaps, the page is armed while B still holds it, and a write through B reads
//! as a finding. Counting live mappings per page would answer it, and it is not
//! done here because the count would have to be complete to be worth anything —
//! a page mapped by a route this guard does not see would make every later
//! answer for it wrong in the quiet direction. So the residue is stated, and the
//! discriminator for a specific finding is whether the page is still reachable
//! by any task at the time it fires.
//!
//! # What a finding means, and what it does not
//!
//! `released_write_after_release` means: between the guest releasing this page
//! and the write that reported it, this device wrote to it, and that write named
//! its pages. A write that names no pages cannot implicate a page or clear one;
//! those are counted separately rather than guessed at in either direction.
//!
//! It does **not** mean the write reached a page table. It means the ordering
//! this device relies on did not hold, which is the precondition for that.

/// Report the writes that landed on pages the guest had taken back.
///
/// Runs on the drain tranche. Unlike the sweep this replaced, the work here is
/// draining a queue that is empty on every tranche of every healthy boot; the
/// detection itself happened at the write.
pub fn sweep(state: &mut crate::model::DeviceState) {
    let writes = &mut state.host_writes;
    for hit in writes.take_released_writes() {
        crate::runtime::drain::note_store_route("released_write_after_release");
        if !crate::observe::first_sight("released_write_after_release", hit.gpa) {
            continue;
        }
        crate::observe::fail(format!(
            "released_pages reason=released_write_after_release gpa={:#x} \
             released_at={} wrote_at={} armed={} (this device wrote to a guest page after the \
             guest released it; the guest is entitled to have given that page to something \
             else, including its own page table)",
            hit.gpa,
            hit.released_at,
            hit.wrote_at,
            writes.armed_pages(),
        ));
    }
}

/// The guard's own levels, at most once per census interval.
///
/// `armed` is the population the guard is watching, and unlike the watch this
/// replaced it is the guest's number rather than a capacity. `dropped` non-zero
/// means findings arrived faster than the drain and the reported set is smaller
/// than the real one; `unnamed` counts the writes that could neither implicate
/// an armed page nor clear it.
///
/// **The cadence is enforced here and cannot be left to the call site.** This is
/// called from the drain tranche, beside [`sweep`], which genuinely wants to run
/// every tranche — so a levels line with no gate of its own inherits the tranche
/// rate. It did: a driven macos-13 window ran 363 tranches a second and this
/// emitted 8 297 lines over 25 s, **62 % of every line in the log**, while its
/// own doc said "on the census cadence".
///
/// The gate is [`crate::runtime::surface_cache::note_cache_levels`]'s, deliberately:
/// sharing the one-second interval is what lets a boot read this row-for-row
/// against `store_routes` and `drain_duty`.
pub fn note_levels(state: &crate::model::DeviceState) {
    static LAST_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let writes = &state.host_writes;
    if writes.armed_pages() == 0
        && writes.dropped_released_writes() == 0
        && writes.unnamed_writes_while_armed() == 0
    {
        return;
    }
    if !claim_census_interval(&LAST_MS, crate::observe::elapsed_ms() as u64) {
        return;
    }
    crate::observe::off(format!(
        "released_pages_levels armed={} dropped={} unnamed={}",
        writes.armed_pages(),
        writes.dropped_released_writes(),
        writes.unnamed_writes_while_armed(),
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
    use crate::runtime::host_writes::HostWrites;

    use super::*;

    const P: u64 = 4096;

    /// A released page nobody writes to reports nothing and stays armed — it is
    /// waiting for a write that may still come.
    #[test]
    fn a_released_page_nobody_wrote_to_stays_armed_and_quiet() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.note_pages(vec![3 * P, 4 * P]);
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.armed_pages(), 1);
    }

    /// A write after the release is the finding, it carries both epochs, and it
    /// is reported exactly once.
    #[test]
    fn a_write_after_the_release_is_reported_once() {
        let mut w = HostWrites::default();
        w.note_pages(vec![P]);
        let released_at = w.epoch();
        w.release_page(9 * P);
        w.note_pages(vec![9 * P]);

        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 9 * P);
        assert_eq!(found[0].released_at, released_at);
        assert_eq!(found[0].wrote_at, w.epoch());
        assert_eq!(
            w.armed_pages(),
            0,
            "a page that reported is not armed again"
        );

        w.note_pages(vec![9 * P]);
        assert!(
            w.take_released_writes().is_empty(),
            "one late write is one finding, not one per write for the rest of the boot"
        );
    }

    /// A write *before* the release says nothing — that is ordinary work on a
    /// page the guest still wanted us in.
    #[test]
    fn a_write_before_the_release_is_not_a_finding() {
        let mut w = HostWrites::default();
        w.note_pages(vec![9 * P]);
        w.release_page(9 * P);
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.armed_pages(), 1, "still armed, still waiting");
    }

    /// A page the guest maps again leaves the guard, so the writes that follow
    /// are ordinary work and not findings. Without this every recycled page
    /// would report.
    #[test]
    fn a_remapped_page_leaves_the_guard() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.remap_page(9 * P);
        assert_eq!(w.armed_pages(), 0);
        w.note_pages(vec![9 * P]);
        assert!(w.take_released_writes().is_empty());
    }

    /// A page released by one task and mapped by **another** is disarmed.
    ///
    /// This is the whole reason the guard is not keyed by task. A shared surface
    /// is mapped by more than one task, so keying by task would arm the page on
    /// the first unmap and then report every legitimate write arriving through
    /// the other task's live mapping — a finding per shared page per boot, all
    /// of them wrong, in an instrument whose only value is that a hit is a
    /// proof. The guard is page-keyed, so there is no task to key by.
    #[test]
    fn a_page_mapped_again_by_a_different_task_is_disarmed() {
        let mut w = HostWrites::default();
        w.release_page(9 * P);
        w.remap_page(9 * P);
        w.note_pages(vec![9 * P]);
        assert!(
            w.take_released_writes().is_empty(),
            "a write through the other task's live mapping is ordinary work"
        );
    }

    /// Releasing a page twice keeps the first epoch, so a write that already
    /// happened is not forgiven by the second release.
    #[test]
    fn a_second_release_does_not_forgive_a_write_that_already_landed() {
        let mut w = HostWrites::default();
        w.note_pages(vec![P]);
        let first = w.epoch();
        w.release_page(9 * P);
        w.note_pages(vec![7 * P]);
        w.release_page(9 * P);
        w.note_pages(vec![9 * P]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(
            found[0].released_at, first,
            "the finding is measured from the first release, not the second"
        );
    }

    /// A write that named no pages cannot judge an armed one, in either
    /// direction, and says so rather than reading as a clean sheet.
    #[test]
    fn an_unnamed_write_is_counted_and_is_not_a_finding() {
        let mut w = HostWrites::default();
        w.note_unknown();
        assert_eq!(
            w.unnamed_writes_while_armed(),
            0,
            "nothing armed, nothing to be undecided about"
        );
        w.release_page(9 * P);
        w.note_unknown();
        assert!(w.take_released_writes().is_empty());
        assert_eq!(w.unnamed_writes_while_armed(), 1);
        assert_eq!(
            w.armed_pages(),
            1,
            "an unnamed write neither clears nor arms"
        );
    }

    /// **The population is the guest's, not a capacity.** The watch this
    /// replaced held 4096 pages and every macos-13 boot refused 58 595 of the
    /// 62 666 the guest released — 93 % of the instrument's own subject, unseen.
    /// A release of that size is now armed in full and each page still answers.
    #[test]
    fn sixty_thousand_released_pages_are_all_armed_and_all_answer() {
        const N: u64 = 62_666;
        let mut w = HostWrites::default();
        w.note_pages(vec![0]);
        for page in 1..=N {
            w.release_page(page * P);
        }
        assert_eq!(w.armed_pages(), N);
        // The last page released is well past any capacity the watch had.
        w.note_pages(vec![N * P]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, N * P);
        assert_eq!(w.armed_pages(), N - 1);
    }

    /// A page armed before this device has ever written is still armed. Epoch 0
    /// means "never written", so an arm that stored it would be indistinguishable
    /// from an unarmed cell and the guard would be blind for the whole first
    /// write of a boot.
    #[test]
    fn a_page_released_before_the_first_write_is_still_armed() {
        let mut w = HostWrites::default();
        assert_eq!(w.epoch(), 0, "nothing written yet");
        w.release_page(9 * P);
        assert_eq!(w.armed_pages(), 1);
        w.note_pages(vec![9 * P]);
        assert_eq!(w.take_released_writes().len(), 1);
    }

    /// A whole-chunk write is the one path that does not touch cells one by one,
    /// and it must still find an armed page inside the chunk it covers.
    #[test]
    fn a_whole_chunk_write_finds_an_armed_page_inside_it() {
        let pages: std::sync::Arc<[u64]> =
            (0..512u64).map(|page| page * P).collect::<Vec<_>>().into();
        let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(pages, P)
            .expect("one contiguous allocation");
        let mut w = HostWrites::default();
        w.release_page(300 * P);
        w.note_footprint(&footprint);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 300 * P);
    }

    /// The report queue bounds the alarms, not the guard. What it cannot hold is
    /// counted, and every page it did not report is still disarmed exactly once.
    #[test]
    fn a_flood_of_findings_is_counted_rather_than_lost_silently() {
        let mut w = HostWrites::default();
        w.note_pages(vec![0]);
        let pages: Vec<u64> = (1..=200u64).map(|page| page * P).collect();
        for &gpa in &pages {
            w.release_page(gpa);
        }
        w.note_pages(pages);
        let found = w.take_released_writes();
        assert_eq!(found.len() as u64 + w.dropped_released_writes(), 200);
        assert!(
            w.dropped_released_writes() > 0,
            "200 findings, a queue of 64"
        );
        assert_eq!(
            w.armed_pages(),
            0,
            "a dropped report still disarms its page"
        );
    }

    /// Arm64's wider pages occupy one cell, so a release and a write of the same
    /// guest page cannot land in different cells.
    #[test]
    fn arm64_pages_arm_and_report_at_arm64_geometry() {
        const ARM_PAGE: u64 = 1 << crate::model::PAGE_SHIFT_ARM64E;
        let mut w = HostWrites::new(crate::model::PAGE_SHIFT_ARM64E);
        w.release_page(3 * ARM_PAGE);
        // An address inside the same 16 KiB page, not at its base.
        w.note_pages(vec![3 * ARM_PAGE + 4096]);
        let found = w.take_released_writes();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].gpa, 3 * ARM_PAGE);
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
