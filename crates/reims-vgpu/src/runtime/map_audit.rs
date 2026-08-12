//! The guest's map/unmap intervals, audited against the invariant its own page
//! table asserts.
//!
//! # Why this exists
//!
//! The guest keeps a three-level page table in guest pages this device reads on
//! every translation. Its teardown walks the range it is given one page at a
//! time and **refuses to clear a leaf entry that is already zero** — an
//! assertion that panics the guest. On one guest line it fires on about half of
//! undriven boots.
//!
//! This device never writes those pages, so it cannot violate that invariant
//! directly. What it *can* do is drive the guest into violating it, and the wire
//! gives us the means to check the most reachable way:
//!
//! **The guest's own `allocate` and `deallocate` take their length from the same
//! call whose value the map packet carries**, and their address from the getter
//! the packet's address field is built from. So the intervals in the FIFO are
//! exactly the intervals the guest applies to its tree. A range unmapped that was
//! never mapped, unmapped twice, mapped over itself, or unmapped at a different
//! length reaches that assertion directly.
//!
//! # What it has answered, and the two things it cannot see
//!
//! Clean on every boot it has been read on, including a dozen that panicked. So
//! no *drained* teardown has ever been a double.
//!
//! **Read that only as far as the census lets you.** Every verdict is counted
//! into `store_routes` under [`MapAudit::slug`], `map_audit_consistent`
//! included, and that counter is the whole basis for the sentence above. The
//! fail line fires only on a finding and is deduped per `(task, channel)` on top
//! of that, so its absence on its own says nothing about whether a single packet
//! was ever audited. A boot whose log carries no `map_audit_consistent` audited
//! nothing, and its silence is not a clean reading. Check for the counter before
//! quoting a zero: the readings taken before the counter existed were quoted
//! that way, and could not tell the two apart.
//!
//! **The fatal one is not drained.** The guest submits an unmap and unwires
//! immediately afterwards — measured, it beats this device to the range about
//! nineteen times in twenty — so a release that panics the guest does so inside
//! the window between its submit and this device's read. That packet is never
//! decoded, never audited and never counted. Read this instrument's zero as
//! covering every teardown that did not end the boot, and nothing more.
//!
//! # This is an instrument, not a repair
//!
//! Nothing here changes what the device does. It watches the packet stream and
//! names a disagreement, so that the question "is this a pairing bug or a race?"
//! is answered by a reading rather than by argument. A clean run does not prove
//! the pairing is right — it moves the weight onto the other hypotheses, which
//! is what it is for.
//!
//! A sub-page length is worth naming for its own reason: the guest's `allocate`
//! returns silently for anything under one page, so a range mapped at less than a
//! page and released at more than one is unmapped having never been mapped.

use std::collections::BTreeMap;

/// The guest page size the audit reasons in.
///
/// Taken as a parameter rather than assumed: this device serves both a 4 KiB and
/// a 16 KiB guest, and the sub-page rule is about the guest's own allocator, not
/// about the host's.
pub type PageSize = u64;

/// What a map or unmap packet did to a task's interval set.
///
/// Everything except [`Self::Consistent`] is a disagreement between the guest's
/// map stream and its own page-table lifetime, and each names the shape rather
/// than a slug, so the caller decides how loudly to say it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapAudit {
    /// The packet is consistent with everything seen for this task.
    Consistent,
    /// A map whose range intersects one already mapped and not released.
    ///
    /// Two live mappings over one guest page means one `deallocate` clears a PTE
    /// the other still believes in.
    OverlapsLive { gva: u64, len: u64 },
    /// An unmap for a range with no live mapping at that address.
    UnmapOfUnmapped,
    /// An unmap at an address that is mapped, but with a different length.
    ///
    /// The guest walks `len` pages from `gva`; a longer unmap clears PTEs beyond
    /// what was mapped and a shorter one leaves a node non-empty that the
    /// mapping side expected to collapse.
    LengthMismatch { mapped_len: u64 },
    /// A length below one guest page, which the guest's allocator ignores.
    SubPage { len: u64 },
    /// Tracking for this task was abandoned because the live set exceeded
    /// [`MapIntervals::MAX_LIVE`].
    ///
    /// Reported rather than dropped silently: once this fires, a later
    /// [`Self::Consistent`] for the task means nothing, and a reader must not
    /// take the absence of findings as evidence.
    TrackingAbandoned,
}

impl MapAudit {
    /// Whether this reading is a disagreement worth reporting.
    pub fn is_finding(self) -> bool {
        !matches!(self, Self::Consistent)
    }

    /// A stable name for this verdict, used both as the fail-channel reason and
    /// as the `store_routes` counter.
    ///
    /// One spelling on purpose. The fail line is emitted only for a finding and
    /// deduped on top of that, while the counter is bumped for every verdict —
    /// so the two say different things about the same reading and a second
    /// spelling would let them drift apart silently.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Consistent => "map_audit_consistent",
            Self::OverlapsLive { .. } => "map_audit_overlaps_live",
            Self::UnmapOfUnmapped => "map_audit_unmap_of_unmapped",
            Self::LengthMismatch { .. } => "map_audit_length_mismatch",
            Self::SubPage { .. } => "map_audit_sub_page",
            Self::TrackingAbandoned => "map_audit_tracking_abandoned",
        }
    }
}

/// One task's live guest-VA mappings, keyed by start address.
///
/// The bound is declared and its exhaustion is a reported state rather than a
/// silent drop — see [`MapAudit::TrackingAbandoned`]. It is generous because a
/// live mapping set is proportional to what the guest is actually using, and
/// exceeding it is itself worth knowing about.
#[derive(Clone, Debug, Default)]
pub struct MapIntervals {
    live: BTreeMap<u64, u64>,
    abandoned: bool,
}

impl MapIntervals {
    /// The most live mappings one task may hold before the audit stops tracking.
    pub const MAX_LIVE: usize = 1 << 16;

    pub const fn new() -> Self {
        Self {
            live: BTreeMap::new(),
            abandoned: false,
        }
    }

    /// How many mappings this task currently holds.
    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    /// Record a `MapMemory2` and say whether it disagrees with what is live.
    pub fn map(&mut self, gva: u64, len: u64, page_size: PageSize) -> MapAudit {
        if self.abandoned {
            return MapAudit::TrackingAbandoned;
        }
        if len < page_size {
            // Recorded anyway: the guest's allocator ignored it, so a later
            // unmap of the same range is the interesting event and it needs
            // this entry to be recognised as such.
            self.insert(gva, len);
            return MapAudit::SubPage { len };
        }
        if let Some((o_gva, o_len)) = self.first_overlap(gva, len) {
            // Still recorded, replacing the old extent, because the guest's tree
            // now reflects this map whatever we think of it. Refusing to track
            // it would turn one finding into a cascade of unmap-of-unmapped.
            self.insert(gva, len);
            return MapAudit::OverlapsLive {
                gva: o_gva,
                len: o_len,
            };
        }
        self.insert(gva, len);
        if self.live.len() > Self::MAX_LIVE {
            self.abandoned = true;
            self.live.clear();
            return MapAudit::TrackingAbandoned;
        }
        MapAudit::Consistent
    }

    /// Record an `UnmapMemory` and say whether it disagrees with what is live.
    pub fn unmap(&mut self, gva: u64, len: u64) -> MapAudit {
        if self.abandoned {
            return MapAudit::TrackingAbandoned;
        }
        match self.live.remove(&gva) {
            None => MapAudit::UnmapOfUnmapped,
            Some(mapped_len) if mapped_len != len => MapAudit::LengthMismatch { mapped_len },
            Some(_) => MapAudit::Consistent,
        }
    }

    /// Drop everything for a task the guest deleted.
    ///
    /// A task's whole address space goes with it, so the mappings that were live
    /// are not leaks and must not be reported as such when the id is reused.
    pub fn clear(&mut self) {
        self.live.clear();
        self.abandoned = false;
    }

    fn insert(&mut self, gva: u64, len: u64) {
        self.live.insert(gva, len);
    }

    /// The first live mapping intersecting `[gva, gva+len)`, if any.
    ///
    /// Checks the nearest mapping at or below `gva` and then walks forward from
    /// `gva`, which is sufficient because the ranges are half-open and stored by
    /// start: anything intersecting either starts before `gva` and reaches into
    /// it, or starts within it.
    fn first_overlap(&self, gva: u64, len: u64) -> Option<(u64, u64)> {
        let end = gva.saturating_add(len);
        if let Some((&p_gva, &p_len)) = self.live.range(..=gva).next_back() {
            if p_gva.saturating_add(p_len) > gva {
                return Some((p_gva, p_len));
            }
        }
        self.live
            .range(gva..end)
            .next()
            .map(|(&g, &l)| (g, l))
            .filter(|_| len != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    #[test]
    fn a_matched_map_and_unmap_is_consistent() {
        let mut m = MapIntervals::new();
        assert_eq!(m.map(0x1000, PAGE * 4, PAGE), MapAudit::Consistent);
        assert_eq!(m.unmap(0x1000, PAGE * 4), MapAudit::Consistent);
        assert_eq!(m.live_count(), 0);
    }

    /// The shape that produces the guest's assertion: a range released twice.
    ///
    /// The second release walks a leaf whose PTEs this device already saw
    /// cleared, so the guest's `entriesSet` underflows past the point its
    /// interior node expected to collapse.
    #[test]
    fn a_second_unmap_of_one_range_is_named() {
        let mut m = MapIntervals::new();
        m.map(0x8000, PAGE, PAGE);
        assert_eq!(m.unmap(0x8000, PAGE), MapAudit::Consistent);
        assert_eq!(m.unmap(0x8000, PAGE), MapAudit::UnmapOfUnmapped);
    }

    #[test]
    fn an_unmap_of_a_range_never_mapped_is_named() {
        let mut m = MapIntervals::new();
        assert_eq!(m.unmap(0xdead_0000, PAGE), MapAudit::UnmapOfUnmapped);
    }

    /// A release longer than its map clears PTEs past what was ever set.
    #[test]
    fn an_unmap_at_a_different_length_is_named() {
        let mut m = MapIntervals::new();
        m.map(0x2000, PAGE, PAGE);
        assert_eq!(
            m.unmap(0x2000, PAGE * 2),
            MapAudit::LengthMismatch { mapped_len: PAGE }
        );
    }

    #[test]
    fn a_map_over_a_live_range_is_named_whichever_end_overlaps() {
        // A later map starting inside an earlier one.
        let mut m = MapIntervals::new();
        m.map(0x10000, PAGE * 4, PAGE);
        assert_eq!(
            m.map(0x11000, PAGE, PAGE),
            MapAudit::OverlapsLive {
                gva: 0x10000,
                len: PAGE * 4
            }
        );

        // An earlier map reaching forward into a later one.
        let mut m = MapIntervals::new();
        m.map(0x20000, PAGE, PAGE);
        assert_eq!(
            m.map(0x1f000, PAGE * 4, PAGE),
            MapAudit::OverlapsLive {
                gva: 0x20000,
                len: PAGE
            }
        );
    }

    /// Adjacent ranges share no page, so tiling an address space is not an
    /// overlap — the common case must stay quiet or the instrument is noise.
    #[test]
    fn adjacent_ranges_do_not_overlap() {
        let mut m = MapIntervals::new();
        assert_eq!(m.map(0x1000, PAGE, PAGE), MapAudit::Consistent);
        assert_eq!(m.map(0x2000, PAGE, PAGE), MapAudit::Consistent);
        assert_eq!(m.map(0x3000, PAGE * 8, PAGE), MapAudit::Consistent);
        assert_eq!(m.live_count(), 3);
    }

    /// The guest's allocator ignores a sub-page length, so the map never
    /// happened in its tree even though the packet says it did.
    #[test]
    fn a_sub_page_length_is_named_and_still_tracked() {
        let mut m = MapIntervals::new();
        assert_eq!(m.map(0x5000, 8, PAGE), MapAudit::SubPage { len: 8 });
        // Tracked, so the release of the same range is recognised rather than
        // reported as an unmap of something unmapped.
        assert_eq!(m.unmap(0x5000, 8), MapAudit::Consistent);
    }

    /// A 16 KiB guest's page is the sub-page threshold, not the host's.
    #[test]
    fn the_sub_page_rule_follows_the_guest_page_size() {
        let mut m = MapIntervals::new();
        assert_eq!(m.map(0x4000, 4096, 16384), MapAudit::SubPage { len: 4096 });
        let mut m = MapIntervals::new();
        assert_eq!(m.map(0x4000, 16384, 16384), MapAudit::Consistent);
    }

    /// Deleting a task takes its address space with it.
    #[test]
    fn clearing_a_task_is_not_a_leak_and_does_not_poison_a_reused_id() {
        let mut m = MapIntervals::new();
        m.map(0x1000, PAGE, PAGE);
        m.clear();
        assert_eq!(m.live_count(), 0);
        assert_eq!(m.map(0x1000, PAGE, PAGE), MapAudit::Consistent);
    }

    /// The bound announces itself instead of quietly dropping entries, and it
    /// keeps announcing — so no later reading of this task can be mistaken for
    /// a clean one.
    #[test]
    fn exhausting_the_bound_is_reported_and_latches() {
        let mut m = MapIntervals::new();
        let mut saw_abandon = false;
        for i in 0..=(MapIntervals::MAX_LIVE as u64) {
            if m.map(0x1_0000_0000 + i * PAGE, PAGE, PAGE) == MapAudit::TrackingAbandoned {
                saw_abandon = true;
                break;
            }
        }
        assert!(saw_abandon, "the bound must announce itself");
        assert_eq!(m.map(0x10, PAGE, PAGE), MapAudit::TrackingAbandoned);
        assert_eq!(m.unmap(0x10, PAGE), MapAudit::TrackingAbandoned);
    }

    #[test]
    fn every_verdict_has_its_own_slug() {
        let all = [
            MapAudit::Consistent,
            MapAudit::OverlapsLive { gva: 0, len: 0 },
            MapAudit::UnmapOfUnmapped,
            MapAudit::LengthMismatch { mapped_len: 0 },
            MapAudit::SubPage { len: 0 },
            MapAudit::TrackingAbandoned,
        ];
        let mut slugs: Vec<&str> = all.iter().map(|a| a.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), before, "two verdicts share a slug");
        assert!(!MapAudit::Consistent.is_finding());
        assert!(all[1..].iter().all(|a| a.is_finding()));
    }
}
