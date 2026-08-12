//! Whether a range's page-table entries are in the state the guest's own next
//! step requires of them.
//!
//! # The invariant, and whose it is
//!
//! One guest line's GPU page table asserts in both directions, one page at a
//! time, and each assertion panics the guest:
//!
//! - **tearing a range down**, it refuses to clear a leaf entry that is
//!   **already zero**, and refuses to descend through an interior entry that is
//!   zero — two sites, one outcome;
//! - **building a range up**, it refuses to write an entry that is **already
//!   non-zero**.
//!
//! So the requirement is exactly opposite per direction, over the same walk:
//! every page of an unmap's range must have an entry, and every page of a map's
//! range must not.
//!
//! # Only one direction is ordered, and it is the map
//!
//! Both edits sit on the same side of their packet: the guest wires the range
//! and **then** submits the map, and it submits the unmap and **then** unwires.
//! So the two readings are not symmetric at all.
//!
//! - **Map — ordered.** The wiring is complete before the packet exists, so the
//!   range must be fully covered when this device sees it. [`Coverage::Absent`]
//!   and [`Coverage::Partial`] are findings: the guest published a mapping its
//!   own tree does not hold, and the eventual teardown walks the hole.
//! - **Unmap — a race, and nothing here is a finding.** The unwiring is only
//!   *started* by the time the packet is readable, so what this device finds
//!   depends on whether the drain or the guest got there first. Measured on
//!   macos-26 the guest wins about nineteen times in twenty, and a boot's
//!   `unmap_coverage_absent` is that race and not a defect.
//!
//! The unmap side is still counted, for two reasons. The ratio measures the race
//! itself, which is a real property of this device's drain latency; and its
//! disagreeing with the map side is what proves the walk works at all. A walk
//! that had a stale root, resolved the wrong task, or was off by a page shift
//! would report absence on **both** sides, and the map side reading covered is
//! the only thing that separates a working instrument from that.
//!
//! **Do not restore a finding on the unmap side.** It was one for exactly one
//! boot, on the premise that the guest blocks on this device's reply before
//! unwiring. It does not; the submit and the unwire are adjacent, with the reply
//! not between them.
//!
//! # Why this is not the same question as the two guards next door
//!
//! [`crate::runtime::node_guard`] and [`crate::runtime::released_pages`] both
//! ask whether this device *wrote* somewhere it should not have. Neither
//! mechanism is required for the assertion above: a range that was torn down
//! twice, or torn down without ever having been built, reaches it with no host
//! write anywhere in the story. Both guards read clean on boots that panicked,
//! and this is the reading that explains how they could.
//!
//! # What the readings mean
//!
//! [`Coverage`] describes the tree and nothing else; [`Op`] decides which
//! reading is the alarm.
//!
//! - [`Coverage::Absent`] — **no** page of the range has an entry. `level`
//!   separates two shapes: a zero at the deepest level is a leaf entry that was
//!   never written or has been cleared, and a zero above it is a subtree that is
//!   not there at all.
//! - [`Coverage::Partial`] — some pages have entries and some do not, and
//!   `first_absent` is the page an in-order walk reaches first, which is the page
//!   the guest's own walk would die on.
//! - [`Coverage::Covered`] — every page has one. On a map that is the healthy
//!   reading and the whole population.
//!
//! [`Coverage::Undecidable`] is **not** a finding: a table page that would not
//! read says nothing about what the guest will find there, and reporting it as
//! absence is how an alarm costs a session for being wrong.
//!
//! # What it costs
//!
//! The walk reuses upper levels across the run and reads the deepest level a
//! batch at a time, so a mapping of `n` pages costs on the order of `n / 64`
//! guest reads rather than `n * depth`. The guest's own teardown walks every one
//! of those pages unconditionally, so the reach asked for is never more than the
//! reach the guest has already committed to.
//!
//! # And it is still too expensive to leave on, which was measured the hard way
//!
//! Always-on for four commits, this walk **changed the thing it was watching**.
//! One undriven macos-15 boot:
//!
//! | build | this probe | `no_list_entry` | `list_miss_slot_empty` |
//! |---|---|---|---|
//! | previous tip | absent | **0** | **0** |
//! | with it, guards on | walking | **47** | **182** |
//! | same binary, guards off | silent | **0** | **0** |
//!
//! A rail that had never lost a draw to an empty object-list slot started losing
//! forty-seven of them, and switching the probe off on the same binary gave them
//! back. The walk is on the drain thread while it holds the device lock, and the
//! guest clears those slots by writing its own memory — which nothing orders
//! against the ring except how fast this device reads it. Slowing the drain lost
//! the race.
//!
//! So this is a **probe, not a guard**, and it is off unless
//! [`crate::env::RANGE_COVERAGE`] asks for it. Turn it on to ask its question and
//! off before quoting any other counter from the same boot.
//!
//! That accident is also the most useful thing this module has produced. It is a
//! *manipulated variable*: drain latency was raised deliberately and lost draws
//! followed, on a rail with none. Every earlier attempt on `no_list_entry` was a
//! correlation. See `kb/macos-26-the-guest-runs-ahead-of-the-drain.md`.

use reims_vgpu_paging::resolve::RangeCoverage;

/// The most pages one packet will be walked for.
///
/// **Not a fidelity bound.** The guest walks every page of its own range, so
/// there is no reach this can be too small for in the sense of missing something
/// the guest does not also do. It bounds the drain thread's cost against a
/// length field this device has not validated, which is the only reason it
/// exists: a corrupt or absurd length would otherwise spin the walk for as long
/// as the number says.
///
/// A million pages is four gigabytes at a 4 KiB page and sixteen at 16 KiB —
/// larger than any mapping this device has been observed handed. When it does
/// bite it is **counted**, not silent: see `unmap_coverage_truncated`. A bound
/// that trims a reading without saying so is how a scan reports a clean sweep of
/// a population it could not see.
pub const MAX_SCAN_PAGES: u64 = 1 << 20;

/// Whether this probe runs at all, which by default it does not.
///
/// **The only instrument here that is off unless asked for, and the reason is a
/// measurement rather than caution.** See the module doc's cost section and
/// [`crate::env::RANGE_COVERAGE`].
///
/// Read once and cached: the alternative is an environment lookup on the drain
/// thread for every map and unmap packet, which is the kind of cost this gate
/// exists to avoid.
///
/// [`crate::runtime::node_guard::enabled`] still silences it, so the one switch
/// that takes every page instrument out of a boot keeps doing that.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        enabled_from(
            crate::env::switch(crate::env::RANGE_COVERAGE),
            crate::runtime::node_guard::enabled(),
        )
    })
}

/// The decision [`enabled`] caches, split out so it can be tested.
///
/// [`enabled`] latches in a `OnceLock` and the environment is process-global, so
/// a test of the real function could assert one arm per process at best. This is
/// the whole rule and it is total over both inputs.
fn enabled_from(asked: crate::env::Switch, guards_on: bool) -> bool {
    asked == crate::env::Switch::On && guards_on
}

/// Which way the guest is about to edit the range.
///
/// Carried as a type rather than a bool so the two requirements are named where
/// they are read, and so a third direction cannot be added without every
/// consumer being told about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// The guest has **already** written an entry per page, and then submitted.
    /// The range must be fully covered; a hole is the guest publishing a mapping
    /// its own tree does not hold.
    Map,
    /// The guest submitted, and unwires afterwards. What the tree holds when the
    /// packet is read is a race between the drain and the guest, so nothing on
    /// this side is a finding.
    Unmap,
}

impl Op {
    /// The counter for a range longer than [`MAX_SCAN_PAGES`], so a bound that
    /// trims a reading says so rather than shrinking the population silently.
    pub fn truncated_route(self) -> &'static str {
        match self {
            Self::Map => "map_coverage_truncated",
            Self::Unmap => "unmap_coverage_truncated",
        }
    }
}

/// What the range looked like, as a verdict rather than counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every page of the range has an entry. The teardown will not assert.
    Covered,
    /// **No page of the range has an entry.**
    ///
    /// `level` is where the first page's descent stopped, zero-based from the
    /// root; `depth` is the tree's. `level + 1 == depth` is a leaf entry that
    /// reads zero, and anything shallower is an absent subtree.
    Absent { level: u32, depth: u32 },
    /// **Some pages have entries and some do not.**
    ///
    /// `first_absent` is the index within the range of the first page without
    /// one, which is the page the guest's own in-order walk reaches first.
    Partial {
        first_absent: u64,
        absent: u64,
        level: u32,
        depth: u32,
    },
    /// The tree could not be read for any page. Not a finding.
    Undecidable,
    /// The task has no readable root, so there is no range to have an opinion
    /// about. Not a finding.
    Unwalkable,
}

impl Coverage {
    /// Read counts from a walk into the verdict they support.
    ///
    /// Absence outranks undecidability on purpose: a range with one readable
    /// zero entry and a hundred unreadable tables still has a zero entry, and
    /// that zero is what the guest will assert on. The reverse — treating a
    /// range as decided because most of it read — would be the mistake.
    pub fn of(c: &RangeCoverage) -> Self {
        if c.absent == 0 {
            return if c.present == 0 {
                Self::Undecidable
            } else {
                Self::Covered
            };
        }
        if c.present == 0 && c.undecidable == 0 {
            return Self::Absent {
                level: c.first_absent_level,
                depth: c.depth,
            };
        }
        Self::Partial {
            first_absent: c.first_absent_index,
            absent: c.absent,
            level: c.first_absent_level,
            depth: c.depth,
        }
    }

    /// Whether this reading is a defect for `op`.
    ///
    /// Only the map direction has any, because only it is ordered: the guest
    /// finishes wiring before the packet exists, so a range that is not covered
    /// is a mapping its own tree does not hold. The unmap direction is a race
    /// with the guest's own unwiring and every reading of it is expected — see
    /// this module's docs before making one of them an alarm.
    pub fn is_finding(self, op: Op) -> bool {
        if op == Op::Unmap {
            return false;
        }
        matches!(self, Self::Absent { .. } | Self::Partial { .. })
    }

    /// The counter name, one per direction per variant and exhaustive, so a new
    /// verdict cannot reach a census under a borrowed name and the two
    /// directions can never be summed into one.
    pub fn route(self, op: Op) -> &'static str {
        match (op, self) {
            (Op::Map, Self::Covered) => "map_coverage_covered",
            (Op::Map, Self::Absent { .. }) => "map_coverage_absent",
            (Op::Map, Self::Partial { .. }) => "map_coverage_partial",
            (Op::Map, Self::Undecidable) => "map_coverage_undecidable",
            (Op::Map, Self::Unwalkable) => "map_coverage_unwalkable",
            (Op::Unmap, Self::Covered) => "unmap_coverage_covered",
            (Op::Unmap, Self::Absent { .. }) => "unmap_coverage_absent",
            (Op::Unmap, Self::Partial { .. }) => "unmap_coverage_partial",
            (Op::Unmap, Self::Undecidable) => "unmap_coverage_undecidable",
            (Op::Unmap, Self::Unwalkable) => "unmap_coverage_unwalkable",
        }
    }

    /// Whether the zero this found sits at the deepest level of the tree.
    ///
    /// That is the one the observed guest assertion fires on — it refuses to
    /// clear a leaf entry that is already zero. A shallower zero ends the guest
    /// too, at its other assertion, and the two are different defects.
    pub fn is_leaf_level(self) -> Option<bool> {
        match self {
            Self::Absent { level, depth } | Self::Partial { level, depth, .. } => {
                Some(level + 1 == depth)
            }
            _ => None,
        }
    }
}

/// How many pages a range spans, and how many of them will be walked.
///
/// Returns `(spanned, scanned)`. They differ only when [`MAX_SCAN_PAGES`] bites,
/// and a caller that finds them different must say so rather than report the
/// scan as covering the range.
///
/// A length below one page spans no pages: the guest's own teardown returns
/// immediately for one, so there is nothing to predict.
pub fn pages_of(length: u64, page_shift: u32) -> (u64, u64) {
    let page = 1u64 << page_shift;
    if length < page {
        return (0, 0);
    }
    let spanned = length >> page_shift;
    (spanned, spanned.min(MAX_SCAN_PAGES))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The probe runs only when an operator asked for it by name, and the page
    /// guards' own switch still silences it.
    ///
    /// Off-by-default is the load-bearing half and it is not a preference: this
    /// walk was measured changing the counters of the rail it ran on, so an
    /// unset environment — which is every boot nobody configured — must not
    /// walk. A `Switch::Unrecognized` counts as not asked, so a typo leaves the
    /// quiet default rather than silently costing a boot its readings.
    #[test]
    fn the_probe_is_off_unless_asked_for_by_name_and_the_guards_agree() {
        use crate::env::Switch;
        assert!(enabled_from(Switch::On, true));
        for asked in [Switch::Unset, Switch::Off, Switch::Unrecognized] {
            assert!(!enabled_from(asked, true), "{asked:?} started the probe");
        }
        for asked in [Switch::On, Switch::Unset, Switch::Off, Switch::Unrecognized] {
            assert!(
                !enabled_from(asked, false),
                "{asked:?} outran the page-guard switch"
            );
        }
    }

    fn counts(present: u64, absent: u64, undecidable: u64) -> RangeCoverage {
        RangeCoverage {
            pages: present + absent + undecidable,
            present,
            absent,
            undecidable,
            first_absent_index: 3,
            first_absent_level: 2,
            depth: 3,
        }
    }

    /// The four decided shapes each map to their own verdict, and the routes are
    /// distinct.
    #[test]
    fn every_shape_of_counts_reaches_its_own_verdict() {
        assert_eq!(Coverage::of(&counts(4, 0, 0)), Coverage::Covered);
        assert_eq!(
            Coverage::of(&counts(0, 4, 0)),
            Coverage::Absent { level: 2, depth: 3 }
        );
        assert_eq!(
            Coverage::of(&counts(2, 2, 0)),
            Coverage::Partial {
                first_absent: 3,
                absent: 2,
                level: 2,
                depth: 3,
            }
        );
        assert_eq!(Coverage::of(&counts(0, 0, 4)), Coverage::Undecidable);

        let mut routes = Vec::new();
        for (op, prefix) in [(Op::Map, "map_coverage_"), (Op::Unmap, "unmap_coverage_")] {
            assert!(op.truncated_route().starts_with(prefix));
            routes.push(op.truncated_route());
            for v in every_verdict() {
                let route = v.route(op);
                assert!(
                    route.starts_with(prefix),
                    "{route} does not name its direction"
                );
                routes.push(route);
            }
        }
        for (i, a) in routes.iter().enumerate() {
            for b in &routes[i + 1..] {
                assert_ne!(a, b, "two verdicts share a counter name");
            }
        }
    }

    fn every_verdict() -> [Coverage; 5] {
        [
            Coverage::Covered,
            Coverage::Absent { level: 2, depth: 3 },
            Coverage::Partial {
                first_absent: 0,
                absent: 1,
                level: 2,
                depth: 3,
            },
            Coverage::Undecidable,
            Coverage::Unwalkable,
        ]
    }

    /// A range that is entirely absent except for tables that would not read is
    /// reported as partial, not as wholly absent.
    ///
    /// The distinction is the whole reason `undecidable` is counted separately:
    /// "the range was torn down twice" is a claim about every page of it, and an
    /// unread table is not evidence for that claim.
    #[test]
    fn an_unread_table_beside_a_zero_entry_downgrades_absent_to_partial() {
        assert!(matches!(
            Coverage::of(&counts(0, 3, 1)),
            Coverage::Partial { .. }
        ));
    }

    /// Nothing on the unmap side is ever a finding, whatever the tree held.
    ///
    /// This is the property a measured boot bought: the guest submits the unmap
    /// and unwires afterwards, so an absent range there is the drain losing a
    /// race — 124 of them against 7 covered on one macos-26 boot. An alarm on
    /// that reads as a boot full of proof and is worth none of it.
    #[test]
    fn the_unmap_direction_has_no_findings_at_all() {
        for v in every_verdict() {
            assert!(!v.is_finding(Op::Unmap), "{v:?} became an unmap finding");
        }
    }

    /// On the map side the guest has finished wiring before the packet exists,
    /// so anything short of full coverage is a defect and full coverage is the
    /// healthy reading.
    #[test]
    fn a_map_is_a_finding_exactly_when_its_range_is_not_fully_covered() {
        assert!(Coverage::Absent { level: 2, depth: 3 }.is_finding(Op::Map));
        assert!(Coverage::Partial {
            first_absent: 0,
            absent: 1,
            level: 0,
            depth: 3,
        }
        .is_finding(Op::Map));
        assert!(!Coverage::Covered.is_finding(Op::Map));
        assert!(!Coverage::Undecidable.is_finding(Op::Map));
        assert!(!Coverage::Unwalkable.is_finding(Op::Map));
    }

    /// The leaf-level question is answered only where there is a zero to ask it
    /// about, and it separates the two guest assertions.
    #[test]
    fn the_leaf_level_question_separates_the_two_assertions() {
        assert_eq!(
            Coverage::Absent { level: 2, depth: 3 }.is_leaf_level(),
            Some(true)
        );
        assert_eq!(
            Coverage::Absent { level: 0, depth: 3 }.is_leaf_level(),
            Some(false)
        );
        assert_eq!(Coverage::Covered.is_leaf_level(), None);
        assert_eq!(Coverage::Undecidable.is_leaf_level(), None);
    }

    /// A range shorter than a page spans nothing, and a range longer than the
    /// walk's reach reports both numbers so the caller can say it was trimmed.
    #[test]
    fn the_page_count_spans_the_range_and_reports_its_own_trim() {
        for shift in [12u32, 14] {
            let page = 1u64 << shift;
            assert_eq!(pages_of(0, shift), (0, 0));
            assert_eq!(pages_of(page - 1, shift), (0, 0));
            assert_eq!(pages_of(page, shift), (1, 1));
            assert_eq!(pages_of(page * 5 + 7, shift), (5, 5));

            let over = (MAX_SCAN_PAGES + 9) << shift;
            assert_eq!(pages_of(over, shift), (MAX_SCAN_PAGES + 9, MAX_SCAN_PAGES));
        }
    }
}
