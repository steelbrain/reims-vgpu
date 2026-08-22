//! Observation-only state for detecting host writes to page-table node pages.

use std::collections::BTreeMap;

use crate::{HostWriteVerdict, HostWrites};

const WATCH_CAP: usize = 512;

/// What changed between two sightings of one page-table node page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeVerdict {
    FirstSight,
    Quiet,
    Wrote { gap_us: u64 },
    Undecidable,
    NotWatched,
}

impl NodeVerdict {
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Wrote { .. })
    }

    pub fn route(self) -> &'static str {
        match self {
            Self::FirstSight => "node_guard_first_sight",
            Self::Quiet => "node_guard_quiet",
            Self::Wrote { .. } => "node_guard_wrote_node_page",
            Self::Undecidable => "node_guard_undecidable",
            Self::NotWatched => "node_guard_not_watched",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Sighting {
    epoch: u64,
    at_us: u64,
}

/// One task's sampled set of page-table node pages.
///
/// Capacity exhaustion is returned and counted, never treated as a quiet
/// result. The watch is diagnostic state and does not select guest behavior.
#[derive(Default, Debug)]
pub struct NodeWatch {
    seen: BTreeMap<u64, Sighting>,
    refused: u64,
}

impl NodeWatch {
    pub fn observe(&mut self, writes: &HostWrites, gpa: u64, now_us: u64) -> NodeVerdict {
        let epoch = writes.epoch();
        let full = self.seen.len() >= WATCH_CAP;
        match self.seen.get_mut(&gpa) {
            Some(previous) => {
                let verdict = match writes.wrote_any_since(previous.epoch, &[gpa]) {
                    HostWriteVerdict::Quiet => NodeVerdict::Quiet,
                    HostWriteVerdict::Overlap => NodeVerdict::Wrote {
                        gap_us: now_us.saturating_sub(previous.at_us),
                    },
                    _ => NodeVerdict::Undecidable,
                };
                *previous = Sighting {
                    epoch,
                    at_us: now_us,
                };
                verdict
            }
            None if full => {
                self.refused = self.refused.saturating_add(1);
                NodeVerdict::NotWatched
            }
            None => {
                self.seen.insert(
                    gpa,
                    Sighting {
                        epoch,
                        at_us: now_us,
                    },
                );
                NodeVerdict::FirstSight
            }
        }
    }

    pub fn refused(&self) -> u64 {
        self.refused
    }

    pub fn watched(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    #[test]
    fn a_write_between_sightings_is_exactly_one_finding() {
        let mut writes = HostWrites::default();
        let mut watch = NodeWatch::default();
        assert_eq!(watch.observe(&writes, PAGE, 100), NodeVerdict::FirstSight);
        writes.note_pages(vec![PAGE]);
        assert_eq!(
            watch.observe(&writes, PAGE, 350),
            NodeVerdict::Wrote { gap_us: 250 }
        );
        assert_eq!(watch.observe(&writes, PAGE, 400), NodeVerdict::Quiet);
    }

    #[test]
    fn unrelated_page_writes_stay_quiet() {
        let mut writes = HostWrites::default();
        let mut watch = NodeWatch::default();
        watch.observe(&writes, PAGE, 0);
        writes.note_pages(vec![2 * PAGE]);
        assert_eq!(watch.observe(&writes, PAGE, 1), NodeVerdict::Quiet);
    }

    #[test]
    fn capacity_refusal_does_not_displace_watched_pages() {
        let writes = HostWrites::default();
        let mut watch = NodeWatch::default();
        for index in 0..WATCH_CAP as u64 {
            assert_eq!(
                watch.observe(&writes, index * PAGE, 0),
                NodeVerdict::FirstSight
            );
        }
        assert_eq!(
            watch.observe(&writes, WATCH_CAP as u64 * PAGE, 0),
            NodeVerdict::NotWatched
        );
        assert_eq!(watch.watched(), WATCH_CAP);
        assert_eq!(watch.refused(), 1);
        assert_eq!(watch.observe(&writes, 0, 1), NodeVerdict::Quiet);
    }

    #[test]
    fn routes_are_exhaustive_and_distinct() {
        let all = [
            NodeVerdict::FirstSight,
            NodeVerdict::Quiet,
            NodeVerdict::Wrote { gap_us: 0 },
            NodeVerdict::Undecidable,
            NodeVerdict::NotWatched,
        ];
        let mut routes: Vec<_> = all.iter().map(|verdict| verdict.route()).collect();
        routes.sort_unstable();
        routes.dedup();
        assert_eq!(routes.len(), all.len());
        assert!(all[2].is_finding());
        assert!(all[..2].iter().all(|verdict| !verdict.is_finding()));
    }
}
