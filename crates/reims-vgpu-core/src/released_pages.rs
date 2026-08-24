//! Observation-only state for writes to guest pages after their release.

use std::collections::BTreeMap;

use crate::{HostWriteVerdict, HostWrites};

pub const RELEASED_PAGE_WATCH_CAP: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleasedVerdict {
    Quiet,
    Wrote { since_us: u64 },
    Undecidable,
}

impl ReleasedVerdict {
    pub fn is_finding(self) -> bool {
        matches!(self, Self::Wrote { .. })
    }

    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "released_quiet",
            Self::Wrote { .. } => "released_write_after_release",
            Self::Undecidable => "released_undecidable",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Released {
    epoch: u64,
    at_us: u64,
    task_id: u32,
}

/// Released guest-physical pages across every task.
///
/// The key is global because another task mapping a shared page makes writes
/// legitimate again. A second release keeps the first epoch, and a reported or
/// remapped page leaves the watch. Capacity refusals are counted without
/// evicting an existing observation.
#[derive(Default, Debug)]
pub struct ReleasedPages {
    pages: BTreeMap<u64, Released>,
    refused: u64,
}

impl ReleasedPages {
    pub fn release(&mut self, writes: &HostWrites, task_id: u32, gpa: u64, now_us: u64) {
        if self.pages.contains_key(&gpa) {
            return;
        }
        if self.pages.len() >= RELEASED_PAGE_WATCH_CAP {
            self.refused = self.refused.saturating_add(1);
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

    pub fn remapped(&mut self, gpa: u64) {
        self.pages.remove(&gpa);
    }

    /// Return only decided pages. Quiet pages remain armed and are not emitted.
    pub fn sweep(&mut self, writes: &HostWrites, now_us: u64) -> Vec<(u64, u32, ReleasedVerdict)> {
        let mut decided = Vec::new();
        self.pages.retain(|&gpa, released| {
            let verdict = match writes.wrote_any_since(released.epoch, &[gpa]) {
                HostWriteVerdict::NoWrites | HostWriteVerdict::Disjoint => return true,
                HostWriteVerdict::Overlap => ReleasedVerdict::Wrote {
                    since_us: now_us.saturating_sub(released.at_us),
                },
                _ => ReleasedVerdict::Undecidable,
            };
            decided.push((gpa, released.task_id, verdict));
            false
        });
        decided
    }

    pub fn watched(&self) -> usize {
        self.pages.len()
    }

    pub fn refused(&self) -> u64 {
        self.refused
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    #[test]
    fn quiet_pages_stay_armed_without_becoming_output() {
        let writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        released.release(&writes, 7, PAGE, 0);
        assert!(released.sweep(&writes, 1).is_empty());
        assert_eq!(released.watched(), 1);
    }

    #[test]
    fn a_write_after_release_reports_once_with_origin_and_time() {
        let mut writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        released.release(&writes, 7, PAGE, 100);
        writes.note_pages(vec![PAGE]);
        assert_eq!(
            released.sweep(&writes, 700),
            vec![(PAGE, 7, ReleasedVerdict::Wrote { since_us: 600 })]
        );
        assert!(released.sweep(&writes, 800).is_empty());
    }

    #[test]
    fn remapping_by_any_task_disarms_the_page() {
        let mut writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        released.release(&writes, 3, PAGE, 0);
        released.remapped(PAGE);
        writes.note_pages(vec![PAGE]);
        assert!(released.sweep(&writes, 1).is_empty());
    }

    #[test]
    fn a_second_release_does_not_forgive_an_intervening_write() {
        let mut writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        released.release(&writes, 7, PAGE, 0);
        writes.note_pages(vec![PAGE]);
        released.release(&writes, 7, PAGE, 10);
        assert_eq!(
            released.sweep(&writes, 20),
            vec![(PAGE, 7, ReleasedVerdict::Wrote { since_us: 20 })]
        );
    }

    #[test]
    fn an_unnamed_write_is_decided_as_undecidable_not_a_finding() {
        let mut writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        released.release(&writes, 7, PAGE, 0);
        writes.note_unknown();
        assert_eq!(
            released.sweep(&writes, 1),
            vec![(PAGE, 7, ReleasedVerdict::Undecidable)]
        );
    }

    #[test]
    fn a_full_watch_refuses_without_evicting() {
        let writes = HostWrites::default();
        let mut released = ReleasedPages::default();
        for index in 0..RELEASED_PAGE_WATCH_CAP as u64 {
            released.release(&writes, 7, index * PAGE, 0);
        }
        released.release(&writes, 7, RELEASED_PAGE_WATCH_CAP as u64 * PAGE, 0);
        assert_eq!(released.watched(), RELEASED_PAGE_WATCH_CAP);
        assert_eq!(released.refused(), 1);
    }

    #[test]
    fn verdict_routes_are_exhaustive_and_distinct() {
        let all = [
            ReleasedVerdict::Quiet,
            ReleasedVerdict::Wrote { since_us: 0 },
            ReleasedVerdict::Undecidable,
        ];
        let mut routes: Vec<_> = all.iter().map(|verdict| verdict.route()).collect();
        routes.sort_unstable();
        routes.dedup();
        assert_eq!(routes.len(), all.len());
        assert_eq!(all.iter().filter(|verdict| verdict.is_finding()).count(), 1);
    }
}
