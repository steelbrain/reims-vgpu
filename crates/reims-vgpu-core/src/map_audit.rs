//! Observation-only state for checking guest map/unmap interval pairing.
//!
//! This state machine has no host-memory, logging, scheduling, or backend
//! dependency. It records the intervals stated by the guest and returns a
//! typed verdict; the composition layer decides how to observe that verdict.

use std::collections::BTreeMap;

/// The guest page size against which sub-page mappings are judged.
pub type PageSize = u64;

/// What one map or unmap did to a task's stated live interval set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapAudit {
    Consistent,
    OverlapsLive {
        gva: u64,
        len: u64,
    },
    UnmapOfUnmapped,
    LengthMismatch {
        mapped_len: u64,
    },
    SubPage {
        len: u64,
    },
    /// The instrument refused further state after its declared capacity.
    TrackingAbandoned,
}

impl MapAudit {
    pub fn is_finding(self) -> bool {
        !matches!(self, Self::Consistent)
    }

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

/// One task's stated live guest-virtual mappings, keyed by start address.
///
/// This is an instrument rather than guest-visible authority. Its capacity is
/// explicit, and exhaustion latches a refusal instead of silently evicting an
/// interval and continuing to report misleadingly clean results.
#[derive(Clone, Debug, Default)]
pub struct MapIntervals {
    live: BTreeMap<u64, u64>,
    abandoned: bool,
}

impl MapIntervals {
    pub const MAX_LIVE: usize = 1 << 16;

    pub const fn new() -> Self {
        Self {
            live: BTreeMap::new(),
            abandoned: false,
        }
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    pub fn map(&mut self, gva: u64, len: u64, page_size: PageSize) -> MapAudit {
        if self.abandoned {
            return MapAudit::TrackingAbandoned;
        }
        if len < page_size {
            self.live.insert(gva, len);
            return MapAudit::SubPage { len };
        }
        if let Some((overlap_gva, overlap_len)) = self.first_overlap(gva, len) {
            self.live.insert(gva, len);
            return MapAudit::OverlapsLive {
                gva: overlap_gva,
                len: overlap_len,
            };
        }
        self.live.insert(gva, len);
        if self.live.len() > Self::MAX_LIVE {
            self.abandoned = true;
            self.live.clear();
            return MapAudit::TrackingAbandoned;
        }
        MapAudit::Consistent
    }

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

    pub fn clear(&mut self) {
        self.live.clear();
        self.abandoned = false;
    }

    fn first_overlap(&self, gva: u64, len: u64) -> Option<(u64, u64)> {
        let end = gva.saturating_add(len);
        if let Some((&prior_gva, &prior_len)) = self.live.range(..=gva).next_back() {
            if prior_gva.saturating_add(prior_len) > gva {
                return Some((prior_gva, prior_len));
            }
        }
        self.live
            .range(gva..end)
            .next()
            .map(|(&start, &span)| (start, span))
            .filter(|_| len != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    #[test]
    fn pairing_and_distinct_failures_are_preserved() {
        let mut intervals = MapIntervals::new();
        assert_eq!(intervals.map(0x1000, PAGE, PAGE), MapAudit::Consistent);
        assert_eq!(intervals.unmap(0x1000, PAGE), MapAudit::Consistent);
        assert_eq!(intervals.unmap(0x1000, PAGE), MapAudit::UnmapOfUnmapped);

        intervals.map(0x2000, PAGE, PAGE);
        assert_eq!(
            intervals.unmap(0x2000, PAGE * 2),
            MapAudit::LengthMismatch { mapped_len: PAGE }
        );
    }

    #[test]
    fn overlap_checks_both_sides_and_not_abutting_ranges() {
        let mut intervals = MapIntervals::new();
        intervals.map(0x2000, PAGE, PAGE);
        assert_eq!(intervals.map(0x3000, PAGE, PAGE), MapAudit::Consistent);
        assert_eq!(
            intervals.map(0x1800, PAGE, PAGE),
            MapAudit::OverlapsLive {
                gva: 0x2000,
                len: PAGE,
            }
        );
    }

    #[test]
    fn sub_page_threshold_is_guest_geometry() {
        let mut intervals = MapIntervals::new();
        assert_eq!(
            intervals.map(0x4000, 4096, 16384),
            MapAudit::SubPage { len: 4096 }
        );
        assert_eq!(intervals.unmap(0x4000, 4096), MapAudit::Consistent);
    }

    #[test]
    fn capacity_refusal_latches_until_clear() {
        let mut intervals = MapIntervals::new();
        for index in 0..=MapIntervals::MAX_LIVE as u64 {
            let verdict = intervals.map(0x1_0000_0000 + index * PAGE, PAGE, PAGE);
            if verdict == MapAudit::TrackingAbandoned {
                break;
            }
        }
        assert_eq!(intervals.map(0x10, PAGE, PAGE), MapAudit::TrackingAbandoned);
        intervals.clear();
        assert_eq!(intervals.map(0x1000, PAGE, PAGE), MapAudit::Consistent);
    }

    #[test]
    fn verdict_slugs_are_exhaustive_and_distinct() {
        let all = [
            MapAudit::Consistent,
            MapAudit::OverlapsLive { gva: 0, len: 0 },
            MapAudit::UnmapOfUnmapped,
            MapAudit::LengthMismatch { mapped_len: 0 },
            MapAudit::SubPage { len: 0 },
            MapAudit::TrackingAbandoned,
        ];
        let mut slugs: Vec<_> = all.iter().map(|verdict| verdict.slug()).collect();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), all.len());
        assert!(!all[0].is_finding());
        assert!(all[1..].iter().all(|verdict| verdict.is_finding()));
    }
}
