//! The pages blocks whose guest RAM overlaps another live pages block's (`pages-block-overlaps`):
//! a diagnostic for the runtime's import of guest pages, the guest sees nothing of it.
//!
//! Each pages block's buffers lie in an import of its own view (the parent module's docs, "Pages
//! blocks"), so two blocks over some of the same guest pages are two host views of those pages,
//! and the runtime holds two imports of them, which its hazard tracking cannot tie together. The
//! guest shares a block only for the same bytes; partial overlaps, read-only pages beside writable
//! ones and memory shared between processes each take a block of their own. So the blocks whose
//! runs meet another live block's as their view's import is made, or taken from the cache, are
//! counted once each ([`OVERLAPS`]), and the most such blocks live at once is counted as its
//! highest value ([`OVERLAPS_LIVE_MOST`]). A block of pages its process maps read-only holds no
//! import, and is not counted. A block is live from then until its entry goes (its release, its
//! context's close, the device's reset): [`Covered`] holds its runs until it drops.

use std::collections::BTreeMap;
use std::sync::Arc;

use reims_vgpu_vulkan::core::sync::Mutex;

/// The code of a pages block whose runs met another live pages block's as its import was made,
/// counted once a block.
pub(crate) const OVERLAPS: &str = "pages-block-overlaps";

/// The code whose count is the most blocks counted as [`OVERLAPS`] live at once.
pub(crate) const OVERLAPS_LIVE_MOST: &str = "pages-block-overlaps-live-most";

/// The guest RAM the live pages blocks' runs cover, as disjoint ranges, each with how many runs
/// cover it, and how many overlapping blocks live.
#[derive(Default)]
pub(crate) struct Coverage {
    /// By its first byte, each covered range's end and its runs.
    ranges: BTreeMap<u64, (u64, u32)>,
    /// The blocks counted as [`OVERLAPS`] that live.
    overlapping: u64,
    /// The most of them that lived at once.
    most: u64,
}

/// A live pages block's runs in its coverage, which its drop takes out.
pub(crate) struct Covered {
    coverage: Arc<Mutex<Coverage>>,
    runs: Vec<(u64, u64)>,
    overlapped: bool,
}

impl Drop for Covered {
    fn drop(&mut self) {
        let mut coverage = self.coverage.lock();
        for &(start, length) in &self.runs {
            coverage.take(start, start.saturating_add(length));
        }
        if self.overlapped {
            coverage.overlapping = coverage.overlapping.saturating_sub(1);
        }
    }
}

/// The runs `runs`, `(guest address, length)` each, of a pages block whose import was just made
/// or taken, added to `coverage`, and counted where one meets a run another live block holds.
pub(crate) fn cover(coverage: &Arc<Mutex<Coverage>>, runs: &[(u64, u64)]) -> Covered {
    let mut c = coverage.lock();
    let meets = |&(start, length): &(u64, u64)| c.meets(start, start.saturating_add(length));
    let overlapped = runs.iter().any(meets);
    for &(start, length) in runs {
        c.add(start, start.saturating_add(length));
    }
    if overlapped {
        reims_vgpu_vulkan::answer::count_kind(OVERLAPS);
        c.overlapping += 1;
        if c.overlapping > c.most {
            reims_vgpu_vulkan::answer::count_kind_by(OVERLAPS_LIVE_MOST, c.overlapping - c.most);
            c.most = c.overlapping;
        }
    }
    drop(c);
    Covered {
        coverage: Arc::clone(coverage),
        runs: runs.to_vec(),
        overlapped,
    }
}

impl Coverage {
    /// Whether a run covers any byte of `start..end`.
    fn meets(&self, start: u64, end: u64) -> bool {
        let last = self.ranges.range(..end).next_back();
        start < end && last.is_some_and(|(_, &(last_end, _))| last_end > start)
    }

    /// Splits the range that holds `at` inside it, so that one starts at `at`.
    fn split(&mut self, at: u64) {
        let Some((&start, &(end, runs))) = self.ranges.range(..at).next_back() else {
            return;
        };
        if end > at {
            self.ranges.insert(start, (at, runs));
            self.ranges.insert(at, (end, runs));
        }
    }

    /// One more run over `start..end`.
    fn add(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        self.split(start);
        self.split(end);
        let inside: Vec<(u64, u64)> = (self.ranges.range(start..end))
            .map(|(&s, &(e, _))| (s, e))
            .collect();
        let mut at = start;
        for (s, e) in inside {
            if at < s {
                self.ranges.insert(at, (s, 1));
            }
            if let Some((_, runs)) = self.ranges.get_mut(&s) {
                *runs += 1;
            }
            at = e;
        }
        if at < end {
            self.ranges.insert(at, (end, 1));
        }
        self.merge(start, end);
    }

    /// One run fewer over `start..end`, which [`Self::add`] added.
    fn take(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        self.split(start);
        self.split(end);
        let inside: Vec<u64> = self.ranges.range(start..end).map(|(&s, _)| s).collect();
        for s in inside {
            match self.ranges.get_mut(&s) {
                Some((_, runs)) if *runs > 1 => *runs -= 1,
                _ => {
                    self.ranges.remove(&s);
                }
            }
        }
        self.merge(start, end);
    }

    /// Joins the ranges from the one before `start` to the one at `end` that touch a neighbour of
    /// as many runs, so that ranges split by runs gone do not pile up.
    fn merge(&mut self, start: u64, end: u64) {
        let first = self
            .ranges
            .range(..start)
            .next_back()
            .map_or(start, |(&s, _)| s);
        let keys: Vec<u64> = self.ranges.range(first..=end).map(|(&s, _)| s).collect();
        let mut kept: Option<(u64, u64, u32)> = None;
        for s in keys {
            let Some(&(e, runs)) = self.ranges.get(&s) else {
                continue;
            };
            match kept {
                Some((ks, ke, kr)) if ke == s && kr == runs => {
                    self.ranges.remove(&s);
                    self.ranges.insert(ks, (e, runs));
                    kept = Some((ks, e, runs));
                }
                _ => kept = Some((s, e, runs)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges(c: &Coverage) -> Vec<(u64, u64, u32)> {
        c.ranges.iter().map(|(&s, &(e, r))| (s, e, r)).collect()
    }

    /// Runs add as disjoint ranges with their counts, merge back as they go, and only a range a
    /// run covers is met.
    #[test]
    fn coverage_counts_runs_over_each_range() {
        let mut c = Coverage::default();
        c.add(0x1000, 0x3000);
        c.add(0x2000, 0x5000);
        assert_eq!(
            ranges(&c),
            [
                (0x1000, 0x2000, 1),
                (0x2000, 0x3000, 2),
                (0x3000, 0x5000, 1)
            ]
        );
        assert!(c.meets(0x2fff, 0x3001) && c.meets(0x4fff, 0x9000));
        assert!(!c.meets(0x5000, 0x6000) && !c.meets(0, 0x1000) && !c.meets(0x1800, 0x1800));
        c.take(0x2000, 0x5000);
        assert_eq!(ranges(&c), [(0x1000, 0x3000, 1)]);
        c.add(0x3000, 0x4000);
        assert_eq!(ranges(&c), [(0x1000, 0x4000, 1)]);
        c.take(0x3000, 0x4000);
        c.take(0x1000, 0x3000);
        assert!(c.ranges.is_empty());
    }

    /// A block whose runs meet a live block's is counted once, and the most live at once as the
    /// second code's count; a block over pages no live block holds is not, nor one whose own
    /// runs alone touch.
    #[test]
    fn a_block_over_a_live_block_s_pages_is_counted() {
        let counted = |code: &str| {
            (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let (before, most) = (counted(OVERLAPS), counted(OVERLAPS_LIVE_MOST));
        let coverage = Arc::new(Mutex::new(Coverage::default()));
        let a = cover(&coverage, &[(0x10000, 0x4000), (0x14000, 0x1000)]);
        assert!(!a.overlapped);
        let b = cover(&coverage, &[(0x20000, 0x1000)]);
        assert!(!b.overlapped);
        let c = cover(&coverage, &[(0x30000, 0x1000), (0x13000, 0x2000)]);
        let d = cover(&coverage, &[(0x20000, 0x1000)]);
        assert!(c.overlapped && d.overlapped);
        assert!(counted(OVERLAPS) >= before + 2);
        assert!(counted(OVERLAPS_LIVE_MOST) >= most);
        assert_eq!(coverage.lock().overlapping, 2);
        drop((a, b, c, d));
        let c = coverage.lock();
        assert!(c.ranges.is_empty());
        assert_eq!((c.overlapping, c.most), (0, 2));
    }
}
