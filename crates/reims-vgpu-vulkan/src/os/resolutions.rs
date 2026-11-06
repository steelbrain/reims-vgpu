//! The ranges of an import the dirty log resolved to segments of the guest's RAM, kept so that a
//! range asked again, or one inside a kept one, is answered without the backing table
//! (`os/linux/dirty.rs`'s module docs, "Resolved once").
//!
//! **Wholes.** An import may name, as it is made, the ranges it is asked within for its life
//! ([`Resolutions::with_wholes`]): the guest's RAM handed to a device whole, whose log's offsets
//! are host addresses, names the RAM's ranges. A whole is resolved at its first ask and kept for
//! the import's life, so it is read with no lock. The runtime reports a command buffer's writes
//! into the RAM once for each run of it that a written surface's pages are (**read**,
//! metal2vulkan's `PagesLog::wrote_but`, at pin 14): about 9,000 runs for a 3840x2400 surface
//! over scattered 4 KiB guest pages, at the commit, at a completion turn and at the completion,
//! three threads at once. Each such run missed the kept ranges below, was resolved under the
//! backing table's lock and was kept under theirs. In a stock-lane present loop on the RTX 5080
//! Laptop GPU the reports were 78% of QEMU's samples, and the locks' slow path 23% (**measured**
//! 2026-11-05, round 20261105-153142, perf at 499 Hz).
//!
//! **Kept ranges.** Any other range is cut from the latest of [`KEPT`] ranges kept that holds it,
//! or resolved and kept. Their lock is held only to find an entry or to put one in: the cut, the
//! resolution and the drop of what goes are made outside it.
//!
//! An answer ([`Segments`]) holds the segments it is cut from and cuts them as it is read, so a
//! range inside a kept one allocates nothing.
//!
//! Plain data over a resolver its caller passes, so how it behaves under the runtime's reporting
//! threads is measured on any host: the ignored test `reports_cost`, run by hand in a release
//! build (`cargo test -p reims-vgpu-vulkan --release --lib -- --ignored --exact
//! os::resolutions::tests::reports_cost --nocapture`).

use std::ops::Range;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use reims_vgpu_core::memory::dirty::Segment;
use reims_vgpu_core::sync::Mutex;

/// The ranges of an import whose segments are kept: the oldest goes when another is resolved.
pub(crate) const KEPT: usize = 4;

/// A range of an import, resolved, and its segments.
type Kept = (Range<u64>, Arc<[Segment]>);

/// A range an import is asked within for its life, and its segments once resolved (module docs,
/// "Wholes").
#[derive(Debug)]
struct Whole {
    range: Range<u64>,
    segments: OnceLock<Box<[Segment]>>,
}

/// An import's kept resolutions (module docs).
#[derive(Debug, Default)]
pub(crate) struct Resolutions {
    /// The wholes, in order and apart.
    wholes: Box<[Whole]>,
    /// The other ranges resolved so far, each with its segments, the latest last.
    kept: Mutex<Vec<Kept>>,
    /// How many times a resolver was asked.
    #[cfg(test)]
    resolutions: AtomicU64,
}

impl Resolutions {
    /// Resolutions with the wholes `ranges` (module docs), those that meet or touch joined.
    pub(crate) fn with_wholes(ranges: impl IntoIterator<Item = Range<u64>>) -> Resolutions {
        let mut ranges: Vec<Range<u64>> =
            (ranges.into_iter()).filter(|r| r.start < r.end).collect();
        ranges.sort_by_key(|r| r.start);
        let mut joined: Vec<Range<u64>> = Vec::new();
        for r in ranges {
            match joined.last_mut() {
                Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                _ => joined.push(r),
            }
        }
        Resolutions {
            wholes: (joined.into_iter())
                .map(|range| Whole {
                    range,
                    segments: OnceLock::new(),
                })
                .collect(),
            ..Resolutions::default()
        }
    }

    /// The import's bytes `range` as segments: cut from the whole that holds it, resolved at its
    /// first ask; else from the latest kept range that holds it; else `resolve`d now and kept.
    pub(crate) fn segments(
        &self,
        range: &Range<u64>,
        resolve: impl FnOnce(&Range<u64>) -> Vec<Segment>,
    ) -> Segments<'_> {
        let holds = |r: &Range<u64>| r.start <= range.start && range.end <= r.end;
        let at = self.wholes.partition_point(|w| w.range.end <= range.start);
        if let Some(whole) = self.wholes.get(at).filter(|w| holds(&w.range)) {
            let segments =
                (whole.segments).get_or_init(|| self.resolve(&whole.range, resolve).into());
            return Segments {
                held: Held::Whole(segments),
                range: range.clone(),
            };
        }
        let holding = {
            let kept = self.kept.lock();
            (kept.iter().rev())
                .find(|(r, _)| holds(r))
                .map(|(_, segments)| Arc::clone(segments))
        };
        if let Some(segments) = holding {
            return Segments {
                held: Held::Kept(segments),
                range: range.clone(),
            };
        }
        let segments: Arc<[Segment]> = self.resolve(range, resolve).into();
        let gone = {
            let mut kept = self.kept.lock();
            // A range the new one holds is cut from it from now on.
            let mut gone: Vec<Kept> = (kept
                .extract_if(.., |(r, _)| range.start <= r.start && r.end <= range.end))
            .collect();
            if kept.len() >= KEPT {
                gone.push(kept.remove(0));
            }
            kept.push((range.clone(), Arc::clone(&segments)));
            gone
        };
        // What went is let go of outside the lock.
        drop(gone);
        Segments {
            held: Held::Kept(segments),
            range: range.clone(),
        }
    }

    /// `resolve`'s answer for `range`, counted.
    fn resolve(
        &self,
        range: &Range<u64>,
        resolve: impl FnOnce(&Range<u64>) -> Vec<Segment>,
    ) -> Vec<Segment> {
        #[cfg(test)]
        self.resolutions.fetch_add(1, Ordering::Relaxed);
        resolve(range)
    }

    /// How many times a resolver was asked.
    #[cfg(test)]
    pub(crate) fn resolutions(&self) -> u64 {
        self.resolutions.load(Ordering::Relaxed)
    }
}

/// A range's segments ([`Resolutions::segments`]): the segments that hold them, cut to the range
/// as they are read.
#[derive(Debug)]
pub(crate) struct Segments<'a> {
    held: Held<'a>,
    range: Range<u64>,
}

/// The segments an answer is cut from: a whole's, or a kept range's.
#[derive(Debug)]
enum Held<'a> {
    Whole(&'a [Segment]),
    Kept(Arc<[Segment]>),
}

impl Segments<'_> {
    /// The segments, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = Segment> + '_ {
        let held: &[Segment] = match &self.held {
            Held::Whole(segments) => segments,
            Held::Kept(segments) => segments,
        };
        cut(held, &self.range)
    }

    /// The segments, to keep: the kept range's own where they are its every segment.
    pub(crate) fn shared(&self) -> Arc<[Segment]> {
        match &self.held {
            Held::Kept(segments) if spans(segments, &self.range) => Arc::clone(segments),
            _ => self.iter().collect(),
        }
    }
}

/// Whether `segments` start where `range` does and end where it does, so a cut to it keeps them.
fn spans(segments: &[Segment], range: &Range<u64>) -> bool {
    segments.first().map(|s| s.offset) == Some(range.start)
        && segments.last().map(|s| s.offset.saturating_add(s.length)) == Some(range.end)
}

/// The segments of `range`, which lies in the range `kept` was resolved for: those of `kept` that
/// meet it, the first and the last cut to it.
pub(crate) fn cut<'a>(
    kept: &'a [Segment],
    range: &Range<u64>,
) -> impl Iterator<Item = Segment> + 'a {
    let (first, end) = (range.start, range.end);
    let from = kept.partition_point(|s| s.offset.saturating_add(s.length) <= first);
    (kept.get(from..).unwrap_or(&[]).iter())
        .take_while(move |s| s.offset < end)
        .map(move |s| {
            let start = s.offset.max(first);
            let stop = s.offset.saturating_add(s.length).min(end);
            Segment {
                offset: start,
                length: stop.saturating_sub(start),
                guest: s
                    .guest
                    .map(|g| g.saturating_add(start.saturating_sub(s.offset))),
            }
        })
}

#[cfg(test)]
mod tests;
