//! The ranges of an import the dirty log resolved to segments of the guest's RAM, kept so that a
//! range asked again, or one inside a kept one, is answered without the backing table
//! (`os/linux/dirty.rs`'s module docs, "Resolved once").
//!
//! Plain data over a resolver its caller passes, so how it behaves under the runtime's reporting
//! threads is measured on any host: the ignored test `reports_cost`, run by hand in a release
//! build (`cargo test -p reims-vgpu-vulkan --release --lib -- --ignored --exact
//! os::resolutions::tests::reports_cost --nocapture`).

use std::ops::Range;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_core::memory::dirty::Segment;
use reims_vgpu_core::sync::Mutex;

/// The ranges of an import whose segments are kept: the oldest goes when another is resolved.
pub(crate) const KEPT: usize = 4;

/// A range of an import, resolved, and its segments.
type Kept = (Range<u64>, Arc<[Segment]>);

/// An import's kept resolutions (module docs).
#[derive(Debug, Default)]
pub(crate) struct Resolutions {
    /// The ranges resolved so far, each with its segments, the latest last.
    kept: Mutex<Vec<Kept>>,
    /// How many times a resolver was asked.
    #[cfg(test)]
    resolutions: AtomicU64,
}

impl Resolutions {
    /// The import's bytes `range` as segments: cut from a range resolved before that holds it, or
    /// `resolve`d now and kept.
    pub(crate) fn segments(
        &self,
        range: &Range<u64>,
        resolve: impl FnOnce(&Range<u64>) -> Vec<Segment>,
    ) -> Arc<[Segment]> {
        {
            let kept = self.kept.lock();
            let holding =
                (kept.iter().rev()).find(|(r, _)| r.start <= range.start && range.end <= r.end);
            match holding {
                Some((r, segments)) if r == range => return Arc::clone(segments),
                Some((_, segments)) => return cut(segments, range).into(),
                None => {}
            }
        }
        #[cfg(test)]
        self.resolutions.fetch_add(1, Ordering::Relaxed);
        let segments: Arc<[Segment]> = resolve(range).into();
        let mut kept = self.kept.lock();
        // A range the new one holds is cut from it from now on.
        kept.retain(|(r, _)| !(range.start <= r.start && r.end <= range.end));
        if kept.len() >= KEPT {
            kept.remove(0);
        }
        kept.push((range.clone(), Arc::clone(&segments)));
        segments
    }

    /// How many times a resolver was asked.
    #[cfg(test)]
    pub(crate) fn resolutions(&self) -> u64 {
        self.resolutions.load(Ordering::Relaxed)
    }
}

/// The segments of `range`, which lies in the range `kept` was resolved for: those of `kept` that
/// meet it, the first and the last cut to it.
pub(crate) fn cut(kept: &[Segment], range: &Range<u64>) -> Vec<Segment> {
    let first = kept.partition_point(|s| s.offset.saturating_add(s.length) <= range.start);
    (kept.get(first..).unwrap_or(&[]).iter())
        .take_while(|s| s.offset < range.end)
        .map(|s| {
            let start = s.offset.max(range.start);
            let end = s.offset.saturating_add(s.length).min(range.end);
            Segment {
                offset: start,
                length: end.saturating_sub(start),
                guest: s
                    .guest
                    .map(|g| g.saturating_add(start.saturating_sub(s.offset))),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
