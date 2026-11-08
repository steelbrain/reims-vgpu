//! The guest's RAM that the host's own devices write, outside any guest GPU work: the frames the
//! graphics device's video arms decode into an IOSurface's pages, by the CPU or by a device of the
//! host.
//!
//! A graphics device keeps the GPU's copy of an IOSurface in its GPU's memory where the kext asks
//! (`graphics/decode/surface_cache.rs`), filled from the pages only when the guest's CPU
//! generation of the surface moves, which such a write does not move. Each graphics device
//! watches its caches' pages here, and a write over any of them marks that cache stale: its next
//! use fills it from the pages, and no download copies it over them.
//!
//! The pages are named by their host addresses in the mapping of the guest's RAM the host gave
//! the device's memory map, which a graphics device reaches through its own map of the RAM. A watcher is
//! called on the writer's thread once the write is in the pages, before the guest is told of it.
//! The list's lock is a leaf: watchers are called after it is let go.

use std::ops::Range;
use std::sync::{Arc, Weak};

use reims_vgpu_core::sync::Mutex;

/// What watches the host's devices' writes.
pub(crate) trait Watcher: Send + Sync {
    /// The host's devices wrote `runs`, host addresses in the guest's RAM, in address order and
    /// apart ([`sorted`]).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the graphics device's video arms write the guest's pages"
        )
    )]
    fn wrote(&self, runs: &[Range<u64>]);
}

/// The live watchers, each held weakly: a watcher's last drop ends its watch.
static WATCHERS: Mutex<Vec<Weak<dyn Watcher>>> = Mutex::new(Vec::new());

/// Watches the host's devices' writes with `watcher` until its last drop.
pub(crate) fn watch(watcher: &Arc<dyn Watcher>) {
    let mut watchers = WATCHERS.lock();
    watchers.retain(|w| w.strong_count() > 0);
    watchers.push(Arc::downgrade(watcher));
}

/// A device of the host wrote `runs`, [`sorted`] (module docs): each live watcher is told.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the graphics device's video arms write the guest's pages"
    )
)]
pub(crate) fn wrote(runs: &[Range<u64>]) {
    if runs.is_empty() {
        return;
    }
    let live: Vec<Arc<dyn Watcher>> = {
        let watchers = WATCHERS.lock();
        watchers.iter().filter_map(Weak::upgrade).collect()
    };
    for watcher in live {
        watcher.wrote(runs);
    }
}

/// Host addresses of pages of `page` bytes each as runs, [`sorted`].
pub(crate) fn page_runs(pages: &[u64], page: u64) -> Vec<Range<u64>> {
    sorted(
        pages
            .iter()
            .map(|&at| at..at.saturating_add(page))
            .collect(),
    )
}

/// `runs` in address order, those that overlap or touch joined, empty ones dropped.
pub(crate) fn sorted(mut runs: Vec<Range<u64>>) -> Vec<Range<u64>> {
    runs.retain(|r| r.start < r.end);
    runs.sort_unstable_by_key(|r| r.start);
    let mut out: Vec<Range<u64>> = Vec::with_capacity(runs.len());
    for r in runs {
        match out.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => out.push(r),
        }
    }
    out
}

/// Whether any of `a` overlaps any of `b`, both [`sorted`].
pub(crate) fn overlap(a: &[Range<u64>], b: &[Range<u64>]) -> bool {
    let (mut i, mut j) = (0, 0);
    while let (Some(x), Some(y)) = (a.get(i), b.get(j)) {
        if x.start < y.end && y.start < x.end {
            return true;
        }
        if x.end <= y.end {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Seen(Mutex<Vec<Range<u64>>>);

    impl Watcher for Seen {
        fn wrote(&self, runs: &[Range<u64>]) {
            self.0.lock().extend_from_slice(runs);
        }
    }

    /// Each live watcher is told of each write; one dropped is told nothing more
    #[test]
    fn a_watcher_sees_writes_until_its_drop() {
        let seen = Arc::new(Seen(Mutex::new(Vec::new())));
        let watcher: Arc<dyn Watcher> = Arc::clone(&seen) as Arc<dyn Watcher>;
        watch(&watcher);
        // Addresses no other test writes, as the list is the process's.
        let at = 0x7e57_0000_0000;
        let runs = [at..at + 0x1000, at + 0x3000..at + 0x4000];
        wrote(&runs);
        wrote(&[]);
        drop(watcher);
        assert_eq!(seen.0.lock().as_slice(), runs);
        let weak = Arc::downgrade(&seen);
        drop(seen);
        assert!(weak.upgrade().is_none());
        wrote(&runs);
    }

    /// Pages make runs in address order, adjacent ones joined
    #[test]
    fn pages_make_runs() {
        assert_eq!(
            page_runs(&[0x5000, 0x1000, 0x2000, 0x6000, 0x3000, 0x2000], 0x1000),
            [0x1000..0x4000, 0x5000..0x7000]
        );
        assert!(page_runs(&[], 0x1000).is_empty());
        assert_eq!(sorted(vec![4..4, 2..3, 0..1]), [0..1, 2..3]);
    }

    /// Runs overlap where they share a byte, not where they only touch
    #[test]
    fn runs_overlap_where_they_share_a_byte() {
        let a = [0x1000..0x3000, 0x8000..0x9000];
        assert!(overlap(&a, std::slice::from_ref(&(0x2fff..0x3000))));
        assert!(overlap(&a, &[0..0x1000, 0x3000..0x8000, 0x8fff..0xa000]));
        assert!(!overlap(&a, &[0x3000..0x8000, 0x9000..0xa000]));
        assert!(overlap(std::slice::from_ref(&(0..u64::MAX)), &a));
        assert!(!overlap(&[], &a));
    }
}
