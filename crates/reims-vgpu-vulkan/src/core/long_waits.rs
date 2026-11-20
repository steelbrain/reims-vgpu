//! Our own waits of [`LONG`] or more, by name, for the host's summary of each ten seconds
//! (reims-vgpu's `graphics::summary`). A site that waits for another thread, the GPU, the runtime
//! or the window system times its wait ([`begin`], [`Began::end`]) under a name of its own, and
//! the summary takes each name's count, total and longest every period ([`take`]). A diagnostic:
//! nothing is counted until the summary turns counting on ([`count`]), where the log names its
//! target; off, a wait costs one relaxed load. A wait is timed on the platform's clock, the trace's
//! ([`crate::core::trace::now`]).
//!
//! The names, each a wait of the calling thread:
//! - `lane-turn`: a command buffer's wait for the buffers before it in its queue's lane.
//! - `sequencer-tickets` and `sequencer-ranges`: a commit's wait for the tickets before its mark,
//!   and for its context's range messages, to be carried out.
//! - `commit-order`: a commit's wait for buffers of other queues to commit; `commit-order-begun`:
//!   a buffer's wait, before it takes its place among the commit order's tickets, for those
//!   buffers to take theirs.
//! - `deal-chunk`: a walk's wait for a chunk it dealt to a helper; `helper-run`: a walk's wait for
//!   a helper's run of its job to end.
//! - `library-make`: a wait for a library being made.
//! - `surface-worker`: a wait for the surface worker's jobs of a block.
//! - `video-fill`: a video op's wait for a cache's fill.
//! - `surface-order-cpu`: a wait on the CPU for a buffer of another queue to finish.
//! - `pool-take`: the pool's take of a block from rt.
//! - `window-take`, `window-fence`, `window-scheduled` and `window-present`: a window's take of an
//!   image, its wait for the image's fence, its wait for its frame's command buffer to be
//!   scheduled, and its present, with the window system's lock.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::core::sync::Mutex;
use crate::core::trace;

/// The shortest wait counted.
pub const LONG: Duration = Duration::from_millis(5);

/// Whether waits are counted ([`count`]).
static ON: AtomicBool = AtomicBool::new(false);

/// The waits counted since the last [`take`], by name.
static WAITS: Mutex<BTreeMap<&'static str, Waits>> = Mutex::new(BTreeMap::new());

/// The waits of one name, of [`LONG`] or more.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Waits {
    /// How many.
    pub count: u64,
    /// Their time in all.
    pub total: Duration,
    /// The longest.
    pub longest: Duration,
}

/// Turns counting on, for the rest of the process.
pub fn count() {
    ON.store(true, Ordering::Relaxed);
}

/// Whether waits are counted.
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// A wait's start, where counting is on.
#[must_use = "a wait is counted only at its end"]
pub struct Began(Option<u64>);

/// A wait starts now.
pub fn begin() -> Began {
    Began(on().then(trace::now))
}

impl Began {
    /// The wait has ended: counted under `name` where it took [`LONG`] or more.
    pub fn end(self, name: &'static str) {
        if let Some(began) = self.0 {
            waited(
                name,
                Duration::from_nanos(trace::now().saturating_sub(began)),
            );
        }
    }
}

/// A wait of `name` took `took`, as the site timed it: counted where it took [`LONG`] or more and
/// counting is on.
pub fn waited(name: &'static str, took: Duration) {
    if took < LONG || !on() {
        return;
    }
    let mut waits = WAITS.lock();
    let of = waits.entry(name).or_default();
    of.count = of.count.saturating_add(1);
    of.total = of.total.saturating_add(took);
    of.longest = of.longest.max(took);
}

/// The waits counted since the last take, by name in its order; they are counted from none again.
pub fn take() -> Vec<(&'static str, Waits)> {
    std::mem::take(&mut *WAITS.lock()).into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off, no wait is counted; on, a wait of [`LONG`] or more is, by its name, and one shorter
    /// is not; a take answers them once.
    #[test]
    fn waits_of_5_ms_or_more_are_counted_by_name_once_on() {
        let ms = Duration::from_millis;
        waited("lane-turn", ms(9));
        assert_eq!(take(), []);
        count();
        waited("lane-turn", ms(9));
        waited("lane-turn", ms(6));
        waited("lane-turn", ms(4));
        waited("commit-order", ms(5));
        let three = |count, total, longest| Waits {
            count,
            total: ms(total),
            longest: ms(longest),
        };
        assert_eq!(
            take(),
            [
                ("commit-order", three(1, 5, 5)),
                ("lane-turn", three(2, 15, 9))
            ]
        );
        assert_eq!(take(), []);
    }
}
