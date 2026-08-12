//! How many distinct guest buffer windows the workload gathers, and how often
//! it gathers the same one twice.
//!
//! # The number `AGENTS.md` asks for before a cache exists
//!
//! The draw-time buffer gather is the largest remaining GPU cost on the x86
//! rail: a driven macos-13 boot issues ~34 000 gathers a second moving ~5.8 GB
//! of guest bytes, and `runtime::draw::vulkan`'s `try_buffer_zero_copy_resolved`
//! records that **99.2 % of them re-copy a span this device already copied**.
//! That number is what makes a content cache look like a 100x lever rather than
//! a few percent.
//!
//! It is also not enough to build one on, and the census it came from is gone.
//! A recurrence rate says the same *key* comes back; it does not say how many
//! distinct keys have to be held at once to catch them, and that is the number
//! a cap is chosen from. The sampled cache learned this the expensive way —
//! [`super::sampled_working_set`]'s own doc opens on a bound that was argued
//! from an instrument whose reach the workload had already exceeded.
//!
//! So this counts the requested set directly, before any cache exists to bias
//! it, and it is deliberately not bounded by one.
//!
//! # What identifies a window here
//!
//! The `Arc<Vec<GuestWindowRun>>` behind the gather's [`super::super::types::GuestRunSource`],
//! by address, paired with the window's byte length. That allocation is owned by
//! [`crate::runtime::bound_buffers`], which resolves a bind once and holds it
//! until the guest moves an address — so the pointer is stable across
//! submissions for exactly as long as the resolution it names is, which is the
//! span a cache would have to survive. It is the same key the engine's
//! within-command-buffer bind map already dedupes on, one scope wider.
//!
//! Holding the `Arc` is **not** required to make the address sound as a key
//! here, because nothing is looked up by it: two windows colliding on a recycled
//! address would merge two entries in a count, which is a censused number and
//! not a bind. A cache keyed this way would owe the `Arc`; this does not.
//!
//! # Read `distinct` and `mib` together, and neither alone
//!
//! A count cap chosen without the byte cap hands every eviction to the other
//! one. Both are on the same line for that reason, and `gathers` beside them is
//! what makes the recurrence rate readable as `1 - distinct / gathers` for this
//! window rather than quoted from a boot nobody can re-run.
//!
//! # What it said
//!
//! Two driven macos-13 sustained-animation boots, a census second each:
//!
//! ```text
//! distinct=1897  mib=287.7  gathers=20771  mib_moved=3520.5  recur=0.909  dropped=0
//! distinct=1781  mib=271.7  gathers=19452  mib_moved=3307.5  recur=0.908  dropped=0
//! ```
//!
//! Three things follow and they do not all point the same way.
//!
//! * **A cache of unbounded size would be *asked* for 91 % of the gathers.** It
//!   would not serve them. `crate::runtime::buffer_gather_freshness`'s content
//!   audit — 21 204 comparisons over two driven boots — found only **~27 % of
//!   repeats unchanged**, so the ceiling on a content cache with a perfect
//!   oracle is `0.91 x 0.27 = 25 %` of the gathers, not 91 %. A repeat whose
//!   bytes moved has to be re-copied by any cache.
//!
//!   Read `recur` as *the population a cache would be asked about* and never as
//!   the population it would serve. This distinction is the whole reason that
//!   audit exists: the number here was being quoted as the size of the prize.
//! * **It is 91 % and not the 99.2 % the buffer rail is usually argued from.**
//!   The older figure came off a window-drag probe and a census that no longer
//!   exists; this is the sustained-animation population. Quote the one whose
//!   probe is named, per `AGENTS.md`: they are two workloads and both are real.
//! * **Unbounded means 288 MiB of device-local memory**, held across
//!   submissions, on a host whose whole gather pool is transient today. That is
//!   the number a cap has to be argued against, and it is large enough that the
//!   cap will bind — so the eviction policy is part of the design and not a
//!   detail to add later. `dropped=0` says the 1 897 is a count and not a floor.
//!
//! Recurrence is about **keys**, not bytes: it says the same window comes back,
//! not that its contents are unchanged. What would make a hit sound is
//! [`crate::runtime::gather_witness`], which already carries exactly this
//! argument for the sampled rails — the hypervisor dirty bitmap for guest CPU
//! stores and [`crate::runtime::host_writes`] for this device's own. Its
//! `MAX_TRACKED_WINDOWS` is 256 against the 1 897 here, so adopting it is a
//! resize as well as a wiring.

use std::collections::HashMap;

/// What one census window asked for.
#[derive(Default)]
struct Window {
    /// Distinct `(runs allocation, byte length)` windows, each with its length.
    /// A repeat overwrites the same number rather than adding, which is the
    /// whole point.
    wanted: HashMap<(usize, u64), u64>,
    /// Gathers this window, including the repeats `wanted` collapses.
    gathers: u64,
    /// Bytes those gathers moved, including the repeats.
    bytes: u64,
    /// Distinct windows refused because [`Window::CAPACITY`] was reached.
    ///
    /// **Non-zero makes `distinct` a floor rather than a count**, and the line
    /// says so itself rather than leaving a reader to infer it.
    dropped: u64,
}

impl Window {
    /// The most distinct windows one census second may track.
    ///
    /// Deliberately far above any plausible answer rather than fitted to one:
    /// the question this exists to answer is how large a cache would have to be,
    /// so a bound that could itself be the binding constraint would beg it. A
    /// driven boot holds 704 `bound_buffers` resolutions at once, and every
    /// gathered window is one of those, so this is ~23x the only related number
    /// anyone has measured. `dropped` is what says if that stops being enough.
    const CAPACITY: usize = 16384;

    fn want(&mut self, runs: usize, len: u64) {
        self.gathers += 1;
        self.bytes = self.bytes.saturating_add(len);
        let entry = (runs, len);
        if !self.wanted.contains_key(&entry) && self.wanted.len() >= Self::CAPACITY {
            self.dropped += 1;
            return;
        }
        self.wanted.insert(entry, len);
    }

    /// The line, or `None` when nothing gathered this window.
    ///
    /// Takes and clears: this is a per-window set and not a high-water. A reader
    /// summing these across a boot is summing overlapping sets and gets a number
    /// larger than anything that was ever wanted at once.
    fn take(&mut self) -> Option<String> {
        if self.gathers == 0 {
            return None;
        }
        let distinct = self.wanted.len();
        let held: u64 = self.wanted.values().sum();
        // Sound because the early return above rules out a zero denominator.
        // Overstated while `dropped` is non-zero, since a censored `distinct` is
        // a floor — which is the reason both are on the line.
        let recur = 1.0 - (distinct as f64 / self.gathers as f64);
        let line = format!(
            "buffer_gather_working_set distinct={distinct} mib={:.1} gathers={} mib_moved={:.1} \
             recur={:.3} dropped={} \
             (distinct guest buffer windows gathered this census second and what holding all of \
              them would cost, beside the gathers that asked; recur is the share a content cache \
              of unbounded size would have served. A per-window set, not a high-water — do not \
              sum across windows.)",
            held as f64 / (1024.0 * 1024.0),
            self.gathers,
            self.bytes as f64 / (1024.0 * 1024.0),
            recur,
            self.dropped,
        );
        *self = Self::default();
        Some(line)
    }
}

fn window() -> &'static std::sync::Mutex<Window> {
    use std::sync::{Mutex, OnceLock};
    static WINDOW: OnceLock<Mutex<Window>> = OnceLock::new();
    WINDOW.get_or_init(|| Mutex::new(Window::default()))
}

/// Record that a draw-time buffer bind gathered this window.
///
/// Called where the gather is *taken*, not where it is considered: a window the
/// engine binds in place costs nothing and would be a window no cache has to
/// hold.
pub(crate) fn note_gathered(runs: usize, len: u64) {
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .want(runs, len);
}

/// Drain the window's set into a census line.
pub fn census() -> Option<String> {
    window().lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The set is the point: gathering one window a thousand times wants one
    /// window, and a cache sized from the gather count rather than the set would
    /// be sized from the frame rate.
    #[test]
    fn gathering_one_window_repeatedly_wants_one_window() {
        let mut w = Window::default();
        for _ in 0..1000 {
            w.want(0xdead_0000, 64 << 10);
        }
        let line = w.take().expect("a gather happened");
        assert!(line.contains("distinct=1"), "{line}");
        assert!(line.contains("gathers=1000"), "{line}");
        assert!(line.contains("mib=0.1"), "{line}");
        assert!(line.contains("recur=0.999"), "{line}");
    }

    /// A resolution rebound at a different length is a different window: a cache
    /// serving the shorter one across would hand the GPU a buffer that stops
    /// before the bind does.
    #[test]
    fn one_allocation_at_two_lengths_is_two_windows() {
        let mut w = Window::default();
        w.want(0xbeef_0000, 1 << 10);
        w.want(0xbeef_0000, 2 << 10);
        let line = w.take().expect("two gathers happened");
        assert!(line.contains("distinct=2"), "{line}");
        assert!(line.contains("gathers=2"), "{line}");
    }

    /// Nothing gathered is no line, so a boot's idle seconds do not publish a
    /// zero that reads like a measured working set.
    #[test]
    fn a_window_with_no_gather_publishes_nothing() {
        let mut w = Window::default();
        assert!(w.take().is_none());
    }

    /// Taking clears. Two census seconds are two sets, and the second must not
    /// carry the first's.
    #[test]
    fn the_set_does_not_carry_into_the_next_census_second() {
        let mut w = Window::default();
        w.want(1, 4096);
        w.take().expect("a gather happened");
        w.want(2, 4096);
        let line = w.take().expect("a gather happened");
        assert!(line.contains("distinct=1"), "{line}");
        assert!(line.contains("gathers=1"), "{line}");
    }

    /// Past the capacity the count is a floor, and the line has to say so —
    /// a censored working set that reads like a measured one is the failure
    /// this module exists to replace.
    #[test]
    fn a_set_past_the_capacity_reports_what_it_dropped() {
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64 + 5) {
            w.want(i as usize, 4096);
        }
        let line = w.take().expect("gathers happened");
        assert!(
            line.contains(&format!("distinct={}", Window::CAPACITY)),
            "{line}"
        );
        assert!(line.contains("dropped=5"), "{line}");
    }

    /// A window already in the set is still counted past the capacity, so the
    /// gather count stays exact even when `distinct` is censored.
    #[test]
    fn a_repeat_past_the_capacity_still_counts_as_a_gather() {
        let mut w = Window::default();
        for i in 0..(Window::CAPACITY as u64) {
            w.want(i as usize, 4096);
        }
        w.want(0, 4096);
        w.want(999_999, 4096);
        let line = w.take().expect("gathers happened");
        assert!(line.contains("dropped=1"), "{line}");
        assert!(
            line.contains(&format!("gathers={}", Window::CAPACITY + 2)),
            "{line}"
        );
    }
}
