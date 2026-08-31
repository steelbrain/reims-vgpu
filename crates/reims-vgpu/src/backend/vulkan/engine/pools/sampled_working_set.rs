//! How many distinct sampled windows the workload wants live at once.
//!
//! # The number `AGENTS.md` asks for before a bound moves
//!
//! The sampled cache has two caps and a victim ledger that bands the *eviction
//! route* — which cap fired. `sampled_evict_route`'s own doc says what that
//! cannot answer and refuses to act without it:
//!
//! > nothing yet counts how many distinct `(key, identity)` windows the workload
//! > wants live at once, and that is the number `AGENTS.md` requires before a
//! > bound moves.
//!
//! The victim ledger is the closest thing to it and it is censored: it remembers
//! `SAMPLED_REACH_BAND * 8` = 512 evictions, and a driven macos-26 boot reports
//! `sampled_reach_beyond_ledger` **6 704** times. A reading that falls off the
//! end of its instrument is not a large reading, it is no reading, and "raise
//! the cap" cannot be argued from it in either direction.
//!
//! This counts the requested set directly and is not bounded by the cache: every
//! sampled bind names a `(SampledKey, SampledContentIdentity)` before any cache
//! is consulted, and the distinct count of those in one census window is the
//! working set — what a cache would have had to hold to serve the window with no
//! miss at all.
//!
//! # Read the two series together or not at all
//!
//! `distinct` alone re-runs the mistake the eviction route was built to expose:
//! a count cap raised without the byte cap hands every eviction straight to the
//! other route. So the bytes those distinct windows would have cost are carried
//! beside the count, and both are compared against both caps on the same line.
//!
//! # A bind with no identity is not part of the answer
//!
//! It could not have hit at any cache size, so it is counted apart rather than
//! given a synthetic key. Folding those in would inflate the working set with
//! windows no cap change could help.

use std::collections::HashMap;

use super::{SampledKey, SAMPLED_CACHE_BYTE_CAP, SAMPLED_REACH_BAND};
use crate::backend::vulkan::engine::SampledContentIdentity;

/// What one census window asked for.
#[derive(Default)]
struct Window {
    /// Distinct `(key, identity)` windows, each with the bytes it would occupy.
    /// The value is the content length; a repeat of the same pair overwrites it
    /// with the same number rather than adding, which is the whole point.
    wanted: HashMap<(SampledKey, SampledContentIdentity), usize>,
    /// Binds naming no identity. Counted, never inserted — see the module doc.
    no_identity: u64,
    /// Distinct windows refused because [`Window::CAPACITY`] was reached.
    ///
    /// **Non-zero makes the window's `distinct` a floor rather than a count.**
    /// The line says so itself rather than leaving a reader to infer it, because
    /// a censored working set that reads like a measured one is exactly the
    /// failure this module exists to replace.
    dropped: u64,
}

impl Window {
    /// The most distinct windows one census second may track.
    ///
    /// Deliberately far above any plausible answer rather than fitted to one:
    /// the question is whether the workload wants more than the 64-entry cache,
    /// so a bound that could itself be the binding constraint would beg it.
    /// Eight thousand is 125x the cache and ~16x the victim ledger this
    /// replaces, and `dropped` is what says if that ever stops being enough.
    const CAPACITY: usize = 8192;

    fn want(&mut self, key: SampledKey, identity: Option<SampledContentIdentity>, bytes: usize) {
        let Some(identity) = identity else {
            self.no_identity += 1;
            return;
        };
        let entry = (key, identity);
        if !self.wanted.contains_key(&entry) && self.wanted.len() >= Self::CAPACITY {
            self.dropped += 1;
            return;
        }
        self.wanted.insert(entry, bytes);
    }

    /// The line, or `None` when no sampled bind happened this window.
    ///
    /// Takes and clears: this is a per-window set, not a high-water, and the two
    /// are read differently. A reader summing these across a boot is summing
    /// overlapping sets and will get a number larger than anything that was ever
    /// wanted at once.
    fn take(&mut self) -> Option<String> {
        if self.wanted.is_empty() && self.no_identity == 0 {
            return None;
        }
        let distinct = self.wanted.len();
        let bytes: usize = self.wanted.values().sum();
        let line = format!(
            "sampled_working_set distinct={distinct} mib={:.1} no_identity={} dropped={} \
             count_cap={SAMPLED_REACH_BAND} byte_cap_mib={} \
             (distinct (key, identity) windows this census second asked for, and what holding \
              all of them would cost; a per-window set, not a high-water — do not sum across \
              windows)",
            bytes as f64 / (1024.0 * 1024.0),
            self.no_identity,
            self.dropped,
            SAMPLED_CACHE_BYTE_CAP / (1024 * 1024),
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

/// Record that a sampled bind wanted this window, before any cache is consulted.
pub(crate) fn note_wanted(key: SampledKey, identity: Option<SampledContentIdentity>, bytes: usize) {
    window()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .want(key, identity, bytes);
}

/// Drain the window's set into a census line.
pub fn census() -> Option<String> {
    window().lock().unwrap_or_else(|e| e.into_inner()).take()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One fixed key: every test here varies the *identity*, which is the half
    /// that decides whether two binds are the same window for a cache keyed on
    /// the image.
    fn key() -> SampledKey {
        SampledKey {
            width: 64,
            height: 64,
            layers: 1,
            volume: false,
            cube: false,
            arrayed: false,
            one_dim: false,
            format: ash::vk::Format::R8G8B8A8_UNORM,
            swizzle: Default::default(),
        }
    }

    fn id(key: u64, generation: u64) -> Option<SampledContentIdentity> {
        Some(SampledContentIdentity { key, generation })
    }

    /// The set is the point: asking for one window a thousand times wants one
    /// window, and a cache sized from the bind count rather than the set would
    /// be sized from the frame rate.
    #[test]
    fn asking_for_one_window_repeatedly_wants_one_window() {
        let mut w = Window::default();
        for _ in 0..1000 {
            w.want(key(), id(7, 1), 2 << 20);
        }
        let line = w.take().expect("a bind happened");
        assert!(line.contains("distinct=1"), "{line}");
        assert!(line.contains("mib=2.0"), "{line}");
    }

    /// A new generation of the same resource is a different window — it is what
    /// the cache must hold separately, so it must be what this counts.
    #[test]
    fn a_new_generation_of_one_resource_is_a_second_window() {
        let mut w = Window::default();
        w.want(key(), id(7, 1), 1 << 20);
        w.want(key(), id(7, 2), 1 << 20);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("distinct=2"), "{line}");
        assert!(line.contains("mib=2.0"), "{line}");
    }

    /// A bind with no identity could not have hit at any cache size, so it must
    /// not inflate the set a cap decision is read off.
    #[test]
    fn a_bind_with_no_identity_is_counted_apart_from_the_set() {
        let mut w = Window::default();
        w.want(key(), None, 4 << 20);
        let line = w.take().expect("a bind happened");
        assert!(line.contains("distinct=0"), "{line}");
        assert!(line.contains("no_identity=1"), "{line}");
        assert!(line.contains("mib=0.0"), "{line}");
    }

    /// The window clears on read. Summing successive windows would be summing
    /// overlapping sets; a stale residue would make that worse by making the
    /// second window look like the first plus itself.
    #[test]
    fn taking_a_window_clears_it() {
        let mut w = Window::default();
        w.want(key(), id(1, 1), 1 << 20);
        assert!(w.take().is_some());
        assert!(w.take().is_none());
    }

    /// The bound refuses new windows and says so on the line, rather than
    /// silently reporting a censored set as a measured one — which is precisely
    /// what the 512-entry victim ledger did on macos-26.
    #[test]
    fn a_full_window_reports_that_its_count_is_a_floor() {
        let mut w = Window::default();
        for i in 0..Window::CAPACITY as u64 {
            w.want(key(), id(i, 0), 0);
        }
        w.want(key(), id(u64::MAX, 0), 0);
        let line = w.take().expect("a bind happened");
        assert!(
            line.contains(&format!("distinct={}", Window::CAPACITY)),
            "{line}"
        );
        assert!(line.contains("dropped=1"), "{line}");
    }
}
