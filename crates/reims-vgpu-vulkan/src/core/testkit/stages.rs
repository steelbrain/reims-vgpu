//! Fake: a platform's stage fills ([`FakeStages`]), what the fake's refusals of a stage
//! ([`Step::Stage`](super::script::Step::Stage), [`FakeStages::refuse_next_prepare`]) hand their
//! stage to, and the test's hands on each fill.
//!
//! The fills are a [`StageFills`] of `u64` stages whose start runs nothing: each fill runs until
//! the test ends it ([`FakeStages::end`]) or a cancel ends it. The fake answers a stage's refusal
//! as the rt platform does: a dispatch waits for its stage's fill, and is made once more after the
//! fill filled the store, else answered as refused ([`FakeStages::refused`]); a make ahead hands
//! its stage over and waits for nothing.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::core::platform::stage::{Cancel, FillDone, FillEnd, Site, StageFills};
use crate::core::sync::Mutex;

/// The fake's stage fills, and what the test sees of them.
pub struct FakeStages {
    fills: StageFills<u64>,
    seen: Arc<Seen>,
}

impl std::fmt::Debug for FakeStages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeStages")
            .field("started", &self.started())
            .field("cancels", &self.cancels())
            .finish_non_exhaustive()
    }
}

/// What the test sees of the fills.
#[derive(Default)]
struct Seen {
    /// Each fill started, by stage, with the site it was started for, in order, with its end until
    /// the test ends it.
    started: Mutex<Vec<(u64, Site, Option<FillDone>)>>,
    /// How many fills a cancel ended.
    cancels: AtomicUsize,
    /// The calls answered as refused after their stage's fill: their selector and stage.
    refused: Mutex<Vec<(&'static str, u64)>>,
    /// The stages the next makes ahead refuse, one each, in order.
    prepares: Mutex<VecDeque<u64>>,
}

impl Default for FakeStages {
    fn default() -> FakeStages {
        FakeStages::new()
    }
}

impl FakeStages {
    /// Fills that run until the test ends them.
    pub fn new() -> FakeStages {
        let seen = Arc::new(Seen::default());
        let theirs = Arc::clone(&seen);
        let start = move |&stage: &u64, site: Site, done: FillDone| -> Cancel {
            theirs.started.lock().push((stage, site, Some(done)));
            let seen = Arc::clone(&theirs);
            Box::new(move || {
                seen.cancels.fetch_add(1, Ordering::SeqCst);
            })
        };
        let fills = StageFills::new(Box::new(start));
        FakeStages { fills, seen }
    }

    /// The fills, which the fake's refusals hand their stage to.
    pub fn fills(&self) -> &StageFills<u64> {
        &self.fills
    }

    /// The stage of each fill started, in order.
    pub fn started(&self) -> Vec<u64> {
        let started = self.seen.started.lock();
        started.iter().map(|&(stage, _, _)| stage).collect()
    }

    /// The stage of each fill started and the site it was started for, in order.
    pub fn started_at(&self) -> Vec<(u64, Site)> {
        let started = self.seen.started.lock();
        started
            .iter()
            .map(|&(stage, site, _)| (stage, site))
            .collect()
    }

    /// Ends the first fill of `stage` the test has not ended with `end`, on this thread; false
    /// where there is none.
    pub fn end(&self, stage: u64, end: FillEnd) -> bool {
        let done = {
            let mut started = self.seen.started.lock();
            started
                .iter_mut()
                .find(|(of, _, done)| *of == stage && done.is_some())
                .and_then(|(_, _, done)| done.take())
        };
        match done {
            Some(done) => {
                done.end(end);
                true
            }
            None => false,
        }
    }

    /// How many fills a cancel ended.
    pub fn cancels(&self) -> usize {
        self.seen.cancels.load(Ordering::SeqCst)
    }

    /// The calls answered as refused after their stage's fill, each its selector and stage, in
    /// order.
    pub fn refused(&self) -> Vec<(&'static str, u64)> {
        self.seen.refused.lock().clone()
    }

    /// Notes the call `sel` answered as refused after the fill of `stage`.
    pub(crate) fn note_refused(&self, sel: &'static str, stage: u64) {
        self.seen.refused.lock().push((sel, stage));
    }

    /// Has the next make ahead not yet refused refuse for `stage`, which it hands to the fills.
    pub fn refuse_next_prepare(&self, stage: u64) {
        self.seen.prepares.lock().push_back(stage);
    }

    /// The stage a make ahead refuses for, where the test asked ([`Self::refuse_next_prepare`]).
    pub(crate) fn prepare_refusal(&self) -> Option<u64> {
        self.seen.prepares.lock().pop_front()
    }
}
