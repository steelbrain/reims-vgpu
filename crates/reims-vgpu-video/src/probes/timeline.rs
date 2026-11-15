//! A record of the driver calls a thread makes while some work runs ([`record`]), for the probes
//! that time them and the tests that check a call is not made:
//! each call by its name, at the time its answer is checked (the VA-API module's `check`, CUDA's
//! `Cuda::check`, which [`stamp`] each answer), so that the time from one answer to the next is
//! that call's, with whatever the crate did between the two.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::time::Duration;

use crate::core::runtime::{MonoTime, Runtime as _, StdRuntime};

thread_local! {
    /// The calls of the record running on this thread, if one is.
    static CALLS: RefCell<Option<Vec<(&'static str, MonoTime)>>> = const { RefCell::new(None) };
}

/// Notes that `call`'s answer is in, where a record runs on this thread.
pub(crate) fn stamp(call: &'static str) {
    CALLS.with_borrow_mut(|calls| {
        if let Some(calls) = calls {
            calls.push((call, StdRuntime.now()));
        }
    });
}

/// `work`'s answer, and the time it took with the calls it made on this thread.
pub(crate) fn record<T>(work: impl FnOnce() -> T) -> (T, Span) {
    CALLS.with_borrow_mut(|calls| *calls = Some(Vec::new()));
    let started = StdRuntime.now();
    let answer = work();
    let ended = StdRuntime.now();
    let calls = CALLS.with_borrow_mut(Option::take).unwrap_or_default();
    let mut steps = Vec::with_capacity(calls.len());
    let mut last = started;
    for (call, at) in calls {
        steps.push((call, at.saturating_duration_since(last)));
        last = at;
    }
    let after = ended.saturating_duration_since(last);
    let span = Span {
        took: ended.saturating_duration_since(started),
        steps,
        after,
    };
    (answer, span)
}

/// What [`record`] timed.
pub(crate) struct Span {
    /// The work's time, from its start to its answer.
    pub(crate) took: Duration,
    /// Each call the work made, in order, with the time from the answer before it (or the work's
    /// start) to its own.
    pub(crate) steps: Vec<(&'static str, Duration)>,
    /// From the last call's answer to the work's.
    pub(crate) after: Duration,
}

impl Span {
    /// The time of the steps named `call`, together.
    pub(crate) fn of(&self, call: &str) -> Duration {
        (self.steps.iter())
            .filter(|(name, _)| *name == call)
            .map(|(_, took)| *took)
            .sum()
    }

    /// `took [call t, call t xN, ..., rest t]`, in milliseconds, the steps of one call in a row
    /// summed with their count.
    pub(crate) fn describe(&self) -> String {
        let mut text = format!("{} [", ms(self.took));
        let mut steps = self.steps.iter().peekable();
        while let Some(&(call, took)) = steps.next() {
            let (mut sum, mut count) = (took, 1);
            while let Some(&&(next, more)) = steps.peek()
                && next == call
            {
                sum += more;
                count += 1;
                steps.next();
            }
            let _ = write!(text, "{call} {}", ms(sum));
            if count > 1 {
                let _ = write!(text, " x{count}");
            }
            text.push_str(" | ");
        }
        let _ = write!(text, "rest {}]", ms(self.after));
        text
    }
}

/// `d` in milliseconds, to the hundredth.
pub(crate) fn ms(d: Duration) -> String {
    format!("{:.2}", d.as_secs_f64() * 1e3)
}
