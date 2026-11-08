//! `Timer`: a repeating timer whose handler runs as a block of a serial queue.
//!
//! The one timer is the EFI present timer: start at the wall time of the call, interval 100 ms,
//! target `reims-vgpu.efi-present`. It fires at `start`, then at `start + k * interval`. A thread of
//! the timer's own waits on the platform's clock (virtual time in the fake) and merges each fire
//! into a DATA_ADD [`CoalescingSignal`] on the queue: fires the queue has not yet run coalesce
//! into one handler call (the data counts them), and a late wake skips to the next point of the
//! schedule.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::core::platform::os::{Deadline, MonoTime, TimerClock, WallTime};
use crate::core::raise::{Kept, keep_forever};
use crate::core::rt::queue::{Executor, SerialQueue};
use crate::core::rt::signal::CoalescingSignal;
use crate::core::sync::{Condvar, Mutex};

/// What the timer's thread and its handle share.
#[derive(Default)]
struct Shared {
    cancelled: Mutex<bool>,
    cv: Condvar,
}

/// The running timer: its thread's state, and the source its fires merge into.
struct Source {
    shared: Arc<Shared>,
    signal: CoalescingSignal,
}

/// A running timer.
///
/// Dropping it without [`Timer::cancel`] does not cancel it: the timer keeps firing, holding its
/// handler (a timer not cancelled is never released).
pub struct Timer {
    /// The source; `None` once dropped.
    source: Option<Source>,
    cancelled: AtomicBool,
    interval: Duration,
    leeway: Duration,
}

impl Drop for Timer {
    /// A timer not cancelled is kept for good: it keeps firing, holding its handler
    /// (`dropped_uncancelled_it_keeps_firing`).
    fn drop(&mut self) {
        if let Some(source) = self.source.take()
            && !self.cancelled.load(Ordering::Acquire)
        {
            keep_forever(source, Kept::TimerKeepsFiring);
        }
    }
}

impl std::fmt::Debug for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Timer")
            .field("interval", &self.interval)
            .field("leeway", &self.leeway)
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl Timer {
    /// A timer that runs `handler` as a block of `queue` at `start` and every `interval` after,
    /// with the number of fires coalesced into the call. The leeway is kept for the record: the
    /// timer fires on the dot.
    pub fn start(
        queue: &SerialQueue,
        start: Deadline,
        interval: Duration,
        leeway: Duration,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Timer {
        let Some(clock) = queue.timer_clock() else {
            // No production platform lacks a clock; a fake made without one refuses here.
            crate::core::raise::death::fatal(crate::core::raise::death::DeathId::Platform {
                kind: crate::core::raise::death::DeathKind::Abort,
                call: "timer: no clock",
            })
        };
        let target: Arc<dyn Executor> = Arc::new(queue.clone());
        // The signal holds the handler alone: its cancel callout releases it on the queue.
        let signal = CoalescingSignal::new_add(target, handler);
        let shared = Arc::new(Shared::default());
        let (sh, sig) = (Arc::clone(&shared), signal.clone());
        let step = u64::try_from(interval.as_nanos()).unwrap_or(u64::MAX);
        // The timer's thread runs no Objective-C: it waits and merges.
        #[allow(clippy::disallowed_methods)]
        let r = std::thread::Builder::new()
            .name("reims-vgpu.timer".into())
            .stack_size(TIMER_STACK)
            .spawn(move || {
                crate::core::rt::thread::become_ours();
                run(&*clock, &sh, &sig, start, step)
            });
        if r.is_err() {
            crate::core::raise::death::fatal(crate::core::raise::death::DeathId::Platform {
                kind: crate::core::raise::death::DeathKind::Abort,
                call: "pthread_create",
            })
        }
        Timer {
            source: Some(Source { shared, signal }),
            cancelled: AtomicBool::new(false),
            interval,
            leeway,
        }
    }

    /// Cancels the timer: no handler call starts after it, and the signal's cancel callout
    /// releases the handler on the queue, after a call already running.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(s) = &self.source {
            *s.shared.cancelled.lock() = true;
            s.signal.cancel();
            s.shared.cv.notify_all();
        }
    }

    /// Whether the timer was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// The timer thread's stack.
const TIMER_STACK: usize = 256 * 1024;

/// The number of whole intervals from `start` to `now`, and the next schedule point after
/// `now` (`start + (k + 1) * interval`).
fn fires_due(start: u64, now: u64, interval: u64) -> (u64, u64) {
    if now < start {
        return (0, start);
    }
    let k = now
        .wrapping_sub(start)
        .checked_div(interval.max(1))
        .unwrap_or(0);
    let fires = k.saturating_add(1);
    let next = start.saturating_add(fires.saturating_mul(interval.max(1)));
    (fires, next)
}

/// The timer's thread: waits for each point of the schedule and merges the fires due.
fn run(clock: &dyn TimerClock, sh: &Shared, sig: &CoalescingSignal, start: Deadline, step: u64) {
    let (start_ns, make): (u64, fn(u64) -> Deadline) = match start {
        Deadline::Mono(t) => (t.as_nanos(), |n| Deadline::Mono(MonoTime::from_nanos(n))),
        Deadline::Wall(t) => (t.as_nanos(), |n| Deadline::Wall(WallTime::from_nanos(n))),
    };
    let mut next = start_ns;
    let mut g = sh.cancelled.lock();
    loop {
        if *g {
            return;
        }
        let deadline = make(next);
        if clock.now_on(deadline) < next {
            clock.wait_until(&sh.cv, &mut g, deadline);
            continue;
        }
        let (fires, after) = fires_due(next, clock.now_on(deadline), step);
        next = after;
        drop(g);
        sig.merge(fires);
        g = sh.cancelled.lock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::os::{Clock, Threads};
    use crate::core::rt::queue::current_label;
    use crate::core::rt::thread::is_our_thread;
    use crate::core::sync::test_clock::ManualClock;
    use crate::core::testkit::threads::FakeThreads;
    use std::sync::atomic::AtomicU64;
    use std::sync::mpsc;

    const MS: Duration = Duration::from_millis(1);

    fn wait_for(v: &AtomicU64, want: u64) -> u64 {
        for _ in 0..2000 {
            if v.load(Ordering::SeqCst) >= want {
                break;
            }
            std::thread::sleep(MS);
        }
        v.load(Ordering::SeqCst)
    }

    /// A queue whose timers wait on `clock`.
    fn queue_on(clock: &Arc<ManualClock>) -> SerialQueue {
        let threads = FakeThreads::with_clock(Arc::clone(clock) as Arc<dyn TimerClock>);
        SerialQueue::new(&threads as &dyn Threads, "reims-vgpu.efi-present")
    }

    fn start_on(
        clock: &Arc<ManualClock>,
        queue: &SerialQueue,
        handler: impl Fn(u64) + Send + Sync + 'static,
    ) -> Timer {
        let start = Deadline::Wall(clock.now_wall());
        Timer::start(queue, start, 100 * MS, MS, handler)
    }

    /// The first fire is at the start; each later one a whole interval on; a late wake
    /// coalesces the fires it missed into one call
    #[test]
    fn fires_on_the_schedule_and_coalesces_late_ones() {
        let clock = Arc::new(ManualClock::default());
        let queue = queue_on(&clock);
        let fired = Arc::new(AtomicU64::new(0));
        let f2 = Arc::clone(&fired);
        let timer = start_on(&clock, &queue, move |n| {
            f2.fetch_add(n, Ordering::SeqCst);
        });
        assert_eq!(wait_for(&fired, 1), 1);
        clock.advance(350 * MS);
        assert_eq!(wait_for(&fired, 4), 4);
        timer.cancel();
    }

    /// Each call runs as a block of the queue: the queue current, owning it, marked ours
    #[test]
    fn the_handler_runs_as_a_block_of_the_queue() {
        let clock = Arc::new(ManualClock::default());
        let queue = queue_on(&clock);
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        let q2 = queue.clone();
        let timer = start_on(&clock, &queue, move |n| {
            let _ = tx
                .lock()
                .send((n, current_label(), q2.is_current(), is_our_thread()));
        });
        let got = rx.recv_timeout(Duration::from_secs(10)).ok();
        assert_eq!(got, Some((1, Some("reims-vgpu.efi-present"), true, true)));
        timer.cancel();
    }

    /// Cancel stops the fires, and the cancel callout releases the handler (and what it holds)
    /// on the queue
    #[test]
    fn cancel_stops_it_and_frees_the_handler_on_the_queue() {
        let clock = Arc::new(ManualClock::default());
        let queue = queue_on(&clock);
        let held = Arc::new(());
        let h2 = Arc::clone(&held);
        let fired = Arc::new(AtomicU64::new(0));
        let f2 = Arc::clone(&fired);
        let timer = start_on(&clock, &queue, move |n| {
            let _ = &h2;
            f2.fetch_add(n, Ordering::SeqCst);
        });
        assert_eq!(wait_for(&fired, 1), 1);
        timer.cancel();
        assert!(timer.is_cancelled());
        queue.sync(|| ());
        assert_eq!(Arc::strong_count(&held), 1);
        clock.advance(500 * MS);
        std::thread::sleep(20 * MS);
        queue.sync(|| ());
        assert_eq!(fired.load(Ordering::SeqCst), 1);
        drop(timer);
    }

    /// A timer dropped without a cancel keeps firing, holding its handler
    #[test]
    fn dropped_uncancelled_it_keeps_firing() {
        let clock = Arc::new(ManualClock::default());
        let queue = queue_on(&clock);
        let fired = Arc::new(AtomicU64::new(0));
        let f2 = Arc::clone(&fired);
        let timer = start_on(&clock, &queue, move |n| {
            f2.fetch_add(n, Ordering::SeqCst);
        });
        assert_eq!(wait_for(&fired, 1), 1);
        drop(timer);
        clock.advance(200 * MS);
        assert_eq!(wait_for(&fired, 3), 3);
    }

    /// The schedule's arithmetic: fires due, and the next point
    #[test]
    fn fires_due_counts_whole_intervals() {
        assert_eq!(fires_due(100, 50, 10), (0, 100));
        assert_eq!(fires_due(100, 100, 10), (1, 110));
        assert_eq!(fires_due(100, 135, 10), (4, 140));
        assert_eq!(fires_due(0, 5, 0), (6, 6));
    }
}
