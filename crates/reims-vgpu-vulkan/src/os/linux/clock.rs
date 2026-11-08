//! `Clock` over Linux's clocks, as macOS's answers them.
//!
//! The monotonic clock is `CLOCK_BOOTTIME`: like `CLOCK_MONOTONIC_RAW` on macOS (libc++'s
//! `steady_clock` there), it counts the time the host sleeps, which Linux's `CLOCK_MONOTONIC`
//! does not. The wall clock is `CLOCK_REALTIME`, in nanoseconds since the Unix epoch. A condvar's
//! timed wait blocks for its timeout in awake time (`parking_lot`'s `wait_for` hands the kernel a
//! relative timeout), and answers as libc++'s does on macOS: after the wait, `now < deadline ?
//! no_timeout : timeout`, read on the deadline's own clock, where a host sleep counts.

use std::time::Duration;

use crate::core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use crate::core::sync::{Condvar, MutexGuard};

/// `clock_gettime(clock)` in nanoseconds.
fn nanos(clock: libc::clockid_t) -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock id and a live out-parameter; these two clocks cannot fail.
    let _ = unsafe { libc::clock_gettime(clock, &mut ts) };
    let secs = u64::try_from(ts.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(ts.tv_nsec).unwrap_or(0);
    secs.saturating_mul(1_000_000_000).saturating_add(nanos)
}

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct LinuxClock;

impl LinuxClock {
    /// The process's Clock.
    pub fn new() -> LinuxClock {
        LinuxClock
    }

    /// How long until `deadline`, on its own clock; zero once reached.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for LinuxClock {
    /// `CLOCK_BOOTTIME`, which counts a host sleep as macOS's monotonic clock does.
    fn now_mono(&self) -> MonoTime {
        MonoTime::from_nanos(nanos(libc::CLOCK_BOOTTIME))
    }

    /// `CLOCK_REALTIME`.
    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(nanos(libc::CLOCK_REALTIME))
    }

    /// One wait, as libc++'s is on macOS: a notify or a spurious wakeup ends it; the answer is
    /// whether the deadline has passed afterwards. A deadline already reached answers timeout
    /// without releasing the lock.
    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let d = self.remaining(deadline);
        if d.is_zero() {
            return TimedOut(true);
        }
        let _ = cv.wait_for(g, d);
        TimedOut(deadline.reached(self))
    }

    /// One untimed wait; the caller loops.
    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
        cv.wait(g);
    }

    /// Sleeps until `deadline` is reached on its own clock.
    fn sleep_until(&self, deadline: Deadline) {
        loop {
            let d = self.remaining(deadline);
            if d.is_zero() {
                return;
            }
            std::thread::sleep(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::core::sync::Mutex;

    use super::*;

    const MS: Duration = Duration::from_millis(1);

    /// The monotonic clock is CLOCK_BOOTTIME, and moves forward at the rate Instant counts while
    /// awake
    #[test]
    fn mono_is_boottime_and_moves_forward() {
        let c = LinuxClock::new();
        let before = nanos(libc::CLOCK_BOOTTIME);
        let ours = c.now_mono().as_nanos();
        let after = nanos(libc::CLOCK_BOOTTIME);
        assert!(before <= ours && ours <= after);
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let i0 = std::time::Instant::now();
        let a = c.now_mono();
        std::thread::sleep(5 * MS);
        let b = c.now_mono();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let i1 = std::time::Instant::now();
        let elapsed = b.saturating_duration_since(a);
        assert!(elapsed >= 5 * MS);
        assert!(elapsed <= i1.duration_since(i0) + MS);
    }

    /// The wall clock is nanoseconds since the Unix epoch
    #[test]
    fn wall_is_unix_epoch_nanos() {
        let c = LinuxClock::new();
        #[expect(
            clippy::disallowed_methods,
            reason = "the test compares with std's clock"
        )]
        let sys = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let ours = Duration::from_nanos(c.now_wall().as_nanos());
        assert!(ours.abs_diff(sys) < Duration::from_secs(1));
    }

    /// A wait with nobody notifying times out at its deadline, on either clock; a past deadline
    /// answers timeout at once
    #[test]
    fn deadlines_time_out() {
        let c = LinuxClock::new();
        let m = Mutex::new(());
        let cv = Condvar::new();
        let mut g = m.lock();
        for deadline in [
            Deadline::Mono(c.now_mono().saturating_add(20 * MS)),
            Deadline::Wall(c.now_wall().saturating_add(20 * MS)),
        ] {
            while !c.wait_until(&cv, &mut g, deadline).timed_out() {}
            assert!(deadline.reached(&c));
        }
        let past = Deadline::Mono(MonoTime::from_nanos(0));
        assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
        let past = Deadline::Wall(WallTime::from_nanos(0));
        assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
    }

    /// A notify before the deadline answers no_timeout, and an untimed wait returns on it
    #[test]
    fn a_notify_ends_a_wait() {
        let c = Arc::new(LinuxClock::new());
        for timed in [true, false] {
            let state = Arc::new((Mutex::new(false), Condvar::new()));
            let s2 = Arc::clone(&state);
            let mut g = state.0.lock();
            let h = std::thread::Builder::new()
                .spawn(move || {
                    *s2.0.lock() = true;
                    s2.1.notify_all();
                })
                .unwrap();
            let deadline = Deadline::Mono(c.now_mono().saturating_add(Duration::from_secs(10)));
            while !*g {
                if timed {
                    assert_eq!(c.wait_until(&state.1, &mut g, deadline), TimedOut(false));
                } else {
                    c.wait(&state.1, &mut g);
                }
            }
            drop(g);
            h.join().unwrap();
        }
    }

    /// sleep_until returns once its own clock reaches the deadline
    #[test]
    fn sleeps_to_the_deadline_on_each_clock() {
        let c = LinuxClock::new();
        let d = Deadline::Mono(c.now_mono().saturating_add(10 * MS));
        c.sleep_until(d);
        assert!(d.reached(&c));
        let d = Deadline::Wall(c.now_wall().saturating_add(10 * MS));
        c.sleep_until(d);
        assert!(d.reached(&c));
        c.sleep_until(Deadline::Mono(MonoTime::from_nanos(0)));
    }
}
