//! `Clock` over the host's clocks, as libc++'s `std::chrono` and libdispatch read them.
//!
//! The monotonic clock is libc++'s `steady_clock` on Apple platforms:
//! `clock_gettime(CLOCK_MONOTONIC_RAW)` (`steady_clock::now()` in the system's
//! `libc++.1.dylib` passes clock id 4; its error string reads "clock_gettime(CLOCK_MONOTONIC_RAW)
//! failed"), which counts the time the host sleeps, unlike `CLOCK_UPTIME_RAW`
//! (`mach_absolute_time`, `std::time::Instant`). The wall clock is `system_clock`:
//! `clock_gettime(CLOCK_REALTIME)` in nanoseconds since the Unix epoch.
//! A condvar's timed wait blocks as libc++'s does, for its timeout in awake time (libpthread
//! hands the kernel a relative timeout, which `parking_lot`'s `wait_for` does too), and answers
//! as libc++'s does: after the wait, `now < deadline ? no_timeout : timeout`, read on the
//! deadline's own clock, where a host sleep counts.

use std::time::Duration;

use crate::core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use crate::core::sync::{Condvar, MutexGuard};

/// `clockid_t` of `CLOCK_MONOTONIC_RAW` (`<time.h>`): libc++'s `steady_clock` on Apple.
const CLOCK_MONOTONIC_RAW: libc::clockid_t = 4;
/// `clockid_t` of `CLOCK_REALTIME` (`<time.h>`).
const CLOCK_REALTIME: libc::clockid_t = 0;

unsafe extern "C" {
    /// `uint64_t clock_gettime_nsec_np(clockid_t)` (`<time.h>`, macOS 10.12+).
    fn clock_gettime_nsec_np(clock_id: libc::clockid_t) -> u64;
}

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct MacClock;

impl MacClock {
    /// The process's Clock.
    pub fn new() -> MacClock {
        MacClock
    }

    /// How long until `deadline`, on its own clock; zero once reached.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for MacClock {
    /// `std::chrono::steady_clock::now()` (libc++: `CLOCK_MONOTONIC_RAW`).
    fn now_mono(&self) -> MonoTime {
        // SAFETY: a libSystem call with a valid clock id; it cannot fail for this clock.
        MonoTime::from_nanos(unsafe { clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) })
    }

    /// `dispatch_walltime(NULL, 0)`.
    fn now_wall(&self) -> WallTime {
        // SAFETY: as above.
        WallTime::from_nanos(unsafe { clock_gettime_nsec_np(CLOCK_REALTIME) })
    }

    /// `std::condition_variable::wait_for` and `wait_until` on `system_clock`. One wait, as
    /// libc++'s is: a notify or a spurious wakeup ends it; the answer is whether the deadline has
    /// passed afterwards. A deadline already reached answers timeout without releasing the lock:
    /// libc++'s `wait_until` returns `timeout` when `__t <= _Clock::now()` and `wait_for` when
    /// `__d <= 0`, before any `pthread_cond_timedwait`
    /// (`<__condition_variable/condition_variable.h>:120-123` Xcode 26.6 SDK).
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

    /// `std::condition_variable::wait` (one wait; the caller loops).
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

    use super::*;
    use crate::core::sync::Mutex;
    use std::sync::Arc;

    const MS: Duration = Duration::from_millis(1);

    mod mono_now {
        use super::*;

        /// steady_clock::now is CLOCK_MONOTONIC_RAW, read as libc++ reads it
        #[test]
        fn mono_is_libcxx_steady_clock() {
            let c = MacClock::new();
            // SAFETY: a libSystem call with a valid clock id.
            let before = unsafe { clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) };
            let ours = c.now_mono().as_nanos();
            // SAFETY: as above.
            let after = unsafe { clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW) };
            assert!(before <= ours && ours <= after);
        }

        /// steady_clock::now moves forward at the rate Instant counts while awake
        #[test]
        fn mono_moves_forward() {
            let c = MacClock::new();
            #[allow(clippy::disallowed_methods)] // the test compares against std's clock
            let i0 = std::time::Instant::now();
            let a = c.now_mono();
            std::thread::sleep(5 * MS);
            let b = c.now_mono();
            #[allow(clippy::disallowed_methods)]
            let i1 = std::time::Instant::now();
            let ours = b.saturating_duration_since(a);
            assert!(ours >= 5 * MS);
            assert!(ours <= i1.duration_since(i0));
        }
    }

    mod wall_now {
        use super::*;

        /// system_clock::now is nanoseconds since the Unix epoch
        #[test]
        fn wall_is_unix_epoch_nanos() {
            let c = MacClock::new();
            #[allow(clippy::disallowed_methods)] // the test compares against std's clock
            let sys = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let ours = Duration::from_nanos(c.now_wall().as_nanos());
            let diff = ours.abs_diff(sys);
            assert!(diff < Duration::from_secs(1), "{diff:?}");
        }
    }

    mod wait_until {
        use super::*;

        /// wait_for(2 s) with nobody notifying times out at the deadline (mono)
        #[test]
        fn mono_deadline_times_out() {
            let c = MacClock::new();
            let m = Mutex::new(());
            let cv = Condvar::new();
            let mut g = m.lock();
            let deadline = Deadline::Mono(c.now_mono().saturating_add(20 * MS));
            let start = c.now_mono();
            let mut r = c.wait_until(&cv, &mut g, deadline);
            // A spurious wakeup answers no_timeout; the deadline decides.
            while !r.timed_out() {
                r = c.wait_until(&cv, &mut g, deadline);
            }
            assert!(c.now_mono().saturating_duration_since(start) >= 20 * MS);
            assert!(deadline.reached(&c));
        }

        /// wait_until(system_clock) times out at a wall deadline
        #[test]
        fn wall_deadline_times_out() {
            let c = MacClock::new();
            let m = Mutex::new(());
            let cv = Condvar::new();
            let mut g = m.lock();
            let deadline = Deadline::Wall(c.now_wall().saturating_add(20 * MS));
            while !c.wait_until(&cv, &mut g, deadline).timed_out() {}
            assert!(deadline.reached(&c));
        }

        /// A past deadline answers timeout (mono and wall)
        #[test]
        fn past_deadline_is_timeout() {
            let c = MacClock::new();
            let m = Mutex::new(());
            let cv = Condvar::new();
            let mut g = m.lock();
            let past = Deadline::Mono(MonoTime::from_nanos(0));
            assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
            let past = Deadline::Wall(WallTime::from_nanos(0));
            assert_eq!(c.wait_until(&cv, &mut g, past), TimedOut(true));
        }

        /// Libc++ wait_until/wait_for return timeout without touching the lock when the deadline
        /// has already passed
        #[test]
        fn past_deadline_keeps_the_lock_held() {
            use std::sync::atomic::{AtomicBool, Ordering};
            for past in [
                Deadline::Mono(MonoTime::from_nanos(0)),
                Deadline::Wall(WallTime::from_nanos(0)),
            ] {
                let c = MacClock::new();
                let state = Arc::new((Mutex::new(false), Condvar::new()));
                let mut g = state.0.lock();
                let started = Arc::new(AtomicBool::new(false));
                let (s2, st2) = (state.clone(), started.clone());
                #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
                let h = std::thread::spawn(move || {
                    st2.store(true, Ordering::SeqCst);
                    *s2.0.lock() = true;
                });
                while !started.load(Ordering::SeqCst) {}
                std::thread::sleep(20 * MS);
                let mut entered = false;
                for _ in 0..2000 {
                    assert_eq!(c.wait_until(&state.1, &mut g, past), TimedOut(true));
                    if *g {
                        entered = true;
                        break;
                    }
                }
                drop(g);
                h.join().unwrap();
                assert!(
                    !entered,
                    "another thread took the mutex inside a past-deadline wait_until"
                );
            }
        }

        /// A notify before the deadline answers no_timeout
        #[test]
        fn notify_is_no_timeout() {
            let c = Arc::new(MacClock::new());
            let state = Arc::new((Mutex::new(false), Condvar::new()));
            let s2 = state.clone();
            let mut g = state.0.lock();
            #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
            let h = std::thread::spawn(move || {
                *s2.0.lock() = true;
                s2.1.notify_all();
            });
            let deadline = Deadline::Mono(c.now_mono().saturating_add(Duration::from_secs(10)));
            while !*g {
                assert_eq!(c.wait_until(&state.1, &mut g, deadline), TimedOut(false));
            }
            drop(g);
            h.join().unwrap();
        }
    }

    mod wait {
        use super::*;

        /// An untimed wait returns on notify
        #[test]
        fn wait_returns_on_notify() {
            let c = MacClock::new();
            let state = Arc::new((Mutex::new(false), Condvar::new()));
            let s2 = state.clone();
            let mut g = state.0.lock();
            #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
            let h = std::thread::spawn(move || {
                *s2.0.lock() = true;
                s2.1.notify_all();
            });
            while !*g {
                c.wait(&state.1, &mut g);
            }
            drop(g);
            h.join().unwrap();
        }
    }

    mod sleep_until {
        use super::*;

        /// sleep_until returns once its own clock reaches the deadline
        #[test]
        fn sleeps_to_the_deadline_on_each_clock() {
            let c = MacClock::new();
            let d = Deadline::Mono(c.now_mono().saturating_add(10 * MS));
            c.sleep_until(d);
            assert!(d.reached(&c));
            let d = Deadline::Wall(c.now_wall().saturating_add(10 * MS));
            c.sleep_until(d);
            assert!(d.reached(&c));
            // A reached deadline returns at once.
            c.sleep_until(Deadline::Mono(MonoTime::from_nanos(0)));
        }
    }
}
