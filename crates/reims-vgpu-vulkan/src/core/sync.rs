//! Locks: re-exports of `parking_lot::{Mutex, MutexGuard, Condvar, ReentrantMutex}` and the
//! writer-preferring [`WriterPrefRwLock`] (`parking_lot::RwLock`).
//!
//! Every timed wait on a `Condvar` goes through the platform's `Clock` (`Clock::wait_until`) so
//! tests run on virtual time (7.1).

pub mod rwlock;

pub use rwlock::{ReadGuard, WriteGuard, WriterPrefRwLock};

pub use parking_lot::{
    Condvar, Mutex, MutexGuard, ReentrantMutex, ReentrantMutexGuard, WaitTimeoutResult,
};

#[cfg(test)]
pub(crate) mod test_clock {
    //! A manual virtual clock for the unit tests of `sync` (the fake platform's is the testkit's
    //! `VirtualClock`).

    use super::{Condvar, Mutex, MutexGuard};
    use crate::core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
    use std::time::Duration;

    /// Virtual time. Timed waits poll in 1 ms real slices, so an `advance` is seen promptly.
    #[derive(Debug, Default)]
    pub(crate) struct ManualClock {
        now: Mutex<u64>,
        tick: Condvar,
    }

    /// Wall time starts here so wall and mono differ.
    pub(crate) const WALL_EPOCH: u64 = 1_000_000_000_000;

    impl ManualClock {
        pub(crate) fn advance(&self, d: Duration) {
            {
                let mut n = self.now.lock();
                *n = n.saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
            }
            self.tick.notify_all();
        }
        fn now(&self) -> u64 {
            *self.now.lock()
        }
        fn reached(&self, d: Deadline) -> bool {
            match d {
                Deadline::Mono(t) => self.now() >= t.as_nanos(),
                Deadline::Wall(t) => self.now().saturating_add(WALL_EPOCH) >= t.as_nanos(),
            }
        }
    }

    impl Clock for ManualClock {
        fn now_mono(&self) -> MonoTime {
            MonoTime::from_nanos(self.now())
        }
        fn now_wall(&self) -> WallTime {
            WallTime::from_nanos(self.now().saturating_add(WALL_EPOCH))
        }
        fn wait_until<T>(
            &self,
            cv: &Condvar,
            g: &mut MutexGuard<'_, T>,
            deadline: Deadline,
        ) -> TimedOut {
            if self.reached(deadline) {
                return TimedOut(true);
            }
            // Poll in short real slices so an `advance` is noticed without a registry of the
            // caller's condvar; a notify on `cv` wakes it at once.
            let _ = cv.wait_for(g, Duration::from_millis(1));
            TimedOut(self.reached(deadline))
        }
        fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
            cv.wait(g)
        }
        fn sleep_until(&self, deadline: Deadline) {
            let mut n = self.now.lock();
            while !match deadline {
                Deadline::Mono(t) => *n >= t.as_nanos(),
                Deadline::Wall(t) => n.saturating_add(WALL_EPOCH) >= t.as_nanos(),
            } {
                self.tick.wait(&mut n);
            }
        }
    }

    /// A clock whose host can sleep: both clocks count the sleep; a condvar's wait counts
    /// awake time only (the kernel's relative timeout). See `Clock::wait_until`.
    /// Blocking waits poll in 1 ms real slices; a notify ends them at once.
    #[derive(Debug, Default)]
    pub(crate) struct SleepClock {
        /// (awake, slept) nanoseconds.
        t: Mutex<(u64, u64)>,
    }

    impl SleepClock {
        fn awake(&self) -> u64 {
            self.t.lock().0
        }
        fn now(&self) -> u64 {
            let t = self.t.lock();
            t.0.saturating_add(t.1)
        }
        /// The host sleeps for `d`: both clocks move, awake time does not.
        pub(crate) fn host_sleep(&self, d: Duration) {
            let mut t = self.t.lock();
            t.1 =
                t.1.saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        }
        fn remaining(&self, deadline: Deadline) -> u64 {
            match deadline {
                Deadline::Mono(t) => t.as_nanos().saturating_sub(self.now()),
                Deadline::Wall(t) => t
                    .as_nanos()
                    .saturating_sub(self.now().saturating_add(WALL_EPOCH)),
            }
        }
    }

    impl Clock for SleepClock {
        fn now_mono(&self) -> MonoTime {
            MonoTime::from_nanos(self.now())
        }
        fn now_wall(&self) -> WallTime {
            WallTime::from_nanos(self.now().saturating_add(WALL_EPOCH))
        }
        fn wait_until<T>(
            &self,
            cv: &Condvar,
            g: &mut MutexGuard<'_, T>,
            deadline: Deadline,
        ) -> TimedOut {
            let rem = self.remaining(deadline);
            if rem == 0 {
                return TimedOut(true);
            }
            let until = self.awake().saturating_add(rem);
            while self.awake() < until && cv.wait_for(g, Duration::from_millis(1)).timed_out() {}
            TimedOut(deadline.reached(self))
        }
        fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
            cv.wait(g);
        }
        fn sleep_until(&self, deadline: Deadline) {
            while !deadline.reached(self) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_clock::ManualClock;
    use super::*;
    use crate::core::platform::os::{Clock, Deadline, MonoTime};
    use std::sync::Arc;
    use std::time::Duration;

    /// A timed wait times out only when virtual time reaches the deadline
    #[test]
    fn wait_until_follows_virtual_time() {
        let clock = ManualClock::default();
        let m = Mutex::new(());
        let cv = Condvar::new();
        let mut g = m.lock();
        let d = Deadline::Mono(MonoTime::from_nanos(1_000));
        assert!(!clock.wait_until(&cv, &mut g, d).timed_out());
        clock.advance(Duration::from_nanos(1_000));
        assert!(clock.wait_until(&cv, &mut g, d).timed_out());
    }

    /// `_signalCond.wait_for(lock, std::chrono::seconds(2))`: libc++: `pthread_cond_timedwait` on a
    /// system_clock deadline, the answer read on steady_clock
    mod timed_wait {
        use super::super::test_clock::SleepClock;
        use super::*;
        use std::sync::atomic::{AtomicBool, Ordering};

        /// A host sleep past the deadline does not end the wait (the kernel counts its timeout in
        /// awake time), but whatever ends it next answers timeout, read on the deadline's clock,
        /// where the sleep counts
        #[test]
        fn a_wake_past_the_deadline_answers_timeout_at_the_next_notify() {
            let clock = Arc::new(SleepClock::default());
            let state = Arc::new((Mutex::new(false), Condvar::new()));
            let (c2, s2) = (clock.clone(), state.clone());
            let deadline = Deadline::Mono(clock.now_mono().saturating_add(Duration::from_secs(2)));
            let done = Arc::new(AtomicBool::new(false));
            let d2 = done.clone();
            #[allow(clippy::disallowed_methods)] // test helper thread: no Objective-C runs here
            let h = std::thread::spawn(move || {
                let mut g = s2.0.lock();
                *g = true;
                let r = c2.wait_until(&s2.1, &mut g, deadline);
                d2.store(true, Ordering::SeqCst);
                r
            });
            // The waiter holds the mutex until it is inside the wait.
            while !*state.0.lock() {
                std::thread::sleep(Duration::from_millis(1));
            }
            clock.host_sleep(Duration::from_secs(3));
            std::thread::sleep(Duration::from_millis(50));
            assert!(
                !done.load(Ordering::SeqCst),
                "the wake ended a condvar wait"
            );
            // The test clock polls in slices, so a notify can land between two of them: notify
            // until the waiter has returned.
            while !done.load(Ordering::SeqCst) {
                {
                    let _g = state.0.lock();
                    state.1.notify_all();
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            let r = h.join();
            assert!(matches!(r, Ok(t) if t.timed_out()));
        }
    }

    /// Notify wakes an untimed wait
    #[test]
    fn notify_wakes_wait() {
        let clock = Arc::new(ManualClock::default());
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let s2 = state.clone();
        let c2 = clock.clone();
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut g = s2.0.lock();
                while !*g {
                    c2.wait(&s2.1, &mut g);
                }
            });
            std::thread::sleep(Duration::from_millis(20));
            *state.0.lock() = true;
            state.1.notify_all();
        });
        assert!(!state.1.notify_one());
    }
}
