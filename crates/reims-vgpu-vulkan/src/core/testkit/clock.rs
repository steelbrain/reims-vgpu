//! Fake: `VirtualClock`: time that moves only when a test advances it (architecture 13.2).
//!
//! Every timed wait in core goes through `Clock` (7.1), so the device's 2 s wait report, the 5000
//! ms event waits and the display's system-clock wait run on virtual time: a test advances the
//! clock past the deadline instead of sleeping. A timed wait returns only when its condvar is
//! notified or virtual time reaches its deadline, never early on its own, so single waits stay
//! single (a drain's one wait on its thread's exit returns on that exit's notify).
//!
//! A timed wait parks untimed on the caller's condvar, as `pthread_cond_timedwait` does, so a
//! notify is never lost (`unsafe` allowed here, architecture 17.3). It first registers the
//! condvar and the caller's mutex, and `advance` delivers the timeout: it takes the caller's
//! mutex (so the waiter is parked, not between its check and its park), marks the wait expired
//! and notifies the condvar. An earlier design polled virtual time in 1 ms real slices; a notify
//! that landed while a slice's timeout was re-taking the mutex found no parked thread and was
//! lost, and a virtual deadline never elapses by itself, so the wait hung.
//!
//! The one liberty: the timeout notify wakes every thread parked on that condvar, so another
//! waiter on the same condvar whose deadline has not passed (or an untimed one) returns as a
//! spurious wakeup, which `std::condition_variable` allows. It never loses a notify.

#![allow(unsafe_code)] // architecture 17.3: fake/clock.rs (a timed wait's registered condvar and mutex)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use parking_lot::RawMutex;
use parking_lot::lock_api::RawMutex as _;

use crate::core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use crate::core::sync::{Condvar, Mutex, MutexGuard};
use crate::core::testkit::capture::VirtualTime;

/// Wall time at virtual zero (nanoseconds since the Unix epoch): 2026-01-01T00:00:00Z. Mono
/// and wall differ, so a test can tell which clock a wait used.
pub const WALL_EPOCH_NANOS: u64 = 1_767_225_600_000_000_000;

/// The real time `wait_for_waiters` sleeps between looks.
pub const POLL_SLICE: Duration = Duration::from_millis(1);

/// How many times `advance` retries a waiter's mutex before it notifies without it. A registered
/// waiter holds its mutex only for the few instructions between registering and parking, so the
/// retries end at once unless another thread holds that mutex for long (a locked hang); then the
/// waiter is parked and a notify without the mutex reaches it.
const DELIVER_TRIES: u32 = 2_000;

/// The real pause between two delivery tries.
const DELIVER_PAUSE: Duration = Duration::from_micros(50);

/// One thread parked in `wait_until`.
#[derive(Debug)]
struct TimedWait {
    id: u64,
    /// The caller's condvar and mutex. Valid while the entry is registered: the waiter removes
    /// it, under the registry lock, before `wait_until` returns.
    cv: *const Condvar,
    mutex: *const RawMutex,
    deadline: Deadline,
    /// `advance` delivered the timeout.
    expired: bool,
}

#[derive(Debug, Default)]
struct TimedWaits {
    next_id: u64,
    entries: Vec<TimedWait>,
}

// SAFETY: the pointers are only dereferenced under the registry's lock while the entry is
// registered, and the waiter that owns the pointees removes its entry before they can end.
unsafe impl Send for TimedWaits {}

/// A virtual monotonic and wall clock.
#[derive(Debug, Default)]
pub struct VirtualClock {
    /// Nanoseconds since virtual zero.
    now: Mutex<u64>,
    /// Wakes `sleep_until`.
    tick: Condvar,
    /// Threads inside `wait`, `wait_until` or `sleep_until`.
    waiters: AtomicUsize,
    /// Parked timed waits. Lock order: `timed`, then `now`; a caller's mutex is only ever
    /// try-locked under it.
    timed: Mutex<TimedWaits>,
}

struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn enter(n: &'a AtomicUsize) -> Waiting<'a> {
        n.fetch_add(1, Ordering::SeqCst);
        Waiting(n)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl VirtualClock {
    /// A clock at virtual zero.
    pub fn new() -> VirtualClock {
        VirtualClock::default()
    }

    /// Nanoseconds since virtual zero.
    pub fn elapsed_nanos(&self) -> u64 {
        *self.now.lock()
    }

    /// Moves time forward by `d` and wakes every wait whose deadline passed.
    pub fn advance(&self, d: Duration) {
        {
            let mut n = self.now.lock();
            *n = n.saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        }
        self.tick.notify_all();
        self.deliver_timeouts();
    }

    /// Moves time to `mono` if it is later.
    pub fn advance_to(&self, mono: MonoTime) {
        {
            let mut n = self.now.lock();
            if mono.as_nanos() > *n {
                *n = mono.as_nanos();
            }
        }
        self.tick.notify_all();
        self.deliver_timeouts();
    }

    /// Ends every registered timed wait whose deadline has passed: under the waiter's mutex
    /// (so it is parked), marks it expired and notifies its condvar.
    fn deliver_timeouts(&self) {
        let mut tries = 0u32;
        loop {
            let mut pending = false;
            {
                let mut reg = self.timed.lock();
                let now = self.elapsed_nanos();
                for e in reg
                    .entries
                    .iter_mut()
                    .filter(|e| !e.expired && Self::reached_at(now, e.deadline))
                {
                    // SAFETY: the entry is registered and the registry is locked (TimedWait).
                    let (cv, mutex) = unsafe { (&*e.cv, &*e.mutex) };
                    if mutex.try_lock() {
                        e.expired = true;
                        cv.notify_all();
                        // SAFETY: this thread took the lock just above.
                        unsafe { mutex.unlock() };
                    } else if tries >= DELIVER_TRIES {
                        // Another thread holds the waiter's mutex for long, so the waiter is
                        // parked (or already woken): a notify without the mutex reaches it.
                        e.expired = true;
                        cv.notify_all();
                    } else {
                        pending = true;
                    }
                }
            }
            if !pending {
                return;
            }
            tries = tries.saturating_add(1);
            std::thread::sleep(DELIVER_PAUSE);
        }
    }

    /// How many threads are blocked in a clock wait now.
    pub fn waiters(&self) -> usize {
        self.waiters.load(Ordering::SeqCst)
    }

    /// Blocks (in real time, up to `real_timeout`) until at least `n` threads are blocked in a
    /// clock wait: a test's way to know a thread reached core's wait before advancing time.
    pub fn wait_for_waiters(&self, n: usize, real_timeout: Duration) -> bool {
        let mut left = real_timeout;
        loop {
            if self.waiters() >= n {
                return true;
            }
            if left.is_zero() {
                return false;
            }
            std::thread::sleep(POLL_SLICE);
            left = left.saturating_sub(POLL_SLICE);
        }
    }

    fn reached_at(now: u64, d: Deadline) -> bool {
        match d {
            Deadline::Mono(t) => now >= t.as_nanos(),
            Deadline::Wall(t) => now.saturating_add(WALL_EPOCH_NANOS) >= t.as_nanos(),
        }
    }

    fn reached(&self, d: Deadline) -> bool {
        Self::reached_at(self.elapsed_nanos(), d)
    }
}

impl Clock for VirtualClock {
    fn now_mono(&self) -> MonoTime {
        MonoTime::from_nanos(self.elapsed_nanos())
    }

    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(self.elapsed_nanos().saturating_add(WALL_EPOCH_NANOS))
    }

    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let _w = Waiting::enter(&self.waiters);
        let id = {
            let mut reg = self.timed.lock();
            if self.reached(deadline) {
                return TimedOut(true);
            }
            let id = reg.next_id;
            reg.next_id = reg.next_id.wrapping_add(1);
            // SAFETY: only the address is kept; `deliver_timeouts` try-locks it while the entry
            // is registered, and this thread holds it until `cv.wait` parks.
            let mutex: *const RawMutex = unsafe { MutexGuard::mutex(g).raw() };
            reg.entries.push(TimedWait {
                id,
                cv,
                mutex,
                deadline,
                expired: false,
            });
            id
        };
        // Registered while holding the mutex, so `advance` cannot deliver before this parks.
        cv.wait(g);
        let expired = {
            let mut reg = self.timed.lock();
            let pos = reg.entries.iter().position(|e| e.id == id);
            pos.is_some_and(|i| reg.entries.remove(i).expired)
        };
        // libc++'s `wait_until` reads the clock after the wake: a notify that comes after the
        // deadline still reports `timeout`.
        TimedOut(expired || self.reached(deadline))
    }

    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
        let _w = Waiting::enter(&self.waiters);
        cv.wait(g);
    }

    fn sleep_until(&self, deadline: Deadline) {
        let _w = Waiting::enter(&self.waiters);
        let mut n = self.now.lock();
        while !Self::reached_at(*n, deadline) {
            self.tick.wait(&mut n);
        }
    }
}

impl VirtualTime for VirtualClock {
    fn advance(&self, d: Duration) {
        VirtualClock::advance(self, d);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const REAL: Duration = Duration::from_secs(10);

    /// Mono and wall move together from their epochs
    #[test]
    fn now_moves_on_advance() {
        let c = VirtualClock::new();
        assert_eq!(c.now_mono(), MonoTime::from_nanos(0));
        assert_eq!(c.now_wall(), WallTime::from_nanos(WALL_EPOCH_NANOS));
        c.advance(Duration::from_secs(2));
        assert_eq!(c.now_mono().as_nanos(), 2_000_000_000);
        assert_eq!(c.now_wall().as_nanos(), WALL_EPOCH_NANOS + 2_000_000_000);
        c.advance_to(MonoTime::from_nanos(1));
        assert_eq!(c.elapsed_nanos(), 2_000_000_000);
        c.advance_to(MonoTime::from_nanos(3_000_000_000));
        assert_eq!(c.elapsed_nanos(), 3_000_000_000);
    }

    /// A 2 s wait times out only once virtual time passes 2 s
    #[test]
    fn two_second_timeout_on_virtual_time() {
        let c = Arc::new(VirtualClock::new());
        let state = Arc::new((Mutex::new(()), Condvar::new()));
        let (c2, s2) = (c.clone(), state.clone());
        let h = std::thread::Builder::new()
            .spawn(move || {
                let mut g = s2.0.lock();
                let d = Deadline::Mono(c2.now_mono().saturating_add(Duration::from_secs(2)));
                c2.wait_until(&s2.1, &mut g, d)
            })
            .unwrap();
        assert!(c.wait_for_waiters(1, REAL));
        c.advance(Duration::from_millis(1999));
        std::thread::sleep(Duration::from_millis(20));
        assert!(!h.is_finished());
        c.advance(Duration::from_millis(1));
        assert_eq!(h.join().unwrap(), TimedOut(true));
        assert_eq!(c.waiters(), 0);
    }

    /// A notify ends a timed wait without a timeout
    #[test]
    fn notify_ends_timed_wait() {
        let c = Arc::new(VirtualClock::new());
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let (c2, s2) = (c.clone(), state.clone());
        let h = std::thread::Builder::new()
            .spawn(move || {
                let mut g = s2.0.lock();
                let d = Deadline::Wall(c2.now_wall().saturating_add(Duration::from_secs(1)));
                let r = c2.wait_until(&s2.1, &mut g, d);
                (r, *g)
            })
            .unwrap();
        assert!(c.wait_for_waiters(1, REAL));
        *state.0.lock() = true;
        state.1.notify_all();
        assert_eq!(h.join().unwrap(), (TimedOut(false), true));
    }

    /// A notify made under the mutex while the waiter is between looks is never lost, and one made
    /// after the unlock neither (1 ms slices dropped such notifies, and the wait hung)
    #[test]
    fn notify_is_never_lost() {
        for under_lock in [true, false] {
            for _ in 0..200 {
                let c = Arc::new(VirtualClock::new());
                let state = Arc::new((Mutex::new(false), Condvar::new()));
                let (c2, s2) = (c.clone(), state.clone());
                let h = std::thread::Builder::new()
                    .spawn(move || {
                        let mut g = s2.0.lock();
                        let d = Deadline::Mono(MonoTime::from_nanos(1_000));
                        let r = c2.wait_until(&s2.1, &mut g, d);
                        (r, *g)
                    })
                    .unwrap();
                assert!(c.wait_for_waiters(1, REAL));
                {
                    let mut g = state.0.lock();
                    *g = true;
                    if under_lock {
                        state.1.notify_all();
                    }
                }
                if !under_lock {
                    state.1.notify_all();
                }
                assert_eq!(h.join().unwrap(), (TimedOut(false), true));
            }
        }
    }

    /// An advance delivers the timeout even while another thread holds the waiter's mutex; the
    /// waiter then returns once it gets the mutex back
    #[test]
    fn advance_delivers_while_mutex_is_held() {
        let c = Arc::new(VirtualClock::new());
        let state = Arc::new((Mutex::new(()), Condvar::new()));
        let (c2, s2) = (c.clone(), state.clone());
        let h = std::thread::Builder::new()
            .spawn(move || {
                let mut g = s2.0.lock();
                c2.wait_until(&s2.1, &mut g, Deadline::Mono(MonoTime::from_nanos(10)))
            })
            .unwrap();
        assert!(c.wait_for_waiters(1, REAL));
        let held = state.0.lock();
        c.advance(Duration::from_nanos(10));
        std::thread::sleep(Duration::from_millis(20));
        assert!(!h.is_finished());
        drop(held);
        assert_eq!(h.join().unwrap(), TimedOut(true));
    }

    /// A timeout wakes only waits whose deadline passed, except as a spurious wakeup of others on
    /// the same condvar; waits on other condvars stay parked
    #[test]
    fn timeouts_follow_each_deadline() {
        let c = Arc::new(VirtualClock::new());
        let a = Arc::new((Mutex::new(()), Condvar::new()));
        let b = Arc::new((Mutex::new(()), Condvar::new()));
        let spawn = |s: Arc<(Mutex<()>, Condvar)>, at: u64| {
            let c2 = c.clone();
            std::thread::Builder::new()
                .spawn(move || {
                    let mut g = s.0.lock();
                    c2.wait_until(&s.1, &mut g, Deadline::Mono(MonoTime::from_nanos(at)))
                })
                .unwrap()
        };
        let early = spawn(a.clone(), 100);
        let late = spawn(b.clone(), 200);
        assert!(c.wait_for_waiters(2, REAL));
        c.advance(Duration::from_nanos(100));
        assert_eq!(early.join().unwrap(), TimedOut(true));
        std::thread::sleep(Duration::from_millis(20));
        assert!(!late.is_finished());
        c.advance(Duration::from_nanos(100));
        assert_eq!(late.join().unwrap(), TimedOut(true));
        assert_eq!(c.waiters(), 0);
    }

    /// A deadline already reached returns at once; the wall deadline uses wall time
    #[test]
    fn past_deadlines() {
        let c = VirtualClock::new();
        let m = Mutex::new(());
        let cv = Condvar::new();
        let mut g = m.lock();
        assert!(
            c.wait_until(&cv, &mut g, Deadline::Mono(MonoTime::from_nanos(0)))
                .timed_out()
        );
        assert!(
            c.wait_until(
                &cv,
                &mut g,
                Deadline::Wall(WallTime::from_nanos(WALL_EPOCH_NANOS))
            )
            .timed_out()
        );
        c.sleep_until(Deadline::Mono(MonoTime::from_nanos(0)));
    }

    /// sleep_until wakes on advance; untimed waits count as waiters
    #[test]
    fn sleep_until_and_untimed_wait() {
        let c = Arc::new(VirtualClock::new());
        let c2 = c.clone();
        let h = std::thread::Builder::new()
            .spawn(move || c2.sleep_until(Deadline::Mono(MonoTime::from_nanos(500))))
            .unwrap();
        assert!(c.wait_for_waiters(1, REAL));
        c.advance(Duration::from_nanos(500));
        h.join().unwrap();

        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let (c3, s3) = (c.clone(), state.clone());
        let h = std::thread::Builder::new()
            .spawn(move || {
                let mut g = s3.0.lock();
                while !*g {
                    c3.wait(&s3.1, &mut g);
                }
            })
            .unwrap();
        assert!(c.wait_for_waiters(1, REAL));
        *state.0.lock() = true;
        state.1.notify_all();
        h.join().unwrap();
        assert!(!c.wait_for_waiters(1, Duration::ZERO));
    }

    /// expect_hang drives the virtual clock
    #[test]
    fn works_with_expect_hang() {
        let c = Arc::new(VirtualClock::new());
        let c2 = c.clone();
        crate::core::testkit::capture::expect_hang(&*c, Duration::from_secs(1), move || {
            c2.sleep_until(Deadline::Mono(MonoTime::from_nanos(5_000_000_000)));
        });
    }
}
