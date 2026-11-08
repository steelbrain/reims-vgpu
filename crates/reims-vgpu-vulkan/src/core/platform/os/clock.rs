//! Time and timed waits: `MonoTime`, `WallTime`, `Deadline`, `TimedOut` and the `Clock`
//! service.

use std::time::Duration;

use crate::core::sync::{Condvar, MutexGuard};

/// A point on the monotonic clock, in nanoseconds (name is ours: architecture 4.5 names the type;
/// the device's 2 s stamp-wait report waits on it). The clock is libc++'s
/// `std::chrono::steady_clock` on Apple platforms, `clock_gettime(CLOCK_MONOTONIC_RAW)`
/// (`steady_clock::now()` in the system's `libc++.1.dylib` passes clock id 4; its error string
/// reads "clock_gettime(CLOCK_MONOTONIC_RAW) failed"), which counts the time the host sleeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MonoTime(u64);

impl MonoTime {
    /// From nanoseconds on the monotonic clock.
    pub const fn from_nanos(n: u64) -> Self {
        MonoTime(n)
    }
    /// Nanoseconds on the monotonic clock.
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
    /// `self + d`, saturating at the end of time.
    pub fn saturating_add(self, d: Duration) -> Self {
        MonoTime(
            self.0
                .saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
        )
    }
    /// `self - earlier`, or zero if `earlier` is later.
    pub fn saturating_duration_since(self, earlier: MonoTime) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

/// A point on the wall clock, in nanoseconds since the Unix epoch (`dispatch_walltime`,
/// `std::chrono::system_clock`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct WallTime(u64);

impl WallTime {
    /// From nanoseconds since the Unix epoch.
    pub const fn from_nanos(n: u64) -> Self {
        WallTime(n)
    }
    /// Nanoseconds since the Unix epoch.
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
    /// `self + d`, saturating.
    pub fn saturating_add(self, d: Duration) -> Self {
        WallTime(
            self.0
                .saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
        )
    }
    /// `self - earlier`, or zero if `earlier` is later.
    pub fn saturating_duration_since(self, earlier: WallTime) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

/// A timed wait's deadline: monotonic, or wall for 's system_clock wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deadline {
    /// On the monotonic clock.
    Mono(MonoTime),
    /// On the wall clock.
    Wall(WallTime),
}

impl Deadline {
    /// Whether `clock` has reached this deadline.
    pub fn reached<C: Clock + ?Sized>(self, clock: &C) -> bool {
        match self {
            Deadline::Mono(t) => clock.now_mono() >= t,
            Deadline::Wall(t) => clock.now_wall() >= t,
        }
    }

    /// The same clock, `d` later (saturating).
    pub fn saturating_add(self, d: Duration) -> Deadline {
        match self {
            Deadline::Mono(t) => Deadline::Mono(t.saturating_add(d)),
            Deadline::Wall(t) => Deadline::Wall(t.saturating_add(d)),
        }
    }
}

/// The result of a timed wait: `TimedOut(true)` when the deadline passed
/// (`std::cv_status::timeout`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimedOut(pub bool);

impl TimedOut {
    /// Whether the wait ended at its deadline.
    pub const fn timed_out(self) -> bool {
        self.0
    }
}

/// Time, and every timed wait (architecture 4.5).
///
/// Time the host spends asleep counts on both clocks (`CLOCK_MONOTONIC_RAW` and `CLOCK_REALTIME`).
/// What counts while a thread is blocked depends on who blocks it, and each wait below says which
/// it is.
pub trait Clock: Send + Sync + 'static {
    /// The monotonic clock: `std::chrono::steady_clock::now()` (see [`MonoTime`]).
    fn now_mono(&self) -> MonoTime;
    /// The wall clock: `std::chrono::system_clock::now()` and `dispatch_walltime`.
    fn now_wall(&self) -> WallTime;
    /// A condvar's timed wait. Timed condvar waits go through the clock so tests run on virtual
    /// time.
    ///
    /// `std::condition_variable::wait_for` and `wait_until`. On Apple, libc++ waits with
    /// `pthread_cond_timedwait` on a `system_clock` deadline (`_LIBCPP_HAS_COND_CLOCKWAIT` is 0:
    /// `<__config>`, `<__condition_variable/condition_variable.h>:141-166`, Xcode 26.6 SDK).
    /// libpthread turns that deadline into a relative timeout before the kernel wait, and the
    /// kernel counts it in awake time, so a host sleep does not end the wait. The answer is then
    /// read on the deadline's own clock (`steady_clock::now() - start < d` for `wait_for`,
    /// `Clock::now() < t` for `wait_until`), where the sleep does count: a wait a notify ends after
    /// a host sleep past the deadline answers timeout.
    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut;
    /// An untimed condvar wait (so a virtual clock can see the waiter).
    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>);
    /// Sleeps until `deadline`.
    fn sleep_until(&self, deadline: Deadline);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Time arithmetic saturates
    #[test]
    fn time_arithmetic() {
        let t = MonoTime::from_nanos(10);
        assert_eq!(t.saturating_add(Duration::from_nanos(5)).as_nanos(), 15);
        assert_eq!(
            MonoTime::from_nanos(u64::MAX)
                .saturating_add(Duration::MAX)
                .as_nanos(),
            u64::MAX
        );
        assert_eq!(
            t.saturating_duration_since(MonoTime::from_nanos(20)),
            Duration::ZERO
        );
        let w = WallTime::from_nanos(1_000);
        assert_eq!(w.saturating_add(Duration::from_micros(1)).as_nanos(), 2_000);
        assert_eq!(
            WallTime::from_nanos(3_000).saturating_duration_since(w),
            Duration::from_nanos(2_000)
        );
        assert_eq!(
            Deadline::Wall(w).saturating_add(Duration::from_nanos(1)),
            Deadline::Wall(WallTime::from_nanos(1_001))
        );
        assert_eq!(
            Deadline::Mono(t).saturating_add(Duration::from_nanos(1)),
            Deadline::Mono(MonoTime::from_nanos(11))
        );
        assert!(TimedOut(true).timed_out());
    }
}
