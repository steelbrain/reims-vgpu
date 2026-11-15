//! `Clock` over Windows's clocks, as macOS's answers them.
//!
//! The monotonic clock is the interrupt time (`QueryInterruptTimePrecise`, in 100 ns units): like
//! `CLOCK_MONOTONIC_RAW` on macOS (libc++'s `steady_clock` there) and Linux's `CLOCK_BOOTTIME`, it
//! counts the time the host sleeps, which the unbiased interrupt time does not (read, Microsoft's
//! documentation of `QueryInterruptTime` and `QueryUnbiasedInterruptTime`). The wall clock is the
//! system time (`GetSystemTimePreciseAsFileTime`, 100 ns units since 1601), in nanoseconds since
//! the Unix epoch. A condvar's timed wait blocks for its timeout in awake time (`parking_lot`'s
//! `wait_for` hands Windows a relative timeout), and answers as libc++'s does on macOS: after the
//! wait, `now < deadline ? no_timeout : timeout`, read on the deadline's own clock, where a host
//! sleep counts.

use std::time::Duration;

use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime;
use windows_sys::Win32::System::WindowsProgramming::QueryInterruptTimePrecise;

use crate::core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use crate::core::sync::{Condvar, MutexGuard};

/// 100 ns units from 1601-01-01 to the Unix epoch, 1970-01-01.
const UNIX_EPOCH_IN_FILETIME: u64 = 116_444_736_000_000_000;

/// The interrupt time, in nanoseconds.
fn interrupt_nanos() -> u64 {
    let mut units = 0u64;
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { QueryInterruptTimePrecise(&mut units) };
    units.saturating_mul(100)
}

/// The system time, in nanoseconds since the Unix epoch.
fn system_nanos() -> u64 {
    let mut ft = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { GetSystemTimePreciseAsFileTime(&mut ft) };
    let units = (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime);
    units
        .saturating_sub(UNIX_EPOCH_IN_FILETIME)
        .saturating_mul(100)
}

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct WindowsClock;

impl WindowsClock {
    /// The process's Clock.
    pub fn new() -> WindowsClock {
        WindowsClock
    }

    /// How long until `deadline`, on its own clock; zero once reached.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for WindowsClock {
    /// The interrupt time, which counts a host sleep as macOS's monotonic clock does.
    fn now_mono(&self) -> MonoTime {
        MonoTime::from_nanos(interrupt_nanos())
    }

    /// The system time.
    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(system_nanos())
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
    use super::*;
    use crate::core::sync::Mutex;

    const MS: Duration = Duration::from_millis(1);

    /// The monotonic clock is the interrupt time, and moves forward at the rate Instant counts
    /// while awake
    #[test]
    fn mono_is_the_interrupt_time_and_moves_forward() {
        let c = WindowsClock::new();
        let before = interrupt_nanos();
        let ours = c.now_mono().as_nanos();
        let after = interrupt_nanos();
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
        assert!(elapsed >= 4 * MS, "{elapsed:?}");
        assert!(elapsed <= i1.duration_since(i0) + MS, "{elapsed:?}");
    }

    /// The wall clock is nanoseconds since the Unix epoch
    #[test]
    fn wall_is_unix_epoch_nanos() {
        let c = WindowsClock::new();
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
        let c = WindowsClock::new();
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
}
