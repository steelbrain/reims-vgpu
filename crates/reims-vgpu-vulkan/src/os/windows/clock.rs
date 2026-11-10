//! Windows clocks with the platform's suspend-inclusive monotonic-time contract.
//!
//! QueryInterruptTimePrecise includes host sleep; QueryUnbiasedInterruptTime does not.
//! The result is a 100 ns counter, not a performance-counter value needing calibration.

use std::time::Duration;

use reims_vgpu_core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use reims_vgpu_core::sync::{Condvar, MutexGuard};

#[link(name = "kernel32")]
unsafe extern "system" {
    fn QueryInterruptTimePrecise(time: *mut u64);
    fn GetSystemTimePreciseAsFileTime(time: *mut u64);
}

/// Windows clock and condition-variable deadlines, in the same units as core's clocks.
#[derive(Debug, Default)]
pub struct WindowsClock;

impl WindowsClock {
    /// The process clock; its origin is Windows boot, shared by every instance.
    pub fn new() -> Self {
        Self
    }

    /// Remaining time on the deadline's own clock, including host suspension.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for WindowsClock {
    fn now_mono(&self) -> MonoTime {
        let mut ticks = 0;
        // SAFETY: QueryInterruptTimePrecise writes one aligned u64 and has no failure result.
        unsafe { QueryInterruptTimePrecise(&mut ticks) };
        MonoTime::from_nanos(ticks * 100)
    }

    fn now_wall(&self) -> WallTime {
        let mut ticks = 0u64;
        // SAFETY: FILETIME is two little-endian DWORDs, exactly this aligned
        // 64-bit output on the supported Windows x86-64 host. The API cannot fail.
        unsafe { GetSystemTimePreciseAsFileTime(&mut ticks) };
        // Windows starts in 1601; core's wall clock starts in 1970. Refuse an
        // unrepresentable date rather than wrapping a deadline or inventing time.
        const UNIX_EPOCH_TICKS: u64 = 116_444_736_000_000_000;
        assert!(ticks >= UNIX_EPOCH_TICKS);
        let unix = ticks - UNIX_EPOCH_TICKS;
        assert!(unix <= u64::MAX / 100);
        WallTime::from_nanos(unix * 100)
    }

    fn wait_until<T>(
        &self,
        cv: &Condvar,
        guard: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let remaining = self.remaining(deadline);
        if remaining.is_zero() {
            return TimedOut(true);
        }
        let _ = cv.wait_for(guard, remaining);
        TimedOut(deadline.reached(self))
    }

    fn wait<T>(&self, cv: &Condvar, guard: &mut MutexGuard<'_, T>) {
        cv.wait(guard);
    }

    fn sleep_until(&self, deadline: Deadline) {
        loop {
            let remaining = self.remaining(deadline);
            if remaining.is_zero() {
                return;
            }
            std::thread::sleep(remaining);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both instances must share an origin; constructing a clock cannot reset frame deadlines.
    #[test]
    fn instances_share_a_monotonic_origin() {
        let before = WindowsClock::new().now_mono();
        std::thread::sleep(Duration::from_millis(5));
        let after = WindowsClock::new().now_mono();
        assert!(after.saturating_duration_since(before) >= Duration::from_millis(5));
    }

    /// A deadline already reached must return immediately without consuming a notification.
    #[test]
    fn expired_deadline_does_not_wait() {
        let clock = WindowsClock::new();
        let mutex = reims_vgpu_core::sync::Mutex::new(());
        let mut guard = mutex.lock();
        assert!(
            clock
                .wait_until(
                    &Condvar::new(),
                    &mut guard,
                    Deadline::Mono(MonoTime::from_nanos(0))
                )
                .0
        );
    }
}
