//! How long a pool's thread with nothing to do is kept before it goes ([`Keep`]).
//!
//! A pool that keeps each thread with nothing to do for [`KEEP`], and gives the next job to the
//! thread that went idle last, keeps as many threads as it ran at once in the last [`KEEP`]: the
//! threads a steady load runs take its jobs and stay through its pauses, and those only a burst
//! ran wait longest and go once they have had no job for [`KEEP`].
//!
//! A thread's start may map a stack (2 MiB) and its end unmap one, writes of the process's memory
//! map that a guest vCPU's page fault waits behind. With the device's and the decoder's threads
//! kept 1 s, 345 graphics threads were made in 30 s of a WindowServer probe (car 12's B1), and 7
//! to 10 every 10 s in games, and a vCPU sat 30 ms or more in a page fault while one of those
//! threads was in `munmap` of 0x201000 bytes, inside an 81 ms guest frame (msi RTX 5080 host;
//! measured).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::{Condvar, MutexGuard};

/// How long a pool's thread with nothing to do is kept before it goes (the module doc).
pub const KEEP: Duration = Duration::from_secs(10);

/// How long a thread with nothing to do waits for a job before it goes, timed on the host's
/// monotonic clock, or on a [`TestClock`] the tests move.
#[derive(Clone, Debug)]
pub struct Keep {
    keep: Duration,
    clock: Option<Arc<TestClock>>,
}

impl Default for Keep {
    /// [`KEEP`], on the host's monotonic clock.
    fn default() -> Keep {
        Keep::new(KEEP)
    }
}

impl Keep {
    /// A thread waits `keep`, on the host's monotonic clock.
    pub fn new(keep: Duration) -> Keep {
        Keep { keep, clock: None }
    }

    /// A thread waits `keep` of `clock`'s time, which passes only as the test moves it.
    pub fn on(clock: &Arc<TestClock>, keep: Duration) -> Keep {
        Keep {
            keep,
            clock: Some(Arc::clone(clock)),
        }
    }

    /// The time on its clock, in nanoseconds: what a thread that begins to wait passes to
    /// [`Self::wait`].
    pub fn now(&self) -> u64 {
        match &self.clock {
            Some(clock) => clock.now(),
            None => mono_ns(),
        }
    }

    /// Waits on `wake`, whose lock `guard` holds, until it is notified or the keep has passed
    /// since `since` (a [`Self::now`]); whether it has passed. On a [`TestClock`] it looks again
    /// every millisecond, so a move of the clock is seen at once.
    pub fn wait<T>(&self, wake: &Condvar, guard: &mut MutexGuard<'_, T>, since: u64) -> bool {
        let keep = u64::try_from(self.keep.as_nanos()).unwrap_or(u64::MAX);
        let waited = self.now().saturating_sub(since);
        if waited >= keep {
            return true;
        }
        let slice = match self.clock {
            Some(_) => Duration::from_millis(1),
            None => Duration::from_nanos(keep - waited),
        };
        let _ = wake.wait_for(guard, slice);
        self.now().saturating_sub(since) >= keep
    }
}

/// A clock for the tests: a [`Keep`] on it passes only as the test moves it ([`Self::advance`]).
#[derive(Debug, Default)]
pub struct TestClock {
    now: AtomicU64,
}

impl TestClock {
    /// A clock at 0.
    pub fn new() -> Arc<TestClock> {
        Arc::default()
    }

    /// Moves the clock on by `by`.
    pub fn advance(&self, by: Duration) {
        let by = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.now.fetch_add(by, Ordering::SeqCst);
    }

    fn now(&self) -> u64 {
        self.now.load(Ordering::SeqCst)
    }
}

/// The host's monotonic clock, in nanoseconds.
fn mono_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes the timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    let secs = u64::try_from(ts.tv_sec).unwrap_or(0);
    let nanos = u64::try_from(ts.tv_nsec).unwrap_or(0);
    secs.saturating_mul(1_000_000_000).saturating_add(nanos)
}

#[cfg(test)]
mod tests {
    use parking_lot::Mutex;

    use super::*;

    /// A keep on a test clock passes only as the clock is moved past it, counted from the wait's
    /// start.
    #[test]
    fn a_keep_on_a_test_clock_passes_only_as_the_clock_moves_past_it() {
        let clock = TestClock::new();
        let keep = Keep::on(&clock, Duration::from_secs(10));
        let (lock, wake) = (Mutex::new(()), Condvar::new());
        let mut guard = lock.lock();
        clock.advance(Duration::from_secs(3));
        let since = keep.now();
        clock.advance(Duration::from_secs(9));
        assert!(!keep.wait(&wake, &mut guard, since), "9 s of 10");
        clock.advance(Duration::from_secs(1));
        assert!(keep.wait(&wake, &mut guard, since), "10 s of 10");
    }

    /// A keep on the host's clock passes once its time has.
    #[test]
    fn a_keep_on_the_host_s_clock_passes_with_its_time() {
        let keep = Keep::new(Duration::from_millis(1));
        let (lock, wake) = (Mutex::new(()), Condvar::new());
        let mut guard = lock.lock();
        let since = keep.now();
        while !keep.wait(&wake, &mut guard, since) {}
        assert!(keep.now() - since >= 1_000_000);
    }
}
