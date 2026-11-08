//! The threads and the clock the video crates run on: [`Runtime`], which their host gives them.
//!
//! This crate is independent of reims-vgpu-core and of QEMU, so it does not link the library's
//! platform: it is handed a runtime of three calls instead, a thread's start, the monotonic
//! clock's time and one timed wait, which each backend that starts a thread, waits or times a
//! call keeps from its making.
//! - **The library's** is its platform's (reims-vgpu's `video/runtime.rs`): a backend's thread
//!   is the platform's, named, with the scheduling hints the platform gives a queue's worker,
//!   and marked as the library's, so that a panic on it ends the process at once
//!   as on every thread of the library's (architecture 6.5); and every bound is kept, and every
//!   time measured, on the platform's monotonic clock, which a test makes virtual.
//! - **std's**, [`StdRuntime`], is the examples' and the fuzzers': std's threads, and a clock
//!   read from std's `Instant`.
//!
//! What the guest observes of a bound is its length on the monotonic clock, the same on either.
//! The waits a bound ends are [`Bell::wait_until`]'s, and [`answer`]'s on it: a [`Bell`] is the
//! one place a video crate waits on a condition variable with a deadline. Locks are
//! [`crate::core::sync`]'s.

use std::fmt;
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::core::sync::{Condvar, Mutex, MutexGuard};

/// A point on a runtime's monotonic clock, in nanoseconds from an origin of the runtime's own:
/// only a difference of two of one runtime means anything.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonoTime(u64);

impl MonoTime {
    /// The end of time: a deadline no clock reaches, for a wait with no bound.
    pub const END: MonoTime = MonoTime(u64::MAX);

    /// From nanoseconds on the clock.
    pub const fn from_nanos(nanos: u64) -> Self {
        MonoTime(nanos)
    }

    /// Nanoseconds on the clock.
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// `self + d`, saturating at [`MonoTime::END`].
    pub fn saturating_add(self, d: Duration) -> Self {
        MonoTime(
            self.0
                .saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)),
        )
    }

    /// `self - earlier`, or zero where `earlier` is later.
    pub fn saturating_duration_since(self, earlier: MonoTime) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(earlier.0))
    }
}

/// Whether a timed wait ended at its deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimedOut(pub bool);

impl TimedOut {
    /// Whether the wait ended at its deadline.
    pub const fn timed_out(self) -> bool {
        self.0
    }
}

/// What a [`Bell`]'s lock guards: how many times it has rung. [`Runtime::wait_until`] waits with
/// its guard held, and does nothing else with it.
#[derive(Debug, Default)]
pub struct Rings(u64);

/// The threads and the clock a video crate runs on (module doc).
pub trait Runtime: Send + Sync {
    /// Starts a detached thread named `name` that runs `body`. `Err` where no thread starts: the
    /// caller does without it, as it says. The library's platform starts every thread it is
    /// asked for, or ends the process.
    fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()>;

    /// Now, on the monotonic clock, which counts the time the host sleeps.
    fn now(&self) -> MonoTime;

    /// One wait on `cv`, whose lock `rings` holds, until a notify, a spurious wakeup or
    /// `deadline` on [`Runtime::now`]'s clock, after which it answers whether the deadline has
    /// passed. A deadline already passed answers so at once, the lock held throughout.
    /// [`Bell::wait_until`] is its caller.
    fn wait_until(
        &self,
        cv: &Condvar,
        rings: &mut MutexGuard<'_, Rings>,
        deadline: MonoTime,
    ) -> TimedOut;

    /// How long since `start`, a time of this runtime's.
    fn elapsed_since(&self, start: MonoTime) -> Duration {
        self.now().saturating_duration_since(start)
    }

    /// The time `d` from now: a deadline.
    fn deadline_in(&self, d: Duration) -> MonoTime {
        self.now().saturating_add(d)
    }
}

/// What a timed wait waits on: rung by each thread that changes what its waiters wait for, after
/// the change. A waiter asks its own question of its own state ([`Bell::wait_until`]'s `ready`),
/// under its own locks, so a bell guards nothing but its count of rings, and a ring between a
/// waiter's question and its wait is never lost: the waiter sees the count has moved, and asks
/// again.
#[derive(Debug, Default)]
pub struct Bell {
    rings: Mutex<Rings>,
    rung: Condvar,
}

impl Bell {
    /// A bell no one has rung.
    pub fn new() -> Self {
        Self::default()
    }

    /// Wakes every wait on the bell, to ask its question again: after the change it answers, from
    /// any thread, with any lock held.
    pub fn ring(&self) {
        {
            let mut rings = self.rings.lock();
            rings.0 = rings.0.wrapping_add(1);
        }
        self.rung.notify_all();
    }

    /// `ready`'s answer, asked now and after each ring until it has one, or `None` once
    /// `deadline` has passed on `runtime`'s clock. `ready` is asked once more at the deadline, so
    /// an answer given before it is taken however late the waiter sees it. `ready` is called with
    /// no lock of the bell's held.
    pub fn wait_until<R>(
        &self,
        runtime: &dyn Runtime,
        deadline: MonoTime,
        mut ready: impl FnMut() -> Option<R>,
    ) -> Option<R> {
        loop {
            let seen = self.rings.lock().0;
            if let Some(answer) = ready() {
                return Some(answer);
            }
            let mut rings = self.rings.lock();
            while rings.0 == seen {
                if runtime
                    .wait_until(&self.rung, &mut rings, deadline)
                    .timed_out()
                {
                    drop(rings);
                    return ready();
                }
            }
        }
    }

    /// `ready`'s answer, waited for with no bound.
    pub fn wait<R>(&self, runtime: &dyn Runtime, ready: impl FnMut() -> Option<R>) -> Option<R> {
        self.wait_until(runtime, MonoTime::END, ready)
    }
}

/// One value one thread hands another, which the other waits for on a runtime's clock: the
/// [`Answerer`] that gives it, and the [`Answer`] that takes it.
pub fn answer<T>() -> (Answerer<T>, Answer<T>) {
    let shared = Arc::new(Slot {
        given: Mutex::new(Held::Waiting),
        bell: Bell::new(),
    });
    (Answerer(Arc::clone(&shared)), Answer(shared))
}

/// Where an answer is, as both halves see it.
enum Held<T> {
    Waiting,
    Given(T),
    /// Taken by the [`Answer`], or never to come: its [`Answerer`] went without giving it.
    Gone,
}

struct Slot<T> {
    given: Mutex<Held<T>>,
    bell: Bell,
}

/// The half of [`answer`] that gives the value. Dropped without giving it, it tells the
/// [`Answer`] that none is to come.
pub struct Answerer<T>(Arc<Slot<T>>);

impl<T> Answerer<T> {
    /// Gives the answer.
    pub fn give(self, value: T) {
        *self.0.given.lock() = Held::Given(value);
        self.0.bell.ring();
    }
}

impl<T> Drop for Answerer<T> {
    fn drop(&mut self) {
        let mut given = self.0.given.lock();
        if matches!(*given, Held::Waiting) {
            *given = Held::Gone;
            drop(given);
            self.0.bell.ring();
        }
    }
}

impl<T> fmt::Debug for Answerer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Answerer")
    }
}

/// The half of [`answer`] that takes the value.
pub struct Answer<T>(Arc<Slot<T>>);

/// Why [`Answer::wait_until`] has no answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unanswered {
    /// The deadline passed first.
    TimedOut,
    /// None is to come: the [`Answerer`] went without giving it, or it was taken already.
    Gone,
}

impl<T> Answer<T> {
    /// The answer, once it is given, or why there is none: [`Unanswered::TimedOut`] once
    /// `deadline` has passed on `runtime`'s clock with none given, [`Unanswered::Gone`] where
    /// none is to come.
    pub fn wait_until(&self, runtime: &dyn Runtime, deadline: MonoTime) -> Result<T, Unanswered> {
        let slot = &*self.0;
        let taken = slot.bell.wait_until(runtime, deadline, || {
            let mut given = slot.given.lock();
            match std::mem::replace(&mut *given, Held::Gone) {
                Held::Waiting => {
                    *given = Held::Waiting;
                    None
                }
                Held::Given(value) => Some(Ok(value)),
                Held::Gone => Some(Err(Unanswered::Gone)),
            }
        });
        taken.unwrap_or(Err(Unanswered::TimedOut))
    }
}

impl<T> fmt::Debug for Answer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Answer")
    }
}

/// std's threads and clock: the runtime of the examples, the hardware tests and the fuzzers
/// (module doc), never the library's. Its threads are std's, named as asked, with std's default
/// stack; its clock is std's `Instant`, from the first time a `StdRuntime` reads it.
#[derive(Clone, Copy, Debug, Default)]
pub struct StdRuntime;

impl StdRuntime {
    /// The process's, as a runtime to hand a backend.
    pub fn shared() -> Arc<dyn Runtime> {
        static SHARED: OnceLock<Arc<StdRuntime>> = OnceLock::new();
        let shared: Arc<StdRuntime> = Arc::clone(SHARED.get_or_init(|| Arc::new(StdRuntime)));
        shared
    }

    /// A std thread named `name` that runs `body`, which the caller may join: the stand-ins'
    /// own threads, which play the host's and the guest's.
    pub fn thread<T: Send + 'static>(
        name: &str,
        body: impl FnOnce() -> T + Send + 'static,
    ) -> std::io::Result<JoinHandle<T>> {
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(body)
    }

    /// The instant the clock counts from.
    #[expect(
        clippy::disallowed_methods,
        reason = "std's runtime reads std's clock, once: its origin, which every `now` is \
                  measured from"
    )]
    fn origin() -> Instant {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        *ORIGIN.get_or_init(Instant::now)
    }
}

impl Runtime for StdRuntime {
    fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        StdRuntime::thread(name, body).map(drop)
    }

    fn now(&self) -> MonoTime {
        let since = StdRuntime::origin().elapsed();
        MonoTime::from_nanos(u64::try_from(since.as_nanos()).unwrap_or(u64::MAX))
    }

    fn wait_until(
        &self,
        cv: &Condvar,
        rings: &mut MutexGuard<'_, Rings>,
        deadline: MonoTime,
    ) -> TimedOut {
        let left = deadline.saturating_duration_since(self.now());
        if left.is_zero() {
            return TimedOut(true);
        }
        let _ = cv.wait_for(rings, left);
        TimedOut(self.now() >= deadline)
    }
}

#[cfg(test)]
pub(crate) mod virtual_time {
    //! A runtime on virtual time, for the tests: std's threads, and a clock that moves only
    //! when a test advances it.

    use super::*;

    /// How long a wait on virtual time blocks in real time before it looks at the clock again.
    const SLICE: Duration = Duration::from_millis(1);

    /// std's threads, and virtual time: a wait ends at its deadline only once a test has
    /// advanced the clock past it.
    #[derive(Debug, Default)]
    pub(crate) struct VirtualRuntime {
        now: Mutex<u64>,
    }

    impl VirtualRuntime {
        /// Moves the clock `d` on.
        pub(crate) fn advance(&self, d: Duration) {
            let mut now = self.now.lock();
            *now = now.saturating_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
        }
    }

    impl Runtime for VirtualRuntime {
        fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
            StdRuntime.spawn(name, body)
        }

        fn now(&self) -> MonoTime {
            MonoTime::from_nanos(*self.now.lock())
        }

        /// One slice of real time, then the answer on virtual time: a spurious wakeup to the
        /// caller, who asks again.
        fn wait_until(
            &self,
            cv: &Condvar,
            rings: &mut MutexGuard<'_, Rings>,
            deadline: MonoTime,
        ) -> TimedOut {
            if self.now() >= deadline {
                return TimedOut(true);
            }
            let _ = cv.wait_for(rings, SLICE);
            TimedOut(self.now() >= deadline)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::virtual_time::VirtualRuntime;
    use super::*;

    const LONG: Duration = Duration::from_secs(10);

    /// Time arithmetic saturates, and a runtime's provided calls measure on its clock.
    #[test]
    fn time_arithmetic_saturates() {
        let t = MonoTime::from_nanos(10);
        assert_eq!(t.saturating_add(Duration::from_nanos(5)).as_nanos(), 15);
        assert_eq!(MonoTime::END.saturating_add(Duration::MAX), MonoTime::END);
        assert_eq!(
            t.saturating_duration_since(MonoTime::from_nanos(20)),
            Duration::ZERO
        );
        let clock = VirtualRuntime::default();
        let start = clock.now();
        clock.advance(Duration::from_millis(3));
        assert_eq!(clock.elapsed_since(start), Duration::from_millis(3));
        assert_eq!(
            clock.deadline_in(Duration::from_millis(1)),
            MonoTime::from_nanos(4_000_000)
        );
    }

    /// std's clock moves forward.
    #[test]
    fn std_time_moves_forward() {
        let rt = StdRuntime;
        let a = rt.now();
        std::thread::sleep(Duration::from_millis(2));
        assert!(rt.elapsed_since(a) >= Duration::from_millis(2));
    }

    /// A ring from another thread ends the wait with the waiter's answer.
    #[test]
    fn a_ring_wakes_a_wait() {
        let rt = StdRuntime::shared();
        let bell = Arc::new(Bell::new());
        let flag = Arc::new(Mutex::new(false));
        let ringer = {
            let (bell, flag) = (Arc::clone(&bell), Arc::clone(&flag));
            StdRuntime::thread("ringer", move || {
                std::thread::sleep(Duration::from_millis(5));
                *flag.lock() = true;
                bell.ring();
            })
        };
        let got = bell.wait_until(&*rt, rt.deadline_in(LONG), || flag.lock().then_some(7));
        assert_eq!(got, Some(7));
        assert!(ringer.is_ok_and(|t| t.join().is_ok()));
    }

    /// A wait with no answer ends at its deadline, on the runtime's clock: on virtual time only
    /// once the clock passes it.
    #[test]
    fn a_wait_ends_at_its_deadline_on_virtual_time() {
        let rt = Arc::new(VirtualRuntime::default());
        let bell = Arc::new(Bell::new());
        let (done, finished) = answer::<Option<()>>();
        let waiter = {
            let (rt, bell) = (Arc::clone(&rt), Arc::clone(&bell));
            StdRuntime::thread("waiter", move || {
                let got = bell.wait_until(&*rt, rt.deadline_in(LONG), || None::<()>);
                done.give(got);
            })
        };
        std::thread::sleep(Duration::from_millis(20));
        let real = StdRuntime;
        assert_eq!(
            finished.wait_until(&real, real.deadline_in(Duration::from_millis(1))),
            Err(Unanswered::TimedOut),
            "it ended before virtual time passed its deadline"
        );
        rt.advance(LONG);
        assert_eq!(finished.wait_until(&real, real.deadline_in(LONG)), Ok(None));
        assert!(waiter.is_ok_and(|t| t.join().is_ok()));
    }

    /// An answer given before the deadline is taken, however late; one is taken once; and an
    /// answerer that goes without giving one says none is to come.
    #[test]
    fn answers_are_given_once_or_gone() {
        let rt = VirtualRuntime::default();
        let (give, take) = answer();
        give.give(5);
        rt.advance(LONG);
        assert_eq!(take.wait_until(&rt, MonoTime::default()), Ok(5));
        assert_eq!(take.wait_until(&rt, MonoTime::END), Err(Unanswered::Gone));
        let (give, take) = answer::<u8>();
        drop(give);
        assert_eq!(take.wait_until(&rt, MonoTime::END), Err(Unanswered::Gone));
        let (_give, take) = answer::<u8>();
        assert_eq!(
            take.wait_until(&rt, MonoTime::default()),
            Err(Unanswered::TimedOut)
        );
    }
}
