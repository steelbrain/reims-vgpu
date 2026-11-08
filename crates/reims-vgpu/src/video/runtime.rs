//! A platform as the video crates' runtime ([`Runtime`]): the threads the host sessions and the
//! backends start, and the clock their bounds and their timings are on.
//!
//! - **A thread** is a worker of the platform's ([`Threads::spawn_worker`]), as a serial queue's
//!   is: a session's thread runs the messages of its queue one at a time, as a queue's worker
//!   runs its blocks. It is named as the video crate asks, has a worker's 512 KiB of stack, and
//!   takes the hints the platform gives a thread that is not a FIFO thread (on Linux it drops
//!   those it inherited from the thread that started it). Its body runs marked as the
//!   library's ([`enter_ours`]), so that a panic on it ends the process before anything
//!   unwinds, as on every thread of the library's (architecture 6.5). The platform starts every
//!   thread it is asked for, or ends the process, so a video crate never meets the refusal it
//!   handles from std's threads.
//! - **The clock** is the platform's monotonic clock, and a wait is the platform's
//!   [`Clock::wait_until`] on a deadline on it: a bound counts the time the host sleeps, as the
//!   platform's own bounds do, and a test's virtual clock is a session's too. A backend waits
//!   for its GPU on the same clock, so a test on a virtual clock opens none.

use std::sync::Arc;

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::os::{Clock as _, Deadline, MonoTime as CoreTime};
#[cfg(doc)]
use reims_vgpu_core::platform::os::{Clock, Threads};
use reims_vgpu_core::rt::enter_ours;
use reims_vgpu_video_core::runtime::{MonoTime, Rings, Runtime, TimedOut};
use reims_vgpu_video_core::sync::{Condvar, MutexGuard};

/// `P` as a video crate's runtime (module doc).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the graphics device's video arms run their sessions on it"
    )
)]
pub(crate) struct PlatformRuntime<P: Platform>(Arc<P>);

impl<P: Platform> PlatformRuntime<P> {
    /// `platform`'s runtime, to hand a session or a backend.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the graphics device's video arms run their sessions on it"
        )
    )]
    pub(crate) fn of(platform: &Arc<P>) -> Arc<dyn Runtime> {
        Arc::new(PlatformRuntime(Arc::clone(platform)))
    }
}

impl<P: Platform> Runtime for PlatformRuntime<P> {
    fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        self.0.threads().spawn_worker(
            name,
            Box::new(move || {
                let _ours = enter_ours();
                body();
            }),
        );
        Ok(())
    }

    fn now(&self) -> MonoTime {
        MonoTime::from_nanos(self.0.clock().now_mono().as_nanos())
    }

    fn wait_until(
        &self,
        cv: &Condvar,
        rings: &mut MutexGuard<'_, Rings>,
        deadline: MonoTime,
    ) -> TimedOut {
        let deadline = Deadline::Mono(CoreTime::from_nanos(deadline.as_nanos()));
        TimedOut(self.0.clock().wait_until(cv, rings, deadline).timed_out())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use reims_vgpu_core::rt::is_our_thread;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_video_core::runtime::{Bell, StdRuntime, Unanswered, answer};

    use super::*;

    /// A thread is the platform's queue worker, named for the video crate and marked as ours,
    /// and the clock is the platform's: on a fake platform, its virtual clock.
    #[test]
    fn a_runtime_is_its_platforms_threads_and_clock() {
        let platform = Arc::new(FakePlatform::new());
        let runtime = PlatformRuntime::of(&platform);
        let (ran, ours) = answer();
        let spawned = runtime.spawn(
            "reims-vgpu.video-test",
            Box::new(move || ran.give(is_our_thread())),
        );
        assert!(spawned.is_ok());
        let real = StdRuntime;
        let waited = ours.wait_until(&real, real.deadline_in(Duration::from_secs(10)));
        assert_eq!(waited, Ok(true));
        assert_eq!(platform.fake_threads().queues(), ["reims-vgpu.video-test"]);
        let start = runtime.now();
        platform.virtual_clock().advance(Duration::from_secs(3));
        assert_eq!(runtime.elapsed_since(start), Duration::from_secs(3));
    }

    /// A wait's bound is on the platform's clock: on virtual time, it passes only once the
    /// clock does.
    #[test]
    fn a_bound_passes_on_the_platforms_clock() {
        let platform = Arc::new(FakePlatform::new());
        let runtime = PlatformRuntime::of(&platform);
        let deadline = runtime.deadline_in(Duration::from_secs(10));
        let (done, waited) = answer();
        let waiting = Arc::clone(&runtime);
        let started = StdRuntime::thread("waiter", move || {
            let bell = Bell::new();
            done.give(bell.wait_until(&*waiting, deadline, || None::<()>));
        });
        assert!(started.is_ok());
        let real = StdRuntime;
        assert!(
            platform
                .virtual_clock()
                .wait_for_waiters(1, Duration::from_secs(10))
        );
        assert_eq!(
            waited.wait_until(&real, real.deadline_in(Duration::from_millis(20))),
            Err(Unanswered::TimedOut)
        );
        platform.virtual_clock().advance(Duration::from_secs(10));
        assert_eq!(
            waited.wait_until(&real, real.deadline_in(Duration::from_secs(10))),
            Ok(None)
        );
    }
}
