//! Threads: the platform starts the threads core runs on (its named threads and each serial
//! queue's worker), and gives the clock timers wait on ([`TimerClock`]).

use std::sync::Arc;

use super::{Clock, Deadline};
use crate::core::sync::{Condvar, MutexGuard};

/// How core gets its threads (architecture 7.1). Core creates no OS thread that runs
/// Objective-C itself: the platform does, so that in production no Rust unwind handler sits at
/// the thread's base (D35) and an exception nothing catches ends the process in phase 1, with no
/// drops. The fake uses `std::thread` (no Objective-C runs there).
pub trait Threads: Send + Sync {
    /// Starts a detached thread named `name`, with `stack_size` bytes of stack (rounded up to
    /// whole pages), that runs `body`: a named thread of core's (`rt::thread::spawn_named`), a
    /// display window's.
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>);

    /// Starts a detached worker thread for the serial queue `label` (`rt::SerialQueue`), as
    /// [`Threads::spawn`] starts one.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>);

    /// Gives the calling thread, one this service did not start, the scheduling hints of a
    /// thread whose speed is the frame rate: a walk thread of the graphics device's lanes, which
    /// walks a command buffer's ops one after another. None where the platform gives none.
    fn give_walk_hints(&self) {}

    /// An owned handle to this service: a queue keeps one to restart its worker.
    fn handle(&self) -> Arc<dyn Threads>;

    /// The clock `rt::Timer`s wait on: the platform's [`Clock`]. `None` where the platform makes
    /// no timers (a fake with no clock).
    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>>;
}

/// A [`Clock`] as an object, for a timer's thread: the time on a deadline's own clock, and one
/// timed wait.
pub trait TimerClock: Send + Sync {
    /// Nanoseconds now, on `d`'s own clock.
    fn now_on(&self, d: Deadline) -> u64;
    /// One timed wait on `cv` until `d` (the clock's [`Clock::wait_until`]).
    fn wait_until(&self, cv: &Condvar, g: &mut MutexGuard<'_, bool>, d: Deadline);
}

impl<C: Clock> TimerClock for C {
    fn now_on(&self, d: Deadline) -> u64 {
        match d {
            Deadline::Mono(_) => self.now_mono().as_nanos(),
            Deadline::Wall(_) => self.now_wall().as_nanos(),
        }
    }

    fn wait_until(&self, cv: &Condvar, g: &mut MutexGuard<'_, bool>, d: Deadline) {
        let _ = Clock::wait_until(self, cv, g, d);
    }
}
