//! A runtime with an exit gate of its own, as the host's is (`reims_vgpu_vulkan::exit_gate`,
//! shaped as that gate is), for the tests of what the crate does as a process exits: its hook,
//! registered by [`Runtime::arm_exit`], closes the gate and waits out the steps in progress, at
//! most 2 s; a step after that is refused. One gate a process, so a test that closes it runs in
//! a child process of its own.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use crate::core::runtime::{ExitStep, MonoTime, Rings, Runtime, StdRuntime, TimedOut};
use crate::core::sync::{Condvar, MutexGuard};

static CLOSED: AtomicBool = AtomicBool::new(false);
static IN_PROGRESS: AtomicUsize = AtomicUsize::new(0);

/// std's threads and clock, and the stand-in gate's steps.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GatedRuntime;

struct InProgress;

impl Drop for InProgress {
    fn drop(&mut self) {
        IN_PROGRESS.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The hook: closes the gate, then waits for the steps in progress, at most 2 s.
pub(crate) extern "C" fn close_the_gate() {
    CLOSED.store(true, Ordering::SeqCst);
    for _ in 0..2000 {
        if IN_PROGRESS.load(Ordering::SeqCst) == 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

impl Runtime for GatedRuntime {
    fn spawn(&self, name: &'static str, body: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
        StdRuntime.spawn(name, body)
    }

    fn now(&self) -> MonoTime {
        StdRuntime.now()
    }

    fn wait_until(
        &self,
        cv: &Condvar,
        rings: &mut MutexGuard<'_, Rings>,
        deadline: MonoTime,
    ) -> TimedOut {
        StdRuntime.wait_until(cv, rings, deadline)
    }

    fn exit_step(&self) -> Option<ExitStep> {
        // Counted before the gate is read, as the host's gate counts: either the hook sees this
        // step, or this step sees the gate closed.
        IN_PROGRESS.fetch_add(1, Ordering::SeqCst);
        if CLOSED.load(Ordering::SeqCst) {
            IN_PROGRESS.fetch_sub(1, Ordering::SeqCst);
            return None;
        }
        Some(ExitStep::of(Box::new(InProgress)))
    }

    fn arm_exit(&self) {
        // SAFETY: a plain function that never unwinds.
        assert_eq!(unsafe { libc::atexit(close_the_gate) }, 0);
    }
}
