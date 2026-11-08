//! `Threads` over `std::thread`: `FakeThreads`.

use std::sync::Arc;

use crate::platform::os::{Threads, TimerClock};
use crate::sync::{Condvar, Mutex, MutexGuard};
use crate::testkit::objects::World;

/// `Threads` over `std::thread`: no Objective-C runs on a fake thread, so a Rust thread base is
/// harmless here (4.5). Named spawns are recorded `(name, stack size)`, and each queue worker's
/// start by its queue's label. Workers are unnamed. Timers wait on the clock it was made with.
#[derive(Clone)]
pub struct FakeThreads {
    inner: Arc<Inner>,
}

struct Inner {
    spawned: Mutex<Vec<(&'static str, usize)>>,
    queues: Mutex<Vec<&'static str>>,
    clock: Option<Arc<dyn TimerClock>>,
}

impl std::fmt::Debug for FakeThreads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeThreads")
            .field("spawned", &*self.inner.spawned.lock())
            .field("queues", &*self.inner.queues.lock())
            .finish_non_exhaustive()
    }
}

impl Default for FakeThreads {
    /// Threads that make no timers.
    fn default() -> Self {
        FakeThreads::on(None)
    }
}

/// A fake process's virtual clock, as a timer's clock.
struct WorldClock(Arc<World>);

impl TimerClock for WorldClock {
    fn now_on(&self, d: crate::platform::os::Deadline) -> u64 {
        TimerClock::now_on(&self.0.clock, d)
    }

    fn wait_until(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, bool>,
        d: crate::platform::os::Deadline,
    ) {
        TimerClock::wait_until(&self.0.clock, cv, g, d)
    }
}

impl FakeThreads {
    fn on(clock: Option<Arc<dyn TimerClock>>) -> FakeThreads {
        FakeThreads {
            inner: Arc::new(Inner {
                spawned: Mutex::new(Vec::new()),
                queues: Mutex::new(Vec::new()),
                clock,
            }),
        }
    }

    /// Threads whose timers wait on `clock`.
    pub fn with_clock(clock: Arc<dyn TimerClock>) -> FakeThreads {
        FakeThreads::on(Some(clock))
    }

    /// Threads whose timers wait on `world`'s virtual clock.
    pub fn for_world(world: Arc<World>) -> FakeThreads {
        FakeThreads::with_clock(Arc::new(WorldClock(world)))
    }

    /// Every named spawn so far.
    pub fn spawned(&self) -> Vec<(&'static str, usize)> {
        self.inner.spawned.lock().clone()
    }

    /// The label of every queue worker started so far, in order.
    pub fn queues(&self) -> Vec<&'static str> {
        self.inner.queues.lock().clone()
    }
}

/// A detached std thread; a refused spawn dies as the system's refusal would end the process.
fn spawn_std(name: Option<&'static str>, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    // The fake's threads carry no Objective-C, so a std thread is allowed (4.5, 17.4). The
    // stack gets headroom: the unoptimised test build needs more than 512 KiB.
    let mut b = std::thread::Builder::new().stack_size(stack_size.max(8 << 20));
    if let Some(name) = name {
        b = b.name(name.to_owned());
    }
    #[allow(clippy::disallowed_methods)] // see above
    let r = b.spawn(body);
    if r.is_err() {
        crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "testkit/threads.rs")
    }
}

impl Threads for FakeThreads {
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        self.inner.spawned.lock().push((name, stack_size));
        spawn_std(Some(name), stack_size, body);
    }

    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        self.inner.queues.lock().push(label);
        spawn_std(None, 0, body);
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(self.clone())
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        self.inner.clock.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Threads are named and recorded
    #[test]
    fn threads() {
        let th = FakeThreads::default();
        let (tx, rx) = mpsc::channel();
        th.spawn(
            "reims-vgpu.test",
            512 * 1024,
            Box::new(move || {
                let _ = tx.send(std::thread::current().name().map(str::to_owned));
            }),
        );
        assert_eq!(rx.recv().unwrap(), Some("reims-vgpu.test".to_string()));
        assert_eq!(th.spawned(), vec![("reims-vgpu.test", 512 * 1024)]);
    }

    /// A queue's worker is recorded by label, not as a named spawn, and is unnamed
    #[test]
    fn queue_workers_are_recorded_by_label() {
        let th = FakeThreads::default();
        let q = crate::rt::queue::SerialQueue::new(&th, "reims-vgpu.test-queue");
        assert_eq!(th.queues(), vec!["reims-vgpu.test-queue"]);
        assert!(th.spawned().is_empty());
        let (tx, rx) = mpsc::channel();
        q.exec_async(move || {
            let _ = tx.send(std::thread::current().name().map(str::to_owned));
        });
        assert_eq!(rx.recv().unwrap(), None);
        assert!(th.spawned().is_empty());
    }
}
