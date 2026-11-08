//! `spawn_named` (a named thread of core's) and the "our thread" marker the panic hook reads.
//!
//! Core asks the platform's `Threads` for the threads it runs on (D35): named threads here,
//! and each serial queue's worker (`rt::queue`). This wrapper adds what such a thread needs:
//! the "ours" mark the panic hook reads (6.5), the thread's own last-resort holder and pool, and,
//! in tests, the creator's death slot. Its body starts with no handler, and holds under
//! [`Hold::NONE`](crate::rt::Hold::NONE): a hold is not `Send`, so none reaches it.

use std::cell::Cell;
use std::marker::PhantomData;

use crate::platform::os::Threads;

/// A named thread's stack: 576 KiB, a queue worker's
/// [`WORKER_STACK`](crate::rt::queue::WORKER_STACK) and 64 KiB more.
pub const FIFO_THREAD_STACK: usize = crate::rt::queue::WORKER_STACK + 64 * 1024;

thread_local! {
    static OURS: Cell<bool> = const { Cell::new(false) };
}

#[cfg(any(test, feature = "testkit"))]
thread_local! {
    /// A thread core started (no destructor).
    static CORE: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is one core started for its whole life (the FIFO thread, a
/// queue's worker, a timer's thread).
#[cfg(any(test, feature = "testkit"))]
pub(crate) fn is_core_thread() -> bool {
    CORE.with(Cell::get)
}

/// Whether the current thread is ours (a named thread, a queue's worker, or inside an `rvg_`
/// entry point). The panic hook aborts before unwinding on such a thread (6.5).
pub fn is_our_thread() -> bool {
    OURS.with(Cell::get)
}

/// Marks the current thread as ours until the guard drops (every `rvg_` entry point, 6.5).
/// Nesting restores the previous mark. Thread-bound (`!Send`).
#[derive(Debug)]
pub struct OursGuard {
    prev: bool,
    _not_send: PhantomData<*const ()>,
}

/// Enters an "ours" region on this thread.
pub fn enter_ours() -> OursGuard {
    let prev = OURS.with(|o| o.replace(true));
    OursGuard {
        prev,
        _not_send: PhantomData,
    }
}

impl Drop for OursGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        OURS.with(|o| o.set(prev));
    }
}

/// A detached thread named `name` with `stack_size` bytes of stack, from the platform's
/// [`Threads::spawn`], whose body runs marked as ours, with the thread's
/// last-resort holder and real pool. The thread ends in this order: `body` returns, the
/// last-resort holder is released and the pool popped ([`crate::rt::pool::drain_thread_end`]),
/// and then `target` is dropped, on the thread.
pub fn spawn_named(
    threads: &dyn Threads,
    name: &'static str,
    stack_size: usize,
    target: Box<dyn std::any::Any + Send>,
    body: impl FnOnce() + Send + 'static,
) {
    #[cfg(any(test, feature = "testkit"))]
    let slot = crate::testkit::capture::current_slot();
    // The target's release is the thread's too: a death there reaches the test's slot.
    #[cfg(any(test, feature = "testkit"))]
    let target: Box<dyn std::any::Any + Send> = Box::new(InSlot {
        slot: slot.clone(),
        value: Some(target),
    });
    let run = move || {
        crate::rt::pool::drain_idle();
        body();
        crate::rt::pool::drain_thread_end();
    };
    threads.spawn(
        name,
        stack_size,
        Box::new(move || {
            become_ours();
            #[cfg(any(test, feature = "testkit"))]
            crate::testkit::capture::with_slot(slot, run);
            #[cfg(not(any(test, feature = "testkit")))]
            run();
            drop(target);
        }),
    );
}

/// A value dropped with a test's death slot current (`spawn_named`'s target).
#[cfg(any(test, feature = "testkit"))]
struct InSlot {
    slot: Option<std::sync::Arc<crate::testkit::capture::DeathSlot>>,
    value: Option<Box<dyn std::any::Any + Send>>,
}

#[cfg(any(test, feature = "testkit"))]
impl Drop for InSlot {
    fn drop(&mut self) {
        let value = self.value.take();
        crate::testkit::capture::with_slot(self.slot.take(), move || drop(value));
    }
}

/// Marks the current thread, for its whole life, as one core made: ours (6.5) and core's (the
/// testkit's `is_core_thread`). A named thread, a queue's worker and a timer's thread.
pub(crate) fn become_ours() {
    OURS.with(|o| o.set(true));
    #[cfg(any(test, feature = "testkit"))]
    CORE.with(|c| c.set(true));
}

#[cfg(test)]
pub(crate) mod test_threads {
    //! `Threads` over `std::thread` for this module's and its siblings' tests.

    use std::sync::Arc;

    use crate::platform::os::{Threads, TimerClock};

    /// Spawns detached std threads: named ones with the requested stack, and unnamed queue
    /// workers. It makes no timers.
    #[derive(Debug, Default)]
    pub(crate) struct StdThreads;

    impl Threads for StdThreads {
        fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
            #[allow(clippy::disallowed_methods)] // test Threads: no Objective-C runs here (4.5)
            let r = std::thread::Builder::new()
                .name(name.into())
                .stack_size(stack_size.max(8 << 20))
                .spawn(body);
            assert!(r.is_ok());
        }

        fn spawn_worker(&self, _label: &'static str, body: Box<dyn FnOnce() + Send>) {
            #[allow(clippy::disallowed_methods)] // test Threads: no Objective-C runs here (4.5)
            let r = std::thread::Builder::new().stack_size(8 << 20).spawn(body);
            assert!(r.is_ok());
        }

        fn handle(&self) -> Arc<dyn Threads> {
            Arc::new(StdThreads)
        }

        fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_threads::StdThreads;
    use super::*;
    use std::sync::mpsc;

    /// The spawned body is ours and named
    #[test]
    fn spawned_thread_is_ours() {
        let (tx, rx) = mpsc::channel();
        spawn_named(
            &StdThreads,
            "reims-vgpu.fifo",
            FIFO_THREAD_STACK,
            Box::new(()),
            move || {
                let name = std::thread::current().name().map(str::to_owned);
                let _ = tx.send((is_our_thread(), name));
            },
        );
        let got = rx.recv().ok();
        assert_eq!(got, Some((true, Some("reims-vgpu.fifo".to_string()))));
        assert!(!is_our_thread());
    }

    /// Sends its name when dropped.
    struct Tell(&'static str, mpsc::Sender<(&'static str, bool)>);

    impl Drop for Tell {
        fn drop(&mut self) {
            let _ = self.1.send((self.0, is_our_thread()));
        }
    }

    /// The thread's end: the body returns, and only then is the target dropped, on
    /// the thread
    #[test]
    fn the_thread_drops_its_target_after_its_body() {
        let (tx, rx) = mpsc::channel();
        let t2 = tx.clone();
        spawn_named(
            &StdThreads,
            "reims-vgpu.fifo",
            FIFO_THREAD_STACK,
            Box::new(Tell("target", tx)),
            move || {
                let _ = t2.send(("body", is_our_thread()));
            },
        );
        let got: Vec<_> = (0..2).map_while(|_| rx.recv().ok()).collect();
        assert_eq!(got, [("body", true), ("target", true)]);
    }

    /// A death in the target's release lands in the creator's slot
    #[test]
    fn death_in_the_targets_release_is_captured() {
        struct Dies;
        impl Drop for Dies {
            fn drop(&mut self) {
                crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "fifo")
            }
        }
        crate::testkit::capture::expect_death(
            crate::raise::death::DeathSite {
                kind: crate::raise::death::DeathKind::Abort,
                file: "fifo",
                line: 0,
            },
            || {
                spawn_named(
                    &StdThreads,
                    "reims-vgpu.fifo",
                    FIFO_THREAD_STACK,
                    Box::new(Dies),
                    || {},
                );
                loop {
                    std::thread::park();
                }
            },
        );
    }

    /// `enter_ours` nests and restores
    #[test]
    fn ours_guard_nests() {
        assert!(!is_our_thread());
        {
            let _a = enter_ours();
            assert!(is_our_thread());
            {
                let _b = enter_ours();
                assert!(is_our_thread());
            }
            assert!(is_our_thread());
        }
        assert!(!is_our_thread());
    }

    /// A death on a spawned thread lands in the creator's slot
    #[test]
    fn death_on_spawned_thread_is_captured() {
        crate::testkit::capture::expect_death(
            crate::raise::death::DeathSite {
                kind: crate::raise::death::DeathKind::Abort,
                file: "fifo",
                line: 0,
            },
            || {
                spawn_named(
                    &StdThreads,
                    "reims-vgpu.fifo",
                    FIFO_THREAD_STACK,
                    Box::new(()),
                    || crate::testkit::capture::die(crate::raise::death::DeathKind::Abort, "fifo"),
                );
                loop {
                    std::thread::park();
                }
            },
        );
    }
}
