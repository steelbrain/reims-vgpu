//! The platform's real autorelease pool around a unit of work: [`Pool`] and [`around`].
//!
//! A `Pool` pushes the real pool when it is made and pops it when it drops, so what Metal
//! autoreleases inside it (its `+0` returns: an uncommitted command buffer, its encoders) is freed
//! there. Without it an uncommitted buffer lives on, and the queue's 65th `-commandBuffer` blocks.
//!
//! A raise out of [`around`] leaves the pool pushed, as clang's `@autoreleasepool` does (its pop is
//! a normal cleanup only): the enclosing pop, which pops every pool pushed inside it, frees what
//! Metal autoreleased there after the catcher's report. With no real pool (the testkit) a `Pool`
//! does nothing.
//!
//! The threads core starts (a queue's worker, a named thread) also keep a last-resort pool, for
//! what is autoreleased on them outside any unit: [`drain_idle`] pops it and pushes the next once
//! the worker has left its queue, and [`drain_thread_end`] pops it at the thread's end. The
//! thread's last-resort holder ([`crate::rt::held`]) is released first at both points.

use std::cell::Cell;
use std::ffi::c_void;

use crate::platform::os::Autorelease;
use crate::platform::os::autorelease::NativePool;
use crate::raise::RaiseResult;

#[cfg(test)]
thread_local! {
    static TEST_NATIVE: Cell<Option<NativePool>> = const { Cell::new(None) };
}

/// The thread's real pool: a test's recording one (the tests' `test_native`), else the platform's.
pub(crate) fn native() -> Option<NativePool> {
    #[cfg(test)]
    if let Some(n) = TEST_NATIVE.with(Cell::get) {
        return Some(n);
    }
    crate::platform::os::autorelease::installed()
}

/// One real pool, pushed while it lives. Thread-bound (`!Send`: the pool is the thread's).
#[derive(Debug)]
pub struct Pool {
    /// The pushed pool (`objc_autoreleasePoolPush`'s token); `None` on a platform with no real
    /// pool, and once a raise has left it pushed.
    token: Option<*mut c_void>,
}

impl Pool {
    /// Pushes the platform's real pool.
    pub fn push(_os: &impl Autorelease) -> Pool {
        Pool::here()
    }

    /// [`Pool::push`], for a site with no platform at hand (a resource's last release, an API
    /// entry).
    pub fn here() -> Pool {
        Pool {
            token: native().map(|n| (n.push)()),
        }
    }

    /// Leaves the pool pushed, for the enclosing pop to take (a raise).
    fn leave_pushed(mut self) {
        self.token = None;
    }
}

impl Drop for Pool {
    /// Pops the pool. A death unwinding (a test's captured one) leaves it
    /// pushed, as a death that ends the process does.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if let (Some(token), Some(n)) = (self.token.take(), native()) {
            (n.pop)(token);
        }
    }
}

/// Runs `f` inside a real pool: popped when `f` returns `Ok`, left pushed when it returns `Err`
/// (module doc).
pub fn around<T>(_os: &impl Autorelease, f: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
    around_here(f)
}

/// [`around`], for a site with no platform at hand (a resource's paging walk).
pub(crate) fn around_here<T>(f: impl FnOnce() -> RaiseResult<T>) -> RaiseResult<T> {
    let pool = Pool::here();
    let r = f();
    if r.is_err() {
        pool.leave_pushed();
    } else {
        drop(pool);
    }
    r
}

thread_local! {
    /// A core thread's last-resort real pool ([`drain_idle`]); no destructor.
    static LAST_RESORT: Cell<Option<*mut c_void>> = const { Cell::new(None) };
}

/// A thread core started (a queue's worker, a named thread), outside any unit: releases the
/// thread's last-resort holder (what was held under a hold with no holder) and makes the next,
/// then pops its last-resort real pool, releasing what Metal autoreleased on it outside any unit,
/// and pushes the next. A worker calls it once it has left its queue (as libdispatch's pool
/// around a worker's drain), a named thread at its start.
pub(crate) fn drain_idle() {
    crate::rt::held::release_last_resort(true);
    let Some(n) = native() else { return };
    if let Some(token) = LAST_RESORT.with(Cell::take) {
        (n.pop)(token);
    }
    let next = (n.push)();
    LAST_RESORT.with(|l| l.set(Some(next)));
}

/// A core thread's end, outside any unit: releases its last-resort holder, then pops its
/// last-resort real pool.
pub(crate) fn drain_thread_end() {
    crate::rt::held::release_last_resort(false);
    if let (Some(token), Some(n)) = (LAST_RESORT.with(Cell::take), native()) {
        (n.pop)(token);
    }
}

/// A recording real pool for other modules' tests: `push 0x…` and `pop 0x…` events, in order,
/// on the current thread.
#[cfg(test)]
pub(crate) mod test_native {
    use super::TEST_NATIVE;
    use crate::platform::os::autorelease::NativePool;
    use std::cell::{Cell, RefCell};
    use std::ffi::c_void;

    thread_local! {
        static EVENTS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
        static NEXT_TOKEN: Cell<usize> = const { Cell::new(0x100) };
    }

    fn t_push() -> *mut c_void {
        let t = NEXT_TOKEN.with(|n| {
            let t = n.get();
            n.set(t.wrapping_add(1));
            t
        });
        EVENTS.with(|e| e.borrow_mut().push(format!("push {t:#x}")));
        std::ptr::without_provenance_mut(t)
    }

    fn t_pop(t: *mut c_void) {
        EVENTS.with(|e| e.borrow_mut().push(format!("pop {:#x}", t.addr())));
    }

    /// Records a note among the pool events (a test's own marker, e.g. a Metal call).
    pub(crate) fn note(s: impl Into<String>) {
        EVENTS.with(|e| e.borrow_mut().push(s.into()));
    }

    /// Runs `f` with the recording pool installed as this thread's real pool. Tokens restart at
    /// 0x100 on each call.
    pub(crate) fn with_test_native(f: impl FnOnce()) {
        NEXT_TOKEN.with(|n| n.set(0x100));
        let _ = events();
        TEST_NATIVE.with(|n| {
            n.set(Some(NativePool {
                push: t_push,
                pop: t_pop,
            }))
        });
        f();
        TEST_NATIVE.with(|n| n.set(None));
    }

    /// The events recorded since the last call.
    pub(crate) fn events() -> Vec<String> {
        EVENTS.with(|e| std::mem::take(&mut *e.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use super::test_native::{events, note, with_test_native};
    use super::*;
    use crate::raise::{Raise, RaiseId};
    use crate::rt::Hold;
    use crate::testkit::FakePlatform;

    /// A pool pops at its drop, after the work inside it
    #[test]
    fn a_pool_pops_at_its_drop() {
        with_test_native(|| {
            let p = FakePlatform::new();
            let pool = Pool::push(&p);
            note("body");
            drop(pool);
            assert_eq!(events(), ["push 0x100", "body", "pop 0x100"]);
        });
    }

    /// `around` pops its pool when the work returns, and leaves it pushed on a raise: the
    /// enclosing pop takes it
    #[test]
    fn around_pops_on_ok_and_leaves_the_pool_pushed_on_a_raise() {
        with_test_native(|| {
            let p = FakePlatform::new();
            assert_eq!(around(&p, || Ok(7)), Ok(7));
            let outer = Pool::push(&p);
            let r: RaiseResult<()> = around(&p, || Err(Raise::new(RaiseId::Test { name: "p" })));
            assert!(r.is_err());
            note("caught");
            drop(outer);
            assert_eq!(
                events(),
                [
                    "push 0x100",
                    "pop 0x100",
                    "push 0x101",
                    "push 0x102",
                    "caught",
                    "pop 0x101"
                ]
            );
        });
    }

    /// With no real pool (the testkit) a pool does nothing
    #[test]
    fn with_no_real_pool_a_pool_does_nothing() {
        let p = FakePlatform::new();
        drop(Pool::push(&p));
        assert_eq!(around(&p, || Ok(1)), Ok(1));
        assert!(events().is_empty());
    }

    /// Notes its name when released.
    struct Rel(&'static str);
    impl Drop for Rel {
        fn drop(&mut self) {
            note(format!("release {}", self.0));
        }
    }

    /// A core thread's drain releases what was held there under a hold with no holder, then
    /// pops its last-resort pool; its end does the same and pushes none. With no real pool (the
    /// testkit) the last-resort holder is kept all the same
    #[test]
    fn a_drain_releases_the_last_resort_holder_before_the_pool_pops() {
        #[allow(clippy::disallowed_methods)] // a fresh thread: no last resort yet, as a core one
        std::thread::spawn(|| {
            with_test_native(|| {
                drain_idle();
                Hold::NONE.keep(Rel("first"));
                drain_idle();
                Hold::NONE.keep(Rel("second"));
                drain_thread_end();
                assert_eq!(
                    events(),
                    [
                        "push 0x100",
                        "release first",
                        "pop 0x100",
                        "push 0x101",
                        "release second",
                        "pop 0x101"
                    ]
                );
            });
            drain_idle();
            Hold::NONE.keep(Rel("no pool"));
            note("held");
            drain_thread_end();
            assert_eq!(events(), ["held", "release no pool"]);
        })
        .join()
        .unwrap();
    }
}
