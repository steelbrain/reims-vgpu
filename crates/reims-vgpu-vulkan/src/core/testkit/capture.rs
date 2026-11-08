//! In-process capture of deaths and hangs: `DeathSlot`, `catch_death`, `expect_death`,
//! `expect_hang`, and a test's own raises and deaths ([`raise`], [`foreign`], [`die`]). Only under
//! `cfg(any(test, feature = "testkit"))`.
//!
//! Under capture, `raise::death::fatal` records its site in the thread's current [`DeathSlot`]
//! and panics with a [`Death`] payload, so the `Drop`s a real death never runs do run: the
//! subprocess harness (`testkit::death`) is the one that shows what a death leaves behind. A slot follows the work: `rt::spawn_named` hands the creator's slot
//! to the new thread, and every `SerialQueue` / `CoalescingSignal` job carries its submitter's
//! slot, so a death on any thread that works for a test lands in that test's slot. Fakes that
//! run callbacks on their own threads propagate it with [`current_slot`] and [`with_slot`].

#![allow(clippy::disallowed_methods)] // catch_unwind and thread spawns are this module's job
#![allow(clippy::panic)] // a test harness reports failure by panicking (17.4: test-only code)

use std::any::Any;
use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::core::raise::death::{DeathKind, DeathSite};
use crate::core::raise::{OnFault, Raise, RaiseId, RaiseResult};

/// The panic payload a captured death unwinds with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Death(pub DeathSite);

static CAPTURE: AtomicBool = AtomicBool::new(true);

/// Whether a death panics with a `Death` (true, the default) or aborts.
pub fn capture_enabled() -> bool {
    CAPTURE.load(Ordering::Relaxed)
}

/// Turns capture on or off for the whole process.
pub fn set_capture_enabled(on: bool) {
    CAPTURE.store(on, Ordering::Relaxed);
}

/// Where the first death of one test's "process" is recorded: the slot follows the process's
/// work onto every thread, so a death on a named thread or in a queue's block lands in it.
#[derive(Default, Debug)]
pub struct DeathSlot {
    site: Mutex<Option<DeathSite>>,
    cv: Condvar,
}

impl DeathSlot {
    /// An empty slot.
    pub fn new() -> Arc<DeathSlot> {
        Arc::new(DeathSlot::default())
    }

    /// The first death recorded, if any.
    pub fn death(&self) -> Option<DeathSite> {
        *self.site.lock()
    }

    fn record(&self, site: DeathSite) {
        let mut g = self.site.lock();
        // The process ends at its first death; later ones are threads still running in a test.
        if g.is_none() {
            *g = Some(site);
        }
        self.cv.notify_all();
    }
}

thread_local! {
    static SLOT: RefCell<Option<Arc<DeathSlot>>> = const { RefCell::new(None) };
}

/// This thread's slot.
pub fn current_slot() -> Option<Arc<DeathSlot>> {
    SLOT.with(|s| s.borrow().clone())
}

/// Runs `f` with `slot` as this thread's slot, then restores the previous one.
pub fn with_slot<R>(slot: Option<Arc<DeathSlot>>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<DeathSlot>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            SLOT.with(|s| *s.borrow_mut() = prev);
        }
    }
    let prev = SLOT.with(|s| std::mem::replace(&mut *s.borrow_mut(), slot));
    let _restore = Restore(prev);
    f()
}

fn install_quiet_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if info.payload().is::<Death>() {
                return;
            }
            prev(info)
        }));
    });
}

/// A death under capture: record, then unwind with the payload.
pub(crate) fn capture(site: DeathSite) -> ! {
    if let Some(slot) = current_slot() {
        slot.record(site);
    }
    install_quiet_hook();
    std::panic::panic_any(Death(site))
}

fn death_of(payload: &(dyn Any + Send)) -> Option<DeathSite> {
    payload.downcast_ref::<Death>().map(|d| d.0)
}

/// Runs `f` on this thread and returns its value, or the site where it died. Other panics
/// propagate.
pub fn catch_death<R>(f: impl FnOnce() -> R) -> Result<R, DeathSite> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(p) => match death_of(p.as_ref()) {
            Some(site) => Err(site),
            None => resume_unwind(p),
        },
    }
}

/// A test's own raise at the site `name` with `detail`, decided by `on` as a raise in a function
/// reached both ways is: [`OnFault::Die`] dies here (`UnhandledRaise` at `name`), and
/// [`OnFault::Return`] answers the raise.
#[track_caller]
pub fn raise(on: OnFault, name: &'static str, detail: impl std::fmt::Display) -> Raise {
    returned(on.throw(Raise::new(RaiseId::Test { name }).with_detail(detail)))
}

/// What a Metal send at `sel` raises, with `detail`, decided by `on`: [`OnFault::Die`] dies at
/// `sel`, and [`OnFault::Return`] answers the raise.
#[track_caller]
pub fn foreign(on: OnFault, sel: &'static str, detail: impl std::fmt::Display) -> Raise {
    returned(on.throw(Raise::foreign(sel).with_detail(detail)))
}

/// The raise a throw that returned answers.
fn returned(thrown: RaiseResult<std::convert::Infallible>) -> Raise {
    match thrown {
        Err(r) => r,
        Ok(never) => match never {},
    }
}

/// A test's own death of `kind` at the site `name`.
pub fn die(kind: DeathKind, name: &'static str) -> ! {
    crate::core::raise::death::fatal(crate::core::raise::death::DeathId::Test { kind, name })
}

/// How long `expect_death` waits, in real time, for `f` to die or return.
pub const DEATH_TIMEOUT: Duration = Duration::from_secs(30);

enum Outcome {
    Returned,
    Panicked(Box<dyn Any + Send>),
}

struct Run {
    slot: Arc<DeathSlot>,
    outcome: Arc<Mutex<Option<Outcome>>>,
}

fn run_detached(f: impl FnOnce() + Send + 'static) -> Run {
    let slot = DeathSlot::new();
    let outcome: Arc<Mutex<Option<Outcome>>> = Arc::new(Mutex::new(None));
    let (s2, o2) = (slot.clone(), outcome.clone());
    let spawned = std::thread::Builder::new()
        .name("capture".into())
        .spawn(move || {
            let r = with_slot(Some(s2.clone()), || catch_unwind(AssertUnwindSafe(f)));
            let out = match r {
                Ok(()) => Outcome::Returned,
                Err(p) if death_of(p.as_ref()).is_some() => return, // recorded in the slot
                Err(p) => Outcome::Panicked(p),
            };
            *o2.lock() = Some(out);
            // Wake the waiter; it rechecks the outcome under the slot's lock.
            let _g = s2.site.lock();
            s2.cv.notify_all();
        });
    if let Err(e) = spawned {
        panic!("expect_death: cannot spawn: {e}");
    }
    Run { slot, outcome }
}

/// Passes when `site` fires on any thread working for `f` before `f` returns, or while it is
/// blocked. Fails if `f` returns first, if another site fires, or if nothing happens within
/// [`DEATH_TIMEOUT`]. `f` runs on its own thread, which is leaked if it blocks.
pub fn expect_death(site: DeathSite, f: impl FnOnce() + Send + 'static) {
    let run = run_detached(f);
    let deadline = Instant::now().checked_add(DEATH_TIMEOUT);
    let mut g = run.slot.site.lock();
    loop {
        if let Some(got) = *g {
            assert_eq!(got, site, "expect_death: died at another site");
            return;
        }
        if let Some(out) = run.outcome.lock().take() {
            match out {
                Outcome::Returned => panic!("expect_death: returned without dying at {site:?}"),
                Outcome::Panicked(p) => {
                    drop(g);
                    resume_unwind(p)
                }
            }
        }
        let timed_out = match deadline {
            Some(at) => run.slot.cv.wait_until(&mut g, at).timed_out(),
            None => {
                run.slot.cv.wait(&mut g);
                false
            }
        };
        if timed_out && g.is_none() {
            panic!("expect_death: no death at {site:?} within {DEATH_TIMEOUT:?}");
        }
    }
}

/// Virtual time a test can advance: implemented by the fake's `VirtualClock`.
pub trait VirtualTime: Send + Sync {
    /// Advances virtual time by `d`, waking every wait whose deadline passes.
    fn advance(&self, d: Duration);
}

/// Real time allowed for threads to settle before and after advancing virtual time.
pub const HANG_SETTLE: Duration = Duration::from_millis(100);

/// Passes when `f` is still blocked after virtual time `d` has passed: it neither returned
/// nor died. `f` runs on its own thread, which is leaked.
pub fn expect_hang(clock: &dyn VirtualTime, d: Duration, f: impl FnOnce() + Send + 'static) {
    let run = run_detached(f);
    let check = |when: &str| {
        std::thread::sleep(HANG_SETTLE);
        if let Some(site) = run.slot.death() {
            panic!("expect_hang: died at {site:?} {when}");
        }
        match run.outcome.lock().take() {
            Some(Outcome::Returned) => panic!("expect_hang: returned {when}"),
            Some(Outcome::Panicked(p)) => resume_unwind(p),
            None => {}
        }
    };
    check("before time advanced");
    clock.advance(d);
    check("after time advanced");
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "site_t",
        line: 0,
    };

    /// A death on the calling thread
    #[test]
    fn expect_death_same_thread() {
        expect_death(SITE, || die(DeathKind::Abort, "site_t"));
    }

    /// A death on a thread that inherited the slot, while `f` blocks
    #[test]
    fn expect_death_other_thread_while_blocked() {
        expect_death(SITE, || {
            let slot = current_slot();
            let _ = std::thread::spawn(move || with_slot(slot, || die(DeathKind::Abort, "site_t")));
            loop {
                std::thread::park();
            }
        });
    }

    /// Returning without the death fails
    #[test]
    #[should_panic(expected = "returned without dying")]
    fn expect_death_fails_on_return() {
        expect_death(SITE, || {});
    }

    /// Another site fails
    #[test]
    #[should_panic(expected = "another site")]
    fn expect_death_fails_on_other_site() {
        expect_death(SITE, || die(DeathKind::Abort, "site_u"));
    }

    /// A non-death panic propagates
    #[test]
    #[should_panic(expected = "boom")]
    fn expect_death_propagates_panic() {
        expect_death(SITE, || panic!("boom"));
    }

    /// catch_death returns the value or the site
    #[test]
    fn catch_death_both_ways() {
        assert_eq!(catch_death(|| 3), Ok(3));
        assert_eq!(
            catch_death(|| die(DeathKind::Abort, "site_t")),
            Err::<(), _>(SITE)
        );
    }

    /// The first death wins
    #[test]
    fn first_death_wins() {
        let slot = DeathSlot::new();
        with_slot(Some(slot.clone()), || {
            let _ = catch_death(|| die(DeathKind::Abort, "site_t"));
            let _ = catch_death(|| die(DeathKind::Assert, "site_v"));
        });
        assert_eq!(slot.death(), Some(SITE));
        assert!(current_slot().is_none());
    }

    struct NoClock;
    impl VirtualTime for NoClock {
        fn advance(&self, _d: Duration) {}
    }

    /// A blocked closure hangs; a returning one does not
    #[test]
    fn expect_hang_blocked() {
        expect_hang(&NoClock, Duration::from_secs(2), || {
            loop {
                std::thread::park();
            }
        });
    }

    /// Returning is not a hang
    #[test]
    #[should_panic(expected = "returned")]
    fn expect_hang_fails_on_return() {
        expect_hang(&NoClock, Duration::from_secs(2), || {});
    }

    /// Dying is not a hang
    #[test]
    #[should_panic(expected = "died")]
    fn expect_hang_fails_on_death() {
        expect_hang(&NoClock, Duration::from_secs(2), || {
            die(DeathKind::Abort, "site_t")
        });
    }
}
