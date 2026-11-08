//! Values held to a named point: [`Held`], [`Held::raising`], [`Hold`], [`Deferred`], a core
//! thread's last-resort holder and [`release_off_thread`].
//!
//! Some values' last release reaches the host or the guest: a task's `destroyTask` and root-page
//! unmap, a guest display's page unmap, a mapper surface's retain count, a display's end. Such a
//! value is released where the API's order puts it, after the work that used it: a [`Held`] keeps
//! it to its own drop, which releases what it holds most recent first. Every other value drops
//! where Rust's scoping puts it.
//!
//! Every public entry makes a holder on the host's thread, and each unit of work one on its
//! thread; a landing frame ([`Held::raising`]) makes one inside its unit. Code with no
//! handle to its holder (a lookup's reference to the kernel task, a page-table read's load of the
//! device) holds through a [`Hold`] ([`Hold::keep`]), which names the holder of its unit or of
//! the landing frame it runs in.
//!
//! A hold with no holder ([`Hold::NONE`]: a queue's block, a thread's body, a drop) holds to the
//! thread's last-resort holder: a thread core started (a queue's worker, a named thread) keeps
//! one, released once the worker has left its queue ([`crate::core::rt::pool::drain_idle`]) and at the
//! thread's end, just before its last-resort real pool pops. On any other thread (Metal's, a
//! test's) the value drops there.
//!
//! A raise carries values outward too: [`crate::core::raise::Raise::defer`] puts one in the raise's
//! [`Deferred`], and the catch that ends the raise ([`Hold::land`]) hands them to its hold's
//! holder. A landing frame whose body raises puts what it holds in the raise the same way. A raise
//! that dies releases none of them, and one dropped with values still in it is a bug, which builds
//! with debug assertions report.
//!
//! Every value keeps the point it was first held at, and a holder or a raise keeps its values in
//! that order, whatever order they reach it in. So values that travel with a raise land among the
//! catcher's holder's own values at the point the landing frame held them: after what the holder
//! took before the frame, before what it took after the frame returned.
//!
//! A death that unwinds through a holder releases nothing it holds, as a death that ends the
//! process releases nothing: a test's captured death.

use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::raise::{Raise, RaiseResult};

/// A held value.
struct Value {
    /// When it was first held, in the process's order: its place among a holder's values.
    at: u64,
    /// The value itself: its drop is the release.
    _value: Box<dyn Any + Send>,
}

/// The next [`Value::at`]. A raise may end on another thread than the one it was made on, so the
/// order is the process's.
static NEXT_AT: AtomicU64 = AtomicU64::new(0);

impl Value {
    /// `v`, held from now.
    fn new<T: Any + Send>(v: T) -> Value {
        Value {
            at: NEXT_AT.fetch_add(1, Ordering::Relaxed),
            _value: Box::new(v),
        }
    }
}

/// Puts `v` among `values` (oldest first) at the point it was first held: last, unless it
/// travelled with a raise past values held after it.
fn insert(values: &mut Vec<Value>, v: Value) {
    let at = values.partition_point(|e| e.at <= v.at);
    values.insert(at, v);
}

thread_local! {
    /// A core thread's last-resort holder ([`release_last_resort`]).
    static LAST_RESORT: RefCell<Option<Held>> = const { RefCell::new(None) };
}

/// The values held to one point: released, most recent first, when it drops. Thread-bound
/// (`!Send`): it releases on the thread of the unit that made it.
pub struct Held {
    /// Oldest first.
    values: RefCell<Vec<Value>>,
    _not_send: PhantomData<*const ()>,
}

impl Held {
    /// An empty holder.
    pub fn new() -> Held {
        Held {
            values: RefCell::new(Vec::new()),
            _not_send: PhantomData,
        }
    }

    /// Holds `v` to this holder's drop.
    pub fn keep<T: Any + Send>(&self, v: T) {
        self.take(Value::new(v));
    }

    /// Holds `v`, already held elsewhere, at the point it was first held.
    fn take(&self, v: Value) {
        insert(&mut self.values.borrow_mut(), v);
    }

    /// A landing frame: runs `body` with a new holder, which `body` names in the hold its work
    /// runs under ([`Hold::of`]). When `body` returns a value the holder releases what it holds
    /// there, most recent first. When it raises, what the holder holds goes on with
    /// the raise instead, as [`crate::core::raise::Raise::defer`]'s values do, and is released where the
    /// raise ends, after the catcher's report. A death releases none of it.
    pub fn raising<T>(body: impl FnOnce(&Held) -> RaiseResult<T>) -> RaiseResult<T> {
        let held = Held::new();
        match body(&held) {
            Ok(v) => {
                drop(held);
                Ok(v)
            }
            Err(mut r) => {
                let values = std::mem::take(&mut *held.values.borrow_mut());
                r.deferred_mut().carry(values);
                Err(r)
            }
        }
    }
}

impl Default for Held {
    fn default() -> Held {
        Held::new()
    }
}

impl fmt::Debug for Held {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Held")
            .field("values", &self.values.borrow().len())
            .finish()
    }
}

impl Drop for Held {
    /// Releases what it holds, most recent first (module doc).
    fn drop(&mut self) {
        if std::thread::panicking() {
            // A death unwinding (module doc).
            std::mem::forget(std::mem::take(self.values.get_mut()));
            return;
        }
        while let Some(v) = self.values.get_mut().pop() {
            drop(v);
        }
    }
}

/// Which holder a deep hold ([`Hold::keep`]), and what a raise ended under it carries
/// ([`Hold::land`]), go to: a unit's or a landing frame's inside it ([`Hold::of`]), or none
/// ([`Hold::NONE`]), the thread's last-resort holder (module doc).
/// A function takes one only where it holds deep or ends a raise, or passes it to one that does.
/// It decides no raise.
#[derive(Debug, Clone, Copy)]
pub struct Hold<'u> {
    held: Option<&'u Held>,
    /// Thread-bound: it can never reach a queue's block, a thread or a completion handler.
    _not_send: PhantomData<*const ()>,
}

impl Hold<'static> {
    /// No holder: a thread's body, a queue's block, a drop. What it is handed goes to the
    /// thread's last-resort holder.
    pub const NONE: Hold<'static> = Hold {
        held: None,
        _not_send: PhantomData,
    };
}

impl<'u> Hold<'u> {
    /// Into `held`: a unit of work's holder (a job's, an API entry's), or a
    /// landing frame's ([`Held::raising`]).
    pub fn of(held: &'u Held) -> Hold<'u> {
        Hold {
            held: Some(held),
            _not_send: PhantomData,
        }
    }

    /// Ends the raise `r`, a catch's: what it carries goes to this holder, each value at the
    /// point it was first held, to be released at the holder's end, after the catcher's report.
    pub fn land(self, mut r: Raise) {
        r.deferred_mut().land(self.held);
    }

    /// One step of an R frame's cleanup, after its caught body (`body`): a raise from the step
    /// goes on in place of the body's, which ends here ([`Hold::land`]), its values going to this
    /// holder before the frame's locals drop.
    pub fn cleanup_step<T>(
        self,
        body: &mut RaiseResult<()>,
        step: RaiseResult<T>,
    ) -> RaiseResult<T> {
        if step.is_err()
            && let Err(r) = std::mem::replace(body, Ok(()))
        {
            self.land(r);
        }
        step
    }

    /// A deep hold: `v` is kept to the end of this holder, or with none, of the thread's
    /// last-resort holder.
    pub fn keep<T: Any + Send>(self, v: T) {
        match self.held {
            Some(h) => h.keep(v),
            None => keep_last_resort(v),
        }
    }
}

/// Holds `v` to the end of the thread's last-resort holder, else drops it: a deep hold under a
/// hold with no holder (module doc).
pub(crate) fn keep_last_resort<T: Any + Send>(v: T) {
    to_last_resort(Value::new(v));
}

/// [`keep_last_resort`] for a value already held elsewhere, placed where it was first held.
fn to_last_resort(v: Value) {
    // During the thread's exit the holder is gone: the closure, and `v` with it, goes at once.
    let unheld = LAST_RESORT.try_with(move |l| match &*l.borrow() {
        Some(held) => {
            held.take(v);
            None
        }
        None => Some(v),
    });
    if let Ok(Some(v)) = unheld {
        drop(v);
    }
}

/// Releases the thread's last-resort holder, most recent first, and, when `again`, makes the
/// next one (the thread goes on idle; at its end, none).
pub(in crate::core::rt) fn release_last_resort(again: bool) {
    let last = LAST_RESORT.try_with(RefCell::take).ok().flatten();
    drop(last);
    if again {
        let _ = LAST_RESORT.try_with(|l| *l.borrow_mut() = Some(Held::new()));
    }
}

/// The values a raise carries outward ([`crate::core::raise::Raise::defer`], [`Held::raising`]), in the
/// order they were first held, until the catch that ends the raise ([`Hold::land`]) lands them
/// in its hold's holder, among whose own values they take their place, and whose
/// drop releases them most recent first. A death forgets them.
///
/// A raise is shared between threads as any answer is (`Sync`), so the values sit behind a lock;
/// only the raise's owner reaches them, through `&mut`, and never waits on it.
#[derive(Default)]
pub struct Deferred {
    values: parking_lot::Mutex<Vec<Value>>,
}

impl Deferred {
    /// Adds `v`, the most recent.
    pub(crate) fn push<T: Any + Send>(&mut self, v: T) {
        self.values.get_mut().push(Value::new(v));
    }

    /// Adds a landing holder's `values`, each at the point it was first held.
    fn carry(&mut self, values: Vec<Value>) {
        let mine = self.values.get_mut();
        for v in values {
            insert(mine, v);
        }
    }

    /// Hands the values to `held`, each at the point it was first held, or with none to the
    /// thread's last-resort holder: the raise ends.
    pub(crate) fn land(&mut self, held: Option<&Held>) {
        let values = std::mem::take(self.values.get_mut());
        match held {
            Some(held) => {
                for v in values {
                    held.take(v);
                }
            }
            None => {
                for v in values {
                    to_last_resort(v);
                }
            }
        }
    }

    /// Whether it carries nothing.
    pub(crate) fn is_empty(&self) -> bool {
        self.values.lock().is_empty()
    }

    /// Lets go of the values without releasing them: the process ends at a death.
    pub(crate) fn forget(&mut self) {
        std::mem::forget(std::mem::take(self.values.get_mut()));
    }
}

impl fmt::Debug for Deferred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Deferred({})", self.values.lock().len())
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // A death unwinding (module doc).
            self.forget();
            return;
        }
        debug_assert!(
            self.values.get_mut().is_empty(),
            "a raise dropped with the values it carries: a catch ends it with `Hold::land`"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::raise::{OnFault, Raise, RaiseId, RaiseResult};
    use crate::core::rt::pool::test_native::{events, note, with_test_native};
    use crate::core::testkit::capture::catch_death;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    /// Notes its name when released.
    struct Rel(&'static str);
    impl Drop for Rel {
        fn drop(&mut self) {
            note(format!("release {}", self.0));
        }
    }

    fn a_raise() -> Raise {
        Raise::new(RaiseId::Test { name: "held" })
    }

    /// The releases `run` shows.
    fn releases(run: impl FnOnce()) -> Vec<String> {
        let mut seen = Vec::new();
        with_test_native(|| {
            run();
            seen = events();
        });
        seen
    }

    /// A holder releases what it keeps at its drop, most recent first
    #[test]
    fn a_holder_releases_most_recent_first_at_its_drop() {
        let seen = releases(|| {
            let held = Held::new();
            held.keep(Rel("a"));
            held.keep(Rel("b"));
            note("body");
            drop(held);
        });
        assert_eq!(seen, ["body", "release b", "release a"]);
    }

    /// Under a hold with no holder, a deep hold, and what a raise ended there carries, goes
    /// at once on a thread with no last-resort holder (the host's, Metal's, a test's)
    #[test]
    fn with_no_holder_a_deep_hold_drops_at_once() {
        let seen = releases(|| {
            Hold::NONE.keep(Rel("deep"));
            note("held");
            Hold::NONE.land(a_raise().defer(Rel("carried")));
            note("ended");
        });
        assert_eq!(seen, ["release deep", "held", "release carried", "ended"]);
    }

    /// A raise's deferred values land, in order, in the holder they are landed in, which
    /// releases them most recent first after the catcher's own work
    #[test]
    fn deferred_values_land_in_their_holder() {
        let seen = releases(|| {
            let held = Held::new();
            let mut d = Deferred::default();
            d.push(Rel("first"));
            d.push(Rel("second"));
            d.land(Some(&held));
            drop(d);
            note("caught");
            drop(held);
        });
        assert_eq!(seen, ["caught", "release second", "release first"]);
    }

    /// A raise carries its values past the holders it leaves; the catch that ends it lands them
    /// in its hold's holder, released after the catcher's report
    #[test]
    fn a_caught_raise_lands_its_values_where_it_ends() {
        let seen = releases(|| {
            let held = Held::new();
            let hold = Hold::of(&held);
            let r = {
                let inner = Held::new();
                {
                    let _ = Hold::of(&inner);
                    Err::<(), _>(a_raise().defer(Rel("mapping")))
                }
            };
            note("report");
            hold.land(r.unwrap_err());
            note("caught");
            drop(held);
        });
        assert_eq!(seen, ["report", "caught", "release mapping"]);
    }

    /// A raise that dies releases nothing it carries, under a handler or with none
    #[test]
    fn a_raise_that_dies_releases_nothing() {
        let a = std::sync::Arc::new(());
        let (b, c) = (std::sync::Arc::clone(&a), std::sync::Arc::clone(&a));
        let d = catch_death(move || OnFault::Die.throw::<()>(a_raise().defer(b)));
        assert_eq!(d.map_err(|s| s.file), Err("held"));
        let d = catch_death(move || a_raise().defer(c).fatal());
        assert_eq!(d.map_err(|s| s.file), Err("held"));
        assert_eq!(std::sync::Arc::strong_count(&a), 3);
    }

    /// A raise dropped with values still in it is a bug: no catch ended it
    #[test]
    #[should_panic(expected = "a raise dropped with the values it carries")]
    fn a_raise_dropped_with_its_values_is_reported() {
        drop(a_raise().defer(()));
    }

    /// Two raises are equal by identity, detail and origin, whatever they carry; a clone carries
    /// nothing
    #[test]
    fn raises_compare_without_what_they_carry() {
        let a = std::sync::Arc::new(());
        let r = a_raise().defer(std::sync::Arc::clone(&a));
        assert_eq!(r, a_raise());
        assert_ne!(r, a_raise().with_detail("other"));
        let bare = a_raise();
        assert_eq!(bare.clone(), bare);
        Hold::NONE.land(r);
        assert_eq!(std::sync::Arc::strong_count(&a), 1);
    }

    /// A landing frame whose body returns releases what it holds at its end, most recent first,
    /// before the enclosing holder goes on
    #[test]
    fn a_landing_frame_releases_at_its_end_when_its_body_returns() {
        let seen = releases(|| {
            let enclosing = Held::new();
            let hold = Hold::of(&enclosing);
            hold.keep(Rel("a"));
            let r = Held::raising(|held| {
                held.keep(Rel("b"));
                Hold::of(held).keep(Rel("c"));
                Ok(7)
            });
            assert_eq!(r, Ok(7));
            note("landed");
            drop(enclosing);
        });
        assert_eq!(seen, ["release c", "release b", "landed", "release a"]);
    }

    /// A landing frame that raises hands its values on with the raise; they land in the
    /// catcher's holder after what it held before the frame and before what it took after the
    /// frame returned, and so are released after the catcher's report, in that order
    #[test]
    fn a_raising_landing_frame_lands_its_values_at_the_frame_s_place() {
        let seen = releases(|| {
            let enclosing = Held::new();
            let hold = Hold::of(&enclosing);
            hold.keep(Rel("a"));
            let r = {
                let r = Held::raising(|held| {
                    let hold = Hold::of(held);
                    hold.keep(Rel("b"));
                    hold.keep(Rel("c"));
                    Err::<(), _>(a_raise())
                });
                hold.keep(Rel("d"));
                r
            };
            note("report");
            hold.land(r.unwrap_err());
            hold.keep(Rel("e"));
            drop(enclosing);
        });
        assert_eq!(
            seen,
            [
                "report",
                "release e",
                "release d",
                "release c",
                "release b",
                "release a",
            ]
        );
    }

    /// Nested landing frames: the inner one's values sit between what the outer one held before
    /// and after it
    #[test]
    fn nested_landing_frames_keep_the_order_values_were_held_in() {
        let seen = releases(|| {
            let enclosing = Held::new();
            let hold = Hold::of(&enclosing);
            hold.keep(Rel("a"));
            let r = Held::raising(|held| {
                let hold = Hold::of(held);
                hold.keep(Rel("b"));
                let inner = Held::raising(|held| {
                    let hold = Hold::of(held);
                    hold.keep(Rel("c"));
                    Err::<(), _>(a_raise())
                });
                hold.keep(Rel("d"));
                inner
            });
            hold.land(r.unwrap_err());
            note("caught");
            drop(enclosing);
        });
        assert_eq!(
            seen,
            ["caught", "release d", "release c", "release b", "release a"]
        );
    }

    /// A value deferred at the raise takes its place among the landing frame's values at that
    /// point, as a value the frame held there does
    #[test]
    fn a_deferred_value_keeps_its_place_among_the_landing_frame_s() {
        let seen = [true, false].map(|defer| {
            releases(|| {
                let enclosing = Held::new();
                let hold = Hold::of(&enclosing);
                let r = Held::raising(|held| {
                    let hold = Hold::of(held);
                    hold.keep(Rel("b"));
                    let r = if defer {
                        Err::<(), _>(a_raise().defer(Rel("m")))
                    } else {
                        hold.keep(Rel("m"));
                        Err(a_raise())
                    };
                    hold.keep(Rel("d"));
                    r
                });
                hold.land(r.unwrap_err());
                drop(enclosing);
            })
        });
        let want = ["release d", "release m", "release b"];
        assert_eq!(seen, [want, want]);
    }

    /// A landing frame whose raise dies releases nothing it or the raise holds
    #[test]
    fn a_landing_frame_whose_raise_dies_releases_nothing() {
        let a = std::sync::Arc::new(());
        let (b, c) = (std::sync::Arc::clone(&a), std::sync::Arc::clone(&a));
        let d = catch_death(move || {
            Held::raising(|held| -> RaiseResult<()> {
                held.keep(b);
                OnFault::Die.throw(a_raise().defer(c))
            })
        });
        assert_eq!(d.map_err(|s| s.file), Err("held"));
        assert_eq!(std::sync::Arc::strong_count(&a), 3);
    }

    /// A forgotten deferral releases nothing
    #[test]
    fn forgotten_deferred_values_are_never_released() {
        let a = std::sync::Arc::new(());
        let mut d = Deferred::default();
        d.push(std::sync::Arc::clone(&a));
        d.forget();
        drop(d);
        assert_eq!(std::sync::Arc::strong_count(&a), 2);
    }

    /// The last-resort holder takes a deep hold made under a hold with no holder, and
    /// releases it at the drain
    #[test]
    fn the_last_resort_holder_releases_at_the_drain() {
        release_last_resort(true);
        Hold::NONE.keep(Rel("deep"));
        note("held");
        release_last_resort(false);
        Hold::NONE.keep(Rel("after"));
        assert_eq!(events(), ["held", "release deep", "release after"]);
    }

    /// Counts its releases.
    struct Counted(Arc<AtomicU32>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A deep hold goes to the hold's holder, a landing frame's inside it to that one, and with
    /// none on a thread core did not start it goes at once
    #[test]
    fn keep_goes_to_the_holder() {
        let released = Arc::new(AtomicU32::new(0));
        let held = Held::new();
        let frame = Held::new();
        let hold = Hold::of(&held);
        hold.keep(Counted(released.clone()));
        Hold::of(&frame).keep(Counted(released.clone()));
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(frame);
        assert_eq!(released.load(Ordering::SeqCst), 1);
        Hold::NONE.keep(Counted(released.clone()));
        assert_eq!(released.load(Ordering::SeqCst), 2);
        drop(held);
        assert_eq!(released.load(Ordering::SeqCst), 3);
    }

    /// A catch that ends a raise hands what it carries to the hold's holder, released at the
    /// holder's end
    #[test]
    fn land_puts_what_the_raise_carries_in_the_holder() {
        let released = Arc::new(AtomicU32::new(0));
        let held = Held::new();
        let hold = Hold::of(&held);
        let r = OnFault::Return.throw::<()>(a_raise().defer(Counted(released.clone())));
        hold.land(r.unwrap_err());
        hold.land(a_raise().defer(Counted(released.clone())));
        assert_eq!(released.load(Ordering::SeqCst), 0);
        drop(held);
        assert_eq!(released.load(Ordering::SeqCst), 2);
    }

    /// A cleanup step that returns leaves the caught body's raise to go on; one that raises goes
    /// on in its place, and the body's ends there, what it carries released at the holder's end
    #[test]
    fn a_cleanup_raise_ends_the_raise_it_replaces() {
        let released = Arc::new(AtomicU32::new(0));
        let held = Held::new();
        let hold = Hold::of(&held);
        let mut body: RaiseResult<()> = Err(a_raise().defer(Counted(released.clone())));
        assert_eq!(hold.cleanup_step(&mut body, Ok(3)), Ok(3));
        assert_eq!(body.as_ref().map_err(Raise::id), Err(a_raise().id()));
        let step: RaiseResult<u32> = Err(Raise::platform("commit"));
        let r = hold.cleanup_step(&mut body, step).unwrap_err();
        assert_eq!(r, Raise::platform("commit"));
        assert_eq!(body, Ok(()));
        assert_eq!(released.load(Ordering::SeqCst), 0);
        // A body that returned: the step's raise is all there is
        let step: RaiseResult<u32> = Err(Raise::platform("commit"));
        assert_eq!(hold.cleanup_step(&mut body, step), Err(r));
        drop(held);
        assert_eq!(released.load(Ordering::SeqCst), 1);
    }
}
