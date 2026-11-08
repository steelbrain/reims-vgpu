//! Fake: `Script`: queued results per call kind (architecture 13.2).
//!
//! A test queues a [`Step`] for a selector (optionally for one receiver, optionally after letting
//! some calls pass); the fake consults the script at every call whose position is observable, the
//! calls it logs, and otherwise runs its model. A free Metal call (a getter, a query, a setter no
//! other property aliases) is never scripted: a test changes what it answers through the model's
//! setters. The steps are the outcomes a platform call can have:
//! - `Nil` / `Error`: Metal returned nil or an `NSError` (the expected channel);
//! - `Value`: a scripted return value (`false`, a count, a struct) in place of the model's;
//! - `Raise`: Metal raised an `NSException`, a raise at the selector (`Raise::foreign`), decided
//!   by the send's [`Policy`], the label the platform's answer layer gives the same send:
//!   `Err(Raise)`, or a death at the send;
//! - `Die`: Metal, IOGPU or the driver crashed, which ends the process whatever thread it is on:
//!   a death marker;
//! - `Hold`: the call takes as long as the test makes it, on the thread that made it (a driver's
//!   cold pipeline make inside a draw, say), and then the model answers ([`HeldCall`]).

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use crate::sync::{Condvar, Mutex};

use crate::platform::ObjId;
use crate::platform::types::NsErrorText;
use crate::raise::death::{DeathId, DeathKind, DeathSite, fatal};
use crate::raise::{OnFault, Raise, RaiseResult};

/// What a scripted raise at a send does: the label the platform's answer layer gives the same send
/// (the Vulkan crate's `answer::Policy`), so that the fake decides each raise as the real platform
/// does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    /// A send no path reaches without a handler: the raise is `Err`, the caller's to catch.
    Caught,
    /// A send no path reaches under a handler, or one whose raise dies whatever handler is live:
    /// the raise dies at the send.
    Fatal,
    /// A send in a method reached both ways: the caller's [`OnFault`] decides.
    On(OnFault),
}

impl Policy {
    /// The raise `r` at a send of this label: `Err`, or a death at the send.
    fn decide<T>(self, r: Raise) -> RaiseResult<T> {
        match self {
            Policy::Caught => Err(r),
            Policy::Fatal => r.fatal(),
            Policy::On(on) => on.throw(r),
        }
    }
}

/// One scripted outcome.
#[derive(Clone)]
pub enum Step {
    /// No script: the fake's model answers.
    Default,
    /// Metal returned nil (a creator), or a getter's zero.
    Nil,
    /// A scripted return value, downcast to the method's return type; a value of another type is
    /// ignored and the model answers. [`Step::value`] builds one.
    Value(Arc<dyn Any + Send + Sync>),
    /// Metal returned nil with this `NSError` (`newLibraryWithData:error:` and friends); a
    /// method without an error out-parameter treats it as `Nil`.
    Error(NsErrorText),
    /// A creator the platform refuses for a gap of its own, where Metal makes the object: noted
    /// as a relayed gap ([`crate::platform::device::note_relayed_gap`]), then as `Error`, nil
    /// with this `NSError` where the creator has an error out-parameter, else `Nil`.
    Gap(NsErrorText),
    /// Metal raised `name` with `reason` at the send (6.3).
    Raise {
        /// The exception's name (`NSInvalidArgumentException`, …).
        name: String,
        /// The exception's reason.
        reason: String,
    },
    /// A crash inside Metal, IOGPU or the driver (class b). `None` is the fake's marker site,
    /// [`metal_crash_site`]; `Some` names another.
    Die(Option<DeathSite>),
    /// The call is held, with nothing of the fake's locked, until the test lets it go; then the
    /// model answers.
    Hold(HeldCall),
}

/// A call the fake holds on the thread that made it until the test lets it go ([`Step::Hold`]).
/// `Clone` shares it.
#[derive(Clone, Default)]
pub struct HeldCall {
    /// Whether a call is held, and whether the test let it go; and their moves.
    state: Arc<(Mutex<(bool, bool)>, Condvar)>,
}

impl std::fmt::Debug for HeldCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (reached, open) = *self.state.0.lock();
        f.debug_struct("HeldCall")
            .field("reached", &reached)
            .field("open", &open)
            .finish()
    }
}

impl HeldCall {
    /// A hold no call has reached yet.
    pub fn new() -> HeldCall {
        HeldCall::default()
    }

    /// Waits, `timeout` at most, for a call to be held; whether one is or was.
    pub fn reached(&self, timeout: Duration) -> bool {
        let (lock, moved) = &*self.state;
        let mut s = lock.lock();
        if !s.0 {
            let _ = moved.wait_while_for(&mut s, |s| !s.0, timeout);
        }
        s.0
    }

    /// Lets the held call go on, and any later call this hold reaches.
    pub fn open(&self) {
        let (lock, moved) = &*self.state;
        lock.lock().1 = true;
        moved.notify_all();
    }

    /// Holds the calling thread until the test lets the call go.
    pub(crate) fn hold(&self) {
        let (lock, moved) = &*self.state;
        let mut s = lock.lock();
        s.0 = true;
        moved.notify_all();
        while !s.1 {
            moved.wait(&mut s);
        }
    }
}

impl std::fmt::Debug for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::Default => f.write_str("Default"),
            Step::Nil => f.write_str("Nil"),
            Step::Value(_) => f.write_str("Value(..)"),
            Step::Error(e) => f.debug_tuple("Error").field(e).finish(),
            Step::Gap(e) => f.debug_tuple("Gap").field(e).finish(),
            Step::Raise { name, reason } => f
                .debug_struct("Raise")
                .field("name", name)
                .field("reason", reason)
                .finish(),
            Step::Die(s) => f.debug_tuple("Die").field(s).finish(),
            Step::Hold(h) => f.debug_tuple("Hold").field(h).finish(),
        }
    }
}

/// The death site of a scripted class-b crash at `sel`: the fake's marker, since the real crash
/// happens inside Metal. `DeathKind::Abort` stands for Metal's abort or fault.
pub const fn metal_crash_site(sel: &'static str) -> DeathSite {
    DeathSite {
        kind: DeathKind::Abort,
        file: sel,
        line: 0,
    }
}

impl Step {
    /// A scripted return value.
    pub fn value<T: Any + Send + Sync>(v: T) -> Step {
        Step::Value(Arc::new(v))
    }

    /// A raise of `name` with `reason`.
    pub fn raise(name: &str, reason: &str) -> Step {
        Step::Raise {
            name: name.to_owned(),
            reason: reason.to_owned(),
        }
    }

    /// An `NSInvalidArgumentException` for an unrecognized selector (an SPI the device lacks).
    pub fn unrecognized(sel: &str) -> Step {
        Step::raise(
            "NSInvalidArgumentException",
            &format!("-[FakeObject {sel}]: unrecognized selector sent to instance"),
        )
    }

    /// A class-b crash at the fake's marker site.
    pub fn crash() -> Step {
        Step::Die(None)
    }

    /// The scripted value, if this is a `Value` of type `T`.
    pub fn get<T: Clone + 'static>(&self) -> Option<T> {
        match self {
            Step::Value(v) => v.downcast_ref::<T>().cloned(),
            _ => None,
        }
    }

    /// Whether the call should return nil.
    pub fn is_nil(&self) -> bool {
        matches!(self, Step::Nil | Step::Error(_) | Step::Gap(_))
    }

    /// The death this step scripts: a crash inside Metal is the platform's, at the marker
    /// site ([`metal_crash_site`]); a named site is the code's own.
    fn death(&self, sel: &'static str) -> Option<DeathId> {
        match self {
            Step::Die(None) => Some(DeathId::Platform {
                kind: DeathKind::Abort,
                call: sel,
            }),
            Step::Die(Some(site)) => Some(DeathId::from(*site)),
            _ => None,
        }
    }

    fn raised(&self, sel: &'static str) -> Option<Raise> {
        match self {
            Step::Raise { reason, .. } => Some(Raise::foreign(sel).with_detail(reason)),
            _ => None,
        }
    }

    /// Applies the death and raise outcomes of a method that returns `RaiseResult<T>`: a death dies
    /// here, and a raise is decided by `policy`, the send's label.
    pub fn throws(&self, policy: Policy, sel: &'static str) -> RaiseResult<()> {
        if let Some(site) = self.death(sel) {
            fatal(site)
        }
        match self.raised(sel) {
            Some(r) => policy.decide(r),
            None => Ok(()),
        }
    }

    /// Applies the death and raise outcomes of a zero-argument getter (`send_get`, 4.6): a raise
    /// there is a death at the send, never a `Raise`.
    pub fn getter(&self, sel: &'static str) {
        if let Some(site) = self.death(sel) {
            fatal(site)
        }
        if self.raised(sel).is_some() {
            fatal(DeathId::Platform {
                kind: DeathKind::UncaughtObjc,
                call: sel,
            })
        }
    }

    /// A `RaiseResult<T>` result: the scripted value, else `model()`.
    pub fn throws_value<T: Clone + 'static>(
        &self,
        policy: Policy,
        sel: &'static str,
        model: impl FnOnce() -> T,
    ) -> RaiseResult<T> {
        self.throws(policy, sel)?;
        Ok(self.get::<T>().unwrap_or_else(model))
    }

    /// A getter's result: the scripted value, else `model()`.
    pub fn getter_value<T: Clone + 'static>(
        &self,
        sel: &'static str,
        model: impl FnOnce() -> T,
    ) -> T {
        self.getter(sel);
        self.get::<T>().unwrap_or_else(model)
    }

    /// A creator's result (`RaiseResult<Option<T>>`): nil when scripted, else `make()`.
    pub fn create<T>(
        &self,
        policy: Policy,
        sel: &'static str,
        make: impl FnOnce() -> Option<T>,
    ) -> RaiseResult<Option<T>> {
        self.throws(policy, sel)?;
        if let Step::Gap(_) = self {
            crate::platform::device::note_relayed_gap();
        }
        if self.is_nil() {
            return Ok(None);
        }
        Ok(make())
    }

    /// A creator with an `NSError **` (`RaiseResult<Result<Option<T>, NsErrorText>>`): the scripted
    /// error, nil, or `make()`.
    pub fn create_or_error<T>(
        &self,
        policy: Policy,
        sel: &'static str,
        make: impl FnOnce() -> Option<T>,
    ) -> RaiseResult<Result<Option<T>, NsErrorText>> {
        self.throws(policy, sel)?;
        match self {
            Step::Error(e) => Ok(Err(e.clone())),
            Step::Gap(e) => {
                crate::platform::device::note_relayed_gap();
                Ok(Err(e.clone()))
            }
            Step::Nil => Ok(Ok(None)),
            _ => Ok(Ok(make())),
        }
    }
}

/// What an entry matches: a selector string (any receiver class), or a selector sent to one
/// receiver class ([`super::faults::Faults`]).
#[derive(Debug, PartialEq)]
pub(super) enum Key {
    Sel(String),
    Call(&'static str, &'static str),
}

#[derive(Debug)]
struct Entry {
    key: Key,
    obj: Option<ObjId>,
    skip: u64,
    /// `None`: forever.
    remaining: Option<u64>,
    step: Step,
}

/// The queued outcomes of one fake world. `Clone` shares the script.
#[derive(Debug, Clone, Default)]
pub struct Script {
    entries: Arc<Mutex<Vec<Entry>>>,
}

impl Script {
    /// An empty script: every call runs the model.
    pub fn new() -> Script {
        Script::default()
    }

    fn add(&self, sel: &str, obj: Option<ObjId>, skip: u64, remaining: Option<u64>, step: Step) {
        self.add_key(Key::Sel(sel.to_owned()), obj, skip, remaining, step);
    }

    pub(super) fn add_key(
        &self,
        key: Key,
        obj: Option<ObjId>,
        skip: u64,
        remaining: Option<u64>,
        step: Step,
    ) {
        self.entries.lock().push(Entry {
            key,
            obj,
            skip,
            remaining,
            step,
        });
    }

    /// The next call of `sel` (any receiver) gets `step`.
    pub fn push(&self, sel: &str, step: Step) {
        self.add(sel, None, 0, Some(1), step);
    }

    /// The next `n` calls of `sel` get `step`.
    pub fn push_n(&self, sel: &str, n: u64, step: Step) {
        self.add(sel, None, 0, Some(n), step);
    }

    /// Every call of `sel` gets `step` (until [`Script::clear`]).
    pub fn always(&self, sel: &str, step: Step) {
        self.add(sel, None, 0, None, step);
    }

    /// The next call of `sel` on `obj` gets `step`.
    pub fn push_on(&self, obj: ObjId, sel: &str, step: Step) {
        self.add(sel, Some(obj), 0, Some(1), step);
    }

    /// Every call of `sel` on `obj` gets `step`.
    pub fn always_on(&self, obj: ObjId, sel: &str, step: Step) {
        self.add(sel, Some(obj), 0, None, step);
    }

    /// Lets `skip` calls of `sel` run the model, then the next one gets `step`: the death or
    /// fault at call K (13.2).
    pub fn after(&self, sel: &str, skip: u64, step: Step) {
        self.add(sel, None, skip, Some(1), step);
    }

    /// Forgets every entry.
    pub fn clear(&self) {
        self.entries.lock().clear();
    }

    /// How many one-shot entries are still unconsumed (a test's check that its script ran).
    pub fn pending(&self) -> usize {
        self.entries
            .lock()
            .iter()
            .filter(|e| e.remaining.is_some())
            .count()
    }

    /// The step for this call of `sel` on `obj`: the first matching entry decides. An entry that
    /// still has calls to let pass consumes one of them and yields `Default`. An entry for one
    /// receiver class matches only through [`Script::next_for`], which knows the class.
    pub fn next(&self, obj: ObjId, sel: &str) -> Step {
        self.next_for(obj, "", sel)
    }

    /// [`Script::next`] for a call of `sel` on an object (or class) of `class`.
    pub fn next_for(&self, obj: ObjId, class: &str, sel: &str) -> Step {
        let mut g = self.entries.lock();
        let pos = g.iter().position(|e| {
            e.obj.is_none_or(|o| o == obj)
                && match &e.key {
                    Key::Sel(s) => s == sel,
                    Key::Call(c, s) => *c == class && *s == sel,
                }
        });
        let Some(pos) = pos else {
            return Step::Default;
        };
        let Some(e) = g.get_mut(pos) else {
            return Step::Default;
        };
        if e.skip > 0 {
            e.skip = e.skip.wrapping_sub(1);
            return Step::Default;
        }
        let step = e.step.clone();
        if let Some(n) = e.remaining.as_mut() {
            *n = n.saturating_sub(1);
            if *n == 0 {
                let _ = g.remove(pos);
            }
        }
        drop(g);
        if let Step::Hold(held) = &step {
            held.hold();
            return Step::Default;
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::capture::catch_death;

    /// One-shot, counted and permanent entries
    #[test]
    fn push_push_n_always() {
        let s = Script::new();
        s.push("a", Step::Nil);
        s.push_n("b", 2, Step::value(5u64));
        s.always("c", Step::value(true));
        assert!(matches!(s.next(ObjId(1), "a"), Step::Nil));
        assert!(matches!(s.next(ObjId(1), "a"), Step::Default));
        assert_eq!(s.next(ObjId(1), "b").get::<u64>(), Some(5));
        assert_eq!(s.next(ObjId(1), "b").get::<u64>(), Some(5));
        assert!(matches!(s.next(ObjId(1), "b"), Step::Default));
        for _ in 0..3 {
            assert_eq!(s.next(ObjId(1), "c").get::<bool>(), Some(true));
        }
        assert_eq!(s.pending(), 0);
        s.clear();
        assert!(matches!(s.next(ObjId(1), "c"), Step::Default));
    }

    /// Receiver filters and call-K skipping
    #[test]
    fn on_and_after() {
        let s = Script::new();
        s.push_on(ObjId(2), "x", Step::Nil);
        assert!(matches!(s.next(ObjId(1), "x"), Step::Default));
        assert!(matches!(s.next(ObjId(2), "x"), Step::Nil));
        s.always_on(ObjId(3), "y", Step::value(1u32));
        assert_eq!(s.next(ObjId(3), "y").get::<u32>(), Some(1));
        assert!(s.next(ObjId(4), "y").get::<u32>().is_none());
        s.after("z", 2, Step::Nil);
        assert!(matches!(s.next(ObjId(1), "z"), Step::Default));
        assert!(matches!(s.next(ObjId(1), "z"), Step::Default));
        assert_eq!(s.pending(), 1);
        assert!(matches!(s.next(ObjId(1), "z"), Step::Nil));
        assert_eq!(s.pending(), 0);
    }

    /// A value of the wrong type falls back to the model
    #[test]
    fn value_types() {
        let v = Step::value(3u64);
        assert_eq!(v.get::<u64>(), Some(3));
        assert_eq!(v.get::<u32>(), None);
        assert_eq!(v.getter_value("g", || 9u32), 9);
        assert_eq!(v.getter_value("g", || 9u64), 3);
        assert_eq!(
            Step::Default.throws_value(Policy::Caught, "g", || 4u8),
            Ok(4)
        );
    }

    /// Nil and error for plain creators and creators with an NSError
    #[test]
    fn creators() {
        assert_eq!(Step::Nil.create(Policy::Caught, "n", || Some(1)), Ok(None));
        assert_eq!(
            Step::Default.create(Policy::Caught, "n", || Some(1)),
            Ok(Some(1))
        );
        let e = NsErrorText {
            domain: "MTLLibraryErrorDomain".into(),
            code: 1,
            description: "bad".into(),
        };
        assert_eq!(
            Step::Error(e.clone()).create(Policy::Caught, "n", || Some(1)),
            Ok(None)
        );
        assert_eq!(
            Step::Error(e.clone()).create_or_error(Policy::Caught, "n", || Some(1)),
            Ok(Err(e))
        );
        assert_eq!(
            Step::Nil.create_or_error(Policy::Caught, "n", || Some(1)),
            Ok(Ok(None))
        );
        assert_eq!(
            Step::Default.create_or_error(Policy::Caught, "n", || Some(1)),
            Ok(Ok(Some(1)))
        );
    }

    /// A raise at a `Caught` send is a Raise with the selector as origin, and at an `On(Return)`
    /// send too
    #[test]
    fn a_caught_raise_is_thrown() {
        let r = Step::raise("NSInvalidArgumentException", "bad")
            .throws(Policy::Caught, "setLineWidth:");
        let t = r.unwrap_err();
        assert_eq!(t.origin(), crate::raise::Origin::Boundary("setLineWidth:"));
        let u = Step::unrecognized("setLineWidth:").create::<u8>(
            Policy::On(OnFault::Return),
            "setLineWidth:",
            || None,
        );
        assert!(u.is_err());
    }

    /// A raise at a `Fatal` or `On(Die)` send, and a raise in a logged getter, die at the send
    #[test]
    fn a_fatal_raise_dies_at_the_send() {
        let commit = Err(DeathSite {
            kind: DeathKind::UncaughtObjc,
            file: "commit",
            line: 0,
        });
        let site = catch_death(|| {
            let _ = Step::raise("N", "r").throws(Policy::Fatal, "commit");
        });
        assert_eq!(site, commit);
        let site = catch_death(|| {
            let _ = Step::raise("N", "r").throws(Policy::On(OnFault::Die), "commit");
        });
        assert_eq!(site, commit);
        let site = catch_death(|| Step::raise("N", "r").getter("iosurface"));
        assert_eq!(site.map_err(|s| s.kind), Err(DeathKind::UncaughtObjc));
    }

    /// A scripted crash dies at the marker, or at a named site
    #[test]
    fn die_steps() {
        let site = catch_death(|| {
            let _ =
                Step::crash().throws(Policy::Caught, "newTextureWithDescriptor:iosurface:plane:");
        });
        assert_eq!(
            site,
            Err(metal_crash_site(
                "newTextureWithDescriptor:iosurface:plane:"
            ))
        );
        let other = DeathSite {
            kind: DeathKind::NullDeref,
            file: "site_x",
            line: 0,
        };
        let site = catch_death(|| Step::Die(Some(other)).getter("iosurface"));
        assert_eq!(site, Err(other));
        assert!(format!("{:?}", Step::crash()).starts_with("Die"));
    }
}
