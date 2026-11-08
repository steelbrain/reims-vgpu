//! The message layer: where a raise inside Metal is decided.
//!
//! Metal is called through objc2-metal's typed methods, and through the typed declarations in
//! [`crate::spi`] for what objc2-metal does not declare (private selectors, and the few whose
//! objc2-metal signature cannot carry a value core sends: a nil where the header says nonnull, a
//! NULL pointer, an `error:` out-parameter left nil). Every call core asks for runs here, named
//! by its selector (a Metal C function by its name with `()`), which a raise and a death name:
//! - [`send`]: the call's [`Policy`], its label, decides a raise inside it. A `Caught` call
//!   answers `Err(Raise::platform(sel))`, the caller's to catch; a `Fatal` call dies at the call;
//!   an `On` call does one or the other as its caller's [`OnFault`] says.
//! - [`get`]: a call whose raise dies at the call (a getter, a property setter, a Metal C
//!   function).
//! - [`catch`]: the call's exception handed back as [`Caught`], the caller's to decide.
//!
//! The call runs inside objc2's `exception::catch`, so an Objective-C exception stops at the
//! call before any Rust frame of the caller unwinds. The closure owns nothing with a `Drop`
//! while Metal runs (D34): it borrows its receiver and arguments, and the `Retained` a creator
//! answers exists only once the method has returned. The closure's own frame lies between the
//! catch and the send, and the foreign unwind runs its landing pads: a value local to the closure,
//! or moved into it, is dropped before `catch` returns (measured, objc2 0.6.4, macOS 26.7).
//!
//! **Foundation and IOSurface sends** go through [`fatal_send`], which catches a raise in one
//! and dies at the send, named by its selector. A send that only makes or reads a value (a
//! string, a number, a collection, an error's text, an `+alloc`) goes through [`value`], which
//! does the same; [`new_string`] and [`string_text`] are its two strings.
//!
//! So no Objective-C exception leaves a call the crate makes: each is caught where it starts, and
//! no frame of the caller unwinds. `tests/message_layer.rs` keeps every call inside these.
//!
//! **objc2.** The crate builds on objc2-metal, with objc2's `exception` feature on and `catch-all`
//! never, so a raise is caught only where this layer catches it. objc2's own checks (a nil
//! receiver, a missing method, an encoding the runtime disagrees with) panic before the send in dev
//! builds and are off in the `checked` and release profiles (D36). So every declaration carries the
//! runtime's types (`crate::spi`, checked by `tests/spi_encodings.rs`), and the crate sends no
//! selector the host's Metal does not implement: core raises the render records'
//! unrecognized-selector fault itself, with no send.

pub mod encode;

use std::panic::AssertUnwindSafe;

use objc2::exception::{self, Exception};
use objc2::rc::Retained;
use objc2_foundation::{NSException, NSString};
use reims_vgpu_core::raise::death::{DeathId, DeathKind};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};

/// What a raise inside Metal at a call does: the call's label, which decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    /// A call no path reaches without a handler: the raise is `Err`, the caller's to catch.
    Caught,
    /// A call whose raise dies whatever handler is live: the raise dies at the send. A call no
    /// path reaches under a handler goes through [`get`] instead, whose result has no raise.
    Fatal,
    /// A call in a platform method reached both ways, whose caller says by the [`OnFault`] which
    /// way this call is: `Die` dies at the send, `Return` is the caller's raise.
    On(OnFault),
}

/// An Objective-C exception raised inside a call, caught at the call before any Rust frame
/// unwound. Its name and reason are diagnostics: they reach logs, never a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Caught {
    /// The selector whose call raised.
    pub sel: &'static str,
    /// `-[NSException name]`, or the thrown object's class name.
    pub name: String,
    /// `-[NSException reason]` (empty for nil).
    pub reason: String,
}

impl Caught {
    /// Reads a caught exception: its `-name` and `-reason` when it is an `NSException` (empty
    /// strings for nil), otherwise the object's class name and an empty reason. `@throw nil`
    /// gives an empty name. The exception itself is not kept.
    pub fn read(sel: &'static str, exc: Option<&Exception>) -> Caught {
        let (name, reason) = name_and_reason(exc);
        Caught { sel, name, reason }
    }

    /// The raise this is.
    pub fn raise(&self) -> Raise {
        Raise::platform(self.sel).with_detail(&self.reason)
    }

    /// The exception decided by `policy`, the call's label ([`Policy`]): `Err` with
    /// [`Caught::raise`], the caller's to catch, or a death here ([`Caught::fatal`]).
    pub fn decide<T>(self, policy: Policy) -> RaiseResult<T> {
        let on = match policy {
            Policy::Caught => OnFault::Return,
            Policy::Fatal => OnFault::Die,
            Policy::On(on) => on,
        };
        match on {
            OnFault::Die => self.fatal(),
            OnFault::Return => Err(self.raise()),
        }
    }

    /// Dies with the raise, here, at the selector. The death's line names the exception.
    pub fn fatal(self) -> ! {
        let raise = Raise::platform(self.sel);
        match exception_detail(&self.name, &self.reason) {
            Some(detail) => raise.with_detail(detail).fatal(),
            None => raise.fatal(),
        }
    }
}

/// `-name` and `-reason` of a caught exception, as [`Caught::read`] describes. Each read runs
/// inside a catch of its own: a read that raises in turn gives an empty string (an object whose
/// `-isKindOfClass:` raises, its class name), and the caller goes on with what was read.
pub(crate) fn name_and_reason(exc: Option<&Exception>) -> (String, String) {
    /// `f`, a read of the exception; `None` when it raises.
    fn read<R>(f: impl FnOnce() -> R) -> Option<R> {
        exception::catch(AssertUnwindSafe(f)).ok()
    }
    /// The text of `s`, or empty when reading it raises.
    fn text(s: &NSString) -> String {
        read(|| s.to_string()).unwrap_or_default()
    }
    let Some(exc) = exc else {
        return (String::new(), String::new());
    };
    match read(|| (**exc).downcast_ref::<NSException>()).flatten() {
        Some(ns) => (
            read(|| ns.name()).map_or_else(String::new, |n| text(&n)),
            read(|| ns.reason())
                .flatten()
                .map_or_else(String::new, |r| text(&r)),
        ),
        None => (
            (**exc).class().name().to_string_lossy().into_owned(),
            String::new(),
        ),
    }
}

/// A caught exception as a death's detail: `exception <name>: <reason>`, leaving out what is
/// empty, or `None` when both are.
fn exception_detail(name: &str, reason: &str) -> Option<String> {
    match (name.is_empty(), reason.is_empty()) {
        (true, true) => None,
        (false, true) => Some(format!("exception {name}")),
        (true, false) => Some(format!("exception: {reason}")),
        (false, false) => Some(format!("exception {name}: {reason}")),
    }
}

/// Runs `call`, a send of `sel`, and hands an exception inside it back to the caller, who
/// decides it.
pub fn catch<R>(sel: &'static str, call: impl FnOnce() -> R) -> Result<R, Caught> {
    exception::catch(AssertUnwindSafe(call)).map_err(|exc| Caught::read(sel, exc.as_deref()))
}

/// Runs `call`, a send of `sel`: `policy`, the call's label, decides a raise inside it
/// ([`Caught::decide`]).
pub fn send<R>(sel: &'static str, policy: Policy, call: impl FnOnce() -> R) -> RaiseResult<R> {
    catch(sel, call).or_else(|c| c.decide(policy))
}

/// Runs `call`, a send of `sel` whose raise dies at the call (a getter, a property setter, a
/// Metal C function, or a send of a platform method no path reaches under a handler): [`send`]
/// with [`Policy::Fatal`], whose result has no raise.
pub fn get<R>(sel: &'static str, call: impl FnOnce() -> R) -> R {
    catch(sel, call).unwrap_or_else(|c| c.fatal())
}

/// A Foundation or IOSurface send (a getter, a property setter, a configuration): a raise inside
/// `f` dies at the send, named by `sel`, whatever handler is live.
///
/// `f` holds no value with a `Drop` (D34): a foreign unwind runs the landing pads of its frame,
/// and a `Retained` dropped there would be a release the process never makes otherwise.
pub(crate) fn fatal_send<R>(sel: &'static str, f: impl FnOnce() -> R) -> R {
    value(sel, f)
}

/// A send that makes or reads a value and has no effect anyone sees (a string, a number, a
/// collection, an error's text, an `+alloc`): a raise inside `f` dies at the send, named by
/// `sel`, whatever handler is live, as at [`fatal_send`].
///
/// `f` holds no value with a `Drop` (D34), as for [`fatal_send`].
pub(crate) fn value<R>(sel: &'static str, f: impl FnOnce() -> R) -> R {
    match exception::catch(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(exc) => {
            let (name, reason) = name_and_reason(exc.as_deref());
            let id = DeathId::Platform {
                kind: DeathKind::UncaughtObjc,
                call: sel,
            };
            match exception_detail(&name, &reason) {
                Some(detail) => reims_vgpu_core::raise::death::fatal_with_detail(id, &detail),
                None => reims_vgpu_core::raise::fatal(id),
            }
        }
    }
}

/// A new `NSString` of `s` (`-initWithBytes:length:encoding:`, a [`value`] send).
pub(crate) fn new_string(s: &str) -> Retained<NSString> {
    value("initWithBytes:length:encoding:", || NSString::from_str(s))
}

/// The text of `s`, copied out (`-UTF8String`, a [`value`] send).
pub(crate) fn string_text(s: &NSString) -> String {
    value("UTF8String", || s.to_string())
}

/// A constructor whose contract has no nil (`+new`, the address-range array) returned nil. Metal
/// never does on the reference host; were it to, the process dies at the call as for an uncaught
/// exception there, rather than invent a handle.
pub fn nil_result(sel: &'static str) -> ! {
    Caught {
        sel,
        name: "nil".to_owned(),
        reason: "constructor returned nil".to_owned(),
    }
    .fatal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;
    use objc2_foundation::{NSArray, NSString};
    use reims_vgpu_core::raise::death::DeathSite;
    use reims_vgpu_core::raise::{Origin, RaiseId};
    use reims_vgpu_core::testkit::capture::catch_death;

    fn array() -> Retained<NSArray<NSObject>> {
        NSArray::new()
    }

    /// `-objectAtIndex:` past the end of an empty array raises an `NSRangeException`: run as
    /// the render pipeline's colour-attachment lookup, labelled `policy`.
    fn out_of_range(policy: Policy, a: &NSArray<NSObject>) -> RaiseResult<Retained<NSObject>> {
        send("objectAtIndexedSubscript:", policy, || a.objectAtIndex(5))
    }

    /// A call returns the method's value
    #[test]
    fn send_returns_the_value() {
        let text = NSString::from_str("four");
        assert_eq!(get("length", || text.length()), 4);
        assert_eq!(send("setLabel:", Policy::Fatal, || 7).unwrap(), 7);
    }

    /// A `Caught` call's raise is `Raise::platform(sel)`, the caller's, with the exception's
    /// reason as its detail
    #[test]
    fn a_caught_raise_is_the_callers() {
        let a = array();
        let r = out_of_range(Policy::Caught, &a).unwrap_err();
        assert_eq!(r.id(), RaiseId::Platform);
        assert_eq!(r.origin(), Origin::Platform("objectAtIndexedSubscript:"));
        assert!(!r.detail().unwrap_or_default().is_empty(), "{r}");
    }

    /// A `Fatal` call's raise dies at the call: `UncaughtObjc` at the selector
    #[test]
    fn a_fatal_raise_dies_at_the_send() {
        let a = array();
        let site = catch_death(|| out_of_range(Policy::Fatal, &a)).unwrap_err();
        assert_eq!(
            site,
            DeathSite {
                kind: DeathKind::UncaughtObjc,
                file: "objectAtIndexedSubscript:",
                line: 0
            }
        );
    }

    /// `Caught::decide` decides an exception caught at a call by its label, as `send` does
    #[test]
    fn decide_decides_by_the_label() {
        let a = array();
        let r = catch("objectAtIndexedSubscript:", || a.objectAtIndex(5))
            .or_else(|c| c.decide(Policy::Caught))
            .unwrap_err();
        assert_eq!(r.origin(), Origin::Platform("objectAtIndexedSubscript:"));
        let site = catch_death(|| {
            catch("objectAtIndexedSubscript:", || a.objectAtIndex(5))
                .or_else(|c| c.decide(Policy::Fatal))
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::UncaughtObjc, "objectAtIndexedSubscript:")
        );
    }

    /// An `On` call's raise is the caller's with `Return` and dies at the call with `Die`
    #[test]
    fn on_calls_decide_by_on() {
        let a = array();
        let r = send(
            "objectAtIndexedSubscript:",
            Policy::On(OnFault::Return),
            || a.objectAtIndex(5),
        )
        .unwrap_err();
        assert_eq!(r.origin(), Origin::Platform("objectAtIndexedSubscript:"));
        let site = catch_death(|| {
            send(
                "objectAtIndexedSubscript:",
                Policy::On(OnFault::Die),
                || a.objectAtIndex(5),
            )
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::UncaughtObjc, "objectAtIndexedSubscript:")
        );
    }

    /// A getter's raise dies at the call, as a `Fatal` send's does
    #[test]
    fn a_get_raise_dies_at_the_send() {
        let a = array();
        let site = catch_death(|| {
            get("setWidth:", || {
                let _ = a.objectAtIndex(5);
            });
        })
        .unwrap_err();
        assert_eq!(site.kind, DeathKind::UncaughtObjc);
        assert_eq!((site.file, site.line), ("setWidth:", 0));
        let site = catch_death(|| {
            let _ = send("setWidth:", Policy::Fatal, || a.objectAtIndex(5));
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::UncaughtObjc, "setWidth:")
        );
    }

    /// A value send passes its value through and dies at the send
    #[test]
    fn a_value_send_dies_at_the_send() {
        assert_eq!(value("count", || 3), 3);
        let a = array();
        let site = catch_death(|| {
            let _ = value("objectAtIndex:", || a.objectAtIndex(5));
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::UncaughtObjc, "objectAtIndex:", 0)
        );
    }

    /// `new_string` and `string_text` carry text both ways
    #[test]
    fn strings_carry_their_text() {
        let s = new_string("reims-vgpu.fifo");
        assert_eq!(s.length(), 15);
        assert_eq!(string_text(&s), "reims-vgpu.fifo");
        assert_eq!(string_text(&new_string("")), "");
    }

    objc2::define_class!(
        /// An `NSException` whose `-name` raises in turn.
        #[unsafe(super(NSException))]
        #[name = "RVGTestRaisingNameException"]
        struct RaisingName;

        impl RaisingName {
            #[unsafe(method(name))]
            fn raising_name(&self) -> *mut NSString {
                // SAFETY: any object may be thrown; `Exception` wraps one.
                objc2::exception::throw(unsafe {
                    Retained::cast_unchecked::<Exception>(NSObject::new())
                })
            }
        }
    );

    /// A read of the caught exception that raises in turn reads as empty; the other reads go on
    #[test]
    fn a_raising_read_reads_as_empty() {
        use objc2::AnyThread as _;
        let name = NSString::from_str("Ignored");
        let reason = NSString::from_str("the reason");
        // SAFETY: NSException's initializer, with a name and a reason and no userInfo, on a
        // fresh allocation of the subclass.
        let e: Retained<RaisingName> = unsafe {
            objc2::msg_send![
                RaisingName::alloc(),
                initWithName: &*name,
                reason: &*reason,
                userInfo: std::ptr::null::<objc2::runtime::AnyObject>()
            ]
        };
        // SAFETY: an NSException is an Objective-C object, which `Exception` wraps.
        let exc = unsafe { Retained::cast_unchecked::<Exception>(e) };
        let c = Caught::read("length", Some(&exc));
        assert_eq!((c.name.as_str(), c.reason.as_str()), ("", "the reason"));
    }

    /// `catch` hands the exception back unjudged
    #[test]
    fn catch_hands_the_exception_back() {
        let a = array();
        let c = catch("length", || a.objectAtIndex(5)).unwrap_err();
        assert_eq!(c.sel, "length");
        assert_eq!(c.name, "NSRangeException");
    }

    /// fatal_send dies at the send, and passes a value through
    #[test]
    fn fatal_send_dies_at_the_send() {
        let a = array();
        let site = catch_death(|| {
            let _ = fatal_send("objectAtIndex:", || a.objectAtIndex(5));
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::UncaughtObjc, "objectAtIndex:", 0)
        );
        assert!(fatal_send("isProxy", || true));
    }

    /// A raise inside a Metal C function call dies there
    #[test]
    fn a_function_raise_is_fatal() {
        assert!(get("MTLCopyDeviceForRegistryID()", || true));
        let a = array();
        let site = catch_death(|| {
            get("MTLCopyDeviceForRegistryID()", || {
                let _ = a.objectAtIndex(5);
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "MTLCopyDeviceForRegistryID()");
    }

    /// A death's detail names the caught exception, leaving out what is empty
    #[test]
    fn the_death_detail_names_the_exception() {
        assert_eq!(
            exception_detail("NSRangeException", "index 5 beyond bounds").as_deref(),
            Some("exception NSRangeException: index 5 beyond bounds")
        );
        assert_eq!(
            exception_detail("NSRangeException", "").as_deref(),
            Some("exception NSRangeException")
        );
        assert_eq!(
            exception_detail("", "why").as_deref(),
            Some("exception: why")
        );
        assert_eq!(exception_detail("", ""), None);
    }

    /// `Caught::read` handles nil and an object that is not an `NSException`
    #[test]
    fn read_handles_any_thrown_object() {
        let c = Caught::read("length", None);
        assert_eq!((c.name.as_str(), c.reason.as_str()), ("", ""));
        let s = NSString::from_str("not an exception");
        // SAFETY: an Exception is any object; this one is an NSString.
        let e: &Exception = unsafe { &*(Retained::as_ptr(&s).cast::<Exception>()) };
        let c = Caught::read("length", Some(e));
        assert!(c.name.contains("String"), "{c:?}");
        assert_eq!(c.reason, "");
    }

    /// A constructor's nil dies at the call (`UncaughtObjc`, the selector)
    #[test]
    fn nil_result_dies_at_the_send() {
        let site = catch_death(|| nil_result("new")).unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::UncaughtObjc, "new", 0)
        );
    }
}
