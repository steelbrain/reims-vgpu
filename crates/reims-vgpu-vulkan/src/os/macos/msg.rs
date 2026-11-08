//! The message layer of the macOS side: where a raise inside an Objective-C send is decided,
//! named by its selector, which a death names. The window's AppKit side sends through [`get`] and
//! [`catch`], and IOSurface's creation (`iosurface`) through [`fatal_send`] and [`value`]:
//! - [`get`]: a call whose raise dies at the call (a getter, a property setter, a C function).
//! - [`catch`]: the call's exception handed back as [`Caught`], the caller's to decide.
//! - [`fatal_send`] and [`value`]: a Foundation or IOSurface send, which dies at the send; a send
//!   that only makes or reads a value (a string, a number, a collection, an `+alloc`) goes through
//!   [`value`], and [`string_text`] is its string.
//!
//! The call runs inside objc2's `exception::catch`, so an Objective-C exception stops at the
//! call before any Rust frame of the caller unwinds. The closure owns nothing with a `Drop`
//! while the send runs (D34): it borrows its receiver and arguments, and the `Retained` a creator
//! answers exists only once the method has returned. The closure's own frame lies between the
//! catch and the send, and the foreign unwind runs its landing pads: a value local to the closure,
//! or moved into it, is dropped before `catch` returns (measured, objc2 0.6.4, macOS 26.7).
//!
//! objc2's `exception` feature is on and `catch-all` never, so a raise is caught only where this
//! layer catches it.

use std::panic::AssertUnwindSafe;

use crate::core::raise::Raise;
use crate::core::raise::death::{DeathId, DeathKind};
use objc2::exception::{self, Exception};
use objc2_foundation::{NSException, NSString};

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
fn name_and_reason(exc: Option<&Exception>) -> (String, String) {
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

/// Runs `call`, a send of `sel` whose raise dies at the call (a getter, a property setter, a
/// C function): a death at
/// the send, so its result has no raise.
pub fn get<R>(sel: &'static str, call: impl FnOnce() -> R) -> R {
    catch(sel, call).unwrap_or_else(|c| c.fatal())
}

/// A Foundation or IOSurface send (a getter, a property setter, a configuration): a raise inside
/// `f` dies at the send, named by `sel`, whatever handler is live.
///
/// `f` holds no value with a `Drop` (D34): a foreign unwind runs the landing pads of its frame,
/// and a `Retained` dropped there would be a release the process never makes otherwise.
pub fn fatal_send<R>(sel: &'static str, f: impl FnOnce() -> R) -> R {
    value(sel, f)
}

/// A send that makes or reads a value and has no effect anyone sees (a string, a number, a
/// collection, an error's text, an `+alloc`): a raise inside `f` dies at the send, named by
/// `sel`, whatever handler is live, as at [`fatal_send`].
///
/// `f` holds no value with a `Drop` (D34), as for [`fatal_send`].
pub fn value<R>(sel: &'static str, f: impl FnOnce() -> R) -> R {
    match exception::catch(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(exc) => {
            let (name, reason) = name_and_reason(exc.as_deref());
            let id = DeathId::Platform {
                kind: DeathKind::UncaughtObjc,
                call: sel,
            };
            match exception_detail(&name, &reason) {
                Some(detail) => crate::core::raise::death::fatal_with_detail(id, &detail),
                None => crate::core::raise::fatal(id),
            }
        }
    }
}

/// The text of `s`, copied out (`-UTF8String`, a [`value`] send).
pub fn string_text(s: &NSString) -> String {
    value("UTF8String", || s.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testkit::capture::catch_death;
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;
    use objc2_foundation::{NSArray, NSString};

    fn array() -> Retained<NSArray<NSObject>> {
        NSArray::new()
    }

    /// A getter's raise dies at the call, and its value passes through
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
        assert_eq!(get("count", || 3), 3);
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

    /// `string_text` copies a string's text out
    #[test]
    fn strings_carry_their_text() {
        assert_eq!(
            string_text(&NSString::from_str("reims-vgpu.test")),
            "reims-vgpu.test"
        );
        assert_eq!(string_text(&NSString::from_str("")), "");
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
}
