//! The catch on macOS: whatever Objective-C object the host throws, and `@throw nil`, ends the
//! process at the call.

use objc2::exception::{self, Exception};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, NSObject};
use objc2_foundation::{NSException, NSString};
use reims_vgpu_core::testkit::capture::catch_death;

use super::*;

/// `@throw object` from the host's code.
fn throw(object: Retained<AnyObject>) -> ! {
    // SAFETY: any Objective-C object may be thrown; `Exception` wraps one.
    exception::throw(unsafe { Retained::cast_unchecked::<Exception>(object) })
}

fn ns_exception() -> Retained<AnyObject> {
    let name = NSString::from_str("HostRefusal");
    let reason = NSString::from_str("the host's reason");
    // SAFETY: a name and a reason, no userInfo.
    let e = unsafe { NSException::exceptionWithName_reason_userInfo(&name, Some(&reason), None) };
    Retained::into_super(Retained::into_super(e))
}

/// Whatever the host throws ends the process at the call
#[test]
fn a_host_raise_is_fatal_at_the_call() {
    let d = catch_death(|| host_call("raise_interrupt", || throw(ns_exception())));
    assert_eq!(d.unwrap_err(), host_raised("raise_interrupt"));
    let d = catch_death(|| host_call("cursor_move", || throw(NSObject::new().into_super())));
    assert_eq!(d.unwrap_err(), host_raised("cursor_move"));
}

unsafe extern "C-unwind" {
    /// `@throw nil` (objc2's `throw` takes an object).
    fn objc_exception_throw(exception: *mut AnyObject) -> !;
}

/// `@throw nil` is fatal too
#[test]
fn a_nil_throw_is_fatal() {
    // SAFETY: objc_exception_throw(nil).
    let d = catch_death(|| {
        host_call("cursor_show", || unsafe {
            objc_exception_throw(std::ptr::null_mut())
        })
    });
    assert_eq!(d.unwrap_err(), host_raised("cursor_show"));
}
