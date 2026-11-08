//! The Objective-C runtime's autorelease pool, which core's `rt::pool::Pool` pushes and pops
//! around each unit of work (D22): what MoltenVK, IOSurface and Foundation autorelease inside one
//! is freed at its end. Core's own values never go to it: those a release point holds are in
//! core's holders (`rt::held`).

use std::ffi::c_void;

use crate::core::platform::os::autorelease::{NativePool, install_native};

/// `objc_autoreleasePoolPush()`.
fn push() -> *mut c_void {
    // SAFETY: no preconditions; the token is popped on this thread by `pop`.
    unsafe { objc2::ffi::objc_autoreleasePoolPush() }
}

/// `objc_autoreleasePoolPop(token)`.
fn pop(token: *mut c_void) {
    // SAFETY: `token` came from `push` on this thread and has not been popped (core pops a
    // token once, and never one an outer pop already took).
    unsafe { objc2::ffi::objc_autoreleasePoolPop(token) }
}

/// Installs the real pool for core's pools (once per process).
pub fn install() {
    install_native(NativePool { push, pop });
}
