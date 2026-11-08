//! The platform's real autorelease pool, which `rt::pool::Pool` pushes and pops around each unit
//! of work (architecture 7.4).
//!
//! The platform installs it once, at its construction ([`install_native`]): a [`NativePool`] of
//! `objc_autoreleasePoolPush` and `objc_autoreleasePoolPop`. It is process-wide rather than a
//! method of the platform because a pool is pushed where no platform is at hand (a resource's
//! last release, a core thread's last-resort pool). [`Autorelease`] is what `rt::pool::Pool::push`
//! and `rt::pool::around` take. Only what Metal autoreleases goes to it: core's own values are
//! held by its holders (`rt::held`).

use std::ffi::c_void;
use std::sync::OnceLock;

use crate::core::platform::Platform;

/// The platform's real autorelease pool.
#[derive(Debug, Clone, Copy)]
pub struct NativePool {
    /// `objc_autoreleasePoolPush()`: a token for `pop`.
    pub push: fn() -> *mut c_void,
    /// `objc_autoreleasePoolPop(token)`: releases everything autoreleased since `push`.
    pub pop: fn(*mut c_void),
}

static NATIVE: OnceLock<NativePool> = OnceLock::new();

/// Installs the platform's real pool (once per process; later calls are ignored).
pub fn install_native(native: NativePool) {
    let _ = NATIVE.set(native);
}

/// The installed real pool, if any (the testkit installs none).
pub(crate) fn installed() -> Option<NativePool> {
    NATIVE.get().copied()
}

/// The platform's autorelease pools, as `rt::pool::Pool::push` and `rt::pool::around` take them.
pub trait Autorelease {}

impl<P: Platform> Autorelease for P {}
