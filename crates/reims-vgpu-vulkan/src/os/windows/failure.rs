//! Unrecoverable OS ownership failures use core's existing fatal boundary.

use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};

/// A broken mapping invariant cannot unwind through an FFI call or continue with
/// stale aliases. Match the Unix platform's fatal path, naming the failed call.
#[cold]
pub(super) fn abort(call: &'static str) -> ! {
    fatal(DeathId::Platform {
        kind: DeathKind::Abort,
        call,
    })
}

/// Preserve the OS error before entering the same non-unwinding fatal boundary.
#[cold]
pub(super) fn failed(error: impl std::fmt::Debug, call: &'static str) -> ! {
    log::error!("{call}: {error:?}");
    abort(call)
}
