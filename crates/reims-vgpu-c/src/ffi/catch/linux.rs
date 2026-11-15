//! Linux, and Windows: std's `catch_unwind`. A Rust panic that unwinds this far (one the library's panic hook
//! did not end first, as `resume_unwind` does not call it) stops at this frame. A foreign
//! exception, a C++ throw through the host's `extern "C-unwind"` function, cannot be caught by
//! Rust: std ends the process where it reaches this frame. Either way the process ends at the
//! call.

use std::panic::AssertUnwindSafe;

/// Runs `f` inside `catch_unwind`: its value, or `Err(())` for a Rust unwind out of it.
#[expect(
    clippy::disallowed_methods,
    reason = "the C API's catch around a host call on Linux and Windows, which ends the process (6.7)"
)]
pub(crate) fn catch<R>(f: impl FnOnce() -> R) -> Result<R, ()> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(drop)
}
