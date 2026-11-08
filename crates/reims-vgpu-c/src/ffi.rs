//! The boundary with the host, both ways: [`export`] around the body of every `rvg_` function,
//! and [`host_call`] around every call of a host function.
//!
//! **Into the library.** Every `rvg_` function is `extern "C"`: nothing unwinds out of it into
//! the host. Its body runs inside [`export`], which first installs the library's panic hook and
//! then marks the thread as the library's for the call, so a panic anywhere under an export ends
//! the process in the hook, before any frame unwinds and before any drop runs (architecture 6.5).
//! The `extern "C"` frame is only the backstop: an unwind that reached it would end the process
//! too, but only after the frames in between had run their drops, a table's `destroy(ctx)` among
//! them. `tests/exports.rs` checks that every export has this shape.
//!
//! **Out to the host.** Every call of a host function runs inside [`host_call`], the innermost
//! frame around the function pointer: whatever the host throws ends the process at the call. The
//! catch is the OS's ([`catch`]): objc2's `exception::catch` on macOS, std's `catch_unwind` on
//! Linux. The host's function-pointer types stay `extern "C-unwind"`, since both are sound only
//! around an ABI that unwinds.

use reims_vgpu_vulkan::core::raise::death::{DeathId, fatal};
use reims_vgpu_vulkan::core::rt::enter_ours;

mod catch;

/// Runs the body of one export: the panic hook installed (once per process), then `body` with the
/// thread marked as the library's until it returns (module doc). An export reached inside another
/// (a host callback that calls back in) nests, and restores the mark it found.
pub(crate) fn export<R>(body: impl FnOnce() -> R) -> R {
    // A thread already unwinding (a Rust host's drop during its own panic) cannot install a hook,
    // and a panic of ours there is a double panic, which aborts anyway.
    if !std::thread::panicking() {
        reims_vgpu::singleton::install_panic_hook();
    }
    let _ours = enter_ours();
    body()
}

/// Calls the host's code through `f`, and answers what it returns. Anything the host throws ends
/// the process here, at `call`, whatever handlers of the library's are live: the call runs inside
/// the OS's [`catch`], so the throw stops at this frame and no frame of the library unwinds. What
/// the catch stops is `DeathId::HostRaised`. QEMU and the suite never raise from a callback.
///
/// `f` must own no value with a `Drop`: a foreign unwind runs its frame's landing pads.
pub(crate) fn host_call<R>(call: &'static str, f: impl FnOnce() -> R) -> R {
    match catch::catch(f) {
        Ok(r) => r,
        Err(()) => fatal(DeathId::HostRaised { call }),
    }
}

#[cfg(test)]
mod tests;
