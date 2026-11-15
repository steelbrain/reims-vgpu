//! The catch around a call of a host function ([`super::host_call`]), in one file per OS with the
//! same function: `catch(f)` runs `f` and answers its value, or `Err(())` for what the host threw.
//! Windows takes Linux's (`linux.rs`): std's `catch_unwind`, which a foreign exception ends the
//! process at, as on Linux.

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(super) use linux::catch;
#[cfg(target_os = "macos")]
pub(super) use macos::catch;
