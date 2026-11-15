//! The system calls and constants the harness uses, in one file per OS with the same names where
//! they mean the same: the C library's on Linux and macOS, Win32's on Windows.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
pub use linux::*;
#[cfg(target_os = "macos")]
pub use macos::*;
#[cfg(windows)]
pub use windows::*;
