//! The `Platform` implementation over Metal, IOSurface and mach_vm. It writes no log line of its
//! own.
//! Metal is sent to through the message layer ([`msg`]), the one path to
//! `objc_msgSend` for Metal; `extern "C"` and `#[link]` stay in this crate, but for the window
//! crate's own: Carbon's keyboard type and AppKit's `NSApp`, beside the system hot keys it
//! looks up at run time with `dlsym`.
//!
//! Metal is macOS's: on any other OS the crate is empty, so that the workspace builds there.

// The crate doc names this crate's macOS items, which other OSes have none of: its links
// resolve, and are checked, on macOS.
#![cfg_attr(not(target_os = "macos"), allow(rustdoc::broken_intra_doc_links))]
#![cfg(target_os = "macos")]

pub mod gpu;
pub mod msg;
pub mod os;
pub mod platform;
pub mod spi;

pub use platform::{MetalPlatform, Obj};
