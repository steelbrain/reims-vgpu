//! What the host video backends share, independent of host, driver and
//! QEMU: the threads and locks they run on, the guest surfaces they write,
//! and how long they wait for their devices.
//!
//! - [`runtime`] is the threads and the clock the host gives the video
//!   crates, and [`sync`] the locks they take.
//! - [`surfaces`] is a held guest surface and its layout, and [`fourcc`]
//!   the four-character codes that name its pixel format.
//! - [`device_memory`] is how a backend's device writes a guest surface's
//!   pages itself, where the host lends them as a dma-buf or a view, and
//!   [`device_wait`] how long a backend waits for its device, and what it
//!   keeps of work the device has not finished by then.
//! - [`backend`] is what a backend answers: the pixel formats it writes,
//!   whether its device wrote a guest surface, and why it failed.

// The library's lint set, the Vulkan crate's core's and the codec crate's: no
// unsafe code but what is marked, no indexing that can panic, and no silent
// `as` conversion outside tests, where clippy.toml lets code unwrap, index
// and panic.
#![deny(
    unsafe_code,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::mod_module_files
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )
)]

pub mod backend;
pub mod device_memory;
pub mod device_wait;
pub mod fourcc;
mod num;
pub mod runtime;
pub mod surfaces;
pub mod sync;
