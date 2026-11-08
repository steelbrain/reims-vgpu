//! The reims-graphics device's video on the host: the codecs above the hardware, and the host's
//! video APIs under them, independent of the library and QEMU.
//!
//! - [`core`] is what the video code shares: the threads, clock and locks the host gives it, a
//!   guest surface's memory as a device reaches it, and its bounded waits on the hardware.
//! - [`codec`] is H.264's and HEVC's syntax and decoding processes above the slice data, and the
//!   encoders' portable half, in plain Rust.
//! - `vaapi` is VA-API through libva, and `nvidia` NVDEC and NVENC over the CUDA driver API, each
//!   with its libraries loaded at run time: Linux's alone.

pub mod codec;
pub mod core;
#[cfg(target_os = "linux")]
pub mod nvidia;
#[cfg(target_os = "linux")]
pub mod vaapi;
