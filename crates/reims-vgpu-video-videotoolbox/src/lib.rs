//! The VideoToolbox backend: decode, encode and motion estimation on a
//! macOS host through VideoToolbox alone, with no FFmpeg or other layer
//! between.
//!
//! - [`VideoToolboxBackend`] is core's `DecoderBackend` on
//!   VTDecompressionSession: H.264, HEVC and AV1 where the host's hardware
//!   decodes them.
//! - [`VideoToolboxEncoderBackend`] is core's `EncoderBackend` on
//!   VTCompressionSession: H.264 and HEVC on the hardware encoders.
//! - [`VideoToolboxMotionBackend`] is core's `MotionEstimationBackend` on
//!   VTMotionEstimationSession, where macOS has it (26 and later).
//! - [`backend`] is all three as core's `Backend`, with no JPEG decoder.
//!
//! Everything here is macOS's alone: on any other host the crate builds
//! and offers nothing.

// The crate doc names this crate's macOS items, which other OSes have none of: its links
// resolve, and are checked, on macOS.
#![cfg_attr(not(target_os = "macos"), allow(rustdoc::broken_intra_doc_links))]
// Every unsafe block says why it is sound: the crate is FFI, in the
// library's process, where a fault ends the guest's VM.
#![deny(clippy::undocumented_unsafe_blocks)]

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "macos")]
pub use macos::{
    Decoder, DecoderChoice, NAME, VideoToolboxBackend, VideoToolboxEncoderBackend,
    VideoToolboxMotionBackend, backend,
};
