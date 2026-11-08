//! The codecs above the hardware, for every backend: each codec's syntax
//! and the part of its decoding process a stateless accelerator is not
//! given, in plain Rust with no FFI.
//!
//! - [`encode`] is the H.264 and HEVC encoders' portable half: their
//!   settings, the GOP and decode times, the parameter sets, slice headers
//!   and configuration records the VA-API backend packs, and the rate
//!   control of a stream held to a hard limit, over an `Accelerator` that
//!   codes the slice data.
//! - [`h264`] is H.264's: NAL units, `avcC`, parameter sets, slice
//!   headers, order counts and the DPB, over an `Accelerator` that decodes
//!   the slice data.
//! - [`hevc`] is HEVC's: NAL units, `hvcC`, parameter sets with their
//!   reference picture sets, slice segment headers, order counts and the
//!   DPB, over an `Accelerator` likewise. It shares H.264's bit reader.

// The crate's lint set, core's but its division lint: the decoding
// processes divide by what the standards divide by. Tests are exempt as in
// core: clippy.toml lets them unwrap, index and panic, and the conversion
// and arithmetic lints apply outside cfg(test) only. Under
// `arithmetic_side_effects` each operation that can overflow says what the
// standard means there: checked where an overflow is invalid input, which
// the stream is refused for; wrapping where the standard wraps; saturating
// where it clips, or where what the value is compared with comes out as
// the standard's unbounded arithmetic would. One whose operands' bounds
// keep it from overflowing at all is written plainly where the lint sees
// those bounds, as for a byte widened before a constant is added to it;
// checked against its function's overflow or truncation error where the
// function has one; and saturating, with the bound stated at the site,
// where it has none. A division by what a stream codes is by a NonZero
// type, or checked.
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
        clippy::cast_possible_wrap,
        clippy::arithmetic_side_effects
    )
)]

mod bits;
mod dpb;
pub mod encode;
pub mod h264;
pub mod hevc;
mod num;

/// The fuzzers' harness: each codec's tests have a `fuzz` module of their
/// own.
#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "its counting allocator implements GlobalAlloc, and it reads the CPU time"
)]
mod fuzz_support;

/// The unit tests count their allocations, so that each fuzz input is
/// held to a memory bound.
#[cfg(test)]
#[global_allocator]
static ALLOCATOR: fuzz_support::CountingAllocator = fuzz_support::CountingAllocator;
