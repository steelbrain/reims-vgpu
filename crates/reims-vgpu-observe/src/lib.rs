//! Failure visibility and measurement shared by device and backend crates.
//!
//! This crate reports decisions; it is never an input to guest-visible policy.

pub mod decline;
pub mod driver_watch;
pub mod emit;
pub mod footprint;
pub mod phase_clock;
pub mod sink;
pub mod srgb_census;

pub use decline::{Decline, Refusal};
pub use emit::{first_sight, state_changed, Emit};
pub use sink::{
    bgra_present_stats, bgra_rgb_stats, draw_log_enabled, elapsed_ms, elapsed_us, fail, line,
    nonzero_stats, off, redirect_logs_for_tests, rgba_rgb_stats,
};

#[cfg(any(test, feature = "test-fixtures"))]
pub use sink::{fail_log_path, FailCapture};
