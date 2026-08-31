//! This crate's half of observability, and the door to the rest of it.
//!
//! # Where the parts live
//!
//! Everything below the device is `reims_vgpu_observe`: the always-on sink, the
//! [`Decline`] and [`Refusal`] traits every subsystem names its refusals
//! through, the `Emit` builder that cannot render a line without a reason, and
//! the slug registry that keeps two checks from spelling one slug. It is a
//! separate crate so the layers below this one can name their own refusal type
//! without depending on the device, and so nothing in it can reach back up into
//! `runtime`, `model`, or a backend — none of them is in scope there. Read that
//! crate's own doc for the obligation, the slug-uniqueness failure it exists to
//! prevent, and why it may describe a decision but never select one.
//!
//! What is left here is the two emitters that are *about this crate's* types:
//!
//! - [`crate::observe::ladder`] — the four object-list resolution rungs, so a rail spells the
//!   condition the same way every other rail does.
//! - [`crate::observe::panic`] — a `catch_unwind` at a `reims_vgpu_qemu_*` entry point, which is
//!   the one failure the sink cannot describe from below because the entry point
//!   is this crate's.
//!
//! Both name `runtime` types, which is exactly why they did not move.
//!
//! # The paths did not change
//!
//! The crate's surface is re-exported under the paths callers already write, so
//! `crate::observe::fail(…)`, `crate::observe::Decline` and
//! `crate::observe::sink::…` mean what they always did. That is deliberate: a
//! layering change that also rewrote three hundred call sites would be two
//! changes reviewed as one.

pub mod ladder;
pub mod panic;

/// The fail line a loader whose event name carries the domain emits for a rung.
pub(crate) use ladder::RungReport;
/// The four object-list resolution rungs, so a rail spells the condition the
/// same way every other rail does. See [`crate::observe::ladder`] for why it is a
/// macro.
pub(crate) use ladder::{ladder_slug, ladder_slugs};
/// Re-exported so call sites write `crate::observe::decline_display!(..)`
/// next to the trait it implements, rather than reaching into the submodule.
pub(crate) use reims_vgpu_observe::decline_display;
pub use reims_vgpu_observe::{
    decline, driver_watch, emit, footprint, phase_clock, sink, slugs, Decline, Refusal,
};
pub(crate) use reims_vgpu_observe::{first_sight, state_changed, Emit};

// The sink's surface is re-exported flat so call sites read `observe::fail(…)`
// rather than `observe::sink::fail(…)`.
pub use reims_vgpu_observe::{
    bgra_present_stats, bgra_rgb_stats, fail, line, nonzero_stats, off, redirect_logs_for_tests,
    rgba_rgb_stats, verbose, when_verbose, RgbaRgbStats,
};
pub(crate) use reims_vgpu_observe::{elapsed_ms, elapsed_us};

// Path accessors and the line matcher exist so tests can assert against the
// real sink rather than a mock; production never reads them back.
#[cfg(test)]
pub(crate) use reims_vgpu_observe::{fail_log_path, FailCapture};
