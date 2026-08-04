//! Crate-wide observability: the always-on log sink, and the decline
//! vocabulary every subsystem reports failures through.
//!
//! # Why this is not under `runtime/`
//!
//! Fail-visibility is not a runtime concern — `backend/`, `contract/`,
//! `model/` and `host_window/` all reject guest work and all owe the reader a
//! reason. It lived under `runtime/` only because that is where the first
//! caller happened to be, and the result was the lapse this module exists to
//! close: 451 fail sites in `runtime/` against 0 in `backend/metal/`,
//! `contract/` and `qemu/`.
//!
//! `translate/` and `caps/` are the other half of the argument. They are pure —
//! they return typed declines and log nothing, which is correct — so the sink
//! must sit somewhere they can name their reason type without depending on
//! `runtime/`. This module is that place.
//!
//! # The parts
//!
//! - [`sink`] — the always-on writer behind `/tmp/reims-vgpu-fail.log`, its background
//!   thread, flood self-detector and test isolation. Moved here verbatim from
//!   `runtime/draw_log.rs`; the machinery was never the problem, the vocabulary
//!   on top of it was.
//!
//! - [`decline`] — the [`Decline`] and [`Refusal`] traits every subsystem
//!   names its refusals through.
//! - [`emit`] — the one builder that renders `reason=<slug> k=v …`, and cannot
//!   produce a line without a reason.
//!
//! # The obligation
//!
//! Per `AGENTS.md`: every path that rejects, drops, degrades or mis-executes a
//! decoded guest command returns a typed decline whose slug is unique crate-wide
//! and reaches the sink at some call site. Both halves are the author's
//! obligation — nothing enforces either. A typed decline nobody logs is still a
//! silent failure.
//!
//! Uniqueness is not a tidiness rule, and this is the failure it prevents.
//! [`Emit::fail_once`] latches on `(slug, discriminant)` through
//! [`first_sight`], whose set is one process-global `HashSet`. Two declines
//! sharing a slug therefore share a latch: whichever fires first for a given
//! discriminant silences the other for that discriminant for the life of the
//! boot, and the loss is invisible because the log looks healthy. That is not
//! hypothetical — `mapping_gpa_span` had exactly this shape between its two
//! emitters, and the silence it produced had already been written up as a
//! finding about the device before the collision was noticed.
//!
//! A scan of every `Decline::slug` body reads 609 distinct slugs and no
//! duplicate. That is a measurement of one tree state, not a guarantee about
//! the next one; a `gate` module that checked it by scanning source text was
//! removed in `db80389` because the check was over text rather than behaviour,
//! and it has not been replaced.
//!
//! The judgement no gate can make stays with the author: do **not** log
//! speculative returns (a resolver legitimately answering "not ready yet" every
//! poll, a genuinely-unbound `ref==0`). Those flood the log.

pub mod decline;
pub mod emit;
pub mod footprint;
pub mod phase_clock;
pub mod sink;

/// Re-exported so call sites write `crate::observe::decline_display!(..)`
/// next to the trait it implements, rather than reaching into the submodule.
pub(crate) use decline::decline_display;
pub use decline::{Decline, Refusal};
pub use emit::{first_sight, state_changed, Emit};

// The sink's surface is re-exported flat so call sites read `observe::fail(…)`
// rather than `observe::sink::fail(…)`. `sink` stays public for the gate and
// for readers who want the machinery.
pub use sink::{
    bgra_present_stats, bgra_present_stats_scalar, bgra_rgb_stats, dump_flush_surfaces, fail, line,
    nonzero_stats, off, redirect_logs_for_tests, rgba_rgb_stats,
};
pub(crate) use sink::{draw_log_enabled, elapsed_ms, elapsed_us};

// Every host-side artifact this crate drops — the sinks themselves, the GOP
// console proxy, the compute-stall SPIR-V dump, the metal2vulkan handoff last
// resort — resolves through one directory, so a host whose spelling of "a
// writable scratch directory" differs is a single edit in `sink` rather than a
// literal repeated across `runtime/`.
pub(crate) use sink::log_dir;

// Path accessors and the line matcher exist so tests can assert against the
// real sink rather than a mock; production never reads them back.
#[cfg(test)]
pub(crate) use sink::{fail_log_path, FailCapture};
