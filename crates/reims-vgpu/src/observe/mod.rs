//! Crate-wide observability: the always-on log sink, and the decline
//! vocabulary every subsystem reports failures through.
//!
//! # Why this is not under `runtime/`
//!
//! Fail-visibility is not a runtime concern — `backend/`, `contract/`,
//! `model/` and `host_window/` all reject guest work and all owe the reader a
//! reason. It lived under `runtime/` only because that is where the first
//! caller happened to be, and the result was the lapse this module exists to
//! close: failure reporting was concentrated in `runtime/`, while `backend/`,
//! `contract/` and `qemu/` had none.
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
//! - [`driver_watch`] — the one failure a census cannot report, because a census
//!   line is written at the end of a drain tranche and this one is a tranche
//!   that never ends: a host driver call that does not return while the drain
//!   thread holds the device lock.
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
//! **Nothing checks this.** A source scan over every `slug()` and `refusal()`
//! body used to, and it went with the rest of them; a `gate` module before that
//! checked it alongside a 2 700-line restatement of the vocabulary and was
//! removed whole in `db80389`. Two shapes are the defect, at different radii:
//! one slug claimed by two impls, and one slug returned by two arms of the same
//! impl. Prefix every slug with the rail that owns it — that is what makes a
//! collision unlikely by construction, since the audit that would catch one is
//! gone.
//!
//! The judgement no gate can make stays with the author: do **not** log
//! speculative returns (a resolver legitimately answering "not ready yet" every
//! poll, a genuinely-unbound `ref==0`). Those flood the log.

pub mod ladder;
pub(crate) mod model;
pub mod panic;
pub use reims_vgpu_observe::{decline, driver_watch, emit, footprint, phase_clock, sink};

/// The fail line a loader whose event name carries the domain emits for a rung.
pub(crate) use ladder::RungReport;
/// The four object-list resolution rungs, so a rail spells the condition the
/// same way every other rail does. See [`ladder`] for why it is a macro.
pub(crate) use ladder::{ladder_slug, ladder_slugs};
/// Re-exported so call sites write `crate::observe::decline_display!(..)`
/// next to the trait it implements, rather than reaching into the submodule.
pub(crate) use reims_vgpu_observe::decline_display;
pub(crate) use reims_vgpu_observe::{first_sight, Emit};
pub use reims_vgpu_observe::{Decline, Refusal};

// The sink's surface is re-exported flat so call sites read `observe::fail(…)`
// rather than `observe::sink::fail(…)`. `sink` stays public for readers who
// want the machinery.
pub use reims_vgpu_observe::{
    bgra_present_stats, bgra_rgb_stats, fail, line, nonzero_stats, off, redirect_logs_for_tests,
    rgba_rgb_stats,
};
pub(crate) use reims_vgpu_observe::{draw_log_enabled, elapsed_ms, elapsed_us};

// Path accessors and the line matcher exist so tests can assert against the
// real sink rather than a mock; production never reads them back.
#[cfg(test)]
pub(crate) use reims_vgpu_observe::{fail_log_path, FailCapture};
