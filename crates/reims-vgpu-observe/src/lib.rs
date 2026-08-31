//! Reims vGPU observability: the always-on sink, the decline vocabulary every
//! subsystem reports failures through, and the registry that keeps their slugs
//! apart.
//!
//! # Why this is a crate
//!
//! Fail-visibility is not a runtime concern. Protocol decode, contract
//! arithmetic, device state, backend translation and presentation all reject
//! guest work, and all of them owe the reader a reason. When the vocabulary
//! lived beside the first caller the result was measurable: 451 fail sites in
//! `runtime/` against 0 in `backend/metal/`, `contract/` and `qemu/`.
//!
//! Being a crate makes two claims a compiler can check rather than two habits.
//! The first is that the layer *below* the device — the backend-neutral
//! protocol and contract vocabulary — can name its own refusal type without
//! depending on the device. The second is the direction: nothing here can reach
//! back up into runtime, model, or a backend, because none of them is in scope.
//!
//! # It describes and does not select
//!
//! An observability service that a product path can query is one a product path
//! can come to depend on. So the surface offers no question to ask. A caller
//! states a refusal ([`Decline`], [`Refusal`]), renders it ([`emit::Emit`]), or
//! hands the sink work it may decline to run ([`sink::verbose`],
//! [`sink::when_verbose`]). Nothing hands back a value describing the log's own
//! state.
//!
//! # The parts
//!
//! - [`sink`] — the always-on writer behind `/tmp/reims-vgpu-fail.log`, its
//!   background thread, flood self-detector, and test isolation.
//! - [`decline`] — the [`Decline`] and [`Refusal`] traits every subsystem names
//!   its refusals through.
//! - [`emit`] — the one builder that renders `reason=<slug> k=v …`, and cannot
//!   produce a line without a reason.
//! - [`slugs`] — which type claims each slug, and the collision report when two
//!   do.
//! - [`driver_watch`] — the one failure a census cannot report, because a census
//!   line is written at the end of a drain tranche and this one is a tranche
//!   that never ends: a host driver call that does not return while the drain
//!   thread holds the device lock.
//! - [`footprint`], [`phase_clock`] — shared rendering for sizes and for the
//!   per-phase clocks a census quotes.
//!
//! # The obligation
//!
//! Per `AGENTS.md`: every path that rejects, drops, degrades or mis-executes a
//! decoded guest command returns a typed decline whose slug is unique crate-wide
//! and reaches the sink at some call site. Both halves are the author's
//! obligation. A typed decline nobody logs is still a silent failure.
//!
//! Uniqueness is not a tidiness rule, and this is the failure it prevents.
//! [`emit::Emit::fail_once`] latches on `(slug, discriminant)` through
//! [`emit::first_sight`], whose set is one process-global `HashSet`. Two
//! declines sharing a slug therefore share a latch: whichever fires first for a
//! given discriminant silences the other for that discriminant for the life of
//! the boot, and the loss is invisible because the log looks healthy. That is
//! not hypothetical — `mapping_gpa_span` had exactly this shape between its two
//! emitters, and the silence it produced had already been written up as a
//! finding about the device before the collision was noticed.
//!
//! [`slugs`] observes it. Every line [`emit::Emit`] renders claims its slug for
//! the concrete type that spelled it, and a second type claiming a slug some
//! other type already holds is reported by name on the always-on channel — and
//! panics in a test build, where one process renders thousands of declines
//! across a whole suite. That proves a collision when both sides emit; it cannot
//! prove their absence, and it does not see the narrower shape of one slug
//! returned by two arms of the same impl. Prefix every slug with the rail that
//! owns it, which is what makes a collision unlikely in the first place.
//!
//! The judgement no gate can make stays with the author: do **not** log
//! speculative returns (a resolver legitimately answering "not ready yet" every
//! poll, a genuinely-unbound `ref==0`). Those flood the log.

pub mod decline;
pub mod driver_watch;
pub mod emit;
pub mod footprint;
pub mod phase_clock;
pub mod sink;
pub mod slugs;

pub use decline::{Decline, Refusal};
pub use emit::{first_sight, state_changed, Emit};

// The sink's surface is re-exported flat so call sites read `observe::fail(…)`
// rather than `observe::sink::fail(…)`. `sink` stays public for readers who
// want the machinery.
pub use sink::{
    bgra_present_stats, bgra_rgb_stats, elapsed_ms, elapsed_us, fail, line, nonzero_stats, off,
    redirect_logs_for_tests, rgba_rgb_stats, verbose, when_verbose, RgbaRgbStats,
};

// Path accessors and the line matcher exist so tests can assert against the
// real sink rather than a mock; production never reads them back. Gated on
// `testing` as well as `test` because the tests that use them are compiled
// separately from this crate — see the feature's comment in `Cargo.toml`.
#[cfg(any(test, feature = "testing"))]
pub use sink::{fail_log_path, FailCapture};
