//! Always-on declines whose reason needs state the raising site does not hold.
//!
//! # What is here, and what is deliberately not
//!
//! [`crate::observe`] is the **sink**: `observe::fail`, `observe::off` and
//! `observe::Emit` are where every always-on line lands, and the execution path
//! calls them directly for its own declines. Filing the sink under `census/`
//! would suggest a draw decline is a measurement, which is the distinction the
//! ground rules turn on.
//!
//! These modules are the cases where the *reason* needs state the raising site
//! does not have — a dedup set spanning draws, or a slug vocabulary shared by
//! several call sites — so the line is written here instead. They are still
//! declines. The execution path calls them, never the reverse.
//!
//! Every one of them names a loss that is still happening. A census whose
//! question has been answered is not another kind of decline, it is a probe
//! that outlived its investigation; see "Removing one" below.
//!
//! # The rule these all obey
//!
//! **Measuring is allowed; branching on the measurement is not.** Nothing in
//! the device or backend may read one of these back to decide what to present,
//! decode or execute. A proxy that changes behaviour has become a content
//! heuristic, which the ground rules forbid outright.
//!
//! # What each one reports
//!
//! | Module | Class it reports |
//! |---|---|
//! | [`present_proxy`] | `secondary_mrt_drop` — a multi-RT draw degraded to single-RT — plus `stale_online_pending` and [`present_proxy::window_publish`], the sole record that a captured frame never reached the host window |
//! | [`srgb_census`] | which rails drop the sRGB transfer function |
//! | [`view_swizzle_census`] | type-8 view swizzles dropped, or served by rewriting texels on the CPU |
//!
//! # Adding one
//!
//! A module belongs here only when the loss it names is otherwise invisible. If
//! the refusal already emits a typed decline at the point it refuses, a second
//! count of its *rate* has no claim under the fail-visible rule — and a tally of
//! successful work never had one. Modules and rate-halves have been deleted on
//! exactly that test more often than they have been added; run it before writing
//! the next one.
//!
//! # Removing one
//!
//! A census that reports guest *input* rather than a device decline is a
//! measurement, and a measurement is done when it has an answer. `exec_res_table`
//! was the one such entry: it counted every field of the `EXEC_INDIRECT2`
//! resource table for one boot, established that `set_host_valid` licenses
//! exactly the resources a submission stores into, that the ids are the task's
//! object-ref space and not the mapping space, and that the trailing 16 bytes
//! are zero — then went on recounting all three for every submission of every
//! boot after. The findings live where they are acted on, in
//! `runtime/exec::consume_resource_table`; the only one that could still
//! change is a guest that starts populating the tail, and that is now a typed
//! decline raised at the record, not a counter nobody reads.
//!
//! `deferred_windows` was the third: a population census for deferred state.
//! State whose loss costs guest work now follows guest resource/content
//! lifetimes rather than a capacity, so there is no eviction policy for this
//! census to justify.
//!
//! The test to apply: name the reading the next window could produce that the
//! last thousand did not. If there isn't one, the census has become a probe.

pub mod present_proxy;
pub mod srgb_census;
pub mod view_swizzle_census;
