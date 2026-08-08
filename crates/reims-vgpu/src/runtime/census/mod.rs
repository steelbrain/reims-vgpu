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
//! `t11_decline` was the second: an eight-way reason enum over the type-11
//! sampled rail's zero-copy declines. Across every recorded boot, 1 051 sampled
//! declines named `below_floor` and nothing else — the other seven variants
//! never fired once, including the three that sit *after* the floor test and so
//! were never shadowed by it. That answer is what set
//! `ZERO_COPY_SAMPLED_MIN_BYTES`, and it is recorded on the constant. The rail
//! now returns `Option` like its type-2/3 sibling: falling back to the CPU byte
//! loader is expected control flow that yields the same pixels, so it stays
//! quiet.
//!
//! `deferred_windows` was the third: peak population and forced-eviction count
//! for the three deferred-window caps (GVA 16, surface 16, storage 8), built to
//! answer whether any of them had ever bound. Across every boot in a 72 MB
//! accumulated log it emitted exactly two distinct lines, differing only in
//! `storage_peak` (1 vs 2) — every peak far under its cap, every `evicted` zero.
//! That answer is recorded on the three constants. The alarm it was standing in
//! for survives at each enforcing site, where it belongs: the storage rail's
//! evictions are `compute_mirror_evicted`. Those fire when a cap binds instead of
//! restating a level once a second forever.
//!
//! The test to apply: name the reading the next window could produce that the
//! last thousand did not. If there isn't one, the census has become a probe.

pub mod present_proxy;
pub mod srgb_census;
pub mod view_swizzle_census;
