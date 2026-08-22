//! The split of [`crate::runtime::chain_phase`]'s `sampled_us`, over the same
//! window.
//!
//! # Why
//!
//! A stale comment inside `push_tex` asked for exactly this division — it
//! described a "measure-only setup_tex sub-split" against a post-resolve stats
//! scan that no longer exists. This is that measurement, against the code as it
//! is now. One `sampled_us` bar could not choose between the four things inside
//! it, which is the mistake `draw_phase`'s doc records having made once with
//! `setup_us` and the one `bind_phase` undid for `binds_us`.
//!
//! # This column is fourth, and reading it as second is a measured trap
//!
//! It is **not** the largest undivided column, and the boot that said it was had
//! this device's own coverage instrumentation attached. Both boots below are
//! driven x86/PCI Safari drags on the same tree; the only difference is the
//! instrumentation, and it inverts the ranking of the two columns nothing
//! divided:
//!
//! ```text
//!                   engine_us  binds_us  store_us  sampled_us
//! coverage boot         88749    115363     14895       54106
//! clean boot           106083     51234     38314       21487
//! ```
//!
//! `sampled_us` reads 2.5x high under coverage and `store_us` 2.5x low, so the
//! instrumented boot ranks the sampled phase 3.6x *above* the Store routing and
//! the clean boot ranks it 1.8x *below*. `AGENTS.md` warns that contention
//! "inverts the ranking between the device's two largest costs"; coverage
//! instrumentation does the same to two smaller ones, and the log is well-formed
//! and self-consistent either way, so nothing in it says which you have. Rank
//! columns from a clean boot only. `scripts/runtime-dead`'s reports answer "what
//! never ran", which survives instrumentation; they do not answer "what costs
//! most", which does not.
//!
//! So on a clean driven second `sampled_us` is 21 ms of a 249 ms chain — 8.6%,
//! fourth behind `engine_us`, `binds_us` and `store_us`. The split still earns
//! its line, because it says where inside those 21 ms to look and the answer is
//! lopsided enough to act on: `resolve_us` is 72% of the phase and the other
//! three share 25%.
//!
//! # Why these split points
//!
//! Same rule the other two use: split where the fix changes, not where the code
//! happens to be indented.
//!
//! The `us/s` column is twelve steady-state windows of the clean driven boot
//! above, and it is why the split was worth building: one lever carries the
//! phase and the other three together do not reach a sixth of it.
//!
//! | part | us/s | % | what it brackets | what would fix it |
//! |---|---|---|---|---|
//! | `Resolve` | 15498 | 72.1 | the attachment-alias branch and `resolve_sampled_source`, per texture bind | the sampled content cache and the gather witness |
//! | [`Part::Reflect`] | 2556 | 11.9 | the AIR constexpr static-sampler walk and, in this measurement, the residual SPIR-V sampler-interface scan | carrying the reflected interface with each `m2v_cache` variant |
//! | [`Part::Lookup`] | 1526 | 7.1 | `lookup_list_entry` + `resolve_texture_view`, per texture bind | caching the guest object-list walk and the type-8 view descriptor read |
//! | [`Part::Samplers`] | 1263 | 5.9 | `load_vulkan_sampler` over the record's own sampler binds | the task-scoped retained sampler registry |
//!
//! The sampler registry is now the contract rather than a prospective cache:
//! construction snapshots one immutable sampler under `(task, ref)`, binds
//! retrieve it, and the sampler-delete record or task teardown retires it. In a
//! 34-window macOS 13 Safari drag it constructed 3,144 samplers for 62,609
//! sampler binds, paired 3,143 explicit deletes with zero absent deletes, and
//! reduced this bar from the earlier 5.4 ms/s to 0.89 ms/s. The native backend
//! cache still deduplicates equivalent sampler state; this registry removes the
//! object-list walk and descriptor decode above it.
//!
//! # That table is a light second, and the phase does not stay fourth
//!
//! Everything above is a clean boot whose whole chain cost 249 ms/s. A clean
//! driven **Safari title-bar drag** on the same tree runs a 610 ms/s chain, and
//! the phase does not scale with it — it overtakes everything:
//!
//! ```text
//!                   engine_us  binds_us  store_us  sampled_us   chain
//! light second         106083     51234     38314       21487   249 ms
//! drag second          190004      8025     57080      314037   610 ms
//! ```
//!
//! `sampled_us` goes from 8.6 % of the chain to **51 %** — 15x the absolute
//! cost against a 2.5x heavier chain — and `resolve_us` carries 97 % of it. So
//! the table's ranking is a property of the drive and not of the device, and the
//! phase is first, not fourth, on the workload this device is trying to make
//! smooth. Neither reading is wrong; quoting either without its drive is.
//!
//! That is what split `Resolve` in two. One 306 ms/s bar could not say whether a
//! drag pays it in [`Part::ResolveAlias`], where a draw sampling the target it
//! is drawing into materialises a fresh `w * h` buffer per bind, or in
//! [`Part::ResolveSource`], where the rung ladder reads guest pages — and those
//! have nothing in common but a line number.
//!
//! The last two are one part on purpose. They are different data structures —
//! a small reflection `Vec` and a full SPIR-V word array — but they answer the
//! same question ("which sampler bindings has nothing provisioned yet") and they
//! have the same fix, which is to answer it once per translated shader instead
//! of once per draw. Splitting them would produce two bars a reader could not
//! act on separately.
//!
//! # What it does not report
//!
//! No `rest_us`, for the reason [`crate::runtime::bind_phase`] gives: the four
//! parts do **not** claim to sum to `sampled_us`. The phase also reads the
//! shader reflection for each bind's image dimensionality, folds a 1D image's
//! axes and pushes the engine resources, none of which is bracketed. Divide
//! against `chain_phase`'s `sampled_us` by hand; what the parts do not cover is
//! the answer to "is there a fifth".
//!
//! On the clean boot they cover **97.0%** — 20844 of 21487 µs/s — so the answer
//! is no, and the 3.0% is the unbracketed reflection read and pushes. Had a
//! computed `rest_us` been emitted instead, that 3.0% would have been reported
//! as a fifth cost centre with a name, which is the failure mode leaving it out
//! avoids.
//!
//! `sampled` equals `chain_phase`'s `chains` in every window of that boot, so
//! every draw chain reaches this phase; a chain declining earlier would show up
//! here as a gap between the two, and none did.
//!
//! Like every phase census here it reports no loss. A slow resolve is not a
//! declined one, and the decline paths inside the phase keep their own typed
//! reasons. A bind that returns early from inside a span charges its remainder
//! to that span, because the commit is in `Drop` — deliberate, for the reason
//! `chain_phase` states: an exit is not a phase.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};

/// The parts of the sampled phase that are worth telling apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// The per-bind guest reads: `objects::lookup_list_entry` for the texture's
    /// object-list entry, and `resolve_texture_view` for a type-8 view's
    /// channel remap.
    Lookup = 0,
    /// The fragment attachment-alias branch: the `req.colors` probe, plus
    /// materialising whatever it found. Both of its byte-producing arms build a
    /// fresh `Vec` per bind — a `w * h` RGBA8 fill for a Clear attachment, a
    /// `to_vec` of the prior record's seed for a Load one — so this is where a
    /// draw sampling the target it is drawing into pays for the copy.
    ResolveAlias = 1,
    /// `resolve_sampled_source`: the IOSurface texture rung ladder and the linear guest
    /// rungs, for every bind the alias branch above did not claim.
    ResolveSource = 2,
    /// `load_vulkan_sampler` over the record's vertex and fragment sampler
    /// binds.
    Samplers = 3,
    /// Provisioning for sampler state the guest did not name: AIR constexpr
    /// samplers from reflection plus defaults for the reflected interface each
    /// shader variant already carries.
    Reflect = 4,
    /// Nested inside [`Part::ResolveSource`]: retaining or borrowing the packed
    /// allocation for one linear texture.
    LinearPacked = 5,
    /// Nested inside [`Part::ResolveSource`]: querying or reusing exact backend
    /// image-binding admission for one linear texture.
    LinearAdmission = 6,
    /// Nested inside [`Part::ResolveSource`]: content identity and coherency
    /// witnessing for any zero-copy sampled source.
    GatherWitness = 7,
}

impl Part {
    /// Highest ordinal, so [`PARTS`] is derived from the enum rather than
    /// hand-counted beside it.
    const LAST: Part = Part::GatherWitness;
}

const PARTS: usize = Part::LAST as usize + 1;

/// Nanoseconds, per [`crate::observe::phase_clock`]. The lookup is the reason:
/// it is a pair of map reads per bind at tens of thousands of binds a second,
/// so a microsecond-truncating accumulator would report it as free — the same
/// shape `bind_phase`'s attribute walk has.
static ACC: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static SAMPLED: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SampledPhaseWindow {
    pub lookup_us: u64,
    pub alias_us: u64,
    pub resolve_us: u64,
    pub samplers_us: u64,
    pub reflect_us: u64,
    pub linear_packed_us: u64,
    pub linear_admission_us: u64,
    pub gather_witness_us: u64,
    /// Sampled phases entered in the window — the denominator the four share.
    pub sampled: u64,
}

/// Take and clear the window. `None` when no sampled phase ran, so an idle
/// second costs no line.
pub fn take_window() -> Option<SampledPhaseWindow> {
    let sampled = SAMPLED.swap(0, Ordering::Relaxed);
    let w = SampledPhaseWindow {
        lookup_us: to_us(ACC[Part::Lookup as usize].swap(0, Ordering::Relaxed)),
        alias_us: to_us(ACC[Part::ResolveAlias as usize].swap(0, Ordering::Relaxed)),
        resolve_us: to_us(ACC[Part::ResolveSource as usize].swap(0, Ordering::Relaxed)),
        samplers_us: to_us(ACC[Part::Samplers as usize].swap(0, Ordering::Relaxed)),
        reflect_us: to_us(ACC[Part::Reflect as usize].swap(0, Ordering::Relaxed)),
        linear_packed_us: to_us(ACC[Part::LinearPacked as usize].swap(0, Ordering::Relaxed)),
        linear_admission_us: to_us(ACC[Part::LinearAdmission as usize].swap(0, Ordering::Relaxed)),
        gather_witness_us: to_us(ACC[Part::GatherWitness as usize].swap(0, Ordering::Relaxed)),
        sampled,
    };
    (sampled > 0).then_some(w)
}

/// Count one entry into the sampled phase, so the parts have a denominator that
/// is theirs rather than `chain_phase`'s `chains`.
///
/// Separate from the spans for the reason `bind_phase::note_bind` gives: a draw
/// that samples nothing still entered the phase, and dividing by a count that
/// only rose when a span opened would report every draw as having bound a
/// texture.
pub fn note_sampled() {
    SAMPLED.fetch_add(1, Ordering::Relaxed);
}

/// Charges the wall clock of one scope to one [`Part`].
///
/// A plain RAII span rather than the open/close phase machine
/// [`crate::runtime::chain_phase`] uses, for the reason
/// [`crate::runtime::bind_phase::Span`] gives: the parts are lexical scopes and
/// nothing needs to switch part from inside a call several frames down.
pub struct Span {
    part: Part,
    started: Instant,
}

impl Span {
    pub fn open(part: Part) -> Self {
        Self {
            part,
            started: Instant::now(),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        ACC[self.part as usize].fetch_add(charge_ns(self.started.elapsed()), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An idle second emits nothing rather than a row of zeros.
    #[test]
    fn a_window_with_no_sampled_phase_is_none() {
        let _ = take_window();
        assert!(take_window().is_none());
    }

    /// Each part is charged only its own scope. The whole value of the split is
    /// that a resolve cost cannot hide inside the lookup column, which is what
    /// a single `sampled_us` bar let it do.
    #[test]
    fn each_part_is_charged_only_its_own_scope() {
        let _ = take_window();
        note_sampled();
        {
            let _s = Span::open(Part::ResolveSource);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        {
            let _s = Span::open(Part::Lookup);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let w = take_window().expect("a sampled phase was noted");
        assert_eq!(w.sampled, 1);
        assert!(w.resolve_us >= 3_000, "{w:?}");
        assert_eq!(w.alias_us, 0, "{w:?}");
        assert_eq!(w.samplers_us, 0, "{w:?}");
        assert_eq!(w.reflect_us, 0, "{w:?}");
        assert_eq!(w.linear_packed_us, 0, "{w:?}");
        assert_eq!(w.linear_admission_us, 0, "{w:?}");
        assert_eq!(w.gather_witness_us, 0, "{w:?}");
        assert!(w.lookup_us >= 1_000 && w.lookup_us < w.resolve_us, "{w:?}");
    }

    /// The two halves of the old `Resolve` must not both be charged for one
    /// bind. They are a partition of one lexical scope, and the hand-off is a
    /// `drop` on one branch — the shape where a forgotten `drop` reads as a
    /// phase that got slower rather than as a span that stayed open.
    #[test]
    fn the_two_resolve_halves_are_charged_apart() {
        let _ = take_window();
        note_sampled();
        {
            let _s = Span::open(Part::ResolveAlias);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        let w = take_window().expect("a sampled phase was noted");
        assert!(w.alias_us >= 3_000, "{w:?}");
        assert_eq!(w.resolve_us, 0, "{w:?}");
    }

    /// The denominator counts phases entered, not spans opened. A draw that
    /// binds no texture still entered the phase, and dividing by a span count
    /// would report every draw as having resolved one.
    #[test]
    fn the_denominator_counts_phases_not_spans() {
        let _ = take_window();
        note_sampled();
        note_sampled();
        {
            let _s = Span::open(Part::ResolveSource);
        }
        let w = take_window().expect("sampled phases were noted");
        assert_eq!(w.sampled, 2);
    }

    /// A texture bind opens two spans and a draw binds several, so this
    /// population is large and individually sub-microsecond. It has to sum to
    /// something, which is the whole reason [`crate::observe::phase_clock`]
    /// accumulates nanoseconds: an empty span is a pair of `Instant::now()`
    /// calls, and a microsecond-truncating accumulator charges that exactly
    /// zero however many times it happens.
    ///
    /// The threshold carries `bind_phase`'s measured basis rather than a fresh
    /// guess: the same span shape reads ~15 ns there, so 20 000 of them is a
    /// few hundred microseconds under nanosecond accumulation and single
    /// digits under truncation. 100 sits well below the true reading and well
    /// above the false one, and load can only raise the true one.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            note_sampled();
            let _s = Span::open(Part::Lookup);
        }
        let w = take_window().expect("sampled phases were noted");
        assert!(w.lookup_us > 100, "{w:?}");
    }

    /// Taking the window resets it, so the line is a rate and not a running
    /// total since boot.
    #[test]
    fn taking_the_window_resets_it() {
        let _ = take_window();
        note_sampled();
        {
            let _s = Span::open(Part::Samplers);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(take_window().is_some());
        assert!(take_window().is_none());
        note_sampled();
        let w = take_window().expect("the second window emits");
        assert_eq!(w.samplers_us, 0, "{w:?}");
    }
}
