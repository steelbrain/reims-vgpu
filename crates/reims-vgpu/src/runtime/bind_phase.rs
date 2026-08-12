//! The split of [`crate::runtime::chain_phase`]'s `binds_us`, over the same
//! window.
//!
//! # Why
//!
//! With the render writeback removed, the draw path is what caps this device,
//! and `binds_us` is its largest column: 23.5 µs of a 103 µs draw on a driven
//! x86/PCI control second, within a microsecond of `draw_phase`'s `stage_us`.
//! One column covering `load_buffer_content` for every vertex buffer, the same
//! for every fragment buffer, and the whole stage-in attribute walk is three
//! costs with three different fixes, and no line could tell them apart.
//!
//! This stands to `chain_phase`'s `binds_us` exactly as `draw_phase` stands to
//! its `engine_us`: a division, emitted on the same cadence, read against the
//! line above it.
//!
//! # What it does not report
//!
//! No `rest_us`. The three parts do **not** claim to sum to `binds_us` — the
//! phase also clones two shader `Arc`s and builds a `BTreeSet` outside them —
//! and a computed remainder would go silently wrong the moment a fourth cost
//! were added between them. Divide against `chain_phase`'s `binds_us` by hand;
//! what the parts do not cover is the answer to "is there a fourth".
//!
//! Like every phase census here it reports no loss. A slow bind is not a
//! declined one, and a chain that returns early from inside a span charges its
//! remainder to that span, because the commit is in `Drop`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};
use crate::runtime::spirv_bind::ReflectedBufferAccess;

/// The parts of the bind phase that are worth telling apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    /// `load_buffer_content` over `req.vertex_buffers`.
    VertexLoad = 0,
    /// `load_buffer_content` over `req.fragment_buffers`.
    FragmentLoad = 1,
    /// The stage-in attribute walk over the pipeline's vertex block.
    Attrs = 2,
}

const PARTS: usize = 3;

/// Nanoseconds, per [`crate::observe::phase_clock`]. The attribute walk is the
/// reason: it is sub-microsecond per draw at tens of thousands of draws a
/// second, so a microsecond accumulator reported it as free.
static ACC: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static BINDS: AtomicU64 = AtomicU64::new(0);

/// One slot per [`ReflectedBufferAccess`], indexed by its ordinal.
const ACCESS_CLASSES: usize = ReflectedBufferAccess::Undeclared as usize + 1;

/// The table is wide enough for every class the parse can return. Derived from
/// the highest ordinal rather than hand-counted, so a new class fails the build
/// here instead of wrapping into another class's tally at runtime.
const _: () = assert!(ACCESS_CLASSES == 3);

static ACCESS: [AtomicU64; ACCESS_CLASSES] = [const { AtomicU64::new(0) }; ACCESS_CLASSES];

/// One window of the split, as taken by the per-second census.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindPhaseWindow {
    pub vertex_us: u64,
    pub fragment_us: u64,
    pub attrs_us: u64,
    /// Bind phases entered in the window — the denominator the three share.
    pub binds: u64,
    /// Buffer binds whose stage's reflection says the shader never dereferences
    /// them. These are the ones whose guest bytes need not be staged at all.
    pub access_unused: u64,
    /// Buffer binds reflection says the shader does touch.
    pub access_dereferenced: u64,
    /// Buffer binds reflection gives no answer for. Not a synonym for
    /// [`Self::access_unused`] — see [`ReflectedBufferAccess::Undeclared`].
    pub access_undeclared: u64,
    /// Of [`Self::access_unused`], those whose guest bytes were staged anyway.
    ///
    /// With the rail on this is the stage-in exclusion refusing a substitution,
    /// which is expected rather than an error. With
    /// [`crate::env::UNUSED_BINDS`] off it equals [`Self::access_unused`], which
    /// is how the off arm is confirmed to have taken.
    pub access_unused_staged: u64,
    /// Binds served the neutral page instead of the guest's bytes.
    ///
    /// Together with [`Self::access_unused_staged`] this partitions
    /// [`Self::access_unused`], so the saving is read rather than assumed.
    pub neutral_served: u64,
}

impl BindPhaseWindow {
    /// The three access classes partition the buffer binds resolved in the
    /// window, so they sum to that count. Stated as a method rather than a
    /// field so no emitter can publish a total that disagrees with its parts.
    pub fn access_total(&self) -> u64 {
        self.access_unused + self.access_dereferenced + self.access_undeclared
    }
}

/// Take and clear the window. `None` when no bind phase ran, so an idle second
/// costs no line.
pub fn take_window() -> Option<BindPhaseWindow> {
    let binds = BINDS.swap(0, Ordering::Relaxed);
    let w = BindPhaseWindow {
        vertex_us: to_us(ACC[Part::VertexLoad as usize].swap(0, Ordering::Relaxed)),
        fragment_us: to_us(ACC[Part::FragmentLoad as usize].swap(0, Ordering::Relaxed)),
        attrs_us: to_us(ACC[Part::Attrs as usize].swap(0, Ordering::Relaxed)),
        binds,
        access_unused: ACCESS[ReflectedBufferAccess::Unused as usize].swap(0, Ordering::Relaxed),
        access_dereferenced: ACCESS[ReflectedBufferAccess::Dereferenced as usize]
            .swap(0, Ordering::Relaxed),
        access_undeclared: ACCESS[ReflectedBufferAccess::Undeclared as usize]
            .swap(0, Ordering::Relaxed),
        access_unused_staged: UNUSED_STAGED.swap(0, Ordering::Relaxed),
        neutral_served: NEUTRAL_SERVED.swap(0, Ordering::Relaxed),
    };
    (binds > 0).then_some(w)
}

/// Count one buffer bind against what its stage's reflection said about it.
///
/// Called once per resolved `[[buffer(n)]]` bind on the render path, in both the
/// vertex and the fragment loop, so the three classes partition that population
/// — see [`BindPhaseWindow::access_total`].
pub fn note_access(class: ReflectedBufferAccess) {
    ACCESS[class as usize].fetch_add(1, Ordering::Relaxed);
}

/// Count one bind that reflection called unused and that was staged from guest
/// memory regardless.
///
/// Called on the staging arm for *every* class and filtered here, rather than
/// called only for `Unused` at the one site that knows, so that the two arms of
/// the branch cannot drift: every bind goes through exactly one of
/// [`note_neutral_served`] and this, and their sum over unused binds is
/// `access_unused`.
///
/// A non-zero reading with the rail on is the reliance signal — the stage-in
/// exclusion refusing a substitution — and not an error.
pub fn note_unused_staged(class: ReflectedBufferAccess) {
    if matches!(class, ReflectedBufferAccess::Unused) {
        UNUSED_STAGED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count one bind served the neutral page instead of the guest's bytes.
pub fn note_neutral_served() {
    NEUTRAL_SERVED.fetch_add(1, Ordering::Relaxed);
}

static UNUSED_STAGED: AtomicU64 = AtomicU64::new(0);
static NEUTRAL_SERVED: AtomicU64 = AtomicU64::new(0);

/// Count one entry into the bind phase, so the parts have a denominator that
/// is theirs rather than `chain_phase`'s `chains`.
///
/// Separate from the spans because a draw with no vertex buffers still entered
/// the phase, and dividing by a count that only rose when a span opened would
/// read as though every draw loaded one.
pub fn note_bind() {
    BINDS.fetch_add(1, Ordering::Relaxed);
}

/// Charges the wall clock of one scope to one [`Part`].
///
/// A plain RAII span rather than the open/close phase machine
/// [`crate::runtime::chain_phase`] uses, because the bind phase is a straight
/// sequence: the parts are lexical scopes and nothing needs to switch phase
/// from inside a call several frames down.
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
    fn a_window_with_no_bind_is_none() {
        let _ = take_window();
        assert!(take_window().is_none());
    }

    /// Each part is charged only its own scope. The whole value of the split is
    /// that a vertex-load cost cannot hide inside the attribute column.
    #[test]
    fn each_part_is_charged_only_its_own_scope() {
        let _ = take_window();
        note_bind();
        {
            let _s = Span::open(Part::VertexLoad);
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        {
            let _s = Span::open(Part::Attrs);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let w = take_window().expect("a bind was noted");
        assert_eq!(w.binds, 1);
        assert!(w.vertex_us >= 3_000, "{w:?}");
        assert_eq!(w.fragment_us, 0, "{w:?}");
        assert!(w.attrs_us >= 1_000 && w.attrs_us < w.vertex_us, "{w:?}");
    }

    /// The denominator counts phases entered, not spans opened. A draw that
    /// binds nothing still entered the phase, and dividing by a span count
    /// would report every draw as having loaded a buffer.
    #[test]
    fn the_denominator_counts_phases_not_spans() {
        let _ = take_window();
        note_bind();
        note_bind();
        {
            let _s = Span::open(Part::FragmentLoad);
        }
        let w = take_window().expect("binds were noted");
        assert_eq!(w.binds, 2);
    }

    /// A large population of sub-microsecond spans has to sum to something.
    /// This is the shape the attribute walk is, and it is the whole reason
    /// [`crate::observe::phase_clock`] exists: an empty span here is a pair of
    /// `Instant::now()` calls, which a microsecond-truncating accumulator
    /// charges exactly zero.
    ///
    /// The threshold is measured rather than guessed. Nanosecond accumulation
    /// reads 302-308 µs over three runs (~15 ns a span); truncating
    /// accumulation reads 3, from the handful of spans a scheduling hiccup
    /// pushed over a microsecond. 100 sits a factor of three below the true
    /// reading and thirty above the false one, and load can only raise the
    /// true reading.
    #[test]
    fn twenty_thousand_sub_microsecond_spans_are_not_free() {
        let _ = take_window();
        for _ in 0..20_000 {
            note_bind();
            let _s = Span::open(Part::Attrs);
        }
        let w = take_window().expect("binds were noted");
        assert!(w.attrs_us > 100, "{w:?}");
    }

    /// The three access classes partition the binds counted, so they sum to the
    /// total the line publishes. That identity is what makes the line
    /// self-checking: a reader who divides gets an answer that holds or a bug
    /// that shows.
    #[test]
    fn the_access_classes_sum_to_the_total() {
        let _ = take_window();
        note_bind();
        for _ in 0..5 {
            note_access(ReflectedBufferAccess::Unused);
        }
        for _ in 0..11 {
            note_access(ReflectedBufferAccess::Dereferenced);
        }
        for _ in 0..2 {
            note_access(ReflectedBufferAccess::Undeclared);
        }

        let w = take_window().expect("a bind was noted");
        assert_eq!(w.access_unused, 5, "{w:?}");
        assert_eq!(w.access_dereferenced, 11, "{w:?}");
        assert_eq!(w.access_undeclared, 2, "{w:?}");
        assert_eq!(w.access_total(), 18, "{w:?}");
    }

    /// Each class lands in its own slot. Indexing the table by the enum's
    /// ordinal is only safe while the ordinals are distinct and in range, and a
    /// class quietly tallying into another's slot would read as a real
    /// population rather than as a bug.
    #[test]
    fn each_access_class_lands_in_its_own_slot() {
        for class in [
            ReflectedBufferAccess::Unused,
            ReflectedBufferAccess::Dereferenced,
            ReflectedBufferAccess::Undeclared,
        ] {
            let _ = take_window();
            note_bind();
            note_access(class);
            let w = take_window().expect("a bind was noted");
            assert_eq!(w.access_total(), 1, "{class:?} counted once: {w:?}");
            let landed = match class {
                ReflectedBufferAccess::Unused => w.access_unused,
                ReflectedBufferAccess::Dereferenced => w.access_dereferenced,
                ReflectedBufferAccess::Undeclared => w.access_undeclared,
            };
            assert_eq!(landed, 1, "{class:?} landed in its own slot: {w:?}");
        }
    }

    /// The neutral and staged tallies partition the unused binds, so the saving
    /// the line reports is the substitutions that actually happened rather than
    /// the classification restated.
    ///
    /// The staging arm is charged for every class and filters internally, which
    /// is what this asserts: a `Dereferenced` bind that was staged must not
    /// appear in `access_unused_staged`, or the identity would read as reliance
    /// on guest bytes by binds the rail never had a claim on.
    #[test]
    fn neutral_and_staged_partition_the_unused_binds() {
        let _ = take_window();
        note_bind();
        // Two unused binds substituted, one refused (a stage-in index).
        note_access(ReflectedBufferAccess::Unused);
        note_neutral_served();
        note_access(ReflectedBufferAccess::Unused);
        note_neutral_served();
        note_access(ReflectedBufferAccess::Unused);
        note_unused_staged(ReflectedBufferAccess::Unused);
        // A bind the rail has no claim on, staged as always.
        note_access(ReflectedBufferAccess::Dereferenced);
        note_unused_staged(ReflectedBufferAccess::Dereferenced);

        let w = take_window().expect("a bind was noted");
        assert_eq!(w.access_unused, 3, "{w:?}");
        assert_eq!(w.neutral_served, 2, "{w:?}");
        assert_eq!(
            w.access_unused_staged, 1,
            "only the refused unused bind, not the dereferenced one: {w:?}"
        );
        assert_eq!(
            w.access_unused_staged + w.neutral_served,
            w.access_unused,
            "the two arms partition the unused binds: {w:?}"
        );
    }

    /// Taking the window resets it, so the line is a rate and not a running
    /// total since boot.
    #[test]
    fn taking_the_window_resets_it() {
        let _ = take_window();
        note_bind();
        {
            let _s = Span::open(Part::VertexLoad);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert!(take_window().is_some());
        assert!(take_window().is_none());
        note_bind();
        let w = take_window().expect("the second window emits");
        assert_eq!(w.vertex_us, 0, "{w:?}");
    }
}
