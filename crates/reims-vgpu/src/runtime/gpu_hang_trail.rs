//! What this device had just asked the GPU to do, kept so that a hang can be
//! named.
//!
//! # The gap this closes
//!
//! When the host GPU wedges, the kernel logs `GPU HANG: ecode …` and resets the
//! context, and this device sees a fence that never signals. By then the batch
//! that hung is gone: `wait_error` knows only that a wait timed out, and every
//! refusal reason downstream of it — `frame_bgra_short`, `no_resident_content`,
//! `read_target_unknown_identity` — is a consequence of the device loss rather
//! than a description of what caused it. Two full sessions have looked for the
//! cause by bisecting the device's narrowing switches, which is a way of asking
//! *whether a rail* is responsible and not *which submission* was.
//!
//! Nor does the kernel's own line say. Its process name identifies the thread
//! that created the i915 context, not the one that submitted the batch — see
//! `kb/the-comm-in-a-gpu-hang-line-names-who-created-the-context.md`.
//!
//! So this keeps the last few pieces of work this device recorded, in memory,
//! and prints them when a drain tranche is caught holding the engine past
//! [`crate::runtime::drain::SYNC_EXEC_STALL_US`] — which on the boots measured
//! so far is the moment the engine wedges, and which fires on the drain thread
//! while it is blocked, so the tail of the trail is the work it is blocked on.
//!
//! # Why a fixed ring and not a counter
//!
//! A counter answers "how much", and the question here is "which". A pipeline
//! that hangs does so because of what it *is* — its fragment module, its
//! geometry, its instance count — and none of that survives as a number
//! averaged over a second. [`CAPACITY`] entries is the bound, chosen so the
//! whole ring is one log line: a hang holds the engine for seconds and the drain
//! blocks inside it, so the work that matters is the last handful and not the
//! last thousand.
//!
//! It is deliberately *not* gated behind [`crate::env::DRAW_LOG`]. That switch
//! turns on a per-draw log flood, which is itself a drain cost heavy enough to
//! change what it measures; this writes seven integers into a fixed array and
//! prints nothing until something has already gone wrong.
//!
//! # Only the Vulkan draw path writes it, and that is not an oversight
//!
//! Two of the fields — the translated module word counts — do not exist on the
//! Metal-direct arm, where the guest's own AIR is handed to the Metal compiler
//! and never becomes SPIR-V. A Metal producer would have to record different
//! fields for a different failure, so it belongs in its own trail if a Metal
//! host is ever seen wedging, not in this one with two columns zeroed. The type
//! and its reader are ungated so that arm compiles; on it the trail stays empty
//! and [`trail`] answers `None`, which emits no line.

use std::sync::Mutex;

/// Entries kept. One log line's worth.
const CAPACITY: usize = 12;

/// One piece of work this device recorded for the GPU.
///
/// The fields are what tells one hanging candidate from another: which guest
/// pipeline object it was, how large each of its two translated modules is —
/// the discriminator for a compositing uber shader against an ordinary blit —
/// and how much geometry it asked for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawNote {
    pub pipeline_ref: u32,
    pub vert_words: u32,
    pub frag_words: u32,
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
    /// Distinct set-0 bindings the fragment module carries a `Binding`
    /// decoration for.
    pub frag_declared: u32,
    /// Distinct bindings this draw will put in the descriptor set layout.
    pub frag_provided: u32,
    /// Of [`Self::frag_declared`], how many the layout will not describe.
    pub frag_gap: u32,
    /// The lowest [`GAP_KEPT`] of them, so the line names the binding and not
    /// only its count. Zero-padded past `min(frag_gap, GAP_KEPT)`.
    pub frag_gap_lo: [u32; GAP_KEPT],
    /// Fragment sampled bindings this draw provided, lowest [`SAMPLED_KEPT`]
    /// first. See [`SampledNote`] for why a trail that names only geometry could
    /// not answer the question this was extended for.
    pub sampled: [SampledNote; SAMPLED_KEPT],
    /// How many the draw provided, so the array reads as a truncation rather
    /// than as the whole list.
    pub sampled_count: u32,
    /// Samplers this draw provided, lowest [`SAMPLER_KEPT`] first. See
    /// [`SamplerNote`] for the hypothesis a trail of textures alone cannot
    /// separate.
    pub samplers: [SamplerNote; SAMPLER_KEPT],
    /// How many the draw provided, for [`Self::sampled_count`]'s reason.
    pub sampler_count: u32,
}

/// One fragment sampled binding, as the draw handed it to the engine.
///
/// # Why the trail needed this
///
/// The wedging draw is a compositing fragment module that walks a pointer chain
/// *through a sampled image* — `uv <- sample(uv).xy`, continuing while
/// `sample(uv).x > 0`, with no counter and no second exit. Zero exits on the
/// first iteration and garbage never terminates, so the whole question is what
/// this device put in that image. The trail recorded the draw's geometry and its
/// two module sizes, which distinguishes the uber shader from a blit and says
/// nothing at all about its inputs — so three live hypotheses (the wrong
/// texture resolved, the wrong format bound, the wrong sampler built) were all
/// equally unobservable from a wedged boot.
///
/// The fields are the three that separate them. `kind` says which rail supplied
/// the texels; `format` says what the shader will read them *as*, which is the
/// hypothesis that a chain stored as float pairs is being quantised by a
/// narrower layout; the extent says whether the image is the one the guest meant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SampledNote {
    /// Device-numbering set-0 binding. `0` means the slot is unused — no sampled
    /// resource is ever bound below `TEXTURE_BINDING_BASE`.
    pub binding: u32,
    /// Which rail supplied the texels: see [`SampledNote::kind_name`].
    pub kind: u8,
    /// `VkFormat`'s raw value. Kept as the number rather than a name because
    /// this type compiles on the Metal-direct arm, where `ash` does not exist.
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// The first four bytes of a CPU-bytes source, or `0` for a rail that put
    /// no bytes on the CPU (`t`, `g`) and for an empty payload.
    ///
    /// For a **1x1** image this is the entire content, and that is what it is
    /// here for. The uber shader's walk continues while `sample(uv).x > 0`, and
    /// a 1x1 texture returns the same texel at every coordinate — so its red
    /// channel decides, on its own and with no undefined memory anywhere in the
    /// story, whether the loop exits immediately or never. The wedging draw
    /// binds one at fragment index 7, which is one of the two the loops walk,
    /// and nothing had ever read it.
    ///
    /// Four bytes and not a hash: the question is a *value*, not whether two
    /// binds match. Larger images get their first texel too, which is worth
    /// little on its own and costs nothing beside it.
    pub texel0: u32,
}

/// Sampled bindings kept per note, lowest binding number first.
///
/// Sixteen and not four, and the reason is the same trap [`GAP_KEPT`] records
/// one version of. The two bindings the uber shader walks are fragment texture
/// indices **6 and 7**, which is `TEXTURE_BINDING_BASE + 6` and `+ 7` — so a
/// list truncated at four would hold the four bindings nobody is asking about
/// and drop both of the ones this exists to name. Sixteen clears them with room
/// for the band to move, and [`DrawNote::sampled_count`] says when even that
/// truncated.
///
/// The cost is `16 * size_of::<SampledNote>()` per entry over a ring of a few
/// dozen, which is kilobytes, paid once and printed only after something has
/// already gone wrong.
pub const SAMPLED_KEPT: usize = 16;

impl SampledNote {
    /// One letter per supplying rail, so a note fits a log line.
    ///
    /// `b` CPU bytes, `t` a resident render target, `g` a guest window gathered
    /// or imported, `?` a kind this predates.
    pub fn kind_name(kind: u8) -> char {
        match kind {
            1 => 'b',
            2 => 't',
            3 => 'g',
            _ => '?',
        }
    }
}

impl std::fmt::Display for SampledNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "b{}:{}:f{}:{}x{}",
            self.binding,
            Self::kind_name(self.kind),
            self.format,
            self.width,
            self.height
        )?;
        if self.texel0 != 0 {
            write!(f, ":t{:08x}", self.texel0)?;
        }
        Ok(())
    }
}

/// One sampler this draw provided, as the draw handed it to the engine.
///
/// # Why the trail needed this too
///
/// [`SampledNote`] answered two of the three hypotheses the wedge left open —
/// which rail supplied the texels, and what format the shader reads them as —
/// and could not touch the third. The uber shader's four unbounded loops all
/// sample through **one** sampler (set 0, binding 67 in m2v numbering, the
/// guest's fragment sampler index 3), and what that sampler *is* decides
/// whether `uv <- sample(uv).xy` walks the cells the guest wrote or a blend of
/// their neighbours.
///
/// A pointer chain stored in a texture is the one content class for which
/// `LINEAR` is not a quality choice: every sample between two cells returns a
/// value that is in neither, so the walk leaves the graph the guest built and
/// the terminating zero cell is never reached. `NEAREST` reads the stored
/// value. This device has two ways to bind `LINEAR` where the guest did not ask
/// for it — a bind whose `sampler_ref` is `0`, and a binding the residual
/// SPIR-V scan provisions a default for — and both produce
/// [`SamplerResource::normalized_default`], which is `Linear`/`Linear`. Nothing
/// distinguished those from a translated guest sampler after the fact.
///
/// [`Self::provenance`] is therefore the field this exists for. The filters and
/// address modes say what was bound; the provenance says whether the guest
/// asked for it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplerNote {
    /// Device-numbering set-0 binding. `0` means the slot is unused — no
    /// sampler is ever bound below `SAMPLER_BINDING_BASE`.
    pub binding: u32,
    /// `N` nearest, `L` linear. ASCII rather than the backend's enum because
    /// this type compiles on the Metal-direct arm, where
    /// `backend::vulkan::engine` does not exist — the same constraint that
    /// keeps [`SampledNote::format`] a raw number.
    pub min_filter: u8,
    pub mag_filter: u8,
    /// `n` not mipmapped, `N` nearest, `L` linear.
    pub mip_filter: u8,
    /// Address mode on U and V: `e` clamp-to-edge, `E` mirror-clamp-to-edge,
    /// `r` repeat, `R` mirror-repeat, `z` clamp-to-zero, `b` clamp-to-border.
    pub address_u: u8,
    pub address_v: u8,
    /// Where the state came from: `g` a translated guest sampler object, `c` an
    /// AIR constexpr sampler carried in reflection, `d` this device's own
    /// [`SamplerResource::normalized_default`] — which is `LINEAR` and which no
    /// guest asked for.
    pub provenance: u8,
    /// Unnormalized texel coordinates, which changes what a UV in `[0, 1]`
    /// addresses and would move a chain walk off its cells on its own.
    pub unnormalized: bool,
}

impl std::fmt::Display for SamplerNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "s{}:{}{}{}{}{}{}{}",
            self.binding,
            self.provenance as char,
            self.min_filter as char,
            self.mag_filter as char,
            self.mip_filter as char,
            self.address_u as char,
            self.address_v as char,
            if self.unnormalized { "!" } else { "" }
        )
    }
}

/// Samplers kept per entry.
///
/// Eight rather than [`SAMPLED_KEPT`]'s sixteen: a draw binds far fewer sampler
/// states than textures — Metal's own limit is 16 and the compositing modules on
/// record use a handful — and [`DrawNote::sampler_count`] says when even this
/// truncated.
pub const SAMPLER_KEPT: usize = 8;

/// Gap binding numbers carried per entry.
///
/// Counts, not a bitmask: this device relocates a fragment stage's buffers to
/// 256+ and its sampled resources to 288+, so the numbers reach past 320 and a
/// mask wide enough to hold them would be four words per entry to answer a
/// question whose interesting answer is "which one". Four is what fits the log
/// line beside everything else, and `frag_gap` says when it is a truncation.
///
/// The first version of this *was* a `u32` mask and it read `fdecl=0x0` on every
/// draw of the shader it was built to look at — not because the module declares
/// nothing, but because every binding it declares is above 255. The count is
/// recorded here rather than in a commit body because that is the shape of
/// mistake a reader of this field is most likely to repeat.
pub const GAP_KEPT: usize = 4;

impl std::fmt::Display for DrawNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pipe={} vw={} fw={} {}x{} vtx={} inst={} fdecl={} fprov={} fgap={}{}",
            self.pipeline_ref,
            self.vert_words,
            self.frag_words,
            self.width,
            self.height,
            self.vertex_count,
            self.instance_count,
            self.frag_declared,
            self.frag_provided,
            self.frag_gap,
            if self.frag_gap == 0 {
                String::new()
            } else {
                format!(
                    "{:?}",
                    &self.frag_gap_lo[..(self.frag_gap as usize).min(GAP_KEPT)]
                )
            }
        )?;
        if self.sampled_count > 0 {
            let shown = (self.sampled_count as usize).min(SAMPLED_KEPT);
            let body = self.sampled[..shown]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            write!(f, " smp={}[{}]", self.sampled_count, body)?;
        }
        if self.sampler_count > 0 {
            let shown = (self.sampler_count as usize).min(SAMPLER_KEPT);
            let body = self.samplers[..shown]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(",");
            write!(f, " smpl={}[{}]", self.sampler_count, body)?;
        }
        Ok(())
    }
}

/// The bindings a module declares that a layout will not describe: the count,
/// and the lowest [`GAP_KEPT`] of them.
///
/// Vulkan requires a descriptor for every resource a shader **statically uses**,
/// and a declared-but-never-referenced variable is legal to omit — so this is a
/// superset of the violation and not the violation itself. It is the cheap
/// question, asked per draw. `spirv_bind::descriptor_static_use` is the exact
/// one and answers `NotDeclared` for anything that is not a `UniformConstant`,
/// which by construction excludes every storage buffer — and a storage buffer is
/// precisely the class this was built to look at.
pub fn gap(declared: &[u32], provided: &[u32]) -> (u32, [u32; GAP_KEPT]) {
    let mut count = 0u32;
    let mut lo = [0u32; GAP_KEPT];
    let mut missing: Vec<u32> = declared
        .iter()
        .copied()
        .filter(|b| !provided.contains(b))
        .collect();
    missing.sort_unstable();
    missing.dedup();
    for (slot, binding) in lo.iter_mut().zip(missing.iter()) {
        *slot = *binding;
    }
    count += missing.len() as u32;
    (count, lo)
}

/// The ring. A `Mutex` rather than a lock-free structure because every writer is
/// the drain worker and the one reader is the same thread inside a stall it has
/// already lost seconds to.
static TRAIL: Mutex<Trail> = Mutex::new(Trail {
    notes: [None; CAPACITY],
    next: 0,
    total: 0,
    seen_pipes: Vec::new(),
    firsts: [None; FIRST_DRAW_KEPT],
    firsts_next: 0,
    submits: [None; SUBMIT_SLOTS],
    pending: Accumulating {
        draws: 0,
        heaviest: None,
        kept: [None; SUBMIT_DRAWS_KEPT],
    },
    submit_seq: 0,
});

/// Pipeline refs whose *first* draw is remembered.
///
/// The ring above holds the last twelve draws, which at twenty thousand draws a
/// second is the last **half millisecond**. The wedge this exists to name begins
/// within a few hundred milliseconds of an application's first frame, so twelve
/// draws lands entirely after it: on the two archived macos-11 boots the device's
/// last census and its first fence timeout are 5 s apart, and the trail printed
/// beside that timeout holds ordinary compositing draws while the two pipelines
/// the guest had just translated are nowhere in it.
///
/// A first-draw record answers the question the ring cannot reach without holding
/// a hundred thousand entries. It is one push per *distinct* pipeline rather than
/// one per draw, so sixteen of them span the whole of an application launching.
const FIRST_DRAW_KEPT: usize = 16;

/// Distinct pipeline refs tracked before the first-draw record stops recording.
///
/// A bound rather than a growing set, because the ref is a guest value and a
/// guest that creates pipelines without bound would otherwise grow this without
/// bound. Past it the set stops admitting *and* the ring stops recording, so a
/// full set cannot turn every later draw into a "new pipeline" line — the
/// failure direction that would flood the one log line this exists to produce.
/// `pipe_firsts_full` says it happened.
const SEEN_PIPES_MAX: usize = 4096;

struct Trail {
    notes: [Option<DrawNote>; CAPACITY],
    next: usize,
    /// Every note ever recorded, so a trail can say how much it is *not*
    /// showing. A trail of twelve out of twelve is the whole boot; twelve out of
    /// four hundred thousand is a tail.
    total: u64,
    /// Pipeline refs already drawn, sorted. A `Vec` and a `binary_search`
    /// rather than a set: the population is a real guest's pipeline count, which
    /// is hundreds, and at that size the compare is a cache line where a tree is
    /// a pointer chase per level. This is on the per-draw path.
    seen_pipes: Vec<u32>,
    /// The first draw of each pipeline ref, with the draw ordinal it happened
    /// at. Oldest first once wrapped.
    ///
    /// The whole note and not just the ref, because the ref alone sends the
    /// reader to a log line that may not exist: `linux_m2v_resources` is emitted
    /// only for a pipeline with a decoded fixed-state gap, so a boot naming
    /// `pipe=91` eighteen draws before its wedge has nothing anywhere that says
    /// how large that pipeline's modules are or what geometry it drew. That is
    /// one more boot to answer a question this record was already holding the
    /// answer to.
    firsts: [Option<(DrawNote, u64)>; FIRST_DRAW_KEPT],
    firsts_next: usize,
    /// One entry per submission ring slot, live between its submit and its
    /// fence retiring. See [`SubmitNote`].
    submits: [Option<SubmitNote>; SUBMIT_SLOTS],
    /// Draws noted since the last submission, not yet attributed to one.
    pending: Accumulating,
    submit_seq: u64,
}

/// Record one draw this device is about to hand the engine.
pub fn note_draw(note: DrawNote) {
    crate::runtime::drain::note_store_route(frag_words_band(note.frag_words));
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let slot = trail.next;
    trail.notes[slot] = Some(note);
    trail.next = (slot + 1) % CAPACITY;
    trail.total = trail.total.wrapping_add(1);
    trail.pending.admit(note);

    if let Err(at) = trail.seen_pipes.binary_search(&note.pipeline_ref) {
        if trail.seen_pipes.len() >= SEEN_PIPES_MAX {
            crate::runtime::drain::note_store_route("pipe_firsts_full");
        } else {
            trail.seen_pipes.insert(at, note.pipeline_ref);
            let total = trail.total;
            let slot = trail.firsts_next;
            trail.firsts[slot] = Some((note, total));
            trail.firsts_next = (slot + 1) % FIRST_DRAW_KEPT;
        }
    }
}

/// The pipelines this device has drawn for the first time most recently, oldest
/// first, each with how many draws ago that was.
///
/// The reading this exists for: a wedge that begins in the second an application
/// opens its first window is a wedge with new pipelines in front of it, and
/// nothing else in the device says which. `None` before any draw, so a caller
/// emits no line rather than an empty one.
pub fn recent_pipeline_firsts() -> Option<String> {
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.total == 0 {
        return None;
    }
    let total = trail.total;
    let body = (0..FIRST_DRAW_KEPT)
        .filter_map(|i| trail.firsts[(trail.firsts_next + i) % FIRST_DRAW_KEPT])
        .map(|(note, at)| format!("[-{} {note}]", total.saturating_sub(at)))
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!("distinct={} {body}", trail.seen_pipes.len()))
}

/// Which size band a draw's fragment module falls in, as a route name.
///
/// The trail answers "what was running when it wedged" and this answers the
/// question that comes straight after it: **how often does that thing run at
/// all**. A trail naming a 95 212-word module in the window before a stall means
/// two very different things depending on whether that module ran twelve times
/// in the boot or twelve thousand — a module that hangs every time it runs and a
/// module that is merely marginal have different fixes, and the trail alone
/// cannot tell them apart because it only ever prints after a stall.
///
/// Banded rather than a peak-plus-count pair because a boot has more than one
/// large module (the pool this device translates holds five over 29 000 words)
/// and a high-water would report only the largest. The bands are decade-ish
/// powers of two over the observed population: everything this workload draws
/// routinely is under a thousand words, and the compositing uber shaders are two
/// orders of magnitude above that, so the interesting boundary is anywhere in
/// between and the exact cuts do not matter.
fn frag_words_band(words: u32) -> &'static str {
    match words {
        0..=1_023 => "fragwords_lt1k",
        1_024..=4_095 => "fragwords_lt4k",
        4_096..=16_383 => "fragwords_lt16k",
        16_384..=65_535 => "fragwords_lt64k",
        _ => "fragwords_ge64k",
    }
}

/// Submission slots this can describe.
///
/// The submission ring is [`crate::backend::vulkan::engine::pools::RING_DEPTH`]
/// deep, and that constant is not nameable here: this module compiles on the
/// Metal-direct arm, where `backend::vulkan` does not exist. So the relation is
/// asserted on the Vulkan side, where both names are in scope — see the
/// `const _` beside `RING_DEPTH`. A capacity larger than the ring is harmless
/// (the extra entries are never written); one smaller would silently drop the
/// slot the wedge is on, which is the entire reading.
pub const SUBMIT_SLOTS: usize = 16;

/// One submission this device handed the queue, kept until its fence retires.
///
/// [`DrawNote`] answers "what was this device drawing", and the trail of the
/// last twelve of them is the last half-millisecond. This answers the question
/// that outlives it: **which submission never came back**. The ring retires in
/// order and a slot's note is cleared only when its fence has signalled, so a
/// slot still holding one is a submission still outstanding, and the lowest
/// [`Self::seq`] among them is the one everything else is queued behind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SubmitNote {
    /// Submit ordinal, so "oldest outstanding" is a comparison and not an
    /// inference from ring positions that wrap.
    pub seq: u64,
    /// The draw ordinal this submission closed at, for lining it up against
    /// [`trail`]'s `kept=n/total`.
    pub at_draw: u64,
    /// Draws recorded into it.
    ///
    /// Zero is a real and interesting reading, not a gap: most submissions on
    /// this device carry copies, resolves or stamp words rather than draws. A
    /// wedge on a `draws=0` slot says the hang is not in a draw at all, which no
    /// other instrument here can distinguish.
    pub draws: u32,
    /// The heaviest fragment module in it, whole rather than by reference.
    ///
    /// The same argument [`Trail::firsts`] makes: a pipeline ref alone sends the
    /// reader to a log line that may not exist for that pipeline, and this
    /// record already holds the answer.
    pub heaviest: Option<DrawNote>,
    /// The first [`SUBMIT_DRAWS_KEPT`] draws in arrival order.
    ///
    /// [`Self::heaviest`] alone names one draw out of [`Self::draws`], and a
    /// wedged submission carrying two of them leaves the reader inferring which
    /// one hung from which is larger. That inference has been right so far —
    /// the heaviest draw of the wedging submission is identical field for field
    /// across independent boots, down to an unusual render extent, which a
    /// merely-adjacent draw would have no reason to be — but it is an inference,
    /// and this turns it into a reading. `draws <= SUBMIT_DRAWS_KEPT` means the
    /// list is the whole submission; above it, `draws` says how much is missing.
    pub kept: [Option<DrawNote>; SUBMIT_DRAWS_KEPT],
}

/// Draws recorded per submission, in arrival order.
///
/// Four rather than [`super::super::backend`]'s batch cap of 32: the wedged
/// submissions measured on this rail carry two, so four holds the whole of the
/// case this exists for while keeping the log line readable. `draws` is always
/// the true count, so a truncation can never read as a complete list.
pub const SUBMIT_DRAWS_KEPT: usize = 4;

impl std::fmt::Display for SubmitNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "#{} at_draw={} draws={}",
            self.seq, self.at_draw, self.draws
        )?;
        if let Some(note) = &self.heaviest {
            write!(f, " heaviest=[{note}]")?;
        }
        for (ordinal, note) in self
            .kept
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.map(|n| (i, n)))
        {
            write!(f, " d{ordinal}=[{note}]")?;
        }
        Ok(())
    }
}

/// What has been recorded since the last submission, waiting to be attributed
/// to one.
#[derive(Clone, Copy, Debug, Default)]
struct Accumulating {
    draws: u32,
    heaviest: Option<DrawNote>,
    kept: [Option<DrawNote>; SUBMIT_DRAWS_KEPT],
}

impl Accumulating {
    fn admit(&mut self, note: DrawNote) {
        // Arrival order, and the *first* few rather than the last: a submission
        // is attributed at its flush, so keeping the last would drop exactly the
        // draw that opened the batch.
        if let Some(slot) = self.kept.get_mut(self.draws as usize) {
            *slot = Some(note);
        }
        self.draws = self.draws.saturating_add(1);
        let heavier = match &self.heaviest {
            Some(seen) => note.frag_words > seen.frag_words,
            None => true,
        };
        if heavier {
            self.heaviest = Some(note);
        }
    }
}

/// Record that the slot `slot` has been submitted, carrying everything noted
/// since the previous submission.
///
/// Called from the one place both submit paths converge — a batch flush and a
/// lone draw's own submit both reach `finish_entry_async`, which is also where
/// the slot starts owing cleanup. Attributing at any earlier point would charge
/// draws to a submission that had not closed yet.
pub fn note_submit(slot: usize) {
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if slot >= trail.submits.len() {
        // Unreachable while the `const _` beside `RING_DEPTH` holds. Counted
        // rather than panicked: this is an instrument, and it may not be the
        // thing that takes a boot down. The accumulator is deliberately left
        // alone — dropping it here would misattribute those draws to the *next*
        // submission rather than losing them visibly.
        drop(trail);
        crate::runtime::drain::note_store_route("hang_submit_slot_over_capacity");
        return;
    }
    let acc = std::mem::take(&mut trail.pending);
    let seq = trail.submit_seq.wrapping_add(1);
    let at_draw = trail.total;
    trail.submit_seq = seq;
    trail.submits[slot] = Some(SubmitNote {
        seq,
        at_draw,
        draws: acc.draws,
        heaviest: acc.heaviest,
        kept: acc.kept,
    });
}

/// Record that the slot `slot`'s fence has signalled and its work is done.
pub fn note_retired(slot: usize) {
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(entry) = trail.submits.get_mut(slot) {
        *entry = None;
    }
}

/// Every submission still outstanding, oldest first, as one line's worth.
///
/// `None` when nothing is outstanding — which is itself a reading, and a
/// surprising one at a stall: it says the wait this device is inside is not
/// waiting on anything this ring submitted.
pub fn outstanding() -> Option<String> {
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    render_outstanding(&trail.submits)
}

/// [`outstanding`]'s ordering and formatting, over a slice rather than the
/// process-wide ring, so it can be tested for the property that matters.
///
/// Ordered by [`SubmitNote::seq`] and never by slot index: the ring wraps, so
/// slot 0 is as likely to hold the newest submission as the oldest, and reading
/// the array in index order would put an arbitrary rotation of the queue in
/// front of the reader with the wedge somewhere in the middle of it.
fn render_outstanding(submits: &[Option<SubmitNote>]) -> Option<String> {
    let mut live: Vec<(usize, SubmitNote)> = submits
        .iter()
        .enumerate()
        .filter_map(|(slot, note)| note.map(|n| (slot, n)))
        .collect();
    if live.is_empty() {
        return None;
    }
    live.sort_unstable_by_key(|(_, note)| note.seq);
    let body = live
        .iter()
        .map(|(slot, note)| format!("[slot={slot} {note}]"))
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!("outstanding={} {body}", live.len()))
}

/// One slot's outstanding submission, for the wait that just failed on it.
pub fn submission(slot: usize) -> Option<SubmitNote> {
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    trail.submits.get(slot).copied().flatten()
}

/// The trail, oldest first, as one line's worth of text.
///
/// `None` when nothing has been recorded, so a caller emits no line rather than
/// an empty one — a stall with no draws behind it is a real and different
/// reading from a stall whose draws are unremarkable, and an empty list would
/// spell them the same way.
pub fn trail() -> Option<String> {
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.total == 0 {
        return None;
    }
    let kept = trail.notes.iter().filter(|n| n.is_some()).count();
    // Oldest first: start at the write cursor, which is the oldest live slot
    // once the ring has wrapped and an empty one before that.
    let body = (0..CAPACITY)
        .filter_map(|i| trail.notes[(trail.next + i) % CAPACITY].as_ref())
        .map(|n| format!("[{n}]"))
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!("kept={kept}/{} {body}", trail.total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pipeline_ref: u32) -> DrawNote {
        DrawNote {
            pipeline_ref,
            ..DrawNote::default()
        }
    }

    /// Nothing recorded is not the same reading as nothing interesting
    /// recorded, so the caller gets to say so.
    #[test]
    fn an_empty_trail_reports_nothing_rather_than_an_empty_list() {
        // Shares the process-wide ring with the tests below, so this one asserts
        // only what is true whatever they have written: a trail is `Some` once
        // anything has been noted, and this test cannot un-note it.
        if trail().is_none() {
            note_draw(note(1));
            assert!(trail().is_some(), "a note makes the trail readable");
        }
    }

    /// A pipeline is recorded on its **first** draw and never again, so the
    /// record reaches back past what the twelve-entry ring can hold.
    ///
    /// That reach is the whole point. Twelve draws is half a millisecond on this
    /// rail, and the wedge these instruments exist for begins in the few hundred
    /// milliseconds after an application opens its first window — so the ring
    /// lands entirely after it while a first-draw record still names the
    /// pipelines that arrived with the window.
    #[test]
    fn a_pipeline_is_recorded_once_and_the_record_outreaches_the_ring() {
        // The ring is process-wide and shared with the tests around this one, so
        // everything here is asserted about ids only this test uses.
        let old = 7_000_001u32;
        note_draw(note(old));
        // Far more draws than the ring holds, all of one already-seen pipeline:
        // the ring has forgotten `old` entirely and the first-draw record has
        // not, and no repeat has pushed a second entry for it.
        for _ in 0..(CAPACITY * 4) {
            note_draw(note(old));
        }
        let line = recent_pipeline_firsts().expect("draws were recorded");
        assert_eq!(
            line.matches(&format!("pipe={old} ")).count(),
            1,
            "a repeat must not re-record: {line}"
        );
        assert!(
            !trail().expect("draws were recorded").contains("pipe=8_"),
            "sanity: the ring holds only this test's repeats"
        );

        // A genuinely new pipeline lands at the newest end, and the ring keeps
        // the newest `FIRST_DRAW_KEPT` of them.
        for i in 0..(FIRST_DRAW_KEPT as u32) {
            note_draw(note(7_100_000 + i));
        }
        let line = recent_pipeline_firsts().expect("draws were recorded");
        assert!(
            !line.contains(&format!("pipe={old} ")),
            "the oldest first-draw ages out once {FIRST_DRAW_KEPT} newer ones arrive: {line}"
        );
        assert!(
            line.contains(&format!("pipe={} ", 7_100_000 + FIRST_DRAW_KEPT as u32 - 1)),
            "the newest first-draw is always present: {line}"
        );
        assert!(line.starts_with("distinct="), "{line}");
    }

    /// The ring keeps the *newest* entries and reports them oldest first, which
    /// is the order a reader reconstructs a submission sequence in.
    #[test]
    fn the_ring_keeps_the_newest_entries_in_arrival_order() {
        for i in 0..(CAPACITY as u32 * 2) {
            note_draw(note(1000 + i));
        }
        let line = trail().expect("notes were recorded");
        let first = 1000 + CAPACITY as u32;
        let last = 1000 + CAPACITY as u32 * 2 - 1;
        let pipes: Vec<&str> = line
            .match_indices("pipe=")
            .map(|(i, _)| &line[i..])
            .collect();
        assert_eq!(pipes.len(), CAPACITY, "the ring is full: {line}");
        assert!(
            line.contains(&format!("pipe={first} ")),
            "the oldest kept entry is the first of the last {CAPACITY}: {line}"
        );
        assert!(
            line.contains(&format!("pipe={last} ")),
            "the newest entry is kept: {line}"
        );
        assert!(
            !line.contains(&format!("pipe={} ", first - 1)),
            "the entry before the window was evicted: {line}"
        );
        let first_at = line.find(&format!("pipe={first} ")).unwrap();
        let last_at = line.find(&format!("pipe={last} ")).unwrap();
        assert!(first_at < last_at, "oldest first: {line}");
    }

    /// A gap is counted in one direction only: a layout may legally carry a
    /// descriptor the module never mentions, and only the reverse is a
    /// specification violation. The kept list is a truncation of the count and
    /// says so by disagreeing with it.
    #[test]
    fn the_gap_is_declared_minus_provided_and_names_its_lowest_members() {
        let g = gap(&[0, 256, 258, 288, 290, 320], &[0, 256, 288]);
        assert_eq!(g.0, 3, "258, 290 and 320 are unlaid out");
        assert_eq!(g.1[..3], [258, 290, 320]);

        let none = gap(&[1, 2], &[1, 2, 3]);
        assert_eq!(none.0, 0, "provided-but-not-declared is not a gap");

        let many: Vec<u32> = (100..110).collect();
        let over = gap(&many, &[]);
        assert_eq!(over.0, 10, "the count is the whole gap");
        assert_eq!(over.1, [100, 101, 102, 103], "the list is its lowest four");
    }

    /// The bands tile the whole `u32`, so no module size is uncounted, and the
    /// boundaries are exclusive on the way up: a reader summing them gets the
    /// draw count.
    #[test]
    fn the_fragment_size_bands_tile_every_module_size() {
        assert_eq!(frag_words_band(0), "fragwords_lt1k");
        assert_eq!(frag_words_band(1_023), "fragwords_lt1k");
        assert_eq!(frag_words_band(1_024), "fragwords_lt4k");
        assert_eq!(frag_words_band(4_095), "fragwords_lt4k");
        assert_eq!(frag_words_band(4_096), "fragwords_lt16k");
        assert_eq!(frag_words_band(16_383), "fragwords_lt16k");
        assert_eq!(frag_words_band(16_384), "fragwords_lt64k");
        assert_eq!(frag_words_band(65_535), "fragwords_lt64k");
        assert_eq!(frag_words_band(65_536), "fragwords_ge64k");
        // The one this exists to separate: the compositing uber shader measured
        // in the trail of a wedged macos-11 boot.
        assert_eq!(frag_words_band(95_212), "fragwords_ge64k");
        assert_eq!(frag_words_band(u32::MAX), "fragwords_ge64k");
    }

    /// The total is what says whether a full ring is the whole boot or its tail.
    #[test]
    fn the_trail_says_how_much_it_is_not_showing() {
        for i in 0..(CAPACITY as u32 + 5) {
            note_draw(note(2000 + i));
        }
        let line = trail().expect("notes were recorded");
        let kept = line
            .split_whitespace()
            .next()
            .and_then(|f| f.strip_prefix("kept="))
            .and_then(|f| f.split_once('/'))
            .map(|(k, t)| (k.to_string(), t.to_string()))
            .expect("the line leads with kept=N/M");
        assert_eq!(kept.0, CAPACITY.to_string());
        assert!(
            kept.1.parse::<u64>().expect("a total") > CAPACITY as u64,
            "the total counts every note, not the kept ones: {line}"
        );
    }

    fn submit(seq: u64, draws: u32) -> Option<SubmitNote> {
        Some(SubmitNote {
            seq,
            at_draw: seq * 10,
            draws,
            heaviest: None,
            kept: [None; SUBMIT_DRAWS_KEPT],
        })
    }

    /// The wedge is the *oldest* outstanding submission, and the ring wraps —
    /// so slot order is a rotation of submit order and reading the array as it
    /// lies puts an arbitrary entry at the head of the list.
    #[test]
    fn outstanding_submissions_are_ordered_by_submit_and_not_by_slot() {
        // Slot 0 holds the newest and slot 2 the oldest: exactly the rotation a
        // ring that has wrapped produces.
        let submits = [submit(9, 1), submit(10, 2), submit(7, 3), None];
        let line = render_outstanding(&submits).expect("three are outstanding");
        assert!(
            line.starts_with("outstanding=3 "),
            "the count leads the line: {line}"
        );
        let slots: Vec<&str> = line
            .match_indices("slot=")
            .map(|(at, _)| &line[at + 5..at + 6])
            .collect();
        assert_eq!(
            slots,
            ["2", "0", "1"],
            "oldest submit first, whatever slot it landed in: {line}"
        );
    }

    /// A ring with nothing outstanding is a real reading — it says the wait is
    /// not on anything this ring submitted — and it must not spell the same as
    /// an empty list.
    #[test]
    fn a_ring_with_nothing_outstanding_reports_nothing() {
        assert!(render_outstanding(&[None, None]).is_none());
        assert!(render_outstanding(&[]).is_none());
    }

    /// A submission carrying no draws is the discriminating reading: it says the
    /// wedge is not in a draw. It must render, and it must not claim a heaviest.
    #[test]
    fn a_submission_with_no_draws_still_names_itself() {
        let line = render_outstanding(&[submit(1, 0)]).expect("one is outstanding");
        assert!(line.contains("draws=0"), "{line}");
        assert!(
            !line.contains("heaviest"),
            "no draw means no heaviest module to name: {line}"
        );
    }

    /// A wedged submission carrying two draws must name both, or the reader is
    /// left inferring which of them hung from which is larger.
    #[test]
    fn a_submission_keeps_its_draws_in_arrival_order() {
        let mut acc = Accumulating::default();
        for pipeline_ref in 1..=3 {
            acc.admit(DrawNote {
                pipeline_ref,
                ..DrawNote::default()
            });
        }
        let kept: Vec<u32> = acc.kept.iter().flatten().map(|n| n.pipeline_ref).collect();
        assert_eq!(kept, [1, 2, 3], "arrival order, first-in");
    }

    /// Past the bound the list truncates, and `draws` is what says so — a
    /// truncated list must never read as a complete submission.
    #[test]
    fn a_submission_past_the_kept_bound_still_reports_its_true_count() {
        let mut acc = Accumulating::default();
        for pipeline_ref in 0..(SUBMIT_DRAWS_KEPT as u32 + 3) {
            acc.admit(DrawNote {
                pipeline_ref,
                ..DrawNote::default()
            });
        }
        assert_eq!(acc.draws, SUBMIT_DRAWS_KEPT as u32 + 3);
        assert_eq!(acc.kept.iter().flatten().count(), SUBMIT_DRAWS_KEPT);
        let kept: Vec<u32> = acc.kept.iter().flatten().map(|n| n.pipeline_ref).collect();
        assert_eq!(
            kept,
            (0..SUBMIT_DRAWS_KEPT as u32).collect::<Vec<_>>(),
            "the first are kept, not the last — a submission is attributed at \
             its flush and the opening draw is the one worth keeping"
        );
    }

    /// The heaviest fragment module in a submission is what tells a compositing
    /// uber shader from the ordinary blits around it, so the accumulator keeps
    /// the largest and not the most recent.
    #[test]
    fn a_submission_keeps_its_heaviest_module_and_not_its_last() {
        let mut acc = Accumulating::default();
        acc.admit(DrawNote {
            pipeline_ref: 7,
            frag_words: 95_212,
            ..DrawNote::default()
        });
        acc.admit(DrawNote {
            pipeline_ref: 8,
            frag_words: 412,
            ..DrawNote::default()
        });
        assert_eq!(acc.draws, 2);
        let heaviest = acc.heaviest.expect("two draws were admitted");
        assert_eq!(heaviest.frag_words, 95_212);
        assert_eq!(
            heaviest.pipeline_ref, 7,
            "the heaviest draw's own pipeline, not the last draw's"
        );
    }
}
