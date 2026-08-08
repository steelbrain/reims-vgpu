//! Where one draw's wall clock goes, split at the boundaries that mean
//! different fixes.
//!
//! `drain_duty` established that `draw_us` is 93-99% of the drain worker's busy
//! time, and `engine_delta` priced the bytes that cross the bus. Neither can
//! separate the two shapes a slow draw comes in, and they need opposite work:
//!
//! - **Bytes.** `stage` and `readback` dominate → the fix is to move less, which
//!   is the guest-page writeback family: render into the order the destination
//!   stores and copy straight into its pages, so no byte crosses host memory.
//! - **Latency.** `wait` dominates → the fix is to stop round-tripping the GPU
//!   per draw, and moving bytes faster buys nothing.
//!
//! Measured on one x86/Vulkan boot under the standing soak (442 206 draws over
//! 342 s): `wait` 43%, the four setup phases 37%, `readback` 13%, and `record` —
//! encoding the commands — 1.1%. Joined per window against `engine_delta`, only
//! **14%** of draws read back at all and each of those blocks **1.2 ms** in the
//! fence wait, while the readback copy itself runs at 8.9 GB/s. So the cost is
//! latency, not bytes: 61 847 readbacks spent 74 s of that boot waiting for a
//! single queue to drain.
//!
//! One draw's total is charged to exactly one phase at a time, so the nine
//! numbers sum to the draw. The split points are the calls that change what the
//! CPU is doing:
//!
//! | phase | from | to |
//! |---|---|---|
//! | `prep` | entry | `begin_entry` returns a ring slot |
//! | `pipeline` | there | shaders, layout, pass and pipeline are resolved |
//! | `stage` | there | vertex/index/storage/seed bytes are in staging |
//! | `stage_pass` | there | the primary render pass is resolved |
//! | `acquire` | there | the render target, its framebuffer and any transient depth are held |
//! | `acquire_sampled` | there | every sampled image the draw binds is *decided and created* |
//! | `sampled_upload` | inside it | the staging buffer is held and the guest bytes are gathered into it |
//! | `acquire_readback` | there | this draw's readback buffer is held |
//! | `descriptors` | there | the descriptor set is written |
//! | `record` | there | the CB is ended |
//! | `submit` | there | `queue_submit` returns |
//! | `wait` | there | this draw's fence signals |
//! | `readback` | there | the mapped buffer is copied out |
//!
//! The four middle phases are what a first pass called `setup`, split because
//! that one number came out at 37% of all draw time while `record` — encoding
//! the actual commands — came out at 1%, and the four have nothing in common
//! but their position. `stage` is host memcpy into mapped staging and scales
//! with bytes; `pipeline` is driver compiles; `descriptors` is pool pressure.
//! Each has a different fix and one bar cannot choose between them.
//!
//! # Why `acquire` is three numbers and not one
//!
//! `acquire` used to cover the render target, the sampled images and the
//! draw's readback buffer, and its description here said it "scales with
//! churn" — meaning
//! `vkCreateImage`/`vkAllocateMemory`/slab. On a driven x86/Vulkan boot that
//! description sent a reader after the wrong cost, so the phase was split where
//! the two populations meet.
//!
//! What convicted the churn reading is a regression of the two counters against
//! each other across 32 driven one-second windows. If `acquire` were creates,
//! `acquire_us / creates` would be flat; it is not, and it moves the wrong way:
//!
//! ```text
//! draws   acquire_us   creates   us/create   us/draw
//!   980        81715        96         851      83.4
//!   118        17205       168         102     145.8
//! ```
//!
//! The window with **more** creates and an eighth of the draws spent a fifth of
//! the time. Across all 32 windows `us/create` ranges 402-1141 while `us/draw`
//! holds 73-146, so the phase is paid per draw, not per creation — and it is
//! paid on the cache-*hit* path, because those windows report `gen_mismatch=0`,
//! `target_evicts=0` and every `*_misses` counter at 0. `registry_ensure`'s hit
//! arm is a `HashMap` get and a touch, which cannot be 85 us.
//!
//! That leaves the rest of the phase, and splitting it is what distinguishes
//! "the target was expensive to hold" from "the textures were" — the same
//! argument that split `setup`.
//!
//! **The first split's reading settled the target half and nothing else.** On a
//! driven x86/Vulkan boot, five consecutive one-second windows at ~660 draws
//! read `acquire_us` 0, 0, 0, 0, 52 against `acquire_sampled_us` 66798, 66579,
//! 63803, 67030, 64430. So holding the render target, its framebuffer and its
//! depth costs *nothing* — the churn reading is refuted outright, not merely
//! doubted — and the entire ~100 us per draw is downstream of it.
//!
//! That reading is also what forced the third number. The block after the
//! sampled loop ends with `acquire_readback`, which holds a `width * height * 4`
//! buffer — 8.3 MB at 1920x1080 — once per draw that reads back. Leaving it
//! inside `acquire_sampled` would have charged an 8 MB buffer acquisition to
//! "sampled textures" and invited exactly the misreading this section exists to
//! correct, so it gets its own slot. The two are separated by what fixes them:
//! `acquire_sampled` is per bound texture and is attacked by binding fewer or
//! caching better, `acquire_readback` is per full-frame buffer and is attacked
//! by not reading back.
//!
//! **The three-way reading settles it.** Five consecutive one-second windows at
//! 660 draws on a driven x86/PCI boot:
//!
//! ```text
//! acquire_us              5      1      0      0   3026
//! acquire_sampled_us  43572  43585  43276  43128  43916
//! acquire_readback_us     0      0      0      0      0
//! ```
//!
//! Both neighbours are zero and the sampled loop is the whole of it — ~66 us
//! per draw. `acquire_readback` reading a flat 0 is consistent with
//! `engine_delta allocs=0`: `create_readback_buffer` bumps `note_alloc`, so a
//! zero there proves the readback pool always hits and the acquire is a pop.
//!
//! # What the sampled loop's own cost is *not*
//!
//! The eliminations below are worth keeping because they are what makes the
//! remaining candidate stark, but read them as being about the **driven**
//! regime — see "and none of it holds when the guest is quiet" at the end. In
//! those
//! same windows `sampled_gpu_binds`, `sampled_cache_misses`, `sampled_reuploads`
//! and `sampled_cache_hits` are **all 0**, and only `sampled_identity_hits`
//! (420/s) moves. So:
//!
//! - The `SampledSource::Target` arm is never taken — it is the only writer of
//!   `sampled_gpu_binds`.
//! - No bind uploads or re-uploads bytes; the `Bytes` arm always hits cache.
//! - The hit it takes is `find_cached_sampled`'s identity fast path, which
//!   scans a `SAMPLED_CACHE_CAP`-bounded list of **64** entries and moves one to
//!   the back. That cannot be the ~100 us per bind the arithmetic demands.
//!
//! **Do not read the third bullet as "the content path never fires".** A later
//! driven x86/Vulkan boot under a 30 s Safari *drag*, 42 census windows, reads
//! `sampled_cache_hits=26697` against `sampled_identity_hits=75994` — the
//! content fallback carries about a quarter of all hits, moving 277 MB at
//! ~10 KB a hit. The windows above and these are different workloads, and both
//! readings stand; what does not stand is concluding from the zeros here that
//! the content rail is dead and can go. It is load-bearing, and
//! `ResidentSampledSlot::content` is what makes taking it safe.
//!
//! What is left is `SampledSource::GuestRuns`, and the reason it was invisible
//! is that it incremented no counter of its own. That arm calls
//! `acquire_sampled`, then `acquire_staging`, then `write_staging_from_runs` —
//! a real memcpy gather out of scattered guest RAM into mapped staging, once
//! per bind. Every other arm reported itself; the one that moves bytes did not.
//!
//! # It is the gather, and the gather is a second gigabyte-per-second rail
//!
//! `sampled_gathers` / `sampled_gather_bytes` closed it. Eight consecutive
//! one-second windows at 660 draws on a driven x86/PCI boot:
//!
//! ```text
//! acquire_sampled_us  72710  61495  66565  64560  65432  68702  61644  65713
//! sampled_gathers       360    360    360    360    360    360    360    360
//! sampled_gather_MB   842.4  842.4  842.4  842.4  842.4  842.4  842.4  842.4
//! us per gather         202    171    185    179    182    191    171    183
//! ```
//!
//! 360 gathers a second at ~180 us each is ~65 ms, which is the phase total.
//! The gather is not part of the cost; it is the cost, and nothing else in the
//! sampled loop is measurable beside it.
//!
//! The bytes are the finding. **842 MB/s of guest memory read into staging, at
//! 2.34 MB per bind**, every second, for a Safari page that is only animating.
//! The render deferred-flush writeback is this device's largest single cost at
//! ~1 GB/s into guest pages — `AGENTS.md` says as much where it explains what
//! retires that rail. This is a second rail of the same order running the other
//! way, and it was undocumented because the arm that drives it was uncounted.
//!
//! Note what the constancy says: 360 and 842.4 MB repeat to the digit across
//! all eight windows, so this is the *same* content re-gathered every frame
//! rather than a changing working set.
//!
//! ## That repair was built, and it is the cache half that fails
//!
//! The paragraph this replaces said the gather path had no content cache at all
//! and that it was *not* established whether the type-11 seed witness could
//! cover these run lists, "the first thing to check before building on this".
//! Both are answered. [`crate::runtime::gather_witness`] covers them —
//! `GatherWindow` carries the window's `gpas` alongside its `runs`, so the
//! page-set question the witness needs is asked in the address space it needs —
//! and a vouch becomes a `GatheredIdentity` the engine binds a retained image
//! on with nothing read and nothing compared. `sampled_gather_skips` counts it
//! working.
//!
//! Which half fails is **not yet established**. A driven x86/PCI drag over 73
//! windows read `sampled_gather_unvouched` at **0** against
//! `sampled_gather_unretained` at 6296, and that was taken to mean every gather
//! had a vouch it could not spend — so the lever was retention rather than the
//! witness. The zero was structural: the emitter was handed
//! `resource.identity.is_some()`, and the producer names every window the
//! witness is asked about, so the arm could not fire on any bind of any boot.
//! The counter now takes the witness's own
//! [`crate::runtime::gather_witness::GatherVouch`], which makes the question
//! answerable for the first time; see
//! [`super::counters::EngineCounters::sampled_gather_unvouched`].
//!
//! The boot that answered it put the split at **5389 compulsory against 2524
//! retention losses** — 68/32, the reverse of the reading above. Two thirds of
//! this rail cannot be reached by any cache, because the witness had spent the
//! generation and no retained image could have matched. And what spends it is
//! this device: `gw_refused_host_write` 5156 against `gw_refused_guest_store`
//! 14 on the same boot. The deferred writeback puts a render target into guest
//! pages and this rail gathers those same pages back at 1.4 MB a bind.
//!
//! So the lever here is neither the witness nor the cache. It is that a window
//! this device just wrote is re-read from guest RAM instead of bound from the
//! image it was written out of — which is what `SampledSource::Target` is for,
//! and that arm is never taken. See
//! [`super::counters::EngineCounters::sampled_gather_unvouched`] for the full
//! reading and its qualifiers.
//!
//! # And none of it holds when the guest is quiet
//!
//! Everything above is one regime. The same phase behaves completely differently
//! in the other, and the difference is where this device's hitches live.
//! Measured on one x86/PCI boot, a driven second against a near-idle one:
//!
//! ```text
//!             draws  acquire_sampled_us  gathers  gather_MB  creates  us/draw
//! driven        660               32399      175      558.4        0      ~49
//! near-idle       9               19233        5        8.9       21    ~2137
//! ```
//!
//! One of those nine draws held **19.2 ms** on its own (`max_us=19475`). Read as
//! bytes that is 0.46 GB/s against 17.2 GB/s driven — 37x apart for the same
//! memcpy, which is not a thing a memcpy does. The arithmetic was wrong, and it
//! was wrong because this phase pooled two populations:
//! `counters.note_sampled_gather` brackets `write_staging_from_runs` alone,
//! while `acquire_sampled_us` also contained `acquire_sampled`'s
//! `vkCreateImage` + `vkAllocateMemory` + `vkCreateImageView` and
//! `acquire_staging`'s `vkCreateBuffer` + `vkAllocateMemory` + `vkMapMemory`.
//! With `creates=21` over nine draws, the second population is not a rounding
//! error there — while in the driven windows it is exactly zero, which is why
//! the eliminations above were sound for that regime and silently wrong for
//! this one.
//!
//! `sampled_upload` is that split: it opens at `acquire_staging` and closes
//! after the gather, so `acquire_sampled` keeps the deciding and creating half
//! and `sampled_upload` gets the byte-moving half. `acquire_sampled_us` staying
//! large with `sampled_upload_us` small convicts object creation on a cold bind;
//! the other way round convicts the gather.
//!
//! The doc for the split above used to say the trail ended here. It ended
//! because one bar was two things.
//!
//! Do not reach for `zc_buffer_gathered` to close this. It is bumped in
//! `try_buffer_zero_copy_resolved` while the *request* is built, covers buffers
//! rather than sampled images, and is therefore not this phase — it was checked
//! and rejected for exactly that reason.
//!
//! A draw that returns early — a decline, a batched deferred submit, a
//! `skip_readback` target — charges its remainder to whichever phase was open,
//! because [`DrawTimer`] commits from `Drop`. That is deliberate: an exit is not
//! a phase, and threading a commit through every `?` would be the one thing
//! guaranteed to go stale.
//!
//! # Why this is a tally and not a decline
//!
//! Per `AGENTS.md`, a census must not be the only record that guest work was
//! lost. Nothing here reports a loss: a slow draw still draws. The one line this
//! module emits outside the per-second aggregate is the *stall* report, and that
//! one is bounded per boot rather than latched per key, because the distribution
//! is the signal — one 950 ms draw and two hundred of them are different bugs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::observe::phase_clock::{charge_ns, to_us};

/// A draw at or above this is a stall rather than a slow frame: at 60 Hz it has
/// already cost six frames, and the guest's compositor blocks on the same lock
/// for the duration. Reported individually with its phase split. Nanoseconds,
/// like the accumulator it is compared against; 100 ms.
const STALL_NS: u64 = 100_000_000;

/// Cap on individual stall reports per boot. The aggregate below keeps counting
/// after this; only the per-event lines stop, so a pathological boot cannot
/// flood the sink it is diagnosed through.
const STALL_REPORT_CAP: u64 = 256;

/// Phase slots, in the order a draw passes through them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    Prep = 0,
    /// Claiming the submission ring slot — `begin_entry`, or a batch joiner
    /// taking the open batch's slot.
    ///
    /// Split out of [`Phase::Prep`] because it is the one part of that span
    /// that **blocks on the GPU**: `begin_entry` advances onto the next slot and
    /// waits when its fence is still unsignaled. `ring_retire_blocks` counts
    /// those, and a count cannot rank them — 18 000 blocks of one microsecond
    /// and 18 000 of four hundred read identically, and only the second says the
    /// ring is the cap. The comment at the claim site has promised a
    /// `retire_wait_us` for a long time; this is it.
    ///
    /// # It was all of it
    ///
    /// First reading, driven Safari drag: `prep_us` **2 525 us/s** against
    /// `slot_us` **314 491 us/s** over 3 845 draws. So the 111 -> 306 ms/s rise
    /// in `prep_us` that followed the resident rungs was not preparation getting
    /// slower by any amount — preparation is free, and the whole column is this
    /// wait. With `ring_retire_blocks` at 17 863 that is roughly **425 us per
    /// block**, which is not the jitter a deeper ring absorbs.
    ///
    /// What that says about `RING_DEPTH` is: probably not the lever. Eight slots
    /// against ~2 100 submissions a second (`batch_flushes` 53 451 over the
    /// boot) gives each submission 3.8 ms to retire, and something is exceeding
    /// it. Doubling the ring doubles that budget once; halving the submissions
    /// would do the same and keep the latency.
    ///
    /// The submission count was the more promising end, and splitting the join
    /// rule by refusing term found it: `!samples_own_target` alone forced
    /// 29.7 % of all draws into their own command buffer, and it was standing
    /// in for a barrier rather than for a real ordering constraint. Dropping it
    /// took `batch_flushes` 55 334 -> 33 538 on the same probe.
    ///
    /// # What that bought, and what it did not
    ///
    /// `slot_us` 6 870 ms -> 5 640 ms and `ring_retire_blocks` 17 533 ->
    /// 14 565 across the boot, so the ring does block less. The frame rate did
    /// not move: 63/s before and after, 24 of 24 seconds below 100 Hz.
    ///
    /// A 39 % cut in submissions buying an 18 % cut in the blocking and no
    /// frames says the ring was not queued behind submission *overhead*. In
    /// the same second the device moves 4.46 GB of guest buffer runs into
    /// device-local memory and writes 4.33 GB of rendered surface back to
    /// guest pages — 8.8 GB/s across the bus on a discrete host, against a
    /// worker that holds the engine lock 671 ms of every second. Look there
    /// before shortening this span again.
    Slot = 1,
    Pipeline = 2,
    Stage = 3,
    StagePass = 4,
    Acquire = 5,
    AcquireSampled = 6,
    SampledUpload = 7,
    AcquireReadback = 8,
    Descriptors = 9,
    Record = 10,
    Submit = 11,
    Wait = 12,
    Readback = 13,
}

impl Phase {
    /// Highest ordinal, so [`PHASES`] is derived from the enum rather than
    /// hand-counted beside it.
    const LAST: Phase = Phase::Readback;
}

const PHASES: usize = Phase::LAST as usize + 1;

/// Nanoseconds, per [`crate::observe::phase_clock`]. `prep_us` and
/// `pipeline_us` are single-digit microseconds over a whole draw, so the spans
/// inside them are well under what a microsecond accumulator can resolve.
static ACC: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
static DRAWS: AtomicU64 = AtomicU64::new(0);
static MAX_NS: AtomicU64 = AtomicU64::new(0);
static STALLS: AtomicU64 = AtomicU64::new(0);
static STALL_LINES: AtomicU64 = AtomicU64::new(0);

/// One window of the split, as taken by the per-second census.
///
/// # These phases account for less than a third of `draw_us`
///
/// Read against `drain_duty` from the same second and the fields below sum to
/// far less than the draw time they sit inside. One driven Safari window-drag
/// second, 1 902 draws: `draw_us=152641` on the `drain_duty` line, against
/// 45 800 us summed here — `record_us=20475`, `pipeline_us=10251`,
/// `stage_us=7501`, `prep_us=2157`, `descriptors_us=1892`, `submit_us=1188`,
/// `acquire_sampled_us=1136`, `sampled_upload_us=596`, `acquire_us=528`, and
/// `wait_us` and `readback_us` both zero. 24 us of the 80 us a draw costs.
///
/// The other 56 us is real and it is not missing from the clock: `draw_us`
/// brackets the whole of `draw::encode_draw_chain`, and every span here is
/// inside the engine call at the end of it. What no field names is the work
/// before that call — binding resolution, the Metal-to-Vulkan translate,
/// texture and buffer resolution, the guest-memory walks. So the largest
/// unowned cost in this device, once the writeback rail's per-window fence was
/// removed, is in `encode_draw_chain` ahead of the engine, and **no phase here
/// can be ranked against it**.
///
/// Which is the point of writing it down rather than acting on it. Naming a
/// suspect inside those 56 us would be a guess, and the ranking above is exactly
/// the shape that makes a guess look informed — `record_us` is the biggest
/// number on the line and it is 13 % of a draw. Add a span to the pre-engine
/// work before optimising any of it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrawPhaseWindow {
    pub prep_us: u64,
    /// Blocked claiming a ring slot. See [`Phase::Slot`].
    pub slot_us: u64,
    pub pipeline_us: u64,
    pub stage_us: u64,
    pub stage_pass_us: u64,
    pub acquire_us: u64,
    pub acquire_sampled_us: u64,
    pub sampled_upload_us: u64,
    pub acquire_readback_us: u64,
    pub descriptors_us: u64,
    pub record_us: u64,
    pub submit_us: u64,
    pub wait_us: u64,
    pub readback_us: u64,
    pub draws: u64,
    pub max_us: u64,
    pub stalls: u64,
}

/// Take and clear the window. `None` when no draw ran, so an idle second costs
/// no line.
pub fn take_window() -> Option<DrawPhaseWindow> {
    let draws = DRAWS.swap(0, Ordering::Relaxed);
    let w = DrawPhaseWindow {
        prep_us: to_us(ACC[Phase::Prep as usize].swap(0, Ordering::Relaxed)),
        slot_us: to_us(ACC[Phase::Slot as usize].swap(0, Ordering::Relaxed)),
        pipeline_us: to_us(ACC[Phase::Pipeline as usize].swap(0, Ordering::Relaxed)),
        stage_us: to_us(ACC[Phase::Stage as usize].swap(0, Ordering::Relaxed)),
        stage_pass_us: to_us(ACC[Phase::StagePass as usize].swap(0, Ordering::Relaxed)),
        acquire_us: to_us(ACC[Phase::Acquire as usize].swap(0, Ordering::Relaxed)),
        acquire_sampled_us: to_us(ACC[Phase::AcquireSampled as usize].swap(0, Ordering::Relaxed)),
        sampled_upload_us: to_us(ACC[Phase::SampledUpload as usize].swap(0, Ordering::Relaxed)),
        acquire_readback_us: to_us(ACC[Phase::AcquireReadback as usize].swap(0, Ordering::Relaxed)),
        descriptors_us: to_us(ACC[Phase::Descriptors as usize].swap(0, Ordering::Relaxed)),
        record_us: to_us(ACC[Phase::Record as usize].swap(0, Ordering::Relaxed)),
        submit_us: to_us(ACC[Phase::Submit as usize].swap(0, Ordering::Relaxed)),
        wait_us: to_us(ACC[Phase::Wait as usize].swap(0, Ordering::Relaxed)),
        readback_us: to_us(ACC[Phase::Readback as usize].swap(0, Ordering::Relaxed)),
        draws,
        max_us: to_us(MAX_NS.swap(0, Ordering::Relaxed)),
        stalls: STALLS.swap(0, Ordering::Relaxed),
    };
    (draws > 0).then_some(w)
}

/// Charges a draw's wall clock to one phase at a time.
///
/// Held by value in `execute_draw_inner`; [`DrawTimer::enter`] closes the open
/// phase and opens the next. The commit is in `Drop` so every exit — including
/// the `?` on a decline — lands its time somewhere.
pub(crate) struct DrawTimer {
    started: Instant,
    last: Instant,
    open: Phase,
    ns: [u64; PHASES],
    /// Set once the draw knows what it is drawing, so a stall report can say
    /// what was on screen rather than only how long it took.
    geom: (u32, u32),
    readback_bytes: u64,
}

impl DrawTimer {
    pub(crate) fn start() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            last: now,
            open: Phase::Prep,
            ns: [0; PHASES],
            geom: (0, 0),
            readback_bytes: 0,
        }
    }

    /// Close the open phase and open `next`.
    pub(crate) fn enter(&mut self, next: Phase) {
        let now = Instant::now();
        self.ns[self.open as usize] += charge_ns(now.duration_since(self.last));
        self.last = now;
        self.open = next;
    }

    /// Context for a stall report. Cheap enough to set unconditionally.
    pub(crate) fn note_target(&mut self, width: u32, height: u32, readback_bytes: u64) {
        self.geom = (width, height);
        self.readback_bytes = readback_bytes;
    }
}

impl Drop for DrawTimer {
    fn drop(&mut self) {
        let now = Instant::now();
        self.ns[self.open as usize] += charge_ns(now.duration_since(self.last));
        let total = charge_ns(now.duration_since(self.started));
        for (slot, acc) in ACC.iter().enumerate() {
            acc.fetch_add(self.ns[slot], Ordering::Relaxed);
        }
        DRAWS.fetch_add(1, Ordering::Relaxed);
        MAX_NS.fetch_max(total, Ordering::Relaxed);
        if total < STALL_NS {
            return;
        }
        STALLS.fetch_add(1, Ordering::Relaxed);
        let line = STALL_LINES.fetch_add(1, Ordering::Relaxed);
        if line >= STALL_REPORT_CAP {
            return;
        }
        let (w, h) = self.geom;
        let latched = if line + 1 == STALL_REPORT_CAP {
            " (last: report cap reached)"
        } else {
            ""
        };
        crate::observe::off(format!(
            "draw_stall us={} prep_us={} slot_us={} pipeline_us={} stage_us={} stage_pass_us={} \
             acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
             descriptors_us={} \
             record_us={} submit_us={} wait_us={} readback_us={} geom={w}x{h} \
             readback_bytes={} exit={:?}{latched}",
            to_us(total),
            to_us(self.ns[Phase::Prep as usize]),
            to_us(self.ns[Phase::Slot as usize]),
            to_us(self.ns[Phase::Pipeline as usize]),
            to_us(self.ns[Phase::Stage as usize]),
            to_us(self.ns[Phase::StagePass as usize]),
            to_us(self.ns[Phase::Acquire as usize]),
            to_us(self.ns[Phase::AcquireSampled as usize]),
            to_us(self.ns[Phase::SampledUpload as usize]),
            to_us(self.ns[Phase::AcquireReadback as usize]),
            to_us(self.ns[Phase::Descriptors as usize]),
            to_us(self.ns[Phase::Record as usize]),
            to_us(self.ns[Phase::Submit as usize]),
            to_us(self.ns[Phase::Wait as usize]),
            to_us(self.ns[Phase::Readback as usize]),
            self.readback_bytes,
            self.open,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every exit commits, and the phase left open at the exit is the one the
    /// remainder is charged to. This is the property that lets `?` returns keep
    /// their time — the alternative (a commit before each return) would silently
    /// drop a phase the next time someone adds an early return.
    #[test]
    fn an_early_exit_charges_its_remainder_to_the_open_phase() {
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Record);
            t.enter(Phase::Wait);
            // Dropped here with `Wait` open — as a `?` on a failed fence would.
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert_eq!(w.draws, 1);
        // Readback never opened, so it must be exactly zero rather than
        // inheriting the tail.
        assert_eq!(w.readback_us, 0);
        assert_eq!(w.submit_us, 0);
    }

    /// The ring claim is charged apart from the bookkeeping above it.
    ///
    /// [`Phase::Slot`] exists because it is the only part of [`Phase::Prep`]
    /// that blocks on the GPU, and the whole point is that a boot can tell the
    /// two apart. A `Slot` whose time landed in `prep_us` would read exactly
    /// like a draw that got more expensive to prepare, and those want opposite
    /// repairs — a deeper ring against less work per draw.
    ///
    /// Walks the ordinals too. They are array indices into [`ACC`], and
    /// inserting a variant renumbers every one below it; a phase left pointing
    /// at its old slot would silently add its time to a neighbour's column.
    #[test]
    fn the_ring_claim_is_charged_apart_from_preparing_the_draw() {
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Slot);
            std::thread::sleep(std::time::Duration::from_millis(3));
            t.enter(Phase::Pipeline);
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert!(w.slot_us >= 2_000, "{w:?}");
        assert_eq!(w.prep_us, 0, "the claim's wait may not read as prepare time");

        // Every ordinal distinct and contiguous from zero, so `PHASES` covers
        // them and no two share an accumulator.
        let all = [
            Phase::Prep,
            Phase::Slot,
            Phase::Pipeline,
            Phase::Stage,
            Phase::StagePass,
            Phase::Acquire,
            Phase::AcquireSampled,
            Phase::SampledUpload,
            Phase::AcquireReadback,
            Phase::Descriptors,
            Phase::Record,
            Phase::Submit,
            Phase::Wait,
            Phase::Readback,
        ];
        assert_eq!(all.len(), PHASES);
        for (want, phase) in all.iter().enumerate() {
            assert_eq!(*phase as usize, want, "{phase:?} indexes the wrong slot");
        }
    }

    /// Holding the render target and holding the sampled images are separate
    /// accumulators, so a draw that spends its time in the sampled loop cannot
    /// be read as an expensive target.
    ///
    /// This is the whole point of the split: the pooled number said "acquire"
    /// and was read as `vkCreateImage` churn, when the per-draw regression said
    /// it could not be. Re-pooling the two would restore exactly that ambiguity.
    ///
    /// The failure this actually guards is a **readout** mis-wiring —
    /// `acquire_sampled_us` taken from `ACC[Phase::Acquire]` — which compiles,
    /// and which reads as a sampled loop that costs nothing. Verified by making
    /// that edit: the assertion below fires. A duplicate discriminant is *not*
    /// what this covers, because rustc rejects it (E0081) before the test runs.
    #[test]
    fn the_sampled_loop_is_charged_apart_from_the_target() {
        // The phases themselves take nanoseconds, so without a measurable sleep
        // every slot reads 0 and the assertions below would pass under a slot
        // collision too. The sleep is what gives the test its power: it is spent
        // entirely inside the sampled loop, so only that slot may carry it.
        const SAMPLED_SLEEP: std::time::Duration = std::time::Duration::from_millis(4);
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Acquire);
            t.enter(Phase::AcquireSampled);
            std::thread::sleep(SAMPLED_SLEEP);
            // Dropped with the sampled loop open, as a decline on a texture
            // that cannot be resolved would leave it.
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert_eq!(w.draws, 1);
        // The sleep landed, so the three slots are genuinely being compared.
        assert!(
            w.acquire_sampled_us >= 2_000,
            "sampled loop lost its own time: {w:?}"
        );
        // The remainder belongs to the phase that was open. `Acquire` closed
        // when the sampled loop opened, so it must not have absorbed it.
        //
        // A ceiling, not zero. `Acquire` was genuinely open across one `enter`
        // call, so its slot legitimately carries however long that took — under
        // a microsecond in a warm process, but 8 µs the first time this test ran
        // as the only test in a cold one. `== 0` was therefore an assertion a
        // correct implementation could fail, and it passed only because the rest
        // of the suite warmed the process first. The bound is half the sleep, so
        // it is derived from the one constant here and cannot be satisfied by a
        // slot that absorbed it: an `Acquire` that swallowed the sampled loop
        // reads the full `SAMPLED_SLEEP`, and one that merely timed an `enter`
        // cannot approach half of it.
        let absorbed_the_sleep = (SAMPLED_SLEEP.as_micros() / 2) as u64;
        assert!(
            w.acquire_us < absorbed_the_sleep,
            "target acquisition charged time the sampled loop spent: {w:?}"
        );
        // Nor may the readback buffer, which is the slot the sampled loop's own
        // time would land in if the two were re-pooled the other way.
        assert_eq!(
            w.acquire_readback_us, 0,
            "readback acquisition charged time the sampled loop spent: {w:?}"
        );
        // Every later phase stays clean, so the tail did not simply spill.
        assert_eq!(w.descriptors_us, 0);
        assert_eq!(w.record_us, 0);
        assert_eq!(w.readback_us, 0);
    }

    /// An idle second must produce no line at all: the census divides against
    /// `drain_duty`, and a zero row there is already reported by `draws=0`.
    #[test]
    fn a_window_with_no_draw_is_none() {
        let _ = take_window();
        assert_eq!(take_window(), None);
    }

    /// The window is a delta, not a running total — two reads of one draw must
    /// not both report it.
    #[test]
    fn taking_the_window_clears_it() {
        let _ = take_window();
        drop(DrawTimer::start());
        assert_eq!(take_window().map(|w| w.draws), Some(1));
        assert_eq!(take_window(), None);
    }
}
