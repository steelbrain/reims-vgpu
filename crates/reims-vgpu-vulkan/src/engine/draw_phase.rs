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
//! | `post_target` | there | target state is published |
//! | `post_store` | inside it | exact guest Store bookkeeping is published |
//! | `post_sampled` | there | sampled-resource state and retains are prepared |
//! | `post_park` | there | cache admission and async cleanup are parked |
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
//! those same historical windows `sampled_gpu_binds`, `sampled_cache_misses`,
//! `sampled_reuploads` and `sampled_cache_hits` were **all 0**, while the former
//! identity-only counter moved at 420/s. That lookup has since been removed:
//! a producer name is not proof that live guest bytes still match a retained
//! copied image. At the time, the reading meant:
//!
//! - The `SampledSource::Target` arm is never taken — it is the only writer of
//!   `sampled_gpu_binds`.
//! - No bind uploads or re-uploads bytes; the `Bytes` arm always hits cache.
//! - The former identity-only lookup was not the ~100 us per bind the arithmetic
//!   demanded.
//!
//! **Do not read the third bullet as "the content path never fires".** A later
//! driven x86/Vulkan boot under a 30 s Safari *drag*, 42 census windows, recorded
//! 26 697 exact-content hits moving 277 MB at ~10 KB a hit. The content rail is
//! load-bearing, and
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
//! ## Guest-page gathers are not retained copies
//!
//! A producer name is not proof that live guest bytes still match a copied
//! image. Neither is the resource-validity transition: the guest consumes its
//! dirty bit when producing a submission table, but unified-memory CPU writes
//! can change the shared storage without emitting a new transition for every
//! change. The full-content audit has observed bytes move while the diagnostic
//! witness reported `Vouched`.
//!
//! Apple samples that storage live. A host whose reported linear image layout
//! matches the guest can do the same through `GuestImage`; where the layouts
//! disagree, this backend must gather every bind until it has another live
//! representation. Retaining the copied image by validity generation is stale
//! content, not a cache.
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

use reims_vgpu_observe::phase_clock::{charge_ns, to_us};

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
    ///
    /// # This span is no longer large, and the reading above is retired
    ///
    /// Everything above is a 2026-07 reading. Two driven macos-13
    /// sustained-animation boots on 2026-08-11, quiesced host, agreeing to 1 %:
    ///
    /// ```text
    ///                 anim1     anim2
    /// slot_us        32.7 ms/s  17.9 ms/s
    /// gpu_span busy  516.9      512.3
    /// draws          29 180/s   28 958/s
    /// drain duty     0.56       0.58
    /// ```
    ///
    /// **`slot_us` is 18-33 ms a second, not 314**, and the GPU is busy 512-517
    /// ms of that second — so the worker's wait is a *twentieth* of the GPU's own
    /// occupancy and the ring is not the constraint on this rail. Whatever the
    /// 314 ms/s was, the join-rule change above and everything since removed it.
    ///
    /// Do not read a large `slot_us` into this device from the numbers above it.
    /// The GPU-side figure they were all inferring is measured directly now, by
    /// [`super::gpu_span`], and it says something different from all of them: at
    /// 51 % occupancy and duty 0.56 **neither the GPU nor the worker is the
    /// pacer**, and the five CPU wins that bought no frames were not absorbed by
    /// `slot_us` — there was nothing to absorb them, because the guest sets the
    /// rate.
    ///
    /// # "The guest sets the rate" is true of that workload and not of this one
    ///
    /// The paragraph above is the correct account of the boots it was taken on,
    /// and it does not generalize. Driven **fullscreen** Maps on the x86 iGPU
    /// runs the drain worker at `busy_us/win` **0.94-0.95** with `gap_idle`
    /// 0.05, against a GPU busy 0.37-0.41. There the worker *is* the pacer,
    /// every microsecond taken out of a draw converts, and a CPU win that buys
    /// no frames means something else is wrong.
    ///
    /// Two readings that follow, because both have misled a session here:
    ///
    /// - **Duty is `busy_us / win_ms`, not `draw_us / win_ms`.** The second is
    ///   the encode span alone and reads about 0.7 on the same boots that are
    ///   at 0.95.
    /// - **GPU `us/draw` on this host is a price, not a cost.** It tracks the
    ///   draw rate at r = -0.956 over thirteen boots while `gpu busy` stays
    ///   flat — the governor clocks up when fed harder. Multiplying it by a
    ///   target draw rate to derive a frame ceiling produces a number that
    ///   assumes today's clocks survive a five-fold higher feed, and nothing
    ///   has measured that.
    ///
    /// So read the workload before reading either paragraph: these are two
    /// populations, and the pacer is not the same one in both.
    Slot = 1,
    /// What is left of the pipeline span once the five below are taken out of
    /// it: building the layout, pass and attribute keys, and resolving the load
    /// action. See [`Phase::PipelineShader`] for why the split exists.
    Pipeline = 2,
    /// `acquire_depth_view` — the one span in the pipeline group that can touch
    /// the image registry rather than a cache.
    PipelineDepth = 14,
    /// Both `get_or_create_shader` calls.
    ///
    /// # Why the pipeline span is split at all
    ///
    /// It is the second-largest phase this device has — 22 % of all draw work on
    /// a driven macos-13 hammer boot — and it is spent almost entirely on cache
    /// **hits**: `pipeline_misses` was 215 of 29 300 draws. A hit is a lookup, so
    /// 37-41 µs of it is a lookup costing what a compile should, and no field
    /// named which of the six lookups in the span it was.
    ///
    /// `shader_hash_words` priced this one first, because keying a module by its
    /// contents means every hit re-walks the contents, and it came back too small
    /// to be the answer: 3 KiB modules, 6 271 bytes a draw, single-digit
    /// microseconds of SipHash. That left ~25 µs a draw unattributed across the
    /// other five, which is what these bars divide.
    PipelineShader = 15,
    /// `get_or_create_layout` and `get_or_create_pass`, the two keyed by a
    /// freshly built key rather than by a handle.
    PipelineLayoutPass = 16,
    /// `get_or_create_pipeline` — the only lookup here whose miss is a driver
    /// compile, so a large bar with `pipeline_misses` near zero is a lookup cost
    /// and a large bar tracking the misses is not.
    PipelineCompile = 17,
    /// The per-sampler `get_or_create_sampler` loop.
    PipelineSampler = 18,
    /// Target resident-state publication after submission (or after choosing
    /// deferred submit).
    PostTarget = 19,
    /// Exact guest Store-footprint publication.
    PostStore = 20,
    /// Sampled-resource state publication and retained-image preparation.
    PostSampled = 21,
    /// Cache admission and parking asynchronous cleanup on the submission slot.
    PostPark = 22,
    /// `buffer_gather_roles` — the pre-pass that classifies every bind before
    /// any of them is resolved.
    StageRoles = 23,
    /// The vertex-attribute bind loop.
    StageVertex = 24,
    /// The index buffer bind.
    StageIndex = 25,
    /// The storage-buffer bind loop.
    StageStorage = 26,
    /// Resolving the colour Load seed into a staging slot.
    StageSeed = 27,
    /// What is left of [`Phase::Stage`] once the five above are taken out of it.
    ///
    /// Carved out rather than added beside, exactly as the pipeline split is, so
    /// the six together are what `stage_us` used to be alone and the total still
    /// divides against `chain_phase`'s `engine_us`.
    ///
    /// This split exists because `stage_us` is the largest bar in the slow half
    /// of a Maps boot — 17.29 µs/draw against 1.57 in the fast half — while
    /// `stage_phase`, which tiles the actual staging *work*, reports under
    /// 100 µs of a whole census second for it, with `gather_us` and `runs_us` at
    /// zero. So almost none of the bar is the byte movement anyone would assume,
    /// and no field named what the rest is.
    Stage = 3,
    StagePass = 4,
    Acquire = 5,
    AcquireSampled = 6,
    SampledUpload = 7,
    AcquireReadback = 8,
    Descriptors = 9,
    /// What is left of the recording span once the five below are taken out of
    /// it. See [`Phase::RecordBarrier`] for why the split exists.
    Record = 10,
    Submit = 11,
    Wait = 12,
    Readback = 13,
    /// Opening the command buffer: `begin_slot_recording`. Zero for a batch
    /// joiner, whose command buffer is already recording.
    RecordBegin = 28,
    /// Everything between the command buffer opening and the render pass
    /// beginning: the guest-gather copies, the resident transitions, the sampled
    /// transitions and the seed copies. **Twenty of this device's
    /// `cmd_pipeline_barrier` call sites are inside this span.**
    ///
    /// # Why the recording span is split at all
    ///
    /// `record_us` is the largest phase this device has — 1.310 µs/draw summed
    /// over 7 396 783 draws across three driven Maps boots, against
    /// `sg_storage_us`'s 1.137 and nothing else above 0.5 — and it was one
    /// undivided bar spanning roughly twelve hundred lines. No field named which
    /// part of the encode it was, so every statement about it was a guess.
    ///
    /// This is the division the pipeline group and the `sg_*` group already got,
    /// for the same reason and in the same shape: carved out of [`Phase::Record`]
    /// rather than added beside it, appended after the existing ordinals so that
    /// none of them moved, and summing back to what `record_us` was alone.
    ///
    /// # And the answer was not the barriers
    ///
    /// First reading, driven Maps boot F0, summed over 2 622 704 draws
    /// (`throttle_ms=0`, `sum` 18.07 µs/draw, 52.2 fps at 1128 draws a frame):
    ///
    /// ```text
    /// rec_draw_us     0.528     vertex/index binds and the draw call
    /// rec_state_us    0.318     pipeline bind, dynamic state, descriptors
    /// rec_pass_us     0.202     continue-or-begin, and cmd_begin_render_pass
    /// rec_begin_us    0.155     begin_slot_recording
    /// rec_barrier_us  0.104     all twenty barrier sites
    /// record_us       0.044     the remainder
    /// ```
    ///
    /// **The barrier region is 8 % of the span it was assumed to dominate**, and
    /// the twenty `cmd_pipeline_barrier` sites that made it look expensive are
    /// nearly free — most of them skip. Do not go hunting for barriers to remove
    /// here; that theory is measured and dead.
    ///
    /// What is left is mostly a floor. `rec_draw` is 39 % of the span and is one
    /// vertex bind plus one draw call per guest draw, which Apple's driver also
    /// issues; `rec_state` is the pipeline and descriptor binds Vulkan requires.
    /// So `record` as a whole is close to irreducible, and a per-draw CPU win has
    /// to come from somewhere else — `sg_storage_us` (1.137 µs/draw) is now the
    /// largest genuinely reducible item this device has.
    ///
    /// # That reading no longer holds, and the seven fields below are why
    ///
    /// A driven macos-13 Maps boot reads `rec_barrier_us` at **4.28 µs/draw**,
    /// 29 % of `engine_us` and 3.7x what `rec_draw_us` costs beside it. That is
    /// 41x the 0.104 above. The paragraph above is kept rather than deleted
    /// because it is still the correct account of the boot it was taken on, and
    /// because it is the reason this span is now divided instead of guessed at
    /// a second time: one number spanning eleven hundred lines told two
    /// sessions two different stories and neither could name a line.
    ///
    /// So the same division the pipeline, staging and recording groups already
    /// got. Seven regions, carved out of this phase rather than added beside
    /// it, appended after the existing ordinals so none of them moved, and
    /// summing back to what `rec_barrier_us` was alone.
    RecordBarrier = 29,
    /// Deciding whether the predecessor's render pass can be continued, and
    /// beginning one when it cannot. The pass-merge census is charged here, so
    /// this is the bar to read against `passdiff_*`.
    RecordPass = 30,
    /// Pipeline bind, dynamic viewport/scissor/stencil, and the descriptor push
    /// or bind — the state a draw needs that is not the draw.
    RecordState = 31,
    /// The vertex and index binds and the `cmd_draw`/`cmd_draw_indexed` itself.
    ///
    /// This is the floor. Whatever else this device stops doing, it still issues
    /// one draw call per guest draw, exactly as Apple's driver does.
    RecordDraw = 32,
    /// Asking the write ledger what outstanding guest writes reach the pages
    /// this draw reads, once the read set is built and the draw is known to read
    /// imported guest memory.
    ///
    /// **This is the largest single cost in the device.** A driven fullscreen
    /// Maps boot on the macos-13 rail charges the region 3.59 µs/draw, 2.3x the
    /// next phase and 16 % of the whole 22.6 µs draw. Three candidates sat
    /// inside it and each has a different fix, so each is now its own ordinal
    /// and this is the residue: [`Phase::RecBarrierReadSet`] is the set build,
    /// [`Phase::RecBarrierImportedTest`] is the predicate that decides whether
    /// to ask at all, and [`Phase::RecBarrierPassBreak`] is what a positive
    /// answer costs.
    ///
    /// What remains here is the ledger question itself — the short-circuit on
    /// the outstanding-write flag, and, when that flag is set, an exact
    /// page-membership walk of every page the read set names under one mutex.
    /// The Maps boot puts that at ~513 pages per draw across 8.2 sets, against
    /// verdicts that are 97.1 % `host_only`: almost every draw pays the ask and
    /// buys nothing. It is attacked by asking a cheaper way, not by asking less
    /// often.
    RecBarrierVisibility = 33,
    /// The attachment-feedback fallback: copying a resident's prior content
    /// into a same-format image before the attachment changes, for the loops
    /// the optional native feedback contract cannot represent. Both the
    /// before-the-pass and after-the-primary-seed halves.
    RecBarrierSnapshot = 34,
    /// Seeding the draw target with what the pass will LOAD — the CPU import,
    /// the GPU present-boundary copy, and the colour-write wait a Clear pass
    /// owes whoever last read the target.
    RecBarrierSeed = 35,
    /// The birth copy an image aliasing guest pages owes, laundered through
    /// staging because the import buffer and the image are two aliases of one
    /// allocation.
    RecBarrierMaterialize = 36,
    /// Transitioning sampled residents in place. The loop is per bound sampled
    /// image and most iterations skip, so this is a walk over the draw's
    /// bindings more than it is a barrier cost.
    RecBarrierResident = 37,
    /// Sampled uploads: the CPU-origin copies, the guest gathers, and the
    /// buffer-to-image copy each one records.
    RecBarrierUpload = 38,
    /// Attachment loads Vulkan's render-pass clear cannot express because Metal
    /// applies a load action to the whole image, plus the MRT secondaries that
    /// must transition back to colour-attachment use.
    RecBarrierAttachment = 39,
    /// Assembling the guest-visibility read set: one entry per guest-backed
    /// vertex attribute, index buffer, storage buffer, sampled source, target
    /// seed, imported target and MRT secondary the draw names.
    ///
    /// Carved out of [`Phase::RecBarrierVisibility`] because the two halves have
    /// opposite fixes and the parent could not choose between them. This half is
    /// a `Vec` and a refcount bump per binding — about 8.5 of them per draw on a
    /// Maps boot — and is attacked by not materializing the set. The residue is
    /// the ledger question that consumes it, ~334 pages per draw under one
    /// mutex, and is attacked by asking it a cheaper way.
    ///
    /// Measured at 0.163 µs/draw on the Maps boot — 4.5 % of the region. The
    /// set build is not where the cost is, so "do not materialize the set" is
    /// the fix this measurement retired.
    RecBarrierReadSet = 40,
    /// The predicate that decides whether this draw reads imported guest memory
    /// at all: a scan of the draw's vertex buffers, index slot, storage slots,
    /// gathers, target, MRT secondaries and sampled bindings, the last of which
    /// asks the registry per binding whether the resident it names aliases guest
    /// pages.
    ///
    /// Carved out of [`Phase::RecBarrierVisibility`] because it is charged on
    /// **every** draw, including the ones that read no guest memory and ask the
    /// ledger nothing, while the rest of the region is charged only on the draws
    /// that get past it. A cost that scales with bindings-per-draw and a cost
    /// that scales with pages-per-draw need different repairs, and averaged
    /// together neither is legible.
    RecBarrierImportedTest = 41,
    /// What a positive visibility answer costs: closing the open render pass so
    /// the global memory dependency can be recorded outside it, the
    /// `cmd_pipeline_barrier` itself, and the reopen the next draw then pays.
    ///
    /// Carved out of [`Phase::RecBarrierVisibility`] because it is the one part
    /// of the region whose cost is not paid per draw but per *overlap*, and the
    /// Maps boot has 46 471 overlaps against 1 550 927 `host_only` verdicts —
    /// 2.9 %. A small rate against a large per-event cost produces the same
    /// per-draw average as a large rate against a small one, and only the split
    /// says which this is. If it is this half, the fix is to stop breaking the
    /// pass; if it is the residue, the fix is in how the ledger is asked.
    RecBarrierPassBreak = 42,
}

impl Phase {
    /// Highest ordinal, so [`PHASES`] is derived from the enum rather than
    /// hand-counted beside it.
    ///
    /// The pipeline, staging and recording sub-phases were each appended after
    /// the ordinals that existed when they were added, rather than inserted next
    /// to the phase they divide, so that every existing ordinal kept its value
    /// and this stayed the only place the count is written.
    const LAST: Phase = Phase::RecBarrierPassBreak;
}

const PHASES: usize = Phase::LAST as usize + 1;

/// Nanoseconds, per [`reims_vgpu_observe::phase_clock`]. `prep_us` and
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
    /// The residue of the pipeline span; the five below are carved out of it and
    /// the six together are what `pipeline_us` used to be alone.
    pub pipeline_us: u64,
    pub pipeline_depth_us: u64,
    pub pipeline_shader_us: u64,
    pub pipeline_layout_pass_us: u64,
    pub pipeline_compile_us: u64,
    pub pipeline_sampler_us: u64,
    pub stage_us: u64,
    /// The five below are carved out of `stage_us`; the six together are what it
    /// used to be alone.
    pub stage_roles_us: u64,
    pub stage_vertex_us: u64,
    pub stage_index_us: u64,
    pub stage_storage_us: u64,
    pub stage_seed_us: u64,
    pub stage_pass_us: u64,
    pub acquire_us: u64,
    pub acquire_sampled_us: u64,
    pub sampled_upload_us: u64,
    pub acquire_readback_us: u64,
    pub descriptors_us: u64,
    pub record_us: u64,
    pub rec_begin_us: u64,
    /// The residue of the barrier span; the nine below are carved out of it and
    /// the ten together are what `rec_barrier_us` used to be alone.
    pub rec_barrier_us: u64,
    pub rb_visibility_us: u64,
    pub rb_snapshot_us: u64,
    pub rb_seed_us: u64,
    pub rb_materialize_us: u64,
    pub rb_resident_us: u64,
    pub rb_upload_us: u64,
    pub rb_attachment_us: u64,
    /// See [`Phase::RecBarrierReadSet`].
    pub rb_read_set_us: u64,
    /// See [`Phase::RecBarrierImportedTest`].
    pub rb_imported_test_us: u64,
    /// See [`Phase::RecBarrierPassBreak`].
    pub rb_pass_break_us: u64,
    pub rec_pass_us: u64,
    pub rec_state_us: u64,
    pub rec_draw_us: u64,
    pub submit_us: u64,
    pub post_target_us: u64,
    pub post_store_us: u64,
    pub post_sampled_us: u64,
    pub post_park_us: u64,
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
        pipeline_depth_us: to_us(ACC[Phase::PipelineDepth as usize].swap(0, Ordering::Relaxed)),
        pipeline_shader_us: to_us(ACC[Phase::PipelineShader as usize].swap(0, Ordering::Relaxed)),
        pipeline_layout_pass_us: to_us(
            ACC[Phase::PipelineLayoutPass as usize].swap(0, Ordering::Relaxed),
        ),
        pipeline_compile_us: to_us(ACC[Phase::PipelineCompile as usize].swap(0, Ordering::Relaxed)),
        pipeline_sampler_us: to_us(ACC[Phase::PipelineSampler as usize].swap(0, Ordering::Relaxed)),
        stage_us: to_us(ACC[Phase::Stage as usize].swap(0, Ordering::Relaxed)),
        stage_roles_us: to_us(ACC[Phase::StageRoles as usize].swap(0, Ordering::Relaxed)),
        stage_vertex_us: to_us(ACC[Phase::StageVertex as usize].swap(0, Ordering::Relaxed)),
        stage_index_us: to_us(ACC[Phase::StageIndex as usize].swap(0, Ordering::Relaxed)),
        stage_storage_us: to_us(ACC[Phase::StageStorage as usize].swap(0, Ordering::Relaxed)),
        stage_seed_us: to_us(ACC[Phase::StageSeed as usize].swap(0, Ordering::Relaxed)),
        stage_pass_us: to_us(ACC[Phase::StagePass as usize].swap(0, Ordering::Relaxed)),
        acquire_us: to_us(ACC[Phase::Acquire as usize].swap(0, Ordering::Relaxed)),
        acquire_sampled_us: to_us(ACC[Phase::AcquireSampled as usize].swap(0, Ordering::Relaxed)),
        sampled_upload_us: to_us(ACC[Phase::SampledUpload as usize].swap(0, Ordering::Relaxed)),
        acquire_readback_us: to_us(ACC[Phase::AcquireReadback as usize].swap(0, Ordering::Relaxed)),
        descriptors_us: to_us(ACC[Phase::Descriptors as usize].swap(0, Ordering::Relaxed)),
        record_us: to_us(ACC[Phase::Record as usize].swap(0, Ordering::Relaxed)),
        rec_begin_us: to_us(ACC[Phase::RecordBegin as usize].swap(0, Ordering::Relaxed)),
        rec_barrier_us: to_us(ACC[Phase::RecordBarrier as usize].swap(0, Ordering::Relaxed)),
        rb_visibility_us: to_us(
            ACC[Phase::RecBarrierVisibility as usize].swap(0, Ordering::Relaxed),
        ),
        rb_snapshot_us: to_us(ACC[Phase::RecBarrierSnapshot as usize].swap(0, Ordering::Relaxed)),
        rb_seed_us: to_us(ACC[Phase::RecBarrierSeed as usize].swap(0, Ordering::Relaxed)),
        rb_materialize_us: to_us(
            ACC[Phase::RecBarrierMaterialize as usize].swap(0, Ordering::Relaxed),
        ),
        rb_resident_us: to_us(ACC[Phase::RecBarrierResident as usize].swap(0, Ordering::Relaxed)),
        rb_upload_us: to_us(ACC[Phase::RecBarrierUpload as usize].swap(0, Ordering::Relaxed)),
        rb_attachment_us: to_us(
            ACC[Phase::RecBarrierAttachment as usize].swap(0, Ordering::Relaxed),
        ),
        rb_read_set_us: to_us(ACC[Phase::RecBarrierReadSet as usize].swap(0, Ordering::Relaxed)),
        rb_imported_test_us: to_us(
            ACC[Phase::RecBarrierImportedTest as usize].swap(0, Ordering::Relaxed),
        ),
        rb_pass_break_us: to_us(
            ACC[Phase::RecBarrierPassBreak as usize].swap(0, Ordering::Relaxed),
        ),
        rec_pass_us: to_us(ACC[Phase::RecordPass as usize].swap(0, Ordering::Relaxed)),
        rec_state_us: to_us(ACC[Phase::RecordState as usize].swap(0, Ordering::Relaxed)),
        rec_draw_us: to_us(ACC[Phase::RecordDraw as usize].swap(0, Ordering::Relaxed)),
        submit_us: to_us(ACC[Phase::Submit as usize].swap(0, Ordering::Relaxed)),
        post_target_us: to_us(ACC[Phase::PostTarget as usize].swap(0, Ordering::Relaxed)),
        post_store_us: to_us(ACC[Phase::PostStore as usize].swap(0, Ordering::Relaxed)),
        post_sampled_us: to_us(ACC[Phase::PostSampled as usize].swap(0, Ordering::Relaxed)),
        post_park_us: to_us(ACC[Phase::PostPark as usize].swap(0, Ordering::Relaxed)),
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
        reims_vgpu_observe::off(format!(
            "draw_stall us={} prep_us={} slot_us={} pipeline_us={} stage_us={} stage_pass_us={} \
             acquire_us={} acquire_sampled_us={} sampled_upload_us={} acquire_readback_us={} \
             descriptors_us={} \
             record_us={} submit_us={} post_target_us={} post_store_us={} post_sampled_us={} \
             post_park_us={} wait_us={} readback_us={} geom={w}x{h} \
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
            to_us(self.ns[Phase::PostTarget as usize]),
            to_us(self.ns[Phase::PostStore as usize]),
            to_us(self.ns[Phase::PostSampled as usize]),
            to_us(self.ns[Phase::PostPark as usize]),
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
        assert_eq!(
            w.prep_us, 0,
            "the claim's wait may not read as prepare time"
        );

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
            // Appended after `Readback` rather than placed next to `Pipeline`,
            // so the ordinals above kept their values when the pipeline span was
            // divided. Declaration order here is ordinal order, not reading
            // order.
            Phase::PipelineDepth,
            Phase::PipelineShader,
            Phase::PipelineLayoutPass,
            Phase::PipelineCompile,
            Phase::PipelineSampler,
            Phase::PostTarget,
            Phase::PostStore,
            Phase::PostSampled,
            Phase::PostPark,
            // The staging split, appended for the same reason the pipeline one
            // was: every ordinal above keeps its value.
            Phase::StageRoles,
            Phase::StageVertex,
            Phase::StageIndex,
            Phase::StageStorage,
            Phase::StageSeed,
            // The recording split, appended for the same reason the pipeline and
            // staging ones were.
            Phase::RecordBegin,
            Phase::RecordBarrier,
            Phase::RecordPass,
            Phase::RecordState,
            Phase::RecordDraw,
            // The barrier split, appended for the same reason all three above
            // were.
            Phase::RecBarrierVisibility,
            Phase::RecBarrierSnapshot,
            Phase::RecBarrierSeed,
            Phase::RecBarrierMaterialize,
            Phase::RecBarrierResident,
            Phase::RecBarrierUpload,
            Phase::RecBarrierAttachment,
            Phase::RecBarrierReadSet,
            Phase::RecBarrierImportedTest,
            Phase::RecBarrierPassBreak,
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

    /// The seven barrier sub-phases are **carved out of** `rec_barrier_us`, not
    /// added beside it.
    ///
    /// This is the invariant that makes the division readable at all. A reader
    /// sums the eight to get what `rec_barrier_us` alone used to be, and ranks
    /// them against each other to find where the 4.28 µs a draw goes. A
    /// sub-phase that also charged the residue would double its own time and
    /// read as the largest bar on the line — which is precisely the shape of the
    /// wrong answer this division exists to stop being possible.
    #[test]
    fn a_barrier_sub_phase_is_carved_out_of_the_residue() {
        const REGION_SLEEP: std::time::Duration = std::time::Duration::from_millis(4);
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::RecordBarrier);
            t.enter(Phase::RecBarrierResident);
            std::thread::sleep(REGION_SLEEP);
            t.enter(Phase::RecordPass);
        }
        let w = take_window().expect("one draw ran");
        assert_eq!(w.draws, 1);
        // The sleep landed where it was spent.
        assert!(
            w.rb_resident_us >= 2_000,
            "the resident region lost its own time: {w:?}"
        );
        // And the residue did not also charge it. Same bound and same reasoning
        // as `the_sampled_loop_is_charged_apart_from_the_target`: a ceiling
        // rather than zero, because `RecordBarrier` was legitimately open across
        // one `enter` call.
        let absorbed_the_sleep = (REGION_SLEEP.as_micros() / 2) as u64;
        assert!(
            w.rec_barrier_us < absorbed_the_sleep,
            "the barrier residue charged time a sub-phase spent: {w:?}"
        );
        // Nor did any sibling. A slot collision between two of the nine is the
        // failure this catches that reading the enum cannot.
        for (name, value) in [
            ("rb_visibility_us", w.rb_visibility_us),
            ("rb_snapshot_us", w.rb_snapshot_us),
            ("rb_seed_us", w.rb_seed_us),
            ("rb_materialize_us", w.rb_materialize_us),
            ("rb_upload_us", w.rb_upload_us),
            ("rb_attachment_us", w.rb_attachment_us),
            ("rb_read_set_us", w.rb_read_set_us),
            ("rb_imported_test_us", w.rb_imported_test_us),
            ("rb_pass_break_us", w.rb_pass_break_us),
        ] {
            assert_eq!(value, 0, "{name} shares a slot with the resident region");
        }
        // And the phase that follows the span did not absorb the sleep either,
        // so the tail did not simply spill past the division. A ceiling rather
        // than zero for the same reason as the residue: `RecordPass` was open
        // when the timer dropped and legitimately carries that instant.
        assert!(
            w.rec_pass_us < absorbed_the_sleep,
            "the pass phase charged time a sub-phase spent: {w:?}"
        );
    }

    /// The three costs inside the visibility region must land in three
    /// different accumulators.
    ///
    /// They are charged at different rates — the predicate on every draw, the
    /// ledger question on the draws that read guest memory, the pass break on
    /// the 2.9 % of those that overlap — so averaging any two together hides
    /// which of a large per-event cost and a large event rate is present. That
    /// ambiguity is the reason for the carve, and a slot collision between two
    /// of them would reintroduce it while still reading as a plausible profile.
    #[test]
    fn the_visibility_region_charges_its_three_costs_apart() {
        const REGION_SLEEP: std::time::Duration = std::time::Duration::from_millis(4);
        let absorbed = (REGION_SLEEP.as_micros() / 2) as u64;
        for (name, sleep_in) in [
            ("rb_imported_test_us", Phase::RecBarrierImportedTest),
            ("rb_read_set_us", Phase::RecBarrierReadSet),
            ("rb_visibility_us", Phase::RecBarrierVisibility),
            ("rb_pass_break_us", Phase::RecBarrierPassBreak),
        ] {
            let _ = take_window();
            {
                let mut t = DrawTimer::start();
                t.enter(Phase::RecordBarrier);
                t.enter(sleep_in);
                std::thread::sleep(REGION_SLEEP);
                t.enter(Phase::RecordPass);
            }
            let w = take_window().expect("one draw ran");
            let charged = [
                ("rb_imported_test_us", w.rb_imported_test_us),
                ("rb_read_set_us", w.rb_read_set_us),
                ("rb_visibility_us", w.rb_visibility_us),
                ("rb_pass_break_us", w.rb_pass_break_us),
            ];
            for (field, value) in charged {
                if field == name {
                    assert!(value >= 2_000, "{field} lost the time it was given: {w:?}");
                } else {
                    assert!(value < absorbed, "{field} shares a slot with {name}: {w:?}");
                }
            }
        }
    }

    /// Store publication must not be charged to the driver submit bar. The
    /// split exists because the two require different fixes, and a swapped
    /// readout would send a real-traffic profile back toward queue batching.
    #[test]
    fn post_submit_store_work_has_its_own_bar() {
        let _ = take_window();
        {
            let mut t = DrawTimer::start();
            t.enter(Phase::Submit);
            t.enter(Phase::PostTarget);
            t.enter(Phase::PostStore);
            std::thread::sleep(std::time::Duration::from_millis(3));
            t.enter(Phase::PostTarget);
        }
        let w = take_window().expect("a dropped timer counts a draw");
        assert!(w.post_store_us >= 2_000, "Store bar lost its work: {w:?}");
        assert!(
            w.submit_us < 1_500,
            "Store bookkeeping was charged to queue submission: {w:?}"
        );
        assert_eq!(w.post_sampled_us, 0);
        assert_eq!(w.post_park_us, 0);
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
