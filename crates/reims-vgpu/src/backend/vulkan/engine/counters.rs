//! Always-on create/alloc and cache hit/miss counters (reuse-gate proxies).
//!
//! # The vocabulary is declared once
//!
//! [`engine_counters!`] takes the counter names and generates the six things
//! that used to spell them out separately: the atomic [`EngineCounters`], the
//! plain-`u64` [`CounterSnapshot`], and the four whole-vocabulary walks
//! [`EngineCounters::snapshot`], [`EngineCounters::reset`],
//! [`CounterSnapshot::delta_since`] and [`CounterSnapshot::delta_fields`].
//!
//! Writing a hundred names six times is how a counter silently stops working,
//! and none of the failure modes is a compile error or a log line:
//!
//! * missing from `reset` — the counter reports a lifetime total into a reader
//!   that asked for a window, so a per-second rate reads as monotonically rising;
//! * missing from `delta_since` — the field reads **zero in every delta**, which
//!   is indistinguishable from "this path never ran". That is the
//!   "an event count is not a state" trap in `AGENTS.md` with the count itself
//!   broken.
//! * missing from the emitted line — the counter is correct and nobody can read
//!   it, which is the same thing from the outside. This is the one that actually
//!   happened: `engine_delta` named its fields by hand and had fallen **35
//!   counters** behind the vocabulary, so a run built to read four of them
//!   measured nothing and reported a clean zero. `delta_fields` is generated for
//!   that reason, and the emitter now walks it instead of naming anything.
//!
//! All five lists were checked against each other before this collapse and all
//! five agreed, so the macro changes no behaviour. What it changes is that they
//! can no longer disagree.
//!
//! The four groups are a real distinction, not a formatting one. `windowed` is
//! zeroed by `reset()`; `cumulative` deliberately survives it, because a
//! device-loss count is a fact about the boot and not about the measurement
//! window, and only `reset_all()` clears it; `pool_sourced` has no atomic at all
//! and is merged in from `ResourcePools` by `engine::counter_snapshot`.
//!
//! `pool_levels` is merged in the same way as `pool_sourced` but is not a rate:
//! every field in it is an absolute high-water or an instantaneous level, so
//! subtracting two readings of one yields nonsense that reads as a plausible
//! zero. That is why the group exists rather than a comment saying so —
//! `delta_since` carries these through unchanged, `delta_fields` omits them, and
//! `census::emit_registry_pressure` prints them absolute beside the prose that
//! says how each is read. Putting a level in the wrong group is now the only way
//! to get it onto the per-interval line, and that is a visible edit rather than
//! an omission.
//!
//! # A field with no named reader is not dead
//!
//! [`CounterSnapshot`] is named field-by-field only by the integration tests in
//! `tests/`; the census reaches it through the generated `delta_fields`, which
//! mentions no name. So a sweep for "fields nobody references" still reports
//! most of this struct. Twenty-seven of the
//! seventy-one came back that way. Do not act on that sweep: the struct derives
//! `Debug` and every assertion in those tests prints the *whole* snapshot on
//! failure (`"...: {d:?}"`), so the unasserted fields are the diagnostic context
//! that makes a failing assertion readable.
//!
//! That is not hypothetical. The ring-wrap defect in the sampled content cache
//! was diagnosed from `sampled_free_allocs` and `sampled_recycle_cap_drops` in
//! one such dump — neither is asserted by any test, and both named the
//! mechanism. Deleting them would have cost more than the lines saved.
//!
//! The exception, and the only thing removed on those grounds, is a field **no
//! code path increments**: it is zero by construction, so it carries no
//! information in a dump either. `seed_imports` and `target_stale_import` were
//! removed for exactly that. Before deleting any other field here, check that
//! something can still make it nonzero.
//!
//! Adding one is still governed by the census-versus-decline rule in
//! `AGENTS.md`: a counter must not be the only record that guest work was lost.
//! These are reuse and cache proxies — tallies of successful work — and the
//! refusal paths they sit near report themselves with typed declines.

use std::sync::atomic::{AtomicU64, Ordering};

/// How much of its render target one draw could have written.
///
/// The standing instrument for "would bounding what a writeback copies pay
/// anything?" — the deferred-flush rail's largest named lever, until it was
/// built, measured at zero, and removed. It is a
/// property of the *guest's* draws and not of any rail here, which is why it
/// outlives the rail: the answer changes with the workload, and nothing else in
/// this device can be read for it.
///
/// See [`EngineCounters::note_draw_coverage`] for the arithmetic that turns
/// these into a verdict.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrawCoverage {
    /// The pass did not load the target, so it rewrote the whole attachment
    /// before the draw began — a CLEAR load action, or a whole-frame CPU seed
    /// standing in for one.
    Full,
    /// The pass loaded the target and bound a scissor covering all of it. The
    /// draw could have written any texel.
    LoadedFullScissor,
    /// The pass loaded the target and bound a scissor smaller than it. The only
    /// arm whose writes are bounded by anything.
    LoadedPartialScissor,
}

/// Declare the engine counter vocabulary once; see the module docs for why.
///
/// Doc comments written on a name here land on *both* the atomic field and its
/// snapshot field, which is why the snapshot no longer has to repeat them.
macro_rules! engine_counters {
    (
        windowed { $($(#[$wm:meta])* $win:ident,)* }
        cumulative { $($(#[$cm:meta])* $cum:ident,)* }
        pool_sourced { $($(#[$pm:meta])* $pool:ident,)* }
        pool_levels { $($(#[$lm:meta])* $lvl:ident,)* }
    ) => {
        /// Process-wide product-path counters (resettable for tests).
        #[derive(Debug, Default)]
        pub struct EngineCounters {
            $($(#[$wm])* pub $win: AtomicU64,)*
            $($(#[$cm])* pub $cum: AtomicU64,)*
        }

        /// One reading of [`EngineCounters`], plus the pool-owned tallies
        /// `engine::counter_snapshot` merges in.
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct CounterSnapshot {
            $($(#[$wm])* pub $win: u64,)*
            $($(#[$cm])* pub $cum: u64,)*
            $($(#[$pm])* pub $pool: u64,)*
            $($(#[$lm])* pub $lvl: u64,)*
        }

        impl EngineCounters {
            /// Read every counter at once. The `pool_sourced` fields have no
            /// atomic here and stay zero; `engine::counter_snapshot` fills them
            /// from `ResourcePools` immediately after calling this.
            pub fn snapshot(&self) -> CounterSnapshot {
                CounterSnapshot {
                    $($win: self.$win.load(Ordering::Relaxed),)*
                    $($cum: self.$cum.load(Ordering::Relaxed),)*
                    $($pool: 0,)*
                    $($lvl: 0,)*
                }
            }

            /// Zero the windowed counters, leaving the cumulative ones alone.
            pub fn reset(&self) {
                $(self.$win.store(0, Ordering::Relaxed);)*
            }

            /// Zero everything, including the counters `reset` preserves.
            pub fn reset_all(&self) {
                self.reset();
                $(self.$cum.store(0, Ordering::Relaxed);)*
            }
        }

        impl CounterSnapshot {
            /// This reading minus an earlier one, field by field. Saturating
            /// because a `reset` between the two readings makes `earlier`
            /// larger, and a window of "no work" must read 0 rather than wrap.
            ///
            /// `pool_levels` fields are **carried through unchanged**, not
            /// subtracted: a high-water mark minus an earlier high-water mark is
            /// not a high-water mark, and it reads as 0 for the rest of a boot
            /// once the true maximum is behind the window. So the difference of
            /// two readings still holds the level, and a caller that prints one
            /// gets the answer rather than a plausible zero.
            pub fn delta_since(&self, earlier: &CounterSnapshot) -> CounterSnapshot {
                CounterSnapshot {
                    $($win: self.$win.saturating_sub(earlier.$win),)*
                    $($cum: self.$cum.saturating_sub(earlier.$cum),)*
                    $($pool: self.$pool.saturating_sub(earlier.$pool),)*
                    $($lvl: self.$lvl,)*
                }
            }

            /// Every field that is a **rate** — one `(name, value)` pair per
            /// counter that means "this many since the last reading", in
            /// declaration order.
            ///
            /// This is what `runtime::drain::census` prints as `engine_delta`,
            /// and it exists so that line cannot drift from the vocabulary. It
            /// used to be a hand-written format string naming its fields twice,
            /// and it had silently fallen 35 counters behind: a name added to
            /// this macro got a field, a reset and a delta, and then printed
            /// nowhere. That reads exactly like a path that never runs, which is
            /// the same trap this module's docs describe for a missing
            /// `delta_since` arm, one step further along — and it cost a
            /// twelve-boot A/B run that could not answer its own question,
            /// because the four counters it was built to read were among them.
            ///
            /// `pool_levels` is excluded on purpose. Those are absolute
            /// high-waters, they are not per-interval, and `registry_pressure`
            /// prints them beside the prose that says how to read each one.
            pub fn delta_fields(&self) -> Vec<(&'static str, u64)> {
                vec![
                    $((stringify!($win), self.$win),)*
                    $((stringify!($cum), self.$cum),)*
                    $((stringify!($pool), self.$pool),)*
                ]
            }
        }
    };
}

engine_counters! {
    windowed {
        creates,
        allocs,
        shader_hits,
        shader_misses,
        /// SPIR-V words walked by [`super::caches::Caches::get_or_create_shader`]
        /// before it can look anything up, summed over every call including the
        /// hits.
        ///
        /// Keying a module by its contents means the key costs a pass over the
        /// contents, and this device asks for two of them — the storage-image
        /// capability derivation and the digest — on every draw, for both stages,
        /// whether or not the module is already cached. `shader_hits` alone reads
        /// as a working cache and says nothing about that, the same way
        /// `sampled_cache_hits` read as working until `sampled_cache_hit_bytes`
        /// priced it: a hit over a 2 KiB module and a hit over an 88 KiB one are
        /// the same count and forty times the work.
        ///
        /// Divided by the census window this is a bandwidth, which is the form
        /// that can be compared against what a hashing pass over memory costs.
        shader_hash_words,
        /// `shader_hits` that never walked the module at all, because
        /// `get_or_create_shader_memoized` recognised the allocation its words
        /// live in and already knew the digest.
        ///
        /// Read as a *fraction of* `shader_hits`, which is the only form that
        /// says anything: the two together are the front index's hit rate, and
        /// `shader_hash_words` beside them is what the walks that remain cost.
        /// A boot where this sits well below `shader_hits` has a draw path
        /// handing the walking form an allocation it does not hold — which is a
        /// correctness-neutral regression that nothing else would report.
        shader_digest_hits,
        layout_hits,
        layout_misses,
        pass_hits,
        pass_misses,
        pipeline_hits,
        /// Positive render-pipeline lookups answered by the exact one-entry
        /// front index without hashing the composite pipeline key.
        pipeline_front_hits,
        /// Positive render-pipeline lookups answered by the exact last variant
        /// attached to this retained guest pipeline object's identity.
        pipeline_object_hits,
        pipeline_misses,
        sampler_hits,
        sampler_misses,

        // --- compute ---
        compute_pipeline_hits,
        compute_pipeline_misses,
        dispatches,
        fence_timeouts,
        /// Compute sampled-image bytes staged for host→device upload.
        compute_sampled_uploads,
        compute_sampled_upload_bytes,
        /// Compute storage-image seed bytes staged for host→device upload.
        compute_storage_seed_uploads,
        compute_storage_seed_upload_bytes,
        /// Sampled inputs seeded by a device-local copy of a resident storage
        /// image (copy-on-sample) — bytes are the elided host upload size.
        compute_sampled_resident_copies,
        compute_sampled_resident_copy_bytes,
        /// Compute storage images whose post-dispatch readback was deferred —
        /// the pinned resident stays authoritative; bytes are the elided
        /// device→host readback size (the CPU writeback of the same size is
        /// elided too).
        compute_deferred_writebacks,
        compute_deferred_writeback_bytes,
        /// Deferred-flush reads (read_resident_storage): the on-access GPU→host
        /// copy that lands deferred content in guest pages.

        // --- residency / oracle I/O ---
        /// Device→host copies taken as the tail of a draw or a compute dispatch,
        /// i.e. work a submission did for itself.
        ///
        /// Deliberately *not* pooled with `target_reads`. A composite Store that
        /// takes `skip_readback` moves its copy from here to there rather than
        /// deleting it, so one number over both populations cannot say whether the
        /// deferral worked — it reads the same either way.
        ///
        /// **This is three rails, not one**, and the split below is what tells
        /// them apart. A driven macos-13 x86/Vulkan boot reads `computes` at
        /// 1288, so the claim this doc used to carry — that `computes` is 0 on a
        /// desktop workload and therefore this is the draw rail alone — is false
        /// on at least one first-class pathway. The total is still the total;
        /// what it cannot do is say which rail to go and look at.
        readbacks,
        readback_bytes,
        /// Readback taken as the tail of a draw ([`ReadbackSource::DrawTail`]).
        readback_draw,
        readback_draw_bytes,
        /// Readback of a compute dispatch's writable storage *buffers*
        /// ([`ReadbackSource::ComputeBuffer`]).
        ///
        /// These bytes end in guest pages: `ComputeOutput::buffers` is consumed
        /// by `writeback_buffer`, so the rail is GPU → host staging → `Vec<u8>`
        /// → guest, two host passes for a guest destination.
        readback_compute_buffer,
        readback_compute_buffer_bytes,
        /// Readback of a compute dispatch's storage *images*
        /// ([`ReadbackSource::ComputeImage`]).
        ///
        /// Also guest-destined, via `writeback_texture`. This is the rail that
        /// had a direct arm — the dispatch wrote an imported view of the
        /// caller's guest window and the caller skipped its own writeback —
        /// which went with the deferred-flush window and was not replaced when
        /// the render side gained `render_writeback`. Sizing that regression is
        /// what this counter exists for.
        readback_compute_image,
        readback_compute_image_bytes,
        /// Full-frame reads of a pinned resident through `read_target`: the present
        /// capture and the deferred render window's on-access flush.
        ///
        /// These are the copies a deferred rail *keeps*, paid once when a consumer
        /// asks instead of once per Store. `target_reads / readbacks` is what
        /// separates "the readback moved" from "the readback went away".
        ///
        /// **Host deliveries only.** A full-frame read that lands in the guest's
        /// own pages on the GPU queue is charged to `target_gpu_copies` below,
        /// and [`TargetReadDelivery`] is what keeps the two apart.
        target_reads,
        target_read_bytes,
        /// Full-frame reads of a pinned resident that never entered host address
        /// space: `copy_target_to_guest_pages` puts them straight into the
        /// guest's imported pages on the queue, so the bytes are what the host
        /// copy *would* have cost.
        ///
        /// Split out because pooling them with `target_reads` made the one
        /// instrument that answers "is this device zero-copy?" read the same
        /// whether the frame crossed into this process or not — the GPU rail
        /// inflated the host total by gigabytes a boot and looked like the host
        /// rail still running. Read the pair: `target_reads` falling while
        /// `target_gpu_copies` rises is the rail moving, and both falling is the
        /// work going away.
        target_gpu_copies,
        target_gpu_copy_bytes,
        /// Completion stamps whose word was recorded into the GPU queue behind
        /// the writebacks they follow, rather than stored by this thread after
        /// blocking on them.
        ///
        /// Read against `readback_split`'s `fence`: together they say which rail
        /// each stamp took. Zero while windows are flushing means every stamp
        /// fell back to the blocking rail — no host-pointer import, no
        /// `timelineSemaphore`, or a stamp page that would not resolve — and the
        /// reason is on the fail channel.
        gpu_stamps,
        seed_uploads,
        seed_upload_bytes,
        /// Present-boundary seeds satisfied by a GPU resident→target image copy
        /// (no CPU front-frame read, no seed upload); bytes = elided upload size.
        seed_gpu_copies,
        seed_gpu_copy_bytes,
        sampled_reuploads,
        sampled_reupload_bytes,
        /// Actual sampled upload bytes, partitioned by decoded API resource.
        /// These fields sum to `sampled_reupload_bytes`.
        sampled_reupload_attachment_bytes,
        sampled_reupload_buffer_texture_bytes,
        sampled_reupload_surface_view_bytes,
        sampled_reupload_surface_cache_bytes,
        sampled_reupload_surface_guest_bytes,
        sampled_reupload_linear_texture_bytes,
        sampled_reupload_synthetic_bytes,
        /// Sampled binds served by gathering scattered guest pages into staging
        /// (`SampledSource::GuestRuns`), and the bytes those gathers moved.
        ///
        /// Every other arm of the sampled loop already reported itself and this
        /// one did not, which is how `acquire_sampled` came to be measured at
        /// the whole of a draw's acquire cost with no counter accounting for
        /// it. See `draw_phase`'s "What the sampled loop's own cost is *not*".
        sampled_gathers,
        sampled_gather_bytes,
        /// Sampled binds that would have gathered and did not, because both
        /// halves of the guest-write witness vouched that the retained image's
        /// bytes could not have moved. Bytes = the gather that did not happen,
        /// so `sampled_gather_bytes + sampled_gather_skip_bytes` is what this
        /// rail would cost with no cache.
        sampled_gather_skips,
        sampled_gather_skip_bytes,
        /// Sampled binds the GPU read straight out of the guest's own pages
        /// through the imported RAMBlock — no CPU gather, no staging scratch.
        ///
        /// The third disposition of a `SampledSource::GuestRuns` bind, ranked
        /// against `sampled_gather_skips` (bound a retained image, moved
        /// nothing) and `sampled_gathers` (the CPU packed the texels). Bytes are
        /// what the copy names, which is what the CPU no longer moves.
        sampled_guest_imports,
        sampled_guest_import_bytes,
        /// Why a `SampledSource::GuestRuns` bind moved bytes instead of binding
        /// a retained image, split at the only two things that can go wrong.
        ///
        /// `sampled_gather_skips` says how often the elision worked; on a driven
        /// boot it says 0 for twenty-three consecutive windows while the rail
        /// moves 424 MB/s, and neither the engine nor the witness could say
        /// which half was at fault. The witness's own `gw_vouched` is a
        /// *runtime*-side per-window tally over every rail, so it cannot be
        /// subtracted from an engine-side per-bind one — that is the "a counter
        /// and a fail line count different things" trap with two counters.
        ///
        /// These two are engine-side and per-bind, so they divide the same
        /// population `sampled_gather_skips` is drawn from and the split adds
        /// up:
        ///
        /// ```text
        /// unvouched + unretained + skips == gathers + imports + skips
        /// ```
        ///
        /// which is the identity `the_unskipped_reason_is_exactly_one_counter_per_bind`
        /// holds the emitter to. The fixes are opposite, which is why one bar
        /// could not serve: `unvouched` is the witness refusing, while
        /// `unretained` is a vouch this device could not spend because no image
        /// answered to it.
        ///
        /// # The first boot's zero was the instrument, not the witness
        ///
        /// Driven x86/PCI Safari drag, quiesced, one `vk_caps`, 73 census
        /// windows with a gather in them:
        ///
        /// ```text
        /// sampled_gather_unvouched      0
        /// sampled_gather_unretained  6296   (== gathers 6292 + imports 4)
        /// ```
        ///
        /// That zero was read as "the witness vouches for every guest-run bind
        /// it is asked about; not one gather happened because it refused", and
        /// the rail's whole cost was attributed to retention on the strength of
        /// it. **It proved nothing.** The emitter was fed
        /// `resource.identity.is_some()`, which is not the witness's verdict but
        /// the producer's: `note_gather` returned `Option<GatheredIdentity>` and
        /// built it by reading the generation back out of the witness map, which
        /// holds an entry for every key `observe` is handed. There was no path
        /// through it that returned `None`. The counter could not fire, and a
        /// counter that cannot fire reading zero is exactly the "a drop counter
        /// reading zero is not a measurement" trap.
        ///
        /// It now takes [`crate::runtime::gather_witness::GatherVouch`], decided
        /// beside the assignment that spends the generation, so `unvouched`
        /// means the witness spent one and the following miss was compulsory.
        ///
        /// # The measured split, and it is mostly not the cache
        ///
        /// Driven x86/PCI Safari drag, quiesced, one `vk_caps`, 166 census
        /// windows, 25 s of real compositing:
        ///
        /// ```text
        /// sampled_gather_unvouched   5389   68.1%
        /// sampled_gather_unretained  2524   31.9%
        ///                            ----
        ///                            7913  == gathers 7909 + imports 4
        /// sampled_gather_skips       3526   (of 11 439 guest-run binds)
        /// ```
        ///
        /// Both arms fire and the identity holds exactly, so the instrument is
        /// non-vacuous for the first time. **Roughly two thirds of this rail's
        /// re-gathers are compulsory**: the witness spent the generation, so no
        /// retained image could have answered and no size of sampled cache
        /// reaches them. Only the 2524 are a cache result. That is the reverse
        /// of what the structural zero was read as, and it is consistent with
        /// the `SAMPLED_REACH_BAND` A/B, where four times the entries and six
        /// times the bytes left the miss rate where it started.
        ///
        /// The 9876 count-cap evictions on the same boot are not evidence of a
        /// cache too small either. A compulsory miss admits an entry under its
        /// fresh identity exactly as a lost one does, and nothing ever looks
        /// that entry up again, so most of the churn is the rail working.
        ///
        /// # What spends the generation is this device, not the guest
        ///
        /// Same boot, the witness's own verdict routes:
        ///
        /// ```text
        /// gw_vouched             6050
        /// gw_refused_host_write  5156
        /// gw_refused_guest_store   14
        /// gw_unarmed              212
        /// gw_rearm                128
        /// gw_audit_unsound          0
        /// ```
        ///
        /// 368:1. The guest barely writes the windows it samples; **this device
        /// writes them**, and each write is what forces the next bind to read
        /// 1.4 MB back out of guest RAM. The deferred writeback rail puts a
        /// render target into guest pages and this rail gathers those same pages
        /// back, so the two largest byte movers in the device are feeding each
        /// other. `gw_audit_unsound` at 0 says the witness is sound while it
        /// does so — nothing vouched had moved.
        ///
        /// Read the ratio, not an attribution. These `gw_*` tallies count
        /// `note_gather` calls (window resolutions) while the two counters above
        /// count binds, so the populations differ even though all three
        /// [`crate::runtime::gather_witness::GatherRail`] variants are sampled
        /// rails and no other caller exists. Subtracting one from the other is
        /// still invalid, which is why the split had to be taken engine-side.
        /// One workload, one boot, one pathway: x86/PCI on a discrete GPU, where
        /// the render target lives in VRAM and the writeback is real. A unified
        /// host has no such round trip to make.
        ///
        /// The skip rate is a property of the workload and not of the rail, so
        /// do not carry a number for it: one boot's drag ran 23 windows at 0%
        /// while the next ran ~41% (168 skips against 244 gathers a window) on
        /// the same build. What is stable across both is this split.
        sampled_gather_unvouched,
        /// Vouched by the witness, but the sampled cache held no image under
        /// `(key, identity)`. See [`sampled_gather_unvouched`].
        ///
        /// [`sampled_gather_unvouched`]: EngineCounters::sampled_gather_unvouched
        sampled_gather_unretained,
        /// How much of its target each draw could have written, split three
        /// ways. See [`DrawCoverage`] and [`EngineCounters::note_draw_coverage`].
        draw_cover_full,
        draw_cover_loaded_full_scissor,
        draw_cover_loaded_partial_scissor,
        /// Vertex/storage/index buffer binds the draw pointed straight at the
        /// guest's own pages through the imported RAMBlock, with no copy in
        /// either direction. Ranked against `buffer_snapshot_binds` and the
        /// `stage_phase` `runs_*` bars, which are what the CPU still gathers.
        buffer_guest_imports,
        buffer_guest_import_bytes,
        /// Index-buffer subset of the direct-import totals above. Kept apart
        /// because indexed draws previously copied this resource through a CPU
        /// `Vec` and therefore appeared in neither zero-copy disposition.
        buffer_guest_index_imports,
        buffer_guest_index_import_bytes,
        /// Vertex/storage buffer binds the GPU assembled out of the guest's own
        /// pages, one `VkBufferCopy` per GPA-contiguous stretch, into a
        /// device-local destination the draw then binds.
        ///
        /// The disposition between `buffer_guest_imports` (nothing copied) and
        /// the `stage_phase` `runs_*` bars (the CPU copied). Bytes are what the
        /// copies name, which is what the CPU no longer moves; regions divided
        /// by gathers is the mean stretch count.
        ///
        /// There is no ceiling on the region count and there must not be one.
        /// A run is a whole number of guest pages, so every region this rail
        /// adds also removes at least one page from the `memcpy` in the
        /// `runs_*` bars — the two costs move in opposite directions, and no
        /// region count exists at which the CPU arm becomes the cheaper one. A
        /// cap here refuses the widest windows, which are exactly the ones with
        /// the most to gain. See `zc_buf_runs_*` for the live distribution.
        buffer_guest_gathers,
        buffer_guest_gather_bytes,
        buffer_guest_gather_regions,
        /// Gathered windows consumed only by fixed-function vertex fetch.
        buffer_guest_gather_vertex_bytes,
        /// Gathered windows consumed only through a shader storage-buffer bind.
        buffer_guest_gather_storage_bytes,
        /// Gathered windows consumed by both fixed-function vertex fetch and a
        /// shader storage-buffer bind. Counted once, like the physical gather.
        buffer_guest_gather_shared_bytes,
        /// Gather bytes consumed only by fixed-function index fetch.
        buffer_guest_gather_index_bytes,
        /// Compute dispatches the buffer gather issued in place of those
        /// regions, and the plans that could not become one.
        ///
        /// **`buffer_guest_gather_regions` above counts regions *planned*, not
        /// issued**, because it is charged where the window is planned and
        /// before either form is chosen. A dispatch boot therefore still reports
        /// ~245 000 of them while issuing none — do not read that column as
        /// transfer traffic. It is a property of the workload (how scattered the
        /// guest's buffers are) and stays comparable across the two arms and
        /// across builds, which is what makes it useful; it is simply not the
        /// column that says which form ran.
        ///
        /// **This one is.** `dispatches > 0` means the dispatch path ran for the
        /// whole batch; `declined > 0` means a window's arithmetic was refused
        /// and every gather in that command buffer took the transfer regions,
        /// which is all-or-nothing because the two forms need different
        /// barriers. Both zero on a boot with gathers means the switch is off or
        /// the host cannot import.
        buffer_gather_dispatches,
        buffer_gather_declined,
        /// Buffer binds served from a copy the command buffer being recorded
        /// already holds — see `ResourcePools::cb_bound_buffers`.
        ///
        /// Read against `buffer_guest_gathers + buffer_guest_imports` plus
        /// itself: the three sum to the binds, so this is the share of them
        /// that cost nothing. It was a per-*draw* map before, and the reuse it
        /// found then was invisible because it never reached a counter at all;
        /// a reading of this is only about reuse *across* draws of one command
        /// buffer if it is taken against a boot's `batch_flush_draws /
        /// batch_flushes`, which says how many draws there were to reuse over.
        buffer_bind_reuses,
        /// Reuses above whose consumer is the indexed-draw input.
        buffer_index_bind_reuses,
        /// Graphics state a draw did **not** record because the command buffer
        /// it joined was already carrying exactly it — see
        /// `ResourcePools::CbGraphicsState`.
        ///
        /// Every draw asks all four questions, so each of these is out of
        /// `chain_phase chains` and none can exceed it. `dynstate_pipeline_held`
        /// is the one to read first: a pipeline change clears the other three by
        /// construction, so it is the ceiling on them and a boot where it is
        /// near zero is a boot where consecutive draws never share a pipeline
        /// and this whole cache is inert.
        ///
        /// `dynstate_stencil_held` is out of the *stencil* draws rather than all
        /// of them — a draw with no stencil state asks nothing and counts
        /// nowhere — so it is the one that does not belong to the same
        /// denominator.
        dynstate_pipeline_held,
        dynstate_viewport_held,
        dynstate_scissor_held,
        dynstate_stencil_held,
        /// Vertex-buffer binding slots requested by draws. This must equal
        /// `vertex_buffer_bind_emitted`; compare either with
        /// `vertex_buffer_bind_calls` to measure contiguous bulk encoding.
        vertex_buffer_bind_slots,
        /// Requested vertex-buffer slots actually handed to Vulkan. Kept
        /// separately so a future optimization cannot silently turn bulk
        /// encoding into dropped guest state.
        vertex_buffer_bind_emitted,
        /// `vkCmdBindVertexBuffers` calls used to emit those slots.
        vertex_buffer_bind_calls,
        /// Draw/dispatch descriptor state recorded directly into the command
        /// buffer through `VK_KHR_push_descriptor`.
        descriptor_pushes,
        /// Graphics pushes omitted because the exact layout and descriptor
        /// values were already present in the recording command buffer.
        descriptor_push_held,
        /// Descriptor sets updated through the Vulkan 1.2 fallback path.
        descriptor_set_updates,
        /// Updated descriptor sets subsequently bound for execution.
        descriptor_set_binds,
        sampled_cache_hits,
        sampled_identity_hits,
        sampled_cache_hit_bytes,
        sampled_cache_misses,
        sampled_gpu_binds,
        /// Batched-draw guest-run buffer binds the CPU had to gather, because
        /// the host could not import the pages' RAMBlock or the span sits at an
        /// offset this device will not bind at.
        ///
        /// A subset of the `stage_phase` `runs_*` bars, distinguished by *when*
        /// the bytes were read: a batched CB reads them at record time and an
        /// immediate one effectively at submit, which is a real difference in
        /// how stale a snapshot can be. `buffer_guest_imports` is the other
        /// disposition of the same bind, where nothing was read at all.
        buffer_snapshot_binds,
        gpu_load_hits,
        target_evicts,
        /// Descriptor-arena growth events: a new pool block was appended because
        /// every existing block was exhausted (cap-pressure signal; 0 = no growth).
        desc_pool_grow,
        gen_mismatch,
        /// Post-submit fence waits skipped by all-deferred compute dispatches.
        compute_post_wait_skips,
        /// Post-submit fence waits skipped by no-readback (resident-target) draws.
        render_post_wait_skips,
        /// Entries that found the ring full and had to block on the oldest
        /// in-flight fence in begin_entry. This fires only when RING_DEPTH
        /// consecutive no-wait entries outrun the GPU.
        ring_retire_blocks,
        /// Draw batching (deferred submit): draws that OPENED a batch (left their
        /// CB recording), draws that JOINED an open batch (skipped
        /// begin_entry+submit entirely), batch submits, and total draws carried by
        /// those submits (avg batch length = batch_flush_draws / batch_flushes).
        batch_opens,
        batch_joins,
        batch_flushes,
        batch_flush_draws,
        /// End-of-drain-tranche batching cost, in microseconds. `lock` is time
        /// spent acquiring the engine mutex and `call` is the complete
        /// `ResourcePools::batch_flush` beneath it. Read their sum against
        /// `drain_duty tail_us`; the remainder is the cheap latch/recovery
        /// control flow around the call.
        batch_tail_lock_us,
        batch_tail_call_us,
        /// Successful end-of-tranche flushes, and the elapsed-time band in
        /// which the next draw batch opened. The five reopen bands partition
        /// the tail-flush population that was followed by another batch.
        batch_tail_flushes,
        batch_tail_reopen_le100us,
        batch_tail_reopen_le1ms,
        batch_tail_reopen_le4ms,
        batch_tail_reopen_le16ms,
        batch_tail_reopen_gt16ms,
        /// Phase attribution for every submitted draw batch, in microseconds.
        /// Close includes ending the open render pass and sealing its GPU-span
        /// query; end is `vkEndCommandBuffer`; submit is the ordered-owner
        /// handoff; finish parks cleanup and publishes recorded sampled
        /// resources. `queue_async_driver_us` separately retains the host queue
        /// call cost moved off the drain worker.
        batch_flush_close_us,
        batch_flush_end_us,
        batch_flush_submit_us,
        batch_flush_finish_us,
        /// Draws recorded inside the render pass instance their preceding draw
        /// left open for the same decoded Metal render encoder.
        render_pass_continuations,
        /// Readbacks that appended their copy to a batch that was still
        /// recording, and so were submitted with it instead of behind it.
        ///
        /// This counted the opposite before the append path existed: the same
        /// population, as *flushes a readback forced*. A driven boot read it at
        /// 58.8 % of all `batch_flushes`, with batches averaging 1.77 draws
        /// against a `BATCH_MAX_DRAWS` of 8 — so nearly every readback was
        /// ending a run of draws to buy itself a second `vkQueueSubmit`. Each
        /// one counted here is now one submission rather than two.
        ///
        /// Read against `batch_flushes` for the share that collapses. Do **not**
        /// expect `batch_flush_draws / batch_flushes` to move with it — it read
        /// 1.77 before the append path and 1.78 after, because a readback still
        /// ends the batch it joined. A readback arriving with no batch open is
        /// not counted and has nothing to collapse.
        ///
        /// Counted at the readback sites rather than inside `batch_flush`,
        /// because that function cannot see who called it and threading a reason
        /// through `begin_entry` would put a diagnostic in the signature of the
        /// device's hottest slot claim.
        batch_readback_joins,
        /// Completion stamps parked for the open draw batch's eventual
        /// submission point.
        gpu_stamp_batch_points,
        /// Completion stamps that found their FIFO's bounded pending ring full
        /// and therefore submitted the open batch to make room.
        gpu_stamp_pressure_flushes,
        /// Completion stamps that reused the newest successful FIFO submission
        /// because no draw batch remained open.
        gpu_stamp_reused_points,
        /// Guest-page writebacks that detiled through the device-local scratch
        /// and scattered with plain buffer copies — one region per guest
        /// stretch instead of up to three rectangles per stretch.
        ///
        /// Read as a share of `guest_write_linear + guest_write_rects`. A boot
        /// where the second term dominates is one whose guest pitches carry row
        /// padding, the single case the linear form cannot express: a run's
        /// bytes then include padding this copy must not write, and a buffer
        /// copy has no way to skip it.
        guest_write_linear,
        /// Guest-page writebacks that went straight to guest RAM as image-copy
        /// rectangles, because the window's rows carry padding.
        guest_write_rects,
        /// Copy regions submitted by the two above, summed.
        ///
        /// The number the linear path exists to reduce, and — since nothing
        /// caps a writeback's width any more — the only account of how wide one
        /// gets. Divide by the writeback count for regions per frame: a 1080p
        /// window is ~507 stretches, so ~507 here means every frame took the
        /// linear path and ~1500 means none did.
        guest_write_regions,
        /// Compute dispatches submitted by the linear path's scatter, one per
        /// destination buffer.
        ///
        /// Read against `guest_write_linear`, which it equals on an ordinary
        /// one-RAMBlock machine where every linear writeback dispatched. The
        /// pair with `guest_write_regions` is the whole reading: a boot on the
        /// dispatch reads ~1 region per linear writeback (the detile) and ~1
        /// dispatch, where one on the transfer scatter reads ~507 regions and
        /// zero.
        guest_write_dispatches,
        /// Linear writebacks that planned a dispatch, could not, and took the
        /// transfer regions.
        ///
        /// A healthy zero. Any firing is a run whose geometry the kernel cannot
        /// express or a window wider than the driver binds, and each one is a
        /// whole frame on the expensive path — the fail-channel record names
        /// which check refused.
        guest_write_scatter_declined,
    }

    cumulative {
        /// Cumulative across the boot: a device loss is a fact about this run,
        /// not about the measurement window, so `reset()` leaves it standing.
        device_lost,
        /// Cumulative across the boot, for the same reason as `device_lost`.
        recreates,
    }

    pool_sourced {
        /// Ended draw-batch submissions executed by the queue owner, time they
        /// spent queued before its host call, and time spent inside that call.
        /// `batch_flush_submit_us` is the drain worker's handoff cost; these
        /// fields retain the driver cost moved off that worker.
        queue_async_submits,
        queue_async_queue_us,
        queue_async_driver_us,
        /// Ordered window display transactions, time queued behind earlier GPU
        /// work, and time in their submit-plus-present host driver calls.  The
        /// queue and driver times are deliberately separate from
        /// `engine_lock`: neither is paid while the resource registry is held.
        queue_present_transactions,
        queue_present_queue_us,
        queue_present_driver_us,
        /// Sampled-cache pool recycle diagnostics (workstream D lag tail). These
        /// four come from `ResourcePools`, not the atomic counters — merged in by
        /// `engine::counter_snapshot`. `free_hits` = `acquire_sampled` reused a
        /// recycled slot (no `vkAllocateMemory`); `free_allocs` = it had to create
        /// a fresh image; `recycle_admits` = an evicted slot rejoined the per-key
        /// free list; `recycle_cap_drops` = an evicted slot was destroyed because
        /// the per-key cap was full (raising the cap would have kept it). A high
        /// `free_allocs` with a high `recycle_cap_drops` means the cap is the
        /// limiter; a high `free_allocs` with low admits means the drain timing is.
        sampled_free_hits,
        sampled_free_allocs,
        sampled_recycle_admits,
        sampled_recycle_cap_drops,
        /// Resident render-target recycle diagnostics (same shape as the sampled
        /// ones). `target_free_hits` = a create reused a recycled image (no
        /// `vkCreateImage`/`vkAllocateMemory`); `target_free_allocs` = it had to
        /// allocate fresh; `target_recycle_admits`/`target_recycle_cap_drops` =
        /// displaced images that rejoined / overflowed the per-key free list. Owned
        /// by `ResourcePools`; merged in by `engine::counter_snapshot` (zero here).
        target_free_hits,
        target_free_allocs,
        target_recycle_admits,
        target_recycle_cap_drops,
    }

    pool_levels {
        /// Current resident registry population and attachment bytes.
        registry_current_count,
        registry_current_bytes,
        /// Current unpinned residents whose content is reproducible.
        registry_recoverable_count,
        registry_recoverable_bytes,
        /// Current residents held by one or more deferred-work pins.
        registry_pinned_count,
        registry_pinned_bytes,
        /// High-water mark of the non-pinned resident population, in slots.
        ///
        /// The demand, with no ceiling to read it against: this population is
        /// bounded by the allocator refusing, not by a count — see
        /// `ResourcePools::recoverable_residents`. A peak exists rather than an
        /// instantaneous population because the interesting shape is a
        /// compositing *burst*, and a burst that rises and drains between two
        /// census samples is exactly what an instantaneous reading misses.
        ///
        /// Cumulative and never reset by the windowed reset, because the
        /// question is "how far did this boot ever reach", not "where is it
        /// now". `EngineCounters::reset_all` clears it for tests.
        ///
        /// Sampled at both admit paths, so every admission that could grow the
        /// population is seen. Two prior readings are quoted in this module's
        /// neighbours — a non-pinned peak of ~260 under a YouTube page-load, and
        /// `reg=512/512 evicts=168` before pinned slots were excluded from the
        /// since-retired count — and **neither is reproducible today**: nothing
        /// in the tree emitted them, so they are historical probe output rather
        /// than something a boot can be asked for. That is the gap this closes.
        registry_non_pinned_peak,
        /// Worst gap, in milliseconds, between a resident being touched and
        /// being read again. Diagnostic only; residency does not branch on it.
        ///
        /// Cumulative high-water, like `registry_non_pinned_peak` and for the
        /// same reason: the question is how close this boot ever came, and a gap
        /// that peaks between two census samples is what an instantaneous
        /// reading misses.
        ///
        /// Read beside the `resident_resample_*` bands, which give the shape of
        /// the distribution this is the tail of. The bands alone could not tell
        /// a worst case of 1.0 s from one of 1.9 s against a 2 s cutoff.
        resident_resample_peak_ms,
        /// `VkDeviceMemory` the DEVICE_LOCAL image slab holds right now, and the
        /// carved half of it, in bytes.
        ///
        /// A *level*, not a total: this is what the device is holding at the
        /// sample, and it can go down. Every other memory reading in this crate
        /// is one of the two things that cannot answer "did that policy change
        /// cost VRAM" — `vk_alloc_sites` is cumulative-allocated and only ever
        /// grows, and `registry_non_pinned_peak_bytes` is an attachment
        /// footprint computed from geometry, blind to tiling padding, slab
        /// rounding and the empty blocks the pool deliberately retains.
        ///
        /// `held` minus `carved` is that retention: blocks the driver has given
        /// this device that hold nothing.
        slab_held_bytes,
        slab_carved_bytes,
        /// The same high-water in attachment bytes, sampled from the same
        /// population at the same instant as `registry_non_pinned_peak`.
        ///
        /// A slot count once bounded this population, and this is the counter
        /// that retired it: 320 slots is 5 MiB of 16x16 scratch or 10 GiB of 4K,
        /// and nothing in this device could tell those apart until the bytes were
        /// sampled beside the slots. A lower bound on VRAM (attachment
        /// footprint, no tiling padding, and a format with no single texel size
        /// contributes nothing), which is the safe direction for a figure that
        /// exists to decide whether a bound measures the right quantity.
        registry_non_pinned_peak_bytes,
        /// High-water of the population both reclaim paths refuse to take
        /// because the image is the only place its pixels exist, in slots and in
        /// the same attachment bytes as `registry_non_pinned_peak_bytes`.
        ///
        /// Read as a ratio against `registry_non_pinned_peak`. That ratio is the
        /// price of never losing a frame: near 0 the reclaim paths have their
        /// usual freedom and the protection costs nothing; near 1 they have
        /// nothing left to take, the allocation-failure retry has nothing to
        /// give back, and the copy-out sites are what needs work.
        registry_sole_copy_peak,
        registry_sole_copy_peak_bytes,
        /// The two above over the compute-storage registry. Separate rather than
        /// summed with them: that registry holds standalone `VkDeviceMemory`
        /// where this one holds slab suballocations, and a boot needs to know
        /// which of the two an allocation failure would have found something in.
        compute_storage_sole_copy_peak,
        compute_storage_sole_copy_peak_bytes,
    }
}
/// Where a full-frame read of a pinned resident put the bytes.
///
/// The two are the same `vkCmdCopyImage*` shape and the same byte count, and
/// they are opposite answers to this project's only question: one crosses into
/// this process's address space and one does not. A single total over both is
/// unreadable, and `copy_target_to_guest_pages` charged the host counter for
/// gigabytes a boot that no CPU ever touched.
///
/// An argument rather than two methods, for the reason [`CreateSite`] is one:
/// a new read site cannot be added without saying which of the two it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetReadDelivery {
    /// Into a mapped staging slot the CPU then reads — a real device→host copy.
    Host,
    /// Into the guest's imported pages, on the queue, with no host pass at all.
    GuestPagesOnGpu,
}

/// Which rail a device→host readback was the tail of.
///
/// `readbacks` pools three call sites that answer different questions, and the
/// pooled number was read for years as the draw rail alone on the strength of a
/// doc claim that `computes` is 0 on a desktop workload. A driven macos-13
/// x86/Vulkan boot puts `computes` at 1288, so that reading was wrong on a
/// first-class pathway and nothing in the census could show it.
///
/// The distinction is not cosmetic. Two of the three end in **guest pages** —
/// `ComputeOutput`'s buffers and images are consumed by `writeback_buffer` and
/// `writeback_texture` — so they are a zero-copy failure with a known repair
/// ([`super::copy_target_to_guest_pages`]), while a draw tail may not be. A
/// single total cannot be used to size that repair.
///
/// An argument rather than three methods, for the reason [`TargetReadDelivery`]
/// is one: a new readback site cannot be added without saying which rail it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadbackSource {
    /// The tail of a draw submission.
    DrawTail,
    /// A compute dispatch's writable storage buffers.
    ComputeBuffer,
    /// A compute dispatch's storage images.
    ComputeImage,
}

macro_rules! create_sites {
    ($($variant:ident => $slug:literal),+ $(,)?) => {
        /// The Vulkan object lifetime a successful create call belongs to.
        ///
        /// A single `creates` total cannot distinguish a pipeline compiled once
        /// from a framebuffer rebuilt per draw. Every create charged to the
        /// total carries this type, so adding an unclassified charge fails to
        /// compile instead of silently widening the remainder of a census.
        #[repr(usize)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum CreateSite {
            $($variant),+
        }

        const CREATE_SITES: &[(CreateSite, &str)] = &[
            $((CreateSite::$variant, $slug)),+
        ];

        impl CreateSite {
            fn index(self) -> usize {
                self as usize
            }
        }
    };
}

create_sites! {
    ShaderModule => "shader_module",
    DescriptorSetLayout => "descriptor_set_layout",
    PipelineLayout => "pipeline_layout",
    RenderPass => "render_pass",
    Sampler => "sampler",
    GraphicsPipeline => "graphics_pipeline",
    ComputePipeline => "compute_pipeline",
    StorageImage => "storage_image",
    StorageImageView => "storage_image_view",
    RegistryFramebuffer => "registry_framebuffer",
    RegistryImportedImage => "registry_imported_image",
    RegistryImage => "registry_image",
    RegistryImageView => "registry_image_view",
    MrtImage => "mrt_image",
    MrtImageView => "mrt_image_view",
    DepthImage => "depth_image",
    DepthImageView => "depth_image_view",
    MrtFramebuffer => "mrt_framebuffer",
    CommandPool => "command_pool",
    DescriptorPool => "descriptor_pool",
    Fence => "fence",
    StagingBuffer => "staging_buffer",
    GatherBuffer => "gather_buffer",
    ReadbackBuffer => "readback_buffer",
    TargetImage => "target_image",
    TargetImageView => "target_image_view",
    TargetFramebuffer => "target_framebuffer",
    GuestSampledImage => "guest_sampled_image",
    GuestSampledImageView => "guest_sampled_image_view",
    SampledImage => "sampled_image",
    SampledImageView => "sampled_image_view",
    QueryPool => "query_pool",
}

static CREATE_SITE_COUNTS: [AtomicU64; CREATE_SITES.len()] =
    [const { AtomicU64::new(0) }; CREATE_SITES.len()];
static CREATE_COUNT: AtomicU64 = AtomicU64::new(0);
const CREATE_EMIT_EVERY: u64 = 512;

fn emit_create_site_census() {
    use std::fmt::Write as _;
    let mut line = String::from("vk_create_sites");
    for (site, name) in CREATE_SITES {
        let _ = write!(
            line,
            " {name}={}",
            CREATE_SITE_COUNTS[site.index()].load(Ordering::Relaxed)
        );
    }
    crate::observe::off(line);
}

/// The `note_*` helpers: the increments that are not a bare `fetch_add(1)` at
/// the call site, because they move a count and a byte total together.
impl EngineCounters {
    pub(crate) fn note_create(&self, site: CreateSite) {
        self.creates.fetch_add(1, Ordering::Relaxed);
        CREATE_SITE_COUNTS[site.index()].fetch_add(1, Ordering::Relaxed);
        let count = CREATE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(CREATE_EMIT_EVERY) {
            emit_create_site_census();
        }
    }

    pub fn note_alloc(&self) {
        self.allocs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_readback(&self, bytes: u64, source: ReadbackSource) {
        self.readbacks.fetch_add(1, Ordering::Relaxed);
        self.readback_bytes.fetch_add(bytes, Ordering::Relaxed);
        let (count, total) = match source {
            ReadbackSource::DrawTail => (&self.readback_draw, &self.readback_draw_bytes),
            ReadbackSource::ComputeBuffer => (
                &self.readback_compute_buffer,
                &self.readback_compute_buffer_bytes,
            ),
            ReadbackSource::ComputeImage => (
                &self.readback_compute_image,
                &self.readback_compute_image_bytes,
            ),
        };
        count.fetch_add(1, Ordering::Relaxed);
        total.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_target_read(&self, bytes: u64, delivery: TargetReadDelivery) {
        let (count, total) = match delivery {
            TargetReadDelivery::Host => (&self.target_reads, &self.target_read_bytes),
            TargetReadDelivery::GuestPagesOnGpu => {
                (&self.target_gpu_copies, &self.target_gpu_copy_bytes)
            }
        };
        count.fetch_add(1, Ordering::Relaxed);
        total.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_seed_upload(&self, bytes: u64) {
        self.seed_uploads.fetch_add(1, Ordering::Relaxed);
        self.seed_upload_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_reupload(&self, bytes: u64, origin: super::types::SampledByteOrigin) {
        self.sampled_reuploads.fetch_add(1, Ordering::Relaxed);
        self.sampled_reupload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        let split = match origin {
            super::types::SampledByteOrigin::AttachmentAlias => {
                &self.sampled_reupload_attachment_bytes
            }
            super::types::SampledByteOrigin::BufferBackedTexture => {
                &self.sampled_reupload_buffer_texture_bytes
            }
            super::types::SampledByteOrigin::SerializedSurfaceView => {
                &self.sampled_reupload_surface_view_bytes
            }
            super::types::SampledByteOrigin::SurfaceHostCache => {
                &self.sampled_reupload_surface_cache_bytes
            }
            super::types::SampledByteOrigin::SurfaceGuestFallback => {
                &self.sampled_reupload_surface_guest_bytes
            }
            super::types::SampledByteOrigin::LinearTexture => {
                &self.sampled_reupload_linear_texture_bytes
            }
            super::types::SampledByteOrigin::Synthetic => &self.sampled_reupload_synthetic_bytes,
        };
        split.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_gather(&self, bytes: u64) {
        self.sampled_gathers.fetch_add(1, Ordering::Relaxed);
        self.sampled_gather_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub(super) fn note_buffer_bind_reused(&self, role: super::exec::BufferGatherRole) {
        self.buffer_bind_reuses.fetch_add(1, Ordering::Relaxed);
        if role.includes_index() {
            self.buffer_index_bind_reuses
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn note_buffer_guest_import(&self, bytes: u64, role: super::exec::BufferGatherRole) {
        self.buffer_guest_imports.fetch_add(1, Ordering::Relaxed);
        self.buffer_guest_import_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        if role.includes_index() {
            self.buffer_guest_index_imports
                .fetch_add(1, Ordering::Relaxed);
            self.buffer_guest_index_import_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
    }

    pub(super) fn note_buffer_guest_gather(
        &self,
        bytes: u64,
        regions: u64,
        role: super::exec::BufferGatherRole,
    ) {
        self.buffer_guest_gathers.fetch_add(1, Ordering::Relaxed);
        self.buffer_guest_gather_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.buffer_guest_gather_regions
            .fetch_add(regions, Ordering::Relaxed);
        let counter = if role.is_shared() {
            &self.buffer_guest_gather_shared_bytes
        } else if role.includes_index() {
            &self.buffer_guest_gather_index_bytes
        } else if role.is_storage_only() {
            &self.buffer_guest_gather_storage_bytes
        } else {
            &self.buffer_guest_gather_vertex_bytes
        };
        counter.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record how much of its target one draw could have written.
    ///
    /// # Reading the three against `surface_flush`
    ///
    /// A damage rect can only pay when a flushed surface receives *no*
    /// whole-surface write between two flushes, because any one of those makes
    /// the union total. So the verdict is a comparison of rates, not a ratio of
    /// these three to each other:
    ///
    /// ```text
    ///   (draw_cover_full + draw_cover_loaded_full_scissor) per second
    ///   ---------------------------------------------------------- << 1
    ///                    flushes per second
    /// ```
    ///
    /// On a driven x86 Safari window-drag boot it was **4.4**, not «1: 840
    /// clears and 1 637 full-scissor draws against 560 flushes, so every flush
    /// interval held several whole-surface writes and a rect built over this
    /// would have copied whole surfaces anyway. That is what was measured when
    /// the rail existed, by a `flush_rows` / `flush_surface_rows` pair that read
    /// exactly equal on every census line of the boot.
    ///
    /// `draw_cover_loaded_partial_scissor` was 1 718 in the same second — 41% of
    /// all draws — which is why the ratio and not that number is the test. A
    /// workload can bind mostly-partial scissors and still leave nothing to
    /// narrow.
    pub fn note_draw_coverage(&self, coverage: DrawCoverage) {
        let field = match coverage {
            DrawCoverage::Full => &self.draw_cover_full,
            DrawCoverage::LoadedFullScissor => &self.draw_cover_loaded_full_scissor,
            DrawCoverage::LoadedPartialScissor => &self.draw_cover_loaded_partial_scissor,
        };
        field.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_sampled_guest_import(&self, bytes: u64) {
        self.sampled_guest_imports.fetch_add(1, Ordering::Relaxed);
        self.sampled_guest_import_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_gather_skipped(&self, bytes: u64) {
        self.sampled_gather_skips.fetch_add(1, Ordering::Relaxed);
        self.sampled_gather_skip_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record why a guest-run sampled bind is about to move bytes, taken at the
    /// one site that already knows both halves: the elision lookup returned
    /// nothing, and `vouch` says whether the identity it looked up could ever
    /// have matched. Call exactly once per bind that falls through the skip,
    /// before the import/gather disposition is decided — the two questions are
    /// independent and the identity in
    /// [`EngineCounters::sampled_gather_unvouched`] holds only if this is not
    /// also called on the skip path.
    ///
    /// Takes the witness's own verdict rather than a `bool` the caller derives.
    /// The caller derived it from `identity.is_some()` for one boot, which is
    /// not the witness's answer but the producer's, and the producer names every
    /// window it is handed.
    pub fn note_sampled_gather_unskipped(
        &self,
        vouch: crate::runtime::gather_witness::GatherVouch,
    ) {
        let field = if vouch.is_vouched() {
            &self.sampled_gather_unretained
        } else {
            &self.sampled_gather_unvouched
        };
        field.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_upload(&self, bytes: u64) {
        self.compute_sampled_uploads.fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_storage_seed_upload(&self, bytes: u64) {
        self.compute_storage_seed_uploads
            .fetch_add(1, Ordering::Relaxed);
        self.compute_storage_seed_upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_compute_sampled_resident_copy(&self, bytes: u64) {
        self.compute_sampled_resident_copies
            .fetch_add(1, Ordering::Relaxed);
        self.compute_sampled_resident_copy_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::gather_witness::GatherVouch;

    /// The pooled total is what the census used to carry alone, and it read the
    /// same whether the bytes were a draw's tail or a compute output bound for
    /// guest pages. Each rail is asserted **by name** rather than by the sum:
    /// a sum-only assertion passes with any two of the three transposed, which
    /// is exactly the confusion the split exists to prevent.
    #[test]
    fn a_readback_lands_on_the_rail_its_source_names() {
        let counters = EngineCounters::default();
        counters.note_readback(4096, ReadbackSource::DrawTail);
        counters.note_readback(8192, ReadbackSource::ComputeBuffer);
        counters.note_readback(16384, ReadbackSource::ComputeImage);

        let s = counters.snapshot();
        assert_eq!((s.readback_draw, s.readback_draw_bytes), (1, 4096));
        assert_eq!(
            (s.readback_compute_buffer, s.readback_compute_buffer_bytes),
            (1, 8192)
        );
        assert_eq!(
            (s.readback_compute_image, s.readback_compute_image_bytes),
            (1, 16384)
        );
        // The pooled pair still counts every rail, so an existing reading of
        // `readback_bytes` keeps its meaning across this change.
        assert_eq!((s.readbacks, s.readback_bytes), (3, 4096 + 8192 + 16384));
    }

    #[test]
    fn note_helpers_update_event_and_byte_counters_together() {
        let counters = EngineCounters::default();
        counters.note_create(CreateSite::ShaderModule);
        counters.note_alloc();
        counters.note_readback(4096, ReadbackSource::DrawTail);
        counters.note_seed_upload(1024);
        counters.note_sampled_gather(2048);
        counters.note_sampled_gather_skipped(512);

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.creates, 1);
        assert_eq!(snapshot.allocs, 1);
        assert_eq!((snapshot.readbacks, snapshot.readback_bytes), (1, 4096));
        assert_eq!(
            (snapshot.seed_uploads, snapshot.seed_upload_bytes),
            (1, 1024)
        );
        // The gather is the sampled loop's only byte-moving arm, and it went
        // uncounted long enough to hide the whole of `acquire_sampled`. Pairing
        // it here keeps the event and its bytes from drifting apart the way a
        // count-only counter would.
        assert_eq!(
            (snapshot.sampled_gathers, snapshot.sampled_gather_bytes),
            (1, 2048)
        );
        // And the gathers that did not happen, whose bytes are the other half of
        // what this rail would cost with no cache.
        assert_eq!(
            (
                snapshot.sampled_gather_skips,
                snapshot.sampled_gather_skip_bytes
            ),
            (1, 512)
        );
    }

    /// A full-frame read charged to the wrong half is the difference between
    /// "this device is zero-copy" and "it is not", so the two deliveries are
    /// asserted by name and with different byte counts.
    ///
    /// Asserting only the sum would pass with the arms swapped, and swapped is
    /// exactly the state this split was written to end: `copy_target_to_guest_pages`
    /// charged `target_reads` for gigabytes a boot that never entered host
    /// address space, and the host total read the same whether the GPU rail was
    /// carrying the frames or not.
    #[test]
    fn a_target_read_lands_on_the_half_its_delivery_names() {
        let counters = EngineCounters::default();
        counters.note_target_read(4096, TargetReadDelivery::Host);
        counters.note_target_read(8192, TargetReadDelivery::GuestPagesOnGpu);

        let snapshot = counters.snapshot();
        assert_eq!(
            (snapshot.target_reads, snapshot.target_read_bytes),
            (1, 4096),
            "host delivery: {snapshot:?}"
        );
        assert_eq!(
            (snapshot.target_gpu_copies, snapshot.target_gpu_copy_bytes),
            (1, 8192),
            "guest-pages delivery: {snapshot:?}"
        );
    }

    /// The split of "the elision did not fire" has to be exhaustive and
    /// exclusive, or it stops being a division of the gathers and becomes two
    /// unrelated tallies that happen to sit next to each other.
    ///
    /// The two counts are deliberately different, and asserted by name rather
    /// than only through their sum: a sum-only assertion passes with the arms
    /// swapped, and swapping them inverts the verdict this instrument exists to
    /// give — "the witness refused" and "the cache dropped it" want opposite
    /// fixes.
    #[test]
    fn the_unskipped_reason_is_exactly_one_counter_per_bind() {
        let counters = EngineCounters::default();
        for _ in 0..3 {
            counters.note_sampled_gather_unskipped(GatherVouch::Vouched);
        }
        for _ in 0..5 {
            counters.note_sampled_gather_unskipped(GatherVouch::Fresh);
        }

        let s = counters.snapshot();
        assert_eq!(s.sampled_gather_unretained, 3, "vouched, nothing retained");
        assert_eq!(s.sampled_gather_unvouched, 5, "the witness gave no vouch");
        // Exhaustive and exclusive: eight calls, eight increments in total.
        assert_eq!(
            s.sampled_gather_unretained + s.sampled_gather_unvouched,
            8,
            "a bind that fell through the skip must land in exactly one arm: {s:?}"
        );
        // And neither is the skip, which counts the population these two divide
        // the complement of. Nothing above took the skip path.
        assert_eq!(s.sampled_gather_skips, 0);
    }

    /// Source attribution is a partition of uploads that actually happened,
    /// not a second population counted earlier at resource resolution.
    #[test]
    fn sampled_reupload_source_bytes_partition_the_total() {
        use super::super::types::SampledByteOrigin;

        let counters = EngineCounters::default();
        let origins = [
            SampledByteOrigin::AttachmentAlias,
            SampledByteOrigin::BufferBackedTexture,
            SampledByteOrigin::SerializedSurfaceView,
            SampledByteOrigin::SurfaceHostCache,
            SampledByteOrigin::SurfaceGuestFallback,
            SampledByteOrigin::LinearTexture,
            SampledByteOrigin::Synthetic,
        ];
        for (index, origin) in origins.into_iter().enumerate() {
            counters.note_sampled_reupload((index + 1) as u64, origin);
        }

        let s = counters.snapshot();
        assert_eq!(s.sampled_reuploads, 7);
        assert_eq!(s.sampled_reupload_bytes, 28);
        assert_eq!(s.sampled_reupload_attachment_bytes, 1);
        assert_eq!(s.sampled_reupload_buffer_texture_bytes, 2);
        assert_eq!(s.sampled_reupload_surface_view_bytes, 3);
        assert_eq!(s.sampled_reupload_surface_cache_bytes, 4);
        assert_eq!(s.sampled_reupload_surface_guest_bytes, 5);
        assert_eq!(s.sampled_reupload_linear_texture_bytes, 6);
        assert_eq!(s.sampled_reupload_synthetic_bytes, 7);
        assert_eq!(
            s.sampled_reupload_attachment_bytes
                + s.sampled_reupload_buffer_texture_bytes
                + s.sampled_reupload_surface_view_bytes
                + s.sampled_reupload_surface_cache_bytes
                + s.sampled_reupload_surface_guest_bytes
                + s.sampled_reupload_linear_texture_bytes
                + s.sampled_reupload_synthetic_bytes,
            s.sampled_reupload_bytes
        );
    }

    #[test]
    fn reset_clears_draw_gate_counters_but_preserves_lifetime_loss_counts() {
        let counters = EngineCounters::default();
        counters.readbacks.store(4, Ordering::Relaxed);
        counters.desc_pool_grow.store(3, Ordering::Relaxed);
        counters.device_lost.store(2, Ordering::Relaxed);
        counters.recreates.store(1, Ordering::Relaxed);

        counters.reset();
        let reset = counters.snapshot();
        assert_eq!(reset.readbacks, 0);
        assert_eq!(reset.desc_pool_grow, 0);
        assert_eq!(reset.device_lost, 2);
        assert_eq!(reset.recreates, 1);

        counters.reset_all();
        assert_eq!(counters.snapshot(), CounterSnapshot::default());
    }

    #[test]
    fn snapshot_delta_saturates_after_a_counter_reset() {
        let earlier = CounterSnapshot {
            creates: 10,
            readback_bytes: 4096,
            ..Default::default()
        };
        let later = CounterSnapshot {
            creates: 13,
            readback_bytes: 1024,
            ..Default::default()
        };

        let delta = later.delta_since(&earlier);
        assert_eq!(delta.creates, 3);
        assert_eq!(delta.readback_bytes, 0);
        assert_eq!(delta.allocs, 0);
    }

    #[test]
    fn atomic_snapshot_leaves_pool_owned_counters_for_the_pool_merge() {
        let snapshot = EngineCounters::default().snapshot();
        assert_eq!(snapshot.sampled_free_hits, 0);
        assert_eq!(snapshot.sampled_free_allocs, 0);
        assert_eq!(snapshot.target_free_hits, 0);
        assert_eq!(snapshot.target_free_allocs, 0);
    }
}
