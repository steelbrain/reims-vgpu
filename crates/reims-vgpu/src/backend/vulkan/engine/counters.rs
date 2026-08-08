//! Always-on create/alloc and cache hit/miss counters (reuse-gate proxies).
//!
//! # The vocabulary is declared once
//!
//! [`engine_counters!`] takes the counter names and generates the five things
//! that used to spell them out separately: the atomic [`EngineCounters`], the
//! plain-`u64` [`CounterSnapshot`], and the three whole-vocabulary walks
//! [`EngineCounters::snapshot`], [`EngineCounters::reset`] and
//! [`CounterSnapshot::delta_since`].
//!
//! Writing seventy names five times is how a counter silently stops working, and
//! neither failure mode is a compile error or a log line:
//!
//! * missing from `reset` — the counter reports a lifetime total into a reader
//!   that asked for a window, so a per-second rate reads as monotonically rising;
//! * missing from `delta_since` — the field reads **zero in every delta**, which
//!   is indistinguishable from "this path never ran". That is the
//!   "an event count is not a state" trap in `AGENTS.md` with the count itself
//!   broken.
//!
//! All five lists were checked against each other before this collapse and all
//! five agreed, so the macro changes no behaviour. What it changes is that they
//! can no longer disagree.
//!
//! The three groups are a real distinction, not a formatting one. `windowed` is
//! zeroed by `reset()`; `cumulative` deliberately survives it, because a
//! device-loss count is a fact about the boot and not about the measurement
//! window, and only `reset_all()` clears it; `pool_sourced` has no atomic at all
//! and is merged in from `ResourcePools` by `engine::counter_snapshot`.
//!
//! # A field with no named reader is not dead
//!
//! [`CounterSnapshot`] is consumed only by the integration tests in `tests/`.
//! No product code reads it and no log line emits it, so a sweep for "fields
//! nobody references" reports most of this struct. Twenty-seven of the
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
            pub fn delta_since(&self, earlier: &CounterSnapshot) -> CounterSnapshot {
                CounterSnapshot {
                    $($win: self.$win.saturating_sub(earlier.$win),)*
                    $($cum: self.$cum.saturating_sub(earlier.$cum),)*
                    $($pool: self.$pool.saturating_sub(earlier.$pool),)*
                }
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
        layout_hits,
        layout_misses,
        pass_hits,
        pass_misses,
        pipeline_hits,
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
        /// deferral worked — it reads the same either way. On a desktop workload
        /// `computes` is 0, so this is the draw rail alone.
        readbacks,
        readback_bytes,
        /// Full-frame reads of a pinned resident through `read_target`: the present
        /// capture and the deferred render window's on-access flush.
        ///
        /// These are the copies a deferred rail *keeps*, paid once when a consumer
        /// asks instead of once per Store. `target_reads / readbacks` is what
        /// separates "the readback moved" from "the readback went away".
        target_reads,
        target_read_bytes,
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
        /// the `SAMPLED_CACHE_CAP` A/B, where four times the entries and six
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
        /// Vertex/storage buffer binds the draw pointed straight at the guest's
        /// own pages through the imported RAMBlock, with no copy in either
        /// direction. Ranked against `buffer_snapshot_binds` and the
        /// `stage_phase` `runs_*` bars, which are what the CPU still gathers.
        buffer_guest_imports,
        buffer_guest_import_bytes,
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
    }

    cumulative {
        /// Cumulative across the boot: a device loss is a fact about this run,
        /// not about the measurement window, so `reset()` leaves it standing.
        device_lost,
        /// Cumulative across the boot, for the same reason as `device_lost`.
        recreates,
    }

    pool_sourced {
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
        /// being read again — the margin against `IDLE_TARGET_AGE_MS`, the age
        /// at which the idle drain destroys a resident terminally.
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
/// The `note_*` helpers: the increments that are not a bare `fetch_add(1)` at
/// the call site, because they move a count and a byte total together.
impl EngineCounters {
    pub fn note_create(&self) {
        self.creates.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_alloc(&self) {
        self.allocs.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_readback(&self, bytes: u64) {
        self.readbacks.fetch_add(1, Ordering::Relaxed);
        self.readback_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_target_read(&self, bytes: u64) {
        self.target_reads.fetch_add(1, Ordering::Relaxed);
        self.target_read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_seed_upload(&self, bytes: u64) {
        self.seed_uploads.fetch_add(1, Ordering::Relaxed);
        self.seed_upload_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_reupload(&self, bytes: u64) {
        self.sampled_reuploads.fetch_add(1, Ordering::Relaxed);
        self.sampled_reupload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_sampled_gather(&self, bytes: u64) {
        self.sampled_gathers.fetch_add(1, Ordering::Relaxed);
        self.sampled_gather_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_buffer_bind_reused(&self) {
        self.buffer_bind_reuses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn note_buffer_guest_import(&self, bytes: u64) {
        self.buffer_guest_imports.fetch_add(1, Ordering::Relaxed);
        self.buffer_guest_import_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn note_buffer_guest_gather(&self, bytes: u64, regions: u64) {
        self.buffer_guest_gathers.fetch_add(1, Ordering::Relaxed);
        self.buffer_guest_gather_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.buffer_guest_gather_regions
            .fetch_add(regions, Ordering::Relaxed);
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

    #[test]
    fn note_helpers_update_event_and_byte_counters_together() {
        let counters = EngineCounters::default();
        counters.note_create();
        counters.note_alloc();
        counters.note_readback(4096);
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
