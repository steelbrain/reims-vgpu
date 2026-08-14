//! Every environment variable this device reads, and the one way they parse.
//!
//! # Why they all live here
//!
//! An override is a rule the operator states from outside the process, so it has
//! the same problem the ABI header has: nothing in the toolchain finds the second
//! copy. A variable read at its point of use is invisible to everyone who does not
//! already know it exists, two sites spelling one variable's "off" differently is
//! a divergence no test can see, and a name that gets renamed in one place keeps
//! working in the other. Naming them here makes the set greppable and makes the
//! parse shared.
//!
//! # What an override may do
//!
//! **An override may only narrow what this device does. It may never widen it.**
//!
//! A switch can turn a rail *off* that the host was capable of running, because
//! that is a statement about policy and is always satisfiable. A switch may not
//! turn a rail *on* that the host reported it cannot run: capability is measured
//! from the device, and a variable that could override the measurement would turn
//! "this host has no such extension" into a crash or, worse, undefined behavior
//! inside a driver. Every gate stays where it is; a switch can only add a reason
//! to refuse.
//!
//! That rule is why [`Switch::On`] exists but is nowhere sufficient on its own.
//! Reading it is how a caller notices an operator asked for something the host
//! cannot give and says so, rather than ignoring the request in silence.

/// Guest RAM reaches the GPU as a host-pointer import over whole RAMBlocks.
/// Setting this off makes the device take the copying rails on a host that
/// could have imported — see
/// [`crate::backend::vulkan::caps::host_pointer`].
///
/// This is the switch that matters for verification. Where the import works
/// every guest window takes it and the copying rails run zero times, so a green
/// boot says nothing about them — and they are the only rails on a host without
/// the extension, and the rails a discrete GPU takes regardless.
///
/// # What the copying rails cost, driven macos-13, two boots each arm
///
/// One binary, interleaved, same probe. The gate held: `disabled_by_env` once
/// per boot, `guest_ram_map_no_backend_import` ~1 000 times, and
/// `sampled_guest_imports`/`buffer_guest_imports` **zero in every one of 77 and
/// 75 windows** — against non-zero on the import-on arm, which is what says the
/// counter would have caught a bind running past a closed gate. No panics; the
/// desktop renders correctly.
///
/// ```text
///                    import on    import off
/// present_hz            14.80     6.80 / 6.85
/// duty                   0.81     0.91 / 0.92
/// draw_us per draw      41.4 us   126.3 / 127.8 us
/// exec_phase finish    639.6 ms/s  810.9 / 821.1 ms/s
/// ```
///
/// **Less than half the frame rate and 3x the per-draw cost**, with the whole
/// difference landing in `ExecPhase::Finish` — the writeback copies. That is the
/// rail working, not a regression: the copy is the point on a host that cannot
/// import, and the guest observes the same pixels. It stays a *performance*
/// difference, which is what the support matrix requires of it.
///
/// **This measures the no-import column, not an iGPU.** A unified-memory host
/// *with* the extension binds a `GuestSlice` directly and is the fastest cell of
/// that matrix, not this one. Nothing in this reading was taken on Intel or AMD
/// hardware.
///
/// # What the GPU-side clock says, and it inverts the reading above
///
/// Everything above is CPU wall clock. `gpu_span` times the submission on the
/// GPU's own clock, and on that clock the copying rails are the **cheaper** arm.
/// One regime-matched pair, driven macos-13, same pin:
///
/// ```text
///                        import off    import on    import on
/// draws per frame           249.6        252.4        253.5
/// window_publish fresh       59.5         59.0         59.0
/// GPU us per draw            6.04        15.26        16.03
/// draw us per submission    78.42       226.66       230.92
/// drain duty                 0.66         0.37         0.36
/// gather regions/draw         0.0         15.4         13.4
/// ```
///
/// **Same frames, 61 % less GPU work per draw, and nearly twice the drain duty.**
/// The gather is what moves: with the import on, every scattered guest window is
/// assembled by the GPU out of guest RAM, which on a discrete host is a PCIe copy
/// — 4.46 GB/s of it, running at ~18 GB/s effective, and about 55 % of all the
/// GPU time this device spends. With the import off there is no gather at all;
/// the CPU packs the same bytes into staging and the drain worker pays for it.
///
/// So the two arms are not "fast" and "slow", they are **which engine does the
/// copy**. On this host, with the GPU at 45 % occupancy and the worker at 0.37
/// duty, moving it to the GPU is free and the import wins on the unmatched
/// regimes. On a host where the GPU is the constraint and the CPU is not — which
/// is the iGPU column of the support matrix — the trade plausibly inverts, and
/// `=off` is already the switch that takes it. That is a hypothesis this host
/// cannot test and it is written here so an operator on an iGPU knows there is a
/// second arm worth two boots.
///
/// One matched pair for the frame rate; the per-draw GPU cost has since
/// replicated across **four boots an arm and three compositing regimes**, and the
/// arms do not come close to touching:
///
/// ```text
/// import off   6.89  6.04  6.61  6.60
/// import on   15.37 13.72 13.84 16.03  (and 14.90, 15.26, 15.85, 13.94)
/// ```
///
/// Drain duty runs 0.66-0.82 on the copying arm against 0.36-0.64 on the import
/// arm, which is the same trade seen from the other side: the worker pays what
/// the GPU does not.
///
/// The frame-rate half stays a single matched pair, because `fresh` is not
/// comparable across compositing regimes and only one pair matched.
pub const GUEST_IMPORT: &str = "REIMS_VGPU_GUEST_IMPORT";

/// Verbose per-draw logging on top of the always-on fail sink.
pub const DRAW_LOG: &str = "REIMS_VGPU_DRAW_LOG";

/// Setting this off makes a completion stamp that follows a guest-page writeback
/// block the drain worker on that writeback and then write the stamp word
/// itself, instead of recording the word into the same GPU queue behind the
/// copy and letting the completion thread raise the interrupt.
///
/// A narrowing, like every switch here: the GPU-ordered stamp needs a
/// host-pointer import to reach the stamp page and `timelineSemaphore` to be
/// waited off-thread, so `off` selects the rail a host lacking either takes
/// regardless. It exists because the two rails answer "when may the guest
/// observe this stamp" with different mechanisms — a CPU wait versus a pipeline
/// barrier plus a thread — and a hang or a torn frame has to be attributable to
/// one of them without rebuilding.
pub const GPU_STAMP: &str = "REIMS_VGPU_GPU_STAMP";

/// Setting this off stops the two guest-page write guards —
/// [`crate::runtime::node_guard`] and [`crate::runtime::released_pages`] — from
/// observing anything. They decide nothing, so this changes no guest-visible
/// behavior; what it removes is the page-table descent and the page-list resolve
/// that each map and unmap packet pays for them, on the drain thread, while it
/// holds the device lock.
///
/// A narrowing, like every switch here: it turns an observation off and can
/// never turn one on.
///
/// It exists because these guards watch an intermittent guest kernel panic that
/// is a **race**, so the honest question "does watching it change the rate?" has
/// to be answerable without rebuilding. A measurement that cannot be controlled
/// is the failure this whole instrument was built to avoid, and an instrument on
/// the drain thread is exactly the kind that could perturb its own subject.
pub const PAGE_GUARDS: &str = "REIMS_VGPU_PAGE_GUARDS";

/// Setting this **on** makes [`crate::runtime::range_coverage`] walk the guest's
/// page table across every page of every map and unmap range. Default off, and
/// it is the only variable here whose default is the quiet one.
///
/// # Why it defaults off, which is a measurement and not a preference
///
/// Always-on, this walk costs the drain enough to lose the guest a race it
/// otherwise wins. One undriven macos-15 boot went from **0** `no_list_entry` to
/// **47**, and from 0 `list_miss_slot_empty` to 182, purely by adding it — and
/// back to 0 with the guards switched off on the same binary. The other guards
/// descend a single path per packet; this one walks a whole range, sixteen
/// thousand pages for one 64 MiB mapping, on the drain thread while it holds the
/// device lock.
///
/// So it is a probe rather than a guard, under the rule its own module states:
/// an instrument that watches a race must not be the reason the race moves.
///
/// # Why it is not the widening this module forbids
///
/// The rule above is about **capability** — a switch may not turn on a rail the
/// host reported it cannot run, because binding an unadvertised extension is a
/// crash and importing an undeclared handle is undefined behavior in a driver.
/// There is no host that cannot walk a page table it is already reading, and
/// nothing this gates changes what the guest observes. What it changes is how
/// much work the drain does, and the default is the side that does less.
///
/// It gets its own name rather than riding on [`DRAW_LOG`] for the same reason
/// it exists: that variable turns on a per-draw log flood that is itself a drain
/// cost, so gating a latency probe behind it would guarantee the perturbation
/// the probe is trying to measure.
pub const RANGE_COVERAGE: &str = "REIMS_VGPU_RANGE_COVERAGE";

/// `off` stops narrowing a guest buffer bind to the extent the shader's
/// reflection proved it can read, so the bind walks the rest of the allocation
/// exactly as it did before that rail existed.
///
/// This is the A/B instrument for the rail, and it is why the rail can be
/// measured at all: the two arms differ by one branch in one process, so a
/// driven boot of each on one build and one rail attributes a change in gathered
/// bytes to the narrowing rather than to a rebuild. Without it the comparison is
/// a boot of `HEAD` against a boot of `HEAD~1`, which also moves every other
/// difference between the two binaries into the result.
///
/// It only ever *widens the window this device reads*, never what the guest may
/// see, so it obeys the rule the module doc states: it turns a rail off, and
/// there is no spelling of it that turns one on. `on` and unset are the same
/// arm — the default — because a capability that is not measured is not a
/// capability this switch may grant.
pub const BUFFER_EXTENT: &str = "REIMS_VGPU_BUFFER_EXTENT";

/// `off` narrows the draw batch back to one render target, so a draw whose
/// target differs from the open batch's stops joining it and submits its own
/// command buffer.
///
/// The wider arm — the default — is that the target does not key the batch at
/// all: every batched draw begins and ends its own render pass inside the
/// command buffer, and nothing between `batch_append` and `batch_flush` reads
/// which image those passes wrote. A run alternating between two surfaces
/// therefore costs one submission per draw under the narrow arm and one per
/// `BATCH_MAX_DRAWS` draws under the wide one.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does: the two
/// arms differ by one comparison in one process, so a driven boot of each on one
/// build and one guest rail attributes a change in submissions, ring blocking
/// and gathered bytes to the batching rule rather than to a rebuild. Off is a
/// refusal (`nojoin_target_switch`) and never a permission.
pub const BATCH_MIXED_TARGETS: &str = "REIMS_VGPU_BATCH_MIXED_TARGETS";

/// `off` stages the guest bytes of a `[[buffer(n)]]` bind even when the stage's
/// own reflection says the shader never dereferences it.
///
/// The wider arm — the default — binds a neutral page for such a bind instead of
/// gathering the guest's, because
/// [`crate::runtime::spirv_bind::ReflectedBufferAccess::Unused`] means no shader
/// invocation reads through the descriptor. The descriptor is still written, so
/// the pipeline layout is byte-for-byte what it was; only the contents change,
/// and only for binds nothing reads.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does, and with
/// more at stake. This is the one rail here whose failure mode is **silent wrong
/// pixels**: if the translator ever says `Unused` about a buffer a shader does
/// dereference, the shader reads the neutral page and nothing anywhere reports
/// it. So the arm that copies has to stay reachable in one process, both to A/B
/// the saving and to answer "is this rail why that surface is wrong" without a
/// rebuild.
///
/// Off is a refusal and never a permission: it makes this device read *more* of
/// the guest's memory, never less, and there is no spelling that grants a
/// capability.
///
/// # It buys frames now, and did not when it was written
///
/// The twelve-boot A/B that landed this rail established the saving —
/// `kib_per_draw` −6.27 %, disjoint at 45x — and explicitly established **no
/// frame gain**, correctly, because the host window presenter was then a hard
/// ~41 Hz ceiling and no device-side saving could appear past it.
///
/// Re-run on the multi-flight presenter, eight driven macos-13 boots, n=3 vs
/// n=2 after regime exclusion: `kib_per_draw` **−6.31 %** — the same number to
/// within a twentieth of a percent, disjoint at 44x, which is what says the two
/// runs measured one rail — and now `frames_s` **+3.73 %, disjoint**, with
/// `draws_s` +2.93 % also disjoint.
///
/// So the saving was real all along and was being spent into a ceiling. Two
/// readings follow, and the second is the one worth carrying: a per-draw saving
/// measured before that presenter fix is owed a re-run rather than believed
/// worthless, and `frames_s` is the metric that reports it — not `presents_s`,
/// which still overlapped here at this sample size.
pub const UNUSED_BINDS: &str = "REIMS_VGPU_UNUSED_BINDS";

/// `off` returns the host window presenter to one present in flight at a time.
///
/// The wider arm — the default — lets several of the presenter's blits be in
/// flight at once, because with one the presenter was a **ceiling** rather than
/// a pacer: twelve driven macos-13 boots put its output at 1599-1696 frames
/// while the device published 1760-2015 to it, `busy_acquire` 0 throughout. The
/// swapchain always had an image free; the refusals were all the previous
/// blit's fence, which retires behind queued guest work because the blit shares
/// a queue with every guest draw.
///
/// It exists as a switch for the same reason [`BUFFER_EXTENT`] does — one
/// binary, one branch, two arms — and because presentation depth is the kind of
/// change whose failure is a stutter or a torn frame rather than a decline, so
/// the previous behavior has to stay reachable without a rebuild.
///
/// Off is a refusal and never a permission: one present in flight is strictly
/// less concurrency than several, never more.
pub const PRESENT_DEPTH: &str = "REIMS_VGPU_PRESENT_DEPTH";

/// **Default on.** Setting this off restores one completion-stamp write per
/// packet, which is what `drain_child_fifo` did before the stamps in a single
/// drain of one channel were collapsed into a single write at its end.
///
/// Off is a refusal and never a permission: one write per packet is strictly
/// more work than one per drain, never less, and the coalesced arm needs no
/// capability the per-packet arm does not.
///
/// It exists because the stamp is **12.3 us a packet** on the Vulkan arm —
/// `write_stamp` submits a command buffer rather than writing a word — and
/// because a change to when a completion becomes visible to the guest is the
/// kind whose failure is a stall or a stutter rather than a decline. The
/// previous behavior has to stay reachable without a rebuild so the two can be
/// A/B'd on one binary.
pub const STAMP_COALESCE: &str = "REIMS_VGPU_STAMP_COALESCE";

/// **Probe, default off.** Setting this *on* cuts every guest-page scatter run
/// into four contiguous sub-ranges that tile it exactly.
///
/// The guest bytes written are byte-for-byte identical either way — only the
/// number of `VkBufferCopy` regions changes, by 4x. It is the controlled form of
/// the question the writeback rail's cost turns on: whether that rail is bound
/// by the bytes it moves or by the number of copy regions it issues. The two
/// predict opposite things about replacing the scatter with a compute dispatch,
/// and a host GPU at 86-91 % busy on 3-4 % memory utilization says it is not the
/// bytes.
///
/// It is a probe and not a rail, in the sense [`RANGE_COVERAGE`] is: it changes
/// nothing the guest observes, and its default is the side that does less work.
/// It does not widen anything — there is no host that can issue one copy region
/// and not four.
///
/// # What it measured, and why it is kept rather than deleted
///
/// Eight driven macos-13 boots, four per arm, one binary: 203 regions per
/// writeback against 806, and `present_hz` **49.15/49.45/56.45/56.40 against
/// 26.90/23.80/23.00/23.70**. Eight boots for eight with no overlap — four times
/// the regions for byte-identical output **halves the frame rate**, and
/// `slot_us` roughly doubles, which is the drain worker blocking longer on a ring
/// fence the GPU takes longer to signal.
///
/// That answers the question for this host class and it is written up where the
/// rail is, in [`crate::runtime::render_writeback`]. The probe stays because the
/// answer is a property of the **host**, not of this device: a discrete GPU
/// crossing PCIe per region and a unified-memory host writing into the same
/// physical pages have no reason to agree, and only one of the two has been
/// measured. A future unified-memory boot re-runs this in one command instead of
/// rebuilding the experiment from the module doc.
pub const SCATTER_SPLIT: &str = "REIMS_VGPU_SCATTER_SPLIT";

/// `off` narrows the guest-page writeback's scatter back to one transfer region
/// per guest run, from the compute dispatch that replaces them.
///
/// The dispatch writes the same guest bytes — the kernel copies `uint`s and
/// carries no format, row or texel semantics at all — so this switch chooses
/// between two byte-identical implementations of one copy and can never change
/// what the guest observes. It narrows in the sense the module doc requires:
/// the transfer form is the only form on a host without the guest-RAM import,
/// and it stays the form for a run whose geometry the dispatch cannot express.
///
/// It exists because it is the A/B. The region count is measured to be ~35 % of
/// frame time (see [`crate::runtime::render_writeback`]), and the only way to
/// hold that number against this repair on a given host is to run the host both
/// ways in one binary.
pub const COMPUTE_SCATTER: &str = "REIMS_VGPU_COMPUTE_SCATTER";

/// `off` narrows a type-11 surface Store back to copying its frame into the
/// guest's pages at the Store, instead of holding it in the engine resident and
/// copying when something reads those pages.
///
/// **The two arms are not byte-identical, and this switch is the only reason
/// that is acceptable.** They land the same bytes in the same pages; what
/// differs is *when*, and therefore what a reader this device failed to account
/// for would see. The failure mode is a stale frame — which no counter can
/// report, because a copy correctly skipped and a copy wrongly skipped are the
/// same absence — so the arm that has to be believed is the one where such a
/// reader exists, and the only way to hold the two against each other on a given
/// guest is a binary that runs both. The A/B harness photographs both arms for
/// this and nothing else.
///
/// It narrows in the sense this module requires: the eager Store is what the
/// lazy one defers, `writeback_debt::pay` calls the identical
/// `render_writeback::store_render_frame`, and switching this off cannot reach a
/// copy the lazy rail would not eventually have made.
///
/// Default on, because it is the measured winner and by a margin no other rail
/// here has produced. Twelve interleaved driven macos-13 boots: 90 % of type-11
/// Stores superseded before anything read their pages, `draw_us` 14.62 against
/// 26.52 a draw, and five of six on-arm boots presenting at 105.8-109.2 Hz
/// against a 77.2-78.6 Hz baseline. `crate::runtime::writeback_debt` carries the
/// census, the reader argument and the standing alarm.
pub const LAZY_WRITEBACK: &str = "REIMS_VGPU_LAZY_WRITEBACK";

/// `off` narrows the idle drain back to returning every empty image-slab block
/// to the driver on each fired pass, instead of only once idle has settled.
///
/// It narrows in the sense this module requires: `off` retains strictly less
/// device memory, never more, and reaches no allocation the settled arm would
/// not also reach — a block trimmed early is re-allocated on the next carve, and
/// the arms differ only in how many `vkAllocateMemory` calls one workload makes.
///
/// It exists because the size of the churn it removes is a property of the
/// *workload's duty cycle*, not of the device: a load that saturates the drain
/// worker never lets a pass fire, and one that leaves 100 ms gaps between frames
/// pays a block allocation in every one of them. Ranking that needs both arms in
/// one binary on one guest.
pub const SLAB_RETAIN: &str = "REIMS_VGPU_SLAB_RETAIN";

/// `off` narrows the CLEAR-seed Store at the head of a draw chain out of
/// existence: the solid colour is not written into the guest's pages before the
/// encode, and only what the draw's own Store lands reaches them.
///
/// It narrows in the sense this module requires — strictly fewer writes, and no
/// write it can reach that the wider arm does not also make.
///
/// **It is an ablation and not a shipping arm.** The seed is what the guest sees
/// outside the region a draw covers, so switching it off can lose pixels, and
/// the failure mode is content rather than a counter. It exists because
/// `prep_seed_us` is 8.6 µs of a 41 µs chain on the load probe's `blur=40` dial
/// — 21 %, second only to the engine — and no elision of it can be designed
/// against a cost nobody has priced. The A/B harness photographs both arms,
/// which is the only way this arm's damage is visible at all.
pub const CLEAR_SEED: &str = "REIMS_VGPU_CLEAR_SEED";

/// `off` narrows a draw chain's pipeline resolution back to the full walk —
/// object list, descriptor, decode, MTLB read, AIR carve and content hash, for
/// the pipeline and both of its functions, on every draw.
///
/// It narrows in the sense this module requires: the full walk is what the memo
/// is a cache in front of, and every resolution the memo serves came out of it.
/// Switching it off cannot reach a resolution the walk would not have produced.
///
/// It exists because the memo's correctness rests on a stated claim about what a
/// guest does to a live pipeline object — see
/// [`crate::runtime::pipeline_resolve`] — and a claim about a guest is worth a
/// binary that can be run both ways against that guest.
pub const PIPELINE_MEMO: &str = "REIMS_VGPU_PIPELINE_MEMO";

/// `on` issues the draw-time guest buffer gather as one compute dispatch per
/// gathered window instead of one transfer region per guest run.
///
/// The gather direction of what [`COMPUTE_SCATTER`] does for the writeback, and
/// byte-identical for the same reason: the kernel copies `uint`s and carries no
/// format, row or direction semantics at all, and the run table is built from
/// the very `VkBufferCopy` regions the transfer form would have issued. So this
/// chooses between two implementations of one copy and can never change what
/// the guest observes.
///
/// # It is default **off**, and this is the measurement that says so
///
/// Ten driven macos-13 sustained-animation boots, interleaved, on the tree that
/// gave a dispatch a shared run-table arena and a recycled descriptor set. The
/// mechanism columns are per draw, so they are comparable across boots that drew
/// different amounts, and they are disjoint in every one of the four compositing
/// regimes the ten boots landed in:
///
/// ```text
///                        on            off
/// slot_us/draw          1.94          4.06     -52 %   GPU blocking
/// record_us/draw        3.10          2.01     +54 %   CPU recording
/// dispatches/draw       1.44          0.00
/// ```
///
/// **Frames did not move, and the honest statement is that they could not be
/// read.** The rank test put `frames_s` at +9.0 % — and that number is an
/// artifact of the band mix, which is worth stating because it is the trap this
/// project keeps walking into. The ten boots sort by draws per frame as
///
/// ```text
/// 257.0 257.2 | 335.9 340.4 341.4 344.0 | 361.6 362.2 363.9 367.5
/// ```
///
/// — a **fourth** regime the harness's band edges did not yet separate, running
/// at ~79 frames a second against the next one's ~72. The on arm drew three
/// boots from the fast side and the off arm one. Matched by draws per frame the
/// two arms are level, and with the edges re-derived no band holds two boots of
/// both arms, so the frame question is *unresolved* rather than answered either
/// way.
///
/// So the arena and the recycle took the recording penalty from +85 % to +54 %,
/// the GPU saving held at ~-52 %, and it is still not enough. What is left is
/// +31.6 ms a second of `record_us` over ~36 700 dispatches — **0.86 us each**,
/// against the ~1.05 us this started at.
///
/// # What would flip it, and what would not
///
/// **Not fewer dispatches.** The obvious move is to batch a command buffer's
/// gathers into one, and the arithmetic refuses it: ~40 000 dispatches against
/// ~26 500 draws is **1.5 per draw**, not 18 per command buffer. `guest_gathers`
/// is a local of `execute_draw_inner`, so a plan covers one draw's windows and
/// there is no command-buffer-wide batch to make. Merging within a draw is real
/// but it is 1.4x on the count, not 18x.
///
/// **The per-dispatch cost, and the next reading is an instrument and not a
/// guess.** [`crate::backend::vulkan::engine::gather_phase`] splits the 0.86 us
/// four ways — the run-table planning, the shared staging arena, the descriptor
/// set, and the command-buffer calls — because guessing which of them is the
/// next 0.8 us is how a session spends a day on `vkCmdBindPipeline` and finds it
/// was never the cost. Read `gather_phase` from a driven boot before touching
/// any of them.
///
/// The candidate that survives the arithmetic without a reading is the
/// **destination arena**: each gathered window takes its own pooled slot, so
/// `Dst` is the only binding that still varies within a draw. Suballocating a
/// draw's destinations from one buffer makes all three constant, which is one
/// descriptor set and one dispatch per draw rather than 1.4 of each.
/// `BoundBuffer` already carries an offset, so the bind side of that is free.
///
/// # A region-count threshold is not the shortcut it looks like
///
/// The dispatch's value scales with how many regions each one replaces, so the
/// obvious cheap interim is to dispatch only for windows above some run count.
/// Six driven boots of `gpu-load-probe` at `layers=24&boxes=6`, which runs
/// **23.5 gather regions per draw** against the sustained probe's 15.8, say the
/// ceiling on that is low. Comparing the boots that reached the same drain duty
/// (~0.8):
///
/// ```text
///            frames/s   slot_us   record_us
/// on               86.96    13 007      64 357
/// off              84.76    95 715      42 487
/// off              83.72   107 146      42 279
/// ```
///
/// The sign does flip — `slot_us` falls **86 %** here against 56 % on the
/// lighter load, exactly as amortising a fixed cost over more regions predicts
/// — and it is still only ~+3 % of the frames, because the per-dispatch cost is
/// unchanged and the recording penalty is the same +52 %. A threshold buys the
/// tail of a distribution whose mean is the problem.
///
/// That load is also a poor A/B vehicle and should not be used as one: the same
/// six boots ranged over drain duty 0.25 to 0.80 and 24.7 to 87.0 frames a
/// second, while `draws/frame` sat at 132.7-133.0 on every one of them. The
/// regime discriminator that works on the sustained probe is flat here, so
/// nothing separates a fast boot from a slow one before the fact.
///
/// # 2026-08-11: this is now default ON, and the switch is an ordinary refusal
///
/// Everything above is the case for leaving it off, and it was sound on the tree
/// that measured it. Two things changed.
///
/// **The GPU cost is now measured directly rather than inferred from a wall-clock
/// wait.** Every reading above reaches for `slot_us`, which is the drain worker
/// blocked on a ring fence. [`crate::backend::vulkan::engine::gpu_span`] times the
/// submission on the GPU's own clock instead. Driven macos-13 sustained boots,
/// same pin, matched compositing regime, with the *planned* region count carried
/// as the control so "byte-identical output" is checkable rather than asserted:
///
/// ```text
///                        off      off      on       on
/// draws per frame       280.2    276.4    269.5    256.1
/// gather regions/draw    15.7     15.5     15.9     16.8   <- control, flat
/// GPU us per draw        18.33    18.40    13.88    14.32   -24 %
/// draw us per submission 266.4    264.1    195.0    203.8   -25 %
/// drain duty              0.56     0.58     0.55     0.37
/// window_publish fresh  105.0/s  104.5/s  110.0/s   59.0/s
/// ```
///
/// The arms are disjoint with no overlap, and the control says both arms planned
/// the same ~16 regions a draw — so this is one workload done for a quarter less
/// GPU, which is exactly what a dispatch replacing ~13 transfer regions should
/// buy.
///
/// Those four boots were taken in blocks, two per arm, which cannot separate the
/// change from anything that drifted between the blocks. Six more, **interleaved**
/// on/off/on/off/on/off on one pin, hold it:
///
/// ```text
///                          on     on     on     off    off    off
/// draws per frame        291.2  299.3  249.4   254.5  301.4  192.6
/// gather regions/draw     15.6   15.4   16.1    16.4   15.0   13.4  <- control
/// GPU us per draw        15.64  15.55  14.56   20.51  19.59  17.50
/// drain duty              0.63   0.63   0.39    0.39   0.64   0.35
/// ```
///
/// Still disjoint — the worst `on` boot beats the best `off` boot — for **-20.6 %**
/// on the arm means, against the -24 % the block-ordered four read. The control
/// overlaps between the arms and duty does not rise on the dispatch arm, which are
/// the two readings that would void the comparison. Take the -20.6 % as the
/// figure: it is the one measured under interleaving, and it spans three
/// compositing regimes rather than one.
///
/// **The CPU cost it was rejected for is now affordable.** +31.6 ms/s of
/// `record_us` mattered because the drain worker was saturated at duty 0.90; the
/// stamp coalescing and the preflight memo have since taken ~148 ms/s off that
/// thread, and duty is 0.55-0.58 here and does not move between the arms. The
/// earlier rejection was not wrong, it was conditional on a premise that two
/// commits removed.
///
/// **Why this matters more than the frames say.** `fresh` moves +4.8 % on the one
/// boot where it could, and on this rail frames are set by the guest — 54 % GPU
/// occupancy beside duty 0.56 means neither side of the device is the pacer. The
/// quantity that matters is GPU work per unit of guest work, because the support
/// matrix's other column is an iGPU, where the same recorded commands cost roughly
/// an order of magnitude more and this workload is hard GPU-bound. A 24 % cut
/// there is a 24 % cut in the thing that binds it. This host cannot boot an iGPU,
/// so `us/draw` is the closest available measurement and it is the one quoted.
///
/// So `off` is now an ordinary refusal — strictly ~13 transfer regions where the
/// wider arm issues one dispatch — and the paragraph below about the switch being
/// a "permission" no longer applies. The arms remain byte-identical in what the
/// guest observes, which is why either may be the default at all: the kernel
/// copies `uint`s and carries no format, row or texel semantics.
///
/// The narrowing arm stays because the answer is a property of the **host**. A
/// discrete GPU crossing PCIe per region and a unified-memory host writing into
/// the same physical pages have no reason to agree, and only the first has been
/// measured. On a host where the dispatch loses, `=off` is the one command that
/// says so.
pub const COMPUTE_GATHER: &str = "REIMS_VGPU_COMPUTE_GATHER";

/// **Default on.** `off` stops the device writing the two GPU timestamps that
/// bound a draw submission's command buffer, which is the only way it knows how
/// long the GPU spent executing one.
///
/// Off is a refusal and never a permission: no query is created, reset, written
/// or read on that arm, and the census publishes no line rather than a zero. The
/// probe is on by default because two timestamps per submission is ~4 000 a
/// second against the readback rail's existing three per composite, and because a
/// reading nobody has to ask for is the one that gets read — every session before
/// this one inferred GPU occupancy from `slot_us`, a wall-clock wait, and five of
/// them concluded the rail was GPU-bound without a GPU-side number existing
/// anywhere in the device.
///
/// It is a switch rather than a constant because it is not free. A timestamp is a
/// pipeline flush point on some hardware, so an A/B that needs the absolute floor
/// — anything ranking submission shape or ring depth — should take it out on both
/// arms and say that it did. See
/// [`crate::backend::vulkan::engine::gpu_span`] for what the pair measures and
/// the two caveats that belong to the reading rather than to the code.
pub const GPU_SPANS: &str = "REIMS_VGPU_GPU_SPANS";

/// **Default off, and a probe rather than a rail.** `on` makes every draw send
/// its colour attachment out of `TRANSFER_SRC_OPTIMAL` and straight back into it
/// after the render pass has ended.
///
/// It prices a pair this device already pays on every loading draw and has never
/// measured. A draw that loads its target barriers the image to
/// `COLOR_ATTACHMENT_OPTIMAL` on the way in, and the render pass's `final_layout`
/// returns it to `TRANSFER_SRC_OPTIMAL` on the way out — so a run of draws into
/// one target moves a full-size image between two layouts twice per draw, for a
/// transfer reader that on this workload arrives a few times a second against
/// tens of thousands of draws. Whether that is free or a whole-attachment resolve
/// is a property of the host's colour compression, and nothing else in this
/// device can distinguish the two.
///
/// The arm is byte-identical in what the guest observes: both layouts preserve
/// contents, and nothing is recorded between the two barriers. So it may only add
/// GPU time, and `gpu_span`'s `us/draw` moving is that time.
///
/// It never widens anything — `on` records *more* work — which is why it is safe
/// as an on-switch where every other variable here is an off-switch.
///
/// # What it read: the pair is free on this host
///
/// Six driven macos-13 sustained-animation boots on one pin, interleaved
/// on/off/on/off/on/off. The `on` arm records **four** full-attachment layout
/// transitions a draw where the `off` arm records two:
///
/// ```text
///                       on     on     on      off    off    off
/// draws per frame     284.5  252.4  271.8    299.9  269.4  253.5
/// GPU us per draw     14.90  15.26  13.84    15.85  13.94  16.03
/// ```
///
/// The arms interleave completely, and in each of the three regime-matched pairs
/// the arm doing **more** work reads **lower** — which is the sign an effect
/// cannot have, so it is boot-to-boot noise. Doubling this device's per-draw
/// layout transitions costs less than the ~5 % spread between two boots of one
/// binary.
///
/// The reading is that this host does not compress what it is asked to transfer:
/// the colour attachment is created with `TRANSFER_SRC` usage, so the driver has
/// no compression to resolve when the layout moves. **So do not remove the
/// existing pair on this evidence.** Every loading draw barriers its target into
/// `COLOR_ATTACHMENT_OPTIMAL` and the pass's `final_layout` puts it back, for a
/// transfer reader that arrives a few times a second against tens of thousands of
/// draws — it looks wasteful and it measures free, and the barrier family it sits
/// in has had five live dependency bugs, so the refactor is all risk and no
/// measured gain *here*.
///
/// # Why it stays
///
/// "Here" is the whole caveat, and it is why this is a kept probe rather than a
/// deleted one. AMD and Intel parts keep colour compression metadata that a
/// transfer layout can force a whole-attachment resolve of, and this project has
/// no such host to boot. Anyone who has one can answer the question in two boots
/// without writing any code: if `us/draw` rises on the `on` arm there, the pair
/// is worth removing and this doc is wrong about their machine, not about this
/// one.
pub const LAYOUT_CHURN: &str = "REIMS_VGPU_LAYOUT_CHURN";

/// **Default off.** `on` records one extra empty render pass instance per
/// loading draw, on the target the draw just finished with. A probe, never a
/// change: it adds work and removes none, and the pixels are identical because
/// the extra instance loads and stores the attachment and draws nothing into it.
///
/// # What it prices, and why nothing else can
///
/// Every batched draw opens and closes its own render pass. `passmerge_*` /
/// `passheld_*` (see [`crate::backend::vulkan::engine`]'s `PassObstacles`) say
/// how many of them *could* share one: 82 % of draws, once the guest gathers
/// they record between the draws are hoisted out of the way. Hoisting them
/// needs a second command buffer per batch, which is a large change to the ring,
/// and nothing in this device says what the pass pair it would remove actually
/// costs.
///
/// This is the positive control for that number, on the pattern
/// [`LAYOUT_CHURN`] set: an arm that pays the cost *twice* prices it once. The
/// probe records the transition back into `COLOR_ATTACHMENT_OPTIMAL` that a
/// second instance needs, then a `vkCmdBeginRenderPass` / `vkCmdEndRenderPass`
/// pair on the same render pass, framebuffer and area. The extra transition is
/// not a confound: [`LAYOUT_CHURN`]'s six boots measured that this host's
/// full-attachment transitions cost less than the boot-to-boot spread, so what
/// separates the arms here is the pass pair.
///
/// Read it on `present_hz` over the fast population, which is what ranks a
/// per-draw change on this device — the arithmetic and the elasticity are beside
/// `crate::runtime::drain::census::VBL_REPORT_EARLY`. A pair that costs ~1.5 us
/// of the ~14 us this device spends per draw is a ~10 % arm and separates
/// cleanly; a pair that costs 0.2 us does not, and the merge is then an
/// iGPU-and-tiler lever only, which is exactly what [`LAYOUT_CHURN`] concluded
/// about the transitions.
///
/// Only loading draws take it. A `CLEAR` pass instance replayed after the draw
/// would clear away what the draw just rendered, so the probe would not be
/// pixel-neutral — and loading draws are the population a merge applies to
/// anyway, since a clearing joiner gets a different `VkRenderPass` and could
/// never have shared the instance.
///
/// # What it read: the pair costs 3 % of a draw and no frames
///
/// Twenty interleaved driven macos-13 sustained-animation boots, one pin,
/// quiesced host, ten an arm. Scored on the driven window only and within the
/// fast population only, per the rules beside
/// `crate::runtime::drain::census::VBL_REPORT_EARLY`:
///
/// ```text
///                        n   mean     range
/// draw_us/draw   off     7   13.33    13.01 .. 14.12
///                on      4   13.75    13.63 .. 13.83
/// present_hz     off     7  113.54   111.40 ..115.80
///                on      4  112.83   112.00 ..113.50
/// ```
///
/// Six of the seven `off` boots read below every `on` boot, so the per-draw
/// column separates: **+3.2 %, about 0.42 us of the ~13.3 us this device spends
/// per draw.** The frame rate does not: −0.6 % with the ranges nested, which is
/// what a 3.2 % per-draw arm is expected to look like at this elasticity and is
/// the reason the sizing note says 2 % is not measurable here.
///
/// So the arithmetic for the merge, which is the only reason this probe exists:
/// `passheld_*` puts the reachable share at 82 % of draws, 0.82 x 0.42 us is
/// ~0.34 us, ~2.6 % of per-draw CPU, **~1.5 % of frames.** Against that, the
/// change is a second command buffer per batch, a rewrite of the ring's
/// submission, and a deferred pass end whose failure mode is a Vulkan
/// render-pass-scope violation on a host with no validation layer installed.
///
/// **Do not build it for this host.** The verdict is the same shape as
/// [`LAYOUT_CHURN`]'s and for the same reason: an immediate-mode discrete GPU
/// with a fast CPU beside it does not care, and this project has no other host
/// to boot. It is a real lever on a tile-based renderer, where a pass boundary
/// is a load and store of the whole attachment through tile memory rather than
/// a driver call — which is every iGPU, and both Apple Silicon pathways in the
/// support matrix. Anyone with one of those answers it in the boots this took.
///
/// Two things about the run that are *not* findings, recorded so nobody reads
/// them as such. The `on` arm drew 4 fast boots of 10 against the `off` arm's 7;
/// a slow rate is a Bernoulli draw whose base rate drifts, and ten boots an arm
/// cannot separate 0.4 from 0.7. And one `on` boot wedged at 5.7 Hz and 622
/// us/draw — excluded as slow, and a single such boot says nothing either.
pub const PASS_CHURN: &str = "REIMS_VGPU_PASS_CHURN";

/// **Default off.** `on` lets a declared object size narrow the guest bytes the
/// gather rail *moves*, instead of only narrowing binds that stay above the
/// zero-copy floor after narrowing — which is none of them.
///
/// # Why the rail is inert without it, and why the floor is the reason
///
/// `spirv_bind::reflected_buffer_extent` answers `Object` for **57 % of buffer
/// binds** at the current translator pin, and essentially every answer is 512
/// bytes or less. `ZERO_COPY_BUFFER_MIN_BYTES` is 16 KiB, so
/// `load_buffer_content` drops every one of those caps before the rail sees it,
/// and `try_buffer_zero_copy_resolved` then resolves the whole rest of the
/// allocation. Both narrowing counters read **zero** across a driven boot of 2.57
/// million binds.
///
/// The bytes behind that, two driven macos-13 boots, summed over the boot:
///
/// ```text
///                            boot 1      boot 2
/// bext_capped_span_bytes     274.1 GB    272.9 GB
/// bext_would_save_bytes      274.0 GB    272.9 GB   (99.97 % of the capped)
/// bext_uncapped_span_bytes   287.3 GB    283.7 GB
/// ```
///
/// That reads as "49 % of every byte this rail moves sits behind a cap that
/// would remove 99.97 % of it", and the gather is ~55 % of all the GPU time this
/// device spends (see [`GUEST_IMPORT`]). **Both halves of that sentence are
/// true and the conclusion drawn from it was still wrong**; the six boots below
/// say why.
///
/// # What it is actually worth: bytes yes, time no
///
/// Six interleaved driven macos-13 boots, one binary, quiesced host,
/// `sustained-animation-probe`. The byte columns are the mechanism's own
/// control and they are **fully disjoint**:
///
/// ```text
/// arm   gather KB/draw    gathers/draw   zc_buffer_held   narrowed
/// on         95.5             0.578         2 671 242       7 054
/// on         98.4             0.585         1 521 450       6 868
/// on         98.4             0.584         1 496 094       6 679
/// off       112.6             0.652         1 538 409           0
/// off       111.2             0.645         1 121 638           0
/// off       109.4             0.643         2 711 284           0
/// ```
///
/// So the rail moves **12.3 % fewer bytes** in **10 % fewer gathers**, the gate
/// holds at zero on the `off` arm, and `zc_buffer_held` does *not* fall — the
/// registry churn that sank the earlier attempt is genuinely avoided, and both
/// arms' screenshots show a correct desktop.
///
/// `us/draw` does not move. Banded within regime, because the guest picks its
/// own draw rate per boot and the two populations are not comparable:
///
/// ```text
/// regime ~245-255 draws/frame (GPU 23 % busy)   on 15.45, 15.55 | off 15.24, 15.54
/// regime ~272-274 draws/frame (GPU 41-43 % busy) on 13.77       | off 14.53
/// ```
///
/// At the idle regime the arms interleave, at n=2 against n=2. The one loaded
/// pair reads −5.2 % for `on`, which is close to the ~6.8 % a 12.3 % byte cut
/// off a 55 %-of-GPU-time rail predicts — but it is one pair against one pair,
/// and the same table shows `us/draw` varying by more than that between two
/// boots of the *same* arm. It is a lead, not a result.
///
/// **The 49 % headroom figure is an over-count, and this is the trap to avoid
/// repeating.** `note_extent_headroom` charges every bind that reaches the
/// gather rail — about 2.6 M of them — while the GPU only gathers ~640 k times
/// a boot, because `buffer_bind_reuses` collapses repeats inside one command
/// buffer before any copy happens. `bext_would_save_bytes` is therefore an upper
/// bound over *binds*, not over *gathers*, and the honest reduction is the 12.3 %
/// measured above. Do not flip a default on it.
///
/// # What `on` changes, and what it deliberately does not
///
/// Only which length is compared against the floor. The floor decides whether a
/// bind is worth the zero-copy rail *at all*, and that is a question about the
/// window the guest bound — so it is asked of the full span on both arms. The
/// narrowed length then decides how much of that window is walked and copied.
///
/// That distinction is the whole point. `load_buffer_content`'s doc records an
/// earlier attempt that applied the cap to the rail *decision*: it saved 2.2 GB a
/// run, halved `zc_buffer_held` by pushing binds off the registry onto CPU reads,
/// and bought no time. This arm cannot do that — every bind stays on the rail it
/// was on, with the same registry entry, and only the extent moves.
///
/// It was also ranked on `draw_us/draw`, a CPU wall clock, before this device
/// could time the GPU. The gather is a PCIe copy; no CPU counter was going to see
/// it.
///
/// # Why it is off by default
///
/// Reading fewer bytes than a shader touches is wrong pixels with no error
/// anywhere, and the entire safety argument is the translator's `Object` verdict
/// meaning what it says. `robustBufferAccess` bounds-clamps an over-read rather
/// than leaving it undefined, so the failure mode is visibly wrong rather than
/// unsound — but visibly wrong is still wrong, and this stays off until an A/B
/// has both the `gpu_span` reading and a screenshot.
///
/// It now has both, and the screenshot passed while the number did not: the six
/// boots above bought no time on the host that can be measured here. Reading
/// fewer guest bytes than the guest declared is a correctness risk taken on the
/// translator's word, and a risk with no measured return is not a default.
///
/// What would overturn this is a host where the gather is the constraint rather
/// than 55 % of a device that is idle three-quarters of the second — which is
/// exactly the iGPU column of the support matrix, and see [`GUEST_IMPORT`] for
/// why that host wants a different answer on this whole family. Two boots there
/// settle it, and the arm and its `bext_*` census stay so that they can.
pub const EXTENT_NARROW: &str = "REIMS_VGPU_EXTENT_NARROW";

/// `on` opens the host presentation window borderless fullscreen instead of
/// windowed, on the monitor winit picks.
///
/// A host-side presentation preference and nothing else. It changes the
/// geometry of the window the device presents into — never the frames it
/// presents, never a rail the guest observes, never what the device is
/// capable of. The guest framebuffer, the pointer mapping and the
/// `WindowConfig` size all stay exactly what they would be in a window; the
/// window system just places the window over the whole display, without
/// decorations.
///
/// It is read by `host_window::present` at window creation, which is why it
/// is the only variable here that names a presentation mode rather than a
/// device rail. Default off, because a windowed window is the state an
/// operator can inspect and leave alone; `on` is the state a VM is *used*
/// in. Like every on-switch in this module it adds host-side work (a
/// fullscreen mode change) and can never grant the device a capability.
pub const FULLSCREEN: &str = "REIMS_VGPU_FULLSCREEN";

/// What one variable says, including the two ways it says nothing usable.
///
/// Four states rather than a `bool` because "unset", "explicitly on" and
/// "spelled wrong" are three different operator intents and a `bool` collapses
/// them into the default. The last one matters most: a typo that silently reads
/// as the default is how an operator concludes a switch does not work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    /// Not in the environment, or exported empty — which is how a shell says
    /// "not set" when a variable is assigned from an unset variable.
    Unset,
    /// An affirmative spelling. Never sufficient by itself; see the module doc.
    On,
    /// A negative spelling. This is the state that may change behavior.
    Off,
    /// Present, non-empty, and not one of the spellings below. Carries nothing:
    /// the value is handed back by [`read`] for the caller to name in its own
    /// refusal, because only the caller knows which variable this was.
    Unrecognized,
}

/// The spellings accepted for each state, ASCII-case-insensitively.
///
/// The conventional shell set rather than a chosen one, so an operator does not
/// have to look up which of `0`/`false`/`no` this particular program wanted. The
/// two lists are disjoint and every entry is lowercase, which
/// `the_spellings_are_disjoint_and_lowercase` pins.
const ON_SPELLINGS: [&str; 4] = ["1", "on", "true", "yes"];
const OFF_SPELLINGS: [&str; 4] = ["0", "off", "false", "no"];

/// Classify `name`'s value, and hand back the raw value for a caller that needs
/// to quote it.
///
/// Pure: it reads the environment and parses, and emits nothing. Deliberately —
/// [`crate::observe`] itself reads a variable through here, so an emit on this
/// path would recurse through the sink that is asking whether it is enabled.
/// The caller emits, and it is better placed to: it knows which rail the answer
/// gates and what the consequence of refusing is.
pub fn read(name: &str) -> (Switch, Option<String>) {
    let Some(raw) = std::env::var_os(name) else {
        return (Switch::Unset, None);
    };
    let value = raw.to_string_lossy().into_owned();
    let folded = value.trim().to_ascii_lowercase();
    if folded.is_empty() {
        return (Switch::Unset, None);
    }
    let state = if ON_SPELLINGS.contains(&folded.as_str()) {
        Switch::On
    } else if OFF_SPELLINGS.contains(&folded.as_str()) {
        Switch::Off
    } else {
        Switch::Unrecognized
    };
    (state, Some(value))
}

/// [`read`] for a caller that has nothing to say about the value.
pub fn switch(name: &str) -> Switch {
    read(name).0
}

/// Every variable this device reads.
///
/// The one place the set is enumerable. A boot line built from this reports what
/// an operator actually set, which is the difference between a bug report that
/// says "it is slow" and one that says "it is slow with a rail switched off" —
/// and an operator who mistyped a value learns it from the same line, because
/// [`Switch::Unrecognized`] has its own spelling here.
///
/// Nothing enforces that a new `pub const` above is added to this list; the rule
/// is stated and honestly unenforced. What keeps it small is that the list is
/// next to the constants, and [`report_line`] is the only consumer.
pub const ALL: [&str; 21] = [
    LAZY_WRITEBACK,
    SLAB_RETAIN,
    CLEAR_SEED,
    GUEST_IMPORT,
    DRAW_LOG,
    GPU_STAMP,
    PAGE_GUARDS,
    RANGE_COVERAGE,
    BUFFER_EXTENT,
    BATCH_MIXED_TARGETS,
    UNUSED_BINDS,
    PRESENT_DEPTH,
    STAMP_COALESCE,
    SCATTER_SPLIT,
    COMPUTE_SCATTER,
    // Absent from this list until 2026-08-11, which made the boot line silent
    // about the arm of the one switch here whose two arms are byte-identical
    // implementations — so a `COMPUTE_GATHER` A/B could not be told from a pair of
    // default boots by reading the log afterwards. That is the "compare arms, not
    // pins" trap with the evidence removed.
    COMPUTE_GATHER,
    GPU_SPANS,
    LAYOUT_CHURN,
    PASS_CHURN,
    EXTENT_NARROW,
    FULLSCREEN,
];

/// The state of every variable in [`ALL`], for the one-shot boot line.
///
/// Unset variables are on the line too, and deliberately: the reading a report
/// needs is "these five are the whole set and four of them are default", not a
/// line that goes empty and leaves a reader unsure whether it ran.
pub fn report_line() -> String {
    let mut out = String::from("vgpu_env");
    for name in ALL {
        let (state, value) = read(name);
        let short = name.strip_prefix("REIMS_VGPU_").unwrap_or(name);
        let state = match state {
            Switch::Unset => "unset".to_owned(),
            Switch::On => "on".to_owned(),
            Switch::Off => "off".to_owned(),
            // The raw value, because an operator who typed `REIMS_VGPU_GPU_STAMP=disabled`
            // needs to see what the parse rejected, not just that it did.
            Switch::Unrecognized => format!("unrecognized({})", value.unwrap_or_default()),
        };
        out.push_str(&format!(" {}={state}", short.to_ascii_lowercase()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One process-wide lock for every test that mutates the environment.
    /// `set_var` is process-global and unsynchronized; two tests setting
    /// different variables concurrently is fine, but two setting the *same* one
    /// is not, and these all touch the same probe name.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Set `PROBE` to `value` (or unset it), run `body`, and restore.
    fn with_probe<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        const PROBE: &str = "REIMS_VGPU_TEST_PROBE";
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock above serializes every mutation of this variable in
        // this process, and nothing outside these tests reads it.
        unsafe {
            match value {
                Some(v) => std::env::set_var(PROBE, v),
                None => std::env::remove_var(PROBE),
            }
        }
        let out = body();
        unsafe { std::env::remove_var(PROBE) };
        out
    }

    fn probe(value: Option<&str>) -> Switch {
        with_probe(value, || switch("REIMS_VGPU_TEST_PROBE"))
    }

    /// Both directions, in every spelling the module claims to accept. A
    /// spelling that silently reads as `Unrecognized` is a switch an operator
    /// sets and watches do nothing.
    #[test]
    fn every_documented_spelling_parses() {
        for on in ON_SPELLINGS {
            assert_eq!(probe(Some(on)), Switch::On, "{on}");
            assert_eq!(probe(Some(&on.to_ascii_uppercase())), Switch::On, "{on}");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(probe(Some(off)), Switch::Off, "{off}");
            assert_eq!(probe(Some(&off.to_ascii_uppercase())), Switch::Off, "{off}");
        }
    }

    /// An unset variable and one exported empty are the same answer. `FOO=$BAR`
    /// with `BAR` unset produces the second, and reading it as a value would
    /// make an unrelated typo elsewhere in a boot script silently flip a rail.
    #[test]
    fn unset_and_empty_are_the_same_answer() {
        assert_eq!(probe(None), Switch::Unset);
        assert_eq!(probe(Some("")), Switch::Unset);
        assert_eq!(probe(Some("   ")), Switch::Unset);
    }

    /// A typo is its own answer and keeps its value, so the caller's refusal can
    /// quote what was actually written. Collapsing this into `Unset` is how a
    /// misspelled switch reads as working.
    #[test]
    fn a_value_that_is_neither_keeps_itself_for_the_message() {
        let (state, value) = with_probe(Some("mabye"), || read("REIMS_VGPU_TEST_PROBE"));
        assert_eq!(state, Switch::Unrecognized);
        assert_eq!(value.as_deref(), Some("mabye"));
    }

    /// Surrounding whitespace is not a value. A trailing space picked up from a
    /// heredoc or a `docker run -e` line would otherwise read as a typo.
    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(probe(Some(" off ")), Switch::Off);
        assert_eq!(probe(Some("\t1\n")), Switch::On);
    }

    /// The two lists cannot overlap and are compared lowercased, so an entry
    /// with a capital in it would never match anything.
    #[test]
    fn the_spellings_are_disjoint_and_lowercase() {
        for on in ON_SPELLINGS {
            assert!(!OFF_SPELLINGS.contains(&on), "{on} is in both lists");
            assert_eq!(on, on.to_ascii_lowercase(), "{on} would never match");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(off, off.to_ascii_lowercase(), "{off} would never match");
        }
    }

    /// Every variable the crate honors is named here, spelled consistently. A
    /// name that does not carry the crate prefix is one an operator cannot find
    /// by grepping their own environment.
    #[test]
    fn every_name_carries_the_crate_prefix() {
        // `ALL` rather than a second list: a list written twice is the thing
        // this module exists to stop, and the boot line reads the same one.
        let names = ALL;
        for name in names {
            assert!(name.starts_with("REIMS_VGPU_"), "{name}");
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "{name}"
            );
        }
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                assert_ne!(a, b, "two variables share a name");
            }
        }
    }

    /// The boot line names every variable, including the ones nobody set.
    ///
    /// A line that only reported what was set would go empty on a default boot,
    /// and an empty line cannot be told from an absent one — so a report from a
    /// machine with a rail switched off would look exactly like a report from a
    /// machine with a build that never emitted it.
    #[test]
    fn the_boot_line_names_every_variable_set_or_not() {
        let line = report_line();
        assert!(line.starts_with("vgpu_env "), "{line}");
        for name in ALL {
            let short = name
                .strip_prefix("REIMS_VGPU_")
                .expect("the prefix is asserted above")
                .to_ascii_lowercase();
            assert!(line.contains(&format!(" {short}=")), "{short} in {line}");
        }
    }

    /// A value the parse rejects reaches the line verbatim. An operator who
    /// wrote `disabled` instead of `off` otherwise reads `unset` and concludes
    /// the switch does not work.
    #[test]
    fn an_unrecognized_value_reaches_the_boot_line() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock serializes every mutation of this variable in this
        // process; `report_line` below is the only reader.
        unsafe { std::env::set_var(GUEST_IMPORT, "disabled") };
        let line = report_line();
        unsafe { std::env::remove_var(GUEST_IMPORT) };
        assert!(
            line.contains("guest_import=unrecognized(disabled)"),
            "{line}"
        );
    }
}
