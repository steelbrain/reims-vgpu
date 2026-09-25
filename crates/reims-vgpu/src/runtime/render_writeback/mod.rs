//! Materialize a host-authoritative render target in guest pages.
//!
//! Two halves. This module owns the neutral one: the settle sites, and the wait
//! every host reader of guest bytes owes a rail that may still be writing them.
//! [`vulkan`] owns the copy, because a resident is that rail's — every entry
//! point there takes a `TargetIdentity`. The design below is the rail's as a
//! whole and reads across both.
//!
//! These functions implement the transfer half of resource synchronization.
//! The render Store itself normally leaves the frame in its engine resident;
//! [`crate::runtime::writeback_debt`] calls back here when an explicit
//! synchronize or an actual guest-byte reader requires a guest copy. An eager
//! fallback may call the same functions at Store time, so both timings produce
//! identical bytes through one implementation.
//!
//! # Both render-target representations can owe the copy
//!
//! [`crate::runtime::writeback_debt`] is the rail this doc spent four sections
//! designing and one section burying, built in the one shape the four guest
//! panics do not rule out: mapper-ref-texture mappings are resolved at payment, while a
//! live GVA resource retains the physical-page identity of its transfer
//! backing without retaining raw host pointers. Read that
//! module's doc first if you are here about deferral; what follows is the
//! measurement it stands on, and three of its numbers are now known to be wrong
//! in the ways below.
//!
//! Twelve driven macos-13 sustained-animation boots, six an arm: **90 % of
//! mapper-ref-texture Stores are superseded before anything reads their pages**, `store_us`
//! falls 0.89 against 9.56 us a chain, `draw_us` 14.62 against 26.52, and five
//! of six on-arm boots present at 105.8-109.2 Hz against a 77.2-78.6 Hz
//! baseline. GVA targets use the same ownership rule now; their debt is keyed by
//! the task-local resource reference and paid into its retained transfer backing.
//!
//! ## What that rail corrected here
//!
//! * **"Six reads a second" is the wrong denominator, the same way
//!   `target_reads` was.** Every `settle_*` counter quoted below counts settles
//!   that *waited*; the settle *calls* are three orders larger.
//!   `draw::vulkan::load_linear_guest_memoized` alone reaches its settle ~1 700
//!   times a second and reads the guest pages every one of them. The prize was
//!   never 1 556-to-6.
//! * **Those readers are nameable, so the deferral does not need the seam this
//!   doc calls hard.** They walk raw task GVAs, but they hold a texture
//!   reference and a debt is keyed by a mapping id, so
//!   `writeback_debt::pay_for_texture` resolves one to the other through the two
//!   tables `resource_validity::apply` already uses. `settle_guest_writes`'s
//!   signature never had to change.
//! * **The redundancy is not "supersession across rails and frames" in the sense
//!   below.** It is supersession within *one* rail across frames — which the two
//!   censuses further down could not see, because both measured the spacing of
//!   Stores rather than the spacing of reads.
//!
//! # Why no resolved transfer is held across the ownership window
//!
//! An older design deferred a resolved transfer. A Store armed a window naming
//! the pinned resident, and the window was landed either by the next completion
//! stamp or by a host path that touched the mapping's bytes first. The argument
//! for that shape was
//! coalescing: several passes fully covering one surface inside one submission
//! would land once instead of once each.
//!
//! That coalescing was measured and it never occurred. Arms and lands were
//! equal on every census line of an accumulated driven x86/Vulkan log — 193 458
//! each across 1 780 lines, not one line differing — because no later Store
//! ever fully covered a live window.
//!
//! **That 1.0 is a property of the land policy, not of the workload, and this
//! doc used to read it the other way.** The window landed at the next completion
//! stamp, and a stamp arrives so much more often than a repeat Store that no
//! second Store could reach a window still live. A land point that frequent
//! makes coalescing structurally impossible, so the ratio cannot distinguish "no
//! coalescing was available" from "none was reachable". The measurement stands;
//! the conclusion drawn from it does not, and the ablation below is what says so.
//!
//! # A second census agrees: this rail is close to one Store per surface
//!
//! A later census counted how many *distinct* surfaces a run of Stores names.
//! Sampling the surface rail in fixed batches of 1 024 Stores over a driven
//! Safari drag, each batch touched about **six** distinct mapping ids, and the
//! rail ran at 640 Stores a second (`surface_resident` on a 1 001 ms
//! `store_routes` window) against a 75.8 Hz median present. That is
//! 640 / 6.2 / 75.8 ≈ **1.3 full-target Stores into each surface per frame the
//! user sees** — near the floor of one, and consistent with the paragraph above
//! rather than against it.
//!
//! The arithmetic is worth stating because getting it wrong is easy and it was
//! got wrong once here: dividing the same six surfaces into `target_reads`
//! (~1 560/s) instead gives ~3 Stores per surface per frame and reads as a 2:1
//! redundancy waiting to be collapsed. `target_reads` counts **every** rail's
//! full-frame copy — this one, the GVA Store, and the present capture — so it is
//! the wrong denominator for a per-surface-rail ratio by about 2.4x. Use the
//! route counter for the rail being reasoned about.
//!
//! # And the redundancy is supersession, never identical bytes
//!
//! There is a cheaper repair than any deferral — refuse the copy when the guest's
//! pages already hold these exact pixels — and the witness for it is already
//! built and already maintained from both ends. `finish` stamps the resident
//! with the mapping's `surface_content_epoch` after every Store, and
//! `registry_mark_ready` clears that stamp on every draw that renders into the
//! resident, so `resident_content_epoch(identity) == m.surface_content_epoch` at
//! the top of a Store means nothing has changed the pixels since the last one.
//! It is the same comparison the mapper-ref-texture attachment LOAD already elides its CPU
//! seed on, read from the writing side.
//!
//! **It is zero, and not nearly zero.** A census partitioning every
//! `surface_flush` four ways over two driven macos-13 sustained-animation boots:
//!
//! ```text
//! boot 1   surface_flush 30 239   current 0   stale 0   unstamped 30 239   absent 0
//! boot 2   surface_flush 29 164   current 0   stale 0   unstamped 29 164   absent 0
//! ```
//!
//! 59 403 Stores and not one of them found a resident still stamped. That is
//! structural rather than a property of this workload: a Store is what a draw
//! chain ends in, so a draw has always just rendered into the resident and
//! always just cleared the stamp. The `absent` column reading zero says the
//! sampling is sound — every Store did find its slot to ask.
//!
//! So do not build the identity elision, and do not read the ~5 GB/s as
//! containing duplicate frames. What the traffic contains is **superseded**
//! frames: this rail writes ~414 full surfaces a second into guest pages that
//! only ~18 non-stamp settles a second ever read, and of those only ~7 overlap
//! the pages they are settling for. Every frame written between two reads of a
//! surface is replaced before anything looks at it. That is what a deferral
//! collapses and what an identity check cannot see.
//!
//! So there is no burst of redundant Stores to collapse *inside* this rail, and
//! the deferred window would still have nothing to coalesce. What is left is the
//! rail's own cost at the rate the guest asks for it, and that cost is this
//! device's largest single item: removing only its copy commands, with every
//! barrier, flush and stamp left in place, took a driven drag from 76 Hz to
//! 104 Hz.
//!
//! # It is the bytes, not the queue they are submitted to
//!
//! The obvious reading of that ablation — that the copies are expensive because
//! they sit in the graphics queue ahead of the draws — was tested by building
//! the alternative and measuring it. It is wrong, and
//! `backend::vulkan::engine::context`'s `dedicated_transfer_family` carries the
//! four boots that say so: putting the bus-crossing half of this copy on a host
//! that has an idle copy engine moves the block between three different counters
//! and leaves the frame rate where it was. A narrower ablation isolates why —
//! skipping the image read alone, with the bytes still crossing, is worth 4 Hz of
//! the 30.
//!
//! What is expensive is the traffic: this rail and the GVA Store together put
//! **~5.0 GB/s into guest RAM**, about 21 full-surface writebacks for every frame
//! the user sees. Six surfaces at ~70 Hz would be a third of that even at one
//! write each, so the redundancy is real — it is simply not *within* one rail,
//! which is what the two censuses above were each measuring. Whatever removes it
//! has to look across the rails and across frames, not at the spacing of Stores
//! inside one.
//!
//! ## But "traffic" is regions, not bytes, and that is a different lever
//!
//! The host GPU runs **86-91 % busy at 3-4 % memory utilization** under this
//! load. A rail that were bandwidth bound could not produce those two numbers
//! together; this one is bound by the *number of copy operations* it issues.
//!
//! The count comes from the guest's own allocator. It backs a surface in 16 KiB
//! physically-contiguous granules, so a 1920x1080 window is 2 025 pages in **507
//! runs** (see [`crate::runtime::guest_ram_map::references_for_runs`]), and the
//! `Linear` plan's scatter is one `VkBufferCopy` region per run. At ~414 Stores a
//! second that is ~210 000 regions a second from this rail alone. The runs cannot
//! be merged — non-adjacent in GPA means non-adjacent in the import — so the
//! count is a property of guest memory layout and not of anything this device
//! chooses.
//!
//! Which is what makes the two ablations in
//! `backend::vulkan::engine::context`'s `dedicated_transfer_family` read the way
//! they do: removing the copies entirely was worth ~30 Hz, and skipping only the
//! image *read* while the bytes still crossed was worth 4 Hz of it. The 26 Hz is
//! in the scatter, and the scatter is where the regions are.
//!
//! ## Measured, by making it worse on purpose
//!
//! That reading was tested directly, and the controlled form is the useful one:
//! [`crate::config::SCATTER_SPLIT`] cuts every scatter run into four contiguous
//! sub-ranges that tile it exactly. The guest bytes written are byte-for-byte
//! identical; only the region count changes. Eight driven macos-13 boots, four
//! per arm, one binary:
//!
//! ```text
//! regions/writeback   present_hz per boot        draw_us   duty   slot_us
//! 203 (shipping)      49.15 49.45 56.45 56.40    34-42     0.76-0.88   143-244 ms/s
//! 806 (4x)            26.90 23.80 23.00 23.70    50-63     0.93        365-489 ms/s
//! ```
//!
//! **Four times the regions for the same bytes halves the frame rate** — 49.95
//! to 24.37 median `frames_s`, disjoint at 7x the arms' own spread, eight boots
//! for eight with no overlap. `draws_s` falls 46 %. And it lands exactly where
//! the mechanism predicts: `slot_us` roughly doubles, which is the drain worker
//! blocking longer on a ring fence the GPU takes longer to signal.
//!
//! **The cost is on the GPU, not in the driver's recording of it**, and that is
//! the part which decides what can fix it. Per draw, across the same boots:
//!
//! ```text
//!               slot_us        record_us
//! shipping      10.40  11.54   2.12  2.07
//! 4x regions    24.69  29.07   2.72  2.10
//! ```
//!
//! `record_us` — building the region arrays and calling `vkCmdCopyBuffer` — does
//! not move, while the wait for the GPU nearly triples. Four times the
//! `VkBufferCopy` structs cost almost nothing to write down and a great deal to
//! execute. So a fix has to remove GPU-side per-region work; batching the same
//! regions into fewer calls would not touch this.
//!
//! So this rail is bound by the **number of copy regions it issues**, and the
//! bytes are close to free. That is the opposite of what the ~5.0 GB/s figure
//! above suggests on its face, and it is why every attempt aimed at the bytes —
//! a second queue, damage rects, the parked deferral — either measured nothing
//! or cost more than it saved.
//!
//! The lever it points at carries none of the deferral's hazards: **issue the
//! scatter as one compute dispatch over a run table instead of ~200 transfer
//! regions.** Same bytes, same destination, byte-identical result, nothing held
//! across any window. That is what `backend::vulkan::engine::guest_scatter` now
//! does — a private module, so this names it rather than links it — and the
//! paragraphs below are its measurement rather than the expected value they used
//! to be.
//!
//! ## What it was worth: +48 % frames, measured
//!
//! Eight driven macos-13 sustained-animation boots, interleaved arms of
//! `REIMS_VGPU_COMPUTE_SCATTER`, three excluded by the standing regime rule.
//! Exact Mann-Whitney over the survivors (on n=3, off n=2):
//!
//! ```text
//!                        on        off      delta   separation
//! frames/s            74.92      50.51    +48.3 %       7.1x  disjoint
//! presents/s          76.08      51.74    +47.1 %      10.8x  disjoint
//! draws/s          26 655.9   21 123.0    +26.2 %      15.3x  disjoint
//! regions/writeback     1.0      187                    (the mechanism)
//! ```
//!
//! `p` floors at 0.20 for n=3 vs n=2 and reads that way; the separations are
//! what carry it. Three independent things say the reading is not an artifact:
//! the off arm reproduces the 49.95 Hz baseline recorded above from a different
//! session and a different probe, `present_hz` equals `offered_hz` on both arms
//! so the presenter is not the thing that moved, and the on arm lands between
//! the prediction below and the ablation ceiling above it.
//!
//! The prediction it replaces is kept because it was *right*, which is the part
//! worth trusting next time. Fitting `frame_time = fixed + k * regions` through
//! the two ablation arms — 203 regions at 49.95 Hz and 806 at 24.37 Hz — gave
//! ~35 us of frame time per region, a region-free floor of 12.94 ms, and **~77
//! Hz against 50**: the region count was 35 % of frame time at the shipping
//! ~200. Measured 74.9. Two straight lines through two points is the weakest
//! kind of model and this one landed within 3 %.
//!
//! It was bracketed above as well as below. The ablation in
//! `backend::vulkan::engine::context` that removed this rail's GPU work
//! *entirely* — regions, detile and bytes — reached 104 Hz, so a model putting
//! the region-free point above that would have been refuted on its face. The
//! 74.9 to 104 Hz that remains is the detile and the traffic the compute path
//! still has to do, and is where the next reading of this rail starts.
//!
//! ## The two gates a compute scatter needs, both already open
//!
//! * The imported guest buffer must be bindable as a storage buffer. It already
//!   is: `caps::host_pointer::GUEST_IMPORT_USAGE` includes `STORAGE_BUFFER`, and
//!   that is the exact usage set the capability query asks the driver about — so
//!   a host that admits the import admits this.
//! * The offsets must be addressable in 4-byte units. Run offsets and lengths are
//!   texel-aligned, which is 4 bytes for the eight-bit-per-channel formats this
//!   rail serves, but it is *not* guaranteed for a narrower texel. That is a
//!   check, not an assumption: a run that fails it falls back to the transfer
//!   regions, which stay as the path for it and for any host that declines.
//!
//! # The contract does not ask for this copy at all
//!
//! Nothing in a render Store carries a region, and the search for one is over:
//! the record has no origin, rect, row range or sequence field, and the guest
//! driver's own dirty model has no sub-rect at any layer — a texture is dirtied
//! by *(face, level)* and a buffer by *(start, length)*. The two candidate damage
//! sources on our side were each measured and each said the same thing: the
//! pass's stated render-target extent is the attachment restated (99.97 % `full`,
//! see `exec::report::note_pass_extent_coverage`) and the union of a pass's
//! scissors covers the attachment 99.92 % of the time
//! (`draw::vulkan::note_pass_scissor_union`).
//!
//! The reason there is no region is that the reference host does not copy here.
//! It builds the render target's own GPU resource directly over the guest's
//! surface backing, so a Store makes the pixels guest-visible as a side effect of
//! rendering, at no bandwidth. The only host-to-guest copy in the contract is a
//! **whole-resource synchronize the guest asks for**, guarded per resource by a
//! host-valid flag the guest also owns. This device already decodes both halves —
//! the validity quad in `runtime::resource_validity`, and the synchronize command
//! in `runtime::drain`.
//!
//! **A driven x86/Vulkan Safari-drag boot issues zero of them.** Not few: no
//! resource-synchronize and no resource-invalidate command appears in the whole
//! log. So on this workload the contract asks for no host-to-guest copy at all,
//! and this device performs about 1 556 a second.
//!
//! # What ablating both rails measured
//!
//! A probe returned from the entry of each rail before writing anything, so no
//! guest page was written and no copy was recorded, against the 67.8 Hz baseline
//! of the same tree and machine that hour:
//!
//! | ablated | `present_hz` med | peak |
//! |---|---|---|
//! | nothing (shipping) | 67.8 | 71.9 |
//! | the GVA Store only | 71.9 | 76.8 |
//! | both rails | **86.0** | **108.3** |
//!
//! So ~4 Hz sits in the GVA rail's 928 Stores a second and ~14 Hz in this rail's
//! 628 — this one is the smaller count and by far the larger cost.
//!
//! And the guest still draws. With **no** guest page written at all the desktop
//! at rest is correct to the eye: menu bar, wallpaper, dock, and Safari's start
//! page with every favicon. It degrades under a drag — the composite displaces
//! and regions go black. Read that as "most of this traffic does not feed the
//! display", not as "none of it does": the probe skipped the write *and* the
//! bookkeeping a write publishes (`surface_cache::forget`, the residency-window
//! invalidate, the write footprint), so some of the degradation is the probe's
//! own and the split is not yet apportioned.
//!
//! The lever is therefore not a damage rect and not a different queue — see
//! `backend::vulkan::engine::context`'s `dedicated_transfer_family` for the rail
//! that was built to test the queue and measured nothing. It is landing this copy
//! when something actually reads the bytes, which is what the contract does and
//! what the deferred window above tried to do with the wrong land point.
//!
//! # Who reads these guest pages, counted
//!
//! A deferral is only as good as the list of readers it has to land for, so here
//! is the list with a driven boot's rates beside it. All three are ours to
//! trigger or the guest's to announce; none of them is an unobservable read.
//!
//! * **This device's own colour LOAD seed**, reading an attachment's guest pages
//!   to seed `MTLLoadActionLoad`. [`SettleSite::LinearTextureSeed`] is where it
//!   blocks and it is the device's largest wait — 4 701 in one drag, 99.8 % of
//!   them genuine overlaps. Already elidable through
//!   [`crate::runtime::gva_store_witness`].
//! * **The host console**, painting a mapping's bytes into the host window.
//!   `scanout_paint` fires **six times in a whole boot**; the window presents
//!   from the resident image, not from guest pages.
//!
//!   Read that against [`SettleSite::ScanoutPaint`] and not against the slug it
//!   used to share. `scanout::paint_mapping` has two callers — the console, and
//!   `read_mapping_bgra8`, which is a draw materialising a sampled mapper-ref-texture
//!   texture — and until the sampled arm got [`SettleSite::SampledMappingRead`]
//!   they charged one route. A macos-11 Safari-torture leg read 985 waits and
//!   1.42 s on it, which is three orders of magnitude off the console's rate and
//!   was entirely the second caller.
//! * **The guest CPU**, which announces itself. Zero all boot.
//!
//! Against 1 556 writebacks a second. The `settle_linear_memo_read` pair says
//! the same thing from the other side: 3 796 disjointness checks a second, of
//! which **six** found a read overlapping an outstanding write.
//!
//! # The deferral below was built, and it corrupts the guest's page tables
//!
//! **Read this before building what the next four sections describe.** They are
//! kept because their reasoning about the seam and the land point is sound and
//! was confirmed; what they underestimate is one hazard and overestimate is the
//! prize.
//!
//! It was implemented as specified — the plan resolved and parked at the Store,
//! landed at every settle site except the three completion stamps, replaced on a
//! second Store into the same pages, dropped on `clear_host_valid`, pinned and
//! page-armed at the arm rather than at the record. It ran, and the mechanism
//! worked: one boot armed 2 951 plans, replaced 865, landed 2 782 and lost none.
//!
//! Four driven macos-13 boots, and a fifth on the **same binary** with the rail
//! switched off:
//!
//! ```text
//! rail on    PANIC  "hitting assertion" @AppleParavirtPageTable.cpp:200
//! rail on    PANIC  PTE Corruption detected: ptep ...
//! rail on    PANIC  Possible memory corruption: pmap_pv_remove(...)
//! rail on    PANIC  "hitting assertion" @AppleParavirtPageTable.cpp:200
//! rail off   ok
//! ```
//!
//! Four for four, against eighteen clean macos-13 boots of the same probe that
//! day, and the off arm of the same binary clean. This is the **page recycling**
//! hazard, which the list below names third and treats as one of four: the guest
//! reassigns a surface's backing inside the park-to-land window — its own
//! `MappingEntry::page_entries` doc measures id recycling at ~20 ms under scroll,
//! and that window is ~55 ms — and the landed copy writes a full surface into
//! whatever now owns those pages. Here that was the guest's page tables.
//! `gpuwb_pages_not_ours` fires on the same boot, which is this device saying so
//! from the other side.
//!
//! Landing at the Store is what makes that hazard *not exist*: the pages cannot
//! move inside one call, which is the fourth bullet under "What the window cost
//! that this cannot" and is load-bearing rather than incidental. Any future
//! attempt needs a page-ownership guard held **across** the window — the
//! re-validation at the land has to be that the pages still belong to the
//! mapping, and neither `arm_guest_write_pages` nor the registry pin is that.
//!
//! **And the prize is not the one computed below.** That estimate divides the
//! Stores by the settles that *waited* (~6 a second) and gets ~36 copies a second
//! against 1 556. The land has to run on every settle **call** at a landing site,
//! not on the ones that block, and those are far more frequent. Measured over the
//! four boots, copies actually removed were **27 %, 65 %, 66 %, 65 %** — a factor
//! of about 2.9 at best, not 43. Worth having, and not worth what it cost here.
//!
//! ## Defer the decision, not the plan
//!
//! The hazard is not deferral as such — it is that a *parked plan* holds guest
//! page references resolved at a moment that has passed. A shape that keeps the
//! saving and cannot have the bug: skip the copy at the Store, and at the settle
//! do a **fresh** Store — re-walk the mapping's page tables *then*, resolve runs
//! *then*, copy from the resident *then*. Nothing stale is held across the
//! window, so a surface whose backing the guest recycled is either gone (skip) or
//! re-resolves to the pages it now owns, which is what a Store landing at that
//! moment would have written anyway.
//!
//! What that costs is the thing the seam analysis below already identifies as
//! hard, and it is now the *only* hard part: the land needs `DeviceState` and
//! `HostMemory`, and [`settle_guest_writes`] takes a [`SettleSite`] and nothing
//! else. Threading both through its call sites is the work. It is untested — no
//! boot has run it — and it is recorded here because it is the one variant the
//! four panics above do not rule out.
//!
//! # What a deferral has to answer, and where the seam is
//!
//! Arming instead of writing is the easy half, and a second Store into one
//! mapping should *replace* the armed copy rather than refuse it — the later
//! frame is the fresher answer, and that replacement is the coalescing the
//! stamp-shaped land point made unreachable. The hazards are the four this doc
//! already lists for the old window: resident drift, pin leaks, page recycling,
//! and ordering against the guest's own CPU write. The last one has a signal
//! already decoded — `clear_host_valid` means the guest wrote those bytes, so an
//! armed copy for that mapping must be **dropped**, not landed.
//!
//! **The stamp is not a land point, and that is a contract statement rather than
//! a risk taken.** A completion stamp says the submission is done; it does not
//! say the guest may read the resource's bytes. What says that is the host-valid
//! flag the guest itself sets and clears, and the synchronize it issues before a
//! CPU read. Landing at the stamp is what makes coalescing unreachable, and the
//! contract does not ask for it.
//!
//! **The seam is the plan, not the call graph.** The obvious shape — land from
//! [`settle_guest_writes`] — does not fit: that function takes a [`SettleSite`]
//! and nothing else, no `DeviceState` and no `HostMemory`, and threading both
//! through its sixteen call sites would be the bulk of the work. It does not have
//! to. Split [`crate::backend::vulkan::engine::copy_target_to_guest_pages`] at
//! the point where it stops needing the guest's page tables: everything up to and
//! including `references_for_runs` is `DeviceState`/`HostMemory` work and stays at
//! the Store, and what is left — acquire a scratch, plan the regions, record,
//! submit — needs only the engine, which is a process-global behind its own lock.
//!
//! So a Store resolves its plan and parks it; a settle records and submits every
//! parked plan before it waits. `settle_guest_writes` can reach that with the
//! signature it already has. The per-Store `vouch` and `resolve` cost (12 and
//! 17 ms/s) is unchanged, which is fine — they are not what the ablation
//! measured. The ~5 GB/s is, and it is entirely on the other side of the seam.
//!
//! Two consequences to keep in view while building it. Parking a plan holds
//! resolved host pointers into guest RAM, so [`crate::runtime::guest_ram`]'s bound
//! and the PTE guard have to be armed at the *arm*, not at the record — earlier
//! than today, which is the safe direction. And the pin that keeps the resident
//! alive until the copy executes has to be taken at the arm too, because between
//! arm and land the reclaim paths would otherwise be free to take the image the
//! parked plan reads from.
//!
//! # How big the cut is, and the one variant that must not take it
//!
//! A settle is far rarer than a Store, which is the whole reason this works. One
//! driven Safari-drag census window:
//!
//! ```text
//! gwdebt_merged           1 529     writebacks that found the debt already set
//! settle_linear_memo_read     6     settles that actually waited
//! settle_*                    0     every other site
//! ```
//!
//! Six waits a second against 1 556 writebacks. Parking against about six
//! distinct surfaces and landing at a settle is therefore of order **36 copies a
//! second instead of 1 556** — the same territory the ablation measured at 86 Hz,
//! reached without losing a frame the guest asked for.
//!
//! **That last arithmetic is wrong and the rail that was built measured it.**
//! `settle_linear_memo_read` reading 6 is six settles that *waited*; the site is
//! called ~1 700 times a second and reads the guest's pages every time. The
//! delivered figure is ~3 300 copies a second against ~21 000, not 36 against
//! 1 556 — a factor of six rather than forty-three, and still worth 45 % of the
//! per-draw cost. See [`crate::runtime::writeback_debt`].
//!
//! **That factor is only available if the two fallback stamp sites do not land parked
//! plans**, and it is a contract statement rather than a shortcut. A completion
//! stamp says a submission finished; it does not say the guest may read the
//! resource. [`SettleSite::CompletionStamp`] and [`SettleSite::RootStamp`] are
//! fallback waits for a completion the asynchronous rail could not carry; they
//! still must not turn a parked plan into a submission. Every other variant is
//! a host toucher of guest bytes and lands everything parked before it reads.
//!
//! `engine::write_completion_stamp` needs no change for this: it orders
//! the stamp word behind outstanding copies with a GPU barrier in the same queue
//! and never calls the settle, so a plan that is still parked is simply not
//! something it claims anything about.
//!
//! One caveat for whoever reads the witness this rail feeds:
//! `MappingEntry::render_flush`'s doc quotes `render_flush_age_sub_ms` /
//! `_sub_frame` / `_frame_plus` figures, and **those counters exist nowhere in
//! the tree but that comment** — they were retired without it. Its conclusion may
//! still be right; it is simply no longer reproducible from a boot, so do not
//! read those three numbers as something a fresh log can confirm.
//!
//! What the rail did buy is real and is kept: the Store does not read the frame
//! back off the GPU. [`crate::runtime::mapping_write::vulkan::write_bgra8_from_resident_gpu`]
//! makes the guest's own pages the destination of the copy the GPU was going to
//! make anyway, so nothing crosses host memory on the arm that runs. Landing at
//! the Store keeps that and drops the window.
//!
//! # What the window cost that this cannot
//!
//! Every hazard the deferred rail had to answer came from the window outliving
//! the Store, and none of them can arise here:
//!
//! * **Resident drift.** A window promised pixels from a slot a later draw
//!   could render over, so the land compared a content epoch and refused on a
//!   mismatch — losing the frame. Here the resident is the one the draw just
//!   produced and nothing runs in between.
//! * **Pin leaks.** A window held a registry pin that the reclaim paths skip by
//!   design, so a pin dropped on any early return stranded a framebuffer for the
//!   guest's lifetime. Nothing is pinned here.
//! * **Page recycling.** The guest could hand a window's pages to a different
//!   allocation before it landed, which is the PTE-corruption class the window
//!   guards existed for. The pages cannot move inside this call.
//! * **Write ordering against the guest's own claim.** A window could hold
//!   pixels rendered *before* a guest CPU write to the same resource, and
//!   landing it afterwards clobbered the guest's bytes with stale ones. The
//!   Store and the write are now ordered by when they happen.
//!
//! # Ordering against the guest
//!
//! The copy is recorded into the engine's command stream, not waited on. It is
//! ordered before the guest can observe it by the completion stamp: the stamp
//! word is written behind an `ALL_COMMANDS -> TRANSFER` barrier and every
//! submitted guest-page write settles before the stamp moves. See
//! `backend::vulkan::engine::write_completion_stamp`.

// The backend the process executes on, reached only through the trait.
use crate::backend::Backend as _;
use crate::model::DeviceState;

/// The rail that can copy a resident straight into guest pages.
///
/// Gated on the build — whether this binary carries a Vulkan rail at all is a
/// fact about the build, and every signature in there names a
/// `TargetIdentity`. *Which rail is running* is never asked here: the module's
/// callers are already inside that rail.
#[cfg(feature = "backend-vulkan")]
pub mod vulkan;

/// Declare the settle sites once, and derive both census route names from one
/// slug each: `concat!` builds `<slug>_us` from `<slug>`, so the count route
/// and the cost route cannot drift into naming different sites.
macro_rules! settle_sites {
    ($($(#[$doc:meta])* $variant:ident => $slug:literal,)*) => {
        /// Which host-side toucher of guest bytes is settling.
        ///
        /// # Why the settle names its caller
        ///
        /// The settle blocks this thread until every submitted guest-page
        /// writeback has executed on the GPU, and on a driven boot that block is
        /// the largest single item in the drain worker's wall clock — a
        /// Safari-drag boot spent 15.6 of the worker's 24.7 busy seconds inside
        /// it. It has sixteen call sites and, until this enum, one flag and one
        /// `fence_us` total served all of them, so no boot could say which site
        /// paid it. A fix aimed at that number was aimed by guess.
        ///
        /// Counting *calls* would rank the sites by how often they ask, which is
        /// the wrong ranking: the flag is clear on most calls and those return
        /// without touching a queue. [`settle_guest_writes`] therefore counts
        /// only the calls that actually waited, and charges the microseconds to
        /// the same site, because a site that settles rarely and expensively and
        /// a site that settles constantly and cheaply read identically in a bare
        /// count and want opposite fixes.
        ///
        /// The per-site microseconds and the `readback_split` `fence_us` total
        /// are the same wait attributed twice, and that is the identity worth
        /// checking: every settle in the device comes through here, so
        /// `sum(settle_*_us)` and `fence_us` agree to within the sampling
        /// window. Their diverging means a new caller reached
        /// `engine::quiesce_guest_writes` directly.
        ///
        /// **The identity holds only where the host-pointer import works**, and
        /// a boot that forgets that will read a huge unattributed remainder as
        /// a missing caller. On the copying arm — a host without the extension,
        /// or `REIMS_VGPU_GUEST_IMPORT=off` — no writeback is ever submitted
        /// without waiting, so every site here reads zero while `fence_us` is
        /// the copying rail's own blocking readback, reported as the same
        /// `ReadbackPhase::Fence`. One measured import-off boot: `fence_us`
        /// 8.38 s against zero settles at every site.
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum SettleSite {
            $($(#[$doc])* $variant,)*
        }

        impl SettleSite {
            /// Every site, for the tests that must be exhaustive over them.
            pub const ALL: &'static [SettleSite] = &[$(SettleSite::$variant,)*];

            /// Census route counting the settles at this site that waited.
            pub fn route(self) -> &'static str {
                match self { $(Self::$variant => $slug,)* }
            }

            /// Census cost charged to this site, in microseconds blocked.
            pub fn route_us(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_us"),)* }
            }

            /// Waits [`settle_guest_writes_unless_disjoint`] skipped here
            /// because nothing outstanding lands in what this site reads.
            pub fn route_disjoint(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_disjoint"),)* }
            }

            /// Waits genuinely owed: the outstanding writeback lands in a page
            /// this site is about to read.
            pub fn route_overlap(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_overlap"),)* }
            }

            /// Waits taken because nothing could be ruled out — this site could
            /// not name its pages, or more than one writeback was outstanding.
            pub fn route_unnamed(self) -> &'static str {
                match self { $(Self::$variant => concat!($slug, "_unnamed"),)* }
            }
        }
    };
}

settle_sites! {
    /// `draw::texture_view::load_linear_texture_impl` — CPU read of a linear
    /// texture's guest pages, reached from the Metal-only `load_sampled_rgba`
    /// ladder. The two arms that reach it on the Vulkan pathway name themselves
    /// below.
    LinearTextureLoad => "settle_linear_texture_load",
    /// The same leaf, reached from `draw::seed_color_load` — the colour LOAD
    /// seed reading the attachment's own guest pages to seed a
    /// `MTLLoadActionLoad`.
    ///
    /// **This is the whole of it.** The split was taken to divide 4 438 waits
    /// between this arm and the sampled one below, and a driven Safari drag put
    /// **4 701 here against 0 there**, 4 692 of them genuine overlaps. So the
    /// device's largest remaining wait is one thing and not two: a colour LOAD
    /// blocking on the render Store that published the pages it is seeding
    /// from. The repair is elision — proving the resident still holds what the
    /// Store put in those pages, which is what
    /// [`crate::runtime::gva_store_witness`] answers — and not narrowing, which
    /// an overlap rate of 99.8 % cannot be improved by.
    LinearTextureSeed => "settle_linear_texture_seed",
    /// The same leaf, reached from `draw::vulkan::resolve_sampled_source`'s
    /// last-resort arm, after every rung above it declined.
    ///
    /// Reads **zero** on a driven drag, and that is a real answer rather than a
    /// gap: the arms above it — the GVA resident rung, the zero-copy gather,
    /// the host caches and the memo — take every sampled bind that gets this
    /// far, so nothing reaches the last resort. A non-zero reading here means a
    /// rung above stopped serving.
    LinearTextureSampled => "settle_linear_texture_sampled",
    /// `draw::vulkan::load_linear_guest_memoized` — the memoized full-span CPU
    /// re-read behind every linear sampled bind the gather rail declines.
    LinearMemoRead => "settle_linear_memo_read",
    /// `draw::read_buffer_bytes_resolved` — the one CPU read of a buffer's
    /// guest bytes, reached by buffer-backed sampled textures, the indirect
    /// command buffer decode and the CPU buffer fallback.
    BufferGuestRead => "settle_buffer_guest_read",
    /// `compute_exec::stage_texture_raw` — staging a compute texture's guest
    /// bytes.
    ComputeStageTexture => "settle_compute_stage_texture",
    /// `scanout::paint_mapping` — the host console reading a mapping to paint.
    ///
    /// The doc on [`crate::runtime::render_writeback`] says this "fires six
    /// times in a whole boot", and that is true of the console. It was not true
    /// of this *slug*, because `paint_mapping` has two callers and both charged
    /// it: the console's `scanout_copy_mapping`, and `read_mapping_bgra8`, which
    /// is sampled mapper-ref-texture bind materialisation on a draw. A macos-11
    /// Safari-torture leg read 985 waits and 1.42 s here, which is the second
    /// caller, and the doc's six-a-boot reading was being applied to a number
    /// three orders of magnitude off it. The sampled arm names itself below.
    ScanoutPaint => "settle_scanout_paint",
    /// `scanout::read_mapping_bgra8` — a draw materialising a sampled mapper-ref-texture
    /// texture out of a mapping's guest pages.
    ///
    /// Shares `paint_mapping`'s leaf with the console above and nothing else:
    /// this is a draw-rate reader on the drain worker, the console is a
    /// once-a-boot reader on the display thread, and folding them cost the
    /// console's rate its meaning. The split is what says which of the two the
    /// wait belongs to before anything is aimed at it.
    SampledMappingRead => "settle_sampled_mapping_read",
    /// `drain::write_stamp` — the completion stamp's blocking fallback, taken
    /// when the GPU-ordered stamp path declined.
    CompletionStamp => "settle_completion_stamp",
    /// `drain::drain_main_fifo` — the root packet's completion stamp.
    RootStamp => "settle_root_stamp",
    /// `mapping_write::write_bgra8_inner` — the copying mapper-ref-texture Store.
    MappingBgra8Write => "settle_mapping_bgra8_write",
    /// `mapping_write::write_rgba8_image_changed`.
    MappingRgba8Write => "settle_mapping_rgba8_write",
    /// `mapping_write::write_native_image`.
    MappingNativeImageWrite => "settle_mapping_native_image_write",
    /// `mapping_write::write_raw_rows`.
    MappingRawRowsWrite => "settle_mapping_raw_rows_write",
    /// `mapping_write::read_raw_rows`.
    MappingRawRowsRead => "settle_mapping_raw_rows_read",
    /// `mapping_write::read_rect_raw_at`.
    MappingRectRead => "settle_mapping_rect_read",
    /// `mapping_write::write_rect_raw_at_impl`.
    MappingRectWrite => "settle_mapping_rect_write",
    /// `mapper::write_mapping_bytes_only`.
    MappingBytesWrite => "settle_mapping_bytes_write",
    /// `mapper::read_mapping_bytes`.
    MappingBytesRead => "settle_mapping_bytes_read",
    /// `drain::apply_map_family` on `UnmapMemory` — the guest handing a GVA
    /// span's pages back to its allocator. Not a reader: the pages are about
    /// to belong to someone else, and a writeback still queued into them lands
    /// in whatever the kernel puts there next. A macOS 15 guest panicked with
    /// `RGBA16Float` texels inside a kqueue object after exactly that, while
    /// every write-after-release guard stayed at zero — they all fire at plan
    /// time, and the GPU landing hundreds of microseconds later calls none of
    /// them. The guest unwires the span before it submits this packet, so this
    /// settle is already late for a copy that has landed; it is the last point
    /// that can hold one that has not.
    GuestRelease => "settle_guest_release",
    /// `drain::apply_map_family` on `DeleteIOSurfaceBacking2` — the same
    /// hand-back for a mapping's pages, keyed by mapping instead of by span.
    BackingRelease => "settle_backing_release",
}

/// Block until every guest-page write this device has submitted has executed.
///
/// The writes above are recorded into the engine's command stream and not
/// waited on, which is what makes a Store cheap. A **host-side** reader of the
/// same guest bytes — a mapping read, a CPU seed, a present capture — is not
/// ordered against them by anything the GPU knows about, so it has to settle
/// them first or it reads the pre-Store bytes.
///
/// The guest is ordered separately and does not come through here: its
/// completion stamp is written behind a barrier that already subsumes these
/// copies (`engine::write_completion_stamp`).
///
/// Free when nothing is outstanding — the engine keeps a debt flag and this
/// returns without touching a queue when it is clear. `site` is what a boot
/// reads to find which caller pays for the waits that are not free; see
/// [`SettleSite`].
pub fn settle_guest_writes(site: SettleSite) {
    let backend = crate::backend::selected();
    // The flag read is one relaxed-acquire load and clear is the common answer,
    // so the census below runs only on the calls that cost something. It can
    // race a writeback armed on another thread between this load and the wait,
    // which makes the site's count a lower bound by at most one per race — the
    // rail re-reads the flag under its own lock and the ordering is unaffected.
    // A rail whose Store has already executed when it returns answers `false`
    // here always, and every caller below is the same shape on it.
    if !backend.guest_writes_outstanding() {
        return;
    }
    let started = std::time::Instant::now();
    backend.quiesce_guest_writes();
    crate::runtime::drain::note_store_route(site.route());
    crate::runtime::drain::note_store_route_us(
        site.route_us(),
        started.elapsed().as_micros() as u64,
    );
}

/// [`settle_guest_writes`], skipped when the outstanding writeback lands nowhere
/// near what this caller is about to read.
///
/// A writeback lands in one surface's pages. Most readers that block on it are
/// reading somewhere else entirely — a glyph atlas, a small linear texture — and
/// the wait they take is for a write that will never touch a byte they read. A
/// driven Safari-drag boot spent 11.5 s in one such reader.
///
/// `pages` is resolved by the closure and the closure runs **only** when
/// something is outstanding, so a caller may put a page-table walk in it: the
/// common answer is the debt flag being clear, and that costs one atomic load
/// exactly as [`settle_guest_writes`] does. It must return every page the caller
/// is about to read, and `None` for "cannot say" — a short list would license a
/// read of pages it had omitted, which is a stale frame.
///
/// Three outcomes, counted apart because they want different fixes:
/// `<site>_disjoint` is the wait this saved, `<site>_overlap` is a wait that was
/// genuinely owed, and `<site>_unnamed` is one taken because nothing could be
/// ruled out — a caller whose walk failed, or a second outstanding writeback
/// (`gwdebt_unnamed`).
pub fn settle_guest_writes_unless_disjoint(
    site: SettleSite,
    pages: impl FnOnce() -> Option<Vec<u64>>,
) {
    use crate::backend::GuestWriteReach as Reach;
    let backend = crate::backend::selected();
    if !backend.guest_writes_outstanding() {
        return;
    }
    let reach = match pages() {
        Some(p) => backend.guest_writes_reaching(&p),
        // The caller could not name its own window, which is the same
        // undecidable as the ledger failing to name the writeback's.
        None => Reach::Unnamed,
    };
    crate::runtime::drain::note_store_route(match reach {
        Reach::Disjoint => site.route_disjoint(),
        Reach::Overlap => site.route_overlap(),
        Reach::Unnamed => site.route_unnamed(),
    });
    if reach == Reach::Disjoint {
        return;
    }
    settle_guest_writes(site);
}

/// Release the engine residents of linear cache entries whose task or object
/// the guest deleted this drain.
///
/// Two releases, and dropping either one is a leak in the opposite direction: an
/// unpin alone leaves the image holding the only copy of content nothing may
/// reclaim, and retiring the content alone leaves a pinned slot no reclaim path
/// may take. Together they make the image ordinarily evictable.
///
/// Task teardown means the GPU VA maps are gone, so nothing here writes guest
/// pages — the deleted object's bytes are not guest work any more.
pub fn retire_linear_residents(state: &mut DeviceState) {
    if state.retired_linear_residents.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_linear_residents);
    // What a release means is the rail's; that the list is emptied here is the
    // device's. A rail that pins nothing takes the whole list and does nothing
    // with it, which is the same answer it always gave.
    crate::backend::selected().retire_linear_residents(&retired);
}

#[cfg(test)]
mod tests {
    use super::SettleSite;

    /// Two sites sharing a slug would silently sum their waits into one census
    /// line, and the reading would name the wrong caller as the device's largest
    /// cost — which is the exact mistake [`SettleSite`] exists to stop. Walks
    /// [`SettleSite::ALL`], so a variant added without a slug of its own fails
    /// here rather than at the next boot's ranking.
    #[test]
    fn every_settle_site_carries_its_own_census_route() {
        let mut seen = std::collections::BTreeSet::new();
        for site in SettleSite::ALL {
            assert!(
                seen.insert(site.route()),
                "{:?} reuses the route {}",
                site,
                site.route()
            );
            assert_eq!(site.route_us(), format!("{}_us", site.route()), "{site:?}");
        }
        assert_eq!(seen.len(), SettleSite::ALL.len());
    }
}
