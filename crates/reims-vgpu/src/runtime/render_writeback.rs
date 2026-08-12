//! Land a render Store's frame in the guest's pages.
//!
//! A type-11 render Store names a mapping and a resident image the draw just
//! rendered into. This module copies the one into the other and returns. There
//! is no window, no pin held across the call, and nothing to land later.
//!
//! # The type-11 rail can now be asked to owe the copy instead
//!
//! [`crate::runtime::writeback_debt`] is the rail this doc spent four sections
//! designing and one section burying, built in the one shape the four guest
//! panics do not rule out: nothing resolved is held, and a payment re-derives
//! the identity and re-walks the page tables at the moment it runs. Read that
//! module's doc first if you are here about deferral; what follows is the
//! measurement it stands on, and three of its numbers are now known to be wrong
//! in the ways below.
//!
//! Twelve driven macos-13 sustained-animation boots, six an arm: **90 % of
//! type-11 Stores are superseded before anything reads their pages**, `store_us`
//! falls 0.89 against 9.56 us a chain, `draw_us` 14.62 against 26.52, and five
//! of six on-arm boots present at 105.8-109.2 Hz against a 77.2-78.6 Hz
//! baseline. `store_gva_frame` below is **not** deferred and still writes its
//! ~850 frames a second eagerly.
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
//! # Why the frame is written here rather than deferred
//!
//! It used to be deferred. A Store armed a window naming the pinned resident,
//! and the window was landed either by the next completion stamp or by a host
//! path that touched the mapping's bytes first. The argument for that shape was
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
//! It is the same comparison the type-11 attachment LOAD already elides its CPU
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
//! [`crate::env::SCATTER_SPLIT`] cuts every scatter run into four contiguous
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
//! **That factor is only available if the three stamp sites do not land parked
//! plans**, and it is a contract statement rather than a shortcut. A completion
//! stamp says a submission finished; it does not say the guest may read the
//! resource. [`SettleSite::CompletionStamp`], [`SettleSite::RootStamp`] and
//! [`SettleSite::ChildStamp`] are the three that fire at that cadence, and a
//! settle from any of them still has to wait what is already *submitted* — it
//! simply must not turn a parked plan into a submission. Every other variant is a
//! host toucher of guest bytes and lands everything parked before it reads.
//!
//! `engine::write_stamp_after_guest_writes` needs no change for this: it orders
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
//! back off the GPU. [`crate::runtime::mapping_write::write_bgra8_from_resident_gpu`]
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
//! `backend::vulkan::engine::write_stamp_after_guest_writes`.

use crate::model::DeviceState;
#[cfg(feature = "backend-vulkan")]
use crate::runtime::host::{HostMemory, HostOps};

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
    ScanoutPaint => "settle_scanout_paint",
    /// `drain::write_stamp` — the completion stamp's blocking fallback, taken
    /// when the GPU-ordered stamp path declined.
    CompletionStamp => "settle_completion_stamp",
    /// `drain::drain_main_fifo` — the root packet's completion stamp.
    RootStamp => "settle_root_stamp",
    /// `drain::process_child_packet` — a child packet's completion stamp.
    ChildStamp => "settle_child_stamp",
    /// `mapping_write::write_bgra8_inner` — the copying type-11 Store.
    MappingBgra8Write => "settle_mapping_bgra8_write",
    /// `mapping_write::write_rgba8_image_changed`.
    MappingRgba8Write => "settle_mapping_rgba8_write",
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
/// copies (`engine::write_stamp_after_guest_writes`).
///
/// Free when nothing is outstanding — the engine keeps a debt flag and this
/// returns without touching a queue when it is clear. `site` is what a boot
/// reads to find which caller pays for the waits that are not free; see
/// [`SettleSite`].
pub fn settle_guest_writes(site: SettleSite) {
    #[cfg(feature = "backend-vulkan")]
    {
        // The flag read is one relaxed-acquire load and clear is the common
        // answer, so the census below runs only on the calls that cost
        // something. It can race a writeback armed on another thread between
        // this load and the wait, which makes the site's count a lower bound by
        // at most one per race — the engine re-reads the flag under its own
        // lock and the ordering is unaffected.
        if !crate::backend::vulkan::engine::guest_writes_outstanding() {
            return;
        }
        let started = std::time::Instant::now();
        crate::backend::vulkan::engine::quiesce_guest_writes();
        crate::runtime::drain::note_store_route(site.route());
        crate::runtime::drain::note_store_route_us(
            site.route_us(),
            started.elapsed().as_micros() as u64,
        );
    }
    #[cfg(not(feature = "backend-vulkan"))]
    let _ = site;
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
    #[cfg(feature = "backend-vulkan")]
    {
        if !crate::backend::vulkan::engine::guest_writes_outstanding() {
            return;
        }
        use crate::backend::vulkan::engine::GuestWriteReach as Reach;
        let reach = match pages() {
            Some(p) => crate::backend::vulkan::engine::guest_writes_reaching(&p),
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
    #[cfg(not(feature = "backend-vulkan"))]
    {
        let _ = (site, pages);
    }
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
    // The engine that holds these pins is the Vulkan one; a `backend-metal`
    // build arms nothing that could have pinned them, so taking the list is the
    // whole of the work there.
    #[cfg(feature = "backend-vulkan")]
    for key in &retired {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::backend::vulkan::engine::retire_resident_storage_content(key);
        crate::observe::off(format!(
            "linear_resident_retired task={} ref={} gva={:#x} {}x{} fmt={:#x}",
            key.map_generation,
            key.texture_ref,
            key.surface_offset,
            key.width,
            key.height,
            key.pixel_format
        ));
    }
    #[cfg(not(feature = "backend-vulkan"))]
    drop(retired);
}

/// Copy `identity`'s pixels into `mapping_id`'s guest pages.
///
/// `true` when the guest's pages hold the frame. `false` is a real loss and is
/// reported on the failure channel by the arm that refused — the caller has no
/// second copy to fall back to, because this rail never made one.
#[cfg(feature = "backend-vulkan")]
pub fn store_render_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    width: u32,
    height: u32,
) -> bool {
    let started = std::time::Instant::now();
    crate::runtime::drain::note_store_route("surface_flush");
    // The GPU writes the guest's pages directly. Tried first because when it
    // works there is nothing left to do: no staging buffer is mapped and no
    // host pass over the frame happens at all.
    match crate::runtime::mapping_write::write_bgra8_from_resident_gpu(
        state, host, mapping_id, identity, width, height,
    ) {
        Ok(bytes) => {
            crate::runtime::drain::note_store_route("render_flush_gpu_direct");
            finish(state, mapping_id, identity, bytes as usize, started);
            return true;
        }
        Err(decline) => {
            // Latched per mapping as well as per reason: a host without
            // `VK_EXT_external_memory_host` declines every Store of every
            // surface, and a line each would drown the channel.
            crate::observe::Emit::decline("render_flush_gpu_declined", &decline)
                .field("mapping", mapping_id)
                .field("geom", format!("{width}x{height}"))
                .fail_once(u64::from(mapping_id));
            crate::runtime::drain::note_store_route("render_flush_gpu_declined");
        }
    }
    // The copying arms. These are the only arms on a host that cannot import
    // guest RAM, and the arm a discrete GPU takes regardless.
    //
    // Borrow the readback where it needs no transformation. The writer below is
    // declared in guest scanout order, so a resident reporting semantic RGBA8
    // owes an R/B exchange first — a whole-frame pass, and `into_bgra8` on an
    // owned copy is its home, so a non-BGRA resident takes the copy rather than
    // teaching the lease to rewrite memory it does not own.
    let bpr = width.saturating_mul(4);
    let write_started = std::time::Instant::now();
    // A refused lease is a *routing* answer, never a loss. The lease is an
    // elision of one whole-frame copy and nothing else, so whatever it declines
    // for, the copying rail below can still serve — and it is strictly the
    // better place to find out that the frame is unreadable, because it is the
    // rail that owns the loss report.
    //
    // Collapsing the refusal into `None` rather than enumerating which refusals
    // are recoverable. That enumeration is what would go stale: the lease
    // refuses a resident wider than four bytes a texel (it hands out a pointer
    // into the slot and so has nowhere to narrow one), and reading that
    // particular `Err` as fatal would lose exactly the frames `read_target`
    // exists to quantize. Any future refusal of a rail that is an optimisation
    // has the same answer.
    //
    // Measured, and only reachable one way. On a host that can import guest RAM
    // every Store lands GPU-direct and this whole tail runs zero times — six
    // driven rails, `render_flush_copied` and `render_flush_leased` exactly zero
    // on all of them. With `REIMS_VGPU_GUEST_IMPORT=off`, macos-26 produces
    // **34 lease refusals and 34 copies, one to one, and zero
    // `render_store_lost`**: every one a Surface resident too wide for the lease
    // to lend. Reading the `Err` as fatal cost 34 frames a boot on exactly the
    // host class that has no second rail, and no boot of a capable host could
    // have shown it.
    let leased = match crate::backend::vulkan::engine::read_target_leased(identity) {
        Ok(leased) => leased,
        Err(decline) => {
            crate::observe::off(format!(
                "render_flush_lease_declined mapping={mapping_id} {width}x{height} err={decline}"
            ));
            None
        }
    };
    let (ok, frame_len) = match leased {
        Some(leased) if leased.bgra => {
            crate::runtime::drain::note_store_route("render_flush_leased");
            let len = leased.bytes().len();
            let ok = crate::runtime::mapping_write::write_bgra8_uncached(
                state,
                host,
                mapping_id,
                leased.bytes(),
                bpr,
                width,
                height,
            );
            // End the lease before anything below reaches the engine again: the
            // re-stamp in `finish` does, and a holder blocking on the engine
            // lock while a teardown waits for this lease is the deadlock
            // `LeasedFrame` forbids.
            drop(leased);
            (ok, len)
        }
        // The pool declined the lease (uncached readback memory, where reading
        // the mapping in place is the slower shape), or it refused outright, or
        // the resident is not in scanout order. Drop any leased frame first so
        // its slot is back in the pool before the second readback asks for one.
        leased => {
            drop(leased);
            crate::runtime::drain::note_store_route("render_flush_copied");
            match crate::backend::vulkan::engine::read_target(identity) {
                Ok(rb) => {
                    // Shared rather than owned outright: the write's tail
                    // publishes this frame to the surface cache, and a cache
                    // entry holds its frame behind an `Arc` precisely so the two
                    // can name one allocation instead of copying it.
                    let bytes = std::sync::Arc::new(rb.into_bgra8());
                    let len = bytes.len();
                    let ok = crate::runtime::mapping_write::write_bgra8_owned(
                        state, host, mapping_id, &bytes, bpr, width, height,
                    );
                    (ok, len)
                }
                Err(e) => {
                    crate::observe::fail(format!(
                        "render_store_lost mapping={mapping_id} {width}x{height} \
                         reason=resident_read err={e}"
                    ));
                    return false;
                }
            }
        }
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    if !ok {
        crate::observe::fail(format!(
            "render_store_lost mapping={mapping_id} {width}x{height} reason=write_refused"
        ));
        return false;
    }
    finish(state, mapping_id, identity, frame_len, started);
    true
}

/// Hand the currency witness back to the image the frame came out of, and score
/// the write.
///
/// `write_bgra8_*` ends in `mark_mapping_written`, which advances the mapping's
/// `surface_content_epoch` — correctly, because its guest pages did change. But
/// the *pixels* did not: they are this resident's, copied out of it one
/// statement ago. Leaving the stamp behind invalidates a resident that holds
/// exactly the mapping's content, which costs the next Load its elision and
/// sends it to a CPU seed for bytes it already has.
#[cfg(feature = "backend-vulkan")]
fn finish(
    state: &mut DeviceState,
    mapping_id: u32,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    frame_len: usize,
    started: std::time::Instant,
) {
    if let Some(epoch) = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.surface_content_epoch)
    {
        crate::backend::vulkan::engine::stamp_resident_content_epoch(identity, epoch);
    }
    // The copy above means this image has stopped being the only place these
    // pixels exist, so the reclaim paths may take it.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_store mapping={mapping_id} bytes={frame_len} us={}",
        started.elapsed().as_micros()
    ));
}

/// Why a GVA render Store could not hand its resident straight to the guest's
/// pages, so it fell back to reading the frame back and converting it row by
/// row.
///
/// Every one of these is a routing answer and not a loss — the copying rail
/// still lands the frame — but each costs a blocking GPU→host readback of a
/// whole framebuffer plus a host pass over it, which is the largest single cost
/// this device pays. They are named individually so a boot says which check is
/// holding the volume.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GvaWritebackDecline {
    /// The guest declared a destination format that [`crate::contract::pixel_format::store_texel_order`] does
    /// not admit as a byte-copy destination, so no image→buffer copy can produce
    /// it and `convert_rgba8_to_row` is the only route.
    ///
    /// This used to say "not four bytes of colour", and to name `RGBA16_FLOAT`
    /// as landing here *always*. Both went stale together: a resident now
    /// carries the format the guest declared rather than always being eight bits
    /// per channel, so a half-float destination can be the same bytes as the
    /// image, and `RGBA16_FLOAT` is an admitted eight-byte member of that table.
    /// The rule was never a width — it is whether the destination's texel and
    /// the resident's are one layout, which is [`crate::contract::pixel_format::store_texel_order`]'s question
    /// and not this doc's to restate.
    ///
    /// `R16_FLOAT` is what lands here now, twice on a driven macos-26 boot: it
    /// is renderable but deliberately not a byte-copy destination, for the
    /// reason `store_texel_order`'s own doc gives for `RG16_FLOAT`.
    FormatNeedsConversion { format: u16 },
    /// The resident's format is not the format the destination stores, so a
    /// byte copy would land the wrong texel. Distinct from the engine's own
    /// check of the same pair: this one is asked before the walk, so a mismatch
    /// costs no page-table work.
    ///
    /// Whole formats and not channel orders. The two differ once a render
    /// target may be wider than eight bits per channel: `RGBA16_FLOAT` and
    /// `RGBA8_UNORM` share an order and are four bytes per texel apart, and this
    /// is the arm that catches a half-float destination whose resident fell back
    /// to eight bits because the host cannot render to the wider format.
    ResidentFormatMismatch {
        held: ash::vk::Format,
        want: ash::vk::Format,
    },
    /// The guest's row pitch is not a whole number of texels, or is narrower
    /// than the frame, so there is no `bufferRowLength` that describes it.
    PitchNotTexels { row_stride: u32 },
    /// The frame's first texel does not start on a texel boundary within its
    /// page. `VkBufferImageCopy::bufferOffset` must be a multiple of the texel
    /// block size, and a copy that ignored this is undefined rather than
    /// misaligned.
    OffsetNotTexelAligned { in_page: u64 },
    /// The command resolved no destination pages before it was submitted, so
    /// there is nothing this rail is authorised to write. The copying rail
    /// treats the same answer as "unbounded" and writes anyway through its own
    /// re-walk; this rail has no second walk to fail closed on, so it declines.
    Unlicensed,
    /// The pre-submit walk did not resolve every page of the destination span,
    /// so its page list cannot be read positionally.
    /// [`crate::runtime::draw::StoreTargetPages::ordered_complete`] states what
    /// a short list would land.
    SpanIncomplete,
    /// The destination span did not become a guest-RAM reference; the inner
    /// refusal names the check, restated here for the reason
    /// `GpuWritebackDecline::GuestRefRefused` restates its own.
    GuestRefRefused {
        refusal: crate::runtime::guest_ram_map::MapRefusal,
    },
    /// The engine declined or the copy failed; the inner error names which.
    Engine {
        inner: crate::backend::vulkan::engine::DrawError,
    },
}

#[cfg(feature = "backend-vulkan")]
impl crate::observe::Decline for GvaWritebackDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::FormatNeedsConversion { .. } => "gvawb_format_needs_conversion",
            Self::ResidentFormatMismatch { .. } => "gvawb_resident_format_mismatch",
            Self::PitchNotTexels { .. } => "gvawb_pitch_not_texels",
            Self::OffsetNotTexelAligned { .. } => "gvawb_offset_not_texel_aligned",
            Self::Unlicensed => "gvawb_unlicensed",
            Self::SpanIncomplete => "gvawb_span_incomplete",
            Self::GuestRefRefused { .. } => "gvawb_guest_ref_refused",
            Self::Engine { inner } => crate::observe::Decline::slug(inner),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unlicensed | Self::SpanIncomplete => Vec::new(),
            Self::FormatNeedsConversion { format } => vec![("fmt", format!("{format:#x}"))],
            Self::ResidentFormatMismatch { held, want } => vec![
                ("resident", format!("{held:?}")),
                ("want", format!("{want:?}")),
            ],
            Self::PitchNotTexels { row_stride } => vec![("bpr", row_stride.to_string())],
            Self::OffsetNotTexelAligned { in_page } => vec![("in_page", in_page.to_string())],
            Self::GuestRefRefused { refusal } => {
                let mut f = vec![("via", crate::observe::Decline::slug(refusal).to_string())];
                f.extend(crate::observe::Decline::fields(refusal));
                f
            }
            Self::Engine { inner } => crate::observe::Decline::fields(inner),
        }
    }
}

#[cfg(feature = "backend-vulkan")]
crate::observe::decline::decline_display!(GvaWritebackDecline);

/// Copy `identity`'s pixels into the guest pages behind a type-2/3 render
/// target's `target_gva`, with no host copy of the frame at any point.
///
/// The GVA twin of [`store_render_frame`]'s first arm, and worth diffing
/// against it: both end in `copy_target_to_guest_pages` and they differ only in
/// how the destination pages are named. A mapping carries its own page list and
/// a page-table vouch licenses it; a GVA carries neither, so the licence is the
/// walk the command took **before it was submitted** — `pages`, which the
/// caller resolved at that point and which this rail may not widen.
///
/// # Why this is the whole cost of a GVA Store
///
/// The rail this stands in front of reads the resident back to the host
/// (`read_resident_chain`, a blocking fence) and then writes it out again a row
/// at a time through `convert_rgba8_to_row`. On a driven desktop-compositing
/// boot that is 59 % of render Stores and most of the time the device spends
/// waiting on a fence. Everything this call declines on pays that instead, so a
/// decline is a cost and never a lost frame.
///
/// # What it does not do
///
/// It publishes no host-side copy of the frame. The two GVA pixel caches are
/// *dropped* instead, for the same reason `store_render_frame` forgets the
/// mapping's: after this call the guest's own pages are the only place the
/// frame exists, and an entry left behind would serve a later sample the
/// previous Store's bytes. The sampled rail re-reads them from guest memory,
/// which is what `store_gva_owned`'s `guest_holds_bytes` already promised.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn store_gva_frame<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    identity: &crate::backend::vulkan::engine::TargetIdentity,
    c0: &crate::runtime::draw::ColorRtRequest,
    texture_ref: u32,
    pages: Option<&crate::runtime::draw::StoreTargetPages>,
) -> Result<u64, GvaWritebackDecline> {
    use crate::contract::pixel_format::store_texel_order;
    // The destination's texel layout, and the whole reason this rail can exist
    // at all: a copy converts nothing, so the guest must already read these
    // bytes exactly as the resident holds them.
    let Some(order) = store_texel_order(c0.format) else {
        return Err(GvaWritebackDecline::FormatNeedsConversion { format: c0.format });
    };
    // Compared as whole formats, not as channel orders.
    //
    // While every resident was eight bits per channel an order was a complete
    // description of one, so `is_bgra() != (order == Bgra8)` decided this. It is
    // not any more: `RGBA16_FLOAT` and `RGBA8_UNORM` are both RGBA-ordered and
    // are four bytes per texel apart, so an order comparison would admit a
    // half-float destination over an eight-bit resident and copy a frame of the
    // wrong size and the wrong texel into the guest's pages.
    let want = crate::backend::vulkan::translate::pixel::vk_texel_layout(order);
    let held = identity.resident_format();
    // A healthy zero on the rail as it stands: `gva_chain_identity` builds the
    // key from this same `c0.format`, so the two agree by construction and this
    // arm is the alarm for an identity that came from somewhere else. It is also
    // the arm that catches the honest disagreement — a guest format whose
    // resident fell back to eight bits because the host cannot render to it —
    // and sends that Store down the CPU conversion rail where it belongs.
    if held != want {
        return Err(GvaWritebackDecline::ResidentFormatMismatch { held, want });
    }
    let bpt = u64::from(order.bytes_per_texel());
    let row_stride = u64::from(c0.row_stride);
    if row_stride == 0 || !row_stride.is_multiple_of(bpt) || row_stride < u64::from(c0.width) * bpt {
        return Err(GvaWritebackDecline::PitchNotTexels {
            row_stride: c0.row_stride,
        });
    }
    let Some(pages) = pages else {
        return Err(GvaWritebackDecline::Unlicensed);
    };
    let page_size = state.page_size();
    let Some(gpas) = pages.ordered_complete(c0.target_gva, page_size) else {
        return Err(GvaWritebackDecline::SpanIncomplete);
    };
    let in_page = c0.target_gva % page_size;
    if !in_page.is_multiple_of(bpt) {
        return Err(GvaWritebackDecline::OffsetNotTexelAligned { in_page });
    }
    // The extent the copy names and nothing past it: padding after the final
    // row belongs to the allocation but is not texels this Store was given, and
    // the copying rail leaves it alone too. The two rails must land identical
    // guest memory or a fallback would be visible.
    let extent = u64::from(c0.height.saturating_sub(1)) * row_stride + u64::from(c0.width) * bpt;
    let runs = crate::runtime::guest_ram_map::references_for_runs(
        host, gpas, page_size, in_page, extent,
    )
    .map_err(|refusal| GvaWritebackDecline::GuestRefRefused { refusal })?;
    let target = crate::backend::vulkan::engine::GuestPageTarget {
        runs,
        // Checked above to divide exactly, so this is the guest's pitch and not
        // a rounding of it.
        row_length_texels: (row_stride / bpt) as u32,
        width: c0.width,
        height: c0.height,
        format: want,
    };
    // This device is about to write these guest pages, and the hypervisor's
    // dirty bitmap is defined not to see it. Without this record a reader
    // holding a gathered image over the same pages
    // (`crate::runtime::gather_witness`) cannot tell "nobody wrote them" from
    // "we wrote them ourselves", and vouches a retained image that no longer
    // matches the pages — a wrong frame that then persists.
    //
    // The copying rail this stands in front of records the identical fact from
    // inside `gva_view`, so the two rails leave the same witness behind and a
    // decline is invisible to every reader. Before the submit and not after it,
    // and over the whole walked span rather than the copy's extent: a spurious
    // bump costs a re-read of bytes that did not change, and the opposite error
    // hands out a stale copy.
    state.note_host_wrote_pages(gpas.to_vec());
    crate::backend::vulkan::engine::copy_target_to_guest_pages(identity, &target, gpas)
        .map_err(|inner| GvaWritebackDecline::Engine { inner })?;
    // Nothing here leaves a host copy of the frame, so neither GVA-keyed cache
    // may go on naming one.
    crate::runtime::surface_cache::evict_gva(state, c0.target_gva);
    if texture_ref != 0 {
        crate::runtime::surface_cache::evict_texture(state, texture_ref);
    }
    // The copy means this image has stopped being the only place these pixels
    // exist, so the reclaim paths may take it — the same handover
    // `store_render_frame` performs in `finish`.
    crate::backend::vulkan::engine::note_resident_content_copied_out(identity);
    // Arm the GVA write witness over the pages this Store just published, the
    // twin of `mapper::stamp_guest_write_gen` on the type-11 rail. It is what
    // lets a later reader ask whether these pages still hold this frame without
    // reading them — see `crate::runtime::gva_store_witness`.
    //
    // After the submit, not before it: a stamp taken ahead of a copy that then
    // declines would claim the guest's pages hold a frame that never reached
    // them. And after `note_host_wrote_pages` above, because the epoch the
    // witness records is compared against that same ring — capturing it first
    // would have every target permanently invalidated by its own Store.
    //
    // Only this rail stamps. The copying arm (`gva_store_sync`) leaves no
    // witness, so its targets never read quiet and never take the shortcut this
    // arms. That is safe and deliberate rather than an oversight: it is the arm
    // a host without the guest-RAM import takes, and it already pays a blocking
    // readback per Store, so the shortcut is worth less there and the rails stay
    // easier to tell apart.
    if let Some(key) = crate::runtime::gva_store_witness::GvaTargetKey::of(identity) {
        crate::runtime::gva_store_witness::note_store(state, host, key, gpas);
    }
    Ok(extent)
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
            assert_eq!(
                site.route_us(),
                format!("{}_us", site.route()),
                "{site:?}"
            );
        }
        assert_eq!(seen.len(), SettleSite::ALL.len());
    }
}
