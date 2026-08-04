//! Deferred compute-writeback flush (flush-on-access).
//!
//! A resident-backed type-11 compute storage output may skip both the engine
//! readback and the CPU guest writeback on the stamp path
//! (`ComputeStorageImageResource::defer_readback`): the pinned engine resident
//! is the authoritative content and the guest window is stale. Every host-side
//! read or write of intersecting mapping bytes calls [`flush_intersecting`]
//! first; the flush copies the resident to the host once
//! (`engine::read_resident_storage`, which also unpins) and lands it in the
//! guest window, then re-establishes the residency mirror so chained seed
//! skips keep working.
//!
//! Guest CPU accesses that never cross our host paths cannot be intercepted by
//! anything in this device — the same accepted exposure as resident render
//! targets under `skip_readback`. That is a statement about the device and not
//! about what is possible; `flush_mapping_windows_before_fence` records what a
//! hypervisor-level witness for them would take, and why it has to be a
//! measurement before it is a mechanism. Choke points: `mapping_write` read/write entries,
//! `mapper::read/write_mapping_bytes`, and the drain unmap/ReplacePhysical
//! sites (which drop-with-fail instead of writing through recycled pages).

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Flush every deferred window intersecting `[lo, hi)` on `mapping_id` into
/// guest pages. Returns `false` when any window could not be flushed (the
/// failure is fail-logged; the guest window keeps its stale-but-coherent
/// pre-dispatch bytes).
///
/// Re-entrancy: intersecting entries are removed from the map up front
/// (fixpoint over window unions), so the nested hook fired by the flush's own
/// `write_full_rect_raw_at` finds nothing and recurses no further.
pub fn flush_intersecting<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    lo: u64,
    hi: u64,
) -> bool {
    if state.compute_deferred_flush.is_empty() {
        return true;
    }
    // A condemned backing is an UNDECIDED window, not a dead one.
    // `condemn_surface_backing` deliberately keeps content state — including
    // these deferred windows — because `DeleteIOSurfaceBacking2` may name a
    // prior incarnation of a recycled id whose slot already carries a live
    // surface with an unflushed paint; `mapper::resolve` settles which by
    // comparing the stashed page fingerprint, and reprieves or drops.
    //
    // Taking the windows here defeats exactly that. The page list is stashed in
    // `condemned_entries`, so the flush cannot write (and must not — the pages
    // may be recycled, the boot-16 PTE-corruption class), and the window is
    // consumed on the way to failing: `flush_intersecting` removes it before
    // `flush_one` runs. The fingerprint decision then has nothing left to
    // reprieve, and the loss is reported as `revalidate_condemned` as though the
    // flush were at fault. Leave the obligation armed for the resolve instead.
    // A second delete with no resolve between still tears down for real
    // (`drop_windows`), and the window cap still bounds the population.
    if state.mapping_backing_condemned(mapping_id) {
        // Latched per mapping. Holding is the *expected* outcome for as long as
        // the condemnation is undecided, and a reader hits this choke point on
        // every access: one boot emitted this 15224 times, 13015 of them for a
        // single mapping that stayed condemned for 121 s, which is 7:1 against
        // every other line in the log put together. That drowns the channel this
        // device is diagnosed through, and the rate was never the signal — which
        // mapping is holding is. A real loss is still reported, by
        // `deferred_flush_lost` if the resolve reprieves and the write then
        // fails, or by `deferred_flush_dropped` if it tears down.
        if crate::observe::first_sight("deferred_flush_held", u64::from(mapping_id)) {
            crate::observe::off(format!(
                "deferred_flush_held mapping={mapping_id} reason=backing_condemned lo={lo} hi={hi} (latched)"
            ));
        }
        return true;
    }
    // Fixpoint: a taken window may extend past [lo, hi) and drag further
    // deferred compute siblings into the flush set.
    let mut pending = state.take_deferred_flush_windows(mapping_id, lo, hi);
    let (mut span_lo, mut span_hi) = (lo, hi);
    loop {
        let new_lo = pending
            .iter()
            .map(|(key, _)| key.surface_offset)
            .fold(span_lo, u64::min);
        let new_hi = pending
            .iter()
            .map(|(key, _)| key.span_end)
            .fold(span_hi, u64::max);
        if new_lo == span_lo && new_hi == span_hi {
            break;
        }
        span_lo = new_lo;
        span_hi = new_hi;
        pending.extend(state.take_deferred_flush_windows(mapping_id, span_lo, span_hi));
    }
    let mut ok = true;
    for (key, owner) in pending {
        ok &= flush_one(state, host, &key, owner);
    }
    ok
}

/// Flush deferred windows whose **physical pages** alias a raw task-GVA span.
///
/// The linear-sample fallback (`load_linear_texture_rgba_host`) reads texture
/// content through task page-table walks that never name a `mapping_id`, so it
/// bypasses every mapping-keyed flush choke point — a sample of a
/// resident-authoritative surface through its GVA alias reads the stale
/// pre-Store bytes (boot-18 `m2v_empty_layer reason=linear_sample` poisoning).
/// Resolve the span's pages, match them against each deferred window's mapping
/// pages, and flush the mappings that hit before the caller reads.
///
/// # Why the per-page walk is unconditional
///
/// The walk visits every page of the span through the task page table, and it
/// only ever runs when at least one deferred window is armed — the `is_empty()`
/// early-outs above return first otherwise. Since the four fence bindings of
/// 2026-07-31 collapsed a deferred window's lifetime to a single submission
/// (mean arm-to-flush age ~351 µs), "at least one window armed" is close to
/// never, so the walk is close to never.
///
/// A per-bind no-intersection memo used to sit here, keyed by `(task, gva,
/// span)` and skipping the walk while the deferred-window signature was
/// unchanged. Its justification was a 78 % skip rate over 408 000 calls; that
/// figure predated the fence bindings. Censusing the memo's three outcomes on a
/// driven x86/Vulkan boot — macOS desktop, 25 s of Safari window compositing
/// (2 727 pointer events at 111 Hz, drain duty 0.97, 499 draws/s), 70
/// `store_routes` lines — read `walk = 1`, `skip` never emitted, `recheck`
/// never emitted. The memo answered nothing because the early-outs answered
/// first, so it and its 1-in-64 sampled self-heal are gone. Do not re-derive
/// it: this walk is not a cost on this rail, and caching it reintroduces a
/// hole that only a sampled walk can close.
pub fn flush_intersecting_task_gva<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) {
    if span == 0
        || (state.deferred_alias_pages.is_empty()
            && state.linear_deferred_flush.is_empty()
            && state.gva_deferred_flush.is_empty())
    {
        return;
    }
    // Fast exact-window path: a sample of the deferred GVA surface itself
    // names the same base GVA — no page walk needed to detect it.
    if state.gva_deferred_flush.contains_key(&gva) {
        flush_gva_exact(state, host, gva, true, "gva_alias");
    }
    if state.deferred_alias_pages.is_empty()
        && state.linear_deferred_flush.is_empty()
        && state.gva_deferred_flush.is_empty()
    {
        return;
    }
    let page = state.page_size();
    let n_pages = crate::runtime::gva_mem::pages_spanned(gva, span, page);
    let mut hits: Vec<u32> = Vec::new();
    let mut linear_hits: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = Vec::new();
    let mut gva_hits: Vec<u64> = Vec::new();
    {
        let index = &state.deferred_alias_pages;
        let linear_index = &state.linear_deferred_flush;
        let gva_index = &state.gva_deferred_flush;
        let total = index.len() + linear_index.len() + gva_index.len();
        crate::runtime::gva_mem::visit_task_gva_page_gpas(
            host,
            &state.tasks,
            task_id,
            gva,
            span,
            state.page_shift,
            1,
            &mut |gpa_page| {
                for (&mid, pages) in index.iter() {
                    if pages.contains(&gpa_page) && !hits.contains(&mid) {
                        hits.push(mid);
                    }
                }
                for (key, entry) in linear_index.iter() {
                    if entry.pages.contains(&gpa_page) && !linear_hits.iter().any(|(k, _)| k == key)
                    {
                        linear_hits.push((*key, entry.generation));
                    }
                }
                for (&window_gva, entry) in gva_index.iter() {
                    if entry.pages.contains(&gpa_page) && !gva_hits.contains(&window_gva) {
                        gva_hits.push(window_gva);
                    }
                }
                hits.len() + linear_hits.len() + gva_hits.len() < total
            },
        );
    }
    let hit_ct = (hits.len() + linear_hits.len() + gva_hits.len()) as u64;
    if hit_ct == 0 {
        return;
    }
    // Always-on: a hit-producing walk is rare (six in a whole repro boot), so
    // there is no flood risk and nothing to sample.
    crate::observe::fail(format!(
        "gva_alias_hit_page task={task_id} gva={gva:#x} span={span} \
         n_pages={n_pages} hits={hit_ct}"
    ));
    for mid in hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias mid={mid} task={task_id} gva={gva:#x} span={span}"
        ));
        let _ = flush_intersecting(state, host, mid, 0, u64::MAX);
    }
    for (key, generation) in linear_hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias kind=linear task={} ref={} gva={gva:#x} span={span}",
            key.map_generation, key.texture_ref
        ));
        let _ = flush_linear_one(state, host, &key, generation);
    }
    for window_gva in gva_hits {
        crate::observe::off(format!(
            "deferred_flush_gva_alias kind=gva window={window_gva:#x} task={task_id} gva={gva:#x} span={span}"
        ));
        flush_gva_exact(state, host, window_gva, true, "gva_alias");
    }
}

/// Land every deferred window the guest is about to CPU-read on `mapping_id`.
///
/// SynchronizeResources (child op 0x35, `synchronizeForUnwire`) is the guest's
/// declaration that it will read/pageoff this resource's pages with the CPU —
/// the one host-visible choke point for guest CPU reads, which no device-side
/// flush hook can see (boot-24/25 black-wallpaper class: the fade snapshot is
/// guest-CPU-composited from device-rendered windows whose writebacks were
/// deferred). Mapping-keyed compute windows flush via [`flush_intersecting`];
/// linear task-GVA windows never name a mapping, so they flush when their
/// defer-time page index aliases the mapping's physical pages. Returns
/// `(all_ok, windows_flushed)`.
///
/// # What its counters can and cannot answer
///
/// `guest_read_declared` counts every call, `guest_read_landed` every window one
/// lands, and `guest_read_dry` the calls that found nothing armed. They exist
/// because this is the only place the guest tells us it is about to read, and
/// until now nothing counted it — so "how often does the guest declare?" had no
/// answer, and the eager fence-bound writeback
/// ([`flush_mapping_windows_before_fence`]) was being weighed against an unknown.
///
/// Read them with the order of events in mind. The fence flush runs first and
/// empties the windows, so `guest_read_dry` dominating is the *expected* reading
/// and does **not** show the declaration would have been too late — it shows the
/// fence got there first, which it always does. What the pair does bound is the
/// declaration *rate*: `guest_read_declared` against the composite rate says
/// whether the guest declares once per frame it reads, rarely, or never. A rate
/// far below the flush rate means most flushes land for nobody, which is the
/// case the demand-driven route is for; a rate close to it means the eager rail
/// is doing work the guest would have asked for anyway.
pub fn flush_mapping_for_guest_read<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> (bool, u32) {
    crate::runtime::drain::note_store_route("guest_read_declared");
    // Does the declaration name a surface the eager rail actually writes back?
    // `guest_read_dry` cannot say — the fence always empties the windows first,
    // so every declaration is dry either way. This split can, and it is the
    // number that decides whether the writeback could be demand-driven at all.
    crate::runtime::drain::note_store_route(
        if state.fence_flushed_mappings.contains(&mapping_id) {
            "guest_read_on_flushed_mid"
        } else {
            "guest_read_on_other_mid"
        },
    );
    let keyed = state
        .compute_deferred_flush
        .keys()
        .filter(|k| k.mapping_id == mapping_id)
        .count();
    let mut ok = true;
    let mut flushed = keyed as u32;
    if keyed > 0 {
        ok &= flush_intersecting(state, host, mapping_id, 0, u64::MAX);
    }
    if !state.linear_deferred_flush.is_empty() || !state.gva_deferred_flush.is_empty() {
        let page = state.page_size();
        let page_shift = state.page_shift;
        if let Some(m) = state.mappings.get(&mapping_id) {
            let pages: std::collections::HashSet<u64> = m
                .page_entries
                .iter()
                .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
                .map(|gpa| gpa & !(page - 1))
                .collect();
            if !pages.is_empty() {
                let hits: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = state
                    .linear_deferred_flush
                    .iter()
                    .filter(|(_, entry)| !entry.pages.is_disjoint(&pages))
                    .map(|(key, entry)| (*key, entry.generation))
                    .collect();
                for (key, generation) in hits {
                    ok &= flush_linear_one(state, host, &key, generation);
                    flushed = flushed.saturating_add(1);
                }
                let gva_hits: Vec<u64> = state
                    .gva_deferred_flush
                    .iter()
                    .filter(|(_, entry)| !entry.pages.is_disjoint(&pages))
                    .map(|(&gva, _)| gva)
                    .collect();
                for gva in gva_hits {
                    ok &= flush_gva_exact(state, host, gva, true, "guest_read");
                    flushed = flushed.saturating_add(1);
                }
            }
        }
    }
    if flushed == 0 {
        crate::runtime::drain::note_store_route("guest_read_dry");
    } else {
        crate::runtime::drain::note_store_route_n("guest_read_landed", u64::from(flushed));
    }
    (ok, flushed)
}

/// Land the deferred GVA render-Store window at exactly `gva`, if armed.
///
/// `guest_write` selects the full landing (guest pages + host caches; the
/// task's PTEs must still be live) vs cache-only (unmap/remap/teardown — the
/// map-notify PTE-corruption class forbids guest writes there; the encode
/// cache alone preserves the wallpaper-retain contract). Returns `true` when
/// nothing was armed or the flush landed.
pub fn flush_gva_exact<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    guest_write: bool,
    trigger: &str,
) -> bool {
    let Some(entry) = state.take_gva_deferred_window(gva) else {
        return true;
    };
    flush_gva_one(state, host, gva, &entry, guest_write, trigger)
}

/// Does this window's GVA still resolve to the pages it was armed with?
///
/// `entry.pages` is the whole point of the page-alias trigger: a new guest write
/// is matched against it to decide the window must land first, so that two
/// writers to the same guest memory are ordered. It was recorded when the window
/// was armed, and nothing re-checks it. If the guest has since re-pointed
/// `[gva, gva+span)` at different pages, then the alias matched pages this window
/// no longer owns *and* the write that follows lands in whatever owns `gva` now —
/// the stale-view class, with our own bookkeeping as the stale part.
///
/// §8.53/§8.54 measured only the case where the guest zeroed the PTEs, which is
/// caught by [`crate::runtime::host::MemError::is_guest_teardown`]. Whether a
/// window's pages can move while still resolving was the open question this used
/// to only report on, on the grounds that a guard for an unmeasured hazard is a
/// guess.
///
/// **It is measured now, and it happens.** One x86/Vulkan boot driving Finder,
/// Calendar and Safari produced fourteen of these, and in most of them *every*
/// armed page had moved — `armed_pages=73 live_pages=73 moved=73` for a 196x381
/// window, and the same total displacement at 5, 4 and 22 pages under the
/// `clear_store`, `rearm` and `gva_alias` triggers. So the guard has its
/// measurement and this decides.
///
/// It returns `true` when the window may still be written to guest RAM. Drift
/// means our own bookkeeping is the stale part: the window was armed against one
/// set of guest pages, the guest has since re-pointed `[gva, gva+span)`
/// somewhere else, and [`crate::runtime::metal_draw::write_gva_rgba8`] walks
/// fresh — so the write lands in whatever owns those pages *now*. On this rail
/// that has been observed as guest heap corruption: WindowServer aborting inside
/// `small_free_list_remove_ptr_no_clear`, and the guest kernel panicking with
/// `element modified after free` on a freed allocation overwritten with white
/// RGBA8 pixels.
///
/// Refusing costs stale bytes at a guest address the guest has already
/// repurposed; permitting costs somebody else's heap. The caller keeps the
/// content either way — `host_cache_store_gva_layer` runs unconditionally — so
/// nothing renderable is lost by refusing.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn window_pages_still_ours<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    trigger: &str,
    outcome: &str,
) -> bool {
    deferred_pages_still_ours(
        state,
        host,
        entry.task_id,
        gva,
        entry.span(),
        &entry.pages,
        &format!("{}x{} trigger={trigger}", entry.width, entry.height),
        outcome,
    )
}

/// The drift decision itself, over any deferred window's armed page set.
///
/// Both deferred rails arm against a page set resolved at defer time and then
/// write guest RAM through a *fresh* walk at flush time, so both have the same
/// hazard and the same answer. Keeping one implementation is what stops the
/// second rail from drifting away from the first: the linear rail carried this
/// hazard with no check at all while the GVA rail had one, purely because the
/// check lived inside the GVA-shaped function.
///
/// Returns `true` when the window still names the pages it was armed on.
///
/// `outcome` names what the caller gives up when this answers `false`, because
/// the question has two consumers that lose different things. A flush asks it
/// to keep a write off somebody else's pages (`guest=refused`). The cross-pass
/// resident Load asks it to keep somebody else's pixels from being loaded as
/// this draw's own prior content (`resident=refused`) — the same drift, read
/// from the other side. One hardcoded outcome word would make one line a lie.
#[cfg(feature = "backend-vulkan")]
#[allow(
    clippy::too_many_arguments,
    reason = "the drift question names the window, its armed pages, and what the caller loses"
)]
pub(crate) fn deferred_pages_still_ours<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    span: u64,
    armed: &std::collections::HashSet<u64>,
    what: &str,
    outcome: &str,
) -> bool {
    // Same accounting the mapping rail carries: this function has three ways to
    // return `true` and only one of them checked anything, so a boot reporting
    // no drift on this rail could not say whether the guard had passed or had
    // nothing to pass on. Counted, never gated — every write that landed before
    // still lands.
    if span == 0 || armed.is_empty() {
        crate::runtime::drain::note_store_route("defw_unwit_no_armed");
        return true;
    }
    let mut live = std::collections::HashSet::new();
    live.extend(crate::runtime::gva_mem::task_gva_page_gpa_set(
        host,
        &state.tasks,
        task_id,
        gva,
        span,
        state.page_shift,
    ));
    // The property that makes the write safe is not "the same number of pages
    // came back", it is "every page this write can reach is one the window was
    // given". `write_gva_rgba8` resolves the destination per row from a fresh
    // walk, so the pages it can reach are exactly the ones this walk resolves —
    // and a page of `live` that is not in `armed` is a page some other owner
    // holds now.
    //
    // A subset is the benign teardown case: the guest dropped part of the range
    // and the rest is still ours, so the rows that still resolve land in our own
    // pages and the rest fail per-row on their own terms. That is what the
    // length test was reaching for, and it is not what it tested — `live` can be
    // shorter than `armed` while containing pages that were never ours, because
    // pages can disappear and reappear pointing somewhere else in the same walk.
    // The strictly-shorter arm returned "still ours" for that case, which is the
    // one arrangement of this range that corrupts another owner's memory.
    if live.iter().all(|p| armed.contains(p)) {
        // `all` over an EMPTY set is true, and that is not a verification: it
        // means the walk resolved no page of this window at all, so there was
        // nothing to compare against `armed`. Harmless — `write_gva_rgba8`
        // resolves per row from the same walk, so no row lands either — but it
        // must not be counted as the guard having agreed, which is exactly the
        // conflation that made the mapping rail's `refused = 0` unreadable.
        crate::runtime::drain::note_store_route(if live.is_empty() {
            "defw_unwit_no_live"
        } else {
            "defw_pages_verified"
        });
        return true;
    }
    crate::runtime::drain::note_store_route("defw_pages_drifted");
    crate::observe::fail(format!(
        "deferred_window_page_drift gva={gva:#x} task={task_id} {what} \
         armed_pages={} live_pages={} moved={} foreign={} {outcome}",
        armed.len(),
        live.len(),
        armed.difference(&live).count(),
        live.difference(armed).count()
    ));
    false
}

/// Land every armed GVA render-Store window, because the guest is about to be
/// told the work is finished.
///
/// This is the deferral rail's contract with the guest, and it is the one thing
/// [`deferred_pages_still_ours`] cannot substitute for. A completion stamp is
/// this device's only statement that a render is done; from the instant it lands
/// the guest may free the target, and its own allocator may hand those pages to
/// anything at all without touching a page table — so no later walk, page-set
/// comparison or content test can tell the memory apart from the target it used
/// to be. The only sound moment to write a render's bytes into guest RAM is
/// before the fence that claims they are already there.
///
/// Apple's device needs no equivalent because it has no equivalent window: the
/// render target *is* the guest allocation, so completion and "the bytes are in
/// guest memory" are the same event. This is that invariant restated for a rail
/// that has to copy.
///
/// What the deferral still buys is everything inside one fence: a chain of
/// passes rendering into the same target reuses the registry resident, and
/// `supersede_gva_window` still drops a window the same submission re-renders.
/// What it stops buying is survival across the fence, which was never the
/// device's to sell.
#[cfg(feature = "backend-vulkan")]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.gva_deferred_flush.is_empty() {
        return;
    }
    // Oldest-first, so windows land in the order they were rendered: a later
    // Store at an address the guest recycled within one submission must not be
    // overwritten by the earlier one.
    while let Some((gva, entry)) = state.take_oldest_gva_deferred_window() {
        crate::runtime::drain::note_store_route("gvaw_fence_flush");
        let _ = flush_gva_one(state, host, gva, &entry, true, "fence");
    }
}

/// Metal-direct builds never arm GVA windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_gva_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed linear compute-storage window, for the same reason and under
/// the same contract as [`flush_gva_windows_before_fence`].
///
/// This rail writes a raw task GVA. `ComputeStorageResidencyKey::linear` sets
/// `mapping_id` to 0 and stores the *task id* in `map_generation`, so there is no
/// mapping incarnation to compare and no lifecycle notify anywhere in the wire
/// format — exactly the position the GVA render rail is in, and exactly why
/// `6bc2220` could clear `flush_render_one` and `flush_storage_one` on
/// `map_generation` drift and could not clear this one.
///
/// # Measured before it was repaired
///
/// One x86/Vulkan boot on the crash-hunt workload (Safari on three compositing
/// pages, Finder windows, then 600 s of Mission Control ×71, Spotlight ×71,
/// window drags ×142):
///
/// ```text
/// linw_stamp_same       0
/// linw_stamp_outlived   1     task=5 ref=52 gva=0x39f000 128x135 stamps=1019
/// ```
///
/// Both halves matter. The rail is late whenever it lands at all — the one
/// landing in ten minutes came 1 019 fences after the guest was told the work was
/// done. And it lands almost never, which is what makes the repair free: the
/// objection that stopped the fence repair from being applied to the render rail
/// was the cost of writing back full-screen frames ~98 % of which nothing reads,
/// and there is no such cost here. One window per ten minutes is not a writeback
/// budget.
///
/// A rate this low cannot on its own convict this rail of any guest crash, and no
/// such claim is made. What it does mean is that the correct behaviour is also
/// the cheap one, so there is nothing to trade.
#[cfg(feature = "backend-vulkan")]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.linear_deferred_flush.is_empty() {
        return;
    }
    // Snapshot the keys first: `flush_linear_one` disarms its own window and may
    // flush others through the cache paths below it, so iterating the live map
    // would borrow it across a mutation. A key whose window is gone by the time
    // it comes up disarms to `None` and the flush is a no-op on the guest.
    let armed: Vec<(crate::model::ComputeStorageResidencyKey, u32)> = state
        .linear_deferred_flush
        .iter()
        .map(|(key, entry)| (*key, entry.generation))
        .collect();
    for (key, generation) in armed {
        if !state.linear_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("linw_fence_flush");
        let _ = flush_linear_one(state, host, &key, generation);
    }
}

/// Metal-direct builds never arm linear windows — nothing to land at the fence.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_linear_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every armed mapping-keyed window — type-11 render Stores and compute
/// storage alike — because the guest is about to be told the work is finished.
///
/// This is the last of the four deferred rails to be bound to the fence, and it
/// is bound for a *different* reason from the other three, which is why it was
/// measured first rather than assumed.
///
/// # The other three rails were bound because they could not name their memory
///
/// A `GvaDeferredEntry` and a `ComputeStorageResidencyKey::linear` name a raw
/// address, so a guest that frees the allocation and reuses the pages leaves the
/// window pointing at somebody else's memory with nothing to refuse on. That is
/// not this rail's position: [`flush_render_one`] and [`flush_storage_one`]
/// compare the mapping's live `map_generation` against `key.map_generation` and
/// refuse before reading, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. [`note_mapping_window_against_fence`]
/// records that argument in full and still holds.
///
/// # This rail is bound because the guest is entitled to the bytes at the fence
///
/// A completion stamp is this device's statement that the render is finished. A
/// guest that has been told so may map the IOSurface and read it — CoreGraphics
/// reading back a layer, a damage forward-copy from the previous buffer, any
/// CPU-side compositing step — and it reads *guest RAM*, through its own
/// mapping, without crossing a single host path this device can intercept.
/// `flush_intersecting` covers every reader that goes through us and there is no
/// mechanism that covers the ones that do not.
///
/// So a deferred window is a bet that nothing reads those pages before we land
/// them, and when the bet loses the guest composites the *pre-Store* bytes: a
/// region of the surface holding whatever was there one frame ago, or nothing at
/// all. That is a stale rectangle in an otherwise correct frame, and it is
/// indistinguishable from the corruption classes this device is chasing.
///
/// Apple's device does not take that bet and does not need to. Its render target
/// *is* the guest allocation, so "the render is complete" and "the bytes are in
/// guest memory" are one event. This is that invariant restated for a rail that
/// has to copy: the copy happens before the statement, not after it.
///
/// # And because it clobbers writes the guest itself made
///
/// A deferred window promises to replay a Store later, and that is only a replay
/// while nothing else writes those pages in between. The guest *is* something
/// else: it maps the same IOSurface and does inter-buffer damage forward-copies
/// and CoreGraphics blits into it. The writeback covers the full attachment
/// extent, so every such guest store inside the deferral interval is gone when
/// the window lands. One x86/Vulkan boot on the icon workload (Safari + Finder,
/// 300 s of Mission Control ×41 / Spotlight ×41 / window drags ×82, then four
/// Finder recomposite rounds) measured that directly:
///
/// ```text
/// surface_resident               49 706
/// surface_flush                  12 343    windows that landed
/// render_flush_over_guest_write   8 968    of those, 73 % clobbered guest bytes
/// rendw_stamp_outlived           12 343    every one landed after the fence
/// storw_stamp_outlived              101
/// ```
///
/// `deferred_flush_clobber` is 8 975 lines of that boot's fail log — the largest
/// self-declared loss of guest work anywhere in it.
///
/// **Those are the numbers that motivated the fence, not current ones.** A
/// driven x86/Vulkan boot on today's binary — 30 s Safari window drag plus two
/// web-content probe runs — reads:
///
/// ```text
/// surface_resident               23 196
/// surface_flush                  23 196
/// render_flush_over_guest_write     152    0.66 % of windows, was 73 %
/// rendw_stamp_outlived                0    was 12 343
/// ```
///
/// `rendw_stamp_outlived` going to zero is the structural half: no window is
/// landing after `write_stamp` any more, which is the ordering statement this
/// rail can actually make, and the doc below names its non-zero as the defect
/// to watch. The clobber rate falling with it is what that predicts.
///
/// Do not read the two tables as one experiment. The 73 % boot was a heavier
/// and much more guest-CPU-composited workload (Mission Control, Spotlight,
/// Finder recomposite rounds) over 300 s, so workload and binary are
/// confounded and the ratio between them is not a measured improvement. What
/// the second table does establish is that **the clobber class is no longer the
/// largest loss in the log on the workload this project drives**, and that
/// bounding what a flush copies is now motivated by its byte cost rather than
/// by this correctness hazard.
///
/// [`render_flush_guest_written_ranges`] states why the obvious repair —
/// preserve the pages the guest wrote — is not available: `page_gen[p]` is
/// stamped at the *harvest* that saw page `p` dirty, not at the write, so the
/// witness cannot say whether a store happened before or after the Store this
/// window defers. Preserving on it withheld the device's own frames and turned
/// the screen black (`13ae46d`, 0 of 14 rounds).
///
/// The fence deletes the question rather than answering it. A window that lands
/// before [`crate::runtime::drain::write_stamp`] covers only the interval a
/// synchronous Store would itself have covered, so there is no interval left in
/// which a guest write can be both after the Store and before the writeback.
/// Nothing has to be preserved because nothing is clobbered.
///
/// # What the deferral still buys, and what it stops buying
///
/// Everything inside one fence survives: a chain of passes into the same surface
/// still reuses one resident, and `supersede_covered_render_windows` still drops
/// a window a later Store in the same submission fully covers. What it stops
/// buying is survival *across* the fence, and that is where this rail's cost is,
/// because unlike the linear rail it is not free.
///
/// `arm_surface_resident_store` exists to skip the whole-framebuffer GPU→host
/// readback entirely on the ~86 % of windows nothing ever flushes — `draw_phase`
/// prices that skip at 565 ms per second of wall clock. Landing every window at
/// its fence pays a readback for each: `surface_resident 49 706` against
/// `surface_flush 12 343` bounds it at 4× the current landings. That is the trade
/// this binding makes, and it is a trade rather than a regression only if the
/// measurement says so, and `present_hz` and `draw_us` were read both ways to
/// settle it.
///
/// The GVA rail's binding was expected to cost frame rate and paid back instead
/// (5.9 → 9.5 Hz, `draw_us` 524 ms → 156 ms), because the unbounded rail spent
/// its time in oldest-first `window_cap` eviction storms holding residents pinned
/// across hundreds of frames. `evict_render_windows_to_cap` is the same shape and
/// may go the same way, but that is a prediction and not a reading.
///
/// The endgame removes the trade rather than choosing a side of it: a resident
/// whose image memory *is* the guest pages has nothing to write back, which is
/// why Apple's device has neither this rail nor this cost. That is a backend
/// allocation change, not a scheduling one.
///
/// # What this costs, measured
///
/// It is the single largest cost in the device. On a driven x86 boot (Safari
/// WebGL, 120 Hz) `flush_rails` reads `render_us=688003 render=100` with the
/// gva, linear and storage rails all at zero: **69% of the drain worker's
/// entire second**, against 21% for draws. `readback_split` divides each 6.9 ms
/// flush into submit 7 µs, fence 3.04 ms, staging memcpy 0.83 ms and guest-page
/// write 2.68 ms.
///
/// `fence_us` owning that line reads like latency, and it is not. The GPU
/// timestamp pair taken inside the fence divides it. On a driven Safari
/// window-drag boot, one 1063 ms window reads `render_us=717130` split
/// `fence_us=410022 write_us=290863 submit_us=6163 map_us=430`, and inside the
/// fence `gpu_us=324787 bar_us=729`. So **79% of `fence_us` is the readback
/// command buffer's own execution** — the copy — and the barrier waiting on the
/// draw batch ahead of it is 729 µs across 720 fences, one microsecond each.
/// Summing what scales with bytes (`gpu_us` + `write_us` + `map_us`) against
/// what does not (`bar_us` + `submit_us` + the fence's non-GPU remainder) puts
/// the rail at **86% bytes and 13% latency**.
///
/// That matters because it decides which of the two endgames below is worth
/// building first, and the naive reading decides it wrong. 720 fences that
/// second each copied a full surface, to produce the 11-17 fresh frames the
/// window presented. The rail is not waiting on the GPU; it is moving whole
/// frames that nobody asked for, so bounding *what* each flush copies attacks
/// six sevenths of it and is not blocked on the host being able to address
/// guest memory.
///
/// All of it is speculative. In the same second `mapw_fence_flush` equals
/// `surface_flush` exactly (104 = 104), so **every flush is this fence and none
/// is a guest demand** — [`flush_mapping_for_guest_read`], the
/// `SynchronizeResources` path that fires when the guest actually declares a CPU
/// read, contributes nothing while driving.
///
/// Nothing reads what it produces, either. `RenderFlushWitness` marks each of
/// the two copies a flush lands — the mapping's guest pages and its host surface
/// cache entry — and clears the mark when a host reader takes that copy, so the
/// next flush of the same mapping reports what became of the previous one. A
/// 30 s driven Safari probe scored 3766 landings:
///
/// ```text
/// render_flush_cache_used      15    render_flush_cache_unread   3751
/// render_flush_pages_used      26    render_flush_pages_unread   3740
/// ```
///
/// **0.4% of the cache copies and 0.7% of the guest-page writes are read by
/// anything in the device before the next flush replaces them.** That is not
/// surprising once stated: every device-side reader of these bytes sits below a
/// rung that prefers the GPU resident (`t11rung_resident`, the LOAD elision, the
/// window's resident-carried present), and the resident is exactly what the
/// flush is a copy of. The readers only fall through to a copy when there is no
/// resident to read — and then there was nothing to write back either.
///
/// That is still not licence to drop the writeback, and the witness says so
/// itself: it can only see readers *inside* the device. The guest CPU loads
/// these pages with no device operation at all and has been observed doing it
/// without declaring it (the black-wallpaper fade snapshot named in
/// [`flush_mapping_for_guest_read`]), which is why this fence exists, and after
/// the completion stamp the pages may belong to something else entirely. So the
/// 99% is a bound on what a *cheaper* rail could save, not a licence to delete
/// this one: "write now or never write" is the real choice and this side of it
/// is the safe one.
///
/// What the pair of numbers argues for is not flushing less often but not
/// needing to flush at all. Two routes remain, and the witness rules out a third:
///
/// - The zero-copy endgame above — a resident whose image memory *is* the guest
///   pages. Available only where the host GPU can address host memory.
/// - Making the undeclared guest read observable, so the writeback becomes
///   demand-driven everywhere rather than only on `SynchronizeResources`. That
///   is what would make the rail's cost proportional to its 0.7% of consumed
///   work on a discrete host too.
/// - **Not** "flush only the mappings whose copies get read": which flushes were
///   wasted is knowable only in hindsight, and a mapping whose pages are read
///   while stale has already served wrong pixels.
///
/// The async-readback split (release the device lock across the fence wait) is
/// the step that does not require any of them.
///
/// # What witnessing the undeclared read would take
///
/// The second route is the one that pays on every host, so it is worth being
/// exact about why it is not simply the write witness turned around.
///
/// [`crate::runtime::gather_witness`] skips a gather when two halves agree that
/// nothing wrote a page set: [`HostOps::guest_write_gen`] over the hypervisor
/// dirty bitmap for guest CPU stores, and [`crate::runtime::host_writes`] for
/// this device's own writes, which the bitmap is defined not to see. Neither
/// half has a reading counterpart and no third one can be added, because **a
/// read leaves no trace anywhere**. A dirty bitmap is a record of stores; a page
/// the guest loaded and a page it never touched are the same bits in it. That is
/// a property of the hardware, not a gap in the shim.
///
/// The only way a load becomes observable is to make it fault, which means the
/// page must not be present in the guest's mapping when the load happens. On
/// Linux that is `userfaultfd`: register the pages, punch them out, and the
/// access traps to a handler that supplies the bytes before the vCPU resumes.
/// QEMU already runs this combination for post-copy migration, so KVM taking a
/// uffd fault on guest RAM is settled behaviour rather than a research question
/// — but note what is being borrowed. Post-copy fills a page once and is done;
/// this rail would re-arm the same pages every frame, so the per-fault cost and
/// the arm/disarm cost are on the hot path in a way they never are there.
///
/// **The first build of this should be a counter, not a rail, and the reason is
/// in the numbers above rather than in caution.** The measured case for
/// demand-driving is an upper bound on waste, not a measurement of demand: 0.7%
/// of guest-page writes are read by a device reader and 4.2% of landings are
/// declared by `SynchronizeResources`, and neither can see an undeclared guest
/// load. The undeclared load is known to exist — the black-wallpaper fade
/// snapshot named in [`flush_mapping_for_guest_read`] is one — but its *rate*
/// has never been measured, and the entire value of the route depends on it. So
/// arm a sample of windows (the [`crate::runtime::gather_witness::AUDIT_STRIDE`]
/// shape is the precedent), still write the bytes, still fill correct content on
/// fault, and count faults against landings. That answers "how often does the
/// guest read what nobody declared" for a fraction of one rail's cost, and it is
/// the number that decides whether the fast path is worth building at all.
///
/// Three hazards, recorded because each one turns the result into a wrong one
/// rather than a noisy one:
///
/// - A fault is not a *read*. `UFFDIO_REGISTER_MODE_MISSING` traps the first
///   access of either kind, so a fault count is an upper bound that includes the
///   guest's own stores — which happen on 73% of these windows, per
///   `render_flush_over_guest_write` above. Separating them needs the fault's
///   `UFFD_PAGEFAULT_FLAG_WRITE`, and a rail that ignores it will conclude the
///   guest reads everything.
/// - Arming a page this device is about to write itself is a fault this rail
///   caused, so the arming site has to know every rail that writes guest RAM —
///   and a grep for that has already missed `gva_view::map_fresh_span_within`
///   once.
/// - Punching a page out loses whatever the guest had put there, so the content
///   to fill with has to be captured before the punch, not after.
///
/// ## What the rail is worth, and the second-order cost of not doing it
///
/// The ledger prices the rail by attribution — the parts sum to the whole — but
/// "removing it returns 20 ms" is a different claim, and one a probe that
/// dropped every mapping-keyed render window at the fence answered directly.
/// That probe is gone; its result is here. Measured on one settled x86/PCI
/// guest, same workload, host GPU at P8 throughout, one representative second
/// each — the guest asks for 8.2 composites and ~63 draws per frame in **all
/// three**, so these are one workload at three speeds:
///
/// ```text
///                     control     no-writeback #1   no-writeback #2
/// guest frames/s         34            98                34
/// present_hz             17.4          68.6              16.4
/// duty                    0.97          0.77              0.97
/// flush_us              760 ms        122 ms             70 ms
/// draw_us per draw      103 us         97 us            421 us
/// ```
///
/// **#1 is the price: 2.9x the guest frame rate and 3.9x the displayed one, at
/// unchanged per-draw cost, with the worker no longer saturated.** So the rail
/// is the cap, and the read-witness route is worth its cost.
///
/// **#2 gave it all back, and that is the warning.** `flush_us` stayed
/// collapsed while `draw_us` per draw quadrupled: 54 `t11rung_resident_refused`
/// with `gw_rail_t11_kb=437400` — binds that refused their resident on
/// `guest_replaced` and gathered 8 MB each out of guest RAM. The control has
/// zero such refusals and gathers 0.9 MB a bind.
///
/// The counterfactual provokes that by being wrong — pages left holding neither
/// our frame nor a whole guest one — so it is not what a correct rail would do.
/// It is what a correct rail would do *if it got the witness wrong*, and the
/// exchange rate is terrible: a 2.26 GB/s writeback for an 8 MB-per-bind
/// gather. **Skipping a writeback has to keep the guest-write witness and the
/// type-11 resident rung sound. Only stopping the write is a wash.**
///
/// Six runs over three boots, `fresh` against `t11rung_resident_refused`:
/// control 34 / 36 / 37 with no refusal in any of them, counterfactual **99**
/// on its first drag with none, then 35 and 38 with 54 and 49. The control does
/// not decay across runs and its own first drag reads 37, so "the first drag
/// after a boot is fast" is not the explanation.
///
/// The chain is named by counters, not inferred. `gw_refused_guest_store=121`
/// and `type11_seed_guest_wrote=86` appear in the degraded run and in neither
/// the control nor the fast counterfactual, and `gw_vouched` — 40 windows in
/// the control — is **absent from both counterfactual runs**. That last one is
/// the mechanism: [`crate::runtime::gather_witness`] subtracts this device's own
/// page-exact write record to tell its stores from the guest's, and a rail that
/// never lands never writes that record. Once real guest stores accumulate with
/// nothing to re-baseline against, the witness assumes the worst and the
/// type-11 rung above it follows.
///
/// ## Which pages to register, and where the route stops
///
/// A `userfaultfd` registration only traps accesses made through the VMA it was
/// registered on, so "register the pages" has to mean the VMA KVM's memslot
/// `userspace_addr` points at, and not some second alias of the same physical
/// memory. [`crate::runtime::host::HostOps::map_pages`] is the crate's only
/// handle on a host VA for guest pages, and the two shims answer differently:
///
/// - **x86/PCI is the pathway this works on.** Its shim never allocates: it
///   translates and hands back `memory_region_get_ram_ptr(mr) + xlat`, which is
///   QEMU's own RAMBlock pointer, and answers `map_pages_stable = 1` for exactly
///   that reason. `vm/boot-x86.sh` passes plain `-m` with no memory backend
///   object, so guest RAM is a conventional anonymous mapping — the case uffd
///   handles best. A registration on that range does trap the vCPU.
/// - **arm64/MMIO cannot take this route at all**, for two independent reasons.
///   Its shim answers `map_pages_stable = 0` because a page list that is not
///   host-contiguous gets a packed `mach_vm_remap` view — a second alias, which
///   a fault registration on the RAMBlock would not cover and which would not
///   itself trap the vCPU. And its host is macOS, which has no `userfaultfd`.
///
/// So the read witness is a **Linux-host mechanism**, and shipping it would
/// leave the arm64/macOS pathway on the eager rail with no equivalent. That is
/// not a reason to skip it — x86 is where the cost was measured — but it is a
/// reason not to write it as though it were the general answer, and a reason
/// the eager rail cannot be deleted behind it. The dirty bitmap does not have
/// this problem: KVM indexes it by physical address, so a write through any
/// alias is seen.
///
/// ## `userfaultfd` needs a privilege QEMU does not have
///
/// Measured on the development host (Linux 7.1.3), as the user QEMU runs as:
///
/// ```text
/// userfaultfd(0)                    -> EPERM   /proc/sys/vm/unprivileged_userfaultfd = 0
/// userfaultfd(UFFD_USER_MODE_ONLY)  -> ok
/// ```
///
/// The mode that is available is the one that cannot do the job.
/// `UFFD_USER_MODE_ONLY` exists to stop an unprivileged process trapping
/// **kernel-mode** faults, and a vCPU touching a missing guest page is exactly
/// that: KVM takes the EPT violation and resolves the HVA through
/// `get_user_pages`, in kernel context. (Creatability is measured; that
/// user-mode-only misses the vCPU is the flag's documented purpose and has not
/// been tested here.) A full `userfaultfd` needs `CAP_SYS_PTRACE` on the QEMU
/// binary or `vm.unprivileged_userfaultfd=1`, both of them root changes on the
/// host running the VM.
///
/// That is not fatal, but it decides the shape: the witness cannot be a rail
/// this device silently enables. It is opt-in and it fails visibly when the
/// host will not grant it.
///
/// **The privilege-free alternative is KVM's own, and it is worth costing
/// before assuming uffd.** Deleting the memslot that covers a surface's pages
/// makes guest accesses to them exit to userspace as MMIO — a supported KVM
/// path, no capability, and it reports direction, which uffd MISSING mode does
/// not without reading the fault flags. It is far too slow for a rail (an exit
/// per access against 2 M accesses in a full-frame read) but that does not
/// matter for a counter: un-protect on the first fault, which is all the
/// question "did anything read this landing" needs. What it costs instead is
/// memslot churn — splitting the RAM slot around an 8 MB surface is three
/// `KVM_SET_USER_MEMORY_REGION` calls and a VM-wide EPT flush per arm — which
/// at an [`crate::runtime::gather_witness::AUDIT_STRIDE`]-shaped sample rate is
/// a handful a second. Neither mechanism has been built.
///
/// # Ordering
///
/// Render windows first in arm order, then whatever remains, and both through
/// [`flush_intersecting`] rather than by taking entries directly. That choke
/// point runs the fixpoint that drags in every sibling overlapping the same guest
/// bytes, so windows that overlap land together in one pass whatever order this
/// loop reaches them in — the ordering here decides only which *disjoint* window
/// goes first, and disjoint windows cannot overwrite each other.
///
/// A window may legitimately survive: `flush_intersecting` holds every window on
/// a condemned backing so `mapper::resolve` can settle whether the delete named
/// this incarnation. That hold is the existing contract and the fence does not
/// override it — such a window is not owed to guest RAM until the resolve says
/// the memory is still ours.
#[cfg(feature = "backend-vulkan")]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    if state.compute_deferred_flush.is_empty() {
        return;
    }
    // Once per fence that has anything to land, against `mapw_fence_flush`'s
    // once per window. The ratio is how many windows are armed at the same
    // time, which `surface_resident` and `surface_flush` cannot say — both are
    // per-window rates and read 104/s whether that is 104 fences of one window
    // or 52 of two.
    //
    // **Measured, and the answer is one.** A driven x86/PCI second reads
    // `mapw_fence_pass=116 mapw_fence_flush=116 surface_resident=116
    // surface_flush=116` — all four equal, on three consecutive windows. So the
    // concurrency is exactly 1: every armed window is landed by the very next
    // fence, and no fence ever lands two.
    //
    // Two things follow, and both are negative results worth keeping. Batching
    // the readbacks — one submit and one fence for N windows — cannot pay,
    // because N is 1. And the deferral saves no readback at all: arming and
    // flushing are the same rate, so it only moves the readback from the Store
    // to the fence a moment later. That is consistent with `ResidentArmCensus`
    // measuring a mean arm-to-flush age of 351 us against a 2.6 ms fence, which
    // is the same finding from the other side and is why submitting at arm time
    // was refuted: there is no interval to hide the wait in.
    //
    // A narrower version of that idea is *also* refuted, and it is worth naming
    // separately because it is not obviously the same one. At the moment
    // `arm_surface_resident_store` runs, the engine's draw batch is still open
    // and recording, and the render target is already in `TRANSFER_SRC_OPTIMAL`
    // — so the copy could be appended to the render's OWN command buffer rather
    // than submitted as a second one, which the "submit a separate CB earlier"
    // refutation above does not cover. It still does not pay. The second submit
    // costs ~10 us of the ~1.5 ms `fence_us`; `begin_entry` already flushes the
    // open batch without waiting on it, so the GPU runs render and copy
    // back-to-back and one wait covers both. There is no second pipeline drain
    // to save. Worse, arming is not flushing: an icon workload measured 49 706
    // arms against 12 343 flushes, so recording the copy at arm time would pay a
    // full-frame DMA for four windows in five that nothing ever reads.
    //
    // What is left is volume. These 116 flushes are 116 whole 1920x1080 frames,
    // 962 MB/s, read back for ~62 presented frames; every phase in
    // `ReadbackPhase` is proportional to it.
    //
    // **Reading back less than the whole attachment is measured, and it is not a
    // lever.** The guest supplies a damage rect and this device carries it
    // verbatim (`OPCODE_SET_SCISSOR` -> `req.scissor`), so a damage-limited
    // writeback would be the decoded contract rather than a guess — but
    // `note_store_damage_coverage` reads `store_damage_texels /
    // store_attach_texels` at **99.34%** on a driven probe, with half the Stores
    // carrying no scissor at all and the other half one that spans the
    // attachment. Partial scissors belong to the small draws *inside* a pass;
    // the Store that ends a full-screen composite declares the full screen. The
    // whole rect is worth 0.66%.
    //
    // Moving the bytes without a CPU pass is closed too, and by policy rather
    // than by measurement: it needs `VK_EXT_external_memory_host`, which this
    // pathway does not request, because importing a host pointer over guest RAM
    // gives the host GPU write access to guest memory.
    //
    // Flushing *fewer times* is closed as well, and by the same witness.
    // [`crate::model::RenderFlushWitness::landed_us`] buckets how long each
    // landing survived before the next replaced it, and across two driven
    // probes `render_flush_age_sub_ms` is **0** against 3079 and 3090 at
    // `_frame_plus`. Nothing is ever rewritten inside a millisecond, so the 99%
    // nobody reads is not one surface written repeatedly inside a burst — it is
    // one full-screen composite per displayed frame, landed once each, at
    // exactly the rate the guest paints. Superseding windows across fence
    // boundaries would have nothing to collapse.
    //
    // **A window-move workload does not reopen that, and it is worth recording
    // why not, because the numbers look at first as though it does.** Moving a
    // 1000x640 Safari window at ~115 Hz (`scripts/window-drag-probe`), per
    // second, against an idle control on the same guest of zero draws and zero
    // flushes at `duty` 0.001:
    //
    // ```text
    // surface_flush 212   render_flush_age_sub_frame 139   _frame_plus 73
    // write_split   frag=212  bytes=1758412800  (8 294 400 each: 1920x1080x4)
    // host_window_cadence presents=11 offered=11   drain_duty duty=0.98
    // render_flush_pages_used 0-5    render_flush_pages_unread 212
    // guest_read_declared 3
    // ```
    //
    // Two thirds of landings are replaced inside a frame here, where the WebGL
    // probe had 97% surviving one. But the bucket that means "collapsible" is
    // `_sub_ms` — a burst rewriting one surface inside a single drain tranche,
    // which no fence boundary separated and nothing could have observed between
    // — and it is **still absent**. What moved is `_sub_frame`: landings 1 to
    // 8.33 ms apart, each its own composite behind its own fence. Every one of
    // those fences entitles the guest to the bytes, so collapsing them is the
    // undeclared-read question again and not a separate lever.
    //
    // What the workload *does* establish is the size of the problem, which is
    // larger than the WebGL figure suggested: **212 full-frame writebacks and
    // 1.76 GB/s to put eleven frames on the screen**, with the worker at 0.98
    // duty. The device keeps up with 212 fences a second and presents 11, so
    // roughly 200 composites per second are written back to guest RAM and never
    // displayed. Whether that ratio is the guest asking or this device
    // presenting too little is not answered by any counter here, and it is the
    // question to take next — `validity_wb_unstated=180` against
    // `validity_wb_licensed=32` in the same second is where to start.
    //
    // What is left is not doing it. Every landing is speculative
    // (`mapw_fence_flush == surface_flush`) and 99% of what it lands is read by
    // nothing (`RenderFlushWitness`), so the writeback survives on exactly one
    // case: a guest CPU read that was never declared. Making *that* observable
    // is the remaining route, and it is a hypervisor-side change rather than a
    // device-side one.
    //
    // ## How much of it lands for nobody, and why declarations cannot replace it
    //
    // That paragraph rested on one witness, which sees only *device* readers.
    // The guest's own declared reads are now counted too
    // ([`flush_mapping_for_guest_read`]), so both consumers can be bounded at
    // once. One driven x86/PCI boot, three Safari probes, summed over its
    // `store_routes` windows:
    //
    // ```text
    // mapw_fence_flush           8051   windows landed
    // render_flush_pages_used     187   landings a device reader consumed  (2.3%)
    // render_flush_pages_unread  7774
    // guest_read_declared         778   guest declarations of a CPU read
    // guest_read_on_flushed_mid   339   of those, on a mapping this rail writes (4.2% of landings)
    // guest_read_on_other_mid     439
    // ```
    //
    // So **at most ~6.5 % of the writeback has a witnessed consumer**, and the
    // two sets may overlap, so that is a ceiling rather than a total. Ninety-odd
    // per cent of the largest cost in this device lands for a consumer nobody
    // has ever observed. That is the case for building the read witness.
    //
    // It is also the case against the cheap version of it. Declarations cover
    // 4.2 % of landings, so **dropping the eager rail and relying on op 0x35
    // would lose the other 95 %** — the tripwire is required, not an
    // optimisation on top of the declarations. And the two rates do not move
    // together: an earlier, lighter boot read 1035 declarations against 3379
    // landings (0.31 each) where this one reads 778 against 8051 (0.10), so the
    // flush rate scales with rendering and the declaration rate does not. The
    // gap widens exactly when the cost matters most.
    //
    // Note also `guest_read_dry` is 778 of 778 on both boots. That is expected
    // and is not evidence of anything: this fence empties every window before
    // any declaration can arrive, so a declaration can never land one.
    //
    // # One of the two CPU passes over the result is gone; the other is the floor
    //
    // The four closed levers above are all about the readback. The *number of
    // CPU passes over its result* was a separate cost, and it was reducible.
    //
    // A flush used to make two passes over ~8 MB: `readback_split map_us` copied
    // the mapped staging buffer into a `Vec<u8>`, then `write_split land_us`
    // scattered that Vec into guest pages — about 0.82 ms and 1.06 ms per frame,
    // together ~250 ms of a loaded second, as large as the fence. The first
    // existed only so the host surface cache could hold an `Arc<Vec<u8>>`, and
    // `render_flush_cache_used` prices that entry at 0.4 %.
    //
    // It is deleted. `read_target_leased` lends the staging buffer through
    // [`crate::backend::vulkan::engine::LeasedFrame` ] and the scatter reads it
    // in place. Measured on a driven x86/PCI boot, three consecutive one-second
    // windows at 120 flushes each:
    //
    // ```text
    // readback_split  map_us=0 map=120 map_max_us=0
    // write_split     stage_us=0 stage=0 land_us=104832 land=120 cache_us=0
    // store_routes    render_flush_leased=120 surface_flush=120
    // ```
    //
    // `map_max_us=0` is the part worth keeping: not an average that rounded
    // down, but no single flush in 360 spending a microsecond there. What the
    // phase still times on that arm is the `vkInvalidateMappedMemoryRanges` a
    // non-coherent readback owes, and this host's readback memory is coherent.
    //
    // Two things make the borrow sound and both are load-bearing. The slot is
    // taken out of every list that could hand it to a GPU copy — including the
    // ring entry's pending cleanup, which is why the lease is claimed before
    // `seal_entry` — and the holder takes no engine lock, which is why
    // `flush_render_one` runs `flush_windows_under_bgra8_write` *before* it
    // acquires the frame rather than letting the writeback's own
    // `flush_intersecting` read another resident from inside the borrow. Do
    // *not* simply hold the engine lock across the scatter instead: the host
    // window's present path takes it, and adding a millisecond per flush at this
    // rate would move the stall onto the window.
    //
    // The cache entry is **invalidated**, not left behind. A reader that hits a
    // stale entry is served an old frame with no witness saying so, which is the
    // corruption shape the fence binding exists to close. Falling through to the
    // guest pages this flush just wrote is correct by construction, and the boot
    // above logged zero `present_capture FAIL` and zero `deferred_flush_lost` on
    // the leased rail.
    //
    // What is left is the floor. `land_us` is 0.87 ms per frame at ~1 GB/s of
    // cache-cold scattered writes into guest RAM, and there is no second pass to
    // remove — the only way past it is not to write the bytes at all, which is
    // the demand-driven route named above.
    crate::runtime::drain::note_store_route("mapw_fence_pass");
    // Snapshot first: landing one window consumes its overlapping siblings
    // through the fixpoint, so iterating the live map would borrow it across a
    // mutation. A key already consumed by an earlier pass is skipped rather than
    // re-flushed.
    for key in mapping_windows_fence_order(state) {
        if !state.compute_deferred_flush.contains_key(&key) {
            continue;
        }
        crate::runtime::drain::note_store_route("mapw_fence_flush");
        state.fence_flushed_mappings.insert(key.mapping_id);
        flush_intersecting(
            state,
            host,
            key.mapping_id,
            key.surface_offset,
            key.span_end,
        );
    }
}

/// Mark the host surface cache copy of `mapping_id` as taken by a host reader.
///
/// Called beside every mapping-keyed read of [`crate::runtime::surface_cache`],
/// which is the leg [`flush_render_one`] stores through. Unknown mappings are
/// ignored: a cache entry outlives its mapping, and a read of one says nothing
/// about a flush there is no longer an entry to attribute it to.
pub fn note_render_flush_cache_read(state: &mut DeviceState, mapping_id: u32) {
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.render_flush.cache_unread = false;
    }
}

/// Mark this mapping's guest pages as gathered by a host reader — the other leg
/// [`flush_render_one`] writes.
pub fn note_render_flush_pages_read(state: &mut DeviceState, mapping_id: u32) {
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.render_flush.pages_unread = false;
    }
}

/// Report what read the previous landed flush of this mapping, then arm the
/// witness for the one landing now.
///
/// The pair of counts per leg is what makes the flush's cost answerable:
/// `render_flush_cache_used` / `render_flush_cache_unread`, and the `pages_`
/// pair beside them, divide a gigabyte a second of readback into the part
/// something asked for and the part nothing did. A mapping whose first flush is
/// landing now is not counted, so an arriving surface is never scored as unread
/// work.
///
/// Only where the rail exists. A Metal-direct build never arms a mapping-keyed
/// window — `flush_mapping_windows_before_fence` is a no-op there — so there is
/// no landing to score, and the two readers above stay unconditional only
/// because clearing a flag that was never set costs nothing and keeps the
/// scanout and sampled rungs free of a cfg.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn note_render_flush_landed(
    state: &mut DeviceState,
    mapping_id: u32,
    cache_stored: bool,
) -> Option<crate::model::RenderFlushWitness> {
    use crate::runtime::drain::note_store_route;
    let now_us = crate::observe::elapsed_us();
    let m = state.mappings.get_mut(&mapping_id)?;
    let prior = std::mem::replace(
        &mut m.render_flush,
        crate::model::RenderFlushWitness {
            landed: true,
            cache_stored,
            cache_unread: cache_stored,
            pages_unread: true,
            landed_us: now_us,
        },
    );
    if !prior.landed {
        return None;
    }
    // How long the flush being replaced survived. Bucketed against the VBL
    // interval, because that is what separates "the compositor repainted"
    // from "this surface was written twice inside one drain tranche".
    note_store_route(match now_us.saturating_sub(prior.landed_us) {
        0..=999 => "render_flush_age_sub_ms",
        1000..=8332 => "render_flush_age_sub_frame",
        _ => "render_flush_age_frame_plus",
    });
    // Only where the previous flush actually stored one. A borrowed-frame flush
    // publishes nothing to the cache, and counting its absent copy as unread
    // would inflate the very number that prices the cache leg.
    if prior.cache_stored {
        note_store_route(if prior.cache_unread {
            "render_flush_cache_unread"
        } else {
            "render_flush_cache_used"
        });
    }
    note_store_route(if prior.pages_unread {
        "render_flush_pages_unread"
    } else {
        "render_flush_pages_used"
    });
    Some(prior)
}

/// Metal-direct builds never arm mapping-keyed windows — nothing to land.
#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_mapping_windows_before_fence<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
) {
}

/// Land every deferred rail. Call this immediately before any word that tells the
/// guest work has finished.
///
/// There is more than one such word. The child stamp slots go through
/// [`crate::runtime::drain::write_stamp`], but the *root* completion stamp is
/// written straight into slot 0 by the main FIFO drain, and it is the one the
/// guest's root packets wait on. A rail bound only to the child path is not bound:
/// the guest may free a render target the moment the root stamp moves, and its
/// allocator may hand those pages to anything — a kalloc element, another
/// process's heap — which no later check can tell from the target they used to be.
///
/// So the binding belongs to "the guest is about to be told", not to one of the
/// two writers that tell it. Every caller of this function is such a site, and a
/// new completion word is a new caller.
///
/// Each rail early-returns when nothing is armed, so the common case — a root
/// packet completing with no deferred window outstanding — costs three map
/// emptiness checks.
pub fn flush_all_windows_before_fence<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
) {
    // The two address-named rails carry the free-then-reuse hazard: they name raw
    // guest addresses with no mapping incarnation to refuse on, so nothing but
    // this ordering keeps them off memory the guest has reclaimed.
    flush_gva_windows_before_fence(state, host);
    flush_linear_windows_before_fence(state, host);
    // The mapping-keyed rails can refuse a replaced incarnation, so they are here
    // for the other hazard: a deferred writeback covers the whole attachment
    // extent while the guest writes the same IOSurface, and landing inside the
    // fence leaves no interval for that to happen in.
    flush_mapping_windows_before_fence(state, host);
}

/// The order [`flush_mapping_windows_before_fence`] lands windows in: render
/// windows oldest-first by `armed_seq`, then every other window.
///
/// Only the render rail carries an arm sequence, and only the render rail can
/// hold several live windows on one mapping at once (different planes, different
/// geometries at the same offset). Compute storage windows are keyed by the
/// dispatch span that produced them and are appended in key order, which is the
/// order every other flush trigger has always used.
#[cfg(feature = "backend-vulkan")]
fn mapping_windows_fence_order(
    state: &DeviceState,
) -> Vec<crate::model::ComputeStorageResidencyKey> {
    let mut render: Vec<(u64, crate::model::ComputeStorageResidencyKey)> = Vec::new();
    let mut rest: Vec<crate::model::ComputeStorageResidencyKey> = Vec::new();
    for (key, owner) in &state.compute_deferred_flush {
        match owner {
            crate::model::DeferredOwner::Render { armed_seq, .. } => {
                render.push((*armed_seq, *key))
            }
            crate::model::DeferredOwner::Storage { .. } => rest.push(*key),
        }
    }
    render.sort_unstable_by_key(|(seq, _)| *seq);
    render.into_iter().map(|(_, key)| key).chain(rest).collect()
}

/// Score a deferred window about to write guest RAM against the guest's fence.
///
/// [`crate::runtime::drain::write_stamp`] is the only thing this device says to
/// the guest about whether work is finished. Once it has moved, the guest is
/// entitled to free everything it allocated for that work — and the guest's own
/// allocator is then free to hand those pages to anything, without touching a
/// page table. So a window armed at stamp N and landed at stamp N+k, k > 0, is a
/// write to memory the guest was told it could reclaim k fences ago.
///
/// [`deferred_pages_still_ours`] cannot see this. It asks whether the GVA still
/// resolves to the pages the window was armed on, and free-then-reuse inside one
/// process preserves the translation exactly. That is why the guard landed and
/// the WindowServer `small_free_list_remove_ptr_no_clear` aborts continued.
///
/// The counters carry their own denominator — `gvaw_stamp_same` against
/// `gvaw_stamp_outlived` in the per-second `store_routes` line.
///
/// # Measured, and it is not a tail
///
/// One x86/Vulkan boot driving the workload the user's report names (Safari on
/// three compositing-heavy pages, Finder windows, then 600 s of Mission Control
/// ×71, Spotlight ×71 and window drags ×142 — every one of them a window-list
/// capture compositing a backdrop blur, which is the frame the report crashed
/// in):
///
/// ```text
/// gvaw_stamp_same       0
/// gvaw_stamp_outlived 810
/// ```
///
/// **Zero.** Not a minority, not a tail: every deferred GVA window that wrote
/// guest RAM on that boot wrote it after the guest had been fenced. The elapsed
/// stamp counts say how far after — over 227 latched spans, median 133 fences,
/// p90 1 099, max 1 601. The guest was told this work had finished 133 times
/// over before the device put the bytes in its memory.
///
/// The trigger breakdown says why: 215 of 227 land under `window_cap`, the
/// oldest-first eviction that runs when `GVA_DEFERRED_WINDOW_CAP` is reached. So
/// the rail's normal exit is not a flush anything asked for; it is a window
/// sitting until the cap pushes it out, hundreds of fences past the point the
/// guest was free to reclaim it.
///
/// And the geometry names the second defect as well as the first. The largest
/// single population is **64x64, 65 of 227** — a folder icon exactly, the same
/// geometry the surviving Finder icon class corrupts at. The icons that come out
/// wrong are the windows written into guest memory long after the guest was told
/// they were done.
///
/// No userspace crash fired during those 600 s, so this boot does not by itself
/// convict the rail of the WindowServer abort. What it establishes is that the
/// hazard is not rare, not a corner, and not something a page-set guard can see.
///
/// # After the repair, on the same harness
///
/// [`flush_gva_windows_before_fence`] inverts it completely:
///
/// ```text
///                      before repair   after repair
/// gvaw_stamp_same                  0         54 932
/// gvaw_stamp_outlived            810              0
/// ```
///
/// Every landing is now inside the fence that completes it, and
/// `gvaw_fence_flush` equals `gva_deferred` exactly — every window armed is a
/// window landed at the next stamp, which is the whole of the deferral the
/// contract permits.
///
/// The cost was expected to be a frame-rate loss and was the opposite. Same
/// harness, same 600 s drive, mean over ~510 one-second windows:
///
/// ```text
///                 before repair   after repair
/// present_hz                5.9            9.5
/// draw_us              523 895        156 294
/// ```
///
/// Two boots are not a benchmark and load varies, but the direction is not
/// subtle and it has a mechanism: 215 of 227 landings used to come out under
/// `window_cap`, so the old rail spent its time in oldest-first eviction storms
/// while holding residents pinned across hundreds of frames. Landing at the
/// fence keeps the window set nearly empty and the pin churn with it.
///
/// The crash itself is still unscored. `.agents/repros/crash-hunt.sh` has never
/// fired the abort in either arm, so it gates the census and not the class.
#[cfg(feature = "backend-vulkan")]
fn note_window_outlived_its_stamp(
    state: &DeviceState,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    trigger: &str,
) {
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(entry.armed_stamp_seq);
    if elapsed == 0 {
        crate::runtime::drain::note_store_route("gvaw_stamp_same");
        return;
    }
    crate::runtime::drain::note_store_route("gvaw_stamp_outlived");
    // Identity, latched per span+trigger: the count says how often, and this
    // says which windows and which door they came through. A rail that only
    // ever outlives its stamp under one trigger is a different repair from one
    // that does it everywhere.
    if crate::observe::first_sight(
        "gva_window_outlived_stamp",
        gva ^ ((entry.width as u64) << 32) ^ entry.height as u64,
    ) {
        crate::observe::fail(format!(
            "gva_window_outlived_stamp gva={gva:#x} task={} {}x{} trigger={trigger} \
             stamps={elapsed} (guest was fenced before these bytes were written)",
            entry.task_id, entry.width, entry.height
        ));
    }
}

/// Score a deferred **linear compute-storage** landing against the guest's fence.
///
/// [`note_window_outlived_its_stamp`] is the same reading for the GVA render
/// rail, and the hazard is identical because the identity is identical: a
/// `ComputeStorageResidencyKey::linear` names a task and an address
/// (`mapping_id` 0, `map_generation` carrying the task id), so nothing the guest
/// does to reclaim the memory reaches this rail as a notification.
///
/// That distinction is why `6bc2220` cleared the other two deferred rails and
/// cannot clear this one. `flush_render_one` and `flush_storage_one` refuse on
/// `map_generation` drift, and `map_generation` moves on exactly the events that
/// let a guest reuse an IOSurface's storage. This rail has no such generation to
/// compare — [`deferred_pages_still_ours`] is its only guard, and free-then-reuse
/// inside one process preserves the translation the guard reads.
///
/// The rail's own flush already records what that costs when it goes wrong:
/// a `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
/// page bookkeeping. What was missing is how often the landing is late at all,
/// which is what `linw_stamp_same` against `linw_stamp_outlived` says.
#[cfg(feature = "backend-vulkan")]
fn note_linear_window_outlived_its_stamp(
    state: &DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
    window: &crate::model::LinearDeferredEntry,
) {
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(window.armed_stamp_seq);
    if elapsed == 0 {
        crate::runtime::drain::note_store_route("linw_stamp_same");
        return;
    }
    crate::runtime::drain::note_store_route("linw_stamp_outlived");
    if crate::observe::first_sight(
        "linear_window_outlived_stamp",
        key.surface_offset ^ ((key.width as u64) << 32) ^ key.height as u64,
    ) {
        crate::observe::fail(format!(
            "linear_window_outlived_stamp task={} ref={} gva={:#x} {}x{} stamps={elapsed} \
             (guest was fenced before these bytes were written)",
            key.map_generation, key.texture_ref, key.surface_offset, key.width, key.height
        ));
    }
}

/// Engine-resident identity a deferred GVA window is holding pinned.
///
/// Rebuilt from the window's own fields — including the
/// [`crate::model::GvaDeferredEntry::alloc_gen`] the arming draw resolved —
/// rather than from a fresh page walk. The window exists because the guest may
/// hand the address to another allocation before the flush runs; a walk taken
/// now would name that allocation, the registry lookup would miss the slot this
/// window pinned, and the deferred frame would be lost instead of landing.
///
/// Single spelling for every consumer that starts from a window
/// ([`flush_gva_one`], `metal_draw::vulkan::supersede_gva_window`,
/// `metal_draw::vulkan::try_sample_deferred_gva`) so the three cannot drift
/// apart from the producer or from each other.
#[cfg(feature = "backend-vulkan")]
pub fn gva_window_identity(
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Gva {
        gva,
        width: entry.width,
        height: entry.height,
        generation: entry.alloc_gen,
    }
}

/// Land a taken deferred GVA render-Store window: engine resident target →
/// guest pages (when `guest_write` and the span is still map-covered) +
/// `host_gva_surfaces`/texture encode caches (always). Unpins the resident
/// either way; a lost resident is fail-visible and leaves the guest window
/// stale-but-coherent (pre-Store bytes).
#[cfg(feature = "backend-vulkan")]
pub fn flush_gva_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    guest_write: bool,
    trigger: &str,
) -> bool {
    let started = std::time::Instant::now();
    let identity = gva_window_identity(gva, entry);
    // `into_rgba8` rather than the raw bytes: a GVA resident is RGBA today, so
    // this is a no-op, but the writer below (`write_gva_rgba8`) is declared in
    // semantic RGBA and the readback states its own order. Asserting the order
    // here instead would be the caller writing a fact it did not read.
    let rgba = match crate::backend::vulkan::engine::read_target(&identity) {
        Ok(rb) => rb.into_rgba8(),
        Err(e) => {
            crate::backend::vulkan::engine::unpin_resident_target(&identity);
            crate::observe::fail(format!(
                "deferred_flush_lost kind=gva gva={gva:#x} {}x{} fmt={:#x} trigger={trigger} err={e}",
                entry.width, entry.height, entry.format
            ));
            return false;
        }
    };
    crate::backend::vulkan::engine::unpin_resident_target(&identity);
    let mut guest = "skip";
    if guest_write {
        note_window_outlived_its_stamp(state, gva, entry, trigger);
    }
    if guest_write && !window_pages_still_ours(state, host, gva, entry, trigger, "guest=refused") {
        // The window's pages moved under us. Cache-only: see
        // `window_pages_still_ours` for why writing here lands in another
        // owner's memory. This is the REPORT — it walks every page of the window
        // against the pages it was armed on and names the event with counts a
        // reader can score. The BOUND is `Some(&entry.pages)` below, which the
        // writer's own walk enforces; a decision taken before a second walk is
        // a decision about a page table the bytes do not go through.
        guest = "skip_drift";
    } else if guest_write {
        guest = match crate::runtime::metal_draw::write_gva_rgba8_within(
            state,
            host,
            entry.task_id,
            gva,
            entry.width,
            entry.height,
            entry.row_stride,
            entry.format,
            &rgba,
            Some(&entry.pages),
        ) {
            Ok(()) => "written",
            // The guest already tore this window down and its Unmap notify has
            // not drained yet. That is the same state the Unmap/Map notify path
            // lands cache-only for — "on Unmap the PTEs are already gone" — just
            // reached through a different door, because a page-alias flush races
            // ahead of the notify. The caches below hold the content, so the
            // obligation is discharged and nothing is lost. Expected control
            // flow: it does not belong in the failure log.
            Err(err) if err.is_guest_teardown() => "unmapped",
            // A write that refused while the target still existed. The caches
            // below keep the authoritative bytes, so guest RAM is stale rather
            // than wrong — but this one is a real loss of guest work.
            Err(err) => {
                crate::observe::Emit::decline("deferred_flush_lost", &err)
                    .field("kind", "gva")
                    .field("gva", format!("{gva:#x}"))
                    .field("dims", format!("{}x{}", entry.width, entry.height))
                    .field("bpr", entry.row_stride)
                    .field("fmt", format!("{:#x}", entry.format))
                    .field("trigger", trigger)
                    .fail();
                "write_fail"
            }
        };
    }
    // The host cache is stored on all five outcomes, deliberately: on the four
    // that did not reach guest RAM it is what holds the authoritative bytes. But
    // that makes the cache store a poor witness of whether the guest got them,
    // and the `guest=` word below rides `observe::line`, which is off by
    // default — so on a stock boot nothing says how the rail's writes divided.
    // Census it on the always-on counters instead. `written` is the healthy
    // majority; the other four are each explained at their arm above.
    crate::runtime::drain::note_store_route(match guest {
        "written" => "gva_flush_guest_written",
        "skip" => "gva_flush_guest_skip",
        "skip_drift" => "gva_flush_guest_skip_drift",
        "unmapped" => "gva_flush_guest_unmapped",
        _ => "gva_flush_guest_write_fail",
    });
    crate::runtime::metal_draw::host_cache_store_gva_layer(
        state,
        host,
        entry.task_id,
        entry.texture_ref,
        entry.producer_object_type,
        gva,
        entry.width,
        entry.height,
        &rgba,
    );
    // A flush that landed is expected control flow and stays quiet. The two
    // outcomes that are not — a refused write, and a window whose span the guest
    // had already torn down — each emit their own typed line above, so the
    // always-on view keeps the losses and drops the running commentary.
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Gva),
        started,
    );
    crate::observe::line(format!(
        "gva_deferred_flush gva={gva:#x} {}x{} fmt={:#x} guest={guest} trigger={trigger} bytes={} us={}",
        entry.width,
        entry.height,
        entry.format,
        rgba.len(),
        started.elapsed().as_micros()
    ));
    guest != "write_fail"
}

#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_gva_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    gva: u64,
    entry: &crate::model::GvaDeferredEntry,
    _guest_write: bool,
    trigger: &str,
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=gva reason=no_backend gva={gva:#x} {}x{} trigger={trigger}",
        entry.width, entry.height
    ));
    false
}

/// Land GVA windows whose task died (`DeviceState::retired_gva_windows`)
/// **cache-only**: the GVA walk is gone with the task, so guest pages are
/// never written from teardown (boot-16 rule); the encode cache keeps the
/// content for later samples (wallpaper-retain contract).
pub fn retire_gva_windows<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.retired_gva_windows.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_gva_windows);
    for (gva, entry) in &retired {
        let _ = flush_gva_one(state, host, *gva, entry, false, "task_retired");
    }
}

/// Land a deferred linear window: resident → cache entry bytes
/// (`materialize_linear_resident`) → guest pages when the span is still
/// GVA-covered (fresh page-table walks; a write through changed PTEs fails
/// per-row, fail-visibly, and never touches other memory). Drops the
/// obligation either way — the cache entry keeps the authoritative bytes.
#[cfg(feature = "backend-vulkan")]
pub fn flush_linear_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let window = state.disarm_linear_deferred_window(key);
    let armed_pages = window.as_ref().map(|w| w.pages.clone());
    let task_id = key.map_generation;
    let texture_ref = key.texture_ref;
    let started = std::time::Instant::now();
    if let Some(window) = window.as_ref() {
        note_linear_window_outlived_its_stamp(state, key, window);
    }
    let (bytes, texel) =
        match crate::backend::vulkan::engine::read_resident_storage(key, generation) {
            Ok(v) => v,
            Err(e) => {
                crate::observe::Emit::decline("deferred_flush_lost", &e)
                    .field("kind", "linear")
                    .field("task", task_id)
                    .field("ref", texture_ref)
                    .field("geom", format!("{}x{}", key.width, key.height))
                    .field("fmt", format!("{:#x}", key.pixel_format))
                    .field("gen", generation)
                    .fail();
                if let Some(entry) = state.host_linear_textures.get_mut(&(task_id, texture_ref)) {
                    if entry.resident_gen == generation {
                        entry.resident_gen = 0;
                    }
                }
                return false;
            }
        };
    // The `skip_drift` arm below refuses the guest write and calls that
    // lossless, on the grounds that the cache entry holds the frame. That is
    // true only if this call landed it, so a failure here has to reach the log:
    // otherwise the two together drop a frame with no record, which is the
    // whole loss this rail exists to avoid. `Superseded` is the exception —
    // a newer defer already owns the entry, so there is no frame to keep.
    let cached = crate::runtime::surface_cache::materialize_linear_resident(
        state,
        task_id,
        texture_ref,
        generation,
        &bytes,
    );
    if let Err(decline) = &cached {
        if !matches!(
            decline,
            crate::runtime::surface_cache::LinearMaterializeDecline::Superseded { .. }
        ) {
            crate::observe::Emit::decline("linear_materialize_lost", decline)
                .field("task", task_id)
                .field("ref", texture_ref)
                .field("geom", format!("{}x{}", key.width, key.height))
                .field("fmt", format!("{:#x}", key.pixel_format))
                .field("gen", generation)
                .fail();
        }
    }
    let tight = (key.width as usize).saturating_mul(texel as usize);

    // Same hazard, same answer as the GVA rail: this window was armed against a
    // page set at defer time and `write_linear_guest` walks fresh, so a span the
    // guest has since re-pointed sends a compute-storage image into whatever
    // owns those pages now. Observed on this rail as guest heap corruption — a
    // `pmap_page_protect` kernel panic and userspace SIGSEGVs inside libmalloc's
    // own page bookkeeping. Refusing is lossless *when* the cache entry kept
    // the authoritative bytes, which is exactly `cached.is_ok()` — the refusal
    // and the store are one claim, so the emit above is what makes the pair
    // honest rather than the comment that used to state it unconditionally.
    let still_ours = match &armed_pages {
        // `span_end` is a length (`row_stride * height`) for a linear key, not
        // an end address — and the arm site walks `(surface_offset, span_end)`
        // with exactly these two values, so this walk has to as well or the two
        // page sets describe different ranges and every flush reads as drift.
        Some(pages) => deferred_pages_still_ours(
            state,
            host,
            task_id,
            key.surface_offset,
            key.span_end,
            pages,
            &format!(
                "{}x{} trigger=linear_flush ref={texture_ref}",
                key.width, key.height
            ),
            "guest=refused",
        ),
        None => true,
    };
    // Both arms assign, so this is the whole set of outcomes this rail can
    // report — `skip_uncovered` was the third and is gone.
    let guest = if !still_ours {
        "skip_drift"
    } else {
        // Same bound as the GVA rail: the armed page set travels into the
        // writer's own walk, so the decision `still_ours` reached above cannot be
        // invalidated by the guest between that walk and this one. `None` here
        // would be a window with no armed pages, which is a window this rail
        // never bounded in the first place.
        match crate::runtime::compute_exec::write_linear_guest_within(
            state,
            host,
            task_id,
            key.surface_offset,
            key.surface_bpr as u64,
            tight,
            key.height,
            &bytes,
            &format!("flush ref={texture_ref}"),
            armed_pages.as_ref(),
        ) {
            crate::runtime::compute_exec::LinearWrite::Written => "written",
            // Nothing resolves at this GVA, so there is no guest memory to land
            // in. Distinct from `write_fail`, which means a write was attempted:
            // one is the guest having taken the pages away, the other is ours.
            crate::runtime::compute_exec::LinearWrite::Unmapped => "skip_unmapped",
            // The per-row failure is already fail-logged; the cache entry keeps
            // the coherent authoritative bytes.
            crate::runtime::compute_exec::LinearWrite::Failed => "write_fail",
        }
    };
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Linear),
        started,
    );
    crate::observe::off(format!(
        "linear_deferred_flush task={task_id} ref={texture_ref} {}x{} fmt={:#x} gen={generation} guest={guest} bytes={} us={}",
        key.width,
        key.height,
        key.pixel_format,
        bytes.len(),
        started.elapsed().as_micros()
    ));
    true
}

#[cfg(not(feature = "backend-vulkan"))]
pub fn flush_linear_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    state.disarm_linear_deferred_window(key);
    crate::observe::fail(format!(
        "deferred_flush_lost kind=linear reason=no_backend task={} ref={} gen={generation}",
        key.map_generation, key.texture_ref
    ));
    false
}

/// Unpin engine residents whose linear cache entry died (task/object delete —
/// `DeviceState::retired_linear_residents`). The images become LRU-evictable;
/// without this a dead entry leaks its pinned VRAM image for the boot.
pub fn retire_linear_residents(state: &mut DeviceState) {
    if state.retired_linear_residents.is_empty() {
        return;
    }
    let retired = std::mem::take(&mut state.retired_linear_residents);
    for key in &retired {
        // Task teardown = the GPU VA maps are gone; never write guest pages
        // from here (boot-16 rule) — drop any pending guest-flush obligation.
        if state.disarm_linear_deferred_window(key).is_some() {
            crate::observe::off(format!(
                "linear_deferred_dropped reason=retired task={} ref={}",
                key.map_generation, key.texture_ref
            ));
        }
        #[cfg(feature = "backend-vulkan")]
        {
            crate::backend::vulkan::engine::unpin_resident_storage(key);
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
    }
}

/// Land one taken mapping-keyed window, dispatching on which rail holds its
/// pixels. The key names the guest side identically for both; only the read
/// differs (see [`crate::model::DeferredOwner`]).
fn flush_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    owner: crate::model::DeferredOwner,
) -> bool {
    note_mapping_window_against_fence(state, key, &owner);
    match owner {
        crate::model::DeferredOwner::Storage {
            generation,
            armed_stamp_seq: _,
        } => flush_storage_one(state, host, key, generation),
        crate::model::DeferredOwner::Render { source, .. } => {
            flush_render_one(state, host, key, &source)
        }
    }
}

/// Score a mapping-keyed deferred window against the guest's fence, exactly as
/// [`note_window_outlived_its_stamp`] scores the GVA rail.
///
/// Counted at the flush dispatcher rather than at each writer, so the two rails
/// share one denominator; whether a landing actually reached guest RAM is what
/// the existing `deferred_flush_*` lines already say.
///
/// # What this counter did NOT settle, and what did
///
/// The reading that made this rail worth measuring separately still stands. One
/// 14-round x86/Vulkan icon boot:
///
/// ```text
/// rendw_stamp_same    0     rendw_stamp_outlived 1088
/// storw_stamp_same    0     storw_stamp_outlived   24
/// elapsed over 217 latched spans: min 1, p50 66, p90 2551, max 17086
/// ```
///
/// Read as a counter that looks exactly like the GVA rail's 810-of-810, and it
/// does not mean the same thing. **The counter is not the hazard.** Outliving
/// the fence corrupts memory only if the guest can repurpose that memory without
/// the device finding out, and on these rails it cannot:
///
/// - [`flush_render_one`] and [`flush_storage_one`] both compare the mapping's
///   live `map_generation` against `key.map_generation` and refuse with
///   `deferred_flush_lost reason=map_generation_drift` before reading anything.
/// - `map_generation` is bumped by exactly the events that let the guest reuse
///   an IOSurface's storage — MAP, UNMAP, `ReplacePhysical`, MappingInternal
///   reattach, any page-table refresh that changes PFNs.
/// - A `DeleteIOSurfaceBacking2` that has not yet resolved leaves the backing
///   *condemned*, and [`flush_intersecting`] refuses to take those windows at
///   all.
///
/// So these windows name a specific mapping incarnation, and a guest that frees
/// the storage invalidates the name. That is precisely the allocation identity
/// the GVA rail did not have and could not be given: a type-2/3 target is a
/// texture handle shifted into an address, with no lifecycle notify anywhere in
/// the wire format, so `deferred_pages_still_ours` was the only guard available
/// and page identity survives free-then-reuse.
///
/// This rail is nonetheless bound to the fence now
/// ([`flush_mapping_windows_before_fence`]) — for the *other* hazard, which this
/// counter cannot see and `render_flush_over_guest_write` can: the guest holds
/// the same IOSurface mapped and writes it, and a full-extent writeback landing
/// later replaces what it wrote. 8 968 of 12 343 landings on one measured boot.
/// The free-then-reuse argument above is untouched by that and is still the
/// reason this rail needed its own evidence instead of the GVA rail's.
///
/// These counters stay as the standing check on the `map_generation` guard, and
/// as the reading of how much deferral the binding actually removed: with the
/// fence drain wired, `rendw_stamp_same` should carry the traffic and
/// `rendw_stamp_outlived` should fall to the windows a condemned backing holds.
fn note_mapping_window_against_fence(
    state: &DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
    owner: &crate::model::DeferredOwner,
) {
    let rail = match owner {
        crate::model::DeferredOwner::Storage { .. } => "storage",
        crate::model::DeferredOwner::Render { .. } => "render",
    };
    let elapsed = state
        .completion_stamp_seq
        .wrapping_sub(owner.armed_stamp_seq());
    if elapsed == 0 {
        crate::runtime::drain::note_store_route(match rail {
            "storage" => "storw_stamp_same",
            _ => "rendw_stamp_same",
        });
        return;
    }
    crate::runtime::drain::note_store_route(match rail {
        "storage" => "storw_stamp_outlived",
        _ => "rendw_stamp_outlived",
    });
    if crate::observe::first_sight(
        "mapping_window_outlived_stamp",
        u64::from(key.mapping_id) ^ ((key.width as u64) << 32) ^ key.height as u64,
    ) {
        crate::observe::fail(format!(
            "mapping_window_outlived_stamp rail={rail} mapping={} {}x{} stamps={elapsed} \
             (guest was fenced before these bytes were written)",
            key.mapping_id, key.width, key.height
        ));
    }
}

/// Whether the mapping's cached page list still names the guest memory it was
/// walked from, counted so a boot carries the rate and gated so an arm and its
/// control stay one binary apart.
///
/// The check is [`crate::runtime::mapper::type4_pages_witness`]; this is the
/// deferred rails' use of it, and it is the missing half of a guarantee the
/// raw-GVA rails already have. `gva_view::write_span` re-walks the task page
/// table at write time and fails closed, stating outright that a write through a
/// cached view "lands in whatever now owns those host pages (guest heap
/// corruption: the 2026-07-19 WindowServer SIGSEGV class)". The mapping-keyed
/// rails write through `MappingEntry::page_entries`, which for a type-4 surface
/// nothing re-walks between the resolve that filled it and the flush that uses
/// it.
///
/// That is the shape of every crash this device is chasing. The user's report is
/// WindowServer aborting inside `small_free_list_remove_ptr_no_clear` under an
/// allocation made by `AppleParavirtGPUMetal`, and the guest kernel's own poison
/// check found freed elements "filled with 0xFF from offset 0" — opaque white
/// pixels in memory the guest had already reclaimed. The twelve guest panics on
/// disk hit apfs, airportd, tccd, a HID driver and WindowServer, which is not a
/// bug in one path but a device writing where it no longer has title.
///
/// # Drift refuses this write and stops the list being believed again
///
/// Refusing the one window is not enough. The list is what every later reader
/// and writer of this mapping resolves through, so leaving it in place means the
/// next flush asks the same question and the next present serves pixels read
/// through the same wrong pages. `invalidate_mapping_pages` clears it and bumps
/// `map_generation`, which retires the contiguous view and the guest-write token
/// with it, and every window still armed against the old incarnation then
/// refuses on the `map_generation` check it already has.
///
/// Self-healing rather than terminal: the next type-4 bind re-resolves the
/// surface from the object list and adopts a fresh plan, which is the path that
/// would have discovered this eventually anyway. An actively-drawn surface
/// recovers on its next bind; an idle one stays unresolvable, which is the
/// correct state for a mapping this device can no longer name.
///
/// Deliberately NOT a forced `resolve_type4_surface` here. That would be the
/// more informative answer — it re-runs the object search and could say whether
/// the surface merely moved or is gone — but it goes through `map_surface`,
/// which clears `has_geom`, the geometry and `surface_content_epoch` before the
/// adoption restores them. Running that from inside a flush puts a mapping
/// through a destructive half-state while a writeback is in progress, to answer
/// a question the next bind answers for free.
#[cfg(feature = "backend-vulkan")]
fn mapping_pages_still_ours<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) -> bool {
    use crate::runtime::mapper::PagesVerdict;
    match crate::runtime::mapper::mapping_pages_verdict(state, host, mapping_id) {
        PagesVerdict::Ours => {
            crate::runtime::drain::note_store_route("mapping_pages_ours");
            true
        }
        // Lands exactly as `Ours` does; counted apart because it is not the
        // same claim. `mapping_pages_ours` used to include every flush this
        // witness had nothing to say about, so the ratio it appeared to give
        // against `mapping_pages_drifted` was not the guard's hit rate.
        PagesVerdict::Unwitnessed(_) => {
            crate::runtime::drain::note_store_route("mapping_pages_unwitnessed");
            true
        }
        PagesVerdict::Drifted => {
            crate::runtime::drain::note_store_route("mapping_pages_drifted");
            false
        }
    }
}

/// Land a deferred **type-11 render Store**: perform the CPU writeback into the
/// mapping's guest pages that the Store itself skipped.
///
/// The pixels come from `surface_cache`, not from the engine. The Store read
/// its target back as it always did and refreshed the cache with that frame
/// before arming; only the guest-page copy was deferred. That is deliberate and
/// it is what keeps this rail small: the engine resident for a type-11 surface
/// is not authoritative here, so nothing has to be pinned, no `content_ready`
/// has to hold across frames, and the Load seed and present capture keep
/// reading exactly what they read before.
///
/// Deferring is a win rather than a rescheduling because nothing on the
/// host-window present path reads these guest pages — `capture_present_frame`
/// takes the cache or the resident and states in situ that it "never touches
/// guest memory" — so the writeback is owed only to a guest-side reader that
/// may never come.
/// The engine resident a [`crate::model::RenderWindowSource::Resident`] window
/// pinned, rebuilt from the key.
///
/// Not stored on the window, for the same reason `flush_gva_one` rebuilds its
/// own: the key already carries every term of the identity, and two spellings of
/// one value are two things that can disagree. `key.map_generation` is the field
/// `present_identity::surface_identity` keys on, and the flush refuses on
/// generation drift before it reads anything, so the rebuild is always for the
/// generation the arm pinned.
#[cfg(feature = "backend-vulkan")]
pub fn render_window_identity(
    key: &crate::model::ComputeStorageResidencyKey,
) -> crate::backend::vulkan::engine::TargetIdentity {
    crate::backend::vulkan::engine::TargetIdentity::Surface {
        id: key.mapping_id,
        width: key.width,
        height: key.height,
        generation: key.map_generation as u64,
    }
}

/// Report that a landing window is about to overwrite the guest's own stores.
///
/// This returns nothing, and that is the finding rather than an omission. It
/// used to hand back the ranges to preserve; it now preserves none of them, and
/// carrying an always-empty `Vec` out to the writeback made three signatures
/// advertise a narrowing no caller can ever ask for — a reader auditing whether
/// this rail honours guest writes would find a `skip` parameter and conclude it
/// does, when it deliberately does not and says so on every occurrence.
///
/// A deferred window promises to replay a synchronous Store later, and that is
/// only a replay while nothing else writes the pages in between. The writeback
/// covers the whole attachment extent, so a guest CPU store into any page of it
/// — an inter-buffer damage forward-copy, a CoreGraphics blit into the same
/// IOSurface — is gone the moment this window lands. Nothing else in the flush
/// can see that: `map_generation` covers a rebind, `resident_content_epoch`
/// covers a later device draw, and neither is a witness for the surface's own
/// owner. One 14-round composite boot measured `render_flush_over_guest_write`
/// at 68 of every 99 `surface_flush`es.
///
/// This rail did preserve those pages, and it must not, because the witness it
/// would preserve them on cannot answer the question it is being asked.
///
/// `page_gen[p]` is stamped with the generation at the *harvest* that saw page
/// `p` dirty, not at the write. `reims_vgpu_dirty_harvest` returns early when
/// nothing has read a generation since the last one, and does not clear the
/// bitmap when it does, so a guest store can sit unharvested across a Store and
/// be attributed to the generation of a harvest that ran after it. Every such
/// page is then "written since the Store" when the device's own render
/// superseded it, and preserving it withholds the frame from guest memory.
///
/// Bisected on the live rail, x86 / Vulkan, four `icon-composite` rounds each,
/// one binary per arm:
///
/// ```text
/// 22a3346  preserve absent   3 of 4 rounds clean, desktop paints
/// 8178caa  preserve absent   2 of 4 rounds clean, desktop paints
/// 13ae46d  preserve present  0 of 14 rounds, screen black, 19 Hz
/// ```
///
/// So the answer this rail reaches for is the right one and the evidence it
/// would reach for it on is not sound. A full-extent landing that reports what
/// it replaced is strictly better than a partial one that silently withholds the
/// device's frame.
///
/// The ordering repair is what actually removes the loss, and it is upstream of
/// this question rather than an answer to it:
/// [`flush_mapping_windows_before_fence`] lands every armed window before the
/// guest is told the work is done, so the interval in which a guest store can be
/// both after the Store and before the writeback does not exist. Nothing needs
/// preserving because nothing is clobbered, and this function becomes the
/// standing check on that.
///
/// It is a **loose** check, and reading it as a tight one sends a reader after a
/// hole that need not exist. The verdict is `guest_write_gen(token) !=
/// guest_write_gen_at_store`, and that generation moves at the *harvest* that
/// saw the page dirty, not at the write —
/// [`render_flush_guest_written_ranges`] states the same rule for the same
/// reason. `reims_vgpu_dirty_gen` returns the value as of the last harvest and
/// only marks a read as owed; `reims_vgpu_dirty_harvest` then returns early
/// unless a read is owed, and runs at the drain tail. So a guest store made
/// *before* the Store, in a tranche whose harvest had not yet run, is stamped
/// into a generation that moves *after* it, and this fires. That is structural,
/// not a race: it is the same unsoundness that made preserving the pages black
/// the screen, and it points the one way that costs nothing to be wrong about.
///
/// So a surviving occurrence is an **upper bound on** clobbered windows, not a
/// count of them, and a single line is not by itself a defect. What would be a
/// defect is the rate not falling when the fence binding tightens, or a
/// `rendw_stamp_outlived` naming a window that landed after
/// [`crate::runtime::drain::write_stamp`] — that one is an ordering statement
/// the device can actually make.
///
/// [`crate::runtime::mapping_write::write_bgra8_skipping`] and
/// `HostOps::guest_written_pages` stay: the sampled ladder's merge uses both,
/// and it errs the other way — it keeps both halves rather than choosing.
#[cfg(feature = "backend-vulkan")]
fn note_render_flush_over_guest_write<M: HostOps>(
    state: &DeviceState,
    host: &M,
    key: &crate::model::ComputeStorageResidencyKey,
) {
    use crate::runtime::mapper::{mapping_guest_write_verdict, GuestWriteVerdict};
    if mapping_guest_write_verdict(state, host, key.mapping_id) != GuestWriteVerdict::Wrote {
        return;
    }
    crate::runtime::drain::note_store_route("render_flush_over_guest_write");
    crate::observe::fail(format!(
        "deferred_flush_clobber kind=render mapping={} {}x{} fmt={:#x} gen={} \
         (a guest write to this surface was observed since the Store this window \
         defers, and the full-extent writeback replaces it; the witness moves at \
         harvest, not at the write, so this cannot order the two and is an upper \
         bound)",
        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
    ));
}

#[cfg(feature = "backend-vulkan")]
fn flush_render_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    source: &crate::model::RenderWindowSource,
) -> bool {
    let started = std::time::Instant::now();
    // Counted on the same one-line-per-second census as the Store routes, so a
    // boot reads `surface_deferred=N surface_flush=M` on one line.
    //
    // That ratio used to be the only thing separating a deferral from a
    // rescheduling: a reader draining every window every frame arms and flushes
    // at identical rates and is indistinguishable from a working rail by arm
    // count alone, so M << N was the win and M ≈ N meant some guest-page reader
    // was asking for these bytes anyway.
    //
    // `flush_mapping_windows_before_fence` changes what the ratio means, and a
    // census read on the old rule would draw the wrong conclusion from it. Every
    // armed window now lands at the next completion stamp by design, so M ≈ N is
    // the *intended* state and says nothing about guest-page readers.
    //
    // This comment used to send the reader to `surface_resident` against
    // `surface_flush` in one second, on the reasoning that what deferral still
    // buys is coalescing inside one fence. That instrument has now been read,
    // and it cannot answer: the two are equal on **all 1 780 census lines** of
    // the accumulated x86 / Vulkan log, 193 458 each, with not one line
    // differing. Every arm gets exactly one flush, because
    // `surface_deferred_superseded` has never once fired — no later Store in the
    // whole log fully covered a live window — and every arm succeeds, because
    // `surface_resident_sync` is likewise absent. A ratio pinned at 1.0 by the
    // workload is not a measurement of coalescing; it is the statement that no
    // coalescing was available to do.
    //
    // So do not quote that ratio as this rail's payoff. What is measured is the
    // readback: the resident Store arms without reading the frame back off the
    // GPU, and `surface_resident_sync` counts the arms that had to. Coalescing
    // is a guard for a workload shape — several passes fully covering one
    // surface inside one submission — that nine driven boots, including a
    // SceneKit title and a live WebGL scene, never produced. It is kept because
    // landing a covered window costs a full-framebuffer write for nothing, not
    // because anything here has seen it pay.
    crate::runtime::drain::note_store_route("surface_flush");
    // Whether this writeback is owed at all, before any question about where it
    // would land. The guard below asks "are these still our pages"; this asks
    // the prior question the guest itself answers on every submission — has it
    // since declared its own bytes newer than the frame this window holds.
    if crate::runtime::resource_validity::writeback_refused(state, key.mapping_id) {
        release_window_pin_for_key(key, source);
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
             reason=host_copy_superseded (the guest declared its own pages \
             authoritative after the Store this window defers)",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    // Recycled-pages guard, identical in intent to the compute rail's below and
    // to the GVA rail's `deferred_pages_still_ours`: a mapping rebound since
    // arm time (ReplacePhysical, unmap/remap) points at pages this window's
    // pixels do not belong in, and writing them there lands a framebuffer in
    // whatever owns that memory now. Drop rather than write.
    let current = state
        .mappings
        .get(&key.mapping_id)
        .map(|m| m.map_generation);
    if current != Some(key.map_generation) {
        // Release the pin first. This arm returns before touching the frame, and
        // a `Resident` window holds a registry pin that nothing else will drop —
        // `evict_registry_to_cap` and the idle drain both skip pinned slots by
        // design, so a pin leaked here strands a whole framebuffer for the guest
        // lifetime. That is the "~260 stale residents (~516 MiB)" shape, and this
        // drift is not rare: one in 85 s on a driven boot.
        release_window_pin_for_key(key, source);
        // Counted, not just logged. The three resident-mismatch refusals below
        // have carried census routes since they were split apart; these two
        // drift refusals did not, and they are the ones that lose a painted
        // tile. `flush_intersecting` has already TAKEN this window out of
        // `compute_deferred_flush`, and `flush_mapping_windows_before_fence`
        // returns `()`, so the fence advances and nothing re-arms the
        // obligation: the pixels land nowhere, permanently.
        //
        // That is the Goal 3 event, and until now a census could not count it.
        // `mapping_pages_drifted` is not a substitute — it is incremented inside
        // `mapping_pages_still_ours`, which several callers reach, so it counts
        // refusals rather than lost tiles.
        crate::runtime::drain::note_store_route("rendflush_gen_drift");
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} reason=map_generation_drift current={current:?}",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    // `map_generation` is the guest's *declared* incarnation, and a type-4
    // surface can be re-pointed with nothing declared at all. See
    // `mapping_pages_still_ours`.
    if !mapping_pages_still_ours(state, host, key.mapping_id) {
        release_window_pin_for_key(key, source);
        crate::runtime::drain::note_store_route("rendflush_page_drift");
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} reason=mapping_page_drift",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    // Where the frame comes from, in guest scanout order either way.
    //
    // `Owned` carries its own bytes and cannot miss. It used to read
    // `surface_cache::get(mapping_id, key.width, key.height)`, and that is one
    // entry per mapping: a later Store at a different geometry replaced it and
    // every window still armed at the old geometry lost its pixels —
    // `deferred_flush_lost reason=cache_miss`, 15 whole layers in one boot, which
    // is a compositing layer going solid black. The bytes are shared with the
    // cache entry the same readback stored, so owning them costs an `Arc` clone
    // and no copy.
    //
    // `Resident` names the pinned engine image instead, and pays the readback here
    // rather than at every Store. It is checked against the epoch it was published
    // at before being believed: `registry_mark_ready` clears a slot's
    // `content_epoch` on every draw into it, so a mismatch means something rendered
    // over this surface after the Store that armed this window, and the resident no
    // longer holds the frame this window promised the guest. Declining leaves the
    // guest its pre-Store bytes — stale but coherent — where writing would land a
    // different layer's pixels in this one's pages.
    // Set when the frame below came *out of* a resident image, so the write can
    // hand the currency witness back to it. See the re-stamp after the write.
    let mut flushed_from_resident: Option<crate::backend::vulkan::engine::TargetIdentity> = None;
    // Land anything already armed over these pages *before* the frame is
    // acquired, not inside the write that follows.
    //
    // `write_bgra8_*` makes this call itself, and for the copying arms that is
    // where it belongs. The leased arm cannot afford it there: its frame is
    // borrowed from the engine's readback buffer, and a flush reached from
    // inside the write would read another resident — re-entering the engine
    // under a live lease, which is the one thing a holder may not do. Running it
    // here leaves the writer's own call nothing to find, because the only thing
    // that arms a window is a guest Store and no guest command is decoded inside
    // a writeback.
    //
    // Unconditional rather than gated on the arm taken below, so both arms reach
    // the write through the same state. A no-op on all but the rare mapping
    // carrying a second window.
    crate::runtime::mapping_write::flush_windows_under_bgra8_write(
        state,
        host,
        key.mapping_id,
        key.width,
        key.height,
    );
    // Owned rather than borrowed, and shared rather than owned outright: the
    // writeback's tail publishes this frame to the surface cache, and a cache
    // entry stores its frame behind an `Arc` precisely so that it and a window
    // can name one allocation. Handing the frame down as an `Arc` therefore ends
    // in one `Arc` clone where a borrow ended in a whole-frame copy — 1.21 ms of
    // memcpy per flush on the composite, about 100 times a second. `Owned`
    // already has one to clone.
    //
    // `Resident` does not go that way any more. It reads the resident back and
    // then *borrows* the staging buffer the readback landed in, because it has
    // no use for the frame after the scatter: owning it means one whole-frame
    // memcpy (`readback_split map_us`, ~0.82 ms of a 6.9 ms flush) whose only
    // consumer beyond the scatter is the host surface cache, and
    // `render_flush_cache_used` prices that consumer at 0.4 %. See
    // [`crate::backend::vulkan::engine::LeasedFrame`] for what the borrow costs
    // and [`crate::runtime::mapping_write::write_bgra8_uncached`] for what
    // happens to the cache entry instead.
    let frame: FlushFrame = match source {
        crate::model::RenderWindowSource::Owned(bytes) => FlushFrame::Owned(bytes.clone()),
        crate::model::RenderWindowSource::Resident { epoch } => {
            use crate::backend::vulkan::engine::ResidentContent;
            // The close of the interval `note_resident_window_armed` opened at
            // the Store. Taken before the epoch check, not after: a window that
            // the check refuses still consumed the arm, and leaving it counted
            // would make every later flush look like it had two outstanding.
            crate::runtime::drain::note_resident_window_flushed();
            let identity = render_window_identity(key);
            // Three outcomes, not two, and the third used to hide inside the
            // second. `resident_content_epoch` answers `None` both for a slot a
            // later draw un-stamped — expected traffic, the newer pass owns the
            // surface now — and for a slot that is not there at all, which
            // cannot happen to a pinned identity unless the arm and the flush
            // spell that identity differently. One measured boot lost ~150
            // frames here, `live=None` on every one of them, and nothing in the
            // log could say which kind they were. See `engine::ResidentContent`.
            let live = crate::backend::vulkan::engine::resident_content_state(&identity);
            if live != ResidentContent::Epoch(*epoch) {
                crate::backend::vulkan::engine::unpin_resident_target(&identity);
                let (reason, route) = match live {
                    ResidentContent::Absent => (
                        "resident_absent (a pinned slot cannot be evicted, so the arm \
                         and the flush name this target differently)",
                        "rendflush_resident_absent",
                    ),
                    ResidentContent::Unstamped => (
                        "resident_epoch_cleared (a draw landed on this surface after \
                         the Store this window defers)",
                        "rendflush_epoch_cleared",
                    ),
                    ResidentContent::Epoch(_) => ("resident_epoch_drift", "rendflush_epoch_drift"),
                };
                crate::runtime::drain::note_store_route(route);
                crate::observe::fail(format!(
                    "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
                     reason={reason} want={epoch} live={live:?}",
                    key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
                ));
                return false;
            }
            // Borrow first, and only where the borrow needs no transformation.
            //
            // The writer below is declared in guest scanout order, so a resident
            // reporting semantic RGBA8 owes an R/B exchange before its bytes can
            // land — which is a whole-frame pass, and a pass over the staging
            // buffer at that. `into_bgra8` on an owned copy is the existing home
            // for it, so a non-BGRA resident takes the copying arm rather than
            // teaching the lease to rewrite memory it does not own. A `Surface`
            // resident is BGRA and that is the composite rail this rail's cost
            // lives on; reading the reported order rather than asserting one is
            // what keeps a future format change from landing R and B exchanged
            // in guest memory.
            match crate::backend::vulkan::engine::read_target_leased(&identity) {
                Ok(Some(leased)) if leased.bgra => {
                    crate::backend::vulkan::engine::unpin_resident_target(&identity);
                    flushed_from_resident = Some(identity);
                    crate::runtime::drain::note_store_route("render_flush_leased");
                    FlushFrame::Leased(leased)
                }
                // Either the pool declined the lease (uncached readback memory,
                // where reading the mapping in place is the *slower* shape) or
                // the resident is not in scanout order. Both take the copy, and
                // the leased frame — if there is one — is dropped first so its
                // slot is back in the pool before the second readback asks for
                // one.
                Ok(leased) => {
                    drop(leased);
                    crate::runtime::drain::note_store_route("render_flush_copied");
                    match crate::backend::vulkan::engine::read_target(&identity) {
                        Ok(rb) => {
                            crate::backend::vulkan::engine::unpin_resident_target(&identity);
                            flushed_from_resident = Some(identity);
                            FlushFrame::Owned(std::sync::Arc::new(rb.into_bgra8()))
                        }
                        Err(e) => {
                            crate::backend::vulkan::engine::unpin_resident_target(&identity);
                            crate::observe::fail(format!(
                                "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} \
                                 gen={} reason=resident_read err={e}",
                                key.mapping_id,
                                key.width,
                                key.height,
                                key.pixel_format,
                                key.map_generation
                            ));
                            return false;
                        }
                    }
                }
                Err(e) => {
                    crate::backend::vulkan::engine::unpin_resident_target(&identity);
                    crate::observe::fail(format!(
                        "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} \
                         reason=resident_read err={e}",
                        key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
                    ));
                    return false;
                }
            }
        }
    };
    if crate::observe::dump_flush_surfaces() {
        // Opt-in census: the one thing the sink cannot say is what a surface
        // actually contains. A blank icon and a missing material look identical
        // from the decline side once nothing declines, and they separate here:
        // an all-zero frame means the render produced nothing, a non-zero one
        // means the loss is downstream of this read.
        let nonzero = match &frame {
            FlushFrame::Owned(bytes) => bytes.iter().filter(|b| **b != 0).count(),
            FlushFrame::Leased(leased) => leased.bytes().iter().filter(|b| **b != 0).count(),
        };
        crate::observe::fail(format!(
            "flush_surface_census mapping={} {}x{} fmt={:#x} bytes={} nonzero={nonzero}",
            key.mapping_id,
            key.width,
            key.height,
            key.pixel_format,
            frame.len()
        ));
        // The count says how much was drawn; only the pixels say what. Small
        // surfaces are the ones in question (icons, glass material) and cost
        // nothing to keep, so land them raw beside the sinks for off-VM
        // inspection.
        if key.width <= 512 && key.height <= 512 {
            let dir = crate::observe::log_dir();
            let bytes: &[u8] = match &frame {
                FlushFrame::Owned(bytes) => bytes.as_ref().as_slice(),
                FlushFrame::Leased(leased) => leased.bytes(),
            };
            let _ = std::fs::write(
                dir.join(format!(
                    "flush-mid{}-{}x{}-fmt{:x}.bgra",
                    key.mapping_id, key.width, key.height, key.pixel_format
                )),
                bytes,
            );
        }
    }
    note_render_flush_over_guest_write(state, host, key);
    let write_started = std::time::Instant::now();
    let ok = match &frame {
        FlushFrame::Owned(bytes) => crate::runtime::mapping_write::write_bgra8_owned(
            state,
            host,
            key.mapping_id,
            bytes,
            key.width.saturating_mul(4),
            key.width,
            key.height,
        ),
        FlushFrame::Leased(leased) => crate::runtime::mapping_write::write_bgra8_uncached(
            state,
            host,
            key.mapping_id,
            leased.bytes(),
            key.width.saturating_mul(4),
            key.width,
            key.height,
        ),
    };
    crate::runtime::drain::note_readback_phase(
        crate::runtime::drain::ReadbackPhase::Write,
        write_started.elapsed().as_micros() as u64,
    );
    // Whether this flush left a host surface cache copy behind, which decides
    // whether the witness has a cache leg to score at all. A borrowed frame
    // leaves none: it drops the entry because the memory holding it goes back to
    // the pool. The skipping write is the other writeback that leaves none, and
    // it is not reachable from here — this rail preserves nothing, so no store
    // it makes is a skipping one.
    let cache_stored = matches!(&frame, FlushFrame::Owned(_));
    // End the lease before anything below reaches the engine again — the
    // resident re-stamp does. A holder that blocks on the engine lock while a
    // teardown is waiting for exactly this lease is the deadlock `LeasedFrame`
    // forbids, and the frame has no reader left after the write in any case.
    let frame_len = frame.len();
    drop(frame);
    if !ok {
        crate::observe::fail(format!(
            "deferred_flush_lost kind=render mapping={} {}x{} fmt={:#x} gen={} reason=write_refused",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
    }
    // Hand the currency witness back to the image the frame came out of.
    //
    // `write_bgra8` ends in `mark_mapping_written`, which advances
    // `surface_content_epoch` — correctly, since the mapping's guest pages did
    // change. But the *pixels* did not: they are the resident's, copied out of it
    // one statement ago. Leaving the stamp behind therefore invalidates a resident
    // that holds exactly the mapping's content, and on the composite rail that is
    // not a residual — it is a loop. The stale stamp costs the next LOAD its
    // elision, the CPU seed it falls back to finds the host cache ceded to this
    // rail, so it reads the mapping's guest pages, and reading them flushes the
    // window this Store just armed, which advances the epoch again. One boot
    // measured it at `surface_flush / surface_resident` = 1369/1373 — one flush per
    // arm, a rail that had become a rescheduling with a GPU round trip added.
    //
    // Only on the resident path: an `Owned` window's bytes came from an `Arc`, and
    // nothing here establishes that the slot under this identity still holds them.
    // The stamp is refused for a slot that is absent or not content_ready, and a
    // failed write leaves `flushed_from_resident` unused, so both fall back to a
    // seed rather than to a wrong frame.
    if ok {
        if let Some(identity) = flushed_from_resident {
            if let Some(epoch) = state
                .mappings
                .get(&key.mapping_id)
                .map(|m| m.surface_content_epoch)
            {
                crate::backend::vulkan::engine::stamp_resident_content_epoch(&identity, epoch);
            }
        }
        // Every copy this flush just made is unread until something reads it;
        // whatever was left of the previous flush's is scored now.
        let _ = note_render_flush_landed(state, key.mapping_id, cache_stored);
    }
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Render),
        started,
    );
    crate::observe::line(format!(
        "render_deferred_flush mapping={} {}x{} fmt={:#x} ok={} bytes={} us={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        ok as u8,
        frame_len,
        started.elapsed().as_micros()
    ));
    ok
}

/// Where a landing render window's frame lives while it is being written.
///
/// The two differ in what the writeback may leave behind. `Owned` names an
/// allocation that outlives the flush, so the host surface cache can hold it for
/// a refcount; `Leased` names the engine's readback staging buffer, which goes
/// back to the pool a moment later and therefore cannot be what a cache entry
/// points at. See [`crate::runtime::mapping_write::write_bgra8_uncached`].
#[cfg(feature = "backend-vulkan")]
enum FlushFrame {
    Owned(std::sync::Arc<Vec<u8>>),
    Leased(crate::backend::vulkan::engine::LeasedFrame),
}

#[cfg(feature = "backend-vulkan")]
impl FlushFrame {
    fn len(&self) -> usize {
        match self {
            FlushFrame::Owned(bytes) => bytes.len(),
            FlushFrame::Leased(leased) => leased.bytes().len(),
        }
    }
}

#[cfg(not(feature = "backend-vulkan"))]
fn flush_render_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    _source: &crate::model::RenderWindowSource,
) -> bool {
    // No engine ⇒ nothing can have deferred; drop the obligation fail-visibly.
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=render mapping={} reason=no_backend",
        key.mapping_id
    ));
    false
}

#[cfg(feature = "backend-vulkan")]
fn flush_storage_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let started = std::time::Instant::now();
    // Two unrelated `u32` generations are in scope here and they must not be
    // confused in the log: `key.map_generation` is the mapping's lifetime, the
    // quantity this guard compares, and `generation` is the pinned resident's
    // *content* generation, which only `read_resident_storage` uses. The fail
    // line below printed `content_gen` in a field named `gen` next to
    // `reason=map_generation_drift`, so a live boot read out as a mapping
    // lifetime that had gone backwards (3 -> 2) when the two numbers were
    // simply not comparable. `gen=` is the compared value; the other one says
    // so in its name.
    //
    // Same recycled-pages guard as the render flush: a surface window whose
    // defer-time map_generation no longer matches must not write through the
    // rewired pages.
    // Same prior question as the render rail: is this writeback owed at all.
    if crate::runtime::resource_validity::writeback_refused(state, key.mapping_id) {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} {}x{} fmt={:#x} gen={} \
             content_gen={generation} reason=host_copy_superseded (the guest declared \
             its own pages authoritative after the dispatch this window defers)",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    let current = state
        .mappings
        .get(&key.mapping_id)
        .map(|m| m.map_generation);
    if current != Some(key.map_generation) {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} {}x{} fmt={:#x} gen={} content_gen={generation} reason=map_generation_drift current={current:?}",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    if !mapping_pages_still_ours(state, host, key.mapping_id) {
        crate::backend::vulkan::engine::unpin_resident_storage(key);
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} {}x{} fmt={:#x} gen={} content_gen={generation} reason=mapping_page_drift",
            key.mapping_id, key.width, key.height, key.pixel_format, key.map_generation
        ));
        return false;
    }
    let (bytes, texel) =
        match crate::backend::vulkan::engine::read_resident_storage(key, generation) {
            Ok(v) => v,
            Err(e) => {
                // The pinned resident vanished (device loss, guest reset,
                // same-identity key change). The window keeps its coherent
                // pre-dispatch bytes; name the loss.
                crate::observe::Emit::decline("deferred_flush_lost", &e)
                    .field("kind", "compute")
                    .field("mapping", key.mapping_id)
                    .field("geom", format!("{}x{}", key.width, key.height))
                    .field("fmt", format!("{:#x}", key.pixel_format))
                    .field("gen", key.map_generation)
                    .field("content_gen", generation)
                    .fail();
                return false;
            }
        };
    let expected_bpp = crate::contract::pixel_format::bytes_per_pixel(key.pixel_format);
    if expected_bpp != Some(texel) {
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} reason=texel_mismatch engine={texel} guest={expected_bpp:?} fmt={:#x}",
            key.mapping_id, key.pixel_format
        ));
        return false;
    }
    let tight = key.width.saturating_mul(texel);
    if !crate::runtime::mapping_write::write_full_rect_raw_at(
        state,
        host,
        key.mapping_id,
        key.surface_offset,
        key.surface_bpr,
        key.span_end,
        key.width,
        key.height,
        texel,
        &bytes,
        tight,
    ) {
        crate::observe::fail(format!(
            "deferred_flush_lost kind=compute mapping={} reason=guest_write {}x{} off={} bpr={} span_end={}",
            key.mapping_id,
            key.width,
            key.height,
            key.surface_offset,
            key.surface_bpr,
            key.span_end
        ));
        return false;
    }
    // Guest pages now hold exactly the resident content at `generation`:
    // re-establish the mirror entry the write's own invalidation dropped so
    // chained seed skips stay live.
    state.compute_storage_residency.insert(*key, generation);
    crate::runtime::drain::note_drain_phase(
        crate::runtime::drain::DrainPhase::Flush(crate::runtime::drain::FlushRail::Storage),
        started,
    );
    crate::observe::off(format!(
        "compute_deferred_flush mapping={} {}x{} fmt={:#x} gen={generation} bytes={} us={}",
        key.mapping_id,
        key.width,
        key.height,
        key.pixel_format,
        bytes.len(),
        started.elapsed().as_micros()
    ));
    true
}

#[cfg(not(feature = "backend-vulkan"))]
fn flush_storage_one<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    _host: &mut M,
    key: &crate::model::ComputeStorageResidencyKey,
    generation: u32,
) -> bool {
    let _ = state;
    crate::observe::fail(format!(
        "deferred_flush_lost kind=compute mapping={} content_gen={generation} reason=no_backend",
        key.mapping_id
    ));
    false
}

/// Drop (without flushing) every deferred window on `mapping_id` whose pages
/// can no longer be written safely (ReplacePhysical PFN recycling, unmap
/// without host access). Each drop is fail-visible.
pub fn drop_windows(state: &mut DeviceState, mapping_id: u32, reason: &str) {
    let dropped = state.take_deferred_flush_windows(mapping_id, 0, u64::MAX);
    for (key, owner) in dropped {
        crate::observe::fail(format!(
            "deferred_flush_dropped mapping={} reason={reason} {}x{} fmt={:#x} owner={}",
            key.mapping_id,
            key.width,
            key.height,
            key.pixel_format,
            owner_slug(&owner)
        ));
        // The two rails pin different registries, so the release has to follow
        // the owner. Unpinning storage for a render window would leave the
        // target resident pinned for the life of the boot — the "~260 stale
        // residents (~516 MiB)" shape — while reporting a clean teardown.
        #[cfg(feature = "backend-vulkan")]
        release_window_pin(&key, &owner);
    }
}

/// Drop — do not land — every render window whose guest byte range this Store
/// fully covers, releasing what each one held.
///
/// Lives here rather than at the arm site because the *release* lives here, and
/// the arm site got it wrong for exactly that reason: it took each covered window
/// with a bare `take_deferred_flush_window_exact` and discarded it, so a
/// `Resident` window's counted registry pin was never dropped. That is one leaked
/// pin per composite Store on a surface the compositor repaints — and because the
/// re-Store carries the *same* key, it is the same slot's `pin_count` climbing
/// without bound. `evict_registry_to_cap` rotates pinned slots instead of
/// evicting and the idle drain requires `pin_count == 0`, so a slot that gets
/// there can never be reclaimed again: the "~260 stale residents (~516 MiB)
/// pinned for the guest lifetime" shape, arrived at one frame at a time.
///
/// Dropping rather than flushing is what makes the rail a deferral instead of a
/// rescheduling — a compositor painting one surface re-Stores the identical range
/// every frame, so the previous window always intersects, and landing it here
/// would perform exactly the guest write the rail exists to skip. It is sound for
/// the reason it is sound on the GVA rail: those bytes were never observable
/// without a flush, since any reader would have taken the window first, and this
/// Store's pixels cover every byte of the range.
///
/// Returns the identities whose pins were released, so a caller can log them and
/// a test can read the decision. `None` for an `Owned` window is the answer, not
/// a missing one: its pixels are an `Arc` and dropping it *is* the release.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn supersede_covered_render_windows(
    state: &mut DeviceState,
    key: &crate::model::ComputeStorageResidencyKey,
) -> Vec<(
    crate::model::ComputeStorageResidencyKey,
    Option<crate::backend::vulkan::engine::TargetIdentity>,
)> {
    // Matched on the guest byte range, not on geometry: a sibling Store at a
    // different size over the same span writes the same pages, so its window is
    // covered even though its key differs. `release_window_pin` therefore has to
    // rebuild the identity from the *old* key, which is why it takes one.
    let covered: Vec<crate::model::ComputeStorageResidencyKey> = state
        .compute_deferred_flush
        .iter()
        .filter(|(k, o)| {
            k.mapping_id == key.mapping_id
                && k.surface_offset == key.surface_offset
                && k.span_end == key.span_end
                && matches!(o, crate::model::DeferredOwner::Render { .. })
        })
        .map(|(k, _)| *k)
        .collect();
    let mut released = Vec::with_capacity(covered.len());
    for old in covered {
        if let Some(owner) = state.take_deferred_flush_window_exact(&old) {
            released.push((old, release_window_pin(&old, &owner)));
        }
    }
    released
}

/// Release whatever a taken window held, according to its rail.
///
/// Every site that takes a window and does not flush it must go through this
/// rather than calling `unpin_resident_storage` directly. A compute window owns
/// a storage-registry pin; a render window owns nothing on the GPU — its pixels
/// are a `surface_cache` entry, which is LRU-managed and shared with the Load
/// seed, so it must not be evicted here. Unpinning storage for a render window
/// would name a key the storage registry never held and succeed silently.
///
/// Returns the render identity it unpinned, if any. `unpin_resident_target` is a
/// silent no-op for an absent slot and the engine keeps no log of it, so without
/// this return value "the pin was released" is a claim no test and no boot can
/// read — which is how the supersede site went several commits leaking one.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn release_window_pin(
    key: &crate::model::ComputeStorageResidencyKey,
    owner: &crate::model::DeferredOwner,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    match owner {
        crate::model::DeferredOwner::Storage { .. } => {
            crate::backend::vulkan::engine::unpin_resident_storage(key);
            None
        }
        crate::model::DeferredOwner::Render { source, .. } => {
            release_window_pin_for_key(key, source)
        }
    }
}

/// Release whatever GPU hold a render window's source carries.
///
/// An `Owned` window holds nothing — its pixels are an `Arc` and dropping it is
/// the release, so `None` here is the answer and not a miss. A `Resident` window
/// holds a counted registry pin, and **every** exit that abandons the window has
/// to drop it: `evict_registry_to_cap` and the idle drain both skip pinned slots
/// by design, so a leaked pin strands a whole framebuffer for the guest lifetime
/// rather than merely delaying a reclaim.
#[cfg(feature = "backend-vulkan")]
fn release_window_pin_for_key(
    key: &crate::model::ComputeStorageResidencyKey,
    source: &crate::model::RenderWindowSource,
) -> Option<crate::backend::vulkan::engine::TargetIdentity> {
    if !matches!(source, crate::model::RenderWindowSource::Resident { .. }) {
        return None;
    }
    let identity = render_window_identity(key);
    crate::backend::vulkan::engine::unpin_resident_target(&identity);
    Some(identity)
}

pub(crate) fn owner_slug(owner: &crate::model::DeferredOwner) -> &'static str {
    match owner {
        crate::model::DeferredOwner::Storage { .. } => "compute",
        crate::model::DeferredOwner::Render { .. } => "render",
    }
}

#[cfg(all(test, feature = "backend-vulkan"))]
mod render_flush_witness_tests {
    use super::{
        note_render_flush_cache_read, note_render_flush_landed, note_render_flush_pages_read,
    };
    use crate::model::{DeviceId, DeviceState, MappingEntry, PAGE_SHIFT_X86};

    fn state_with_mapping(mid: u32) -> DeviceState {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.mappings.insert(
            mid,
            MappingEntry {
                mapped: true,
                ..Default::default()
            },
        );
        state
    }

    /// Nothing to score on the first landing: a surface that has only just
    /// arrived has no previous flush whose copies anything could have read.
    #[test]
    fn the_first_landing_of_a_mapping_scores_nothing() {
        let mut state = state_with_mapping(7);
        assert_eq!(note_render_flush_landed(&mut state, 7, true), None);
        let w = state.mappings[&7].render_flush;
        assert_eq!(
            (w.landed, w.cache_unread, w.pages_unread),
            (true, true, true)
        );
    }

    /// The age is stamped from the landing, not left at the `Default` zero: a
    /// zero stamp would score every second landing as a frame-plus survivor and
    /// hide exactly the burst case the bucket exists to find.
    #[test]
    fn a_landing_stamps_the_time_it_landed() {
        let mut state = state_with_mapping(7);
        note_render_flush_landed(&mut state, 7, true);
        let first = state.mappings[&7].render_flush.landed_us;
        assert!(first > 0, "landing must stamp a live clock reading");
        note_render_flush_landed(&mut state, 7, true);
        assert!(
            state.mappings[&7].render_flush.landed_us >= first,
            "each landing re-stamps"
        );
    }

    /// The whole point of the witness: a flush neither leg was read from is
    /// reported as unread, which is what says the readback bought nothing.
    #[test]
    fn a_landing_nothing_read_scores_both_legs_unread() {
        let mut state = state_with_mapping(7);
        note_render_flush_landed(&mut state, 7, true);
        let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
        assert!(scored.cache_unread && scored.pages_unread);
    }

    /// Each reader clears only the copy it took. A cache hit must not excuse
    /// the guest-page write, or a flush whose pages nothing reads would be
    /// scored as consumed and the write leg would look owed when it is not.
    #[test]
    fn each_leg_is_cleared_only_by_its_own_reader() {
        let mut state = state_with_mapping(7);
        note_render_flush_landed(&mut state, 7, true);
        note_render_flush_cache_read(&mut state, 7);
        let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
        assert!(!scored.cache_unread, "cache read must clear the cache leg");
        assert!(
            scored.pages_unread,
            "cache read must not clear the pages leg"
        );

        note_render_flush_pages_read(&mut state, 7);
        let scored = note_render_flush_landed(&mut state, 7, true).expect("third landing scores");
        assert!(!scored.pages_unread, "pages read must clear the pages leg");
        assert!(
            scored.cache_unread,
            "pages read must not clear the cache leg"
        );
    }

    /// A flush that stored no cache copy has no cache leg to score.
    ///
    /// `render_flush_cache_unread` is the number a future reader would use to
    /// decide whether the cache leg is worth keeping, and a borrowed-frame flush
    /// stores nothing — it drops the entry, because the memory holding its frame
    /// goes straight back to the readback pool. Arming the leg anyway would
    /// report a copy that was never made, once per flush, at the rate the guest
    /// paints: a counter that looks like a measurement and is an artefact.
    ///
    /// The pages leg is asserted alongside it, because it is the one that stays
    /// meaningful and the two must not be conflated.
    #[test]
    fn a_flush_that_stored_no_cache_copy_arms_no_cache_leg() {
        let mut state = state_with_mapping(7);
        note_render_flush_landed(&mut state, 7, false);
        let w = state.mappings[&7].render_flush;
        assert!(!w.cache_stored, "no copy was stored");
        assert!(
            !w.cache_unread,
            "an absent copy must not be armed as an unread one"
        );
        assert!(w.pages_unread, "the guest pages were still written");

        // And the scoring of the previous landing skips the leg rather than
        // reporting it either way.
        let scored = note_render_flush_landed(&mut state, 7, true).expect("second landing scores");
        assert!(!scored.cache_stored);

        // A stored copy still scores normally, so the gate narrows the count
        // rather than silencing it.
        note_render_flush_cache_read(&mut state, 7);
        let scored = note_render_flush_landed(&mut state, 7, true).expect("third landing scores");
        assert!(scored.cache_stored && !scored.cache_unread);
    }

    /// A read attributed to a mapping the mapper no longer holds is dropped
    /// rather than resurrecting an entry: the cache outlives its mapping, so a
    /// late read of a stale entry must not create mapping state.
    #[test]
    fn a_read_of_an_unknown_mapping_creates_nothing() {
        let mut state = state_with_mapping(7);
        note_render_flush_cache_read(&mut state, 9);
        note_render_flush_pages_read(&mut state, 9);
        assert!(!state.mappings.contains_key(&9));
        assert_eq!(note_render_flush_landed(&mut state, 9, true), None);
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{ComputeStorageResidencyKey, DeviceId, DeviceState, PAGE_SHIFT_X86};

    fn key(mapping_id: u32, lo: u64, hi: u64) -> ComputeStorageResidencyKey {
        ComputeStorageResidencyKey {
            mapping_id,
            map_generation: 1,
            surface_offset: lo,
            surface_bpr: 64,
            span_end: hi,
            width: 4,
            height: 4,
            pixel_format: 0x46,
            texture_ref: 0,
        }
    }

    /// A render window carrying its own 4x4 BGRA frame — the geometry [`key`]
    /// names, since the flush writes `key.width x key.height` from these bytes.
    fn render_owner(armed_seq: u64) -> crate::model::DeferredOwner {
        crate::model::DeferredOwner::Render {
            armed_seq,
            armed_stamp_seq: 0,
            source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                0u8;
                4 * 4 * 4
            ])),
        }
    }

    #[test]
    fn condemn_keeps_content_state_and_lifecycle_clears_it() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let m = state.mappings.entry(7).or_default();
        m.mapped = true;
        m.has_geom = true;
        m.width = 100;
        m.height = 50;
        m.format = 0x46;
        m.map_generation = 4;
        m.page_entries = vec![5, 9, 13];
        assert!(state.condemn_surface_backing(7));
        let e = state.mappings.get(&7).unwrap();
        assert!(e.mapped, "condemn must not unmap");
        assert!(e.has_geom, "condemn must keep geometry");
        assert_eq!(e.map_generation, 4, "condemn must not bump the generation");
        assert!(e.page_entries.is_empty(), "live bindings must be retired");
        assert_eq!(e.condemned_entries.as_deref(), Some(&[5u32, 9, 13][..]));
        assert!(state.mapping_backing_condemned(7));
        // Second condemn with no resolve between: nothing left to stash — the
        // caller falls back to full teardown (genuinely dead).
        // (mapping_backing_condemned gates that in the drain handler.)
        // A fresh MAP notify does NOT settle the pending decision (the notify
        // may trail our eager resolve of the same surface): the fingerprint
        // survives; only a resolve (or unmap/new-internal) settles it.
        assert!(state.map_surface(7));
        assert!(state.mapping_backing_condemned(7));
        assert!(state.unmap_surface(7));
        assert!(!state.mapping_backing_condemned(7));
        // Pageless mapping: condemn declines (caller tears down).
        let m = state.mappings.entry(8).or_default();
        m.mapped = true;
        assert!(!state.condemn_surface_backing(8));
    }

    #[test]
    fn map_notify_stashes_fingerprint_instead_of_bumping() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let m = state.mappings.entry(5).or_default();
        m.mapped = true;
        m.map_generation = 7;
        m.page_entries = vec![1, 2, 3];
        // The MAP notify often trails the eager resolve that established the
        // same surface: it must not bump (the resolve-time fingerprint compare
        // decides), so a deferred paint's resident/window stay live.
        assert!(state.map_surface(5));
        let e = state.mappings.get(&5).unwrap();
        assert_eq!(e.map_generation, 7, "late MAP notify must not bump");
        assert_eq!(e.condemned_entries.as_deref(), Some(&[1u32, 2, 3][..]));
        assert!(!e.has_geom, "geometry must re-resolve after MAP");
        // Same MappingInternal re-statement: full no-op for content state.
        let m = state.mappings.entry(6).or_default();
        m.mapped = true;
        m.map_generation = 9;
        m.mapping_internal = 0xabc;
        m.page_entries = vec![4, 5];
        m.has_geom = true;
        assert!(state.attach_mapping_internal(6, 0xabc));
        let e = state.mappings.get(&6).unwrap();
        assert_eq!(e.map_generation, 9);
        assert_eq!(e.page_entries, vec![4, 5]);
        assert!(e.has_geom, "same-internal re-statement keeps geometry");
        // Different MappingInternal: genuine new surface — full reset + bump.
        assert!(state.attach_mapping_internal(6, 0xdef));
        let e = state.mappings.get(&6).unwrap();
        assert_eq!(e.map_generation, 10);
        assert!(e.page_entries.is_empty());
    }

    /// The compute flush's drift line must print the generation its guard
    /// compared, not the other one in scope.
    ///
    /// `flush_one` holds two unrelated `u32`s: `key.map_generation` (the
    /// mapping lifetime it compares) and the pinned resident's *content*
    /// generation. The line printed the content generation in a field named
    /// `gen`, adjacent to `reason=map_generation_drift current=…`, and a boot
    /// was read as showing a mapping lifetime running backwards (`gen=3
    /// current=Some(2)`) when the two numbers were never comparable.
    #[test]
    fn the_compute_drift_line_names_the_generation_it_compared() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        // Distinct on purpose: the window's map_generation is 1 (from `key`),
        // the mapping is at 5, and the content generation is 3. Only one pair
        // of those is what the guard compares.
        m.map_generation = 5;
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        let cap = crate::observe::FailCapture::start();
        assert!(!super::flush_intersecting(
            &mut state,
            &mut host,
            9,
            0,
            u64::MAX
        ));
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("reason=map_generation_drift"),
            "wrong refusal: {line}"
        );
        assert!(
            line.contains(" gen=1 ") && line.contains("current=Some(5)"),
            "`gen=` must be the compared window generation: {line}"
        );
        assert!(
            line.contains("content_gen=3"),
            "the resident's content generation must say so in its name: {line}"
        );
        assert!(
            line.contains("kind=compute"),
            "every deferred_flush_lost names its path: {line}"
        );
    }

    /// A type-11 render window is found and landed by the *same* mapping-keyed
    /// trigger the compute rail uses, and is read as a render window.
    ///
    /// This is the property the whole deferred type-11 rail rests on. Its
    /// pixels live in a target resident that `ComputeStorageResidencyKey`
    /// cannot name, so the flush has to dispatch on the owner; if it did not,
    /// `flush_intersecting` would hand a render window to the storage read and
    /// report a compute loss for a window the compute rail never armed. Driving
    /// it through `flush_intersecting` — rather than calling the flush directly
    /// — is deliberate: that call is the choke point every guest-page reader
    /// goes through, so this also pins the trigger wiring.
    ///
    /// The map-generation drift is the cheap way to make the flush take a
    /// decisive branch with no engine present. It doubles as coverage of the
    /// recycled-pages guard: a mapping rebound since arm time must never have a
    /// stale framebuffer written through its new pages.
    #[test]
    fn a_render_window_flushes_through_the_shared_trigger_and_names_its_rail() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        // The window latched map_generation 1 (from `key`); the mapping has
        // since moved to 5, so its pages are not the ones the Store rendered
        // for.
        m.map_generation = 5;
        state
            .compute_deferred_flush
            .insert(key(9, 0, 256), render_owner(1));
        // Route counts are process-global and this suite runs serially, so take
        // a baseline rather than assuming this is the first window to drift.
        let before_gen_drift = crate::runtime::drain::store_route_count("rendflush_gen_drift");
        let cap = crate::observe::FailCapture::start();
        assert!(
            !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window that cannot be written must report the loss"
        );
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("kind=render"),
            "a render window must not be reported as a compute one: {line}"
        );
        assert!(
            line.contains("reason=map_generation_drift") && line.contains("current=Some(5)"),
            "the rebound mapping must be the stated refusal: {line}"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the trigger must consume the window it took"
        );
        // A lost tile has to be countable, not just loggable. The window is gone
        // from `compute_deferred_flush` (asserted just above) and nothing
        // re-arms it, so this is a permanent loss of painted pixels — the Goal 3
        // event — and a census that cannot count it cannot score an arm against
        // it.
        //
        // `mapping_pages_drifted` is not a substitute: it is incremented inside
        // `mapping_pages_still_ours`, which more than one caller reaches, so it
        // counts refusals rather than lost tiles.
        assert_eq!(
            crate::runtime::drain::store_route_count("rendflush_gen_drift"),
            before_gen_drift + 1,
            "the generation-drift loss must be counted on the store-route census"
        );
    }

    /// The other drift refusal, and the one a live boot actually takes.
    ///
    /// `map_generation` drift is the guest's *declared* rebind; page drift is a
    /// type-4 surface re-pointed with nothing declared, which is the shape
    /// traced end to end on a control boot — a 1225x512 WebKit tile whose
    /// backing was fabricated at its own GVA, then refused when the live walk
    /// disagreed. Both refusals are correct and both lose the tile, so both have
    /// to be countable apart; testing only the sibling would leave the branch a
    /// live boot exercises uncovered.
    #[test]
    fn a_render_window_over_repointed_pages_is_refused_and_counted() {
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::model::Type4Walk;
        use crate::runtime::host::{FakeHost, HostMemory};

        const BACKING_PFN: u32 = 9;
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data0 = 4u64 << PAGE_SHIFT_X86;
        for gpa in [dir_gpa, root_gpa, data0] {
            host.map_range(gpa, page as usize, 0);
        }
        let st32 = |b: &mut [u8], v: u32| b[..4].copy_from_slice(&v.to_le_bytes());
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        // Depth-1 table: the live translation of GVA page 9 is `data0`.
        let mut pte = [0u8; 4];
        st32(&mut pte, (data0 >> PAGE_SHIFT_X86) as u32);
        host.write_gpa(root_gpa + u64::from(BACKING_PFN) * 4, &pte)
            .unwrap();

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, page, 2));
        {
            let m = state.mappings.entry(9).or_default();
            m.mapped = true;
            // The generation still matches the window's, so this test cannot
            // pass through the sibling's branch: the only thing wrong is where
            // the cached entry points.
            m.map_generation = 1;
            m.page_entries = vec![(0x77u32 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
            m.type4_walk = Some(Type4Walk {
                task_id: 1,
                backing_pfn: BACKING_PFN,
                map_generation: 1,
            });
        }
        state
            .compute_deferred_flush
            .insert(key(9, 0, 256), render_owner(1));
        let before = crate::runtime::drain::store_route_count("rendflush_page_drift");
        let cap = crate::observe::FailCapture::start();
        assert!(
            !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window whose pages moved must report the loss"
        );
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("kind=render") && line.contains("reason=mapping_page_drift"),
            "the re-pointed pages must be the stated refusal, not the generation: {line}"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the trigger consumes the window it took, so the obligation is gone"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("rendflush_page_drift"),
            before + 1,
            "the page-drift loss must be counted on the store-route census"
        );
    }

    /// A window landing over pages the guest wrote preserves nothing and says
    /// so; one landing over untouched pages preserves nothing and stays quiet.
    ///
    /// Both halves are the test. The report has to be keyed on the guest write
    /// and not on the landing — the writeback runs on every landing and the
    /// interesting population is the subset the guest also wrote — so the
    /// untouched arm is what makes the reporting arm mean anything.
    ///
    /// This test asserted the opposite of its first half until the rail was
    /// bisected on live boots: the preserving behaviour turned the screen black
    /// (0 of 14 rounds, against 3 of 4 and 2 of 4 clean on the two commits
    /// before it), because `page_gen` is stamped at the harvest and not at the
    /// write, so a store the device's own render superseded can still be named
    /// "written since the Store". See
    /// [`super::note_render_flush_over_guest_write`], which returns nothing at
    /// all now — "preserves nothing" is in its signature and no longer only in
    /// this assertion.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_render_window_landing_over_guest_writes_reports_them_and_preserves_nothing() {
        use crate::runtime::host::{FakeHost, HostOps};
        let page = 1u64 << PAGE_SHIFT_X86;
        for guest_wrote in [false, true] {
            let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
            let mut host = FakeHost::new();
            let token = host
                .track_guest_writes(&[page], 1usize << PAGE_SHIFT_X86)
                .unwrap();
            let stamped = host.guest_write_gen(token).unwrap();
            let m = state.mappings.entry(9).or_default();
            m.mapped = true;
            m.map_generation = 1;
            // The one tracked page IS this surface's page 0, so the report the
            // host gives back has somewhere in the mapping to land.
            let pfn = (page >> PAGE_SHIFT_X86) as u32;
            m.page_entries = vec![
                (pfn << crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
                    | crate::contract::iosurface_pages::PAGE_ENTRY_VALID,
            ];
            m.guest_write_token = token;
            m.guest_write_token_gen = 1;
            m.guest_write_gen_at_store = stamped;
            if guest_wrote {
                host.guest_wrote_page(page);
            }
            let cap = crate::observe::FailCapture::start();
            super::note_render_flush_over_guest_write(&state, &host, &key(9, 0, 256));
            let clobbers: Vec<String> = cap
                .lines()
                .into_iter()
                .filter(|l| l.split_whitespace().next() == Some("deferred_flush_clobber"))
                .collect();
            assert_eq!(
                clobbers.len(),
                usize::from(guest_wrote),
                "guest_wrote={guest_wrote} must decide whether the loss is reported: {clobbers:?}"
            );
        }
    }

    /// All the report knows is that the generation moved since the stamp. It
    /// carries no ordering against the Store, so the line must not claim one.
    ///
    /// Why that matters is at the shim, not here. `reims_vgpu_dirty_gen` answers
    /// with the generation as of the last harvest and only marks a read as owed;
    /// `reims_vgpu_dirty_harvest` returns early unless one is owed, and runs at
    /// the drain tail. A guest store in a tranche whose harvest has not yet run
    /// is therefore stamped into a generation that moves *after* the Store, and
    /// arrives here indistinguishable from a store that genuinely followed it —
    /// the same unsoundness that made preserving the pages black the screen.
    ///
    /// `FakeHost` cannot stage that: `guest_wrote_page` is both the write and
    /// the observation, so the harvest lag has no double here and this fixture
    /// does not reproduce it. What it does pin is the consequence — the verdict
    /// is a bare generation comparison — and that the emitted line stops short of
    /// an ordering claim. `note_render_flush_over_guest_write`'s doc used to make
    /// that claim two paragraphs after `render_flush_guest_written_ranges` stated
    /// the opposite rule.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn the_clobber_report_claims_no_ordering_against_the_store() {
        use crate::runtime::host::{FakeHost, HostOps};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let token = host
            .track_guest_writes(&[page], 1usize << PAGE_SHIFT_X86)
            .unwrap();

        let stamped = host.guest_write_gen(token).unwrap();
        // The only input the verdict has: the generation is no longer `stamped`.
        // Whether the store behind it preceded or followed the Store is exactly
        // what neither this fixture nor the product witness can say.
        host.guest_wrote_page(page);

        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.map_generation = 1;
        m.page_entries = vec![
            (((page >> PAGE_SHIFT_X86) as u32)
                << crate::contract::iosurface_pages::PAGE_ENTRY_PFN_SHIFT)
                | crate::contract::iosurface_pages::PAGE_ENTRY_VALID,
        ];
        m.guest_write_token = token;
        m.guest_write_token_gen = 1;
        m.guest_write_gen_at_store = stamped;

        let cap = crate::observe::FailCapture::start();
        super::note_render_flush_over_guest_write(&state, &host, &key(9, 0, 256));
        let clobbers: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some("deferred_flush_clobber"))
            .collect();
        assert_eq!(
            clobbers.len(),
            1,
            "the witness cannot order the write against the Store, so it reports \
             either way — the line is an upper bound, not a defect count"
        );
        assert!(
            !clobbers[0].contains("wrote pages of this surface after"),
            "the line must not claim an ordering the witness cannot establish: {:?}",
            clobbers[0]
        );
    }

    /// A `Resident` window whose resident no longer vouches for the frame
    /// declines, and leaves the guest's pages exactly as it found them.
    ///
    /// This is the whole safety argument for the `skip_readback` rail. An `Owned`
    /// window carries its pixels and cannot be wrong about them; a `Resident`
    /// window carries only a *claim* that a GPU image still holds them, and the
    /// epoch is what tests the claim. `registry_mark_ready` clears a slot's
    /// `content_epoch` on every draw into it, so a mismatch means another layer
    /// rendered over this surface after the Store that armed the window — and
    /// writing then lands that other layer's pixels in these pages, which is the
    /// black/torn-layer class rather than a merely stale frame.
    ///
    /// No engine is initialized here, so the registry has no slot at the
    /// reconstructed identity at all, and the refusal this asserts is therefore
    /// `resident_absent` rather than `resident_epoch_cleared`. Those two used to
    /// be one `reason=resident_epoch_drift` with `live=None`, and separating
    /// them is what `engine::ResidentContent` exists for: an un-stamped slot is
    /// expected traffic, a missing one cannot happen to a pinned identity and
    /// means the arm and the flush name the target differently.
    ///
    /// The assertion that matters either way is the *guest memory*: a decline
    /// that still wrote would pass a log-only check.
    #[test]
    fn a_resident_window_that_cannot_be_vouched_for_declines_without_writing() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostMemory};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x4500_0000u64;
        host.map_range(gpa, page as usize, 0);
        // A recognizable pre-Store pattern, so "did not write" is checkable
        // rather than indistinguishable from a zeroed page.
        let pre = [0x5Cu8; 256];
        host.write_gpa(gpa, &pre).unwrap();
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Resident { epoch: 7 },
            },
        );
        let cap = crate::observe::FailCapture::start();
        assert!(
            !super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window whose resident cannot be vouched for must report the loss"
        );
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("kind=render")
                && line.contains("reason=resident_absent")
                && line.contains("want=7"),
            "the epoch witness must be the stated refusal, naming which kind of \
             absence it was and the value it wanted: {line}"
        );
        let mut after = [0u8; 256];
        host.read_gpa(gpa, &mut after).unwrap();
        assert_eq!(
            &after[..],
            &pre[..],
            "a declined resident window must leave the guest's own bytes untouched"
        );
        assert!(
            state.compute_deferred_flush.is_empty(),
            "the trigger must consume the window it took"
        );
    }

    /// The identity a `Resident` window's flush rebuilds from its key is the one
    /// the draw rendered into, pinned and stamped.
    ///
    /// Four separate places name this slot — the draw's `target_identity`, the
    /// arm's `pin_resident_target`, the arm's `stamp_resident_content_epoch`, and
    /// the flush's `read_target` — and all four resolve through
    /// `present_identity::surface_identity` except the last, which has only the
    /// key. If those two spellings ever disagree the pin protects one image while
    /// the flush reads another: the frame is silently the wrong one, and no
    /// assertion in the crate is watching for it because both lookups *succeed*.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_render_windows_key_rebuilds_the_identity_the_draw_rendered_into() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        for generation in [1u32, 5, u32::MAX] {
            let m = state.mappings.entry(9).or_default();
            m.map_generation = generation;
            let mut k = key(9, 0, 256);
            k.map_generation = generation;
            assert_eq!(
                super::render_window_identity(&k),
                crate::runtime::present_identity::surface_identity(&state, 9, k.width, k.height),
                "the flush's rebuilt identity must equal the one the draw and the pin used"
            );
        }
    }

    /// A render window lands its own pixels even when `surface_cache` has moved
    /// on to another geometry for the same mapping.
    ///
    /// The flush used to source its bytes from
    /// `surface_cache::get(mapping_id, key.width, key.height)`, and that cache
    /// holds exactly one entry per mapping. A guest that re-Stores the surface at
    /// a new size therefore orphaned every window still armed at the old one:
    /// the flush missed, emitted `deferred_flush_lost reason=cache_miss` and the
    /// guest kept its stale pixels. One boot lost 15 whole layers that way —
    /// including a 1920x1080 desktop surface and a 1920x24 menu bar — which on
    /// screen is a compositing layer rendering solid black.
    #[test]
    fn a_render_window_lands_its_own_pixels_after_the_cache_moved_geometry() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::{FakeHost, HostMemory};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page = 1u64 << PAGE_SHIFT_X86;
        let gpa = 0x4400_0000u64;
        host.map_range(gpa, page as usize, 0);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.mapped = true;
            m.map_generation = 1;
            m.has_geom = true;
            m.width = 4;
            m.height = 4;
            m.format = crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
            m.page_entries =
                vec![(((gpa >> PAGE_SHIFT_X86) as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        // The window's own frame — every byte 0xA7.
        let frame = vec![0xA7u8; 4 * 4 * 4];
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(frame.clone())),
            },
        );
        // A later Store re-Stored this mapping at 8x8, replacing the one cache
        // entry it has. The 4x4 window above is now unreachable through it.
        crate::runtime::surface_cache::store(&mut state, 9, 8, 8, vec![0x11u8; 8 * 8 * 4]);

        let cap = crate::observe::FailCapture::start();
        assert!(
            super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX),
            "a window carrying its own pixels is always landable"
        );
        assert!(
            cap.lines()
                .iter()
                .all(|l| !l.contains("deferred_flush_lost")),
            "nothing may be lost: {:?}",
            cap.lines()
        );
        // The guest side is row-strided at the mapping's own bytes-per-row, so
        // read it the way the writeback wrote it.
        let (base_off, bpr, _) = {
            let m = state.mappings.get(&9).unwrap();
            crate::runtime::mapping_write::type11_sample_window(
                m,
                9,
                4,
                4,
                crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            )
            .expect("the mapping has a type-11 sample window")
        };
        for y in 0..4u64 {
            let mut row = [0u8; 4 * 4];
            host.read_gpa(gpa + base_off + y * bpr as u64, &mut row)
                .unwrap();
            assert_eq!(
                &row[..],
                &frame[(y as usize) * 16..(y as usize) * 16 + 16],
                "row {y} of the guest pages must hold the window's frame, not the cache's"
            );
        }
    }

    /// A render window fully covered by a later writer is *dropped*, not
    /// flushed, and dropping it takes its alias-index refs with it.
    ///
    /// This is the difference between a deferral and a rescheduling. A guest
    /// compositing into one surface re-Stores the identical guest range every
    /// frame, so the previous window always intersects the new one; landing it
    /// there performs exactly the guest write the rail exists to skip, once per
    /// Store, and `surface_flush` would track `surface_deferred` at a ratio of 1.
    ///
    /// The alias-index half is the part that is easy to get wrong: taking the
    /// entry with a bare `remove` leaves `deferred_alias_pages` holding page
    /// refs for a mapping with no windows left, and the raw-GVA sampling guard
    /// then walks pages nothing defers on.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_superseded_render_window_is_dropped_and_releases_its_alias_pages() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        let k = key(9, 0, 256);
        state.compute_deferred_flush.insert(k, render_owner(1));
        state.index_deferred_alias_pages(9);
        assert!(
            state.deferred_alias_pages.contains_key(&9),
            "arming indexes the mapping's pages for the raw-GVA guard"
        );

        let released = super::supersede_covered_render_windows(&mut state, &k);
        assert_eq!(
            released.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            vec![k],
            "the exact key is the one taken"
        );
        assert!(state.compute_deferred_flush.is_empty());
        assert!(
            !state.deferred_alias_pages.contains_key(&9),
            "the last window leaving must drop the mapping's alias-page refs"
        );
    }

    /// The other half of dropping a superseded window: a `Resident` one holds a
    /// counted registry pin, and the supersede is one of the exits
    /// `release_window_pin` names.
    ///
    /// The arm site got this wrong. It took each covered window with a bare
    /// `take_deferred_flush_window_exact` and discarded it, so every composite
    /// Store on a repainted surface leaked one pin — and since the re-Store
    /// carries the same key, it is the same slot's `pin_count` climbing without
    /// bound until nothing can ever reclaim it. `unpin_resident_target` is a
    /// silent no-op with no engine here, so the assertion is on the *identity*
    /// the release named: it has to be rebuilt from the superseded window's own
    /// key, since a covered sibling may carry a different geometry over the same
    /// guest range.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn superseding_a_resident_window_releases_the_pin_it_held() {
        use crate::backend::vulkan::engine::TargetIdentity;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut k = key(9, 0, 256);
        k.map_generation = 3;
        state.compute_deferred_flush.insert(
            k,
            crate::model::DeferredOwner::Render {
                armed_seq: 1,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Resident { epoch: 11 },
            },
        );

        let released = super::supersede_covered_render_windows(&mut state, &k);
        assert_eq!(
            released,
            vec![(
                k,
                Some(TargetIdentity::Surface {
                    id: 9,
                    width: k.width,
                    height: k.height,
                    generation: 3,
                })
            )],
            "a resident window's pin must be released, under the identity its own key names"
        );

        // An `Owned` window holds nothing on the GPU, so `None` is the answer and
        // not a missed release — unpinning for one would name a slot the arm never
        // pinned and succeed silently.
        state.compute_deferred_flush.insert(k, render_owner(2));
        assert_eq!(
            super::supersede_covered_render_windows(&mut state, &k),
            vec![(k, None)],
            "an owned window releases nothing"
        );
    }

    /// Superseding one window must not disturb a sibling covering a different
    /// guest range on the same mapping — that one holds bytes the new Store does
    /// not write, and dropping it would lose them.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn superseding_one_window_leaves_a_disjoint_sibling_armed() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let covered = key(9, 0, 256);
        let sibling = key(9, 256, 512);
        state
            .compute_deferred_flush
            .insert(covered, render_owner(1));
        state
            .compute_deferred_flush
            .insert(sibling, render_owner(2));

        assert_eq!(
            super::supersede_covered_render_windows(&mut state, &covered).len(),
            1
        );
        assert!(
            state.compute_deferred_flush.contains_key(&sibling),
            "a different range is a different obligation"
        );
        assert_eq!(state.compute_deferred_flush.len(), 1);
    }

    /// Teardown must name the render rail, because the two rails pin different
    /// registries and the drop is where the pin is released.
    ///
    /// Unpinning storage for a render window succeeds silently and leaves the
    /// target resident pinned for the life of the boot — a display-sized image
    /// per window, which is the "~260 stale residents (~516 MiB)" shape. The
    /// slug on this line is the only always-on evidence that the right registry
    /// was chosen.
    #[test]
    fn dropping_a_render_window_reports_the_render_rail() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state
            .compute_deferred_flush
            .insert(key(9, 0, 256), render_owner(7));
        state.compute_deferred_flush.insert(
            key(9, 256, 512),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        let cap = crate::observe::FailCapture::start();
        super::drop_windows(&mut state, 9, "unit");
        let lines: Vec<String> = cap
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some("deferred_flush_dropped"))
            .collect();
        assert_eq!(lines.len(), 2, "both windows drop: {lines:?}");
        assert!(
            lines.iter().any(|l| l.contains("owner=render")),
            "the render window must say so: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("owner=compute")),
            "the compute window must say so: {lines:?}"
        );
        assert!(state.compute_deferred_flush.is_empty());
    }

    /// `condemn_surface_backing` keeps a mapping's deferred windows on purpose:
    /// `DeleteIOSurfaceBacking2` may name a prior incarnation of a recycled id,
    /// and `mapper::resolve` settles it later by fingerprint compare. A flush
    /// trigger arriving inside that undecided window must therefore leave the
    /// obligation armed — consuming it destroys the very thing the fingerprint
    /// decision exists to reprieve, and reports a loss the flush did not cause.
    #[test]
    fn flush_holds_windows_while_the_backing_is_condemned() {
        use crate::contract::iosurface_pages::{PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID};
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let k = key(9, 0, 4096);
        state.map_surface(9);
        {
            let m = state.mappings.get_mut(&9).unwrap();
            m.map_generation = 2;
            m.page_entries = vec![(0x300 << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID];
        }
        state.compute_deferred_flush.insert(
            k,
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        // The guest deletes the backing; the window is kept for the fingerprint
        // decision and the page list moves to `condemned_entries`.
        assert!(state.condemn_surface_backing(9));
        assert!(state.mapping_backing_condemned(9));
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(ok, "an undecided window is not a loss");
        assert!(
            state.compute_deferred_flush.contains_key(&k),
            "the window must survive for mapper::resolve to reprieve or drop"
        );
    }

    /// A window whose mapping the guest has since declared it owns must not
    /// land, and the refusal must name itself.
    ///
    /// The window's frame is what the device rendered *before* the guest's CPU
    /// write. Landing it replaces the guest's own bytes over the full attachment
    /// extent with a copy the guest has already said is stale. Every other guard
    /// on this path asks where the bytes would land; this one asks whether they
    /// are owed at all, which is why it runs first.
    #[test]
    fn a_window_the_guest_superseded_is_refused_by_name() {
        use crate::runtime::host::FakeHost;
        use crate::runtime::resource_validity::{apply, ValiditySite};
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let k = key(9, 0, 4096);
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.map_generation = k.map_generation;
        state.compute_deferred_flush.insert(
            k,
            crate::model::DeferredOwner::Render {
                armed_seq: 0,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                    0u8;
                    4096
                ])),
            },
        );
        // The device published this surface's pixels, and the guest then claimed
        // a CPU write to it — so the guest's bytes are the newer ones.
        state.note_surface_content_published(9);
        apply(
            &mut state,
            0,
            9,
            crate::runtime::decode::fifo::InvalidateValidityOps {
                clear_host_valid: 1,
                set_host_valid: 0,
                clear_guest_valid: 0,
                set_guest_valid: 0,
            },
            ValiditySite::ExecTable,
        );
        // The claim also drops the window, which is the repair upstream of this
        // gate; re-arm it so the gate itself is what this exercises.
        state.compute_deferred_flush.insert(
            k,
            crate::model::DeferredOwner::Render {
                armed_seq: 0,
                armed_stamp_seq: 0,
                source: crate::model::RenderWindowSource::Owned(std::sync::Arc::new(vec![
                    0u8;
                    4096
                ])),
            },
        );

        let cap = crate::observe::FailCapture::start();
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(!ok, "a refused writeback is a reported loss, not a success");
        let line = cap.one("deferred_flush_lost");
        assert!(
            line.contains("reason=host_copy_superseded"),
            "the refusal must name itself: {line}"
        );
        assert!(state.compute_deferred_flush.is_empty());
    }

    #[test]
    fn flush_intersecting_takes_windows_and_reports_loss() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        // Window over an unmapped mapping: the flush must fail closed
        // (fail-visible loss), remove the window, and return false.
        state.compute_deferred_flush.insert(
            key(9, 0, 4096),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        let ok = super::flush_intersecting(&mut state, &mut host, 9, 0, u64::MAX);
        assert!(!ok, "lost window must report failure");
        assert!(
            state.compute_deferred_flush.is_empty(),
            "taken windows never return to the map"
        );
        // Disjoint mapping id: untouched.
        state.compute_deferred_flush.insert(
            key(10, 0, 4096),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        assert!(super::flush_intersecting(
            &mut state,
            &mut host,
            11,
            0,
            u64::MAX
        ));
        assert_eq!(state.compute_deferred_flush.len(), 1);
    }

    /// A raw task-GVA span whose physical pages alias a deferred window's
    /// mapping pages must take (and attempt to flush) that window; a window
    /// on non-aliased pages stays. Locks the boot-18 linear_sample poisoning
    /// channel: GVA reads bypassing the mapping-keyed hooks.
    #[test]
    fn gva_alias_takes_only_aliased_windows() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        // Task 1 directory at pfn 2 → root table pfn 3 → gva page 0 =
        // pfn 0x2000. Data pfns sit past the default task object list
        // (pfn 1 + 0x100000 slots = 4096 pages), which the mapping
        // control-page collision check treats as reserved.
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 0x2000u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        // Mapping 9 is backed by pfn 0x2000 (the page the GVA span resolves
        // to); mapping 10 is backed by pfn 0x2001 (disjoint).
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![page_entry(pfn)];
        }
        let ckey = |mapping_id: u32| key(mapping_id, 0, 0x1000);
        state.compute_deferred_flush.insert(
            ckey(9),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        state.compute_deferred_flush.insert(
            ckey(10),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        // Product defer sites index pages at defer time.
        state.index_deferred_alias_pages(9);
        state.index_deferred_alias_pages(10);
        assert_eq!(state.deferred_alias_pages.len(), 2);

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.compute_deferred_flush.contains_key(&ckey(9)),
            "aliased window must be taken for flush"
        );
        assert!(
            state.compute_deferred_flush.contains_key(&ckey(10)),
            "non-aliased window must stay deferred"
        );
        assert!(
            !state.deferred_alias_pages.contains_key(&9),
            "alias index must drop with the mapping's last window"
        );
        assert!(
            state.deferred_alias_pages.contains_key(&10),
            "alias index for the untouched mapping must stay"
        );
    }

    /// SynchronizeResources choke point: the guest names a mapping it is
    /// about to CPU-read; every deferred window on it — mapping-keyed
    /// (compute) and linear windows whose defer-time page index aliases the
    /// mapping's physical pages — must be taken for flush.
    /// Windows on disjoint mappings/pages stay deferred. Locks the
    /// boot-25 black-wallpaper class (guest-CPU composite of stale pages).
    #[test]
    fn guest_read_flush_takes_keyed_and_linear_alias_windows() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        for (mid, pfn) in [(9u32, 0x2000u32), (10, 0x2001)] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![page_entry(pfn)];
        }
        state.compute_deferred_flush.insert(
            key(9, 0, 256),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        let disjoint = key(10, 0, 0x1000);
        state.compute_deferred_flush.insert(
            disjoint,
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        // Linear windows never name the mapping: one aliases mapping 9's
        // physical page, one sits on a disjoint page.
        let mut lin_aliased = key(0, 0, 0x1000);
        lin_aliased.texture_ref = 42;
        let mut lin_disjoint = key(0, 0, 0x1000);
        lin_disjoint.texture_ref = 43;
        let aliased_pages: std::collections::HashSet<u64> =
            [(0x2000u64) << PAGE_SHIFT_X86].into_iter().collect();
        let disjoint_pages: std::collections::HashSet<u64> =
            [(0x3000u64) << PAGE_SHIFT_X86].into_iter().collect();
        state.arm_linear_deferred_window(lin_aliased, 1, aliased_pages);
        state.arm_linear_deferred_window(lin_disjoint, 1, disjoint_pages);

        // No windows on mapping 11: clean no-op.
        assert_eq!(
            super::flush_mapping_for_guest_read(&mut state, &mut host, 11),
            (true, 0)
        );

        let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        // Nothing is engine-pinned / host-mapped in this fixture, so every
        // flush reports a fail-visible loss — but every aliased window must
        // still be taken (obligations never return to the maps).
        assert!(!ok, "losses must be reported");
        assert_eq!(flushed, 2, "compute@9 + linear alias");
        assert!(!state.compute_deferred_flush.contains_key(&key(9, 0, 256)));
        assert!(
            state.compute_deferred_flush.contains_key(&disjoint),
            "disjoint mapping's window must stay deferred"
        );
        assert!(
            !state.linear_deferred_flush.contains_key(&lin_aliased),
            "page-aliased linear window must be taken"
        );
        assert!(
            state.linear_deferred_flush.contains_key(&lin_disjoint),
            "disjoint-page linear window must stay deferred"
        );
    }

    fn gva_entry(task_id: u32, w: u32, h: u32, pages: &[u64]) -> crate::model::GvaDeferredEntry {
        crate::model::GvaDeferredEntry {
            task_id,
            texture_ref: 5,
            producer_object_type: 2,
            width: w,
            height: h,
            row_stride: w * 4,
            format: 0x46,
            armed_seq: 0,
            armed_stamp_seq: 0,
            pages: pages.iter().copied().collect(),
            alloc_gen: 0,
        }
    }

    /// A linear compute-storage window records the fence it was armed under, and
    /// a re-arm records the fence it was re-armed under.
    ///
    /// This rail writes a raw task GVA with no mapping incarnation to name, so
    /// the only thing that can say a landing is late is the stamp counter at arm
    /// time. Without it every linear landing is unscoreable, which is the state
    /// `6bc2220` left it in while clearing the two rails that *do* carry an
    /// allocation identity.
    #[test]
    fn a_linear_window_records_the_fence_it_was_armed_under() {
        use crate::model::{ComputeStorageResidencyKey, DeviceState, PAGE_SHIFT_X86};
        let mut state = DeviceState::new(crate::model::DeviceId(1), PAGE_SHIFT_X86);
        let p = |pfn: u64| pfn << PAGE_SHIFT_X86;
        let key = ComputeStorageResidencyKey::linear(1, 7, 0x4000, 256, 0x1000, 64, 64, 0x46);

        state.completion_stamp_seq = 41;
        state.arm_linear_deferred_window(key, 1, [p(0xA)].into_iter().collect());
        assert_eq!(
            state
                .linear_deferred_flush
                .get(&key)
                .unwrap()
                .armed_stamp_seq,
            41,
            "the window must carry the fence it was armed under"
        );

        // The guest is fenced twice, then the same key re-arms: the window is a
        // NEW obligation and must be scored against the new fence, not the one
        // its predecessor was armed under.
        state.completion_stamp_seq = 43;
        state.arm_linear_deferred_window(key, 2, [p(0xB)].into_iter().collect());
        let window = state.disarm_linear_deferred_window(&key).unwrap();
        assert_eq!(window.armed_stamp_seq, 43, "a re-arm re-stamps the window");
        assert_eq!(window.generation, 2);
        assert_eq!(window.pages, [p(0xB)].into_iter().collect());
    }

    /// A raw task-GVA span aliasing a deferred GVA render-Store window's
    /// pages (or naming its base GVA exactly) must take the window; windows
    /// on disjoint pages stay armed. Same channel as the linear windows —
    /// GVA reads that bypass every mapping-keyed hook.
    #[test]
    fn task_gva_alias_takes_gva_store_windows() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 0x2000u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        // Window A aliases the page the span resolves to; window B does not.
        state.arm_gva_deferred_window(0x9000_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        // No engine in this fixture: the flush reports a fail-visible loss,
        // but the aliased window must be taken (obligations never return).
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9000_0000),
            "page-aliased GVA window must be taken"
        );
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "disjoint GVA window must stay armed"
        );

        // Exact-base fast path: a read naming the window's own GVA takes it
        // without any page walk.
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0x9100_0000, 0x10);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "exact-base read must take the window"
        );
    }

    /// PT builder shared by the alias-walk tests: task 1's GVA `0..0x1000`
    /// resolves to data page `0x2000<<shift`, and page `0x3000<<shift` is mapped
    /// but unreferenced so a test can point a PTE at it. Returns the root PTE
    /// GPA so the caller can remap and simulate a task page-table change.
    fn alias_pt_fixture() -> (crate::runtime::host::FakeHost, DeviceState, u64, u32) {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page_shift = PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(0x2000u64 << page_shift, 0x1000, 0xab);
        host.map_range(0x3000u64 << page_shift, 0x1000, 0xcd);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x2000);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = DeviceState::new(DeviceId(1), page_shift);
        assert!(state.define_task(1, 0x1000, 2));
        (host, state, root_gpa, page_shift)
    }

    /// A large bind's alias must be found wherever it sits, not only where a
    /// sample point happens to land.
    ///
    /// This walk used to sample every 16th page once a span passed 64 pages, on
    /// the stated grounds that "real aliases are same-surface, so the first page
    /// hits". Measured on the rail, no alias hit page 0 — the three observed
    /// landed at 16, 32 and 48 of 127- and 256-page spans, i.e. partial overlaps
    /// somewhere below each sample point. So the miss window was live, and this
    /// is what falls through it: a 65-page bind overlapping a window on page 1
    /// alone, which a stride of 16 steps straight over.
    #[test]
    fn a_large_bind_alias_is_found_off_the_sample_points() {
        use crate::contract::endian::st32;
        use crate::runtime::host::HostMemory;
        let (mut host, mut state, root_gpa, page_shift) = alias_pt_fixture();
        // 65 pages, so the old rule ran a strided walk. Page i -> pfn 0x4000+i.
        const N: u64 = 65;
        for i in 0..N {
            let pfn = 0x4000 + i;
            host.map_range(pfn << page_shift, 0x1000, 0);
            let mut pte = [0u8; 4];
            st32(&mut pte, pfn as u32);
            host.write_gpa(root_gpa + 4 * i, &pte).unwrap();
        }
        // The deferred window covers page 1 and nothing else. A stride-16 walk
        // visits 0, 16, 32, 48, 64 — never 1.
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x4001u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, N << page_shift);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "a window aliasing page 1 of a 65-page bind must be found and flushed"
        );
    }

    /// The alias walk is never skipped, so a bind that has already been walked
    /// still finds a window armed onto its pages afterwards.
    ///
    /// This used to be answered by the no-intersection memo's cheap page
    /// recheck. With the memo gone the same bind simply walks again, and the
    /// repeat is what this pins: identical `(task, gva, span)`, walked once with
    /// nothing to find, then walked again with a window on its resolved page.
    #[test]
    fn a_repeat_bind_walks_again_and_takes_a_newly_armed_window() {
        let (mut host, mut state, _root, page_shift) = alias_pt_fixture();
        // Disjoint window on page 0x3000 keeps the deferred set non-empty (so
        // the early-out does not answer) but never aliases the [0,0x100) bind,
        // which resolves to page 0x2000.
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "a disjoint window must stay armed"
        );

        // Arm a window ON the bind's resolved page and repeat the same bind.
        state.arm_gva_deferred_window(0x9300_0000, gva_entry(1, 4, 4, &[0x2000u64 << page_shift]));
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9300_0000),
            "the repeat bind must walk again and take the newly armed window"
        );
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "the disjoint window must still stay armed"
        );
    }

    /// A task page-table remap that nothing told the device about is seen by the
    /// very next bind.
    ///
    /// The deferred set does not change here and no invalidation hook fires —
    /// only the guest's PTE moves, so the bind's pages land under an
    /// already-armed window. The memo that used to cache this bind's resolved
    /// pages could not see that; it closed the hole with a 1-in-64 sampled walk,
    /// which left up to 63 binds reading stale bytes. An unconditional walk has
    /// no such hole, and this is the test that would fail if one came back.
    #[test]
    fn a_task_pt_remap_is_seen_by_the_very_next_bind() {
        use crate::contract::endian::st32;
        use crate::runtime::host::HostMemory;
        let (mut host, mut state, root_gpa, page_shift) = alias_pt_fixture();
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(1, 4, 4, &[0x3000u64 << page_shift]));
        // Disjoint at first: the bind resolves to page 0x2000.
        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));

        // Remap gva page 0 -> 0x3000 directly in guest RAM. No retire, no
        // deferred-set change: the bind now aliases the armed window.
        let mut pte = [0u8; 4];
        st32(&mut pte, 0x3000);
        host.write_gpa(root_gpa, &pte).unwrap();

        super::flush_intersecting_task_gva(&mut state, &mut host, 1, 0, 0x100);
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9100_0000),
            "the bind after the remap must flush the window it now aliases"
        );
    }

    /// SynchronizeResources choke point: GVA windows whose defer-time pages
    /// alias the named mapping's physical pages must be taken for flush.
    #[test]
    fn guest_read_flush_takes_gva_store_alias_windows() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let page_entry = |pfn: u32| (pfn << 2) | 1;
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.page_entries = vec![page_entry(0x2000)];
        state.arm_gva_deferred_window(
            0x9000_0000,
            gva_entry(1, 4, 4, &[0x2000u64 << PAGE_SHIFT_X86]),
        );
        state.arm_gva_deferred_window(
            0x9100_0000,
            gva_entry(1, 4, 4, &[0x3000u64 << PAGE_SHIFT_X86]),
        );

        let declared = crate::runtime::drain::store_route_count("guest_read_declared");
        let landed = crate::runtime::drain::store_route_count("guest_read_landed");
        let dry = crate::runtime::drain::store_route_count("guest_read_dry");

        let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        assert!(!ok, "engine-less flush reports the loss");
        assert_eq!(flushed, 1, "exactly the aliased GVA window");
        assert!(!state.gva_deferred_flush.contains_key(&0x9000_0000));
        assert!(state.gva_deferred_flush.contains_key(&0x9100_0000));

        // The declaration rate is the number the demand-driven writeback design
        // turns on, and until these counters existed nothing measured it. Assert
        // them here rather than trusting the wiring: `guest_read_landed` must
        // agree with the returned count, and the dry counter must stay put on a
        // call that landed something — a route on the wrong side of that branch
        // would read as "the guest never declares" and close the question the
        // wrong way. Deltas, not absolutes: the census window is process-wide
        // and other tests share it.
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_declared") - declared,
            1,
            "every call must count as a declaration"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_landed") - landed,
            u64::from(flushed),
            "landed must count windows, not calls"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_dry") - dry,
            0,
            "a call that landed a window is not dry"
        );
    }

    /// A declaration that finds nothing armed counts as dry and lands nothing.
    ///
    /// The complement of the case above, and the one that will dominate live:
    /// the fence-bound writeback runs first and empties the windows, so most
    /// declarations arrive to an empty set. That reading is only interpretable
    /// if "dry" is known to mean *nothing was armed* rather than *the counter
    /// never fired*, which is what this pins.
    #[test]
    fn guest_read_flush_with_nothing_armed_counts_as_dry() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        let m = state.mappings.entry(9).or_default();
        m.mapped = true;
        m.page_entries = vec![(0x2000u32 << 2) | 1];

        let declared = crate::runtime::drain::store_route_count("guest_read_declared");
        let landed = crate::runtime::drain::store_route_count("guest_read_landed");
        let dry = crate::runtime::drain::store_route_count("guest_read_dry");

        let (ok, flushed) = super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        assert!(ok, "nothing armed is not a failure");
        assert_eq!(flushed, 0, "nothing armed lands nothing");
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_declared") - declared,
            1,
            "a dry call is still a declaration"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_dry") - dry,
            1,
            "nothing armed must count as dry"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_landed") - landed,
            0,
            "nothing armed lands nothing"
        );
    }

    /// A declaration is attributed by whether the fence rail has ever written
    /// back that mapping, and the two arms must be exclusive.
    ///
    /// This is the split that decides whether the eager writeback could become
    /// demand-driven, and it is the one `guest_read_dry` cannot make — the fence
    /// empties the windows before any declaration arrives, so a declaration on a
    /// surface the fence just wrote and one on an unrelated surface look
    /// identical from the dry count. A mis-wired split here would read as "the
    /// guest declares on surfaces we never write back" and close a month of work
    /// the wrong way, so both arms are driven through the same fixture.
    #[test]
    fn a_declaration_is_attributed_to_whether_the_fence_writes_that_mapping() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        for mid in [7u32, 9u32] {
            let m = state.mappings.entry(mid).or_default();
            m.mapped = true;
            m.page_entries = vec![(0x2000u32 << 2) | 1];
        }

        let other0 = crate::runtime::drain::store_route_count("guest_read_on_other_mid");
        let flushed0 = crate::runtime::drain::store_route_count("guest_read_on_flushed_mid");

        // Nothing has been written back yet: both mappings are "other".
        super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_on_other_mid") - other0,
            1,
            "a mapping the fence never wrote is not a flushed mid"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_on_flushed_mid") - flushed0,
            0,
            "nothing has been fence-flushed yet"
        );

        // Once the fence has landed a window on 9, a declaration on 9 counts as
        // covered and one on 7 still does not.
        state.fence_flushed_mappings.insert(9);
        super::flush_mapping_for_guest_read(&mut state, &mut host, 9);
        super::flush_mapping_for_guest_read(&mut state, &mut host, 7);
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_on_flushed_mid") - flushed0,
            1,
            "a declaration on a fence-written mapping must count as covered"
        );
        assert_eq!(
            crate::runtime::drain::store_route_count("guest_read_on_other_mid") - other0,
            2,
            "the arms must be exclusive — every call lands in exactly one"
        );
    }

    /// Page drift must distinguish the cases it exists to separate, and now
    /// **decide** them.
    ///
    /// A probe that reports nothing is indistinguishable from a probe that
    /// cannot fire, and this codebase has already paid for three of those. So
    /// drive both controls through the same fixture: a window whose GVA still
    /// resolves to its armed pages must stay silent and stay writable, and one
    /// whose pages moved under it must produce the line and be refused — same
    /// task, same geometry, only the armed set differs.
    ///
    /// The decision is asserted alongside the line because they are two separate
    /// claims. Logging drift while still writing is exactly what this used to
    /// do, and the guest heap corruption that allowed — WindowServer aborting in
    /// `small_free_list_remove_ptr_no_clear` — is why it decides now.
    /// The mapping-keyed rails get the same reading as the GVA rail, per rail,
    /// so a boot can say whether `map_generation` in the key is already enough
    /// to make deferral here safe — rather than the two being assumed alike.
    #[test]
    fn each_mapping_rail_is_scored_against_the_fence_under_its_own_name() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let k = key(7, 0, 256);

        let render = render_owner(1);
        let storage = crate::model::DeferredOwner::Storage {
            generation: 3,
            armed_stamp_seq: 0,
        };

        // Inside the fence: neither rail may be reported.
        let before = [
            store_route_count("rendw_stamp_outlived"),
            store_route_count("storw_stamp_outlived"),
        ];
        super::note_mapping_window_against_fence(&state, &k, &render);
        super::note_mapping_window_against_fence(&state, &k, &storage);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            before,
            "a window landed inside its own fence is the safe case on both rails"
        );

        // Past the fence: each rail reports under its own counter, so a boot can
        // tell a render-Store window from a compute-storage one.
        state.completion_stamp_seq = 5;
        super::note_mapping_window_against_fence(&state, &k, &render);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            [before[0] + 1, before[1]],
            "the render rail must not be counted under the storage rail's name"
        );
        super::note_mapping_window_against_fence(&state, &k, &storage);
        assert_eq!(
            [
                store_route_count("rendw_stamp_outlived"),
                store_route_count("storw_stamp_outlived")
            ],
            [before[0] + 1, before[1] + 1],
            "and the storage rail must not be counted under the render rail's"
        );
    }

    /// The window and the resident it pinned must be the same slot, and the two
    /// spellings that name it must agree by construction.
    ///
    /// `arm_surface_resident_store` pins `render_chain_identity`;
    /// `flush_render_one` rebuilds `render_window_identity` from
    /// `key.width`/`key.height`. Both now read color0's declared geometry —
    /// the draw request has only that one — so the geometry axis is closed by
    /// construction rather than by this check. It was not: the arm's spelling
    /// preferred a whole-request pass extent, and a record whose extent
    /// differed from its attachment produced two different
    /// `TargetIdentity::Surface` values. The arm pinned one slot, the flush
    /// looked up another, `registry_get` missed, and the frame was lost —
    /// reported as `live=Absent` — while the pin leaked, because eviction skips
    /// pinned slots by design. One measured boot lost ~135 frames at 1920x1080
    /// with `live=None`, the whole desktop compositing layer keeping pre-Store
    /// bytes in guest memory.
    ///
    /// The first two assertions hold that closure: with one geometry the two
    /// spellings are the same value, and a different extent is a different slot
    /// that must never be pinned on a window's behalf. The third is the axis
    /// still live at runtime, and the reason the equality check stays.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_window_and_the_resident_it_pins_cannot_be_named_at_two_geometries() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let k = key(7, 0, 256);
        state.map_surface(k.mapping_id);
        state
            .mappings
            .get_mut(&k.mapping_id)
            .unwrap()
            .map_generation = k.map_generation;
        let from_key = super::render_window_identity(&k);

        // The spelling the arm uses, when the pass extent equals the attachment
        // and the mapping still carries the incarnation the key was built at.
        assert_eq!(
            from_key,
            crate::runtime::present_identity::surface_identity(
                &state,
                k.mapping_id,
                k.width,
                k.height
            ),
            "with one geometry the two spellings must be the same value, or the \
             rail is broken for every window rather than only the split ones"
        );

        // And when it does not. This is the value the arm would have pinned for
        // a record whose pass extent is larger than its color0 attachment; the
        // flush cannot find it from the key, which is why the arm refuses.
        assert_ne!(
            from_key,
            crate::runtime::present_identity::surface_identity(
                &state,
                k.mapping_id,
                k.width + 1,
                k.height
            ),
            "geometry is part of the resident's shape, so a pass-extent identity \
             is a different slot and must not be pinned on a window's behalf"
        );

        // Generation is the second axis, and it is the one that can move
        // *inside* the arm. `arm_surface_resident_store` takes the identity from
        // the live mapping before it builds the key, and the step between them —
        // `prepare_surface_deferred_window` — lands intersecting windows, whose
        // writeback re-resolves the mapping and can bump `map_generation`. The
        // arm would then pin the pre-bump slot and hand the window a post-bump
        // key. Same miss, same lost frame, same leaked pin.
        crate::model::DeviceState::bump_map_generation(
            state.mappings.get_mut(&k.mapping_id).unwrap(),
        );
        assert_ne!(
            from_key,
            crate::runtime::present_identity::surface_identity(
                &state,
                k.mapping_id,
                k.width,
                k.height
            ),
            "a generation that moved during the arm names a different slot, and \
             the equality check is what stops the window being armed across it"
        );
    }

    /// A completion stamp is the guest's licence to free everything it allocated
    /// for the work being completed, so the stamp must leave nothing owed to
    /// guest RAM. Asserted through [`crate::runtime::drain::write_stamp`] itself
    /// rather than against the helper, because the claim that matters is the
    /// wiring: a helper nothing calls at the fence is the bug this fixes.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_completion_stamp_leaves_no_window_still_owing_guest_ram() {
        use crate::runtime::host::FakeHost;
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let stamp_pfn = 9u32;
        host.map_range(u64::from(stamp_pfn) << PAGE_SHIFT_X86, page as usize, 0);
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.gfx.fifo_base_page = stamp_pfn;

        state.arm_gva_deferred_window(0x1000, gva_entry(1, 4, 4, &[]));
        state.arm_gva_deferred_window(0x2000, gva_entry(1, 4, 4, &[]));
        assert_eq!(state.gva_deferred_flush.len(), 2, "two windows armed");

        crate::runtime::drain::write_stamp(&mut state, &mut host, 1, 0x55);

        assert!(
            state.gva_deferred_flush.is_empty(),
            "the guest may free every one of these targets the instant it reads \
             the stamp, so none of them may still be waiting to be written"
        );
        assert_eq!(
            state.completion_stamp_seq, 1,
            "the fence the windows are measured against must have moved"
        );
    }

    /// The guest's fence is the only thing that separates a deferred write from
    /// a write into somebody else's allocation, and the page-set guard cannot
    /// see it: free-then-reuse inside one process leaves the translation
    /// identical, so `deferred_pages_still_ours` says yes to exactly the window
    /// that corrupts the guest heap.
    ///
    /// Both directions are asserted. A census that fires on every landing is as
    /// useless as one that never fires — the whole point is that it separates
    /// the windows landed inside their own fence from the ones that outlived it.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_window_landed_after_its_fence_is_counted_apart_from_one_landed_inside_it() {
        use crate::runtime::drain::store_route_count;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);

        // Negative control: armed and landed under the same stamp. The guest
        // has not been told this render finished, so it cannot have freed the
        // target, and the write is the one the Store promised.
        let mut inside = gva_entry(1, 4, 4, &[]);
        inside.armed_stamp_seq = state.completion_stamp_seq;
        let same_before = store_route_count("gvaw_stamp_same");
        let outlived_before = store_route_count("gvaw_stamp_outlived");
        super::note_window_outlived_its_stamp(&state, 0x1000, &inside, "rearm");
        assert_eq!(
            store_route_count("gvaw_stamp_same"),
            same_before + 1,
            "a window landed inside its own fence is the safe case and must be counted as one"
        );
        assert_eq!(
            store_route_count("gvaw_stamp_outlived"),
            outlived_before,
            "a guard that fires on every landing cannot price the repair"
        );

        // Positive control: the same window, landed after the guest was fenced.
        state.completion_stamp_seq = state.completion_stamp_seq.wrapping_add(3);
        let same_before = store_route_count("gvaw_stamp_same");
        super::note_window_outlived_its_stamp(&state, 0x1000, &inside, "gva_alias");
        assert_eq!(
            store_route_count("gvaw_stamp_outlived"),
            outlived_before + 1,
            "a window whose stamp moved before it landed writes memory the guest was \
             told it could reclaim, and that is the class the page-set guard is blind to"
        );
        assert_eq!(
            store_route_count("gvaw_stamp_same"),
            same_before,
            "the two outcomes must not both be counted for one landing"
        );
    }

    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn window_page_drift_refuses_the_guest_write_and_is_silent_without_it() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let (dir_gpa, root_gpa, data0) = (2 * page, 3 * page, 4 * page);
        for gpa in [dir_gpa, root_gpa, data0] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, page, 2));

        crate::observe::redirect_logs_for_tests();
        let drift_lines = |from: usize| -> usize {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .get(from..)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("deferred_window_page_drift "))
                .count()
        };
        let mark = || {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len()
        };

        // Negative control: armed on the page the GVA resolves to right now.
        let at = mark();
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0]),
                "gva_alias",
                "guest=refused",
            ),
            "an unmoved window must stay writable — a guard that refuses every \
             flush means the guest never sees a Store"
        );
        assert_eq!(
            drift_lines(at),
            0,
            "a window that did not move must be quiet"
        );

        // Positive control: same window, armed on a page it no longer maps to.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[9 * page]),
                "gva_alias",
                "guest=refused",
            ),
            "a window whose pages moved must be refused, not merely reported"
        );
        assert_eq!(drift_lines(at), 1, "a window whose pages moved must report");

        // A window armed on TWO pages whose range now resolves ONE page, and
        // that page is not one of the two. This is the arrangement a guest
        // produces by releasing a GPU allocation and letting part of the virtual
        // range be re-pointed: fewer pages come back, and what does come back
        // belongs to somebody else.
        //
        // A guard keyed on page COUNT reads this as "shorter walk, therefore
        // teardown, therefore nothing to protect" and permits it, and the writer
        // then lands rows in `data0` — which this window never owned. Keyed on
        // membership it is refused, which is what the guest's own crash reports
        // say has to happen.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[7 * page, 8 * page]),
                "clear_store",
                "guest=refused",
            ),
            "a short walk that resolves a page the window was never armed on is \
             not a teardown — it is a write into another owner's pages"
        );
        assert_eq!(
            drift_lines(at),
            1,
            "the refusal must be visible; a silent one cannot be scored"
        );

        // The benign half of the same shape: fewer pages come back, and every
        // one of them is still ours. Refusing this would drop live Stores whose
        // destination never moved, so the guard must not simply require equal
        // sets.
        let at = mark();
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0, 8 * page]),
                "clear_store",
                "guest=refused",
            ),
            "a walk that came back short but entirely inside the armed pages is \
             the teardown case, and its rows land in this window's own memory"
        );
        assert_eq!(drift_lines(at), 0, "a subset walk must stay quiet");
    }

    /// The same window, asked by the reader instead of the writer.
    ///
    /// The cross-pass resident Load in `encode_draw_chain` trusts a GVA resident
    /// as a draw's *prior content*, gated on a deferred window existing at the
    /// address with matching geometry — conditions a different allocation
    /// reusing the address satisfies exactly. The flush path had refused that
    /// drift since it was written; the read path did not ask, which left the two
    /// sides of one window disagreeing about whether it still belonged to its
    /// name.
    ///
    /// What this pins is that the reader gets the same verdict *and its own
    /// outcome word*. A drift line is the only record either consumer leaves,
    /// and `guest=refused` on a line emitted by a Load would say guest memory
    /// was protected when what was actually refused was a stale picture.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn the_resident_load_reader_gets_the_same_drift_verdict_under_its_own_name() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let (dir_gpa, root_gpa, data0) = (2 * page, 3 * page, 4 * page);
        for gpa in [dir_gpa, root_gpa, data0] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, page, 2));

        crate::observe::redirect_logs_for_tests();
        let tail = |from: usize| -> String {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .get(from..)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("deferred_window_page_drift "))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let mark = || {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len()
        };

        // The address still names the pages the window was armed on, so the
        // resident behind it is this allocation's own prior frame and the Load
        // may take it. Refusing here would cost a seed on every chained pass.
        let at = mark();
        assert!(
            super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[data0]),
                "xpass_load",
                "resident=refused",
            ),
            "an unmoved window's resident is the draw's own prior content"
        );
        assert_eq!(
            tail(at),
            "",
            "an unmoved window must be quiet on both sides"
        );

        // The guest handed this address to a different allocation. The resident
        // still exists, still has the geometry, and still reports content_ready
        // — every gate the Load had before this check. It holds the previous
        // allocation's pixels.
        let at = mark();
        assert!(
            !super::window_pages_still_ours(
                &state,
                &host,
                0,
                &gva_entry(1, 4, 4, &[9 * page]),
                "xpass_load",
                "resident=refused",
            ),
            "a reallocated address must not load the previous owner's pixels as \
             this draw's prior content"
        );
        let line = tail(at);
        assert!(
            line.contains("trigger=xpass_load"),
            "the line must name the reader that asked: {line}"
        );
        assert!(
            line.contains("resident=refused"),
            "the reader refuses a resident, not a guest write: {line}"
        );
        assert!(
            !line.contains("guest=refused"),
            "a Load must not claim it protected guest memory: {line}"
        );
    }

    /// The linear compute-storage rail gets the same drift decision as the GVA
    /// rail, and takes its span the way the arm site does.
    ///
    /// `flush_linear_one` needs a live Vulkan engine to reach its guest write, so
    /// this exercises the decision itself with a linear key's geometry. The span
    /// argument is the subtle part and the positive control is what pins it: a
    /// linear key's `span_end` is a *length* (`row_stride * height`), not an end
    /// address, and the arm site walks `(surface_offset, span_end)` with exactly
    /// those two values. Reading `span_end` as an end address here would make the
    /// span `page - page == 0`, the walk would come back empty, the short-walk arm
    /// would permit, and the positive control below would fail — which is the
    /// point of siting it at a nonzero offset rather than at GVA 0, where both
    /// readings coincide.
    #[cfg(feature = "backend-vulkan")]
    #[test]
    fn a_linear_window_whose_pages_moved_is_refused_and_reads_its_span_as_a_length() {
        use crate::contract::endian::st32;
        use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
        use crate::runtime::host::{FakeHost, HostMemory};
        let page = 1u64 << PAGE_SHIFT_X86;
        let mut host = FakeHost::new();
        let (dir_gpa, root_gpa, data0, data1) = (2 * page, 3 * page, 4 * page, 5 * page);
        for gpa in [dir_gpa, root_gpa, data0, data1] {
            host.map_range(gpa, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        // Two PTEs: GVA page 0 → data0, GVA page 1 → data1.
        let mut ptes = [0u8; 8];
        st32(&mut ptes[0..], 4);
        st32(&mut ptes[4..], 5);
        host.write_gpa(root_gpa, &ptes).unwrap();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        assert!(state.define_task(1, 8 * page, 2));

        crate::observe::redirect_logs_for_tests();
        let drift_lines = |from: usize| -> usize {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .get(from..)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.starts_with("deferred_window_page_drift "))
                .count()
        };
        let mark = || {
            std::fs::read_to_string(crate::observe::fail_log_path())
                .unwrap_or_default()
                .len()
        };
        // One page long, sited at GVA `page` so a length/end confusion is visible.
        let (offset, span) = (page, page);

        // Negative control: armed on the page GVA `page` resolves to right now.
        let at = mark();
        assert!(
            super::deferred_pages_still_ours(
                &state,
                &host,
                1,
                offset,
                span,
                &[data1].into_iter().collect(),
                "8x8 trigger=linear_flush ref=5",
                "guest=refused",
            ),
            "an unmoved linear window must stay writable — a guard that refuses \
             every flush means the guest never sees a compute Store"
        );
        assert_eq!(
            drift_lines(at),
            0,
            "a linear window that did not move must be quiet"
        );

        // Positive control: same window, armed on a page it no longer maps to.
        // This is also the assertion that the span is read as a length.
        let at = mark();
        assert!(
            !super::deferred_pages_still_ours(
                &state,
                &host,
                1,
                offset,
                span,
                &[9 * page].into_iter().collect(),
                "8x8 trigger=linear_flush ref=5",
                "guest=refused",
            ),
            "a linear window whose pages moved must be refused — and a zero-length \
             walk from misreading span_end as an end address would permit it"
        );
        assert_eq!(
            drift_lines(at),
            1,
            "a linear window whose pages moved must report"
        );

        // A walk that resolves NOTHING also returns "still ours", because
        // `all` over an empty set is true — and that is not the guard agreeing,
        // it is the guard having nothing to compare. Counted apart, or this
        // rail's "no drift" reads as a verification it never made.
        use crate::runtime::drain::store_route_count;
        let verified_before = store_route_count("defw_pages_verified");
        let unwit_before = store_route_count("defw_unwit_no_live");
        let at = mark();
        assert!(
            super::deferred_pages_still_ours(
                &state,
                &host,
                1,
                // Page index 3: inside the root page, but its PTE was never
                // written, so it is zero and translates to nothing. An index
                // past the root page's own extent would instead read whatever
                // GPA follows it, which is a different (and resolvable) thing.
                3 * page,
                span,
                &[data1].into_iter().collect(),
                "8x8 trigger=linear_flush ref=5",
                "guest=refused",
            ),
            "an unresolvable window is not drift — no row can land through it"
        );
        assert_eq!(drift_lines(at), 0, "and it is not reported as drift");
        assert_eq!(
            store_route_count("defw_unwit_no_live"),
            unwit_before + 1,
            "it must be counted as unwitnessed"
        );
        assert_eq!(
            store_route_count("defw_pages_verified"),
            verified_before,
            "and must NOT be counted as a verification"
        );

        // An empty armed set is the other unchecked exit, and it is its own slug
        // because it is a different gap.
        let no_armed_before = store_route_count("defw_unwit_no_armed");
        assert!(super::deferred_pages_still_ours(
            &state,
            &host,
            1,
            offset,
            span,
            &std::collections::HashSet::new(),
            "8x8 trigger=linear_flush ref=5",
            "guest=refused",
        ));
        assert_eq!(
            store_route_count("defw_unwit_no_armed"),
            no_armed_before + 1
        );
    }

    /// Task teardown moves the task's GVA windows to the retired list (model)
    /// and the runtime lands them cache-only — obligations never write guest
    /// pages from teardown and never linger.
    #[test]
    fn task_delete_retires_gva_windows_cache_only() {
        use crate::runtime::host::FakeHost;
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut host = FakeHost::new();
        assert!(state.define_task(6, 0x1000, 2));
        state.arm_gva_deferred_window(0x9000_0000, gva_entry(6, 4, 4, &[]));
        state.arm_gva_deferred_window(0x9100_0000, gva_entry(7, 4, 4, &[]));
        assert!(state.delete_task(6));
        assert!(
            !state.gva_deferred_flush.contains_key(&0x9000_0000),
            "dead task's window must leave the armed map"
        );
        assert!(
            state.gva_deferred_flush.contains_key(&0x9100_0000),
            "other task's window must stay armed"
        );
        assert_eq!(state.retired_gva_windows.len(), 1);
        super::retire_gva_windows(&mut state, &mut host);
        assert!(state.retired_gva_windows.is_empty());
    }

    /// The window cap lands the oldest-armed window first.
    #[test]
    fn oldest_gva_window_is_taken_first() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut newer = gva_entry(1, 4, 4, &[]);
        newer.armed_seq = 9;
        let mut older = gva_entry(1, 4, 4, &[]);
        older.armed_seq = 3;
        state.arm_gva_deferred_window(0x1000, newer);
        state.arm_gva_deferred_window(0x2000, older);
        let (gva, entry) = state.take_oldest_gva_deferred_window().unwrap();
        assert_eq!(gva, 0x2000);
        assert_eq!(entry.armed_seq, 3);
        assert_eq!(state.gva_deferred_flush.len(), 1);
    }

    #[test]
    fn take_deferred_windows_is_exact_intersection() {
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.compute_deferred_flush.insert(
            key(7, 0, 256),
            crate::model::DeferredOwner::Storage {
                generation: 3,
                armed_stamp_seq: 0,
            },
        );
        state.compute_deferred_flush.insert(
            key(7, 256, 512),
            crate::model::DeferredOwner::Storage {
                generation: 4,
                armed_stamp_seq: 0,
            },
        );
        state.compute_deferred_flush.insert(
            key(8, 0, 256),
            crate::model::DeferredOwner::Storage {
                generation: 5,
                armed_stamp_seq: 0,
            },
        );

        // Disjoint range takes nothing.
        assert!(state.take_deferred_flush_windows(7, 512, 1024).is_empty());
        assert_eq!(state.compute_deferred_flush.len(), 3);

        // Intersecting range takes only the touching window on that mapping.
        let taken = state.take_deferred_flush_windows(7, 200, 257);
        assert_eq!(taken.len(), 2, "both mapping-7 windows intersect [200,257)");
        assert_eq!(state.compute_deferred_flush.len(), 1);
        assert!(state.compute_deferred_flush.contains_key(&key(8, 0, 256)));
    }
}
