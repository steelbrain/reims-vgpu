//! Owe a type-11 surface's guest pages a frame, and pay only when something
//! reads them.
//!
//! # Why an owed frame is not a deferred plan
//!
//! [`crate::runtime::render_writeback`]'s doc carries the measurement this rail
//! exists for and the wreckage of the shape that must not be built again. The
//! short form of both:
//!
//! * A driven macos-13 sustained-animation boot issues **~840 type-11 surface
//!   Stores a second** and **~850 GVA Stores**, each copying a whole frame into
//!   guest pages. Over the same second, everything that *reads* those pages
//!   totals **six** — five colour LOAD seeds and one host-console paint. Every
//!   frame written between two reads is replaced before anything looks at it.
//! * The rail that tried to collect that before parked a **resolved plan** —
//!   host pointers walked at the Store — and landed it later. Four driven boots,
//!   four guest kernel panics, all of them page-table corruption: the guest
//!   recycles a surface's backing inside the park window and the land wrote a
//!   full frame into whatever now owned those pages.
//!
//! So this module holds **no** resolved memory. A debt is four integers naming
//! the mapping, its geometry and the `map_generation` the Store was taken under.
//! Paying it re-derives the identity from [`DeviceState`] *at that moment*, walks
//! the page tables *at that moment*, and calls the ordinary
//! [`crate::runtime::render_writeback::store_render_frame`]. A surface whose
//! backing the guest recycled either fails the generation check and is dropped,
//! or resolves to the pages it now owns — which is what a Store landing at that
//! moment would have written anyway. There is nothing stale to hold because
//! nothing is held.
//!
//! # What keeps the pixels alive
//!
//! An unpaid debt says the frame exists only in the engine's resident image, and
//! the engine already has that concept: `ResidentTargetSlot::gpu_only_content`
//! is what the reclaim paths skip. The eager Store clears it through
//! `note_resident_content_copied_out` as soon as the copy is recorded, because
//! the guest's pages then hold the pixels too. Arming a debt is exactly the case
//! where that is *not* true yet, so the arm does not clear it and the flag is
//! load-bearing rather than incidental — no separate pin, and no pin to leak.
//!
//! # The reader set is the settle set, and that is not a coincidence
//!
//! A host-side reader of guest bytes this device wrote is already obliged to
//! call [`crate::runtime::render_writeback::settle_guest_writes`] first, or it
//! reads the pre-Store bytes. That obligation predates this rail and its call
//! sites are enumerated by [`crate::runtime::render_writeback::SettleSite`], so
//! the set of places that must pay a debt is the set of places that already
//! settle — with two amendments:
//!
//! * The three completion-stamp sites must **not** pay. A stamp says a
//!   submission finished; what says the guest may read a resource's bytes is the
//!   host-valid flag the guest itself owns and the synchronize it issues before a
//!   CPU read. Paying at the stamp is what made the old deferred window's
//!   coalescing structurally unreachable, and the contract does not ask for it.
//! * A reader that names a `mapping_id` pays only that mapping's debt
//!   ([`pay_for_mapping`]); one that cannot name a mapping — a GVA span, a
//!   buffer read, an aliasing walk — pays everything ([`pay_all`]). Aliasing
//!   across the id namespaces is real rather than theoretical (see
//!   [`crate::runtime::host_writes`]), so "cannot name one" resolves to "owes all
//!   of them" and never to "owes none".
//!
//! Missing a reader costs a **stale frame**, not a corrupted guest: the reader
//! sees the previous Store's pixels. That is a visible defect and the A/B
//! harness photographs both arms for exactly this reason.
//!
//! # The guest's own write is answered at the payment, not at the claim
//!
//! `clear_host_valid` is the guest saying it wrote the resource's bytes itself,
//! and an owed frame older than that write must be abandoned rather than landed
//! on top of the guest's work. Nothing in [`crate::runtime::resource_validity`]
//! is hooked to do it: the arm publishes through
//! `DeviceState::note_surface_content_published`, which stamps
//! `ResourceValidity::host_published_seq`, and the guest's claim stamps
//! `host_cleared_seq` as it already did. Which happened last is then
//! [`crate::runtime::resource_validity::licence_of`]'s existing answer, read once
//! at the payment.
//!
//! Deciding it there rather than at the claim is what keeps one exit path. A
//! payment holds the target identity — it has just re-derived it — so the arm
//! that abandons a frame can also hand the resident back to the reclaim, which a
//! hook on the claim could not: `clear_host_valid` fires ~1 600 times a second
//! and knows only a mapping id.
//!
//! # The ledger is bounded by a type, not by a sweep
//!
//! [`PendingWritebacks`] is the only container, its insert is the only way in,
//! and it holds at most [`crate::runtime::writeback_debt::MAX_DEBTS`] entries: a
//! mapping arming past that limit
//! pays the oldest debt on the spot rather than growing the map. Six distinct
//! surfaces carry a driven boot, so the bound is ~5x the observed working set
//! and a boot that reaches it is reporting something about the guest.

use crate::model::DeviceState;
use crate::runtime::host::{HostMemory, HostOps};

/// Debts held at once, before an arm pays the oldest to make room.
///
/// A driven macos-13 sustained-animation boot names about six distinct type-11
/// surfaces (`render_writeback`'s second census), so this is a ceiling with
/// headroom rather than a working figure. `wbdebt_evicted` firing is the boot
/// saying the guest's surface set outgrew it.
pub const MAX_DEBTS: usize = 32;

/// A frame owed to one type-11 mapping's guest pages.
///
/// Deliberately four integers and no memory. See the module doc: the rail this
/// replaces held resolved host pointers and corrupted the guest's page tables
/// with them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WritebackDebt {
    /// Geometry the Store was taken at, and the geometry the payment writes.
    pub width: u32,
    pub height: u32,
    /// `MappingEntry::map_generation` at the arm.
    ///
    /// The identity the payment re-derives carries this, so a mapping the guest
    /// has since remapped produces a different identity — a different resident —
    /// and the debt is void rather than paid into the wrong pages.
    pub map_generation: u32,
    /// Arm order, for choosing which debt an over-full ledger pays first.
    pub seq: u64,
}

/// Every type-11 mapping owed a frame, keyed by mapping id.
///
/// One debt per mapping by construction: a second Store into a mapping replaces
/// the first, which is the coalescing the whole rail exists for and the reason
/// the key is the mapping rather than the Store.
#[derive(Debug, Default)]
pub struct PendingWritebacks {
    debts: std::collections::BTreeMap<u32, WritebackDebt>,
    next_seq: u64,
}

impl PendingWritebacks {
    /// Mappings currently owed a frame.
    pub fn len(&self) -> usize {
        self.debts.len()
    }

    /// Whether anything is owed at all — the check every reader makes, and the
    /// one that has to be free.
    pub fn is_empty(&self) -> bool {
        self.debts.is_empty()
    }

    /// What `mapping_id` is owed, if anything.
    pub fn get(&self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.get(&mapping_id).copied()
    }

    /// Take `mapping_id`'s debt, leaving it owed nothing.
    pub fn take(&mut self, mapping_id: u32) -> Option<WritebackDebt> {
        self.debts.remove(&mapping_id)
    }

    /// Every mapping owed a frame, oldest arm first.
    pub fn mappings_by_age(&self) -> Vec<u32> {
        let mut all: Vec<(u64, u32)> = self.debts.iter().map(|(id, d)| (d.seq, *id)).collect();
        all.sort_unstable();
        all.into_iter().map(|(_, id)| id).collect()
    }

    /// The mapping whose debt has been owed longest.
    fn oldest(&self) -> Option<u32> {
        self.debts
            .iter()
            .min_by_key(|(_, d)| d.seq)
            .map(|(id, _)| *id)
    }

    /// Record that `mapping_id` is owed a frame, returning the mapping whose
    /// debt the caller must pay to bring the ledger back under [`MAX_DEBTS`] —
    /// `None` in the ordinary case.
    ///
    /// A mapping already owed a frame is *replaced*: the later frame is the
    /// fresher answer and the earlier one has been superseded on the GPU
    /// already. That replacement is the whole saving, so it is counted.
    ///
    /// The over-full entry is left in the ledger and handed back by name rather
    /// than removed here, so [`PendingWritebacks::take`] stays the only way a
    /// debt leaves — a removal that is not a payment is a frame the guest asked
    /// for and never received.
    #[must_use = "an evicted mapping still owes a frame and the caller must pay it"]
    pub fn arm(
        &mut self,
        mapping_id: u32,
        width: u32,
        height: u32,
        map_generation: u32,
    ) -> Option<u32> {
        let evict = match self.debts.len() >= MAX_DEBTS && !self.debts.contains_key(&mapping_id) {
            true => self.oldest(),
            false => None,
        };
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        let previous = self.debts.insert(
            mapping_id,
            WritebackDebt {
                width,
                height,
                map_generation,
                seq,
            },
        );
        if previous.is_some() {
            crate::runtime::drain::note_store_route("wbdebt_superseded");
        }
        crate::runtime::drain::note_store_route("wbdebt_armed");
        evict
    }
}

/// Whether the lazy rail is on for this process.
///
/// Read once. The rail changes *when* a frame reaches guest pages, not what the
/// bytes are, so a boot that flipped it midway would be two devices in one log.
pub fn lazy_writeback_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let (state, value) = crate::env::read(crate::env::LAZY_WRITEBACK);
        // Only an explicit `off` narrows to the eager Store. Unset, `on` and an
        // unrecognized value are all the shipping rail, which is what makes
        // `Switch::Unrecognized` — an operator's typo — fail toward the measured
        // default rather than silently selecting the arm it is 45 % slower on.
        let on = !matches!(state, crate::env::Switch::Off);
        crate::observe::off(format!(
            "lazy_writeback on={on} switch={state:?} value={}",
            value.unwrap_or_else(|| "<unset>".into())
        ));
        on
    })
}

/// Pay `mapping_id`'s owed frame, if it owes one.
///
/// The one call a reader of a named mapping's guest bytes makes before it reads
/// them. Free when nothing is owed — one `BTreeMap` emptiness check, which is
/// the answer on nearly every call.
pub fn pay_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let Some(debt) = state.pending_writebacks.take(mapping_id) else {
        return;
    };
    pay(state, host, mapping_id, debt, "wbdebt_paid_named");
}

/// Pay every owed frame.
///
/// For a reader that cannot name the mapping it is about to read — a GVA span, a
/// buffer, a page walk that may alias a surface. Aliasing across the id
/// namespaces is real, so "cannot say" resolves to "owes all of them".
///
/// # Why the disjointness closures those readers already carry do not narrow it
///
/// The three GVA readers each build the exact page list they are about to touch,
/// and hand it to
/// [`crate::runtime::render_writeback::settle_guest_writes_unless_disjoint`] so a
/// reader somewhere else entirely does not wait for a surface's writeback. That
/// narrowing cannot be reused here, and the reason is the rail itself: the test
/// runs only when a copy is **outstanding**, and an owed frame has not been
/// submitted at all. With the lazy rail on, the common state is a clear debt
/// flag and a full ledger, where the closure never runs.
///
/// Narrowing this would need a page-extent hint held per debt, and a hint is the
/// beginning of holding resolved memory — which is what the module doc says this
/// rail must not do. [`note_unnamed_reach`] is the instrument that says whether
/// it would be worth it; read its doc before building one.
pub fn pay_all<M: HostMemory + HostOps>(state: &mut DeviceState, host: &mut M) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    for mapping_id in state.pending_writebacks.mappings_by_age() {
        let Some(debt) = state.pending_writebacks.take(mapping_id) else {
            continue;
        };
        pay(state, host, mapping_id, debt, "wbdebt_paid_all");
    }
}

/// One call in [`REACH_SAMPLE`] does the walk; the rest cost one modulo.
///
/// The walk is ~2 000 page-table descents for a 1080p span and the site that
/// dominates [`pay_all`] runs about 1 700 times a second, so measuring every call
/// would cost more than the rail saves and would be measuring the instrument.
/// A census wants a rate and not a total, and a rate converges on a 1-in-64
/// sample of a population this size: ~26 walks a second against ~1 700 calls.
const REACH_SAMPLE: u64 = 64;

/// Calls to [`note_unnamed_reach`], for the sample.
static REACH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Does an unnameable reader that pays every debt actually read the pages it is
/// paying for?
///
/// # The question, and why it decides the rail's ceiling
///
/// The premise the lazy rail was built on read the settle census wrong. Those
/// counters count settles that **waited**, and on a driven macos-13
/// sustained-animation boot they total six a second — which is what "840 writes
/// consumed six times" was derived from. The *calls* are a different population:
/// `draw::vulkan::load_linear_guest_memoized` alone reaches its settle about
/// 1 700 times a second, reads the guest pages every one of them, and cannot name
/// a mapping, so it pays every owed frame. That is why the first driven on-arm
/// boot coalesced 130 Stores of 577 rather than the ~95 % the premise predicted.
///
/// But paying is only *owed* where the read and the surface share pages. A
/// compositor sampling a glyph atlas while three windows owe frames pays three
/// copies it will not look at. This counts which it is:
///
/// * `wbdebt_reach_overlap` — the sampled read touched a page some debt's
///   mapping holds. The payment was owed and no narrowing can remove it.
/// * `wbdebt_reach_disjoint` — it did not. The payment was pure waste, and the
///   ratio of these two is the prize a page-extent hint per debt would collect.
/// * `wbdebt_reach_unnamed` — the reader's own walk came back short, so nothing
///   could be ruled out. A narrowing must treat this as overlap.
///
/// `pages` is the reader's own closure, the same one it hands the disjointness
/// test, so both ends of the comparison stay one rule. It runs only on a sampled
/// call and only while something is owed.
pub fn note_unnamed_reach(state: &DeviceState, pages: impl FnOnce() -> Option<Vec<u64>>) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    let n = REACH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !n.is_multiple_of(REACH_SAMPLE) {
        return;
    }
    let Some(read) = pages() else {
        crate::runtime::drain::note_store_route("wbdebt_reach_unnamed");
        return;
    };
    let read: std::collections::BTreeSet<u64> = read.into_iter().collect();
    let overlap = state
        .pending_writebacks
        .mappings_by_age()
        .into_iter()
        .any(|mapping_id| {
            state
                .mapping_reach_pages(mapping_id)
                .is_some_and(|owed| owed.iter().any(|page| read.contains(page)))
        });
    match overlap {
        true => crate::runtime::drain::note_store_route("wbdebt_reach_overlap"),
        false => crate::runtime::drain::note_store_route("wbdebt_reach_disjoint"),
    }
}

/// Pay whatever a *texture* reference names, for a reader that reaches guest
/// bytes through a task GVA but knows which resource it is reading.
///
/// # Why a GVA reader is nameable after all, and what that measured
///
/// The three linear readers walk raw task GVAs, so the first cut of this rail
/// had them pay every owed frame. [`note_unnamed_reach`] priced that: **173
/// sampled payments over one driven macos-13 sustained-animation boot, 173
/// disjoint from every owed surface and not one overlap**, at a 1-in-64 sample
/// of ~11 000 payments. Meanwhile `wbdebt_paid_all` was 20 391 against
/// `wbdebt_paid_named` 755, so 96 % of all payments were the ones that read
/// nothing they paid for, and they cost `sampled_us` 1.64 → 8.49 us a chain.
///
/// They are nameable because the guest names them. A debt is keyed by mapping
/// id, and this device holds two ways from a texture reference to one:
/// `DeviceState::texture_to_mapping` for the per-task registration, and the id
/// itself where the guest uses one namespace for both —
/// [`crate::runtime::resource_validity::apply`] resolves a validity statement
/// through exactly this pair, and this is the same question asked of the same
/// two tables.
///
/// A reference that resolves to neither names no mapping this device holds, so
/// no debt can be about it. That is a statement about the registries and not
/// about a workload — but it is not a statement about raw *page* aliasing, where
/// a surface's pages are re-used as some other resource's backing with no
/// mapping entry. [`note_unnamed_reach`] stays wired at these sites as the
/// standing alarm for exactly that: it samples the read's own page walk against
/// every owed surface's pages, and `wbdebt_reach_overlap` above zero is a
/// payment this naming skipped and should not have.
pub fn pay_for_texture<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
) {
    if state.pending_writebacks.is_empty() {
        return;
    }
    // Both spellings, in the order `resource_validity::apply` uses: a reference
    // that is itself a mapping id, and the per-task registration. Paying one
    // leaves the ledger holding the other, so asking twice costs a map lookup
    // and cannot pay the wrong surface.
    let mapped = state
        .texture_to_mapping
        .get(&(task_id, texture_ref))
        .copied();
    let mut named = false;
    if state.pending_writebacks.get(texture_ref).is_some() {
        named = true;
        pay_for_mapping(state, host, texture_ref);
    }
    if let Some(mapping_id) = mapped.filter(|&id| id != texture_ref) {
        if state.pending_writebacks.get(mapping_id).is_some() {
            named = true;
            pay_for_mapping(state, host, mapping_id);
        }
    }
    if !named {
        crate::runtime::drain::note_store_route("wbdebt_texture_owes_nothing");
    }
}

/// Pay `mapping_id`'s owed frame and then wait for every guest-page write this
/// device has submitted — the whole obligation of a host-side reader or writer
/// of one named mapping's bytes, in one call.
///
/// The two halves are one obligation and are spelled as one function so a new
/// site cannot discharge half of it. A site that settles without paying reads
/// the frame *before* the one the guest's own driver last asked for; a site that
/// pays without settling reads the frame it just submitted and has not waited
/// for. Both are stale pixels and neither shows up as a refusal.
///
/// Free when nothing is owed and nothing is outstanding, which is the answer on
/// nearly every call: one emptiness check and one relaxed atomic load.
pub fn settle_for_mapping<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_mapping(state, host, mapping_id);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// [`settle_for_mapping`] for a caller that cannot name the mapping it is about
/// to touch, so it owes every debt.
pub fn settle_unnamed<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_all(state, host);
    crate::runtime::render_writeback::settle_guest_writes(site);
}

/// [`settle_for_mapping`] over
/// [`crate::runtime::render_writeback::settle_guest_writes_unless_disjoint`].
///
/// The disjointness test narrows only the *wait*, never the payment: an owed
/// frame has not been submitted, so there is nothing outstanding for the test to
/// find disjoint from and a debt left unpaid here would be read straight past.
///
/// The page set is walked here rather than taken as a closure, and both of those
/// are deliberate. Walked *here* because the payment needs `state` mutably and
/// the disjointness test needs it shared, so a caller-supplied closure cannot
/// hold `state` across both. `DeviceState::mapping_reach_pages` because that is
/// the same function the writeback builds its own destination from, so the two
/// ends of the comparison are one rule rather than two spellings of it. It stays
/// lazy: `settle_guest_writes_unless_disjoint` runs the closure only when
/// something is outstanding.
pub fn settle_for_mapping_unless_disjoint<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    site: crate::runtime::render_writeback::SettleSite,
) {
    pay_for_mapping(state, host, mapping_id);
    let s = &*state;
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(site, || {
        s.mapping_reach_pages(mapping_id)
    });
}

/// Count a reader that reaches guest bytes while a frame is owed and cannot pay
/// it, because it holds `DeviceState` immutably.
///
/// There is one — `draw::read_buffer_bytes_resolved`, the CPU
/// read of a *buffer's* guest bytes. A buffer and a type-11 render surface are
/// separate guest allocations, so this fires only where the two alias, and
/// aliasing across id namespaces is real rather than theoretical (see
/// [`crate::runtime::host_writes`]). The gap is therefore counted rather than
/// argued away: a boot reading `wbdebt_unpaid_buffer_read` above zero is a boot
/// where a buffer read *may* have seen a superseded surface frame, and that is
/// the signal to thread `&mut DeviceState` down to it.
///
/// A driven macos-13 sustained-animation boot puts `settle_buffer_guest_read` at
/// zero, which is the same call site counted for the waits it took.
pub fn note_unpaid_buffer_read(state: &DeviceState) {
    if !state.pending_writebacks.is_empty() {
        crate::runtime::drain::note_store_route("wbdebt_unpaid_buffer_read");
    }
}

/// Run the Store the debt stands for, now.
///
/// Everything the copy needs is resolved here and not at the arm — the identity
/// from the mapping's *current* generation, the page walk inside
/// `store_render_frame`. Two answers other than writing, and both release the
/// resident's `gpu_only_content` where they can, because that flag is what keeps
/// the reclaim off an image holding pixels nothing else has:
///
/// * **The guest superseded the frame.** `clear_host_valid` after the arm means
///   the guest wrote these pages itself, and landing an older frame on top of
///   its work is the write-ordering hazard `render_writeback`'s doc names
///   fourth. [`crate::runtime::resource_validity::licence_of`] is the existing
///   happens-before and it is read rather than re-derived.
/// * **The mapping's generation moved.** The guest remapped the surface, so the
///   identity this debt was armed under names a resident that is now an orphan,
///   and the pages it would be written into belong to something else. There is
///   no way to name that orphan from here — the current generation resolves to a
///   different identity — so its `gpu_only_content` outlives it and one image
///   leaks per occurrence. `wbdebt_generation_moved` is how a boot says how many;
///   a reading above single digits is the signal to carry the arm's whole
///   identity rather than the four integers that re-derive it.
#[cfg(feature = "backend-vulkan")]
fn pay<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
    debt: WritebackDebt,
    route: &'static str,
) {
    let Some(entry) = state.mappings.get(&mapping_id) else {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    };
    let (map_generation, validity) = (entry.map_generation, entry.validity);
    if map_generation != debt.map_generation {
        crate::runtime::drain::note_store_route("wbdebt_generation_moved");
        return;
    }
    let identity = crate::runtime::present_identity::surface_identity(
        state,
        mapping_id,
        debt.width,
        debt.height,
    );
    if crate::runtime::resource_validity::licence_of(validity)
        == crate::runtime::resource_validity::WritebackLicence::Superseded
    {
        crate::runtime::drain::note_store_route("wbdebt_abandoned_guest_wrote");
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
        return;
    }
    crate::runtime::drain::note_store_route(route);
    if !crate::runtime::render_writeback::store_render_frame(
        state,
        host,
        mapping_id,
        &identity,
        debt.width,
        debt.height,
    ) {
        // `store_render_frame` reports its own loss on the failure channel; this
        // names the rail that owed it, because a debt paid late and refused is a
        // different investigation from a Store refused where it was issued.
        crate::observe::fail(format!(
            "wbdebt_pay_lost mapping={mapping_id} {}x{} reason=store_refused",
            debt.width, debt.height
        ));
        crate::backend::vulkan::engine::note_resident_content_copied_out(&identity);
    }
    crate::runtime::mapper::stamp_guest_write_gen(state, host, mapping_id);
}

/// [`pay`] on an arm with no Vulkan engine to owe a frame to.
///
/// Unreachable rather than merely unused: the only arm site is the type-11
/// surface Store in `draw::vulkan`, so the ledger is empty on this arm and both
/// callers return at their emptiness check before reaching here. It exists so
/// the reader-side helpers can be one set of functions on both arms instead of
/// two spellings the settle sites would have to choose between.
#[cfg(not(feature = "backend-vulkan"))]
fn pay<M: HostMemory + HostOps>(
    _state: &mut DeviceState,
    _host: &mut M,
    _mapping_id: u32,
    _debt: WritebackDebt,
    _route: &'static str,
) {
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The coalescing the rail exists for, at the container: a second arm into
    /// one mapping replaces the first rather than queueing beside it, so N
    /// Stores between two reads cost one copy and not N.
    #[test]
    fn a_second_arm_into_one_mapping_replaces_the_first() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(7, 1920, 1080, 3), None);
        assert_eq!(pending.arm(7, 1920, 1080, 3), None);
        assert_eq!(pending.len(), 1, "one mapping owes one frame");
        let debt = pending.take(7).expect("mapping 7 owes a frame");
        assert_eq!(debt.seq, 1, "the later Store is the one owed");
        assert!(pending.is_empty());
    }

    /// Geometry travels with the debt, because the payment writes at the
    /// geometry the Store was taken at and the mapping may have been re-declared
    /// since.
    #[test]
    fn a_debt_carries_the_geometry_its_store_was_taken_at() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(4, 800, 600, 11), None);
        let debt = pending.get(4).expect("mapping 4 owes a frame");
        assert_eq!(
            (debt.width, debt.height, debt.map_generation),
            (800, 600, 11)
        );
    }

    /// The bound is the container's, and it hands the caller the mapping that
    /// has to be paid rather than dropping a frame to stay under it.
    #[test]
    fn arming_past_the_bound_evicts_the_oldest_and_says_so() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(pending.arm(id, 64, 64, 1), None, "under the bound");
        }
        assert_eq!(pending.len(), MAX_DEBTS);
        let evicted = pending.arm(MAX_DEBTS as u32, 64, 64, 1);
        assert_eq!(evicted, Some(0), "the oldest arm is the one handed back");
        assert_eq!(
            pending.len(),
            MAX_DEBTS + 1,
            "the named debt is still owed until the caller pays it"
        );
        assert!(
            pending.take(0).is_some(),
            "and paying it is what brings the ledger back under the bound"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Re-arming a mapping already at the head of a full ledger must not evict:
    /// it is a replacement and the entry count does not grow.
    #[test]
    fn re_arming_a_held_mapping_never_evicts() {
        let mut pending = PendingWritebacks::default();
        for id in 0..MAX_DEBTS as u32 {
            assert_eq!(pending.arm(id, 64, 64, 1), None);
        }
        assert_eq!(
            pending.arm(0, 64, 64, 1),
            None,
            "a replacement makes no room"
        );
        assert_eq!(pending.len(), MAX_DEBTS);
    }

    /// Age order is arm order and not mapping id, because `pay_all` walks it and
    /// the oldest owed frame is the one most likely to be read.
    #[test]
    fn mappings_come_back_in_arm_order() {
        let mut pending = PendingWritebacks::default();
        assert_eq!(pending.arm(9, 1, 1, 1), None);
        assert_eq!(pending.arm(2, 1, 1, 1), None);
        assert_eq!(pending.arm(5, 1, 1, 1), None);
        assert_eq!(pending.mappings_by_age(), vec![9, 2, 5]);
    }
}
