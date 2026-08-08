//! Read task GPU-virtual addresses via the task page directory.
//!
//! The device's adapter over `reims_vgpu_paging`: the walk, the span cutting
//! and the geometry table live there, and what this module adds is the three
//! things that crate structurally cannot see — the device's [`TaskTable`], its
//! [`HostMemory`] (as the [`HostPhys`] seam), and the mapping of the walk's
//! typed refusals onto [`MemError`] and the failure channel.
//!
//! Geometry always requires an explicit create-time page_shift (12 = x86_64,
//! 14 = arm64e). There is no arm-default overload — callers must choose.

use crate::model::{TaskEntry, TaskTable};
use crate::runtime::host::{HostMemory, MemError};
use reims_vgpu_paging::resolve::{
    geometry_for_page_shift, read_task_root, resolve_status_name, translate_root, ResolveStatus,
    Task,
};
use reims_vgpu_paging::span::{visit_span_chunks, walk_span, SpanRefusal};
use reims_vgpu_wire::mem::GuestMemory;

/// A span refusal as this device's memory error.
///
/// One spelling, because every rail that reads or writes bytes across a span
/// ends here and each would otherwise decide for itself which refusals mean
/// "the directory did not read", which mean "the walk refused", and which mean
/// "that address does not translate".
///
/// A [`SpanRefusal::Page`] is the last of those: the walk ran and the guest's
/// own table had no mapping, so the status is about the address. A
/// [`SpanRefusal::Setup`] is one of the first two, and which one is the whole
/// content of that arm — `ErrZeroRootPfn` and `ErrZeroDepth` are the walk's
/// answer about a directory it *could* read, and everything else there is a
/// failure to get as far as the directory at all.
pub(crate) fn span_refusal_error(refusal: SpanRefusal) -> MemError {
    match refusal {
        SpanRefusal::Setup(
            status @ (ResolveStatus::ErrZeroRootPfn | ResolveStatus::ErrZeroDepth),
        ) => MemError::Unresolved(status),
        SpanRefusal::Setup(_) => MemError::TaskRootRead,
        SpanRefusal::Page(status) => MemError::Unresolved(status),
    }
}

/// [`HostMemory`]'s guest-physical reads as the wire crate's guest-memory
/// seam. One address space — guest-physical — per that trait's hard rule.
///
/// The one spelling in the crate. There were two, and the second was declared
/// inside a function body in `gva_view`, which is how it stayed invisible: a
/// reader of either site saw a complete four-line adapter and no reason to look
/// for another. They agreed, but nothing made them — and the seam they
/// implement is the one place where "which address space is this" is decided,
/// so a copy that grew a second method or read a different accessor would put
/// two answers in the crate with no diff to catch it.
pub(crate) struct HostPhys<'a, M: HostMemory>(pub &'a M);

impl<M: HostMemory> GuestMemory for HostPhys<'_, M> {
    fn read_at(&self, gpa: u64, dst: &mut [u8]) -> bool {
        self.0.read_gpa(gpa, dst).is_ok()
    }
}

/// Translate `gva` under `task` and copy `buf.len()` bytes into `buf`.
///
/// `page_shift` must be the device create-time guest page shift (12 or 14).
pub fn read_task_gva<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active || task.directory_pfn == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // Streams rather than collecting the chunks first: this sits one level
    // below per-row blit loops, and a read that resolves to a single page —
    // which most of them do — would otherwise allocate a one-element Vec per
    // row. The write path cannot do the same, because the walk holds the host
    // shared and the write needs it exclusively.
    let mut result: Result<(), MemError> = Ok(());
    visit_span_chunks(&reader, geom, &gr_task, gva, buf.len(), &mut |chunk| {
        match host.read_gpa(chunk.gpa, &mut buf[chunk.range()]) {
            Ok(()) => true,
            Err(e) => {
                // The host's own error, not a walk status: the address resolved
                // and the transaction is what failed, and which transaction it
                // was is the finding.
                result = Err(e);
                false
            }
        }
    })
    .map_err(span_refusal_error)?;
    result
}

/// Read `[gva, gva+len)` under the task the guest named. **That task, or an
/// error.**
///
/// This used to fall back to walking `task_id >> 1`'s page table at the same
/// address, and it was the last of the three `>> 1` arms this crate improvised.
/// The other two were deleted after measuring zero. This one measured **9-11
/// substitutions per boot**, every boot, all from `objects::lookup_list_entry` —
/// and the contract says every one of them was wrong:
///
/// A GVA has no meaning apart from the page table it is resolved against.
/// `lookup_list_entry` builds its address from the **named** task's own
/// `object_list_pfn`, so the same number under a different task's table is a
/// different location that merely happens to be readable. And it always is:
/// tasks put their object lists in low pages, so the neighbour's table has
/// something mapped there on essentially every attempt. The fallback therefore
/// did not fail loudly when it was wrong — it succeeded, and returned the
/// neighbour's object-list entry as if it were this task's.
///
/// The failure mode is now a typed refusal the caller already handles
/// (`lookup_list_entry` returns `None`, which is its "the guest has not told us"
/// answer), carrying **which** of the walk's checks refused.
/// `#[track_caller]` names the site.
#[track_caller]
pub fn read_task_gva_by_id<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    let r = try_read_task_gva_by_id(host, tasks, task_id, gva, buf, page_shift);
    if let Err(named) = r {
        note_read_refusal(task_id, gva, named);
    }
    r
}

/// [`read_task_gva_by_id`] without the refusal line, for a caller whose miss is
/// an **answer** rather than a failure.
///
/// There is exactly one such shape in this device and it is worth naming,
/// because using the loud read for it put 18 lines per boot on the fail channel
/// that meant nothing. `objects::type4_probe_order` walks the live tasks asking
/// "does this one own surface N?", and a task that does not own it has no entry
/// at that slot — so the walk *must* miss on every task before the owner. The
/// miss is how the search works.
///
/// This is not a way to quieten a noisy path. The caller has to be able to say
/// what the miss means, which is why it is a second function rather than a flag:
/// a read whose failure the caller cannot interpret must stay on the loud one.
pub fn try_read_task_gva_by_id<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    let Some(task) = tasks.get(task_id) else {
        return Err(MemError::NoSuchTask);
    };
    read_task_gva(host, task, gva, buf, page_shift)
}

/// Record a refused read, latched per `(reason, task, site)`.
///
/// The reason is the [`MemError`] the walk itself returned, so the line names
/// which of the walk's checks refused rather than a label chosen here.
///
/// The latch is taken before the line is built: `Emit::field` renders eagerly,
/// and this sits one level below per-row blit loops, so building and dropping
/// strings on every refused read would make the probe cost scale with the
/// traffic it is measuring.
#[track_caller]
fn note_read_refusal(task_id: u32, gva: u64, named: MemError) {
    use crate::observe::Decline;
    // Key off the raw location, not its rendering — a refused read can repeat
    // per row, and formatting before the latch would allocate on every one.
    let loc = std::panic::Location::caller();
    if !crate::observe::first_sight(named.slug(), latch_key(task_id, 0, loc)) {
        return;
    }
    let via = via_caller();
    crate::observe::Emit::decline("gva_read_refused", &named)
        .field("task", task_id)
        .field("gva", format!("{gva:#x}"))
        .field("via", via)
        .fail();
}

/// Fixture write at the arm64e page shift, panicking if it does not land.
///
/// The page shift is fixed in the name, per the crate rule that portable code
/// takes `page_shift` and arch-fixed helpers say so. Every unit-test fixture in
/// this crate writes arm64e and treats a failed write as a broken fixture
/// rather than a result, which is why the assertion lives here instead of at
/// each call site.
///
/// # The `#[cfg(test)]` is the enforcement — do not remove it
///
/// "Product code must not call a helper with a page shift baked into its name"
/// is not a rule a reader has to hold, and it is not something to go looking
/// for: this gate and the one on [`define_task_pages_arm64e`] are the only two
/// arch-fixed functions in the crate, and behind them a product call is a
/// `cannot find function` from rustc rather than a finding. `contract::gva`
/// exposes the arch-fixed *constants* ungated, which is fine — a shift is
/// picked from `state.page_shift` at the call site, and a constant cannot
/// silently walk a page table at the wrong stride the way a helper can.
///
/// Ungating either one to share it with an integration test would take the
/// enforcement away and leave nothing, so a caller outside the crate is a
/// reason to move the fixture, not to widen the gate.
#[cfg(test)]
#[track_caller]
pub fn write_task_gva_arm64e<M: HostMemory>(host: &mut M, task: &TaskEntry, gva: u64, buf: &[u8]) {
    assert!(
        write_task_gva(host, task, gva, buf, crate::model::PAGE_SHIFT_ARM64E).is_ok(),
        "fixture write of {} bytes at {gva:#x} failed",
        buf.len()
    );
}

/// Define task 1 with an arm64e page table covering `pages` data pages from
/// `data_base_pfn`: a one-level directory at PFN 2 pointing at a root table at
/// PFN 3, whose first `pages` entries map consecutive PFNs.
///
/// The directory and root PFNs are fixed at 2 and 3 because every fixture in
/// the crate that walks a task GVA uses exactly this shape — it was defined
/// verbatim inside nine separate test bodies across four modules, differing
/// only in `pages`. Callers that also need an object list assert
/// `set_object_list` themselves; a page table is not one.
#[cfg(test)]
#[track_caller]
pub fn define_task_pages_arm64e(
    host: &mut crate::runtime::host::FakeHost,
    state: &mut crate::model::DeviceState,
    data_base_pfn: u32,
    pages: u32,
) {
    use crate::contract::endian::st32;
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::model::PAGE_SHIFT_ARM64E;
    let dir_pfn = 2u32;
    let root_pfn = 3u32;
    let dir_gpa = (dir_pfn as u64) << PAGE_SHIFT_ARM64E;
    let root_gpa = (root_pfn as u64) << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], root_pfn);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    for i in 0..pages {
        let pfn = data_base_pfn + i;
        host.map_range((pfn as u64) << PAGE_SHIFT_ARM64E, 0x4000, 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, pfn);
        let _ = host.write_gpa(root_gpa + (i as u64) * 4, &pte);
    }
    state.define_task(1, 0x1000, dir_pfn);
}

/// Translate `gva` under `task` and write `buf` into guest RAM via `write_gpa`.
///
/// **Tests / fixtures only.** Product paths must use [`write_task_gva_product`]
/// (contig HostOps view). Do not call from product encode/blit/compute.
#[cfg(test)]
pub fn write_task_gva<M: HostMemory>(
    host: &mut M,
    task: &TaskEntry,
    gva: u64,
    buf: &[u8],
    page_shift: u32,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    if !task.active || task.directory_pfn == 0 {
        return Err(MemError::NoTaskDirectory);
    }
    let geom = geometry_for_page_shift(page_shift).ok_or(MemError::UnsupportedPageShift)?;
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // Resolve first, then write: the walk borrows the host to read page tables
    // and the writes need it mutably, so the two cannot interleave. The
    // collecting form of the cutter has this one caller, so its import is here
    // rather than at the module head where it would be dead on a product build.
    use reims_vgpu_paging::span::span_chunks;
    let chunks = {
        let reader = HostPhys(&*host);
        span_chunks(&reader, geom, &gr_task, gva, buf.len()).map_err(span_refusal_error)?
    };
    for chunk in chunks {
        host.write_gpa(chunk.gpa, &buf[chunk.range()])?;
    }
    Ok(())
}

/// `file:line` of whoever called the `#[track_caller]` function above this one.
///
/// Rendered as the repo-relative tail so the field stays short enough to sit on
/// an always-on line: `runtime/blit_exec/mod.rs:1039`. The tail is whatever
/// `Location::file()` gives after `/src/`, so a module that becomes a
/// directory changes what this field reads — as `blit_exec` just did.
#[track_caller]
fn via_caller() -> String {
    let loc = std::panic::Location::caller();
    let file = loc.file();
    let tail = file.rfind("/src/").map_or(file, |i| &file[i + 5..]);
    format!("{tail}:{}", loc.line())
}

/// Dedup key for the guest-memory censuses: two task ids **and** the call site.
///
/// The call site belongs in the identity. Without it the second site to reach a
/// given `(arm, task, other)` is silent for the life of the process, and
/// `first_sight` is per-process rather than per-boot — the hazard that has
/// already caused one census here to be read as a behavioural difference.
///
/// Hashed rather than bit-packed because both ids can carry a raw wire word, so
/// neither has a bound worth relying on. This is a set key for suppressing
/// repeats, not a value anything reads back. Takes the `Location` rather than
/// its rendering so callers on a per-row path can key without allocating.
fn latch_key(task_id: u32, other: u32, loc: &std::panic::Location<'_>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    task_id.hash(&mut h);
    other.hash(&mut h);
    loc.file().hash(&mut h);
    loc.line().hash(&mut h);
    h.finish()
}

/// Whether any page of `[gva, gva+span)` resolves under `task_id`'s tables.
///
/// Separates "there is nowhere to put this" from "putting it there went wrong",
/// which callers that degrade gracefully need and a writer returning one status
/// for both cannot give them. Stops at the first hit, so the common answer costs
/// one translate rather than a walk of the whole span.
pub fn any_task_gva_page_resolves<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> bool {
    let mut found = false;
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span.max(1), page_shift, &mut |_| {
        found = true;
        false
    });
    found
}

/// Product GVA write: HostOps `map_pages` only (no `write_gpa` walk).
///
/// Full-span packed view when possible; otherwise **multi-import** maximal
/// packed GPA runs ([`crate::runtime::gva_view::write_span_within`]). Fails closed when
/// any page is unmapped or a run cannot be mapped — that walk is the whole
/// bound on this write. Always-on: `gva_write fail reason=…`, carrying the
/// check `write_span_within` actually refused on rather than a reason chosen
/// here.
///
/// `#[track_caller]` so the always-on lines can name **which** of the fifteen
/// product call sites issued the write. The reason and the writer were both
/// unattributable before: a refusal or a gate census named a task, an address
/// and a length, and finding the code that produced them meant guessing from
/// the size. Reading `Location::caller()` keeps that a reading — the callee
/// asks who called it, rather than each caller passing a label it chose.
/// The guest pages a row loop's destination span resolves to, taken before the
/// loop that writes them.
///
/// A blit does not wait for the GPU, which is why these writes were argued to be
/// authorised by the page table at the moment they run. That argument confuses
/// "synchronous with the device thread" with "instantaneous": a full-screen
/// texture copy is tens of MiB of per-row guest read and guest write, the guest's
/// own vCPUs run throughout, and the destination is re-resolved from scratch on
/// every row. The pages `gva + 8000 * stride` names at the end of the loop need
/// not be the pages the same expression named at the start, and a copy that runs
/// off its resource paints whatever the guest handed those pages to next — the
/// heap and kernel corruption this class is made of.
///
/// Capturing the whole destination span once, up front, makes every row's write
/// authorised by the walk the command itself would have been authorised by.
///
/// This lives beside [`write_task_gva_product_within`] rather than in one of its
/// callers because it is the bound that writer takes: every multi-row guest
/// writer owes it, and a second copy of the rule is how one of them ends up not
/// taking it.
///
/// `None` when the span resolves no page at all. That leaves the writer to fail
/// closed on its own terms rather than refusing a whole copy because the capture
/// failed for an unrelated reason; it is counted so the arm is measurable
/// instead of assumed.
pub fn dest_window<M: HostMemory>(
    state: &crate::model::DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Option<std::collections::HashSet<u64>> {
    if gva == 0 || span == 0 {
        return None;
    }
    let pages = task_gva_page_gpa_set(host, &state.tasks, task_id, gva, span, state.page_shift);
    if pages.is_empty() {
        crate::runtime::drain::note_store_route("blit_dest_unbounded");
        return None;
    }
    crate::runtime::drain::note_store_route("blit_dest_bound");
    Some(pages)
}

/// [`write_task_gva_product`] bounded to the guest pages a deferred window was
/// armed on.
///
/// `allowed` is `None` for every writer whose authorisation is the command it
/// is executing. It is `Some` only where the write is landing content that was
/// captured earlier against a page set, which is the one case where the live
/// page table answers a different question from the one that matters.
///
/// # Every row loop passes `Some`, and there is no shorter way to say it
///
/// A loop that re-derives its destination on each row must capture
/// [`dest_window`] once, before the loop, and pass it here every iteration. The
/// pages `gva + 8000 * stride` names at the end of a full-screen copy need not
/// be the pages that expression named at the start, and an unbounded write
/// reports success while painting whatever the guest handed those pages to next
/// — `a_blit_destination_is_bounded_against_a_guest_that_repoints_it_mid_copy`
/// is that failure written down.
///
/// This function used to have a five-argument wrapper, `write_task_gva_product`,
/// that supplied `None` for you. Two row loops drifted onto it — `blit_exec`'s
/// staged texture-to-buffer arm and `mipmap`'s level writeback — each surrounded
/// by siblings doing it right, because the two spellings were one suffix apart
/// and the wrong one was shorter and invisible in review.
///
/// A `#[cfg(test)]` scan over four hand-listed module directories used to guard
/// that. It was a source-grep scanner of the kind `AGENTS.md` bans, its own doc
/// conceded the rule "cannot be spelled reliably from source", and its watched
/// list could not see the callers outside those four modules at all. Deleting
/// the wrapper is the structural replacement: no shorter spelling is left to
/// drift onto, every call site states its authority as an argument, and the old
/// spelling is now a compile error rather than a scan that a newly-added
/// directory silently escapes.
///
/// What that does **not** do is prove a given `Some` names the right window, or
/// that a loop captured it outside itself. That much is unenforced, and
/// honestly so — which `AGENTS.md` prefers to a scanner reporting success over a
/// population it cannot see.
#[track_caller]
pub fn write_task_gva_product_within<H: HostMemory + crate::runtime::host::HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    let via = via_caller();
    let Err(err) =
        crate::runtime::gva_view::write_span_within(state, host, task_id, gva, buf, allowed)
    else {
        return Ok(());
    };
    crate::observe::Emit::decline("gva_write", &err)
        .field("task", task_id)
        .field("gva", format!("{gva:#x}"))
        .field("len", format!("{:#x}", buf.len()))
        .field("via", via)
        .fail();
    Err(err)
}

/// Resolve pages of `[gva, gva + span)` under the task the guest named — the
/// same selection as [`read_task_gva_by_id`] and
/// [`crate::runtime::gva_view::write_span_within`]'s resolver — and call `visit` with
/// each page-aligned GPA. Stops early when `visit` returns `false`.
///
/// This is a lookup, not a validator: pages that fail to translate are
/// skipped silently — the content read that follows fails (and fail-logs) on
/// its own terms. One root read and one descent span the whole range.
///
/// **The named task, or no pages.** This was the last of four sites that fell
/// back to `task_id >> 1` when the named slot had no page table to walk. The
/// other three are gone — `resolve_task_word` decides raw-only
/// (`raw_live.then_some(raw)`), `read_task_gva_by_id` refuses, and
/// `gva_view::resolve_task_for_walk` returns `None` — all on the same contract
/// argument: a GVA has no meaning apart from the page table it is resolved
/// against, and slots run densely from 0, so `task_id >> 1` is almost always
/// some *other* live task whose table happens to have something mapped there.
///
/// Here the substitution was invisible rather than merely wrong, because
/// the page-drift guard that decides whether a resolved span may still be
/// written to guest RAM re-resolves
/// through *this* function with the *same* task id the window was armed under.
/// A window indexed under the neighbour's table was therefore re-indexed under
/// the neighbour's table, the two sets matched, and the guard reported "still
/// ours". It could not see a hazard it reproduced.
///
/// A short walk is what every caller already fails closed on: the guest-run
/// builder and the deferred-Store arm both compare the page count against the
/// span and decline, and the compute rail reports its count as `pages=` on an
/// always-on line.
pub fn visit_task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(u64) -> bool,
) {
    visit_task_gva_pages(host, tasks, task_id, gva, span, page_shift, &mut |gpa| {
        match gpa {
            Some(gpa) => visit(gpa),
            None => true,
        }
    });
}

/// The resolved page GPAs of `[gva, gva+span)` under `task_id`'s page table, in
/// GVA order, with unresolved pages dropped.
///
/// The ordered form, for callers that walk the result as a window —
/// neighbouring entries differing by exactly one page is what lets a gather
/// coalesce them. Compare `len()` against
/// [`reims_vgpu_paging::span::pages_spanned`] to learn whether
/// anything was dropped.
pub fn task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> Vec<u64> {
    let mut out = Vec::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, &mut |gpa| {
        out.push(gpa);
        true
    });
    out
}

/// The distinct page GPAs of `[gva, gva+span)` under `task_id`'s page table.
///
/// The set form, for callers that only ask "is this page one of mine?" — the
/// deferred-window page indexes and the blit/Store destination bounds. Order
/// is not preserved and repeats collapse, so `len()` is a lower bound on the
/// pages walked; that is what every caller compares against
/// [`reims_vgpu_paging::span::pages_spanned`].
pub fn task_gva_page_gpa_set<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> std::collections::HashSet<u64> {
    let mut out = std::collections::HashSet::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, &mut |gpa| {
        out.insert(gpa);
        true
    });
    out
}

/// Shared page-table walk behind [`visit_task_gva_page_gpas`]: one root read
/// and one descent for the whole range, visiting every page in order. Reports
/// an unresolved page as `None` rather than dropping it, which is what a caller
/// recording *which* pages it read needs.
fn visit_task_gva_pages<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(Option<u64>) -> bool,
) {
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return;
    };
    let reader = HostPhys(host);
    let Some(task) = tasks.get(task_id) else {
        return;
    };
    if !task.active || task.directory_pfn == 0 {
        return;
    }
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // Every page of the run, which is the shape the licence check and the
    // guest-run resolvers ask for. One descent is shared across the pages whose
    // upper indices match, instead of `depth` guest reads per page.
    //
    // A setup refusal visits nothing and is dropped rather than reported: this
    // function's contract is that a caller compares what it saw against what it
    // expected, so it is the only one of the span readers for which "no pages"
    // is an answer rather than an error.
    let _ = walk_span(&reader, geom, &gr_task, gva, span, &mut |_, r| {
        visit(r.ok())
    });
}

/// Every page of `[gva, gva+span)` in order, resolved through one root read and
/// one descent, with `None` for a page the table cannot translate.
///
/// [`visit_task_gva_page_gpas`] drops the unresolved pages; a caller checking a
/// cached page list against the live table needs them, because "page 40 does not
/// translate" and "page 40 translates elsewhere" are different findings and only
/// one of them is about the guest. Stride is fixed at one page for the same
/// reason: a check that samples cannot conclude anything about the pages it
/// skipped.
///
/// The visitor stops when `visit` answers `false`, and it visits nothing at all
/// for an inactive task, an absent directory or an unwalkable page geometry — so
/// a caller must compare what it saw against what it expected rather than
/// treating a quiet return as agreement.
pub fn visit_task_gva_pages_in_order<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    visit: &mut dyn FnMut(Option<u64>) -> bool,
) {
    visit_task_gva_pages(host, tasks, task_id, gva, span, page_shift, visit);
}

/// Translate one GVA to a GPA under the task directory (single page).
pub fn translate_task_gva<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    page_shift: u32,
) -> Option<u64> {
    if !task.active || task.directory_pfn == 0 {
        return None;
    }
    let geom = geometry_for_page_shift(page_shift)?;
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    // A one-byte span, so the single chunk's `gpa` is this GVA's own address —
    // page base plus its offset within the page. Going through the span cutter
    // rather than `read_task_root` + `translate_root` by hand is what keeps the
    // zero-root and zero-depth refusals here identical to the ones every other
    // rail gets: written out at this call site they were a fifth copy, and this
    // copy was the one that did not have them, reaching the same answer only
    // because the descent refuses a zero root a second time further down.
    let mut gpa = None;
    visit_span_chunks(&reader, geom, &gr_task, gva, 1, &mut |chunk| {
        gpa = Some(chunk.gpa);
        false
    })
    .ok()?;
    gpa
}

/// One-line walk diagnosis for a single task slot (measure-only; no product gates).
///
/// Example: `tid=2 act=1 dir=0xabc root=0xdef depth=2 st=zero-pfn pte=0 lvl=1 idx=4`
pub fn diagnose_task_slot<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    if !task.active {
        return format!(
            "tid={task_id} act=0 dir={:#x} st=inactive",
            task.directory_pfn
        );
    }
    if task.directory_pfn == 0 {
        return format!("tid={task_id} act=1 dir=0 st=no-directory");
    }
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return format!(
            "tid={task_id} act=1 dir={:#x} st=bad-page-shift({page_shift})",
            task.directory_pfn
        );
    };
    let reader = HostPhys(host);
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let root = match read_task_root(&reader, &gr_task, geom) {
        Ok(r) => r,
        Err(st) => {
            return format!(
                "tid={task_id} act=1 dir={:#x} st=root({})",
                task.directory_pfn,
                resolve_status_name(st)
            );
        }
    };
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva);
    if t.status == ResolveStatus::Ok {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st=ok gpa={:#x} leaf_pfn={:#x}",
            task.directory_pfn, root.root_pfn, root.depth, t.gpa, t.leaf_pfn
        )
    } else {
        format!(
            "tid={task_id} act=1 dir={:#x} root={:#x} depth={} st={} pte={:#x} lvl={} idx={}",
            task.directory_pfn,
            root.root_pfn,
            root.depth,
            resolve_status_name(t.status),
            t.raw_pte,
            t.level,
            t.entry_index
        )
    }
}

/// Diagnose walk under wire `task_id`, `task_id>>1`, and a few active peers.
///
/// Compact multi-clause string for one fail-log line (MapMemory2 / stage Unmapped).
pub fn diagnose_gva_walk<M: HostMemory>(
    host: &M,
    tasks: &TaskTable,
    task_id: u32,
    gva: u64,
    page_shift: u32,
) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(6);
    let mut tried = std::collections::BTreeSet::new();
    let try_id = |id: u32, parts: &mut Vec<String>, tried: &mut std::collections::BTreeSet<u32>| {
        if !tried.insert(id) {
            return;
        }
        let Some(task) = tasks.get(id) else {
            // No task under this id at all. `st=undefined` rather than the
            // `st=oob` this printed against the old fixed array: there is no
            // range to be outside of now, and the two say different things —
            // one was "the id is too large", this is "the guest never defined
            // it", which is the only way to reach here.
            parts.push(format!("tid={id} st=undefined"));
            return;
        };
        parts.push(diagnose_task_slot(host, task, id, gva, page_shift));
    };
    try_id(task_id, &mut parts, &mut tried);
    try_id(task_id >> 1, &mut parts, &mut tried);
    // Peer scan: active tasks with a directory (cap 4 extras) — catches wrong-task walks.
    let peer_ids: Vec<u32> = tasks
        .live()
        .filter(|(id, t)| !tried.contains(id) && t.directory_pfn != 0)
        .map(|(id, _)| id)
        .take(4)
        .collect();
    for id in peer_ids {
        try_id(id, &mut parts, &mut tried);
    }
    format!(
        "gva={gva:#x} page_shift={page_shift} | {}",
        parts.join(" || ")
    )
}

/// Snapshot of active task directories (for periodic map census).
pub fn format_active_tasks(tasks: &TaskTable) -> String {
    let mut bits = Vec::new();
    for (i, t) in tasks.live() {
        bits.push(format!(
            "t{i}:dir={:#x},len={:#x},ol_pfn={:#x},ol_n={}",
            t.directory_pfn, t.length, t.object_list_pfn, t.object_list_count
        ));
    }
    if bits.is_empty() {
        "tasks=none".into()
    } else {
        format!("tasks[{}]={}", bits.len(), bits.join(";"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::endian::st32;
    use crate::observe::Decline;

    /// The collapse this migration ended: every distinct check — the walk's own
    /// and four more here — answered `MemError::Unmapped`, and
    /// `MemError` reached the always-on log at no site in the crate. So a
    /// malformed PTE, a zero root PFN, an inactive task and a genuinely unmapped
    /// GPA were one value, invisibly, on the guest-memory hot path.
    ///
    /// Asserted as "no two of these share a slug" rather than by naming each
    /// one, because the property that matters is the absence of aliasing.
    #[test]
    fn no_two_guest_memory_checks_answer_with_the_same_reason() {
        use reims_vgpu_paging::resolve::ResolveStatus as R;
        const WALK: &[R] = &[
            R::ErrArgs,
            R::ErrInactiveTask,
            R::ErrNoDirectory,
            R::ErrDirectoryRead,
            R::ErrZeroRootPfn,
            R::ErrZeroDepth,
            R::ErrDepthTooDeep,
            R::ErrPageTableRead,
            R::ErrZeroPfn,
            R::ErrMalformedPte,
            R::ErrUnsupportedGeometry,
        ];
        let mut slugs: Vec<&str> = WALK
            .iter()
            .map(|r| MemError::Unresolved(*r).slug())
            .chain(
                [
                    MemError::Unmapped,
                    MemError::NoCpu,
                    MemError::Overflow,
                    MemError::BadArgs,
                    MemError::QemuReadGpaCallbackMissing,
                    MemError::QemuReadGpaCallbackFailed(-1),
                    MemError::QemuWriteGpaCallbackMissing,
                    MemError::QemuWriteGpaCallbackFailed(-1),
                    MemError::QemuReadKvaCallbackMissing,
                    MemError::QemuReadKvaCallbackFailed(-1),
                    MemError::XregUnavailable,
                    MemError::QemuReadXregCallbackMissing,
                    MemError::QemuReadXregCallbackFailed(-1),
                    MemError::NoTaskDirectory,
                    MemError::UnsupportedPageShift,
                    MemError::TaskRootRead,
                    MemError::NoSuchTask,
                    MemError::NotRam,
                    MemError::MapPagesRefused,
                    MemError::RunOutOfRange,
                ]
                .iter()
                .map(|e| e.slug()),
            )
            .collect();
        let total = slugs.len();
        assert_eq!(total, 31, "11 walk reasons + 20 memory reasons");
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(
            slugs.len(),
            total,
            "two guest-memory checks share a reason slug"
        );

        // `Unresolved` must forward, not invent: the walk already named the
        // check, and a second name here would make two log lines disagree about
        // one event.
        assert_eq!(
            MemError::Unresolved(R::ErrMalformedPte).slug(),
            "gva_malformed_pte"
        );
        // And `Ok` inside `Unresolved` is a construction bug, named as one
        // rather than reported as a plausible walk reason.
        assert_eq!(MemError::Unresolved(R::Ok).slug(), "mem_unresolved_ok");
    }

    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::model::{DeviceId, DeviceState, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::decode::resource::RESOURCE_PAGE_SHIFT;
    use crate::runtime::host::FakeHost;

    #[test]
    fn diagnose_reports_ok_and_zero_pfn() {
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        // leaf PTE for page index 0 → pfn 4
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        let mut tasks = TaskTable::default();
        tasks.define(1, TaskEntry::define(0x1000, 2));
        let ok = diagnose_gva_walk(&host, &tasks, 1, 0, PAGE_SHIFT_X86);
        assert!(ok.contains("st=ok"), "{ok}");
        assert!(
            ok.contains("gpa=0x4000") || ok.contains("leaf_pfn=0x4"),
            "{ok}"
        );
        // unmapped page index 1
        let miss = diagnose_gva_walk(&host, &tasks, 1, 0x1000, PAGE_SHIFT_X86);
        assert!(
            miss.contains("zero-pfn") || miss.contains("st=zero"),
            "{miss}"
        );
    }

    #[test]
    fn one_level_gva_read() {
        let mut host = FakeHost::new();
        // directory at pfn 2, root table at pfn 3, leaf data at pfn 4
        let dir_gpa = 2u64 << RESOURCE_PAGE_SHIFT;
        let root_gpa = 3u64 << RESOURCE_PAGE_SHIFT;
        let data_gpa = 4u64 << RESOURCE_PAGE_SHIFT;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        // directory: root_pfn=3, depth=1
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // PTE for gva page 0: pfn 4
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.define_task(1, 0x1000, 2);
        let mut buf = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut buf, PAGE_SHIFT_ARM64E).is_ok());
        assert_eq!(buf, [0xab; 4]);
        // Round-trip write.
        let out = [1u8, 2, 3, 4];
        assert!(write_task_gva(&mut host, &state.tasks[1], 0, &out, PAGE_SHIFT_ARM64E).is_ok());
        let mut back = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut back, PAGE_SHIFT_ARM64E).is_ok());
        assert_eq!(back, out);
    }

    #[test]
    fn x86_4k_geometry_read() {
        let mut host = FakeHost::new();
        let page_shift = PAGE_SHIFT_X86;
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data_gpa = 4u64 << page_shift;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xcd);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        let mut state = DeviceState::new(DeviceId(1), page_shift);
        state.define_task(1, 0x1000, 2);
        let mut buf = [0u8; 4];
        assert!(read_task_gva(&host, &state.tasks[1], 0, &mut buf, page_shift).is_ok());
        assert_eq!(buf, [0xcd; 4]);
    }

    #[test]
    fn unknown_page_shift_rejected() {
        assert!(geometry_for_page_shift(13).is_none());
        assert!(geometry_for_page_shift(0).is_none());
        assert!(geometry_for_page_shift(PAGE_SHIFT_X86).is_some());
        assert!(geometry_for_page_shift(PAGE_SHIFT_ARM64E).is_some());
    }

    /// What a task may write is what its own page table maps, and nothing else.
    ///
    /// The guest allocates, installs its own PTEs, uploads, and only afterwards
    /// notifies the range with `MapMemory2` — measured at 0-29 ms after the
    /// write. So a notification cannot authorise anything, and the walk this
    /// writer performs at write time is the whole bound. Both halves are
    /// asserted here because a writer that lands everything and a writer that
    /// refuses everything each satisfy one of them alone.
    #[test]
    fn the_page_table_is_the_only_bound_on_a_product_gva_write() {
        use crate::model::PAGE_SHIFT_ARM64E;
        let mut host = crate::runtime::host::FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // Task 1's own page tables map GVA 0.. onto four data pages.
        define_task_pages_arm64e(&mut host, &mut state, 0x100, 4);
        let page = 1u64 << PAGE_SHIFT_ARM64E;

        assert!(
            write_task_gva_product_within(&mut state, &mut host, 1, page, &[1, 2, 3, 4], None)
                .is_ok(),
            "the range is mapped for the writing task, so the write lands \
             whatever the guest has or has not notified"
        );

        // Inside the task's one-level table but with a zero PTE — the fixture
        // installs four data pages, so index 10 resolves to nothing. Chosen
        // over an index past the end of the table because a one-level walk masks
        // its index to the entry count and `4096 * page` aliases index 0, which
        // would have made this assertion pass for the wrong reason.
        assert!(
            write_task_gva_product_within(&mut state, &mut host, 1, 10 * page, &[1, 2, 3, 4], None)
                .is_err(),
            "unmapped for this task, so the writer fails closed"
        );
    }

    /// The `via=` field must name the **caller**, not `gva_mem.rs` itself, or it
    /// reports where the log line is written and nothing about who wrote.
    ///
    /// Also pins the rendering: a bare `Location::file()` is the whole build
    /// path, which is long enough to push the load-bearing fields off the end of
    /// a scanned log line.
    #[test]
    fn the_via_field_names_the_call_site_and_not_the_logging_site() {
        #[track_caller]
        fn relay() -> String {
            via_caller()
        }
        let here = relay();
        assert!(
            here.starts_with("runtime/gva_mem.rs:"),
            "expected a repo-relative caller, got {here}"
        );
        assert!(
            !here.contains("crates/") && !here.starts_with('/'),
            "the crate prefix must be trimmed off, got {here}"
        );
        let line: u32 = here.rsplit(':').next().unwrap().parse().unwrap();
        assert!(line > 0);
    }

    /// The latch key must separate call sites, or the second site to reach a
    /// given `(arm, task, by)` is silent for the life of the process — the same
    /// per-process latching hazard that has already misread one census.
    #[test]
    fn the_refusal_latch_key_separates_call_sites() {
        #[track_caller]
        fn key(task: u32, other: u32) -> u64 {
            latch_key(task, other, std::panic::Location::caller())
        }
        let a = key(1, 0);
        let b = key(1, 0);
        assert_ne!(a, b, "two call sites, same ids, must be two sightings");
        assert_ne!(key(1, 0), key(2, 0));
        assert_ne!(key(1, 0), key(1, 1));
        let loc = std::panic::Location::caller();
        assert_eq!(
            latch_key(1, 0, loc),
            latch_key(1, 0, loc),
            "and it is stable"
        );
    }

    /// A read the named task cannot serve is **refused**, even when the
    /// neighbouring task's page table would have resolved the same address.
    ///
    /// This is the deletion itself. Task 2 maps GVA page 1; task 5 (`5 >> 1 == 2`)
    /// maps nothing. The old code walked task 2 here and returned its bytes,
    /// which is why the substitution never surfaced as an error — a GVA under
    /// the wrong page table is a different location that merely happens to be
    /// readable, and low pages essentially always are.
    #[test]
    fn a_read_the_named_task_cannot_serve_is_refused_not_redirected() {
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa + 4, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.define_task(2, 0x1_0000, 2);
        state.define_task(5, 0x1_0000, 9);

        // The donor really can serve it — otherwise this test would pass for
        // the wrong reason.
        let mut buf = [0u8; 4];
        assert!(
            read_task_gva_by_id(&host, &state.tasks, 2, 0x1000, &mut buf, PAGE_SHIFT_X86).is_ok()
        );
        assert_eq!(buf, [0xab; 4]);

        let mut buf = [0u8; 4];
        let err = read_task_gva_by_id(&host, &state.tasks, 5, 0x1000, &mut buf, PAGE_SHIFT_X86)
            .unwrap_err();
        assert!(
            matches!(err, MemError::Unresolved(_)),
            "task 5's own walk must be what answers, got {err:?}"
        );
        assert_eq!(
            buf, [0u8; 4],
            "and no neighbour's bytes may reach the caller"
        );
    }

    /// A page walk for a task with no page table yields **no pages**, even when
    /// the `>> 1` neighbour's table resolves the same address.
    ///
    /// This is the fourth and last `>> 1` arm, and the one whose substitution no
    /// guard could see: the page-drift guard re-resolves a resolved span
    /// through this same function under the same task id, so a
    /// window indexed under the neighbour's table was re-indexed under the
    /// neighbour's table and the drift check reported the pages "still ours".
    ///
    /// Task 2 maps GVA page 1; task 5 (`5 >> 1 == 2`) is live with no directory,
    /// which is the state `define_task` really produces — the slot is active and
    /// only the walk fails.
    #[test]
    fn a_page_walk_for_a_task_with_no_page_table_visits_nothing() {
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x100, 0xab);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa + 4, &pte).unwrap();

        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.define_task(2, 0x1_0000, 2);
        state.define_task(5, 0x1_0000, 0);
        assert!(
            state.tasks.is_active(5),
            "the slot is live; only the page table is missing"
        );

        // The donor really can serve it — otherwise this test would pass for the
        // wrong reason.
        let mut donor = Vec::new();
        visit_task_gva_page_gpas(&host, &state.tasks, 2, 0x1000, 4, PAGE_SHIFT_X86, &mut |gpa| {
            donor.push(gpa);
            true
        });
        assert_eq!(donor, vec![data_gpa], "task 2 resolves GVA page 1");

        let mut pages = Vec::new();
        visit_task_gva_page_gpas(&host, &state.tasks, 5, 0x1000, 4, PAGE_SHIFT_X86, &mut |gpa| {
            pages.push(gpa);
            true
        });
        assert!(
            pages.is_empty(),
            "no neighbour's pages may be indexed under task 5, got {pages:x?}"
        );
    }

    /// When neither task can serve the read, the caller must receive the
    /// **named** task's own walk error, not a `NoSuchTask` this function chose.
    ///
    /// The task exists and is active here; what fails is the walk, with no
    /// directory installed. Reporting `NoSuchTask` for that would name a check
    /// that never ran — the collapse the typed-decline vocabulary exists to
    /// prevent, and it regrew here because the fallback discarded both errors.
    #[test]
    fn a_failed_fallback_read_carries_the_named_tasks_own_refusal() {
        let host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        state.define_task(6, 0x1_0000, 0);
        assert!(
            state.tasks.is_active(6),
            "the slot is live; only the walk fails"
        );
        let mut buf = [0u8; 4];
        let err = read_task_gva_by_id(&host, &state.tasks, 6, 0x1000, &mut buf, PAGE_SHIFT_X86)
            .unwrap_err();
        assert_eq!(
            err,
            MemError::NoTaskDirectory,
            "the walk's own refusal, not a blanket NoSuchTask"
        );
    }

    /// A word naming no task at all still reports `NoSuchTask` — that one IS
    /// the check that refused.
    ///
    /// `u32::MAX` rather than "one past the table": there is no table to be past
    /// now, and an undefined id is the only way to reach this. The largest id
    /// the wire can carry is the strongest case, and it would have been refused
    /// by a range check before, which is a different refusal wearing this one's
    /// name.
    /// The quiet read must return exactly what the loud one returns.
    ///
    /// `try_read_task_gva_by_id` exists so a speculative caller does not put a
    /// line on the fail channel for a miss that is its answer. That is only
    /// sound while the two agree on the answer itself — a quiet read that took
    /// a different path would silence a real refusal instead of an expected
    /// one, which is the failure this split could introduce and nothing else
    /// would catch.
    #[test]
    fn the_quiet_read_answers_exactly_as_the_reporting_one_does() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        host.write_gpa(data_gpa, &[0xab; 8]).unwrap();
        state.define_task(2, 0x1000, 2);

        for (task, gva, what) in [
            (2u32, 0x1000u64, "a page the task maps"),
            (2, 0x2000, "a page the task does not map"),
            (2, 0, "the null page"),
            (77, 0x1000, "a task nothing defined"),
        ] {
            let mut loud = [0u8; 8];
            let mut quiet = [0u8; 8];
            let a = read_task_gva_by_id(&host, &state.tasks, task, gva, &mut loud, PAGE_SHIFT_X86);
            let b =
                try_read_task_gva_by_id(&host, &state.tasks, task, gva, &mut quiet, PAGE_SHIFT_X86);
            assert_eq!(a, b, "verdicts differ on {what}");
            assert_eq!(loud, quiet, "bytes differ on {what}");
        }

        // The fixture reads something, so the loop cannot pass by refusing
        // everything identically.
        let mut buf = [0u8; 8];
        assert!(
            try_read_task_gva_by_id(&host, &state.tasks, 2, 0x1000, &mut buf, PAGE_SHIFT_X86)
                .is_ok()
        );
        assert_eq!(buf, [0xab; 8]);
    }

    #[test]
    fn a_read_for_an_undefined_task_word_still_reports_no_such_task() {
        let host = FakeHost::new();
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut buf = [0u8; 4];
        let err = read_task_gva_by_id(
            &host,
            &state.tasks,
            u32::MAX,
            0x1000,
            &mut buf,
            PAGE_SHIFT_X86,
        )
        .unwrap_err();
        assert_eq!(err, MemError::NoSuchTask);
    }
}
