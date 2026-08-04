//! Read task GPU-virtual addresses via the task page directory.
//!
//! Thin wrapper over [`crate::contract::gva_resolve`] + [`HostMemory`].
//! Geometry always requires an explicit create-time page_shift (12 = x86_64,
//! 14 = arm64e). There is no arm-default overload — callers must choose.

use crate::contract::gva_resolve::{
    read_task_root, resolve_status_name, translate_root, Cache, Geometry, PhysReader,
    ResolveStatus, Task, ARM64E_GEOMETRY, X86_64_GEOMETRY,
};
use crate::model::TaskEntry;
use crate::runtime::host::{HostMemory, MemError};

struct HostPhys<'a, M: HostMemory>(&'a M);

impl<M: HostMemory> PhysReader for HostPhys<'_, M> {
    fn read_phys(&self, gpa: u64, dst: &mut [u8]) -> bool {
        self.0.read_gpa(gpa, dst).is_ok()
    }
}

/// Select page-table geometry for a known guest page size.
///
/// Only 12 (x86_64) and 14 (arm64e) are valid. Unknown shifts return `None`
/// (no silent arm fallback).
#[inline]
pub fn geometry_for_page_shift(page_shift: u32) -> Option<&'static Geometry> {
    if page_shift == X86_64_GEOMETRY.page_shift {
        Some(&X86_64_GEOMETRY)
    } else if page_shift == ARM64E_GEOMETRY.page_shift {
        Some(&ARM64E_GEOMETRY)
    } else {
        None
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
    let root = read_task_root(&reader, &gr_task, geom).map_err(|_| MemError::TaskRootRead)?;
    let mut cache = Cache::default();
    let mut filled = 0usize;
    while filled < buf.len() {
        let cur = gva.saturating_add(filled as u64);
        let t = translate_root(
            &reader,
            geom,
            root.root_pfn,
            root.depth,
            cur,
            Some(&mut cache),
        );
        if t.status != ResolveStatus::Ok {
            return Err(MemError::Unresolved(t.status));
        }
        let page_left = geom.page_size as u64 - (cur & geom.page_offset_mask as u64);
        let n = (buf.len() - filled).min(page_left as usize);
        host.read_gpa(t.gpa, &mut buf[filled..filled + n])?;
        filled += n;
    }
    Ok(())
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
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
    page_shift: u32,
) -> Result<(), MemError> {
    let named = if (task_id as usize) < tasks.len() {
        match read_task_gva(host, &tasks[task_id as usize], gva, buf, page_shift) {
            Ok(()) => return Ok(()),
            Err(e) => e,
        }
    } else {
        MemError::NoSuchTask
    };
    note_read_refusal(task_id, gva, named);
    Err(named)
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
    assert!(state.define_task(1, 0x1000, dir_pfn));
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
    let root = {
        let reader = HostPhys(&*host);
        read_task_root(&reader, &gr_task, geom).map_err(|_| MemError::TaskRootRead)?
    };
    let mut cache = Cache::default();
    let mut written = 0usize;
    while written < buf.len() {
        let cur = gva.saturating_add(written as u64);
        let t = {
            let reader = HostPhys(&*host);
            translate_root(
                &reader,
                geom,
                root.root_pfn,
                root.depth,
                cur,
                Some(&mut cache),
            )
        };
        if t.status != ResolveStatus::Ok {
            return Err(MemError::Unresolved(t.status));
        }
        let page_left = geom.page_size as u64 - (cur & geom.page_offset_mask as u64);
        let n = (buf.len() - written).min(page_left as usize);
        host.write_gpa(t.gpa, &buf[written..written + n])?;
        written += n;
    }
    Ok(())
}

/// `file:line` of whoever called the `#[track_caller]` function above this one.
///
/// Rendered as the repo-relative tail so the field stays short enough to sit on
/// an always-on line: `runtime/blit_exec.rs:1039`.
#[track_caller]
fn via_caller() -> String {
    let loc = std::panic::Location::caller();
    format!("{}:{}", via_file_tail(loc.file()), loc.line())
}

/// The tail of a `Location::file()` from the crate's `src/` onward, always
/// spelled with `/`.
///
/// `file!()` carries the path separator of the host that compiled the crate, so
/// a Windows build spells it `crates\reims-vgpu\src\runtime\gva_mem.rs`.
/// Matching only `/src/` found nothing there and emitted the whole build path —
/// precisely what the trimming exists to prevent, on the rail where the log line
/// is the only evidence there is.
///
/// The result is normalised rather than merely trimmed so `via=` reads the same
/// on every host: a grep or offline parser written against one boot's log keeps
/// working on another's.
fn via_file_tail(file: &str) -> String {
    let norm = file.replace('\\', "/");
    match norm.rfind("/src/") {
        Some(i) => norm[i + 5..].to_string(),
        None => norm,
    }
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
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> bool {
    let mut found = false;
    visit_task_gva_page_gpas(
        host,
        tasks,
        task_id,
        gva,
        span.max(1),
        page_shift,
        1,
        &mut |_| {
            found = true;
            false
        },
    );
    found
}

/// Product GVA write: HostOps `map_pages` only (no `write_gpa` walk).
///
/// Full-span packed view when possible; otherwise **multi-import** maximal
/// packed GPA runs ([`crate::runtime::gva_view::write_span_within`]). Fails closed when
/// any page is unmapped or a run cannot be mapped — that walk is the whole
/// bound on this write. Always-on: `gva_write fail reason=…`, carrying the
/// check `write_span` actually refused on rather than a reason chosen here.
///
/// `#[track_caller]` so the always-on lines can name **which** of the fifteen
/// product call sites issued the write. The reason and the writer were both
/// unattributable before: a refusal or a gate census named a task, an address
/// and a length, and finding the code that produced them meant guessing from
/// the size. Reading `Location::caller()` keeps that a reading — the callee
/// asks who called it, rather than each caller passing a label it chose.
#[track_caller]
pub fn write_task_gva_product<H: HostMemory + crate::runtime::host::HostOps>(
    state: &mut crate::model::DeviceState,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
) -> Result<(), MemError> {
    write_task_gva_product_within(state, host, task_id, gva, buf, None)
}

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
/// `stride_pages` visits every Nth page plus always the last (1 = every page);
/// callers trade probe density against walk cost.
///
/// This is a lookup, not a validator: pages that fail to translate are
/// skipped silently — the content read that follows fails (and fail-logs) on
/// its own terms. One page-walk cache and one root read span the whole range.
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
/// `storage_flush::deferred_pages_still_ours` — the drift guard that decides
/// whether a deferred window may still be written to guest RAM — re-resolves
/// through *this* function with the *same* task id the window was armed under.
/// A window indexed under the neighbour's table was therefore re-indexed under
/// the neighbour's table, the two sets matched, and the guard reported "still
/// ours". It could not see a hazard it reproduced.
///
/// A short walk is what every caller already fails closed on: the guest-run
/// builder and the deferred-Store arm both compare the page count against the
/// span and decline, and the compute rail reports its count as `pages=` on an
/// always-on line.
#[allow(
    clippy::too_many_arguments,
    reason = "the visitor API exposes task, span, page geometry, and callback state explicitly"
)]
pub fn visit_task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    stride_pages: u64,
    visit: &mut dyn FnMut(u64) -> bool,
) {
    visit_task_gva_pages(
        host,
        tasks,
        task_id,
        gva,
        span,
        page_shift,
        stride_pages,
        &mut |gpa| match gpa {
            Some(gpa) => visit(gpa),
            None => true,
        },
    );
}

/// How many guest pages `[gva, gva+span)` touches, given `page_size`.
///
/// The `gva % page_size` term is the whole content: a span that starts
/// mid-page reaches one page further than its length alone implies. Callers
/// compare a walk's result against this to decide whether the *whole* span
/// resolved, and getting it wrong reads as "fully covered" for exactly the
/// windows that straddle a page boundary — which is most of them.
pub fn pages_spanned(gva: u64, span: u64, page_size: u64) -> u64 {
    ((gva % page_size) + span).div_ceil(page_size)
}

/// The resolved page GPAs of `[gva, gva+span)` under `task_id`'s page table, in
/// GVA order, with unresolved pages dropped.
///
/// The ordered form, for callers that walk the result as a window —
/// neighbouring entries differing by exactly one page is what lets a gather
/// coalesce them. Compare `len()` against [`pages_spanned`] to learn whether
/// anything was dropped.
pub fn task_gva_page_gpas<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> Vec<u64> {
    let mut out = Vec::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, 1, &mut |gpa| {
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
/// pages walked; that is what every caller compares against [`pages_spanned`].
pub fn task_gva_page_gpa_set<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
) -> std::collections::HashSet<u64> {
    let mut out = std::collections::HashSet::new();
    visit_task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift, 1, &mut |gpa| {
        out.insert(gpa);
        true
    });
    out
}

/// Shared page-table walk behind [`visit_task_gva_page_gpas`]: one root read and
/// one walk cache for the whole range, visiting every `stride_pages`-th page
/// plus the exact last page. Reports an unresolved page as `None` rather than
/// dropping it, which is what a caller recording *which* pages it read needs.
#[allow(
    clippy::too_many_arguments,
    reason = "the visitor API exposes task, span, page geometry, and callback state explicitly"
)]
fn visit_task_gva_pages<M: HostMemory>(
    host: &M,
    tasks: &[TaskEntry],
    task_id: u32,
    gva: u64,
    span: u64,
    page_shift: u32,
    stride_pages: u64,
    visit: &mut dyn FnMut(Option<u64>) -> bool,
) {
    if span == 0 || stride_pages == 0 {
        return;
    }
    let Some(geom) = geometry_for_page_shift(page_shift) else {
        return;
    };
    let reader = HostPhys(host);
    let Some(task) = tasks.get(task_id as usize) else {
        return;
    };
    if !task.active || task.directory_pfn == 0 {
        return;
    }
    let gr_task = Task {
        active: true,
        directory_pfn: task.directory_pfn,
    };
    let Ok(root) = read_task_root(&reader, &gr_task, geom) else {
        return;
    };
    let page = geom.page_size as u64;
    let first = gva & !(page - 1);
    let last = gva.saturating_add(span - 1) & !(page - 1);
    let step = page.saturating_mul(stride_pages);
    let mut cache = Cache::default();
    let mut cur = first;
    loop {
        let t = translate_root(
            &reader,
            geom,
            root.root_pfn,
            root.depth,
            cur,
            Some(&mut cache),
        );
        let resolved = (t.status == ResolveStatus::Ok).then(|| t.gpa & !(page - 1));
        if !visit(resolved) {
            return;
        }
        if cur == last {
            return;
        }
        // Always end on the exact last page so span tails are covered.
        cur = cur.saturating_add(step).min(last);
    }
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
    let root = read_task_root(&reader, &gr_task, geom).ok()?;
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva, None);
    if t.status != ResolveStatus::Ok {
        return None;
    }
    Some(t.gpa)
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
    let t = translate_root(&reader, geom, root.root_pfn, root.depth, gva, None);
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
    tasks: &[TaskEntry],
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
        if (id as usize) >= tasks.len() {
            parts.push(format!("tid={id} st=oob"));
            return;
        }
        parts.push(diagnose_task_slot(
            host,
            &tasks[id as usize],
            id,
            gva,
            page_shift,
        ));
    };
    try_id(task_id, &mut parts, &mut tried);
    try_id(task_id >> 1, &mut parts, &mut tried);
    // Peer scan: active tasks with a directory (cap 4 extras) — catches wrong-task walks.
    let mut peers = 0u32;
    for (i, t) in tasks.iter().enumerate() {
        if peers >= 4 {
            break;
        }
        let id = i as u32;
        if tried.contains(&id) || !t.active || t.directory_pfn == 0 {
            continue;
        }
        try_id(id, &mut parts, &mut tried);
        peers += 1;
    }
    format!(
        "gva={gva:#x} page_shift={page_shift} | {}",
        parts.join(" || ")
    )
}

/// Snapshot of active task directories (for periodic map census).
pub fn format_active_tasks(tasks: &[TaskEntry]) -> String {
    let mut bits = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        if !t.active {
            continue;
        }
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

    /// Every guest writer that loops over rows takes [`dest_window`] first.
    ///
    /// The bounded/unbounded pair is one character apart at the call site, the
    /// unbounded form is the shorter one, and the difference is invisible in
    /// review: `a_blit_destination_is_bounded_against_a_guest_that_repoints_it_mid_copy`
    /// shows what the unbounded arm does to a guest that re-points the range —
    /// it reports success and paints a page the command never named. Two row
    /// loops had drifted onto it (`blit_exec`'s staged texture-to-buffer arm and
    /// `mipmap`'s level writeback), each surrounded by siblings doing it right.
    ///
    /// So this is a gate on the class rather than on those two: a loop writer is
    /// a `write_task_gva_product` call whose enclosing file also re-derives a
    /// destination per row, and there is no way to spell that reliably from
    /// source. What is reliable is that the modules that own row loops must not
    /// name the unbounded form at all. `drain` is not one of them — its three
    /// calls are single completion stamps, authorised by the packet being
    /// retired — and `gva_view`/`compute_exec` name it only from tests.
    #[test]
    fn row_loop_writers_take_the_bounded_form() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/runtime");
        for relative in [
            "blit_exec.rs",
            "mipmap.rs",
            "mapping_write.rs",
            "storage_flush.rs",
        ] {
            let src = std::fs::read_to_string(root.join(relative))
                .unwrap_or_else(|e| panic!("{relative}: {e}"));
            // `_within` is the bounded form and shares this prefix, so match the
            // open paren that only the unbounded form can present.
            let unbounded = src.match_indices("write_task_gva_product(").count();
            assert_eq!(
                unbounded, 0,
                "{relative} calls the unbounded write_task_gva_product; a row loop \
                 that re-resolves its destination must capture dest_window once \
                 up front and pass it to write_task_gva_product_within"
            );
        }
    }

    /// A span's page count is decided by where it *starts*, not only by how long
    /// it is.
    ///
    /// Four rails compare a walk's page count against this to decide whether the
    /// whole span resolved. Drop the offset term and a window that straddles a
    /// page boundary — which is most of them, since a texture row rarely starts
    /// page-aligned — reports fully covered while missing its last page. The
    /// gather then hands the GPU a short buffer, which is a wrong frame.
    #[test]
    fn pages_spanned_counts_the_page_the_offset_pushes_a_span_into() {
        const PAGE: u64 = 4096;
        // Page-aligned: exactly what the length implies.
        assert_eq!(pages_spanned(0, PAGE, PAGE), 1);
        assert_eq!(pages_spanned(PAGE * 7, PAGE * 3, PAGE), 3);
        // Offset by one byte: the same length now reaches one page further.
        assert_eq!(pages_spanned(1, PAGE, PAGE), 2);
        assert_eq!(pages_spanned(PAGE * 7 + 1, PAGE * 3, PAGE), 4);
        // A span wholly inside one page stays at one, wherever it starts.
        assert_eq!(pages_spanned(PAGE - 1, 1, PAGE), 1);
        // …and one byte longer crosses.
        assert_eq!(pages_spanned(PAGE - 1, 2, PAGE), 2);
        // The arm64 pathway's 16 KiB pages take the same rule.
        assert_eq!(pages_spanned(16384 * 3 + 5, 16384, 16384), 2);
    }

    /// The collapse this migration ended: nineteen distinct checks — the walk's
    /// fifteen plus four of its own — all answered `MemError::Unmapped`, and
    /// `MemError` reached the always-on log at no site in the crate. So a
    /// malformed PTE, a zero root PFN, an inactive task and a genuinely unmapped
    /// GPA were one value, invisibly, on the guest-memory hot path.
    ///
    /// Asserted as "no two of these share a slug" rather than by naming each
    /// one, because the property that matters is the absence of aliasing.
    #[test]
    fn no_two_guest_memory_checks_answer_with_the_same_reason() {
        use crate::contract::gva_resolve::ResolveStatus as R;
        const WALK: &[R] = &[
            R::ErrArgs,
            R::ErrInactiveTask,
            R::ErrNoDirectory,
            R::ErrDirectoryRead,
            R::ErrZeroRootPfn,
            R::ErrZeroDepth,
            R::ErrDepthTooDeep,
            R::ErrAddressOutOfRange,
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
        assert_eq!(total, 32, "12 walk reasons + 20 memory reasons");
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
        let mut tasks: [TaskEntry; 4] = std::array::from_fn(|_| TaskEntry::default());
        tasks[1] = TaskEntry::define(0x1000, 2);
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
        assert!(state.define_task(1, 0x1000, 2));
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
        assert!(state.define_task(1, 0x1000, 2));
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
            write_task_gva_product(&mut state, &mut host, 1, page, &[1, 2, 3, 4]).is_ok(),
            "the range is mapped for the writing task, so the write lands \
             whatever the guest has or has not notified"
        );

        // Inside the task's one-level table but with a zero PTE — the fixture
        // installs four data pages, so index 10 resolves to nothing. Chosen
        // over an index past the end of the table because a one-level walk masks
        // its index to the entry count and `4096 * page` aliases index 0, which
        // would have made this assertion pass for the wrong reason.
        assert!(
            write_task_gva_product(&mut state, &mut host, 1, 10 * page, &[1, 2, 3, 4]).is_err(),
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

    /// The trimming must not depend on which host compiled the crate.
    ///
    /// `file!()` carries the building host's separator, so on Windows this
    /// field arrived as the full `crates\...` build path — the log line was
    /// wide enough to push the load-bearing fields off the end, on the rail
    /// where that line is the only evidence. Both spellings are fed in
    /// explicitly so a Unix host still fails if the backslash arm is dropped.
    #[test]
    fn the_via_field_is_trimmed_whichever_separator_the_build_host_used() {
        assert_eq!(
            via_file_tail("/home/u/reims-vgpu/crates/reims-vgpu/src/runtime/gva_mem.rs"),
            "runtime/gva_mem.rs"
        );
        assert_eq!(
            via_file_tail(r"crates\reims-vgpu\src\runtime\gva_mem.rs"),
            "runtime/gva_mem.rs"
        );
        assert_eq!(
            via_file_tail(r"C:\Users\u\reims-vgpu\crates\reims-vgpu\src\runtime\gva_mem.rs"),
            "runtime/gva_mem.rs"
        );
        // The deepest `src/` wins, so a checkout that itself lives under a
        // directory called `src` does not swallow the crate-relative part.
        assert_eq!(
            via_file_tail("/src/reims-vgpu/crates/reims-vgpu/src/runtime/gva_mem.rs"),
            "runtime/gva_mem.rs"
        );
        // Nothing to trim: pass it through rather than emitting an empty field,
        // and still normalise the separator.
        assert_eq!(via_file_tail(r"lib\thing.rs"), "lib/thing.rs");
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
        assert!(state.define_task(2, 0x1_0000, 2));
        assert!(state.define_task(5, 0x1_0000, 9));

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
    /// guard could see: `storage_flush::deferred_pages_still_ours` re-resolves an
    /// armed window through this same function under the same task id, so a
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
        assert!(state.define_task(2, 0x1_0000, 2));
        assert!(state.define_task(5, 0x1_0000, 0));
        assert!(
            state.tasks[5].active,
            "the slot is live; only the page table is missing"
        );

        // The donor really can serve it — otherwise this test would pass for the
        // wrong reason.
        let mut donor = Vec::new();
        visit_task_gva_page_gpas(
            &host,
            &state.tasks,
            2,
            0x1000,
            4,
            PAGE_SHIFT_X86,
            1,
            &mut |gpa| {
                donor.push(gpa);
                true
            },
        );
        assert_eq!(donor, vec![data_gpa], "task 2 resolves GVA page 1");

        let mut pages = Vec::new();
        visit_task_gva_page_gpas(
            &host,
            &state.tasks,
            5,
            0x1000,
            4,
            PAGE_SHIFT_X86,
            1,
            &mut |gpa| {
                pages.push(gpa);
                true
            },
        );
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
        assert!(state.define_task(6, 0x1_0000, 0));
        assert!(
            state.tasks[6].active,
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

    /// A word naming no slot at all still reports `NoSuchTask` — that one IS
    /// the check that refused.
    #[test]
    fn a_fallback_read_for_an_out_of_range_word_still_reports_no_such_task() {
        let host = FakeHost::new();
        let state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let mut buf = [0u8; 4];
        let oob = state.tasks.len() as u32 + 4;
        let err = read_task_gva_by_id(&host, &state.tasks, oob, 0x1000, &mut buf, PAGE_SHIFT_X86)
            .unwrap_err();
        assert_eq!(err, MemError::NoSuchTask);
    }
}
