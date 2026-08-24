//! Task-GVA HostOps views — MapMemory2 / UnmapMemory lifecycle.
//!
//! Apple's host path maps guest pages into a task VA window (`mapMemory`) and
//! tears that mapping down on unmap (`unmapMemory` / guest `CmdUnmapMemory`).
//! Our analogue is a registry of contiguous host-VA views obtained via
//! [`HostOps::map_pages`] after walking the guest task page table for a GVA
//! span. MapMemory2 eagerly materializes stable backend allocations; CPU-only
//! callers can still create a view on demand through [`ensure_gva_view`]. Every
//! view is retired when the guest unmaps or remaps that range.
//!
//! Distinct from:
//! - the mapping materialization's contiguous view — iosfc `mapping_id` page list
//!   (MAP/UNMAP ring)
//! - [`HostReplicaState::gva_surfaces`] — discrete encode cache (retained on Unmap)
//!
//! See [[map-memory2]] GPU-import model and HostOps `map_pages` / `unmap_pages`.

use crate::model::{GvaHostView, TaskEntry, TaskTable};
use crate::runtime::gva_mem::HostPhys;
use crate::runtime::host::{HostMemory, HostOps, MemError};
use crate::runtime::mapper::{RectStride, RunCopy};
use crate::runtime::Device;
use reims_vgpu_paging::resolve::{geometry_for_page_shift, Task};
use reims_vgpu_paging::runs::{contig_page_runs, contig_run_count};
use reims_vgpu_paging::span::span_page_bases;
use std::sync::Arc;

pub(crate) type MappingImportSpan = (Arc<reims_vgpu_memory::GuestRamImport>, u64, u64, Arc<[u64]>);

/// True if half-open ranges `[a, a+la)` and `[b, b+lb)` overlap.
#[inline]
fn ranges_overlap(a: u64, la: u64, b: u64, lb: u64) -> bool {
    if la == 0 || lb == 0 {
        return false;
    }
    let a_end = a.saturating_add(la);
    let b_end = b.saturating_add(lb);
    a < b_end && b < a_end
}

/// Retire every registered GVA view that overlaps `[gva, gva+length)` under `task_id`.
///
/// Emits a typed view-release effect for [`mapper::flush_retired_views`].
/// Does **not** evict `host_gva_surfaces` (encode content is retained across
/// Unmap — a mapping that churns and comes back must not black out the
/// wallpaper); it marks the overlapping entries suspect instead, so the next
/// reader re-walks and finds out whether the GVA still names the same pages.
///
/// Returns the number of views retired. Always-on proxy when `n > 0` is logged by caller.
pub fn retire_gva_views_overlapping(
    state: &mut Device,
    task_id: u32,
    gva: u64,
    length: u64,
) -> u32 {
    if gva == 0 || length == 0 {
        return 0;
    }
    let n = state.host_materializations.retire_gva_views_where(|view| {
        view.task_id == task_id && ranges_overlap(view.gva, view.length, gva, length)
    });
    // The GVA-keyed encode cache survives this deliberately — a mapping that
    // churns and comes back must not black out the wallpaper — and its entries
    // used to be marked "suspect" here so the next reader would re-walk and
    // prove the address still named the same allocation. Nothing does that
    // re-walk any more: the only reader that asked for the proof never ran, so
    // the mark, the flag and the revalidation went with it.
    //
    // What that leaves open is stated where the cache lives: a remap can leave
    // the same virtual address naming a different allocation, and nothing on
    // this rail can currently tell. It is a contract question — what the guest's
    // statement of ownership is for a surface whose pages it has re-pointed —
    // and `gva_backing_moved` in the `cache_levels` line still counts how often
    // the condition arises.
    n
}

/// Find a covering view for `task_id` + `[gva, gva+length)` if one is registered.
fn find_covering_view(state: &Device, task_id: u32, gva: u64, length: u64) -> Option<&GvaHostView> {
    if length == 0 {
        return None;
    }
    // Every command other than task creation carries the task slot directly.
    // A view can therefore be reused only inside the exact task namespace that
    // created it.
    state.host_materializations.find_gva_view(|v| {
        v.task_id == task_id
            && v.gva <= gva
            && gva.saturating_add(length) <= v.gva.saturating_add(v.length)
            && v.host_view.is_some()
    })
}

/// Resolve which task slot to walk. **The wire id, or nothing.**
///
/// The wire id is the slot id. Only `DefineTask2` (`0x38`) carries the doubled
/// form — it registers under `raw >> 1`
/// ([`crate::model::DEFINE_TASK_ID_SHIFT`]) — and every other opcode names the
/// slot directly, so `task_id >> 1` is never the intended task and this does not
/// consider it.
///
/// That used to be an open question, hedged with a fallback to `task_id >> 1`
/// and then with a census counting how often both slots were live. It is closed:
/// the `DefineTask2` wire space is `(slot << 1) | is_kernel_task`, which
/// contains exactly one odd word — `0x1`, the kernel task, whose id is 0 — and
/// is otherwise strictly even. The words the other opcodes were measured
/// receiving include `0x5`, `0x7` and `0x9`, all odd and all greater than one,
/// so they cannot be `DefineTask2` words and are slot ids. Two live slots are
/// therefore not an ambiguity, only a dense table, and there is nothing left for
/// a census to decide.
///
/// A word naming no live slot still refuses rather than landing on a neighbour:
/// slots run densely from 0, so `task_id >> 1` is almost always some *other*
/// live task, and walking it would return that task's bytes on a read and put
/// host bytes at a GPA the named task does not own on a write. Callers turn the
/// `None` into a typed, always-on refusal (`MemError::NoSuchTask`).
fn resolve_task_for_walk(tasks: &TaskTable, task_id: u32) -> Option<(u32, &TaskEntry)> {
    let t = tasks.get(task_id)?;
    (t.active && t.directory_pfn != 0).then_some((task_id, t))
}

/// Collect one GPA per guest page covering `[gva, gva+length)` under the task PT.
///
/// Returns page-aligned GPAs in GVA order. Fails closed on any unmapped page,
/// carrying **which** check refused — the walk's own status when the walk is
/// what said no. Callers that only need "yes or no" take `.ok()`; the guest-write
/// path propagates the reason so the always-on line names it.
fn collect_span_gpas<M: HostMemory>(
    host: &M,
    task: &TaskEntry,
    gva: u64,
    length: u64,
    page_shift: u32,
) -> Result<Vec<u64>, MemError> {
    if length == 0 {
        return Err(MemError::BadArgs);
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
    // Page bases, which is what HostOps `map_pages` takes, and fail-closed,
    // which is what makes them safe to hand it.
    let gpas = span_page_bases(&reader, geom, &gr_task, gva, length)
        .map_err(crate::runtime::gva_mem::span_refusal_error)?;
    if gpas.is_empty() {
        return Err(MemError::BadArgs);
    }
    Ok(gpas)
}

/// Build or reuse a contiguous host-VA view of guest pages for `[gva, gva+length)`.
///
/// Walks the task page table (PPNs already installed by the guest before MapMemory2),
/// then [`crate::runtime::host::HostPageViews::map_pages`]. Returns `(ptr,
/// host_len)`. The host may
/// return a direct RAM run or construct a packed alias over scattered pages;
/// either result preserves task-virtual byte order. Does not invent PTEs.
fn ensure_gva_view<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<(usize, usize)> {
    if gva == 0 || length == 0 {
        return None;
    }
    if let Some(v) = find_covering_view(state, task_id, gva, length) {
        let (vptr, vlen) = (v.ptr(), v.ptr_len());
        let view_task_id = v.task_id;
        let view_gva = v.gva;
        let view_length = v.length;
        let view_is_current = view_gpas_current(host, state, v);
        // Staleness verify, on EVERY reuse: re-translate the view's first/last
        // leaf and compare GPAs. A guest PT rewire that the Unmap/Map2 notifies
        // missed (or that raced ahead of the FIFO) makes the cached view alias
        // pages the guest already recycled — reads through it see freshly
        // zeroed memory, which is the black-tile class. A mismatch retires the
        // view fail-visibly and rebuilds fresh below.
        //
        // This used to run on 1 reuse in 32, which meant a view known to be
        // stale was still handed back for the other 31 — the check could only
        // ever bound how long wrong bytes were served, not prevent them. A
        // sampling rate on a correctness test is a guess about how often the
        // guest rewires under us, and we have no such number from the contract.
        // Two leaf walks is also far cheaper than the full-span walk the cache
        // exists to avoid, so verifying always is what the cache can afford.
        //
        crate::runtime::drain::note_store_route("view_reuse");
        if view_is_current {
            return Some((vptr, vlen));
        }
        crate::runtime::drain::note_store_route("view_stale");
        state.observations.view_stale_reads = state.observations.view_stale_reads.saturating_add(1);
        let n = state.observations.view_stale_reads;
        if n == 1 || n.is_multiple_of(256) {
            crate::observe::fail(format!(
                "gva_view_stale task={} gva={:#x} len={:#x} count={n}",
                view_task_id, view_gva, view_length
            ));
        }
        state.host_materializations.retire_gva_views_where(|view| {
            view.ptr() == vptr && view.gva == view_gva && view.task_id == view_task_id
        });
    }
    // Flush any pending unmaps before allocating a new view (Darwin private VA).
    crate::runtime::mapper::flush_retired_views(state, host);

    let page_shift = state.page_shift;
    let (resolved_tid, gpas) = {
        let (tid, task) = resolve_task_for_walk(&state.tasks, task_id)?;
        let gpas = collect_span_gpas(host, task, gva, length, page_shift).ok()?;
        (tid, gpas)
    };
    let page_sz = state.page_size() as usize;
    // Reject non-RAM leaf GPAs (mapper / wild-PFN class) before map_pages.
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return None;
    }
    let ptr = host.map_pages(&gpas, page_sz)?;
    let page_sz = (1usize) << page_shift;
    let ptr_len = gpas.len().saturating_mul(page_sz);
    let Some(host_view) = crate::model::HostPageView::new(ptr, ptr_len) else {
        host.unmap_pages(ptr, ptr_len);
        return None;
    };
    state
        .host_materializations
        .publish_gva_view(GvaHostView::new(
            resolved_tid,
            gva,
            length,
            host_view,
            Arc::from(gpas),
        ));
    Some((ptr, ptr_len))
}

/// Materialize the allocation published by one task page-table mapping.
///
/// The guest has already installed every page before issuing MapMemory2. A
/// stable host can therefore create and import the exact directed-page alias at
/// this lifetime boundary; later resource views only slice it.
pub fn publish_mapping_import<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> bool {
    if !host.map_pages_stable() || gva == 0 || length == 0 {
        return false;
    }
    let Some((_ptr, ptr_len)) = ensure_gva_view(state, host, task_id, gva, length) else {
        return false;
    };
    let page = state.page_size();
    let (ptr, page_gpas, existing) = {
        let Some(view) = find_covering_view(state, task_id, gva, length) else {
            return false;
        };
        (
            view.ptr(),
            Arc::clone(&view.page_gpas),
            view.import().cloned(),
        )
    };
    if let Some(import) = existing {
        let _ = state
            .executor
            .warm_guest_ram_imports(std::slice::from_ref(&import));
        return true;
    }
    let map_len = ptr_len as u64;
    let ramblock = (contig_run_count(&page_gpas, page) == 1)
        .then(|| {
            crate::runtime::guest_ram_map::reference_for_pages(host, &page_gpas, page, 0, map_len)
                .ok()
        })
        .flatten();
    let (import, import_head) = match ramblock {
        Some(guest) => {
            let import = Arc::clone(guest.import());
            let Some(base) = import.gpa_base() else {
                return false;
            };
            let Some(head) = page_gpas[0].checked_sub(base) else {
                return false;
            };
            (import, head)
        }
        None => {
            let align = match crate::runtime::guest_ram::host_allocation_import_align(map_len) {
                Ok(align) => align,
                Err(refusal) => {
                    crate::runtime::guest_ram::report_host_allocation_import_refusal(
                        "task_mapping_alias_import",
                        &refusal,
                    );
                    return false;
                }
            };
            let Ok(import) =
                crate::runtime::guest_ram::GuestRamImport::new_host_allocation(ptr, map_len, align)
            else {
                return false;
            };
            (Arc::new(import), 0)
        }
    };
    let installed = state
        .host_materializations
        .find_gva_view_mut(|view| {
            view.task_id == task_id
                && view.gva <= gva
                && gva.saturating_add(length) <= view.gva.saturating_add(view.length)
        })
        .is_some_and(|view| {
            view.install_import(Arc::clone(&import), import_head);
            true
        });
    if !installed {
        return false;
    }
    let _ = state
        .executor
        .warm_guest_ram_imports(std::slice::from_ref(&import));
    true
}

/// A retained MapMemory2 allocation covering one resource span.
pub(crate) fn mapping_import_for_span(
    state: &Device,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<MappingImportSpan> {
    let view = find_covering_view(state, task_id, gva, length)?;
    let page_base = view.gva & !(state.page_size() - 1);
    let in_view = gva.checked_sub(page_base)?;
    let offset = view.import_head().checked_add(in_view)?;
    let import = Arc::clone(view.import()?);
    offset
        .checked_add(length)
        .filter(|end| *end <= import.len())?;
    Some((import, offset, page_base, Arc::clone(&view.page_gpas)))
}

/// True when every page still translates to the exact frames the alias owns.
fn view_gpas_current<H: HostMemory>(
    host: &H,
    state: &Device,
    v: &crate::model::GvaHostView,
) -> bool {
    if v.page_gpas.is_empty() {
        return true;
    }
    let Some((_tid, task)) = resolve_task_for_walk(&state.tasks, v.task_id) else {
        return false;
    };
    let page_shift = state.page_shift;
    let page = 1u64 << page_shift;
    let first_page = v.gva & !(page - 1);
    let span = (v.gva - first_page).saturating_add(v.length);
    let Ok(gpas) = collect_span_gpas(host, task, first_page, span, page_shift) else {
        return false;
    };
    gpas.as_slice() == v.page_gpas.as_ref()
}

/// Always-on line when views are retired (proxy for Unmap/Map teardown).
///
/// `op` names the guest operation that retired them, and the key is `op=` rather
/// than `reason=` deliberately: retiring a view on Unmap/Map is *correct*
/// behaviour, not a refusal, so it has no registered decline slug and must not
/// claim one. It was `reason=UnmapMemory` — CamelCase, so
/// `grep 'reason=[a-z_]*'` silently missed it while a reader scanning for
/// refusals found a line that was not one.
pub fn log_retire(op: &str, task_id: u32, gva: u64, length: u64, n: u32) {
    if n == 0 {
        return;
    }
    crate::observe::off(format!(
        "gva_view_drop op={op} task={task_id} gva={gva:#x} len={length:#x} n={n}"
    ));
}

/// Host pointer to the first byte of guest `gva` for a span of `length` bytes.
///
/// Builds/reuses a contig HostOps view over the task page table. Returns
/// `(host_ptr, available_bytes_from_ptr)` covering at least `length`, or `None`
/// if any page is unmapped / non-contiguous on the host.
pub fn host_ptr_for_span<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
) -> Option<(*mut u8, usize)> {
    if gva == 0 || length == 0 {
        return None;
    }
    let (ptr, ptr_len) = ensure_gva_view(state, host, task_id, gva, length)?;
    let page_size = state.page_size();
    let page_mask = page_size - 1;
    // ensure_gva_view maps from the page base of the registered span base.
    // Prefer the covering view's registered gva for offset math.
    let view_gva = find_covering_view(state, task_id, gva, length)
        .map(|v| v.gva)
        .unwrap_or(gva);
    let view_page_base = view_gva & !page_mask;
    let off = (gva.saturating_sub(view_page_base)) as usize;
    if off >= ptr_len {
        return None;
    }
    let avail = ptr_len - off;
    if (avail as u64) < length {
        return None;
    }
    // SAFETY: ensure_gva_view returns ptr for ptr_len host-mapped bytes.
    let p = unsafe { (ptr as *mut u8).add(off) };
    Some((p, avail))
}

/// Pages a deferred write is allowed to reach, or `None` for a write whose
/// authorisation is the command that issued it.
pub type WindowPages<'a> = Option<&'a std::collections::HashSet<u64>>;

/// Every page of a resolved span is one the caller may write.
///
/// The check belongs beside the walk rather than in the caller because the walk
/// that authorises has to be the walk that writes. A caller that walks, checks,
/// and then calls a writer which walks again has proved something about a page
/// table it is no longer using: between the two walks the guest can re-point
/// the range, and the second walk — the one whose answer the bytes actually go
/// to — was never checked. Passing the set down means there is only one walk
/// and its result is both the authorisation and the destination.
///
/// `gpas` are page bases ([`collect_span_gpas`] masks them), so no masking is
/// needed here.
fn span_within_window(gpas: &[u64], allowed: WindowPages<'_>) -> bool {
    match allowed {
        None => true,
        Some(pages) => gpas.iter().all(|g| pages.contains(g)),
    }
}

/// Write `buf` into guest `[gva, gva+buf.len())`, bounded to the pages a
/// deferred window was armed on.
///
/// **Writes never reuse a cached view.** A registered task-GVA host view
/// goes stale the moment the guest rewires its task PT (tile/page recycle)
/// and is only retired when the Unmap/Map2 notify drains — a write through
/// it lands in whatever now owns those host pages (guest heap corruption:
/// the 2026-07-19 WindowServer SIGSEGV class). Every write walks the PT at
/// write time: packed spans map once, fragmented spans multi-import per run.
pub fn write_span_within<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &[u8],
    allowed: WindowPages<'_>,
) -> Result<(), MemError> {
    if buf.is_empty() {
        return Ok(());
    }
    span_multi(state, host, task_id, gva, RunCopy::Write(buf), allowed)
}

/// Write a packed `src` into the guest rectangle at `gva`, bounded to the pages
/// a deferred window was armed on.
///
/// The rectangle is resolved **once** — one task lookup, one page-table walk,
/// one run split — and every row is placed into the runs that walk produced.
/// The alternative this replaces is the caller doing its own `0..row_count`
/// loop over [`write_span_within`], which re-pays all of that per row for a
/// destination it has already described.
///
/// Fragmentation is not a refusal and must not become one: guest linear
/// textures are routinely scattered in guest-physical space, and a rectangle
/// primitive that insisted on a single contiguous run declined **every** blit
/// on a driven macos-13 boot while reading as though it had been installed.
/// The run walk is what makes the general case the fast case.
///
/// Every other property is [`write_span_within`]'s, unchanged and for the same
/// reasons: the walk that authorises is the walk that writes, no cached view is
/// reused, and the pages are recorded as host-written before any byte lands.
pub(crate) fn write_rect_within<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    rect: RectStride,
    src: &[u8],
    allowed: WindowPages<'_>,
) -> Result<(), MemError> {
    let copy = RunCopy::write_rect(src, rect).ok_or(MemError::BadArgs)?;
    span_multi(state, host, task_id, gva, copy, allowed)
}

/// Read the guest rectangle at `gva` into a packed `dst`.
///
/// The read counterpart of [`write_rect_within`], and the same one-walk
/// argument. A read authorises nothing, so it carries no window.
pub(crate) fn read_rect<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    rect: RectStride,
    dst: &mut [u8],
) -> Result<(), MemError> {
    let copy = RunCopy::read_rect(dst, rect).ok_or(MemError::BadArgs)?;
    settle_before_read(
        state,
        host,
        task_id,
        gva,
        copy.len() as u64,
        crate::runtime::render_writeback::SettleSite::GvaRectRead,
    );
    span_multi(state, host, task_id, gva, copy, None)
}

/// Ephemeral fresh-walk host mapping of `[gva, gva+length)` for guest writes.
///
/// Same write-freshness rule as [`write_span_within`]: walks the task PT at call
/// time and maps the packed span without consulting or registering
/// the registered task-GVA view set. The caller must release it with [`unmap_fresh_span`]
/// (product Linux unmap is a no-op alias; Darwin unmaps a real region).
/// Fragmented spans return `None` — callers fall back to their per-row
/// multi-import path, which is also fresh.
pub struct FreshSpan {
    /// First byte of `gva` inside the mapped span.
    pub ptr: *mut u8,
    /// Writable bytes available at `ptr` (>= the requested length).
    pub avail: usize,
    map_base: usize,
    map_len: usize,
}

/// Build a [`FreshSpan`] over `[gva, gva+length)` — fresh PT walk, packed map,
/// bounded to the pages a deferred window was armed on.
///
/// See [`span_within_window`]: the check sits on this walk because this walk is
/// what the returned pointer aliases.
pub fn map_fresh_span_within<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    length: u64,
    allowed: WindowPages<'_>,
) -> Option<FreshSpan> {
    if gva == 0 || length == 0 {
        return None;
    }
    let page_shift = state.page_shift;
    let gpas = {
        let (_tid, task) = resolve_task_for_walk(&state.tasks, task_id)?;
        collect_span_gpas(host, task, gva, length, page_shift).ok()?
    };
    if !span_within_window(&gpas, allowed) {
        return None;
    }
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return None;
    }
    let page_size = state.page_size();
    crate::runtime::mapper::flush_retired_views(state, host);
    let page_sz = page_size as usize;
    let ptr_base = host.map_pages(&gpas, page_sz)?;
    let map_len = gpas.len().saturating_mul(page_sz);
    let off = (gva & (page_size - 1)) as usize;
    if off >= map_len || ((map_len - off) as u64) < length {
        host.unmap_pages(ptr_base, map_len);
        return None;
    }
    // Recorded here, on the requested `[gva, gva+length)`, because the bytes go
    // through the returned pointer in a caller this function never sees. That
    // makes it the one hook in the set that records an *intent* rather than a
    // completed copy: a caller that takes the span and writes less than it asked
    // for leaves frames marked that no byte reached.
    //
    // This is the safe direction and only this direction. Over-marking can only
    // turn a miss into a hit, and a hit is the weak verdict — it never asserts
    // the device wrote somewhere it did not, it only declines to exonerate. The
    // mirror — under-marking a real write — would manufacture the clean "we
    // never wrote there" that this whole set exists to be trusted for.
    //
    let mut remaining = length;
    let mut in_page = (gva & (page_size - 1)) as usize;
    for &page_gpa in &gpas {
        let n = remaining.min((page_size as usize - in_page) as u64);
        crate::observe::footprint::note_written_range(page_gpa + in_page as u64, n);
        remaining -= n;
        if remaining == 0 {
            break;
        }
        in_page = 0;
    }
    // Same reasoning as the footprint mark above, for the other reader of these
    // writes: what this hands back is a writable alias of guest pages, and every
    // caller of it writes. Recorded on the acquisition rather than in each caller
    // because that is where the resolved page list exists, and because a new
    // caller then inherits the record instead of having to remember it.
    state.note_host_wrote_pages(gpas.clone());
    Some(FreshSpan {
        // SAFETY: map_pages returned `map_len` mapped bytes at `ptr_base`.
        ptr: unsafe { (ptr_base as *mut u8).add(off) },
        avail: map_len - off,
        map_base: ptr_base,
        map_len,
    })
}

/// Release a [`map_fresh_span_within`] mapping.
pub fn unmap_fresh_span<H: HostOps>(host: &mut H, span: FreshSpan) {
    host.unmap_pages(span.map_base, span.map_len);
}

/// Order a host read of `[gva, gva+span)` against this device's own submitted
/// GPU writes to the same guest pages.
///
/// The GVA rail hands out a raw `memcpy` over guest RAM. Nothing the GPU knows
/// about orders that copy against a render Store this device has already
/// recorded into the command stream and deliberately not waited on, so a reader
/// that does not settle first reads the pre-Store bytes — silently, and only
/// when the race happens to be lost.
///
/// It sits on the two read entry points rather than in their callers because
/// this is where the span being read is named. A caller cannot reach guest bytes
/// through this module without passing one of them, so a new reader inherits the
/// ordering instead of having to remember it, and there is nothing for a sweep
/// to go looking for.
///
/// Free in the common case: [`settle_guest_writes_unless_disjoint`] returns on a
/// flag read when nothing is outstanding, so the page walk below runs only when
/// this device actually owes writes, and the wait only when they land in pages
/// this read touches.
fn settle_before_read<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    span: u64,
    site: crate::runtime::render_writeback::SettleSite,
) {
    if span == 0 {
        return;
    }
    let (tasks, page_shift, page_size) = (&state.tasks, state.page_shift, state.page_size());
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        state.executor.as_ref(),
        site,
        || {
            let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
            let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
                host, tasks, task_id, gva, span, page_shift,
            );
            (gpas.len() as u64 == want).then_some(gpas)
        },
    );
}

/// Read `buf.len()` bytes from guest `gva` via HostOps map_pages (multi-import).
pub fn read_span<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    settle_before_read(
        state,
        host,
        task_id,
        gva,
        buf.len() as u64,
        crate::runtime::render_writeback::SettleSite::GvaSpanRead,
    );
    if let Some((ptr, avail)) = host_ptr_for_span(state, host, task_id, gva, buf.len() as u64) {
        if avail >= buf.len() {
            // SAFETY: host_ptr_for_span guarantees `avail` readable bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(ptr as *const u8, buf.as_mut_ptr(), buf.len());
            }
            return true;
        }
    }
    let len = buf.len();
    let Err(err) = span_multi(state, host, task_id, gva, RunCopy::Read(buf), None) else {
        return true;
    };
    // A refused read hands its caller a buffer indistinguishable from a
    // successful one, and every caller of this returns a bare `bool` upward, so
    // this is the last place that still holds the reason. Emitted the same way
    // and on the same channel as the write direction's `gva_write` — the two
    // lose the guest's bytes equally, and the read side only looked cheaper
    // because it had nothing to say. Undeduped for the same reason `gva_write`
    // is: the sink's flood detector is what bounds a repeating failure, and a
    // latch here would hide a span that starts refusing after a rewire.
    crate::observe::Emit::decline("gva_read_span", &err)
        .field("task", task_id)
        .field("gva", format!("{gva:#x}"))
        .field("len", format!("{len:#x}"))
        .fail();
    false
}

/// Multi-import span copy: map each packed GPA run, move bytes, unmap. No
/// `write_gpa`/`read_gpa`.
///
/// Ephemeral per-run maps (do not register partial views — Darwin unmap needs
/// the full map_pages base; product Linux alias is a no-op unmap).
///
/// **This is the only implementation of the GVA rail's run walk.** The write and
/// read directions were two functions, ~57 % identical, and the read one had
/// already drifted: it returned a bare `false` at six sites where the write one
/// named six [`MemError`]s — including at `collect_span_gpas`, where it threw
/// away a refusal the walk had already computed. Every one of those loses the
/// caller's bytes, so every one is now named. The `buf_off`/`host_off`/`n`
/// arithmetic and the bound that guards it existed once per direction too, and
/// one of those directions writes guest memory.
///
/// Every refusal names its own check. Fragmentation is **not** one of them:
/// a gapped span is split into packed runs and mapped a run at a time, so a
/// caller reporting "not contiguous" for a failure of this function is reporting
/// a condition the function does not test.
///
/// Three steps are write-only and are keyed on the direction, not duplicated:
/// the [`WindowPages`] containment check, which only a deferred write carries an
/// authorisation set for; [`Device::note_host_wrote_pages`], which
/// invalidates retained derived content reached through another alias; and the
/// per-run footprint mark. `allowed` is ignored in the read direction because a
/// read authorises nothing.
fn span_multi<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    gva: u64,
    mut copy: RunCopy<'_>,
    allowed: WindowPages<'_>,
) -> Result<(), MemError> {
    let length = copy.len() as u64;
    let page_shift = state.page_shift;
    let page_size = state.page_size();
    let page_sz = page_size as usize;
    let gpas = {
        let Some((_tid, task)) = resolve_task_for_walk(&state.tasks, task_id) else {
            return Err(MemError::NoSuchTask);
        };
        collect_span_gpas(host, task, gva, length, page_shift)?
    };
    if copy.is_write() {
        // Record the device write after the walk names its exact pages and before any
        // byte is written, so a refusal below costs a spurious invalidation
        // rather than a missing one.
        state.note_host_wrote_pages(gpas.clone());
        if !span_within_window(&gpas, allowed) {
            return Err(MemError::WriteOutsideWindow);
        }
    }
    if gpas.iter().any(|&g| !host.is_ram_gpa(g)) {
        return Err(MemError::NotRam);
    }
    let runs = contig_page_runs(&gpas, page_size);
    if runs.is_empty() {
        return Err(MemError::BadArgs);
    }
    crate::runtime::mapper::flush_retired_views(state, host);
    let span_page_base = gva & !(page_size - 1);
    let end = gva.saturating_add(length);
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            return Err(MemError::MapPagesRefused);
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let run_gva = span_page_base.saturating_add((run.start as u64).saturating_mul(page_size));
        let run_end = run_gva.saturating_add(total as u64);
        let copy_lo = gva.max(run_gva);
        let copy_hi = end.min(run_end);
        if copy_lo >= copy_hi {
            host.unmap_pages(ptr, total);
            continue;
        }
        let buf_off = (copy_lo - gva) as usize;
        let host_off = (copy_lo - run_gva) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > copy.len() {
            host.unmap_pages(ptr, total);
            return Err(MemError::RunOutOfRange);
        }
        // SAFETY: map_pages packed `total` bytes, and the bound above puts
        // `host_off + n` inside it and `buf_off + n` inside the caller's buffer.
        unsafe { copy.apply(ptr, host_off, buf_off, n) };
        if copy.is_write() {
            // A run is packed by construction, so the `n` bytes at `host_off`
            // are the `n` bytes at `run_gpas[0] + host_off` in guest-physical
            // space — the exact destination, not the run's hull.
            crate::observe::footprint::note_written_range(
                run_gpas[0].saturating_add(host_off as u64),
                n as u64,
            );
        }
        host.unmap_pages(ptr, total);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;
    #[cfg(not(target_os = "macos"))]
    use crate::runtime::host::HostPageViews;
    use reims_vgpu_core::endian::st32;
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use reims_vgpu_paging::resolve::ResolveStatus;

    fn state_x86() -> Device {
        Device::new(DeviceId(1), 12)
    }

    #[test]
    fn contig_page_runs_splits_gaps() {
        let page = 0x1000u64;
        let gpas = [0x1000u64, 0x2000, 0x4000, 0x5000, 0x8000];
        let runs = contig_page_runs(&gpas, page);
        assert_eq!(runs, vec![0..2, 2..4, 4..5]);
        assert_eq!(contig_page_runs(&[0x1000], page), vec![0..1]);
        assert!(contig_page_runs(&[], page).is_empty());
    }

    /// Counting and constructing physical runs must classify the same list.
    #[test]
    fn contig_run_count_agrees_with_the_runs_it_does_not_build() {
        let page = 0x1000u64;
        let shapes: &[&[u64]] = &[
            &[],
            &[0x1000],
            &[0x1000, 0x2000, 0x3000],
            &[0x1000, 0x3000],
            &[0x1000, 0x2000, 0x4000, 0x5000, 0x8000],
            // Descending and repeated GPAs are breaks like any other.
            &[0x3000, 0x2000, 0x1000],
            &[0x1000, 0x1000],
        ];
        for shape in shapes {
            assert_eq!(
                contig_run_count(shape, page),
                contig_page_runs(shape, page).len(),
                "shape {shape:x?}"
            );
        }
        let fragmented: Vec<u64> = (0..2040u64)
            .map(|i| (i / 4) * 0x8000 + (i % 4) * page)
            .collect();
        assert_eq!(contig_run_count(&fragmented, page), 510);
        assert_eq!(
            contig_run_count(&fragmented, page),
            contig_page_runs(&fragmented, page).len()
        );
        // `page_size == 0` is the no-runs guard, not a one-run answer.
        assert_eq!(contig_run_count(&[0x1000, 0x2000], 0), 0);
        assert_eq!(
            contig_run_count(&[0x1000, 0x2000], 0),
            contig_page_runs(&[0x1000, 0x2000], 0).len()
        );
    }

    /// Physical scattering does not change the task-virtual byte order.
    #[test]
    fn fragmented_gva_view_asks_the_host_for_a_packed_alias() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        // Wire PTE[1] → data1 (pfn 10) so the two-page span is gapped.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        let mut state = state_x86();
        state.define_task(1, page, 2);

        let before = host.map_pages_calls;
        assert!(ensure_gva_view(&mut state, &mut host, 1, page - 4, 8).is_some());
        assert_eq!(host.map_pages_calls, before + 1);

        // The packed case still goes to the host and still resolves.
        let before = host.map_pages_calls;
        assert!(ensure_gva_view(&mut state, &mut host, 1, 8, 4).is_some());
        assert_eq!(host.map_pages_calls, before + 1);
    }

    #[test]
    fn map_memory_materializes_once_and_resource_views_slice_its_import() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        host.stable_map_pages = true;
        reims_vgpu_memory::latch_import_limits(1, 1 << 30, 1 << 30);

        let mut state = state_x86();
        state.define_task(1, page, 2);
        assert!(publish_mapping_import(
            &mut state,
            &mut host,
            1,
            page - 4,
            8,
        ));
        let mapped_calls = host.map_pages_calls;
        assert!(crate::runtime::bound_buffers::ensure_packed_resource(
            &mut state,
            &mut host,
            1,
            7,
            page - 4,
            8,
            crate::runtime::bound_buffers::PackedResourceUse::Buffer,
        ));
        assert_eq!(
            host.map_pages_calls, mapped_calls,
            "a resource inside the mapping slices the retained allocation"
        );
        assert!(state.bound_buffers.packed(1, 7).is_some());

        assert_eq!(retire_gva_views_overlapping(&mut state, 1, page - 4, 8), 1);
        crate::runtime::mapper::flush_retired_views(&mut state, &mut host);
        reims_vgpu_memory::forget_import_limits();
    }

    /// A cached view's reuse verify has to check every page it aliases, not the
    /// two a partial remap is least likely to move.
    ///
    /// The guest recycles allocations. One that keeps its head and its tail
    /// while its middle goes somewhere else passes an ends-only check forever,
    /// and every read through the middle of that view returns another owner's
    /// bytes -- freshly zeroed memory, if the guest has not filled them yet,
    /// which is a blank tile on screen rather than a crash.
    ///
    /// The view here is three pages so it HAS a middle. Moving only the middle
    /// page is the exact case the ends-only form could not see.
    #[test]
    fn a_view_reuse_verify_checks_every_page_it_aliases() {
        let page_shift = PAGE_SHIFT_X86;
        let page = 1u64 << page_shift;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        // Virtual page i maps to frame 4+i; the view is virtual pages 1..4, so
        // frames 5,6,7. Frame 9 is the allocation the guest moves to. Virtual
        // page 0 is left out of the view because gva 0 is not a view address.
        for pfn in [4u64, 5, 6, 7, 9] {
            host.map_range(pfn << page_shift, page as usize, 0);
        }
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        for i in 0..4u32 {
            st32(&mut pte, 4 + i);
            host.write_gpa(root_gpa + (i as u64) * 4, &pte).unwrap();
        }
        let mut state = state_x86();
        state.define_task(1, page, 2);

        let (ptr, ptr_len) = ensure_gva_view(&mut state, &mut host, 1, page, 3 * page)
            .expect("three packed pages map as one view");
        assert_eq!(ptr_len as u64, 3 * page);
        let view = crate::model::GvaHostView::new(
            1,
            page,
            3 * page,
            crate::model::HostPageView::new(ptr, ptr_len).expect("mapped test view"),
            Arc::from([5 << page_shift, 6 << page_shift, 7 << page_shift]),
        );
        assert!(
            view_gpas_current(&host, &state, &view),
            "nothing has moved yet"
        );

        // The guest re-points the MIDDLE page only -- virtual page 2, frame 6.
        // Head and tail are untouched, which is all an ends-only check looked
        // at.
        st32(&mut pte, 9);
        host.write_gpa(root_gpa + 8, &pte).unwrap();
        assert!(
            !view_gpas_current(&host, &state, &view),
            "the view now aliases frame 9 for its middle page and called itself current"
        );

        // A page dropping out of the walk is a change too: the view keeps
        // aliasing three frames whatever the page table says.
        st32(&mut pte, 6);
        host.write_gpa(root_gpa + 8, &pte).unwrap();
        assert!(view_gpas_current(&host, &state, &view), "restored");
        st32(&mut pte, 0);
        host.write_gpa(root_gpa + 12, &pte).unwrap();
        assert!(
            !view_gpas_current(&host, &state, &view),
            "the last page no longer resolves and the view still spans it"
        );
    }

    #[test]
    fn ranges_overlap_basic() {
        assert!(ranges_overlap(0x1000, 0x1000, 0x1800, 0x1000));
        assert!(!ranges_overlap(0x1000, 0x1000, 0x2000, 0x1000));
        assert!(!ranges_overlap(0x1000, 0, 0x1000, 0x1000));
    }

    /// A host that refuses a packed alias still has the per-run write path.
    #[test]
    fn multi_import_fragmented_gva_write() {
        let page_shift = PAGE_SHIFT_X86;
        let page = 1u64 << page_shift;
        let mut host = FakeHost::new();
        host.strict_linux_map = true;
        // dir pfn=2, root pfn=3, data page0 pfn=4, data page1 pfn=10 (gap).
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data0 = 4u64 << page_shift;
        let data1 = 10u64 << page_shift;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        host.map_range(data0, page as usize, 0);
        host.map_range(data1, page as usize, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();

        #[cfg(not(target_os = "macos"))]
        {
            // Full span map of [data0, data1] must fail under strict Linux semantics.
            assert!(
                host.map_pages(&[data0, data1], page as usize).is_none(),
                "strict map must reject non-packed GPA list"
            );
        }

        let mut state = state_x86();
        state.define_task(1, page, 2);
        let payload = [0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        // Write 4 bytes at end of page0 + 4 at start of page1 (crosses gap).
        let gva = page - 4;
        assert!(
            crate::runtime::gva_mem::write_task_gva_product_within(
                &mut state, &mut host, 1, gva, &payload, None
            )
            .is_ok(),
            "multi-import product write must succeed across fragmented PFNs"
        );
        let mut back = [0u8; 8];
        assert!(host.read_gpa(data0 + page - 4, &mut back[..4]).is_ok());
        assert!(host.read_gpa(data1, &mut back[4..]).is_ok());
        assert_eq!(back, payload);
    }

    /// PT fixture: dir pfn 2 (root pfn 3, depth 1), PTE[0] → data0 (pfn 4).
    /// data1 (pfn 10) is mapped but initially unreferenced by the PT.
    fn pt_fixture(page_shift: u32) -> (FakeHost, u64, u64, u64, u64) {
        let page = 1u64 << page_shift;
        let mut host = FakeHost::new();
        let dir_gpa = 2u64 << page_shift;
        let root_gpa = 3u64 << page_shift;
        let data0 = 4u64 << page_shift;
        let data1 = 10u64 << page_shift;
        host.map_range(dir_gpa, page as usize, 0);
        host.map_range(root_gpa, page as usize, 0);
        host.map_range(data0, page as usize, 0);
        host.map_range(data1, page as usize, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        host.write_gpa(dir_gpa, &d).unwrap();
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        host.write_gpa(root_gpa, &pte).unwrap();
        (host, root_gpa, data0, data1, page)
    }

    /// The raw-GVA rail's writes reach `observe::footprint`, at the guest
    /// physical address the walk resolved and nowhere else.
    ///
    /// The rail's whole shape is that the destination is not knowable from the
    /// call — it is whatever the task's page table says at write time — so the
    /// footprint has to be marked from the *resolved* GPA. Marking from the GVA
    /// instead would be the same substitution the surface backing identity guard exists
    /// to refuse, and it would fill the set with addresses in low guest RAM,
    /// which is exactly where the panic census's victims live. Every such frame
    /// would then read as a hit.
    #[test]
    fn a_raw_gva_write_marks_the_resolved_gpa_and_not_the_virtual_address() {
        use crate::observe::footprint;

        let _fp = footprint::exclusive_for_tests();
        let (mut host, _root_gpa, data0, data1, page) = pt_fixture(PAGE_SHIFT_X86);
        let mut state = state_x86();
        state.define_task(1, page, 2);

        let gva = 0x100u64;
        write_span_within(&mut state, &mut host, 1, gva, &[0x7Eu8; 64], None)
            .expect("the walk resolves");

        assert!(
            footprint::wrote_gpa(data0),
            "the frame the page table named must be marked"
        );
        assert!(
            !footprint::wrote_gpa(gva),
            "the guest VIRTUAL address is not a physical frame; marking it would \
             put low guest RAM — where the panic census's victims live — into the \
             set on every write"
        );
        assert!(
            !footprint::wrote_gpa(data1),
            "the page the guest did not point at is not ours"
        );
        assert_eq!(footprint::counts(), (1, 0));
    }

    /// A deferred write is bounded by the pages it was armed on, and the bound
    /// is enforced by the same walk that produces the destination.
    ///
    /// The scenario is the one the guest actually produces: a window armed on
    /// `data0`, and by flush time the guest has re-pointed that GVA at `data1`.
    /// A guard that walked separately and then called an unbounded writer would
    /// have to win a race against the guest's own vCPUs to catch this; a bound
    /// carried into the walk cannot lose it, because there is only one walk.
    ///
    /// Both writers are covered — `write_span_within` for the fragmented
    /// per-row path, `map_fresh_span_within` for the packed one — because
    /// which of the two a real Store takes depends on how the guest happened to
    /// lay the pages out, and a bound present on only one of them is a bound
    /// that holds on some machines.
    #[test]
    fn a_deferred_write_cannot_reach_a_page_its_window_was_not_armed_on() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        let armed: std::collections::HashSet<u64> = [data0].into_iter().collect();

        // Control: while the GVA still resolves inside the window, the write
        // lands. A bound that refused everything would silently blank the guest.
        assert!(
            write_span_within(&mut state, &mut host, 1, 8, &[0xaa; 4], Some(&armed)).is_ok(),
            "a window writing its own page must still write"
        );
        let mut back = [0u8; 4];
        host.read_gpa(data0 + 8, &mut back).unwrap();
        assert_eq!(back, [0xaa; 4]);
        assert!(map_fresh_span_within(&mut state, &mut host, 1, 8, 16, Some(&armed)).is_some());

        // The guest re-points the range. Nothing about the walk is unhealthy —
        // it resolves, it is RAM, it is current. It is simply not ours.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();

        assert_eq!(
            write_span_within(&mut state, &mut host, 1, 8, &[0xbb; 4], Some(&armed)),
            Err(crate::runtime::host::MemError::WriteOutsideWindow),
            "a re-pointed range is another owner's memory, not this window's"
        );
        assert!(
            map_fresh_span_within(&mut state, &mut host, 1, 8, 16, Some(&armed)).is_none(),
            "the packed path must carry the same bound as the per-row path"
        );
        host.read_gpa(data1 + 8, &mut back).unwrap();
        assert_eq!(back, [0u8; 4], "the new owner's page must be untouched");

        // Unbounded callers are unaffected: a synchronous Store's authorisation
        // is the page table at the moment it runs, and it passes `None`.
        assert!(
            write_span_within(&mut state, &mut host, 1, 8, &[0xbb; 4], None).is_ok(),
            "an unbounded write must keep following the live page table"
        );
        host.read_gpa(data1 + 8, &mut back).unwrap();
        assert_eq!(back, [0xbb; 4]);
    }

    /// Read the always-on log from `from` to the end.
    fn log_tail(from: usize) -> String {
        let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        body[from.min(body.len())..].to_string()
    }

    /// Start capturing the always-on log; returns the offset to slice from.
    fn log_mark() -> usize {
        crate::observe::redirect_logs_for_tests();
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    }

    /// A host view belongs to exactly one task address space.
    #[test]
    fn a_view_registered_by_another_task_does_not_satisfy_this_tasks_lookup() {
        let mut state = state_x86();
        state
            .host_materializations
            .publish_gva_view(crate::model::GvaHostView::fixture(
                20, 0x1000, 0x1000, 0x4000, 0x1000,
            ));

        assert!(find_covering_view(&state, 20, 0x1000, 0x10).is_some());
        assert!(
            find_covering_view(&state, 41, 0x1000, 0x10).is_none(),
            "a view registered by task 20 must not satisfy a lookup for task 41"
        );
    }

    /// A live `task_id >> 1` is a dense task table, not a second reading of the
    /// word, and must not perturb the walk.
    ///
    /// Only `DefineTask2` carries the doubled form; the opcodes that later name
    /// a task carry slot ids. So with slots 9 and 4 both live, a write naming 9
    /// walks slot 9 — silently, because there is nothing to report. This used to
    /// emit a `task_walk_ambiguous` census on the reading that 4 might be what
    /// the guest meant.
    #[test]
    fn a_live_shifted_slot_does_not_perturb_the_walk() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, _root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(4, page, 2);
        state.define_task(9, page, 2);
        let before = log_mark();
        assert!(write_span_within(&mut state, &mut host, 9, 8, &[1, 2, 3, 4], None).is_ok());
        let tail = log_tail(before);
        assert!(
            !tail.contains("task_walk "),
            "two live slots are an ordinary table, so the walk must say nothing: {tail}"
        );
    }

    /// A write naming a slot that is not live must **refuse**, not silently walk
    /// `task_id >> 1`'s page tables.
    ///
    /// Slot 9 is not defined here and slot 4 is. The deleted fallback would have
    /// resolved this write through task 4 and landed guest bytes at GPAs task 9
    /// does not own — and it would have looked like success. Slots run densely
    /// from 0 on the real rail, so `>> 1` almost always finds *some* live task;
    /// that is why the arm could never fail safely and why refusing is the whole
    /// point of removing it.
    #[test]
    fn a_write_naming_a_dead_slot_refuses_instead_of_walking_its_neighbour() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, _root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(4, page, 2);
        let before = log_mark();
        let err = write_span_within(&mut state, &mut host, 9, 8, &[1, 2, 3, 4], None)
            .expect_err("slot 9 is not live, so this write has no address space");
        assert_eq!(err, MemError::NoSuchTask);
        let tail = log_tail(before);
        assert!(
            !tail.contains("task_walk_ambiguous"),
            "one live slot is not an ambiguous wire word: {tail}"
        );
    }

    /// Guest writes must land where the PT points **now**, not where a
    /// registered view pointed when it was built (stale-view heap-corruption
    /// class — the guest recycles pages before the Unmap notify drains).
    #[test]
    fn write_span_ignores_stale_registered_view() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        let gva = 8u64;
        assert!(write_span_within(&mut state, &mut host, 1, gva, &[1, 2, 3, 4], None).is_ok());
        let mut back = [0u8; 4];
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);

        // Register a view over the span (as a read would), then rewire the
        // PTE to data1 WITHOUT any Unmap notify — the view is now stale.
        let (vptr, vlen) = ensure_gva_view(&mut state, &mut host, 1, gva, 4).unwrap();
        assert!(vptr != 0 && vlen != 0);
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();

        assert!(write_span_within(&mut state, &mut host, 1, gva, &[5, 6, 7, 8], None).is_ok());
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back, [5, 6, 7, 8], "write must follow the live PT");
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4], "stale page must not be touched");
    }

    /// The product guest-write path must name the check that refused, not
    /// assume contiguity. [`span_multi`] has six distinct refusals — no
    /// task, an unresolved page, a non-RAM GPA, an empty run list, a `map_pages`
    /// refusal and an out-of-range run window — and every one of them used to
    /// reach the always-on log as `mem_not_contiguous`. That is the same "one
    /// status for N checks" collapse that
    /// `no_two_guest_memory_checks_answer_with_the_same_reason` ended for the
    /// read paths, still standing on the write path.
    ///
    /// Here the span's second page has no PTE, so the walk refuses and
    /// contiguity is never even in question.
    #[test]
    fn product_gva_write_reports_the_check_that_refused() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, _root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        crate::observe::redirect_logs_for_tests();
        let before = std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len();
        // pt_fixture wires PTE[0] only, so page 1 of this two-page span is
        // unresolved. `page - 4` straddles the boundary.
        assert!(crate::runtime::gva_mem::write_task_gva_product_within(
            &mut state,
            &mut host,
            1,
            page - 4,
            &[0u8; 8],
            None
        )
        .is_err());
        let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        let tail = &body[before.min(body.len())..];
        let line = tail
            .lines()
            .find(|l| l.starts_with("gva_write "))
            .expect("a refused product guest write is always-on");
        assert!(
            line.contains("reason=gva_zero_pfn"),
            "the walk's own check must be the reason: {line}"
        );
    }

    /// A host read of guest pages is ordered against this device's own
    /// submitted-but-unexecuted GPU writes to those same pages.
    ///
    /// The GVA rail is a raw `memcpy` over guest RAM, and nothing the GPU knows
    /// about orders it against a render Store already recorded into the command
    /// stream. Without the settle this reads the pre-Store bytes — silently, and
    /// only on the boots where the race is lost.
    ///
    /// Both arms are here on purpose. Asserting only that a wait happens would
    /// pass on a rail that waits unconditionally, which is a different bug: the
    /// disjoint arm is what says the ordering is taken because the writes reach
    /// these pages and not because every read blocks.
    #[test]
    fn a_host_read_of_guest_pages_settles_the_writes_that_reach_them() {
        use crate::runtime::executor::*;
        use reims_vgpu_core::{
            CapabilityService, ComputeResidencyService, ExecutionPort, GuestWriteReach,
            PresentationService, ReadbackService, ResidentService,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        #[derive(Debug)]
        struct SettleProbe {
            reach: GuestWriteReach,
            quiesced: AtomicUsize,
        }

        impl ExecutionPort for SettleProbe {
            type Submission = ResolvedSubmission;
            type Completion = ExecutionCompletion;
            type Error = DrawError;

            fn execute(
                &self,
                _submission: Self::Submission,
            ) -> Result<Self::Completion, Self::Error> {
                unreachable!("this test executes no command buffer")
            }
        }
        impl reims_vgpu_core::GuestWriteService for SettleProbe {
            fn guest_writes_outstanding(&self) -> bool {
                true
            }
            fn guest_writes_reaching(&self, _pages: &[u64]) -> GuestWriteReach {
                self.reach
            }
            fn quiesce_guest_writes(&self) {
                self.quiesced.fetch_add(1, Ordering::AcqRel);
            }
        }
        impl ResidentService for SettleProbe {}
        impl ComputeResidencyService for SettleProbe {}
        impl CapabilityService for SettleProbe {}
        impl PresentationService for SettleProbe {}
        impl ReadbackService for SettleProbe {
            type Error = DrawError;

            fn read_target(
                &self,
                _identity: &crate::model::TargetIdentity,
            ) -> Result<reims_vgpu_core::TargetReadback, Self::Error> {
                unreachable!("this test reads no target")
            }
        }
        impl GuestPageTransferService for SettleProbe {}
        impl ResidentCopyService for SettleProbe {}
        impl CompletionService for SettleProbe {}
        impl SubmissionBatchService for SettleProbe {}
        impl GuestImportService for SettleProbe {}
        impl MaintenanceService for SettleProbe {}
        impl SessionService for SettleProbe {}
        impl ObservationService for SettleProbe {}
        impl ShaderTranslationService for SettleProbe {}
        impl RenderBufferPlanningService for SettleProbe {}
        impl GuestImagePlanningService for SettleProbe {}
        impl WindowPresentationService for SettleProbe {}
        impl Executor for SettleProbe {}

        // One page of the fixture's data0, read both ways. The rect is strided
        // so its guest reach is the span the walk resolves and not the packed
        // length the caller receives.
        for (reach, expected) in [
            (GuestWriteReach::Overlap, 2),
            (GuestWriteReach::Disjoint, 0),
        ] {
            let (mut host, _root, _data0, _data1, page) = pt_fixture(PAGE_SHIFT_X86);
            let mut state = state_x86();
            state.define_task(1, page, 2);
            let probe = Arc::new(SettleProbe {
                reach,
                quiesced: AtomicUsize::new(0),
            });
            state.executor = probe.clone();

            let mut buf = [0u8; 32];
            assert!(
                read_span(&mut state, &mut host, 1, 0x100, &mut buf),
                "the fixture's first page resolves"
            );
            let rect = crate::runtime::mapper::RectStride::new(16, 8, 4).expect("a valid rect");
            read_rect(&mut state, &mut host, 1, 0x100, rect, &mut buf[..32])
                .expect("the fixture's first page resolves");

            assert_eq!(
                probe.quiesced.load(Ordering::Acquire),
                expected,
                "{reach:?}: a host read of guest pages must settle exactly when \
                 this device's outstanding writes reach them"
            );
        }
    }

    /// The read direction must name the check that refused, exactly as the
    /// write direction does.
    ///
    /// It did not: `read_span_multi` returned a bare `false` at six sites where
    /// its write twin named a [`MemError`], and at the walk it threw away a
    /// refusal that had already been computed. Every caller of [`read_span`]
    /// returns a `bool` upward, so those bytes went missing with nothing in the
    /// log to say why. This is the read-side mirror of
    /// [`product_gva_write_reports_the_check_that_refused`], on the same fixture
    /// and the same straddling span, and it fails on any tree where the read
    /// walk drops the reason again.
    #[test]
    fn product_gva_read_reports_the_check_that_refused() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, _root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        crate::observe::redirect_logs_for_tests();
        let before = std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len();
        // pt_fixture wires PTE[0] only, so page 1 of this two-page span is
        // unresolved. `page - 4` straddles the boundary.
        let mut buf = [0u8; 8];
        assert!(
            !read_span(&mut state, &mut host, 1, page - 4, &mut buf),
            "a span whose second page has no PTE cannot be read"
        );
        let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        let tail = &body[before.min(body.len())..];
        let line = tail
            .lines()
            .find(|l| l.starts_with("gva_read_span "))
            .expect("a refused product guest read is always-on");
        assert!(
            line.contains("reason=gva_zero_pfn"),
            "the walk's own check must be the reason: {line}"
        );
    }

    /// The three refusals [`span_multi`] owns must stay distinguishable
    /// from each other and from the walk's. Asserted as "no two share a slug"
    /// rather than by naming each, because the property that matters is the
    /// absence of aliasing — the same shape
    /// `no_two_guest_memory_checks_answer_with_the_same_reason` asserts crate-wide.
    #[test]
    fn write_span_refusals_do_not_share_a_reason() {
        use crate::observe::Decline;
        let mut slugs: Vec<&str> = [
            MemError::NoSuchTask,
            MemError::NotRam,
            MemError::MapPagesRefused,
            MemError::RunOutOfRange,
            MemError::BadArgs,
            MemError::Unresolved(ResolveStatus::ErrZeroPfn),
        ]
        .iter()
        .map(|e| e.slug())
        .collect();
        let total = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), total, "two write-span refusals share a slug");
    }

    /// A gapped span is the one thing the multi-import path is *for*, so it must
    /// succeed — the reason it can no longer be reported as is now impossible to
    /// name. Guards the direction the vocabulary change could have gone wrong in:
    /// renaming the refusal without checking fragmentation still works.
    #[test]
    fn fragmented_write_succeeds_so_no_refusal_can_mean_fragmented() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        // PTE[1] → pfn 10, leaving a gap after PTE[0]'s pfn 4.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        let mut state = state_x86();
        state.define_task(1, page, 2);
        assert!(
            write_span_within(
                &mut state,
                &mut host,
                1,
                page - 4,
                &[1, 2, 3, 4, 5, 6, 7, 8],
                None
            )
            .is_ok(),
            "the multi-import path exists to write a gapped span"
        );
        let mut back = [0u8; 4];
        host.read_gpa(data0 + page - 4, &mut back).unwrap();
        assert_eq!(back, [1, 2, 3, 4]);
        host.read_gpa(data1, &mut back).unwrap();
        assert_eq!(back, [5, 6, 7, 8]);
    }

    /// A fresh writable view preserves task-virtual order across scattered
    /// physical pages.
    #[test]
    fn fragmented_map_fresh_span_uses_one_packed_host_view() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, _data0, _data1, page) = pt_fixture(page_shift);
        // Wire PTE[1] → data1 (pfn 10) so the two-page span is gapped.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa + 4, &pte).unwrap();
        let mut state = state_x86();
        state.define_task(1, page, 2);

        let before = host.map_pages_calls;
        let span = map_fresh_span_within(&mut state, &mut host, 1, page - 4, 8, None)
            .expect("the host packs the directed page list");
        assert_eq!(host.map_pages_calls, before + 1);
        unsafe { std::ptr::copy_nonoverlapping([1u8; 8].as_ptr(), span.ptr, 8) };
        unmap_fresh_span(&mut host, span);

        // The packed case still goes to the host and still resolves.
        let before = host.map_pages_calls;
        let s =
            map_fresh_span_within(&mut state, &mut host, 1, 8, 4, None).expect("packed span maps");
        assert!(s.avail >= 4);
        unmap_fresh_span(&mut host, s);
        assert_eq!(host.map_pages_calls, before + 1);
    }

    /// map_fresh_span re-walks per call: after a PT rewire, writes through
    /// the returned pointer land in the newly wired page.
    #[test]
    fn map_fresh_span_follows_pt_rewire() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        let gva = 8u64;
        let s = map_fresh_span_within(&mut state, &mut host, 1, gva, 16, None).unwrap();
        assert!(s.avail >= 16);
        // SAFETY: map_fresh_span guarantees ≥16 writable bytes at ptr.
        unsafe { std::ptr::copy_nonoverlapping([0xaau8; 4].as_ptr(), s.ptr, 4) };
        unmap_fresh_span(&mut host, s);
        let mut back = [0u8; 4];
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [0xaa; 4]);

        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();
        let s = map_fresh_span_within(&mut state, &mut host, 1, gva, 16, None).unwrap();
        // SAFETY: as above.
        unsafe { std::ptr::copy_nonoverlapping([0xbbu8; 4].as_ptr(), s.ptr, 4) };
        unmap_fresh_span(&mut host, s);
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back, [0xbb; 4], "fresh span must follow the rewired PT");
        host.read_gpa(data0 + gva, &mut back).unwrap();
        assert_eq!(back, [0xaa; 4], "old page must not see the second write");
    }

    /// Every reuse verifies, so a PT rewire under a cached view is caught on
    /// the FIRST read after it happens — the view is retired and rebuilt, and
    /// no read is ever served through the stale mapping.
    #[test]
    fn stale_covering_view_detected_and_rebuilt() {
        let page_shift = PAGE_SHIFT_X86;
        let (mut host, root_gpa, data0, data1, page) = pt_fixture(page_shift);
        let mut state = state_x86();
        state.define_task(1, page, 2);
        let gva = 8u64;
        let (p0, _) = ensure_gva_view(&mut state, &mut host, 1, gva, 16).unwrap();
        assert_eq!(state.host_materializations.views().len(), 1);
        assert_eq!(
            state.host_materializations.views()[0].page_gpas.as_ref(),
            &[data0]
        );

        // Rewire the PTE (no Unmap notify) — the registered view is stale.
        let mut pte = [0u8; 4];
        st32(&mut pte, 10);
        host.write_gpa(root_gpa, &pte).unwrap();

        // The very next reuse detects, retires, and rebuilds. Under the old
        // 1-in-32 gate this same call returned `p0` — the stale mapping — and
        // left `view_stale_reads` at 0.
        let (p2, _) = ensure_gva_view(&mut state, &mut host, 1, gva, 16).unwrap();
        assert_ne!(p2, p0, "a stale view must never be handed back");
        assert_eq!(state.observations.view_stale_reads, 1);
        assert_eq!(state.host_materializations.views().len(), 1);
        assert_eq!(
            state.host_materializations.views()[0].page_gpas.as_ref(),
            &[data1]
        );
        // Writes through the rebuilt view land in the newly wired page.
        // SAFETY: ensure_gva_view mapped the page; gva page offset is 8.
        unsafe { *((p2 as *mut u8).add(gva as usize)) = 0xcc };
        let mut back = [0u8; 1];
        host.read_gpa(data1 + gva, &mut back).unwrap();
        assert_eq!(back[0], 0xcc);
    }

    #[test]
    fn unmap_retires_overlapping_view_only() {
        let mut state = state_x86();
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(2, 0x1000, 0x2000, 0xaaaa, 0x2000));
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(2, 0x10000, 0x1000, 0xbbbb, 0x1000));
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(3, 0x1000, 0x2000, 0xcccc, 0x2000));

        let n = retire_gva_views_overlapping(&mut state, 2, 0x1500, 0x100);
        assert_eq!(n, 1);
        assert_eq!(state.host_materializations.views().len(), 2);
        assert!(state
            .host_materializations
            .views()
            .iter()
            .any(|v| v.ptr() == 0xbbbb && v.task_id == 2));
        assert!(state
            .host_materializations
            .views()
            .iter()
            .any(|v| v.ptr() == 0xcccc && v.task_id == 3));
        assert_eq!(
            state.host_materializations.queued_views(),
            vec![(0xaaaa, 0x2000)]
        );
    }

    #[test]
    fn unmap_does_not_cross_task_namespaces() {
        let mut state = state_x86();
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(1, 0x2000, 0x1000, 0x1111, 0x1000));
        let n = retire_gva_views_overlapping(&mut state, 2, 0x2000, 0x1000);
        assert_eq!(n, 0);
        assert_eq!(state.host_materializations.views().len(), 1);
        assert!(state.host_materializations.queued_views().is_empty());
    }

    #[test]
    fn delete_task_retires_views() {
        let mut state = state_x86();
        state.define_task(1, 0x1_0000, 0x100);
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(1, 0x3000, 0x1000, 0xdddd, 0x1000));
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(2, 0x3000, 0x1000, 0xeeee, 0x1000));
        assert!(state.delete_task(1).is_some());
        assert_eq!(state.host_materializations.views().len(), 1);
        assert_eq!(state.host_materializations.views()[0].ptr(), 0xeeee);
        assert_eq!(
            state.host_materializations.queued_views(),
            vec![(0xdddd, 0x1000)]
        );
    }

    #[test]
    fn ensure_gva_view_none_without_task() {
        let mut state = state_x86();
        let mut host = FakeHost::new();
        assert!(ensure_gva_view(&mut state, &mut host, 1, 0x1000, 0x1000).is_none());
    }

    #[test]
    fn covering_view_reuse() {
        let mut state = state_x86();
        state
            .host_materializations
            .publish_gva_view(GvaHostView::fixture(1, 0x1000, 0x4000, 0x9000, 0x4000));
        let v = find_covering_view(&state, 1, 0x1800, 0x100).unwrap();
        assert_eq!(v.ptr(), 0x9000);
        assert!(find_covering_view(&state, 1, 0x1000, 0x5000).is_none());
    }
}
