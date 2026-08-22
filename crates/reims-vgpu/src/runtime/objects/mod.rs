//! Object-list lookup, IOSurface texture registration, and x86 surface backing surface backing.
//!
//! Live layout (reims-vgpu-resource-format): entry `ref` is at
//! `(object_list_pfn << PAGE_SHIFT) + ref * 12` in the task GVA space —
//! `[type|desc_len packed u32][desc_gva u64]`.
//!
//! **x86 surface backing present path (Ventura 13.7 RE):**
//! `AppleParavirtResource::allocateBackingHandle` calls
//! `ResourceHeap::addObject(type=4, objectId=IOSurface::getSurfaceID(), …)` so
//! the object-list index for a surface-backed resource **is** the present
//! `surface_id`. The descriptor's own layout is
//! [`reims_vgpu_wire::device_desc`], which states it once with `offset_of!`
//! rather than as the eight literal offsets that used to sit in this module.

use crate::model::{SurfaceMappingEntry, TaskResource, TaskTable};
use crate::runtime::decode::resource::{
    decode_list_object_entry, list_object_entry_offset, ListObjectEntry, ObjectKind,
    OBJECT_LIST_ENTRY_LEN,
};
use crate::runtime::gva_mem;
use crate::runtime::host::HostMemory;
use crate::runtime::texture;
use crate::runtime::Device;
use reims_vgpu_core::endian::{ld32, st16, st32, st64};
use reims_vgpu_paging::geometry::{
    mapper_entry_gpa as entry_gpa_shift, page_size as page_size_of,
    MAPPER_PAGE_ENTRY_PFN_SHIFT as PAGE_ENTRY_PFN_SHIFT,
    MAPPER_PAGE_ENTRY_VALID as PAGE_ENTRY_VALID,
};
use reims_vgpu_protocol::{
    DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BASE_OFFSET, DEVICE_DESC_BPE, DEVICE_DESC_BPR,
    DEVICE_DESC_DIMS, DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT, DEVICE_DESC_PLANES,
    DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE, DEVICE_PLANE_BPR, DEVICE_PLANE_DESC_LEN,
    DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
};
use std::sync::Arc;

pub mod slot_recheck;

/// Resolve and retain one object's semantic construction descriptor.
///
/// The resource model owns the immutable cache; this boundary module owns the
/// byte parser used to fill it.
pub(crate) fn decoded_resource(
    resource: &TaskResource,
) -> &Result<reims_vgpu_protocol::ResourceDescriptor, reims_vgpu_protocol::ResourceDecodeError> {
    resource.decoded_with(|| {
        crate::runtime::decode::resource::decode_descriptor(
            resource.entry().kind,
            resource.descriptor(),
        )
    })
}

/// Fail-visible, de-duplicated per `(task_id, ref)`, for the IOSurface texture resolve
/// blind spot: an object ref that IS an IOSurface-backed texture but whose
/// descriptor cannot be read, cannot register a host texture, or carries
/// `mapping_id==0` used to collapse into a bare `None` → a coarse
/// `MissingTexture` at the draw site with no reason. `resolve_iosurface_texture_ref` runs
/// per-draw per-ref (very hot), so a bare fail line would flood; the latch logs
/// each `(task,ref,reason)` once and is cleared when the ref resolves
/// ([`clear_iosurface_texture_fail`]). Only genuine failures for a *confirmed IOSurface*
/// ref are routed here — the legitimate "ref is a different object type" and
/// unbound-slot returns stay silent. Runs on the drain worker (off the QEMU main
/// core).
type IOSurfaceTextureFailure = (u32, u32, &'static str);
type IOSurfaceTextureFailureSet = std::collections::HashSet<IOSurfaceTextureFailure>;

fn iosurface_texture_fail_latch() -> &'static std::sync::Mutex<IOSurfaceTextureFailureSet> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<IOSurfaceTextureFailureSet>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(IOSurfaceTextureFailureSet::new()))
}

fn note_iosurface_texture_fail(task_id: u32, ref_: u32, reason: &'static str, detail: String) {
    let mut guard = iosurface_texture_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((task_id, ref_, reason)) {
        crate::observe::fail(detail);
    }
}

/// Re-arm the fail latch for a ref that just resolved, so a later genuine
/// failure on the same ref is logged again (catches flapping).
fn clear_iosurface_texture_fail(task_id: u32, ref_: u32) {
    let mut guard = iosurface_texture_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(t, r, _)| !(*t == task_id && *r == ref_));
}

/// Fail-visible, de-duplicated per `(surface_id, reason)`, for the surface backing
/// backing blind spot: a surface whose object-list descriptor decoded fine (an
/// active task, a valid `SurfaceBackingDescriptor`) but whose page-backing construction then
/// failed — every downstream present/Store for that surface paints **stale or
/// black** with no reason. `apply_surface_backing` is reached from the per-present
/// scanout path (`ensure_surface_for_present`, ~48/s under scroll), so a persistent
/// backing failure would flood; the latch logs each `(surface_id, reason)` once
/// and re-arms when the surface next resolves cleanly ([`clear_surface_backing_fail`]), so a
/// flapping backing is re-logged. Only genuine surface backing candidate failures are
/// routed here — the caller's speculative per-task `continue`s (surface absent
/// from this task or a non-surface object type) stay silent. Runs on the drain
/// worker (off the QEMU main core).
///
/// # What the latch remembers, and why it is not just a set
///
/// The backing this device asked for, and when it asked. A refusal here is
/// frequently **transient** — the guest had not finished mapping the surface
/// when the per-present path first walked it, and the next attach resolves the
/// same pages — and the log gave no way to tell that from a surface this device
/// never managed to back. See [`clear_surface_backing_fail`] for what the pair enables and
/// what it measured.
///
/// `gva` is the backing base the refusal named, or `None` for the refusals
/// raised before one can be computed (`sid_zero`, `page_size_zero`). It is the
/// backing and not `surface_id` that identifies a recovery, because surface ids
/// recycle within a boot and across geometries — the same caveat
/// `apply_surface_backing`'s census line carries `gva0` for.
#[derive(Clone, Copy)]
struct ReportedSurfaceBackingFail {
    gva: Option<u64>,
    /// When this refusal was first raised. **Never refreshed**, so its age is
    /// how long the surface has been unbacked.
    first_at_ms: u64,
    /// When the device last asked for this backing and was refused again. Its
    /// age is how long since anything asked at all.
    last_at_ms: u64,
    /// How many times the device has asked and been refused, the first
    /// included. The whole point of the pair: see
    /// [`surface_backing_outstanding_census`].
    attempts: u32,
}

type SurfaceBackingFailLatch =
    std::collections::HashMap<(u32, &'static str), ReportedSurfaceBackingFail>;

fn surface_backing_fail_latch() -> &'static std::sync::Mutex<SurfaceBackingFailLatch> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<SurfaceBackingFailLatch>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(SurfaceBackingFailLatch::new()))
}

fn note_surface_backing_fail(
    surface_id: u32,
    reason: &'static str,
    gva: Option<u64>,
    detail: String,
) {
    let at_ms = crate::observe::elapsed_ms() as u64;
    let mut guard = surface_backing_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match guard.entry((surface_id, reason)) {
        std::collections::hash_map::Entry::Occupied(mut slot) => {
            // A repeat is the retry the design asks for, and it stays quiet on
            // the fail channel — `apply_surface_backing` is the per-present path
            // and one line per frame would flood. It is counted instead, which
            // is what makes the silence readable.
            let held = slot.get_mut();
            held.gva = gva;
            held.last_at_ms = at_ms;
            held.attempts = held.attempts.saturating_add(1);
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(ReportedSurfaceBackingFail {
                gva,
                first_at_ms: at_ms,
                last_at_ms: at_ms,
                attempts: 1,
            });
            crate::observe::fail(detail);
        }
    }
}

/// The first probe failure of the search in progress, per surface.
///
/// A surface lives in exactly one task's object list, so a search that walks
/// tasks in order meets non-owners before it meets the owner. Those misses are
/// the search working, not a backing failure, and reporting them as one is what
/// put ~95 `surface_backing_fail reason=translate` lines on a driven boot's
/// always-on channel for surfaces that then backed perfectly — the resolve
/// succeeded on a later task and the line stayed behind to be read as a defect.
///
/// So a probe records its reason here and nothing is emitted until the search
/// runs out of tasks. The first reason is kept rather than the last: it is the
/// most specific one available, and the tail of a search is dominated by tasks
/// that simply do not list the surface.
struct PendingSurfaceBackingFail {
    reason: &'static str,
    gva: Option<u64>,
    detail: String,
}

type SurfaceBackingPendingLatch = std::collections::HashMap<u32, PendingSurfaceBackingFail>;

fn surface_backing_pending_latch() -> &'static std::sync::Mutex<SurfaceBackingPendingLatch> {
    use std::sync::{Mutex, OnceLock};
    static PENDING: OnceLock<Mutex<SurfaceBackingPendingLatch>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(SurfaceBackingPendingLatch::new()))
}

/// Record why one task's probe refused, to be reported only if none succeeds.
///
/// `gva` is the backing base the probe was walking, threaded through so a later
/// clean attach on the *same* backing can be recognised as a recovery rather
/// than guessed at from `surface_id`. `None` where the refusal precedes any
/// computable address.
fn defer_surface_backing_fail(
    surface_id: u32,
    reason: &'static str,
    gva: Option<u64>,
    detail: String,
) {
    let mut guard = surface_backing_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard
        .entry(surface_id)
        .or_insert(PendingSurfaceBackingFail {
            reason,
            gva,
            detail,
        });
}

/// The search found no task that could back this surface: report the first
/// probe's reason through the flood latch.
fn flush_surface_backing_fail(surface_id: u32) {
    let pending = {
        let mut guard = surface_backing_pending_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.remove(&surface_id)
    };
    if let Some(pending) = pending {
        note_surface_backing_fail(surface_id, pending.reason, pending.gva, pending.detail);
    }
}

/// Re-arm the surface backing fail latch for a surface that just backed cleanly, drop the
/// probe reasons the successful search left behind, and — when the backing that
/// landed is the one a refusal named — say so.
///
/// # A refusal here is usually a retry that then worked, and the log did not say
///
/// `surface_backing_fail reason=translate` reads as lost guest work: the surface
/// could not be backed, so every present for it paints stale or black. It is
/// frequently nothing of the sort. `st=zero-pfn pte=0x0` means the guest had not
/// filled the leaf PTE when the per-present path walked it, and the refusal is
/// what makes the device ask again next frame instead of substituting a guess.
/// The retry is the design; a *silent* retry is what made the class unreadable.
///
/// Measured, driven x86/PCI, `web-content-probe -n 10 --churn 1` — six
/// refusals, and every one recovered on the same backing:
///
/// ```text
///   sid   refused at   backed at   delta
///    11        23652       23673    21 ms
///    72        49139       49146     7 ms
///    73        49165       49171     6 ms
///    48        51770       51771     1 ms
/// ```
///
/// Each was confirmed the *same* backing by matching the refusal's `gva=`
/// against the later attach's `gva0=`, not by surface id — ids recycle within a
/// boot and across geometries, so "the same sid resolved a frame later" can be a
/// different surface wearing the same number. That is why `gva` is threaded
/// through the latch rather than reconstructed here, and why a mismatch claims
/// nothing.
///
/// A prior session read this class as "the best-defined live failure class left"
/// and queued work against it. It was six successful retries. The recovery line
/// is what stops that from happening again, and a refusal that stays unpaired is
/// now the signal that something really did not back.
///
/// # The emitter confirms it on its own output
///
/// The table above was reconstructed by hand from a log that did not carry the
/// pairing. A second driven boot with this line in place produces it directly —
/// three refusals, three recoveries, **no unpaired refusal**, matched on the
/// backing:
///
/// ```text
///   surface_backing_recovered sid=25 reason=translate gva=0x42ab000 after_ms=5
///   surface_backing_recovered sid=25 reason=translate gva=0x4282000 after_ms=7
///   surface_backing_recovered sid=46 reason=translate gva=0x93b1000 after_ms=20
/// ```
///
/// Note the two `sid=25` at different backings: the id really does get reused
/// inside one boot, and matching on it would have paired the second refusal with
/// the first attach. `surface_backing_translate_refused` and `surface_backing_recovered` sum
/// to 3 and 3 across the census windows, so the counters agree with the lines.
fn clear_surface_backing_fail(surface_id: u32, backed_gva: u64) {
    surface_backing_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&surface_id);
    let mut superseded = 0usize;
    let recovered: Vec<(&'static str, u64)> = {
        let mut guard = surface_backing_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Every entry for this surface is dropped, exactly as before: the latch
        // must re-arm or a later genuine failure on a recycled id goes unlogged.
        // Only the ones whose backing matches are *claimed* as recoveries.
        let mut out = Vec::new();
        guard.retain(|(s, reason), reported| {
            if *s != surface_id {
                return true;
            }
            if reported.gva == Some(backed_gva) {
                out.push((*reason, reported.first_at_ms));
            } else {
                // Dropped without being claimed: this surface backed, but at a
                // different address than the refusal named — the guest
                // re-pointed it, so the refusal is moot rather than repaired.
                // Counted because it used to be the *third* way a refusal could
                // leave the latch and the only one that said nothing, which made
                // an unpaired `surface_backing_fail` unreadable: a reader diffing
                // the two line counts could not tell a refusal the guest walked
                // away from apart from one that never came back.
                superseded += 1;
            }
            false
        });
        out
    };
    if superseded > 0 {
        crate::runtime::drain::note_store_route_n("surface_backing_superseded", superseded as u64);
    }
    // After the early return, not before it: this runs on every clean attach
    // (the per-present scanout path) and almost every one of those has nothing
    // latched, so the clock read stays off the hot path.
    if recovered.is_empty() {
        return;
    }
    let now = crate::observe::elapsed_ms() as u64;
    for (reason, at_ms) in recovered {
        crate::runtime::drain::note_store_route("surface_backing_recovered");
        crate::observe::fail(format!(
            "surface_backing_recovered sid={surface_id} reason={reason} gva={backed_gva:#x} \
             after_ms={} (the earlier refusal for this backing was a retry; the guest \
             finished mapping and it landed)",
            now.saturating_sub(at_ms)
        ));
    }
}

/// The surface backing refusals still latched, for the census — or `None` if there are
/// none.
///
/// # The reading this exists to make possible
///
/// A refusal leaves [`surface_backing_fail_latch`] exactly three ways, and until this
/// existed only one of them said so. It is *recovered* when the surface backs at
/// the address the refusal named, which emits `surface_backing_recovered`. It is
/// *superseded* when the surface backs somewhere else — the guest re-pointed it,
/// so the refusal is moot — which [`clear_surface_backing_fail`] now counts. Or it is
/// still here, which is the only one that can be lost guest work, and it was the
/// silence the other two were mistaken for.
///
/// The three add up: `surface_backing_fail` lines equal
/// `surface_backing_recovered + surface_backing_superseded + n` from the last
/// census window. That identity is the point — a reader who finds fewer
/// recoveries than refusals now has a line to check instead of a hand-matched
/// diff of backing GVAs, which is how this hole was found.
///
/// # Reading the three numbers, because two of them used to say the opposite
///
/// `oldest_ms` is the age of the longest-outstanding refusal's **first** raise,
/// `since_last_ms` the age of the most recent one, and `attempts` how many times
/// the device has asked. Together they separate the two states a bare count
/// cannot:
///
/// - `attempts` climbing, `since_last_ms` near zero — the device is asking every
///   frame and being refused every frame. **This is the one that is lost guest
///   work**, and every present for that surface is painting stale or black.
/// - `attempts=1`, `since_last_ms` tracking `oldest_ms` — the device asked once,
///   was refused, and **nothing has asked since**. `apply_surface_backing` is
///   reached from the per-present path, so nothing asking means nothing is
///   presenting that surface: the guest is done with it. Nothing is lost.
///
/// This carried one number, `oldest_ms`, and its doc read it exactly backwards —
/// "`oldest_ms=12` is a refusal caught mid-retry; `oldest_ms=83000` is a surface
/// this device never backed". [`note_surface_backing_fail`] used a plain `insert`, so a
/// retry *overwrote* the timestamp: an actively-retried refusal pins that age
/// near zero forever, and a large one means the retries stopped. The sentence
/// had it the wrong way round for both states, which is why the reading below
/// went unmade across three boots.
///
/// Silent when the latch is empty, because an empty latch is the expected state
/// and this rides a one-second cadence.
///
/// # What a driven boot reads
///
/// x86/PCI, `web-content-probe -n 10 --churn 1`, 12 951 log lines, probe green.
/// The identity closes exactly:
///
/// ```text
///   surface_backing_fail        8
///   surface_backing_recovered   7
///   surface_backing_superseded  0
///   outstanding n             1      8 = 7 + 0 + 1
/// ```
///
/// Fifteen census lines for the boot, so the cadence costs nothing. The first
/// reads `oldest_ms=294` and the last `oldest_ms=14317` — which is the argument
/// for carrying the age at all, because one of those `n=1`s is a retry still in
/// flight and the other is a surface that was never backed, and the count alone
/// cannot tell them apart.
///
/// # The standing `sid=27` reading, and what it turned out to be
///
/// Three driven boots each ended with exactly one outstanding refusal, every
/// time `sid=27`, `reason=translate st=zero-pfn pte=0x0`, a 2-3 page surface, at
/// a different backing each boot. It was carried as an open question across two
/// sessions: abandoned by the guest, or a retry that never came?
///
/// **Abandoned.** It is answerable from the third boot's own census series
/// without any new signal, once `insert`'s refresh is accounted for. The 23
/// `surface_backing_outstanding` lines run `oldest_ms` 204 → 22242 while `t` runs
/// 401037 → 423075: the two deltas are **equal at every sample**, 22038 apiece.
/// A timestamp that tracks wall clock exactly is a timestamp nothing refreshed,
/// so the device asked once, at t=400833, and never again.
///
/// The surface agrees. `sid=27` is 93×21 and then 99×29 — a tooltip — set up at
/// t≈26 s and untouched for the following six minutes. `zero-pfn` says the
/// guest's own leaf PTE is empty: it took the backing away. Nothing presented it
/// afterwards, which is why nothing asked again.
///
/// `attempts` is in the line so this costs a glance rather than an afternoon,
/// and so the *other* state — a surface retried every frame and refused every
/// frame — is not mistaken for it. That one is real lost work and this boot does
/// not contain one.
pub(crate) fn surface_backing_outstanding_census() -> Option<String> {
    let now = crate::observe::elapsed_ms() as u64;
    let guard = surface_backing_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    // The oldest entry is the one worth naming: it is the one least likely to
    // be a retry still in flight, and a single line cannot carry them all.
    let oldest = guard
        .iter()
        .min_by_key(|(_, reported)| reported.first_at_ms)
        .map(|((sid, reason), reported)| (*sid, *reason, *reported))?;
    let (sid, reason, held) = oldest;
    Some(format!(
        "surface_backing_outstanding n={} oldest_ms={} since_last_ms={} attempts={} \
         sid={sid} reason={reason} gva={}",
        guard.len(),
        now.saturating_sub(held.first_at_ms),
        now.saturating_sub(held.last_at_ms),
        held.attempts,
        gva_text(held.gva),
    ))
}

fn gva_text(gva: Option<u64>) -> String {
    gva.map_or_else(|| "none".to_string(), |g| format!("{g:#x}"))
}

/// Wire object type for surface / IOSurface backing (x86 Tahoe/Ventura).
pub const OBJECT_TYPE_SURFACE: u8 = 4;
/// RefTextureHandle: surfaceID@0 + cookie@4 + guest blob@8 (texture-ref 28-06-26).
pub const OBJECT_TYPE_REF_TEXTURE: u8 = 5;
/// IOSurface plane view RefTexture descriptor (RE `allocateRefTextureHandle` + Metal
/// `initWithDevice:descriptor:iosurface:plane:field:`):
/// - `surfaceID@0` = `IOSurface::getSurfaceID()` = surface backing heap object id / mid
/// - `ownerTask@4` = the task whose object list holds that surface
/// - `args@8..` = **serialized texture args** length `desc_len-8` (MTLTextureDescriptor
///   stream for the **plane** view; plane is applied guest-side before serialize)
///
/// See [[reims-vgpu-resource-paging]] IOSurface plane view section.
/// IOSurface plane view descriptor geometry, from the wire crate's Tier-2 view of it.
///
/// The ten offsets these used to be are `offset_of!` on
/// [`reims_vgpu_wire::device_desc::IOSurfacePlaneViewHeader`], `IOSurfacePlaneViewArgsHeader` and
/// `IOSurfacePlaneViewTextureRecord`, asserted there against the numbers this module used to
/// state. The two record tags come with them, since a tag is part of the
/// layout's identity and not of this device's policy.
// These five are named only by the tests that
// assert the layout, so they are gated with those tests rather than left
// reachable from the staticlib. `IOSURFACE_PLANE_VIEW_RECORD_TAG_PLANE` is not among them:
// its one caller builds the descriptor with `wire::device_desc::IOSurfacePlaneViewBuilder`
// on the line above and now names the tag from the same module.
#[cfg(test)]
pub(crate) use reims_vgpu_wire::device_desc::{
    IOSURFACE_PLANE_VIEW_ARG_RECORD, IOSURFACE_PLANE_VIEW_OWNER_TASK,
    IOSURFACE_PLANE_VIEW_RECORD_PLANE, IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW,
    IOSURFACE_PLANE_VIEW_SURFACE_ID,
};

#[cfg(test)]
pub const IOSURFACE_PLANE_VIEW_MIN_LEN: usize =
    reims_vgpu_wire::device_desc::IOSURFACE_PLANE_VIEW_ARGS;

/// Texture view named by a IOSurface plane view descriptor's serialized args record.
///
/// This is not limited to IOSurface planes. The live desktop also uses
/// row-byte-equivalent reinterpretations such as a 480-wide RGBA32Uint view
/// over a 1920-wide BGRA8 surface.
pub use reims_vgpu_protocol::IOSurfacePlaneViewDescriptor;

/// Report the owner task a IOSurface plane view descriptor names, once per distinct value.
///
/// Every surface backing surface this device has resolved lived in task 0 — measured
/// (`surface_backing_claimants`: `claims=1 winner=0` on every surface id of two driven
/// boots) and structural, since the guest registers IOSurface backings in the
/// accelerator's kernel task whose id is a hardcoded 0.
/// [`reims_vgpu_wire::device_desc::IOSURFACE_PLANE_VIEW_OWNER_TASK`] is
/// the guest saying the same thing on the wire, so this reads 0 forever and
/// stays on the quiet channel.
///
/// A non-zero value is the one reading that matters, and it is a failure line
/// because two things would follow from it at once: the surface backing search's "task 0
/// first" probe order is no longer the guest's answer, and the field's decoded
/// meaning is wrong. `first_sight` is keyed on the value alone, so the whole
/// boot costs one line whichever way it goes.
fn note_iosurface_plane_view_owner_task(desc: &[u8]) {
    let Ok(h) = reims_vgpu_wire::device_desc::iosurface_plane_view_header(desc) else {
        return;
    };
    let task = h.owner_task.get();
    if !crate::observe::first_sight("iosurface_plane_view_owner_task", task as u64) {
        return;
    }
    let line = format!("iosurface_plane_view_owner_task task={task}");
    if task == 0 {
        crate::observe::off(line);
    } else {
        crate::observe::fail(format!(
            "{line} (a IOSurface plane view view names a surface owner other than the kernel task; \
             the surface backing search probes task 0 first on the reading that this is always 0)"
        ));
    }
}

/// Decode the serialized texture-view record from a full IOSurface plane view descriptor.
///
/// Fail-closed: `None` unless the record tag matches and geometry is sane
/// (2D, nonzero). The record names the exact Metal view (format + geometry)
/// over the IOSurface bytes; callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable.
pub fn decode_iosurface_plane_view(desc: &[u8]) -> Option<IOSurfacePlaneViewDescriptor> {
    note_iosurface_plane_view_owner_task(desc);
    crate::runtime::decode::resource::decode_iosurface_plane_view_resource(desc)
        .ok()
        .and_then(|resource| resource.view)
}

/// The stride is named only by the tests that walk a plane table by hand.
#[cfg(test)]
pub(crate) use reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_STRIDE;
/// Surface backing descriptor geometry, from the wire crate's Tier-2 view of it.
///
/// The eight offsets these used to be are now `offset_of!` on
/// [`reims_vgpu_wire::device_desc::SurfaceBackingHeader`] and
/// [`reims_vgpu_wire::device_desc::SurfaceBackingPlaneRecord`], asserted there. Only the
/// four the device still computes with are re-exported.
pub(crate) use reims_vgpu_wire::device_desc::{
    surface_backing_len_for, SURFACE_BACKING_MIN_LEN, SURFACE_BACKING_PLANES,
    SURFACE_BACKING_PLANE_CAP,
};

/// CoreVideo / IOSurface biplanar 420 full-range (`'420f'`).
pub const IOSURFACE_FOURCC_420F: u32 = 0x3432_3066;
/// CoreVideo / IOSurface biplanar 420 video-range (`'420v'`).
pub const IOSURFACE_FOURCC_420V: u32 = 0x3432_3076;

pub use reims_vgpu_protocol::{SurfaceBackingDescriptor, SurfaceBackingPlane};

/// CoreVideo biplanar 8-bit 420 family — **not** a single `MTLPixelFormat`.
///
/// Metal binds planes via `newTextureWithDescriptor:iosurface:plane:` as R8 (Y)
/// and RG8 (UV). Product must not invent BGRA.
#[inline]
pub fn iosurface_fourcc_is_biplanar(pixel_format: u32) -> bool {
    matches!(pixel_format, IOSURFACE_FOURCC_420F | IOSURFACE_FOURCC_420V)
}

/// True when surface backing / mapping cannot be staged as one linear color texture.
#[inline]
pub fn surface_backing_is_multiplanar(surf: &SurfaceBackingDescriptor) -> bool {
    surf.plane_count > 1 || iosurface_fourcc_is_biplanar(surf.pixel_format)
}

/// Mapping has multi-plane device geometry (plane_count≥2) or biplanar FourCC.
pub fn mapping_is_multiplanar(m: &SurfaceMappingEntry) -> bool {
    use reims_vgpu_protocol::decode_device_surface;
    if let Some(s) = decode_device_surface(m.device_desc_bytes()) {
        if s.plane_count > 1 {
            return true;
        }
        if iosurface_fourcc_is_biplanar(s.pixel_format) {
            return true;
        }
    }
    false
}

/// The device descriptor's `pixelFormat` word as a Metal format.
///
/// That field carries **two** encodings and always has. On x86 this device
/// synthesizes the descriptor and [`synthesize_device_desc_from_surface_backing`] writes
/// the MTL ordinal for a known single-plane surface and the raw OSType FourCC
/// otherwise. On arm64 the descriptor is the guest's own and the field holds
/// whatever `getPixelFormat()` returned, which is a FourCC for media surfaces.
///
/// The arm64 mapper used to read it as `raw as u16` — a silent narrowing, and
/// wrong by the rule [`iosurface_pixel_format_to_mtl`] states about this exact
/// operation: `'BGRA'` truncates to `0x5241`, which is not a Metal format, so
/// `bytes_per_pixel` refuses it, every sample window refuses, and every render
/// target on that mapping resolves to nothing. The x86 arm meanwhile read the
/// same conceptual field as a FourCC. Two consumers, one field, two encodings
/// assumed — and the truncation is the arm that loses guest work silently.
///
/// The two encodings are disjoint, and the test between them is not a
/// plausibility one. An MTLPixelFormat is an enum ordinal, and the descriptor's
/// own per-plane format fields are 16 bits wide, so an ordinal fits in 16 bits by
/// construction. An OSType is four character bytes, none of them zero, so it
/// cannot. A value that does not fit therefore *cannot* be an ordinal and goes
/// through the FourCC table; a value that does fit is the ordinal it is.
///
/// Unknown FourCCs and multi-plane OSTypes come back 0 — the same fail-closed
/// refusal the surface backing path latches, never an invented BGRA8.
pub fn device_desc_format_to_mtl(raw: u32) -> u16 {
    if raw <= u16::MAX as u32 {
        return raw as u16;
    }
    iosurface_pixel_format_to_mtl(raw)
}

/// Map IOSurface OSType FourCC (or MTL raw) to a **single-plane** MTL pixel format.
///
/// Live x86 surface backing carries IOSurface `pixelFormat` as a FourCC (e.g. `'BGRA'` =
/// `0x42475241`). Truncating to u16 yields `0x5241` which is not a Metal format.
///
/// Returns **0** when:
/// - format is multi-plane (e.g. `'420f'` / `'420v'`) — no single MTLPixelFormat
/// - format is unknown — fail closed; **do not** invent BGRA8
///
/// Unknown formats fail closed.
pub fn iosurface_pixel_format_to_mtl(pixel_format: u32) -> u16 {
    use reims_vgpu_core::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RG8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
        MTL_FORMAT_RGBA8_UNORM,
    };
    if pixel_format == 0 {
        return 0;
    }
    // Multi-plane OSTypes are not MTLPixelFormats (Metal plane: API).
    if iosurface_fourcc_is_biplanar(pixel_format) {
        return 0;
    }
    // No pass-through for small values. This used to return `pixel_format as
    // u16` for anything at or below 0x200, on the reading that such a value was
    // "already an MTLPixelFormat ordinal". That decided which *encoding* a field
    // was in from the field's magnitude, and the caller already knows: every
    // caller here passes a surface backing `pixelFormat` (+0x0c), which is an IOSurface
    // OSType — a four-character code, so never below 0x20202020. The IOSurface texture and
    // IOSurface plane view rails carry their MTL ordinal in a `u16` field of their own and do
    // not route through this function.
    //
    // The magnitude test was also wrong at its own boundary: MTLPixelFormat
    // BGRA10_XR is 552 (0x228) and its three siblings are 553-555, so a 10-bit
    // XR surface passed 0x200 and fell into the FourCC match below regardless.
    match pixel_format {
        // 'BGRA' / 'ARGB' (kb: ARGB fourcc → BGRA8Unorm 0x50 for render targets)
        0x4247_5241 | 0x4152_4742 => MTL_FORMAT_BGRA8_UNORM,
        // 'RGBA'
        0x5247_4241 => MTL_FORMAT_RGBA8_UNORM,
        // 'RGhA' / half-float variants seen as AhGR in notes
        0x5247_6841 | 0x4168_4752 => MTL_FORMAT_RGBA16_FLOAT,
        // Single-plane R8 / RG8 OSTypes used as plane textures (not biplanar media fourcc).
        // 'L008' / common R8 fourccs are rare on surface backing; MTL ordinals already handled above.
        // 'R8  ' / 'RG08' if ever seen as OSType:
        0x5238_2020 => MTL_FORMAT_R8_UNORM,
        0x5247_3038 => MTL_FORMAT_RG8_UNORM,
        // Unknown FourCC: 0 — callers fail closed (no BGRA invent).
        _ => 0,
    }
}

/// Decode one surface backing plane record.
fn decode_surface_backing_plane(desc: &[u8], plane_index: usize) -> Option<SurfaceBackingPlane> {
    let r = reims_vgpu_wire::device_desc::surface_backing_plane(desc, plane_index).ok()?;
    Some(SurfaceBackingPlane {
        offset: r.offset.get(),
        width: r.width.get(),
        height: r.height.get(),
        bytes_per_row: r.bytes_per_row(),
        bytes_per_element: r.bytes_per_element(),
    })
}

/// The bytes of a surface backing surface descriptor that [`decode_surface_backing`] does
/// **not** read: `+0x11..0x14` and everything past the plane records it
/// consumed (`SURFACE_BACKING_PLANES + plane_count * SURFACE_BACKING_PLANE_STRIDE ..`).
///
/// Decoded today: `length` (+0x00), `backing_pfn` (+0x08), `pixel_format`
/// (+0x0c), `plane_count` (+0x10), and each plane's offset/width/height/packed
/// bpr. That is everything we know about a surface when the guest creates it —
/// and it is not enough to tell a desktop swapchain buffer from a same-geometry
/// offscreen render target, because a WebKit content tile is also 1920x1080
/// 'BGRA'. Membership is therefore reconstructed downstream by compositor-output
/// edges, full-frame-publish detection, output groups, presented-ness, and the
/// a/b seed.
///
/// **Measured: the guest is not telling us here.** Across one 1766 s x86/Vulkan
/// session with a real GUI login (boot `20260728-163046`), the probe below
/// emitted exactly two shapes for ≥5983 decodes over 453 distinct surface ids
/// and 154 distinct geometries — desktop swapchain buffers and never-displayed
/// content tiles alike:
///
/// ```text
/// surface_backing_desc_shape distinct=1 1920x1080 fmt=0x42475241 planes=1 len=36 undecoded_len=3 undecoded_nz=0
/// surface_backing_desc_shape distinct=2   320x320 fmt=0x34323066 planes=2 len=52 undecoded_len=3 undecoded_nz=0
/// ```
///
/// `len` is `SURFACE_BACKING_PLANES + plane_count * SURFACE_BACKING_PLANE_STRIDE` exactly, and it is
/// the *guest's* number — [`read_descriptor`] honours `descriptor_length` with no
/// clamp. The record ends where the plane array ends; the only bytes we skip are
/// the three at `+0x11`, and they were zero every time. There is nowhere in this
/// descriptor for a usage, bind, scanout or role hint to be, so no rule over
/// surface identity can classify a brand-new buffer before its first draw.
///
/// Narrow: this is the surface backing record on the x86 PCI pathway. It says nothing
/// about the mapper IOSurface texture-view envelope decoded on arm, and a
/// create-time record we never read at all would be invisible to it.
///
/// A `plane_count` above [`SURFACE_BACKING_PLANE_CAP`] is clamped by the decoder, so the
/// records past the clamp fall into this span too — which is correct: they are
/// bytes we did not read.
///
/// Public so the probe's notion of "undecoded" is pinned by a test rather than
/// restated in a log format string.
pub fn undecoded_surface_backing_bytes(desc: &[u8]) -> Vec<u8> {
    if desc.len() < SURFACE_BACKING_MIN_LEN {
        return Vec::new();
    }
    let Ok(h) = reims_vgpu_wire::device_desc::surface_backing_header(desc) else {
        return Vec::new();
    };
    let plane_count = (h.plane_count as usize).min(SURFACE_BACKING_PLANE_CAP);
    let planes_end = surface_backing_len_for(plane_count);
    let mut out = Vec::new();
    // The header's undecoded interior, named by the field rather than by the
    // literal it used to be: a field added before it moves this with it.
    let reserved =
        core::mem::offset_of!(reims_vgpu_wire::device_desc::SurfaceBackingHeader, reserved);
    out.extend_from_slice(&desc[reserved..SURFACE_BACKING_PLANES]);
    if planes_end < desc.len() {
        out.extend_from_slice(&desc[planes_end..]);
    }
    out
}

/// One always-on line per distinct `(len, undecoded span)`, capped.
///
/// Keyed on the **content** of the undecoded bytes, never on the record length.
/// The `display_txn_payload` probe keyed its budget on `(opcode, payload_len)`,
/// the length never varied, and it exhausted itself inside the first 400 ms —
/// it answered one question and then went blind for the rest of the session. A
/// new *value* is the interesting event here, so that is the key.
///
/// Runs before the decoder's own validity checks, so a record that fails to
/// decode still reports. An earlier version of this probe on the IOSurface texture
/// descriptor sat after its length check and emitted nothing at all on a live
/// boot; "the decoder never ran" and "the tail is constant" produced the same
/// silence, which is the reading the probe exists to rule out.
///
/// Hitting the cap is reported once. A silent truncation would read like "we
/// saw everything", which is the same class of error as a probe reporting a
/// confident constant.
fn note_surface_backing_shape(desc: &[u8]) {
    const MAX_SHAPES: usize = 24;
    const HEX_MAX: usize = 128;
    use std::sync::Mutex;
    type ShapeKey = (usize, Vec<u8>);
    static SEEN: Mutex<Option<std::collections::BTreeSet<ShapeKey>>> = Mutex::new(None);

    let undecoded = undecoded_surface_backing_bytes(desc);
    let (fresh, distinct) = {
        let mut guard = SEEN.lock().unwrap_or_else(|p| p.into_inner());
        let seen = guard.get_or_insert_with(Default::default);
        if seen.len() > MAX_SHAPES {
            return;
        }
        (seen.insert((desc.len(), undecoded.clone())), seen.len())
    };
    if !fresh {
        return;
    }
    if distinct > MAX_SHAPES {
        crate::observe::fail(format!(
            "surface_backing_desc_shape outcome=cap_reached distinct={distinct} \
             (the undecoded span varies per surface; it is not a constant tail)"
        ));
        return;
    }
    // Plane 0's geometry through the plane decoder, not through `SURFACE_BACKING_PLANES
    // + 4` and `+ 8`. Those two literals were the plane record's interior
    // restated at a second site, so the stride was declared once and its
    // contents twice.
    let (w, h, fmt, pc) = match (
        reims_vgpu_wire::device_desc::surface_backing_header(desc),
        decode_surface_backing_plane(desc, 0),
    ) {
        (Ok(h), Some(p0)) => (p0.width, p0.height, h.pixel_format.get(), h.plane_count),
        _ => (0, 0, 0, 0),
    };
    let hex: String = desc
        .iter()
        .take(HEX_MAX)
        .map(|b| format!("{b:02x}"))
        .collect();
    crate::observe::fail(format!(
        "surface_backing_desc_shape distinct={distinct} {w}x{h} fmt={fmt:#x} planes={pc} len={} \
         undecoded_len={} undecoded_nz={} hex={hex}{}",
        desc.len(),
        undecoded.len(),
        undecoded.iter().filter(|&&b| b != 0).count(),
        if desc.len() > HEX_MAX { "…" } else { "" },
    ));
}

/// Report, once per reason, that the surface backing decoder dropped something the guest
/// declared.
///
/// Deduped rather than sampled: each reason names a distinct shape of blob, and
/// a surface stream re-decodes the same descriptor thousands of times a boot, so
/// an undeduped line would flood while adding nothing. The first occurrence is
/// what a reader needs — after it, `surface_backing_desc_shape` carries the geometry.
fn surface_backing_decode_drop_latch(
) -> &'static std::sync::Mutex<std::collections::HashSet<&'static str>> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<&'static str>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

fn note_surface_backing_decode_drop(reason: &'static str, detail: String) {
    // The latch is flood protection, so the magnitude has to live somewhere it
    // survives: without this, a second malformed descriptor and a thousand of
    // them read identically — one line, and nothing to ask.
    crate::runtime::drain::census::note_store_route("surface_backing_desc_refused");
    let fresh = {
        let mut guard = surface_backing_decode_drop_latch()
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        guard.insert(reason)
    };
    if fresh {
        crate::observe::fail(detail);
    }
}

/// Forget which reasons have been reported, so a test observes the first
/// occurrence rather than whatever an earlier test in the same process left
/// behind.
#[cfg(test)]
fn reset_surface_backing_decode_drops() {
    surface_backing_decode_drop_latch()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
}

/// Decode a surface backing surface descriptor blob.
pub fn decode_surface_backing(desc: &[u8]) -> Option<SurfaceBackingDescriptor> {
    note_surface_backing_shape(desc);
    let decoded = match reims_vgpu_protocol::decode_surface_backing_descriptor(desc) {
        Ok(decoded) => decoded,
        Err(reims_vgpu_protocol::ResourceDecodeError::ErrUnsupported(
            "res_surface_backing_plane_count",
        )) => {
            let header = reims_vgpu_wire::device_desc::surface_backing_header(desc).ok()?;
            note_surface_backing_decode_drop(
                "plane_count_over_cap",
                format!(
                    "surface_backing_decode_drop reason=plane_count_over_cap declared={} \
                     cap={SURFACE_BACKING_PLANE_CAP} fmt={:#x}",
                    header.plane_count,
                    header.pixel_format.get()
                ),
            );
            return None;
        }
        Err(reims_vgpu_protocol::ResourceDecodeError::ErrShort("res_surface_backing_plane")) => {
            let header = reims_vgpu_wire::device_desc::surface_backing_header(desc).ok()?;
            let available = desc.len().saturating_sub(SURFACE_BACKING_PLANES)
                / reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_STRIDE;
            note_surface_backing_decode_drop(
                "plane_record_short",
                format!(
                    "surface_backing_decode_drop reason=plane_record_short plane={available} \
                     planes={} desc_len={} fmt={:#x}",
                    header.plane_count,
                    desc.len(),
                    header.pixel_format.get()
                ),
            );
            return None;
        }
        Err(_) => return None,
    };
    Some(decoded)
}

/// Build `sIOSurfaceDeviceDescriptor` geometry from surface backing wire (no invent).
///
/// Multi-plane: plane records from surface backing planes; sample path selects by
/// geometry. Single-plane: surface-level fields only
/// (`plane_count==0` path in `sample_window_from_device_desc`).
fn synthesize_device_desc_from_surface_backing(surf: &SurfaceBackingDescriptor) -> Vec<u8> {
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    let multi = surface_backing_is_multiplanar(surf);
    let mtl = iosurface_pixel_format_to_mtl(surf.pixel_format);
    // Device desc pixelFormat field: guest stores getPixelFormat() (FourCC for
    // biplanar media). Single-plane product sample uses MTL ordinal when known.
    let fmt_word = if multi {
        surf.pixel_format
    } else if mtl != 0 {
        mtl as u32
    } else {
        surf.pixel_format
    };
    st32(&mut device_desc[DEVICE_DESC_PIXEL_FORMAT..], fmt_word);
    // `allocSize` is a u32 field in the device descriptor and `length` is u64 on
    // the wire, so a surface above 4 GiB cannot be published faithfully. Saying
    // `u32::MAX` is the least wrong answer available — it is the largest size the
    // field can hold, so a reader sizing a mapping from it under-reads rather
    // than walking past the end — but it is still a size the guest did not ask
    // for, and it must not be published as though it were.
    let alloc = if surf.length > u32::MAX as u64 {
        note_surface_backing_decode_drop(
            "alloc_size_over_u32",
            format!(
                "surface_backing_decode_drop reason=alloc_size_over_u32 length={} \
                 published={} (device-descriptor allocSize is 32-bit)",
                surf.length,
                u32::MAX
            ),
        );
        u32::MAX
    } else {
        surf.length as u32
    };
    st32(&mut device_desc[DEVICE_DESC_ALLOC_SIZE..], alloc);
    // Surface-level dims/bpr from plane0 (same as surface backing plane0 convenience).
    let dims = ((surf.width as u64) << 8) | ((surf.height as u64) << 40);
    st64(&mut device_desc[DEVICE_DESC_DIMS..], dims);
    if surf.bytes_per_row > 0 {
        st32(&mut device_desc[DEVICE_DESC_BPR..], surf.bytes_per_row);
    }
    if multi && surf.plane_count > 0 {
        // Multi-plane: publish plane records; sample_window_from_device_desc
        // matches IOSurface texture R8/RG8 binds by (w,h,bpe), and declines when two
        // planes share all three. Do not invent bases from format alone.
        let n = (surf.plane_count as usize).min(SURFACE_BACKING_PLANE_CAP);
        device_desc[DEVICE_DESC_PLANE_COUNT] = n as u8;
        // Surface-level bpe: plane0 element size when wire provides it.
        let bpe0 = surf.planes[0].bytes_per_element;
        if bpe0 != 0 {
            st16(&mut device_desc[DEVICE_DESC_BPE..], bpe0 as u16);
        }
        for i in 0..n {
            let p = &surf.planes[i];
            let base = DEVICE_DESC_PLANES + i * DEVICE_PLANE_DESC_LEN;
            st32(&mut device_desc[base + DEVICE_PLANE_OFFSET..], p.offset);
            // plane_size: 0 = skip size check in sample_window_from_device_plane
            // (surface backing wire has offset/w/h/bpr, not a separate size field).
            st32(&mut device_desc[base + DEVICE_PLANE_SIZE..], 0);
            let pdims = ((p.width as u64) << 8) | ((p.height as u64) << 40);
            st64(&mut device_desc[base + DEVICE_PLANE_DIMS..], pdims);
            st32(&mut device_desc[base + DEVICE_PLANE_BPR..], p.bytes_per_row);
            if p.bytes_per_element != 0 {
                st16(
                    &mut device_desc[base + DEVICE_PLANE_BPE..],
                    p.bytes_per_element as u16,
                );
            } else if iosurface_fourcc_is_biplanar(surf.pixel_format) {
                // Contract: 420 Y bpe=1, UV bpe=2 when wire high-byte is 0.
                // Only fill when FourCC is known biplanar — not a free invent for
                // arbitrary multi-plane. Matches Metal R8/RG8 plane bind bpp.
                let bpe = if i == 0 { 1u16 } else { 2u16 };
                st16(&mut device_desc[base + DEVICE_PLANE_BPE..], bpe);
            }
        }
    } else {
        // Single-plane surface-level sample path (plane_count 0).
        device_desc[DEVICE_DESC_PLANE_COUNT] = 0;
        // Plane 0's offset, which this arm used to decode and then drop.
        //
        // `decode_surface_backing_plane` reads four fields per plane; the surface-level
        // convenience copies took three of them (width, height, bytes-per-row)
        // and left the offset behind, so a single-plane surface whose pixels
        // start past the base of its allocation was read and written at 0. The
        // multi-plane arm above publishes every plane's offset, and the
        // consumers are symmetric: `sample_window_from_device_surface` returns
        // `base_offset` as the window offset and folds it into `span_end`, which
        // is exactly what `sample_window_from_device_plane` does with a plane's.
        // On the arm64 mapper path this same field is read straight out of the
        // guest's descriptor rather than synthesized, so dropping it here also
        // made the two pathways describe one surface differently.
        //
        // Zero is the ordinary value and stays silent; a non-zero one is the
        // population that was being misread, and `surface_backing_base_offset_nonzero`
        // counts how large that is. Read on a driven x86/Vulkan boot: **0**, so
        // no single-plane surface on that workload starts past its base and this
        // is contract fidelity rather than a live repair. It is also why the
        // change was safe to make without a rate: every window it could move is
        // one the counter would have named.
        let base_offset = surf.planes[0].offset;
        if base_offset != 0 {
            crate::runtime::drain::note_store_route("surface_backing_base_offset_nonzero");
            st32(&mut device_desc[DEVICE_DESC_BASE_OFFSET..], base_offset);
        }
        if mtl != 0 {
            if let Some(bpp) = reims_vgpu_core::pixel_format::bytes_per_pixel(mtl) {
                st16(&mut device_desc[DEVICE_DESC_BPE..], bpp as u16);
            }
        }
    }
    device_desc
}

/// Name the object-list geometry behind a refused entry read.
///
/// `gva_mem::read_task_gva_by_id` already reports the refusal and which of the
/// walk's checks produced it, but it is generic over every caller and can only
/// name the address. For this caller the address is *derived* — `(pfn <<
/// page_shift) + ref * entry_len` — and the address alone cannot say which of
/// its three inputs is the surprising one.
///
/// That distinction was the question, and this line answered it. A driven x86
/// boot emits `gva_zero_pfn` from this site for tasks 2..=9, and the geometry
/// behind every one is `list_pfn=1 list_count=1048576` — the guest's own
/// `SetObjectList` values, taken straight off the wire by
/// `drain::apply_set_object_list`, which invents nothing. So the guest reserves
/// a million-slot object list at task-virtual page 1 and maps its pages lazily,
/// and a lookup that lands on an unpopulated page is the resolver saying "the
/// guest has not said yet". That is expected control flow, `None` is the right
/// answer, and no guest work is lost.
///
/// Worth knowing because the address alone reads like the opposite. `pfn = 1`
/// with a million slots is precisely the pair `TaskEntry::define` used to
/// *fabricate* for tasks with no list — `model::state` documents removing it,
/// after it caused this device to walk a neighbouring task's page table and
/// decode its entry as this task's. Seeing those two numbers again invites the
/// conclusion that the fabrication is back. It is not: they are what the guest
/// publishes, which is presumably where the fabricated defaults were copied from.
///
/// Off-channel and latched per task: expected control flow stays quiet, and the
/// measurement above is what establishes it is expected. One line per task per
/// boot (eight on that run), alongside the one the walker already emits per
/// (reason, task, site).
fn note_list_entry_unreadable(
    task_id: u32,
    ref_: u32,
    task: &crate::model::TaskEntry,
    entry_gva: u64,
) {
    if !crate::observe::first_sight("object_list_entry_unreadable", u64::from(task_id)) {
        return;
    }
    crate::observe::off(list_entry_unreadable_detail(task_id, ref_, task, entry_gva));
}

/// The line [`note_list_entry_unreadable`] emits, built separately so a test can
/// assert the geometry reaches it without going through the always-on sink.
fn list_entry_unreadable_detail(
    task_id: u32,
    ref_: u32,
    task: &crate::model::TaskEntry,
    entry_gva: u64,
) -> String {
    format!(
        "object_list_entry_unreadable task={task_id} ref={ref_} gva={entry_gva:#x} \
         list_pfn={} list_count={} entry_len={OBJECT_LIST_ENTRY_LEN} \
         (the walker's own refusal names the check; this names the three inputs \
          the address was built from, which it cannot see)",
        task.object_list_pfn, task.object_list_count
    )
}

/// What a missing object-list entry means to the caller that asked.
///
/// The read underneath is identical; only whether a miss is reportable differs,
/// and only the caller knows. A driven boot measured 18 `gva_read_refused
/// reason=gva_zero_pfn` lines on the fail channel — two per task for tasks 2
/// through 10 — and every one of them was [`Self::Probe`] working exactly as
/// designed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListLookup {
    /// The guest's own command named this ref against this task, so an entry
    /// the device cannot read is guest work it cannot execute. Reportable.
    Named,
    /// This device is asking *whether* the task owns the ref. Every task that
    /// does not own it misses, which is how the search finds the one that does —
    /// so a miss here is the answer, not a failure, and reporting it would put a
    /// line on the fail channel for each task the search correctly stepped over.
    Probe,
}

/// Which of the eight checks in an object-list lookup came back empty.
///
/// They used to be eight `None`s behind one `reason=no_list_entry`, and only the
/// unreadable arm said anything at all. That is why a macos-26 boot losing ~40
/// draws to this refusal could not say which of two opposite things had
/// happened: the device cleared the task's list under the guest
/// ([`Self::NoObjectList`] — `define_task` resets `object_list_pfn`/`count` and
/// macOS 26 re-issues `define_task` for a live tid with a new page-table root,
/// which macOS 13 never does), or the guest had simply not published the object
/// into a list this device read perfectly well ([`Self::SlotEmpty`]). The first
/// is a device defect and the second is a wait; they share a reason string and
/// nothing else.
///
/// The routes are counted only for a ref the guest **named** — see
/// [`ListLookup`], whose `Probe` arm misses by design on every task that does
/// not own the ref.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ListMiss {
    /// No task under this id at all.
    NoTask,
    /// The task exists and has been torn down or not activated.
    TaskInactive,
    /// The task has no object list registered. Either none was ever set, or one
    /// was set and a later `define_task` for the same id reset it.
    NoObjectList,
    /// The ref is past the count the guest declared for the list.
    RefBeyondList,
    /// `pfn << page_shift` plus the slot offset does not fit a `u64`.
    AddressOverflow,
    /// The slot's guest address did not read, carrying **which** of the walk's
    /// checks refused.
    ///
    /// The payload is the same distinction [`crate::runtime::host::MemError`]
    /// draws for every other guest read, and the reason it is here is
    /// [`slot_recheck`]: a slot that read and decoded cleanly at miss time
    /// cannot later be genuinely unmapped, rooted at a zero PFN and outside the
    /// address space all at once, so on a re-read the check that refused *is*
    /// the finding. It was the whole of a driven macos-26 boot's terminal
    /// verdicts while it was still one bare value.
    ///
    /// `object_list_entry_unreadable` stays alongside it, because that line
    /// names the three inputs the address was built from, which the walk's own
    /// refusal cannot see.
    Unreadable(crate::runtime::host::MemError),
    /// The sixteen bytes read and are not an object-list entry.
    Undecodable,
    /// The slot read and is zero: the list is where the guest said and this
    /// entry has not been written yet. Not a device failure — a race with the
    /// guest publishing the object.
    SlotEmpty,
}

impl ListMiss {
    /// Every variant, for the distinctness check below.
    ///
    /// [`Self::route`]'s match is exhaustive, so a new variant cannot skip
    /// having a route — but it can skip this list. Add it here too, or the
    /// distinctness test stops covering the population it names.
    #[cfg(test)]
    const ALL: [Self; 8] = [
        Self::NoTask,
        Self::TaskInactive,
        Self::NoObjectList,
        Self::RefBeyondList,
        Self::AddressOverflow,
        Self::Unreadable(crate::runtime::host::MemError::Unmapped),
        Self::Undecodable,
        Self::SlotEmpty,
    ];

    pub fn route(self) -> &'static str {
        match self {
            Self::NoTask => "list_miss_no_task",
            Self::TaskInactive => "list_miss_task_inactive",
            Self::NoObjectList => "list_miss_no_object_list",
            Self::RefBeyondList => "list_miss_ref_beyond_list",
            Self::AddressOverflow => "list_miss_address_overflow",
            Self::Unreadable(_) => "list_miss_unreadable",
            Self::Undecodable => "list_miss_undecodable",
            Self::SlotEmpty => "list_miss_slot_empty",
        }
    }

    /// The same eight checks, seen a tranche later by
    /// [`slot_recheck`](self::slot_recheck).
    ///
    /// A second table rather than a prefix swap on [`Self::route`], because both
    /// are matched exhaustively and a new variant therefore cannot be added
    /// without giving it a name on both cadences. It sits here rather than in
    /// `slot_recheck` for the same reason the first one does: the two spellings
    /// of one check belong next to each other, where a divergence is visible.
    ///
    /// The recheck's first version collapsed four of these into one
    /// `slot_recheck_unreadable`, and the first driven boot put **20** readings
    /// in it — the whole of that boot's terminal verdicts, and unreadable in
    /// exactly the sense that it could not say which check refused. That is the
    /// failure [`ListMiss`] itself exists to have fixed once.
    pub fn recheck_route(self) -> &'static str {
        match self {
            Self::NoTask => "slot_recheck_no_task",
            Self::TaskInactive => "slot_recheck_task_inactive",
            Self::NoObjectList => "slot_recheck_no_object_list",
            Self::RefBeyondList => "slot_recheck_ref_beyond_list",
            Self::AddressOverflow => "slot_recheck_address_overflow",
            Self::Unreadable(_) => "slot_recheck_unreadable",
            Self::Undecodable => "slot_recheck_undecodable",
            // Not terminal: the watch survives to be asked again. Named anyway
            // so the table is total and the residue has a spelling if a caller
            // ever wants to report it.
            Self::SlotEmpty => "slot_recheck_still_empty",
        }
    }
}

/// Lookup one object-list slot for `task_id` / `ref_`, reporting a miss.
///
/// For a ref the guest named. A speculative caller wants [`probe_list_entry`];
/// see [`ListLookup`] for why the distinction is the caller's to make.
pub fn lookup_list_entry<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    list_entry(state, host, task_id, ref_, ListLookup::Named)
}

/// [`lookup_list_entry`] for a caller asking whether this task owns `ref_`.
///
/// Quiet on a miss. The object list really is where the guest said — a driven
/// boot reads `set_object_list … pfn=0x1 count=1048576 plen=12` for **every**
/// task including the one that then resolves 11 768 surfaces — so a task
/// missing at this slot is a task without that object, which is what the search
/// is asking.
pub fn probe_list_entry<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    list_entry(state, host, task_id, ref_, ListLookup::Probe)
}

fn list_entry<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    ref_: u32,
    lookup: ListLookup,
) -> Option<ListObjectEntry> {
    let found = list_entry_or_miss(state, host, task_id, ref_, lookup);
    match found {
        Ok(entry) => {
            // Only a ref the guest named: a probe's success is the search
            // finding an owner, which says nothing about what this task's own
            // list once held. One atomic bit — see `slot_recheck::ResolvedBits`
            // for why this path cannot take a lock.
            if lookup == ListLookup::Named {
                slot_recheck::note_ref_resolved(task_id, ref_);
                // The control for the banding below, and the reason it is worth
                // reading: a miss skewing late says nothing unless the hits do
                // not. See `census::note_list_lookup_age`.
                crate::runtime::drain::note_list_lookup_age(
                    true,
                    crate::runtime::drain::tranche_elapsed_us(),
                );
            }
            Some(entry)
        }
        Err(miss) => {
            // Only for a ref the guest named. A probe misses on every task that
            // does not own the ref, which is how it finds the one that does —
            // counting those would bury the named misses under the search.
            if lookup == ListLookup::Named {
                crate::runtime::drain::note_store_route(miss.route());
                // How late in its tranche this lookup happened. The guest clears
                // a slot by writing its own memory, so a slot found cleared
                // should be one read late — if these band like the hits do, that
                // story is wrong however good the totals look.
                crate::runtime::drain::note_list_lookup_age(
                    false,
                    crate::runtime::drain::tranche_elapsed_us(),
                );
                if miss == ListMiss::SlotEmpty {
                    note_slot_empty_claimants(state, host, task_id, ref_);
                    // The unconfounded half of the same question — see
                    // `slot_recheck` for why the claimant search above cannot
                    // settle it and this can.
                    slot_recheck::note_slot_empty(state, host, task_id, ref_);
                }
            }
            None
        }
    }
}

/// For a named ref whose own task's slot is empty: how many *other* live tasks
/// hold a real object at that slot, against how many there are?
///
/// [`ListMiss::SlotEmpty`] is the only miss a macos-26 boot produces and the
/// whole of that rail's lost draws. "Does anyone else have it" looked like the
/// question, and a first boot answered *every* miss with yes — which is when the
/// confound became obvious. **Every task registers its object list at the same
/// `pfn = 1`**, and refs are small and dense, so "another task has something at
/// slot 3" is close to a tautology on a busy guest and says nothing about
/// ownership.
///
/// So the reading is the *fraction*, and it is emitted banded against the live
/// task count rather than as a yes/no:
///
/// - **nowhere** — nobody has published it. The guest named a ref before writing
///   the slot, and the answer is for the packet to wait rather than for the draw
///   to be dropped.
/// - **one** — exactly one other task has it. That is a real ownership signal:
///   the object exists in a list this device did not look in.
/// - **many / all** — the slot index is simply populated across the guest's
///   tasks, and this search cannot tell ownership from coincidence. Anything
///   built on it would be built on the confound.
///
/// Costs one probe read per live task, on a miss only.
fn note_slot_empty_claimants<M: HostMemory>(state: &Device, host: &M, task_id: u32, ref_: u32) {
    let live = state.tasks.live_count();
    let claimants: Vec<(u32, ListObjectEntry)> = state
        .tasks
        .live_ids()
        .filter(|&other| other != task_id)
        .filter_map(|other| probe_list_entry(state, host, other, ref_).map(|e| (other, e)))
        .collect();
    crate::runtime::drain::note_store_route(slot_empty_claim_route(claimants.len(), live));
    // The band was built when these lists were believed dense, where naming the
    // claimants would have been naming most of the guest. They are not: a driven
    // boot reads 4 to 18 occupied slots in a 341-entry first page, so a claim is
    // ~2 % likely by coincidence and *which* tasks claim is now a reading rather
    // than noise. Latched per `(task, ref)` — the band above is per miss and this
    // is per slot, which is also why the two counts differ.
    if !crate::observe::first_sight(
        "slot_empty_claimants",
        (u64::from(task_id) << 32) | u64::from(ref_),
    ) {
        return;
    }
    // Occupancy first, because it disqualifies most claims outright: a task
    // holding 316 of 341 slots claims every ref there is. A driven boot found
    // task 1 doing exactly that.
    //
    // Occupancy alone does not qualify the survivors, though. A sparse list
    // grows from index 0, so low refs are more likely occupied in *any* task,
    // and the missing refs here are 3, 4, 9, 10, 14 — the low end. `type=` is
    // the reading that position cannot fake: if the claimant's slot holds an
    // object of a kind the guest's command could not have meant, the claim is
    // coincidence however sparse the claimant is.
    let detail: Vec<String> = claimants
        .iter()
        .map(|&(other, entry)| {
            let held = slot_recheck::first_page_population(state, host, other)
                .map_or(-1i64, |p| p.populated as i64);
            format!("{other}:holds={held}:kind={}", entry.kind)
        })
        .collect();
    crate::observe::off(format!(
        "slot_empty_claimants task={task_id} ref={ref_} live={live} claimants=[{}] \
         (other live tasks holding a real object at this ref, each with how many objects \
          it holds in all and the object type it has here)",
        detail.join(" ")
    ));
}

/// Band a claimant count against the live task count.
///
/// Split out from the walk so the banding is testable without a guest: the walk
/// is a page-table read per task and the band is the only part that can be
/// wrong in a way that changes what the next session believes.
fn slot_empty_claim_route(claimants: usize, live_tasks: usize) -> &'static str {
    // `live_tasks` counts the asking task too, so the most that can claim is
    // one fewer. Comparing against that rather than against `live_tasks` is what
    // makes "all" mean all of them.
    let others = live_tasks.saturating_sub(1);
    match claimants {
        0 => "list_miss_slot_empty_claimed_nowhere",
        1 => "list_miss_slot_empty_claimed_by_one",
        n if others > 0 && n >= others => "list_miss_slot_empty_claimed_by_all",
        _ => "list_miss_slot_empty_claimed_by_many",
    }
}

/// The lookup proper, naming the check that refused.
///
/// Split from [`list_entry`] so the eight ways a slot comes back empty are eight
/// values rather than eight `None`s. See [`ListMiss`] for why that mattered.
fn list_entry_or_miss<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    ref_: u32,
    lookup: ListLookup,
) -> Result<ListObjectEntry, ListMiss> {
    let task = state.tasks.get(task_id).ok_or(ListMiss::NoTask)?;
    if !task.active {
        return Err(ListMiss::TaskInactive);
    }
    if task.object_list_count == 0 {
        return Err(ListMiss::NoObjectList);
    }
    let off =
        list_object_entry_offset(ref_, task.object_list_count).ok_or(ListMiss::RefBeyondList)?;
    let entry_gva = ((task.object_list_pfn as u64) << state.page_shift)
        .checked_add(off)
        .ok_or(ListMiss::AddressOverflow)?;
    let mut raw = [0u8; OBJECT_LIST_ENTRY_LEN];
    let read = match lookup {
        ListLookup::Named => gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            entry_gva,
            &mut raw,
            state.page_shift,
        ),
        ListLookup::Probe => gva_mem::try_read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            entry_gva,
            &mut raw,
            state.page_shift,
        ),
    };
    if let Err(why) = read {
        if lookup == ListLookup::Named {
            note_list_entry_unreadable(task_id, ref_, task, entry_gva);
        }
        return Err(ListMiss::Unreadable(why));
    }
    let e = decode_list_object_entry(&raw).map_err(|_| ListMiss::Undecodable)?;
    if e.descriptor_length == 0 || e.descriptor_gva == 0 {
        return Err(ListMiss::SlotEmpty);
    }
    Ok(e)
}

/// Read the descriptor blob for a list entry.
pub fn read_descriptor<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    entry: &ListObjectEntry,
) -> Option<Vec<u8>> {
    // Guest descriptor_length is authoritative — no product 4 KiB read clamp.
    let len =
        crate::runtime::draw::host_alloc_len(entry.descriptor_length as u64).filter(|&n| n > 0)?;
    let mut buf = vec![0u8; len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        entry.descriptor_gva,
        &mut buf,
        state.page_shift,
    )
    .ok()?;
    Some(buf)
}

/// Which rung of the object-list ladder refused, as a value rather than a
/// spelling.
///
/// [`crate::observe::ladder_slug`] made the four rungs share one *vocabulary*;
/// this makes the first three share one *implementation*. A rail that matches on
/// this cannot skip the type check, cannot ask the three questions out of order,
/// and cannot invent a fourth condition between them — all three of which a
/// hand-written ladder can do silently.
///
/// The decode rung is deliberately absent. Its decoder differs per object type
/// and returns that decoder's own error, so folding it in here would mean either
/// one resolver per type or a decoder trait, and neither buys anything the call
/// site does not already have.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LadderRung {
    /// The guest has put nothing under this ref. Ordinary while a task's list is
    /// still being populated, which is why several rails answer it quietly.
    NoListEntry,
    /// Something is under the ref and it is not a type this caller accepts.
    ///
    /// Carries the tag it found, because every rail that reports this rung was
    /// formatting `ot={}` by hand from the entry it no longer has by then.
    WrongType { got: ObjectKind },
    /// The entry names descriptor bytes that could not be read — the ref is
    /// live, and its descriptor GVA is not mapped right now.
    ///
    /// Carries the length the entry declared, for the same reason
    /// [`Self::WrongType`] carries the tag: it is the datum that explains this
    /// rung — "the entry said this many bytes at that GVA and they could not be
    /// read" — and the entry it came from is gone by the time a caller reports.
    DescRead { declared_len: u32 },
}

/// Why construction of a retained sampler object did not complete.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplerResolveError {
    Rung(LadderRung),
    Decode {
        status: crate::runtime::decode::resource::DecodeStatus,
        descriptor_len: usize,
        tag: Option<u32>,
        declared_len: Option<u32>,
    },
}

/// Retrieve or construct the sampler named by `task_id` / `sampler_ref`.
///
/// A sampler is an immutable object in its own task-local reference space.
/// Successful construction snapshots and decodes its descriptor once; failed
/// construction remains retryable because nothing is registered until every
/// rung, including decode, succeeds. Its explicit sampler-delete command and
/// task teardown are the only events that retire this entry.
pub fn resolve_sampler_state<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    sampler_ref: u32,
) -> Result<Arc<reims_vgpu_protocol::SamplerDescriptor>, SamplerResolveError> {
    use crate::runtime::decode::resource::decode_sampler_descriptor;

    if let Some(sampler) = state.task_objects.samplers.get(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(sampler_ref),
    ) {
        return Ok(sampler);
    }

    let entry = lookup_list_entry(state, host, task_id, sampler_ref)
        .ok_or(SamplerResolveError::Rung(LadderRung::NoListEntry))?;
    if entry.kind != ObjectKind::SerializerResource {
        return Err(SamplerResolveError::Rung(LadderRung::WrongType {
            got: entry.kind,
        }));
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(SamplerResolveError::Rung(
        LadderRung::DescRead {
            declared_len: entry.descriptor_length,
        },
    ))?;
    let descriptor_len = bytes.len();
    let tag = bytes.get(..4).map(ld32);
    let declared_len = bytes.get(4..8).map(ld32);
    let descriptor =
        decode_sampler_descriptor(&bytes).map_err(|status| SamplerResolveError::Decode {
            status,
            descriptor_len,
            tag,
            declared_len,
        })?;
    let sampler = state.task_objects.samplers.register(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(sampler_ref),
        Arc::new(descriptor),
    );
    crate::runtime::drain::note_store_route("sampler_state_constructed");
    Ok(sampler)
}

/// Whether a semantic object is owned by the task resource registry.
fn object_kind_is_resource(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::Buffer
            | ObjectKind::Texture
            | ObjectKind::SurfaceBacking
            | ObjectKind::IOSurfacePlaneView
            | ObjectKind::TextureView
            | ObjectKind::IOSurfaceTexture
            | ObjectKind::MemorylessTexture
            | ObjectKind::DualPlaneTexture
            | ObjectKind::ResourceHandle
            | ObjectKind::HeapBuffer
            | ObjectKind::ExternalBuffer
    )
}

/// Retrieve or construct the resource named by `task_id` / `obj_ref`.
///
/// A successful construction snapshots the object-list entry and descriptor
/// bytes for the lifetime of that reference. Subsequent binds retrieve the
/// retained resource; guest memory is consulted again only after an explicit
/// resource deletion or task teardown. Failed constructions are not retained,
/// so a descriptor that is still being published can succeed on retry.
pub fn resolve_resource<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    obj_ref: u32,
) -> Result<Arc<TaskResource>, LadderRung> {
    if let Some(resource) = state.task_objects.resources.get(task_id, obj_ref) {
        ensure_resource_relations(state, host, task_id, obj_ref, &resource);
        return Ok(resource);
    }

    let entry = lookup_list_entry(state, host, task_id, obj_ref).ok_or(LadderRung::NoListEntry)?;
    if !object_kind_is_resource(entry.kind) {
        return Err(LadderRung::WrongType { got: entry.kind });
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(LadderRung::DescRead {
        declared_len: entry.descriptor_length,
    })?;
    let descriptor: Arc<[u8]> = Arc::from(bytes);
    let candidate = Arc::new(TaskResource::new(entry, descriptor));
    let resource = state
        .task_objects
        .resources
        .register(task_id, obj_ref, Arc::clone(&candidate));
    ensure_resource_relations(state, host, task_id, obj_ref, &resource);
    // A validity record may precede this lazy construction. Normalize that
    // statement only after descriptor relations have attached the resource to
    // its storage authority; doing it before attachment would advance the
    // temporary private authority that attachment replaces.
    if Arc::ptr_eq(&resource, &candidate)
        && state
            .content
            .preconstruction_writes
            .has_write(task_id, obj_ref)
    {
        let id = resource
            .semantic_id()
            .expect("a published task resource has a semantic identity");
        state.task_objects.resources.note_guest_write_by_id(id);
    }
    Ok(resource)
}

fn ensure_resource_relations<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    object_ref: u32,
    resource: &Arc<TaskResource>,
) {
    if !resource.begin_relation_publication() {
        return;
    }
    let published = publish_resource_relations(state, host, task_id, object_ref, resource);
    resource.finish_relation_publication(published);
}

/// Publish construction relations that are explicit in decoded descriptors.
fn publish_resource_relations<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    object_ref: u32,
    resource: &Arc<TaskResource>,
) -> bool {
    match resource.entry().kind {
        ObjectKind::Buffer => {
            let Ok(crate::runtime::decode::resource::Descriptor::Buffer(buffer)) =
                decoded_resource(resource)
            else {
                return false;
            };
            let Some((address, length)) = buffer.backing_gva_size(state.page_shift) else {
                return false;
            };
            state
                .task_objects
                .resources
                .attach_task_address(task_id, object_ref, address, length)
        }
        ObjectKind::Texture => {
            let Ok(crate::runtime::decode::resource::Descriptor::Texture(texture)) =
                decoded_resource(resource)
            else {
                return false;
            };
            let Some(address) = texture.allocation_base_gva(state.page_shift) else {
                return false;
            };
            if texture.allocation_size == 0 {
                return false;
            }
            state.task_objects.resources.attach_task_address(
                task_id,
                object_ref,
                address,
                texture.allocation_size,
            )
        }
        ObjectKind::SurfaceBacking => {
            let Ok(crate::runtime::decode::resource::Descriptor::SurfaceBacking(_)) =
                decoded_resource(resource)
            else {
                return false;
            };
            state
                .task_objects
                .resources
                .attach_registered_surface(task_id, object_ref, object_ref)
        }
        ObjectKind::IOSurfacePlaneView => {
            let Ok(crate::runtime::decode::resource::Descriptor::IOSurfacePlaneView(view)) =
                decoded_resource(resource)
            else {
                return false;
            };
            let parent_task = view.owner_task.get();
            let parent_ref = view.surface.get();
            if parent_ref == 0 || (parent_task == task_id && parent_ref == object_ref) {
                return false;
            }
            let Ok(parent) = resolve_resource(state, host, parent_task, parent_ref) else {
                return false;
            };
            if parent.entry().kind == ObjectKind::SurfaceBacking {
                state
                    .task_objects
                    .resources
                    .link_view(task_id, object_ref, parent_task, parent_ref)
            } else {
                false
            }
        }
        ObjectKind::TextureView => {
            if let Ok(crate::runtime::decode::resource::Descriptor::BufferTexture(buffer_texture)) =
                decoded_resource(resource)
            {
                if buffer_texture.buffer_ref == 0 || buffer_texture.buffer_ref == object_ref {
                    return false;
                }
                let Ok(buffer) = resolve_resource(state, host, task_id, buffer_texture.buffer_ref)
                else {
                    return false;
                };
                if buffer.entry().kind != ObjectKind::Buffer {
                    return false;
                }
                return state.task_objects.resources.link_buffer_texture(
                    task_id,
                    object_ref,
                    buffer_texture.buffer_ref,
                    buffer_texture.offset,
                    buffer_texture.bytes_per_row,
                );
            }
            let Ok(crate::runtime::decode::resource::Descriptor::TextureView(view)) =
                decoded_resource(resource)
            else {
                return false;
            };
            let parent_ref = view.base_texture_ref;
            if parent_ref == 0 || parent_ref == object_ref {
                return false;
            }
            let Ok(parent) = resolve_resource(state, host, task_id, parent_ref) else {
                return false;
            };
            if matches!(
                parent.entry().kind,
                ObjectKind::Texture
                    | ObjectKind::TextureView
                    | ObjectKind::IOSurfaceTexture
                    | ObjectKind::IOSurfacePlaneView
            ) {
                state
                    .task_objects
                    .resources
                    .link_view(task_id, object_ref, task_id, parent_ref)
            } else {
                false
            }
        }
        _ => true,
    }
}

/// Look `obj_ref` up in `task_id`'s list, require its type to be one of `want`,
/// and read its descriptor bytes.
///
/// The first three rungs of the ladder about twenty rails open with, in the one
/// order they can be asked: a type tag cannot be checked before the entry is
/// found, and a descriptor cannot be read before the entry says where it is.
///
/// `want` is a slice because several rails accept more than one tag —
/// `OBJECT_TYPE_TEXTURE` and `OBJECT_TYPE_TEXTURE_VARIANT` are the standing
/// pair — and a single-tag caller passes a one-element slice rather than a
/// second entry point that would drift from this one.
///
/// `ref_ == 0` is **not** handled here. An unbound ref is a different statement
/// from a ref naming nothing, several rails treat it as expected control flow
/// and stay silent, and `AGENTS.md` names it as one of the things that must not
/// be logged. Callers that care test it before calling.
pub fn resolve_descriptor<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    obj_ref: u32,
    want: &[ObjectKind],
) -> Result<(ListObjectEntry, Arc<[u8]>), LadderRung> {
    if let Some(resource) = state.task_objects.resources.get(task_id, obj_ref) {
        if !want.contains(&resource.entry().kind) {
            return Err(LadderRung::WrongType {
                got: resource.entry().kind,
            });
        }
        ensure_resource_relations(state, host, task_id, obj_ref, &resource);
        return Ok((resource.entry(), Arc::clone(resource.descriptor())));
    }

    // Keep the ladder's type check ahead of the descriptor read. A caller
    // asking for the wrong kind must not turn an unreadable descriptor into a
    // different refusal merely because resource retention is enabled.
    let entry = lookup_list_entry(state, host, task_id, obj_ref).ok_or(LadderRung::NoListEntry)?;
    if !want.contains(&entry.kind) {
        return Err(LadderRung::WrongType { got: entry.kind });
    }
    let bytes = read_descriptor(state, host, task_id, &entry).ok_or(LadderRung::DescRead {
        declared_len: entry.descriptor_length,
    })?;
    if !object_kind_is_resource(entry.kind) {
        return Ok((entry, Arc::from(bytes)));
    }
    let descriptor: Arc<[u8]> = Arc::from(bytes);
    let resource = state.task_objects.resources.register(
        task_id,
        obj_ref,
        Arc::new(TaskResource::new(entry, descriptor)),
    );
    ensure_resource_relations(state, host, task_id, obj_ref, &resource);
    if !want.contains(&resource.entry().kind) {
        return Err(LadderRung::WrongType {
            got: resource.entry().kind,
        });
    }
    Ok((resource.entry(), Arc::clone(resource.descriptor())))
}

/// Why a type-1 buffer ref did not yield a backing span.
///
/// The three ways past [`resolve_descriptor`]'s rungs, which is the whole of
/// what [`resolve_buffer_span`] can refuse for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferSpanRefusal {
    /// One of the first three object-list rungs.
    Rung(LadderRung),
    /// The descriptor bytes read, and are not a buffer descriptor.
    Decode,
    /// The descriptor decoded and names no backing allocation — a zero handle
    /// or a zero size. The resource exists; it has nowhere to read from.
    NoBacking,
}

/// Resolve a type-1 buffer ref to its `(guest base address, allocation size)`.
///
/// Three rails needed this and each wrote it out: `compute_exec`'s buffer-window
/// read, `icb`'s type-1 bind, and `draw`'s vertex/fragment buffer load — which
/// was found last, after this function already existed, and whose five refusals
/// carried no `reason=` field at all, so none of them was in the log's ranking.
/// They agreed on the four steps — resolve as
/// `OBJECT_TYPE_BUFFER`, decode, derive the span from the handle and the
/// device's own `page_shift` — and disagreed on what to say afterwards. The ICB
/// copy named all four refusals; the compute copy returned `Option` and its
/// caller labelled every one of them `no_backing`, which is the *last* of the
/// four and therefore wrong about the other three.
///
/// So the refusal is returned as a value and each rail maps it into its own
/// status vocabulary, rather than one rail's answer being the other's guess.
///
/// `page_shift` comes from the device (x86 12, arm64e 14) and never from the
/// resource decoder's arm-only default; the two place the handle differently.
///
/// `buffer_ref == 0` is not handled here, for the reason [`resolve_descriptor`]
/// gives: an unbound ref is a different statement from a ref naming nothing, and
/// callers that care test it first.
pub fn resolve_buffer_span<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<(u64, u64), BufferSpanRefusal> {
    let resource =
        resolve_resource(state, host, task_id, buffer_ref).map_err(BufferSpanRefusal::Rung)?;
    resolve_buffer_span_from_resource(state, &resource)
}

/// Resolve the backing carried by an encoder-retained buffer object.
pub fn resolve_buffer_span_from_resource(
    state: &Device,
    resource: &crate::model::TaskResource,
) -> Result<(u64, u64), BufferSpanRefusal> {
    if resource.entry().kind != ObjectKind::Buffer {
        return Err(BufferSpanRefusal::Rung(LadderRung::WrongType {
            got: resource.entry().kind,
        }));
    }
    let desc = match decoded_resource(resource) {
        Ok(crate::runtime::decode::resource::Descriptor::Buffer(desc)) => desc,
        Ok(_) => {
            return Err(BufferSpanRefusal::Rung(LadderRung::WrongType {
                got: resource.entry().kind,
            }));
        }
        Err(_) => return Err(BufferSpanRefusal::Decode),
    };
    desc.backing_gva_size(state.page_shift)
        .ok_or(BufferSpanRefusal::NoBacking)
}

/// Resolve object ref and, if IOSurface texture, latch mapping geometry + cache the entry.
///
/// Returns the mapping_id for IOSurface textures, or None.
pub fn resolve_iosurface_texture_ref<M: HostMemory>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<u32> {
    let resource = match resolve_resource(state, host, task_id, ref_) {
        Ok(resource) => resource,
        Err(LadderRung::DescRead { .. }) => {
            // Keep this failure scoped to a confirmed IOSurface texture object. The
            // second lookup is only on the failed-construction path; successful
            // binds retrieve the retained resource without a guest read.
            if let Some(entry) = lookup_list_entry(state, host, task_id, ref_)
                .filter(|entry| entry.kind == ObjectKind::IOSurfaceTexture)
            {
                note_iosurface_texture_fail(
                    task_id,
                    ref_,
                    crate::observe::ladder_slug!("iosurface_texture", desc_read),
                    format!(
                        "iosurface_texture_resolve_fail reason=iosurface_texture_desc_read task={task_id} ref={ref_} obj_type={} desc_gva={:#x} desc_len={}",
                        entry.kind, entry.descriptor_gva, entry.descriptor_length
                    ),
                );
            }
            return None;
        }
        Err(_) => return None,
    };
    resolve_iosurface_texture_resource(state, task_id, ref_, &resource)
}

/// Resolve an already-retained IOSurface texture resource to its mapping.
///
/// Draw preparation resolves each bound reference once and threads the
/// resulting object through all of its consumers. Keeping this half separate
/// prevents the IOSurface texture branch from looking the same reference up again and
/// reparsing immutable construction bytes on every bind.
pub fn resolve_iosurface_texture_resource(
    state: &mut Device,
    task_id: u32,
    ref_: u32,
    resource: &TaskResource,
) -> Option<u32> {
    let entry = resource.entry();
    let desc = &resource.descriptor();
    if entry.kind != ObjectKind::IOSurfaceTexture {
        // Legitimate: this ref is a different object type, not a texture. Normal
        // control flow (resolve_iosurface_texture_refs skips it) — never a failure.
        return None;
    }
    if let Some(mapping_id) = resource.registered_iosurface_mapping() {
        return Some(mapping_id.get());
    }
    let (mapping_id, mapper_ref, plane) = match decoded_resource(resource) {
        Ok(crate::runtime::decode::resource::Descriptor::MapperIOSurfaceTextureView(view)) => {
            let Ok(mapping_id) = texture::register_iosurface_texture_geom(
                state,
                view.mapper_surface,
                view.declaration.width,
                view.declaration.height,
                view.declaration.pixel_format,
            ) else {
                note_iosurface_texture_fail(
                    task_id,
                    ref_,
                    "iosurface_mapper_surface_unresolved",
                    format!(
                        "iosurface_texture_resolve_fail reason=iosurface_mapper_surface_unresolved task={task_id} ref={ref_} mapper_ref={:#x}",
                        view.mapper_surface.get()
                    ),
                );
                return None;
            };
            (mapping_id, view.mapper_surface, view.plane)
        }
        Err(error) => {
            note_iosurface_texture_fail(
                task_id,
                ref_,
                "iosurface_texture_decode",
                format!(
                    "iosurface_texture_resolve_fail reason=iosurface_texture_decode task={task_id} ref={ref_} desc_len={} decode={error:?}",
                    desc.len()
                ),
            );
            return None;
        }
        Ok(_) => return None,
    };
    if mapping_id == 0 {
        // Defensive for a compatibility decoder that ever accepts the sentinel
        // mapping id without registering it.
        note_iosurface_texture_fail(
            task_id,
            ref_,
            "iosurface_mapping_zero",
            format!(
                "iosurface_texture_resolve_fail reason=iosurface_mapping_zero task={task_id} ref={ref_} desc_len={}",
                desc.len()
            ),
        );
        return None;
    }
    let mapping_id =
        resource.register_iosurface_mapping(reims_vgpu_protocol::SurfaceId::new(mapping_id));
    #[cfg(test)]
    state
        .state
        .fixtures
        .texture_to_mapping
        .insert((task_id, ref_), mapping_id.get());
    if !state
        .task_objects
        .resources
        .attach_mapper_storage(task_id, ref_, mapper_ref, plane)
    {
        note_iosurface_texture_fail(
            task_id,
            ref_,
            "iosurface_texture_resource_graph",
            format!(
                "iosurface_texture_resolve_fail reason=iosurface_texture_resource_graph task={task_id} ref={ref_} mapping={}",
                mapping_id.get()
            ),
        );
        return None;
    }
    // Resolved: re-arm so a later genuine failure on this ref logs again.
    clear_iosurface_texture_fail(task_id, ref_);
    Some(mapping_id.get())
}

/// Make the physical backing of a retained texture bindable.
///
/// A texture bind does not revalidate immutable surface construction input.
/// The guest announces a physical re-point explicitly; that path clears the
/// mapping's page entries and bumps its generation. Consequently a warm bind
/// only checks the retained mapping, while the first bind and the first bind
/// after a re-point rebuild the backing through the full resolver.
pub fn ensure_surface_for_texture_bind<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
) -> bool {
    let ready = state
        .surfaces
        .mappings
        .get(&surface_id)
        .is_some_and(|m| m.lifecycle.active && !m.pages.entries.is_empty() && m.has_geometry());
    ready || ensure_surface_for_present(state, host, surface_id)
}

/// The detail line a refused page walk reports.
///
/// # Why the walk status is on it
///
/// `reims_vgpu_paging::resolve::ResolveStatus` distinguishes every check
/// in the guest page-table walk and has done since it was written; this site
/// collapsed all of them into the single word `translate`. Two refusals with
/// opposite remedies were therefore indistinguishable in the log: a leaf PTE the
/// guest has not filled in yet (`zero-pfn` — the surface is mid-map, and the
/// next frame resolves it) and a task root this device could not read at all
/// (`no-directory`, `root(...)` — the walk is aimed at the wrong table and
/// waiting will never help).
///
/// The distinction had to be reconstructed by hand for a whole A/B's worth of
/// refusals, by matching each against later attaches of the same surface id and
/// inferring which it had been. `walk=` states it outright.
///
/// Pure and separate from the emit so the composition is testable: the always-on
/// sink has no in-memory capture, so a test can only reach this line by building
/// it.
fn surface_backing_translate_fail_detail(
    surface_id: u32,
    task_id: u32,
    page: u64,
    page_count: u64,
    gva: u64,
    walk: &str,
) -> String {
    format!(
        "surface_backing_fail reason=translate sid={surface_id} task={task_id} \
         page={page}/{page_count} gva={gva:#x} walk=[{walk}] \
         (no translation in this task; not substituting the GVA)"
    )
}

/// Apply a decoded surface backing surface as page-table backing for `surface_id`.
///
/// `backing_pfn` is a GPU-VA page (same source as type-2/3 textures). Translate
/// each consecutive GVA page through the task page table into GPA page entries
/// the scanout path already understands.
fn apply_surface_backing<M: HostMemory>(
    state: &mut Device,
    host: &M,
    task_id: u32,
    surface_id: u32,
    surf: &SurfaceBackingDescriptor,
) -> bool {
    if !crate::model::is_surface_mapping_id(surface_id) {
        defer_surface_backing_fail(
            surface_id,
            "sid_zero",
            None,
            format!(
                "surface_backing_fail reason=sid_zero sid={surface_id} task={task_id} \
                 (0 is the unbound-mapping sentinel; backing it would store pixels \
                 no attachment reader can address)"
            ),
        );
        return false;
    }
    let page_shift = state.page_shift;
    let page_size = page_size_of(page_shift);
    if page_size == 0 {
        defer_surface_backing_fail(
            surface_id,
            "page_size_zero",
            None,
            format!("surface_backing_fail reason=page_size_zero sid={surface_id} task={task_id} page_shift={page_shift}"),
        );
        return false;
    }
    // The backing base this attempt is about. It identifies the refusal for
    // `clear_surface_backing_fail`, which is what lets a later clean attach on the *same*
    // backing be recognised as a recovery — surface ids recycle, addresses do
    // not.
    let backing_base_gva = (surf.backing_pfn as u64) << page_shift;
    let page_count = ((surf.length.saturating_sub(1)) / page_size) + 1;
    // No host MiB budget: page count follows guest `surf.length` only.
    // Fail if zero or not host-addressable as a page-entry vector.
    if page_count == 0 || crate::runtime::draw::host_alloc_len(page_count).is_none() {
        defer_surface_backing_fail(
            surface_id,
            "page_count_oob",
            Some(backing_base_gva),
            format!(
                "surface_backing_fail reason=page_count_oob sid={surface_id} task={task_id} len={:#x} page_count={page_count}",
                surf.length
            ),
        );
        return false;
    }
    let task = match state.tasks.get(task_id) {
        Some(t) if t.active => t,
        _ => {
            defer_surface_backing_fail(
                surface_id,
                "task_inactive",
                Some(backing_base_gva),
                format!(
                    "surface_backing_fail reason=task_inactive sid={surface_id} task={task_id}"
                ),
            );
            return false;
        }
    };

    // Contract: backing_pfn is getGPUVirtualAddress>>page_shift (GPU-VA page).
    // Translate each consecutive GVA page through the task directory.
    //
    // A failed walk is not an address. The device used to substitute the guest
    // *virtual* address as a guest *physical* one whenever `read_gpa` could
    // touch it, but that probe asks "is this RAM", which nearly all of low
    // guest memory answers yes to. Two things follow, and the second is why
    // this refuses rather than guesses harder.
    //
    // The fabricated PFN goes into `m.pages.entries`, which is the address list
    // every later reader and writer resolves through, so a guess aims real
    // pixel writes at memory the guest allocated for something else — and it
    // stays there, because the guess is cached as the surface's backing.
    //
    // What refusing buys, measured, is a retry. On boot 20260731-192622 both
    // refusals were followed by a full real-walk resolve of the same surface on
    // the same task within one or two frames: the guest had not finished
    // mapping the backing when the device first asked. The callers are
    // per-frame (scanout, bind, draw), so re-asking is already the shape of the
    // code; the guess was standing in for an answer about to be available.
    //
    // Refusing also lets the task search do its job, which a guess ended.
    // `apply_surface_backing` returning `true` stops the loop in
    // `resolve_surface_backing_ex`, and task 0 is probed first, so a guess made
    // task 0 claim surfaces it could not translate. That path is covered by
    // `the_task_search_reaches_the_owner_when_task_zero_cannot_translate`; it
    // has not been observed on the rig, where every attach resolves on task 0.
    let mut entries = Vec::with_capacity(page_count as usize);
    let mut gva_hits = 0u32;
    for i in 0..page_count {
        let gva = ((surf.backing_pfn as u64) + i) << page_shift;
        let Some(gpa) = gva_mem::translate_task_gva(host, task, gva, page_shift) else {
            crate::runtime::drain::note_store_route("surface_backing_translate_refused");
            defer_surface_backing_fail(
                surface_id,
                "translate",
                Some(backing_base_gva),
                surface_backing_translate_fail_detail(
                    surface_id,
                    task_id,
                    i,
                    page_count,
                    gva,
                    &gva_mem::diagnose_task_slot(host, task, task_id, gva, page_shift),
                ),
            );
            return false;
        };
        gva_hits = gva_hits.saturating_add(1);
        let pfn = gpa >> page_shift;
        if pfn > u32::MAX as u64 {
            defer_surface_backing_fail(
                surface_id,
                "pfn_oob",
            Some(backing_base_gva),
                format!("surface_backing_fail reason=pfn_oob sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} pfn={pfn:#x}"),
            );
            return false;
        }
        let entry = ((pfn as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        // Sanity: entry_gpa must round-trip.
        if entry_gpa_shift(entry, page_shift) != Some(gpa & !(page_size - 1)) {
            defer_surface_backing_fail(
                surface_id,
                "entry_roundtrip",
            Some(backing_base_gva),
                format!("surface_backing_fail reason=entry_roundtrip sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} entry={entry:#x}"),
            );
            return false;
        }
        entries.push(entry);
    }
    // Bring-up probe once per surface_id (first attach).
    let first_attach = state
        .surfaces
        .mappings
        .get(&surface_id)
        .map(|m| m.pages.entries.is_empty())
        .unwrap_or(true);
    if first_attach && page_count >= 1 {
        let g0 = entry_gpa_shift(entries[0], page_shift).unwrap_or(0);
        // Three fields that used to ride on this line are gone, all of them
        // probe residue rather than census:
        //
        // - `sample0_nz`: a 16-byte `read_gpa` of the first backing page, per
        //   first attach, to count non-zero bytes. It fed no decision, and it
        //   read `0/16` on every one of the 131 attaches across two driven
        //   boots — a content sniff answering a bring-up question that has been
        //   answered.
        // - `plane0_bytes` and its `bpe0`: where the wire did not state
        //   bytes-per-element, a four-branch ladder guessed one from the format
        //   so a log field could be filled. Deriving a number nothing consumes
        //   is how a guess becomes a rule later.
        // - `gpa1`/`gpa2`: the second and third pages, sampled for no stated
        //   reason. `n` and `gva_hits` already say how many pages resolved.
        //
        // What remains is the census the comment below defends: the identity of
        // the backing, so a refusal and a later resolve can be matched.
        //
        // Bring-up census (dims/fmt), not a drop — the genuine
        // surface backing failures route through note_surface_backing_fail with reason=. On the
        // always-on `off()` sink, not `fail()`: under surface recycling this
        // "first attach" re-fires per recycle (page_entries cleared by the
        // teardown), so on fail() it floods the curated real-error view (~4k
        // lines under a continuously-animating app, burying genuine failures).
        // `gva0` is what the refusal line above prints as `gva=`, so a refusal
        // and a later resolve can be matched by the *backing* they name. Matching
        // them by `sid` alone is unsound: surface ids recycle within a boot and
        // across geometries — sid 145 was a 15x622 scrollbar at t=332488 and a
        // 1225x512 tile 2.8 s later — so "the same surface resolved a frame
        // later" can be a different surface wearing the same id.
        let gva0 = (surf.backing_pfn as u64) << page_shift;
        crate::observe::off(format!(
            "surface_backing pages sid={surface_id} task={task_id} n={page_count} gva_hits={gva_hits} gva0={gva0:#x} gpa0={g0:#x} w={} h={} bpr={} len={:#x} fmt={:#x} planes={} multi={}",
            surf.width,
            surf.height,
            surf.bytes_per_row,
            surf.length,
            surf.pixel_format,
            surf.plane_count,
            surface_backing_is_multiplanar(surf) as u8
        ));
    }

    if !state.ensure_surface_slot(surface_id) {
        defer_surface_backing_fail(
            surface_id,
            "map_surface",
            Some(backing_base_gva),
            format!("surface_backing_fail reason=map_surface sid={surface_id} task={task_id} n={page_count}"),
        );
        return false;
    }
    // Device desc from surface backing wire only (single- or multi-plane). No BGRA invent.
    let device_desc = synthesize_device_desc_from_surface_backing(surf);

    let state_page_shift = state.page_shift;
    let page_count = entries.len();
    let span = crate::runtime::mapper::entry_gpa_span(&entries, state_page_shift);
    let adoption = state
        .adopt_registered_surface_plan(
            reims_vgpu_protocol::SurfaceId::new(surface_id),
            entries,
            reims_vgpu_protocol::TaskId::new(task_id),
            surf.backing_pfn,
            &device_desc,
        )
        .expect("map_surface published the registered-surface slot");
    if adoption.replaced {
        // Recycled-mid backing-refresh census — not a drop. Off the curated
        // fail() view: per-recycle under animation churn it floods the
        // real-error view, at 793 lines in one measured boot.
        crate::observe::off(format!(
            "surface_backing_pages_refreshed sid={surface_id} task={task_id} n={page_count} map_gen={}",
            adoption.lifecycle_generation
        ));
    }
    debug_assert!(!adoption.replaced || adoption.changed);
    // The guest-physical footprint this incarnation authorises us to write.
    // See `mapper::entry_gpa_span`; this is the surface backing adoption site, and it
    // is the one that carried every span in the x86 log.
    //
    // That reading used to be stated as "the page list arrives here, the
    // mapper's own adoption stays silent". It could not have come out any
    // other way: both sites deduped through one `first_sight` namespace on
    // the same key, so this site claimed each footprint it reached first and
    // silenced its peer for that footprint. The namespaces are now
    // `mapper::SPAN_SEEN_SURFACE_BACKING` and `SPAN_SEEN_MAPPER`, so each site's
    // silence is its own.
    //
    // No `changed=` field, though `changed` is in scope and the mapper's
    // peer emitter prints its own. Here it could only ever be 1: the dedup
    // is `first_sight` on the span, and an unchanged plan has by definition
    // the same span as the plan before it, so the unchanged case is filtered
    // out before reaching this line. The one way to arrive here unchanged is
    // the first visit for a surface, and there `prior` is empty, which makes
    // `changed` true.
    if let Some((lo, hi)) = span {
        let key =
            crate::runtime::mapper::span_first_sight_key(surface_id, lo, hi, state_page_shift);
        if crate::observe::first_sight(crate::runtime::mapper::SPAN_SEEN_SURFACE_BACKING, key) {
            crate::observe::off(format!(
                "mapping_gpa_span mid={surface_id} gen={} pages={} src=surface_backing \
                     lo={lo:#x} hi={:#x} pn_lo={:#x} pn_hi={:#x}",
                adoption.lifecycle_generation,
                page_count,
                hi + (1u64 << state_page_shift),
                lo >> state_page_shift,
                hi >> state_page_shift,
            ));
        }
    }

    // Dims come from plane 0 for a multi-plane surface, which is bookkeeping;
    // the format is `latched_mapping_format`'s, which is a contract.
    if surf.width > 0 && surf.height > 0 {
        let _ = state.set_mapping_geom(
            surface_id,
            surf.width,
            surf.height,
            latched_mapping_format(surf),
        );
    }

    // Backing built cleanly — re-arm the fail latch so a later genuine failure
    // on this surface (flapping backing) is logged again, and report the earlier
    // refusal for *this* backing as the recovery it turned out to be.
    clear_surface_backing_fail(surface_id, backing_base_gva);
    true
}

/// Resolve present `surface_id` to surface backing backing pages + geometry.
///
/// Scans active tasks: object-list slot `surface_id` must be surface backing (heap is
/// indexed by IOSurface surface ID). Returns true when pages were latched.
pub fn resolve_surface_backing<M: HostMemory>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_surface_backing_ex(state, host, surface_id, false)
}

/// Like [`resolve_surface_backing`] but always re-reads the object list / PT.
pub fn resolve_surface_backing_force<M: HostMemory>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_surface_backing_ex(state, host, surface_id, true)
}

/// Latch the task that owns `surface_id` as its surface backing backing so the next
/// present-path scan tries it right after task 0.
fn record_surface_backing_owner(state: &mut Device, surface_id: u32, task_id: u32) {
    state.note_surface_owner_hint(
        reims_vgpu_protocol::SurfaceId::new(surface_id),
        reims_vgpu_protocol::TaskId::new(task_id),
    );
}

/// Apply `CmdReplacePhysical` (`0x3c`): the guest re-pointed this resource's
/// GPU-VA range at different physical pages.
///
/// The packet is the announcement, and it is the only one there is. The guest
/// releases the range, rewires the pages, re-commits the *same* GPU-VA with the
/// new PFNs, and then emits one of these per attached resource. Nothing else on
/// the wire says the translation moved — GVA, surface id, geometry and length
/// are all unchanged — so a cached GPA list not dropped here stays trusted while
/// naming pages that now back something else.
///
/// Dropping the list is the whole action. It bumps `map_generation`, which is
/// what retires the [`crate::model::SurfaceBackingWalk`] latch and the resident/deferred
/// state keyed on that incarnation, and the next resolve re-walks the page table
/// the guest has already rewritten.
///
/// The list is cleared immediately because the packet says the PFNs under this
/// GPU-VA have changed. The next access must resolve the replacement backing.
///
/// # The packet names a task-local resource
///
/// The task and object fields are obtained from the same resource object. The
/// object is therefore resolved in the task's namespace; its integer must never
/// be tried as a global mapping id first. Surface ids and resource refs overlap
/// numerically. Treating a ref in task B as mapping `n` can retire the pages of
/// an unrelated surface backing surface `n` owned by task A, after which a compositor
/// draw sees an unbound texture until that surface happens to be mapped again.
///
/// A latched surface backing walk is the exact provenance of a direct surface mapping.
/// IOSurface texture resources carry the task/ref-to-mapping association established at
/// construction. Both routes require the packet's task; a bare integer match is
/// not ownership.
///
/// What the *unreached* re-points turn out to be is measured rather than
/// assumed, and it is mostly benign: 44 of 46 on a driven boot were
/// `replace_physical_unmapped_no_state` — this device holds nothing at all for
/// the object, so the first resolve of that ref reads the page table the guest
/// has already rewritten. Two held a ref-keyed host copy.
/// [`note_replace_physical_unmapped_after_invalidation`] is what separates
/// those, and it exists
/// because a bare "reached nothing" cannot: an announcement with nothing to
/// apply it to and an announcement that missed a live host copy read the same.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReplacePhysicalTransition {
    pub gva_resource_retired: bool,
    pub bound_buffers_retired: usize,
}

pub fn replace_physical<H: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    object_id: u32,
) -> ReplacePhysicalTransition {
    let transition = ReplacePhysicalTransition {
        gva_resource_retired: crate::runtime::writeback_debt::retire_gva_resource(
            state, task_id, object_id,
        ),
        bound_buffers_retired: state.retire_bound_buffers_for_ref(task_id, object_id),
    };
    let target = state
        .surfaces
        .mappings
        .get(&object_id)
        .and_then(|mapping| {
            mapping
                .pages
                .surface_walk
                .filter(|walk| walk.task_id == task_id)
                .map(|_| object_id)
        })
        .or_else(|| {
            state
                .registered_texture_mapping(task_id, object_id)
                .filter(|mid| state.surfaces.mappings.contains_key(mid))
                .inspect(|_| crate::runtime::drain::note_store_route("replace_physical_routed_ref"))
        });

    // A re-point changes every host representation of this resource, whether
    // or not it also owns a mapping. Mapping and ref-keyed caches are different
    // indexes over the same resource lifetime; reaching one does not excuse
    // leaving the other stale.
    let (texture_cache, linear_cache) = state.invalidate_object_host_copies(task_id, object_id);

    let Some(target) = target else {
        note_replace_physical_unmapped_after_invalidation(
            state,
            host,
            task_id,
            object_id,
            texture_cache,
            linear_cache,
        );
        return transition;
    };
    // A deferred window still owed on this mapping is riding the page plan the
    // guest has just replaced. The generation bump below would refuse it anyway,
    // so taking it here changes no outcome — it changes whether the loss has a
    // name. `drop_windows` reports each one against this packet instead of
    // letting it disappear into a generation mismatch nothing attributes.
    let effect = state.invalidate_mapping_pages(target);
    crate::runtime::note_mapping_invalidation(effect);
    let had = effect.had_page_state;
    crate::runtime::drain::note_store_route(if had {
        "replace_physical_dropped"
    } else {
        "replace_physical_no_pages"
    });
    // The views the invalidation retired name pages the guest is re-pointing.
    // Handing them back now, rather than at the next poll, keeps this device from
    // holding a host mapping over memory the guest has already rewired — the same
    // reason the trailing delete flushes here.
    crate::runtime::mapper::flush_retired_views(state, host);
    if had {
        crate::observe::off(format!(
            "replace_physical task={task_id} object={object_id} mid={target} \
             (guest re-pointed the backing; cached page list dropped)"
        ));
    }
    transition
}

/// Say what a re-point named when no mapping belongs to the resource, and whether this
/// device is holding anything else for it.
///
/// A mapping is not the only place a resource's bytes can be cached. The type-2/3
/// rails key their host copies by object-list ref (`host_texture_surfaces`,
/// `host_linear_textures`) rather than by mapping id, and neither carries a page
/// list to notice a move — so "no mapping for this task-local resource" does
/// not settle whether the re-point had anything to invalidate. It only settles
/// that the *mapping* rail had nothing.
///
/// The counters split three ways. `_unmapped_no_state` is a re-point of a
/// resource this device holds nothing for, which is genuinely a no-op — the
/// first resolve of that ref will read the page table the guest has already
/// rewritten. `_unmapped_texture_invalidated` and `_unmapped_linear_invalidated`
/// are the re-points that reached a live host copy, and they now name a repair
/// rather than a loss.
///
/// They used to be `_unmapped_texture_cache` / `_unmapped_linear_cache` and to
/// count only. A host copy whose pages the guest has re-pointed is a copy of
/// memory that is no longer the object's, and leaving it trusted served the
/// guest a stale frame from bytes it had already rewired — with nothing refusing
/// and nothing to read but these two counters. That is what they were added to
/// measure, and they measured it. Driven x86/PCI boot, `web-content-probe -n 10
/// --churn 1`, run on to a settled desktop:
///
/// ```text
///   replace_physical_unknown_object                 56
///   replace_physical_unmapped_no_state              43
///   replace_physical_unmapped_texture_invalidated   13
///   replace_physical_unmapped_linear_invalidated     0
/// ```
///
/// So roughly a quarter of the re-points reaching this branch find a live host
/// copy. The class is intermittent across boots — an earlier one on the same
/// image and probe read 8 — because it depends on which objects hold a host copy
/// at the moment the packet arrives, and the events cluster late in a boot
/// rather than during the probe.
///
/// **These are `store_routes` counters and they are per-window: sum the samples,
/// do not take the maximum.** The series descends across a boot
/// (`unknown_object` = 30, 12, 8), so a `sort -n | tail -1` reads the busiest
/// window and calls it the total — which is how this measurement was first
/// misreported as 37/32/7/1. The check that catches it is the arithmetic:
/// `no_state + texture + linear` must equal `unknown_object`. It does above
/// (43 + 13 = 56); on the maxima it did not (32 + 7 + 1 vs 37).
///
/// Re-taken on the host-pointer-import tree, same probe, 54 windows summed:
/// `17 + 7 + 0 = 24 == 24`, alongside `replace_physical_dropped` 19 and
/// `_no_pages` 5. So the identity survives the guest-memory rail change, which
/// is the plan's §7 line for it.
///
/// **A window drag cannot check this.** A `window-drag-probe` boot reads all
/// four at zero, so the identity holds as `0 == 0` and proves nothing —
/// verifying it needs `web-content-probe --churn 1`, which is what produced
/// both readings above.
///
/// `invalidate_object_host_copies` is the discharge, and it is the same one
/// `delete_object` has always performed for the same two maps — the difference
/// being that a delete also unnames the object while a re-point only moves its
/// bytes. Kept fail-visible after the repair so the reliance stays measurable;
/// a rising count is now this device correctly following the guest, and its
/// disappearance would mean the packet stopped arriving, not that the bug was
/// fixed twice.
///
/// The guest's own object list supplies the type, which is the only authority on
/// what the id names: `lookup_list_entry` reads it at use time and this device
/// caches no copy of it. The type is on the line rather than in a counter
/// because the interesting reading is which types show up at all, and that is a
/// small set a boot enumerates in a handful of lines.
fn note_replace_physical_unmapped_after_invalidation<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    object_id: u32,
    texture_cache: bool,
    linear_cache: bool,
) {
    let object_type = lookup_list_entry(state, host, task_id, object_id).map(|e| e.kind);
    crate::runtime::drain::note_store_route("replace_physical_unknown_object");
    if texture_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_texture_invalidated");
    }
    if linear_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_linear_invalidated");
    }
    if !texture_cache && !linear_cache {
        crate::runtime::drain::note_store_route("replace_physical_unmapped_no_state");
    }
    // Deduped per `(task, object)`: a compositor re-pointing the same resource
    // does it every frame, and the class is what matters here.
    if crate::observe::first_sight(
        "replace_physical_unknown_object",
        (u64::from(task_id) << 32) | u64::from(object_id),
    ) {
        let kind = object_type
            .map(|t| t.to_string())
            .unwrap_or_else(|| "absent".to_string());
        crate::observe::fail(format!(
            "replace_physical_unknown_object task={task_id} object={object_id} \
             obj_type={kind} tex_dropped={} lin_dropped={} \
             (no mapping belongs to this task-local resource; ref-keyed host copies dropped)",
            texture_cache as u8, linear_cache as u8
        ));
    }
}

/// The active tasks whose object list holds an `OBJECT_TYPE_SURFACE` at slot
/// `surface_id` — every task the search could legitimately have stopped on.
///
/// `lookup_list_entry` already refuses an inactive task, an out-of-range slot
/// and an entry with no descriptor, so a task reaching the type test is one with
/// a real object at that slot.
fn surface_backing_claimant_tasks<M: HostMemory>(
    state: &Device,
    host: &M,
    surface_id: u32,
) -> Vec<u32> {
    // The live ids, not a fixed range. This walked `0..256` while the task table
    // was an array; `lookup_list_entry` refuses an inactive task before reading
    // anything, so the ids in between were never claimants and the answer is
    // unchanged. The walk is now the size of the guest's task set.
    state
        .tasks
        .live_ids()
        .filter(|&task_id| {
            probe_list_entry(state, host, task_id, surface_id)
                .is_some_and(|e| e.kind == ObjectKind::SurfaceBacking)
        })
        .collect()
}

/// Report how many active tasks claim `surface_id`, once per surface id.
///
/// The search below takes the first task that produces a translatable backing.
/// If two tasks can, probe order decides which of them the guest gets, and
/// nothing on the wire would say it chose wrong — there is no field to verify a
/// candidate against. The object-list entry is `[type | desc_len]` plus
/// `desc_gva` and carries no identity ([`decode_list_object_entry`]), and the
/// surface backing descriptor is fully consumed: its only undecoded span is the three
/// bytes at `0x11`, which read zero on every distinct shape a driven boot
/// produces (`surface_backing_desc_shape … undecoded_nz=0`).
///
/// So the question the wire cannot answer directly is answered by counting
/// instead. A surface id only ever claimed by one task is a surface whose owner
/// probe order cannot have gotten wrong, whatever order it used.
///
/// The claim test is the object-list slot's type alone, not a descriptor read or
/// a translation: a task that lists a surface backing surface at this slot is a task the
/// search could have stopped on. That keeps the sweep to one 12-byte guest read
/// per active task, and it is taken once per surface id — whether a surface id
/// is claimed twice is a property of the guest's allocation, not of this
/// resolve.
///
/// `claims=1` is the healthy reading and stays on the quiet channel. More than
/// one claimant is the case that makes the search's tie-break load-bearing, so
/// that one is a failure line naming the tasks involved.
fn note_surface_backing_claimants<M: HostMemory>(
    state: &Device,
    host: &M,
    surface_id: u32,
    winner: u32,
) {
    if !crate::observe::first_sight("surface_backing_claimants", surface_id as u64) {
        return;
    }
    let claimants = surface_backing_claimant_tasks(state, host, surface_id);
    let line = format!(
        "surface_backing_claimants sid={surface_id} winner={winner} claims={} tasks={claimants:?}",
        claimants.len()
    );
    if claimants.len() > 1 {
        crate::observe::fail(format!(
            "{line} (more than one task lists this surface id, so probe order \
             chose between them and no wire field can say it chose right)"
        ));
    } else {
        crate::observe::off(line);
    }
}

/// The mapping format a surface backing backing latches: single-plane MTL only.
///
/// Multi-plane and unknown-FourCC surfaces get `0`, and that zero is a decoded
/// refusal rather than an absence — stage and paint must not invent BGRA, and
/// IOSurface texture selects planes through `device_desc` instead.
/// [`iosurface_pixel_format_to_mtl`] states the same rule for the conversion.
///
/// Named rather than inlined at [`apply_surface_backing`] because
/// [`backing_matches_latched_geom`] has to compute the *same* value: it compares
/// a freshly-read descriptor against `m.format_or_zero()`, which is whatever this returned
/// last time.
fn latched_mapping_format(surf: &SurfaceBackingDescriptor) -> u16 {
    if surface_backing_is_multiplanar(surf) {
        return 0;
    }
    iosurface_pixel_format_to_mtl(surf.pixel_format)
}

/// Whether the geometry already latched on this mapping is still the geometry
/// the freshly-read descriptor declares.
///
/// Both arms of [`resolve_surface_backing_ex`]'s freshness test ask this, and they
/// used to ask it differently: the non-force arm compared width **and** height,
/// the force arm compared width only. They are the same question — "may this
/// resolve return without rebuilding" — and the force arm is the one that cannot
/// afford to be looser, because `force_fresh` returns through
/// [`win_surface_backing_search`] *without* calling [`apply_surface_backing`], so neither
/// `set_mapping_geom` nor `synthesize_device_desc_from_surface_backing` runs. A height
/// change that stays inside the same page count therefore left `m.height_or_zero()` and
/// the whole device descriptor describing the previous incarnation, on the exact
/// path `ensure_surface_for_present` calls to catch a wire geometry change.
///
/// Format is compared too, and neither arm used to. A surface id can be recycled
/// at identical dimensions with a different pixel format, and the format is what
/// every read window's bytes-per-pixel comes from — so keeping the old one
/// samples the new backing at the wrong stride. The comparison goes through
/// [`latched_mapping_format`] rather than the wire FourCC, because `m.format_or_zero()` is
/// whatever that function last returned. Comparing the FourCC would report every
/// surface as changed, and comparing the raw conversion would report every
/// multi-plane surface as changed forever, since the latch deliberately discards
/// it in favour of 0.
fn backing_matches_latched_geom(m: &SurfaceMappingEntry, surf: &SurfaceBackingDescriptor) -> bool {
    m.width_or_zero() == surf.width
        && m.height_or_zero() == surf.height
        && m.format_or_zero() == latched_mapping_format(surf)
}

/// The order [`resolve_surface_backing_ex`] probes task object lists in: task 0,
/// then the cached owner hint, then every other task once.
///
/// An iterator rather than a materialised list. The order is unchanged, but
/// building it as a `Vec` allocated 257 elements on every call, and this runs
/// from `ensure_surface_for_present` on every present for every resident
/// mapping — thousands of times a boot to read element 0 and stop.
///
/// A hint of 0 contributes nothing: 0 is already first, so admitting it would
/// only cost the `!= hint` filter its meaning. A hint naming no live task is
/// admitted and then skipped by the liveness test at the probe, which is where
/// that question was always answered.
///
/// The tail is the **live** ids rather than `1..256`. It walked the fixed range
/// while the task table was an array; the probe refuses an inactive task, so
/// those ids were only ever skipped, and yielding them cost the caller a
/// liveness test per id per present. Task 0 leads whether or not it is live —
/// see the caller for why the guest's kernel task is named rather than found —
/// so it is filtered out of the tail rather than left to `live_ids` to omit.
fn surface_backing_probe_order(tasks: &TaskTable, hint: u32) -> Vec<u32> {
    std::iter::once(0)
        .chain(Some(hint).filter(|&h| h != 0))
        .chain(tasks.live_ids().filter(move |&tid| tid != 0 && tid != hint))
        .collect()
}

/// Take `task_id` as the owner of `surface_id` and report the search's exposure.
fn win_surface_backing_search<M: HostMemory>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
    task_id: u32,
) -> bool {
    record_surface_backing_owner(state, surface_id, task_id);
    note_surface_backing_claimants(state, host, surface_id, task_id);
    true
}

fn resolve_surface_backing_ex<M: HostMemory>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
    force: bool,
) -> bool {
    if !crate::model::is_surface_mapping_id(surface_id) {
        return false;
    }
    // Task probe order: task 0 first, then the cached owner-task hint (so a hot
    // present-path re-scan short-circuits on the owning task instead of walking
    // all 256 slots), then the remaining tasks.
    //
    // Task 0 leads because the guest says so, not because it is where surfaces
    // have happened to be. A IOSurface plane view view carries the owning task at
    // [`IOSURFACE_PLANE_VIEW_OWNER_TASK`] and it is the accelerator's kernel task, whose id is a
    // hardcoded 0 and whose slot the task-id allocator reserves before any client
    // task exists. `note_iosurface_plane_view_owner_task` fails loudly if that ever reads
    // otherwise.
    //
    // The remaining 255 probes are not dead weight on that reading. They cost
    // nothing on the path that matters — every successful resolve measured has
    // stopped on the first probe — and they are what makes `surface_backing_claimants` able
    // to say a second task claims the id at all.
    //
    // Built as an iterator rather than a `Vec`. The order is the same one, but
    // materialising it allocated a 257-element vector on every call, and this is
    // called from `ensure_surface_for_present` on every present for every
    // resident mapping — thousands of times a boot to read element 0 and stop.
    let hint = state
        .surfaces
        .mappings
        .get(&surface_id)
        .map(|m| m.owner_task_hint)
        .unwrap_or(0);

    for task_id in surface_backing_probe_order(&state.tasks, hint) {
        // Count the guest-read cost of one active-task object-list probe.
        let Some(entry) = probe_list_entry(state, host, task_id, surface_id) else {
            continue;
        };
        if entry.kind != ObjectKind::SurfaceBacking {
            continue;
        }
        let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
            defer_surface_backing_fail(
                surface_id,
                crate::observe::ladder_slug!("", desc_read),
                None,
                format!(
                    "surface_backing_fail reason=desc_read sid={surface_id} task={task_id} desc_gva={:#x} desc_len={}",
                    entry.descriptor_gva, entry.descriptor_length
                ),
            );
            continue;
        };
        let Some(surf) = decode_surface_backing(&desc) else {
            defer_surface_backing_fail(
                surface_id,
                crate::observe::ladder_slug!("", desc_decode),
                None,
                format!(
                    "surface_backing_fail reason=desc_decode sid={surface_id} task={task_id} desc_len={} backing_pfn={:#x} length={:#x}",
                    desc.len(),
                    reims_vgpu_wire::device_desc::surface_backing_header(&desc)
                        .map(|h| h.backing_pfn.get())
                        .unwrap_or(0),
                    reims_vgpu_wire::device_desc::surface_backing_header(&desc)
                        .map(|h| h.length.get())
                        .unwrap_or(0)
                ),
            );
            continue;
        };
        // Surface backing construction joins the same canonical registry as
        // every other resource family. The old live-object set is not an
        // ownership authority and is no longer populated by product resolve.
        let candidate = Arc::new(TaskResource::new(entry, Arc::from(desc)));
        candidate.decoded_with(|| {
            Ok(crate::runtime::decode::resource::Descriptor::SurfaceBacking(surf.clone()))
        });
        let resource = state
            .task_objects
            .resources
            .register(task_id, surface_id, candidate);
        ensure_resource_relations(state, host, task_id, surface_id, &resource);
        // Force path validated the cached pages are still fresh → keep them.
        let mut force_fresh = false;
        // Skip rebuild when pages already match this backing (hot present path).
        if !force {
            let same_geom = state
                .surfaces
                .mappings
                .get(&surface_id)
                .map(|m| {
                    m.lifecycle.active
                        && !m.pages.entries.is_empty()
                        && m.has_geometry()
                        && backing_matches_latched_geom(m, &surf)
                })
                .unwrap_or(false);
            if same_geom {
                // Same geom + non-empty pages: keep (guest double-buffer
                // may still rewrite page *content* without changing pfn).
                return win_surface_backing_search(state, host, surface_id, task_id);
            }
        } else if let Some(m) = state.surfaces.mappings.get(&surface_id) {
            // Force: keep the cached table only while the CURRENT task
            // page-table translation of the descriptor's first and last
            // backing pages still matches it. `backing_pfn` is a GPU-VA page;
            // the guest may remap that GVA range onto new physical pages
            // without changing surface id, geometry, or length (early-boot
            // console FB vs the WindowServer reallocation). A same-size guard
            // here kept boot-time pages forever, so presents froze on pages
            // nobody writes.
            if m.lifecycle.active && !m.pages.entries.is_empty() {
                let page_shift = state.page_shift;
                let page_size = page_size_of(page_shift);
                let need = ((surf.length.saturating_sub(1)) / page_size) + 1;
                if m.pages.entries.len() as u64 == need && backing_matches_latched_geom(m, &surf) {
                    let task = state.tasks.get(task_id).filter(|t| t.active);
                    let entry_fresh = |idx: u64, entry: u32| -> bool {
                        let gva = ((surf.backing_pfn as u64) + idx) << page_shift;
                        let cached = entry_gpa_shift(entry, page_shift);
                        match task
                            .and_then(|t| gva_mem::translate_task_gva(host, t, gva, page_shift))
                        {
                            Some(gpa) => cached == Some(gpa & !(page_size - 1)),
                            // No translation now, so nothing here can vouch for
                            // the cached table. The device never caches a
                            // GVA-as-GPA entry, so say stale and let the rebuild
                            // refuse, which is what moves the task search on to
                            // the task that can translate.
                            None => false,
                        }
                    };
                    let last = m.pages.entries.len() - 1;
                    if entry_fresh(0, m.pages.entries[0])
                        && entry_fresh(last as u64, m.pages.entries[last])
                    {
                        force_fresh = true;
                    } else {
                        crate::observe::fail(format!(
                            "surface_backing_pages_stale sid={surface_id} task={task_id} n={} gpa0={:#x} (task PT translation moved; rebuilding)",
                            m.pages.entries.len(),
                            entry_gpa_shift(m.pages.entries[0], page_shift).unwrap_or(0)
                        ));
                    }
                }
            }
        }
        if force_fresh {
            return win_surface_backing_search(state, host, surface_id, task_id);
        }
        if apply_surface_backing(state, host, task_id, surface_id, &surf) {
            return win_surface_backing_search(state, host, surface_id, task_id);
        }
    }
    // No task could back it. Only now is a probe's refusal a backing failure.
    flush_surface_backing_fail(surface_id);
    false
}

/// Ensure surface backing for present: surface backing pages when needed, else keep arm
/// MappingInternal path.
///
/// Resolves surface backing once pages are empty; guest double-buffering uses distinct
/// surface_ids (content updates land in-place on an already-mapped pfn).
pub fn ensure_surface_for_present<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut Device,
    host: &M,
    surface_id: u32,
) -> bool {
    if surface_id == 0 {
        return false;
    }
    let need = state
        .surfaces
        .mappings
        .get(&surface_id)
        .map(|m| !m.lifecycle.active || m.pages.entries.is_empty())
        .unwrap_or(true);
    if need {
        let _ = resolve_surface_backing(state, host, surface_id);
    } else {
        // Opportunistic refresh if wire geom changed (mode switch).
        let _ = resolve_surface_backing_force(state, host, surface_id);
    }
    // Arm/iosfc path: MappingInternal resolve when captured.
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, surface_id);
    state
        .surfaces
        .mappings
        .get(&surface_id)
        .map(|m| m.lifecycle.active && !m.pages.entries.is_empty() && m.has_geometry())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
