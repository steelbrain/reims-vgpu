//! Object-list lookup, type-11 registration, and x86 type-4 surface backing.
//!
//! Live layout (reims-vgpu-resource-format): entry `ref` is at
//! `(object_list_pfn << PAGE_SHIFT) + ref * 12` in the task GVA space —
//! `[type|desc_len packed u32][desc_gva u64]`.
//!
//! **x86 type-4 present path (Ventura 13.7 RE):**
//! `AppleParavirtResource::allocateBackingHandle` calls
//! `ResourceHeap::addObject(type=4, objectId=IOSurface::getSurfaceID(), …)` so
//! the object-list index for a surface-backed resource **is** the present
//! `surface_id`. Descriptor layout:
//! length@0, backing_pfn@8, format@0xc, plane_count@0x10, planes@0x14.

use crate::contract::endian::{ld32, ld64, st16, st32, st64};
use crate::contract::iosurface_pages::{
    entry_gpa_shift, page_size_of, DEVICE_DESC_ALLOC_SIZE, DEVICE_DESC_BASE_OFFSET,
    DEVICE_DESC_BPE, DEVICE_DESC_BPR, DEVICE_DESC_DIMS, DEVICE_DESC_LEN, DEVICE_DESC_PIXEL_FORMAT,
    DEVICE_DESC_PLANES, DEVICE_DESC_PLANE_COUNT, DEVICE_PLANE_BPE, DEVICE_PLANE_BPR,
    DEVICE_PLANE_DESC_LEN, DEVICE_PLANE_DIMS, DEVICE_PLANE_OFFSET, DEVICE_PLANE_SIZE,
    PAGE_ENTRY_PFN_SHIFT, PAGE_ENTRY_VALID,
};
use crate::model::{DeviceState, MappingEntry, MAX_MAPPINGS, MAX_TASKS};
use crate::runtime::decode::resource::{
    decode_list_object_entry, list_object_entry_offset, ListObjectEntry, OBJECT_LIST_ENTRY_LEN,
    OBJECT_TYPE_IOSURFACE,
};
use crate::runtime::gva_mem;
use crate::runtime::host::HostMemory;
use crate::runtime::texture;

/// Fail-visible, de-duplicated per `(task_id, ref)`, for the type-11 resolve
/// blind spot: an object ref that IS a type-11 IOSurface texture but whose
/// descriptor cannot be read, cannot register a Metal/Vulkan texture, or carries
/// `mapping_id==0` used to collapse into a bare `None` → a coarse
/// `MissingTexture` at the draw site with no reason. `resolve_type11_ref` runs
/// per-draw per-ref (very hot), so a bare fail line would flood; the latch logs
/// each `(task,ref,reason)` once and is cleared when the ref resolves
/// ([`clear_type11_fail`]). Only genuine failures for a *confirmed IOSurface*
/// ref are routed here — the legitimate "ref is a different object type" and
/// unbound-slot returns stay silent. Runs on the drain worker (off the QEMU main
/// core).
type Type11Failure = (u32, u32, &'static str);
type Type11FailureSet = std::collections::HashSet<Type11Failure>;

fn type11_fail_latch() -> &'static std::sync::Mutex<Type11FailureSet> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<Type11FailureSet>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(Type11FailureSet::new()))
}

fn note_type11_fail(task_id: u32, ref_: u32, reason: &'static str, detail: String) {
    let mut guard = type11_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((task_id, ref_, reason)) {
        crate::observe::fail(detail);
    }
}

/// Re-arm the fail latch for a ref that just resolved, so a later genuine
/// failure on the same ref is logged again (catches flapping).
fn clear_type11_fail(task_id: u32, ref_: u32) {
    let mut guard = type11_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(t, r, _)| !(*t == task_id && *r == ref_));
}

/// Fail-visible, de-duplicated per `(surface_id, reason)`, for the type-4
/// backing blind spot: a surface whose object-list descriptor decoded fine (an
/// active task, a valid `Type4Surface`) but whose page-backing construction then
/// failed — every downstream present/Store for that surface paints **stale or
/// black** with no reason. `apply_type4_backing` is reached from the per-present
/// scanout path (`ensure_surface_for_present`, ~48/s under scroll), so a persistent
/// backing failure would flood; the latch logs each `(surface_id, reason)` once
/// and re-arms when the surface next resolves cleanly ([`clear_type4_fail`]), so a
/// flapping backing is re-logged. Only genuine type-4 candidate failures are
/// routed here — the caller's speculative per-task `continue`s (surface absent
/// from this task or a non-surface object type) stay silent. Runs on the drain
/// worker (off the QEMU main core).
fn type4_fail_latch() -> &'static std::sync::Mutex<std::collections::HashSet<(u32, &'static str)>> {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn note_type4_fail(surface_id: u32, reason: &'static str, detail: String) {
    let mut guard = type4_fail_latch().lock().unwrap_or_else(|e| e.into_inner());
    if guard.insert((surface_id, reason)) {
        crate::observe::fail(detail);
    }
}

/// The first probe failure of the search in progress, per surface.
///
/// A surface lives in exactly one task's object list, so a search that walks
/// tasks in order meets non-owners before it meets the owner. Those misses are
/// the search working, not a backing failure, and reporting them as one is what
/// put ~95 `type4_backing_fail reason=translate` lines on a driven boot's
/// always-on channel for surfaces that then backed perfectly — the resolve
/// succeeded on a later task and the line stayed behind to be read as a defect.
///
/// So a probe records its reason here and nothing is emitted until the search
/// runs out of tasks. The first reason is kept rather than the last: it is the
/// most specific one available, and the tail of a search is dominated by tasks
/// that simply do not list the surface.
fn type4_pending_latch(
) -> &'static std::sync::Mutex<std::collections::HashMap<u32, (&'static str, String)>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static PENDING: OnceLock<Mutex<HashMap<u32, (&'static str, String)>>> = OnceLock::new();
    PENDING.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record why one task's probe refused, to be reported only if none succeeds.
fn defer_type4_fail(surface_id: u32, reason: &'static str, detail: String) {
    let mut guard = type4_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.entry(surface_id).or_insert((reason, detail));
}

/// The search found no task that could back this surface: report the first
/// probe's reason through the flood latch.
fn flush_type4_fail(surface_id: u32) {
    let pending = {
        let mut guard = type4_pending_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guard.remove(&surface_id)
    };
    if let Some((reason, detail)) = pending {
        note_type4_fail(surface_id, reason, detail);
    }
}

/// Re-arm the type-4 fail latch for a surface that just backed cleanly, so a
/// later genuine backing failure on the same surface is logged again, and drop
/// the probe reasons the successful search left behind.
fn clear_type4_fail(surface_id: u32) {
    type4_pending_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&surface_id);
    let mut guard = type4_fail_latch().lock().unwrap_or_else(|e| e.into_inner());
    guard.retain(|(s, _)| *s != surface_id);
}

/// Wire object type for surface / IOSurface backing (x86 Tahoe/Ventura).
pub const OBJECT_TYPE_SURFACE: u8 = 4;
/// RefTextureHandle: surfaceID@0 + cookie@4 + guest blob@8 (texture-ref 28-06-26).
pub const OBJECT_TYPE_REF_TEXTURE: u8 = 5;
/// Type-5 RefTexture descriptor (RE `allocateRefTextureHandle` + Metal
/// `initWithDevice:descriptor:iosurface:plane:field:`):
/// - `surfaceID@0` = `IOSurface::getSurfaceID()` = type-4 heap object id / mid
/// - `ownerTask@4` = the task whose object list holds that surface
/// - `args@8..` = **serialized texture args** length `desc_len-8` (MTLTextureDescriptor
///   stream for the **plane** view; plane is applied guest-side before serialize)
///
/// See [[reims-vgpu-resource-paging]] type-5 section.
pub const TYPE5_SURFACE_ID: usize = 0x00;
/// The task the guest names as the surface's owner, and the answer to the
/// question [`resolve_type4_surface_ex`]'s search is asking.
///
/// This field used to be documented as a "device-side field dword", i.e. as
/// opaque. It is not. `allocateRefTextureHandle` writes it from the *accelerator's*
/// task rather than from its own — the same field the type-4 registration path
/// reads its heap out of — so a type-5 view carries the id of the task whose
/// object list holds the surface it references, not the id of the task that
/// created the view. Those differ by construction: an IOSurface backing is
/// registered in the accelerator's kernel task, while the view is registered in
/// the calling client's, which is why threading the *naming* task into type-4
/// resolution regresses the boot (`AGENTS.md` records that dead end).
///
/// The kernel task's id is an immediate 0 and index 0 is reserved out of the
/// 256-entry task id allocator before any client task exists, so every value
/// seen here is expected to be 0 — which is why the search's "task 0 first" probe
/// order works, and why it is a decoded fact rather than a lucky constant.
/// [`note_type5_owner_task`] is the standing check on that.
pub const TYPE5_OWNER_TASK: usize = 0x04;
pub const TYPE5_ARGS: usize = 0x08;
pub const TYPE5_MIN_LEN: usize = 0x08;

/// Type-5 args blob layout (live wire census 2026-07-14, `compute_stage_tex
/// type5 … args_hex`; 48-byte blob on Ventura 13.7.8 x86):
/// - `+0` u32 kind tag (`0x2f` observed)
/// - `+4` u32 blob length (== `desc_len - TYPE5_ARGS`)
/// - `+8` u32 the type-5 object's **own ref** (same convention as the type-11
///   texture descriptor's object-ref field)
/// - `+12` serialized **plane texture record** — the guest-side
///   `newTextureWithDescriptor:iosurface:plane:` view (plane already applied
///   before serialization; see the `TYPE5_ARGS` doc above):
///   `[+0 u8 tag=0x42][+1 u8 unknown][+2 u16 MTLPixelFormat][+4 u32 width]`
///   `[+8 u32 height][+12 u32 depth][+0x10 trailer][+0x20 u32 IOSurface plane]`
///   Live: `R8 1024×1024 depth=1` = Y plane of a `'420f'` 1024×1024 surface;
///   `BGRA8 68×58`, `RGBA32Uint 482×1928` (uint4 view of a BGRA 1928×1928
///   surface — byte-identical rows) also observed.
///   The **plane index at record `+0x20`** is the
///   `newTextureWithDescriptor:iosurface:plane:` plane argument — live v0a8
///   3-plane blob census (boot 20260717-063043, 10 mappings): Y blobs carry 0,
///   the RG8 chroma blob carries 1, and the second R8 view of identical
///   geometry carries 2 (the alpha plane). Geometry cannot disambiguate Y from
///   alpha; this field is the only wire key. (The type-11 texture descriptor
///   carries no such field — that finding is unchanged.)
pub const TYPE5_ARG_KIND: usize = TYPE5_ARGS;
pub const TYPE5_ARG_BLOB_LEN: usize = TYPE5_ARGS + 0x04;
pub const TYPE5_ARG_OWN_REF: usize = TYPE5_ARGS + 0x08;
pub const TYPE5_ARG_RECORD: usize = TYPE5_ARGS + 0x0c;
pub const TYPE5_RECORD_TAG: u8 = 0x42;
/// Sibling record tag observed live on the blit copy-source path (x86 Ventura
/// 13.7.8, 2026-07-19 six-app launch): full-color texture views (BGRA8_sRGB
/// 1024×768 window backings) carry tag `0x62` where biplanar plane views carry
/// `0x42`. The record layout (format@+2, width@+4, height@+8, depth@+0xc) is
/// byte-identical — the tag distinguishes a variant, not a different geometry
/// encoding — so both decode through the same field offsets.
pub const TYPE5_RECORD_TAG_COLOR_VIEW: u8 = 0x62;
pub const TYPE5_RECORD_FORMAT: usize = 0x02;
pub const TYPE5_RECORD_WIDTH: usize = 0x04;
pub const TYPE5_RECORD_HEIGHT: usize = 0x08;
pub const TYPE5_RECORD_DEPTH: usize = 0x0c;
pub const TYPE5_RECORD_PLANE: usize = 0x20;
pub const TYPE5_RECORD_MIN_LEN: usize = 0x10;

/// Texture view named by a type-5 descriptor's serialized args record.
///
/// This is not limited to IOSurface planes. The live desktop also uses
/// row-byte-equivalent reinterpretations such as a 480-wide RGBA32Uint view
/// over a 1920-wide BGRA8 surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type5TextureView {
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// IOSurface plane the serialized view binds (record `+0x20`); 0 when the
    /// record is too short to carry the field (pre-plane blobs, tests).
    pub plane_index: u32,
}

/// Report the owner task a type-5 descriptor names, once per distinct value.
///
/// Every type-4 surface this device has resolved lived in task 0 — measured
/// (`type4_claimants`: `claims=1 winner=0` on every surface id of two driven
/// boots) and structural, since the guest registers IOSurface backings in the
/// accelerator's kernel task whose id is a hardcoded 0. [`TYPE5_OWNER_TASK`] is
/// the guest saying the same thing on the wire, so this reads 0 forever and
/// stays on the quiet channel.
///
/// A non-zero value is the one reading that matters, and it is a failure line
/// because two things would follow from it at once: the type-4 search's "task 0
/// first" probe order is no longer the guest's answer, and the field's decoded
/// meaning is wrong. `first_sight` is keyed on the value alone, so the whole
/// boot costs one line whichever way it goes.
fn note_type5_owner_task(desc: &[u8]) {
    let Some(bytes) = desc.get(TYPE5_OWNER_TASK..TYPE5_OWNER_TASK + 4) else {
        return;
    };
    let task = ld32(bytes);
    if !crate::observe::first_sight("type5_owner_task", task as u64) {
        return;
    }
    let line = format!("type5_owner_task task={task}");
    if task == 0 {
        crate::observe::off(line);
    } else {
        crate::observe::fail(format!(
            "{line} (a type-5 view names a surface owner other than the kernel task; \
             the type-4 search probes task 0 first on the reading that this is always 0)"
        ));
    }
}

/// Decode the serialized texture-view record from a full type-5 descriptor.
///
/// Fail-closed: `None` unless the record tag matches and geometry is sane
/// (2D, nonzero). The record names the exact Metal view (format + geometry)
/// over the IOSurface bytes; callers must not replace it with base mapping
/// geometry merely because the surface itself is otherwise stageable.
pub fn decode_type5_texture_view(desc: &[u8]) -> Option<Type5TextureView> {
    note_type5_owner_task(desc);
    if desc.len() < TYPE5_ARG_RECORD + TYPE5_RECORD_MIN_LEN {
        return None;
    }
    let rec = &desc[TYPE5_ARG_RECORD..];
    // Accept both the biplanar-plane record tag (0x42) and the full-color
    // texture-view variant (0x62); both share the field layout below. Any other
    // tag stays unknown → fail closed (no invented geometry).
    if rec[0] != TYPE5_RECORD_TAG && rec[0] != TYPE5_RECORD_TAG_COLOR_VIEW {
        return None;
    }
    let pixel_format = u16::from_le_bytes([rec[TYPE5_RECORD_FORMAT], rec[TYPE5_RECORD_FORMAT + 1]]);
    let width = ld32(&rec[TYPE5_RECORD_WIDTH..]);
    let height = ld32(&rec[TYPE5_RECORD_HEIGHT..]);
    let depth = ld32(&rec[TYPE5_RECORD_DEPTH..]);
    if pixel_format == 0 || width == 0 || height == 0 || depth != 1 {
        return None;
    }
    let plane_index = if rec.len() >= TYPE5_RECORD_PLANE + 4 {
        ld32(&rec[TYPE5_RECORD_PLANE..])
    } else {
        0
    };
    Some(Type5TextureView {
        pixel_format,
        width,
        height,
        depth,
        plane_index,
    })
}

/// Type-4 descriptor field offsets (RE allocateBackingHandle / tahoe §9.4).
pub const TYPE4_LEN: usize = 0x00;
pub const TYPE4_BACKING_PFN: usize = 0x08;
pub const TYPE4_PIXEL_FORMAT: usize = 0x0c;
pub const TYPE4_PLANE_COUNT: usize = 0x10;
pub const TYPE4_PLANES: usize = 0x14;
pub const TYPE4_PLANE_STRIDE: usize = 0x10;
pub const TYPE4_MIN_LEN: usize = 0x24;
/// Max plane records in type-4 wire / device desc (IOSurface getPlaneCount cap).
pub const TYPE4_PLANE_CAP: usize = 8;

/// CoreVideo / IOSurface biplanar 420 full-range (`'420f'`).
pub const IOSURFACE_FOURCC_420F: u32 = 0x3432_3066;
/// CoreVideo / IOSurface biplanar 420 video-range (`'420v'`).
pub const IOSURFACE_FOURCC_420V: u32 = 0x3432_3076;

/// One type-4 plane record (stride 0x10 @ +0x14): offset, w, h, bpr|bpe<<24.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Type4Plane {
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    /// From packed high 8 bits (`getPlaneBytesPerElement`); 0 if wire left it 0.
    pub bytes_per_element: u8,
}

/// Decoded type-4 surface backing descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Type4Surface {
    pub length: u64,
    pub backing_pfn: u32,
    /// Wire `pixelFormat@0xc` — OSType FourCC or small MTL ordinal.
    pub pixel_format: u32,
    pub plane_count: u8,
    pub planes: [Type4Plane; TYPE4_PLANE_CAP],
    /// Plane0 convenience (present / single-plane geom).
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

/// CoreVideo biplanar 8-bit 420 family — **not** a single `MTLPixelFormat`.
///
/// Metal binds planes via `newTextureWithDescriptor:iosurface:plane:` as R8 (Y)
/// and RG8 (UV). Product must not invent BGRA.
#[inline]
pub fn iosurface_fourcc_is_biplanar(pixel_format: u32) -> bool {
    matches!(pixel_format, IOSURFACE_FOURCC_420F | IOSURFACE_FOURCC_420V)
}

/// True when type-4 / mapping cannot be staged as one linear color texture.
#[inline]
pub fn type4_is_multiplanar(surf: &Type4Surface) -> bool {
    surf.plane_count > 1 || iosurface_fourcc_is_biplanar(surf.pixel_format)
}

/// Mapping has multi-plane device geometry (plane_count≥2) or biplanar FourCC.
pub fn mapping_is_multiplanar(m: &MappingEntry) -> bool {
    use crate::contract::iosurface_pages::decode_device_surface;
    if let Some(s) = decode_device_surface(&m.device_desc) {
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
/// synthesizes the descriptor and [`synthesize_device_desc_from_type4`] writes
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
/// refusal the type-4 path latches, never an invented BGRA8.
pub fn device_desc_format_to_mtl(raw: u32) -> u16 {
    if raw <= u16::MAX as u32 {
        return raw as u16;
    }
    iosurface_pixel_format_to_mtl(raw)
}

/// Map IOSurface OSType FourCC (or MTL raw) to a **single-plane** MTL pixel format.
///
/// Live x86 type-4 carries IOSurface `pixelFormat` as a FourCC (e.g. `'BGRA'` =
/// `0x42475241`). Truncating to u16 yields `0x5241` which is not a Metal format.
///
/// Returns **0** when:
/// - format is multi-plane (e.g. `'420f'` / `'420v'`) — no single MTLPixelFormat
/// - format is unknown — fail closed; **do not** invent BGRA8
///
/// Unknown formats fail closed.
pub fn iosurface_pixel_format_to_mtl(pixel_format: u32) -> u16 {
    use crate::contract::pixel_format::{
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
    // caller here passes a type-4 `pixelFormat` (+0x0c), which is an IOSurface
    // OSType — a four-character code, so never below 0x20202020. The type-11 and
    // type-5 rails carry their MTL ordinal in a `u16` field of their own and do
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
        // 'L008' / common R8 fourccs are rare on type-4; MTL ordinals already handled above.
        // 'R8  ' / 'RG08' if ever seen as OSType:
        0x5238_2020 => MTL_FORMAT_R8_UNORM,
        0x5247_3038 => MTL_FORMAT_RG8_UNORM,
        // Unknown FourCC: 0 — callers fail closed (no BGRA invent).
        _ => 0,
    }
}

/// Decode one type-4 plane at `TYPE4_PLANES + i*TYPE4_PLANE_STRIDE`.
fn decode_type4_plane(desc: &[u8], plane_index: usize) -> Option<Type4Plane> {
    let base = TYPE4_PLANES + plane_index * TYPE4_PLANE_STRIDE;
    if desc.len() < base + TYPE4_PLANE_STRIDE {
        return None;
    }
    let offset = ld32(&desc[base..]);
    let width = ld32(&desc[base + 4..]);
    let height = ld32(&desc[base + 8..]);
    let packed = ld32(&desc[base + 12..]);
    let bytes_per_row = packed & 0x00ff_ffff;
    let bytes_per_element = ((packed >> 24) & 0xff) as u8;
    Some(Type4Plane {
        offset,
        width,
        height,
        bytes_per_row,
        bytes_per_element,
    })
}

/// The bytes of a type-4 surface descriptor that [`decode_type4_surface`] does
/// **not** read: `+0x11..0x14` and everything past the plane records it
/// consumed (`TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE ..`).
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
/// type4_desc_shape distinct=1 1920x1080 fmt=0x42475241 planes=1 len=36 undecoded_len=3 undecoded_nz=0
/// type4_desc_shape distinct=2   320x320 fmt=0x34323066 planes=2 len=52 undecoded_len=3 undecoded_nz=0
/// ```
///
/// `len` is `TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE` exactly, and it is
/// the *guest's* number — [`read_descriptor`] honours `descriptor_length` with no
/// clamp. The record ends where the plane array ends; the only bytes we skip are
/// the three at `+0x11`, and they were zero every time. There is nowhere in this
/// descriptor for a usage, bind, scanout or role hint to be, so no rule over
/// surface identity can classify a brand-new buffer before its first draw.
///
/// Narrow: this is the type-4 record on the x86 PCI pathway. It says nothing
/// about type-11 (`decode_iosurface_texture_descriptor`, which does not run
/// here and whose 0x38/0x58 blobs are still read only to 0x20), and a
/// create-time record we never read at all would be invisible to it.
///
/// A `plane_count` above [`TYPE4_PLANE_CAP`] is clamped by the decoder, so the
/// records past the clamp fall into this span too — which is correct: they are
/// bytes we did not read.
///
/// Public so the probe's notion of "undecoded" is pinned by a test rather than
/// restated in a log format string.
pub fn undecoded_type4_surface_bytes(desc: &[u8]) -> Vec<u8> {
    if desc.len() < TYPE4_MIN_LEN {
        return Vec::new();
    }
    let plane_count = (desc[TYPE4_PLANE_COUNT] as usize).min(TYPE4_PLANE_CAP);
    let planes_end = TYPE4_PLANES + plane_count * TYPE4_PLANE_STRIDE;
    let mut out = Vec::new();
    out.extend_from_slice(&desc[0x11..TYPE4_PLANES]);
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
/// decode still reports. An earlier version of this probe on the type-11
/// descriptor sat after its length check and emitted nothing at all on a live
/// boot; "the decoder never ran" and "the tail is constant" produced the same
/// silence, which is the reading the probe exists to rule out.
///
/// Hitting the cap is reported once. A silent truncation would read like "we
/// saw everything", which is the same class of error as a probe reporting a
/// confident constant.
fn note_type4_surface_shape(desc: &[u8]) {
    const MAX_SHAPES: usize = 24;
    const HEX_MAX: usize = 128;
    use std::sync::Mutex;
    type ShapeKey = (usize, Vec<u8>);
    static SEEN: Mutex<Option<std::collections::BTreeSet<ShapeKey>>> = Mutex::new(None);

    let undecoded = undecoded_type4_surface_bytes(desc);
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
            "type4_desc_shape outcome=cap_reached distinct={distinct} \
             (the undecoded span varies per surface; it is not a constant tail)"
        ));
        return;
    }
    let (w, h, fmt, pc) = if desc.len() >= TYPE4_MIN_LEN {
        (
            ld32(&desc[TYPE4_PLANES + 4..]),
            ld32(&desc[TYPE4_PLANES + 8..]),
            ld32(&desc[TYPE4_PIXEL_FORMAT..]),
            desc[TYPE4_PLANE_COUNT],
        )
    } else {
        (0, 0, 0, 0)
    };
    let hex: String = desc
        .iter()
        .take(HEX_MAX)
        .map(|b| format!("{b:02x}"))
        .collect();
    crate::observe::fail(format!(
        "type4_desc_shape distinct={distinct} {w}x{h} fmt={fmt:#x} planes={pc} len={} \
         undecoded_len={} undecoded_nz={} hex={hex}{}",
        desc.len(),
        undecoded.len(),
        undecoded.iter().filter(|&&b| b != 0).count(),
        if desc.len() > HEX_MAX { "…" } else { "" },
    ));
}

/// Report, once per reason, that the type-4 decoder dropped something the guest
/// declared.
///
/// Deduped rather than sampled: each reason names a distinct shape of blob, and
/// a surface stream re-decodes the same descriptor thousands of times a boot, so
/// an undeduped line would flood while adding nothing. The first occurrence is
/// what a reader needs — after it, `type4_desc_shape` carries the geometry.
fn type4_decode_drop_latch() -> &'static std::sync::Mutex<std::collections::HashSet<&'static str>> {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<&'static str>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

fn note_type4_decode_drop(reason: &'static str, detail: String) {
    let fresh = {
        let mut guard = type4_decode_drop_latch()
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
fn reset_type4_decode_drops() {
    type4_decode_drop_latch()
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
}

/// Decode a type-4 surface descriptor blob.
pub fn decode_type4_surface(desc: &[u8]) -> Option<Type4Surface> {
    note_type4_surface_shape(desc);
    if desc.len() < TYPE4_MIN_LEN {
        return None;
    }
    let length = ld64(&desc[TYPE4_LEN..]);
    let backing_pfn = ld32(&desc[TYPE4_BACKING_PFN..]);
    let pixel_format = ld32(&desc[TYPE4_PIXEL_FORMAT..]);
    let plane_count_raw = desc[TYPE4_PLANE_COUNT];
    if backing_pfn == 0 || length == 0 {
        return None;
    }
    let plane_count = (plane_count_raw as usize).min(TYPE4_PLANE_CAP) as u8;
    if plane_count_raw as usize > TYPE4_PLANE_CAP {
        // The bound itself is right — IOSurface's own `getPlaneCount` caps at
        // eight — but dropping the surplus quietly is not. The guest declared
        // planes this device will never look at, and every later reader sees a
        // surface that simply has eight.
        note_type4_decode_drop(
            "plane_count_over_cap",
            format!(
                "type4_decode_drop reason=plane_count_over_cap declared={plane_count_raw} \
                 cap={TYPE4_PLANE_CAP} fmt={pixel_format:#x}"
            ),
        );
    }
    let mut planes = [Type4Plane::default(); TYPE4_PLANE_CAP];
    for (i, plane) in planes.iter_mut().enumerate().take(plane_count as usize) {
        match decode_type4_plane(desc, i) {
            Some(p) => *plane = p,
            // A declared plane whose record the blob does not reach. Leaving the
            // default in place publishes a 0x0 plane as if the guest had asked
            // for one, which reads downstream as a surface with no content
            // rather than as a descriptor we could not decode.
            None => note_type4_decode_drop(
                "plane_record_short",
                format!(
                    "type4_decode_drop reason=plane_record_short plane={i} \
                     planes={plane_count} desc_len={} fmt={pixel_format:#x}",
                    desc.len()
                ),
            ),
        }
    }
    let (width, height, bpr) = if plane_count > 0 {
        let p0 = planes[0];
        (p0.width, p0.height, p0.bytes_per_row)
    } else {
        (0, 0, 0)
    };
    Some(Type4Surface {
        length,
        backing_pfn,
        pixel_format,
        plane_count,
        planes,
        width,
        height,
        bytes_per_row: bpr,
    })
}

/// Build `sIOSurfaceDeviceDescriptor` geometry from type-4 wire (no invent).
///
/// Multi-plane: plane records from type-4 planes; sample path selects by
/// geometry. Single-plane: surface-level fields only
/// (`plane_count==0` path in `sample_window_prefer_device`).
fn synthesize_device_desc_from_type4(surf: &Type4Surface) -> Vec<u8> {
    let mut device_desc = vec![0u8; DEVICE_DESC_LEN];
    let multi = type4_is_multiplanar(surf);
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
        note_type4_decode_drop(
            "alloc_size_over_u32",
            format!(
                "type4_decode_drop reason=alloc_size_over_u32 length={} \
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
    // Surface-level dims/bpr from plane0 (same as type-4 plane0 convenience).
    let dims = ((surf.width as u64) << 8) | ((surf.height as u64) << 40);
    st64(&mut device_desc[DEVICE_DESC_DIMS..], dims);
    if surf.bytes_per_row > 0 {
        st32(&mut device_desc[DEVICE_DESC_BPR..], surf.bytes_per_row);
    }
    if multi && surf.plane_count > 0 {
        // Multi-plane: publish plane records; sample_window_prefer_device matches
        // type-11 R8/RG8 binds by (w,h,bpe). Do not invent bases from format alone.
        let n = (surf.plane_count as usize).min(TYPE4_PLANE_CAP);
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
            // (type-4 wire has offset/w/h/bpr, not a separate size field).
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
        // `decode_type4_plane` reads four fields per plane; the surface-level
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
        // population that was being misread, and `type4_base_offset_nonzero`
        // counts how large that is. Read on a driven x86/Vulkan boot: **0**, so
        // no single-plane surface on that workload starts past its base and this
        // is contract fidelity rather than a live repair. It is also why the
        // change was safe to make without a rate: every window it could move is
        // one the counter would have named.
        let base_offset = surf.planes[0].offset;
        if base_offset != 0 {
            crate::runtime::drain::note_store_route("type4_base_offset_nonzero");
            st32(&mut device_desc[DEVICE_DESC_BASE_OFFSET..], base_offset);
        }
        if mtl != 0 {
            if let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(mtl) {
                st16(&mut device_desc[DEVICE_DESC_BPE..], bpp as u16);
            }
        }
    }
    device_desc
}

/// Lookup one object-list slot for `task_id` / `ref_`.
pub fn lookup_list_entry<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<ListObjectEntry> {
    if task_id as usize >= MAX_TASKS {
        return None;
    }
    let task = &state.tasks[task_id as usize];
    if !task.active {
        return None;
    }
    if task.object_list_count == 0 {
        return None;
    }
    // A ref past the end of a published list is not a "not ready" miss; it is a
    // stale count or bad ref and would otherwise hide the lost guest object.
    let Some(off) = list_object_entry_offset(ref_, task.object_list_count) else {
        crate::observe::fail(format!(
            "object_list_miss reason=ref_beyond_count task={task_id} ref={ref_} count={}",
            task.object_list_count
        ));
        return None;
    };
    let entry_gva = ((task.object_list_pfn as u64) << state.page_shift).checked_add(off)?;
    let mut raw = [0u8; OBJECT_LIST_ENTRY_LEN];
    // An unmapped entry page and an empty slot are BOTH expected: the guest
    // allocates a sparse object list and maps only the pages it has filled, so
    // a probe of an unpopulated ref is control flow, not a loss. Measured at
    // 227k unreadable and 73k empty in one login — reporting either would drown
    // the sink in the rail working correctly.
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        entry_gva,
        &mut raw,
        state.page_shift,
    )
    .ok()?;
    let Ok(e) = decode_list_object_entry(&raw) else {
        crate::observe::fail(format!(
            "object_list_miss reason=entry_undecodable task={task_id} ref={ref_}"
        ));
        return None;
    };
    if e.descriptor_length == 0 || e.descriptor_gva == 0 {
        return None;
    }
    Some(e)
}

/// Read the descriptor blob for a list entry.
pub fn read_descriptor<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    entry: &ListObjectEntry,
) -> Option<Vec<u8>> {
    // Guest descriptor_length is authoritative — no product 4 KiB read clamp.
    let len = crate::runtime::metal_draw::host_alloc_len(entry.descriptor_length as u64)
        .filter(|&n| n > 0)?;
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

/// Resolve object ref and, if type-11, latch mapping geometry + cache the entry.
///
/// Returns the mapping_id for type-11 textures, or None.
pub fn resolve_type11_ref<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    ref_: u32,
) -> Option<u32> {
    let entry = lookup_list_entry(state, host, task_id, ref_)?;
    // The list entry passed validation (descriptor_gva != 0, length != 0) but
    // its descriptor blob is unreadable — genuine, only for a bound entry.
    let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
        note_type11_fail(
            task_id,
            ref_,
            "type11_desc_read",
            format!(
                "type11_resolve_fail reason=type11_desc_read task={task_id} ref={ref_} obj_type={} desc_gva={:#x} desc_len={}",
                entry.object_type, entry.descriptor_gva, entry.descriptor_length
            ),
        );
        return None;
    };
    // Record the ref as live; the type and descriptor come from the guest's own
    // list at every use, never from here.
    let _ = state.insert_object(task_id, ref_);
    if entry.object_type != OBJECT_TYPE_IOSURFACE {
        // Legitimate: this ref is a different object type, not a texture. Normal
        // control flow (resolve_type11_refs skips it) — never a failure.
        return None;
    }
    if !texture::register_from_descriptor_bytes(state, OBJECT_TYPE_IOSURFACE, &desc) {
        // A confirmed IOSurface texture whose descriptor could not register —
        // the draw then samples a missing/black texture.
        note_type11_fail(
            task_id,
            ref_,
            "type11_register",
            format!(
                "type11_resolve_fail reason=type11_register task={task_id} ref={ref_} desc_len={}",
                desc.len()
            ),
        );
        return None;
    }
    // mapping_id is first u32 of type-11 desc.
    let mapping_id = u32::from_le_bytes(desc[0..4].try_into().ok()?);
    if mapping_id == 0 {
        note_type11_fail(
            task_id,
            ref_,
            "type11_mapping_zero",
            format!("type11_resolve_fail reason=type11_mapping_zero task={task_id} ref={ref_} desc_len={}", desc.len()),
        );
        return None;
    }
    state.texture_to_mapping.insert((task_id, ref_), mapping_id);
    // Resolved: re-arm so a later genuine failure on this ref logs again.
    clear_type11_fail(task_id, ref_);
    Some(mapping_id)
}

/// The detail line a refused page walk reports.
///
/// # Why the walk status is on it
///
/// [`crate::contract::gva_resolve::ResolveStatus`] distinguishes fifteen checks
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
fn type4_translate_fail_detail(
    surface_id: u32,
    task_id: u32,
    page: u64,
    page_count: u64,
    gva: u64,
    walk: &str,
) -> String {
    format!(
        "type4_backing_fail reason=translate sid={surface_id} task={task_id} \
         page={page}/{page_count} gva={gva:#x} walk=[{walk}] \
         (no translation in this task; not substituting the GVA)"
    )
}

/// Apply a decoded type-4 surface as page-table backing for `surface_id`.
///
/// `backing_pfn` is a GPU-VA page (same source as type-2/3 textures). Translate
/// each consecutive GVA page through the task page table into GPA page entries
/// the scanout path already understands.
fn apply_type4_backing<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    task_id: u32,
    surface_id: u32,
    surf: &Type4Surface,
) -> bool {
    if surface_id == 0 || surface_id as usize >= MAX_MAPPINGS {
        defer_type4_fail(
            surface_id,
            "sid_oob",
            format!("type4_backing_fail reason=sid_oob sid={surface_id} task={task_id} max={MAX_MAPPINGS}"),
        );
        return false;
    }
    let page_shift = state.page_shift;
    let page_size = page_size_of(page_shift);
    if page_size == 0 {
        defer_type4_fail(
            surface_id,
            "page_size_zero",
            format!("type4_backing_fail reason=page_size_zero sid={surface_id} task={task_id} page_shift={page_shift}"),
        );
        return false;
    }
    let page_count = ((surf.length.saturating_sub(1)) / page_size) + 1;
    // No host MiB budget: page count follows guest `surf.length` only.
    // Fail if zero or not host-addressable as a page-entry vector.
    if page_count == 0 || crate::runtime::metal_draw::host_alloc_len(page_count).is_none() {
        defer_type4_fail(
            surface_id,
            "page_count_oob",
            format!(
                "type4_backing_fail reason=page_count_oob sid={surface_id} task={task_id} len={:#x} page_count={page_count}",
                surf.length
            ),
        );
        return false;
    }
    let task = match state.tasks.get(task_id as usize) {
        Some(t) if t.active => t,
        _ => {
            defer_type4_fail(
                surface_id,
                "task_inactive",
                format!("type4_backing_fail reason=task_inactive sid={surface_id} task={task_id}"),
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
    // The fabricated PFN goes into `m.page_entries`, which is the address list
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
    // `apply_type4_backing` returning `true` stops the loop in
    // `resolve_type4_surface_ex`, and task 0 is probed first, so a guess made
    // task 0 claim surfaces it could not translate. That path is covered by
    // `the_task_search_reaches_the_owner_when_task_zero_cannot_translate`; it
    // has not been observed on the rig, where every attach resolves on task 0.
    let mut entries = Vec::with_capacity(page_count as usize);
    let mut gva_hits = 0u32;
    for i in 0..page_count {
        let gva = ((surf.backing_pfn as u64) + i) << page_shift;
        let Some(gpa) = gva_mem::translate_task_gva(host, task, gva, page_shift) else {
            crate::runtime::drain::note_store_route("type4_translate_refused");
            defer_type4_fail(
                surface_id,
                "translate",
                type4_translate_fail_detail(
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
            defer_type4_fail(
                surface_id,
                "pfn_oob",
                format!("type4_backing_fail reason=pfn_oob sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} pfn={pfn:#x}"),
            );
            return false;
        }
        let entry = ((pfn as u32) << PAGE_ENTRY_PFN_SHIFT) | PAGE_ENTRY_VALID;
        // Sanity: entry_gpa must round-trip.
        if entry_gpa_shift(entry, page_shift) != Some(gpa & !(page_size - 1)) {
            defer_type4_fail(
                surface_id,
                "entry_roundtrip",
                format!("type4_backing_fail reason=entry_roundtrip sid={surface_id} task={task_id} page={i}/{page_count} gpa={gpa:#x} entry={entry:#x}"),
            );
            return false;
        }
        entries.push(entry);
    }
    // Bring-up probe once per surface_id (first attach).
    let first_attach = state
        .mappings
        .get(&surface_id)
        .map(|m| m.page_entries.is_empty())
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
        // type-4 failures route through note_type4_fail with reason=. On the
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
            "type4 pages sid={surface_id} task={task_id} n={page_count} gva_hits={gva_hits} gva0={gva0:#x} gpa0={g0:#x} w={} h={} bpr={} len={:#x} fmt={:#x} planes={} multi={}",
            surf.width,
            surf.height,
            surf.bytes_per_row,
            surf.length,
            surf.pixel_format,
            surf.plane_count,
            type4_is_multiplanar(surf) as u8
        ));
    }

    if !state.map_surface(surface_id) {
        defer_type4_fail(
            surface_id,
            "map_surface",
            format!("type4_backing_fail reason=map_surface sid={surface_id} task={task_id} n={page_count}"),
        );
        return false;
    }
    // Device desc from type-4 wire only (single- or multi-plane). No BGRA invent.
    let device_desc = synthesize_device_desc_from_type4(surf);

    let state_page_shift = state.page_shift;
    if let Some(m) = state.mappings.get_mut(&surface_id) {
        // `map_surface` above stashed the prior bindings as the incarnation
        // fingerprint (the notify-vs-eager-resolve rule): compare the fresh
        // plan against it — identical pages are the SAME incarnation (no
        // bump; deferred windows and the resident survive), a change is the
        // recycled-mid rule (bump; stale residents/views must never survive).
        let prior = m
            .condemned_entries
            .take()
            .unwrap_or_else(|| std::mem::take(&mut m.page_entries));
        let changed = prior != entries;
        let replaced = !prior.is_empty() && changed;
        if changed {
            crate::model::DeviceState::bump_map_generation(m);
        }
        if replaced {
            // Recycled-mid backing-refresh census — not a drop. Off the curated
            // fail() view: per-recycle under animation churn it floods the
            // real-error view, at 793 lines in one measured boot.
            crate::observe::off(format!(
                "type4_pages_refreshed sid={surface_id} task={task_id} n={} map_gen={}",
                entries.len(),
                m.map_generation
            ));
            // Present evidence needs no prune here: this branch only runs when
            // the plan changed, which bumped `map_generation` just above, and
            // the evidence is stamped with the incarnation that recorded it.
            // Pruning it unconditionally is what the identical-plan path used
            // to do via `map_surface`, and that demoted a surface the compare
            // had just called the SAME incarnation.
        }
        // The guest-physical footprint this incarnation authorises us to write.
        // See `mapper::entry_gpa_span`; this is the type-4 adoption site, and it
        // is the one that carried every span in the x86 log.
        //
        // That reading used to be stated as "the page list arrives here, the
        // mapper's own adoption stays silent". It could not have come out any
        // other way: both sites deduped through one `first_sight` namespace on
        // the same key, so this site claimed each footprint it reached first and
        // silenced its peer for that footprint. The namespaces are now
        // `mapper::SPAN_SEEN_TYPE4` and `SPAN_SEEN_MAPPER`, so each site's
        // silence is its own.
        //
        // No `changed=` field, though `changed` is in scope and the mapper's
        // peer emitter prints its own. Here it could only ever be 1: the dedup
        // is `first_sight` on the span, and an unchanged plan has by definition
        // the same span as the plan before it, so the unchanged case is filtered
        // out before reaching this line. The one way to arrive here unchanged is
        // the first visit for a surface, and there `prior` is empty, which makes
        // `changed` true. The mapper's copy is not in that position — its
        // reprieve path can repopulate an emptied entry list with no change.
        if let Some((lo, hi)) = crate::runtime::mapper::entry_gpa_span(&entries, state_page_shift) {
            let key =
                crate::runtime::mapper::span_first_sight_key(surface_id, lo, hi, state_page_shift);
            if crate::observe::first_sight(crate::runtime::mapper::SPAN_SEEN_TYPE4, key) {
                crate::observe::off(format!(
                    "mapping_gpa_span mid={surface_id} gen={} pages={} src=type4 \
                     lo={lo:#x} hi={:#x} pn_lo={:#x} pn_hi={:#x}",
                    m.map_generation,
                    entries.len(),
                    hi + (1u64 << state_page_shift),
                    lo >> state_page_shift,
                    hi >> state_page_shift,
                ));
            }
        }
        // The type-4 peer of the adoption in `mapper::resolve_mapping_backing`:
        // these pages are a surface's again, so the write-after-teardown
        // detector must stop reporting writes into them.
        crate::observe::footprint::note_pages_authorized(
            entries.iter().filter_map(|&e| {
                crate::contract::iosurface_pages::entry_gpa_shift(e, state_page_shift)
            }),
            crate::contract::iosurface_pages::page_size_of(state_page_shift),
        );
        m.page_entries = entries;
        m.mapped = true;
        m.page_table_kva = 0;
        m.device_desc = device_desc;
        // Latched with the list rather than near it: this records the walk that
        // produced the entries above, so a later reader can repeat that walk —
        // and find out whether the entries still name the guest's memory —
        // without repeating the object search that found the surface. Written at
        // the assignment so the two cannot be updated independently.
        m.type4_walk = Some(crate::model::Type4Walk {
            task_id,
            backing_pfn: surf.backing_pfn,
            map_generation: m.map_generation,
        });
        // Contiguous view must be rebuilt.
        if m.contig_ptr != 0 {
            state.retired_views.push((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
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
    // on this surface (flapping backing) is logged again.
    clear_type4_fail(surface_id);
    true
}

/// Resolve present `surface_id` to type-4 backing pages + geometry.
///
/// Scans active tasks: object-list slot `surface_id` must be type-4 (heap is
/// indexed by IOSurface surface ID). Returns true when pages were latched.
pub fn resolve_type4_surface<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_type4_surface_ex(state, host, surface_id, false)
}

/// Like [`resolve_type4_surface`] but always re-reads the object list / PT.
pub fn resolve_type4_surface_force<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    resolve_type4_surface_ex(state, host, surface_id, true)
}

/// Latch the task that owns `surface_id` as its type-4 backing so the next
/// present-path scan tries it right after task 0.
fn record_type4_owner(state: &mut DeviceState, surface_id: u32, task_id: u32) {
    if let Some(m) = state.mappings.get_mut(&surface_id) {
        m.owner_task_hint = task_id;
    }
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
/// what retires the [`crate::model::Type4Walk`] latch and the resident/deferred
/// state keyed on that incarnation, and the next resolve re-walks the page table
/// the guest has already rewritten.
///
/// `object_id` is the mapping id, and is used as one directly.
///
/// It is not looked up in `texture_to_mapping` first. That map is keyed by the
/// task object-list *ref*, and a ref and a surface id are different id spaces
/// that collide — `blit_exec`'s type-5 resolve states the rule ("never the task
/// object-list ref — those id spaces collide"). A ref-keyed lookup ahead of the
/// direct reading would therefore silently misroute a packet naming surface `n`
/// onto whatever mapping the same task has registered under ref `n`, and
/// invalidate a surface the guest said nothing about while leaving the one it
/// did name stale. Every observed packet resolves directly: 40 in a driven boot,
/// all with `mid == object_id`, and the ref-keyed arm never answered.
///
/// A type-11 texture that is re-pointed is therefore not handled here yet, and
/// that is deliberate — which id its packet would carry is not established, and
/// guessing costs the packets that do arrive.
pub fn replace_physical(state: &mut DeviceState, task_id: u32, object_id: u32) {
    let had = state.invalidate_mapping_pages(object_id);
    crate::runtime::drain::note_store_route(if had {
        "replace_physical_dropped"
    } else {
        "replace_physical_nothing_cached"
    });
    if had {
        crate::observe::off(format!(
            "replace_physical task={task_id} mid={object_id} \
             (guest re-pointed the backing; cached page list dropped)"
        ));
    }
}

/// The active tasks whose object list holds an `OBJECT_TYPE_SURFACE` at slot
/// `surface_id` — every task the search could legitimately have stopped on.
///
/// `lookup_list_entry` already refuses an inactive task, an out-of-range slot
/// and an entry with no descriptor, so a task reaching the type test is one with
/// a real object at that slot.
fn type4_claimant_tasks<M: HostMemory>(state: &DeviceState, host: &M, surface_id: u32) -> Vec<u32> {
    (0..MAX_TASKS as u32)
        .filter(|&task_id| {
            lookup_list_entry(state, host, task_id, surface_id)
                .is_some_and(|e| e.object_type == OBJECT_TYPE_SURFACE)
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
/// type-4 descriptor is fully consumed: its only undecoded span is the three
/// bytes at `0x11`, which read zero on every distinct shape a driven boot
/// produces (`type4_desc_shape … undecoded_nz=0`).
///
/// So the question the wire cannot answer directly is answered by counting
/// instead. A surface id only ever claimed by one task is a surface whose owner
/// probe order cannot have gotten wrong, whatever order it used.
///
/// The claim test is the object-list slot's type alone, not a descriptor read or
/// a translation: a task that lists a type-4 surface at this slot is a task the
/// search could have stopped on. That keeps the sweep to one 12-byte guest read
/// per active task, and it is taken once per surface id — whether a surface id
/// is claimed twice is a property of the guest's allocation, not of this
/// resolve.
///
/// `claims=1` is the healthy reading and stays on the quiet channel. More than
/// one claimant is the case that makes the search's tie-break load-bearing, so
/// that one is a failure line naming the tasks involved.
fn note_type4_claimants<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    surface_id: u32,
    winner: u32,
) {
    if !crate::observe::first_sight("type4_claimants", surface_id as u64) {
        return;
    }
    let claimants = type4_claimant_tasks(state, host, surface_id);
    let line = format!(
        "type4_claimants sid={surface_id} winner={winner} claims={} tasks={claimants:?}",
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

/// The mapping format a type-4 backing latches: single-plane MTL only.
///
/// Multi-plane and unknown-FourCC surfaces get `0`, and that zero is a decoded
/// refusal rather than an absence — stage and paint must not invent BGRA, and
/// type-11 selects planes through `device_desc` instead.
/// [`iosurface_pixel_format_to_mtl`] states the same rule for the conversion.
///
/// Named rather than inlined at [`apply_type4_backing`] because
/// [`backing_matches_latched_geom`] has to compute the *same* value: it compares
/// a freshly-read descriptor against `m.format`, which is whatever this returned
/// last time.
fn latched_mapping_format(surf: &Type4Surface) -> u16 {
    if type4_is_multiplanar(surf) {
        return 0;
    }
    iosurface_pixel_format_to_mtl(surf.pixel_format)
}

/// Whether the geometry already latched on this mapping is still the geometry
/// the freshly-read descriptor declares.
///
/// Both arms of [`resolve_type4_surface_ex`]'s freshness test ask this, and they
/// used to ask it differently: the non-force arm compared width **and** height,
/// the force arm compared width only. They are the same question — "may this
/// resolve return without rebuilding" — and the force arm is the one that cannot
/// afford to be looser, because `force_fresh` returns through
/// [`win_type4_search`] *without* calling [`apply_type4_backing`], so neither
/// `set_mapping_geom` nor `synthesize_device_desc_from_type4` runs. A height
/// change that stays inside the same page count therefore left `m.height` and
/// the whole device descriptor describing the previous incarnation, on the exact
/// path `ensure_surface_for_present` calls to catch a wire geometry change.
///
/// Format is compared too, and neither arm used to. A surface id can be recycled
/// at identical dimensions with a different pixel format, and the format is what
/// every read window's bytes-per-pixel comes from — so keeping the old one
/// samples the new backing at the wrong stride. The comparison goes through
/// [`latched_mapping_format`] rather than the wire FourCC, because `m.format` is
/// whatever that function last returned. Comparing the FourCC would report every
/// surface as changed, and comparing the raw conversion would report every
/// multi-plane surface as changed forever, since the latch deliberately discards
/// it in favour of 0.
fn backing_matches_latched_geom(m: &MappingEntry, surf: &Type4Surface) -> bool {
    m.width == surf.width && m.height == surf.height && m.format == latched_mapping_format(surf)
}

/// The order [`resolve_type4_surface_ex`] probes task object lists in: task 0,
/// then the cached owner hint, then every other task once.
///
/// An iterator rather than a materialised list. The order is unchanged, but
/// building it as a `Vec` allocated 257 elements on every call, and this runs
/// from `ensure_surface_for_present` on every present for every resident
/// mapping — thousands of times a boot to read element 0 and stop.
///
/// A hint of 0, or one outside the task id space, contributes nothing: 0 is
/// already first, and an out-of-range id would be skipped by the liveness test
/// at the probe anyway, so admitting it here would only cost the `!= hint`
/// filter its meaning.
fn type4_probe_order(hint: u32) -> impl Iterator<Item = u32> {
    let hint = if hint != 0 && (hint as usize) < MAX_TASKS {
        hint
    } else {
        0
    };
    std::iter::once(0)
        .chain(Some(hint).filter(|&h| h != 0))
        .chain((1..MAX_TASKS as u32).filter(move |&tid| tid != hint))
}

/// Take `task_id` as the owner of `surface_id` and report the search's exposure.
fn win_type4_search<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
    task_id: u32,
) -> bool {
    record_type4_owner(state, surface_id, task_id);
    note_type4_claimants(state, host, surface_id, task_id);
    true
}

fn resolve_type4_surface_ex<M: HostMemory>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
    force: bool,
) -> bool {
    if surface_id == 0 || surface_id as usize >= MAX_MAPPINGS {
        return false;
    }
    // Task probe order: task 0 first, then the cached owner-task hint (so a hot
    // present-path re-scan short-circuits on the owning task instead of walking
    // all 256 slots), then the remaining tasks.
    //
    // Task 0 leads because the guest says so, not because it is where surfaces
    // have happened to be. A type-5 view carries the owning task at
    // [`TYPE5_OWNER_TASK`] and it is the accelerator's kernel task, whose id is a
    // hardcoded 0 and whose slot the task-id allocator reserves before any client
    // task exists. `note_type5_owner_task` fails loudly if that ever reads
    // otherwise.
    //
    // The remaining 255 probes are not dead weight on that reading. They cost
    // nothing on the path that matters — every successful resolve measured has
    // stopped on the first probe — and they are what makes `type4_claimants` able
    // to say a second task claims the id at all.
    //
    // Built as an iterator rather than a `Vec`. The order is the same one, but
    // materialising it allocated a 257-element vector on every call, and this is
    // called from `ensure_surface_for_present` on every present for every
    // resident mapping — thousands of times a boot to read element 0 and stop.
    let hint = state
        .mappings
        .get(&surface_id)
        .map(|m| m.owner_task_hint)
        .unwrap_or(0);

    for task_id in type4_probe_order(hint) {
        if task_id as usize >= state.tasks.len() {
            continue;
        }
        if !state.tasks[task_id as usize].active {
            continue;
        }
        // Count the guest-read cost of one active-task object-list probe.
        let Some(entry) = lookup_list_entry(state, host, task_id, surface_id) else {
            continue;
        };
        if entry.object_type != OBJECT_TYPE_SURFACE {
            continue;
        }
        let Some(desc) = read_descriptor(state, host, task_id, &entry) else {
            defer_type4_fail(
                surface_id,
                "desc_read",
                format!(
                    "type4_backing_fail reason=desc_read sid={surface_id} task={task_id} desc_gva={:#x} desc_len={}",
                    entry.descriptor_gva, entry.descriptor_length
                ),
            );
            continue;
        };
        let _ = state.insert_object(task_id, surface_id);
        let Some(surf) = decode_type4_surface(&desc) else {
            defer_type4_fail(
                surface_id,
                "desc_decode",
                format!(
                    "type4_backing_fail reason=desc_decode sid={surface_id} task={task_id} desc_len={} backing_pfn={:#x} length={:#x}",
                    desc.len(),
                    desc.get(TYPE4_BACKING_PFN..TYPE4_BACKING_PFN + 4)
                        .map(ld32)
                        .unwrap_or(0),
                    desc.get(TYPE4_LEN..TYPE4_LEN + 8)
                        .map(ld64)
                        .unwrap_or(0)
                ),
            );
            continue;
        };
        // Force path validated the cached pages are still fresh → keep them.
        let mut force_fresh = false;
        // Skip rebuild when pages already match this backing (hot present path).
        if !force {
            let same_geom = state
                .mappings
                .get(&surface_id)
                .map(|m| {
                    m.mapped
                        && !m.page_entries.is_empty()
                        && m.has_geom
                        && backing_matches_latched_geom(m, &surf)
                })
                .unwrap_or(false);
            if same_geom {
                // Same geom + non-empty pages: keep (guest double-buffer
                // may still rewrite page *content* without changing pfn).
                return win_type4_search(state, host, surface_id, task_id);
            }
        } else if let Some(m) = state.mappings.get(&surface_id) {
            // Force: keep the cached table only while the CURRENT task
            // page-table translation of the descriptor's first and last
            // backing pages still matches it. `backing_pfn` is a GPU-VA page;
            // the guest may remap that GVA range onto new physical pages
            // without changing surface id, geometry, or length (early-boot
            // console FB vs the WindowServer reallocation). A same-size guard
            // here kept boot-time pages forever, so presents froze on pages
            // nobody writes.
            if m.mapped && !m.page_entries.is_empty() {
                let page_shift = state.page_shift;
                let page_size = page_size_of(page_shift);
                let need = ((surf.length.saturating_sub(1)) / page_size) + 1;
                if m.page_entries.len() as u64 == need && backing_matches_latched_geom(m, &surf) {
                    let task = state.tasks.get(task_id as usize).filter(|t| t.active);
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
                    let last = m.page_entries.len() - 1;
                    if entry_fresh(0, m.page_entries[0])
                        && entry_fresh(last as u64, m.page_entries[last])
                    {
                        force_fresh = true;
                    } else {
                        crate::observe::fail(format!(
                            "type4_pages_stale sid={surface_id} task={task_id} n={} gpa0={:#x} (task PT translation moved; rebuilding)",
                            m.page_entries.len(),
                            entry_gpa_shift(m.page_entries[0], page_shift).unwrap_or(0)
                        ));
                    }
                }
            }
        }
        if force_fresh {
            return win_type4_search(state, host, surface_id, task_id);
        }
        if apply_type4_backing(state, host, task_id, surface_id, &surf) {
            return win_type4_search(state, host, surface_id, task_id);
        }
    }
    // No task could back it. Only now is a probe's refusal a backing failure.
    flush_type4_fail(surface_id);
    false
}

/// Ensure surface backing for present: type-4 pages when needed, else keep arm
/// MappingInternal path.
///
/// Resolves type-4 once pages are empty; guest double-buffering uses distinct
/// surface_ids (content updates land in-place on an already-mapped pfn).
pub fn ensure_surface_for_present<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &M,
    surface_id: u32,
) -> bool {
    if surface_id == 0 {
        return false;
    }
    let need = state
        .mappings
        .get(&surface_id)
        .map(|m| !m.mapped || m.page_entries.is_empty())
        .unwrap_or(true);
    if need {
        let _ = resolve_type4_surface(state, host, surface_id);
    } else {
        // Opportunistic refresh if wire geom changed (mode switch).
        let _ = resolve_type4_surface_force(state, host, surface_id);
    }
    // Arm/iosfc path: MappingInternal resolve when captured.
    let _ = crate::runtime::mapper::ensure_resolved_for_scanout(state, host, surface_id);
    state
        .mappings
        .get(&surface_id)
        .map(|m| m.mapped && !m.page_entries.is_empty() && m.has_geom)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::contract::endian::{ld32, st16, st32, st64};
    use crate::contract::gva::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
    use crate::contract::iosurface_pages::DEVICE_DESC_PLANE_COUNT;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
    use crate::runtime::host::FakeHost;

    #[test]
    fn type11_fail_latch_dedups_per_task_ref_and_rearms_on_clear() {
        // Flood guard for the per-draw-per-ref resolve path: a genuinely-broken
        // type-11 ref logs each reason once, isolates per (task,ref), and
        // re-arms on resolve. Unique ids so this never races real refs across
        // the process-global latch.
        let (t, r, r2) = (0xAB01u32, 0xCD01u32, 0xCD02u32);
        clear_type11_fail(t, r);
        clear_type11_fail(t, r2);
        let seen = |task: u32, rf: u32, reason: &'static str| {
            type11_fail_latch()
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&(task, rf, reason))
        };
        note_type11_fail(t, r, "type11_register", "x".into());
        assert!(seen(t, r, "type11_register"));
        // Distinct reason on the same ref tracked independently.
        note_type11_fail(t, r, "type11_desc_read", "x".into());
        assert!(seen(t, r, "type11_desc_read"));
        // A different ref is untouched.
        assert!(!seen(t, r2, "type11_register"));
        note_type11_fail(t, r2, "type11_register", "x".into());
        // Clearing r re-arms only r, leaves r2.
        clear_type11_fail(t, r);
        assert!(!seen(t, r, "type11_register"));
        assert!(!seen(t, r, "type11_desc_read"));
        assert!(seen(t, r2, "type11_register"));
        clear_type11_fail(t, r2);
    }

    fn setup_task_with_list(host: &mut FakeHost, state: &mut DeviceState) {
        // Same 1-level map as gva_mem test: GVA page 0 → data pfn 4.
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
        let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        assert!(state.define_task(1, 0x1000, 2));
        // list base GVA 0 (pfn field 0 allowed)
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 11u32 | (0x20u32 << 8));
        entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 12, &entry);
        let mut desc = [0u8; 0x20];
        st32(&mut desc[0..], 9);
        st16(&mut desc[0x16..], 0x50);
        st32(&mut desc[0x18..], 64);
        st32(&mut desc[0x1c..], 32);
        let _ = host.write_gpa(data_gpa + 0x40, &desc);
    }

    #[test]
    fn resolve_type11_from_list() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_with_list(&mut host, &mut state);
        // Sanity: list entry readable
        let e = lookup_list_entry(&state, &host, 1, 1).expect("list entry");
        assert_eq!(e.object_type, 11);
        assert_eq!(e.descriptor_gva, 0x40);
        let mid = resolve_type11_ref(&mut state, &host, 1, 1).expect("type11");
        assert_eq!(mid, 9);
        let m = state.mappings.get(&9).unwrap();
        assert!(m.has_geom);
        assert_eq!((m.width, m.height, m.format), (64, 32, 0x50));
    }

    /// The type-4 decoder says so when it drops what the guest declared.
    ///
    /// All three of these bounds are correct — IOSurface caps `getPlaneCount`
    /// at eight, a plane record the blob does not reach cannot be decoded, and
    /// the device descriptor's `allocSize` really is 32 bits. What was wrong is
    /// that each one applied in silence, so a surface whose ninth plane this
    /// device will never look at, or whose size it cannot express, reached every
    /// later reader as a surface that simply had eight planes and that size.
    /// Never Fail Silently: a bound the guest crossed is a bound worth naming.
    #[test]
    fn the_type4_decoder_reports_what_it_drops() {
        // `desc` reaches only plane 0's record, so planes 1..=7 are declared
        // and unreachable, and plane 8+ is over the cap.
        let mut desc = vec![0u8; 0x24];
        st64(&mut desc[TYPE4_LEN..], 0x1000);
        st32(&mut desc[TYPE4_BACKING_PFN..], 0x100);
        st32(&mut desc[TYPE4_PIXEL_FORMAT..], 0x4247_5241); // 'BGRA'
        desc[TYPE4_PLANE_COUNT] = 12;

        reset_type4_decode_drops();
        let cap = crate::observe::FailCapture::start();
        let s = decode_type4_surface(&desc).expect("type4 decodes");
        assert_eq!(s.plane_count, TYPE4_PLANE_CAP as u8, "still clamped");
        // Two distinct drops on this descriptor, so select by reason rather
        // than by slug: the surplus planes over the cap, and — separately —
        // the declared planes whose records the blob does not reach.
        let over = cap
            .lines()
            .into_iter()
            .find(|l| l.contains("reason=plane_count_over_cap"))
            .expect("an over-cap plane count must be reported");
        assert!(
            over.contains("declared=12") && over.contains("cap=8"),
            "the line must name what the guest asked for and what it got: {over}"
        );

        // Same reason twice is one line — the latch is what keeps a per-surface
        // stream from flooding the always-on channel.
        let cap2 = crate::observe::FailCapture::start();
        let _ = decode_type4_surface(&desc);
        assert!(
            cap2.lines()
                .iter()
                .all(|l| !l.contains("reason=plane_count_over_cap")),
            "a repeat must not spend a second line: {:?}",
            cap2.lines()
        );

        // A declared plane whose record the blob does not reach.
        reset_type4_decode_drops();
        let cap3 = crate::observe::FailCapture::start();
        let _ = decode_type4_surface(&desc);
        let short = cap3
            .lines()
            .into_iter()
            .find(|l| l.contains("reason=plane_record_short"))
            .expect("an unreachable plane record must be reported");
        assert!(short.contains("plane=1"), "{short}");

        // A surface larger than the 32-bit `allocSize` field can express.
        reset_type4_decode_drops();
        let mut big = vec![0u8; 0x30];
        st64(&mut big[TYPE4_LEN..], (u32::MAX as u64) + 1);
        st32(&mut big[TYPE4_BACKING_PFN..], 0x100);
        st32(&mut big[TYPE4_PIXEL_FORMAT..], 0x4247_5241);
        big[TYPE4_PLANE_COUNT] = 1;
        st32(&mut big[TYPE4_PLANES + 4..], 64);
        st32(&mut big[TYPE4_PLANES + 8..], 32);
        st32(&mut big[TYPE4_PLANES + 12..], 256);
        let surf = decode_type4_surface(&big).expect("type4 decodes");
        let cap4 = crate::observe::FailCapture::start();
        let _ = synthesize_device_desc_from_type4(&surf);
        let sat = cap4
            .lines()
            .into_iter()
            .find(|l| l.contains("reason=alloc_size_over_u32"))
            .expect("a length the 32-bit allocSize cannot hold must be reported");
        assert!(sat.contains("length=4294967296"), "{sat}");
    }

    #[test]
    fn decode_type4_plane0() {
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x100); // backing pfn
        st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
        desc[0x10] = 1;
        st32(&mut desc[0x14..], 0); // plane offset
        st32(&mut desc[0x18..], 64);
        st32(&mut desc[0x1c..], 32);
        st32(&mut desc[0x20..], 256); // bpr
        let s = decode_type4_surface(&desc).expect("type4");
        assert_eq!(s.length, 0x1000);
        assert_eq!(s.backing_pfn, 0x100);
        assert_eq!((s.width, s.height, s.bytes_per_row), (64, 32, 256));
        assert_eq!(s.plane_count, 1);
        assert_eq!(s.planes[0].offset, 0);
        assert!(!type4_is_multiplanar(&s));
        assert_eq!(
            iosurface_pixel_format_to_mtl(s.pixel_format),
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        );
    }

    #[test]
    fn fourcc_420f_not_bgra_and_multiplanar() {
        assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
        assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420V), 0);
        assert!(iosurface_fourcc_is_biplanar(IOSURFACE_FOURCC_420F));
        // Unknown FourCC must not invent BGRA.
        assert_eq!(iosurface_pixel_format_to_mtl(0xdead_beef), 0);
    }

    /// A small value is not an MTLPixelFormat ordinal in disguise.
    ///
    /// The converter used to return `pixel_format as u16` for anything at or
    /// below 0x200, deciding which encoding the field was in from how big the
    /// number was. Every caller passes a type-4 `pixelFormat` (+0x0c), which is
    /// an IOSurface OSType and therefore never below `'    '` (0x20202020), so
    /// a small value arriving here is a bad read — and passing it through
    /// published a format the guest never named. Fail closed instead, which is
    /// what this function already does for every FourCC it does not know.
    #[test]
    fn a_small_value_is_not_read_as_an_mtl_ordinal() {
        // 0x50 is MTLPixelFormatBGRA8Unorm. As a type-4 OSType it is nonsense,
        // and the old magnitude test would have handed it back as a format.
        assert_eq!(iosurface_pixel_format_to_mtl(0x50), 0);
        assert_eq!(iosurface_pixel_format_to_mtl(0x200), 0);
        // Known FourCCs are unaffected — this is the boundary the old test sat
        // on, not a narrowing of what the converter accepts.
        assert_eq!(
            iosurface_pixel_format_to_mtl(0x4247_5241),
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
        );
    }

    #[test]
    fn decode_type4_biplanar_420f_planes() {
        // Wire: plane0 Y 1024×1024 bpr=1024 bpe=1; plane1 UV 512×512 bpr=1024 bpe=2.
        // Live boot: fmt='420f' len=0x180000 plane0 bpr=1024.
        let mut desc = vec![0u8; 0x14 + 2 * 0x10];
        st64(&mut desc[0..], 0x180000);
        st32(&mut desc[8..], 0x200);
        st32(&mut desc[0xc..], IOSURFACE_FOURCC_420F);
        desc[0x10] = 2;
        // plane0
        st32(&mut desc[0x14..], 0); // offset
        st32(&mut desc[0x18..], 1024);
        st32(&mut desc[0x1c..], 1024);
        st32(&mut desc[0x20..], 1024 | (1 << 24)); // bpr | bpe<<24
                                                   // plane1
        st32(&mut desc[0x24..], 1024 * 1024); // offset after Y
        st32(&mut desc[0x28..], 512);
        st32(&mut desc[0x2c..], 512);
        st32(&mut desc[0x30..], 1024 | (2 << 24));
        let s = decode_type4_surface(&desc).expect("type4 420f");
        assert!(type4_is_multiplanar(&s));
        assert_eq!(s.plane_count, 2);
        assert_eq!(
            (
                s.planes[0].width,
                s.planes[0].height,
                s.planes[0].bytes_per_row
            ),
            (1024, 1024, 1024)
        );
        assert_eq!(s.planes[0].bytes_per_element, 1);
        assert_eq!(
            (
                s.planes[1].width,
                s.planes[1].height,
                s.planes[1].bytes_per_element
            ),
            (512, 512, 2)
        );
        let dev = synthesize_device_desc_from_type4(&s);
        assert_eq!(dev[DEVICE_DESC_PLANE_COUNT], 2);
        use crate::contract::iosurface_pages::{
            decode_device_surface, sample_window_prefer_device, DEVICE_DESC_PIXEL_FORMAT,
        };
        assert_eq!(
            ld32(&dev[DEVICE_DESC_PIXEL_FORMAT..]),
            IOSURFACE_FOURCC_420F
        );
        let surf = decode_device_surface(&dev).expect("device");
        assert_eq!(surf.plane_count, 2);
        assert_eq!(surf.alloc_size, 0x180000);
        // Type-11 Y plane: R8 1024×1024 matches plane0 (contract geometry key).
        let y = sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_R8_UNORM,
            1024,
            1024,
        )
        .expect("Y window");
        assert_eq!(y.0, 0); // offset
        assert_eq!(y.1, 1024); // bpr
        assert!(y.3); // from device
                      // UV plane: RG8 half res.
        let uv = sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_RG8_UNORM,
            512,
            512,
        )
        .expect("UV window");
        assert_eq!(uv.0, 1024 * 1024);
        assert_eq!(uv.1, 1024);
        // BGRA invent of full 1024² must still reject (alloc < invent span).
        assert!(sample_window_prefer_device(
            Some(&dev),
            None,
            crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM,
            1024,
            1024,
        )
        .is_none());
    }

    /// A failed page-table walk is not an address. The device used to answer it
    /// with the backing *virtual* address used as a physical one whenever that
    /// number happened to be RAM, which put a fabricated PFN into
    /// `page_entries` — the list every later reader and writer resolves
    /// through. Here the walk cannot resolve the backing GVA and the identity
    /// candidate *is* mapped RAM, so the old path would have accepted it.
    #[test]
    fn resolve_type4_refuses_to_substitute_the_gva_when_the_walk_fails() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // The identity candidate is backed RAM: `read_gpa` succeeds on it, which
        // is the whole of what the old gate checked.
        host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x2000, 0x5a);
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // root[0] carries the object list and descriptors. root[0x20] — the
        // backing GVA page — is left unmapped, so the backing walk fails.
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x20); // backing GVA page — unmapped in this task
        st32(&mut desc[0xc..], 0x50);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(
            !resolve_type4_surface(&mut state, &host, 3),
            "an untranslatable backing must not resolve"
        );
        // The refusal happens before any mutation, so no fabricated entry is
        // left behind for a later writer to aim at.
        let fabricated = state
            .mappings
            .get(&3)
            .map(|m| m.mapped || !m.page_entries.is_empty())
            .unwrap_or(false);
        assert!(!fabricated, "refusal must not cache a fabricated backing");
    }

    /// `resolve_type4_surface_ex` probes task 0 first and returns on the first
    /// task whose backing applies. The identity guess made task 0 succeed for
    /// surfaces it could not translate, so the search stopped there and the
    /// owning task was never tried — the surface was then backed by an address
    /// derived from a virtual one. Refusing is what lets the loop continue.
    ///
    /// Both tasks list the surface, as task 0 (the kernel/global list) and the
    /// owner do in production; only the owner can translate the backing.
    #[test]
    fn the_task_search_reaches_the_owner_when_task_zero_cannot_translate() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
        let root0_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
        let root1_gpa = 8u64 << PAGE_SHIFT_X86;
        let real_page = 9u64 << PAGE_SHIFT_X86;
        for (gpa, len) in [
            (dir0_gpa, 0x20),
            (root0_gpa, 0x1000),
            (data_gpa, 0x200),
            (dir1_gpa, 0x20),
            (root1_gpa, 0x1000),
            (real_page, 0x1000),
        ] {
            host.map_range(gpa, len, 0);
        }
        // The identity candidate for the backing GVA is RAM, so the old path
        // would have taken it on task 0 rather than moving on.
        host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x1000, 0x5a);

        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir0_gpa, &d);
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
        let _ = host.write_gpa(dir1_gpa, &d);
        // Both roots reach the object list at GVA 0; only task 1's maps the
        // backing GVA page 0x20, and it maps it to `real_page`.
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root0_gpa, &d[..4]);
        let _ = host.write_gpa(root1_gpa, &d[..4]);
        st32(&mut d[..4], 9);
        let _ = host.write_gpa(root1_gpa + 0x20 * 4, &d[..4]);

        assert!(state.define_task(0, 0x1000, 2));
        assert!(state.set_object_list(0, 0, 8));
        assert!(state.define_task(1, 0x1000, 7));
        assert!(state.set_object_list(1, 0, 8));

        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 0x20);
        st32(&mut desc[0xc..], 0x50);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(
            resolve_type4_surface(&mut state, &host, 3),
            "the owning task can translate the backing, so the resolve must succeed"
        );
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.page_entries.len(), 1);
        assert_eq!(
            entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
            Some(real_page),
            "the backing must come from the task that could translate it, \
             not from task 0's untranslatable GVA"
        );
    }

    /// The search stops on the first task that can back a surface, so whether
    /// that choice was ever a choice is the thing to count. Nothing on the wire
    /// can verify a candidate — the object-list entry carries no identity and
    /// the type-4 descriptor is fully decoded — so the claimant count is the
    /// only available reading of the search's exposure, and it has to
    /// distinguish "one task lists this id" from "two do".
    #[test]
    fn a_surface_id_claimed_by_two_tasks_is_counted_as_two() {
        // Two tasks, each with its own directory and root, both listing eight
        // object slots at GVA 0. Task 0's list page holds a type-4 surface at
        // slot 3; task 1's holds a type-5 there until the second half of the
        // test rewrites it.
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
        let root0_gpa = 3u64 << PAGE_SHIFT_X86;
        let list0_gpa = 4u64 << PAGE_SHIFT_X86;
        let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
        let root1_gpa = 8u64 << PAGE_SHIFT_X86;
        let list1_gpa = 9u64 << PAGE_SHIFT_X86;
        for (gpa, len) in [
            (dir0_gpa, 0x20),
            (root0_gpa, 0x1000),
            (list0_gpa, 0x200),
            (dir1_gpa, 0x20),
            (root1_gpa, 0x1000),
            (list1_gpa, 0x200),
        ] {
            host.map_range(gpa, len, 0);
        }

        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir0_gpa, &d);
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
        let _ = host.write_gpa(dir1_gpa, &d);
        // Each task's GVA page 0 reaches its own list page.
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root0_gpa, &d[..4]);
        st32(&mut d[..4], 9);
        let _ = host.write_gpa(root1_gpa, &d[..4]);

        assert!(state.define_task(0, 0x1000, 2));
        assert!(state.set_object_list(0, 0, 8));
        assert!(state.define_task(1, 0x1000, 7));
        assert!(state.set_object_list(1, 0, 8));

        // Slot 3 of task 0 is the surface. Both entries carry a descriptor GVA
        // and length, which is what `lookup_list_entry` requires before the type
        // is even looked at.
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], OBJECT_TYPE_SURFACE as u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(list0_gpa + 3 * 12, &entry);

        // Task 1 lists a *different object type* at the same slot, so it is not
        // a claimant even though the slot is populated.
        let mut other = [0u8; 12];
        st32(
            &mut other[0..],
            OBJECT_TYPE_REF_TEXTURE as u32 | (0x30u32 << 8),
        );
        other[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(list1_gpa + 3 * 12, &other);

        assert_eq!(
            type4_claimant_tasks(&state, &host, 3),
            vec![0],
            "a populated slot of another object type is not a claim on this id"
        );

        // Now task 1 lists a type-4 surface at the same slot. The id spaces are
        // per task, so this is a second, unrelated surface wearing the same id —
        // and the search would have to break the tie by probe order alone.
        let _ = host.write_gpa(list1_gpa + 3 * 12, &entry);
        assert_eq!(
            type4_claimant_tasks(&state, &host, 3),
            vec![0, 1],
            "both tasks list a type-4 surface at slot 3, so both are claimants"
        );

        // An inactive task cannot be the one the search stops on, so it is not
        // counted either.
        state.tasks[1].active = false;
        assert_eq!(
            type4_claimant_tasks(&state, &host, 3),
            vec![0],
            "an inactive task is not a claimant"
        );
    }

    /// Force-resolve must rebuild the cached page table when the task PT
    /// translation of the backing GVA moved (same surface id, same geometry,
    /// new physical pages — the early-boot FB vs WindowServer reallocation).
    #[test]
    fn resolve_type4_force_rebuilds_when_task_translation_moves() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        let old_page = 5u64 << PAGE_SHIFT_X86;
        let new_page = 6u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x200, 0);
        host.map_range(old_page, 0x1000, 0x11);
        host.map_range(new_page, 0x1000, 0x22);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        // root[0] = data page (object list + descriptors), root[1] = old backing.
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        st32(&mut d[..4], 5);
        let _ = host.write_gpa(root_gpa + 4, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, 8));
        // Type-4 entry at surface_id=3, descriptor at GVA 0x80.
        let mut entry = [0u8; 12];
        st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
        entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
        let mut desc = vec![0u8; 0x30];
        st64(&mut desc[0..], 0x1000);
        st32(&mut desc[8..], 1); // backing_pfn = GVA page 1
        st32(&mut desc[0xc..], 0x50);
        desc[0x10] = 1;
        st32(&mut desc[0x18..], 16);
        st32(&mut desc[0x1c..], 16);
        st32(&mut desc[0x20..], 64);
        let _ = host.write_gpa(data_gpa + 0x80, &desc);

        assert!(resolve_type4_surface(&mut state, &host, 3));
        {
            let m = state.mappings.get(&3).unwrap();
            assert_eq!(m.page_entries.len(), 1);
            assert_eq!(
                entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
                Some(old_page)
            );
            assert_eq!(m.map_generation, 1);
        }
        // Guest remaps GVA page 1 onto a new physical page (same id/geometry).
        st32(&mut d[..4], 6);
        let _ = host.write_gpa(root_gpa + 4, &d[..4]);
        assert!(resolve_type4_surface_force(&mut state, &host, 3));
        {
            let m = state.mappings.get(&3).unwrap();
            assert_eq!(
                entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
                Some(new_page),
                "force-resolve must follow the moved translation"
            );
            assert_eq!(m.map_generation, 2, "page move bumps map_generation");
        }
        // Unchanged translation: force keeps the table without a rebuild.
        assert!(resolve_type4_surface_force(&mut state, &host, 3));
        let m = state.mappings.get(&3).unwrap();
        assert_eq!(m.map_generation, 2);
        assert_eq!(
            entry_gpa_shift(m.page_entries[0], PAGE_SHIFT_X86),
            Some(new_page)
        );
    }

    /// A genuine backing failure (a surface whose descriptor decoded fine but
    /// whose page-backing construction fails) must be fail-visible with a
    /// `reason=` slug, deduped per `(surface_id, reason)`, and re-armed when the
    /// surface next backs cleanly — never a silent `return false` that paints
    /// stale/black with no log. Locks the type-4 backing blind-spot closure.
    #[test]
    fn apply_type4_backing_fail_latches_reason_and_rearms() {
        let host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        state.page_shift = PAGE_SHIFT_X86;
        // A surface_id other type-4 tests do not touch (they use 3).
        let sid = 11u32;
        clear_type4_fail(sid);
        assert!(!type4_fail_latch()
            .lock()
            .unwrap()
            .contains(&(sid, "task_inactive")));
        // Small valid length (page_count = 1) so the alloc-guard passes, then an
        // undefined/inactive task_id hits the `task_inactive` site — the drain
        // race where a decoded surface's owning task died before backing landed.
        let surf = Type4Surface {
            length: 0x1000,
            backing_pfn: 0x20,
            pixel_format: 0,
            plane_count: 1,
            planes: [Type4Plane::default(); TYPE4_PLANE_CAP],
            width: 16,
            height: 16,
            bytes_per_row: 64,
        };
        assert!(!apply_type4_backing(&mut state, &host, 5, sid, &surf));
        assert!(
            !type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "task_inactive")),
            "one task's probe is not a backing failure: the search has other \
             tasks to try, and reporting here is what put `reason=translate` \
             lines under surfaces that then backed cleanly"
        );
        // The search running out of tasks is what turns the probe's reason into
        // a reported failure.
        flush_type4_fail(sid);
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "task_inactive")),
            "an exhausted search must report the first probe's reason slug"
        );
        // A clean backing on the same surface re-arms the latch.
        clear_type4_fail(sid);
        assert!(
            !type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "task_inactive")),
            "clear_type4_fail must re-arm so a later failure logs again"
        );
    }

    /// A refused walk must say **which** of the walk's checks refused.
    ///
    /// The walk distinguishes fifteen refusals and this rail reported one word,
    /// `translate`, for all of them — so "the guest has not filled in this leaf
    /// PTE yet" and "this device could not read the task root at all" produced
    /// identical log lines while wanting opposite responses. Both halves are
    /// locked here: the walk names its failing check, and the detail line
    /// carries that name verbatim.
    ///
    /// The fixture maps GVA page 0 and nothing else, so the *same* task walks
    /// clean for one address and refuses for the next. Asserting the clean case
    /// too is what keeps this from passing vacuously: a fixture in which every
    /// walk fails would satisfy the refusal assertions on its own.
    #[test]
    fn a_refused_type4_walk_names_the_check_that_refused() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        setup_task_with_list(&mut host, &mut state);
        let task = state.tasks.get(1).expect("fixture defines task 1");

        // Control: the address the fixture does map walks all the way down.
        let mapped = gva_mem::diagnose_task_slot(&host, task, 1, 0, PAGE_SHIFT_ARM64E);
        assert!(
            mapped.contains("st=ok"),
            "fixture must be able to translate, got {mapped:?}"
        );

        // The case the rig produces: a backing whose leaf entry the guest has
        // not written. Page 1 shares the fixture's root and has no PTE.
        let gva = 1u64 << PAGE_SHIFT_ARM64E;
        let walk = gva_mem::diagnose_task_slot(&host, task, 1, gva, PAGE_SHIFT_ARM64E);
        assert!(
            walk.contains("st=zero-pfn"),
            "an unwritten leaf must be reported as zero-pfn, got {walk:?}"
        );
        assert!(
            walk.contains("lvl=") && walk.contains("idx="),
            "the refusal must name where in the walk it stopped, got {walk:?}"
        );

        let line = type4_translate_fail_detail(202, 1, 0, 640, gva, &walk);
        assert!(line.contains("reason=translate"), "{line}");
        assert!(line.contains("sid=202"), "{line}");
        assert!(line.contains("page=0/640"), "{line}");
        assert!(
            line.contains(&format!("walk=[{walk}]")),
            "the refusal must carry the walk diagnosis verbatim, got {line}"
        );
    }

    /// A task the guest has defined but never given an object list to must
    /// resolve **nothing** — not another task's list.
    ///
    /// This reproduces, at unit scale, what the rail was measured doing on every
    /// boot. `TaskEntry::define` used to invent `object_list_pfn = 1` and
    /// `count = 0x100000`, so a task with no `SetObjectList` still computed an
    /// entry address of `0x1000 + off`. Nothing is mapped there for that task,
    /// the walk failed `gva_zero_pfn`, and `read_task_gva_by_id` then walked
    /// task `5 >> 1 == 2`'s page table at the same address — where task 2's
    /// object list genuinely lives — and decoded task 2's entry as task 5's.
    ///
    /// Task 2's own lookup is asserted first so the fixture is known to be real:
    /// a test where the donor list is unreadable would pass for the wrong reason.
    #[test]
    fn a_task_with_no_object_list_resolves_nothing_not_its_neighbours_list() {
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
        let _ = host.write_gpa(dir_gpa, &d);
        // PTE for GVA page 1 (0x1000) → pfn 4, so task 2's list is readable.
        let mut pte = [0u8; 4];
        st32(&mut pte, 4);
        let _ = host.write_gpa(root_gpa + 4, &pte);

        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (0x40u32 << 8),
        );
        entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
        let _ = host.write_gpa(data_gpa, &entry);

        // Task 2 owns a real list at pfn 1. Task 5 has a directory that maps
        // nothing, and `5 >> 1 == 2`.
        assert!(state.define_task(2, 0x1000, 2));
        assert!(state.set_object_list(2, 1, 4));
        assert!(state.define_task(5, 0x1000, 9));

        let donor = lookup_list_entry(&state, &host, 2, 0);
        assert!(
            donor.is_some(),
            "fixture is not real: task 2's own list must be readable"
        );

        // The behavioural claim first, so a regression fails on the corruption
        // itself rather than on the field that causes it.
        assert_eq!(
            lookup_list_entry(&state, &host, 5, 0),
            None,
            "task 5 has no object list, so it must resolve nothing — returning \
             Some here is task 2's entry answering for task 5"
        );
        assert_eq!(
            state.tasks[5].object_list_pfn, 0,
            "a defined task has no list until SetObjectList says so"
        );
        assert_eq!(state.tasks[5].object_list_count, 0);
    }

    fn setup_type4_candidate(
        host: &mut FakeHost,
        state: &mut DeviceState,
        surface_id: u32,
        desc_gva: u64,
        desc_len: u32,
    ) -> u64 {
        let dir_gpa = 2u64 << PAGE_SHIFT_X86;
        let root_gpa = 3u64 << PAGE_SHIFT_X86;
        let data_gpa = 4u64 << PAGE_SHIFT_X86;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x1000, 0);
        host.map_range(data_gpa, 0x1000, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);
        assert!(state.define_task(1, 0x1000, 2));
        assert!(state.set_object_list(1, 0, surface_id + 1));

        let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
        st32(
            &mut entry[0..],
            (OBJECT_TYPE_SURFACE as u32) | (desc_len << 8),
        );
        entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
        let entry_gpa = data_gpa + surface_id as u64 * OBJECT_LIST_ENTRY_LEN as u64;
        let _ = host.write_gpa(entry_gpa, &entry);
        data_gpa
    }

    /// Once task-scan lookup finds an actual type-4 candidate, descriptor read
    /// failure is no longer speculative: the surface has an owner but cannot get
    /// backing. It must be fail-visible with a stable reason slug.
    #[test]
    fn resolve_type4_candidate_logs_descriptor_read_failure() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let sid = 17u32;
        clear_type4_fail(sid);
        let _ = setup_type4_candidate(&mut host, &mut state, sid, 0x3000, 0x30);

        assert!(!resolve_type4_surface(&mut state, &host, sid));
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "desc_read")),
            "surface-type candidate with unreadable descriptor must name desc_read"
        );
        clear_type4_fail(sid);
    }

    /// A readable but invalid type-4 descriptor used to fall through to the
    /// resolver tail with no site reason. Keep it fail-visible without logging
    /// absent/non-surface speculative probes.
    #[test]
    fn resolve_type4_candidate_logs_descriptor_decode_failure() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_X86);
        let sid = 18u32;
        clear_type4_fail(sid);
        let data_gpa = setup_type4_candidate(&mut host, &mut state, sid, 0x80, 0x30);
        let bad_desc = vec![0u8; 0x30];
        let _ = host.write_gpa(data_gpa + 0x80, &bad_desc);

        assert!(!resolve_type4_surface(&mut state, &host, sid));
        assert!(
            type4_fail_latch()
                .lock()
                .unwrap()
                .contains(&(sid, "desc_decode")),
            "surface-type candidate with invalid descriptor must name desc_decode"
        );
        clear_type4_fail(sid);
    }

    /// Live wire bytes (boot 093019 `compute_stage_tex type5 … args_hex`):
    /// R8 1024×1024 = Y plane view of a biplanar 1024×1024 surface.
    #[test]
    fn decode_type5_texture_view_live_r8_y_plane() {
        let mut desc = vec![0u8; 8];
        st32(&mut desc[TYPE5_SURFACE_ID..], 8);
        // args blob: kind 0x2f, len 0x30, own_ref 0x15, record R8 1024×1024 d=1.
        let args = [
            0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x15, 0, 0, 0, // kind, blob_len, own_ref
            0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
            0x00, 0x04, 0x00, 0x00, // width 1024
            0x00, 0x04, 0x00, 0x00, // height 1024
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer (unconsumed)
        ];
        desc.extend_from_slice(&args);
        let rec = decode_type5_texture_view(&desc).expect("live R8 record decodes");
        assert_eq!(rec.pixel_format, 0x0a);
        assert_eq!((rec.width, rec.height, rec.depth), (1024, 1024, 1));
        // Short record (no +0x20 field) defaults to plane 0.
        assert_eq!(rec.plane_index, 0);
    }

    /// Live 56-byte wire blob from the BLIT copy-source path (x86 Ventura
    /// 13.7.8, 2026-07-19 `blit t5_view_decode sid=34`): a full-color
    /// texture view (BGRA8_sRGB 1024×768 window backing) carries the sibling
    /// record tag `0x62`, not the biplanar `0x42`. Same field layout — must
    /// decode, or the blit path drops the copy.
    #[test]
    fn decode_type5_texture_view_live_0x62_color_window_view() {
        // Exact leading 40 bytes observed, zero-padded to the 56-byte desc_len.
        let head: [u8; 40] = [
            0x22, 0x00, 0x00, 0x00, // surface_id = 34
            0x00, 0x00, 0x00, 0x00, // field
            0x2f, 0x00, 0x00, 0x00, // kind 0x2f
            0x30, 0x00, 0x00, 0x00, // blob_len 0x30
            0x0b, 0x00, 0x00, 0x00, // own_ref 0x0b
            0x62, 0x00, 0x51, 0x00, // tag=0x62, unk, fmt=0x51 BGRA8_sRGB
            0x00, 0x04, 0x00, 0x00, // width 1024
            0x00, 0x03, 0x00, 0x00, // height 768
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, // trailer
        ];
        let mut desc = head.to_vec();
        desc.resize(56, 0); // plane field (+0x20 in record) reads 0
        let rec = decode_type5_texture_view(&desc).expect("0x62 color view must decode");
        assert_eq!(rec.pixel_format, 0x51);
        assert_eq!((rec.width, rec.height, rec.depth), (1024, 768, 1));
        assert_eq!(rec.plane_index, 0);
    }

    /// Live 56-byte wire blob (boot 20260717-063043, v0a8 hero): the record
    /// carries the `newTextureWithDescriptor:iosurface:plane:` plane at
    /// `+0x20` — Y views carry 0, the RG8 chroma view 1, the same-geometry
    /// alpha view 2. Geometry cannot separate Y from alpha; this field does.
    #[test]
    fn decode_type5_texture_view_live_v0a8_alpha_plane_index() {
        let mut desc = vec![0u8; 8];
        st32(&mut desc[TYPE5_SURFACE_ID..], 0x6d);
        let args = [
            0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x82, 0x01, 0, 0, // kind, blob_len, own_ref
            0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
            0xb2, 0x03, 0x00, 0x00, // width 946
            0x5e, 0x01, 0x00, 0x00, // height 350
            0x01, 0x00, 0x00, 0x00, // depth 1
            0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
            0x02, 0x00, 0x00, 0x00, // IOSurface plane index = 2 (alpha)
        ];
        desc.extend_from_slice(&args);
        let rec = decode_type5_texture_view(&desc).expect("live v0a8 alpha record decodes");
        assert_eq!(rec.pixel_format, 0x0a);
        assert_eq!((rec.width, rec.height, rec.depth), (946, 350, 1));
        assert_eq!(rec.plane_index, 2);
    }

    /// The owner-task census must read the dword the guest wrote, and must be
    /// able to tell 0 from anything else.
    ///
    /// A census whose extraction is wrong reports 0 forever whatever the wire
    /// says, and 0 is the answer this device already assumes — so the failing
    /// case would be indistinguishable from the healthy one, which is the whole
    /// point of having it. Pinning the offset against a descriptor whose *other*
    /// leading dword is non-zero is what makes an off-by-four visible.
    #[test]
    fn the_type5_owner_task_is_read_from_its_own_dword() {
        let mut desc = [0u8; TYPE5_MIN_LEN];
        st32(&mut desc[TYPE5_SURFACE_ID..], 0xabcd);
        assert_eq!(
            ld32(&desc[TYPE5_OWNER_TASK..]),
            0,
            "the surface id must not be read as the owner task"
        );
        st32(&mut desc[TYPE5_OWNER_TASK..], 7);
        assert_eq!(ld32(&desc[TYPE5_OWNER_TASK..]), 7);
        assert_eq!(
            ld32(&desc[TYPE5_SURFACE_ID..]),
            0xabcd,
            "writing the owner task must not disturb the surface id"
        );
        // Both fields sit inside the minimum descriptor — the array above is
        // exactly `TYPE5_MIN_LEN` and indexing it proves that — so the census can
        // never be silently skipped on a well-formed record.
        assert_eq!(TYPE5_OWNER_TASK, TYPE5_SURFACE_ID + 4);
    }

    #[test]
    fn decode_type5_texture_view_fail_closed() {
        // Short descriptor (no record).
        let mut short = vec![0u8; 8];
        st32(&mut short[TYPE5_SURFACE_ID..], 8);
        assert!(decode_type5_texture_view(&short).is_none());
        // Wrong record tag.
        let mut bad_tag = vec![0u8; 8];
        st32(&mut bad_tag[TYPE5_SURFACE_ID..], 8);
        bad_tag.extend_from_slice(&[0u8; 12]);
        bad_tag.extend_from_slice(&[
            0x41, 0x01, 0x0a, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x01, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&bad_tag).is_none());
        // Non-2D (depth != 1) fails closed.
        let mut vol = vec![0u8; 8];
        st32(&mut vol[TYPE5_SURFACE_ID..], 8);
        vol.extend_from_slice(&[0u8; 12]);
        vol.extend_from_slice(&[
            0x42, 0x07, 0x50, 0x00, 0x40, 0, 0, 0, 0x40, 0, 0, 0, 0x40, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&vol).is_none());
        // Zero width fails closed.
        let mut zw = vec![0u8; 8];
        st32(&mut zw[TYPE5_SURFACE_ID..], 8);
        zw.extend_from_slice(&[0u8; 12]);
        zw.extend_from_slice(&[
            0x42, 0x01, 0x0a, 0x00, 0, 0, 0, 0, 0x00, 0x04, 0, 0, 0x01, 0, 0, 0,
        ]);
        assert!(decode_type5_texture_view(&zw).is_none());
    }

    /// The probe's notion of "undecoded" must be exactly the bytes
    /// `decode_type4_surface` skips, and it must distinguish two surfaces on
    /// those bytes alone.
    ///
    /// This is the measurement that blocks the largest deletion in the present
    /// path: nothing decoded at surface-create time separates a desktop
    /// swapchain buffer from a same-geometry offscreen tile, so membership is
    /// reconstructed by half a dozen downstream mechanisms. If the guest is
    /// telling us in the undecoded span, the probe has to be able to see it.
    /// The two arms of the type-4 freshness test must accept exactly the same
    /// backings, because only one of them rebuilds when it says no.
    ///
    /// The force arm returns through `win_type4_search` **without** calling
    /// `apply_type4_backing`, so `set_mapping_geom` and
    /// `synthesize_device_desc_from_type4` are both skipped. It used to compare
    /// width alone while the non-force arm compared width and height, and
    /// `ensure_surface_for_present` calls the force arm precisely to catch a
    /// wire geometry change — so a height change that stayed inside the same
    /// page count left the mapping describing the previous incarnation, on the
    /// path whose job was to notice.
    ///
    /// Neither arm compared format, and a surface id recycled at identical
    /// dimensions with a different pixel format keeps the old bytes-per-pixel
    /// for every read window built over it.
    #[test]
    fn a_latched_backing_is_stale_when_any_of_geometry_or_format_moved() {
        use crate::contract::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
        let surf = |w: u32, h: u32, fourcc: u32| Type4Surface {
            length: 0x1000,
            backing_pfn: 1,
            pixel_format: fourcc,
            plane_count: 1,
            planes: Default::default(),
            width: w,
            height: h,
            bytes_per_row: w * 4,
        };
        // 'BGRA' and 'RGBA' are distinct single-plane FourCCs at one bpp, so a
        // swap between them is invisible to a dimensions-only test.
        const BGRA: u32 = 0x4247_5241;
        const RGBA: u32 = 0x5247_4241;
        assert_eq!(
            latched_mapping_format(&surf(8, 4, BGRA)),
            MTL_FORMAT_BGRA8_UNORM
        );
        assert_eq!(
            latched_mapping_format(&surf(8, 4, RGBA)),
            MTL_FORMAT_RGBA8_UNORM
        );

        let m = MappingEntry {
            width: 8,
            height: 4,
            format: MTL_FORMAT_BGRA8_UNORM,
            ..Default::default()
        };
        assert!(backing_matches_latched_geom(&m, &surf(8, 4, BGRA)));
        assert!(
            !backing_matches_latched_geom(&m, &surf(8, 5, BGRA)),
            "a height change must be stale on both arms"
        );
        assert!(!backing_matches_latched_geom(&m, &surf(9, 4, BGRA)));
        assert!(
            !backing_matches_latched_geom(&m, &surf(8, 4, RGBA)),
            "same dimensions, different format: every read window's bpp comes from it"
        );
    }

    /// A multi-plane backing must compare equal to itself.
    ///
    /// The latch stores `0` for it — the decoder's refusal to name a single
    /// colour format — while the raw FourCC conversion may well return a real
    /// format. A freshness test that compared the raw conversion would find
    /// `0 != BGRA8` on every present and rebuild the backing forever, which is
    /// the failure a shared `latched_mapping_format` exists to make impossible.
    #[test]
    fn a_multiplane_backing_compares_equal_to_the_zero_it_latched() {
        let mut surf = Type4Surface {
            length: 0x1000,
            backing_pfn: 1,
            pixel_format: 0x4247_5241, // 'BGRA' — a format the converter knows
            plane_count: 2,
            planes: Default::default(),
            width: 8,
            height: 4,
            bytes_per_row: 32,
        };
        assert_ne!(
            iosurface_pixel_format_to_mtl(surf.pixel_format),
            0,
            "the fixture only means something if the raw conversion resolves"
        );
        assert_eq!(latched_mapping_format(&surf), 0, "multi-plane latches 0");

        let m = MappingEntry {
            width: 8,
            height: 4,
            format: 0,
            ..Default::default()
        };
        assert!(backing_matches_latched_geom(&m, &surf));
        // Dropping to one plane makes it a single-plane BGRA8 surface, which is
        // a real change of what the mapping describes.
        surf.plane_count = 1;
        assert!(!backing_matches_latched_geom(&m, &surf));
    }

    /// A single-plane surface must publish plane 0's offset, because both its
    /// consumers fold it in and one of them is the other pathway.
    ///
    /// `decode_type4_plane` reads four fields; the surface-level convenience
    /// copies on `Type4Surface` take three, and the synthesizer's single-plane
    /// arm used to publish only those three. A surface whose pixels start past
    /// the base of its allocation was then read and written at 0 — the
    /// multi-plane arm has always published each plane's offset, and
    /// `sample_window_from_device_surface` treats `base_offset` exactly as
    /// `sample_window_from_device_plane` treats a plane's.
    #[test]
    fn a_single_plane_backing_publishes_the_offset_its_pixels_start_at() {
        use crate::contract::iosurface_pages::{
            decode_device_surface, sample_window_prefer_device,
        };
        use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
        const BASE: u32 = 0x800;
        let (w, h, bpr) = (8u32, 4u32, 32u32);
        let mut surf = Type4Surface {
            length: 0x4000,
            backing_pfn: 1,
            pixel_format: 0x4247_5241, // 'BGRA'
            plane_count: 1,
            planes: Default::default(),
            width: w,
            height: h,
            bytes_per_row: bpr,
        };
        surf.planes[0] = Type4Plane {
            offset: BASE,
            width: w,
            height: h,
            bytes_per_row: bpr,
            bytes_per_element: 4,
        };
        assert!(
            !type4_is_multiplanar(&surf),
            "the single-plane arm is the one under test"
        );

        let desc = synthesize_device_desc_from_type4(&surf);
        let decoded = decode_device_surface(&desc).expect("device descriptor");
        assert_eq!(
            decoded.plane_count, 0,
            "single-plane publishes no plane records"
        );
        assert_eq!(decoded.base_offset, BASE);

        // The consumer, not just the field: the sample window must start at the
        // offset and its span must end past it, or publishing it bought nothing.
        let (off, got_bpr, end, from_device) =
            sample_window_prefer_device(Some(&desc), None, MTL_FORMAT_BGRA8_UNORM, w, h)
                .expect("surface-level window");
        assert!(from_device, "the window must come from the descriptor");
        assert_eq!(off, BASE as u64);
        assert_eq!(got_bpr, bpr);
        assert_eq!(
            end,
            BASE as u64 + (h as u64 - 1) * bpr as u64 + (w as u64 * 4)
        );

        // Zero stays zero — the ordinary case must not gain an offset.
        surf.planes[0].offset = 0;
        let zero = synthesize_device_desc_from_type4(&surf);
        assert_eq!(decode_device_surface(&zero).expect("desc").base_offset, 0);
    }

    /// The device descriptor's format word must survive both of the encodings
    /// it is written in.
    ///
    /// The x86 synthesizer writes an MTL ordinal for a known single-plane
    /// surface and the raw OSType otherwise; the arm64 mapper reads the guest's
    /// own descriptor, where media surfaces carry a FourCC. Narrowing with
    /// `as u16` is correct for one of those and destroys the other — `'BGRA'`
    /// becomes `0x5241`, which no format table accepts, so the mapping ends up
    /// with a format that refuses every sample window and every render target.
    #[test]
    fn the_device_descriptor_format_word_survives_both_of_its_encodings() {
        use crate::contract::pixel_format::{
            bytes_per_pixel, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
        };

        const BGRA_FOURCC: u32 = 0x4247_5241;

        // The failure the narrowing produced, stated as the thing not to return.
        assert!(
            bytes_per_pixel((BGRA_FOURCC & 0xffff) as u16).is_none(),
            "the truncation's output is not a format, which is why it was a bug"
        );
        assert_eq!(
            device_desc_format_to_mtl(BGRA_FOURCC),
            MTL_FORMAT_BGRA8_UNORM
        );

        // An ordinal fits in the descriptor's own 16-bit format fields and is
        // passed through as itself — including one above the old 0x200
        // magnitude boundary, which is why the test is width and not size.
        assert_eq!(
            device_desc_format_to_mtl(MTL_FORMAT_BGRA8_UNORM as u32),
            MTL_FORMAT_BGRA8_UNORM
        );
        assert_eq!(
            device_desc_format_to_mtl(MTL_FORMAT_RGBA16_FLOAT as u32),
            MTL_FORMAT_RGBA16_FLOAT
        );
        // MTLPixelFormatBGRA10_XR is 552, above the 0x200 boundary an earlier
        // magnitude test used and which `iosurface_pixel_format_to_mtl` records
        // as having been wrong for exactly this format. It still fits in 16
        // bits, so the width test carries it where a size test did not.
        assert_eq!(device_desc_format_to_mtl(552), 552);

        // Fail closed, not BGRA8: a multi-plane OSType and an unknown one.
        assert_eq!(device_desc_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
        assert_eq!(device_desc_format_to_mtl(0x5A5A_5A5A), 0);
        assert_eq!(device_desc_format_to_mtl(0), 0);
    }

    /// The type-4 probe order must visit task 0 first, the hint next, and every
    /// other task exactly once.
    ///
    /// It is the thing that makes the search terminate on the first probe for
    /// every surface this device has ever resolved, so its shape is the whole
    /// cost of the search. Two properties are load-bearing and neither is
    /// obvious from the iterator chain: no task may be probed **twice** (a
    /// duplicate is a wasted guest read on the hot present path, and with a
    /// misbehaving hint it would be 256 of them), and no task may be **missed**
    /// (a missed one is a surface that cannot be found at all).
    #[test]
    fn the_type4_probe_order_visits_task_zero_first_and_every_task_once() {
        use std::collections::HashSet;

        for hint in [0u32, 1, 7, MAX_TASKS as u32 - 1] {
            let order: Vec<u32> = type4_probe_order(hint).collect();
            assert_eq!(order[0], 0, "task 0 leads for hint {hint}");
            if hint != 0 {
                assert_eq!(order[1], hint, "the hint is probed second");
            }
            assert_eq!(
                order.len(),
                MAX_TASKS,
                "every task exactly once, no duplicate for hint {hint}"
            );
            let seen: HashSet<u32> = order.iter().copied().collect();
            assert_eq!(seen.len(), MAX_TASKS);
            assert!((0..MAX_TASKS as u32).all(|t| seen.contains(&t)));
        }

        // A hint outside the id space must not add a probe or lose one. It
        // cannot be found, so admitting it would cost a wasted read and — worse
        // — leave the `!= hint` filter matching nothing real.
        for bad in [MAX_TASKS as u32, u32::MAX] {
            let order: Vec<u32> = type4_probe_order(bad).collect();
            assert_eq!(order.len(), MAX_TASKS);
            assert_eq!(order, type4_probe_order(0).collect::<Vec<_>>());
        }
    }

    #[test]
    fn undecoded_type4_span_is_exactly_what_the_decoder_skips() {
        // One plane: the decoder consumes 0x14..0x24, so the tail starts there.
        let mut a = vec![0u8; 0x40];
        st64(&mut a[TYPE4_LEN..], 0x800000);
        st32(&mut a[TYPE4_BACKING_PFN..], 0x1234);
        st32(&mut a[TYPE4_PIXEL_FORMAT..], 0x4247_5241); // 'BGRA'
        a[TYPE4_PLANE_COUNT] = 1;
        st32(&mut a[TYPE4_PLANES..], 0); // plane0 offset
        st32(&mut a[TYPE4_PLANES + 4..], 1920);
        st32(&mut a[TYPE4_PLANES + 8..], 1080);
        st32(&mut a[TYPE4_PLANES + 12..], 1920 * 4);

        // Every decoded field can change without moving the undecoded span.
        let mut b = a.clone();
        st64(&mut b[TYPE4_LEN..], 0x900000);
        st32(&mut b[TYPE4_BACKING_PFN..], 0x9999);
        st32(&mut b[TYPE4_PIXEL_FORMAT..], 0x4c31_3062);
        st32(&mut b[TYPE4_PLANES + 4..], 1280);
        st32(&mut b[TYPE4_PLANES + 8..], 720);
        st32(&mut b[TYPE4_PLANES + 12..], 1280 * 4);
        assert_eq!(
            undecoded_type4_surface_bytes(&a),
            undecoded_type4_surface_bytes(&b),
            "changing only decoded fields must not look like a new shape"
        );

        // The span covers the three bytes after plane_count and the whole tail
        // past the plane records the decoder consumed.
        for probe in [0x11usize, 0x13, 0x24, 0x3f] {
            let mut c = a.clone();
            c[probe] ^= 0xff;
            assert_ne!(
                undecoded_type4_surface_bytes(&a),
                undecoded_type4_surface_bytes(&c),
                "byte {probe:#x} is undecoded and must be visible to the probe"
            );
        }

        // Bytes the decoder DOES read must not be in the span, or ordinary
        // surface-to-surface variation would look like a new shape forever.
        // `plane_count` (+0x10) is excluded on purpose: it is decoded AND it
        // moves the span's own boundary, which the two-plane case below pins.
        for probe in [0x00usize, 0x08, 0x0c, 0x14, 0x23] {
            let mut c = a.clone();
            c[probe] ^= 0xff;
            assert_eq!(
                undecoded_type4_surface_bytes(&a),
                undecoded_type4_surface_bytes(&c),
                "byte {probe:#x} is decoded and must stay out of the span"
            );
        }

        // A second plane moves the boundary: 0x24..0x34 becomes decoded.
        let mut two = a.clone();
        two[TYPE4_PLANE_COUNT] = 2;
        assert_eq!(
            undecoded_type4_surface_bytes(&two).len(),
            undecoded_type4_surface_bytes(&a).len() - TYPE4_PLANE_STRIDE,
            "the span shrinks by exactly one plane record"
        );

        // A record too short to decode reports nothing rather than a partial
        // span that would compare unequal against every real one.
        assert!(undecoded_type4_surface_bytes(&a[..TYPE4_MIN_LEN - 1]).is_empty());
    }
}
