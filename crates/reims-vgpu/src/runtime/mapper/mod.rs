//! IOSurface mapper capture + page-table / geometry resolve.
//!
//! Capture runs on the iosfc producer MMIO path (guest x19/x21/x22 still hold
//! the directed handoff from `do_host_mapping_gated`). Resolve builds
//! `MappingEntry.page_entries` and geometry from MappingInternal + device
//! descriptor via guest KVA reads ([`HostOps::read_kva`]).

use crate::contract::iosurface_pages::{
    self, build_table_plan, decode_device_surface, decode_mapper_request_entry, guest_kernel_va,
    mapper_request_published_entry_offset, mapping_span_bound, read_internal_desc_ptr,
    read_mapper_identity, read_mapper_internal, validate_mapper_internal, PagesMemory,
    DEVICE_DESC_LEN, MAPPER_CAPTURE_REG_MAPPER_DEVICE, MAPPER_CAPTURE_REG_MAPPING_INTERNAL,
    MAPPER_CAPTURE_REG_REQUEST_TYPE, MAPPER_REQUEST_ENTRY_LEN, MAPPER_REQUEST_MAP,
    MAPPER_REQUEST_UNMAP,
};
use crate::model::{DeviceState, MapperCapture};
use crate::runtime::host::{HostMemory, HostOps, MemError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapperDecline {
    CaptureMapperXregRead(MemError),
    CaptureRequestTypeXregRead(MemError),
    CaptureInternalXregRead(MemError),
    CaptureRequestTypeMismatch,
    CaptureInternalZero,
    CaptureInternalKvaInvalid,
    CaptureMapperKvaInvalid,
    DeviceDescriptorRead(MemError),
}

impl crate::observe::Decline for MapperDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CaptureMapperXregRead(_) => "mapper_capture_mapper_xreg_read",
            Self::CaptureRequestTypeXregRead(_) => "mapper_capture_request_type_xreg_read",
            Self::CaptureInternalXregRead(_) => "mapper_capture_internal_xreg_read",
            Self::CaptureRequestTypeMismatch => "mapper_capture_request_type_mismatch",
            Self::CaptureInternalZero => "mapper_capture_internal_zero",
            Self::CaptureInternalKvaInvalid => "mapper_capture_internal_kva_invalid",
            Self::CaptureMapperKvaInvalid => "mapper_capture_mapper_kva_invalid",
            Self::DeviceDescriptorRead(_) => "mapper_device_descriptor_read",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CaptureMapperXregRead(error)
            | Self::CaptureRequestTypeXregRead(error)
            | Self::CaptureInternalXregRead(error)
            | Self::DeviceDescriptorRead(error) => vec![(
                "host_reason",
                crate::observe::Decline::slug(error).to_string(),
            )],
            _ => Vec::new(),
        }
    }
}

fn refusal_reason(status: &iosurface_pages::Status) -> &'static str {
    crate::observe::Refusal::refusal(status)
        .expect("an IOSurface contract error must carry a refusal reason")
}

/// Fail-visible, **de-duplicated per `(mapping_id, reason)`**, for the
/// `resolve_mapping_backing` blind spot: a mapped surface whose page-table /
/// geometry resolve fails leaves the mapping silently un-resolved, and every
/// downstream present/Store/sample paints or writes back **black** for it with
/// no log naming why. `resolve_mapping_backing` runs on the per-present `force`
/// path (drain.rs), so a bare `observe::fail` at a failing site would flood.
/// This latch logs each `(mapping_id, reason)` **once** and is cleared for a
/// mapping the moment it resolves ([`clear_resolve_fail`]), so a genuinely
/// broken mapping logs one line, a flapping one re-logs per transition, and a
/// healthy boot fires nothing. Runs on the drain worker (off the QEMU main
/// core). Speculative/not-ready returns (unmapped, dims-not-yet-landed) are
/// **not** routed here — only genuine anomalies for an already-mapped surface.
fn resolve_fail_latch() -> &'static std::sync::Mutex<std::collections::HashSet<(u32, &'static str)>>
{
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, &'static str)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
}

fn note_resolve_fail(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::fail(detail);
    }
}

fn note_resolve_keep_cached(mapping_id: u32, reason: &'static str, detail: String) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if guard.insert((mapping_id, reason)) {
        crate::observe::off(detail);
    }
}

/// Fail-visible capture miss, sharing the same per-`(mapping_id, reason)` latch
/// as [`note_resolve_fail`]. `capture_at_producer` runs on the publishing vCPU
/// (iosfc producer MMIO write), not the drain worker, but it fires **once per
/// mapper-ring publish** — a rare setup/map event, never per-frame — so a
/// latched genuine-only line here costs nothing on the hot path. A capture miss
/// for an already-decoded MAP/UNMAP request means the mapping's `MappingInternal`
/// never attaches, and every downstream present/Store for it paints **black**
/// with no reason. Speculative returns (producer==0, ring not ready, a non
/// MAP/UNMAP request type) are **not** routed here. Sharing the latch means a
/// mapping that later resolves cleanly re-arms its capture reasons too
/// ([`clear_resolve_fail`] clears all reasons for the id).
fn note_capture_fail(mapping_id: u32, reason: &'static str, detail: String) {
    note_resolve_fail(mapping_id, reason, detail);
}

/// Re-arm every reason latch for a mapping that just resolved, so a later
/// genuine failure on the same mapping is logged again (catches flapping).
fn clear_resolve_fail(mapping_id: u32) {
    let mut guard = resolve_fail_latch()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    guard.retain(|(mid, _)| *mid != mapping_id);
}

fn capture_xreg_failed(
    mapping_id: u32,
    producer: u32,
    decline: MapperDecline,
) -> Option<MapperCapture> {
    note_capture_fail(
        mapping_id,
        crate::observe::Decline::slug(&decline),
        crate::observe::Emit::decline("mapper_capture_fail", &decline)
            .field("mapping", mapping_id)
            .field("producer", producer)
            .render(),
    );
    None
}

/// Adapter: mapper internals are KVA; page content GPAs use HostMemory.
struct MapperMem<'a, H: HostMemory + HostOps> {
    host: &'a H,
    last_error: std::cell::Cell<Option<MemError>>,
}

impl<'a, H: HostMemory + HostOps> MapperMem<'a, H> {
    fn new(host: &'a H) -> Self {
        Self {
            host,
            last_error: std::cell::Cell::new(None),
        }
    }

    fn last_error(&self) -> Option<MemError> {
        self.last_error.get()
    }
}

impl<H: HostMemory + HostOps> PagesMemory for MapperMem<'_, H> {
    fn read(&self, address: u64, dst: &mut [u8]) -> bool {
        if guest_kernel_va(address) {
            match self.host.read_kva(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        } else {
            match self.host.read_gpa(address, dst) {
                Ok(()) => true,
                Err(e) => {
                    self.last_error.set(Some(e));
                    false
                }
            }
        }
    }
    fn is_kernel_va(&self, address: u64) -> bool {
        guest_kernel_va(address)
    }
    fn is_ram_gpa(&self, address: u64) -> bool {
        self.host.is_ram_gpa(address)
    }
}

/// Capture mapper handoff registers while still on the publishing vCPU.
///
/// Call from the iosfc producer MMIO write path before scheduling the drain BH.
pub fn capture_at_producer<H: HostMemory + HostOps>(
    state: &DeviceState,
    host: &H,
    producer: u32,
) -> Option<MapperCapture> {
    if producer == 0 || state.iosfc.ring_base == 0 {
        return None;
    }
    let entry_off = mapper_request_published_entry_offset(producer)?;
    let mut e = [0u8; MAPPER_REQUEST_ENTRY_LEN];
    host.read_gpa(state.iosfc.ring_base + entry_off, &mut e)
        .ok()?;
    let request = decode_mapper_request_entry(&e).ok()?;
    if request.request_type != MAPPER_REQUEST_MAP && request.request_type != MAPPER_REQUEST_UNMAP {
        return None;
    }
    if !crate::model::is_mapping_id(request.mapping_id) {
        return None;
    }

    // From here the ring entry is a decoded MAP/UNMAP for a valid mapping_id, so
    // any failure below is a genuine capture miss (the handoff registers do not
    // corroborate the request), not the speculative not-ready poll — log it once
    // per (mapping_id, reason). The mapping's MappingInternal never attaches and
    // downstream present/Store paints black otherwise.
    let mid = request.mapping_id;
    let mapper = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPER_DEVICE) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureMapperXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let rtype = match host.read_xreg(MAPPER_CAPTURE_REG_REQUEST_TYPE) {
        Ok(value) => value as u32,
        Err(error) => {
            let decline = MapperDecline::CaptureRequestTypeXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    let internal = match host.read_xreg(MAPPER_CAPTURE_REG_MAPPING_INTERNAL) {
        Ok(value) => value,
        Err(error) => {
            let decline = MapperDecline::CaptureInternalXregRead(error);
            return capture_xreg_failed(mid, producer, decline);
        }
    };
    if rtype != request.request_type {
        let decline = MapperDecline::CaptureRequestTypeMismatch;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("rtype", rtype)
                .field("request_type", request.request_type)
                .render(),
        );
        return None;
    }
    if internal == 0 {
        let decline = MapperDecline::CaptureInternalZero;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .render(),
        );
        return None;
    }
    if !guest_kernel_va(internal) {
        let decline = MapperDecline::CaptureInternalKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }
    if mapper != 0 && !guest_kernel_va(mapper) {
        let decline = MapperDecline::CaptureMapperKvaInvalid;
        note_capture_fail(
            mid,
            crate::observe::Decline::slug(&decline),
            crate::observe::Emit::decline("mapper_capture_fail", &decline)
                .field("mapping", mid)
                .field("mapper_kva", format!("{mapper:#x}"))
                .render(),
        );
        return None;
    }

    let mem = MapperMem::new(host);
    let fields = match read_mapper_identity(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            note_capture_fail(
                mid,
                reason,
                crate::observe::Emit::refusal("mapper_capture_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mid)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .render(),
            );
            return None;
        }
    };
    let status = validate_mapper_internal(&mem, mid, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_capture_fail(
            mid,
            reason,
            crate::observe::Emit::refusal("mapper_capture_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mid)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return None;
    }

    Some(MapperCapture {
        producer,
        mapper_device_kva: mapper,
        request_type: rtype,
        mapping_internal: internal,
    })
}

/// Apply a capture to the mapping named by the just-drained ring entry.
pub fn apply_capture(state: &mut DeviceState, cap: &MapperCapture, mapping_id: u32) -> bool {
    // Neither branch below releases a deferred writeback window. A type-11
    // render Store writes guest pages on its own path, so an UNMAP (or a MAP
    // that re-backs the slot with a different MappingInternal, orphaning the
    // old identity) leaves no mapping-keyed render obligation behind, because a
    // render Store lands its frame in guest pages before it returns.
    if cap.request_type == MAPPER_REQUEST_UNMAP {
        return state.unmap_surface(mapping_id);
    }
    if cap.request_type != MAPPER_REQUEST_MAP {
        return false;
    }
    if cap.mapper_device_kva != 0 {
        state.mapper_device_kva = cap.mapper_device_kva;
    }
    state.attach_mapping_internal(mapping_id, cap.mapping_internal)
}

/// Resolve page table + device-descriptor geometry for a mapped slot.
///
/// Safe to call repeatedly; refreshes pages when `mapping_internal` is set.
pub fn resolve_mapping_backing<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if !m.mapped || m.mapping_internal == 0 {
        return false;
    }
    let internal = m.mapping_internal;
    let mapper = state.mapper_device_kva;
    let cached_pages = m.page_entries.len();
    let cached_table = m.page_table_kva;
    let had_cached_pages = cached_pages != 0;
    let mem = MapperMem::new(host);

    let fields = match read_mapper_internal(&mem, internal, mapper != 0, mapper) {
        Ok(f) => f,
        Err(status) => {
            let reason = refusal_reason(&status);
            let host_error = mem.last_error();
            let host_reason = host_error
                .map(|error| crate::observe::Decline::slug(&error))
                .unwrap_or("none");
            if had_cached_pages {
                // QEMU can stop exposing a CPU-backed KVA alias after the
                // mapper handoff while the already-validated GPA page plan
                // remains live. Likewise, a transiently unmapped debug-read
                // alias says nothing about the cached guest-physical pages.
                // Both are the normal revalidation fallback, not a decoded
                // guest-command refusal; emitting them once per recycled
                // mapping id made a healthy boot fire dozens of new Phase-5
                // lines. Keep other cached-plan failures visible because they
                // describe malformed identity fields rather than alias
                // availability.
                if matches!(host_error, Some(MemError::NoCpu | MemError::Unmapped)) {
                    return true;
                }
                note_resolve_keep_cached(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_revalidate_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("pages", cached_pages)
                        .field("table", format!("{cached_table:#x}"))
                        .field("internal", format!("{internal:#x}"))
                        .field("mapper_kva", format!("{mapper:#x}"))
                        .field("host_reason", host_reason)
                        .render(),
                );
                return true;
            }
            // A mapped surface (m.mapped, mapping_internal != 0) whose mapper
            // internal KVA is unreadable is a genuine anomaly, not the
            // not-yet-mapped poll — every downstream present/Store for this
            // mapping then paints black with no reason.
            note_resolve_fail(
                mapping_id,
                reason,
                crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                    .expect("the error arm cannot carry Status::Ok")
                    .field("mapping", mapping_id)
                    .field("internal", format!("{internal:#x}"))
                    .field("mapper_kva", format!("{mapper:#x}"))
                    .field("host_reason", host_reason)
                    .render(),
            );
            return false;
        }
    };
    let status = validate_mapper_internal(&mem, mapping_id, &fields);
    if status != iosurface_pages::Status::Ok {
        let reason = refusal_reason(&status);
        note_resolve_fail(
            mapping_id,
            reason,
            crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                .expect("the non-Ok branch must carry a refusal")
                .field("mapping", mapping_id)
                .field("internal", format!("{internal:#x}"))
                .render(),
        );
        return false;
    }

    // Geometry from device descriptor when present; cache full 0x200 for
    // biplanar plane selection (mapping_span_bound).
    let mut width = 0u32;
    let mut height = 0u32;
    let mut format = 0u16;
    // Guest page size for *this* device — never a bare arm PAGE_SIZE constant.
    let guest_page = state.page_size();
    let mut min_size = guest_page;
    let mut device_desc: Option<Vec<u8>> = None;
    match read_internal_desc_ptr(&mem, internal) {
        Ok(desc_kva) => {
            let mut desc = [0u8; DEVICE_DESC_LEN];
            if !mem.read(desc_kva, &mut desc) {
                let decline = MapperDecline::DeviceDescriptorRead(
                    mem.last_error().unwrap_or(MemError::Unmapped),
                );
                note_resolve_fail(
                    mapping_id,
                    crate::observe::Decline::slug(&decline),
                    crate::observe::Emit::decline("mapper_device_descriptor_fallback", &decline)
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .field("descriptor", format!("{desc_kva:#x}"))
                        .render(),
                );
            } else {
                device_desc = Some(desc.to_vec());
                if let Some(surf) = decode_device_surface(&desc) {
                    if surf.alloc_size as u64 > 0 {
                        min_size = (surf.alloc_size as u64).max(guest_page);
                    }
                    if surf.width > 0 && surf.height > 0 {
                        width = surf.width;
                        height = surf.height;
                        // Not `as u16`: this field carries an MTL ordinal or
                        // an OSType FourCC depending on who wrote the
                        // descriptor, and narrowing a FourCC produces a format
                        // nothing in the device accepts. See
                        // `objects::device_desc_format_to_mtl`.
                        format =
                            crate::runtime::objects::device_desc_format_to_mtl(surf.pixel_format);
                        if let Some(end) = mapping_span_bound(Some(&desc), format, width, height) {
                            min_size = min_size.max(end).max(guest_page);
                        }
                    }
                }
            }
        }
        Err(status) => {
            let reason = refusal_reason(&status);
            // A zero descriptor pointer is the documented "not present" state:
            // geometry can come from the texture object. A failed read or a
            // nonzero invalid pointer is a real fallback decision.
            if reason != "iosurface_mapper_device_desc_pointer_zero" {
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_device_descriptor_fallback", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("internal", format!("{internal:#x}"))
                        .render(),
                );
            }
        }
    }

    // Texture-path geom (type-11 object dims) refines span for single-plane; for
    // multi-plane, prefer alloc_size already latched from the device descriptor.
    if let Some(m) = state.mappings.get(&mapping_id) {
        if m.has_geom && m.width > 0 && m.height > 0 {
            width = m.width;
            height = m.height;
            format = if m.format != 0 { m.format } else { format };
            let desc_slice = device_desc.as_deref().or(m.device_desc_complete());
            if let Some(end) = mapping_span_bound(desc_slice, format, width, height) {
                min_size = min_size.max(end).max(guest_page);
            }
        }
    }

    let plan = match build_table_plan(&mem, mapping_id, &fields, min_size, state.page_shift) {
        Ok(p) => p,
        Err(status) => {
            // Still latch geom / device desc if we decoded them, even without pages yet.
            if let Some(ref d) = device_desc {
                let _ = state.set_mapping_device_desc(mapping_id, d);
            }
            if width > 0 && height > 0 {
                let _ = state.set_mapping_geom(mapping_id, width, height, format);
                // Geometry IS known, yet no page table covers its
                // `min_size` span — the short-page-table → black-tile class
                // (fail-closed Store writeback / sample walk while the geom is
                // set). Distinct from the dims-not-yet-landed poll (width==0),
                // which stays silent as legitimate not-ready control flow.
                let reason = refusal_reason(&status);
                note_resolve_fail(
                    mapping_id,
                    reason,
                    crate::observe::Emit::refusal("mapper_resolve_fail", &status)
                        .expect("the error arm cannot carry Status::Ok")
                        .field("mapping", mapping_id)
                        .field("width", width)
                        .field("height", height)
                        .field("format", format!("{format:#x}"))
                        .field("min_size", min_size)
                        .render(),
                );
            }
            return false;
        }
    };

    // The count that carried `build_table_plan`'s second candidate to its
    // deletion, kept because it is what would falsify that deletion.
    //
    // That function used to chase `MappingInternal` `+0x48` then `+0xb8`, and
    // `+0x50` then `+0x28`, taking whichever parsed first. The question was
    // whether that is a "try both, keep the one that works" ladder or two
    // layouts handled side by side, and it turns on how often both fields are
    // populated at once: never, and it dispatches; always, and it chooses.
    //
    // Measured **223 successful resolves across two driven arm64 workloads**,
    // and both fields held a kernel VA on every single one while `+0x48` won
    // every single one. So it chose, always the same way, and the second chase
    // never carried a resolve — which is a fallback, and it is gone.
    //
    // `iosurface_pt_cand_both` therefore stays as the premise's alarm rather
    // than as a tally: it should keep reading equal to
    // `iosurface_pt_cand_only_48 + itself`, i.e. essentially 100%. A run where
    // `only_48` grows means the two fields are *not* both always populated,
    // which is the reading under which the deleted chase was load-bearing.
    //
    // This rail is arm64-only — it is entered from `capture_at_producer`, which
    // needs `HostOps::read_xreg`, and the x86 PCI shim returns -1 for that
    // unconditionally — so only an arm64 boot can move these.
    crate::runtime::drain::note_store_route(if plan.candidates.other_field_populated {
        "iosurface_pt_cand_both"
    } else {
        "iosurface_pt_cand_only_48"
    });

    // Read before the `get_mut` below takes `state` mutably.
    let page_shift = state.page_shift;
    let mut retired = None;
    let mut retired_import = None;
    let mut incarnation_changed = false;
    let mut reprieved = false;
    let mut pages_changed = false;
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        // A condemned slot (trailing DeleteIOSurfaceBacking2, no resolve
        // since) compares against the stashed fingerprint: the same plan is
        // the SAME incarnation — the delete was stale, keep the generation so
        // the resident and deferred windows stay live (black-band class). A
        // different plan is a genuine new incarnation.
        let condemned = m.condemned_entries.take();
        let prev_pages = m.page_entries.len();
        (pages_changed, incarnation_changed, reprieved) =
            plan_adoption_decision(condemned.as_deref(), &m.page_entries, &plan.entries);
        // New page table ⇒ the contiguous view (and any Metal texture aliasing
        // it) describe the old pages; retire them before adopting the plan.
        if m.contig_ptr != 0 && pages_changed {
            retired = Some((m.contig_ptr, m.contig_len));
            m.contig_ptr = 0;
            m.contig_len = 0;
            m.contig_footprint = None;
            retired_import = m.contig_import.take().map(|import| {
                import.retire();
                import.id()
            });
        }
        if pages_changed {
            DeviceState::bump_map_generation(m);
        }
        // The guest-physical footprint this incarnation authorises us to write.
        //
        // A guest kernel panic names a *physical page* (`pmap_page_protect()
        // ... pn=0x46b53b`), and nothing this device emitted could be compared
        // against it — so "did we write there?" was unanswerable, and the
        // random-victim panic class this project has recorded stayed a signature
        // with no way to confirm or clear this device — see
        // `observe::footprint`, which carries that account. Every mapping-rail write is bounded
        // to the page list adopted here, so the union of these spans over a boot
        // is exactly the set of pages those writes can reach. A `pn` inside it
        // is evidence; a `pn` outside every one of them exonerates the rail.
        //
        // One line per surface incarnation, not per write: the key is
        // (mapping, generation) and the generation only moves when the PFNs do,
        // so this is bounded by how often the guest rewires a surface and is
        // safe to leave on. min/max over the entries is O(pages) once per
        // incarnation, against an O(pages) table build that just ran.
        // Keyed on the ADOPTION, not on `pages_changed`. Two earlier cuts of
        // this line were silent for entire boots, and both were the
        // branch-versus-arm trap `runtime::draw::vulkan`'s store-route reporter
        // states, committed by a change that
        // cites it. The first logged only when a span resolved, so it could not
        // distinguish "no span" from "never ran". The second moved to
        // `pages_changed` on the reasoning that `map_gen` climbing past 100
        // proved that branch ran — but `bump_map_generation` has five other
        // call sites, so the generation was never evidence about this one.
        //
        // `pages_changed` is genuinely false here even on a first population:
        // the reprieve path (`condemned` holds the fingerprint, the plan
        // matches it) repopulates an emptied `page_entries` with
        // `pages_changed == false`. The adoption below is what every write is
        // then bounded by, so that is what has to be reported.
        //
        // Dedup is `first_sight` on the span itself, which keeps it bounded by
        // *distinct footprints* rather than by resolve rate — a mapping
        // re-resolved every frame to the same pages logs once.
        if let Some((lo, hi)) = entry_gpa_span(&plan.entries, page_shift) {
            let key = span_first_sight_key(mapping_id, lo, hi, page_shift);
            if crate::observe::first_sight(SPAN_SEEN_MAPPER, key) {
                // `src=mapper` against the type-4 adoption site's `src=type4`,
                // which says which path a surface's page list arrived through.
                //
                // Read that field with the latch in mind: until the two sites
                // were given separate `first_sight` namespaces they shared one,
                // and on identical keys, so the type-4 site claimed every
                // footprint it reached first and this one was suppressed for
                // that footprint permanently. Every `mapping_gpa_span` line in
                // an x86 boot read `src=type4`, which was taken as evidence that
                // the page list arrives at the type-4 site — but the latch could
                // not have produced any other reading. Whether this site is
                // genuinely quiet is now an open question again, and a driven
                // boot is what answers it.
                crate::observe::off(format!(
                    "mapping_gpa_span mid={mapping_id} gen={} pages={} src=mapper \
                     prev_pages={prev_pages} \
                     changed={} lo={lo:#x} hi={:#x} pn_lo={:#x} pn_hi={:#x}",
                    m.map_generation,
                    plan.entries.len(),
                    pages_changed as u8,
                    hi + (1u64 << page_shift),
                    lo >> page_shift,
                    hi >> page_shift,
                ));
            }
        }
        m.page_entries = plan.entries;
        m.page_table_kva = plan.page_table_kva;
        m.mapping_internal = internal;
        m.mapped = true;
        if let Some(ref d) = device_desc {
            m.device_desc = d.clone();
        }
    }
    if let Some(v) = retired {
        state.retired_views.push(v);
    }
    if let Some(import) = retired_import {
        state.retired_guest_imports.push(import);
    }
    if incarnation_changed {
        // The condemned backing really died and the id now carries a new
        // surface: drop the prior incarnation's deferred windows before any
        // access could flush old content through the new pages.
    } else if reprieved {
        // Wrong-PFN guard on the REPRIEVE path — the blind spot the rewire guard
        // above cannot cover. A reprieve keeps this mapping's page plan WITHOUT
        // bumping `map_generation` (the delete looked stale: the plan still
        // fingerprints the condemned pages). But a guest that FREED the backing
        // and handed the SAME physical pages to another surface — yet has not
        // rewired this mapping's GPU page table away from them — fingerprints
        // identical here, so `pages_changed` is false and the rewire guard never
        // runs. A render Store landing through the kept plan would then write
        // pages another live surface owns, or recycled userspace heap, which is
        // the WindowServer malloc free-list corruption class.
        //
        // This used to run only when the reprieve found an armed deferred
        // window, on the reasoning that a still-armed flush was the thing that
        // would DMA through the stale plan. Stores land at the Store now, so
        // there is no armed set to qualify on and the hazard belongs to the page
        // plan rather than to any pending write — so the check runs on every
        // reprieve. Reprieves are rare; this is not a hot path.
        //
        // A detected cross-surface alias is a proven ownership violation, so
        // fail closed: name it, invalidate the plan, and make this resolve fail.
        if !surface_pages_are_exclusively_owned(state, mapping_id, "reprieve") {
            return false;
        }
    }
    if width > 0 && height > 0 {
        let _ = state.set_mapping_geom(mapping_id, width, height, format);
    }
    // Wrong-PFN rewire-race guard: a freshly adopted page plan must not alias
    // a *different* live surface's guest pages. Two distinct live IOSurface
    // mappings backing the same physical page means one holds a stale/wrong
    // PFN — a pixel writeback through it scribbles the other surface, or (if
    // the page was recycled to userspace) guest heap: the WindowServer malloc
    // free-list corruption class. A detected alias is a proven ownership
    // violation, so fail closed: name it, drop deferred writes, invalidate the
    // adopted plan, and make this resolve fail. Runs only on a genuine rewire
    // (`pages_changed`), on the drain worker.
    if pages_changed && !surface_pages_are_exclusively_owned(state, mapping_id, "rewire") {
        return false;
    }
    // Resolved: re-arm the fail latch so a later genuine failure (a re-map that
    // goes bad, a corrupted descriptor) is logged again rather than swallowed.
    clear_resolve_fail(mapping_id);
    true
}

/// Whether `mapping_id`'s adopted page plan is free of cross-surface aliasing.
///
/// Two distinct live IOSurface mappings backing the same guest physical page
/// means one holds a stale PFN, and a pixel writeback through it scribbles the
/// other surface — or, if the page was recycled to userspace, guest heap. So a
/// detection is a proven ownership violation and this fails closed: it names
/// the collision, drops the deferred writes and invalidates the plan, and the
/// caller must fail its resolve.
///
/// `site` names which rewire reached here (`reprieve` or `rewire`) so the two
/// populations stay separable in the log.
fn surface_pages_are_exclusively_owned(
    state: &mut DeviceState,
    mapping_id: u32,
    site: &str,
) -> bool {
    let Some((gpa, owner)) = first_surface_page_collision(state, mapping_id) else {
        return true;
    };
    let mine_pages = state
        .mappings
        .get(&mapping_id)
        .map(|m| m.page_entries.len())
        .unwrap_or(0);
    fail_closed_surface_page_collision(state, mapping_id, gpa, owner, mine_pages, site);
    false
}

/// Detect the wrong-PFN rewire-race corruption vector: the mapping `mapping_id`
/// just adopted a fresh page plan whose page base is also owned by a
/// *different* currently-live surface mapping. Two distinct live IOSurface
/// mappings must never back the same guest physical page; if they do, one holds
/// a stale/wrong PFN and a writeback through it corrupts memory it does not own
/// (see the WindowServer heap-corruption class).
///
/// A detection **fails the resolve closed** — see
/// [`surface_pages_are_exclusively_owned`], which is how both call sites reach
/// this. It is not measure-only; the doc said so for a while and was wrong,
/// which is the worse way round for a detector on the guest-corruption rail.
///
/// Cost O(this_pages + Σ other live pages); called only on a rewire.
fn first_surface_page_collision(state: &DeviceState, mapping_id: u32) -> Option<(u64, u32)> {
    let page_shift = state.page_shift;
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let mine: std::collections::HashSet<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .map(page_base)
        .collect();
    if mine.is_empty() {
        return None;
    }
    for (&other_id, other) in &state.mappings {
        if other_id == mapping_id || !other.mapped || other.page_entries.is_empty() {
            continue;
        }
        for &e in &other.page_entries {
            if let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift) {
                if mine.contains(&page_base(gpa)) {
                    return Some((page_base(gpa), other_id));
                }
            }
        }
    }
    None
}

/// Always-on, deduped-per-`(mid, owner, gpa)` fail line for a cross-surface
/// page alias. Off-main-core (drain worker resolve path). Fires zero on a
/// healthy boot (distinct live surfaces never share a physical page).
fn note_surface_page_collision(
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u32, u64)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((mapping_id, owner, gpa))
    {
        crate::observe::fail(format!(
            "mapping_pages fail reason=surface_page_collision path={path} mid={mapping_id} \
             owner={owner} gpa={gpa:#x} mine_pages={mine_pages}"
        ));
    }
}

/// A cross-surface page alias is a structural ownership violation: any
/// host-authored pixel write through `mapping_id` can scribble another live
/// surface or, for the recycled-heap variant, a userspace heap page. Keep the
/// always-on forensic line, then fail closed by dropping deferred windows and
/// invalidating the adopted page plan so the next writer must re-resolve instead
/// of writing through known-bad pages.
fn fail_closed_surface_page_collision(
    state: &mut DeviceState,
    mapping_id: u32,
    gpa: u64,
    owner: u32,
    mine_pages: usize,
    path: &str,
) {
    note_surface_page_collision(mapping_id, gpa, owner, mine_pages, path);
    let _ = state.invalidate_mapping_pages(mapping_id);
}

/// Lowest and highest page-aligned GPA a page-entry list resolves to.
///
/// Invalid entries are skipped rather than failing the span: the span is a
/// *bound* on where writes through this list can land, and an entry that does
/// not resolve is one that no write can reach. `None` when nothing resolves.
///
/// The bound is inclusive of `hi` — it names the first byte of the last page,
/// not the end of it — so a caller reporting a range must add one page. The
/// page list is not sorted and is not contiguous, so `[lo, hi]` is a hull and
/// not a promise that every page inside it belongs to this surface.
pub(crate) fn entry_gpa_span(entries: &[u32], page_shift: u32) -> Option<(u64, u64)> {
    let (mut lo, mut hi) = (u64::MAX, 0u64);
    for &e in entries {
        if let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift) {
            lo = lo.min(gpa);
            hi = hi.max(gpa);
        }
    }
    (lo != u64::MAX).then_some((lo, hi))
}

/// `first_sight` namespace for the mapper's own adoption span line.
///
/// Separate from [`SPAN_SEEN_TYPE4`] on purpose, and the separation is the
/// point. Both emitters print `mapping_gpa_span`, both dedup on
/// [`span_first_sight_key`], and the key is built from the same three values at
/// both — mapping id, span low, span high. Sharing one namespace therefore made
/// them share one latch: whichever site reached a given footprint first claimed
/// it, and the other could never report that footprint at all.
///
/// That is not a harmless overlap, because the two lines are read against each
/// other. `src=` exists to say which adoption path a surface's page list
/// arrived through, and the type-4 site wins the race in practice, so the
/// mapper's silence was manufactured by the latch rather than observed. Two
/// namespaces make each site's silence its own evidence.
pub(crate) const SPAN_SEEN_MAPPER: &str = "mapping_gpa_span_mapper";

/// `first_sight` namespace for the type-4 adoption span line. See
/// [`SPAN_SEEN_MAPPER`] for why the two are not one.
pub(crate) const SPAN_SEEN_TYPE4: &str = "mapping_gpa_span_type4";

/// Dedup discriminant for a `mapping_gpa_span` line: the footprint identity.
///
/// Keyed on the span rather than on the resolve, so a mapping re-resolved every
/// frame to the same pages logs once while a mapping that moves logs again.
/// Shared by both emitters so the two cannot drift apart on what counts as the
/// same footprint — they are compared against each other, which only means
/// anything if they agree on the identity.
pub(crate) fn span_first_sight_key(mapping_id: u32, lo: u64, hi: u64, page_shift: u32) -> u64 {
    (u64::from(mapping_id) << 40) ^ (lo >> page_shift) ^ (hi << 20)
}

/// Incarnation decision when adopting a freshly resolved page plan into a
/// mapping slot. `condemned` is the fingerprint a trailing
/// `DeleteIOSurfaceBacking2` stashed (None when the slot is not condemned).
///
/// Returns `(pages_changed, incarnation_changed, reprieved)`:
/// `incarnation_changed` = the condemned backing really died and the id now
/// carries different pages (drop the old windows); `reprieved` = the delete
/// was stale — the plan matches the fingerprint, the same incarnation lives
/// on (keep generation, resident, deferred windows).
pub(crate) fn plan_adoption_decision(
    condemned: Option<&[u32]>,
    current: &[u32],
    plan: &[u32],
) -> (bool, bool, bool) {
    let pages_changed = match condemned {
        Some(old) => old != plan,
        None => current != plan,
    };
    (
        pages_changed,
        condemned.is_some() && pages_changed,
        condemned.is_some() && !pages_changed,
    )
}

/// True when the cached page table covers the type-11 sample/write span for
/// the latched geom (archive table build uses the same min_size).
///
/// Early resolve often runs before type-11 object dims land (`min_size` =
/// PAGE_SIZE only). Leaving a short `page_entries` while `has_geom` is true
/// makes Store writeback and sample page walks fail-closed on tiles (Favourites
/// 249² with ~16 pages vs ~63 required) while the Metal attachment still holds
/// content. Re-resolve when the span no longer fits.
pub fn pages_cover_geom(state: &DeviceState, mapping_id: u32) -> bool {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return false;
    };
    if m.page_entries.is_empty() {
        return false;
    }
    if !m.has_geom || m.width == 0 || m.height == 0 {
        // No geom yet — any non-empty table is acceptable until dims latch.
        return true;
    }
    let format = if m.format != 0 {
        m.format
    } else {
        // Match scanout/writeback default when format not latched.
        crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM
    };
    let Some(span_end) = mapping_span_bound(m.device_desc_complete(), format, m.width, m.height)
    else {
        return false;
    };
    let page_size = crate::contract::iosurface_pages::page_size_of(state.page_shift);
    let covered = (m.page_entries.len() as u64).saturating_mul(page_size);
    covered >= span_end.max(page_size)
}

/// Ensure pages (and geom if possible) before scanout paint / type-11 Store.
///
/// Re-resolves when the table is empty, geom is missing, **or** the cached page
/// count cannot cover the latched W×H sample window (stale early resolve).
pub fn ensure_resolved_for_scanout<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    let (mapped, has_internal, empty_pages, has_geom) = match state.mappings.get(&mapping_id) {
        Some(m) => (
            m.mapped,
            m.mapping_internal != 0,
            m.page_entries.is_empty(),
            m.has_geom,
        ),
        None => return false,
    };
    let needs = mapped
        && has_internal
        && (empty_pages || !has_geom || !pages_cover_geom(state, mapping_id));
    if needs {
        resolve_mapping_backing(state, host, mapping_id)
    } else {
        mapped && !empty_pages
    }
}

/// Fail-closed page-list revalidation before host writeback or import-present.
///
/// When `mapping_internal` is set **and** we previously resolved a live
/// `page_table_kva`, re-walk MappingInternal so we never write through PFNs the
/// guest recycled (zone freelist `0xff000000ff000000` class). Resolve failure
/// **invalidates** a live table rather than writing stale PFNs.
///
/// Manual / unit-test page lists (`page_table_kva == 0`) keep their entries when
/// resolve is not available — product MAP always re-resolves once KVA is known.
pub fn revalidate_mapping_pages<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> bool {
    revalidate_mapping_reason(state, host, mapping_id).is_none()
}

/// Precise reason a revalidate missed, or `None` when the mapping is resolvable.
///
/// The bool [`revalidate_mapping_pages`] collapses four distinct outcomes into
/// one "false", which forces a caller that fail-logs a lost flush to emit a
/// single `reason=revalidate` slug — and a future reader then cannot tell a
/// benign teardown window from a genuine content-drop without hunting for a
/// paired `map_revalidate resolve_fail` line. This returns the specific slug so
/// the two never share a status (AGENTS.md: each distinct check owns its slug):
/// - `revalidate_gone` / `revalidate_unmapped` — the guest already dropped the
///   mapping (pageoff/unwire raced ahead of the flush trigger); nothing to write
///   back to, benign.
/// - `revalidate_resolve_fail` — a live page table turned unreadable; the real
///   content-drop risk, and the only one that also emits the `st=invalidate`
///   line below.
///
/// The empty-page-list outcome is not one outcome. Four different states reach
/// it, and they were all reported as `revalidate_no_pages` with a doc comment
/// calling the class "a transient (re)wire gap" — one of the four, asserted for
/// all of them. 106 render-flush losses across 73 boots carry that slug and none
/// of them says which state produced it. Each check now owns its own:
/// - `revalidate_condemned` — `DeleteIOSurfaceBacking2` moved the page list into
///   `condemned_entries` and no resolve has re-adopted it. The guest deleted the
///   backing; there is nothing safe to write through.
/// - `revalidate_no_internal` — no `MappingInternal`, so **no resolve was ever
///   attempted**, and the page list is empty for some other reason. Note that a
///   zero `mapping_internal` is NOT itself a sign of missing backing: measured on
///   the rail, 2280 render windows in one boot were armed on mappings with
///   `mapping_internal == 0` and `page_entries.len() == 2040`, and all but two
///   flushed normally, because a non-empty page list returns `None` above.
/// - `revalidate_resolve_miss` — resolve ran and missed, with no live page table
///   to condemn (so not `resolve_fail`).
/// - `revalidate_empty_after_resolve` — resolve ran and *succeeded*, and the page
///   list is still empty. The genuinely surprising one.
/// - `revalidate_unmapped_late` / `revalidate_gone_late` — the mapping was
///   mapped on entry and is unmapped or absent after resolve; teardown raced the
///   revalidate, which the entry-side `revalidate_unmapped` cannot see.
pub fn revalidate_mapping_reason<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> Option<&'static str> {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Some("revalidate_gone");
    };
    if !m.mapped {
        return Some("revalidate_unmapped");
    }
    let has_internal = m.mapping_internal != 0;
    let had_live_table = m.page_table_kva != 0;
    let had_pages = !m.page_entries.is_empty();
    // Whether the resolve below ran at all, and whether it reported success —
    // the two facts that separate the empty-page-list outcomes from each other.
    let mut resolve_ran = false;
    let mut resolve_ok = false;
    if has_internal {
        let generation_before = m.map_generation;
        let started = std::time::Instant::now();
        let resolved = resolve_mapping_backing(state, host, mapping_id);
        resolve_ran = true;
        resolve_ok = resolved;
        let elapsed_us = started.elapsed().as_micros() as u64;
        let (pages_after, generation_after) = state
            .mappings
            .get(&mapping_id)
            .map(|entry| (entry.page_entries.len(), entry.map_generation))
            .unwrap_or((0, generation_before));
        if revalidate_timing_is_slow(elapsed_us) {
            crate::observe::off(format!(
                "map_revalidate_slow mid={mapping_id} us={elapsed_us} pages={pages_after} resolved={} generation={} changed={}",
                resolved as u8,
                generation_after,
                (generation_after != generation_before) as u8
            ));
        }
        if !resolved && had_live_table {
            // Product: table was live and is now unreadable — drop PFNs.
            if had_pages {
                let _ = state.invalidate_mapping_pages(mapping_id);
                crate::observe::fail(format!(
                    "map_revalidate mid={mapping_id} st=invalidate reason=resolve_fail"
                ));
            }
            return Some("revalidate_resolve_fail");
        }
        // No prior live KVA (first resolve miss, or test fixture with manual
        // page_entries only) — fall through to accept non-empty manual list.
    }
    match state.mappings.get(&mapping_id) {
        Some(m) if m.mapped && !m.page_entries.is_empty() => None,
        Some(m) if !m.mapped => Some("revalidate_unmapped_late"),
        Some(m) if m.condemned_entries.is_some() => Some("revalidate_condemned"),
        Some(_) if !resolve_ran => Some("revalidate_no_internal"),
        Some(_) if !resolve_ok => Some("revalidate_resolve_miss"),
        Some(_) => Some("revalidate_empty_after_resolve"),
        None => Some("revalidate_gone_late"),
    }
}

/// Which of the two `true`s a bare "are these pages still ours" bool would
/// have returned.
///
/// [`type4_pages_witness`]'s contract says a caller must not read a bare
/// "yes" as "these pages were verified", because four of its five exits check
/// nothing at all. Every caller then collapsed both meanings into
/// [`PagesVerdict::Ours`], and the counters built on it — `mapw_pages_vouched` 29 002 against
/// `mapw_pages_refused` **0** on one boot — cannot say whether that zero is a
/// guard that passed or a guard that was never armed. Those are opposite claims
/// about the write-after-free class and the census reported them identically.
///
/// This is the denominator, and it is measure-only: [`Unwitnessed`] still
/// yields a token and still lets the write through, exactly as before. Nothing
/// here changes policy; it changes what the boot can say about it.
///
/// [`Unwitnessed`]: Type4Witness::Unwitnessed
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Type4Witness {
    /// Every page of the cached list was re-walked and agreed. The only exit
    /// that is evidence the list still names the surface's memory.
    Verified,
    /// Nothing was checked. Carries which of the four states it was, because
    /// they are not one outcome: a surface that never latched a walk is a
    /// different gap from one whose walk was superseded.
    Unwitnessed(&'static str),
    /// The re-walk disagreed, or the owning task is gone.
    Drifted,
}

/// Whether a mapping's cached page list still names the guest memory it was
/// walked from, re-derived rather than remembered.
///
/// # Why the existing revalidation cannot answer this
///
/// [`revalidate_mapping_reason`] re-resolves a mapping *only* when
/// `mapping_internal != 0`. A type-4 surface has no MappingInternal, so that
/// function reaches its final `match` with `resolve_ran == false` and returns
/// `None` — "resolvable" — on the strength of `mapped && !page_entries
/// .is_empty()` alone. The list is accepted because it is non-empty, not
/// because anything checked it. Its own doc records the scale: 2 280 render
/// windows in one boot were armed on mappings with `mapping_internal == 0`.
///
/// Nothing on the wire closes that gap. When the guest re-points a type-4
/// surface's backing in its own page table there is no packet, so
/// `map_generation` does not move and the cached entries stay trusted until the
/// next type-4 command happens to re-resolve — which for an idle surface may be
/// never. `resolve_type4_surface_ex` knows this and checks for it, but only
/// there, and only on the first and last entry:
///
/// ```text
/// type4_pages_stale sid=49 task=0 n=256 gpa0=0x2e8cf6000 (task PT translation moved; rebuilding)
/// ```
///
/// That line is from a live boot. The translations do move.
///
/// # What this checks
///
/// Every page, not two. [`crate::model::Type4Walk`] latched the task and GPU-VA
/// base the entries were walked from, so the walk repeats with no object search:
/// page `i` is `(backing_pfn + i) << page_shift` translated through that task.
/// Any page that translates differently — or no longer translates at all — means
/// the list in hand names memory that is no longer the surface's.
///
/// Answers [`Type4Witness::Unwitnessed`] when there is nothing to check
/// (`type4_walk` absent, or latched at a superseded `map_generation`), because
/// this is a *specific* witness and not a general one. That exit is the reason
/// the return type is an enum rather than a bool: it is not evidence, and a
/// caller must not read it as "these pages were verified".
///
/// This asks the question about a mapping's page list, which is what the
/// mapping-keyed rails write through.
///
/// Re-walks the cached page list and reports which exit it took.
pub fn type4_pages_witness<H: HostMemory>(
    state: &DeviceState,
    host: &H,
    mapping_id: u32,
) -> Type4Witness {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Type4Witness::Unwitnessed("no_mapping");
    };
    let Some(walk) = m.type4_walk else {
        // No type-4 walk was ever latched for this mapping, so this witness has
        // never had anything to say about it. The type-11 rail lives here.
        return Type4Witness::Unwitnessed("no_walk");
    };
    if walk.map_generation != m.map_generation {
        // The list has been replaced since the walk was latched. The new list
        // may be perfectly good — it simply has no witness of its own yet.
        return Type4Witness::Unwitnessed("walk_superseded");
    }
    if m.page_entries.is_empty() {
        return Type4Witness::Unwitnessed("no_pages");
    }
    let page_shift = state.page_shift;
    let page_size = crate::contract::iosurface_pages::page_size_of(page_shift);
    // Checked here as well as by the visitor below, which visits nothing for an
    // inactive task: the two answers are the same refusal, and only this one can
    // say *why* without the reader having to know the visitor's early returns.
    if !state.tasks.is_active(walk.task_id) {
        // The task that owned the translation is gone. Its page table is gone
        // with it, so the cached GPAs are unbacked by anything this device can
        // still read — which is exactly the state a write must not proceed in.
        crate::observe::fail(format!(
            "mapping_page_drift mid={mapping_id} task={} reason=task_inactive pages={} \
             (the page table these entries were walked from no longer exists)",
            walk.task_id,
            m.page_entries.len()
        ));
        return Type4Witness::Drifted;
    }
    // One walk over the whole run, not one per page.
    //
    // The pages this checks are consecutive GVAs by construction — entry `i` is
    // `backing_pfn + i` — so every one of them resolves through the same upper
    // page-table levels, and a per-page `translate_task_gva` re-reads the task
    // directory and re-descends those levels for each. `visit_task_gva_pages`
    // reads the root once and carries a walk cache across the range, which is
    // what it exists for.
    //
    // The cost this removes is not incidental. It is `O(pages)` guest memory
    // reads on the licence check every write to a mapping takes, so a 1080p
    // surface pays it 2025 times per flush; a driven window-drag boot measured
    // `readback_split vouch_us` at 221 µs a flush over 11 854 flushes, which is
    // 2.6 s of the boot spent re-deriving an answer from the same three
    // page-table pages.
    //
    // Same walk and same comparisons — this is the identical page-table read
    // with the redundant descents removed, not a weaker check. A short visit is
    // a refusal, because a walk that stops early has proved nothing about the
    // pages it did not reach.
    let entries = &m.page_entries;
    let base_gva = (walk.backing_pfn as u64) << page_shift;
    let span = (entries.len() as u64).saturating_mul(page_size);
    let mut i = 0usize;
    let mut verdict = None;
    crate::runtime::gva_mem::visit_task_gva_pages_in_order(
        host,
        &state.tasks,
        walk.task_id,
        base_gva,
        span,
        page_shift,
        &mut |walked| {
            let Some(&entry) = entries.get(i) else {
                return false;
            };
            let gva = base_gva + (i as u64) * page_size;
            let cached = crate::contract::iosurface_pages::entry_gpa_shift(entry, page_shift);
            let Some(live) = walked else {
                // No translation now. This used to answer the failed walk with
                // the GVA, to match the identity fallback that produced the
                // entry, and that mirror had to go with it: `apply_type4_backing`
                // no longer adopts a page at its own GVA, so the only entry this
                // could still accept is one the control arm made.
                //
                // The distinction is not cosmetic. Reporting the substitute as
                // `live` made a walk that FAILED indistinguishable from one that
                // succeeded and disagreed, so every such page was written up as
                // `translation_moved` — "the guest re-pointed this surface" —
                // when nothing had moved and the device simply could not
                // translate it. Both outcomes refuse the write; only one of them
                // is about the guest.
                crate::observe::fail(format!(
                    "mapping_page_drift mid={mapping_id} task={} page={i}/{} gva={gva:#x} \
                     cached={cached:?} live=None reason=no_translation \
                     (this task cannot translate the page the entry was walked from)",
                    walk.task_id,
                    entries.len()
                ));
                verdict = Some(Type4Witness::Drifted);
                return false;
            };
            if cached != Some(live) {
                // Every entry in this list came from a walk that succeeded:
                // `apply_type4_backing` refuses the whole surface on the first
                // page it cannot walk, so it cannot leave a fabricated one
                // behind. A disagreement here is therefore between two real
                // translations taken at different times, which is the guest
                // having re-pointed the surface without saying so.
                crate::observe::fail(format!(
                    "mapping_page_drift mid={mapping_id} task={} page={i}/{} gva={gva:#x} \
                     cached={cached:?} live={:?} reason=translation_moved \
                     (the guest re-pointed this surface and no packet said so)",
                    walk.task_id,
                    entries.len(),
                    Some(live)
                ));
                verdict = Some(Type4Witness::Drifted);
                return false;
            }
            i += 1;
            true
        },
    );
    if let Some(verdict) = verdict {
        return verdict;
    }
    if i != entries.len() {
        // The visitor stopped before the list did — an inactive task, a
        // directory that has gone, or a page geometry it cannot walk. Each of
        // those is the state the per-page loop reported as `task_inactive` or as
        // a failed translation, and all of them mean the same thing here: the
        // pages this write was about to land in are not provably the mapping's.
        crate::observe::fail(format!(
            "mapping_page_drift mid={mapping_id} task={} reason=walk_short \
             pages={i}/{} (the page table these entries were walked from cannot \
             be walked now)",
            walk.task_id,
            entries.len()
        ));
        return Type4Witness::Drifted;
    }
    Type4Witness::Verified
}

/// Proof that a mapping's cached page list named the guest memory it was walked
/// from at the moment this was taken, carried by value so a writer cannot reach
/// guest RAM without holding one.
///
/// # Why a token and not a call
///
/// [`type4_pages_witness`] existed for a release with exactly one caller — the
/// render writeback — while every other write
/// through `MappingEntry::page_entries` went unchecked. That is not an oversight
/// that a second call site fixes: the check has to be *reachable only through*
/// the write, or the next rail added to this crate arrives unguarded too, and
/// nothing in review distinguishes it from the guarded ones.
///
/// So [`write_mapping_bytes`] and the contig write view both demand one of
/// these, and the four public writers in [`crate::runtime::mapping_write`] take
/// it once at the head of the operation. "Once" is the other half of the design:
/// the walk is a translation per page, and two of those writers call
/// `write_mapping_bytes` per row, so checking inside the funnel would have cost
/// rows x pages per frame — 2.2 M translations for a 1080p surface. Hoisting the
/// proof to the operation and *presenting* it in the funnel keeps the guarantee
/// without the quadratic.
///
/// # Why it carries the generation
///
/// Six sites clear or replace `page_entries` and all six bump `map_generation`.
/// A token minted before one of them names a list that no longer exists, so it
/// records the generation it was taken at and [`PagesVouched::covers`] re-checks
/// it at the point of use. A carried-over token is then unusable by
/// construction rather than by every future writer remembering a second field —
/// the same rule [`crate::model::Type4Walk`] states for its own latch.
#[derive(Clone, Copy, Debug)]
pub struct PagesVouched {
    mapping_id: u32,
    map_generation: u32,
}

impl PagesVouched {
    /// Whether this proof is about `mapping_id`'s page list *as it is now*.
    ///
    /// False once anything has cleared or replaced the list since the walk, so a
    /// funnel that takes a token still refuses when the flush it performs first
    /// invalidates the mapping underneath it.
    pub fn covers(&self, state: &DeviceState, mapping_id: u32) -> bool {
        self.mapping_id == mapping_id
            && state
                .mappings
                .get(&mapping_id)
                .is_some_and(|m| m.map_generation == self.map_generation)
    }
}

/// Re-walk a mapping's page list and mint the proof a write needs, or refuse.
///
/// `None` means the list in hand does not name the surface's memory any more.
/// The response is not to skip this one write: `page_entries` is what every
/// later reader and writer resolves through, so the list is invalidated, which
/// clears it, bumps `map_generation` and retires the contiguous view and the
/// guest-write token with it. Every window still armed against the old
/// incarnation then refuses on the `map_generation` check it already had, and
/// the next type-4 bind re-resolves the surface from the object list.
///
/// Returning a token when there is nothing to check is deliberate and is why
/// [`type4_pages_witness`] reports "nothing to check" separately rather than
/// "verified": a mapping with no [`crate::model::Type4Walk`] latch has a page
/// list this witness cannot speak about, and refusing every write to it would
/// blank surfaces the device has no evidence against.
/// The verdict is handed back alongside the token, not folded into it, because
/// `Unwitnessed` and `Ours` both yield a token and only one of them is a clean
/// answer. A caller that emits counters needs to tell them apart: folding them
/// together would make a boot that never armed the guard indistinguishable from
/// one with no drift — a count that reads as success when it is the opposite.
pub fn vouch_mapping_pages_verdict<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> (PagesVerdict, Option<PagesVouched>) {
    let verdict = mapping_pages_verdict(state, host, mapping_id);
    if verdict == PagesVerdict::Drifted {
        return (verdict, None);
    }
    let token = state.mappings.get(&mapping_id).map(|m| PagesVouched {
        mapping_id,
        map_generation: m.map_generation,
    });
    (verdict, token)
}

/// What the re-walk found, kept separate from what the device does about it.
///
/// Two rails ask this question — the deferred render flush and the four direct
/// writers — and they want different counters but must not want different
/// *policy*.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PagesVerdict {
    /// The re-walk agreed with the cached list, every page of it.
    Ours,
    /// Nothing was checked; the payload is which of the four states it was, per
    /// [`Type4Witness::Unwitnessed`]. **The write proceeds exactly as it does
    /// for [`Self::Ours`]** — this variant changes no policy. It exists because
    /// `Ours` used to mean both "verified" and "unverifiable", so a boot
    /// reporting `mapw_pages_refused = 0` could not say whether its guard had
    /// passed or had simply never been armed, and those are opposite claims
    /// about the write-after-free class.
    Unwitnessed(&'static str),
    /// The re-walk disagreed. The list has been invalidated — refusing this one
    /// write is not enough, because `page_entries` is what every later reader and
    /// writer resolves through.
    Drifted,
}

/// The single decision both mapping-keyed rails take, so they cannot drift apart
/// in policy or in what the control knob turns off.
///
/// An earlier shape had the deferred flush spell this out inline while the write
/// rails had no gate at all, which would have made a control boot measure only
/// half the change — the kind of divergence that makes an A/B report the rig
/// rather than the code.
pub fn mapping_pages_verdict<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &H,
    mapping_id: u32,
) -> PagesVerdict {
    match type4_pages_witness(state, host, mapping_id) {
        Type4Witness::Verified => return PagesVerdict::Ours,
        // Same policy as `Verified` — the write goes through — and a different
        // counter, because only one of the two is evidence.
        Type4Witness::Unwitnessed(why) => return PagesVerdict::Unwitnessed(why),
        Type4Witness::Drifted => {}
    }
    state.invalidate_mapping_pages(mapping_id);
    PagesVerdict::Drifted
}

/// Report a page-table revalidation that took at least a millisecond.
///
/// A pure observability gate — nothing branches on it, so the value costs
/// nothing but log volume if it is wrong in either direction. One millisecond
/// is the frame budget's own scale: at 60 Hz a frame is 16.7 ms, so a single
/// revalidation spending 6 % of it is worth a line, and anything shorter is
/// noise against a rail that runs per surface per frame.
const REVALIDATE_SLOW_US: u64 = 1_000;

#[inline]
fn revalidate_timing_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= REVALIDATE_SLOW_US
}

/// Release contiguous views whose page tables changed.
///
/// A GPU object can retain a view only when [`HostOps::map_pages_stable`]
/// promises the address until explicit retirement. A transient view is never
/// admitted to a backend import, so its only users are CPU copies that finish
/// inside their own call.
pub fn flush_retired_views<H: HostOps>(state: &mut DeviceState, host: &mut H) {
    // The backend allocation aliases the host view, so revoke the GPU parent
    // first. Existing child images and recorded buffers hold it through their
    // fence-safe retirement; only then is the matching host view unmapped.
    #[cfg(feature = "backend-vulkan")]
    let backend_owned: std::collections::HashSet<_> = state
        .retired_guest_imports
        .drain(..)
        .filter_map(crate::backend::vulkan::engine::retire_guest_import)
        .collect();
    #[cfg(not(feature = "backend-vulkan"))]
    state.retired_guest_imports.clear();

    #[cfg(feature = "backend-vulkan")]
    let mut released: std::collections::HashSet<_> =
        crate::backend::vulkan::engine::take_released_host_aliases()
            .into_iter()
            .collect();
    for (ptr, len) in state.retired_views.drain(..) {
        #[cfg(feature = "backend-vulkan")]
        if backend_owned.contains(&(ptr, len)) {
            continue;
        }
        #[cfg(feature = "backend-vulkan")]
        released.remove(&(ptr, len));
        host.unmap_pages(ptr, len);
    }
    #[cfg(feature = "backend-vulkan")]
    for (ptr, len) in released {
        host.unmap_pages(ptr, len);
    }
    // Same shape and the same reason: a guest-write token is host-side state
    // for a page list that no longer exists, and only the host can free it.
    for token in state.retired_guest_write_tokens.drain(..) {
        host.untrack_guest_writes(token);
    }
}

/// Return Vulkan aliases whose terminal fence-safe destruction has completed
/// to the host. Called from the device heartbeat so release does not depend on
/// another guest mapping event arriving.
#[cfg(feature = "backend-vulkan")]
pub fn drain_deferred_unmaps<H: HostOps>(host: &mut H) -> usize {
    let released = crate::backend::vulkan::engine::take_released_host_aliases();
    let count = released.len();
    for (ptr, len) in released {
        host.unmap_pages(ptr, len);
    }
    count
}

/// The live guest-write token for this mapping's current page list, asking the
/// host for one if the list has none.
///
/// Registration is what makes the host observe writes to these pages at all,
/// so it happens where the device first cares — at the Store that publishes
/// the surface — rather than at every mapping resolve. A host that cannot
/// observe guest writes answers `None` here forever, and every consumer reads
/// that as "assume written".
///
/// Reads `page_entries` directly instead of going through
/// [`mapping_page_gpas`]: the revalidation that function performs is a guest
/// page-table walk, and the caller has already proved the mapping resolvable by
/// rendering into it.
///
/// What keys the token to the surface's *current* pages is
/// [`crate::model::MappingEntry::map_generation`], not the eager retirement in
/// the lifecycle mutators. Two writers replace `page_entries` in place without
/// going near those mutators — the mapper's own plan adoption and the type-4
/// page refresh — and both retired the contiguous view while leaving the token
/// alone. Both do bump the generation exactly when the list changes, so
/// checking it here makes a carried-over token unusable by construction instead
/// of depending on every future writer remembering a second thing to retire.
pub fn ensure_guest_write_token<H: HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<u64> {
    let page_shift = state.page_shift;
    let page_size = state.page_size() as usize;
    let m = state.mappings.get(&mapping_id)?;
    let map_generation = m.map_generation;
    if m.guest_write_token != 0 {
        if m.guest_write_token_gen == map_generation {
            return Some(m.guest_write_token);
        }
        // The list moved underneath the token. Everything recorded against it
        // describes pages this surface may no longer own, so the Store stamp
        // goes with it.
        //
        // Counted because retiring a token reopens its two-harvest arming
        // window, and everything downstream of an unarmed token falls back to
        // the whole-surface answer: the type-11 LOAD elision refuses with
        // `t11_gw_ref_no_stamp` and the draw pays a full-frame seed read plus a
        // full-frame staging upload. `t11_gw_unarmed` says the window was open;
        // this says whether it keeps being reopened, which separates "the token
        // is warming up once" from "the page list churns and it never warms up".
        crate::runtime::drain::note_store_route("gw_token_retired");
        let e = state.mappings.get_mut(&mapping_id)?;
        let stale = std::mem::replace(&mut e.guest_write_token, 0);
        e.guest_write_token_gen = 0;
        e.guest_write_gen_at_store = 0;
        state.retired_guest_write_tokens.push(stale);
    }
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let gpas: Vec<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .collect();
    // A partial list would have the host watch some of the surface and report
    // "unwritten" for the rest, which is the one answer that must never be
    // invented.
    if gpas.len() != m.page_entries.len() {
        return None;
    }
    let token = host.track_guest_writes(&gpas, page_size)?;
    let e = state.mappings.get_mut(&mapping_id)?;
    e.guest_write_token = token;
    e.guest_write_token_gen = map_generation;
    Some(token)
}

/// Record the host's guest-write generation for a mapping whose pixels a Store
/// has just published, registering its pages for tracking the first time.
///
/// Called from every rail that publishes a type-11 surface: the Vulkan Store
/// rails next to their `surface_content_epoch` stamp, and the CPU writer in
/// [`crate::runtime::mapping_write`] next to the host-cache store it makes
/// authoritative. Lives here rather than in the Vulkan draw path because
/// `mapping_write` is backend-agnostic and the witness is not a backend concern.
///
/// Historically only the Vulkan Store rails armed it, which is why the type-4
/// sampled ladder's first census read `t11rung_host_cache_gw_no_stamp` 14 092
/// against `gw_clean` 0: the copy that rung serves is written here, and nothing
/// here had ever asked the host to watch the pages it is a copy of.
///
/// Next to that rail's
/// `surface_content_epoch` stamp. The two are halves of one witness — the epoch
/// covers writers inside this crate, this covers the guest CPU — and a rail that
/// wrote only the epoch would let the elision vouch for a resident on evidence
/// that cannot see the surface's owner. The resident-store rail did exactly that
/// and it was the whole of the rail's traffic: one boot measured
/// `surface_resident` ~210/s against zero calls through the readback rails.
///
/// 0 is written for every unknown, and it is never a live generation (the host's
/// first readable one is 1), so a surface whose host cannot answer fails the
/// currency test instead of passing it by default.
pub fn stamp_guest_write_gen<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    mapping_id: u32,
) {
    let gen_ = match crate::runtime::mapper::ensure_guest_write_token(state, host, mapping_id) {
        None => {
            // The host cannot watch these pages: no dirty bitmap, or the mapping
            // has no page list to name. Counted, because a rail whose
            // registration silently never happens is indistinguishable from a
            // guest that writes every frame, and the two want opposite fixes.
            crate::runtime::drain::note_store_route("t11_gw_untracked");
            0
        }
        Some(token) => match host.guest_write_gen(token) {
            // The host has the pages but cannot vouch for them yet: its report
            // only becomes a fact about the guest once logging has been on for a
            // full interval.
            None => {
                crate::runtime::drain::note_store_route("t11_gw_unarmed");
                0
            }
            Some(gen_) => {
                crate::runtime::drain::note_store_route("t11_gw_armed");
                gen_
            }
        },
    };
    if let Some(m) = state.mappings.get_mut(&mapping_id) {
        m.guest_write_gen_at_store = gen_;
    }
}

/// What the hypervisor's dirty bitmap can say about a type-4 surface's pages
/// since the Store that stamped them.
///
/// Every variant but [`Self::Clean`] means "assume written". They are kept
/// apart because "this rail never got started" and "the guest rewrites this
/// surface every frame" are the same refusal and completely different findings.
#[cfg(feature = "backend-vulkan")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestWriteVerdict {
    /// The host has observed no write to these pages since the stamp.
    Clean,
    /// No mapping under this id.
    NoMapping,
    /// No Store has stamped this surface against a live token for its
    /// *current* page list.
    NoStamp,
    /// The host observed a write.
    Wrote,
    /// The host cannot answer for this token.
    Unreadable,
}

/// The verdict itself, with no counters attached.
///
/// Split from the type-11 LOAD gate's `type11_guest_wrote_since_store` so more
/// than one rail can ask
/// the same question and report it under its own names. A shared counter would
/// pool two rails' refusals into one number, and the number would then be
/// unreadable for either.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn mapping_guest_write_verdict<M: HostOps>(
    state: &DeviceState,
    host: &M,
    mapping_id: u32,
) -> GuestWriteVerdict {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return GuestWriteVerdict::NoMapping;
    };
    // The dominant disjunct is the first, and the reason is the shim's arming
    // window rather than anything about this mapping. `reims_vgpu_dirty_track`
    // sets `arm_at = harvests + 1`, and `reims_vgpu_dirty_harvest` pins
    // `s->gen = 0` until `harvests >= arm_at`; `HostOps::guest_write_gen` maps
    // that 0 to `None`, so `stamp_guest_write_gen` records 0 and counts
    // `t11_gw_unarmed`. Every Store landing inside a token's arming window
    // therefore stamps 0, and every LOAD against that stamp lands here.
    //
    // That window is per *token*, not per boot, and it is entered two ways. A
    // brand new mapping enters it once. `ensure_guest_write_token` also retires
    // a token and builds a new one whenever `guest_write_token_gen` has fallen
    // behind `map_generation`, so a mapping whose page list churns re-enters it
    // each time — counted as `gw_token_retired`, and measured at **0 per second
    // across a driven x86/PCI boot**, so on this pathway churn is not how the
    // window is reached and new surfaces are.
    //
    // It is the structural explanation for `t11_gw_ref_no_stamp` running far
    // ahead of `t11_gw_ref_moved` — the refusals are this device's own startup
    // cost, repeated, not guest writes. What it costs is measured: the window
    // is counted in harvests, harvests are driven by guest doorbells, and a
    // draw that lands in it pays a whole-frame seed read plus a whole-frame
    // staging upload. On a near-idle desktop `chain_phase` reads 12-65 ms per
    // draw with `seed_us` and the engine's `stage_us` holding it, against
    // 0.2 ms per draw driven, which is the hitch class goals 5 and 6 name.
    if m.guest_write_gen_at_store == 0
        // Subsumed by the stamp test above rather than independent: every writer
        // that zeroes the token zeroes the stamp in the same breath
        // (`DeviceState::take_guest_write_token`, and the stale-token path in
        // `ensure_guest_write_token`), and a default `MappingEntry` starts with
        // both at 0. Kept because it is the cheap half of a check whose false
        // "unwritten" is the one answer that produces a wrong frame, so a future
        // writer that clears only the token must not be able to slip past.
        || m.guest_write_token == 0
        // A token built for a different page list watches pages this surface
        // may no longer own, so its generation is not a statement about the
        // pages the resident would be reused for. Checked here and not only in
        // `ensure_guest_write_token` because a LOAD can arrive between the list
        // changing and the next Store rebuilding the token.
        || m.guest_write_token_gen != m.map_generation
    {
        return GuestWriteVerdict::NoStamp;
    }
    match host.guest_write_gen(m.guest_write_token) {
        Some(gen_) if gen_ == m.guest_write_gen_at_store => GuestWriteVerdict::Clean,
        Some(_) => GuestWriteVerdict::Wrote,
        None => GuestWriteVerdict::Unreadable,
    }
}

/// Mapping byte ranges covering the pages of `mapping_id` whose GPAs appear in
/// `written`, ascending and merged.
///
/// The dirty bitmap answers in guest *physical* pages; a writeback lays bytes
/// out in the mapping's own offset space. This is the one place the two meet,
/// and it goes through `page_entries` — the same list the tracking token was
/// built from — so page `i` of the surface is offset `i * page_size` by
/// construction rather than by an address arithmetic that could drift from it.
///
/// A GPA the mapping does not hold is ignored: a token is per page list, but a
/// caller may hand back an answer taken before a rebind, and inventing an offset
/// for a page this surface does not own would exclude bytes at random.
///
/// Adjacent pages merge, so a guest that rewrites a whole surface produces one
/// range rather than thousands.
pub fn mapping_offsets_of_pages(
    state: &DeviceState,
    mapping_id: u32,
    written: &[u64],
) -> Vec<(u64, u64)> {
    let Some(m) = state.mappings.get(&mapping_id) else {
        return Vec::new();
    };
    if written.is_empty() {
        return Vec::new();
    }
    let page_shift = state.page_shift;
    let page_size = 1u64 << page_shift;
    let mut sorted: Vec<u64> = written.to_vec();
    sorted.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (i, &entry) in m.page_entries.iter().enumerate() {
        let Some(gpa) = crate::contract::iosurface_pages::entry_gpa_shift(entry, page_shift) else {
            continue;
        };
        if sorted.binary_search(&gpa).is_err() {
            continue;
        }
        let lo = (i as u64).saturating_mul(page_size);
        let hi = lo.saturating_add(page_size);
        match out.last_mut() {
            Some(last) if last.1 == lo => last.1 = hi,
            _ => out.push((lo, hi)),
        }
    }
    out
}

/// Revalidate + collect page-aligned GPAs for a mapped surface (GVA order).
///
/// Fails closed on empty / invalid entries and known transport/control-page
/// aliases. Does not invent PFNs. Every consumer immediately passes the
/// returned GPAs to `HostOps::map_pages`, whose host callback is the
/// authoritative RAM/range validator; repeating `is_ram_gpa` once per page
/// here makes full-frame surfaces perform thousands of duplicate QEMU address
/// translations before the exact same validation in `map_pages`.
pub fn mapping_page_gpas<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<Vec<u64>> {
    if !{ revalidate_mapping_pages(state, host, mapping_id) } {
        return None;
    }
    let m = state.mappings.get(&mapping_id)?;
    if !m.mapped || m.page_entries.is_empty() {
        return None;
    }
    let page_shift = state.page_shift;
    let gpas: Vec<u64> = m
        .page_entries
        .iter()
        .filter_map(|&e| crate::contract::iosurface_pages::entry_gpa_shift(e, page_shift))
        .collect();
    if gpas.is_empty() || gpas.len() != m.page_entries.len() {
        return None;
    }
    if let Some((gpa, owner)) = { first_control_page_collision(state, &gpas) } {
        crate::observe::fail(format!(
            "mapping_pages fail reason=control_page_collision mid={mapping_id} gpa={gpa:#x} owner={owner} pages={}",
            gpas.len()
        ));
        return None;
    }
    Some(gpas)
}

/// A render surface must never alias pages that the device knows are live
/// transport or task-control structures. `is_ram_gpa` alone cannot distinguish
/// an IOSurface page from a FIFO/page-table page; reject the provable overlap
/// before either CPU or GPU writes touch it.
///
/// Every region compared here is guest-**physical** by construction, which is
/// what makes the comparison mean anything: `gfx.root_page`, `gfx.fifo_base_page`
/// and `directory_pfn` are PFNs the guest writes and every other consumer turns
/// into a GPA with `pfn_gpa`/`pfn_to_gpa` before a physical read, while
/// `iosfc.ring_base` and `child_rings[..].page_gpas` are already GPAs and are
/// read raw.
///
/// **A task's object list is not, and so cannot be checked here.**
/// `object_list_pfn << page_shift` is an address in that task's *virtual* space:
/// [`crate::runtime::objects::lookup_list_entry`] builds it, names it
/// `entry_gva` and reads it through `gva_mem::read_task_gva_by_id`, and
/// `gva_mem`'s own doc states the rule — "a GVA has no meaning apart from the
/// page table it is resolved against". Testing it against surface GPAs compares
/// two different address spaces, so it can only ever produce a coincidence, and
/// the coincidence is not remote: tasks put their object lists in low pages.
/// That arm therefore rejected a legitimate surface — losing real guest work —
/// on a numeric collision, while a genuine alias stayed invisible to it. It also
/// strided the span at 16 bytes where the contract's `OBJECT_LIST_ENTRY_LEN` is
/// 12, oversizing the window it got wrong by a third.
///
/// Making it meaningful would mean walking the task's page table over the whole
/// object-list span on every call, which is the cost this function's range-query
/// shape exists to avoid, to enforce a rule the protocol never states and that
/// no boot has ever seen violated. Do not re-add it without those GPAs in hand.
fn first_control_page_collision(state: &DeviceState, gpas: &[u64]) -> Option<(u64, &'static str)> {
    let page = state.page_size();
    let page_base = |gpa: u64| gpa & !(page - 1);
    // Probe the SURFACE, not the control structures. A live task can advertise a
    // million object-list slots — 4,096 x86 pages — and an object list is one
    // contiguous span, so asking "does the surface hold this page?" once per
    // control page enumerated the whole span page by page. Sorted, the same
    // question is one range query per task.
    //
    // Measured on the x86/Vulkan rail before this shape: 414 µs per call, 71 024
    // calls in a 120 s arm, 29.4 s of wall clock spent proving a full-screen
    // IOSurface does not alias a FIFO ring.
    let mut pages: Vec<u64> = gpas.iter().map(|&gpa| page_base(gpa)).collect();
    pages.sort_unstable();
    let holds = |gpa: u64| pages.binary_search(&page_base(gpa)).is_ok();
    // The lowest surface page in `[start, end)`, if any. `start` is page-aligned
    // and every entry is a page base, so a hit is exactly one of the pages a
    // per-page enumeration would have reported — same page, same reported gpa.
    let holds_range = |start: u64, len: u64| -> Option<u64> {
        let end = start.saturating_add(len);
        let i = pages.partition_point(|&p| p < start);
        pages.get(i).copied().filter(|&p| p < end)
    };

    if state.gfx.root_page != 0 && holds((state.gfx.root_page as u64) << state.page_shift) {
        return Some(((state.gfx.root_page as u64) << state.page_shift, "gfx_root"));
    }
    // The main FIFO is as long as the guest said it is, not one page. The ring
    // spans `fifo_length` bytes from the base page — `drain_main_fifo` reads it
    // over exactly that extent, `fifo_start` being the header offset *inside*
    // it — and it is routinely more than one page, so probing only the first
    // let a surface alias the rest of the ring undetected.
    if state.gfx.fifo_base_page != 0 {
        let base = (state.gfx.fifo_base_page as u64) << state.page_shift;
        if let Some(gpa) = holds_range(base, state.gfx.fifo_length.max(1) as u64) {
            return Some((gpa, "root_fifo"));
        }
    }
    // Still the first page only. `iosfc.capacity` would give the ring's extent
    // the way `fifo_length` does above, but nothing in this crate consumes it —
    // it is written and read back over MMIO and never bounds anything — so its
    // units are not established, and sizing a rejection window from a field
    // whose meaning is a guess is how a legitimate surface gets refused. Bound
    // it when a consumer settles whether it counts entries or bytes.
    if state.iosfc.ring_base != 0 && holds(state.iosfc.ring_base) {
        return Some((page_base(state.iosfc.ring_base), "iosfc_ring"));
    }
    for ring in &state.child_rings {
        for &gpa in &ring.page_gpas {
            if holds(gpa) {
                return Some((page_base(gpa), "child_fifo"));
            }
        }
    }
    for (_, task) in state.tasks.live() {
        if task.directory_pfn != 0 {
            let gpa = (task.directory_pfn as u64) << state.page_shift;
            if holds(gpa) {
                return Some((gpa, "task_directory"));
            }
        }
    }
    None
}

/// Device-wide count of [`ensure_contig_view`] calls the host refused,
/// whether the verdict was derived or served from `contig_refused_gen`.
/// Reported as `served=` on every `contig_view_refused` line so the
/// magnitude the old per-call line carried survives its deduplication.
static CONTIG_REFUSED_SERVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Contiguous host-VA view over the mapping's guest pages (unified memory).
///
/// Builds the view on first use via [`HostOps::map_pages`]. The host may return
/// a direct run or reconstruct a scattered page list as one shared virtual
/// alias; either answer is the same packed byte sequence to the caller.
/// Returns `(ptr, len)`.
///
/// **Safe zero-copy contract:** always [`revalidate_mapping_pages`] first so a
/// cached contig never aliases PFNs after ReplacePhysical / guest recycle.
///
/// A refusal is cached on the page-list generation. Whether scattered pages
/// can be packed is a host capability: pre-rejecting them here would make a
/// shared file-backed alias unreachable and force the scatter paths even when
/// the host can express the exact view.
pub fn ensure_contig_view<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<(usize, usize)> {
    ensure_contig_view_with_pages(state, host, mapping_id).map(|(ptr, len, _)| (ptr, len))
}

/// [`ensure_contig_view`] plus the guest-physical footprint owned by the view.
///
/// The footprint is retained with an imported GPU resource so synchronizing
/// that resource never has to reconstruct its backing from a mapping id.
pub fn ensure_contig_view_with_pages<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<(usize, usize, std::sync::Arc<[u64]>)> {
    // Always revalidate before returning a cached contig (ReplacePhysical /
    // recycle must not leave a live view over freelist PFNs).
    if !revalidate_mapping_pages(state, host, mapping_id) {
        return None;
    }
    flush_retired_views(state, host);
    {
        let m = state.mappings.get(&mapping_id)?;
        if m.contig_ptr != 0 {
            let Some(footprint) = &m.contig_footprint else {
                return None;
            };
            return Some((m.contig_ptr, m.contig_len, footprint.pages_arc()));
        }
        // The negative verdict caches on exactly the key that makes the
        // positive one above safe. Re-deriving it per call collected the page
        // GPAs and rescanned them every time, and said so in the always-on sink
        // every time: 471 757 lines in one 2 900 s boot, the sole prefix ever to
        // trip `log_flood_detected`, at up to 1 826 lines in a one-second window.
        if m.contig_refused_gen == Some(m.map_generation) {
            CONTIG_REFUSED_SERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }
    let gpas = mapping_page_gpas(state, host, mapping_id)?;
    let page_sz = crate::contract::iosurface_pages::page_size_of(state.page_shift) as usize;
    let physical_runs = reims_vgpu_paging::runs::contig_run_count(&gpas, page_sz as u64);
    let Some(ptr) = host.map_pages(&gpas, page_sz) else {
        let served = CONTIG_REFUSED_SERVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let m = state.mappings.get_mut(&mapping_id)?;
        m.contig_refused_gen = Some(m.map_generation);
        let generation = m.map_generation;
        // One line per (mapping, page list) rather than per call. Physical run
        // count remains diagnosis: it distinguishes a host that declined even
        // a direct run from one that cannot reconstruct a scattered list.
        crate::observe::off(format!(
            "contig_view_refused mid={mapping_id} pages={} physical_runs={physical_runs} generation={generation} served={served}",
            gpas.len(),
        ));
        return None;
    };
    let len = gpas.len() * page_sz;
    let gpas: std::sync::Arc<[u64]> = gpas.into();
    let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(
        std::sync::Arc::clone(&gpas),
        page_sz as u64,
    )?;
    let m = state.mappings.get_mut(&mapping_id)?;
    m.contig_ptr = ptr;
    m.contig_len = len;
    m.contig_footprint = Some(footprint);
    Some((ptr, len, gpas))
}

/// The mapping's one checked backend import and its physical-page footprint.
///
/// A mapping is the allocation; planes and texture views are offsets inside
/// it. The import therefore follows the mapping lifetime and is reused by all
/// of those views. Hosts whose page aliases are transient or backends that did
/// not publish host-pointer import limits retain the copy-backed paths.
#[cfg(feature = "backend-vulkan")]
pub fn ensure_contig_import_with_footprint<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
) -> Option<(
    std::sync::Arc<crate::runtime::guest_ram::GuestRamImport>,
    crate::runtime::guest_ram::GuestPageFootprint,
)> {
    if !host.map_pages_stable() {
        return None;
    }
    let (ptr, len, _pages) = ensure_contig_view_with_pages(state, host, mapping_id)?;
    let footprint = state.mappings.get(&mapping_id)?.contig_footprint.clone()?;
    let len = u64::try_from(len).ok()?;
    // The one admission rule, which asks the map's standing refusal before the
    // latches. This site used to ask the three latches directly and so kept
    // importing on a host whose whole RAMBlock map had been refused.
    let align = crate::runtime::guest_ram_map::packed_alias_import_align(host, len)?;
    if let Some(import) = state
        .mappings
        .get(&mapping_id)
        .and_then(|mapping| mapping.contig_import.as_ref())
    {
        if import.host_base() == ptr && import.len() == len && import.align() == align {
            return Some((std::sync::Arc::clone(import), footprint));
        }
    }
    let import = std::sync::Arc::new(
        crate::runtime::guest_ram::GuestRamImport::new_host_allocation(ptr, len, align).ok()?,
    );
    state.mappings.get_mut(&mapping_id)?.contig_import = Some(std::sync::Arc::clone(&import));
    Some((import, footprint))
}

/// Record the guest frames a mapping-rail write of `[off, off+len)` lands in.
///
/// Resolved through the mapping's own page list rather than over the span's
/// hull, because that list is a scatter: a surface's pages are wherever the
/// guest allocator put them, and a hull would claim every frame in between —
/// memory belonging to someone else, every one of which would then read as a
/// hit for the rest of the boot.
///
/// Each page contributes only its intersection with the byte range, so a write
/// of one row into a 16 KiB arm64 page marks the frame that row is in and not
/// the other three.
pub(crate) fn note_mapping_write_footprint(
    state: &DeviceState,
    mapping_id: u32,
    off: u64,
    len: u64,
) {
    if len == 0 {
        return;
    }
    let Some(m) = state.mappings.get(&mapping_id) else {
        return;
    };
    let page_size = state.page_size();
    let page_shift = state.page_shift;
    note_page_write_footprint(page_size, off, len, |i| {
        m.page_entries
            .get(i)
            .map(|&entry| crate::contract::iosurface_pages::entry_gpa_shift(entry, page_shift))
    });
}

/// Record a write through a retained allocation footprint.
///
/// These are the pages admitted with a guest-backed GPU resource. Consuming
/// them directly keeps Store publication tied to the allocation that actually
/// rendered, even if mutable mapping state changes after admission.
#[cfg(any(feature = "backend-vulkan", test))]
pub(crate) fn note_physical_page_write_footprint(
    footprint: &crate::runtime::guest_ram::GuestPageFootprint,
    off: u64,
    len: u64,
) {
    footprint.visit_window(off, len, crate::observe::footprint::note_written_range);
}

fn note_page_write_footprint(
    page_size: u64,
    off: u64,
    len: u64,
    mut page_at: impl FnMut(usize) -> Option<Option<u64>>,
) {
    if len == 0 {
        return;
    }
    let end = off.saturating_add(len);
    let first = off / page_size;
    let last = (end - 1) / page_size;
    let mut physical_run: Option<(u64, u64)> = None;
    for i in first..=last {
        let Some(gpa) = page_at(i as usize) else {
            // A short page list is a refusal the caller reports; there is no
            // frame to name for a page the list does not have.
            break;
        };
        let Some(gpa) = gpa else {
            continue;
        };
        let page_lo = i.saturating_mul(page_size);
        let lo = off.max(page_lo);
        let hi = end.min(page_lo.saturating_add(page_size));
        if lo < hi {
            let segment = (gpa + (lo - page_lo), gpa + (hi - page_lo));
            match physical_run {
                Some((start, run_end)) if run_end == segment.0 => {
                    physical_run = Some((start, segment.1));
                }
                Some((start, run_end)) => {
                    crate::observe::footprint::note_written_range(start, run_end - start);
                    physical_run = Some(segment);
                }
                None => physical_run = Some(segment),
            }
        }
    }
    if let Some((start, run_end)) = physical_run {
        crate::observe::footprint::note_written_range(start, run_end - start);
    }
}

/// Which way a guest-page run copy moves bytes.
///
/// The direction is the only thing a write walk and a read walk disagree about,
/// so it is the only thing that is a parameter. Two rails hold such a pair — the
/// mapping rail here and the GVA rail in [`super::gva_view`] — and both split
/// into twin functions before this type existed, with the read side of each
/// drifting from its write side in the same way: fewer named refusals.
/// A rectangle's shape inside the guest span that holds it: `row_count` rows of
/// `row_bytes` useful bytes, each `row_stride` apart.
///
/// The rectangle is the shape every texture copy actually has, and until this
/// existed the only primitive that spoke it was the mapping rail's. A caller
/// holding a rectangle over the GVA rail had one move available — a row — so it
/// re-entered the guest page table `row_count` times to copy one region, and a
/// driven macos-13 boot charged that re-entry 906 ms of a 916 ms
/// texture-to-texture rail at ~7.6 us for a 4 KiB row.
///
/// The gap between `row_bytes` and `row_stride` is the guest's, not the copy's:
/// a texture's row padding holds whatever the guest put there, so a rectangle
/// copy that filled the span contiguously would land the right pixels and
/// destroy it. That is why this is a shape rather than a length.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RectStride {
    row_stride: usize,
    row_bytes: usize,
    row_count: usize,
}

impl RectStride {
    /// A rectangle, or `None` where the shape cannot describe one.
    ///
    /// Every refusal here is a shape this type must not be able to hold, so it
    /// is refused at construction rather than checked at each use: a zero
    /// extent, rows wider than the stride that separates them (which would make
    /// two rows overlap), or a span that does not fit a `usize`. After this,
    /// [`RectStride::span`] is total and the packed-buffer offsets
    /// [`RunCopy::apply`] computes are in range by construction.
    pub(crate) fn new(row_stride: u64, row_bytes: u64, row_count: u64) -> Option<Self> {
        if row_bytes == 0 || row_count == 0 || row_bytes > row_stride {
            return None;
        }
        let span = row_count
            .checked_sub(1)?
            .checked_mul(row_stride)?
            .checked_add(row_bytes)?;
        let packed = row_bytes.checked_mul(row_count)?;
        usize::try_from(span).ok()?;
        usize::try_from(packed).ok()?;
        Some(Self {
            row_stride: row_stride as usize,
            row_bytes: row_bytes as usize,
            row_count: row_count as usize,
        })
    }

    /// Bytes from the first row's first byte to the last row's last byte —
    /// what the guest-side walk must resolve, padding included.
    pub(crate) fn span(&self) -> usize {
        (self.row_count - 1) * self.row_stride + self.row_bytes
    }

    /// Bytes of the caller's side, which holds rows back to back and no padding.
    pub(crate) fn packed(&self) -> usize {
        self.row_count * self.row_bytes
    }
}

pub(crate) enum RunCopy<'a> {
    Write(&'a [u8]),
    Read(&'a mut [u8]),
    /// A packed buffer into a strided guest rectangle.
    WriteRect(&'a [u8], RectStride),
    /// A strided guest rectangle into a packed buffer.
    ReadRect(&'a mut [u8], RectStride),
}

impl<'a> RunCopy<'a> {
    /// A rectangle write, or `None` when `src` is shorter than the rows.
    pub(crate) fn write_rect(src: &'a [u8], rect: RectStride) -> Option<Self> {
        (src.len() >= rect.packed()).then_some(Self::WriteRect(src, rect))
    }

    /// A rectangle read, or `None` when `dst` is shorter than the rows.
    pub(crate) fn read_rect(dst: &'a mut [u8], rect: RectStride) -> Option<Self> {
        (dst.len() >= rect.packed()).then_some(Self::ReadRect(dst, rect))
    }
}

impl RunCopy<'_> {
    /// How much guest span this copy covers.
    ///
    /// For a rectangle this is the **span**, not the buffer: the walk has to
    /// resolve the padding it steps over even though no byte of it moves, and
    /// the run bound in each caller is stated against this length.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Write(buf) => buf.len(),
            Self::Read(buf) => buf.len(),
            Self::WriteRect(_, rect) | Self::ReadRect(_, rect) => rect.span(),
        }
    }

    pub(crate) fn is_write(&self) -> bool {
        matches!(self, Self::Write(_) | Self::WriteRect(..))
    }

    /// Move `n` bytes between the caller's buffer at `buf_off` and the mapped
    /// host range at `host_off`.
    ///
    /// # Safety
    ///
    /// `host_ptr` must be a live mapping of at least `host_off + n` bytes, and
    /// `buf_off + n` must be within [`RunCopy::len`]. The callers check both
    /// against the run's mapped total before calling.
    ///
    /// For the rectangle forms `buf_off` is an offset into the *span*, and this
    /// splits it into the rows it crosses, touching only their `row_bytes` and
    /// stepping over the padding between them. `RectStride::new` has already
    /// made the packed offsets that produces in range: `buf_off + n <= span`
    /// puts every row index below `row_count`.
    pub(crate) unsafe fn apply(
        &mut self,
        host_ptr: usize,
        host_off: usize,
        buf_off: usize,
        n: usize,
    ) {
        match self {
            Self::Write(buf) => unsafe {
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr().add(buf_off),
                    (host_ptr as *mut u8).add(host_off),
                    n,
                );
            },
            Self::Read(buf) => unsafe {
                std::ptr::copy_nonoverlapping(
                    (host_ptr as *const u8).add(host_off),
                    buf.as_mut_ptr().add(buf_off),
                    n,
                );
            },
            Self::WriteRect(buf, rect) => {
                rect.for_each_piece(buf_off, host_off, n, |packed_off, host_at, len| unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf.as_ptr().add(packed_off),
                        (host_ptr as *mut u8).add(host_at),
                        len,
                    );
                });
            }
            Self::ReadRect(buf, rect) => {
                rect.for_each_piece(buf_off, host_off, n, |packed_off, host_at, len| unsafe {
                    std::ptr::copy_nonoverlapping(
                        (host_ptr as *const u8).add(host_at),
                        buf.as_mut_ptr().add(packed_off),
                        len,
                    );
                });
            }
        }
    }
}

impl RectStride {
    /// The `(packed_offset, host_offset, len)` pieces of `[buf_off, buf_off+n)`
    /// that belong to rows, ascending, with the inter-row padding dropped.
    ///
    /// `buf_off` is a span offset and `host_off` the mapped address of that same
    /// span offset, so the two move together and a run boundary falling mid-row
    /// splits the row rather than the rectangle.
    /// A rectangle with no padding is one piece however many rows it has: the
    /// span and the packed buffer are the same bytes in the same order, so
    /// splitting it per row would issue `row_count` memcpys where one describes
    /// the identical move. Full-plane reads are the common shape on the blit
    /// rail, and they are the widest rectangles here.
    fn for_each_piece(
        &self,
        buf_off: usize,
        host_off: usize,
        n: usize,
        mut piece: impl FnMut(usize, usize, usize),
    ) {
        if self.row_bytes == self.row_stride {
            piece(buf_off, host_off, n);
            return;
        }
        let end = buf_off + n;
        let first = buf_off / self.row_stride;
        let last = end.saturating_sub(1) / self.row_stride;
        for row in first..=last {
            let row_lo = row * self.row_stride;
            let lo = buf_off.max(row_lo);
            let hi = end.min(row_lo + self.row_bytes);
            if lo < hi {
                piece(
                    row * self.row_bytes + (lo - row_lo),
                    host_off + (lo - buf_off),
                    hi - lo,
                );
            }
        }
    }
}

/// The parts of `[lo, hi)` a selection covers, ascending.
///
/// `None` selects the window whole. `Some(&[])` selects **nothing**, which is
/// why this is an `Option` rather than "empty means everything": a caller
/// handing over an explicit list of ranges to write has an empty list exactly
/// when there is nothing to write, and reading that as "write it all" would turn
/// the cheapest landing into the most expensive one.
///
/// `ranges` is ascending and disjoint. The first candidate is found by binary
/// search rather than by walking from the front, so a caller carrying many runs
/// costs `O(log n)` per query instead of `O(n)`. Nothing here assumes successive
/// queries ascend, so a caller may probe windows in any order.
pub(crate) fn selected_within(
    ranges: Option<&[(u64, u64)]>,
    lo: u64,
    hi: u64,
) -> impl Iterator<Item = (u64, u64)> + '_ {
    let whole = ranges.is_none();
    let list: &[(u64, u64)] = ranges.unwrap_or(&[]);
    let start = list.partition_point(|&(_, e)| e <= lo);
    std::iter::once((lo, hi))
        .filter(move |_| whole && lo < hi)
        .chain(
            list[start..]
                .iter()
                .take_while(move |&&(s, _)| s < hi)
                .filter_map(move |&(s, e)| {
                    let a = s.max(lo);
                    let b = e.min(hi);
                    (a < b).then_some((a, b))
                }),
        )
}

/// Copy `[off, off+len)` between a caller buffer and the mapping's guest pages.
///
/// One packed contig view when the mapping has one that covers the range;
/// otherwise the page list is split into maximal packed runs, each mapped,
/// copied and unmapped in turn.
///
/// **This is the only implementation of that walk.** The
/// `buf_off`/`host_off`/`n` arithmetic and the bounds check that guards it used
/// to exist once per direction. One of those directions is the guest-corruption
/// rail, so a bound corrected on the read copy and not the write copy would
/// have been silent — and the two are otherwise identical.
///
/// `only` narrows what is moved without narrowing what is *resolved*: the page
/// list, its packed runs and the imports are worked out for `[off, off+len)`
/// whole, and each mapped run then moves only the parts of itself the selection
/// names. That is the whole reason it is a parameter rather than a loop in the
/// caller — resolving the page list per run costs `O(pages)` each time, so a
/// caller looping over a thousand runs pays it a thousand times.
///
/// `site` names the caller in the refusal lines. Every failure here loses the
/// caller's bytes, so all four are named; the read direction used to return a
/// bare `false` on three of them and left the caller with no reason.
///
/// Callers flush deferred writeback over the range first, and the write
/// direction re-checks its [`PagesVouched`] after that flush.
fn copy_mapping_runs<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    mut copy: RunCopy<'_>,
    only: Option<&[(u64, u64)]>,
    site: &str,
) -> bool {
    if copy.is_write() {
        // Puts bytes into guest pages the hypervisor's dirty bitmap cannot
        // witness. The read direction shares this walk and writes nothing.
        state.note_host_wrote_mapping(mapping_id);
    }
    let page_size = state.page_size();
    let len = copy.len();
    let need_end = off.saturating_add(len as u64);
    // Fast path: one packed view covering the whole range.
    if let Some((ptr, view_len)) = ensure_contig_view(state, host, mapping_id) {
        if (view_len as u64) >= need_end && (off as usize) + len <= view_len {
            for (lo, hi) in selected_within(only, off, need_end) {
                let buf_off = (lo - off) as usize;
                let n = (hi - lo) as usize;
                // SAFETY: the view covers `need_end`, checked directly above,
                // and `[lo, hi)` is within `[off, need_end)` by construction.
                unsafe { copy.apply(ptr, lo as usize, buf_off, n) };
                if copy.is_write() {
                    note_mapping_write_footprint(state, mapping_id, lo, n as u64);
                }
            }
            return true;
        }
    }
    let Some(gpas) = mapping_page_gpas(state, host, mapping_id) else {
        crate::observe::fail(format!(
            "{site} fail reason=revalidate mid={mapping_id} off={off:#x} len={len:#x}"
        ));
        return false;
    };
    let page_sz = page_size as usize;
    let span_end = (gpas.len() as u64).saturating_mul(page_size);
    if need_end > span_end {
        crate::observe::fail(format!(
            "{site} fail reason=short_table mid={mapping_id} off={off:#x} len={len:#x} span={span_end:#x}"
        ));
        return false;
    }
    flush_retired_views(state, host);
    let runs = reims_vgpu_paging::runs::contig_page_runs(&gpas, page_size);
    let import_started = std::time::Instant::now();
    for run in &runs {
        let run_gpas = &gpas[run.clone()];
        let run_mlo = (run.start as u64).saturating_mul(page_size);
        let run_mhi = (run.end as u64).saturating_mul(page_size);
        let copy_lo = off.max(run_mlo);
        let copy_hi = need_end.min(run_mhi);
        if copy_lo >= copy_hi {
            continue;
        }
        let Some(ptr) = host.map_pages(run_gpas, page_sz) else {
            crate::observe::fail(format!(
                "{site} fail reason=map_pages mid={mapping_id} run_pages={} mlo={run_mlo:#x}",
                run_gpas.len()
            ));
            return false;
        };
        let total = run_gpas.len().saturating_mul(page_sz);
        let buf_off = (copy_lo - off) as usize;
        let host_off = (copy_lo - run_mlo) as usize;
        let n = (copy_hi - copy_lo) as usize;
        if host_off + n > total || buf_off + n > len {
            host.unmap_pages(ptr, total);
            crate::observe::fail(format!(
                "{site} fail reason=run_bounds mid={mapping_id} host_off={host_off:#x} \
                 buf_off={buf_off:#x} n={n:#x} total={total:#x} len={len:#x}"
            ));
            return false;
        }
        // The selection is applied *inside* the mapped run, so a run the
        // selection touches nowhere still costs its import — and a run it
        // touches in twenty places costs only one. Splitting the import per
        // selected span instead would map and unmap the same pages repeatedly.
        for (sel_lo, sel_hi) in selected_within(only, copy_lo, copy_hi) {
            let buf_off = (sel_lo - off) as usize;
            let host_off = (sel_lo - run_mlo) as usize;
            let n = (sel_hi - sel_lo) as usize;
            // SAFETY: the map covers `total`, and `host_off + n <= total` and
            // `buf_off + n <= len` were both just checked for the enclosing
            // window, which `[sel_lo, sel_hi)` is inside.
            unsafe { copy.apply(ptr, host_off, buf_off, n) };
            if copy.is_write() {
                // Packed run, so the `n` bytes at `host_off` are the `n` bytes
                // at `run_gpas[0] + host_off`. Marked per span rather than once
                // over the whole range because the runs are what is contiguous;
                // the mapping's list between them is not.
                crate::observe::footprint::note_written_range(
                    run_gpas[0].saturating_add(host_off as u64),
                    n as u64,
                );
            }
        }
        host.unmap_pages(ptr, total);
    }
    let import_us = import_started.elapsed().as_micros() as u64;
    if mapping_run_import_is_slow(import_us) {
        crate::observe::off(format!(
            "{site}_runs mid={mapping_id} us={import_us} bytes={len} pages={} runs={}",
            gpas.len(),
            runs.len()
        ));
    }
    true
}

/// Write `buf` into mapping linear offset `off` via packed map_pages runs.
///
/// Covers fragmented page lists (Linux product): split GPAs into maximal packed
/// runs, map each, poke, unmap. No `write_gpa`. Returns false if revalidate /
/// map fails.
///
/// `vouched` is the caller's proof that the page list still names the surface's
/// guest memory ([`vouch_mapping_pages_verdict`]); it is a parameter rather than a call
/// here because two callers write a row at a time and the walk is per page.
pub fn write_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &[u8],
    vouched: &PagesVouched,
) -> bool {
    write_mapping_bytes_only(state, host, mapping_id, off, buf, None, vouched)
}

/// [`write_mapping_bytes`], storing only the parts of `buf` that `only` names.
///
/// `only` is in the same mapping-linear space as `off`, ascending and disjoint;
/// `None` stores the buffer whole. Everything the write owes before the first
/// byte moves — the deferred-writeback flush, the residency invalidation, the
/// re-check of `vouched` and the payload sample — is owed once for the span
/// `buf` covers and is done once here, whatever the selection turns out to
/// name. That is what makes this different from calling the whole-buffer form
/// once per run: `flush_intersecting` alone re-scans the deferred windows, and
/// the walk below re-resolves the mapping's page list, so a thousand-run frame
/// through the loop shape pays both a thousand times.
///
/// A selection that names nothing still does the prologue. It has to: the
/// pages are about to be declared current with `buf` by the caller's
/// `mark_mapping_written`, and a pending deferred window left unflushed under
/// them would land afterwards and put an older frame on top.
pub fn write_mapping_bytes_only<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &[u8],
    only: Option<&[(u64, u64)]>,
    vouched: &PagesVouched,
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: land any pending resident content
    // in these pages first so this write applies on top of it, not under it.
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingBytesWrite,
    );
    // Exact-window residency invalidation: guest pages in this range no
    // longer mirror any resident storage image (disjoint windows survive).
    state.invalidate_storage_residency_window(
        mapping_id,
        off,
        off.saturating_add(buf.len() as u64),
    );
    // The flush above can invalidate this mapping — that is exactly what it does
    // when its own drift check refuses — so the proof is re-checked after it and
    // not before. A stale token here is not a lost frame to mourn: the list it
    // was taken against is gone, and writing through whatever replaced it is the
    // corruption this token exists to prevent.
    if !vouched.covers(state, mapping_id) {
        crate::observe::fail(format!(
            "mapping_write fail reason=vouch_stale mid={mapping_id} off={off:#x} len={:#x} \
             (the page list was cleared or replaced between the walk and this write)",
            buf.len()
        ));
        return false;
    }
    copy_mapping_runs(
        state,
        host,
        mapping_id,
        off,
        RunCopy::Write(buf),
        only,
        "mapping_write",
    )
}

/// Read mapping linear `[off, off+buf.len())` via packed map_pages runs.
pub fn read_mapping_bytes<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    buf: &mut [u8],
) -> bool {
    if buf.is_empty() {
        return true;
    }
    // Deferred-writeback flush-on-access: this read must observe the resident
    // content, not the stale pre-dispatch guest bytes.
    //
    // Narrowed to this mapping's own pages. Unnarrowed, this wait silently
    // defeated the narrowing its callers had already done: `scanout::paint_mapping`
    // rules the outstanding writeback disjoint from the very same mapping and
    // skips its `ScanoutPaint` settle, then reaches here and waits for that same
    // writeback anyway. An inner gate that is wider than the outer one makes the
    // outer one decorative.
    //
    // Measured on driven macos-13 Apple Maps drags, three boots per arm,
    // alternating pinned binaries on a quiesced host. The site was 31-34 waits a
    // boot costing 153-287 ms, ~5 ms each; after narrowing it is **zero**, and the
    // outcome counters say every one of those waits was `_disjoint` with no
    // `_overlap` and no `_unnamed` — not one was owed. `fence_us/fence` and
    // `sampled_us` both fall ~3x with clean separation between the arms.
    //
    // **No throughput claim.** End-to-end per-chain time does not separate at
    // n=3: the `seed_us` and `engine_us` controls, which this cannot touch,
    // drifted upward by more than the total moved down. ~34 waits of ~5 ms in a
    // 25 s window is 0.7 % of wall clock concentrated in ~34 of ~1000 chains, so
    // this is a tail effect and a median is the wrong instrument for it.
    //
    // Correctness is unchanged in the direction that matters: the skip needs the
    // engine to *prove* the pending writeback lands nowhere in the page set, and
    // an unnameable set (`None`) settles exactly as before. The page set comes
    // from the same `mapping_reach_pages` the writeback's own destination is
    // named with, so both ends of the comparison are one rule.
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingBytesRead,
    );
    copy_mapping_runs(
        state,
        host,
        mapping_id,
        off,
        RunCopy::Read(buf),
        None,
        "mapping_read",
    )
}

/// Read a strided rectangle starting at mapping-linear `off` into a packed `dst`.
///
/// The rectangle's rows are `rect.row_stride` apart in the mapping and back to
/// back in `dst`, which is the shape every texture read has. It resolves the
/// mapping's page list and packed runs **once** for the whole rectangle, where a
/// caller looping [`read_mapping_bytes`] per row pays that resolution
/// `row_count` times — `O(pages)` each — and a caller materialising the whole
/// sample window first pays a plane-sized allocation and a second copy out of it.
///
/// `dst` shorter than the rectangle's packed size is a refusal, not a partial
/// read: the shape is checked at [`RunCopy::read_rect`] before any page is
/// touched.
pub(crate) fn read_mapping_rect<H: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut H,
    mapping_id: u32,
    off: u64,
    rect: RectStride,
    dst: &mut [u8],
) -> bool {
    // Same flush-on-access obligation as `read_mapping_bytes`, for the same
    // reason and narrowed the same way: this read must observe the resident
    // content and not the stale pre-dispatch guest bytes. It returns at once
    // when nothing is armed, so a caller that has already settled pays a
    // map-empty check.
    crate::runtime::writeback_debt::settle_for_mapping(
        state,
        host,
        mapping_id,
        crate::runtime::render_writeback::SettleSite::MappingBytesRead,
    );
    if dst.len() < rect.packed() {
        crate::observe::fail(format!(
            "mapping_read_rect fail reason=rect_dst_short mid={mapping_id} off={off:#x} \
             packed={} dst={}",
            rect.packed(),
            dst.len()
        ));
        return false;
    }
    let Some(copy) = RunCopy::read_rect(dst, rect) else {
        return false;
    };
    copy_mapping_runs(
        state,
        host,
        mapping_id,
        off,
        copy,
        None,
        "mapping_read_rect",
    )
}

/// Report a per-run host-pointer import that took at least a millisecond.
///
/// Same gate and same basis as [`REVALIDATE_SLOW_US`]: observability only, and
/// one millisecond is 6 % of a 60 Hz frame. Deliberately the same number as its
/// peer so the two rails' slow lines are comparable without a conversion.
const MAPPING_RUN_IMPORT_SLOW_US: u64 = 1_000;

#[inline]
fn mapping_run_import_is_slow(elapsed_us: u64) -> bool {
    elapsed_us >= MAPPING_RUN_IMPORT_SLOW_US
}

#[cfg(test)]
mod revalidate_tests;

#[cfg(test)]
mod tests;
