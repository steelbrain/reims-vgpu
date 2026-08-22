//! Resident render-target identity, currency, readback, and Store publication.
//!
//! This module owns when a semantic target may reuse a host-resident image and
//! how completed resident content becomes guest-visible. It does not select memory
//! topology or issue native API calls; those remain executor capabilities.

use super::*;

pub(crate) fn render_chain_identity(
    state: &Device,
    req: &DrawEncodeRequest,
    gva_alloc_generation: u64,
) -> Option<crate::model::TargetIdentity> {
    let c0 = req.colors.first()?;
    let (width, height) = (c0.width, c0.height);
    if width == 0 || height == 0 {
        return None;
    }
    if c0.mapping_id() != 0 {
        return Some(crate::runtime::present_identity::surface_identity(
            state,
            c0.mapping_id(),
            width,
            height,
        ));
    }
    gva_chain_identity(state.executor.as_ref(), req, gva_alloc_generation)
}

pub(super) fn color_target_identity<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    color: &ColorRtRequest,
    format: reims_vgpu_core::pixel_format::TexelLayout,
    known_gva_generation: Option<u64>,
) -> Option<crate::model::TargetIdentity> {
    use crate::model::TargetIdentity;

    if color.width == 0 || color.height == 0 {
        return None;
    }
    if color.mapping_id() != 0 {
        return Some(crate::runtime::present_identity::surface_identity(
            state,
            color.mapping_id(),
            color.width,
            color.height,
        ));
    }
    if color.target_gva() == 0 {
        return None;
    }
    let generation = known_gva_generation.unwrap_or_else(|| {
        if color.texture_ref != 0 {
            crate::runtime::writeback_debt::resource_key(state, task_id, color.texture_ref)
                .map(|key| {
                    crate::runtime::writeback_debt::gva_resource_generation(
                        state,
                        host,
                        key,
                        color.target_gva(),
                        u64::from(color.row_stride()).saturating_mul(u64::from(color.height)),
                    )
                })
                .unwrap_or(0)
        } else {
            gva_span_alloc_generation(
                state,
                host,
                task_id,
                color.target_gva(),
                color.row_stride(),
                color.height,
            )
        }
    });
    Some(TargetIdentity::Gva {
        gva: color.target_gva(),
        width: color.width,
        height: color.height,
        generation,
        format,
    })
}

/// The registry resident an IOSurface texture composite Store renders into, if this record
/// is one.
///
/// Unlike the GVA rail this is **not** a `skip_readback` deferral: a composite
/// Store still reads its target back, because that readback is what feeds
/// `surface_cache`, the deferred window's owned frame and the guest writeback.
/// The resident exists so the *next* LOAD on this surface can start from the
/// image that already holds these pixels instead of re-uploading them through
/// the staging span — `draw_phase` prices that upload at ~155 ms per second of
/// wall clock under a browser workload, a copy of bytes the GPU just produced.
///
/// Single source of truth on purpose. The draw's `target_identity`, the LOAD's
/// currency check and the Store's epoch stamp must name the same slot; deriving
/// it three times from three spellings of the predicate is how a stamp ends up
/// vouching for an image the draw never rendered into.
///
/// `chain_from_resident` is **not** a refusal, and used to be. The last record of
/// a resident render-pass chain is both the chain's consumer and the packet's
/// guest-visible Store, and refusing it here cost the whole composite readback
/// population: `iosurface_keep_chain_from_resident` measured equal to
/// `surface_deferred` in all twelve windows of one boot, 112-132 Stores a second
/// at 366-372 MB/s, with every other keep-reason at zero. Nothing about the chain
/// changes which slot this record renders into — `retarget_render_pass_draw`
/// builds every record of a packet from one attachment template, so records N-1
/// and N carry the same `mapping_id` and geometry and therefore the same
/// [`render_chain_identity`] — and the intermediates already render into that
/// resident under `skip_readback` with `LoadOp::LoadFromTarget`. The last record
/// differs only in what happens *after* the draw.
pub(super) fn iosurface_texture_store_identity(
    state: &Device,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
) -> Option<crate::model::TargetIdentity> {
    if !writeback_guest {
        return None;
    }
    iosurface_texture_render_identity(state, req)
}

/// The registry resident this record renders its IOSurface texture color0 *into*, whatever
/// its role in the packet — the strict superset of [`iosurface_texture_store_identity`],
/// which is this same slot restricted to the record that also stores it for the
/// guest.
///
/// The two are separate because the LOAD and the Store ask different questions of
/// the same slot. "May I start from what is already in this image?" is answerable
/// by any record that renders into it; "may I leave my frame there instead of
/// copying it to guest pages?" is only answerable by the packet's last record.
/// Conflating them cost the whole seed elision on multi-record packets: record 1
/// of a chain has `writeback_guest == false`, so the currency check keyed on the
/// Store identity never ran for it, and its LOAD fell through to
/// `resolve_iosurface_texture_load_seed` — which, with the host cache ceded to the resident
/// rail, reads the mapping's guest pages and therefore lands the very window the
/// rail armed. One boot measured that loop directly: `surface_flush /
/// surface_resident` = 1369/1373, one flush per arm, with `iosurface_texture_load_seed`
/// reporting `outcome=guest_pages` 110 times against 17 `cache_hit` and
/// `hostgen=0` on every one.
///
/// A record with `mapping_id != 0` and a real Store action renders into this slot
/// on every route: the chain block claims it for `chain_from_resident` and for a
/// `!writeback_guest` intermediate, and the composite-Store rail claims it for the
/// last record. So the condition here is the same one those blocks share, asked
/// once.
pub(super) fn iosurface_texture_render_identity(
    state: &Device,
    req: &DrawEncodeRequest,
) -> Option<crate::model::TargetIdentity> {
    let c0 = req.colors.first()?;
    if c0.mapping_id() == 0 || !c0.store_action.publishes_single_sample() {
        return None;
    }
    render_chain_identity(state, req, 0)
}

/// Stable packed allocation behind the primary type-2/3 attachment.
///
/// The decoded texture declaration already names the allocation and plane.
/// This function resolves that allocation once under the resource reference
/// that owns it and carries the resulting import and physical footprint into
/// the engine. A missing resource, incomplete GVA walk, transient host alias,
/// or changed declaration returns `None`; the ordinary device-local resident
/// remains the complete fallback.
pub(super) fn gva_guest_target_backing<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    req: &DrawEncodeRequest,
) -> Option<reims_vgpu_memory::GuestTargetMemory> {
    let c0 = req.colors.first()?;
    if c0.target_gva() == 0 {
        return None;
    }
    color_target_guest_backing(state, host, req.task_id, c0, None)
}

/// Resolve the canonical guest allocation of any declared colour attachment.
///
/// This is deliberately attachment-generic: MRT slot number does not alter the
/// allocation contract. The caller supplies the already-decoded resident
/// layout for IOSurface records; linear records carry their own declaration.
pub(super) fn color_target_guest_backing<H: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut H,
    task_id: u32,
    color: &ColorRtRequest,
    surface_layout: Option<reims_vgpu_core::pixel_format::TexelLayout>,
) -> Option<reims_vgpu_memory::GuestTargetMemory> {
    if color.target_gva() == 0 {
        return try_iosurface_texture_target_guest_memory(
            state,
            host,
            color.mapping_id(),
            color.width,
            color.height,
            surface_layout?,
        );
    }
    let decline = |route| {
        crate::runtime::drain::note_store_route(route);
        None
    };
    if !host.map_pages_stable() {
        return decline("gvatarget_alias_unstable");
    }
    let Some(linear) = color.linear_target() else {
        return decline("gvatarget_declaration_missing");
    };
    if color.texture_ref == 0 {
        return decline("gvatarget_resource_missing");
    }
    let allocation = BufferBacking {
        gva: linear.allocation_gva,
        size: linear.allocation_size,
    };
    if !crate::runtime::bound_buffers::ensure_packed_resource(
        state,
        host,
        task_id,
        color.texture_ref,
        allocation.gva,
        allocation.size,
        crate::runtime::bound_buffers::PackedResourceUse::LinearTarget,
    ) {
        return decline("gvatarget_pack_declined");
    }
    let Some(packed) = state.bound_buffers.packed_available(
        task_id,
        color.texture_ref,
        allocation.gva,
        allocation.size,
    ) else {
        return decline("gvatarget_pack_unavailable");
    };
    let Some(plane_offset) = packed.head.checked_add(linear.plane_offset) else {
        return decline("gvatarget_plane_overflow");
    };
    if plane_offset >= packed.import.len() {
        return decline("gvatarget_plane_outside_import");
    }
    Some(reims_vgpu_memory::GuestTargetMemory {
        backing: reims_vgpu_memory::GuestTargetBacking {
            allocation_host_ptr: packed.import.host_base(),
            allocation_len: packed.import.len(),
            resource_offset: packed.head,
            resource_len: linear.allocation_size,
            plane_offset,
            row_pitch: u64::from(linear.row_stride),
        },
        import: std::sync::Arc::clone(&packed.import),
        footprint: packed.footprint.clone(),
    })
}

/// Whether this record's color0 LOAD is one the resident could serve at all —
/// it must be a LOAD, and no explicit seed may already have been selected for
/// it by RT provenance. Separate from the currency question so the two counters
/// on the branch below divide candidates, not all draws.
pub(super) fn iosurface_texture_load_is_a_seed_candidate(c0: &ColorRtRequest) -> bool {
    c0.load_action == reims_vgpu_protocol::pass_action::LoadAction::Load
        && c0.target_seed_rgba.is_none()
}

/// The `(resident, mapping epoch)` pair a record's IOSurface texture LOAD has to compare to
/// decide whether the image it is about to render into already holds the seed —
/// or `None` when this record's LOAD is not one a resident could serve at all.
///
/// **Takes no `writeback_guest`, deliberately.** The record's role in the packet
/// is not part of this question, and a signature that cannot see the role cannot
/// be keyed on it. It was: the check read the *Store* identity, so record 1 of a
/// chain — `writeback_guest == false` — never asked, took a CPU seed, found the
/// host cache ceded to the resident rail, read the mapping's guest pages, and by
/// reading them landed the window the packet's Store had just armed. That advanced
/// the epoch and cost the next LOAD its elision in turn, so the rail degraded into
/// a rescheduling with a GPU round trip added: `surface_flush / surface_resident`
/// = 1369/1373 on one boot. Structure rather than a test, because a unit test on
/// the resolver passes whatever the call site then decides to pass it.
pub(super) fn iosurface_texture_load_currency_query(
    state: &Device,
    req: &DrawEncodeRequest,
) -> Option<(crate::model::TargetIdentity, Option<u32>)> {
    let c0 = req.colors.first()?;
    if !iosurface_texture_load_is_a_seed_candidate(c0) {
        return None;
    }
    let identity = iosurface_texture_render_identity(state, req)?;
    let mapping_epoch = state
        .surfaces
        .mappings
        .get(&c0.mapping_id())
        .map(|m| m.content.surface_epoch);
    Some((identity, mapping_epoch))
}

/// Count which contract-owned rung served an IOSurface texture sampled bind.
pub(super) fn note_iosurface_texture_sample_rung(rung: &'static str) {
    crate::runtime::drain::note_store_route(rung);
}

/// Whether a resident's stamp still vouches for the mapping's current content.
///
/// The `is_some` guard is the whole function. Both values are `Option`, and
/// `None == None` is `true` in Rust — so a bare equality would read "the
/// mapping has no entry" and "this image was never stamped" as agreement and
/// load undefined memory as though it were the guest's prior frame. That is
/// precisely the black-layer class. Absence on either side is a refusal.
pub(super) fn iosurface_texture_resident_is_current(
    mapping_epoch: Option<u32>,
    resident_epoch: Option<u32>,
) -> bool {
    mapping_epoch.is_some() && mapping_epoch == resident_epoch
}

pub(super) fn iosurface_texture_load_resident_is_current(
    copied_currency: impl FnOnce() -> bool,
) -> bool {
    copied_currency()
}

/// Record that the resident this Store rendered into holds the mapping's
/// content as of `epoch`, so the surface's next LOAD can skip its CPU seed.
///
/// Keyed through [`iosurface_texture_store_identity`] — the same call the draw's
/// `target_identity` came from — so the slot stamped is the slot rendered into.
/// A miss is expected and silent: the identity resolves to `None` when this
/// record never took the resident path, and `stamp_resident_content_epoch`
/// refuses a slot that was evicted between the draw and here. Both leave the
/// stamp absent, which costs a seed and never a wrong frame.
///
pub(super) fn stamp_iosurface_texture_resident(
    state: &mut Device,
    req: &DrawEncodeRequest,
    writeback_guest: bool,
    epoch: u32,
) {
    if let Some(identity) = iosurface_texture_store_identity(state, req, writeback_guest) {
        state
            .executor
            .stamp_resident_content_epoch(&identity, epoch);
        // Both callers reach here only on a route that has already put these
        // pixels outside the image — the synchronous one through `write_bgra8`
        // into the mapping's guest pages, the deferred-readback one through
        // `publish_surface_store` into `surface_cache` plus the window's own
        // `Arc`. So the resident is no longer the sole copy and the reclaim
        // paths may take it, which is the whole reason those routes pay a
        // readback.
        //
        // Deliberately not on `arm_surface_resident_store`: that route skips the
        // readback precisely so no copy is made, keeps the frame in the image
        // alone, and holds it with a pin instead.
        state.executor.note_resident_content_copied_out(&identity);
    }
}

/// Order-independent hash of the guest physical pages behind a GVA span.
///
/// The set is the identity, not the sequence: a walk that reports the same pages
/// in another order must produce the same value, or re-rendering one buffer
/// would read as a new allocation. `0x9E37_79B9_7F4A_7C15` is the odd 64-bit
/// golden-ratio constant (2^64 / phi rounded to odd), used here only to spread
/// page GPAs — which are dense multiples of the page size, so their low bits are
/// all zero — across the whole word before the XOR fold.
pub(crate) fn gva_page_set_hash(pages: &std::collections::HashSet<u64>) -> u64 {
    let mut hash: u64 = 0;
    for p in pages {
        hash ^= p.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    hash
}

/// Host-texture identity of this draw's color0 GVA resource.
///
/// A task-local texture reference keeps one generation from creation through
/// ordinary task map changes and transfer-backing discard. Explicit resource
/// delete ends that lifetime. A target with no resource reference falls back to
/// the page-set identity because it has no protocol lifetime to name instead.
pub(super) fn gva_alloc_generation<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
) -> u64 {
    let Some(c0) = req.colors.first() else {
        return 0;
    };
    if c0.mapping_id() != 0 || c0.target_gva() == 0 {
        return 0;
    }
    // Same span as the deferred arm walks (`arm_gva_deferred_store`) so the two
    // describe one region: the guest bytes a Store into this target writes.
    if c0.texture_ref != 0 {
        crate::runtime::writeback_debt::resource_key(state, req.task_id, c0.texture_ref)
            .map(|key| {
                crate::runtime::writeback_debt::gva_resource_generation(
                    state,
                    host,
                    key,
                    c0.target_gva(),
                    u64::from(c0.row_stride()).saturating_mul(u64::from(c0.height)),
                )
            })
            .unwrap_or(0)
    } else {
        gva_span_alloc_generation(
            state,
            host,
            req.task_id,
            c0.target_gva(),
            c0.row_stride(),
            c0.height,
        )
    }
}

/// The page-set generation of one `row_stride * height` GVA span under one task.
///
/// Every GVA render target is named this way, not only color0: the secondary MRT
/// attachments go through here too ([`build_secondary_targets`]). One spelling,
/// so no two callers can disagree about what names an allocation.
///
/// A short walk yields 0. An incomplete walk names no allocation, and hashing the
/// pages that happened to resolve would be an identity the guest never had.
pub(crate) fn gva_span_alloc_generation<M: HostMemory + HostOps>(
    state: &Device,
    host: &mut M,
    task_id: u32,
    gva: u64,
    row_stride: u32,
    height: u32,
) -> u64 {
    if gva == 0 {
        return 0;
    }
    let span = (row_stride as u64).saturating_mul(height as u64);
    if span == 0 {
        return 0;
    }
    let pages =
        gva_mem::task_gva_page_gpa_set(host, &state.tasks, task_id, gva, span, state.page_shift);
    if (pages.len() as u64) < reims_vgpu_paging::span::pages_spanned(gva, span, state.page_size()) {
        return 0;
    }
    gva_page_set_hash(&pages)
}

/// Engine-resident identity for a GVA (type-2/3) color0 render target.
///
/// Single source of truth for the resident GVA chain rail: the draw path,
/// the alias-sample bind, and the abandon-path landing must all agree on the
/// exact identity or the registry lookups miss.
///
/// `generation` is resolved once from the guest pages backing the target,
/// because a GVA is only a name and the
/// guest recycles names. Without it the registry keys this resident on
/// `(gva, width, height)` alone, and the cross-pass resident Load below hands a
/// new allocation the previous one's pixels as its prior content.
pub(crate) fn gva_chain_identity(
    executor: &dyn crate::runtime::executor::Executor,
    req: &DrawEncodeRequest,
    generation: u64,
) -> Option<crate::model::TargetIdentity> {
    let c0 = req.colors.first()?;
    if c0.mapping_id() != 0 || c0.target_gva() == 0 {
        return None;
    }
    let (w, h) = (c0.width, c0.height);
    if w == 0 || h == 0 {
        return None;
    }
    Some(crate::model::TargetIdentity::Gva {
        gva: c0.target_gva(),
        width: w,
        height: h,
        generation,
        format: gva_resident_format(executor, c0.format),
    })
}

/// The format the resident behind a GVA render target must hold: the one the
/// guest declared for that attachment.
///
/// This is [`crate::model::TargetIdentity::resident_layout`]'s
/// rule applied to the one namespace that has a declaration to follow, and it is
/// a function rather than an expression at each producer because the producers
/// key *the same registry slot*. A primary and a secondary attachment that
/// spelled it differently would render into one identity claiming two formats,
/// which `registry_ensure` answers by recreating the image every frame.
///
/// The guest's declaration is followed whenever the host can follow it. A
/// layout this device has no `TexelLayout` for, or one the host cannot render
/// to and blend into, falls back to the engine's neutral resident colour
/// format — which is what *every* render target got while the key held a
/// `bgra: bool` and could express nothing else.
///
/// The fallback is a fidelity loss and not a refusal. The draw still runs, and
/// the Store still lands correctly-shaped bytes for the guest's declared texel,
/// because [`reims_vgpu_core::pixel_format::convert_rgba8_to_row`] expands them
/// from eight bits — the guest reads a well-formed half-float frame carrying
/// eight bits of information. What the fallback costs is the range and the
/// precision the guest asked for: anything above 1.0 in a half-float
/// compositing target is clamped away before the compositor sees it.
///
/// Capability, never an API-version assumption.
/// `VK_FORMAT_R16G16B16A16_SFLOAT` is in Vulkan's mandatory format table for
/// both `COLOR_ATTACHMENT` and `COLOR_ATTACHMENT_BLEND`, and it is still asked
/// for per host: a widening that reads the spec's table instead of the device
/// is the shape AGENTS.md names, and this one would fail at `vkCreateImage`
/// rather than decline.
pub(crate) fn gva_resident_format(
    executor: &dyn crate::runtime::executor::Executor,
    format: u16,
) -> reims_vgpu_core::pixel_format::TexelLayout {
    // The declaration the *attachment* will be built from, folded onto its
    // allocation family, which is what `registry_ensure` creates the image in.
    //
    // Asked of `color_attachment` and not of `store_texel_order`, which is what
    // this used to ask and which is a different question — that one says whether
    // a resident's texels can be byte-copied into guest pages, and it answers
    // for three formats where `render_target_bpp` admits six. A guest render
    // target declared `R8Unorm`, `R16Float` or `RG16Float` therefore built its
    // image at one to four bytes a texel through `color_attachment` while its
    // identity claimed `RESIDENT_RGBA_FORMAT` and four, so everything keyed off
    // the identity — `bytes_per_texel`, the readback size, `stored_bytes_agree`
    // — read the wrong width for the image it was describing. `R16Float` and
    // `RG16Float` are both in the guest's vocabulary on boots on record.
    let Ok(attachment) = pixel_format::color_attachment_format_checked(format) else {
        return reims_vgpu_core::pixel_format::TexelLayout::Rgba8;
    };
    let layout = attachment.layout();
    // Capability, never an API-version assumption: the host is asked whether
    // it renders to and blends this layout.
    if executor.render_target_layout_supported(layout) {
        layout
    } else {
        reims_vgpu_core::pixel_format::TexelLayout::Rgba8
    }
}

/// Read a resident render-pass chain back to host memory so the exec loop can
/// land its pixels. Every failure is fail-visible; the guest keeps its pre-pass
/// bytes on loss.
///
/// # Every caller is now a refusal path
///
/// This used to be the ordinary Store of a GVA-targeted render, and the doc here
/// used to say so — 13 653 reads on a driven boot, 59 % of every render Store,
/// each one submitting and then blocking on a fence. That is no longer the
/// shape, and reading it as the hot path sends the next reader at a premise that
/// has already been fixed.
///
/// Both Store rails normally leave the frame host-authoritative. Their payments
/// use the GPU-direct guest-page copies in `render_writeback`; this function is
/// what either Store falls back to when it cannot arm or materialize that copy,
/// plus `writeback_chain_rgba`. The ordinary path therefore has no framebuffer
/// readback or CPU conversion at Store time.
///
/// So this is genuinely the abandon path, and it is a cost rather than a lost
/// frame — but it is the expensive one, and the reason to keep it narrow: it
/// reads the whole framebuffer back across the bus and blocks on a fence to do
/// it. A change that pushes traffic back onto it will not show up as a refusal,
/// only as `slot_us` and the fail-visible decline that sent it here.
///
/// What made the GPU-direct arm reachable was format. A buffer→image copy
/// converts nothing, so the resident has to already hold the bytes its
/// destination stores; the order used to be derived from the identity's *kind*,
/// which made every GVA resident RGBA and every Store a per-row conversion.
/// `TargetIdentity::Gva` now carries the order the guest declared for that
/// render target — see `is_bgra` on
/// [`crate::model::TargetIdentity`] for why it is keyed on the
/// identity and what a change there has to keep true.
///
/// The identity is the caller's, never re-derived here: both callers hold the
/// key their own draw registered, carried out of `M2vDrawSpan`, and
/// `render_chain_identity` asked again after the draw can answer at a newer
/// mapping generation than the one the registry holds. See
/// [`M2vDrawSpan::ResidentSurfaceStore`].
pub(crate) fn read_resident_chain(
    executor: &dyn crate::runtime::executor::Executor,
    req: &DrawEncodeRequest,
    identity: &crate::model::TargetIdentity,
) -> Option<Vec<u8>> {
    match executor.read_target(identity) {
        // Every caller of this function — `writeback_chain_rgba` and the GVA
        // arm-refusal fallback — has an RGBA contract, so the exchange happens
        // here, once, rather than at three call sites that would each have to
        // remember which namespace they were reading. `into_rgba8` uses the order
        // the engine reports for the image it copied, so it is a no-op on a
        // pooled resident and a whole-frame pass on anything the guest declared
        // in BGRA order — a surface, and a GVA target whose declared format
        // `gva_resident_format` could honour, which is most of them.
        //
        // That last clause used to read "a no-op on the pooled and GVA
        // residents", and it was wrong rather than imprecise: a GVA render
        // target declared `BGRA8Unorm` or `BGRA8Unorm_sRGB` is resident in BGRA
        // and owes the exchange. It was a true description of what the code did
        // — `ResidentReadSnapshot::bgra` answered "not BGRA" for the sRGB
        // spelling — so the comment documented the defect instead of catching
        // it, and the Store below exchanged R and B on its way into guest pages.
        Ok(rb) => Some(rb.into_rgba8()),
        Err(e) => {
            crate::observe::fail(format!(
                "chain_resident_land_fail reason=read_target target={identity:?} \
                 mid={} gva={:#x} {}x{} err={e}",
                req.colors.first().map(|c| c.mapping_id()).unwrap_or(0),
                req.colors.first().map(|c| c.target_gva()).unwrap_or(0),
                identity.width(),
                identity.height()
            ));
            None
        }
    }
}
/// Land an IOSurface texture render Store's frame in the guest's pages, from the resident
/// the draw just rendered into.
///
/// The Store never reads the frame back off the GPU: the copy's destination is
/// the guest's own pages, recorded into the engine's command stream and ordered
/// against the guest by the completion stamp. See
/// [`crate::runtime::render_writeback`] for what this replaced, and why the
/// deferred window it used to arm bought nothing on any measured workload.
///
/// `false` means the frame is not in the guest's pages and the caller must
/// materialize it the slow way — read the resident back and run the synchronous
/// Store block. That is a cost, never a lost frame.
///
/// The identity is the caller's, carried out of the draw on
/// [`M2vDrawSpan::ResidentSurfaceStore`], so the image read here is the image
/// the draw rendered into by construction.
///
/// It used to be [`iosurface_texture_store_identity`] called again here, on the argument
/// that this is the same function that produced the draw's `target_identity`.
/// The same function is not the same value: that identity carries
/// the mapping lifecycle generation, and the draw mutates `Device` between
/// the two calls. Read that variant's doc before reintroducing a derivation
/// anywhere on this path — deriving it a second *time* is as wrong as deriving
/// it a second *way*.
pub(super) fn store_surface_resident<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    identity: &crate::model::TargetIdentity,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> bool {
    // The union belongs to the draws that just ran whether or not the write
    // below succeeds, and leaving it un-reset on a refused write would fold this
    // pass into the next Store's reading.
    super::execution::note_pass_scissor_union(width, height);
    if !crate::runtime::render_writeback::store_render_frame(
        state, host, mapping_id, identity, width, height,
    ) {
        return false;
    }
    true
}

/// Readback-skip gate for the final/single record of a GVA render Store: the
/// record may leave its pixels on the engine registry resident and let the
/// caller read them back once, instead of taking a readback plus a fence wait
/// inside the record. All gates are protocol-shape checks (never content): the
/// caller must be able to replay the sync `write_gva_rgba8` exactly — identity
/// geometry == c0 geometry, convertible format, sane BPR.
pub(super) fn gva_store_defer_eligible(req: &DrawEncodeRequest, gva_alloc_generation: u64) -> bool {
    let Some(c0) = req.colors.first() else {
        return false;
    };
    if c0.mapping_id() != 0
        || c0.target_gva() == 0
        || c0.row_stride() == 0
        || c0.texture_ref == 0
        || gva_alloc_generation == 0
    {
        return false;
    }
    pixel_format::tight_row_bytes(c0.width, c0.format)
        .and_then(|bytes| bytes.checked_mul(c0.sample_count.max(1)))
        .is_some_and(|tight| c0.row_stride() >= tight)
}
