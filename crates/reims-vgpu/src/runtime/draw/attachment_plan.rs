//! Render-pass colour attachment planning.
//!
//! This module converts decoded attachment records into one complete semantic
//! target set. Expected absence is distinct from a typed refusal; callers never
//! decide whether a malformed or partially resolved pass may execute.

use super::*;

/// Build an MRT draw request from the pass's color slots.
#[allow(
    clippy::too_many_arguments,
    reason = "the MRT builder combines explicit pass, pipeline, and draw state"
)]
pub fn mrt_draw_request<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    pipeline_ref: u32,
    color_slots: &[(u32, crate::runtime::decode::render::ColorAttachment)],
    clears: &[crate::runtime::decode::render::ColorAttachment],
    draw: reims_vgpu_core::draw::DrawArgs,
) -> Result<Option<DrawEncodeRequest>, reims_vgpu_core::AttachmentPlanDecline> {
    use reims_vgpu_core::AttachmentPlanDecline;
    if color_slots.is_empty() {
        return Ok(None);
    }
    // Linear allocation dimensions expose mip and array geometry, but do not
    // repeat a texture's immutable creation sample count. At render time the
    // bound pipeline supplies the missing contract: every color attachment
    // must match its raster sample count. Resolve that before LOAD/CLEAR seed
    // policy and before this request is cloned by either encoder.
    let pipeline_sample_count = crate::runtime::pipeline_resolve::attachment_sample_count(
        state,
        host,
        task_id,
        pipeline_ref,
    );
    let mut colors = Vec::new();
    // Colour0's LOAD seed was skipped in favour of the engine resident. Declared
    // out here because it belongs to the request, not to the slot that set it.
    let mut gva_load_source = GvaLoadSource::None;
    for &(slot, att) in color_slots {
        if att.texture_ref == 0 {
            // An empty colour slot is the guest declining to attach one, not a
            // loss. Counted anyway, because it is the difference between the
            // slots the pass *has* and the slots it *uses*, and the census
            // below is unreadable without it.
            crate::runtime::drain::note_store_route("mrt_slot_empty");
            continue;
        }
        crate::runtime::drain::note_store_route("mrt_slot_attached");
        let load_action = reims_vgpu_protocol::pass_action::load_action(att.load_action)
            .map_err(|reason| AttachmentPlanDecline::PassAction { slot, reason })?;
        let store_action = reims_vgpu_protocol::pass_action::store_action(att.store_action)
            .map_err(|reason| AttachmentPlanDecline::PassAction { slot, reason })?;
        // Resolve both sides independently. The source proves the multisample
        // attachment's shape; the destination becomes the guest-visible target
        // that the backend stores and reads back.
        let Some(source_target) = lookup_render_target(state, host, task_id, att) else {
            crate::runtime::drain::note_store_route("mrt_slot_unresolved");
            return Err(AttachmentPlanDecline::TargetUnresolved {
                slot,
                texture_ref: att.texture_ref,
                role: reims_vgpu_core::AttachmentTargetRole::Source,
            });
        };
        let (target_ref, multisample_source_ref, target) = if att.resolve_texture_ref != 0 {
            let resolve_attachment = ColorAttachment {
                texture_ref: att.resolve_texture_ref,
                resolve_texture_ref: 0,
                level: 0,
                ..att
            };
            let Some(resolve_target) =
                lookup_render_target(state, host, task_id, resolve_attachment)
            else {
                crate::runtime::drain::note_store_route("mrt_resolve_target_unresolved");
                return Err(AttachmentPlanDecline::TargetUnresolved {
                    slot,
                    texture_ref: att.resolve_texture_ref,
                    role: reims_vgpu_core::AttachmentTargetRole::Resolve,
                });
            };
            if crate::observe::first_sight(
                "render_resolve_contract",
                (u64::from(att.texture_ref) << 32) | u64::from(att.resolve_texture_ref),
            ) {
                crate::observe::off(format!(
                    "render_resolve_contract task={task_id} pipe={pipeline_ref} \
                     source_ref={} source_mid={} source_gva={:#x} source={}x{} \
                     source_fmt={:#x} resolve_ref={} resolve_mid={} resolve_gva={:#x} \
                     resolve={}x{} resolve_fmt={:#x} load={} store={} raster_samples={}",
                    att.texture_ref,
                    source_target.storage.mapping_id(),
                    source_target.storage.target_gva(),
                    source_target.width,
                    source_target.height,
                    source_target.format,
                    att.resolve_texture_ref,
                    resolve_target.storage.mapping_id(),
                    resolve_target.storage.target_gva(),
                    resolve_target.width,
                    resolve_target.height,
                    resolve_target.format,
                    att.load_action,
                    att.store_action,
                    pipeline_sample_count.unwrap_or(1),
                ));
            }
            if source_target.width != resolve_target.width
                || source_target.height != resolve_target.height
                || source_target.format != resolve_target.format
            {
                return Err(AttachmentPlanDecline::ResolveTargetMismatch {
                    slot,
                    source_ref: att.texture_ref,
                    resolve_ref: att.resolve_texture_ref,
                    source_width: source_target.width,
                    source_height: source_target.height,
                    resolve_width: resolve_target.width,
                    resolve_height: resolve_target.height,
                    source_format: source_target.format,
                    resolve_format: resolve_target.format,
                });
            }
            (att.resolve_texture_ref, att.texture_ref, resolve_target)
        } else {
            (att.texture_ref, 0, source_target)
        };
        let ResolvedRenderTarget {
            storage,
            width: mw,
            height: mh,
            format: mfmt,
            sample_count: target_sample_count,
        } = target;
        let mapping_id = storage.mapping_id();
        let gva = storage.target_gva();
        let bpr = storage.row_stride();
        let attachment_sample_count = pipeline_sample_count.unwrap_or(target_sample_count);
        let mut load_action = load_action;
        let mut clear_color = att.clear_color;
        let mut seed = None;
        if let Some(cl) = clears.iter().find(|a| a.texture_ref == att.texture_ref) {
            // Clear-only stream record for this attachment: real Metal Clear.
            load_action = reims_vgpu_protocol::pass_action::LoadAction::Clear;
            clear_color = cl.clear_color;
            if mapping_id == 0 {
                seed = Some(solid_rgba8(mw, mh, &cl.clear_color));
            }
        } else if load_action == reims_vgpu_protocol::pass_action::LoadAction::Clear {
            if mapping_id == 0 {
                seed = Some(solid_rgba8(mw, mh, &att.clear_color));
            }
        } else if load_action == reims_vgpu_protocol::pass_action::LoadAction::Load
            && mapping_id == 0
        {
            // A GVA linear target needs a CPU seed when no mapping is available.
            // IOSurface texture is seeded later instead, at the attachment site in
            // `encode_draw` — the same place the guest-backed alias used to be
            // built, and the same seed it already took whenever the alias was
            // refused. Seeding here would need the mapping read twice.
            //
            {
                // Before the read, not after it: the seed this is about to build
                // is the one a resident rung would replace, and a probe placed
                // downstream of here measures an empty population by
                // construction — see `note_gva_load_seed_probe`.
                // Before the read, not after it. The engine may still hold
                // exactly what the render Store published into these pages, in
                // which case reading them back costs a full-frame CPU walk and a
                // block on that same Store's writeback — the device's largest
                // remaining wait. See `draw::execution::gva_resident_if_current`;
                // the encode side honours the flag or re-seeds.
                // Only colour0. `gva_chain_identity` names the first attachment
                // and the chain rail carries that one, so a second slot whose
                // seed was skipped would reach the pass with nothing to load.
                // `colors.is_empty()` is "this push becomes `colors[0]`", taken
                // from the vector the identity will read rather than from the
                // slot number, which is the guest's and need not start at zero.
                let is_color0 = colors.is_empty();
                let resident = is_color0
                    && gva_load_seed_elidable(
                        state,
                        host,
                        task_id,
                        GvaSpan {
                            texture_ref: att.texture_ref,
                            gva,
                            row_stride: bpr,
                            width: mw,
                            height: mh,
                            format: mfmt,
                        },
                    );
                if is_color0 {
                    gva_load_source = if resident {
                        GvaLoadSource::Resident
                    } else {
                        GvaLoadSource::GuestPages
                    };
                } else {
                    seed = seed_color_load(state, host, task_id, att.texture_ref, gva, mw, mh);
                    if seed.is_none() {
                        crate::observe::fail(format!(
                            "color LOAD seed miss ref={} {}x{} fmt={:#x} gva={:#x} (archive: still encode)",
                            att.texture_ref, mw, mh, mfmt, gva
                        ));
                    }
                }
            }
        }
        colors.push(ColorRtRequest {
            slot,
            texture_ref: target_ref,
            resource: objects::resolve_resource(state, host, task_id, target_ref).ok(),
            storage,
            width: mw,
            height: mh,
            format: mfmt,
            sample_count: attachment_sample_count,
            load_action,
            store_action,
            clear_color,
            target_seed_rgba: seed,
            multisample_source_ref,
        });
    }
    if colors.is_empty() {
        return Ok(None);
    }
    Ok(Some(DrawEncodeRequest {
        task_id,
        pipeline_ref,
        vertex_count: draw.vertex_count,
        instance_count: draw.instance_count,
        primitive_topology: draw.primitive_topology,
        first_vertex: draw.first_vertex,
        base_instance: draw.base_instance,
        colors,
        gva_load_source,
        ..Default::default()
    }))
}
