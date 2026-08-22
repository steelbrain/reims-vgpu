//! Final semantic executor-request assembly for a resolved draw.
//!
//! All resource families and target policy meet here exactly once. The result
//! contains an immutable core request and its sole completion route; Store and
//! diagnostics consume the completed draw outside this module.

use super::*;

pub(super) struct RequestPlanInputs {
    pub(super) blend_states: Vec<(u32, reims_vgpu_protocol::BlendStateResource)>,
    pub(super) attributes: Vec<reims_vgpu_core::VertexAttributeResource>,
    pub(super) storage_buffers: Vec<reims_vgpu_core::StorageBufferResource>,
    pub(super) sampled_images: Vec<reims_vgpu_core::SampledImageResource>,
    pub(super) samplers: Vec<reims_vgpu_core::SamplerResource>,
    pub(super) target_rgba8: Option<std::sync::Arc<Vec<u8>>>,
    pub(super) target_guest: Option<reims_vgpu_memory::GuestTargetPlan>,
    pub(super) target_clear: [f32; 4],
    pub(super) color_load_action: reims_vgpu_core::ColorLoadAction,
    pub(super) target_seed_order: reims_vgpu_core::SeedOrder,
    pub(super) color_input: bool,
    pub(super) gva_alloc_generation: u64,
    pub(super) gpu_only_content_allowed: bool,
    pub(super) writeback_guest: bool,
    pub(super) iosurface_texture_resident_target: Option<crate::model::TargetIdentity>,
    pub(super) chain_load_from_target: bool,
    pub(super) gva_load_identity: Option<crate::model::TargetIdentity>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) program: reims_vgpu_core::PreparedRenderProgram,
}

pub(super) struct ExecutorRequestPlan {
    pub(super) request: reims_vgpu_core::DrawRequest,
    pub(super) completion_route: DrawCompletionRoute,
    pub(super) vertex_count: u32,
    pub(super) secondary_targets_built: bool,
}

pub(super) fn plan_executor_request<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    inputs: RequestPlanInputs,
) -> Result<ExecutorRequestPlan, DrawPreparationDecline> {
    let pd = &resolved.desc;
    let RequestPlanInputs {
        blend_states,
        attributes: attrs,
        storage_buffers: storage,
        sampled_images: images,
        samplers,
        target_rgba8,
        target_guest,
        target_clear,
        color_load_action,
        target_seed_order: seed_order,
        color_input: frag_color_input,
        gva_alloc_generation,
        gpu_only_content_allowed,
        writeback_guest,
        iosurface_texture_resident_target,
        chain_load_from_target,
        gva_load_identity,
        width: w,
        height: h,
        program,
    } = inputs;
    let mut gva_load_identity = gva_load_identity;

    let mut resources = reims_vgpu_core::DrawRequest {
        pipeline_lifetime: resolved.pipeline_lifetime.clone(),
        // Honor the guest's face-culling state, its winding, and its
        // primitive type. All three come from the protocol decoder, and all
        // three fall back to a Metal default when the guest bound nothing —
        // but an out-of-contract *value* is a different thing from an unbound
        // one, and it says its own name before falling back. Silently
        // coercing here is how a guest that asked for lines got triangles
        // with nothing in the log to say so.
        cull_mode: req.cull_mode,
        // MTLWinding: CounterClockwise == 1; Metal defaults to Clockwise.
        front_face_ccw: req.front_face_ccw,
        // MTLTriangleFillMode / MTLDepthClipMode, both defaulting to 0.
        // Unlike the two above, the non-default arm of each needs a device
        // feature, so the engine may still decline the pipeline by name
        // after this maps cleanly: the mapping says what the guest asked
        // for, the capability check says whether the host can spell it.
        fill_mode: req.fill_mode,
        line_width: req.line_width,
        depth_bias: req.depth_bias,
        depth_clip: req.depth_clip_mode,
        blend_constants: req.blend_color.unwrap_or([0.0; 4]),
        render_target_extent: req.render_target_extent,
        first_vertex: req.first_vertex,
        // Passed through. `decode::render`'s `wire_instance_count` is where
        // a zero instance count is decided, and it is decided once — a
        // second `.max(1)` here would re-apply that rule on this arm alone,
        // so a change made at the decode site would appear to take effect
        // everywhere while this path quietly kept the old answer.
        instance_count: Some(req.instance_count),
        primitive_topology: req.primitive_topology,
        raster_sample_count: pd.raster_sample_count.max(1),
        color_sample_count: req
            .colors
            .first()
            .map(|color| color.sample_count.max(1))
            .unwrap_or(1),
        multisample_resolve: req
            .colors
            .first()
            .is_some_and(|color| color.multisample_source_ref != 0),
        ..reims_vgpu_core::DrawRequest::default()
    };
    let resolving_colors = req
        .colors
        .iter()
        .filter(|color| color.multisample_source_ref != 0)
        .count();
    if resolving_colors != 0
        && (resolving_colors != 1
            || req
                .colors
                .first()
                .is_none_or(|color| color.multisample_source_ref == 0))
    {
        return Err(DrawPreparationDecline::MultisampleResolveShapeUnsupported {
            color_targets: req.colors.len() as u32,
            depth: req.depth_attach.is_some(),
            color_input: false,
        });
    }
    if let Some(color) = req
        .colors
        .first()
        .filter(|color| color.multisample_source_ref != 0)
    {
        if color.store_action != reims_vgpu_protocol::pass_action::StoreAction::MultisampleResolve {
            return Err(DrawPreparationDecline::MultisampleStoreActionUnsupported {
                store_action: color.store_action.guest_ordinal(),
            });
        }
        if color.load_action == reims_vgpu_protocol::pass_action::LoadAction::Load {
            return Err(DrawPreparationDecline::MultisampleLoadActionUnsupported {
                load_action: color.load_action.guest_ordinal(),
            });
        }
    }
    resources.viewports = req
        .viewports
        .iter()
        .map(|vp| reims_vgpu_core::ViewportResource {
            x: vp[0] as f32,
            y: vp[1] as f32,
            width: vp[2] as f32,
            height: vp[3] as f32,
            min_depth: vp[4] as f32,
            max_depth: vp[5] as f32,
        })
        .collect();
    resources.occlusion_query = req.visibility.map(|arming| arming.mode);
    resources.scissors = req
        .scissors
        .iter()
        .map(|s| reims_vgpu_core::ScissorResource {
            x: s.x,
            y: s.y,
            width: s.width,
            height: s.height,
        })
        .collect();
    if let Some(idx) = req.indexed.as_ref() {
        let index_type = idx
            .index_type
            .map_err(|_| DrawPreparationDecline::IndexLoad {
                reason: IndexLoadReason::TypeUnsupported,
            })?;
        let content = load_index_content_reason(state, host, req.task_id, idx)
            .map_err(|reason| DrawPreparationDecline::IndexLoad { reason })?;
        resources.indexed = Some(reims_vgpu_core::IndexedDrawResource {
            index_type,
            index_count: idx.index_count,
            // Vulkan's vertexOffset is a signed 32-bit field where Metal's
            // baseVertex is 64-bit, so a value that cannot fit is declined
            // rather than wrapped into an index somewhere else in the
            // buffer. The guest cannot express one: Apple's serializer
            // truncates baseVertex to 16 bits in the compact records and
            // this device's own decode is the only other source.
            vertex_offset: i32::try_from(idx.base_vertex).map_err(|_| {
                DrawPreparationDecline::IndexLoad {
                    reason: crate::runtime::draw::IndexLoadReason::BaseVertexOutOfRange,
                }
            })?,
            content,
        });
    }
    // Vulkan's `firstInstance` is Metal's `baseInstance`. The field has
    // always been here and always read 0, because nothing upstream decoded
    // the draw forms that carry one; the engine's Constant-step-rate vertex
    // prefix rebuild already reads it.
    resources.base_instance = req.base_instance;
    resources.vertex_attributes = attrs;
    resources.storage_buffers = storage;
    resources.sampled_images = images;
    resources.color_input = frag_color_input;
    resources.continues_render_pass = req.continues_render_pass;
    resources.render_pass_continues = req.render_pass_continues;
    resources.samplers = samplers;
    // Load seed always goes to the GPU (workstream D3). Premult One/OMSA is
    // hardware blend over the Load-seeded target — identical math to the
    // retired software `src + seed*(1-src.a)` path. Sampled alpha is
    // protocol data and must not be rewritten from an RGB content census;
    // content-gated keep-seed / alpha0-holes composites are retired.
    let store_is_store = req
        .colors
        .first()
        .map(|c| c.store_action.publishes_single_sample())
        .unwrap_or(true);
    resources.target_rgba8 = target_rgba8;
    resources.target_guest = target_guest;
    resources.target_clear = target_clear;
    resources.color_load_action = color_load_action;
    resources.target_seed_order = seed_order;
    // Start from the portable Store answer. Target planning may turn readback
    // off only while assigning the matching resident completion route; the
    // executor then reports whether that resident was guest-backed or copied.
    resources.skip_readback = !store_is_store;
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleTarget);
    let completion_route = plan_target_completion(
        state,
        req,
        &mut resources,
        gva_alloc_generation,
        gpu_only_content_allowed,
        store_is_store,
        writeback_guest,
        iosurface_texture_resident_target,
        chain_load_from_target,
        gva_load_identity.take(),
        w,
        h,
    )?;
    // The target plan already couples guest backing, LOAD content, resident
    // identity, and completion. No parallel front-frame policy is assembled
    // here.
    // The engine previously left `resources.blend = None`, selecting opaque replace for every
    // draw, so Load seeds (gray/wallpaper/logo bases) were wiped by sparse
    // dock/chrome layers that Metal would alpha-blend over the attachment.
    // Contract: render-pipeline color attachment blend tags (decode/resource).
    // Outside the `blending_enabled` guard below, and deliberately: an
    // unblended attachment with a mask still leaves its unwritten channels
    // alone, so gating the mask on blending would drop it exactly where the
    // guest is replacing rather than compositing.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
    resources.color_write_mask = pd.color0.write_mask;
    resources.blend = blend_states
        .iter()
        .find(|(slot, _)| *slot == 0)
        .map(|(_, state)| *state);

    // Preserve the decoded count. Non-indexed draws use it as their exact
    // invocation count; indexed draws are governed by `index_count`.
    let vertex_count = req.vertex_count;

    // Honor a bound NON-TRIVIAL depth-stencil state: attach a transient depth
    // buffer + enable the depth test. Decoded once per depth draw; the whole
    // 2D UI binds no depth-stencil (`depth_stencil_ref == 0`, 0 decodes), so
    // this is inert there. A trivial state (compare Always, no write, no
    // stencil) stays `None` — no depth attachment, byte-identical 2D path.
    // Descriptor resolution and enum normalization are fail-closed: a
    // bound state either becomes one semantic engine state or refuses the
    // draw by name. A trivial state remains the exact no-depth operation.
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleDepth);
    if req.depth_stencil_ref != 0 {
        let ds = load_depth_stencil_descriptor(state, host, req.task_id, req.depth_stencil_ref)
            .map_err(|detail| DrawPreparationDecline::DepthStencilStateMissing {
                task_id: req.task_id,
                state_ref: req.depth_stencil_ref,
                detail,
            })?;
        resources.depth = semantic_depth_state(&ds, req)?;
    }
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);
    resources.program = program;
    resources.width = w;
    resources.height = h;
    resources.vertex_count = vertex_count;
    if let Some(c0) = req.colors.first() {
        resources.color_attachment_format = Some(
            pixel_format::color_attachment_format_checked(c0.format)
                .map_err(|reason| DrawPreparationDecline::ColorAttachmentFormat { reason })?,
        );
    }
    // True MRT: render every color attachment (slot 1.. as engine secondary
    // residents) instead of dropping the shader's secondary outputs. Gated
    // on a resident primary; an `Ok(empty)` is the guest's own single-RT
    // draw and is byte-identical to the classic path.
    let mut secondary_targets_built = false;
    if let Some(primary_id) = resources.target_identity.clone() {
        let secs = build_secondary_targets(
            state,
            host,
            req.task_id,
            &req.colors,
            pd,
            &primary_id,
            &blend_states,
        );
        // A secondary attachment this device cannot build refuses the draw.
        // Executing it against slot 0 alone would render a frame whose
        // `location` 1.. outputs went nowhere, and nothing downstream — not
        // the guest, not this log — could tell that from a draw the guest
        // had only ever asked one target for.
        resources.secondary_targets =
            secs.map_err(
                |refusal| DrawPreparationDecline::SecondaryTargetUnbuildable {
                    pipeline_ref: req.pipeline_ref,
                    refusal,
                },
            )?;
        secondary_targets_built = req.colors.len() > 1;
    }

    Ok(ExecutorRequestPlan {
        request: resources,
        completion_route,
        vertex_count,
        secondary_targets_built,
    })
}
