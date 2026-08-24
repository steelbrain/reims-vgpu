//! Buffer and vertex stage-in planning for one resolved draw.
//!
//! Guest buffer identities, reflected access, zero-copy eligibility, dynamic
//! stride, and vertex-step semantics are resolved together. Execution receives
//! a complete resource set or one typed draw-preparation refusal.

use super::*;

pub(super) struct BoundBufferPlan {
    pub(super) vertex: Vec<(u32, reims_vgpu_core::BufferContent)>,
    pub(super) fragment: Vec<(u32, reims_vgpu_core::BufferContent)>,
    pub(super) attributes: Vec<reims_vgpu_core::VertexAttributeResource>,
    pub(super) stage_in_buffers: std::collections::BTreeSet<u32>,
}

/// Whether this draw has an executable consumer for a bound buffer.
///
/// Static use comes from the translated module, independently of source
/// reflection. A vertex stage-in fetch is the other consumer and is declared
/// by the pipeline descriptor rather than by a direct shader argument.
fn buffer_bind_needs_content(
    executable_use: reims_vgpu_core::DescriptorUse,
    feeds_stage_in: bool,
) -> bool {
    feeds_stage_in
        || matches!(
            executable_use,
            reims_vgpu_core::DescriptorUse::Used | reims_vgpu_core::DescriptorUse::Ambiguous
        )
}

pub(super) fn plan_bound_buffers<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
) -> Result<BoundBufferPlan, DrawPreparationDecline> {
    let pd = &resolved.desc;
    let v_shader = &resolved.vertex;
    let f_shader = &resolved.fragment;

    // Materialize stream buffer binds (vertex + fragment). Large spans
    // ride the zero-copy rail (the GPU gathers them from imported guest
    // RAM at execute time); the rest stay on the CPU staging read.
    // Constant-step attribute streams stay CPU: the engine prepends a
    // base-instance prefix to those bytes at prepare time.
    //
    // Which indices those are, and which the attribute list names at all,
    // are both functions of the pipeline's attribute list and nothing else,
    // so they are resolved with the pipeline rather than rebuilt per draw —
    // see [`reims_vgpu_core::VertexBindPlan`], which also
    // carries why the second set is deliberately unfiltered. Not to be
    // confused with `stage_in_bufs` further down: that one is filled during
    // the attribute walk, holds only the indices that actually carried
    // bytes, and decides storage binding.
    let bind_plan = &resolved.bind_plan;
    let mut vtx_storage: Vec<(u32, reims_vgpu_core::BufferContent)> = Vec::new();
    // The three `bind_phase` spans below divide `chain_phase`'s `binds_us`,
    // which is this draw path's largest column and covered three costs with
    // one number. Each is a lexical scope so an early `return Err` charges
    // the span it left from rather than losing the time.
    let vertex_span =
        crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::VertexLoad);
    for b in req.vertex_buffers.iter() {
        if b.buffer_ref == 0 {
            continue;
        }
        let allow_zc = !bind_plan.is_constant_step(b.index);
        // A vertex buffer is read twice on this path — as the declared
        // argument reflection describes, and as the byte source for every
        // stage-in attribute naming this index, which it does not. Only the
        // first is what `Unused` is about, so an index the pipeline's
        // attribute list names keeps its guest bytes whatever reflection
        // says about the argument.
        let feeds_stage_in = bind_plan.feeds_stage_in(b.index);
        let executable_use = v_shader.variant().buffer_use(b.index);
        let access = v_shader.interface.buffer_access(b.index);
        crate::runtime::bind_phase::note_access(access);
        if !buffer_bind_needs_content(executable_use, feeds_stage_in) {
            crate::runtime::drain::note_store_route("render_buffer_executable_unused");
            continue;
        }
        // The vertex shader's own reflection bounds its own `[[buffer(n)]]`
        // binds, and a stage-in index is excluded — see that function's doc
        // for why the exclusion is not implied by the translator's output.
        let cap = state.executor.render_buffer_extent(
            &v_shader.interface,
            b.index,
            feeds_stage_in,
            req.first_vertex,
            req.vertex_count,
            req.base_instance,
            req.instance_count,
            req.indexed.is_some(),
        );
        crate::runtime::bind_phase::note_unused_staged(access);
        let Some(content) = load_buffer_content(
            state,
            host,
            req.task_id,
            b.buffer_ref,
            b.resource.as_deref(),
            b.offset,
            allow_zc,
            cap,
        ) else {
            return Err(DrawPreparationDecline::VertexBufferMissing {
                index: b.index,
                buffer_ref: b.buffer_ref,
                offset: b.offset,
            });
        };
        vtx_storage.push((b.index, content));
    }
    drop(vertex_span);
    let mut frag_storage: Vec<(u32, reims_vgpu_core::BufferContent)> = Vec::new();
    let fragment_span =
        crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::FragmentLoad);
    for b in req.fragment_buffers.iter() {
        if b.buffer_ref == 0 {
            continue;
        }
        let executable_use = f_shader.variant().buffer_use(b.index);
        let access = f_shader.interface.buffer_access(b.index);
        crate::runtime::bind_phase::note_access(access);
        if !buffer_bind_needs_content(executable_use, false) {
            crate::runtime::drain::note_store_route("render_buffer_executable_unused");
            continue;
        }
        // The fragment shader's reflection, for the same reason. The two
        // stages are looked up separately because one Metal buffer index
        // names a different argument in each, and a cap taken from the
        // wrong stage would bound a bind the other stage never declared.
        let cap = state.executor.render_buffer_extent(
            &f_shader.interface,
            b.index,
            true,
            req.first_vertex,
            req.vertex_count,
            req.base_instance,
            req.instance_count,
            req.indexed.is_some(),
        );
        // No stage-in exclusion here: `[[stage_in]]` is a vertex-stage
        // concept and `pd.vertex_attributes` names vertex buffer indices,
        // which are a different index space from the fragment stage's.
        crate::runtime::bind_phase::note_unused_staged(access);
        // Zero-copy, for the same reason the vertex loop above allows it and
        // with none of that loop's one exclusion. A fragment bind is the
        // guest's buffer as the guest bound it: nothing on this path prepends
        // a prefix to the bytes or otherwise needs to own a mutable copy, and
        // `is_constant_step` — the only reason a vertex index is held back —
        // is a stage-in concept the fragment index space does not have. Both
        // stages' binds land in the same `storage_buffers` vector, so the
        // descriptor side already consumes imported content.
        //
        // The call is a ladder: an import that cannot be formed falls to the
        // CPU read that used to be unconditional here, so this widens which
        // rung a bind reaches and cannot change what the shader sees.
        let Some(content) = load_buffer_content(
            state,
            host,
            req.task_id,
            b.buffer_ref,
            b.resource.as_deref(),
            b.offset,
            true,
            cap,
        ) else {
            return Err(DrawPreparationDecline::FragmentBufferMissing {
                index: b.index,
                buffer_ref: b.buffer_ref,
                offset: b.offset,
            });
        };
        frag_storage.push((b.index, content));
    }
    drop(fragment_span);
    // Stage-in attributes from pipeline vertex block + bound buffer bytes.
    let mut attrs: Vec<reims_vgpu_core::VertexAttributeResource> = Vec::new();
    let mut stage_in_bufs: std::collections::BTreeSet<u32> = Default::default();
    let attrs_span =
        crate::runtime::bind_phase::Span::open(crate::runtime::bind_phase::Part::Attrs);
    for a in &pd.vertex_attributes {
        // `setVertexBuffer:offset:attributeStride:atIndex:` overrides what
        // the pipeline's `MTLVertexBufferLayoutDescriptor` declared for this
        // buffer index, so it is resolved before the stride is read — a
        // pipeline built for a dynamic stride declares one this device
        // cannot use, and the guard below would drop the attribute for it.
        //
        // On this backend the stride reaches the pipeline through
        // `AttrKey::stride`, which is already part of the key: Vulkan's
        // per-binding stride is `VkVertexInputBindingDescription::stride`
        // and is not dynamic below `vkCmdBindVertexBuffers2`, core in 1.3
        // against this device's 1.2 floor. So two draws sharing shaders and
        // differing only in a guest-supplied stride already get their own
        // pipelines, with no change to the key.
        let stride =
            super::super::bind_attribute_stride(&req.vertex_buffers, a.buffer_index, a.stride);
        if a.format == 0 || stride == 0 {
            continue;
        }
        let format = prepare_vertex_attribute_format(a)?;
        let content = vtx_storage
            .iter()
            .find(|(idx, _)| *idx == a.buffer_index)
            .map(|(_, d)| d.clone())
            .unwrap_or_else(|| reims_vgpu_core::BufferContent::from(Vec::new()));
        if !content.is_empty() {
            stage_in_bufs.insert(a.buffer_index);
        } else if a.format != 0 {
            // Pipeline declares stage-in but stream did not bind bytes — fail
            // visibly rather than raster black garbage that wipes CLEAR.
            return Err(DrawPreparationDecline::StageInBytesMissing {
                location: a.location,
                buffer_index: a.buffer_index,
                raw_format: a.format,
                stride,
            });
        }
        let step = prepare_vertex_step_function(a)?;
        let step_rate = a.step_rate();
        attrs.push(reims_vgpu_core::VertexAttributeResource {
            location: a.location,
            // One Vulkan binding per location (archive render_draw_core).
            binding: a.location,
            format,
            offset: a.offset,
            stride,
            step_function: step,
            step_rate,
            content,
        });
    }
    drop(attrs_span);

    Ok(BoundBufferPlan {
        vertex: vtx_storage,
        fragment: frag_storage,
        attributes: attrs,
        stage_in_buffers: stage_in_bufs,
    })
}

#[cfg(test)]
mod tests {
    use super::buffer_bind_needs_content;
    use reims_vgpu_core::DescriptorUse;

    #[test]
    fn only_an_executable_or_stage_in_consumer_needs_buffer_content() {
        assert!(!buffer_bind_needs_content(
            DescriptorUse::NotDeclared,
            false
        ));
        assert!(!buffer_bind_needs_content(
            DescriptorUse::DeclaredUnused,
            false
        ));
        assert!(buffer_bind_needs_content(DescriptorUse::Used, false));
        assert!(buffer_bind_needs_content(DescriptorUse::Ambiguous, false));
        assert!(buffer_bind_needs_content(
            DescriptorUse::DeclaredUnused,
            true
        ));
    }
}
