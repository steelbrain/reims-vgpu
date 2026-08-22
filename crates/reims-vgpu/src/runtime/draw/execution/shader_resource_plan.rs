//! Shader numbering and directly-bound resource occupancy for one draw.
//!
//! Buffer loading, reflected storage binding projection, descriptor gap
//! classification, and framebuffer-fetch admission are one decision. A
//! consumer receives variants and resources which were validated together.

use super::*;

pub(super) struct ShaderResourcePlan {
    pub attributes: Vec<reims_vgpu_core::VertexAttributeResource>,
    pub storage_buffers: Vec<reims_vgpu_core::StorageBufferResource>,
    pub vertex_variant: reims_vgpu_core::PreparedShaderVariant,
    pub fragment_variant: reims_vgpu_core::PreparedShaderVariant,
    pub fragment_null_textures: Vec<u32>,
    pub fragment_color_input: bool,
}

pub(super) fn plan_shader_resources<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    request: &DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    width: u32,
    height: u32,
) -> Result<ShaderResourcePlan, DrawPreparationDecline> {
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Binds);
    crate::runtime::bind_phase::note_bind();
    let bind_plan::BoundBufferPlan {
        vertex,
        fragment,
        attributes,
        stage_in_buffers,
    } = bind_plan::plan_bound_buffers(state, host, request, resolved)?;

    let vertex_variant = resolved.vertex.variant().clone();
    let fragment_variant = resolved.fragment.variant().clone();

    let mut storage_buffers = Vec::new();
    for (index, content) in &vertex {
        if !vertex_buffer_needs_storage_binding(
            &resolved.vertex.interface,
            *index,
            stage_in_buffers.contains(index),
        ) {
            continue;
        }
        storage_buffers.push(reims_vgpu_core::StorageBufferResource {
            binding: *index,
            content: content.clone(),
        });
    }
    for (index, content) in &fragment {
        storage_buffers.push(reims_vgpu_core::StorageBufferResource {
            binding: fragment_variant.buffer_binding(*index),
            content: content.clone(),
        });
    }

    let unbound = frag_unbound_scan(
        &resolved.fragment.interface.bindings,
        |index| fragment.iter().any(|(bound, _)| *bound == index),
        |index| {
            request
                .fragment_textures
                .iter()
                .any(|texture| texture.index == index && texture.texture_ref != 0)
        },
        |index| {
            request
                .fragment_samplers
                .iter()
                .any(|sampler| sampler.index == index && sampler.sampler_ref != 0)
        },
        |index| fragment_variant.declares_descriptor(fragment_variant.texture_binding(index, None)),
    );
    let uses = unbound
        .iter()
        .map(|gap| (*gap, frag_unbound_static_use(gap, &fragment_variant)))
        .collect::<Vec<_>>();
    for (_, use_) in &uses {
        crate::runtime::drain::note_store_route(use_.slug());
    }
    let reportable = uses
        .iter()
        .copied()
        .filter(|(gap, use_)| !(gap.class == FragUnboundClass::Texture && use_.is_violation()))
        .collect::<Vec<_>>();
    if !reportable.is_empty() {
        report_fragment_gaps(request, &fragment, &reportable, width, height);
    }
    let fragment_null_textures = frag_unbound_textures_to_bind_null(&uses);

    let mut fragment_color_input = false;
    for binding in &resolved.fragment.interface.bindings {
        if binding.kind != reims_vgpu_core::ShaderResourceKind::ColorInput {
            continue;
        }
        if binding.metal_index != 0 {
            return Err(DrawPreparationDecline::ColorInputMrtUnsupported {
                destination_index: binding.metal_index,
            });
        }
        fragment_color_input = true;
    }

    Ok(ShaderResourcePlan {
        attributes,
        storage_buffers,
        vertex_variant,
        fragment_variant,
        fragment_null_textures,
        fragment_color_input,
    })
}

fn report_fragment_gaps(
    request: &DrawEncodeRequest,
    fragment: &[(u32, reims_vgpu_core::BufferContent)],
    uses: &[(FragUnbound, reims_vgpu_core::DescriptorUse)],
    width: u32,
    height: u32,
) {
    let buffers = fragment
        .iter()
        .map(|(index, _)| *index)
        .collect::<std::collections::BTreeSet<_>>();
    let textures = request
        .fragment_textures
        .iter()
        .filter(|texture| texture.texture_ref != 0)
        .map(|texture| texture.index)
        .collect::<std::collections::BTreeSet<_>>();
    let samplers = request
        .fragment_samplers
        .iter()
        .filter(|sampler| sampler.sampler_ref != 0)
        .map(|sampler| sampler.index)
        .collect::<std::collections::BTreeSet<_>>();
    let detail = uses
        .iter()
        .map(|(gap, use_)| format!("{gap}:{}", use_.slug()))
        .collect::<Vec<_>>()
        .join(",");
    let violations = uses.iter().filter(|(_, use_)| use_.is_violation()).count();
    crate::observe::fail(format!(
        "shader_resource_declared_unbound reason=frag_declared_descriptor_unbound \
         pipe={} unbound=[{detail}] violations={violations}/{} \
         provided_buf={buffers:?} provided_tex={textures:?} \
         provided_smp={samplers:?} {width}x{height}",
        request.pipeline_ref,
        uses.len(),
    ));
}
