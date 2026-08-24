//! Complete executor resource planning for one validated pipeline.
//!
//! Direct buffers, reflected shader layouts, sampled images, and samplers cross
//! back into orchestration only as one complete set. Internal null-descriptor
//! obligations cannot be consumed independently.

use super::*;

pub(super) struct DrawResourcePlan {
    pub attributes: Vec<reims_vgpu_core::VertexAttributeResource>,
    pub storage_buffers: Vec<reims_vgpu_core::StorageBufferResource>,
    pub sampled_images: Vec<reims_vgpu_core::SampledImageResource>,
    pub samplers: Vec<reims_vgpu_core::SamplerResource>,
    pub sampler_provenance: std::collections::BTreeMap<u32, u8>,
    pub vertex_variant: reims_vgpu_core::PreparedShaderVariant,
    pub fragment_variant: reims_vgpu_core::PreparedShaderVariant,
    pub fragment_color_input: bool,
}

pub(super) fn plan_draw_resources<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    request: &DrawEncodeRequest,
    resolved: &reims_vgpu_core::ResolvedRenderPipeline,
    gva_allocation_generation: u64,
    width: u32,
    height: u32,
) -> Result<DrawResourcePlan, DrawPreparationDecline> {
    let shader_resource_plan::ShaderResourcePlan {
        attributes,
        storage_buffers,
        mut vertex_variant,
        mut fragment_variant,
        fragment_null_textures,
        fragment_color_input,
    } = shader_resource_plan::plan_shader_resources(state, host, request, resolved, width, height)?;
    let sampled_images = texture_plan::plan_sampled_textures(
        state,
        host,
        request,
        &resolved.vertex,
        &resolved.fragment,
        &vertex_variant,
        &fragment_variant,
        &fragment_null_textures,
        gva_allocation_generation,
    )?;
    let sampler_plan::SamplerPlan {
        resources: samplers,
        provenance: sampler_provenance,
    } = sampler_plan::plan_samplers(state, host, request, &vertex_variant, &fragment_variant)?;
    vertex_variant = state
        .executor
        .specialize_render_samplers(&vertex_variant, &samplers)
        .map_err(|reason| DrawPreparationDecline::VertexTranslate {
            pipeline_ref: request.pipeline_ref,
            reason,
        })?;
    fragment_variant = state
        .executor
        .specialize_render_samplers(&fragment_variant, &samplers)
        .map_err(|reason| DrawPreparationDecline::FragmentTranslate {
            pipeline_ref: request.pipeline_ref,
            reason,
        })?;

    Ok(DrawResourcePlan {
        attributes,
        storage_buffers,
        sampled_images,
        samplers,
        sampler_provenance,
        vertex_variant,
        fragment_variant,
        fragment_color_input,
    })
}
