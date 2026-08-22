//! Retained render-pipeline resolution and semantic executor admission.
//!
//! A successful plan carries the immutable pipeline, complete blend state, and
//! validated pass extent together. Bind and request planning cannot observe a
//! pipeline whose shader interface, sample contract, or geometry was refused.

use std::sync::Arc;

use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::Device;

use super::{DrawEncodeRequest, DrawPreparationDecline};

pub(super) struct PipelinePlan {
    pub resolved: Arc<reims_vgpu_core::ResolvedRenderPipeline>,
    pub blend_states: Vec<(u32, reims_vgpu_core::BlendStateResource)>,
    pub width: u32,
    pub height: u32,
}

pub(super) fn plan_pipeline<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    request: &DrawEncodeRequest,
) -> Result<Option<PipelinePlan>, DrawPreparationDecline> {
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::PipelineDesc);
    let resolved = crate::runtime::pipeline_resolve::resolve(
        state,
        host,
        request.task_id,
        request.pipeline_ref,
    )?;
    let blend_states = super::semantic_blend_states(&resolved.desc)?;

    let pipeline_sample_count = resolved.desc.raster_sample_count.max(1);
    observe_multisample_contract(request, pipeline_sample_count);
    if let Some(color) = request
        .colors
        .iter()
        .find(|color| color.sample_count != pipeline_sample_count)
    {
        return Err(
            DrawPreparationDecline::MultisampleAttachmentSampleCountMismatch {
                attachment: color.sample_count,
                raster: pipeline_sample_count,
            },
        );
    }

    for (stage, shader, textures) in [
        ("vertex", &resolved.vertex, &request.vertex_textures),
        ("fragment", &resolved.fragment, &request.fragment_textures),
    ] {
        if let Some((index, descriptor)) = shader.interface.first_non_sampled_texture_descriptor() {
            let access = match descriptor.access {
                reims_vgpu_core::ReflectedTextureAccess::Storage => "storage",
                reims_vgpu_core::ReflectedTextureAccess::Unknown => "unknown",
                reims_vgpu_core::ReflectedTextureAccess::Sampled => continue,
            };
            return Err(DrawPreparationDecline::TextureAccessUnsupported {
                stage,
                index,
                texture_ref: textures
                    .iter()
                    .find(|texture| texture.index == index)
                    .map(|texture| texture.texture_ref)
                    .unwrap_or(0),
                binding: descriptor.binding,
                access,
            });
        }
    }

    for (stage, expected_stage, shader) in [
        (
            "vertex",
            reims_vgpu_core::ReflectedShaderStage::Vertex,
            &resolved.vertex,
        ),
        (
            "fragment",
            reims_vgpu_core::ReflectedShaderStage::Fragment,
            &resolved.fragment,
        ),
    ] {
        if let Some(unsupported) = shader.interface.first_unsupported_interface(expected_stage) {
            return Err(DrawPreparationDecline::ReflectedInterfaceUnsupported {
                stage,
                feature: unsupported.feature,
                count: unsupported.count,
            });
        }
        if let Some(resource) = shader.interface.first_unsupported_resource() {
            let kind = resource
                .kind
                .unsupported_vulkan_name()
                .expect("helper returned an unsupported Vulkan resource");
            return Err(DrawPreparationDecline::ReflectedResourceUnsupported {
                stage,
                index: resource.metal_index,
                binding: resource.descriptor.map(|descriptor| descriptor.binding),
                kind,
            });
        }
    }

    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Pipeline);
    let Some((width, height)) = request
        .colors
        .first()
        .map(|color| (color.width, color.height))
    else {
        return Ok(None);
    };
    let max_dimension = state.executor.capabilities().max_render_target_dimension;
    if width == 0 || height == 0 || width > max_dimension || height > max_dimension {
        return Err(DrawPreparationDecline::GeometryUnsupported { width, height });
    }

    Ok(Some(PipelinePlan {
        resolved,
        blend_states,
        width,
        height,
    }))
}

fn observe_multisample_contract(request: &DrawEncodeRequest, pipeline_sample_count: u32) {
    if pipeline_sample_count <= 1 {
        return;
    }
    let color = request.colors.first();
    let key = (u64::from(request.pipeline_ref) << 32)
        | u64::from(color.map_or(0, |color| color.texture_ref));
    if !crate::observe::first_sight("render_multisample_contract", key) {
        return;
    }
    crate::observe::off(format!(
        "render_multisample_contract task={} pipe={} raster_samples={} \
         colors={} color_ref={} source_ref={} mid={} gva={:#x} {}x{} \
         fmt={:#x} load={} store={} depth_ref={}",
        request.task_id,
        request.pipeline_ref,
        pipeline_sample_count,
        request.colors.len(),
        color.map_or(0, |color| color.texture_ref),
        color.map_or(0, |color| color.multisample_source_ref),
        color.map_or(0, |color| color.mapping_id()),
        color.map_or(0, |color| color.target_gva()),
        color.map_or(0, |color| color.width),
        color.map_or(0, |color| color.height),
        color.map_or(0, |color| color.format),
        color.map_or(0, |color| color.load_action.guest_ordinal()),
        color.map_or(0, |color| color.store_action.guest_ordinal()),
        request
            .depth_attach
            .as_ref()
            .map_or(0, |depth| depth.texture_ref),
    ));
}
