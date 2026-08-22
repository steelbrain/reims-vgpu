//! Semantic fixed-function draw state.
//!
//! Raw guest ordinals terminate here. Execution receives complete blend,
//! depth, and stencil state or a typed preparation refusal.

use super::{depth_chain_identity, DrawEncodeRequest, DrawPreparationDecline};

/// Whether one present face makes the native state read or write stencil.
///
/// Across all 16,384 combinations of compare function, three operations, and
/// zero/full masks, native state reports stencil use exactly when a nontrivial
/// comparison can read through `readMask`, or a non-Keep operation can write
/// through `writeMask`. Face presence alone is not enablement: both default
/// faces are present and inert.
fn stencil_face_has_effect(
    present: bool,
    face: &crate::runtime::decode::resource::DepthStencilFace,
) -> bool {
    if !present {
        return false;
    }
    let writes = face.write_mask != 0
        && (face.stencil_failure_operation != 0
            || face.depth_failure_operation != 0
            || face.depth_stencil_pass_operation != 0);
    let compares = face.read_mask != 0 && face.compare_function != 0 && face.compare_function != 7;
    writes || compares
}

pub(super) fn depth_stencil_descriptor_is_trivial(
    descriptor: &crate::runtime::decode::resource::DepthStencilDescriptor,
) -> bool {
    const MTL_COMPARE_ALWAYS: u32 = 7;
    descriptor.depth_compare_function == MTL_COMPARE_ALWAYS
        && !descriptor.depth_write_enabled
        && !stencil_face_has_effect(descriptor.front_stencil_present, &descriptor.front_face)
        && !stencil_face_has_effect(descriptor.back_stencil_present, &descriptor.back_face)
}

fn stencil_face(
    face: &crate::runtime::decode::resource::DepthStencilFace,
) -> Result<reims_vgpu_core::StencilFaceOps, reims_vgpu_protocol::PipelineStateDecodeError> {
    Ok(reims_vgpu_core::StencilFaceOps {
        compare: reims_vgpu_protocol::compare_function(face.compare_function)?,
        fail_op: reims_vgpu_protocol::stencil_operation(face.stencil_failure_operation)?,
        depth_fail_op: reims_vgpu_protocol::stencil_operation(face.depth_failure_operation)?,
        pass_op: reims_vgpu_protocol::stencil_operation(face.depth_stencil_pass_operation)?,
        read_mask: face.read_mask,
        write_mask: face.write_mask,
    })
}

fn stencil_face_ops_has_effect(face: reims_vgpu_core::StencilFaceOps) -> bool {
    use reims_vgpu_core::{SamplerCompareFunction, StencilOp};

    let writes = face.write_mask != 0
        && (face.fail_op != StencilOp::Keep
            || face.depth_fail_op != StencilOp::Keep
            || face.pass_op != StencilOp::Keep);
    let compares = face.read_mask != 0
        && !matches!(
            face.compare,
            SamplerCompareFunction::Never | SamplerCompareFunction::Always
        );
    writes || compares
}

pub(super) fn semantic_blend_state(
    attachment: &reims_vgpu_protocol::resource::PipelineColorAttachment,
) -> Result<reims_vgpu_core::BlendStateResource, reims_vgpu_protocol::PipelineStateDecodeError> {
    reims_vgpu_protocol::blend_state(attachment)
}

pub(super) fn semantic_blend_states(
    pipeline: &reims_vgpu_protocol::resource::RenderPipelineDescriptor,
) -> Result<Vec<(u32, reims_vgpu_core::BlendStateResource)>, DrawPreparationDecline> {
    pipeline
        .color_attachments
        .iter()
        .filter(|attachment| attachment.blending_enabled)
        .map(|attachment| {
            semantic_blend_state(attachment)
                .map(|state| (attachment.slot, state))
                .map_err(|reason| DrawPreparationDecline::BlendState { reason })
        })
        .collect()
}

pub(super) fn semantic_depth_state(
    descriptor: &crate::runtime::decode::resource::DepthStencilDescriptor,
    request: &DrawEncodeRequest,
) -> Result<Option<reims_vgpu_core::DepthState>, DrawPreparationDecline> {
    use reims_vgpu_core::{SamplerCompareFunction, StencilFaceOps, StencilOp, StencilState};

    let compare = reims_vgpu_protocol::compare_function(descriptor.depth_compare_function)
        .map_err(|reason| DrawPreparationDecline::DepthCompare { reason })?;
    let front = descriptor
        .front_stencil_present
        .then(|| stencil_face(&descriptor.front_face))
        .transpose()
        .map_err(|reason| DrawPreparationDecline::StencilState {
            face: "front",
            reason,
        })?;
    let back = descriptor
        .back_stencil_present
        .then(|| stencil_face(&descriptor.back_face))
        .transpose()
        .map_err(|reason| DrawPreparationDecline::StencilState {
            face: "back",
            reason,
        })?;
    if depth_stencil_descriptor_is_trivial(descriptor) {
        return Ok(None);
    }
    let (clear_value, load_action) = request
        .depth_attach
        .as_ref()
        .map(|depth| (depth.clear_depth as f32, depth.load_action))
        .unwrap_or((1.0, reims_vgpu_protocol::pass_action::LoadAction::Clear));
    let stencil = if front.is_some_and(stencil_face_ops_has_effect)
        || back.is_some_and(stencil_face_ops_has_effect)
    {
        const PASS_THROUGH: StencilFaceOps = StencilFaceOps {
            compare: SamplerCompareFunction::Always,
            fail_op: StencilOp::Keep,
            depth_fail_op: StencilOp::Keep,
            pass_op: StencilOp::Keep,
            read_mask: u32::MAX,
            write_mask: u32::MAX,
        };
        let front = front.unwrap_or(PASS_THROUGH);
        let back = back.unwrap_or(PASS_THROUGH);
        let (reference_front, reference_back) = request.stencil_ref.unwrap_or((0, 0));
        let clear_value = request
            .stencil_attach
            .as_ref()
            .map(|stencil| stencil.clear_stencil)
            .unwrap_or(0);
        Some(StencilState {
            front,
            back,
            reference_front,
            reference_back,
            clear_value,
        })
    } else {
        None
    };

    Ok(Some(reims_vgpu_core::DepthState {
        identity: depth_chain_identity(request, stencil.is_some()),
        test_enable: true,
        write_enable: descriptor.depth_write_enabled,
        compare,
        clear_value,
        load: load_action == reims_vgpu_protocol::pass_action::LoadAction::Load,
        stencil,
    }))
}
