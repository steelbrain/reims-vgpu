//! Render encode path: PSO cache, stage-in, textured, fixed-function state.

use crate::backend::blob::BlobKey;
use crate::backend::hash::hash_bytes;
use crate::backend::metal::abi::*;
use crate::backend::metal::cache::{
    depth_stencil_insert, depth_stencil_lookup, render_pso_insert, render_pso_lookup,
    DepthStencilKey,
};
use crate::backend::metal::constants::*;
use crate::backend::metal::format::mtl_pixel_format_bpp;
use crate::backend::metal::function::load_only_function;
use crate::backend::metal::mtl_enum;
use crate::backend::metal::raw_metal::{
    command_buffer_error_description, render_reflection_sampler_mask,
};
use crate::backend::metal::runtime::{new_buffer_from_host, system_device, thread_queue};
use crate::backend::metal::samplers::{make_default_sampler, make_explicit_sampler};
use crate::backend::metal::util::{
    bytes_of, clear_err, sampler_index, set_err, texture_index, valid_buffer_binding, ErrOut,
    Status,
};
use crate::backend::render_pso_key::{RenderPsoKey, RenderPsoLookup};
use crate::contract::extent::tight_image_bytes;
use crate::contract::vertex_step::{step_rate_in_contract, MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE};
use crate::runtime::decode::resource::MTL_COLOR_WRITE_MASK_ALL;
use foreign_types::ForeignType;
use metal::*;
use std::ptr;

struct AttrBufferSlot {
    data: *const u8,
    len: usize,
    stride: u32,
    step_function: u32,
    step_rate: u32,
    index: u64,
    buffer: Buffer,
}

fn apply_blend(
    color: &RenderPipelineColorAttachmentDescriptorRef,
    blend: Option<&ReimsVgpuBlendState>,
    err: ErrOut<'_>,
) -> Status {
    let Some(blend) = blend.filter(|b| b.enable != 0) else {
        return Status::OK;
    };
    // Each factor and operation is validated by being converted, so the
    // accepted set is `MTLBlendFactor`'s own variant list rather than a ceiling
    // written beside it. That matters here specifically: the ceiling this used
    // to carry was `MTLBlendFactor::OneMinusBlendAlpha as u32`, which was the
    // last variant when Metal shipped and is now the fifteenth of nineteen, so
    // the four dual-source factors (`Source1Color` .. `OneMinusSource1Alpha`,
    // 15-18) were refused device-wide. Every sibling bound in this file names
    // its enum's actual last variant; this one had drifted.
    //
    // The six slugs stay one per field: which of the six words the guest got
    // wrong is the whole content of the refusal, and a shared slug would report
    // "a blend state was refused" for six different guest mistakes.
    macro_rules! factor {
        ($field:ident, $slug:literal) => {
            match mtl_enum::blend_factor(blend.$field) {
                Some(v) => v,
                None => {
                    set_err(err, "unsupported Metal blend state");
                    return Status::args($slug).field("value", blend.$field);
                }
            }
        };
    }
    macro_rules! operation {
        ($field:ident, $slug:literal) => {
            match mtl_enum::blend_operation(blend.$field) {
                Some(v) => v,
                None => {
                    set_err(err, "unsupported Metal blend state");
                    return Status::args($slug).field("value", blend.$field);
                }
            }
        };
    }
    let src_rgb = factor!(src_rgb, "metal_render_blend_src_rgb_unsupported");
    let dst_rgb = factor!(dst_rgb, "metal_render_blend_dst_rgb_unsupported");
    let src_alpha = factor!(src_alpha, "metal_render_blend_src_alpha_unsupported");
    let dst_alpha = factor!(dst_alpha, "metal_render_blend_dst_alpha_unsupported");
    let op_rgb = operation!(op_rgb, "metal_render_blend_rgb_operation_unsupported");
    let op_alpha = operation!(op_alpha, "metal_render_blend_alpha_operation_unsupported");
    color.set_blending_enabled(true);
    color.set_source_rgb_blend_factor(src_rgb);
    color.set_destination_rgb_blend_factor(dst_rgb);
    color.set_rgb_blend_operation(op_rgb);
    color.set_source_alpha_blend_factor(src_alpha);
    color.set_destination_alpha_blend_factor(dst_alpha);
    color.set_alpha_blend_operation(op_alpha);
    Status::OK
}

fn apply_blend_color(encoder: &RenderCommandEncoderRef, blend: Option<&ReimsVgpuBlendState>) {
    if let Some(blend) = blend {
        if blend.enable != 0 && blend.has_blend_color != 0 {
            encoder.set_blend_color(
                blend.blend_color[0],
                blend.blend_color[1],
                blend.blend_color[2],
                blend.blend_color[3],
            );
        }
    }
}

fn apply_raster_state(
    encoder: &RenderCommandEncoderRef,
    raster: Option<&ReimsVgpuRasterState>,
    err: ErrOut<'_>,
) -> Status {
    let Some(raster) = raster else {
        return Status::OK;
    };
    // Convert every word the record carries before touching the encoder, the
    // way the range checks this replaced all ran first: a refusal on the
    // winding must not leave the cull mode already applied.
    //
    // Each `has_` flag guards its own conversion, so a word the guest never set
    // is not read at all — converting unconditionally would refuse a raster
    // record for a field it does not carry.
    macro_rules! optional {
        ($has:ident, $field:ident, $convert:ident, $slug:literal, $name:literal) => {
            if raster.$has != 0 {
                match mtl_enum::$convert(raster.$field) {
                    Some(v) => Some(v),
                    None => {
                        set_err(err, "unsupported Metal raster state");
                        return Status::args($slug).field($name, raster.$field);
                    }
                }
            } else {
                None
            }
        };
    }
    let cull = optional!(
        has_cull_mode,
        cull_mode,
        cull_mode,
        "metal_render_cull_mode_unsupported",
        "cull_mode"
    );
    let front_facing = optional!(
        has_front_facing_winding,
        front_facing_winding,
        winding,
        "metal_render_winding_unsupported",
        "winding"
    );
    let fill = optional!(
        has_fill_mode,
        fill_mode,
        fill_mode,
        "metal_render_fill_mode_unsupported",
        "fill_mode"
    );
    let depth_clip = optional!(
        has_depth_clip_mode,
        depth_clip_mode,
        depth_clip_mode,
        "metal_render_depth_clip_mode_unsupported",
        "depth_clip_mode"
    );
    if let Some(cull) = cull {
        encoder.set_cull_mode(cull);
    }
    if let Some(front_facing) = front_facing {
        encoder.set_front_facing_winding(front_facing);
    }
    // Neither of these carries a host capability the way their Vulkan
    // spellings do: `MTLTriangleFillMode` and `MTLDepthClipMode` are plain
    // encoder state on every Metal device, so a converted value always
    // applies.
    if let Some(fill) = fill {
        encoder.set_triangle_fill_mode(fill);
    }
    if let Some(depth_clip) = depth_clip {
        encoder.set_depth_clip_mode(depth_clip);
    }
    Status::OK
}

fn apply_depth_bias(
    encoder: &RenderCommandEncoderRef,
    depth_bias: Option<&ReimsVgpuDepthBiasState>,
) {
    if let Some(db) = depth_bias {
        encoder.set_depth_bias(db.depth_bias, db.slope_scale, db.clamp);
    }
}

/// One stencil face's four enum words, already converted.
///
/// The validity check and the application used to be separate functions over
/// the same raw struct — one comparing four ordinals against a ceiling, the
/// other transmuting the same four. Carrying the converted values instead means
/// the check *is* the conversion, so there is no way to apply a face that was
/// not checked and no second copy of the accepted set.
struct StencilFace {
    compare: MTLCompareFunction,
    stencil_fail: MTLStencilOperation,
    depth_fail: MTLStencilOperation,
    pass: MTLStencilOperation,
}

fn stencil_face_converted(face: &ReimsVgpuDepthStencilFaceState) -> Option<StencilFace> {
    Some(StencilFace {
        compare: mtl_enum::compare_function(face.compare_function)?,
        stencil_fail: mtl_enum::stencil_operation(face.stencil_failure_operation)?,
        depth_fail: mtl_enum::stencil_operation(face.depth_failure_operation)?,
        pass: mtl_enum::stencil_operation(face.depth_stencil_pass_operation)?,
    })
}

fn apply_stencil_face(
    dst: &StencilDescriptorRef,
    src: &ReimsVgpuDepthStencilFaceState,
    face: &StencilFace,
) {
    dst.set_stencil_compare_function(face.compare);
    dst.set_stencil_failure_operation(face.stencil_fail);
    dst.set_depth_failure_operation(face.depth_fail);
    dst.set_depth_stencil_pass_operation(face.pass);
    dst.set_read_mask(src.read_mask);
    dst.set_write_mask(src.write_mask);
}

fn apply_depth_stencil_state(
    device: &Device,
    encoder: &RenderCommandEncoderRef,
    depth_stencil: Option<&ReimsVgpuDepthStencilState>,
    stencil_reference: Option<&ReimsVgpuStencilReferenceState>,
    err: ErrOut<'_>,
) -> Status {
    if let Some(ds) = depth_stencil {
        let Some(depth_compare) = mtl_enum::compare_function(ds.depth_compare_function) else {
            set_err(err, "unsupported Metal depth-stencil enum value");
            return Status::args("metal_render_depth_compare_unsupported")
                .field("compare", ds.depth_compare_function);
        };
        let Some(front) = stencil_face_converted(&ds.front_face) else {
            set_err(err, "unsupported Metal depth-stencil enum value");
            return Status::args("metal_render_front_stencil_state_unsupported")
                .field("compare", ds.front_face.compare_function)
                .field("stencil_fail", ds.front_face.stencil_failure_operation)
                .field("depth_fail", ds.front_face.depth_failure_operation)
                .field("pass", ds.front_face.depth_stencil_pass_operation);
        };
        let Some(back) = stencil_face_converted(&ds.back_face) else {
            set_err(err, "unsupported Metal depth-stencil enum value");
            return Status::args("metal_render_back_stencil_state_unsupported")
                .field("compare", ds.back_face.compare_function)
                .field("stencil_fail", ds.back_face.stencil_failure_operation)
                .field("depth_fail", ds.back_face.depth_failure_operation)
                .field("pass", ds.back_face.depth_stencil_pass_operation);
        };
        let ds_key = DepthStencilKey {
            hash: hash_bytes(bytes_of(ds)),
            desc: *ds,
        };
        let state = if let Some(hit) = depth_stencil_lookup(&ds_key) {
            hit
        } else {
            let descriptor = DepthStencilDescriptor::new();
            descriptor.set_depth_compare_function(depth_compare);
            descriptor.set_depth_write_enabled(ds.depth_write_enabled != 0);
            if ds.front_stencil_enabled != 0 {
                let face = StencilDescriptor::new();
                apply_stencil_face(&face, &ds.front_face, &front);
                descriptor.set_front_face_stencil(Some(&face));
            }
            if ds.back_stencil_enabled != 0 {
                let face = StencilDescriptor::new();
                apply_stencil_face(&face, &ds.back_face, &back);
                descriptor.set_back_face_stencil(Some(&face));
            }
            let state = device.new_depth_stencil_state(&descriptor);
            depth_stencil_insert(ds_key, state)
        };
        encoder.set_depth_stencil_state(&state);
    }
    if let Some(sr) = stencil_reference {
        encoder.set_stencil_front_back_reference_value(sr.front, sr.back);
    }
    Status::OK
}

fn apply_viewports(
    encoder: &RenderCommandEncoderRef,
    viewports: &[ReimsVgpuViewport],
    target_width: u32,
    target_height: u32,
) {
    if viewports.is_empty() {
        encoder.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: target_width as f64,
            height: target_height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        return;
    }
    let mut mtl: Vec<MTLViewport> = Vec::with_capacity(viewports.len());
    for v in viewports {
        mtl.push(MTLViewport {
            originX: v.x as f64,
            originY: v.y as f64,
            width: v.width as f64,
            height: v.height as f64,
            znear: v.znear as f64,
            zfar: v.zfar as f64,
        });
    }
    encoder.set_viewports(&mtl);
}

fn apply_scissors(
    encoder: &RenderCommandEncoderRef,
    scissors: &[ReimsVgpuScissor],
    target_width: u32,
    target_height: u32,
) {
    if scissors.is_empty() {
        encoder.set_scissor_rect(MTLScissorRect {
            x: 0,
            y: 0,
            width: target_width as u64,
            height: target_height as u64,
        });
        return;
    }
    let mut mtl: Vec<MTLScissorRect> = Vec::with_capacity(scissors.len());
    for s in scissors {
        mtl.push(MTLScissorRect {
            x: s.x as u64,
            y: s.y as u64,
            width: s.width as u64,
            height: s.height as u64,
        });
    }
    encoder.set_scissor_rects(&mtl);
}

fn find_or_add_attr_slot(
    device: &Device,
    slots: &mut Vec<AttrBufferSlot>,
    attr: &ReimsVgpuVertexAttr,
    err: ErrOut<'_>,
) -> Result<Option<u64>, Status> {
    // Layout-only attrs (no host bytes yet) still drive the vertex descriptor;
    // buffer data may arrive via regular setVertexBuffer binds.
    if attr.data.is_null() || attr.len == 0 {
        return Ok(None);
    }
    if attr.stride == 0 {
        set_err(err, "invalid vertex attribute buffer");
        return Err(Status::args("metal_render_vertex_attribute_stride_zero")
            .field("location", attr.location)
            .field("buffer", attr.buffer_index));
    }
    // The step pair is admitted by the caller, for every attribute rather than
    // only the ones with host bytes: this function returns above when the
    // attribute carries none, and the descriptor is built from all of them.
    let step_function = attr.step_function;
    let step_rate = attr.step_rate;
    for s in slots.iter() {
        if s.index == attr.buffer_index as u64
            && s.data == attr.data
            && s.len == attr.len
            && s.stride == attr.stride
            && s.step_function == step_function
            && s.step_rate == step_rate
        {
            return Ok(Some(s.index));
        }
        if s.index == attr.buffer_index as u64 {
            set_err(
                err,
                format!(
                    "conflicting vertex attribute buffer index {}",
                    attr.buffer_index
                ),
            );
            return Err(Status::args("metal_render_vertex_buffer_index_conflict")
                .field("buffer", attr.buffer_index));
        }
    }
    if slots.len() >= REIMS_VGPU_METAL_MAX_BUFFERS {
        set_err(err, "too many vertex attribute buffers");
        return Err(Status::args("metal_render_vertex_buffer_count_exceeded")
            .field("count", slots.len())
            .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS));
    }
    let buffer = match new_buffer_from_host(device, attr.data, attr.len) {
        Some(b) => b,
        None => {
            set_err(err, "failed to create vertex attribute buffer");
            return Err(Status::execute("metal_render_vertex_buffer_create_failed")
                .field("buffer", attr.buffer_index)
                .field("len", attr.len));
        }
    };
    let index = attr.buffer_index as u64;
    slots.push(AttrBufferSlot {
        data: attr.data,
        len: attr.len,
        stride: attr.stride,
        step_function,
        step_rate,
        index,
        buffer,
    });
    Ok(Some(index))
}

fn make_vertex_descriptor(
    device: &Device,
    attrs: &[ReimsVgpuVertexAttr],
    err: ErrOut<'_>,
) -> Result<(Option<VertexDescriptor>, Vec<AttrBufferSlot>), Status> {
    if attrs.is_empty() {
        return Ok((None, Vec::new()));
    }
    if attrs.len() > REIMS_VGPU_METAL_MAX_ATTRS {
        set_err(err, "invalid vertex attribute list");
        return Err(Status::args("metal_render_vertex_attribute_count_exceeded")
            .field("count", attrs.len())
            .field("limit", REIMS_VGPU_METAL_MAX_ATTRS));
    }
    let descriptor = VertexDescriptor::new().to_owned();
    let mut slots = Vec::new();
    let mut any_layout = false;
    for attr in attrs {
        if attr.format == 0 || attr.stride == 0 {
            continue;
        }
        if attr.location as usize >= REIMS_VGPU_METAL_MAX_ATTRS {
            set_err(
                err,
                format!("vertex attribute location {} out of range", attr.location),
            );
            return Err(
                Status::args("metal_render_vertex_attribute_location_out_of_range")
                    .field("location", attr.location)
                    .field("limit", REIMS_VGPU_METAL_MAX_ATTRS),
            );
        }
        if !valid_buffer_binding(attr.buffer_index) {
            set_err(
                err,
                format!(
                    "vertex attribute buffer index {} out of range",
                    attr.buffer_index
                ),
            );
            return Err(
                Status::args("metal_render_vertex_attribute_buffer_out_of_range")
                    .field("buffer", attr.buffer_index)
                    .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS),
            );
        }
        // The format and step-function words come straight off the guest's
        // type-7 pipeline descriptor and nothing upstream clamps them, so both
        // are converted before anything is encoded. Neither had any check at
        // all before; the location and buffer index above did.
        let Some(format) = mtl_enum::vertex_format(attr.format) else {
            set_err(
                err,
                format!("unsupported vertex attribute format {}", attr.format),
            );
            return Err(
                Status::args("metal_render_vertex_attribute_format_unsupported")
                    .field("location", attr.location)
                    .field("format", attr.format),
            );
        };
        let step_ordinal = attr.step_function;
        let Some(step) = mtl_enum::vertex_step_function(step_ordinal) else {
            set_err(
                err,
                format!("unsupported vertex step function {step_ordinal}"),
            );
            return Err(
                Status::args("metal_render_vertex_step_function_unsupported")
                    .field("buffer", attr.buffer_index)
                    .field("step", step_ordinal),
            );
        };
        // Recognised, and this pipeline is not one they belong to. `PerPatch`
        // and `PerPatchControlPoint` describe a post-tessellation vertex
        // function; `MTLRenderPipelineDescriptor` validation rejects a vertex
        // descriptor that names one without a tessellation stage, so the draw is
        // lost either way and the only question is whether the log says which
        // kind of loss it was. It is a separate slug from the conversion refusal
        // above for the reason `translate::vertex` gives on the Vulkan arm,
        // where the same split already exists: one reads "the guest ran a
        // tessellation pipeline", the other "something is wrong upstream".
        if step_ordinal > MTL_VERTEX_STEP_FUNCTION_PER_INSTANCE {
            set_err(err, "unsupported vertex attribute step state");
            return Err(Status::args("metal_render_vertex_step_function_per_patch")
                .field("location", attr.location)
                .field("step", step_ordinal));
        }
        // A rate of zero is legal for exactly one step function and required by
        // it, which `contract::vertex_step` states beside the ordinals. Under
        // any other, a zero rate advances nothing and `MTLVertexDescriptor`
        // validation rejects the descriptor, so refuse it here by name instead.
        // Asked here rather than where the buffer slot is allocated, because
        // that runs only for an attribute carrying host bytes while every
        // attribute reaches `set_step_function` below.
        if !step_rate_in_contract(step_ordinal, attr.step_rate) {
            set_err(err, "unsupported vertex attribute step state");
            return Err(Status::args("metal_render_vertex_step_rate_zero")
                .field("location", attr.location)
                .field("step", step_ordinal));
        }
        // Optional host bytes → Metal buffer slot for encode-time bind.
        find_or_add_attr_slot(device, &mut slots, attr, err)?;
        if let Some(a) = descriptor.attributes().object_at(attr.location as u64) {
            a.set_format(format);
            a.set_offset(attr.offset as u64);
            a.set_buffer_index(attr.buffer_index as u64);
        }
        if let Some(layout) = descriptor.layouts().object_at(attr.buffer_index as u64) {
            layout.set_stride(attr.stride as u64);
            layout.set_step_function(step);
            layout.set_step_rate(attr.step_rate as u64);
        }
        any_layout = true;
    }
    if any_layout {
        Ok((Some(descriptor), slots))
    } else {
        Ok((None, slots))
    }
}

/// Color RT slot + format + per-slot blend and write mask for PSO keying.
pub struct ColorRtKey {
    pub slot: u32,
    pub pixel_format: u32,
    pub blend: Option<ReimsVgpuBlendState>,
    /// `MTLColorWriteMask` bits. Not inside `blend`, because the mask applies
    /// to an unblended attachment too.
    pub write_mask: u32,
}

// Every argument is one component of the pipeline-state key, and the point of
// the function is that the key is built from exactly these and nothing else.
//
// The two shader modules are not among them any more. They used to arrive here
// as slices and leave as four `(hash, len)` words folded into the key, which is
// the shape `backend::render_pso_key` explains was wrong: a digest with no
// retained bytes decides two pipelines are one. They now travel to the cache as
// `BlobKey`s beside this descriptor, so there is no pairing left for this
// function to get wrong and no reason for it to see a shader at all.
fn fill_render_pso_key(
    attrs: &[ReimsVgpuVertexAttr],
    blend: Option<&ReimsVgpuBlendState>,
    color_rts: &[ColorRtKey],
    depth_pixel_format: u32,
    stencil_pixel_format: u32,
) -> RenderPsoKey {
    use crate::backend::metal::constants::REIMS_VGPU_METAL_MAX_COLOR_RTS;
    // Exhaustive rather than `..Default::default()`, which is the difference
    // between a field added to `RenderPsoKey` being a compile error here and
    // being silently filled with a zero. The second is not a missing
    // discriminator — it is worse: the stored key would carry the default while
    // every later lookup key carried the same default, so the two would agree
    // and two pipelines differing only in the new field would share one
    // `MTLRenderPipelineState`. `RenderPsoKeyClone::clone_key` is exhaustive for
    // the same reason.
    let mut key = RenderPsoKey {
        // Overwritten at the bottom with the fold over every field below. Zero
        // here would be a bucket, not a key, if that fold were ever removed.
        key_hash: 0,
        // Untruncated on purpose; see `REIMS_VGPU_METAL_MAX_ATTRS`.
        attr_count: attrs.len() as u32,
        attr_location: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_format: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_offset: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_buffer_index: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_stride: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_step_function: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        attr_step_rate: [0; REIMS_VGPU_METAL_MAX_ATTRS],
        blend_enable: 0,
        blend_src_rgb: 0,
        blend_dst_rgb: 0,
        blend_op_rgb: 0,
        blend_src_alpha: 0,
        blend_dst_alpha: 0,
        blend_op_alpha: 0,
        color_count: 0,
        color_formats: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_slot: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_enable: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_src_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_dst_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_op_rgb: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_src_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_dst_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        color_blend_op_alpha: [0; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        // Every slot writes every channel until a `ColorRtKey` says otherwise;
        // a zero here would describe a pipeline that writes no channel at all.
        color_write_mask: [MTL_COLOR_WRITE_MASK_ALL; REIMS_VGPU_METAL_MAX_COLOR_RTS],
        depth_pixel_format: 0,
        stencil_pixel_format: 0,
    };
    for (i, attr) in attrs.iter().enumerate().take(REIMS_VGPU_METAL_MAX_ATTRS) {
        key.attr_location[i] = attr.location;
        key.attr_format[i] = attr.format;
        key.attr_offset[i] = attr.offset;
        key.attr_buffer_index[i] = attr.buffer_index;
        key.attr_stride[i] = attr.stride;
        key.attr_step_function[i] = attr.step_function;
        key.attr_step_rate[i] = attr.step_rate;
    }
    // Global blend (stream blend color path / color0 fallback).
    if let Some(blend) = blend {
        if blend.enable != 0 {
            key.blend_enable = 1;
            key.blend_src_rgb = blend.src_rgb;
            key.blend_dst_rgb = blend.dst_rgb;
            key.blend_op_rgb = blend.op_rgb;
            key.blend_src_alpha = blend.src_alpha;
            key.blend_dst_alpha = blend.dst_alpha;
            key.blend_op_alpha = blend.op_alpha;
        }
    }
    key.color_count = color_rts.len().min(REIMS_VGPU_METAL_MAX_COLOR_RTS) as u32;
    for (i, rt) in color_rts
        .iter()
        .take(REIMS_VGPU_METAL_MAX_COLOR_RTS)
        .enumerate()
    {
        key.color_slot[i] = rt.slot as u8;
        key.color_formats[i] = rt.pixel_format;
        if let Some(b) = rt.blend.as_ref().filter(|b| b.enable != 0) {
            key.color_blend_enable[i] = 1;
            key.color_blend_src_rgb[i] = b.src_rgb;
            key.color_blend_dst_rgb[i] = b.dst_rgb;
            key.color_blend_op_rgb[i] = b.op_rgb;
            key.color_blend_src_alpha[i] = b.src_alpha;
            key.color_blend_dst_alpha[i] = b.dst_alpha;
            key.color_blend_op_alpha[i] = b.op_alpha;
        } else if key.blend_enable != 0 && rt.slot == 0 {
            key.color_blend_enable[i] = 1;
            key.color_blend_src_rgb[i] = key.blend_src_rgb;
            key.color_blend_dst_rgb[i] = key.blend_dst_rgb;
            key.color_blend_op_rgb[i] = key.blend_op_rgb;
            key.color_blend_src_alpha[i] = key.blend_src_alpha;
            key.color_blend_dst_alpha[i] = key.blend_dst_alpha;
            key.color_blend_op_alpha[i] = key.blend_op_alpha;
        }
        // Outside both blend arms: `MTLColorWriteMask` is independent of
        // `blendingEnabled`, so a masked attachment that does not blend still
        // has to leave its unwritten channels alone.
        key.color_write_mask[i] = rt.write_mask;
    }
    key.depth_pixel_format = depth_pixel_format;
    key.stencil_pixel_format = stencil_pixel_format;

    key.rehash();
    key
}

fn get_render_pipeline_state(
    device: &Device,
    vertex: &Function,
    fragment: &Function,
    vertex_descriptor: Option<&VertexDescriptor>,
    lookup: &RenderPsoLookup<'_>,
    err: ErrOut<'_>,
) -> Result<(RenderPipelineState, u32, u32), Status> {
    if let Some(hit) = render_pso_lookup(lookup) {
        return Ok(hit);
    }
    let key = lookup.desc;

    let pipeline_descriptor = RenderPipelineDescriptor::new();
    pipeline_descriptor.set_vertex_function(Some(vertex));
    pipeline_descriptor.set_fragment_function(Some(fragment));
    if let Some(vd) = vertex_descriptor {
        pipeline_descriptor.set_vertex_descriptor(Some(vd));
    }
    for i in 0..key.color_count as usize {
        let slot = key.color_slot[i] as u64;
        if let Some(color) = pipeline_descriptor.color_attachments().object_at(slot) {
            let Some(format) = mtl_enum::pixel_format(key.color_formats[i]) else {
                set_err(
                    err,
                    format!("unknown color pixel format {}", key.color_formats[i]),
                );
                return Err(Status::args("metal_render_pso_color_format_undeclared")
                    .field("slot", slot)
                    .field("format", key.color_formats[i]));
            };
            color.set_pixel_format(format);
            let slot_blend = if key.color_blend_enable[i] != 0 {
                Some(ReimsVgpuBlendState {
                    enable: 1,
                    src_rgb: key.color_blend_src_rgb[i],
                    dst_rgb: key.color_blend_dst_rgb[i],
                    op_rgb: key.color_blend_op_rgb[i],
                    src_alpha: key.color_blend_src_alpha[i],
                    dst_alpha: key.color_blend_dst_alpha[i],
                    op_alpha: key.color_blend_op_alpha[i],
                    has_blend_color: 0,
                    blend_color: [0.0; 4],
                })
            } else {
                None
            };
            let rc = apply_blend(color, slot_blend.as_ref(), err);
            if !rc.is_ok() {
                return Err(rc);
            }
            // Unconditional, unlike the blend above: the guest's mask governs
            // an unblended attachment too. `MTLColorWriteMask`'s bit order is
            // Metal's own here — no exchange, unlike the Vulkan arm — because
            // this descriptor is the same API the value was serialized from.
            color.set_write_mask(MTLColorWriteMask::from_bits_truncate(
                key.color_write_mask[i] as NSUInteger,
            ));
        }
    }
    if key.depth_pixel_format != 0 {
        let Some(format) = mtl_enum::pixel_format(key.depth_pixel_format) else {
            set_err(
                err,
                format!("unknown depth pixel format {}", key.depth_pixel_format),
            );
            return Err(Status::args("metal_render_pso_depth_format_undeclared")
                .field("format", key.depth_pixel_format));
        };
        pipeline_descriptor.set_depth_attachment_pixel_format(format);
    }
    if key.stencil_pixel_format != 0 {
        let Some(format) = mtl_enum::pixel_format(key.stencil_pixel_format) else {
            set_err(
                err,
                format!("unknown stencil pixel format {}", key.stencil_pixel_format),
            );
            return Err(Status::args("metal_render_pso_stencil_format_undeclared")
                .field("format", key.stencil_pixel_format));
        };
        pipeline_descriptor.set_stencil_attachment_pixel_format(format);
    }

    let (pso, reflection) = device
        .new_render_pipeline_state_with_reflection(
            &pipeline_descriptor,
            MTLPipelineOption::ArgumentInfo,
        )
        .map_err(|e| {
            set_err(err, format!("render PSO failed: {e}"));
            Status::execute("metal_render_pso_create_failed").field("key_hash", key.key_hash)
        })?;

    let reflection_ptr = reflection.as_ptr() as *mut objc::runtime::Object;
    // A reflection naming a sampler slot the argument table does not have
    // refuses the PSO. Building it anyway would cache a mask claiming the shader
    // does not sample that slot, and every later draw through this PSO would
    // sample an undefined sampler with nothing to say why.
    let sampler_mask = |vertex: bool| {
        render_reflection_sampler_mask(reflection_ptr, vertex).map_err(|overflow| {
            set_err(
                err,
                format!(
                    "render reflection sampler slot {} past table",
                    overflow.index
                ),
            );
            Status::execute("metal_render_reflection_sampler_past_table")
                .field("index", overflow.index)
                .field("vertex", vertex)
        })
    };
    let vert_mask = sampler_mask(true)?;
    let frag_mask = sampler_mask(false)?;

    Ok(render_pso_insert(lookup, pso, vert_mask, frag_mask))
}

fn bind_storage_buffers(
    device: &Device,
    encoder: &RenderCommandEncoderRef,
    retained: &mut Vec<Buffer>,
    buffers: &[ReimsVgpuBuffer],
    fragment_stage: bool,
    err: ErrOut<'_>,
) -> Status {
    if buffers.is_empty() {
        return Status::OK;
    }
    for buffer in buffers {
        if !valid_buffer_binding(buffer.binding) {
            set_err(
                err,
                format!(
                    "invalid {} buffer binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    buffer.binding
                ),
            );
            return Status::args("metal_render_buffer_binding_out_of_range")
                .field("fragment", fragment_stage)
                .field("binding", buffer.binding)
                .field("limit", REIMS_VGPU_METAL_MAX_BUFFERS);
        }
        if buffer.data.is_null() {
            set_err(
                err,
                format!(
                    "invalid {} buffer binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    buffer.binding
                ),
            );
            return Status::args("metal_render_buffer_data_missing")
                .field("fragment", fragment_stage)
                .field("binding", buffer.binding);
        }
        if buffer.len == 0 {
            set_err(
                err,
                format!(
                    "invalid {} buffer binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    buffer.binding
                ),
            );
            return Status::args("metal_render_buffer_length_zero")
                .field("fragment", fragment_stage)
                .field("binding", buffer.binding);
        }
        let mtl_buffer = match new_buffer_from_host(device, buffer.data, buffer.len) {
            Some(b) => b,
            None => {
                set_err(
                    err,
                    format!(
                        "failed to create {} buffer",
                        if fragment_stage { "fragment" } else { "vertex" }
                    ),
                );
                return Status::execute("metal_render_buffer_create_failed")
                    .field("fragment", fragment_stage)
                    .field("binding", buffer.binding)
                    .field("len", buffer.len);
            }
        };
        if fragment_stage {
            encoder.set_fragment_buffer(buffer.binding as u64, Some(&mtl_buffer), 0);
        } else if buffer.has_attribute_stride != 0 {
            // `setVertexBuffer:offset:attributeStride:atIndex:` is only legal
            // where the pipeline's `MTLVertexBufferLayoutDescriptor.stride` for
            // this index is `MTLBufferLayoutStrideDynamic`, exactly as the
            // compute rail's `metal_compute_attribute_stride_without_dynamic_layout`
            // states for `MTLBufferLayoutDescriptor`. This rail's vertex
            // descriptor is built from the type-7 attribute block and never
            // declares a dynamic layout, so the selector would raise an
            // NSException — a process abort, not an error return.
            //
            // Refused by name rather than bound with the pipeline's own stride.
            // A guest that sent this negotiated `supportsDynamicAttributeStride`
            // and built a pipeline whose layout stride is the sentinel, so
            // fetching at that stride is not "close enough": it is wrong
            // geometry the guest is never told about. Closing this means the
            // render pipeline declaring the dynamic layout, at which point the
            // bind becomes `raw_metal`'s render sibling of the compute setter.
            set_err(
                err,
                format!(
                    "vertex buffer {} carries an attributeStride and this rail's \
                     vertex descriptor declares no dynamic layout",
                    buffer.binding
                ),
            );
            return Status::args("metal_render_attribute_stride_without_dynamic_layout")
                .field("binding", buffer.binding)
                .field("stride", buffer.attribute_stride);
        } else {
            encoder.set_vertex_buffer(buffer.binding as u64, Some(&mtl_buffer), 0);
        }
        retained.push(mtl_buffer);
    }
    Status::OK
}

fn bind_sampled_images(
    device: &Device,
    encoder: &RenderCommandEncoderRef,
    retained: &mut Vec<Texture>,
    images: &[ReimsVgpuSampledImage],
    fragment_stage: bool,
    err: ErrOut<'_>,
) -> Status {
    if images.is_empty() {
        return Status::OK;
    }
    for image in images {
        let Some(texture_index) = texture_index(image.binding) else {
            set_err(
                err,
                format!(
                    "invalid {} sampled image binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    image.binding
                ),
            );
            return Status::args("metal_render_sampled_binding_invalid")
                .field("fragment", fragment_stage)
                .field("binding", image.binding);
        };
        if image.width == 0 {
            set_err(
                err,
                format!(
                    "invalid {} sampled image binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    image.binding
                ),
            );
            return Status::args("metal_render_sampled_width_zero")
                .field("fragment", fragment_stage)
                .field("binding", image.binding);
        }
        if image.height == 0 {
            set_err(
                err,
                format!(
                    "invalid {} sampled image binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    image.binding
                ),
            );
            return Status::args("metal_render_sampled_height_zero")
                .field("fragment", fragment_stage)
                .field("binding", image.binding);
        }

        let (pixel_format, bytes, bytes_per_row) = if image.pixel_format != 0 {
            let Some(bpp) = mtl_pixel_format_bpp(image.pixel_format) else {
                set_err(
                    err,
                    format!(
                        "invalid native {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_format_unsupported")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("format", image.pixel_format);
            };
            if image.data.is_null() {
                set_err(
                    err,
                    format!(
                        "invalid native {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_data_missing")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding);
            }
            if image.data_len == 0 {
                set_err(
                    err,
                    format!(
                        "invalid native {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_data_empty")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding);
            }
            let bpr = if image.bytes_per_row != 0 {
                image.bytes_per_row as u64
            } else {
                image.width as u64 * bpp as u64
            };
            let need = bpr.checked_mul(image.height as u64);
            let Some(need) = need else {
                set_err(
                    err,
                    format!(
                        "native {} sampled image too short binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_span_overflow")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("bytes_per_row", bpr)
                    .field("height", image.height);
            };
            if image.data_len < need as usize {
                set_err(
                    err,
                    format!(
                        "native {} sampled image too short binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_data_too_short")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("len", image.data_len)
                    .field("required", need);
            }
            let Some(format) = mtl_enum::pixel_format(image.pixel_format) else {
                set_err(
                    err,
                    format!(
                        "native {} sampled image names no pixel format, binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_native_format_undeclared")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("format", image.pixel_format);
            };
            (format, image.data, bpr)
        } else {
            let Some(expected_len) = tight_image_bytes(
                image.width,
                image.height,
                crate::contract::pixel_format::RGBA8_BPP as usize,
            ) else {
                set_err(
                    err,
                    format!(
                        "invalid {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_rgba_geometry_invalid")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("width", image.width)
                    .field("height", image.height);
            };
            if image.rgba8.is_null() {
                set_err(
                    err,
                    format!(
                        "invalid {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_rgba_data_missing")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding);
            }
            if image.len < expected_len {
                set_err(
                    err,
                    format!(
                        "invalid {} sampled image binding {}",
                        if fragment_stage { "fragment" } else { "vertex" },
                        image.binding
                    ),
                );
                return Status::args("metal_render_sampled_rgba_data_too_short")
                    .field("fragment", fragment_stage)
                    .field("binding", image.binding)
                    .field("len", image.len)
                    .field("required", expected_len);
            }
            (
                MTLPixelFormat::RGBA8Unorm,
                image.rgba8,
                image.width as u64 * 4,
            )
        };

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(pixel_format);
        descriptor.set_width(image.width as u64);
        descriptor.set_height(image.height as u64);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor)
        else {
            set_err(err, "failed to allocate sampled image texture");
            return Status::execute("metal_render_sampled_texture_alloc_failed")
                .field("fragment", fragment_stage)
                .field("binding", image.binding)
                .field("width", image.width)
                .field("height", image.height);
        };
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: image.width as u64,
                height: image.height as u64,
                depth: 1,
            },
        };
        texture.replace_region(region, 0, bytes as *const _, bytes_per_row);
        if fragment_stage {
            encoder.set_fragment_texture(texture_index as u64, Some(&texture));
        } else {
            encoder.set_vertex_texture(texture_index as u64, Some(&texture));
        }
        retained.push(texture);
    }
    Status::OK
}

fn bind_samplers(
    device: &Device,
    encoder: &RenderCommandEncoderRef,
    sampler_mask: u32,
    samplers: &[ReimsVgpuSampler],
    fragment_stage: bool,
    err: ErrOut<'_>,
) -> Status {
    let mut seen = [false; REIMS_VGPU_METAL_MAX_SAMPLERS];
    for s in samplers {
        let Some(index) = sampler_index(s.binding) else {
            set_err(
                err,
                format!(
                    "invalid {} sampler binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    s.binding
                ),
            );
            return Status::args("metal_render_sampler_binding_invalid")
                .field("fragment", fragment_stage)
                .field("binding", s.binding);
        };
        if seen[index] {
            set_err(
                err,
                format!(
                    "duplicate {} sampler binding {}",
                    if fragment_stage { "fragment" } else { "vertex" },
                    s.binding
                ),
            );
            return Status::args("metal_render_sampler_binding_duplicate")
                .field("fragment", fragment_stage)
                .field("binding", s.binding);
        }
        let sampler = match make_explicit_sampler(device, s, err) {
            Ok(s) => s,
            Err(st) => return st,
        };
        seen[index] = true;
        if s.has_lod_clamp != 0 {
            let lod = f32::from_bits(s.clamp_lod_min_bits)..f32::from_bits(s.clamp_lod_max_bits);
            if fragment_stage {
                encoder.set_fragment_sampler_state_with_lod(index as u64, Some(&sampler), lod);
            } else {
                encoder.set_vertex_sampler_state_with_lod(index as u64, Some(&sampler), lod);
            }
        } else if fragment_stage {
            encoder.set_fragment_sampler_state(index as u64, Some(&sampler));
        } else {
            encoder.set_vertex_sampler_state(index as u64, Some(&sampler));
        }
    }

    // `seen` is already exactly the argument table's width, so its own length is
    // the bound — the `.take(REIMS_VGPU_METAL_MAX_SAMPLERS)` that used to sit
    // here re-stated that in a second place while truncating nothing, and read
    // as though the mask could name a slot this loop declines to serve. It
    // cannot: `render_reflection_sampler_mask` sets no bit outside the table and
    // says so fail-visibly if the reflection ever asks it to.
    for (index, seen) in seen.iter_mut().enumerate() {
        if (sampler_mask & (1u32 << index)) == 0 || *seen {
            continue;
        }
        let sampler = make_default_sampler(device);
        *seen = true;
        if fragment_stage {
            encoder.set_fragment_sampler_state(index as u64, Some(&sampler));
        } else {
            encoder.set_vertex_sampler_state(index as u64, Some(&sampler));
        }
    }
    Status::OK
}

/// A depth or stencil attachment's two actions, converted once by the validator
/// and handed to the configure pass.
///
/// Converting here rather than where they are encoded keeps this device's two
/// layers of narrowing distinct and each stated once. The validator below is
/// *policy*: it accepts only `DontCare`/`Load`/`Clear` and only
/// `DontCare`/`Store`, refusing `MultisampleResolve` and the rest because this
/// device has no resolve path. [`mtl_enum`] is *type validity*: it accepts
/// every variant `metal` declares. Encoding used to transmute the raw ordinal
/// and was sound only because the validator happened to have run first, in a
/// different function, with nothing in the types saying so.
#[derive(Clone, Copy, Debug)]
struct AttachmentActions {
    load: MTLLoadAction,
    store: MTLStoreAction,
}

impl Default for AttachmentActions {
    /// What an absent attachment carries; `configure_*` returns before reading
    /// it, because a `None` attachment produces no texture at all.
    fn default() -> Self {
        Self {
            load: MTLLoadAction::DontCare,
            store: MTLStoreAction::DontCare,
        }
    }
}

fn validate_depth_attachment(
    depth: Option<&ReimsVgpuDepthAttachment>,
    width: u32,
    height: u32,
    err: ErrOut<'_>,
) -> Result<AttachmentActions, Status> {
    let Some(depth) = depth else {
        return Ok(AttachmentActions::default());
    };
    if depth.pixel_format != REIMS_VGPU_MTL_PIXEL_FORMAT_DEPTH32_FLOAT {
        set_err(
            err,
            format!(
                "unsupported depth attachment pixel format {}",
                depth.pixel_format
            ),
        );
        return Err(Status::args("metal_render_depth_format_unsupported")
            .field("format", depth.pixel_format));
    }
    let Some(load) = mtl_enum::load_action(depth.load_action) else {
        set_err(
            err,
            format!(
                "unsupported depth attachment load action {}",
                depth.load_action
            ),
        );
        return Err(Status::args("metal_render_depth_load_action_unsupported")
            .field("load_action", depth.load_action));
    };
    // The two accepted store actions are named rather than bounded: this device
    // has no resolve path, so `MultisampleResolve` and everything above it are
    // refused even though they are legal `MTLStoreAction` values.
    let Some(store @ (MTLStoreAction::DontCare | MTLStoreAction::Store)) =
        mtl_enum::store_action(depth.store_action)
    else {
        set_err(
            err,
            format!(
                "unsupported depth attachment store action {}",
                depth.store_action
            ),
        );
        return Err(Status::args("metal_render_depth_store_action_unsupported")
            .field("store_action", depth.store_action));
    };
    let Some(depth_len) = tight_image_bytes(width, height, std::mem::size_of::<f32>()) else {
        set_err(err, "invalid depth attachment dimensions");
        return Err(Status::args("metal_render_depth_geometry_invalid")
            .field("width", width)
            .field("height", height));
    };
    if !depth.data.is_null() {
        if depth.len != depth_len {
            set_err(err, "invalid depth attachment data length");
            return Err(Status::args("metal_render_depth_data_length_mismatch")
                .field("len", depth.len)
                .field("expected", depth_len));
        }
    } else if depth.len != 0 {
        set_err(err, "depth attachment length without data");
        return Err(Status::args("metal_render_depth_length_without_data").field("len", depth.len));
    }
    if (depth.load_action == REIMS_VGPU_MTL_LOAD_ACTION_LOAD
        || depth.store_action == REIMS_VGPU_MTL_STORE_ACTION_STORE)
        && depth.data.is_null()
    {
        set_err(err, "depth attachment load/store requires data");
        return Err(Status::args("metal_render_depth_data_required")
            .field("load_action", depth.load_action)
            .field("store_action", depth.store_action));
    }
    // `depth_len` is checked above against the attachment's declared length and
    // is not needed beyond that; the caller used to receive it and discard it.
    let _ = depth_len;
    Ok(AttachmentActions { load, store })
}

fn validate_stencil_attachment(
    stencil: Option<&ReimsVgpuStencilAttachment>,
    width: u32,
    height: u32,
    err: ErrOut<'_>,
) -> Result<AttachmentActions, Status> {
    let Some(stencil) = stencil else {
        return Ok(AttachmentActions::default());
    };
    if stencil.pixel_format != REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8 {
        set_err(
            err,
            format!(
                "unsupported stencil attachment pixel format {}",
                stencil.pixel_format
            ),
        );
        return Err(Status::args("metal_render_stencil_format_unsupported")
            .field("format", stencil.pixel_format));
    }
    let Some(load) = mtl_enum::load_action(stencil.load_action) else {
        set_err(
            err,
            format!(
                "unsupported stencil attachment load action {}",
                stencil.load_action
            ),
        );
        return Err(Status::args("metal_render_stencil_load_action_unsupported")
            .field("load_action", stencil.load_action));
    };
    let Some(store @ (MTLStoreAction::DontCare | MTLStoreAction::Store)) =
        mtl_enum::store_action(stencil.store_action)
    else {
        set_err(
            err,
            format!(
                "unsupported stencil attachment store action {}",
                stencil.store_action
            ),
        );
        return Err(
            Status::args("metal_render_stencil_store_action_unsupported")
                .field("store_action", stencil.store_action),
        );
    };
    let Some(stencil_len) = tight_image_bytes(width, height, 1) else {
        set_err(err, "invalid stencil attachment dimensions");
        return Err(Status::args("metal_render_stencil_geometry_invalid")
            .field("width", width)
            .field("height", height));
    };
    if !stencil.data.is_null() {
        if stencil.len != stencil_len {
            set_err(err, "invalid stencil attachment data length");
            return Err(Status::args("metal_render_stencil_data_length_mismatch")
                .field("len", stencil.len)
                .field("expected", stencil_len));
        }
    } else if stencil.len != 0 {
        set_err(err, "stencil attachment len without data");
        return Err(
            Status::args("metal_render_stencil_length_without_data").field("len", stencil.len)
        );
    }
    if (stencil.load_action == REIMS_VGPU_MTL_LOAD_ACTION_LOAD
        || stencil.store_action == REIMS_VGPU_MTL_STORE_ACTION_STORE)
        && stencil.data.is_null()
    {
        set_err(err, "stencil attachment load/store requires data");
        return Err(Status::args("metal_render_stencil_data_required")
            .field("load_action", stencil.load_action)
            .field("store_action", stencil.store_action));
    }
    let _ = stencil_len;
    Ok(AttachmentActions { load, store })
}

/// `Ok(None)` is "the guest attached no depth buffer"; `Err` is "it attached one
/// this device will not build". Those were the same answer while this returned a
/// bare `Option`, and they must not be: a draw that silently loses its depth
/// attachment renders with no depth test rather than refusing, which is guest
/// work quietly executed wrong.
///
/// `validate_depth_attachment` pins the format to `DEPTH32_FLOAT` before the
/// caller gets here, so the refusal below cannot fire today. It is the channel
/// that matters — with it, relaxing that check upstream turns an unbuildable
/// format into a named refusal instead of a missing attachment.
fn configure_depth_attachment(
    device: &Device,
    pass: &RenderPassDescriptorRef,
    retained: &mut Vec<Texture>,
    depth: Option<&ReimsVgpuDepthAttachment>,
    width: u32,
    height: u32,
    actions: AttachmentActions,
) -> Result<Option<Texture>, Status> {
    let Some(depth) = depth else {
        return Ok(None);
    };
    let Some(format) = mtl_enum::pixel_format(depth.pixel_format) else {
        return Err(
            Status::args("metal_render_depth_attachment_format_undeclared")
                .field("format", depth.pixel_format),
        );
    };
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(format);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor) else {
        return Err(
            Status::execute("metal_render_depth_attachment_alloc_failed")
                .field("width", width)
                .field("height", height),
        );
    };
    if depth.load_action == REIMS_VGPU_MTL_LOAD_ACTION_LOAD {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        texture.replace_region(
            region,
            0,
            depth.data as *const _,
            (width as u64) * (std::mem::size_of::<f32>() as u64),
        );
    }
    retained.push(texture.clone());
    if let Some(att) = pass.depth_attachment() {
        att.set_texture(Some(&texture));
        att.set_load_action(actions.load);
        att.set_clear_depth(depth.clear_depth);
        att.set_store_action(actions.store);
    }
    Ok(Some(texture))
}

/// Same split as [`configure_depth_attachment`]: `Ok(None)` is no stencil
/// attachment, `Err` is one this device will not build.
/// `validate_stencil_attachment` pins the format to `STENCIL8` upstream.
fn configure_stencil_attachment(
    device: &Device,
    pass: &RenderPassDescriptorRef,
    retained: &mut Vec<Texture>,
    stencil: Option<&ReimsVgpuStencilAttachment>,
    width: u32,
    height: u32,
    actions: AttachmentActions,
) -> Result<Option<Texture>, Status> {
    let Some(stencil) = stencil else {
        return Ok(None);
    };
    let Some(format) = mtl_enum::pixel_format(stencil.pixel_format) else {
        return Err(
            Status::args("metal_render_stencil_attachment_format_undeclared")
                .field("format", stencil.pixel_format),
        );
    };
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(format);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let Some(texture) = crate::backend::metal::raw_metal::new_texture(device, &descriptor) else {
        return Err(
            Status::execute("metal_render_stencil_attachment_alloc_failed")
                .field("width", width)
                .field("height", height),
        );
    };
    if stencil.load_action == REIMS_VGPU_MTL_LOAD_ACTION_LOAD {
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        texture.replace_region(region, 0, stencil.data as *const _, width as u64);
    }
    retained.push(texture.clone());
    if let Some(att) = pass.stencil_attachment() {
        att.set_texture(Some(&texture));
        att.set_load_action(actions.load);
        att.set_clear_stencil(stencil.clear_stencil);
        att.set_store_action(actions.store);
    }
    Ok(Some(texture))
}

/// One color render target for MRT encode (host RGBA8 seed/readback by default).
pub struct ColorRt<'a> {
    /// Metal color attachment index (`[[color(n)]]`).
    pub slot: u32,
    /// MTL pixel format; 0 = RGBA8Unorm (product writeback path).
    pub pixel_format: u32,
    pub seed_rgba8: Option<&'a [u8]>,
    pub out_rgba8: Option<&'a mut [u8]>,
    pub clear_r: f64,
    pub clear_g: f64,
    pub clear_b: f64,
    pub clear_a: f64,
    /// Guest MTL loadAction (0=DontCare, 1=Load, 2=Clear). Every color target
    /// is an ephemeral host RT, so Load requires a CPU seed (archive
    /// `reims_vgpu_backend_metal`: NULL seed → Clear invent).
    pub load_action: u32,
    /// Per-slot blend from pipeline color-attachment section (overrides global for this RT).
    pub blend: Option<ReimsVgpuBlendState>,
    /// Per-slot `MTLColorWriteMask` from the same section. `0xf` (all) is the
    /// value for an attachment whose entry omits the tag.
    pub write_mask: u32,
}

/// The occlusion query a draw is armed with, and the answer the pass recorded.
///
/// In and out in one struct because the two halves are one question — "how was
/// this draw armed, and what did the hardware count" — and the shape a caller
/// gets wrong is passing the mode and forgetting to read the answer back.
///
/// The offset the guest gave stays with the caller. This pass writes into a
/// host buffer of its own at offset 0 and hands the scalar back, because
/// `render_core_mrt` encodes exactly one draw per pass: a Metal pass spanning
/// several draws that share one guest offset becomes N of these, summed above
/// the backends, the same way the Vulkan rail's per-`DrawRequest` query pool
/// does.
pub struct VisibilityQuery {
    /// `MTLVisibilityResultMode` as the guest sent it. `Disabled` (`0`) is
    /// refused rather than executed: a query struct exists only where the
    /// stream armed one, so a disarming ordinal here is a caller defect and
    /// running the pass unarmed would report a count of zero for a draw that
    /// was never asked about.
    pub mode: u32,
    /// Out: samples the draw passed, or `None` where the pass did not run to
    /// completion. Never written on any refusal path, so a caller that keeps
    /// `None` knows the query is unanswered rather than answered zero.
    pub samples: Option<u64>,
}

/// Resolve Metal loadAction for a color attachment. Archive
/// `reims_vgpu_backend_metal` (fresh Shared RT every job):
/// `loadAction = target_rgba8 ? Load : Clear` — guest Load without a CPU seed
/// Clear-invents. This is now every color attachment: the guest-backed
/// alias that used to honor `load_action` verbatim is gone.
pub(crate) fn color_rt_load_action(guest_load: u32, has_seed: bool) -> u32 {
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_LOAD_ACTION_CLEAR, REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
    };
    match guest_load {
        x if x == REIMS_VGPU_MTL_LOAD_ACTION_CLEAR => REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
        x if x == REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE => REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        x if x == REIMS_VGPU_MTL_LOAD_ACTION_LOAD && has_seed => REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
        _ if has_seed => REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
        _ => REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
    }
}

#[cfg(test)]
mod load_action_tests {
    use super::color_rt_load_action;
    use crate::backend::metal::abi::{
        REIMS_VGPU_MTL_LOAD_ACTION_CLEAR, REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE,
        REIMS_VGPU_MTL_LOAD_ACTION_LOAD,
    };

    /// Archive: Load + CPU seed on fresh RT → Load.
    #[test]
    fn load_with_seed_loads() {
        assert_eq!(
            color_rt_load_action(REIMS_VGPU_MTL_LOAD_ACTION_LOAD, true),
            REIMS_VGPU_MTL_LOAD_ACTION_LOAD
        );
    }

    /// Archive NULL seed on fresh RT → Clear invent.
    #[test]
    fn load_without_seed_fresh_rt_clear_invents() {
        assert_eq!(
            color_rt_load_action(REIMS_VGPU_MTL_LOAD_ACTION_LOAD, false),
            REIMS_VGPU_MTL_LOAD_ACTION_CLEAR
        );
    }

    #[test]
    fn clear_and_dontcare_preserved() {
        assert_eq!(
            color_rt_load_action(REIMS_VGPU_MTL_LOAD_ACTION_CLEAR, false),
            REIMS_VGPU_MTL_LOAD_ACTION_CLEAR
        );
        assert_eq!(
            color_rt_load_action(REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE, true),
            REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE
        );
    }
}

#[cfg(test)]
mod attachment_decline_tests {
    use super::*;
    use crate::observe::Emit;

    fn line(status: Status) -> String {
        Emit::refusal("metal_attachment_test", &status)
            .expect("invalid attachment must carry a refusal")
            .render()
    }

    #[test]
    fn depth_and_stencil_rejections_name_the_attachment_and_check() {
        let depth = ReimsVgpuDepthAttachment {
            pixel_format: 0xfeed,
            load_action: REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
            store_action: REIMS_VGPU_MTL_STORE_ACTION_STORE,
            clear_depth: 1.0,
            data: std::ptr::null_mut(),
            len: 0,
        };
        let depth_status = validate_depth_attachment(Some(&depth), 4, 4, (std::ptr::null_mut(), 0))
            .expect_err("unsupported depth format must fail");
        assert_eq!(
            line(depth_status),
            "metal_attachment_test reason=metal_render_depth_format_unsupported class=args format=65261"
        );

        let stencil = ReimsVgpuStencilAttachment {
            pixel_format: REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8,
            load_action: REIMS_VGPU_MTL_LOAD_ACTION_CLEAR,
            store_action: u32::MAX,
            clear_stencil: 0,
            data: std::ptr::null_mut(),
            len: 0,
        };
        let stencil_status =
            validate_stencil_attachment(Some(&stencil), 4, 4, (std::ptr::null_mut(), 0))
                .expect_err("unsupported stencil store action must fail");
        assert_eq!(
            line(stencil_status),
            "metal_attachment_test reason=metal_render_stencil_store_action_unsupported class=args store_action=4294967295"
        );
    }
}

/// Multi-render-target encode: one Metal pass with color attachments at given slots.
#[allow(clippy::too_many_arguments)]
pub fn render_core_mrt(
    vert_mtlb: &[u8],
    frag_mtlb: &[u8],
    width: u32,
    height: u32,
    draw: crate::contract::draw::DrawArgs,
    primitive_indirect: Option<&ReimsVgpuPrimitiveIndirectDraw>,
    indexed: Option<&ReimsVgpuIndexedDraw>,
    attrs: &[ReimsVgpuVertexAttr],
    buffers: &[ReimsVgpuBuffer],
    frag_buffers: &[ReimsVgpuBuffer],
    vertex_images: &[ReimsVgpuSampledImage],
    vertex_samplers: &[ReimsVgpuSampler],
    images: &[ReimsVgpuSampledImage],
    samplers: &[ReimsVgpuSampler],
    viewports: &[ReimsVgpuViewport],
    scissors: &[ReimsVgpuScissor],
    raster: Option<&ReimsVgpuRasterState>,
    depth_bias: Option<&ReimsVgpuDepthBiasState>,
    depth_stencil: Option<&ReimsVgpuDepthStencilState>,
    stencil_reference: Option<&ReimsVgpuStencilReferenceState>,
    depth_attachment: Option<&mut ReimsVgpuDepthAttachment>,
    stencil_attachment: Option<&mut ReimsVgpuStencilAttachment>,
    blend: Option<&ReimsVgpuBlendState>,
    colors: &mut [ColorRt<'_>],
    visibility: Option<&mut VisibilityQuery>,
    err: ErrOut<'_>,
) -> Status {
    use crate::backend::metal::constants::REIMS_VGPU_METAL_MAX_COLOR_RTS;
    // Widened here rather than at the call, so the `as usize` on each of these
    // happens once and the caller passes the decoded draw whole. These were five
    // positional parameters, four of them `usize`; the sole caller reached them
    // through `mrt_draw_request`, which took the same five values in a different
    // order.
    let vertex_count = draw.vertex_count as usize;
    let first_vertex = draw.first_vertex as usize;
    let instance_count = draw.instance_count as usize;
    let base_instance = draw.base_instance as usize;
    let primitive_type = draw.primitive_type;
    let indexed_indirect = indexed.map(|i| !i.indirect.is_null()).unwrap_or(false);
    if colors.is_empty() {
        set_err(err, "invalid color render target count");
        return Status::args("metal_render_color_targets_empty");
    }
    if colors.len() > REIMS_VGPU_METAL_MAX_COLOR_RTS {
        set_err(err, "invalid color render target count");
        return Status::args("metal_render_color_target_count_exceeded")
            .field("count", colors.len())
            .field("limit", REIMS_VGPU_METAL_MAX_COLOR_RTS);
    }
    // Resolve per-RT format + bpp; require uniform dimensions (Metal pass rule).
    let mut color_meta: Vec<(u32, u32, usize, MTLPixelFormat)> = Vec::with_capacity(colors.len());
    // (slot, fmt_u32, bpp, mtl_fmt)
    for c in colors.iter() {
        if c.slot as usize >= REIMS_VGPU_METAL_MAX_COLOR_RTS {
            set_err(
                err,
                format!("color attachment slot {} out of range", c.slot),
            );
            return Status::args("metal_render_color_slot_out_of_range")
                .field("slot", c.slot)
                .field("limit", REIMS_VGPU_METAL_MAX_COLOR_RTS);
        }
        let mut fmt = c.pixel_format;
        if fmt == 0 {
            fmt = MTLPixelFormat::RGBA8Unorm as u32;
        }
        let Some(bpp) = mtl_pixel_format_bpp(fmt) else {
            set_err(err, format!("unsupported render color pixel format {fmt}"));
            return Status::args("metal_render_color_format_unsupported")
                .field("slot", c.slot)
                .field("format", fmt);
        };
        if width == 0 {
            set_err(err, "invalid render dimensions");
            return Status::args("metal_render_color_width_zero").field("slot", c.slot);
        }
        if height == 0 {
            set_err(err, "invalid render dimensions");
            return Status::args("metal_render_color_height_zero").field("slot", c.slot);
        }
        let Some(need) = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(bpp))
        else {
            set_err(err, "invalid render dimensions");
            return Status::args("metal_render_color_span_overflow")
                .field("slot", c.slot)
                .field("width", width)
                .field("height", height)
                .field("bpp", bpp);
        };
        if let Some(seed) = c.seed_rgba8 {
            if seed.len() != need {
                set_err(err, "invalid color RT seed length");
                return Status::args("metal_render_color_seed_length_mismatch")
                    .field("slot", c.slot)
                    .field("len", seed.len())
                    .field("expected", need);
            }
        }
        if let Some(out) = c.out_rgba8.as_ref() {
            if !out.is_empty() && out.len() != need {
                set_err(err, "invalid color RT out length");
                return Status::args("metal_render_color_output_length_mismatch")
                    .field("slot", c.slot)
                    .field("len", out.len())
                    .field("expected", need);
            }
        }
        let Some(mtl_format) = mtl_enum::pixel_format(fmt) else {
            set_err(err, format!("color RT names no pixel format {fmt}"));
            return Status::args("metal_render_color_format_undeclared")
                .field("slot", c.slot)
                .field("format", fmt);
        };
        color_meta.push((c.slot, fmt, bpp, mtl_format));
    }

    if primitive_indirect.is_none() && !indexed_indirect && vertex_count == 0 {
        set_err(err, "invalid render dimensions or draw count");
        return Status::args("metal_render_vertex_count_zero");
    }
    if primitive_indirect.is_none() && !indexed_indirect && instance_count == 0 {
        set_err(err, "invalid render dimensions or draw count");
        return Status::args("metal_render_instance_count_zero");
    }
    if primitive_indirect.is_some() && indexed.is_some() {
        set_err(
            err,
            "primitive indirect and indexed draw are mutually exclusive",
        );
        return Status::args("metal_render_indirect_and_indexed_conflict");
    }
    let Some(prim) = mtl_enum::primitive_type(primitive_type) else {
        set_err(err, "unsupported Metal primitive type");
        return Status::args("metal_render_primitive_type_unsupported")
            .field("primitive_type", primitive_type);
    };
    if viewports.len() > REIMS_VGPU_BACKEND_MAX_VIEWPORTS {
        set_err(err, "invalid viewport array state");
        return Status::args("metal_render_viewport_count_exceeded")
            .field("count", viewports.len())
            .field("limit", REIMS_VGPU_BACKEND_MAX_VIEWPORTS);
    }
    if scissors.len() > REIMS_VGPU_BACKEND_MAX_SCISSORS {
        set_err(err, "invalid scissor array state");
        return Status::args("metal_render_scissor_count_exceeded")
            .field("count", scissors.len())
            .field("limit", REIMS_VGPU_BACKEND_MAX_SCISSORS);
    }

    let Some(device) = system_device() else {
        set_err(err, "MTLCreateSystemDefaultDevice returned nil");
        return Status::execute("metal_render_device_unavailable");
    };

    let vertex = match load_only_function(device, vert_mtlb, "vertex", err) {
        Ok(f) => f,
        Err(st) => return st,
    };
    let fragment = match load_only_function(device, frag_mtlb, "fragment", err) {
        Ok(f) => f,
        Err(st) => return st,
    };

    let (vertex_descriptor, attr_slots) = match make_vertex_descriptor(device, attrs, err) {
        Ok(v) => v,
        Err(st) => return st,
    };

    let depth_actions =
        match validate_depth_attachment(depth_attachment.as_deref(), width, height, err) {
            Ok(a) => a,
            Err(st) => return st,
        };
    let stencil_actions =
        match validate_stencil_attachment(stencil_attachment.as_deref(), width, height, err) {
            Ok(a) => a,
            Err(st) => return st,
        };

    let color_rt_keys: Vec<ColorRtKey> = colors
        .iter()
        .zip(color_meta.iter())
        .map(|(c, &(slot, fmt, _, _))| ColorRtKey {
            slot,
            pixel_format: fmt,
            blend: c.blend.or_else(|| {
                if slot == 0 {
                    blend.copied()
                } else {
                    blend.filter(|b| b.enable != 0).copied()
                }
            }),
            write_mask: c.write_mask,
        })
        .collect();
    let pso_key = fill_render_pso_key(
        attrs,
        blend,
        &color_rt_keys,
        depth_attachment
            .as_ref()
            .map(|d| d.pixel_format)
            .unwrap_or(0),
        stencil_attachment
            .as_ref()
            .map(|s| s.pixel_format)
            .unwrap_or(0),
    );
    // The shaders join the descriptor here rather than inside it: the cache
    // retains their bytes and compares them, so they must reach it as bytes.
    let pso_lookup = RenderPsoLookup {
        desc: &pso_key,
        vert: BlobKey::new(vert_mtlb),
        frag: BlobKey::new(frag_mtlb),
    };
    let (pso, vert_sampler_mask, frag_sampler_mask) = match get_render_pipeline_state(
        device,
        &vertex,
        &fragment,
        vertex_descriptor.as_ref(),
        &pso_lookup,
        err,
    ) {
        Ok(v) => v,
        Err(st) => return st,
    };

    let mut retained_tex: Vec<Texture> = Vec::new();
    // (slot, tex, bpp)
    let mut color_textures: Vec<(u32, Texture, usize)> = Vec::new();
    for (i, c) in colors.iter().enumerate() {
        let (slot, _fmt_u32, bpp, mtl_fmt) = color_meta[i];
        let target_descriptor = TextureDescriptor::new();
        target_descriptor.set_texture_type(MTLTextureType::D2);
        target_descriptor.set_pixel_format(mtl_fmt);
        target_descriptor.set_width(width as u64);
        target_descriptor.set_height(height as u64);
        target_descriptor.set_storage_mode(MTLStorageMode::Shared);
        target_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let Some(target) =
            crate::backend::metal::raw_metal::new_texture(device, &target_descriptor)
        else {
            return Status::execute("metal_render_color_target_alloc_failed")
                .field("slot", slot)
                .field("width", width)
                .field("height", height);
        };
        // Archive reims_vgpu_backend_metal: upload target_rgba8 before Load
        // (fresh RT every job; NULL seed → Clear invent below).
        if let Some(seed) = c.seed_rgba8 {
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: width as u64,
                    height: height as u64,
                    depth: 1,
                },
            };
            target.replace_region(
                region,
                0,
                seed.as_ptr() as *const _,
                (width as u64) * (bpp as u64),
            );
        }
        retained_tex.push(target.clone());
        color_textures.push((slot, target, bpp));
    }
    let mut retained_buf: Vec<Buffer> = Vec::new();

    let pass = RenderPassDescriptor::new();
    for (i, c) in colors.iter().enumerate() {
        let (slot, target, _) = &color_textures[i];
        if let Some(ca) = pass.color_attachments().object_at(*slot as u64) {
            ca.set_texture(Some(target));
            // Ephemeral host RT: archive Load+seed / Clear invent.
            let resolved = color_rt_load_action(c.load_action, c.seed_rgba8.is_some());
            let mtl_load = match resolved {
                x if x == crate::backend::metal::abi::REIMS_VGPU_MTL_LOAD_ACTION_LOAD => {
                    MTLLoadAction::Load
                }
                x if x == crate::backend::metal::abi::REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE => {
                    MTLLoadAction::DontCare
                }
                _ => MTLLoadAction::Clear,
            };
            ca.set_load_action(mtl_load);
            ca.set_clear_color(MTLClearColor::new(
                c.clear_r, c.clear_g, c.clear_b, c.clear_a,
            ));
            ca.set_store_action(MTLStoreAction::Store);
        }
    }

    // Both builders now separate "no attachment" from "an attachment this
    // device will not build", and this function's channel is a bare `Status`,
    // so the refusal is matched out rather than carried by `?`.
    let depth_texture = match configure_depth_attachment(
        device,
        pass,
        &mut retained_tex,
        depth_attachment.as_deref(),
        width,
        height,
        depth_actions,
    ) {
        Ok(texture) => texture,
        Err(status) => return status,
    };
    let stencil_texture = match configure_stencil_attachment(
        device,
        pass,
        &mut retained_tex,
        stencil_attachment.as_deref(),
        width,
        height,
        stencil_actions,
    ) {
        Ok(texture) => texture,
        Err(status) => return status,
    };

    // Armed before the encoder exists, because `visibilityResultBuffer` is a
    // property of the pass descriptor and Metal reads it when the encoder is
    // created. Both refusals below are therefore reachable without an encoder
    // to end.
    let visibility_mode = match visibility.as_ref() {
        None => None,
        Some(q) => {
            let Some(mode) = mtl_enum::visibility_result_mode(q.mode) else {
                set_err(err, format!("unsupported visibility result mode {}", q.mode));
                return Status::args("metal_render_visibility_result_mode_unsupported")
                    .field("mode", q.mode);
            };
            // A **healthy-zero alarm**, not a guest-reachable refusal: no guest
            // action takes this path. `runtime::exec`'s
            // `SetVisibilityResultMode` arm builds a `VisibilityArming` only
            // where the decoded mode is non-zero, so a query struct exists
            // exactly where the stream armed one. A firing means that invariant
            // stopped holding above this backend, and running the pass unarmed
            // instead would report zero fragments visible for a draw nobody
            // asked about — the one wrong answer occlusion culling acts on.
            //
            // So it is not a never-taken branch to delete: it is the thing that
            // says the caller broke.
            if mode == MTLVisibilityResultMode::Disabled {
                set_err(err, "visibility query armed with the disabling mode");
                return Status::args("metal_render_visibility_result_mode_disabled");
            }
            Some(mode)
        }
    };
    // One `u64` at offset 0: this pass encodes one draw, so the buffer holds
    // exactly the one result the guest asked for. Shared so the CPU can read it
    // after the command buffer completes without a blit.
    let visibility_buffer = match visibility_mode {
        None => None,
        Some(_) => {
            let Some(buffer) = crate::backend::metal::raw_metal::new_buffer(
                device,
                core::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            ) else {
                return Status::execute("metal_render_visibility_buffer_alloc_failed")
                    .field("len", core::mem::size_of::<u64>());
            };
            Some(buffer)
        }
    };
    if let Some(buffer) = visibility_buffer.as_ref() {
        // Metal does not document the buffer as zeroed, and the count is an
        // accumulation the GPU adds into.
        unsafe { core::ptr::write_bytes(buffer.contents() as *mut u8, 0, size_of::<u64>()) };
        pass.set_visibility_result_buffer(Some(buffer));
    }

    let queue = thread_queue(device);
    let Some(command_buffer) = crate::backend::metal::raw_metal::new_command_buffer(&queue) else {
        return Status::execute("metal_render_command_buffer_unavailable");
    };
    let command_buffer = command_buffer.to_owned();
    let Some(encoder) =
        crate::backend::metal::raw_metal::new_render_command_encoder(&command_buffer, pass)
    else {
        return Status::execute("metal_render_encoder_unavailable");
    };
    encoder.set_render_pipeline_state(&pso);
    if let Some(mode) = visibility_mode {
        encoder.set_visibility_result_mode(mode, 0);
    }
    apply_blend_color(encoder, blend);
    let rc = apply_raster_state(encoder, raster, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    apply_depth_bias(encoder, depth_bias);
    let rc = apply_depth_stencil_state(device, encoder, depth_stencil, stencil_reference, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    apply_viewports(encoder, viewports, width, height);
    apply_scissors(encoder, scissors, width, height);

    for slot in &attr_slots {
        encoder.set_vertex_buffer(slot.index, Some(&slot.buffer), 0);
    }
    let rc = bind_storage_buffers(device, encoder, &mut retained_buf, buffers, false, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    let rc = bind_storage_buffers(device, encoder, &mut retained_buf, frag_buffers, true, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    let rc = bind_sampled_images(
        device,
        encoder,
        &mut retained_tex,
        vertex_images,
        false,
        err,
    );
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    let rc = bind_samplers(
        device,
        encoder,
        vert_sampler_mask,
        vertex_samplers,
        false,
        err,
    );
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    let rc = bind_sampled_images(device, encoder, &mut retained_tex, images, true, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }
    let rc = bind_samplers(device, encoder, frag_sampler_mask, samplers, true, err);
    if !rc.is_ok() {
        encoder.end_encoding();
        return rc;
    }

    if let Some(pi) = primitive_indirect {
        let need = std::mem::size_of::<ReimsVgpuPrimitiveIndirectArguments>();
        if pi.arguments.is_null() {
            set_err(err, "invalid primitive indirect draw");
            encoder.end_encoding();
            return Status::args("metal_render_primitive_indirect_arguments_missing");
        }
        if pi.arguments_len < need {
            set_err(err, "invalid primitive indirect draw");
            encoder.end_encoding();
            return Status::args("metal_render_primitive_indirect_arguments_too_short")
                .field("len", pi.arguments_len)
                .field("required", need);
        }
        let indirect = unsafe {
            crate::backend::metal::raw_metal::new_buffer_with_data(
                device,
                pi.arguments as *const _,
                pi.arguments_len as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let Some(indirect) = indirect else {
            encoder.end_encoding();
            return Status::execute("metal_render_indirect_buffer_alloc_failed")
                .field("len", pi.arguments_len);
        };
        retained_buf.push(indirect.clone());
        encoder.draw_primitives_indirect(prim, &indirect, 0);
    } else if let Some(ix) = indexed {
        let converted_index_type = mtl_enum::index_type(ix.index_type);
        if ix.indices.is_null() {
            set_err(err, "invalid indexed draw");
            encoder.end_encoding();
            return Status::args("metal_render_index_data_missing");
        }
        if ix.indices_len == 0 {
            set_err(err, "invalid indexed draw");
            encoder.end_encoding();
            return Status::args("metal_render_index_data_empty");
        }
        if !indexed_indirect && ix.index_count == 0 {
            set_err(err, "invalid indexed draw");
            encoder.end_encoding();
            return Status::args("metal_render_index_count_zero");
        }
        // Kept after the three checks above so the refusal order is unchanged;
        // the conversion itself ran before any of them.
        let Some(index_type) = converted_index_type else {
            set_err(err, "invalid indexed draw");
            encoder.end_encoding();
            return Status::args("metal_render_index_type_unsupported")
                .field("index_type", ix.index_type);
        };
        let base_vertex = ix.base_vertex;
        let index_size: usize = match index_type {
            MTLIndexType::UInt16 => 2,
            MTLIndexType::UInt32 => 4,
        };
        if indexed_indirect {
            let ind = unsafe { &*ix.indirect };
            let need = std::mem::size_of::<ReimsVgpuIndexedIndirectArguments>();
            if ind.arguments.is_null() {
                set_err(err, "invalid indexed indirect draw");
                encoder.end_encoding();
                return Status::args("metal_render_indexed_indirect_arguments_missing");
            }
            if ind.arguments_len < need {
                set_err(err, "invalid indexed indirect draw");
                encoder.end_encoding();
                return Status::args("metal_render_indexed_indirect_arguments_too_short")
                    .field("len", ind.arguments_len)
                    .field("required", need);
            }
            let mut args = ReimsVgpuIndexedIndirectArguments {
                index_count: 0,
                instance_count: 0,
                index_start: 0,
                base_vertex: 0,
                base_instance: 0,
            };
            unsafe {
                ptr::copy_nonoverlapping(ind.arguments, &mut args as *mut _ as *mut u8, need);
            }
            let Some(index_end) =
                (args.index_start as usize).checked_add(args.index_count as usize)
            else {
                set_err(err, "indexed indirect index buffer too short");
                encoder.end_encoding();
                return Status::args("metal_render_indexed_indirect_range_overflow")
                    .field("index_start", args.index_start)
                    .field("index_count", args.index_count);
            };
            let Some(index_bytes) = index_end.checked_mul(index_size) else {
                set_err(err, "indexed indirect index buffer too short");
                encoder.end_encoding();
                return Status::args("metal_render_indexed_indirect_byte_count_overflow")
                    .field("index_end", index_end)
                    .field("index_size", index_size);
            };
            if ix.indices_len < index_bytes {
                set_err(err, "indexed indirect index buffer too short");
                encoder.end_encoding();
                return Status::args("metal_render_indexed_indirect_buffer_too_short")
                    .field("index_start", args.index_start)
                    .field("index_count", args.index_count)
                    .field("index_size", index_size)
                    .field("indices_len", ix.indices_len);
            }
        } else {
            if ix.index_count > usize::MAX / index_size {
                set_err(err, "indexed draw byte count overflows");
                encoder.end_encoding();
                return Status::args("metal_render_index_byte_count_overflow")
                    .field("index_count", ix.index_count)
                    .field("index_size", index_size);
            }
            if ix.indices_len < ix.index_count * index_size {
                set_err(err, "index buffer too short");
                encoder.end_encoding();
                return Status::args("metal_render_index_buffer_too_short")
                    .field("index_count", ix.index_count)
                    .field("index_size", index_size)
                    .field("indices_len", ix.indices_len);
            }
        }
        let index_buffer = unsafe {
            crate::backend::metal::raw_metal::new_buffer_with_data(
                device,
                ix.indices as *const _,
                ix.indices_len as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let Some(index_buffer) = index_buffer else {
            encoder.end_encoding();
            return Status::execute("metal_render_index_buffer_alloc_failed")
                .field("len", ix.indices_len);
        };
        retained_buf.push(index_buffer.clone());
        if indexed_indirect {
            let ind = unsafe { &*ix.indirect };
            let indirect = unsafe {
                crate::backend::metal::raw_metal::new_buffer_with_data(
                    device,
                    ind.arguments as *const _,
                    ind.arguments_len as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            };
            let Some(indirect) = indirect else {
                encoder.end_encoding();
                return Status::execute("metal_render_indexed_indirect_buffer_alloc_failed")
                    .field("len", ind.arguments_len);
            };
            retained_buf.push(indirect.clone());
            encoder.draw_indexed_primitives_indirect(
                prim,
                index_type,
                &index_buffer,
                0,
                &indirect,
                0,
            );
        } else {
            encoder.draw_indexed_primitives_instanced_base_instance(
                prim,
                ix.index_count as u64,
                index_type,
                &index_buffer,
                0,
                instance_count as u64,
                base_vertex,
                base_instance as u64,
            );
        }
    } else if base_instance != 0 {
        encoder.draw_primitives_instanced_base_instance(
            prim,
            first_vertex as u64,
            vertex_count as u64,
            instance_count as u64,
            base_instance as u64,
        );
    } else if instance_count != 1 {
        encoder.draw_primitives_instanced(
            prim,
            first_vertex as u64,
            vertex_count as u64,
            instance_count as u64,
        );
    } else {
        encoder.draw_primitives(prim, first_vertex as u64, vertex_count as u64);
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        let detail = command_buffer_error_description(&command_buffer);
        set_err(err, format!("Metal command buffer failed: {detail}"));
        return Status::execute("metal_render_command_buffer_failed");
    }

    // Read after the completion check, not before it: a command buffer that
    // errored ran no query, and answering the guest out of an untouched Shared
    // buffer would report zero fragments visible for a draw that never
    // rasterized — the one wrong answer occlusion culling acts on.
    if let (Some(query), Some(buffer)) = (visibility, visibility_buffer.as_ref()) {
        query.samples = Some(unsafe { core::ptr::read_unaligned(buffer.contents() as *const u64) });
    }

    for (i, c) in colors.iter_mut().enumerate() {
        if let Some(out) = c.out_rgba8.as_mut() {
            if out.is_empty() {
                continue;
            }
            let (slot, target, bpp) = &color_textures[i];
            let _ = slot;
            let target_len = (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(*bpp);
            let mut readback = vec![0u8; target_len];
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize {
                    width: width as u64,
                    height: height as u64,
                    depth: 1,
                },
            };
            target.get_bytes(
                readback.as_mut_ptr() as *mut _,
                (width as u64) * (*bpp as u64),
                region,
                0,
            );
            let n = out.len().min(target_len);
            out[..n].copy_from_slice(&readback[..n]);
        }
    }
    color_textures.clear();
    if let Some(depth) = depth_attachment {
        if depth.store_action == REIMS_VGPU_MTL_STORE_ACTION_STORE {
            if let Some(tex) = depth_texture {
                let region = MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: width as u64,
                        height: height as u64,
                        depth: 1,
                    },
                };
                tex.get_bytes(
                    depth.data as *mut _,
                    (width as u64) * (std::mem::size_of::<f32>() as u64),
                    region,
                    0,
                );
            }
        }
    }
    if let Some(stencil) = stencil_attachment {
        if stencil.store_action == REIMS_VGPU_MTL_STORE_ACTION_STORE {
            if let Some(tex) = stencil_texture {
                let region = MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: width as u64,
                        height: height as u64,
                        depth: 1,
                    },
                };
                tex.get_bytes(stencil.data as *mut _, width as u64, region, 0);
            }
        }
    }
    clear_err(err);
    let _ = retained_buf;
    Status::OK
}
