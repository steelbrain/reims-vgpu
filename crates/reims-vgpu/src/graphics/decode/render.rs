//! The render-encoder op decoder: one arm per [`reims_graphics::ops::render`] op,
//! turning a wire op into a `RenderCmd` on the open `P::RenderEncoder`
//! (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The command-buffer
//! walk ([`super::cmdbuf`]) routes every render-group op
//! (`tag >> 12 == `[`RGFX_OP_GROUP_RENDER`](reims_graphics::ops::RGFX_OP_GROUP_RENDER))
//! here.
//!
//! Each arm reads its op's body with zerocopy, resolves its object references
//! through the [object-resolution API](super) (`require_*`/`get_*`), and sends the
//! matching [`RenderCmd`] with [`RenderEncoder::encode`]. A body too short for its
//! op faults the exec ([`super::op_fault`]); a bad reference faults through the
//! resolution API.
//!
//! This first slice carries a reference subset — pipeline, vertex/fragment binds,
//! viewport and scissor, and the two common draws — that proves the pattern end to
//! end. The rest of the ~70 render ops extend this `match`, one arm each.

use reims_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES, RGFX_OP_RENDER_DRAW_PRIMITIVES,
    RGFX_OP_RENDER_SET_FRAGMENT_BUFFER, RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RGFX_OP_RENDER_SET_SCISSOR_RECT,
    RGFX_OP_RENDER_SET_VERTEX_BUFFER, RGFX_OP_RENDER_SET_VIEWPORT, RgfxOpDrawIndexedPrimitives,
    RgfxOpDrawPrimitives, RgfxOpSetFragmentBuffer, RgfxOpSetFragmentTexture,
    RgfxOpSetRenderPipelineState, RgfxOpSetScissorRect, RgfxOpSetVertexBuffer, RgfxOpSetViewport,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::types::{MtlScissorRect, MtlViewport};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use zerocopy::FromBytes;

use super::{GraphicsDecoder, op_fault};

/// Reads op body `T` from `payload`, or an exec fault when it is too short.
fn body<T: FromBytes>(tag: u16, payload: &[u8]) -> RaiseResult<T> {
    match T::read_from_prefix(payload) {
        Ok((t, _)) => Ok(t),
        Err(_) => Err(op_fault(format_args!(
            "render op {tag:#06x}: body too short"
        ))),
    }
}

/// Decodes one render-encoder op: `tag` its op, `payload` its body (the bytes
/// after the [`RgfxOpHeader`](reims_graphics::ops::RgfxOpHeader)). `objs` resolves
/// object references, `enc` is the open render encoder, and `on` says how an
/// encode's raise is taken. `Ok(true)` when the op was handled, `Ok(false)` for an
/// unknown tag (the walk then faults); a platform raise or a bad reference
/// propagates as `Err`.
pub(crate) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    enc: &P::RenderEncoder,
    on: OnFault,
) -> RaiseResult<bool> {
    match tag {
        RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE => {
            let m: RgfxOpSetRenderPipelineState = body(tag, payload)?;
            let pipeline = objs.get_render_pipeline(m.pipeline_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetRenderPipelineState {
                    pipeline_state: pipeline.as_ref(),
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VERTEX_BUFFER => {
            let m: RgfxOpSetVertexBuffer = body(tag, payload)?;
            let buffer = objs.get_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetVertexBuffer {
                    buffer: buffer.as_ref(),
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_BUFFER => {
            let m: RgfxOpSetFragmentBuffer = body(tag, payload)?;
            let buffer = objs.get_buffer(m.buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetFragmentBuffer {
                    buffer: buffer.as_ref(),
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE => {
            let m: RgfxOpSetFragmentTexture = body(tag, payload)?;
            let texture = objs.get_texture(m.texture_obj_id)?;
            enc.encode(
                on,
                RenderCmd::SetFragmentTexture {
                    texture: texture.as_ref(),
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_RENDER_SET_VIEWPORT => {
            let m: RgfxOpSetViewport = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetViewport {
                    viewport: MtlViewport {
                        origin_x: f64::from_bits(m.origin_x_bits),
                        origin_y: f64::from_bits(m.origin_y_bits),
                        width: f64::from_bits(m.width_bits),
                        height: f64::from_bits(m.height_bits),
                        znear: f64::from_bits(m.znear_bits),
                        zfar: f64::from_bits(m.zfar_bits),
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_SET_SCISSOR_RECT => {
            let m: RgfxOpSetScissorRect = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::SetScissorRect {
                    rect: MtlScissorRect {
                        x: m.x,
                        y: m.y,
                        width: m.width,
                        height: m.height,
                    },
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_PRIMITIVES => {
            let m: RgfxOpDrawPrimitives = body(tag, payload)?;
            enc.encode(
                on,
                RenderCmd::DrawPrimitives {
                    primitive_type: m.primitive_type,
                    vertex_start: m.vertex_start,
                    vertex_count: m.vertex_count,
                },
            )?;
        }
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES => {
            let m: RgfxOpDrawIndexedPrimitives = body(tag, payload)?;
            // The index buffer must name an object (an indexed draw without one).
            let index_buffer = objs.require_buffer(m.index_buffer_obj_id)?;
            enc.encode(
                on,
                RenderCmd::DrawIndexedPrimitives {
                    primitive_type: m.primitive_type,
                    index_count: m.index_count,
                    index_type: m.index_type,
                    index_buffer: &index_buffer,
                    index_buffer_offset: m.index_buffer_offset,
                },
            )?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}
