//! The render-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_RENDER`](super::RGFX_OP_GROUP_RENDER)):
//! one tag and one body struct per `MTLRenderCommandEncoder` selector the guest
//! serialises. The host's render decoder (`reims-vgpu`'s `graphics::decode::render`)
//! reads the body, resolves its object references and sends the matching
//! `RenderCmd` (docs/graphics-kext/design.md, "Stage 3 wire protocol").
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil). Floating-point arguments travel as the bits of their
//! `f64`/`f32`, so the schema stays all-integer and comparable, as the sampler's
//! LOD clamps do in [`crate::wire`].
//!
//! This first slice carries the reference subset the host decoder implements —
//! enough to prove the pattern end to end; the rest of the ~70 render selectors
//! extend this module, one tag and one body struct each, their tags staying in
//! the `0x1xxx` range.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Op `SET_RENDER_PIPELINE_STATE` ([`RgfxOpSetRenderPipelineState`]):
/// `setRenderPipelineState:`.
pub const RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE: u16 = 0x1000;
/// Op `SET_VERTEX_BUFFER` ([`RgfxOpSetVertexBuffer`]):
/// `setVertexBuffer:offset:atIndex:`.
pub const RGFX_OP_RENDER_SET_VERTEX_BUFFER: u16 = 0x1001;
/// Op `SET_FRAGMENT_BUFFER` ([`RgfxOpSetFragmentBuffer`]):
/// `setFragmentBuffer:offset:atIndex:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_BUFFER: u16 = 0x1002;
/// Op `SET_FRAGMENT_TEXTURE` ([`RgfxOpSetFragmentTexture`]):
/// `setFragmentTexture:atIndex:`.
pub const RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE: u16 = 0x1003;
/// Op `SET_VIEWPORT` ([`RgfxOpSetViewport`]): `setViewport:`.
pub const RGFX_OP_RENDER_SET_VIEWPORT: u16 = 0x1004;
/// Op `SET_SCISSOR_RECT` ([`RgfxOpSetScissorRect`]): `setScissorRect:`.
pub const RGFX_OP_RENDER_SET_SCISSOR_RECT: u16 = 0x1005;
/// Op `DRAW_PRIMITIVES` ([`RgfxOpDrawPrimitives`]):
/// `drawPrimitives:vertexStart:vertexCount:`.
pub const RGFX_OP_RENDER_DRAW_PRIMITIVES: u16 = 0x1006;
/// Op `DRAW_INDEXED_PRIMITIVES` ([`RgfxOpDrawIndexedPrimitives`]):
/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`.
pub const RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES: u16 = 0x1007;

/// `setRenderPipelineState:`. The pipeline may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetRenderPipelineState {
    /// The render pipeline state, or 0 for nil.
    pub pipeline_obj_id: u64,
}

/// `setVertexBuffer:offset:atIndex:`. The buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetVertexBuffer {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setFragmentBuffer:offset:atIndex:`. The buffer may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentBuffer {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setFragmentTexture:atIndex:`. The texture may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetFragmentTexture {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The texture binding index.
    pub index: u64,
}

/// `setViewport:`. The six `MTLViewport` fields travel as the bits of their `f64`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetViewport {
    /// `originX` as `f64` bits.
    pub origin_x_bits: u64,
    /// `originY` as `f64` bits.
    pub origin_y_bits: u64,
    /// `width` as `f64` bits.
    pub width_bits: u64,
    /// `height` as `f64` bits.
    pub height_bits: u64,
    /// `znear` as `f64` bits.
    pub znear_bits: u64,
    /// `zfar` as `f64` bits.
    pub zfar_bits: u64,
}

/// `setScissorRect:`. The four `MTLScissorRect` fields are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpSetScissorRect {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

/// `drawPrimitives:vertexStart:vertexCount:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawPrimitives {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The first vertex.
    pub vertex_start: u64,
    /// The vertices to draw.
    pub vertex_count: u64,
}

/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:`.
/// The index buffer must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxOpDrawIndexedPrimitives {
    /// `MTLPrimitiveType`.
    pub primitive_type: u64,
    /// The indices to draw.
    pub index_count: u64,
    /// `MTLIndexType`.
    pub index_type: u64,
    /// The index buffer (must not be nil).
    pub index_buffer_obj_id: u64,
    /// The offset into the index buffer.
    pub index_buffer_offset: u64,
}

// The render ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpSetRenderPipelineState>() == 8);
    assert!(size_of::<RgfxOpSetVertexBuffer>() == 24);
    assert!(size_of::<RgfxOpSetFragmentBuffer>() == 24);
    assert!(size_of::<RgfxOpSetFragmentTexture>() == 16);
    assert!(size_of::<RgfxOpSetViewport>() == 48);
    assert!(size_of::<RgfxOpSetScissorRect>() == 32);
    assert!(size_of::<RgfxOpDrawPrimitives>() == 24);
    assert!(size_of::<RgfxOpDrawIndexedPrimitives>() == 40);
    // The render tags stay in the render group.
    assert!(
        super::op_group(RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE) == super::RGFX_OP_GROUP_RENDER
    );
    assert!(super::op_group(RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES) == super::RGFX_OP_GROUP_RENDER);
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// Each render op body round-trips, floats travelling as bits.
    #[test]
    fn render_ops_round_trip() {
        let vb = RgfxOpSetVertexBuffer {
            buffer_obj_id: 4,
            offset: 256,
            index: 2,
        };
        assert_eq!(
            RgfxOpSetVertexBuffer::read_from_bytes(vb.as_bytes()).unwrap(),
            vb
        );

        let vp = RgfxOpSetViewport {
            origin_x_bits: 0.0f64.to_bits(),
            origin_y_bits: 0.0f64.to_bits(),
            width_bits: 1920.0f64.to_bits(),
            height_bits: 1080.0f64.to_bits(),
            znear_bits: 0.0f64.to_bits(),
            zfar_bits: 1.0f64.to_bits(),
        };
        assert_eq!(
            RgfxOpSetViewport::read_from_bytes(vp.as_bytes()).unwrap(),
            vp
        );
        assert_eq!(f64::from_bits(vp.width_bits), 1920.0);

        let di = RgfxOpDrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 6,
            index_type: 1,
            index_buffer_obj_id: 9,
            index_buffer_offset: 0,
        };
        assert_eq!(
            RgfxOpDrawIndexedPrimitives::read_from_bytes(di.as_bytes()).unwrap(),
            di
        );
    }
}
