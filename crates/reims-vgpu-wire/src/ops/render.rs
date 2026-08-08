//! Render encoder records.
//!
//! These are the records a `PGSerializerRenderCommandEncoder` writes through
//! `PGSerializerCommandStream`'s `-getCommandBufferBytes:`. Each one is the
//! shared 8-byte [`crate::op::OpHeader`] followed by a per-opcode payload.
//!
//! Every layout below was derived by calling the Metal method with distinctive
//! arguments and reading the bytes back; the fixture that pins each is named in
//! its doc. See `oracle/oracle.m`'s `encoderCases`.
//!
//! # Every draw has two encodings, and the guest picks by magnitude
//!
//! Opcodes `0x00`–`0x0b` are six draw selectors in twelve opcodes: each one has
//! a **compact** form whose counts are 16-bit and a **wide** form whose counts
//! are 64-bit. The serializer emits the wide form when any of `vertexStart`,
//! `vertexCount`, `instanceCount`, `baseInstance`, `indexCount` or
//! `indexBufferOffset` exceeds `0xffff`, and the compact form otherwise. The
//! even opcode is always the wide sibling of the odd one above it:
//!
//! | selector | compact | wide |
//! |---|---|---|
//! | `drawPrimitives:vertexStart:vertexCount:` | [`OPCODE_DRAW`] `0x01` | [`OPCODE_DRAW_WIDE`] `0x00` |
//! | …`:instanceCount:` | [`OPCODE_DRAW_INSTANCED`] `0x03` | [`OPCODE_DRAW_INSTANCED_WIDE`] `0x02` |
//! | …`:instanceCount:baseInstance:` | [`OPCODE_DRAW_INSTANCED_BASE`] `0x05` | [`OPCODE_DRAW_INSTANCED_BASE_WIDE`] `0x04` |
//! | `drawIndexedPrimitives:…:indexBufferOffset:` | [`OPCODE_DRAW_INDEXED`] `0x07` | [`OPCODE_DRAW_INDEXED_WIDE`] `0x06` |
//! | …`:instanceCount:` | [`OPCODE_DRAW_INDEXED_INSTANCED`] `0x09` | [`OPCODE_DRAW_INDEXED_INSTANCED_WIDE`] `0x08` |
//! | …`:instanceCount:baseVertex:baseInstance:` | [`OPCODE_DRAW_INDEXED_INSTANCED_BASE`] `0x0b` | [`OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE`] `0x0a` |
//!
//! The boundary is bracketed rather than assumed: `vertexCount = 0xffff` stays
//! compact (`render_draw_primitives_count_at_16bit_max`) and `0x10000` goes wide
//! (`render_draw_primitives_count_over_16bit`), and each of the other five
//! arguments has a fixture that crosses it alone. One argument crossing widens
//! **every** field in the record, not just its own.
//!
//! `baseVertex` is the exception and is not part of the test: it is Metal's only
//! signed draw argument, and the serializer truncates it to 16 bits in the
//! compact form rather than widening the record. `baseVertex = -70000` came back
//! as `0xee90` in a compact record
//! (`render_draw_indexed_base_vertex_below_i16`), which is that value's low half
//! — Apple's own serializer loses it, so a guest cannot express it and this
//! device cannot recover it.
//!
//! # Relationship to `reims_vgpu::runtime::decode::render`
//!
//! Six opcodes match that module's constants exactly, which is two independent
//! derivations agreeing. The rest of the draw family does not, and each
//! divergence is named in the view that settles it: see [`DrawWide`],
//! [`DrawIndexed`] and [`DrawIndexedWide`].

use crate::le::{F32le, F64le, I16le, I64le, U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

// --- 0x01 drawPrimitives:vertexStart:vertexCount: --------------------------

pub const OPCODE_DRAW: u32 = 0x01;
pub const DRAW_TOTAL_LEN: u32 = 16;

/// Payload of a non-instanced draw.
///
/// Fixtures `render_draw_primitives` (Triangle, 7, 11) and
/// `render_draw_primitives_strip` (TriangleStrip, 2, 5). The type is the
/// `MTLPrimitiveType` ordinal carried straight through — Triangle→3,
/// TriangleStrip→4 — and the two counts are 16-bit, not 32.
#[repr(C)]
#[derive(Debug)]
pub struct Draw {
    pub primitive_type: U32le,
    pub vertex_start: U16le,
    pub vertex_count: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for Draw {}

pub fn draw<'a>(op: &Op<'a>) -> Result<&'a Draw, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW);
    view::<Draw>(op.payload)
}

// --- 0x00 drawPrimitives:vertexStart:vertexCount:, wide --------------------

pub const OPCODE_DRAW_WIDE: u32 = 0x00;
pub const DRAW_WIDE_TOTAL_LEN: u32 = 28;

/// Payload of a non-instanced draw whose counts do not fit 16 bits.
///
/// Decode used to decline this opcode as unsupported while its comment guessed
/// a layout of `u64 · u64 · u32 primitiveType@0x10`. That guess was wrong:
/// `primitiveType` is **first and 32-bit**, exactly as in the compact [`Draw`],
/// and the two counts follow it — the same field order as `0x01` with each
/// count widened. Decode now maps this view. Fixtures
/// `render_draw_primitives_wide` (Triangle, `0x11111`, `0x22222`),
/// `render_draw_primitives_count_over_16bit` and
/// `render_draw_primitives_start_over_16bit`.
#[repr(C)]
#[derive(Debug)]
pub struct DrawWide {
    pub primitive_type: U32le,
    pub vertex_start: U64le,
    pub vertex_count: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawWide {}

pub fn draw_wide<'a>(op: &Op<'a>) -> Result<&'a DrawWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_WIDE);
    view::<DrawWide>(op.payload)
}

// --- 0x03 drawPrimitives:...:instanceCount: --------------------------------

pub const OPCODE_DRAW_INSTANCED: u32 = 0x03;
pub const DRAW_INSTANCED_TOTAL_LEN: u32 = 16;

/// Payload of an instanced draw with no base instance.
///
/// The field order is the instanced family's, not [`Draw`]'s: the counts lead
/// and `primitiveType` is last and 16-bit. Fixtures
/// `render_draw_primitives_instanced` (TriangleStrip, `0x1111`, `0x2222`,
/// `0x3333`) and `render_draw_primitives_instances_over_16bit`.
///
/// Decode maps this layout through [`draw_instanced`]; fixtures and live
/// WebKit captures agree field for field.
#[repr(C)]
#[derive(Debug)]
pub struct DrawInstanced {
    pub vertex_start: U16le,
    pub vertex_count: U16le,
    pub instance_count: U16le,
    pub primitive_type: U16le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawInstanced {}

pub fn draw_instanced<'a>(op: &Op<'a>) -> Result<&'a DrawInstanced, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INSTANCED);
    view::<DrawInstanced>(op.payload)
}

// --- 0x02 drawPrimitives:...:instanceCount:, wide --------------------------

pub const OPCODE_DRAW_INSTANCED_WIDE: u32 = 0x02;
pub const DRAW_INSTANCED_WIDE_TOTAL_LEN: u32 = 36;

/// Wide form of [`DrawInstanced`]: the same field order with 64-bit counts.
///
/// Fixtures `render_draw_primitives_instanced_wide` and
/// `render_draw_primitives_instances_over_16bit`, the second of which leaves
/// both vertex counts small and moves only `instanceCount` across the boundary
/// — so all three widened because one did.
///
/// 26 bytes of a 28-byte payload; the last two stayed `0xAA` poison and are
/// uninitialized on the wire.
#[repr(C)]
#[derive(Debug)]
pub struct DrawInstancedWide {
    pub vertex_start: U64le,
    pub vertex_count: U64le,
    pub instance_count: U64le,
    pub primitive_type: U16le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawInstancedWide {}

pub fn draw_instanced_wide<'a>(op: &Op<'a>) -> Result<&'a DrawInstancedWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INSTANCED_WIDE);
    view::<DrawInstancedWide>(op.payload)
}

// --- 0x05 drawPrimitives:...:instanceCount:baseInstance: -------------------

/// Instanced draw carrying a base instance.
///
/// **`reims_vgpu::runtime::decode::render` has no constant for `0x05`.** Its
/// table runs `0x00, 0x01, 0x03, 0x06, 0x07, 0x09`, and `opcode_supported`
/// accepts everything up to [`OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE`]
/// (`0xa6`), so a `0x05` record
/// reaches the catch-all and becomes `Kind::OtherAccepted` — accepted, reported
/// once by `note_unimplemented_render_opcode`, and executed by nothing. The
/// guest's draw does not happen.
///
/// This is the layout — Metal was asked for `(Triangle, start 1, count 2,
/// instances 3, baseInstance 4)` and those five values came back in this order
/// (fixture `render_draw_primitives_instanced_base`).
pub const OPCODE_DRAW_INSTANCED_BASE: u32 = 0x05;
pub const DRAW_INSTANCED_BASE_TOTAL_LEN: u32 = 20;

/// Payload of an instanced draw with a base instance.
///
/// Ten bytes inside a twelve-byte payload. The serializer never writes the
/// final two — they stayed `0xAA` poison in the capture — so they are
/// uninitialized on the wire and hold whatever the guest's ring last contained.
/// This struct is deliberately 10 bytes so a view cannot read them.
#[repr(C)]
#[derive(Debug)]
pub struct DrawInstancedBase {
    pub vertex_start: U16le,
    pub vertex_count: U16le,
    pub instance_count: U16le,
    pub base_instance: U16le,
    /// Last, unlike [`Draw`], where it is first and 32-bit.
    pub primitive_type: U16le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawInstancedBase {}

pub fn draw_instanced_base<'a>(op: &Op<'a>) -> Result<&'a DrawInstancedBase, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INSTANCED_BASE);
    view::<DrawInstancedBase>(op.payload)
}

// --- 0x04 drawPrimitives:...:instanceCount:baseInstance:, wide -------------

pub const OPCODE_DRAW_INSTANCED_BASE_WIDE: u32 = 0x04;
pub const DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN: u32 = 44;

/// Wide form of [`DrawInstancedBase`]. Fixtures
/// `render_draw_primitives_instanced_base_wide` and
/// `render_draw_primitives_base_over_16bit`, the second moving only
/// `baseInstance` across the boundary.
///
/// 34 bytes of a 36-byte payload; the last two are poison, as in every other
/// draw whose trailing `primitiveType` is 16-bit.
#[repr(C)]
#[derive(Debug)]
pub struct DrawInstancedBaseWide {
    pub vertex_start: U64le,
    pub vertex_count: U64le,
    pub instance_count: U64le,
    pub base_instance: U64le,
    pub primitive_type: U16le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawInstancedBaseWide {}

pub fn draw_instanced_base_wide<'a>(op: &Op<'a>) -> Result<&'a DrawInstancedBaseWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INSTANCED_BASE_WIDE);
    view::<DrawInstancedBaseWide>(op.payload)
}

// --- 0x07 drawIndexedPrimitives:…:indexBufferOffset: -----------------------

pub const OPCODE_DRAW_INDEXED: u32 = 0x07;
pub const DRAW_INDEXED_TOTAL_LEN: u32 = 20;

/// Payload of an indexed draw.
///
/// The index buffer is named by its serializer resource ref, not by an address:
/// the stub buffer's ref `5151` came back at `+4` unchanged.
///
/// **`reims_vgpu::runtime::decode::render` reads the first four bytes as one
/// `u32 primitiveType` and hardcodes `index_type = 0`.** That is right only
/// while `indexType` is `MTLIndexTypeUInt16`, because the ordinal is 0 and the
/// two halves of the word are indistinguishable. Fixture
/// `render_draw_indexed_uint32` is the case that separates them: with
/// `MTLIndexTypeUInt32` the word reads `04 00 01 00`, so that decoder yields
/// `primitiveType = 0x10004` — no such Metal primitive — and still reports
/// UInt16 for a 32-bit index buffer. A guest drawing with 32-bit indices is
/// mis-drawn twice over, silently.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexed {
    pub primitive_type: U16le,
    /// `MTLIndexType`: UInt16→0, UInt32→1 (fixtures `render_draw_indexed` and
    /// `render_draw_indexed_uint32`).
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_count: U16le,
    pub index_buffer_offset: U16le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexed {}

pub fn draw_indexed<'a>(op: &Op<'a>) -> Result<&'a DrawIndexed, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED);
    view::<DrawIndexed>(op.payload)
}

// --- 0x06 drawIndexedPrimitives:…:indexBufferOffset:, wide -----------------

pub const OPCODE_DRAW_INDEXED_WIDE: u32 = 0x06;
pub const DRAW_INDEXED_WIDE_TOTAL_LEN: u32 = 32;

/// Wide form of [`DrawIndexed`]. Fixtures
/// `render_draw_indexed_count_over_16bit` and
/// `render_draw_indexed_offset_over_16bit`, one per widening argument.
///
/// `reims_vgpu::runtime::decode::render` reads this opcode's head correctly —
/// `u16 primitiveType`, `u16 indexType`, `u32 indexBufferRef` — and is the only
/// place in that module where `indexType` comes off the wire at all. It then
/// describes the counts as `u32 indexCount@8`, `u32 pad@0xc`, `u32
/// indexBufferOffset@0x10`, `u32 pad@0x14`. Those "pads" are the upper halves of
/// two 64-bit fields, which reads the same on little-endian for values below
/// 2³² and differently above it.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedWide {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_count: U64le,
    pub index_buffer_offset: U64le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedWide {}

pub fn draw_indexed_wide<'a>(op: &Op<'a>) -> Result<&'a DrawIndexedWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_WIDE);
    view::<DrawIndexedWide>(op.payload)
}

// --- 0x09 drawIndexedPrimitives:…:instanceCount: ---------------------------

pub const OPCODE_DRAW_INDEXED_INSTANCED: u32 = 0x09;
pub const DRAW_INDEXED_INSTANCED_TOTAL_LEN: u32 = 24;

/// [`DrawIndexed`] with an instance count appended. Fixture
/// `render_draw_indexed_instanced`.
///
/// 14 bytes of a 16-byte payload are written; the last two are poison.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedInstanced {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_count: U16le,
    pub index_buffer_offset: U16le,
    pub instance_count: U16le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedInstanced {}

pub fn draw_indexed_instanced<'a>(op: &Op<'a>) -> Result<&'a DrawIndexedInstanced, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_INSTANCED);
    view::<DrawIndexedInstanced>(op.payload)
}

// --- 0x08 drawIndexedPrimitives:…:instanceCount:, wide ---------------------

pub const OPCODE_DRAW_INDEXED_INSTANCED_WIDE: u32 = 0x08;
pub const DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN: u32 = 40;

/// Wide form of [`DrawIndexedInstanced`]. Fixture
/// `render_draw_indexed_instances_over_16bit`, which leaves the index count and
/// offset small and moves only `instanceCount` across the boundary.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedInstancedWide {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_count: U64le,
    pub index_buffer_offset: U64le,
    pub instance_count: U64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedInstancedWide {}

pub fn draw_indexed_instanced_wide<'a>(
    op: &Op<'a>,
) -> Result<&'a DrawIndexedInstancedWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_INSTANCED_WIDE);
    view::<DrawIndexedInstancedWide>(op.payload)
}

// --- 0x0b drawIndexedPrimitives:…:baseVertex:baseInstance: -----------------

pub const OPCODE_DRAW_INDEXED_INSTANCED_BASE: u32 = 0x0b;
pub const DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN: u32 = 28;

/// The full indexed draw, with a base vertex and a base instance.
///
/// **The offset comes before the count here**, which is the opposite of
/// [`DrawIndexed`] and [`DrawIndexedInstanced`] and the reason this record has
/// two fixtures rather than one: `render_draw_indexed_instanced_base` (count
/// `0x1111`, offset `0x2222`) and `render_draw_indexed_instanced_base_alt`
/// (count `0x6666`, offset `0x7777`, and every other field moved with them). A
/// single case cannot tell a swap from a coincidence, and reading this record
/// with the siblings' order swaps a guest's index count and buffer offset.
///
/// 18 bytes of a 20-byte payload are written; the last two are poison.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedInstancedBase {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_buffer_offset: U16le,
    pub index_count: U16le,
    pub instance_count: U16le,
    /// Signed, and **truncated rather than widened**: this argument is not part
    /// of the compact/wide test, so `-70000` arrives here as `0xee90`
    /// (`render_draw_indexed_base_vertex_below_i16`) and `-2` as `0xfffe`
    /// (`render_draw_indexed_negative_base_vertex`). The loss is Apple's
    /// serializer's, upstream of the wire.
    pub base_vertex: I16le,
    pub base_instance: U16le,
}

// SAFETY: eight align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedInstancedBase {}

pub fn draw_indexed_instanced_base<'a>(
    op: &Op<'a>,
) -> Result<&'a DrawIndexedInstancedBase, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_INSTANCED_BASE);
    view::<DrawIndexedInstancedBase>(op.payload)
}

// --- 0x0a drawIndexedPrimitives:…:baseVertex:baseInstance:, wide -----------

pub const OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE: u32 = 0x0a;
pub const DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN: u32 = 56;

/// Wide form of [`DrawIndexedInstancedBase`], keeping its offset-before-count
/// order. Fixtures `render_draw_indexed_base_instances_over_16bit` and
/// `render_draw_indexed_base_instance_over_16bit`, which widen the record
/// through `instanceCount` and `baseInstance` respectively.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedInstancedBaseWide {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub index_buffer_offset: U64le,
    pub index_count: U64le,
    pub instance_count: U64le,
    /// Sign-extended to the full width, unlike the compact form's truncation:
    /// `baseVertex = -70000` in a record widened by another argument reads
    /// `0xfffffffffffeee90` (`render_draw_indexed_wide_negative_base_vertex`).
    pub base_vertex: I64le,
    pub base_instance: U64le,
}

// SAFETY: eight align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedInstancedBaseWide {}

pub fn draw_indexed_instanced_base_wide<'a>(
    op: &Op<'a>,
) -> Result<&'a DrawIndexedInstancedBaseWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE);
    view::<DrawIndexedInstancedBaseWide>(op.payload)
}

// --- 0x75 setScissorRect: --------------------------------------------------

pub const OPCODE_SET_SCISSOR: u32 = 0x75;
pub const SET_SCISSOR_TOTAL_LEN: u32 = 40;

/// `MTLScissorRect` — four `NSUInteger`, so four 64-bit fields.
/// Fixture `render_set_scissor` (1, 2, 300, 400).
#[repr(C)]
#[derive(Debug)]
pub struct ScissorRect {
    pub x: U64le,
    pub y: U64le,
    pub width: U64le,
    pub height: U64le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ScissorRect {}

pub fn set_scissor<'a>(op: &Op<'a>) -> Result<&'a ScissorRect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_SCISSOR);
    view::<ScissorRect>(op.payload)
}

// --- 0x82 setViewport: -----------------------------------------------------

pub const OPCODE_SET_VIEWPORT: u32 = 0x82;
pub const SET_VIEWPORT_TOTAL_LEN: u32 = 56;

/// `MTLViewport` — six `double`. Fixture `render_set_viewport`
/// (0, 0, 640, 480, 0, 1).
///
/// Note the width contrast with [`BlendColor`], which is 32-bit: this protocol
/// carries both float widths and they are not interchangeable.
#[repr(C)]
#[derive(Debug)]
pub struct Viewport {
    pub origin_x: F64le,
    pub origin_y: F64le,
    pub width: F64le,
    pub height: F64le,
    pub znear: F64le,
    pub zfar: F64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for Viewport {}

pub fn set_viewport<'a>(op: &Op<'a>) -> Result<&'a Viewport, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VIEWPORT);
    view::<Viewport>(op.payload)
}

// --- One-`NSUInteger` state records ----------------------------------------

pub const OPCODE_SET_CULL_MODE: u32 = 0x6b;
pub const OPCODE_SET_FRONT_FACING: u32 = 0x73;
pub const OPCODE_SET_DEPTH_CLIP_MODE: u32 = 0x6d;
pub const OPCODE_SET_TRIANGLE_FILL_MODE: u32 = 0x7c;
/// Every single-`NSUInteger` state record is 16 bytes.
pub const SET_MODE_TOTAL_LEN: u32 = 16;

/// A one-`NSUInteger` state record. Six selectors share this shape.
///
/// Fixtures `render_set_cull_mode` (`MTLCullModeBack` = 2),
/// `render_set_front_facing` (`MTLWindingCounterClockwise` = 1),
/// `render_set_depth_clip_mode` (`MTLDepthClipModeClamp` = 1),
/// `render_set_triangle_fill_mode` (`MTLTriangleFillModeLines` = 1),
/// `render_set_depth_store_action` (`MTLStoreActionStore` = 1) and
/// `render_set_stencil_store_action` (`MTLStoreActionDontCare` = 0) — depth and
/// stencil have one attachment each, so unlike a colour store action their
/// records carry no index. The upper four bytes read zero rather than poison,
/// so the field is 64-bit and written whole — not a 32-bit value beside
/// uninitialized padding. Each selector's type encoding declares `Q`, which
/// agrees.
#[repr(C)]
#[derive(Debug)]
pub struct ModeState {
    pub mode: U64le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for ModeState {}

/// Whether `opcode` is one of the one-`NSUInteger` state records.
#[inline]
pub fn is_mode_state(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_CULL_MODE
            | OPCODE_SET_FRONT_FACING
            | OPCODE_SET_DEPTH_CLIP_MODE
            | OPCODE_SET_TRIANGLE_FILL_MODE
            | OPCODE_SET_DEPTH_STORE_ACTION
            | OPCODE_SET_STENCIL_STORE_ACTION
    )
}

/// The mode of any of the four records above. Which state it sets is the
/// opcode's business, not this view's — the record is identical for all four.
pub fn mode_state<'a>(op: &Op<'a>) -> Result<&'a ModeState, WireError> {
    debug_assert!(is_mode_state(op.opcode()));
    view::<ModeState>(op.payload)
}

pub fn set_cull_mode<'a>(op: &Op<'a>) -> Result<&'a ModeState, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_CULL_MODE);
    view::<ModeState>(op.payload)
}

pub fn set_front_facing<'a>(op: &Op<'a>) -> Result<&'a ModeState, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_FRONT_FACING);
    view::<ModeState>(op.payload)
}

// --- One-`float` state records ---------------------------------------------

pub const OPCODE_SET_LINE_WIDTH: u32 = 0x88;
pub const OPCODE_SET_TESSELLATION_FACTOR_SCALE: u32 = 0x7b;
/// A single 32-bit float, in the shortest record this encoder emits.
pub const SET_FLOAT_TOTAL_LEN: u32 = 12;

/// A one-`float` state record, shared by line width and tessellation factor
/// scale. Fixtures `render_set_line_width` (2.5) and
/// `render_set_tessellation_factor_scale` (1.25), both exact in binary so a
/// float/double confusion would read as a wrong number rather than a rounding
/// difference.
///
/// Twelve bytes total: the payload is four, not eight. This is the record that
/// shows the serializer does not pad a short payload out to eight — the length
/// only has to be a multiple of four.
#[repr(C)]
#[derive(Debug)]
pub struct FloatState {
    pub value: F32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for FloatState {}

#[inline]
pub fn is_float_state(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_LINE_WIDTH | OPCODE_SET_TESSELLATION_FACTOR_SCALE
    )
}

pub fn float_state<'a>(op: &Op<'a>) -> Result<&'a FloatState, WireError> {
    debug_assert!(is_float_state(op.opcode()));
    view::<FloatState>(op.payload)
}

// --- 0x77 setStencilReferenceValue: / …Front:back: -------------------------

pub const OPCODE_SET_STENCIL_REFERENCE: u32 = 0x77;
pub const SET_STENCIL_REFERENCE_TOTAL_LEN: u32 = 16;

/// Stencil reference values, front and back.
///
/// **Two selectors emit this one opcode.** The one-argument
/// `setStencilReferenceValue:` writes the same value into *both* fields —
/// fixture `render_set_stencil_reference` asked for `0x11223344` and the
/// payload reads `44 33 22 11 44 33 22 11` — while
/// `setStencilFrontReferenceValue:backReferenceValue:` writes them separately
/// (`render_set_stencil_reference_front_back`, `0x11223344` and `0x55667788`).
/// So there is no "which selector was it" question on the wire, and a decoder
/// does not need one: the two-field form is the whole contract.
///
/// The fields are 32-bit, not the `NSUInteger` the neighbouring state records
/// use. Both selectors declare `I` in their type encodings, which is where that
/// came from rather than from the byte pattern — `0x11223344` would look the
/// same at either width in a record this size.
#[repr(C)]
#[derive(Debug)]
pub struct StencilReference {
    pub front: U32le,
    pub back: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for StencilReference {}

pub fn set_stencil_reference<'a>(op: &Op<'a>) -> Result<&'a StencilReference, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_STENCIL_REFERENCE);
    view::<StencilReference>(op.payload)
}

// --- 0x6c setDepthBias:slopeScale:clamp: -----------------------------------

pub const OPCODE_SET_DEPTH_BIAS: u32 = 0x6c;
pub const SET_DEPTH_BIAS_TOTAL_LEN: u32 = 20;

/// Depth bias, in the selector's own argument order. Fixture
/// `render_set_depth_bias` (0.25, 1.5, 2.25). Three 32-bit floats, as the
/// encoding `f16f20f24` declares.
#[repr(C)]
#[derive(Debug)]
pub struct DepthBias {
    pub bias: F32le,
    pub slope_scale: F32le,
    pub clamp: F32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DepthBias {}

pub fn set_depth_bias<'a>(op: &Op<'a>) -> Result<&'a DepthBias, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_DEPTH_BIAS);
    view::<DepthBias>(op.payload)
}

// --- 0x84 setVisibilityResultMode:offset: ----------------------------------

pub const OPCODE_SET_VISIBILITY_RESULT_MODE: u32 = 0x84;
pub const SET_VISIBILITY_RESULT_MODE_TOTAL_LEN: u32 = 24;

/// Visibility result mode and the offset it writes its counter to.
///
/// **The offset comes first**, which is the reverse of the selector's argument
/// order and of every other two-argument record in this module. Fixture
/// `render_set_visibility_result_mode` asked for
/// (`MTLVisibilityResultModeCounting` = 2, offset `0x1234`) and the payload
/// reads `34 12 …` then `02 …`. Both arguments are `Q`, so nothing about the
/// widths distinguishes them and only distinct values do.
#[repr(C)]
#[derive(Debug)]
pub struct VisibilityResult {
    pub offset: U64le,
    pub mode: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for VisibilityResult {}

pub fn set_visibility_result_mode<'a>(op: &Op<'a>) -> Result<&'a VisibilityResult, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VISIBILITY_RESULT_MODE);
    view::<VisibilityResult>(op.payload)
}

// --- 0x65 setBlendColorRed:green:blue:alpha: -------------------------------

pub const OPCODE_SET_BLEND_COLOR: u32 = 0x65;
pub const SET_BLEND_COLOR_TOTAL_LEN: u32 = 24;

/// Blend colour — four **32-bit** floats, though the Metal method takes them as
/// `float` and a render pass's clear colour is 64-bit. Fixture
/// `render_set_blend_color` (0.25, 0.5, 0.75, 1.0, all exact in binary).
#[repr(C)]
#[derive(Debug)]
pub struct BlendColor {
    pub red: F32le,
    pub green: F32le,
    pub blue: F32le,
    pub alpha: F32le,
}

// SAFETY: four align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BlendColor {}

pub fn set_blend_color<'a>(op: &Op<'a>) -> Result<&'a BlendColor, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_BLEND_COLOR);
    view::<BlendColor>(op.payload)
}

// --- Bind records ----------------------------------------------------------
//
// Every bind the render encoder emits has one shape: `[u32 first][u32 count]`
// followed by `count` entries. The singular selectors are the `count == 1`
// case of the plural ones and share their opcode, which is why the leading
// pair is not two constants — `setVertexTextures:withRange:` over range (2, 3)
// writes `first = 2, count = 3` and three refs (`render_set_vertex_textures_
// range`), while `setVertexTexture:atIndex:` at index 3 writes `first = 3,
// count = 1` and one (`render_set_vertex_texture`).
//
// That pairing is what settles the leading word as a count rather than a
// constant a singular record happens to carry. It also matches
// `reims_vgpu::runtime::decode::render`'s independently derived `BIND_FIRST`,
// `BIND_COUNT` and `BIND_ENTRIES`, field for field.

pub const OPCODE_SET_VERTEX_TEXTURE: u32 = 0x81;
pub const OPCODE_SET_FRAGMENT_TEXTURE: u32 = 0x72;
pub const OPCODE_SET_VERTEX_SAMPLER: u32 = 0x7f;
pub const OPCODE_SET_FRAGMENT_SAMPLER: u32 = 0x70;
pub const OPCODE_SET_VERTEX_BUFFER: u32 = 0x7d;
pub const OPCODE_SET_FRAGMENT_BUFFER: u32 = 0x6e;

/// The `[first][count]` head every bind record starts with.
#[repr(C)]
#[derive(Debug)]
pub struct BindHeader {
    /// The first table slot this record binds.
    pub first: U32le,
    /// How many consecutive slots follow. Guest-controlled, so a reader must
    /// bound it against the record's own length — [`ref_binds`] and
    /// [`buffer_binds`] do.
    pub count: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BindHeader {}

/// One texture or sampler slot: a serializer object ref and nothing else.
///
/// Fixtures `render_set_vertex_texture` / `render_set_fragment_texture` (the
/// stub texture's ref 4242) and `render_set_vertex_sampler` /
/// `render_set_fragment_sampler` (the stub sampler's 6363). The two families
/// differ only in opcode, which is also how the stage is decided — no wire
/// field names it.
#[repr(C)]
#[derive(Debug)]
pub struct RefBind {
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for RefBind {}

/// One buffer slot: a ref and the offset into it.
///
/// Fixtures `render_set_vertex_buffer` (ref 5151, offset `0x1234`, index 5) and
/// `render_set_fragment_buffers_range`, whose two entries carry `0x1111` and
/// `0x2222` and so show the entry stride is 12 rather than 16.
#[repr(C)]
#[derive(Debug)]
pub struct BufferBind {
    pub buffer_ref: U32le,
    pub offset: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BufferBind {}

#[inline]
pub fn is_ref_bind(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_VERTEX_TEXTURE
            | OPCODE_SET_FRAGMENT_TEXTURE
            | OPCODE_SET_VERTEX_SAMPLER
            | OPCODE_SET_FRAGMENT_SAMPLER
    )
}

#[inline]
pub fn is_buffer_bind(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_VERTEX_BUFFER | OPCODE_SET_FRAGMENT_BUFFER
    )
}

// --- 0x80 / 0x71 the sampler binds that carry LOD clamps -------------------

pub const OPCODE_SET_VERTEX_SAMPLER_LOD: u32 = 0x80;
pub const OPCODE_SET_FRAGMENT_SAMPLER_LOD: u32 = 0x71;

/// One sampler slot with its own level-of-detail clamps.
///
/// **A different opcode from the plain sampler bind, not a longer form of it.**
/// `setVertexSamplerState:atIndex:` writes `0x7f` and
/// `setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:` writes `0x80`; the
/// fragment pair is `0x70` and `0x71`. A decoder that knows only the plain
/// opcodes does not merely lose the clamps — it does not see the bind at all,
/// and the sampler stays unbound.
/// `reims_vgpu::runtime::decode::render` was in exactly that state.
///
/// **The clamps are per entry, not per record**, the same as the compute
/// encoder's [`crate::ops::compute::SamplerLodBind`]. With `count == 1` the pair
/// of floats after the ref could be either; `render_set_vertex_samplers_lod_range`
/// binds two slots with four distinct clamps (0.25/0.75 and 0.5/0.875) in a
/// 40-byte record, which is the eight-byte head plus two twelve-byte entries.
/// Fixtures `render_set_vertex_sampler_lod` and `render_set_fragment_sampler_lod`
/// for the singular form.
///
/// The `lodBias:` sibling of both selectors is **refused by the serializer** —
/// it fails an assertion rather than emitting anything — so there is no
/// four-float form of this entry. See the manifest.
#[repr(C)]
#[derive(Debug)]
pub struct SamplerLodBind {
    pub sampler_ref: U32le,
    pub lod_min_clamp: F32le,
    pub lod_max_clamp: F32le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for SamplerLodBind {}

#[inline]
pub fn is_sampler_lod_bind(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_VERTEX_SAMPLER_LOD | OPCODE_SET_FRAGMENT_SAMPLER_LOD
    )
}

/// Head and entries of a sampler bind carrying LOD clamps.
pub fn sampler_lod_binds<'a>(
    op: &Op<'a>,
) -> Result<(&'a BindHeader, &'a [SamplerLodBind]), WireError> {
    debug_assert!(is_sampler_lod_bind(op.opcode()));
    bind_entries::<SamplerLodBind>(op.payload)
}

/// Head and entries of a texture or sampler bind.
pub fn ref_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [RefBind]), WireError> {
    debug_assert!(is_ref_bind(op.opcode()));
    bind_entries::<RefBind>(op.payload)
}

/// Head and entries of a buffer bind.
pub fn buffer_binds<'a>(op: &Op<'a>) -> Result<(&'a BindHeader, &'a [BufferBind]), WireError> {
    debug_assert!(is_buffer_bind(op.opcode()));
    bind_entries::<BufferBind>(op.payload)
}

fn bind_entries<T: Wire>(payload: &[u8]) -> Result<(&BindHeader, &[T]), WireError> {
    let (head, rest) = crate::view::split::<BindHeader>(payload)?;
    let entries = crate::view::view_slice::<T>(rest, head.count.get() as usize)?;
    Ok((head, entries))
}

// --- 0x7e / 0x6f set{Vertex,Fragment}BufferOffset:atIndex: -----------------

pub const OPCODE_SET_VERTEX_BUFFER_OFFSET: u32 = 0x7e;
pub const OPCODE_SET_FRAGMENT_BUFFER_OFFSET: u32 = 0x6f;
pub const SET_BUFFER_OFFSET_TOTAL_LEN: u32 = 20;

/// Re-point an already-bound buffer slot without naming the buffer again.
///
/// Fixtures `render_set_vertex_buffer_offset` (index 5, offset `0x1234`) and
/// `render_set_fragment_buffer_offset` (index 6, `0x5678`). Note this is *not*
/// a [`BindHeader`] — the second word is the 64-bit offset, not a count, which
/// a reader coming from the bind records above would otherwise assume.
#[repr(C)]
#[derive(Debug)]
pub struct BufferOffset {
    pub index: U32le,
    pub offset: U64le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BufferOffset {}

#[inline]
pub fn is_buffer_offset(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_VERTEX_BUFFER_OFFSET | OPCODE_SET_FRAGMENT_BUFFER_OFFSET
    )
}

pub fn buffer_offset<'a>(op: &Op<'a>) -> Result<&'a BufferOffset, WireError> {
    debug_assert!(is_buffer_offset(op.opcode()));
    view::<BufferOffset>(op.payload)
}

// --- 0xa5 / 0xa6 the vertex buffer binds that carry an attribute stride -----

pub const OPCODE_SET_VERTEX_BUFFER_STRIDE: u32 = 0xa5;
pub const OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE: u32 = 0xa6;
pub const SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN: u32 = 28;

/// One vertex buffer slot bound with a dynamic attribute stride.
///
/// **A different opcode from the plain buffer bind, not a longer form of it**,
/// exactly as [`SamplerLodBind`] is to [`RefBind`].
/// `setVertexBuffer:offset:atIndex:` writes `0x7d` and
/// `setVertexBuffer:offset:attributeStride:atIndex:` writes `0xa5`. A decoder
/// that knows only `0x7d` does not merely lose the stride — it does not see the
/// bind at all, and the vertex buffer stays unbound.
///
/// **The stride is per entry**, which only the plural case can show:
/// `render_set_vertex_buffers_range_attribute_stride` binds slots 9 and 10 with
/// offsets `0x3333`/`0x4444` and strides `0x5555`/`0x6666` in a 56-byte record —
/// the eight-byte [`BindHeader`] plus two twenty-byte entries. With `count == 1`
/// the trailing `u64` could have been either.
///
/// `setVertexBytes:length:attributeStride:atIndex:` writes this same opcode:
/// the serializer stages the bytes and records the staging buffer's ref and
/// offset, so there is no inline-data form, which is what the non-stride
/// sibling does too (`render_set_vertex_bytes_attribute_stride` carries a
/// staging ref rather than 5151).
///
/// There is **no fragment sibling**. Metal puts `attributeStride` only on the
/// vertex-stage selectors, and the inventory has no
/// `setFragmentBuffer:offset:attributeStride:atIndex:` to drive — so the
/// absence here is Apple's API surface, not an uncaptured case.
///
/// # This record is gated on a capability, and that is why it looked absent
///
/// The serializer answers `-supportsDynamicAttributeStride` **false** by
/// default, and with it false all four selectors run and write nothing. The
/// first capture of them landed on `silent`, which would have become an
/// `EMITS_NO_OPERATION` manifest row asserting Apple emits nothing here — a
/// false claim about Apple. Driven through `withCapability` with the flag
/// forced on, all four emit. See the crate `AGENTS.md`.
#[repr(C)]
#[derive(Debug)]
pub struct BufferStrideBind {
    pub buffer_ref: U32le,
    pub offset: U64le,
    /// Bytes between consecutive vertex attributes in this buffer. Observed
    /// `0x3456` singular, `0x5555`/`0x6666` across the two plural entries.
    pub attribute_stride: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BufferStrideBind {}

#[inline]
pub fn is_buffer_stride_bind(opcode: u32) -> bool {
    opcode == OPCODE_SET_VERTEX_BUFFER_STRIDE
}

/// Head and entries of a vertex buffer bind carrying attribute strides.
pub fn buffer_stride_binds<'a>(
    op: &Op<'a>,
) -> Result<(&'a BindHeader, &'a [BufferStrideBind]), WireError> {
    debug_assert!(is_buffer_stride_bind(op.opcode()));
    bind_entries::<BufferStrideBind>(op.payload)
}

/// Re-point a bound vertex slot and restate its attribute stride.
///
/// [`BufferOffset`] with the stride appended, and the same trap: the second
/// word is the 64-bit offset rather than a count. Fixture
/// `render_set_vertex_buffer_offset_attribute_stride` (index 8, offset `0x4567`,
/// stride `0x5678` — three distinct values, so no pair can be crossed unseen).
/// Also capability-gated; see [`BufferStrideBind`].
#[repr(C)]
#[derive(Debug)]
pub struct BufferOffsetStride {
    pub index: U32le,
    pub offset: U64le,
    pub attribute_stride: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for BufferOffsetStride {}

pub fn buffer_offset_stride<'a>(op: &Op<'a>) -> Result<&'a BufferOffsetStride, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VERTEX_BUFFER_OFFSET_STRIDE);
    view::<BufferOffsetStride>(op.payload)
}

// --- 0x99 / 0x9a vertex amplification --------------------------------------

pub const OPCODE_SET_VERTEX_AMPLIFICATION_MODE: u32 = 0x99;
pub const OPCODE_SET_VERTEX_AMPLIFICATION_COUNT: u32 = 0x9a;
pub const SET_VERTEX_AMPLIFICATION_MODE_TOTAL_LEN: u32 = 16;

/// `setVertexAmplificationMode:value:`.
///
/// **Both arguments are `Q` in the type encoding and 32 bits on the wire.** That
/// is the serializer narrowing, which the encoding cannot tell you and only a
/// capture can — the record is 16 bytes, and two `u64` would need 24. Fixture
/// `render_set_vertex_amplification_mode` (mode `0x5555`, value `0x6666`).
///
/// Gated on `-supportsVertexAmplification`, which defaults off; driven through
/// `withCapability`. See [`BufferStrideBind`] and the crate `AGENTS.md`.
#[repr(C)]
#[derive(Debug)]
pub struct VertexAmplificationMode {
    pub mode: U32le,
    pub value: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for VertexAmplificationMode {}

pub fn vertex_amplification_mode<'a>(
    op: &Op<'a>,
) -> Result<&'a VertexAmplificationMode, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VERTEX_AMPLIFICATION_MODE);
    view::<VertexAmplificationMode>(op.payload)
}

/// The count leading `setVertexAmplificationCount:viewMappings:`.
///
/// **Four bytes, not the eight-byte [`BindHeader`]** — there is no `first` here,
/// the mappings start immediately after the count, and reading this record with
/// a `BindHeader` takes the first mapping's viewport offset as the count.
#[repr(C)]
#[derive(Debug)]
pub struct VertexAmplificationHeader {
    pub count: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for VertexAmplificationHeader {}

/// One `MTLVertexAmplificationViewMapping`.
///
/// Two `uint32_t`, which the type encoding gives for free as `r^{?=II}`. The
/// mappings do reach the wire and they are per entry: fixture
/// `render_set_vertex_amplification_count` passes two with four distinct offsets
/// (`0x1111`/`0x2222` and `0x3333`/`0x4444`) in a 28-byte record — the header,
/// the count, and two eight-byte entries.
#[repr(C)]
#[derive(Debug)]
pub struct ViewMapping {
    pub viewport_array_index_offset: U32le,
    pub render_target_array_index_offset: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ViewMapping {}

pub fn vertex_amplification_count<'a>(
    op: &Op<'a>,
) -> Result<(&'a VertexAmplificationHeader, &'a [ViewMapping]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VERTEX_AMPLIFICATION_COUNT);
    let (head, rest) = crate::view::split::<VertexAmplificationHeader>(op.payload)?;
    let entries = crate::view::view_slice::<ViewMapping>(rest, head.count.get() as usize)?;
    Ok((head, entries))
}

// --- 0x74 setRenderPipelineState: / 0x68 setDepthStencilState: -------------

pub const OPCODE_SET_RENDER_PIPELINE_STATE: u32 = 0x74;
pub const OPCODE_SET_DEPTH_STENCIL_STATE: u32 = 0x68;
/// A single object ref, in a 12-byte record.
pub const SET_STATE_TOTAL_LEN: u32 = 12;

/// A record that is one serializer object ref and nothing else. Fixtures
/// `render_set_render_pipeline_state` (6161) and
/// `render_set_depth_stencil_state` (6262) — distinct stub refs, so a record
/// that picked up the wrong object would be obvious rather than off by one.
#[repr(C)]
#[derive(Debug)]
pub struct StateRef {
    pub object_ref: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for StateRef {}

#[inline]
pub fn is_state_ref(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_RENDER_PIPELINE_STATE | OPCODE_SET_DEPTH_STENCIL_STATE
    )
}

pub fn state_ref<'a>(op: &Op<'a>) -> Result<&'a StateRef, WireError> {
    debug_assert!(is_state_ref(op.opcode()));
    view::<StateRef>(op.payload)
}

// --- 0x18 updateFence:afterStages: / 0x19 waitForFence:beforeStages: -------

pub const OPCODE_UPDATE_FENCE: u32 = 0x18;
pub const OPCODE_WAIT_FOR_FENCE: u32 = 0x19;
pub const FENCE_TOTAL_LEN: u32 = 16;

/// A fence and the render stages it is ordered against. Fixtures
/// `render_update_fence` (ref 6464, stages 2 = Fragment) and
/// `render_wait_for_fence` (stages 1 = Vertex). Both selectors write the same
/// record; which side of the fence it is comes from the opcode.
///
/// `stages` is 32 bits here where the selector declares `Q`. Contrast
/// [`UseResource`], whose two `Q` arguments are narrowed to 16 bits each —
/// this protocol narrows per record, not per type.
#[repr(C)]
#[derive(Debug)]
pub struct Fence {
    pub fence_ref: U32le,
    pub stages: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for Fence {}

#[inline]
pub fn is_fence(opcode: u32) -> bool {
    matches!(opcode, OPCODE_UPDATE_FENCE | OPCODE_WAIT_FOR_FENCE)
}

pub fn fence<'a>(op: &Op<'a>) -> Result<&'a Fence, WireError> {
    debug_assert!(is_fence(op.opcode()));
    view::<Fence>(op.payload)
}

// --- 0x89 useResource:usage:stages: ----------------------------------------

pub const OPCODE_USE_RESOURCE: u32 = 0x89;

/// Residency declaration for one or more resources.
///
/// `count` leads, as in the bind records, and the refs trail — but there is no
/// `first`, because this names no table slot. Fixtures `render_use_resource`
/// (count 1) and `render_use_resources_count` (count 2, two distinct refs),
/// the pair that shows the trailing array is really an array.
///
/// **`usage` and `stages` are 16 bits each**, sharing one 32-bit word, though
/// the selector declares both `Q`. Neither `MTLResourceUsage` nor
/// `MTLRenderStages` has a value above `0xffff`, so no case can separate two
/// 16-bit fields from one packed 32-bit word by magnitude. What settles it is
/// position: the resource ref sits at `+8`, so both arguments must fit between
/// `+4` and there, and each was seen to move on its own — `(usage 1, stages 2)`
/// reads `01 00 02 00` and `(usage 2, stages 4)` reads `02 00 04 00`.
#[repr(C)]
#[derive(Debug)]
pub struct UseResource {
    pub count: U32le,
    pub usage: U16le,
    pub stages: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for UseResource {}

/// Head and the resource refs that follow it.
pub fn use_resource<'a>(op: &Op<'a>) -> Result<(&'a UseResource, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_USE_RESOURCE);
    let (head, rest) = crate::view::split::<UseResource>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

// --- 0x1b useHeap:stages: --------------------------------------------------

pub const OPCODE_USE_HEAP: u32 = 0x1b;

/// Residency declaration for a heap.
///
/// **Not the same shape as [`UseResource`], and not a neighbouring opcode.**
/// `useResource:usage:stages:` is `0x89` with `usage` and `stages` sharing the
/// word at `+4`; this is `0x1b` with no `usage` at all, so `stages` sits alone
/// at `+4` as a `u16` and the refs begin at `+6` — an odd-of-four offset that
/// only an align-1 view can take. Fixture `render_use_heap` (the stub heap's
/// ref 6565, stages 2).
///
/// # The record is two bytes longer than it is written, and `+6` is still right
///
/// This is the one record here whose *length* disagrees with its layout. The
/// serializer sizes it as `count * 4 + 8` — the shape its `usage`-bearing
/// sibling has — and then writes `count` at `+0`, `stages` at `+4` and the refs
/// from `+6`, leaving the last two bytes untouched. Deriving the head from the
/// record length therefore yields 8 and starts the refs two bytes late, reading
/// every heap ref straddling two entries.
///
/// So a length is an upper bound on a head, never a measurement of one, and this
/// is the record that shows it. Do not "correct" `+6` to `+8` from a size.
///
/// Decode consumes these opcodes through [`use_heap`] / [`use_resource`]
/// (`0x1b` / `0x89`).
///
/// **They are half the family.** `0x1b` and `0x89` are the `stages:`-qualified
/// forms, which the render encoder declares itself. The unqualified
/// `useHeaps:count:` / `useResources:count:usage:` are declared on the encoder
/// base class and inherited by every encoder including this one; they emit
/// `0x86` / `0x87`, with a four-byte and an eight-byte head respectively, and
/// this module has no view for either. `runtime::decode::compute` does decode
/// them; see the inheritance caveat in [`crate::manifest`] for why the coverage
/// instrument cannot see the selectors behind them, and why a reader checking
/// only the render encoder's own method list concludes they have none.
#[repr(C)]
#[derive(Debug)]
pub struct UseHeap {
    pub count: U32le,
    pub stages: U16le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for UseHeap {}

/// Head and the heap refs that follow it.
pub fn use_heap<'a>(op: &Op<'a>) -> Result<(&'a UseHeap, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_USE_HEAP);
    let (head, rest) = crate::view::split::<UseHeap>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

// --- 0x86 / 0x87 the residency forms that take no `stages:` -----------------

pub const OPCODE_USE_HEAPS_NO_STAGES: u32 = 0x86;
pub const OPCODE_USE_RESOURCES_NO_STAGES: u32 = 0x87;

/// `useHeaps:count:` — the residency form with no `stages:` argument.
///
/// # Why there are four residency opcodes and not two
///
/// `useHeap:stages:` and `useResource:usage:stages:` are declared on the render
/// encoder itself and write [`OPCODE_USE_HEAP`] / [`OPCODE_USE_RESOURCE`]. The
/// unqualified `useHeaps:count:` and `useResources:count:usage:` are declared
/// one class up, on the encoder base class every encoder derives from, and write
/// these two. A render encoder answers all four, so a decoder that knows only
/// the qualified pair sees a guest's `useResources:count:usage:` as an
/// unimplemented opcode — the same shape as the `0x7d`/`0xa5` split that
/// [`BufferStrideBind`] warns about, one level up.
///
/// They are invisible to [`crate::manifest`], which is built per class from each
/// class's own method list; see the inheritance caveat there.
///
/// # This head is a size, not a field map
///
/// The record is a four-byte head followed by `count` four-byte refs, and
/// `count` is the only field this device reads — residency is answered by doing
/// nothing, so the head exists here to make the refs start in the right place
/// and to let `count` bound the record. No fixture on a non-Apple checkout pins
/// it; what stands behind the size is the emitted record length, and a wrong one
/// fails the `count == refs.len()` check in `runtime::decode::render` rather
/// than silently reading a ref short.
#[repr(C)]
#[derive(Debug)]
pub struct UseHeapsNoStages {
    pub count: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for UseHeapsNoStages {}

/// `useResources:count:usage:` — see [`UseHeapsNoStages`].
///
/// Eight-byte head against that record's four, the extra word being `usage`.
/// Nothing reads it: the `stages:`-qualified [`UseResource`] carries `usage` as
/// a `u16` beside `stages`, and whether this form widens it to `u32` or leaves
/// two bytes unwritten is not settled here, because no case can tell them apart
/// — `MTLResourceUsage` has no value above `0xffff`, which is the same argument
/// that sized the qualified form's field. It is declared at the width that makes
/// the head the size the record actually has.
#[repr(C)]
#[derive(Debug)]
pub struct UseResourcesNoStages {
    pub count: U32le,
    pub usage: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for UseResourcesNoStages {}

/// Head and the heap refs that follow it.
pub fn use_heaps_no_stages<'a>(
    op: &Op<'a>,
) -> Result<(&'a UseHeapsNoStages, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_USE_HEAPS_NO_STAGES);
    let (head, rest) = crate::view::split::<UseHeapsNoStages>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

/// Head and the resource refs that follow it.
pub fn use_resources_no_stages<'a>(
    op: &Op<'a>,
) -> Result<(&'a UseResourcesNoStages, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_USE_RESOURCES_NO_STAGES);
    let (head, rest) = crate::view::split::<UseResourcesNoStages>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

// --- 0x10 / 0x11 indirect draws --------------------------------------------

pub const OPCODE_DRAW_INDIRECT: u32 = 0x10;
pub const DRAW_INDIRECT_TOTAL_LEN: u32 = 24;

/// A draw whose counts come from a buffer instead of the record.
///
/// The record reverses the selector: the **offset comes first**, then the
/// buffer, then the primitive type — the same inversion
/// [`SetVisibilityResultMode`] has. Fixture `render_draw_primitives_indirect`
/// (offset `0x1111`, buffer 5151, `MTLPrimitiveTypeTriangle`).
///
/// `primitive_type` is 16 bits here where the direct draws give it 32. Two
/// bytes past it are never written; the record is 24 and this body is 14 of the
/// 16 payload bytes.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndirect {
    pub indirect_buffer_offset: U64le,
    pub indirect_buffer_ref: U32le,
    pub primitive_type: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndirect {}

pub fn draw_indirect<'a>(op: &Op<'a>) -> Result<&'a DrawIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDIRECT);
    view::<DrawIndirect>(op.payload)
}

pub const OPCODE_DRAW_INDEXED_INDIRECT: u32 = 0x11;
pub const DRAW_INDEXED_INDIRECT_TOTAL_LEN: u32 = 36;

/// An indexed draw whose counts come from a buffer.
///
/// Both buffers lead as `u32` refs and both offsets trail as `u64`, which is
/// the blit family's shape rather than [`DrawIndirect`]'s. Fixture
/// `render_draw_indexed_indirect`: `MTLPrimitiveTypeTriangleStrip`,
/// `MTLIndexTypeUInt32`, index buffer 5151 at `0x1111`, indirect buffer 5252 at
/// `0x2222` — two distinct refs and two distinct offsets, so a record that
/// crossed them could not read back correct.
///
/// `index_type` is its own 16-bit field beside `primitive_type`. That is the
/// bug `reims_vgpu::runtime::decode::render` had in the *compact indexed* draw
/// and had fixed: reading a `u32` at `+0` absorbs the field at `+2`.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedIndirect {
    pub primitive_type: U16le,
    pub index_type: U16le,
    pub index_buffer_ref: U32le,
    pub indirect_buffer_ref: U32le,
    pub index_buffer_offset: U64le,
    pub indirect_buffer_offset: U64le,
}

// SAFETY: six align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedIndirect {}

pub fn draw_indexed_indirect<'a>(op: &Op<'a>) -> Result<&'a DrawIndexedIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_INDIRECT);
    view::<DrawIndexedIndirect>(op.payload)
}

// --- 0x14 / 0x15 indirect command buffer execution -------------------------

pub const OPCODE_EXECUTE_COMMANDS_INDIRECT: u32 = 0x14;
pub const EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN: u32 = 24;

/// Execute an indirect command buffer, with the command range itself coming
/// from a second buffer.
///
/// Fixture `render_execute_commands_indirect` (ICB 7171, indirect buffer 5151
/// at `0x1111`). The ICB's ref comes from `indirectCommandBufferRef`, not from
/// the accessor a plain buffer answers.
#[repr(C)]
#[derive(Debug)]
pub struct ExecuteCommandsIndirect {
    pub icb_ref: U32le,
    pub indirect_buffer_ref: U32le,
    pub indirect_buffer_offset: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ExecuteCommandsIndirect {}

pub fn execute_commands_indirect<'a>(
    op: &Op<'a>,
) -> Result<&'a ExecuteCommandsIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_EXECUTE_COMMANDS_INDIRECT);
    view::<ExecuteCommandsIndirect>(op.payload)
}

pub const OPCODE_EXECUTE_COMMANDS_RANGE: u32 = 0x15;
pub const EXECUTE_COMMANDS_RANGE_TOTAL_LEN: u32 = 28;

/// Execute a literal range of an indirect command buffer.
///
/// Byte for byte the blit encoder's [`crate::ops::blit::IcbRange`], at a
/// different opcode in a different opcode space. Fixture
/// `render_execute_commands_range` (ICB 7171, range `0x1100`/`0x2200`).
#[repr(C)]
#[derive(Debug)]
pub struct ExecuteCommandsRange {
    pub icb_ref: U32le,
    pub range_location: U64le,
    pub range_length: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ExecuteCommandsRange {}

pub fn execute_commands_range<'a>(op: &Op<'a>) -> Result<&'a ExecuteCommandsRange, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_EXECUTE_COMMANDS_RANGE);
    view::<ExecuteCommandsRange>(op.payload)
}

// --- 0x16 / 0x17 / 0x85 barriers -------------------------------------------

pub const OPCODE_MEMORY_BARRIER_RESOURCES: u32 = 0x16;

/// A barrier over a named list of resources.
///
/// Head plus a trailing ref array, like [`UseResource`] — but the refs start at
/// `+8` here rather than `+6`, because both stage masks are 16 bits and neither
/// is folded away. Fixture `render_memory_barrier_resources` (count 2, after
/// `MTLRenderStageVertex`, before `MTLRenderStageFragment`, refs 5151 and 4343:
/// a buffer and a texture, so the array is shown to hold resources of any kind
/// rather than one kind).
#[repr(C)]
#[derive(Debug)]
pub struct MemoryBarrierResources {
    pub count: U32le,
    pub after_stages: U16le,
    pub before_stages: U16le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for MemoryBarrierResources {}

/// Head and the resource refs that follow it.
pub fn memory_barrier_resources<'a>(
    op: &Op<'a>,
) -> Result<(&'a MemoryBarrierResources, &'a [RefBind]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_MEMORY_BARRIER_RESOURCES);
    let (head, rest) = crate::view::split::<MemoryBarrierResources>(op.payload)?;
    let refs = crate::view::view_slice::<RefBind>(rest, head.count.get() as usize)?;
    Ok((head, refs))
}

pub const OPCODE_MEMORY_BARRIER_SCOPE: u32 = 0x17;
pub const MEMORY_BARRIER_SCOPE_TOTAL_LEN: u32 = 12;

/// A barrier over a scope rather than a list, in four bytes.
///
/// Every field is one byte, though all three are declared `Q` — this record is
/// the protocol's narrowing taken furthest. Fixtures
/// `render_memory_barrier_scope` (scope 4, after 1, before 2 → `04 00 01 02`)
/// and `render_memory_barrier_scope_alt` (scope 1, after 4, before 8 → `01 00
/// 04 08`), which is what separates the three from each other.
#[repr(C)]
#[derive(Debug)]
pub struct MemoryBarrierScope {
    /// `MTLBarrierScope`, one byte.
    pub scope: u8,
    /// Written, `0` under both scopes captured, and not identified.
    ///
    /// **Tried:** `MTLBarrierScopeRenderTargets` (4) and
    /// `MTLBarrierScopeBuffers` (1) both moved byte `+0` and left this at 0.
    /// It cannot be shown to be separate from `scope`, because
    /// `MTLBarrierScope` defines no value above 4 and a two-byte `scope` is
    /// indistinguishable from a one-byte one beside a zero byte.
    ///
    /// **What would settle it:** a scope with bit 8 set. Metal defines none, so
    /// on this API the question may have no answer — which is itself the
    /// finding, and is why the field is named rather than folded into `scope`.
    ///
    /// **What does not settle it, and looks as though it should:** the encoder
    /// writes bytes `+0` and `+1` with a single 16-bit store, and `+2` and `+3`
    /// with a byte store each. That is why this byte is reliably `0` here while
    /// the compute sibling — which stores only two bytes of its four — leaves
    /// ring residue in the rest, and it is the asymmetry
    /// [`crate::ops::compute::MemoryBarrierScope`] describes. It is *not*
    /// evidence of a `u16` field: a one-byte value zero-extended into a
    /// two-byte store is the same instruction, and "declared `Q`, narrowed on
    /// the wire" is what this whole record does. Do not fold the two on the
    /// strength of the store width.
    ///
    /// Keeping it separate also keeps the alarm. The fixture asserts this byte
    /// is still `0`; folded into `scope`, an Apple build that started using it
    /// would be absorbed into a larger scope value with nothing to report.
    pub unidentified_u8: u8,
    /// `MTLRenderStages` the barrier waits on.
    pub after_stages: u8,
    /// `MTLRenderStages` the barrier blocks.
    pub before_stages: u8,
}

// SAFETY: four `u8`s; align 1 and every byte pattern valid.
unsafe impl Wire for MemoryBarrierScope {}

pub fn memory_barrier_scope<'a>(op: &Op<'a>) -> Result<&'a MemoryBarrierScope, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_MEMORY_BARRIER_SCOPE);
    view::<MemoryBarrierScope>(op.payload)
}

pub const OPCODE_TEXTURE_BARRIER: u32 = 0x85;
pub const TEXTURE_BARRIER_TOTAL_LEN: u32 = 8;

/// `textureBarrier` is the header and nothing else — an 8-byte record with an
/// empty payload (fixture `render_texture_barrier`).
///
/// There is no view because there is nothing to view; the opcode is the whole
/// command.
///
/// It sits one below `0x86`/`0x87`, which this serializer assigns to no render
/// selector. `reims_vgpu::runtime::decode::render` used to call those its
/// residency pair and no longer does — it reads `0x1b` and `0x89`, see
/// [`UseHeap`]. The observation that outlives the correction is the one worth
/// keeping: **whatever `0x86` and `0x87` are, they are in the barrier
/// neighbourhood rather than the residency one**, which is why reading them as
/// residency put the device four opcodes away from anything Apple emits.
pub fn texture_barrier_has_no_payload(op: &Op<'_>) -> bool {
    op.opcode() == OPCODE_TEXTURE_BARRIER && op.payload.is_empty()
}

// --- 0x76 / 0x83 the plural viewport and scissor forms ----------------------

pub const OPCODE_SET_SCISSOR_RECTS: u32 = 0x76;

/// Head of the plural scissor record.
///
/// The element is [`ScissorRect`] itself — the singular record's whole payload,
/// unchanged. So the plural form really is "the singular one with a count in
/// front", which is worth stating because it is not true of the bind records
/// (where the singular is the plural at `count == 1` and shares its opcode) and
/// not true of `useHeap:` either.
///
/// `count` is read as eight bytes. It cannot be shown to be eight rather than
/// four followed by four written zeros — on a little-endian wire those decode
/// identically for any count below `2^32`, and no legal count reaches that. The
/// wider read is chosen because it *refuses* a record whose high word is
/// non-zero, where the narrower one would ignore it silently. The sibling
/// [`SetViewports`] puts its array at `+4`, so this is not a family rule.
///
/// Fixture `render_set_scissor_rects`: two rects, eight distinct values
/// `0x11`–`0x88`, so no pair of the eight fields can be confused.
#[repr(C)]
#[derive(Debug)]
pub struct SetScissorRects {
    pub count: U64le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for SetScissorRects {}

/// Head and the rectangles that follow it.
pub fn set_scissor_rects<'a>(
    op: &Op<'a>,
) -> Result<(&'a SetScissorRects, &'a [ScissorRect]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_SCISSOR_RECTS);
    let (head, rest) = crate::view::split::<SetScissorRects>(op.payload)?;
    let rects = crate::view::view_slice::<ScissorRect>(rest, head.count.get() as usize)?;
    Ok((head, rects))
}

pub const OPCODE_SET_VIEWPORTS: u32 = 0x83;

/// Head of the plural viewport record. `count` is four bytes here — see
/// [`SetScissorRects`], whose is eight.
///
/// The element is [`Viewport`], the singular record's whole payload. Because
/// `count` is four bytes, every one of those six `f64` lands at an offset that
/// is `4 mod 8`, and nothing but an align-1 view can take a reference to one.
///
/// Fixture `render_set_viewports`: two viewports with twelve distinct values,
/// including two pairs of depth bounds inside `[0, 1]` that differ from each
/// other, so a record that copied one viewport twice is visible.
#[repr(C)]
#[derive(Debug)]
pub struct SetViewports {
    pub count: U32le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for SetViewports {}

/// Head and the viewports that follow it.
pub fn set_viewports<'a>(op: &Op<'a>) -> Result<(&'a SetViewports, &'a [Viewport]), WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_VIEWPORTS);
    let (head, rest) = crate::view::split::<SetViewports>(op.payload)?;
    let ports = crate::view::view_slice::<Viewport>(rest, head.count.get() as usize)?;
    Ok((head, ports))
}

// --- Store actions ---------------------------------------------------------

pub const OPCODE_SET_COLOR_STORE_ACTION: u32 = 0x66;
pub const SET_COLOR_STORE_ACTION_TOTAL_LEN: u32 = 16;
pub const OPCODE_SET_DEPTH_STORE_ACTION: u32 = 0x69;
pub const OPCODE_SET_STENCIL_STORE_ACTION: u32 = 0x78;

/// A colour attachment's store action and which slot it applies to, in the
/// selector's own argument order. Fixture `render_set_color_store_action`
/// (`MTLStoreActionMultisampleResolve` = 2 at index 3) — deliberately unequal,
/// because a first capture used 2 for both and could not tell them apart.
#[repr(C)]
#[derive(Debug)]
pub struct ColorStoreAction {
    pub store_action: U32le,
    pub index: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ColorStoreAction {}

pub fn set_color_store_action<'a>(op: &Op<'a>) -> Result<&'a ColorStoreAction, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_COLOR_STORE_ACTION);
    view::<ColorStoreAction>(op.payload)
}

// --- 0x0c / 0x0d / 0x0f / 0x12 / 0x13 the patch draws ----------------------
//
// Tessellation. This protocol carries it: all four `drawPatches:` /
// `drawIndexedPatches:` selectors emit, and so do both pieces of state a
// tessellated draw needs ([`OPCODE_SET_TESSELLATION_FACTOR_BUFFER`] and
// [`OPCODE_SET_TESSELLATION_FACTOR_SCALE`]).
//
// **The two wide forms share one opcode and are told apart by length alone.**
// `0x0c` at 56 bytes is the plain patch draw; `0x0c` at 68 is the indexed one.
// Every other draw pair in this family gives the wide form its own opcode
// (`0x01`/`0x00`, `0x07`/`0x06`, …), so this is the exception and `0x0e` is
// simply unused. A decoder that dispatched on `0x0c` alone and then read the
// plain body would take the indexed record's `control_point_index_buffer_ref`
// as its `patch_index_buffer_offset`.

pub const OPCODE_DRAW_PATCHES: u32 = 0x0d;
pub const OPCODE_DRAW_INDEXED_PATCHES: u32 = 0x0f;
/// Both wide forms. See the note above: the length is the discriminator.
pub const OPCODE_DRAW_PATCHES_WIDE: u32 = 0x0c;
pub const OPCODE_DRAW_PATCHES_INDIRECT: u32 = 0x12;
pub const OPCODE_DRAW_INDEXED_PATCHES_INDIRECT: u32 = 0x13;

pub const DRAW_PATCHES_TOTAL_LEN: u32 = 24;
pub const DRAW_PATCHES_WIDE_TOTAL_LEN: u32 = 56;
pub const DRAW_INDEXED_PATCHES_TOTAL_LEN: u32 = 32;
pub const DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN: u32 = 68;
pub const DRAW_PATCHES_INDIRECT_TOTAL_LEN: u32 = 36;
pub const DRAW_INDEXED_PATCHES_INDIRECT_TOTAL_LEN: u32 = 48;

/// `drawPatches:patchStart:patchCount:…`, compact.
///
/// **`control_points` trails**, reversing the selector — it is the *first*
/// argument and the last field, on all six patch records. Every other field is
/// a `Q` narrowed to 16 bits, which is what the wide form exists for.
/// Fixture `render_draw_patches`: seven distinct values `0x11`–`0x55` plus ref
/// 5151 and three control points, so no two fields can be confused.
#[repr(C)]
#[derive(Debug)]
pub struct DrawPatches {
    pub patch_start: U16le,
    pub patch_count: U16le,
    pub patch_index_buffer_ref: U32le,
    pub patch_index_buffer_offset: U16le,
    pub instance_count: U16le,
    pub base_instance: U16le,
    pub control_points: U16le,
}

// SAFETY: seven align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawPatches {}

/// The same draw with every count at full width. Fixture
/// `render_draw_patches_over_16bit` (`patchCount` = `0x10000`), which is what
/// showed the wide form exists at all.
#[repr(C)]
#[derive(Debug)]
pub struct DrawPatchesWide {
    pub patch_start: U64le,
    pub patch_count: U64le,
    pub patch_index_buffer_ref: U32le,
    pub patch_index_buffer_offset: U64le,
    pub instance_count: U64le,
    pub base_instance: U64le,
    pub control_points: U16le,
}

// SAFETY: seven align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawPatchesWide {}

/// `drawIndexedPatches:patchStart:…`, compact.
///
/// Note `control_point_index_buffer_ref` sits at payload `+10` — a `u32` at an
/// odd multiple of two, which is why every struct in this crate must be align-1
/// rather than merely `#[repr(C)]`. Fixture `render_draw_indexed_patches`,
/// whose two buffer refs are 5151 and 5252 so a decoder that crossed them
/// cannot read back correct.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedPatches {
    pub patch_start: U16le,
    pub patch_count: U16le,
    pub patch_index_buffer_ref: U32le,
    pub patch_index_buffer_offset: U16le,
    pub control_point_index_buffer_ref: U32le,
    pub control_point_index_buffer_offset: U16le,
    pub instance_count: U16le,
    pub base_instance: U16le,
    pub control_points: U16le,
}

// SAFETY: nine align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedPatches {}

/// Fixture `render_draw_indexed_patches_over_16bit`. Shares opcode `0x0c` with
/// [`DrawPatchesWide`] and is twelve bytes longer.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedPatchesWide {
    pub patch_start: U64le,
    pub patch_count: U64le,
    pub patch_index_buffer_ref: U32le,
    pub patch_index_buffer_offset: U64le,
    pub control_point_index_buffer_ref: U32le,
    pub control_point_index_buffer_offset: U64le,
    pub instance_count: U64le,
    pub base_instance: U64le,
    pub control_points: U16le,
}

// SAFETY: nine align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedPatchesWide {}

/// `drawPatches:patchIndexBuffer:…indirectBuffer:…`.
///
/// **Both refs lead, then both offsets** — the blit family's shape, and *not*
/// the compact patch draws' interleaving of ref and offset. The offsets are
/// `u64` here rather than narrowed, so there is no wide form of this record and
/// none was found. Fixture `render_draw_patches_indirect` (refs 5151 and 5353).
#[repr(C)]
#[derive(Debug)]
pub struct DrawPatchesIndirect {
    pub patch_index_buffer_ref: U32le,
    pub indirect_buffer_ref: U32le,
    pub patch_index_buffer_offset: U64le,
    pub indirect_buffer_offset: U64le,
    pub control_points: U16le,
}

// SAFETY: five align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawPatchesIndirect {}

/// Three refs, then three offsets. Fixture
/// `render_draw_indexed_patches_indirect` carries 5151, 5252 and 5353 so all
/// three slots are distinguishable — two refs could not have shown the order.
#[repr(C)]
#[derive(Debug)]
pub struct DrawIndexedPatchesIndirect {
    pub patch_index_buffer_ref: U32le,
    pub control_point_index_buffer_ref: U32le,
    pub indirect_buffer_ref: U32le,
    pub patch_index_buffer_offset: U64le,
    pub control_point_index_buffer_offset: U64le,
    pub indirect_buffer_offset: U64le,
    pub control_points: U16le,
}

// SAFETY: seven align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for DrawIndexedPatchesIndirect {}

#[inline]
pub fn is_patch_draw(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_DRAW_PATCHES
            | OPCODE_DRAW_PATCHES_WIDE
            | OPCODE_DRAW_INDEXED_PATCHES
            | OPCODE_DRAW_PATCHES_INDIRECT
            | OPCODE_DRAW_INDEXED_PATCHES_INDIRECT
    )
}

/// Which of the two records `0x0c` is, decided by the operation's own length.
///
/// Returns `None` for any other length, because there is no third reading: a
/// `0x0c` that is neither 56 nor 68 bytes is a record this crate does not know,
/// and guessing between the two would take one record's ref as the other's
/// offset. The caller must refuse it rather than pick.
#[inline]
pub fn patch_draw_wide_is_indexed(op: &Op<'_>) -> Option<bool> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_PATCHES_WIDE);
    match op.length() {
        DRAW_PATCHES_WIDE_TOTAL_LEN => Some(false),
        DRAW_INDEXED_PATCHES_WIDE_TOTAL_LEN => Some(true),
        _ => None,
    }
}

pub fn draw_patches<'a>(op: &Op<'a>) -> Result<&'a DrawPatches, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_PATCHES);
    view::<DrawPatches>(op.payload)
}

pub fn draw_patches_wide<'a>(op: &Op<'a>) -> Result<&'a DrawPatchesWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_PATCHES_WIDE);
    view::<DrawPatchesWide>(op.payload)
}

pub fn draw_indexed_patches<'a>(op: &Op<'a>) -> Result<&'a DrawIndexedPatches, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_PATCHES);
    view::<DrawIndexedPatches>(op.payload)
}

pub fn draw_indexed_patches_wide<'a>(op: &Op<'a>) -> Result<&'a DrawIndexedPatchesWide, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_PATCHES_WIDE);
    view::<DrawIndexedPatchesWide>(op.payload)
}

pub fn draw_patches_indirect<'a>(op: &Op<'a>) -> Result<&'a DrawPatchesIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_PATCHES_INDIRECT);
    view::<DrawPatchesIndirect>(op.payload)
}

pub fn draw_indexed_patches_indirect<'a>(
    op: &Op<'a>,
) -> Result<&'a DrawIndexedPatchesIndirect, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_DRAW_INDEXED_PATCHES_INDIRECT);
    view::<DrawIndexedPatchesIndirect>(op.payload)
}

// --- 0x67 / 0x6a / 0x79 the store-action *options* -------------------------

pub const OPCODE_SET_COLOR_STORE_ACTION_OPTIONS: u32 = 0x67;
pub const SET_COLOR_STORE_ACTION_OPTIONS_TOTAL_LEN: u32 = 20;
pub const OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS: u32 = 0x6a;
pub const OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS: u32 = 0x79;
pub const SET_STORE_ACTION_OPTIONS_TOTAL_LEN: u32 = 16;

/// `MTLStoreActionOptions` for one attachment, separate from its store action.
///
/// **Each store-action opcode has an options sibling one above it**: `0x66`
/// →`0x67`, `0x69`→`0x6a`, `0x78`→`0x79`. That adjacency is the derived shape,
/// not a coincidence to lean on — each of the three was captured.
///
/// Two things do not carry over from the store-action records beside them.
/// The options are a **`u64`** where `ColorStoreAction::store_action` is a
/// `u32`, so the colour form is 20 bytes rather than 16 and its index sits at
/// payload `+8`. And the depth and stencil forms have no index at all, because
/// a pass has one of each — same as their store-action siblings.
///
/// Fixtures `render_set_color_store_action_options` (`0x1111` at index 3),
/// `render_set_depth_store_action_options` (`0x2222`) and
/// `render_set_stencil_store_action_options` (`0x3333`); three different
/// options values, so a decoder reading the wrong record's payload is visible.
#[repr(C)]
#[derive(Debug)]
pub struct StoreActionOptions {
    pub options: U64le,
}

// SAFETY: one align-1 all-bytes-valid `le` scalar.
unsafe impl Wire for StoreActionOptions {}

/// The colour form, which alone carries the attachment index.
#[repr(C)]
#[derive(Debug)]
pub struct ColorStoreActionOptions {
    pub options: U64le,
    pub index: U32le,
}

// SAFETY: two align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for ColorStoreActionOptions {}

#[inline]
pub fn is_store_action_options(opcode: u32) -> bool {
    matches!(
        opcode,
        OPCODE_SET_COLOR_STORE_ACTION_OPTIONS
            | OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS
            | OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS
    )
}

pub fn set_color_store_action_options<'a>(
    op: &Op<'a>,
) -> Result<&'a ColorStoreActionOptions, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_COLOR_STORE_ACTION_OPTIONS);
    view::<ColorStoreActionOptions>(op.payload)
}

/// The depth and stencil forms, which carry no index.
pub fn set_store_action_options<'a>(op: &Op<'a>) -> Result<&'a StoreActionOptions, WireError> {
    debug_assert!(matches!(
        op.opcode(),
        OPCODE_SET_DEPTH_STORE_ACTION_OPTIONS | OPCODE_SET_STENCIL_STORE_ACTION_OPTIONS
    ));
    view::<StoreActionOptions>(op.payload)
}

// --- 0x7a setTessellationFactorBuffer:offset:instanceStride: ---------------

pub const OPCODE_SET_TESSELLATION_FACTOR_BUFFER: u32 = 0x7a;
pub const SET_TESSELLATION_FACTOR_BUFFER_TOTAL_LEN: u32 = 28;

/// The buffer a tessellated draw reads its per-patch factors from.
///
/// Companion to [`OPCODE_SET_TESSELLATION_FACTOR_SCALE`] (`0x7b`), one below
/// it. Note this is **not** a [`BufferBind`] behind a [`BindHeader`]: there is
/// one tessellation factor buffer per encoder, so there is no slot and no
/// count — the record is the ref and its two `u64` directly. A reader that
/// assumed the bind shape would take the ref as `first` and the low half of
/// the offset as `count`.
///
/// Fixture `render_set_tessellation_factor_buffer` (ref 5151, offset `0x3456`,
/// stride `0x4567` — the two `u64` differ so a decoder that crossed them
/// cannot read back correct).
///
/// The draws that consume it are [`OPCODE_DRAW_PATCHES`] and its three
/// siblings, which are real records on this wire. This crate briefly claimed
/// they were refused by the serializer, on the strength of the ray-tracing
/// binds beside them having been; they were not driven, and they emit.
#[repr(C)]
#[derive(Debug)]
pub struct TessellationFactorBuffer {
    pub buffer_ref: U32le,
    pub offset: U64le,
    pub instance_stride: U64le,
}

// SAFETY: three align-1 all-bytes-valid `le` scalars.
unsafe impl Wire for TessellationFactorBuffer {}

pub fn set_tessellation_factor_buffer<'a>(
    op: &Op<'a>,
) -> Result<&'a TessellationFactorBuffer, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_SET_TESSELLATION_FACTOR_BUFFER);
    view::<TessellationFactorBuffer>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::OP_HEADER_LEN;
    use core::mem::{align_of, size_of};

    /// Every record's length beside the size of the view that reads it.
    ///
    /// Kept as one table so a new draw cannot be added without deciding which
    /// of the two tests below it belongs to.
    const RECORDS: &[(&str, u32, usize)] = &[
        ("draw", DRAW_TOTAL_LEN, size_of::<Draw>()),
        ("draw_wide", DRAW_WIDE_TOTAL_LEN, size_of::<DrawWide>()),
        (
            "draw_instanced",
            DRAW_INSTANCED_TOTAL_LEN,
            size_of::<DrawInstanced>(),
        ),
        (
            "draw_instanced_wide",
            DRAW_INSTANCED_WIDE_TOTAL_LEN,
            size_of::<DrawInstancedWide>(),
        ),
        (
            "draw_instanced_base",
            DRAW_INSTANCED_BASE_TOTAL_LEN,
            size_of::<DrawInstancedBase>(),
        ),
        (
            "draw_instanced_base_wide",
            DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN,
            size_of::<DrawInstancedBaseWide>(),
        ),
        (
            "draw_indexed",
            DRAW_INDEXED_TOTAL_LEN,
            size_of::<DrawIndexed>(),
        ),
        (
            "draw_indexed_wide",
            DRAW_INDEXED_WIDE_TOTAL_LEN,
            size_of::<DrawIndexedWide>(),
        ),
        (
            "draw_indexed_instanced",
            DRAW_INDEXED_INSTANCED_TOTAL_LEN,
            size_of::<DrawIndexedInstanced>(),
        ),
        (
            "draw_indexed_instanced_wide",
            DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN,
            size_of::<DrawIndexedInstancedWide>(),
        ),
        (
            "draw_indexed_instanced_base",
            DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN,
            size_of::<DrawIndexedInstancedBase>(),
        ),
        (
            "draw_indexed_instanced_base_wide",
            DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN,
            size_of::<DrawIndexedInstancedBaseWide>(),
        ),
        (
            "set_scissor",
            SET_SCISSOR_TOTAL_LEN,
            size_of::<ScissorRect>(),
        ),
        (
            "set_viewport",
            SET_VIEWPORT_TOTAL_LEN,
            size_of::<Viewport>(),
        ),
        ("set_mode", SET_MODE_TOTAL_LEN, size_of::<ModeState>()),
        ("set_float", SET_FLOAT_TOTAL_LEN, size_of::<FloatState>()),
        ("set_state", SET_STATE_TOTAL_LEN, size_of::<StateRef>()),
        (
            "set_buffer_offset",
            SET_BUFFER_OFFSET_TOTAL_LEN,
            size_of::<BufferOffset>(),
        ),
        ("fence", FENCE_TOTAL_LEN, size_of::<Fence>()),
        (
            "set_color_store_action",
            SET_COLOR_STORE_ACTION_TOTAL_LEN,
            size_of::<ColorStoreAction>(),
        ),
        (
            "set_stencil_reference",
            SET_STENCIL_REFERENCE_TOTAL_LEN,
            size_of::<StencilReference>(),
        ),
        (
            "set_depth_bias",
            SET_DEPTH_BIAS_TOTAL_LEN,
            size_of::<DepthBias>(),
        ),
        (
            "set_visibility_result_mode",
            SET_VISIBILITY_RESULT_MODE_TOTAL_LEN,
            size_of::<VisibilityResult>(),
        ),
        (
            "set_blend_color",
            SET_BLEND_COLOR_TOTAL_LEN,
            size_of::<BlendColor>(),
        ),
    ];

    #[test]
    fn no_view_claims_more_bytes_than_its_record_has() {
        for (name, total, body) in RECORDS {
            assert!(
                body + OP_HEADER_LEN <= *total as usize,
                "{name}: view is {body} bytes, record holds {} after the header",
                *total as usize - OP_HEADER_LEN
            );
        }
    }

    /// A record's length is always a multiple of four, so a payload whose last
    /// written field ends on a 2-byte boundary is followed by exactly two bytes
    /// the serializer never touched. Those stayed `0xAA` poison in the capture,
    /// which means on a real wire they hold whatever the guest's ring last
    /// contained — so every view stops short of them rather than naming them
    /// padding, which would be a claim that nothing is ever there.
    #[test]
    fn a_record_either_fits_its_view_exactly_or_leaves_two_unwritten_bytes() {
        for (name, total, body) in RECORDS {
            let slack = *total as usize - OP_HEADER_LEN - body;
            assert!(
                slack == 0 || slack == 2,
                "{name}: {slack} bytes between the view and the record length; \
                 expected 0 or the 2 that four-byte record alignment leaves"
            );
            assert_eq!(total % 4, 0, "{name}: record length is not 4-aligned");
        }
    }

    /// The draw family is exactly the opcodes `0x00..=0x0b`, in compact/wide
    /// pairs where the wide one is the even member. A gap here would mean a
    /// selector whose record this module cannot read.
    #[test]
    fn the_twelve_draw_opcodes_pair_up_and_leave_no_gap() {
        let pairs = [
            (OPCODE_DRAW_WIDE, OPCODE_DRAW),
            (OPCODE_DRAW_INSTANCED_WIDE, OPCODE_DRAW_INSTANCED),
            (OPCODE_DRAW_INSTANCED_BASE_WIDE, OPCODE_DRAW_INSTANCED_BASE),
            (OPCODE_DRAW_INDEXED_WIDE, OPCODE_DRAW_INDEXED),
            (
                OPCODE_DRAW_INDEXED_INSTANCED_WIDE,
                OPCODE_DRAW_INDEXED_INSTANCED,
            ),
            (
                OPCODE_DRAW_INDEXED_INSTANCED_BASE_WIDE,
                OPCODE_DRAW_INDEXED_INSTANCED_BASE,
            ),
        ];
        let mut seen = [false; 12];
        for (i, (wide, compact)) in pairs.iter().enumerate() {
            assert_eq!(*wide % 2, 0, "pair {i}: wide member is not even");
            assert_eq!(*compact, *wide + 1, "pair {i}: not adjacent");
            seen[*wide as usize] = true;
            seen[*compact as usize] = true;
        }
        assert!(seen.iter().all(|s| *s), "the range 0x00..=0x0b has a gap");

        // A wide record is always longer than its compact sibling, because
        // widening its counts is the whole reason the guest chose it.
        for (name, wide, compact) in [
            ("draw", DRAW_WIDE_TOTAL_LEN, DRAW_TOTAL_LEN),
            (
                "draw_instanced",
                DRAW_INSTANCED_WIDE_TOTAL_LEN,
                DRAW_INSTANCED_TOTAL_LEN,
            ),
            (
                "draw_instanced_base",
                DRAW_INSTANCED_BASE_WIDE_TOTAL_LEN,
                DRAW_INSTANCED_BASE_TOTAL_LEN,
            ),
            (
                "draw_indexed",
                DRAW_INDEXED_WIDE_TOTAL_LEN,
                DRAW_INDEXED_TOTAL_LEN,
            ),
            (
                "draw_indexed_instanced",
                DRAW_INDEXED_INSTANCED_WIDE_TOTAL_LEN,
                DRAW_INDEXED_INSTANCED_TOTAL_LEN,
            ),
            (
                "draw_indexed_instanced_base",
                DRAW_INDEXED_INSTANCED_BASE_WIDE_TOTAL_LEN,
                DRAW_INDEXED_INSTANCED_BASE_TOTAL_LEN,
            ),
        ] {
            assert!(
                wide > compact,
                "{name}: wide form is not longer than the compact one"
            );
        }
    }

    #[test]
    fn the_instanced_draw_view_stops_before_the_bytes_apple_leaves_unwritten() {
        // The payload is 12 bytes; the serializer writes 10. Reading the last
        // two would be reading whatever the guest's ring last held.
        assert_eq!(size_of::<DrawInstancedBase>(), 10);
        assert_eq!(
            size_of::<DrawInstancedBase>() + OP_HEADER_LEN + 2,
            DRAW_INSTANCED_BASE_TOTAL_LEN as usize
        );
    }

    #[test]
    fn every_render_view_is_align_one() {
        assert_eq!(align_of::<Draw>(), 1);
        assert_eq!(align_of::<DrawWide>(), 1);
        assert_eq!(align_of::<DrawInstanced>(), 1);
        assert_eq!(align_of::<DrawInstancedWide>(), 1);
        assert_eq!(align_of::<DrawInstancedBase>(), 1);
        assert_eq!(align_of::<DrawInstancedBaseWide>(), 1);
        assert_eq!(align_of::<DrawIndexed>(), 1);
        assert_eq!(align_of::<DrawIndexedWide>(), 1);
        assert_eq!(align_of::<DrawIndexedInstanced>(), 1);
        assert_eq!(align_of::<DrawIndexedInstancedWide>(), 1);
        assert_eq!(align_of::<DrawIndexedInstancedBase>(), 1);
        assert_eq!(align_of::<DrawIndexedInstancedBaseWide>(), 1);
        assert_eq!(align_of::<ScissorRect>(), 1);
        assert_eq!(align_of::<Viewport>(), 1);
        assert_eq!(align_of::<ModeState>(), 1);
        assert_eq!(align_of::<FloatState>(), 1);
        assert_eq!(align_of::<StencilReference>(), 1);
        assert_eq!(align_of::<DepthBias>(), 1);
        assert_eq!(align_of::<VisibilityResult>(), 1);
        assert_eq!(align_of::<BlendColor>(), 1);
        assert_eq!(align_of::<BindHeader>(), 1);
        assert_eq!(align_of::<RefBind>(), 1);
        assert_eq!(align_of::<BufferBind>(), 1);
        assert_eq!(align_of::<BufferOffset>(), 1);
        assert_eq!(align_of::<StateRef>(), 1);
        assert_eq!(align_of::<Fence>(), 1);
        assert_eq!(align_of::<UseResource>(), 1);
        assert_eq!(align_of::<ColorStoreAction>(), 1);
    }

    /// No opcode may be claimed by two shape predicates.
    ///
    /// Eight of them now route records to a shared view, and an overlap sends a
    /// record to the wrong one — where a length check cannot notice, because
    /// several of these shapes are the same size. `0x6e` is the case that makes
    /// this worth asserting: it is a buffer bind, and `0x6f` one apart from it
    /// is a buffer *offset*, a different record entirely.
    #[test]
    fn no_opcode_belongs_to_two_shapes() {
        /// A shape predicate and the name to report it by.
        type Shape = (&'static str, fn(u32) -> bool);
        let shapes: [Shape; 6] = [
            ("mode_state", is_mode_state),
            ("float_state", is_float_state),
            ("ref_bind", is_ref_bind),
            ("buffer_bind", is_buffer_bind),
            ("buffer_offset", is_buffer_offset),
            ("state_ref", is_state_ref),
        ];
        for opcode in 0..=0x98u32 {
            let mut claimed = 0usize;
            let mut first_name = "";
            for (name, f) in &shapes {
                if f(opcode) {
                    assert_eq!(
                        claimed, 0,
                        "opcode {opcode:#x} is claimed by both {first_name} and {name}"
                    );
                    claimed += 1;
                    first_name = name;
                }
            }
            if is_fence(opcode) {
                assert_eq!(
                    claimed, 0,
                    "opcode {opcode:#x} is a fence and also {first_name}"
                );
            }
        }
        // Spot-check the pair that motivated this.
        assert!(is_buffer_bind(OPCODE_SET_FRAGMENT_BUFFER));
        assert!(is_buffer_offset(OPCODE_SET_FRAGMENT_BUFFER_OFFSET));
        assert_ne!(
            OPCODE_SET_FRAGMENT_BUFFER,
            OPCODE_SET_FRAGMENT_BUFFER_OFFSET
        );
    }

    /// The two shape predicates must agree with the opcodes their views read,
    /// and must not overlap: a record routed to the wrong shared view reads a
    /// float as an integer without any length check noticing.
    #[test]
    fn the_shared_state_shapes_claim_exactly_their_own_opcodes() {
        for op in [
            OPCODE_SET_CULL_MODE,
            OPCODE_SET_FRONT_FACING,
            OPCODE_SET_DEPTH_CLIP_MODE,
            OPCODE_SET_TRIANGLE_FILL_MODE,
        ] {
            assert!(
                is_mode_state(op),
                "{op:#x} is a mode state and is not claimed"
            );
            assert!(!is_float_state(op), "{op:#x} is claimed by both shapes");
        }
        for op in [OPCODE_SET_LINE_WIDTH, OPCODE_SET_TESSELLATION_FACTOR_SCALE] {
            assert!(
                is_float_state(op),
                "{op:#x} is a float state and is not claimed"
            );
            assert!(!is_mode_state(op), "{op:#x} is claimed by both shapes");
        }
        for op in [
            OPCODE_SET_SCISSOR,
            OPCODE_SET_VIEWPORT,
            OPCODE_SET_BLEND_COLOR,
            OPCODE_SET_DEPTH_BIAS,
            OPCODE_SET_STENCIL_REFERENCE,
            OPCODE_SET_VISIBILITY_RESULT_MODE,
            OPCODE_DRAW,
        ] {
            assert!(
                !is_mode_state(op) && !is_float_state(op),
                "{op:#x} over-claimed"
            );
        }
    }

    /// A bind's `count` comes off the wire, so it is attacker-controlled.
    ///
    /// The entries are read from the same record, so a `count` larger than the
    /// record holds must be refused rather than producing a slice that runs off
    /// the end. This is the one place in the module where a length field
    /// governs how much is read.
    #[test]
    fn a_bind_count_larger_than_the_record_is_refused_rather_than_read() {
        // Head plus two entries' worth of bytes.
        let mut payload = [0u8; 8 + 2 * 4];
        payload[0..4].copy_from_slice(&3u32.to_le_bytes()); // first
        payload[4..8].copy_from_slice(&2u32.to_le_bytes()); // count, honest
        let (head, entries) = bind_entries::<RefBind>(&payload).expect("two entries fit");
        assert_eq!(head.first.get(), 3);
        assert_eq!(entries.len(), 2);

        // The same bytes claiming three entries, then a count that would
        // overflow the multiply if it were not checked.
        payload[4..8].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            bind_entries::<RefBind>(&payload),
            Err(WireError::Short { .. })
        ));
        payload[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(bind_entries::<RefBind>(&payload).is_err());

        // A buffer entry is 12 bytes, not 16, so the same record holds fewer.
        let mut buf = [0u8; 8 + 12];
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            bind_entries::<BufferBind>(&buf).expect("one fits").1.len(),
            1
        );
        buf[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(bind_entries::<BufferBind>(&buf).is_err());
    }

    /// The entry widths the record lengths imply.
    ///
    /// A singular bind is the `count == 1` case, so its record length is the
    /// header plus one entry — which is what ties these two sizes to the
    /// fixtures without a separate constant per selector.
    #[test]
    fn a_singular_bind_is_the_header_plus_exactly_one_entry() {
        assert_eq!(size_of::<BindHeader>(), 8);
        assert_eq!(size_of::<RefBind>(), 4);
        assert_eq!(size_of::<BufferBind>(), 12);
        // render_set_vertex_texture: 20 bytes on the wire.
        assert_eq!(
            OP_HEADER_LEN + size_of::<BindHeader>() + size_of::<RefBind>(),
            20
        );
        // render_set_vertex_buffer: 28.
        assert_eq!(
            OP_HEADER_LEN + size_of::<BindHeader>() + size_of::<BufferBind>(),
            28
        );
        // render_use_resource: 20, and its head has no `first`.
        assert_eq!(size_of::<UseResource>(), 8);
        assert_eq!(
            OP_HEADER_LEN + size_of::<UseResource>() + size_of::<RefBind>(),
            20
        );
    }

    /// The plural viewport and scissor records are their singular element with a
    /// count in front, and their two counts are *not* the same width.
    ///
    /// Both facts are load-bearing and neither is guessable: reusing one head
    /// for both would put the scissor array four bytes early, and reusing the
    /// singular record's total length as the plural element size would be right
    /// only by accident if the element ever diverged.
    #[test]
    fn the_plural_state_records_are_a_count_plus_the_singular_payload() {
        assert_eq!(
            size_of::<ScissorRect>() + OP_HEADER_LEN,
            SET_SCISSOR_TOTAL_LEN as usize,
            "the plural scissor element is the singular record's payload"
        );
        assert_eq!(
            size_of::<Viewport>() + OP_HEADER_LEN,
            SET_VIEWPORT_TOTAL_LEN as usize,
            "the plural viewport element is the singular record's payload"
        );
        assert_eq!(size_of::<SetScissorRects>(), 8);
        assert_eq!(size_of::<SetViewports>(), 4);
    }

    /// Every fixed-length record added with the barrier and indirect families
    /// is its body plus the header, except the one that leaves a tail.
    #[test]
    fn the_indirect_and_barrier_records_are_their_body_plus_the_header() {
        assert_eq!(
            size_of::<DrawIndexedIndirect>() + OP_HEADER_LEN,
            DRAW_INDEXED_INDIRECT_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<ExecuteCommandsIndirect>() + OP_HEADER_LEN,
            EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<ExecuteCommandsRange>() + OP_HEADER_LEN,
            EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize
        );
        assert_eq!(
            size_of::<MemoryBarrierScope>() + OP_HEADER_LEN,
            MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize
        );
        // `textureBarrier` is the header alone.
        assert_eq!(TEXTURE_BARRIER_TOTAL_LEN as usize, OP_HEADER_LEN);
        // The indirect draw stops two bytes short of its record, and those two
        // bytes are never written by the serializer.
        assert_eq!(
            size_of::<DrawIndirect>() + OP_HEADER_LEN + 2,
            DRAW_INDIRECT_TOTAL_LEN as usize
        );
    }

    /// A count in a variable-length record is guest-controlled, so every one of
    /// them refuses a count larger than the record holds rather than producing
    /// a slice off the end.
    #[test]
    fn a_variable_length_count_past_the_record_is_refused() {
        // `no_std`: a fixed buffer and a length, not a Vec.
        fn record(buf: &mut [u8; 64], opcode: u32, head: &[u8], tail_len: usize) -> usize {
            let total = OP_HEADER_LEN + head.len() + tail_len;
            assert!(total <= buf.len());
            *buf = [0u8; 64];
            buf[..4].copy_from_slice(&opcode.to_le_bytes());
            buf[4..8].copy_from_slice(&(total as u32).to_le_bytes());
            buf[OP_HEADER_LEN..OP_HEADER_LEN + head.len()].copy_from_slice(head);
            total
        }
        let mut buf = [0u8; 64];

        // memoryBarrierWithResources: head is [count][after][before]; claim two
        // refs and supply one.
        let mut head = [0u8; 8];
        head[..4].copy_from_slice(&2u32.to_le_bytes());
        let n = record(
            &mut buf,
            OPCODE_MEMORY_BARRIER_RESOURCES,
            &head,
            size_of::<RefBind>(),
        );
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(
            memory_barrier_resources(&o),
            Err(WireError::Short { .. })
        ));

        // setScissorRects: claim one rect and supply none.
        let n = record(&mut buf, OPCODE_SET_SCISSOR_RECTS, &1u64.to_le_bytes(), 0);
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(
            set_scissor_rects(&o),
            Err(WireError::Short { .. })
        ));

        // setViewports: same, at the other count width.
        let n = record(&mut buf, OPCODE_SET_VIEWPORTS, &1u32.to_le_bytes(), 0);
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(set_viewports(&o), Err(WireError::Short { .. })));

        // And the multiply that produces the byte count must not wrap.
        let n = record(
            &mut buf,
            OPCODE_SET_SCISSOR_RECTS,
            &u64::MAX.to_le_bytes(),
            0,
        );
        let o = crate::op::op(&buf[..n], 0).expect("header fits");
        assert!(matches!(
            set_scissor_rects(&o),
            Err(WireError::CountOverflow { .. })
        ));
    }

    /// A negative base vertex must not read back as a large positive index.
    #[test]
    fn a_negative_base_vertex_survives_the_view_as_a_negative_number() {
        let mut payload = [0u8; 20];
        payload[14..16].copy_from_slice(&(-2i16).to_le_bytes());
        let v = view::<DrawIndexedInstancedBase>(&payload).expect("fits");
        assert_eq!(v.base_vertex.get(), -2);
    }
}
