//! The records of an `ICB_WRITE` ([`RGFX_CMD_ICB_WRITE`](crate::wire::RGFX_CMD_ICB_WRITE)):
//! the CPU's encodes into an indirect command buffer, one record a setter call of
//! `MTLIndirectRenderCommand`, `MTLIndirectComputeCommand` or the buffer's `resetWithRange:`, in
//! the order the guest made them.
//!
//! Each record is an [`RgfxIcbRecord`] header, then its op's body (none for the barriers and the
//! resets): `size` is the whole record's bytes, a multiple of 8, header included, and
//! `command_index` the command the call was made on (`indirectRenderCommandAtIndex:` or
//! `indirectComputeCommandAtIndex:`), or the range's location for a reset of a range. Objects are
//! named by their `ObjId` (0 for nil), buffers with their offset. Counts and sizes are their low
//! 32 bits, which is what a native command's record holds of them (measured on an M3 Max and an
//! M4, metal2vulkan's `indirect_command_buffer`).
//!
//! The render ops are `0x01` to `0x1f`, the compute ops `0x21` to `0x3f`. The host carries each
//! record out on the platform's buffer, and skips one it cannot (an object the sender does not
//! hold, or the platform's refusal), alone.
//!
//! **Slots** ([`RGFX_CMD_ICB_SLOTS`](crate::wire::RGFX_CMD_ICB_SLOTS), where the guest took
//! [`RGFX_FEATURE_ICB_SLOTS`](crate::wire::RGFX_FEATURE_ICB_SLOTS)): in place of records, the
//! guest writes each command's state into a slot of its own in a block of the process's region,
//! as a GPU's driver writes a command's record, and posts nothing while it encodes. A slot is a
//! fixed part ([`RgfxIcbRenderSlot`], [`RgfxIcbComputeSlot`]), then its bindings
//! ([`RgfxIcbSlotBuffer`]: a render slot's vertex buffers, then its fragment buffers; a compute
//! slot's kernel buffers, then its threadgroup memory lengths, a `u32` each, padded to 8 bytes).
//! Each setter writes its field and sets the field's bit in the slot's `mask` or `mask_b`; a
//! command's `reset` sets the mask to [`RGFX_ICB_SLOT_RESET`] alone and a `resetWithRange:` to
//! [`RGFX_ICB_SLOT_RESET_RANGE`] alone, so what was set before them goes. The guest sets each bit
//! with an atomic OR, after the field's bytes; the host swaps each mask for 0 before it reads the
//! slot, and carries out the bits it took, so it carries out what the guest set since its last
//! take, as the slot holds it then: the latest encode. A slot names objects by their `ObjId`, as
//! records do, and buffers with their offset; a field is the body its record would carry.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A record's header.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbRecord {
    /// An `RGFX_ICB_OP_*`.
    pub op: u16,
    /// The record's bytes, this header included: a multiple of 8.
    pub size: u16,
    /// The command the call was made on, or a reset range's location.
    pub command_index: u32,
}

/// `setRenderPipelineState:` ([`RgfxIcbObject`]).
pub const RGFX_ICB_OP_RENDER_PIPELINE: u16 = 0x01;
/// `setVertexBuffer:offset:atIndex:` ([`RgfxIcbBuffer`]).
pub const RGFX_ICB_OP_VERTEX_BUFFER: u16 = 0x02;
/// `setVertexBuffer:offset:attributeStride:atIndex:` ([`RgfxIcbBufferStride`]).
pub const RGFX_ICB_OP_VERTEX_BUFFER_STRIDE: u16 = 0x03;
/// `setFragmentBuffer:offset:atIndex:` ([`RgfxIcbBuffer`]).
pub const RGFX_ICB_OP_FRAGMENT_BUFFER: u16 = 0x04;
/// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:` ([`RgfxIcbDraw`]).
pub const RGFX_ICB_OP_DRAW: u16 = 0x05;
/// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`
/// ([`RgfxIcbDrawIndexed`]).
pub const RGFX_ICB_OP_DRAW_INDEXED: u16 = 0x06;
/// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:`
/// ([`RgfxIcbDrawPatches`]).
pub const RGFX_ICB_OP_DRAW_PATCHES: u16 = 0x07;
/// `drawIndexedPatches:...` ([`RgfxIcbDrawIndexedPatches`]).
pub const RGFX_ICB_OP_DRAW_INDEXED_PATCHES: u16 = 0x08;
/// A render command's `setBarrier` (no body).
pub const RGFX_ICB_OP_RENDER_SET_BARRIER: u16 = 0x09;
/// A render command's `clearBarrier` (no body).
pub const RGFX_ICB_OP_RENDER_CLEAR_BARRIER: u16 = 0x0a;
/// A render command's `reset` (no body).
pub const RGFX_ICB_OP_RENDER_RESET: u16 = 0x0b;
/// `resetWithRange:` of a buffer of render commands ([`RgfxIcbRange`]): each command from
/// `command_index` reset, as its `reset` does.
pub const RGFX_ICB_OP_RENDER_RESET_RANGE: u16 = 0x0c;

/// `setComputePipelineState:` ([`RgfxIcbObject`]).
pub const RGFX_ICB_OP_COMPUTE_PIPELINE: u16 = 0x21;
/// `setKernelBuffer:offset:atIndex:` ([`RgfxIcbBuffer`]).
pub const RGFX_ICB_OP_KERNEL_BUFFER: u16 = 0x22;
/// `setKernelBuffer:offset:attributeStride:atIndex:` ([`RgfxIcbBufferStride`]).
pub const RGFX_ICB_OP_KERNEL_BUFFER_STRIDE: u16 = 0x23;
/// `concurrentDispatchThreadgroups:threadsPerThreadgroup:` ([`RgfxIcbDispatch`]).
pub const RGFX_ICB_OP_DISPATCH_THREADGROUPS: u16 = 0x24;
/// `concurrentDispatchThreads:threadsPerThreadgroup:` ([`RgfxIcbDispatch`]).
pub const RGFX_ICB_OP_DISPATCH_THREADS: u16 = 0x25;
/// `setThreadgroupMemoryLength:atIndex:` ([`RgfxIcbThreadgroupMemory`]).
pub const RGFX_ICB_OP_THREADGROUP_MEMORY: u16 = 0x26;
/// `setStageInRegion:` ([`RgfxIcbRegion`]).
pub const RGFX_ICB_OP_STAGE_IN_REGION: u16 = 0x27;
/// `setImageblockWidth:height:` ([`RgfxIcbImageblock`]).
pub const RGFX_ICB_OP_IMAGEBLOCK: u16 = 0x28;
/// A compute command's `setBarrier` (no body).
pub const RGFX_ICB_OP_COMPUTE_SET_BARRIER: u16 = 0x29;
/// A compute command's `clearBarrier` (no body).
pub const RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER: u16 = 0x2a;
/// A compute command's `reset` (no body).
pub const RGFX_ICB_OP_COMPUTE_RESET: u16 = 0x2b;
/// `resetWithRange:` of a buffer of compute commands ([`RgfxIcbRange`]).
pub const RGFX_ICB_OP_COMPUTE_RESET_RANGE: u16 = 0x2c;

/// The body of a pipeline state's setter: the state, 0 for nil.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbObject {
    /// The pipeline state.
    pub obj_id: u64,
}

/// The body of a buffer's setter.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbBuffer {
    /// The buffer, 0 for nil.
    pub obj_id: u64,
    /// `offset`.
    pub offset: u64,
    /// `index`.
    pub index: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// The body of a buffer's setter with its attribute stride.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbBufferStride {
    /// The buffer, 0 for nil.
    pub obj_id: u64,
    /// `offset`.
    pub offset: u64,
    /// `attributeStride`.
    pub stride: u64,
    /// `index`.
    pub index: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_1c: u32,
}

/// The body of [`RGFX_ICB_OP_DRAW`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbDraw {
    /// `MTLPrimitiveType`.
    pub primitive_type: u32,
    /// `vertexStart`.
    pub vertex_start: u32,
    /// `vertexCount`.
    pub vertex_count: u32,
    /// `instanceCount`.
    pub instance_count: u32,
    /// `baseInstance`.
    pub base_instance: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// The body of [`RGFX_ICB_OP_DRAW_INDEXED`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbDrawIndexed {
    /// `MTLPrimitiveType`.
    pub primitive_type: u32,
    /// `indexCount`.
    pub index_count: u32,
    /// `MTLIndexType`.
    pub index_type: u32,
    /// `instanceCount`.
    pub instance_count: u32,
    /// The index buffer, 0 for nil.
    pub index_buffer_obj_id: u64,
    /// `indexBufferOffset`.
    pub index_buffer_offset: u64,
    /// `baseVertex` (`NSInteger`; its low 32 bits, signed).
    pub base_vertex: i32,
    /// `baseInstance`.
    pub base_instance: u32,
}

/// The body of [`RGFX_ICB_OP_DRAW_PATCHES`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbDrawPatches {
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u32,
    /// `patchStart`.
    pub patch_start: u32,
    /// `patchCount`.
    pub patch_count: u32,
    /// `instanceCount`.
    pub instance_count: u32,
    /// `baseInstance`.
    pub base_instance: u32,
    /// `tessellationFactorBufferInstanceStride`.
    pub instance_stride: u32,
    /// The patch index buffer, 0 for nil.
    pub patch_index_buffer_obj_id: u64,
    /// `patchIndexBufferOffset`.
    pub patch_index_buffer_offset: u64,
    /// The tessellation factor buffer, 0 for nil.
    pub tessellation_factor_buffer_obj_id: u64,
    /// `tessellationFactorBufferOffset`.
    pub tessellation_factor_buffer_offset: u64,
}

/// The body of [`RGFX_ICB_OP_DRAW_INDEXED_PATCHES`]: a patch draw's, then its control-point
/// index buffer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbDrawIndexedPatches {
    /// The patch draw's fields.
    pub patches: RgfxIcbDrawPatches,
    /// The control-point index buffer, 0 for nil.
    pub control_point_index_buffer_obj_id: u64,
    /// `controlPointIndexBufferOffset`.
    pub control_point_index_buffer_offset: u64,
}

/// The body of a dispatch: the grid (threadgroups or threads), then the threadgroup's size.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbDispatch {
    /// `threadgroupsPerGrid` or `threadsPerGrid`: width, height, depth.
    pub grid: [u32; 3],
    /// `threadsPerThreadgroup`: width, height, depth.
    pub threadgroup: [u32; 3],
}

/// The body of [`RGFX_ICB_OP_THREADGROUP_MEMORY`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbThreadgroupMemory {
    /// `length`.
    pub length: u32,
    /// `index`.
    pub index: u32,
}

/// The body of [`RGFX_ICB_OP_STAGE_IN_REGION`]: `MTLRegion`'s origin, then its size.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbRegion {
    /// x, y, z.
    pub origin: [u32; 3],
    /// width, height, depth.
    pub size: [u32; 3],
}

/// The body of [`RGFX_ICB_OP_IMAGEBLOCK`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbImageblock {
    /// `width`.
    pub width: u32,
    /// `height`.
    pub height: u32,
}

/// The body of a reset of a range: its length (its location is the record's `command_index`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbRange {
    /// `length`.
    pub length: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_04: u32,
}

/// A slot's mask bit: the command's `reset`, carried out before the other bits.
pub const RGFX_ICB_SLOT_RESET: u64 = 1 << 0;
/// A slot's mask bit: the command was reset by its buffer's `resetWithRange:`, carried out before
/// the other bits; a run of commands with this bit alone is reset in one call.
pub const RGFX_ICB_SLOT_RESET_RANGE: u64 = 1 << 1;
/// A slot's mask bit: `setRenderPipelineState:` or `setComputePipelineState:` (`pipeline`).
pub const RGFX_ICB_SLOT_PIPELINE: u64 = 1 << 2;
/// A slot's mask bit: the draw or the dispatch (`op` and its body).
pub const RGFX_ICB_SLOT_ACTION: u64 = 1 << 3;
/// A slot's mask bit: `setBarrier` or `clearBarrier` (`barrier`).
pub const RGFX_ICB_SLOT_BARRIER: u64 = 1 << 4;
/// A compute slot's mask bit: `setStageInRegion:` (`stage_in`).
pub const RGFX_ICB_SLOT_STAGE_IN: u64 = 1 << 5;
/// A compute slot's mask bit: `setImageblockWidth:height:` (`imageblock`).
pub const RGFX_ICB_SLOT_IMAGEBLOCK: u64 = 1 << 6;
/// A slot's mask bit of its first binding of the first kind (a render slot's vertex buffer 0, a
/// compute slot's kernel buffer 0); binding `i` is this bit shifted left by `i`, up to 31.
pub const RGFX_ICB_SLOT_BUFFER_0: u64 = 0x1_0000_0000;
/// The most bindings of each kind a slot holds: a `mask_b` bit each, and the upper half of `mask`.
pub const RGFX_ICB_SLOT_BINDINGS: u32 = 32;

/// A render command's slot, its fixed part; then `vertex_count` [`RgfxIcbSlotBuffer`]s, then
/// `fragment_count`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbRenderSlot {
    /// `RGFX_ICB_SLOT_*` bits; vertex buffer `i`'s is `RGFX_ICB_SLOT_BUFFER_0 << i`.
    pub mask: u64,
    /// Fragment buffer `i`'s bit is `1 << i`.
    pub mask_b: u64,
    /// The pipeline state, 0 for nil.
    pub pipeline: u64,
    /// The draw's op (`RGFX_ICB_OP_DRAW` to `RGFX_ICB_OP_DRAW_INDEXED_PATCHES`).
    pub op: u16,
    /// 1 for `setBarrier`, 0 for `clearBarrier`.
    pub barrier: u16,
    /// Bit `i`: vertex buffer `i` was set with its attribute stride.
    pub strided: u32,
    /// The draw's body, as its record's (72 bytes), zeros past it.
    pub action: [u64; 9],
}

/// A compute command's slot, its fixed part; then `kernel_count` [`RgfxIcbSlotBuffer`]s, then
/// `threadgroup_count` threadgroup memory lengths (`u32`), padded to 8 bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbComputeSlot {
    /// `RGFX_ICB_SLOT_*` bits; kernel buffer `i`'s is `RGFX_ICB_SLOT_BUFFER_0 << i`.
    pub mask: u64,
    /// Threadgroup memory length `i`'s bit is `1 << i`.
    pub mask_b: u64,
    /// The pipeline state, 0 for nil.
    pub pipeline: u64,
    /// The dispatch's op (`RGFX_ICB_OP_DISPATCH_THREADGROUPS` or `RGFX_ICB_OP_DISPATCH_THREADS`).
    pub op: u16,
    /// 1 for `setBarrier`, 0 for `clearBarrier`.
    pub barrier: u16,
    /// Bit `i`: kernel buffer `i` was set with its attribute stride.
    pub strided: u32,
    /// The dispatch's body.
    pub dispatch: RgfxIcbDispatch,
    /// `setStageInRegion:`'s body.
    pub stage_in: RgfxIcbRegion,
    /// `setImageblockWidth:height:`'s body.
    pub imageblock: RgfxIcbImageblock,
}

/// A buffer binding of a slot: the buffer (0 for nil), its offset, and the attribute stride where
/// the slot's `strided` bit says it was set with one.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbSlotBuffer {
    /// The buffer, 0 for nil.
    pub obj_id: u64,
    /// `offset`.
    pub offset: u64,
    /// `attributeStride`.
    pub stride: u64,
}

/// The least bytes a slot of `kind` ([`RGFX_ICB_SLOTS_RENDER`](crate::wire::RGFX_ICB_SLOTS_RENDER)
/// or [`RGFX_ICB_SLOTS_COMPUTE`](crate::wire::RGFX_ICB_SLOTS_COMPUTE)) with `a` bindings of the
/// first kind and `b` of the second takes: its fixed part, then its bindings; `None` for another
/// kind or more than [`RGFX_ICB_SLOT_BINDINGS`] of either.
pub fn slot_bytes(kind: u16, a: u32, b: u32) -> Option<u64> {
    use core::mem::size_of;
    if a > RGFX_ICB_SLOT_BINDINGS || b > RGFX_ICB_SLOT_BINDINGS {
        return None;
    }
    let buffer = size_of::<RgfxIcbSlotBuffer>() as u64;
    let (a, b) = (u64::from(a), u64::from(b));
    match kind {
        crate::wire::RGFX_ICB_SLOTS_RENDER => {
            Some(size_of::<RgfxIcbRenderSlot>() as u64 + buffer * (a + b))
        }
        crate::wire::RGFX_ICB_SLOTS_COMPUTE => {
            Some(size_of::<RgfxIcbComputeSlot>() as u64 + buffer * a + (4 * b).next_multiple_of(8))
        }
        _ => None,
    }
}

const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxIcbRenderSlot>() == 104);
    assert!(size_of::<RgfxIcbComputeSlot>() == 88);
    assert!(size_of::<RgfxIcbSlotBuffer>() == 24);
    assert!(size_of::<RgfxIcbDrawIndexedPatches>() == 72);
    assert!(size_of::<RgfxIcbRecord>() == 8);
    assert!(size_of::<RgfxIcbObject>() == 8);
    assert!(size_of::<RgfxIcbBuffer>() == 24);
    assert!(size_of::<RgfxIcbBufferStride>() == 32);
    assert!(size_of::<RgfxIcbDraw>() == 24);
    assert!(size_of::<RgfxIcbDrawIndexed>() == 40);
    assert!(size_of::<RgfxIcbDrawPatches>() == 56);
    assert!(size_of::<RgfxIcbDrawIndexedPatches>() == 72);
    assert!(size_of::<RgfxIcbDispatch>() == 24);
    assert!(size_of::<RgfxIcbThreadgroupMemory>() == 8);
    assert!(size_of::<RgfxIcbRegion>() == 24);
    assert!(size_of::<RgfxIcbImageblock>() == 8);
    assert!(size_of::<RgfxIcbRange>() == 8);
};
