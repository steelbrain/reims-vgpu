//! The compute-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_COMPUTE`](super::RGFX_OP_GROUP_COMPUTE)):
//! one tag and one body struct per `MTLComputeCommandEncoder` selector the guest
//! serialises (docs/graphics-kext/design.md, "Stage 3 wire protocol"). The host's
//! compute decoder (`reims-vgpu`'s `graphics::decode::compute`) reads the body,
//! resolves its object references and sends the matching `ComputeCmd`.
//!
//! A body struct is the op's bytes **after** its [`RgfxOpHeader`](super::RgfxOpHeader),
//! not the whole op (see the [module doc](super)). Object references are `u64`
//! `obj_id` (0 = nil). Floating-point arguments travel as the bits of their
//! `f32`/`f64`, so the schema stays all-integer and comparable.
//!
//! **Array-tailed ops.** An op that binds or uses a run of objects — `setBuffers:`,
//! `setTextures:`, `setSamplerStates:`, `useHeaps:`, `useResources:`,
//! `memoryBarrierWithResources:` — carries a fixed head (its range or count) and
//! then that many fixed-size element records right after the head, as the render
//! pass's colour attachments do. The element record's struct documents its stride;
//! the head's `length` (an `NSRange`) or `count` is how many follow.
//!
//! **Empty-bodied ops.** `encodeStartElse`, `encodeStartDoWhile` and
//! `insertCompressedTextureReinterpretationFlush` take no arguments, so their op
//! carries no body (the header alone); they have a tag but no body struct.
//!
//! `endEncoding` has no op here: the command-buffer walk's `ENCODER_END` framing op
//! ends the open compute encoder (`reims-vgpu`'s `graphics::decode::cmdbuf`).

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

// The op tags, in the compute group's `0x2xxx` range.

/// Op `SET_COMPUTE_PIPELINE_STATE` ([`RgfxComputeOpSetComputePipelineState`]):
/// `setComputePipelineState:`.
pub const RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE: u16 = 0x2000;
/// Op `SET_BUFFER_OFFSET` ([`RgfxComputeOpSetBufferOffset`]): `setBufferOffset:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_BUFFER_OFFSET: u16 = 0x2001;
/// Op `SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE` ([`RgfxComputeOpSetBufferOffsetAttributeStride`]):
/// `setBufferOffset:attributeStride:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE: u16 = 0x2002;
/// Op `SET_BUFFERS` ([`RgfxComputeOpSetBuffers`] then [`RgfxComputeBufferBinding`] records):
/// `setBuffers:offsets:withRange:`.
pub const RGFX_OP_COMPUTE_SET_BUFFERS: u16 = 0x2003;
/// Op `SET_BUFFERS_ATTRIBUTE_STRIDES` ([`RgfxComputeOpSetBuffersAttributeStrides`] then
/// [`RgfxComputeBufferStrideBinding`] records):
/// `setBuffers:offsets:attributeStrides:withRange:`.
pub const RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES: u16 = 0x2004;
/// Op `SET_TEXTURE` ([`RgfxComputeOpSetTexture`]): `setTexture:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_TEXTURE: u16 = 0x2005;
/// Op `SET_TEXTURES` ([`RgfxComputeOpSetTextures`] then [`RgfxComputeTextureBinding`] records):
/// `setTextures:withRange:`.
pub const RGFX_OP_COMPUTE_SET_TEXTURES: u16 = 0x2006;
/// Op `SET_SAMPLER_STATES` ([`RgfxComputeOpSetSamplerStates`] then
/// [`RgfxComputeSamplerBinding`] records): `setSamplerStates:withRange:`.
pub const RGFX_OP_COMPUTE_SET_SAMPLER_STATES: u16 = 0x2007;
/// Op `SET_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxComputeOpSetSamplerStatesLodClamps`] then
/// [`RgfxComputeSamplerLodBinding`] records):
/// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x2008;
/// Op `SET_THREADGROUP_MEMORY_LENGTH` ([`RgfxComputeOpSetThreadgroupMemoryLength`]):
/// `setThreadgroupMemoryLength:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH: u16 = 0x2009;
/// Op `SET_STAGE_IN_REGION` ([`RgfxComputeOpSetStageInRegion`]): `setStageInRegion:`.
pub const RGFX_OP_COMPUTE_SET_STAGE_IN_REGION: u16 = 0x200a;
/// Op `SET_STAGE_IN_REGION_INDIRECT` ([`RgfxComputeOpSetStageInRegionIndirect`]):
/// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT: u16 = 0x200b;
/// Op `SET_IMAGEBLOCK_WIDTH` ([`RgfxComputeOpSetImageblockWidth`]): `setImageblockWidth:height:`.
pub const RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH: u16 = 0x200c;
/// Op `DISPATCH_THREADGROUPS` ([`RgfxComputeOpDispatchThreadgroups`]):
/// `dispatchThreadgroups:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS: u16 = 0x200d;
/// Op `DISPATCH_THREADGROUPS_INDIRECT` ([`RgfxComputeOpDispatchThreadgroupsIndirect`]):
/// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT: u16 = 0x200e;
/// Op `DISPATCH_THREADS` ([`RgfxComputeOpDispatchThreads`]):
/// `dispatchThreads:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADS: u16 = 0x200f;
/// Op `DISPATCH_THREADS_INDIRECT` ([`RgfxComputeOpDispatchThreadsIndirect`]):
/// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT: u16 = 0x2010;
/// Op `USE_RESOURCES` ([`RgfxComputeOpUseResources`] then [`RgfxComputeResourceRef`] records):
/// `useResources:count:usage:`.
pub const RGFX_OP_COMPUTE_USE_RESOURCES: u16 = 0x2011;
/// Op `USE_HEAPS` ([`RgfxComputeOpUseHeaps`] then [`RgfxComputeHeapBinding`] records):
/// `useHeaps:count:`.
pub const RGFX_OP_COMPUTE_USE_HEAPS: u16 = 0x2012;
/// Op `MEMORY_BARRIER_RESOURCES` ([`RgfxComputeOpMemoryBarrierResources`] then
/// [`RgfxComputeResourceRef`] records): `memoryBarrierWithResources:count:`.
pub const RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES: u16 = 0x2013;
/// Op `MEMORY_BARRIER_SCOPE` ([`RgfxComputeOpMemoryBarrierScope`]): `memoryBarrierWithScope:`.
pub const RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE: u16 = 0x2014;
/// Op `UPDATE_FENCE` ([`RgfxComputeOpUpdateFence`]): `updateFence:`.
pub const RGFX_OP_COMPUTE_UPDATE_FENCE: u16 = 0x2015;
/// Op `WAIT_FOR_FENCE` ([`RgfxComputeOpWaitForFence`]): `waitForFence:`.
pub const RGFX_OP_COMPUTE_WAIT_FOR_FENCE: u16 = 0x2016;
/// Op `EXECUTE_COMMANDS_INDIRECT` ([`RgfxComputeOpExecuteCommandsIndirect`]):
/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT: u16 = 0x2017;
/// Op `EXECUTE_COMMANDS_IN_BUFFER` ([`RgfxComputeOpExecuteCommandsInBuffer`]):
/// `executeCommandsInBuffer:withRange:`.
pub const RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER: u16 = 0x2018;
/// Op `ENCODE_START_IF` ([`RgfxComputeOpEncodeStartIf`]):
/// `encodeStartIf:offset:comparison:referenceValue:` (SPI).
pub const RGFX_OP_COMPUTE_ENCODE_START_IF: u16 = 0x2019;
/// Op `ENCODE_START_WHILE` ([`RgfxComputeOpEncodeStartWhile`]):
/// `encodeStartWhile:offset:comparison:referenceValue:` (SPI).
pub const RGFX_OP_COMPUTE_ENCODE_START_WHILE: u16 = 0x201a;
/// Op `ENCODE_START_ELSE`: `encodeStartElse` (SPI). No body.
pub const RGFX_OP_COMPUTE_ENCODE_START_ELSE: u16 = 0x201b;
/// Op `ENCODE_START_DO_WHILE`: `encodeStartDoWhile` (SPI). No body.
pub const RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE: u16 = 0x201c;
/// Op `COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH`:
/// `insertCompressedTextureReinterpretationFlush` (SPI). No body.
pub const RGFX_OP_COMPUTE_COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH: u16 = 0x201d;

// A resource reference's kind, for the records of `useResources:` and
// `memoryBarrierWithResources:`: the host resolves `obj_id` through the resolver
// the kind names, so one array can mix buffers, textures and indirect command
// buffers, as `id<MTLResource>` does.

// The resource-kind tags are shared across the encoder groups (defined once in the parent
// module) so the guest-facing ABI has a single numbering.
pub use super::{
    RGFX_RESOURCE_KIND_BUFFER, RGFX_RESOURCE_KIND_ICB, RGFX_RESOURCE_KIND_NONE,
    RGFX_RESOURCE_KIND_TEXTURE,
};

/// `setComputePipelineState:`. The pipeline may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetComputePipelineState {
    /// The compute pipeline state, or 0 for nil.
    pub pipeline_obj_id: u64,
}

/// `setBufferOffset:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetBufferOffset {
    /// The new offset into the bound buffer.
    pub offset: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setBufferOffset:attributeStride:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetBufferOffsetAttributeStride {
    /// The new offset into the bound buffer.
    pub offset: u64,
    /// The attribute stride.
    pub stride: u64,
    /// The buffer binding index.
    pub index: u64,
}

/// `setBuffers:offsets:withRange:` head. `length` [`RgfxComputeBufferBinding`]
/// records follow at [`size_of`]`::<RgfxComputeOpSetBuffers>()`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetBuffers {
    /// The first binding index (the range's `location`).
    pub location: u64,
    /// How many bindings follow (the range's `length`).
    pub length: u64,
}

/// One `setBuffers:offsets:withRange:` binding.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeBufferBinding {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
}

/// `setBuffers:offsets:attributeStrides:withRange:` head. `length`
/// [`RgfxComputeBufferStrideBinding`] records follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetBuffersAttributeStrides {
    /// The first binding index (the range's `location`).
    pub location: u64,
    /// How many bindings follow (the range's `length`).
    pub length: u64,
}

/// One `setBuffers:offsets:attributeStrides:withRange:` binding.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeBufferStrideBinding {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The attribute stride.
    pub stride: u64,
}

/// `setTexture:atIndex:`. The texture may be nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetTexture {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
    /// The texture binding index.
    pub index: u64,
}

/// `setTextures:withRange:` head. `length` [`RgfxComputeTextureBinding`] records
/// follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetTextures {
    /// The first binding index (the range's `location`).
    pub location: u64,
    /// How many bindings follow (the range's `length`).
    pub length: u64,
}

/// One `setTextures:withRange:` binding.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeTextureBinding {
    /// The texture, or 0 for nil.
    pub texture_obj_id: u64,
}

/// `setSamplerStates:withRange:` head. `length` [`RgfxComputeSamplerBinding`]
/// records follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetSamplerStates {
    /// The first binding index (the range's `location`).
    pub location: u64,
    /// How many bindings follow (the range's `length`).
    pub length: u64,
}

/// One `setSamplerStates:withRange:` binding.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeSamplerBinding {
    /// The sampler, or 0 for nil.
    pub sampler_obj_id: u64,
}

/// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` head. `length`
/// [`RgfxComputeSamplerLodBinding`] records follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetSamplerStatesLodClamps {
    /// The first binding index (the range's `location`).
    pub location: u64,
    /// How many bindings follow (the range's `length`).
    pub length: u64,
}

/// One `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` binding. The LOD
/// clamps travel as the bits of their `f32`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeSamplerLodBinding {
    /// The sampler, or 0 for nil.
    pub sampler_obj_id: u64,
    /// `lodMinClamp` as `f32` bits.
    pub lod_min_clamp_bits: u32,
    /// `lodMaxClamp` as `f32` bits.
    pub lod_max_clamp_bits: u32,
}

/// `setThreadgroupMemoryLength:atIndex:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetThreadgroupMemoryLength {
    /// The threadgroup memory length in bytes.
    pub length: u64,
    /// The threadgroup memory binding index.
    pub index: u64,
}

/// `setStageInRegion:`. The `MTLRegion`'s six fields are `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetStageInRegion {
    /// `origin.x`.
    pub origin_x: u64,
    /// `origin.y`.
    pub origin_y: u64,
    /// `origin.z`.
    pub origin_z: u64,
    /// `size.width`.
    pub width: u64,
    /// `size.height`.
    pub height: u64,
    /// `size.depth`.
    pub depth: u64,
}

/// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`. The buffer must
/// name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetStageInRegionIndirect {
    /// The indirect buffer (must not be nil).
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `setImageblockWidth:height:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpSetImageblockWidth {
    /// The imageblock width in pixels.
    pub width: u64,
    /// The imageblock height in pixels.
    pub height: u64,
}

/// `dispatchThreadgroups:threadsPerThreadgroup:`. Each `MTLSize` is three
/// `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpDispatchThreadgroups {
    /// `threadgroupsPerGrid.width`.
    pub threadgroups_width: u64,
    /// `threadgroupsPerGrid.height`.
    pub threadgroups_height: u64,
    /// `threadgroupsPerGrid.depth`.
    pub threadgroups_depth: u64,
    /// `threadsPerThreadgroup.width`.
    pub threads_width: u64,
    /// `threadsPerThreadgroup.height`.
    pub threads_height: u64,
    /// `threadsPerThreadgroup.depth`.
    pub threads_depth: u64,
}

/// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`.
/// The indirect buffer may be nil (`0`); the platform aborts on a nil buffer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpDispatchThreadgroupsIndirect {
    /// The indirect buffer, or 0 for nil.
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
    /// `threadsPerThreadgroup.width`.
    pub threads_width: u64,
    /// `threadsPerThreadgroup.height`.
    pub threads_height: u64,
    /// `threadsPerThreadgroup.depth`.
    pub threads_depth: u64,
}

/// `dispatchThreads:threadsPerThreadgroup:`. Each `MTLSize` is three `NSUInteger`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpDispatchThreads {
    /// `threadsPerGrid.width`.
    pub grid_width: u64,
    /// `threadsPerGrid.height`.
    pub grid_height: u64,
    /// `threadsPerGrid.depth`.
    pub grid_depth: u64,
    /// `threadsPerThreadgroup.width`.
    pub threads_width: u64,
    /// `threadsPerThreadgroup.height`.
    pub threads_height: u64,
    /// `threadsPerThreadgroup.depth`.
    pub threads_depth: u64,
}

/// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` (SPI). The indirect
/// buffer may be nil (`0`); the platform aborts on a nil buffer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpDispatchThreadsIndirect {
    /// The indirect buffer, or 0 for nil.
    pub indirect_buffer_obj_id: u64,
    /// The offset into the indirect buffer.
    pub indirect_buffer_offset: u64,
}

/// `useResources:count:usage:` head. `count` [`RgfxComputeResourceRef`] records
/// follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpUseResources {
    /// How many resource records follow.
    pub count: u64,
    /// The `MTLResourceUsage`.
    pub usage: u64,
}

/// One resource reference of `useResources:count:usage:` or
/// `memoryBarrierWithResources:count:`. `kind` picks the resolver `obj_id` is
/// resolved through (`RGFX_RESOURCE_KIND_*`); `RGFX_RESOURCE_KIND_NONE` is a nil
/// entry.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeResourceRef {
    /// The kind of resource (`RGFX_RESOURCE_KIND_*`).
    pub kind: u32,
    /// Padding, 0.
    pub reserved_04: u32,
    /// The resource, resolved through the kind's resolver.
    pub obj_id: u64,
}

/// `useHeaps:count:` head. `count` [`RgfxComputeHeapBinding`] records follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpUseHeaps {
    /// How many heap records follow.
    pub count: u64,
}

/// One `useHeaps:count:` heap.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeHeapBinding {
    /// The heap, or 0 for nil.
    pub heap_obj_id: u64,
}

/// `memoryBarrierWithResources:count:` head. `count` [`RgfxComputeResourceRef`]
/// records follow.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpMemoryBarrierResources {
    /// How many resource records follow.
    pub count: u64,
}

/// `memoryBarrierWithScope:`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpMemoryBarrierScope {
    /// The `MTLBarrierScope`.
    pub scope: u64,
}

/// `updateFence:`. The fence must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpUpdateFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
}

/// `waitForFence:`. The fence must name an object (nil faults the stream).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpWaitForFence {
    /// The fence (must not be nil).
    pub fence_obj_id: u64,
}

/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`. Both the
/// indirect command buffer and the range buffer must name an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpExecuteCommandsIndirect {
    /// The indirect command buffer (must not be nil).
    pub indirect_commandbuffer_obj_id: u64,
    /// The buffer holding the execution range (must not be nil).
    pub indirect_range_buffer_obj_id: u64,
    /// The offset into the range buffer.
    pub indirect_buffer_offset: u64,
}

/// `executeCommandsInBuffer:withRange:`. The indirect command buffer must name
/// an object.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpExecuteCommandsInBuffer {
    /// The indirect command buffer (must not be nil).
    pub indirect_command_buffer_obj_id: u64,
    /// The execution range's `location`.
    pub location: u64,
    /// The execution range's `length`.
    pub length: u64,
}

/// `encodeStartIf:offset:comparison:referenceValue:` (SPI). The buffer may be
/// nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpEncodeStartIf {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The `MTLComparisonFunction`.
    pub comparison: u64,
    /// The reference value.
    pub reference_value: u32,
    /// Padding, 0.
    pub reserved_1c: u32,
}

/// `encodeStartWhile:offset:comparison:referenceValue:` (SPI). The buffer may be
/// nil (`0`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputeOpEncodeStartWhile {
    /// The buffer, or 0 for nil.
    pub buffer_obj_id: u64,
    /// The offset into the buffer.
    pub offset: u64,
    /// The `MTLComparisonFunction`.
    pub comparison: u64,
    /// The reference value.
    pub reference_value: u32,
    /// Padding, 0.
    pub reserved_1c: u32,
}

// The compute ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxComputeOpSetComputePipelineState>() == 8);
    assert!(size_of::<RgfxComputeOpSetBufferOffset>() == 16);
    assert!(size_of::<RgfxComputeOpSetBufferOffsetAttributeStride>() == 24);
    assert!(size_of::<RgfxComputeOpSetBuffers>() == 16);
    assert!(size_of::<RgfxComputeBufferBinding>() == 16);
    assert!(size_of::<RgfxComputeOpSetBuffersAttributeStrides>() == 16);
    assert!(size_of::<RgfxComputeBufferStrideBinding>() == 24);
    assert!(size_of::<RgfxComputeOpSetTexture>() == 16);
    assert!(size_of::<RgfxComputeOpSetTextures>() == 16);
    assert!(size_of::<RgfxComputeTextureBinding>() == 8);
    assert!(size_of::<RgfxComputeOpSetSamplerStates>() == 16);
    assert!(size_of::<RgfxComputeSamplerBinding>() == 8);
    assert!(size_of::<RgfxComputeOpSetSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxComputeSamplerLodBinding>() == 16);
    assert!(size_of::<RgfxComputeOpSetThreadgroupMemoryLength>() == 16);
    assert!(size_of::<RgfxComputeOpSetStageInRegion>() == 48);
    assert!(size_of::<RgfxComputeOpSetStageInRegionIndirect>() == 16);
    assert!(size_of::<RgfxComputeOpSetImageblockWidth>() == 16);
    assert!(size_of::<RgfxComputeOpDispatchThreadgroups>() == 48);
    assert!(size_of::<RgfxComputeOpDispatchThreadgroupsIndirect>() == 40);
    assert!(size_of::<RgfxComputeOpDispatchThreads>() == 48);
    assert!(size_of::<RgfxComputeOpDispatchThreadsIndirect>() == 16);
    assert!(size_of::<RgfxComputeOpUseResources>() == 16);
    assert!(size_of::<RgfxComputeResourceRef>() == 16);
    assert!(size_of::<RgfxComputeOpUseHeaps>() == 8);
    assert!(size_of::<RgfxComputeHeapBinding>() == 8);
    assert!(size_of::<RgfxComputeOpMemoryBarrierResources>() == 8);
    assert!(size_of::<RgfxComputeOpMemoryBarrierScope>() == 8);
    assert!(size_of::<RgfxComputeOpUpdateFence>() == 8);
    assert!(size_of::<RgfxComputeOpWaitForFence>() == 8);
    assert!(size_of::<RgfxComputeOpExecuteCommandsIndirect>() == 24);
    assert!(size_of::<RgfxComputeOpExecuteCommandsInBuffer>() == 24);
    assert!(size_of::<RgfxComputeOpEncodeStartIf>() == 32);
    assert!(size_of::<RgfxComputeOpEncodeStartWhile>() == 32);
    // The compute tags stay in the compute group.
    assert!(
        super::op_group(RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE) == super::RGFX_OP_GROUP_COMPUTE
    );
    assert!(
        super::op_group(RGFX_OP_COMPUTE_COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH)
            == super::RGFX_OP_GROUP_COMPUTE
    );
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// The scalar and region op bodies round-trip, floats travelling as bits.
    #[test]
    fn scalar_ops_round_trip() {
        let p = RgfxComputeOpSetComputePipelineState { pipeline_obj_id: 7 };
        assert_eq!(
            RgfxComputeOpSetComputePipelineState::read_from_bytes(p.as_bytes()).unwrap(),
            p
        );

        let d = RgfxComputeOpDispatchThreadgroups {
            threadgroups_width: 16,
            threadgroups_height: 8,
            threadgroups_depth: 1,
            threads_width: 32,
            threads_height: 1,
            threads_depth: 1,
        };
        assert_eq!(
            RgfxComputeOpDispatchThreadgroups::read_from_bytes(d.as_bytes()).unwrap(),
            d
        );

        let r = RgfxComputeOpSetStageInRegion {
            origin_x: 0,
            origin_y: 1,
            origin_z: 2,
            width: 64,
            height: 64,
            depth: 1,
        };
        assert_eq!(
            RgfxComputeOpSetStageInRegion::read_from_bytes(r.as_bytes()).unwrap(),
            r
        );

        let sif = RgfxComputeOpEncodeStartIf {
            buffer_obj_id: 4,
            offset: 16,
            comparison: 3,
            reference_value: 42,
            reserved_1c: 0,
        };
        assert_eq!(
            RgfxComputeOpEncodeStartIf::read_from_bytes(sif.as_bytes()).unwrap(),
            sif
        );
    }

    /// The array-tailed op bodies and their element records round-trip, the LOD
    /// clamps travelling as bits.
    #[test]
    fn array_tailed_ops_round_trip() {
        let head = RgfxComputeOpSetBuffers {
            location: 2,
            length: 3,
        };
        assert_eq!(
            RgfxComputeOpSetBuffers::read_from_bytes(head.as_bytes()).unwrap(),
            head
        );

        let bind = RgfxComputeBufferBinding {
            buffer_obj_id: 9,
            offset: 256,
        };
        assert_eq!(
            RgfxComputeBufferBinding::read_from_bytes(bind.as_bytes()).unwrap(),
            bind
        );

        let lod = RgfxComputeSamplerLodBinding {
            sampler_obj_id: 5,
            lod_min_clamp_bits: 0.0f32.to_bits(),
            lod_max_clamp_bits: 8.0f32.to_bits(),
        };
        assert_eq!(
            RgfxComputeSamplerLodBinding::read_from_bytes(lod.as_bytes()).unwrap(),
            lod
        );
        assert_eq!(f32::from_bits(lod.lod_max_clamp_bits), 8.0);

        let res = RgfxComputeResourceRef {
            kind: RGFX_RESOURCE_KIND_TEXTURE,
            reserved_04: 0,
            obj_id: 12,
        };
        assert_eq!(
            RgfxComputeResourceRef::read_from_bytes(res.as_bytes()).unwrap(),
            res
        );
    }
}
