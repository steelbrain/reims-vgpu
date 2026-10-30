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

/// The standard derives every op body and element record carries.
macro_rules! op_body {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident : $t:ty),* $(,)? }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(
            Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
        )]
        pub struct $name {
            $($(#[$fm])* pub $f : $t),*
        }
    };
}

// The op tags, in the compute group's `0x2xxx` range.

/// Op `SET_COMPUTE_PIPELINE_STATE` ([`RgfxOpSetComputePipelineState`]):
/// `setComputePipelineState:`.
pub const RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE: u16 = 0x2000;
/// Op `SET_BUFFER_OFFSET` ([`RgfxOpSetBufferOffset`]): `setBufferOffset:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_BUFFER_OFFSET: u16 = 0x2001;
/// Op `SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE` ([`RgfxOpSetBufferOffsetAttributeStride`]):
/// `setBufferOffset:attributeStride:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE: u16 = 0x2002;
/// Op `SET_BUFFERS` ([`RgfxOpSetBuffers`] then [`RgfxComputeBufferBinding`] records):
/// `setBuffers:offsets:withRange:`.
pub const RGFX_OP_COMPUTE_SET_BUFFERS: u16 = 0x2003;
/// Op `SET_BUFFERS_ATTRIBUTE_STRIDES` ([`RgfxOpSetBuffersAttributeStrides`] then
/// [`RgfxComputeBufferStrideBinding`] records):
/// `setBuffers:offsets:attributeStrides:withRange:`.
pub const RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES: u16 = 0x2004;
/// Op `SET_TEXTURE` ([`RgfxOpSetTexture`]): `setTexture:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_TEXTURE: u16 = 0x2005;
/// Op `SET_TEXTURES` ([`RgfxOpSetTextures`] then [`RgfxComputeTextureBinding`] records):
/// `setTextures:withRange:`.
pub const RGFX_OP_COMPUTE_SET_TEXTURES: u16 = 0x2006;
/// Op `SET_SAMPLER_STATES` ([`RgfxOpSetSamplerStates`] then
/// [`RgfxComputeSamplerBinding`] records): `setSamplerStates:withRange:`.
pub const RGFX_OP_COMPUTE_SET_SAMPLER_STATES: u16 = 0x2007;
/// Op `SET_SAMPLER_STATES_LOD_CLAMPS` ([`RgfxOpSetSamplerStatesLodClamps`] then
/// [`RgfxComputeSamplerLodBinding`] records):
/// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:`.
pub const RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS: u16 = 0x2008;
/// Op `SET_THREADGROUP_MEMORY_LENGTH` ([`RgfxOpSetThreadgroupMemoryLength`]):
/// `setThreadgroupMemoryLength:atIndex:`.
pub const RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH: u16 = 0x2009;
/// Op `SET_STAGE_IN_REGION` ([`RgfxOpSetStageInRegion`]): `setStageInRegion:`.
pub const RGFX_OP_COMPUTE_SET_STAGE_IN_REGION: u16 = 0x200a;
/// Op `SET_STAGE_IN_REGION_INDIRECT` ([`RgfxOpSetStageInRegionIndirect`]):
/// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT: u16 = 0x200b;
/// Op `SET_IMAGEBLOCK_WIDTH` ([`RgfxOpSetImageblockWidth`]): `setImageblockWidth:height:`.
pub const RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH: u16 = 0x200c;
/// Op `DISPATCH_THREADGROUPS` ([`RgfxOpDispatchThreadgroups`]):
/// `dispatchThreadgroups:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS: u16 = 0x200d;
/// Op `DISPATCH_THREADGROUPS_INDIRECT` ([`RgfxOpDispatchThreadgroupsIndirect`]):
/// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT: u16 = 0x200e;
/// Op `DISPATCH_THREADS` ([`RgfxOpDispatchThreads`]):
/// `dispatchThreads:threadsPerThreadgroup:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADS: u16 = 0x200f;
/// Op `DISPATCH_THREADS_INDIRECT` ([`RgfxOpDispatchThreadsIndirect`]):
/// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT: u16 = 0x2010;
/// Op `USE_RESOURCES` ([`RgfxOpUseResources`] then [`RgfxComputeResourceRef`] records):
/// `useResources:count:usage:`.
pub const RGFX_OP_COMPUTE_USE_RESOURCES: u16 = 0x2011;
/// Op `USE_HEAPS` ([`RgfxOpUseHeaps`] then [`RgfxComputeHeapBinding`] records):
/// `useHeaps:count:`.
pub const RGFX_OP_COMPUTE_USE_HEAPS: u16 = 0x2012;
/// Op `MEMORY_BARRIER_RESOURCES` ([`RgfxOpMemoryBarrierResources`] then
/// [`RgfxComputeResourceRef`] records): `memoryBarrierWithResources:count:`.
pub const RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES: u16 = 0x2013;
/// Op `MEMORY_BARRIER_SCOPE` ([`RgfxOpMemoryBarrierScope`]): `memoryBarrierWithScope:`.
pub const RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE: u16 = 0x2014;
/// Op `UPDATE_FENCE` ([`RgfxOpUpdateFence`]): `updateFence:`.
pub const RGFX_OP_COMPUTE_UPDATE_FENCE: u16 = 0x2015;
/// Op `WAIT_FOR_FENCE` ([`RgfxOpWaitForFence`]): `waitForFence:`.
pub const RGFX_OP_COMPUTE_WAIT_FOR_FENCE: u16 = 0x2016;
/// Op `EXECUTE_COMMANDS_INDIRECT` ([`RgfxOpExecuteCommandsIndirect`]):
/// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`.
pub const RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT: u16 = 0x2017;
/// Op `EXECUTE_COMMANDS_IN_BUFFER` ([`RgfxOpExecuteCommandsInBuffer`]):
/// `executeCommandsInBuffer:withRange:`.
pub const RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER: u16 = 0x2018;
/// Op `ENCODE_START_IF` ([`RgfxOpEncodeStartIf`]):
/// `encodeStartIf:offset:comparison:referenceValue:` (SPI).
pub const RGFX_OP_COMPUTE_ENCODE_START_IF: u16 = 0x2019;
/// Op `ENCODE_START_WHILE` ([`RgfxOpEncodeStartWhile`]):
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

op_body! {
    /// `setComputePipelineState:`. The pipeline may be nil (`0`).
    RgfxOpSetComputePipelineState {
        /// The compute pipeline state, or 0 for nil.
        pipeline_obj_id: u64,
    }
}

op_body! {
    /// `setBufferOffset:atIndex:`.
    RgfxOpSetBufferOffset {
        /// The new offset into the bound buffer.
        offset: u64,
        /// The buffer binding index.
        index: u64,
    }
}

op_body! {
    /// `setBufferOffset:attributeStride:atIndex:`.
    RgfxOpSetBufferOffsetAttributeStride {
        /// The new offset into the bound buffer.
        offset: u64,
        /// The attribute stride.
        stride: u64,
        /// The buffer binding index.
        index: u64,
    }
}

op_body! {
    /// `setBuffers:offsets:withRange:` head. `length` [`RgfxComputeBufferBinding`]
    /// records follow at [`size_of`](core::mem::size_of)`::<RgfxOpSetBuffers>()`.
    RgfxOpSetBuffers {
        /// The first binding index (the range's `location`).
        location: u64,
        /// How many bindings follow (the range's `length`).
        length: u64,
    }
}

op_body! {
    /// One `setBuffers:offsets:withRange:` binding.
    RgfxComputeBufferBinding {
        /// The buffer, or 0 for nil.
        buffer_obj_id: u64,
        /// The offset into the buffer.
        offset: u64,
    }
}

op_body! {
    /// `setBuffers:offsets:attributeStrides:withRange:` head. `length`
    /// [`RgfxComputeBufferStrideBinding`] records follow.
    RgfxOpSetBuffersAttributeStrides {
        /// The first binding index (the range's `location`).
        location: u64,
        /// How many bindings follow (the range's `length`).
        length: u64,
    }
}

op_body! {
    /// One `setBuffers:offsets:attributeStrides:withRange:` binding.
    RgfxComputeBufferStrideBinding {
        /// The buffer, or 0 for nil.
        buffer_obj_id: u64,
        /// The offset into the buffer.
        offset: u64,
        /// The attribute stride.
        stride: u64,
    }
}

op_body! {
    /// `setTexture:atIndex:`. The texture may be nil (`0`).
    RgfxOpSetTexture {
        /// The texture, or 0 for nil.
        texture_obj_id: u64,
        /// The texture binding index.
        index: u64,
    }
}

op_body! {
    /// `setTextures:withRange:` head. `length` [`RgfxComputeTextureBinding`] records
    /// follow.
    RgfxOpSetTextures {
        /// The first binding index (the range's `location`).
        location: u64,
        /// How many bindings follow (the range's `length`).
        length: u64,
    }
}

op_body! {
    /// One `setTextures:withRange:` binding.
    RgfxComputeTextureBinding {
        /// The texture, or 0 for nil.
        texture_obj_id: u64,
    }
}

op_body! {
    /// `setSamplerStates:withRange:` head. `length` [`RgfxComputeSamplerBinding`]
    /// records follow.
    RgfxOpSetSamplerStates {
        /// The first binding index (the range's `location`).
        location: u64,
        /// How many bindings follow (the range's `length`).
        length: u64,
    }
}

op_body! {
    /// One `setSamplerStates:withRange:` binding.
    RgfxComputeSamplerBinding {
        /// The sampler, or 0 for nil.
        sampler_obj_id: u64,
    }
}

op_body! {
    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` head. `length`
    /// [`RgfxComputeSamplerLodBinding`] records follow.
    RgfxOpSetSamplerStatesLodClamps {
        /// The first binding index (the range's `location`).
        location: u64,
        /// How many bindings follow (the range's `length`).
        length: u64,
    }
}

op_body! {
    /// One `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` binding. The LOD
    /// clamps travel as the bits of their `f32`.
    RgfxComputeSamplerLodBinding {
        /// The sampler, or 0 for nil.
        sampler_obj_id: u64,
        /// `lodMinClamp` as `f32` bits.
        lod_min_clamp_bits: u32,
        /// `lodMaxClamp` as `f32` bits.
        lod_max_clamp_bits: u32,
    }
}

op_body! {
    /// `setThreadgroupMemoryLength:atIndex:`.
    RgfxOpSetThreadgroupMemoryLength {
        /// The threadgroup memory length in bytes.
        length: u64,
        /// The threadgroup memory binding index.
        index: u64,
    }
}

op_body! {
    /// `setStageInRegion:`. The `MTLRegion`'s six fields are `NSUInteger`.
    RgfxOpSetStageInRegion {
        /// `origin.x`.
        origin_x: u64,
        /// `origin.y`.
        origin_y: u64,
        /// `origin.z`.
        origin_z: u64,
        /// `size.width`.
        width: u64,
        /// `size.height`.
        height: u64,
        /// `size.depth`.
        depth: u64,
    }
}

op_body! {
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`. The buffer must
    /// name an object (nil faults the stream).
    RgfxOpSetStageInRegionIndirect {
        /// The indirect buffer (must not be nil).
        indirect_buffer_obj_id: u64,
        /// The offset into the indirect buffer.
        indirect_buffer_offset: u64,
    }
}

op_body! {
    /// `setImageblockWidth:height:`.
    RgfxOpSetImageblockWidth {
        /// The imageblock width in pixels.
        width: u64,
        /// The imageblock height in pixels.
        height: u64,
    }
}

op_body! {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`. Each `MTLSize` is three
    /// `NSUInteger`.
    RgfxOpDispatchThreadgroups {
        /// `threadgroupsPerGrid.width`.
        threadgroups_width: u64,
        /// `threadgroupsPerGrid.height`.
        threadgroups_height: u64,
        /// `threadgroupsPerGrid.depth`.
        threadgroups_depth: u64,
        /// `threadsPerThreadgroup.width`.
        threads_width: u64,
        /// `threadsPerThreadgroup.height`.
        threads_height: u64,
        /// `threadsPerThreadgroup.depth`.
        threads_depth: u64,
    }
}

op_body! {
    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`.
    /// The indirect buffer may be nil (`0`); the platform aborts on a nil buffer.
    RgfxOpDispatchThreadgroupsIndirect {
        /// The indirect buffer, or 0 for nil.
        indirect_buffer_obj_id: u64,
        /// The offset into the indirect buffer.
        indirect_buffer_offset: u64,
        /// `threadsPerThreadgroup.width`.
        threads_width: u64,
        /// `threadsPerThreadgroup.height`.
        threads_height: u64,
        /// `threadsPerThreadgroup.depth`.
        threads_depth: u64,
    }
}

op_body! {
    /// `dispatchThreads:threadsPerThreadgroup:`. Each `MTLSize` is three `NSUInteger`.
    RgfxOpDispatchThreads {
        /// `threadsPerGrid.width`.
        grid_width: u64,
        /// `threadsPerGrid.height`.
        grid_height: u64,
        /// `threadsPerGrid.depth`.
        grid_depth: u64,
        /// `threadsPerThreadgroup.width`.
        threads_width: u64,
        /// `threadsPerThreadgroup.height`.
        threads_height: u64,
        /// `threadsPerThreadgroup.depth`.
        threads_depth: u64,
    }
}

op_body! {
    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` (SPI). The indirect
    /// buffer may be nil (`0`); the platform aborts on a nil buffer.
    RgfxOpDispatchThreadsIndirect {
        /// The indirect buffer, or 0 for nil.
        indirect_buffer_obj_id: u64,
        /// The offset into the indirect buffer.
        indirect_buffer_offset: u64,
    }
}

op_body! {
    /// `useResources:count:usage:` head. `count` [`RgfxComputeResourceRef`] records
    /// follow.
    RgfxOpUseResources {
        /// How many resource records follow.
        count: u64,
        /// The `MTLResourceUsage`.
        usage: u64,
    }
}

op_body! {
    /// One resource reference of `useResources:count:usage:` or
    /// `memoryBarrierWithResources:count:`. `kind` picks the resolver `obj_id` is
    /// resolved through (`RGFX_RESOURCE_KIND_*`); `RGFX_RESOURCE_KIND_NONE` is a nil
    /// entry.
    RgfxComputeResourceRef {
        /// The kind of resource (`RGFX_RESOURCE_KIND_*`).
        kind: u32,
        /// Padding, 0.
        reserved_04: u32,
        /// The resource, resolved through the kind's resolver.
        obj_id: u64,
    }
}

op_body! {
    /// `useHeaps:count:` head. `count` [`RgfxComputeHeapBinding`] records follow.
    RgfxOpUseHeaps {
        /// How many heap records follow.
        count: u64,
    }
}

op_body! {
    /// One `useHeaps:count:` heap.
    RgfxComputeHeapBinding {
        /// The heap, or 0 for nil.
        heap_obj_id: u64,
    }
}

op_body! {
    /// `memoryBarrierWithResources:count:` head. `count` [`RgfxComputeResourceRef`]
    /// records follow.
    RgfxOpMemoryBarrierResources {
        /// How many resource records follow.
        count: u64,
    }
}

op_body! {
    /// `memoryBarrierWithScope:`.
    RgfxOpMemoryBarrierScope {
        /// The `MTLBarrierScope`.
        scope: u64,
    }
}

op_body! {
    /// `updateFence:`. The fence must name an object (nil faults the stream).
    RgfxOpUpdateFence {
        /// The fence (must not be nil).
        fence_obj_id: u64,
    }
}

op_body! {
    /// `waitForFence:`. The fence must name an object (nil faults the stream).
    RgfxOpWaitForFence {
        /// The fence (must not be nil).
        fence_obj_id: u64,
    }
}

op_body! {
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`. Both the
    /// indirect command buffer and the range buffer must name an object.
    RgfxOpExecuteCommandsIndirect {
        /// The indirect command buffer (must not be nil).
        indirect_commandbuffer_obj_id: u64,
        /// The buffer holding the execution range (must not be nil).
        indirect_range_buffer_obj_id: u64,
        /// The offset into the range buffer.
        indirect_buffer_offset: u64,
    }
}

op_body! {
    /// `executeCommandsInBuffer:withRange:`. The indirect command buffer must name
    /// an object.
    RgfxOpExecuteCommandsInBuffer {
        /// The indirect command buffer (must not be nil).
        indirect_command_buffer_obj_id: u64,
        /// The execution range's `location`.
        location: u64,
        /// The execution range's `length`.
        length: u64,
    }
}

op_body! {
    /// `encodeStartIf:offset:comparison:referenceValue:` (SPI). The buffer may be
    /// nil (`0`).
    RgfxOpEncodeStartIf {
        /// The buffer, or 0 for nil.
        buffer_obj_id: u64,
        /// The offset into the buffer.
        offset: u64,
        /// The `MTLComparisonFunction`.
        comparison: u64,
        /// The reference value.
        reference_value: u32,
        /// Padding, 0.
        reserved_1c: u32,
    }
}

op_body! {
    /// `encodeStartWhile:offset:comparison:referenceValue:` (SPI). The buffer may be
    /// nil (`0`).
    RgfxOpEncodeStartWhile {
        /// The buffer, or 0 for nil.
        buffer_obj_id: u64,
        /// The offset into the buffer.
        offset: u64,
        /// The `MTLComparisonFunction`.
        comparison: u64,
        /// The reference value.
        reference_value: u32,
        /// Padding, 0.
        reserved_1c: u32,
    }
}

// The compute ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxOpSetComputePipelineState>() == 8);
    assert!(size_of::<RgfxOpSetBufferOffset>() == 16);
    assert!(size_of::<RgfxOpSetBufferOffsetAttributeStride>() == 24);
    assert!(size_of::<RgfxOpSetBuffers>() == 16);
    assert!(size_of::<RgfxComputeBufferBinding>() == 16);
    assert!(size_of::<RgfxOpSetBuffersAttributeStrides>() == 16);
    assert!(size_of::<RgfxComputeBufferStrideBinding>() == 24);
    assert!(size_of::<RgfxOpSetTexture>() == 16);
    assert!(size_of::<RgfxOpSetTextures>() == 16);
    assert!(size_of::<RgfxComputeTextureBinding>() == 8);
    assert!(size_of::<RgfxOpSetSamplerStates>() == 16);
    assert!(size_of::<RgfxComputeSamplerBinding>() == 8);
    assert!(size_of::<RgfxOpSetSamplerStatesLodClamps>() == 16);
    assert!(size_of::<RgfxComputeSamplerLodBinding>() == 16);
    assert!(size_of::<RgfxOpSetThreadgroupMemoryLength>() == 16);
    assert!(size_of::<RgfxOpSetStageInRegion>() == 48);
    assert!(size_of::<RgfxOpSetStageInRegionIndirect>() == 16);
    assert!(size_of::<RgfxOpSetImageblockWidth>() == 16);
    assert!(size_of::<RgfxOpDispatchThreadgroups>() == 48);
    assert!(size_of::<RgfxOpDispatchThreadgroupsIndirect>() == 40);
    assert!(size_of::<RgfxOpDispatchThreads>() == 48);
    assert!(size_of::<RgfxOpDispatchThreadsIndirect>() == 16);
    assert!(size_of::<RgfxOpUseResources>() == 16);
    assert!(size_of::<RgfxComputeResourceRef>() == 16);
    assert!(size_of::<RgfxOpUseHeaps>() == 8);
    assert!(size_of::<RgfxComputeHeapBinding>() == 8);
    assert!(size_of::<RgfxOpMemoryBarrierResources>() == 8);
    assert!(size_of::<RgfxOpMemoryBarrierScope>() == 8);
    assert!(size_of::<RgfxOpUpdateFence>() == 8);
    assert!(size_of::<RgfxOpWaitForFence>() == 8);
    assert!(size_of::<RgfxOpExecuteCommandsIndirect>() == 24);
    assert!(size_of::<RgfxOpExecuteCommandsInBuffer>() == 24);
    assert!(size_of::<RgfxOpEncodeStartIf>() == 32);
    assert!(size_of::<RgfxOpEncodeStartWhile>() == 32);
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
        let p = RgfxOpSetComputePipelineState { pipeline_obj_id: 7 };
        assert_eq!(
            RgfxOpSetComputePipelineState::read_from_bytes(p.as_bytes()).unwrap(),
            p
        );

        let d = RgfxOpDispatchThreadgroups {
            threadgroups_width: 16,
            threadgroups_height: 8,
            threadgroups_depth: 1,
            threads_width: 32,
            threads_height: 1,
            threads_depth: 1,
        };
        assert_eq!(
            RgfxOpDispatchThreadgroups::read_from_bytes(d.as_bytes()).unwrap(),
            d
        );

        let r = RgfxOpSetStageInRegion {
            origin_x: 0,
            origin_y: 1,
            origin_z: 2,
            width: 64,
            height: 64,
            depth: 1,
        };
        assert_eq!(
            RgfxOpSetStageInRegion::read_from_bytes(r.as_bytes()).unwrap(),
            r
        );

        let sif = RgfxOpEncodeStartIf {
            buffer_obj_id: 4,
            offset: 16,
            comparison: 3,
            reference_value: 42,
            reserved_1c: 0,
        };
        assert_eq!(
            RgfxOpEncodeStartIf::read_from_bytes(sif.as_bytes()).unwrap(),
            sif
        );
    }

    /// The array-tailed op bodies and their element records round-trip, the LOD
    /// clamps travelling as bits.
    #[test]
    fn array_tailed_ops_round_trip() {
        let head = RgfxOpSetBuffers {
            location: 2,
            length: 3,
        };
        assert_eq!(
            RgfxOpSetBuffers::read_from_bytes(head.as_bytes()).unwrap(),
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
