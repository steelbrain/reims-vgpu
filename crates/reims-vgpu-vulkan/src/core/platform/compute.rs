//! Platform trait `ComputeEncoder`: every `MTLComputeCommandEncoder` selector core sends (D4).
//!
//! `ComputeCmd::selector` names each command's selector. `ComputePipeline`'s own selectors are in
//! `platform/resource.rs`.
//!
//! Conventions (architecture 4.2): every method takes arguments or encodes, so every one returns
//! `RaiseResult<T>`; an object argument is `Option<&P::X>` unless every caller is shown never to
//! pass nil, and the `nil:` line of such a method says why. A C array passed with a count or a
//! range is a slice whose length is that count or `range.length`; its elements are `Option` because
//! they are filled from lookups that return nil.

// One method per selector and one argument per keyword (D4).
#![allow(clippy::too_many_arguments)]

use super::blit::ResourceRef;
use super::types::{MtlRegion, MtlSize, NsRange};
use super::{Handle, Platform};
use crate::core::raise::{OnFault, RaiseResult};

/// One `MTLComputeCommandEncoder` command.
///
/// One variant per selector, whose fields are the selector's arguments in order;
/// [`ComputeEncoder::encode`] sends one.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the selector's arguments
pub enum ComputeCmd<'a, P: Platform> {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`
    DispatchThreadgroups {
        threadgroups_per_grid: MtlSize,
        threads_per_threadgroup: MtlSize,
    },
    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`
    /// aborts: a nil indirect buffer (IOGPUResourceListAddResource:
    /// compute_death_threadgroups_indirect_nil,
    /// interplay_decodewalk_23, _52); AGX's compute context on some indirect dispatches
    /// (interplay_decodewalk_27)
    DispatchThreadgroupsIndirect {
        indirect_buffer: Option<&'a P::Buffer>,
        indirect_buffer_offset: u64,
        threads_per_threadgroup: MtlSize,
    },
    /// `dispatchThreads:threadsPerThreadgroup:`
    DispatchThreads {
        threads_per_grid: MtlSize,
        threads_per_threadgroup: MtlSize,
    },
    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` (SPI)
    /// aborts: a nil indirect buffer (IOGPUResourceListAddResource:
    /// compute_death_threads_indirect_nil)
    DispatchThreadsIndirect {
        indirect_buffer: Option<&'a P::Buffer>,
        indirect_buffer_offset: u64,
    },
    /// `encodeStartDoWhile` (SPI)
    EncodeStartDoWhile,
    /// `encodeStartElse` (SPI)
    EncodeStartElse,
    /// `encodeStartIf:offset:comparison:referenceValue:` (SPI)
    EncodeStartIf {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    },
    /// `encodeStartWhile:offset:comparison:referenceValue:` (SPI)
    EncodeStartWhile {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    },
    /// `endEncoding`
    EndEncoding,
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`
    /// nil: `indirect_commandbuffer`,
    /// `indirect_range_buffer` — never nil: the decoder raises on either nil first.
    ExecuteCommandsIndirect {
        indirect_commandbuffer: &'a P::Icb,
        indirect_range_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `executeCommandsInBuffer:withRange:`
    /// nil: `indirect_command_buffer` — never nil:
    /// Core raises on nil first ("no indirect command buffer for reference …").
    ExecuteCommandsInBuffer {
        indirect_command_buffer: &'a P::Icb,
        execution_range: NsRange,
    },
    /// `insertCompressedTextureReinterpretationFlush` (SPI)
    CompressedTextureReinterpretationFlush,
    /// `memoryBarrierWithResources:count:`
    MemoryBarrierWithResources {
        resources: &'a [Option<ResourceRef<'a, P>>],
    },
    /// `memoryBarrierWithScope:`
    MemoryBarrierWithScope { scope: u64 },
    /// `setBufferOffset:atIndex:`
    SetBufferOffset { offset: u64, index: u64 },
    /// `setBufferOffset:attributeStride:atIndex:`
    SetBufferOffsetAttributeStride {
        offset: u64,
        stride: u64,
        index: u64,
    },
    /// `setBuffers:offsets:attributeStrides:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`. `strides`: its length is `range.length`.
    SetBuffersAttributeStrides {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        strides: &'a [u64],
        range: NsRange,
    },
    /// `setBuffers:offsets:withRange:`
    /// `buffers`: its length is `range.length`.
    /// `offsets`: its length is `range.length`.
    SetBuffers {
        buffers: &'a [Option<&'a P::Buffer>],
        offsets: &'a [u64],
        range: NsRange,
    },
    /// `setComputePipelineState:`
    SetComputePipelineState {
        state: Option<&'a P::ComputePipeline>,
    },
    /// `setImageblockWidth:height:`
    SetImageblockWidth { width: u64, height: u64 },
    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:`
    /// `samplers`: its length is `range.length`.
    /// `lod_min_clamps`: its length is `range.length`. `lod_max_clamps`: its length is
    /// `range.length`.
    SetSamplerStatesLodClamps {
        samplers: &'a [Option<&'a P::SamplerState>],
        lod_min_clamps: &'a [f32],
        lod_max_clamps: &'a [f32],
        range: NsRange,
    },
    /// `setSamplerStates:withRange:`
    /// `samplers`: its length is `range.length`.
    SetSamplerStates {
        samplers: &'a [Option<&'a P::SamplerState>],
        range: NsRange,
    },
    /// `setStageInRegion:`
    SetStageInRegion { region: MtlRegion },
    /// `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`
    /// nil: `indirect_buffer` — never nil: core
    /// raises on nil first ("no indirect buffer for reference …").
    SetStageInRegionIndirect {
        indirect_buffer: &'a P::Buffer,
        indirect_buffer_offset: u64,
    },
    /// `setTexture:atIndex:`
    /// nil: kept `Option`; the dual-plane paging binds `_planeTextures[i]`, which `-discard` sets
    /// to nil
    SetTexture {
        texture: Option<&'a P::Texture>,
        index: u64,
    },
    /// `setTextures:withRange:`
    /// `textures`: its length is `range.length`.
    SetTextures {
        textures: &'a [Option<&'a P::Texture>],
        range: NsRange,
    },
    /// `setThreadgroupMemoryLength:atIndex:`
    SetThreadgroupMemoryLength { length: u64, index: u64 },
    /// `updateFence:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    UpdateFence { fence: &'a P::Fence },
    /// `useHeaps:count:`
    UseHeaps { heaps: &'a [Option<&'a P::Heap>] },
    /// `useResources:count:usage:`
    /// aborts: a nil entry (AGX `useResources:count:usage:`: interplay_decodewalk_26)
    UseResources {
        resources: &'a [Option<ResourceRef<'a, P>>],
        usage: u64,
    },
    /// `waitForFence:`
    /// nil: `fence` — never nil: it comes from `getFenceForReferenceNonNull:`, which
    /// throws on nil.
    WaitForFence { fence: &'a P::Fence },
}

impl<P: Platform> ComputeCmd<'_, P> {
    /// The selector this sends.
    pub fn selector(&self) -> &'static str {
        match self {
            Self::DispatchThreadgroups { .. } => "dispatchThreadgroups:threadsPerThreadgroup:",
            Self::DispatchThreadgroupsIndirect { .. } => {
                "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"
            }
            Self::DispatchThreads { .. } => "dispatchThreads:threadsPerThreadgroup:",
            Self::DispatchThreadsIndirect { .. } => {
                "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:"
            }
            Self::EncodeStartDoWhile => "encodeStartDoWhile",
            Self::EncodeStartElse => "encodeStartElse",
            Self::EncodeStartIf { .. } => "encodeStartIf:offset:comparison:referenceValue:",
            Self::EncodeStartWhile { .. } => "encodeStartWhile:offset:comparison:referenceValue:",
            Self::EndEncoding => "endEncoding",
            Self::ExecuteCommandsIndirect { .. } => {
                "executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:"
            }
            Self::ExecuteCommandsInBuffer { .. } => "executeCommandsInBuffer:withRange:",
            Self::CompressedTextureReinterpretationFlush => {
                "insertCompressedTextureReinterpretationFlush"
            }
            Self::MemoryBarrierWithResources { .. } => "memoryBarrierWithResources:count:",
            Self::MemoryBarrierWithScope { .. } => "memoryBarrierWithScope:",
            Self::SetBufferOffset { .. } => "setBufferOffset:atIndex:",
            Self::SetBufferOffsetAttributeStride { .. } => {
                "setBufferOffset:attributeStride:atIndex:"
            }
            Self::SetBuffersAttributeStrides { .. } => {
                "setBuffers:offsets:attributeStrides:withRange:"
            }
            Self::SetBuffers { .. } => "setBuffers:offsets:withRange:",
            Self::SetComputePipelineState { .. } => "setComputePipelineState:",
            Self::SetImageblockWidth { .. } => "setImageblockWidth:height:",
            Self::SetSamplerStatesLodClamps { .. } => {
                "setSamplerStates:lodMinClamps:lodMaxClamps:withRange:"
            }
            Self::SetSamplerStates { .. } => "setSamplerStates:withRange:",
            Self::SetStageInRegion { .. } => "setStageInRegion:",
            Self::SetStageInRegionIndirect { .. } => {
                "setStageInRegionWithIndirectBuffer:indirectBufferOffset:"
            }
            Self::SetTexture { .. } => "setTexture:atIndex:",
            Self::SetTextures { .. } => "setTextures:withRange:",
            Self::SetThreadgroupMemoryLength { .. } => "setThreadgroupMemoryLength:atIndex:",
            Self::UpdateFence { .. } => "updateFence:",
            Self::UseHeaps { .. } => "useHeaps:count:",
            Self::UseResources { .. } => "useResources:count:usage:",
            Self::WaitForFence { .. } => "waitForFence:",
        }
    }
}
/// The compute encoder core gets from `-computeCommandEncoder` or
/// `-computeCommandEncoderWithDescriptor:` (`CommandBuffer`).
pub trait ComputeEncoder<P: Platform>: Handle {
    /// Sends one [`ComputeCmd`]. `on` says what a raise in a command reached both under a
    /// handler and with none (`EndEncoding`) does; the other commands are reached one way only.
    fn encode(&self, on: OnFault, cmd: ComputeCmd<'_, P>) -> RaiseResult<()>;
    /// sel: encodeEndDoWhile:offset:comparison:referenceValue: SPI
    /// Declared on MTLComputeCommandEncoderSPI (protocol).
    fn encode_end_do_while(
        &self,
        buffer: Option<&P::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> RaiseResult<bool>;
    /// sel: encodeEndIf SPI
    /// Declared on MTLComputeCommandEncoderSPI (protocol).
    fn encode_end_if(&self) -> RaiseResult<bool>;
    /// sel: encodeEndWhile SPI
    /// Declared on MTLComputeCommandEncoderSPI (protocol).
    fn encode_end_while(&self) -> RaiseResult<bool>;
}
