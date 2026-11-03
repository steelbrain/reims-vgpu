//! Platform traits for indirect command buffers: `IndirectCommandBuffer`, `IndirectRenderCommand`
//! and `IndirectComputeCommand`, and the values the commands' SPI getters read whole from Metal's
//! private `_MTLIndirect*Arguments` and `_MTLImageBlockArguments` objects.
//!
//! Conventions (architecture 4.2): one method per selector core sends; each doc line is
//! `sel: <selector> [SPI]`. Integer widths
//! follow the runtime's encodings (D41), as `method_getTypeEncoding` reads them on the AGX
//! classes (macOS 26.7), where a declaration in the headers differs. Methods with arguments,
//! creators and actions return `RaiseResult<T>`; zero-argument property getters and descriptor
//! property setters do not (the metal crate wraps them in `send_get`). A handle argument core can
//! pass as nil is `Option<&P::X>`; one it never passes as nil is `&P::X`.

use super::types::{MtlRegion, MtlSize};
use super::{Handle, Platform};
use crate::raise::RaiseResult;

/// `MTLIndirectCommandBuffer` (with `MTLIndirectCommandBufferSPI`).
pub trait IndirectCommandBuffer<P: Platform>: Handle {
    /// sel: gpuResourceID
    /// `MTLResourceID._impl` (runtime `{MTLResourceID=Q}`).
    fn gpu_resource_id(&self) -> u64;
    /// sel: indirectComputeCommandAtIndex:
    fn compute_command(&self, command_index: u64)
    -> RaiseResult<Option<P::IndirectComputeCommand>>;
    /// sel: indirectRenderCommandAtIndex:
    fn render_command(&self, command_index: u64) -> RaiseResult<Option<P::IndirectRenderCommand>>;
    /// sel: resetWithRange:
    /// Commands `location..location + length`, inside the size, as a fresh buffer's: one call
    /// for the range, whose cost is the platform's for the commands it clears. Only the graphics
    /// lane sends it, for its guest's `resetWithRange:`.
    fn reset_with_range(&self, location: u64, length: u64) -> RaiseResult<()>;
    /// sel: resourceIndex SPI
    fn resource_index(&self) -> u64;
    /// sel: size
    fn size(&self) -> u64;
    /// sel: uniqueIdentifier SPI
    /// Undeclared in the headers: core sends it to an `id`, which clang binds to a `- (uint64_t)`
    /// declaration of another protocol. The runtime encoding is `Q16@0:8` (measured on the AGX
    /// class, macOS 26.7).
    fn unique_identifier(&self) -> u64;
}

/// One `MTLIndirectRenderCommand` command.
///
/// One variant per selector, whose fields are the selector's arguments in order;
/// [`IndirectRenderCommand::encode`] sends one.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the selector's arguments
pub enum IcbRenderCmd<'a, P: Platform> {
    /// `clearBarrier`
    ClearBarrier,
    /// `drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:`
    DrawIndexedPatches {
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        control_point_index_buffer: Option<&'a P::Buffer>,
        control_point_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        instance_stride: u64,
    },
    /// `drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:`
    /// `baseVertex` is `NSInteger`.
    DrawIndexedPrimitives {
        primitive_type: u64,
        index_count: u64,
        index_type: u64,
        index_buffer: Option<&'a P::Buffer>,
        index_buffer_offset: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    /// `drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:`
    DrawPatches {
        number_of_patch_control_points: u64,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer: Option<&'a P::Buffer>,
        patch_index_buffer_offset: u64,
        instance_count: u64,
        base_instance: u64,
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        instance_stride: u64,
    },
    /// `drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:`
    DrawPrimitives {
        primitive_type: u64,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// `reset`
    Reset,
    /// `setBarrier`
    SetBarrier,
    /// `setFragmentBuffer:offset:atIndex:`
    SetFragmentBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        index: u64,
    },
    /// `setRenderPipelineState:`
    SetRenderPipelineState {
        pipeline_state: Option<&'a P::RenderPipeline>,
    },
    /// `setVertexBuffer:offset:atIndex:`
    SetVertexBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        index: u64,
    },
    /// `setVertexBuffer:offset:attributeStride:atIndex:`
    SetVertexBufferAttributeStride {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        stride: u64,
        index: u64,
    },
}

impl<P: Platform> IcbRenderCmd<'_, P> {
    /// The selector this sends.
    pub fn selector(&self) -> &'static str {
        match self {
            Self::ClearBarrier => "clearBarrier",
            Self::DrawIndexedPatches { .. } => {
                "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:"
            }
            Self::DrawIndexedPrimitives { .. } => {
                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"
            }
            Self::DrawPatches { .. } => {
                "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:"
            }
            Self::DrawPrimitives { .. } => {
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"
            }
            Self::Reset => "reset",
            Self::SetBarrier => "setBarrier",
            Self::SetFragmentBuffer { .. } => "setFragmentBuffer:offset:atIndex:",
            Self::SetRenderPipelineState { .. } => "setRenderPipelineState:",
            Self::SetVertexBuffer { .. } => "setVertexBuffer:offset:atIndex:",
            Self::SetVertexBufferAttributeStride { .. } => {
                "setVertexBuffer:offset:attributeStride:atIndex:"
            }
        }
    }
}
/// `MTLIndirectRenderCommand` (with `MTLIndirectRenderCommandSPI`, the name
/// `PGResourceManager::getRenderCommandBuffers`'s mangling gives).
// One argument per keyword of the selectors core sends (D4, 4.2).
#[allow(clippy::too_many_arguments)]
pub trait IndirectRenderCommand<P: Platform>: Handle {
    /// Sends one [`IcbRenderCmd`].
    fn encode(&self, cmd: IcbRenderCmd<'_, P>) -> RaiseResult<()>;
    /// sel: getCommandType SPI
    fn command_type(&self) -> u64;
    /// sel: getFragmentBufferAddressAtIndex: SPI
    fn fragment_buffer_address(&self, index: u64) -> RaiseResult<u64>;
    /// sel: getPipelineStateUniqueIdentifier SPI
    fn pipeline_state_unique_identifier(&self) -> u64;
    /// sel: getVertexAttributeStrideAtIndex: SPI
    fn vertex_attribute_stride(&self, index: u64) -> RaiseResult<u64>;
    /// sel: getVertexBufferAddressAtIndex: SPI
    fn vertex_buffer_address(&self, index: u64) -> RaiseResult<u64>;
    /// sel: hasBarrier SPI
    /// C++ `bool`; runtime `B`.
    fn has_barrier(&self) -> bool;

    // The argument objects, read whole: the getter that answers the object, then each of its
    // getters in the order of the value's fields; nil is `None`.
    /// sel: drawArguments SPI, then its getters ([`DrawArgs`]).
    fn draw_args(&self) -> Option<DrawArgs>;
    /// sel: drawIndexedArguments SPI, then its getters ([`DrawIndexedArgs`]).
    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs>;
    /// sel: drawPatchesArguments SPI, then its getters ([`DrawPatchesArgs`]).
    fn draw_patches_args(&self) -> Option<DrawPatchesArgs>;
    /// sel: drawIndexedPatchesArguments SPI, then its getters ([`DrawIndexedPatchesArgs`]).
    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs>;
    /// sel: getTessellationFactorArguments SPI, then its getters ([`TessellationFactorArgs`]).
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs>;
}

/// One `MTLIndirectComputeCommand` command.
///
/// One variant per selector, whose fields are the selector's arguments in order;
/// [`IndirectComputeCommand::encode`] sends one.
#[derive(Debug)]
#[allow(missing_docs)] // the fields are the selector's arguments
pub enum IcbComputeCmd<'a, P: Platform> {
    /// `clearBarrier`
    ClearBarrier,
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
    ConcurrentDispatchThreadgroups {
        threadgroups_per_grid: MtlSize,
        threads_per_threadgroup: MtlSize,
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`
    ConcurrentDispatchThreads {
        threads_per_grid: MtlSize,
        threads_per_threadgroup: MtlSize,
    },
    /// `reset`
    Reset,
    /// `setBarrier`
    SetBarrier,
    /// `setComputePipelineState:`
    SetComputePipelineState {
        pipeline_state: Option<&'a P::ComputePipeline>,
    },
    /// `setImageblockWidth:height:`
    SetImageblockWidth { width: u64, height: u64 },
    /// `setKernelBuffer:offset:atIndex:`
    SetKernelBuffer {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        index: u64,
    },
    /// `setKernelBuffer:offset:attributeStride:atIndex:`
    SetKernelBufferAttributeStride {
        buffer: Option<&'a P::Buffer>,
        offset: u64,
        stride: u64,
        index: u64,
    },
    /// `setStageInRegion:`
    SetStageInRegion { region: MtlRegion },
    /// `setThreadgroupMemoryLength:atIndex:`
    SetThreadgroupMemoryLength { length: u64, index: u64 },
}

impl<P: Platform> IcbComputeCmd<'_, P> {
    /// The selector this sends.
    pub fn selector(&self) -> &'static str {
        match self {
            Self::ClearBarrier => "clearBarrier",
            Self::ConcurrentDispatchThreadgroups { .. } => {
                "concurrentDispatchThreadgroups:threadsPerThreadgroup:"
            }
            Self::ConcurrentDispatchThreads { .. } => {
                "concurrentDispatchThreads:threadsPerThreadgroup:"
            }
            Self::Reset => "reset",
            Self::SetBarrier => "setBarrier",
            Self::SetComputePipelineState { .. } => "setComputePipelineState:",
            Self::SetImageblockWidth { .. } => "setImageblockWidth:height:",
            Self::SetKernelBuffer { .. } => "setKernelBuffer:offset:atIndex:",
            Self::SetKernelBufferAttributeStride { .. } => {
                "setKernelBuffer:offset:attributeStride:atIndex:"
            }
            Self::SetStageInRegion { .. } => "setStageInRegion:",
            Self::SetThreadgroupMemoryLength { .. } => "setThreadgroupMemoryLength:atIndex:",
        }
    }
}
/// `MTLIndirectComputeCommand` (with `MTLIndirectComputeCommandSPI`, the name
/// `PGResourceManager::getComputeCommandBuffers`'s mangling gives).
pub trait IndirectComputeCommand<P: Platform>: Handle {
    /// Sends one [`IcbComputeCmd`].
    fn encode(&self, cmd: IcbComputeCmd<'_, P>) -> RaiseResult<()>;
    /// sel: getCommandType SPI
    fn command_type(&self) -> u64;
    /// sel: getKernelAttributeStrideAtIndex: SPI
    fn kernel_attribute_stride(&self, index: u64) -> RaiseResult<u64>;
    /// sel: getKernelBufferAddressAtIndex: SPI
    fn kernel_buffer_address(&self, index: u64) -> RaiseResult<u64>;
    /// sel: getPipelineStateUniqueIdentifier SPI
    fn pipeline_state_unique_identifier(&self) -> u64;
    /// sel: getStageInRegion SPI
    fn stage_in_region(&self) -> MtlRegion;
    /// sel: getThreadgroupMemoryLengthAtIndex: SPI
    fn threadgroup_memory_length(&self, index: u64) -> RaiseResult<u64>;
    /// sel: hasBarrier SPI
    /// C++ `bool`; runtime `B`.
    fn has_barrier(&self) -> bool;

    // The argument objects, read whole: the getter that answers the object, then each of its
    // getters in the order of the value's fields; nil is `None`.
    /// sel: dispatchThreadgroupsArguments SPI, then its getters ([`DispatchThreadgroupsArgs`]).
    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs>;
    /// sel: dispatchThreadsArguments SPI, then its getters ([`DispatchThreadsArgs`]).
    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs>;
    /// sel: getImageBlockSize SPI, then its getters ([`ImageBlockArgs`]).
    fn image_block_args(&self) -> Option<ImageBlockArgs>;
}

// ---- The eight ICB argument values: each of Metal's private argument objects read whole, its
// fields in the order the ICB walk sends their getters.
/// What `drawArguments`'s `_MTLIndirectDrawArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawArgs {
    /// `primitiveType`.
    pub primitive_type: u64,
    /// `vertexStart`.
    pub vertex_start: u64,
    /// `vertexCount`.
    pub vertex_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
}

/// What `drawIndexedArguments`'s `_MTLIndirectDrawIndexedArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawIndexedArgs {
    /// `indexBufferGPUVirtualAddress`.
    pub index_buffer_gpu_virtual_address: u64,
    /// `primitiveType`.
    pub primitive_type: u64,
    /// `indexType`.
    pub index_type: u64,
    /// `indexCount`.
    pub index_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseVertex`.
    pub base_vertex: u64,
    /// `baseInstance`.
    pub base_instance: u64,
}

/// What `drawPatchesArguments`'s `_MTLIndirectDrawPatchesArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawPatchesArgs {
    /// `patchIndexBufferGPUVirtualAddress`.
    pub patch_index_buffer_gpu_virtual_address: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u64,
    /// `patchStart`.
    pub patch_start: u64,
    /// `patchCount`.
    pub patch_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
}

/// What `drawIndexedPatchesArguments`'s `_MTLIndirectDrawIndexedPatchesArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DrawIndexedPatchesArgs {
    /// `patchIndexBufferGPUVirtualAddress`.
    pub patch_index_buffer_gpu_virtual_address: u64,
    /// `controlPointIndexBufferGPUVirtualAddress`.
    pub control_point_index_buffer_gpu_virtual_address: u64,
    /// `numberOfPatchControlPoints`.
    pub number_of_patch_control_points: u64,
    /// `patchStart`.
    pub patch_start: u64,
    /// `patchCount`.
    pub patch_count: u64,
    /// `instanceCount`.
    pub instance_count: u64,
    /// `baseInstance`.
    pub base_instance: u64,
}

/// What `getTessellationFactorArguments`'s `_MTLIndirectTessellationFactorArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TessellationFactorArgs {
    /// `GPUVirtualAddress`.
    pub gpu_virtual_address: u64,
    /// `instanceStride`.
    pub instance_stride: u64,
    /// `scale`.
    pub scale: f32,
}

/// What `dispatchThreadsArguments`'s `_MTLIndirectDispatchThreadsArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DispatchThreadsArgs {
    /// `threadsPerGrid`.
    pub threads_per_grid: MtlSize,
    /// `threadsPerThreadgroup`.
    pub threads_per_threadgroup: MtlSize,
}

/// What `dispatchThreadgroupsArguments`'s `_MTLIndirectDispatchThreadgroupsArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DispatchThreadgroupsArgs {
    /// `threadgroupsPerGrid`.
    pub threadgroups_per_grid: MtlSize,
    /// `threadsPerThreadgroup`.
    pub threads_per_threadgroup: MtlSize,
}

/// What `getImageBlockSize`'s `_MTLImageBlockArguments` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ImageBlockArgs {
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}
