//! Indirect command buffers: core's `IndirectCommandBuffer`, its render and compute commands
//! (whose commands, `IcbRenderCmd` and `IcbComputeCmd`, are one exhaustive `match` each, one
//! typed call per variant) and their getters. A getter that answers one of Metal's private
//! `_MTLIndirect*Arguments` or `_MTLImageBlockArguments` objects is read whole into core's value.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLIndirectCommandBuffer, MTLIndirectComputeCommand, MTLIndirectRenderCommand, MTLPrimitiveType,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::icb::{
    DispatchThreadgroupsArgs, DispatchThreadsArgs, DrawArgs, DrawIndexedArgs,
    DrawIndexedPatchesArgs, DrawPatchesArgs, IcbComputeCmd, IcbRenderCmd, ImageBlockArgs,
    TessellationFactorArgs,
};
use reims_vgpu_core::platform::types::{MtlRegion, MtlSize};
use reims_vgpu_core::raise::RaiseResult;

use crate::msg::encode::{FromMtl, ToMtl};
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::{
    IcbSpi, IndirectComputeCommandSpi, IndirectRenderCommandSpi, ResourceIndexSpi,
    UniqueIdentifierSpi,
};

/// A getter's `+0` arguments object as a handle, retained; nil is `None`.
fn arguments<T: objc2::Message>(
    sel: &'static str,
    call: impl FnOnce() -> Option<Retained<T>>,
) -> Option<Obj<T>> {
    msg::get(sel, call).map(Obj::from_retained)
}

/// `id<MTLIndirectCommandBuffer>`.
pub type MetalIcb = Obj<ProtocolObject<dyn MTLIndirectCommandBuffer>>;

impl reims_vgpu_core::platform::icb::IndirectCommandBuffer<MetalPlatform> for MetalIcb {
    /// `gpuResourceID`'s one word, `_impl`.
    fn gpu_resource_id(&self) -> u64 {
        let icb = self.get();
        msg::get("gpuResourceID", || icb.gpuResourceID()).to_raw()
    }
    /// `indirectComputeCommandAtIndex:`: a `+0` command, retained, or nil. Metal checks the
    /// index itself.
    fn compute_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::IndirectComputeCommand>> {
        let icb = self.get();
        msg::send("indirectComputeCommandAtIndex:", Policy::Caught, || {
            icb.indirectComputeCommandAtIndex_or_nil(command_index)
        })
        .map(|c| c.map(Obj::from_retained))
    }
    /// `indirectRenderCommandAtIndex:`: a `+0` command, retained, or nil. Metal checks the index
    /// itself.
    fn render_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::IndirectRenderCommand>> {
        let icb = self.get();
        msg::send("indirectRenderCommandAtIndex:", Policy::Caught, || {
            icb.indirectRenderCommandAtIndex_or_nil(command_index)
        })
        .map(|c| c.map(Obj::from_retained))
    }
    /// `resetWithRange:`, a raise inside Metal the caller's to catch, as the commands' setters'.
    fn reset_with_range(&self, location: u64, length: u64) -> RaiseResult<()> {
        let icb = self.get();
        // SAFETY: the caller passes a range inside the buffer's size (the trait's contract).
        msg::send("resetWithRange:", Policy::Caught, || unsafe {
            icb.resetWithRange(NSRange::new(location.to_usize(), length.to_usize()))
        })
    }
    /// `resourceIndex` (SPI).
    fn resource_index(&self) -> u64 {
        let icb = self.get();
        msg::get("resourceIndex", || ResourceIndexSpi::resourceIndex(icb))
    }
    fn size(&self) -> u64 {
        let icb = self.get();
        msg::get("size", || icb.size()).to_u64()
    }
    /// `uniqueIdentifier` (SPI).
    fn unique_identifier(&self) -> u64 {
        let icb = self.get();
        msg::get("uniqueIdentifier", || {
            UniqueIdentifierSpi::uniqueIdentifier(icb)
        })
    }
}

/// `id<MTLIndirectRenderCommand>`.
pub type MetalIndirectRenderCommand = Obj<ProtocolObject<dyn MTLIndirectRenderCommand>>;

impl reims_vgpu_core::platform::icb::IndirectRenderCommand<MetalPlatform>
    for MetalIndirectRenderCommand
{
    /// One call per command: a raise inside Metal is the caller's to catch (every command is
    /// `Caught`: only the ICB's preparation encodes them, always under a handler). A nil buffer
    /// or state goes to Metal as it is.
    fn encode(&self, cmd: IcbRenderCmd<'_, MetalPlatform>) -> RaiseResult<()> {
        let c = self.get();
        match cmd {
            IcbRenderCmd::ClearBarrier => {
                msg::send("clearBarrier", Policy::Caught, || c.clearBarrier())
            }
            IcbRenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let patch_index = patch_index_buffer.map(Obj::get);
                let control_point_index = control_point_index_buffer.map(Obj::get);
                let buffer = buffer.map(Obj::get);
                msg::send(
                    "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:",
                    Policy::Caught,
                    || {
                        IndirectRenderCommandSpi::drawIndexedPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_controlPointIndexBuffer_controlPointIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
                        c,
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        control_point_index,
                        control_point_index_buffer_offset,
                        instance_count,
                        base_instance,
                        buffer,
                        offset,
                        instance_stride,
                    )
                    },
                )
            }
            IcbRenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => {
                let index = index_buffer.map(Obj::get);
                msg::send(
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                    Policy::Caught,
                    || {
                        IndirectRenderCommandSpi::drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        c,
                        primitive_type,
                        index_count,
                        index_type,
                        index,
                        index_buffer_offset,
                        instance_count,
                        base_vertex,
                        base_instance,
                    )
                    },
                )
            }
            IcbRenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let (patch_index, buffer) =
                    (patch_index_buffer.map(Obj::get), buffer.map(Obj::get));
                msg::send(
                    "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:",
                    Policy::Caught,
                    || {
                        IndirectRenderCommandSpi::drawPatches_patchStart_patchCount_patchIndexBuffer_patchIndexBufferOffset_instanceCount_baseInstance_tessellationFactorBuffer_tessellationFactorBufferOffset_tessellationFactorBufferInstanceStride(
                        c,
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        instance_count,
                        base_instance,
                        buffer,
                        offset,
                        instance_stride,
                    )
                    },
                )
            }
            IcbRenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                let primitive = MTLPrimitiveType(primitive_type.to_usize());
                let (start, count) = (vertex_start.to_usize(), vertex_count.to_usize());
                let (instances, base) = (instance_count.to_usize(), base_instance.to_usize());
                // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
                msg::send(
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                    Policy::Caught,
                    || unsafe {
                        c.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                            primitive, start, count, instances, base,
                        )
                    },
                )
            }
            IcbRenderCmd::Reset => msg::send("reset", Policy::Caught, || c.reset()),
            IcbRenderCmd::SetBarrier => msg::send("setBarrier", Policy::Caught, || c.setBarrier()),
            IcbRenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send("setFragmentBuffer:offset:atIndex:", Policy::Caught, || {
                    IndirectRenderCommandSpi::setFragmentBuffer_offset_atIndex(
                        c, buffer, offset, index,
                    )
                })
            }
            IcbRenderCmd::SetRenderPipelineState { pipeline_state } => {
                let state = pipeline_state.map(Obj::get);
                msg::send("setRenderPipelineState:", Policy::Caught, || {
                    IndirectRenderCommandSpi::setRenderPipelineState(c, state)
                })
            }
            IcbRenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send("setVertexBuffer:offset:atIndex:", Policy::Caught, || {
                    IndirectRenderCommandSpi::setVertexBuffer_offset_atIndex(
                        c, buffer, offset, index,
                    )
                })
            }
            IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send(
                    "setVertexBuffer:offset:attributeStride:atIndex:",
                    Policy::Caught,
                    || {
                        IndirectRenderCommandSpi::setVertexBuffer_offset_attributeStride_atIndex(
                            c, buffer, offset, stride, index,
                        )
                    },
                )
            }
        }
    }
    /// `getCommandType` (SPI).
    fn command_type(&self) -> u64 {
        let c = self.get();
        msg::get("getCommandType", || {
            IndirectRenderCommandSpi::getCommandType(c)
        })
    }
    /// `getFragmentBufferAddressAtIndex:` (SPI).
    fn fragment_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getFragmentBufferAddressAtIndex:", Policy::Caught, || {
            c.getFragmentBufferAddressAtIndex(index)
        })
    }
    /// `getPipelineStateUniqueIdentifier` (SPI).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let c = self.get();
        msg::get("getPipelineStateUniqueIdentifier", || {
            IndirectRenderCommandSpi::getPipelineStateUniqueIdentifier(c)
        })
    }
    /// `getVertexAttributeStrideAtIndex:` (SPI).
    fn vertex_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getVertexAttributeStrideAtIndex:", Policy::Caught, || {
            c.getVertexAttributeStrideAtIndex(index)
        })
    }
    /// `getVertexBufferAddressAtIndex:` (SPI).
    fn vertex_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getVertexBufferAddressAtIndex:", Policy::Caught, || {
            c.getVertexBufferAddressAtIndex(index)
        })
    }
    /// `hasBarrier` (SPI).
    fn has_barrier(&self) -> bool {
        let c = self.get();
        msg::get("hasBarrier", || IndirectRenderCommandSpi::hasBarrier(c))
    }

    // The argument objects, read whole: each getter of one, in the order of its value's fields.
    fn draw_args(&self) -> Option<DrawArgs> {
        let c = self.get();
        arguments("drawArguments", || c.drawArguments()).map(|a| {
            let a = a.get();
            DrawArgs {
                primitive_type: msg::get("primitiveType", || a.primitiveType()),
                vertex_start: msg::get("vertexStart", || a.vertexStart()),
                vertex_count: msg::get("vertexCount", || a.vertexCount()),
                instance_count: msg::get("instanceCount", || a.instanceCount()),
                base_instance: msg::get("baseInstance", || a.baseInstance()),
            }
        })
    }
    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs> {
        let c = self.get();
        arguments("drawIndexedArguments", || c.drawIndexedArguments()).map(|a| {
            let a = a.get();
            DrawIndexedArgs {
                index_buffer_gpu_virtual_address: msg::get("indexBufferGPUVirtualAddress", || {
                    a.indexBufferGPUVirtualAddress()
                }),
                primitive_type: msg::get("primitiveType", || a.primitiveType()),
                index_type: msg::get("indexType", || a.indexType()),
                index_count: msg::get("indexCount", || a.indexCount()),
                instance_count: msg::get("instanceCount", || a.instanceCount()),
                base_vertex: msg::get("baseVertex", || a.baseVertex()),
                base_instance: msg::get("baseInstance", || a.baseInstance()),
            }
        })
    }
    fn draw_patches_args(&self) -> Option<DrawPatchesArgs> {
        let c = self.get();
        arguments("drawPatchesArguments", || c.drawPatchesArguments()).map(|a| {
            let a = a.get();
            DrawPatchesArgs {
                patch_index_buffer_gpu_virtual_address: msg::get(
                    "patchIndexBufferGPUVirtualAddress",
                    || a.patchIndexBufferGPUVirtualAddress(),
                ),
                number_of_patch_control_points: msg::get("numberOfPatchControlPoints", || {
                    a.numberOfPatchControlPoints()
                }),
                patch_start: msg::get("patchStart", || a.patchStart()),
                patch_count: msg::get("patchCount", || a.patchCount()),
                instance_count: msg::get("instanceCount", || a.instanceCount()),
                base_instance: msg::get("baseInstance", || a.baseInstance()),
            }
        })
    }
    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs> {
        let c = self.get();
        arguments("drawIndexedPatchesArguments", || {
            c.drawIndexedPatchesArguments()
        })
        .map(|a| {
            let a = a.get();
            DrawIndexedPatchesArgs {
                patch_index_buffer_gpu_virtual_address: msg::get(
                    "patchIndexBufferGPUVirtualAddress",
                    || a.patchIndexBufferGPUVirtualAddress(),
                ),
                control_point_index_buffer_gpu_virtual_address: msg::get(
                    "controlPointIndexBufferGPUVirtualAddress",
                    || a.controlPointIndexBufferGPUVirtualAddress(),
                ),
                number_of_patch_control_points: msg::get("numberOfPatchControlPoints", || {
                    a.numberOfPatchControlPoints()
                }),
                patch_start: msg::get("patchStart", || a.patchStart()),
                patch_count: msg::get("patchCount", || a.patchCount()),
                instance_count: msg::get("instanceCount", || a.instanceCount()),
                base_instance: msg::get("baseInstance", || a.baseInstance()),
            }
        })
    }
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs> {
        let c = self.get();
        arguments("getTessellationFactorArguments", || {
            c.getTessellationFactorArguments()
        })
        .map(|a| {
            let a = a.get();
            TessellationFactorArgs {
                gpu_virtual_address: msg::get("GPUVirtualAddress", || a.GPUVirtualAddress()),
                instance_stride: msg::get("instanceStride", || a.instanceStride()),
                scale: msg::get("scale", || a.scale()),
            }
        })
    }
}

/// `id<MTLIndirectComputeCommand>`.
pub type MetalIndirectComputeCommand = Obj<ProtocolObject<dyn MTLIndirectComputeCommand>>;

impl reims_vgpu_core::platform::icb::IndirectComputeCommand<MetalPlatform>
    for MetalIndirectComputeCommand
{
    /// One call per command: a raise inside Metal is the caller's to catch (every command is
    /// `Caught`: only the ICB's preparation encodes them, always under a handler). A nil buffer
    /// or state goes to Metal as it is.
    fn encode(&self, cmd: IcbComputeCmd<'_, MetalPlatform>) -> RaiseResult<()> {
        let c = self.get();
        match cmd {
            IcbComputeCmd::ClearBarrier => {
                msg::send("clearBarrier", Policy::Caught, || c.clearBarrier())
            }
            IcbComputeCmd::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => {
                let (grid, group) = (threadgroups_per_grid.mtl(), threads_per_threadgroup.mtl());
                msg::send(
                    "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                    Policy::Caught,
                    || c.concurrentDispatchThreadgroups_threadsPerThreadgroup(grid, group),
                )
            }
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => {
                let (grid, group) = (threads_per_grid.mtl(), threads_per_threadgroup.mtl());
                msg::send(
                    "concurrentDispatchThreads:threadsPerThreadgroup:",
                    Policy::Caught,
                    || c.concurrentDispatchThreads_threadsPerThreadgroup(grid, group),
                )
            }
            IcbComputeCmd::Reset => msg::send("reset", Policy::Caught, || c.reset()),
            IcbComputeCmd::SetBarrier => msg::send("setBarrier", Policy::Caught, || c.setBarrier()),
            IcbComputeCmd::SetComputePipelineState { pipeline_state } => {
                let state = pipeline_state.map(Obj::get);
                msg::send("setComputePipelineState:", Policy::Caught, || {
                    IndirectComputeCommandSpi::setComputePipelineState(c, state)
                })
            }
            IcbComputeCmd::SetImageblockWidth { width, height } => {
                let (width, height) = (width.to_usize(), height.to_usize());
                msg::send("setImageblockWidth:height:", Policy::Caught, || {
                    c.setImageblockWidth_height(width, height)
                })
            }
            IcbComputeCmd::SetKernelBuffer {
                buffer,
                offset,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send("setKernelBuffer:offset:atIndex:", Policy::Caught, || {
                    IndirectComputeCommandSpi::setKernelBuffer_offset_atIndex(
                        c, buffer, offset, index,
                    )
                })
            }
            IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let buffer = buffer.map(Obj::get);
                msg::send(
                    "setKernelBuffer:offset:attributeStride:atIndex:",
                    Policy::Caught,
                    || {
                        IndirectComputeCommandSpi::setKernelBuffer_offset_attributeStride_atIndex(
                            c, buffer, offset, stride, index,
                        )
                    },
                )
            }
            IcbComputeCmd::SetStageInRegion { region } => {
                let region = region.mtl();
                msg::send("setStageInRegion:", Policy::Caught, || {
                    c.setStageInRegion(region)
                })
            }
            IcbComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                let (length, index) = (length.to_usize(), index.to_usize());
                // SAFETY: raw guest values go to Metal unvalidated; Metal checks them itself.
                msg::send(
                    "setThreadgroupMemoryLength:atIndex:",
                    Policy::Caught,
                    || unsafe { c.setThreadgroupMemoryLength_atIndex(length, index) },
                )
            }
        }
    }
    /// `getCommandType` (SPI).
    fn command_type(&self) -> u64 {
        let c = self.get();
        msg::get("getCommandType", || {
            IndirectComputeCommandSpi::getCommandType(c)
        })
    }
    /// `getKernelAttributeStrideAtIndex:` (SPI).
    fn kernel_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getKernelAttributeStrideAtIndex:", Policy::Caught, || {
            c.getKernelAttributeStrideAtIndex(index)
        })
    }
    /// `getKernelBufferAddressAtIndex:` (SPI).
    fn kernel_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getKernelBufferAddressAtIndex:", Policy::Caught, || {
            c.getKernelBufferAddressAtIndex(index)
        })
    }
    /// `getPipelineStateUniqueIdentifier` (SPI).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let c = self.get();
        msg::get("getPipelineStateUniqueIdentifier", || {
            IndirectComputeCommandSpi::getPipelineStateUniqueIdentifier(c)
        })
    }
    /// `getStageInRegion` (SPI).
    fn stage_in_region(&self) -> MtlRegion {
        let c = self.get();
        MtlRegion::from_mtl(msg::get("getStageInRegion", || c.getStageInRegion()))
    }
    /// `getThreadgroupMemoryLengthAtIndex:` (SPI).
    fn threadgroup_memory_length(&self, index: u64) -> RaiseResult<u64> {
        let c = self.get();
        msg::send("getThreadgroupMemoryLengthAtIndex:", Policy::Caught, || {
            c.getThreadgroupMemoryLengthAtIndex(index)
        })
    }
    /// `hasBarrier` (SPI).
    fn has_barrier(&self) -> bool {
        let c = self.get();
        msg::get("hasBarrier", || IndirectComputeCommandSpi::hasBarrier(c))
    }

    // The argument objects, read whole: each getter of one, in the order of its value's fields.
    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs> {
        let c = self.get();
        arguments("dispatchThreadgroupsArguments", || {
            c.dispatchThreadgroupsArguments()
        })
        .map(|a| {
            let a = a.get();
            DispatchThreadgroupsArgs {
                threadgroups_per_grid: MtlSize::from_mtl(msg::get("threadgroupsPerGrid", || {
                    a.threadgroupsPerGrid()
                })),
                threads_per_threadgroup: MtlSize::from_mtl(msg::get(
                    "threadsPerThreadgroup",
                    || a.threadsPerThreadgroup(),
                )),
            }
        })
    }
    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs> {
        let c = self.get();
        arguments("dispatchThreadsArguments", || c.dispatchThreadsArguments()).map(|a| {
            let a = a.get();
            DispatchThreadsArgs {
                threads_per_grid: MtlSize::from_mtl(msg::get("threadsPerGrid", || {
                    a.threadsPerGrid()
                })),
                threads_per_threadgroup: MtlSize::from_mtl(msg::get(
                    "threadsPerThreadgroup",
                    || a.threadsPerThreadgroup(),
                )),
            }
        })
    }
    fn image_block_args(&self) -> Option<ImageBlockArgs> {
        let c = self.get();
        arguments("getImageBlockSize", || c.getImageBlockSize()).map(|a| {
            let a = a.get();
            ImageBlockArgs {
                width: msg::get("width", || a.width()),
                height: msg::get("height", || a.height()),
            }
        })
    }
}

#[cfg(test)]
mod tests {

    use reims_vgpu_core::platform::icb::*;
    use reims_vgpu_core::platform::resource::{
        Buffer as _, ComputePipeline as _, RenderPipeline as _,
    };

    use reims_vgpu_core::platform::types::*;

    use super::*;
    use crate::gpu::resource::testkit::*;

    /// Every IndirectCommandBuffer method
    #[test]
    fn icb_sends() {
        let k = kit();
        let x = k.icb_render;
        assert_eq!(
            x.gpu_resource_id(),
            raw::<ResourceId>(id(&x), "gpuResourceID").0
        );
        let r = x.compute_command(0);
        assert!(r.is_ok(), "compute_command: {r:?}");
        let r = x.render_command(0);
        assert!(r.is_ok(), "render_command: {r:?}");
        assert_eq!(x.resource_index(), raw_u64(id(&x), "resourceIndex"));
        assert_eq!(x.size(), raw_u64(id(&x), "size"));
        assert_eq!(x.unique_identifier(), raw_u64(id(&x), "uniqueIdentifier"));
    }

    /// Every IndirectRenderCommand method, pipeline first
    #[test]
    fn indirect_render_command_sends() {
        let k = kit();
        let x = k.icb_render.render_command(0).unwrap().unwrap();
        let r = x.encode(IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&k.render_pipeline),
        });
        assert!(r.is_ok(), "set_render_pipeline_state: {r:?}");
        let r = x.encode(IcbRenderCmd::ClearBarrier);
        assert!(r.is_ok(), "clear_barrier: {r:?}");
        let _ = x.draw_args();
        let _ = x.draw_indexed_args();
        let r = x.encode(IcbRenderCmd::DrawIndexedPatches {
            number_of_patch_control_points: 0,
            patch_start: 0,
            patch_count: 1,
            patch_index_buffer: Some(&k.buffer),
            patch_index_buffer_offset: 0,
            control_point_index_buffer: Some(&k.buffer),
            control_point_index_buffer_offset: 0,
            instance_count: 1,
            base_instance: 0,
            buffer: Some(&k.buffer),
            offset: 0,
            instance_stride: 16,
        });
        assert!(r.is_ok(), "draw_indexed_patches: {r:?}");
        let _ = x.draw_indexed_patches_args();
        let r = x.encode(IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 0,
            index_type: 0,
            index_buffer: Some(&k.buffer),
            index_buffer_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        });
        assert!(r.is_ok(), "draw_indexed_primitives: {r:?}");
        let r = x.encode(IcbRenderCmd::DrawPatches {
            number_of_patch_control_points: 0,
            patch_start: 0,
            patch_count: 1,
            patch_index_buffer: Some(&k.buffer),
            patch_index_buffer_offset: 0,
            instance_count: 1,
            base_instance: 0,
            buffer: Some(&k.buffer),
            offset: 0,
            instance_stride: 16,
        });
        assert!(r.is_ok(), "draw_patches: {r:?}");
        let _ = x.draw_patches_args();
        let r = x.encode(IcbRenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
            base_instance: 0,
        });
        assert!(r.is_ok(), "draw_primitives: {r:?}");
        assert_eq!(x.command_type(), raw_u64(id(&x), "getCommandType"));
        let r = x.fragment_buffer_address(0);
        assert!(r.is_ok(), "fragment_buffer_address: {r:?}");
        assert_eq!(
            x.pipeline_state_unique_identifier(),
            raw_u64(id(&x), "getPipelineStateUniqueIdentifier")
        );
        let _ = x.tessellation_factor_args();
        let r = x.vertex_attribute_stride(0);
        assert!(r.is_ok(), "vertex_attribute_stride: {r:?}");
        let r = x.vertex_buffer_address(0);
        assert!(r.is_ok(), "vertex_buffer_address: {r:?}");
        assert_eq!(x.has_barrier(), raw_bool(id(&x), "hasBarrier"));
        let r = x.encode(IcbRenderCmd::Reset);
        assert!(r.is_ok(), "reset: {r:?}");
        let r = x.encode(IcbRenderCmd::SetBarrier);
        assert!(r.is_ok(), "set_barrier: {r:?}");
        let r = x.encode(IcbRenderCmd::SetFragmentBuffer {
            buffer: Some(&k.buffer),
            offset: 0,
            index: 0,
        });
        assert!(r.is_ok(), "set_fragment_buffer: {r:?}");
        let r = x.encode(IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&k.buffer),
            offset: 0,
            index: 0,
        });
        assert!(r.is_ok(), "set_vertex_buffer: {r:?}");
        let r = x.encode(IcbRenderCmd::SetVertexBufferAttributeStride {
            buffer: Some(&k.buffer),
            offset: 0,
            stride: 16,
            index: 0,
        });
        assert!(r.is_ok(), "set_vertex_buffer_attribute_stride: {r:?}");
    }

    /// Every IndirectComputeCommand method, pipeline first
    #[test]
    fn indirect_compute_command_sends() {
        let k = kit();
        let x = k.icb_compute.compute_command(0).unwrap().unwrap();
        let r = x.encode(IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&k.compute_pipeline),
        });
        assert!(r.is_ok(), "set_compute_pipeline_state: {r:?}");
        let r = x.encode(IcbComputeCmd::ClearBarrier);
        assert!(r.is_ok(), "clear_barrier: {r:?}");
        let r = x.encode(IcbComputeCmd::ConcurrentDispatchThreadgroups {
            threadgroups_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        });
        assert!(r.is_ok(), "concurrent_dispatch_threadgroups: {r:?}");
        let r = x.encode(IcbComputeCmd::ConcurrentDispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        });
        assert!(r.is_ok(), "concurrent_dispatch_threads: {r:?}");
        let _ = x.dispatch_threadgroups_args();
        let _ = x.dispatch_threads_args();
        assert_eq!(x.command_type(), raw_u64(id(&x), "getCommandType"));
        let _ = x.image_block_args();
        let r = x.kernel_attribute_stride(0);
        assert!(r.is_ok(), "kernel_attribute_stride: {r:?}");
        let r = x.kernel_buffer_address(0);
        assert!(r.is_ok(), "kernel_buffer_address: {r:?}");
        assert_eq!(
            x.pipeline_state_unique_identifier(),
            raw_u64(id(&x), "getPipelineStateUniqueIdentifier")
        );
        let _ = x.stage_in_region();
        let r = x.threadgroup_memory_length(0);
        assert!(r.is_ok(), "threadgroup_memory_length: {r:?}");
        assert_eq!(x.has_barrier(), raw_bool(id(&x), "hasBarrier"));
        let r = x.encode(IcbComputeCmd::Reset);
        assert!(r.is_ok(), "reset: {r:?}");
        let r = x.encode(IcbComputeCmd::SetBarrier);
        assert!(r.is_ok(), "set_barrier: {r:?}");
        let r = x.encode(IcbComputeCmd::SetImageblockWidth {
            width: 1,
            height: 1,
        });
        assert!(r.is_ok(), "set_imageblock_width: {r:?}");
        let r = x.encode(IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&k.buffer),
            offset: 0,
            index: 0,
        });
        assert!(r.is_ok(), "set_kernel_buffer: {r:?}");
        let r = x.encode(IcbComputeCmd::SetKernelBufferAttributeStride {
            buffer: Some(&k.buffer),
            offset: 0,
            stride: 16,
            index: 0,
        });
        assert!(r.is_ok(), "set_kernel_buffer_attribute_stride: {r:?}");
        let r = x.encode(IcbComputeCmd::SetStageInRegion {
            region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(1, 1, 1)),
        });
        assert!(r.is_ok(), "set_stage_in_region: {r:?}");
        let r = x.encode(IcbComputeCmd::SetThreadgroupMemoryLength {
            length: 16,
            index: 0,
        });
        assert!(r.is_ok(), "set_threadgroup_memory_length: {r:?}");
    }

    /// The render command's recorded state reads back through the SPI getters and the values read
    /// from Metal's private argument objects (_MTLIndirectDraw*Arguments,
    /// _MTLIndirectTessellationFactorArguments), barrier set and clear, reset
    #[test]
    fn render_command_reads_back() {
        let k = kit();
        let icb = &k.icb_render;
        assert_eq!(icb.size(), 4);
        assert_eq!(
            icb.gpu_resource_id(),
            raw::<ResourceId>(id(icb), "gpuResourceID").0
        );
        let c = icb.render_command(1).unwrap().unwrap();
        c.encode(IcbRenderCmd::SetRenderPipelineState {
            pipeline_state: Some(&k.render_pipeline),
        })
        .unwrap();
        assert_eq!(
            c.pipeline_state_unique_identifier(),
            k.render_pipeline.unique_identifier()
        );
        c.encode(IcbRenderCmd::SetVertexBuffer {
            buffer: Some(&k.buffer),
            offset: 16,
            index: 0,
        })
        .unwrap();
        assert_eq!(
            c.vertex_buffer_address(0).unwrap(),
            k.buffer.gpu_address() + 16
        );
        c.encode(IcbRenderCmd::SetFragmentBuffer {
            buffer: Some(&k.buffer),
            offset: 32,
            index: 1,
        })
        .unwrap();
        assert_eq!(
            c.fragment_buffer_address(1).unwrap(),
            k.buffer.gpu_address() + 32
        );
        c.encode(IcbRenderCmd::SetBarrier).unwrap();
        assert!(c.has_barrier());
        c.encode(IcbRenderCmd::ClearBarrier).unwrap();
        assert!(!c.has_barrier());

        c.encode(IcbRenderCmd::DrawPrimitives {
            primitive_type: 3,
            vertex_start: 5,
            vertex_count: 6,
            instance_count: 2,
            base_instance: 1,
        })
        .unwrap();
        assert_eq!(
            c.draw_args(),
            Some(DrawArgs {
                primitive_type: 3,
                vertex_start: 5,
                vertex_count: 6,
                instance_count: 2,
                base_instance: 1,
            })
        );
        assert_ne!(c.command_type(), 0);

        c.encode(IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type: 3,
            index_count: 6,
            index_type: 1,
            index_buffer: Some(&k.buffer),
            index_buffer_offset: 64,
            instance_count: 2,
            base_vertex: 0,
            base_instance: 1,
        })
        .unwrap();
        assert_eq!(
            c.draw_indexed_args(),
            Some(DrawIndexedArgs {
                index_buffer_gpu_virtual_address: k.buffer.gpu_address() + 64,
                primitive_type: 3,
                index_type: 1,
                index_count: 6,
                instance_count: 2,
                base_vertex: 0,
                base_instance: 1,
            })
        );

        c.encode(IcbRenderCmd::DrawPatches {
            number_of_patch_control_points: 3,
            patch_start: 0,
            patch_count: 4,
            patch_index_buffer: None,
            patch_index_buffer_offset: 0,
            instance_count: 1,
            base_instance: 0,
            buffer: Some(&k.buffer),
            offset: 128,
            instance_stride: 0,
        })
        .unwrap();
        let a = c.draw_patches_args().unwrap();
        assert_eq!(
            (
                a.number_of_patch_control_points,
                a.patch_start,
                a.patch_count,
                a.instance_count,
                a.base_instance
            ),
            (3, 0, 4, 1, 0)
        );
        // No patch index buffer: the word is whatever the command held (not asserted).
        let t = c.tessellation_factor_args().unwrap();
        assert_eq!(t.gpu_virtual_address, k.buffer.gpu_address() + 128);

        c.encode(IcbRenderCmd::DrawIndexedPatches {
            number_of_patch_control_points: 3,
            patch_start: 1,
            patch_count: 2,
            patch_index_buffer: Some(&k.buffer),
            patch_index_buffer_offset: 0,
            control_point_index_buffer: Some(&k.buffer),
            control_point_index_buffer_offset: 256,
            instance_count: 1,
            base_instance: 0,
            buffer: Some(&k.buffer),
            offset: 512,
            instance_stride: 0,
        })
        .unwrap();
        assert_eq!(
            c.draw_indexed_patches_args(),
            Some(DrawIndexedPatchesArgs {
                patch_index_buffer_gpu_virtual_address: k.buffer.gpu_address(),
                control_point_index_buffer_gpu_virtual_address: k.buffer.gpu_address() + 256,
                number_of_patch_control_points: 3,
                patch_start: 1,
                patch_count: 2,
                instance_count: 1,
                base_instance: 0,
            })
        );
        c.encode(IcbRenderCmd::Reset).unwrap();
    }

    /// The compute command's recorded state reads back through the SPI getters and the values
    /// read from _MTLIndirectDispatch*Arguments / _MTLImageBlockArguments
    #[test]
    fn compute_command_reads_back() {
        let k = kit();
        let c = k.icb_compute.compute_command(2).unwrap().unwrap();
        c.encode(IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&k.compute_pipeline),
        })
        .unwrap();
        assert_eq!(
            c.pipeline_state_unique_identifier(),
            k.compute_pipeline.unique_identifier()
        );
        c.encode(IcbComputeCmd::SetKernelBuffer {
            buffer: Some(&k.buffer),
            offset: 48,
            index: 0,
        })
        .unwrap();
        assert_eq!(
            c.kernel_buffer_address(0).unwrap(),
            k.buffer.gpu_address() + 48
        );
        c.encode(IcbComputeCmd::SetThreadgroupMemoryLength {
            length: 64,
            index: 0,
        })
        .unwrap();
        assert_eq!(c.threadgroup_memory_length(0).unwrap(), 64);
        let region = MtlRegion::new(MtlOrigin::new(1, 2, 0), MtlSize::new(3, 4, 1));
        c.encode(IcbComputeCmd::SetStageInRegion { region })
            .unwrap();
        assert_eq!(c.stage_in_region(), region);
        c.encode(IcbComputeCmd::ConcurrentDispatchThreadgroups {
            threadgroups_per_grid: MtlSize::new(2, 3, 1),
            threads_per_threadgroup: MtlSize::new(8, 1, 1),
        })
        .unwrap();
        assert_eq!(
            c.dispatch_threadgroups_args(),
            Some(DispatchThreadgroupsArgs {
                threadgroups_per_grid: MtlSize::new(2, 3, 1),
                threads_per_threadgroup: MtlSize::new(8, 1, 1),
            })
        );
        c.encode(IcbComputeCmd::ConcurrentDispatchThreads {
            threads_per_grid: MtlSize::new(64, 2, 1),
            threads_per_threadgroup: MtlSize::new(16, 1, 1),
        })
        .unwrap();
        assert_eq!(
            c.dispatch_threads_args(),
            Some(DispatchThreadsArgs {
                threads_per_grid: MtlSize::new(64, 2, 1),
                threads_per_threadgroup: MtlSize::new(16, 1, 1),
            })
        );
        assert_ne!(c.command_type(), 0);
        c.encode(IcbComputeCmd::SetImageblockWidth {
            width: 8,
            height: 4,
        })
        .unwrap();
        assert_eq!(
            c.image_block_args(),
            Some(ImageBlockArgs {
                width: 8,
                height: 4
            })
        );
        c.encode(IcbComputeCmd::SetBarrier).unwrap();
        assert!(c.has_barrier());
        c.encode(IcbComputeCmd::ClearBarrier).unwrap();
        c.encode(IcbComputeCmd::Reset).unwrap();
    }
}
