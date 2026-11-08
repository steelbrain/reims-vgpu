//! Fake: the `platform/icb.rs` traits over the fake models (architecture 13.2).
//!
//! An ICB's commands are per-index child objects, made at their first `…CommandAtIndex:`. Each
//! logs its encoded commands, which are ordered, and records what they set in its
//! [`IcbCommandModel`], which the free read-back getters answer from.

use crate::core::testkit::script::Policy;
use std::collections::BTreeMap;

use super::super::calllog::Arg;
use super::super::objects::{Fk, Kind, Model, ResourceCommon, kind};
use super::super::platform::FakePlatform;
use crate::core::platform::Platform;
use crate::core::platform::icb::*;
use crate::core::platform::types::{MtlRegion, MtlSize};
use crate::core::raise::RaiseResult;

/// An indirect command buffer's model.
#[derive(Debug, Default)]
pub struct IcbModel {
    /// Common resource state.
    pub common: ResourceCommon,
    /// `size`: the maximum command count it was made with.
    pub size: u64,
    /// The resource options it was made with.
    pub options: u64,
}

impl Fk<kind::Icb> {
    /// Makes `size` answer `size`.
    pub fn set_size(&self, size: u64) {
        self.with_model_mut(|m: &mut IcbModel| m.size = size);
    }
}

/// What an ICB render or compute command's encoded commands set (zeros for a fresh or reset
/// command).
#[derive(Debug, Default)]
pub struct IcbCommandModel {
    /// `getCommandType`: the last draw's or dispatch's type.
    pub command_type: u64,
    /// `hasBarrier`.
    pub barrier: bool,
    /// `getPipelineStateUniqueIdentifier`: the pipeline state set, by its `uniqueIdentifier`.
    pub pipeline: u64,
    /// A render command's vertex buffer addresses, by index.
    pub vertex_buffers: BTreeMap<u64, u64>,
    /// A render command's vertex attribute strides, by index.
    pub vertex_strides: BTreeMap<u64, u64>,
    /// A render command's fragment buffer addresses, by index.
    pub fragment_buffers: BTreeMap<u64, u64>,
    /// A compute command's kernel buffer addresses, by index.
    pub kernel_buffers: BTreeMap<u64, u64>,
    /// A compute command's kernel attribute strides, by index.
    pub kernel_strides: BTreeMap<u64, u64>,
    /// A compute command's threadgroup memory lengths, by index.
    pub threadgroup_memory: BTreeMap<u64, u64>,
    /// A render command's last draw.
    pub draw: Option<IcbDraw>,
    /// A compute command's last dispatch: the grid, then the threadgroup.
    pub dispatch: Option<[MtlSize; 2]>,
    /// A compute command's image block width and height.
    pub imageblock: [u64; 2],
    /// A compute command's stage-in region.
    pub stage_in_region: MtlRegion,
}

/// What an ICB render command's draw recorded: every draw's fields, 0 where the draw has none.
/// The argument getters read it whatever the draw was.
#[derive(Debug, Default, Clone, Copy)]
#[allow(missing_docs)] // each field is the argument object's field of that name
pub struct IcbDraw {
    pub primitive_type: u64,
    pub vertex_start: u64,
    pub vertex_count: u64,
    pub index_type: u64,
    pub index_count: u64,
    pub index_buffer_gpu_virtual_address: u64,
    pub instance_count: u64,
    pub base_vertex: u64,
    pub base_instance: u64,
    pub number_of_patch_control_points: u64,
    pub patch_start: u64,
    pub patch_count: u64,
    pub patch_index_buffer_gpu_virtual_address: u64,
    pub control_point_index_buffer_gpu_virtual_address: u64,
    pub gpu_virtual_address: u64,
    pub instance_stride: u64,
}

impl IndirectCommandBuffer<FakePlatform> for Fk<kind::Icb> {
    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    fn compute_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::IndirectComputeCommand>> {
        self.index_child(
            Policy::Caught,
            "indirectComputeCommandAtIndex:",
            "compute",
            command_index,
        )
    }

    fn render_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::IndirectRenderCommand>> {
        self.index_child(
            Policy::Caught,
            "indirectRenderCommandAtIndex:",
            "render",
            command_index,
        )
    }

    /// One call for the range: the commands made so far inside it lose what they were set, as
    /// a command's `reset` leaves one, each reached once, whatever the range's length.
    fn reset_with_range(&self, location: u64, length: u64) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "resetWithRange:",
            vec![Arg::U64(location), Arg::U64(length)],
        )?;
        let range = location..location.saturating_add(length);
        for (_, index, child) in self.children() {
            if !range.contains(&index) {
                continue;
            }
            if let Some(c) = child.cast::<kind::IndirectRenderCommand>() {
                command(&c, |m| *m = IcbCommandModel::default());
            } else if let Some(c) = child.cast::<kind::IndirectComputeCommand>() {
                command(&c, |m| *m = IcbCommandModel::default());
            }
        }
        Ok(())
    }

    fn resource_index(&self) -> u64 {
        self.resource_index_value()
    }

    fn size(&self) -> u64 {
        self.with_model(|m: &IcbModel| m.size)
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_id_value()
    }
}

#[allow(clippy::too_many_arguments)] // one argument per keyword (D4)
impl IndirectRenderCommand<FakePlatform> for Fk<kind::IndirectRenderCommand> {
    fn encode(&self, cmd: IcbRenderCmd<'_, FakePlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        match cmd {
            IcbRenderCmd::ClearBarrier => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| m.barrier = false);
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
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::U64(number_of_patch_control_points),
                        Arg::U64(patch_start),
                        Arg::U64(patch_count),
                        Arg::opt(patch_index_buffer),
                        Arg::U64(patch_index_buffer_offset),
                        Arg::opt(control_point_index_buffer),
                        Arg::U64(control_point_index_buffer_offset),
                        Arg::U64(instance_count),
                        Arg::U64(base_instance),
                        Arg::opt(buffer),
                        Arg::U64(offset),
                        Arg::U64(instance_stride),
                    ],
                )?;
                draw(
                    self,
                    8,
                    IcbDraw {
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index_buffer_gpu_virtual_address: gpu_of(
                            patch_index_buffer,
                            patch_index_buffer_offset,
                        ),
                        control_point_index_buffer_gpu_virtual_address: gpu_of(
                            control_point_index_buffer,
                            control_point_index_buffer_offset,
                        ),
                        instance_count,
                        base_instance,
                        gpu_virtual_address: gpu_of(buffer, offset),
                        instance_stride,
                        ..IcbDraw::default()
                    },
                );
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
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::U64(primitive_type),
                        Arg::U64(index_count),
                        Arg::U64(index_type),
                        Arg::opt(index_buffer),
                        Arg::U64(index_buffer_offset),
                        Arg::U64(instance_count),
                        Arg::I64(base_vertex),
                        Arg::U64(base_instance),
                    ],
                )?;
                draw(
                    self,
                    2,
                    IcbDraw {
                        primitive_type,
                        index_count,
                        index_type,
                        index_buffer_gpu_virtual_address: gpu_of(index_buffer, index_buffer_offset),
                        instance_count,
                        base_vertex: base_vertex.cast_unsigned(),
                        base_instance,
                        ..IcbDraw::default()
                    },
                );
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
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::U64(number_of_patch_control_points),
                        Arg::U64(patch_start),
                        Arg::U64(patch_count),
                        Arg::opt(patch_index_buffer),
                        Arg::U64(patch_index_buffer_offset),
                        Arg::U64(instance_count),
                        Arg::U64(base_instance),
                        Arg::opt(buffer),
                        Arg::U64(offset),
                        Arg::U64(instance_stride),
                    ],
                )?;
                draw(
                    self,
                    4,
                    IcbDraw {
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index_buffer_gpu_virtual_address: gpu_of(
                            patch_index_buffer,
                            patch_index_buffer_offset,
                        ),
                        instance_count,
                        base_instance,
                        gpu_virtual_address: gpu_of(buffer, offset),
                        instance_stride,
                        ..IcbDraw::default()
                    },
                );
            }
            IcbRenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::U64(primitive_type),
                        Arg::U64(vertex_start),
                        Arg::U64(vertex_count),
                        Arg::U64(instance_count),
                        Arg::U64(base_instance),
                    ],
                )?;
                draw(
                    self,
                    1,
                    IcbDraw {
                        primitive_type,
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                        ..IcbDraw::default()
                    },
                );
            }
            IcbRenderCmd::Reset => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| *m = IcbCommandModel::default());
            }
            IcbRenderCmd::SetBarrier => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| m.barrier = true);
            }
            IcbRenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![Arg::opt(buffer), Arg::U64(offset), Arg::U64(index)],
                )?;
                command(self, |m| {
                    m.fragment_buffers.insert(index, gpu_of(buffer, offset))
                });
            }
            IcbRenderCmd::SetRenderPipelineState { pipeline_state } => {
                self.act(Policy::Caught, sel, vec![Arg::opt(pipeline_state)])?;
                let pipeline = pipeline_state.map_or(0, |p| p.unique_id_value());
                command(self, |m| m.pipeline = pipeline);
            }
            IcbRenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![Arg::opt(buffer), Arg::U64(offset), Arg::U64(index)],
                )?;
                command(self, |m| {
                    m.vertex_buffers.insert(index, gpu_of(buffer, offset))
                });
            }
            IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::opt(buffer),
                        Arg::U64(offset),
                        Arg::U64(stride),
                        Arg::U64(index),
                    ],
                )?;
                command(self, |m| {
                    m.vertex_buffers.insert(index, gpu_of(buffer, offset));
                    m.vertex_strides.insert(index, stride);
                });
            }
        }
        Ok(())
    }

    fn command_type(&self) -> u64 {
        self.with_model(|m: &IcbCommandModel| m.command_type)
    }

    fn fragment_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getFragmentBufferAddressAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.fragment_buffers),
        )
    }

    fn pipeline_state_unique_identifier(&self) -> u64 {
        self.with_model(|m: &IcbCommandModel| m.pipeline)
    }

    fn vertex_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getVertexAttributeStrideAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.vertex_strides),
        )
    }

    fn vertex_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getVertexBufferAddressAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.vertex_buffers),
        )
    }

    fn has_barrier(&self) -> bool {
        self.with_model(|m: &IcbCommandModel| m.barrier)
    }

    fn draw_args(&self) -> Option<DrawArgs> {
        let f = drawn(self)?;
        Some(DrawArgs {
            primitive_type: f.primitive_type,
            vertex_start: f.vertex_start,
            vertex_count: f.vertex_count,
            instance_count: f.instance_count,
            base_instance: f.base_instance,
        })
    }

    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs> {
        let f = drawn(self)?;
        Some(DrawIndexedArgs {
            index_buffer_gpu_virtual_address: f.index_buffer_gpu_virtual_address,
            primitive_type: f.primitive_type,
            index_type: f.index_type,
            index_count: f.index_count,
            instance_count: f.instance_count,
            base_vertex: f.base_vertex,
            base_instance: f.base_instance,
        })
    }

    fn draw_patches_args(&self) -> Option<DrawPatchesArgs> {
        let f = drawn(self)?;
        Some(DrawPatchesArgs {
            patch_index_buffer_gpu_virtual_address: f.patch_index_buffer_gpu_virtual_address,
            number_of_patch_control_points: f.number_of_patch_control_points,
            patch_start: f.patch_start,
            patch_count: f.patch_count,
            instance_count: f.instance_count,
            base_instance: f.base_instance,
        })
    }

    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs> {
        let f = drawn(self)?;
        Some(DrawIndexedPatchesArgs {
            patch_index_buffer_gpu_virtual_address: f.patch_index_buffer_gpu_virtual_address,
            control_point_index_buffer_gpu_virtual_address: f
                .control_point_index_buffer_gpu_virtual_address,
            number_of_patch_control_points: f.number_of_patch_control_points,
            patch_start: f.patch_start,
            patch_count: f.patch_count,
            instance_count: f.instance_count,
            base_instance: f.base_instance,
        })
    }

    /// The scale is 0: the model keeps none.
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs> {
        let f = drawn(self)?;
        Some(TessellationFactorArgs {
            gpu_virtual_address: f.gpu_virtual_address,
            instance_stride: f.instance_stride,
            scale: 0.0,
        })
    }
}

impl IndirectComputeCommand<FakePlatform> for Fk<kind::IndirectComputeCommand> {
    fn encode(&self, cmd: IcbComputeCmd<'_, FakePlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        match cmd {
            IcbComputeCmd::ClearBarrier => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| m.barrier = false);
            }
            IcbComputeCmd::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::val(&threadgroups_per_grid),
                        Arg::val(&threads_per_threadgroup),
                    ],
                )?;
                command(self, |m| {
                    m.command_type = 32;
                    m.dispatch = Some([threadgroups_per_grid, threads_per_threadgroup]);
                });
            }
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::val(&threads_per_grid),
                        Arg::val(&threads_per_threadgroup),
                    ],
                )?;
                command(self, |m| {
                    m.command_type = 64;
                    m.dispatch = Some([threads_per_grid, threads_per_threadgroup]);
                });
            }
            IcbComputeCmd::Reset => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| *m = IcbCommandModel::default());
            }
            IcbComputeCmd::SetBarrier => {
                self.act(Policy::Caught, sel, vec![])?;
                command(self, |m| m.barrier = true);
            }
            IcbComputeCmd::SetComputePipelineState { pipeline_state } => {
                self.act(Policy::Caught, sel, vec![Arg::opt(pipeline_state)])?;
                let pipeline = pipeline_state.map_or(0, |p| p.unique_id_value());
                command(self, |m| m.pipeline = pipeline);
            }
            IcbComputeCmd::SetImageblockWidth { width, height } => {
                self.act(Policy::Caught, sel, vec![Arg::U64(width), Arg::U64(height)])?;
                command(self, |m| m.imageblock = [width, height]);
            }
            IcbComputeCmd::SetKernelBuffer {
                buffer,
                offset,
                index,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![Arg::opt(buffer), Arg::U64(offset), Arg::U64(index)],
                )?;
                command(self, |m| {
                    m.kernel_buffers.insert(index, gpu_of(buffer, offset))
                });
            }
            IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                self.act(
                    Policy::Caught,
                    sel,
                    vec![
                        Arg::opt(buffer),
                        Arg::U64(offset),
                        Arg::U64(stride),
                        Arg::U64(index),
                    ],
                )?;
                command(self, |m| {
                    m.kernel_buffers.insert(index, gpu_of(buffer, offset));
                    m.kernel_strides.insert(index, stride);
                });
            }
            IcbComputeCmd::SetStageInRegion { region } => {
                self.act(Policy::Caught, sel, vec![Arg::val(&region)])?;
                command(self, |m| m.stage_in_region = region);
            }
            IcbComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                self.act(Policy::Caught, sel, vec![Arg::U64(length), Arg::U64(index)])?;
                command(self, |m| m.threadgroup_memory.insert(index, length));
            }
        }
        Ok(())
    }

    fn command_type(&self) -> u64 {
        self.with_model(|m: &IcbCommandModel| m.command_type)
    }

    fn kernel_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getKernelAttributeStrideAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.kernel_strides),
        )
    }

    fn kernel_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getKernelBufferAddressAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.kernel_buffers),
        )
    }

    fn pipeline_state_unique_identifier(&self) -> u64 {
        self.with_model(|m: &IcbCommandModel| m.pipeline)
    }

    fn stage_in_region(&self) -> MtlRegion {
        self.with_model(|m: &IcbCommandModel| m.stage_in_region)
    }

    fn threadgroup_memory_length(&self, index: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "getThreadgroupMemoryLengthAtIndex:",
            vec![Arg::U64(index)],
            || at(self, index, |m| &m.threadgroup_memory),
        )
    }

    fn has_barrier(&self) -> bool {
        self.with_model(|m: &IcbCommandModel| m.barrier)
    }

    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs> {
        let [grid, group] = dispatched(self, 32)?;
        Some(DispatchThreadgroupsArgs {
            threadgroups_per_grid: grid,
            threads_per_threadgroup: group,
        })
    }

    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs> {
        let [grid, group] = dispatched(self, 64)?;
        Some(DispatchThreadsArgs {
            threads_per_grid: grid,
            threads_per_threadgroup: group,
        })
    }

    /// Never nil: the model's image block size, zeros when none was set.
    fn image_block_args(&self) -> Option<ImageBlockArgs> {
        let [width, height] = self.with_model(|m: &IcbCommandModel| m.imageblock);
        Some(ImageBlockArgs { width, height })
    }
}

/// The GPU address of `offset` into `b`, 0 for nil.
fn gpu_of(b: Option<&Fk<kind::Buffer>>, offset: u64) -> u64 {
    b.map_or(0, |b| {
        let base = match &b.state().model {
            Model::Buffer(m) => m.gpu_address,
            _ => 0,
        };
        base.wrapping_add(offset)
    })
}

/// Changes command `c`'s model with `f`.
fn command<K: Kind, T>(c: &Fk<K>, f: impl FnOnce(&mut IcbCommandModel) -> T) {
    c.with_model_mut(|m: &mut IcbCommandModel| {
        let _ = f(m);
    });
}

/// Records a draw of `command_type` with `draw`'s fields on `c`.
fn draw<K: Kind>(c: &Fk<K>, command_type: u64, draw: IcbDraw) {
    command(c, |m| {
        m.command_type = command_type;
        m.draw = Some(draw);
    });
}

/// `c`'s last draw: nil when no draw was encoded.
fn drawn<K: Kind>(c: &Fk<K>) -> Option<IcbDraw> {
    c.with_model(|m: &IcbCommandModel| m.draw)
}

/// `c`'s last dispatch, its two sizes: nil when the command is not a `command_type` dispatch.
fn dispatched<K: Kind>(c: &Fk<K>, command_type: u64) -> Option<[MtlSize; 2]> {
    c.with_model(|m: &IcbCommandModel| m.dispatch.filter(|_| m.command_type == command_type))
}

/// The value `map` of `c` holds at `index`, 0 when none was set.
fn at<K: Kind>(
    c: &Fk<K>,
    index: u64,
    map: impl FnOnce(&IcbCommandModel) -> &BTreeMap<u64, u64>,
) -> u64 {
    c.with_model(|m: &IcbCommandModel| map(m).get(&index).copied().unwrap_or(0))
}

impl Fk<kind::IndirectRenderCommand> {
    /// Makes `getCommandType` answer `command_type`.
    pub fn set_command_type(&self, command_type: u64) {
        command(self, |m| m.command_type = command_type);
    }
}

impl Fk<kind::IndirectComputeCommand> {
    /// Makes `getCommandType` answer `command_type`.
    pub fn set_command_type(&self, command_type: u64) {
        command(self, |m| m.command_type = command_type);
    }
}
