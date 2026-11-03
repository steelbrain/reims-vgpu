//! Indirect command buffers: `ICB_CREATE`, answered `ICB_STATE` with the platform's
//! `gpuResourceID`, and `ICB_WRITE`, the CPU's encodes into one.
//!
//! The platform's buffer is the one copy of the commands. The guest's CPU encodes reach it as
//! records ([`reims_graphics::icb`]) on ring 0, which the guest posts before any command buffer
//! that may execute them, and a kernel's encodes write it on the GPU, through the
//! `gpuResourceID` the guest's argument buffers hold. So a command buffer executes the commands
//! as both sides last wrote them, as Metal's does.
//!
//! Each record is carried out in its own refusal scope: one that names an object the sender does
//! not hold, of the wrong kind, a command past the buffer's size, or that the platform refuses
//! (the Vulkan platform ends the paravirt process where its runtime refuses a setter) is skipped
//! alone, and the message names how many were in one warning. A record whose framing is broken
//! ends the message there.

use reims_graphics::icb::*;
use reims_graphics::wire::{
    RGFX_EVT_ICB_STATE, RGFX_ICB_INHERIT_BLEND_COLOR, RGFX_ICB_INHERIT_BUFFERS,
    RGFX_ICB_INHERIT_CULL_MODE, RGFX_ICB_INHERIT_DEPTH_BIAS, RGFX_ICB_INHERIT_DEPTH_CLIP_MODE,
    RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE, RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS,
    RGFX_ICB_INHERIT_FRONT_FACING_WINDING, RGFX_ICB_INHERIT_PIPELINE_STATE,
    RGFX_ICB_INHERIT_SCISSOR_RECTS, RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES,
    RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE, RGFX_ICB_INHERIT_VIEWPORTS,
    RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE, RGFX_ICB_SUPPORT_RAY_TRACING,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RgfxIcbCreate, RgfxIcbState,
    RgfxIcbWrite, RgfxMsgHeader,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::icb::{
    IcbComputeCmd, IcbRenderCmd, IndirectCommandBuffer, IndirectComputeCommand,
    IndirectRenderCommand,
};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::values::IcbDesc;
use reims_vgpu_core::raise::RaiseResult;
use zerocopy::FromBytes;

use super::{DecodeError, Entry, GraphicsDecoder, ObjHandle, op_fault, read, refusing};

/// The platform's descriptor of an `ICB_CREATE`: every field the guest's descriptor answered.
pub(super) fn icb_desc(m: &RgfxIcbCreate) -> IcbDesc {
    let flag = |bit: u32| m.flags & bit != 0;
    IcbDesc {
        command_types: m.command_types.into(),
        max_vertex_buffer_bind_count: m.max_vertex_buffer_bind_count.into(),
        max_fragment_buffer_bind_count: m.max_fragment_buffer_bind_count.into(),
        max_kernel_buffer_bind_count: m.max_kernel_buffer_bind_count.into(),
        max_object_buffer_bind_count: m.max_object_buffer_bind_count.into(),
        max_mesh_buffer_bind_count: m.max_mesh_buffer_bind_count.into(),
        max_kernel_threadgroup_memory_bind_count: m.max_kernel_threadgroup_memory_bind_count.into(),
        max_object_threadgroup_memory_bind_count: m.max_object_threadgroup_memory_bind_count.into(),
        inherit_pipeline_state: flag(RGFX_ICB_INHERIT_PIPELINE_STATE),
        inherit_buffers: flag(RGFX_ICB_INHERIT_BUFFERS),
        support_ray_tracing: flag(RGFX_ICB_SUPPORT_RAY_TRACING),
        support_dynamic_attribute_stride: flag(RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE),
        max_scissor_rect_count: m.max_scissor_rect_count.into(),
        max_viewport_count: m.max_viewport_count.into(),
        inherit_stencil_reference_values: flag(RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES),
        inherit_depth_test_bounds: flag(RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS),
        inherit_scissor_rects: flag(RGFX_ICB_INHERIT_SCISSOR_RECTS),
        inherit_viewports: flag(RGFX_ICB_INHERIT_VIEWPORTS),
        inherit_blend_color: flag(RGFX_ICB_INHERIT_BLEND_COLOR),
        allow_override_render_states: m.allow_override_render_states.into(),
        inherit_depth_stencil_state: flag(RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE),
        inherit_depth_bias: flag(RGFX_ICB_INHERIT_DEPTH_BIAS),
        inherit_depth_clip_mode: flag(RGFX_ICB_INHERIT_DEPTH_CLIP_MODE),
        inherit_cull_mode: flag(RGFX_ICB_INHERIT_CULL_MODE),
        inherit_front_facing_winding: flag(RGFX_ICB_INHERIT_FRONT_FACING_WINDING),
        inherit_triangle_fill_mode: flag(RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE),
        // The host's device numbers its buffers itself, which the answer gives the guest.
        resource_index: None,
    }
}

/// A record's body, or an op fault for one too short.
fn body<T: FromBytes>(op: u16, bytes: &[u8]) -> RaiseResult<T> {
    match T::read_from_prefix(bytes) {
        Ok((t, _)) => Ok(t),
        Err(_) => Err(op_fault(format_args!(
            "icb record {op:#04x}: body too short"
        ))),
    }
}

/// A dispatch's grid and threadgroup sizes.
fn sizes(d: &RgfxIcbDispatch) -> (MtlSize, MtlSize) {
    let size = |s: [u32; 3]| MtlSize {
        width: s[0].into(),
        height: s[1].into(),
        depth: s[2].into(),
    };
    (size(d.grid), size(d.threadgroup))
}

/// Whether `op` is a render command's record (a reset range included).
fn renders(op: u16) -> bool {
    (RGFX_ICB_OP_RENDER_PIPELINE..=RGFX_ICB_OP_RENDER_RESET_RANGE).contains(&op)
}

/// Whether `op` is a compute command's record (a reset range included).
fn computes(op: u16) -> bool {
    (RGFX_ICB_OP_COMPUTE_PIPELINE..=RGFX_ICB_OP_COMPUTE_RESET_RANGE).contains(&op)
}

/// The command a run of records is carried out on: consecutive records of one command, a setter
/// run and its draw, take it from the buffer once.
enum Command<P: Platform> {
    None,
    Render(u64, P::IndirectRenderCommand),
    Compute(u64, P::IndirectComputeCommand),
}

impl<P: Platform> GraphicsDecoder<P> {
    /// An `ICB_CREATE`'s answer (`ICB_STATE`): the buffer made, stored under the message's id
    /// before the answer, so an `ICB_WRITE` or a command buffer of its process, which names it
    /// only after the answer, finds it; with its `gpuResourceID`. [`RGFX_STATUS_REFUSED`] with
    /// 0 where the sender's context failed or the platform makes none, and
    /// [`RGFX_STATUS_INVALID_ARGUMENT`] for a short create. `None` for a message too short to
    /// carry its seqno, which no one can be answered by.
    pub(crate) fn icb_state(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Option<RgfxIcbState>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(None);
        };
        let answer = |status, gpu_resource_id| RgfxIcbState {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_ICB_STATE,
                size: size_of::<RgfxIcbState>() as u32,
                seqno: header.seqno,
            },
            status,
            reserved_14: 0,
            gpu_resource_id,
        };
        if self.context_failed() {
            return Ok(Some(answer(RGFX_STATUS_REFUSED, 0)));
        }
        Ok(Some(match self.new_icb(device, message)? {
            Ok(id) => answer(RGFX_STATUS_OK, id),
            Err(DecodeError::Malformed) => answer(RGFX_STATUS_INVALID_ARGUMENT, 0),
            Err(_) => answer(RGFX_STATUS_REFUSED, 0),
        }))
    }

    /// The buffer of an `ICB_CREATE`, stored under its id; its `gpuResourceID`.
    pub(super) fn new_icb(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<u64, DecodeError>> {
        let Some(m) = read::<RgfxIcbCreate>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let made =
            device.new_indirect_command_buffer(&icb_desc(&m), m.max_command_count, m.options)?;
        let Some(icb) = made else {
            log::warn!(
                target: "reims_vgpu::graphics",
                "indirect command buffer {:#x} not made: the platform answered nil for {} \
                 commands of types {:#x}, options {:#x}",
                m.obj_id,
                m.max_command_count,
                m.command_types,
                m.options
            );
            return Ok(Err(DecodeError::Refused));
        };
        let id = icb.gpu_resource_id();
        self.store(m.obj_id, Entry::of(ObjHandle::Icb(icb)));
        Ok(Ok(id))
    }

    /// An `ICB_WRITE`: each record carried out on the buffer, in order, each refused alone.
    /// Refused as a whole where it names no indirect command buffer of the sender's, and
    /// malformed where a record's framing is broken, after the records before it.
    pub(super) fn icb_write(&self, message: &[u8]) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxIcbWrite>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let Some(ObjHandle::Icb(icb)) = self.named(m.obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        let size = icb.size();
        let end = (m.header.size as usize).min(message.len());
        let mut at = size_of::<RgfxIcbWrite>();
        let mut command = Command::None;
        let (mut records, mut refused) = (0u64, 0u64);
        let mut first: Option<(u16, u32, String)> = None;
        let mut framing = Ok(());
        while at < end {
            let rest = &message[at..end];
            let Some(rec) = read::<RgfxIcbRecord>(rest) else {
                framing = Err(DecodeError::Malformed);
                break;
            };
            let len = usize::from(rec.size);
            if len < size_of::<RgfxIcbRecord>() || len % 8 != 0 || len > rest.len() {
                framing = Err(DecodeError::Malformed);
                break;
            }
            let bytes = &rest[size_of::<RgfxIcbRecord>()..len];
            records += 1;
            let why = match refusing(|| self.icb_record(icb, size, &mut command, &rec, bytes)) {
                Ok(Ok(())) => None,
                Ok(Err(e)) => {
                    // A failed lookup leaves no command to carry the next record on.
                    command = Command::None;
                    Some(format!("{e:?}"))
                }
                Err(raise) => {
                    command = Command::None;
                    Some(raise.to_string())
                }
            };
            if let Some(why) = why {
                refused += 1;
                first.get_or_insert((rec.op, rec.command_index, why));
            }
            at += len;
        }
        if let Some((op, index, why)) = first {
            log::warn!(
                target: "reims_vgpu::graphics",
                "indirect command buffer {:#x}: {refused} of {records} records not carried out; \
                 the first, op {op:#04x} at command {index}: {why}",
                m.obj_id
            );
        }
        Ok(framing)
    }

    /// One record of an `ICB_WRITE` on `icb` of `size` commands, its command taken from
    /// `command` where the record before it was of the same one.
    fn icb_record(
        &self,
        icb: &P::Icb,
        size: u64,
        command: &mut Command<P>,
        rec: &RgfxIcbRecord,
        bytes: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let op = rec.op;
        let index = u64::from(rec.command_index);
        if op == RGFX_ICB_OP_RENDER_RESET_RANGE || op == RGFX_ICB_OP_COMPUTE_RESET_RANGE {
            // The guest's `resetWithRange:`, one platform call for the commands of the range
            // inside the size: its cost is the platform's for the commands it clears, not a
            // command taken and reset one at a time.
            let r: RgfxIcbRange = body(op, bytes)?;
            let stop = index.saturating_add(r.length.into()).min(size);
            *command = Command::None;
            if index < stop {
                icb.reset_with_range(index, stop - index)?;
            }
            return Ok(Ok(()));
        }
        if index >= size {
            return Ok(Err(DecodeError::BadReference));
        }
        if renders(op) {
            if !matches!(command, Command::Render(i, _) if *i == index) {
                let Some(c) = icb.render_command(index)? else {
                    return Ok(Err(DecodeError::Refused));
                };
                *command = Command::Render(index, c);
            }
            let Command::Render(_, c) = command else {
                unreachable!("a render command was just taken");
            };
            self.render_record(c, op, bytes)
        } else if computes(op) {
            if !matches!(command, Command::Compute(i, _) if *i == index) {
                let Some(c) = icb.compute_command(index)? else {
                    return Ok(Err(DecodeError::Refused));
                };
                *command = Command::Compute(index, c);
            }
            let Command::Compute(_, c) = command else {
                unreachable!("a compute command was just taken");
            };
            self.compute_record(c, op, bytes)
        } else {
            Ok(Err(DecodeError::UnknownCommand))
        }
    }

    /// A render command's record.
    fn render_record(
        &self,
        c: &P::IndirectRenderCommand,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let cmd = match op {
            RGFX_ICB_OP_RENDER_PIPELINE => {
                let m: RgfxIcbObject = body(op, bytes)?;
                IcbRenderCmd::SetRenderPipelineState {
                    pipeline_state: self.get_render_pipeline(m.obj_id)?,
                }
            }
            RGFX_ICB_OP_VERTEX_BUFFER | RGFX_ICB_OP_FRAGMENT_BUFFER => {
                let m: RgfxIcbBuffer = body(op, bytes)?;
                let buffer = self.get_buffer(m.obj_id)?;
                let (offset, index) = (m.offset, m.index.into());
                if op == RGFX_ICB_OP_VERTEX_BUFFER {
                    IcbRenderCmd::SetVertexBuffer {
                        buffer,
                        offset,
                        index,
                    }
                } else {
                    IcbRenderCmd::SetFragmentBuffer {
                        buffer,
                        offset,
                        index,
                    }
                }
            }
            RGFX_ICB_OP_VERTEX_BUFFER_STRIDE => {
                let m: RgfxIcbBufferStride = body(op, bytes)?;
                IcbRenderCmd::SetVertexBufferAttributeStride {
                    buffer: self.get_buffer(m.obj_id)?,
                    offset: m.offset,
                    stride: m.stride,
                    index: m.index.into(),
                }
            }
            RGFX_ICB_OP_DRAW => {
                let m: RgfxIcbDraw = body(op, bytes)?;
                IcbRenderCmd::DrawPrimitives {
                    primitive_type: m.primitive_type.into(),
                    vertex_start: m.vertex_start.into(),
                    vertex_count: m.vertex_count.into(),
                    instance_count: m.instance_count.into(),
                    base_instance: m.base_instance.into(),
                }
            }
            RGFX_ICB_OP_DRAW_INDEXED => {
                let m: RgfxIcbDrawIndexed = body(op, bytes)?;
                IcbRenderCmd::DrawIndexedPrimitives {
                    primitive_type: m.primitive_type.into(),
                    index_count: m.index_count.into(),
                    index_type: m.index_type.into(),
                    index_buffer: self.get_buffer(m.index_buffer_obj_id)?,
                    index_buffer_offset: m.index_buffer_offset,
                    instance_count: m.instance_count.into(),
                    base_vertex: m.base_vertex.into(),
                    base_instance: m.base_instance.into(),
                }
            }
            RGFX_ICB_OP_DRAW_PATCHES => {
                let m: RgfxIcbDrawPatches = body(op, bytes)?;
                IcbRenderCmd::DrawPatches {
                    number_of_patch_control_points: m.number_of_patch_control_points.into(),
                    patch_start: m.patch_start.into(),
                    patch_count: m.patch_count.into(),
                    patch_index_buffer: self.get_buffer(m.patch_index_buffer_obj_id)?,
                    patch_index_buffer_offset: m.patch_index_buffer_offset,
                    instance_count: m.instance_count.into(),
                    base_instance: m.base_instance.into(),
                    buffer: self.get_buffer(m.tessellation_factor_buffer_obj_id)?,
                    offset: m.tessellation_factor_buffer_offset,
                    instance_stride: m.instance_stride.into(),
                }
            }
            RGFX_ICB_OP_DRAW_INDEXED_PATCHES => {
                let m: RgfxIcbDrawIndexedPatches = body(op, bytes)?;
                let p = m.patches;
                IcbRenderCmd::DrawIndexedPatches {
                    number_of_patch_control_points: p.number_of_patch_control_points.into(),
                    patch_start: p.patch_start.into(),
                    patch_count: p.patch_count.into(),
                    patch_index_buffer: self.get_buffer(p.patch_index_buffer_obj_id)?,
                    patch_index_buffer_offset: p.patch_index_buffer_offset,
                    control_point_index_buffer: self
                        .get_buffer(m.control_point_index_buffer_obj_id)?,
                    control_point_index_buffer_offset: m.control_point_index_buffer_offset,
                    instance_count: p.instance_count.into(),
                    base_instance: p.base_instance.into(),
                    buffer: self.get_buffer(p.tessellation_factor_buffer_obj_id)?,
                    offset: p.tessellation_factor_buffer_offset,
                    instance_stride: p.instance_stride.into(),
                }
            }
            RGFX_ICB_OP_RENDER_SET_BARRIER => IcbRenderCmd::SetBarrier,
            RGFX_ICB_OP_RENDER_CLEAR_BARRIER => IcbRenderCmd::ClearBarrier,
            RGFX_ICB_OP_RENDER_RESET => IcbRenderCmd::Reset,
            _ => return Ok(Err(DecodeError::UnknownCommand)),
        };
        c.encode(cmd)?;
        Ok(Ok(()))
    }

    /// A compute command's record.
    fn compute_record(
        &self,
        c: &P::IndirectComputeCommand,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let cmd = match op {
            RGFX_ICB_OP_COMPUTE_PIPELINE => {
                let m: RgfxIcbObject = body(op, bytes)?;
                IcbComputeCmd::SetComputePipelineState {
                    pipeline_state: self.get_compute_pipeline(m.obj_id)?,
                }
            }
            RGFX_ICB_OP_KERNEL_BUFFER => {
                let m: RgfxIcbBuffer = body(op, bytes)?;
                IcbComputeCmd::SetKernelBuffer {
                    buffer: self.get_buffer(m.obj_id)?,
                    offset: m.offset,
                    index: m.index.into(),
                }
            }
            RGFX_ICB_OP_KERNEL_BUFFER_STRIDE => {
                let m: RgfxIcbBufferStride = body(op, bytes)?;
                IcbComputeCmd::SetKernelBufferAttributeStride {
                    buffer: self.get_buffer(m.obj_id)?,
                    offset: m.offset,
                    stride: m.stride,
                    index: m.index.into(),
                }
            }
            RGFX_ICB_OP_DISPATCH_THREADGROUPS => {
                let m: RgfxIcbDispatch = body(op, bytes)?;
                let (grid, group) = sizes(&m);
                IcbComputeCmd::ConcurrentDispatchThreadgroups {
                    threadgroups_per_grid: grid,
                    threads_per_threadgroup: group,
                }
            }
            RGFX_ICB_OP_DISPATCH_THREADS => {
                let m: RgfxIcbDispatch = body(op, bytes)?;
                let (grid, group) = sizes(&m);
                IcbComputeCmd::ConcurrentDispatchThreads {
                    threads_per_grid: grid,
                    threads_per_threadgroup: group,
                }
            }
            RGFX_ICB_OP_THREADGROUP_MEMORY => {
                let m: RgfxIcbThreadgroupMemory = body(op, bytes)?;
                IcbComputeCmd::SetThreadgroupMemoryLength {
                    length: m.length.into(),
                    index: m.index.into(),
                }
            }
            RGFX_ICB_OP_STAGE_IN_REGION => {
                let m: RgfxIcbRegion = body(op, bytes)?;
                IcbComputeCmd::SetStageInRegion {
                    region: MtlRegion {
                        origin: MtlOrigin {
                            x: m.origin[0].into(),
                            y: m.origin[1].into(),
                            z: m.origin[2].into(),
                        },
                        size: MtlSize {
                            width: m.size[0].into(),
                            height: m.size[1].into(),
                            depth: m.size[2].into(),
                        },
                    },
                }
            }
            RGFX_ICB_OP_IMAGEBLOCK => {
                let m: RgfxIcbImageblock = body(op, bytes)?;
                IcbComputeCmd::SetImageblockWidth {
                    width: m.width.into(),
                    height: m.height.into(),
                }
            }
            RGFX_ICB_OP_COMPUTE_SET_BARRIER => IcbComputeCmd::SetBarrier,
            RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER => IcbComputeCmd::ClearBarrier,
            RGFX_ICB_OP_COMPUTE_RESET => IcbComputeCmd::Reset,
            _ => return Ok(Err(DecodeError::UnknownCommand)),
        };
        c.encode(cmd)?;
        Ok(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::wire::{
        RGFX_CMD_BUFFER_CREATE, RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_WRITE, RgfxBufferCreate,
    };
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::testkit::{Arg, FakePlatform, Step};
    use zerocopy::{Immutable, IntoBytes};

    use super::super::tests::{compute_create, pipeline_decoder, whole_render_create};
    use super::*;

    type Dec = GraphicsDecoder<FakePlatform>;

    /// `MTLIndirectCommandTypeDraw | DrawIndexed | DrawPatches | DrawIndexedPatches`.
    const RENDER_TYPES: u32 = 0xf;
    /// `MTLIndirectCommandTypeConcurrentDispatch | ConcurrentDispatchThreads`.
    const COMPUTE_TYPES: u32 = 0x60;
    const RENDER_PIPELINE: u64 = 100;
    const COMPUTE_PIPELINE: u64 = 101;
    const RENDER_ICB: u64 = 50;
    const COMPUTE_ICB: u64 = 51;

    fn create(obj_id: u64, seqno: u64, command_types: u32, count: u64) -> RgfxIcbCreate {
        RgfxIcbCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_CREATE,
                size: size_of::<RgfxIcbCreate>() as u32,
                seqno,
            },
            obj_id,
            max_command_count: count,
            options: 0x200,
            command_types,
            flags: RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE | RGFX_ICB_INHERIT_CULL_MODE,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 1,
            max_kernel_buffer_bind_count: 3,
            max_kernel_threadgroup_memory_bind_count: 31,
            max_scissor_rect_count: 1,
            max_viewport_count: 1,
            ..RgfxIcbCreate::default()
        }
    }

    /// A record: its header, then `body`.
    fn record<T: IntoBytes + Immutable>(op: u16, index: u32, body: &T) -> Vec<u8> {
        let bytes = body.as_bytes();
        let mut r = RgfxIcbRecord {
            op,
            size: (size_of::<RgfxIcbRecord>() + bytes.len()) as u16,
            command_index: index,
        }
        .as_bytes()
        .to_vec();
        r.extend_from_slice(bytes);
        r
    }

    /// A record with no body.
    fn bare(op: u16, index: u32) -> Vec<u8> {
        record(op, index, &[0u8; 0])
    }

    /// An `ICB_WRITE` of `records` into `icb`.
    fn write(icb: u64, records: &[Vec<u8>]) -> Vec<u8> {
        let len = size_of::<RgfxIcbWrite>() + records.iter().map(Vec::len).sum::<usize>();
        let mut m = RgfxIcbWrite {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_WRITE,
                size: len as u32,
                seqno: 0,
            },
            obj_id: icb,
        }
        .as_bytes()
        .to_vec();
        for r in records {
            m.extend_from_slice(r);
        }
        m
    }

    /// A decoder holding a render pipeline, a compute pipeline, buffers 2 and 3, and a render
    /// and a compute indirect command buffer of 8 commands each.
    fn setup() -> (Arc<FakePlatform>, Dec) {
        let (p, mut dec, _) = pipeline_decoder();
        let device = p.device();
        assert_eq!(
            dec.decode(&device, &whole_render_create(RENDER_PIPELINE, true))
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, &compute_create(COMPUTE_PIPELINE))
                .unwrap(),
            Ok(())
        );
        for id in [2, 3] {
            let b = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                length: 65536,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, b.as_bytes()).unwrap(), Ok(()));
        }
        for (id, types) in [(RENDER_ICB, RENDER_TYPES), (COMPUTE_ICB, COMPUTE_TYPES)] {
            let m = create(id, 1, types, 8);
            assert_eq!(dec.decode(&device, m.as_bytes()).unwrap(), Ok(()));
        }
        dec.end_message();
        (p, dec)
    }

    fn buffer(dec: &Dec, id: u64) -> Arg {
        Arg::Obj(Some(dec.require_buffer(id).unwrap().id()))
    }

    /// The calls since `mark`, as selector and arguments.
    fn calls(p: &FakePlatform, mark: u64) -> Vec<(&'static str, Vec<Arg>)> {
        p.call_log()
            .since(mark)
            .into_iter()
            .map(|c| (c.sel, c.args))
            .collect()
    }

    /// A create carries the guest's whole descriptor to the platform's creator and is answered
    /// with the buffer's `gpuResourceID`, the buffer stored before the answer; a refused one with
    /// 0, a short one invalid, and a failed context's refused, storing nothing.
    #[test]
    fn a_create_carries_its_descriptor_and_answers_the_platforms_id() {
        let (p, mut dec, _) = pipeline_decoder();
        let device = p.device();
        let m = create(21, 4, RENDER_TYPES, 16);
        let mark = p.call_log().mark();
        let answer = dec.icb_state(&device, m.as_bytes()).unwrap().unwrap();
        assert_eq!(answer.header.r#type, RGFX_EVT_ICB_STATE);
        assert_eq!(answer.header.size as usize, size_of::<RgfxIcbState>());
        assert_eq!((answer.header.seqno, answer.status), (4, RGFX_STATUS_OK));
        assert_ne!(answer.gpu_resource_id, 0);
        let expected = IcbDesc {
            command_types: 0xf,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 1,
            max_kernel_buffer_bind_count: 3,
            max_kernel_threadgroup_memory_bind_count: 31,
            max_scissor_rect_count: 1,
            max_viewport_count: 1,
            inherit_depth_stencil_state: true,
            inherit_cull_mode: true,
            ..IcbDesc::default()
        };
        assert_eq!(
            calls(&p, mark),
            vec![(
                "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
                vec![Arg::val(&expected), Arg::U64(16), Arg::U64(0x200)]
            )]
        );
        let Some(ObjHandle::Icb(stored)) = dec.named(21) else {
            panic!("no indirect command buffer under 21");
        };
        assert_eq!(stored.gpu_resource_id(), answer.gpu_resource_id);
        assert_eq!(stored.size(), 16);
        dec.end_message();
        let second = dec
            .icb_state(&device, create(22, 5, COMPUTE_TYPES, 4).as_bytes())
            .unwrap()
            .unwrap();
        assert_ne!(second.gpu_resource_id, answer.gpu_resource_id);

        p.script().push(
            "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
            Step::Nil,
        );
        let refused = dec
            .icb_state(&device, create(23, 6, RENDER_TYPES, 4).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(
            (
                refused.header.seqno,
                refused.status,
                refused.gpu_resource_id
            ),
            (6, RGFX_STATUS_REFUSED, 0)
        );
        assert!(dec.named(23).is_none());
        let short = create(24, 7, RENDER_TYPES, 4);
        let short = &short.as_bytes()[..size_of::<RgfxMsgHeader>() + 8];
        let invalid = dec.icb_state(&device, short).unwrap().unwrap();
        assert_eq!(
            (invalid.header.seqno, invalid.status),
            (7, RGFX_STATUS_INVALID_ARGUMENT)
        );
        assert!(dec.icb_state(&device, &[0u8; 8]).unwrap().is_none());
        dec.set_context(Some(0));
        let contexts = dec.failed_contexts();
        assert!(contexts.fail(0, contexts.resets()));
        let failed = dec
            .icb_state(&device, create(25, 9, RENDER_TYPES, 4).as_bytes())
            .unwrap()
            .unwrap();
        assert_eq!(
            (failed.header.seqno, failed.status, failed.gpu_resource_id),
            (9, RGFX_STATUS_REFUSED, 0)
        );
        assert!(dec.named(25).is_none());
    }

    /// Every render record drives its setter on its command, in the order the guest made them,
    /// with its objects resolved and its counts widened; consecutive records of one command take
    /// it once; a reset of a range is one call for the commands of it inside the buffer's size.
    #[test]
    fn render_records_drive_their_setters_in_order() {
        let (p, mut dec) = setup();
        let device = p.device();
        let pipeline = Arg::Obj(Some(
            dec.require_render_pipeline(RENDER_PIPELINE).unwrap().id(),
        ));
        let (b2, b3) = (buffer(&dec, 2), buffer(&dec, 3));
        let patches = RgfxIcbDrawPatches {
            number_of_patch_control_points: 4,
            patch_start: 1,
            patch_count: 2,
            instance_count: 3,
            base_instance: 4,
            instance_stride: 12,
            patch_index_buffer_obj_id: 0,
            patch_index_buffer_offset: 0,
            tessellation_factor_buffer_obj_id: 3,
            tessellation_factor_buffer_offset: 48,
        };
        let records = vec![
            record(
                RGFX_ICB_OP_RENDER_PIPELINE,
                0,
                &RgfxIcbObject {
                    obj_id: RENDER_PIPELINE,
                },
            ),
            record(
                RGFX_ICB_OP_VERTEX_BUFFER,
                0,
                &RgfxIcbBuffer {
                    obj_id: 2,
                    offset: 16,
                    index: 1,
                    reserved_14: 0,
                },
            ),
            record(
                RGFX_ICB_OP_FRAGMENT_BUFFER,
                0,
                &RgfxIcbBuffer {
                    obj_id: 3,
                    offset: 32,
                    index: 0,
                    reserved_14: 0,
                },
            ),
            record(
                RGFX_ICB_OP_DRAW,
                0,
                &RgfxIcbDraw {
                    primitive_type: 4,
                    vertex_start: 1,
                    vertex_count: 4,
                    instance_count: 2,
                    base_instance: 7,
                    reserved_14: 0,
                },
            ),
            record(
                RGFX_ICB_OP_VERTEX_BUFFER_STRIDE,
                1,
                &RgfxIcbBufferStride {
                    obj_id: 0,
                    offset: 8,
                    stride: 24,
                    index: 0,
                    reserved_1c: 0,
                },
            ),
            record(
                RGFX_ICB_OP_DRAW_INDEXED,
                1,
                &RgfxIcbDrawIndexed {
                    primitive_type: 3,
                    index_count: 6,
                    index_type: 1,
                    instance_count: 1,
                    index_buffer_obj_id: 2,
                    index_buffer_offset: 64,
                    base_vertex: -4,
                    base_instance: 0,
                },
            ),
            record(RGFX_ICB_OP_DRAW_PATCHES, 2, &patches),
            record(
                RGFX_ICB_OP_DRAW_INDEXED_PATCHES,
                3,
                &RgfxIcbDrawIndexedPatches {
                    patches: RgfxIcbDrawPatches {
                        patch_index_buffer_obj_id: 2,
                        patch_index_buffer_offset: 8,
                        ..patches
                    },
                    control_point_index_buffer_obj_id: 3,
                    control_point_index_buffer_offset: 16,
                },
            ),
            bare(RGFX_ICB_OP_RENDER_SET_BARRIER, 3),
            bare(RGFX_ICB_OP_RENDER_CLEAR_BARRIER, 3),
            bare(RGFX_ICB_OP_RENDER_RESET, 4),
            record(
                RGFX_ICB_OP_RENDER_RESET_RANGE,
                6,
                &RgfxIcbRange {
                    length: 5,
                    reserved_04: 0,
                },
            ),
        ];
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &records)).unwrap(),
            Ok(())
        );
        let at = |i| ("indirectRenderCommandAtIndex:", vec![Arg::U64(i)]);
        assert_eq!(
            calls(&p, mark),
            vec![
                at(0),
                ("setRenderPipelineState:", vec![pipeline]),
                (
                    "setVertexBuffer:offset:atIndex:",
                    vec![b2.clone(), Arg::U64(16), Arg::U64(1)]
                ),
                (
                    "setFragmentBuffer:offset:atIndex:",
                    vec![b3.clone(), Arg::U64(32), Arg::U64(0)]
                ),
                (
                    "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                    vec![
                        Arg::U64(4),
                        Arg::U64(1),
                        Arg::U64(4),
                        Arg::U64(2),
                        Arg::U64(7)
                    ]
                ),
                at(1),
                (
                    "setVertexBuffer:offset:attributeStride:atIndex:",
                    vec![Arg::Obj(None), Arg::U64(8), Arg::U64(24), Arg::U64(0)]
                ),
                (
                    "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:",
                    vec![
                        Arg::U64(3),
                        Arg::U64(6),
                        Arg::U64(1),
                        b2.clone(),
                        Arg::U64(64),
                        Arg::U64(1),
                        Arg::I64(-4),
                        Arg::U64(0)
                    ]
                ),
                at(2),
                (
                    "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:",
                    vec![
                        Arg::U64(4),
                        Arg::U64(1),
                        Arg::U64(2),
                        Arg::Obj(None),
                        Arg::U64(0),
                        Arg::U64(3),
                        Arg::U64(4),
                        b3.clone(),
                        Arg::U64(48),
                        Arg::U64(12)
                    ]
                ),
                at(3),
                (
                    "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride:",
                    vec![
                        Arg::U64(4),
                        Arg::U64(1),
                        Arg::U64(2),
                        b2,
                        Arg::U64(8),
                        b3.clone(),
                        Arg::U64(16),
                        Arg::U64(3),
                        Arg::U64(4),
                        b3,
                        Arg::U64(48),
                        Arg::U64(12)
                    ]
                ),
                ("setBarrier", vec![]),
                ("clearBarrier", vec![]),
                at(4),
                ("reset", vec![]),
                ("resetWithRange:", vec![Arg::U64(6), Arg::U64(2)]),
            ]
        );
    }

    /// Every compute record drives its setter on its command, in order.
    #[test]
    fn compute_records_drive_their_setters_in_order() {
        let (p, mut dec) = setup();
        let device = p.device();
        let pipeline = Arg::Obj(Some(
            dec.require_compute_pipeline(COMPUTE_PIPELINE).unwrap().id(),
        ));
        let size = |w, h, d| {
            Arg::val(&MtlSize {
                width: w,
                height: h,
                depth: d,
            })
        };
        let records = vec![
            record(
                RGFX_ICB_OP_COMPUTE_PIPELINE,
                2,
                &RgfxIcbObject {
                    obj_id: COMPUTE_PIPELINE,
                },
            ),
            record(
                RGFX_ICB_OP_KERNEL_BUFFER,
                2,
                &RgfxIcbBuffer {
                    obj_id: 2,
                    offset: 256,
                    index: 0,
                    reserved_14: 0,
                },
            ),
            record(
                RGFX_ICB_OP_KERNEL_BUFFER_STRIDE,
                2,
                &RgfxIcbBufferStride {
                    obj_id: 3,
                    offset: 0,
                    stride: 16,
                    index: 1,
                    reserved_1c: 0,
                },
            ),
            record(
                RGFX_ICB_OP_THREADGROUP_MEMORY,
                2,
                &RgfxIcbThreadgroupMemory {
                    length: 256,
                    index: 0,
                },
            ),
            record(
                RGFX_ICB_OP_STAGE_IN_REGION,
                2,
                &RgfxIcbRegion {
                    origin: [1, 2, 3],
                    size: [4, 5, 6],
                },
            ),
            record(
                RGFX_ICB_OP_IMAGEBLOCK,
                2,
                &RgfxIcbImageblock {
                    width: 8,
                    height: 4,
                },
            ),
            record(
                RGFX_ICB_OP_DISPATCH_THREADGROUPS,
                2,
                &RgfxIcbDispatch {
                    grid: [2, 1, 1],
                    threadgroup: [64, 1, 1],
                },
            ),
            bare(RGFX_ICB_OP_COMPUTE_SET_BARRIER, 3),
            record(
                RGFX_ICB_OP_DISPATCH_THREADS,
                3,
                &RgfxIcbDispatch {
                    grid: [60, 1, 1],
                    threadgroup: [64, 1, 1],
                },
            ),
            bare(RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER, 3),
            bare(RGFX_ICB_OP_COMPUTE_RESET, 5),
            record(
                RGFX_ICB_OP_COMPUTE_RESET_RANGE,
                0,
                &RgfxIcbRange {
                    length: 1,
                    reserved_04: 0,
                },
            ),
        ];
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &write(COMPUTE_ICB, &records)).unwrap(),
            Ok(())
        );
        let at = |i| ("indirectComputeCommandAtIndex:", vec![Arg::U64(i)]);
        assert_eq!(
            calls(&p, mark),
            vec![
                at(2),
                ("setComputePipelineState:", vec![pipeline]),
                (
                    "setKernelBuffer:offset:atIndex:",
                    vec![buffer(&dec, 2), Arg::U64(256), Arg::U64(0)]
                ),
                (
                    "setKernelBuffer:offset:attributeStride:atIndex:",
                    vec![buffer(&dec, 3), Arg::U64(0), Arg::U64(16), Arg::U64(1)]
                ),
                (
                    "setThreadgroupMemoryLength:atIndex:",
                    vec![Arg::U64(256), Arg::U64(0)]
                ),
                (
                    "setStageInRegion:",
                    vec![Arg::val(&MtlRegion {
                        origin: MtlOrigin { x: 1, y: 2, z: 3 },
                        size: MtlSize {
                            width: 4,
                            height: 5,
                            depth: 6
                        },
                    })]
                ),
                ("setImageblockWidth:height:", vec![Arg::U64(8), Arg::U64(4)]),
                (
                    "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                    vec![size(2, 1, 1), size(64, 1, 1)]
                ),
                at(3),
                ("setBarrier", vec![]),
                (
                    "concurrentDispatchThreads:threadsPerThreadgroup:",
                    vec![size(60, 1, 1), size(64, 1, 1)]
                ),
                ("clearBarrier", vec![]),
                at(5),
                ("reset", vec![]),
                ("resetWithRange:", vec![Arg::U64(0), Arg::U64(1)]),
            ]
        );
    }

    /// A reset of a range is one platform call for the range inside the buffer's size, however
    /// many commands it spans: no command is taken from the buffer for it, and the commands made
    /// before it lose what they were set. One at or past the size makes no call.
    #[test]
    fn a_reset_of_a_range_is_one_call_whatever_its_length() {
        let (p, mut dec) = setup();
        let device = p.device();
        const BIG: u64 = 50 + 7;
        const COUNT: u32 = 1 << 24;
        let m = create(BIG, 2, RENDER_TYPES, COUNT.into());
        assert_eq!(dec.decode(&device, m.as_bytes()).unwrap(), Ok(()));
        dec.end_message();
        let range = |index, length| {
            record(
                RGFX_ICB_OP_RENDER_RESET_RANGE,
                index,
                &RgfxIcbRange {
                    length,
                    reserved_04: 0,
                },
            )
        };
        let records = [
            bare(RGFX_ICB_OP_RENDER_SET_BARRIER, 5),
            bare(RGFX_ICB_OP_RENDER_SET_BARRIER, COUNT - 1),
            range(0, u32::MAX),
            range(COUNT, 4),
        ];
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &write(BIG, &records)).unwrap(), Ok(()));
        let at = |i| ("indirectRenderCommandAtIndex:", vec![Arg::U64(i)]);
        assert_eq!(
            calls(&p, mark),
            vec![
                at(5),
                ("setBarrier", vec![]),
                at(u64::from(COUNT - 1)),
                ("setBarrier", vec![]),
                (
                    "resetWithRange:",
                    vec![Arg::U64(0), Arg::U64(COUNT.into())]
                ),
            ]
        );
        let Some(ObjHandle::Icb(icb)) = dec.named(BIG) else {
            panic!("no indirect command buffer under {BIG}");
        };
        for i in [5, u64::from(COUNT - 1)] {
            assert!(!icb.render_command(i).unwrap().unwrap().has_barrier());
        }
    }

    /// A record the host cannot carry out is skipped alone: one naming a buffer the sender does
    /// not hold, one of a command past the buffer's size, one of an unknown op, one the platform
    /// refuses; the records around them stand, and the message is carried out.
    #[test]
    fn a_refused_record_is_skipped_alone() {
        let (p, mut dec) = setup();
        let device = p.device();
        let draw = RgfxIcbDraw {
            primitive_type: 4,
            vertex_count: 4,
            instance_count: 1,
            ..RgfxIcbDraw::default()
        };
        p.script().push(
            "setFragmentBuffer:offset:atIndex:",
            Step::raise("NSInvalidArgumentException", "refused"),
        );
        let records = vec![
            record(
                RGFX_ICB_OP_VERTEX_BUFFER,
                0,
                &RgfxIcbBuffer {
                    obj_id: 99,
                    offset: 0,
                    index: 0,
                    reserved_14: 0,
                },
            ),
            record(RGFX_ICB_OP_DRAW, 0, &draw),
            record(RGFX_ICB_OP_DRAW, 8, &draw),
            bare(0x1f, 1),
            record(
                RGFX_ICB_OP_FRAGMENT_BUFFER,
                1,
                &RgfxIcbBuffer {
                    obj_id: 2,
                    offset: 0,
                    index: 0,
                    reserved_14: 0,
                },
            ),
            record(RGFX_ICB_OP_DRAW, 1, &draw),
            // A record whose body is too short for its op.
            bare(RGFX_ICB_OP_DRAW, 2),
            record(RGFX_ICB_OP_DRAW, 3, &draw),
        ];
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &records)).unwrap(),
            Ok(())
        );
        let draws: Vec<_> = p
            .call_log()
            .since(mark)
            .into_iter()
            .filter(|c| c.sel.starts_with("draw") || c.sel.starts_with("set"))
            .map(|c| c.sel)
            .collect();
        assert_eq!(
            draws,
            vec![
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                "setFragmentBuffer:offset:atIndex:",
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
            ]
        );
    }

    /// A write naming no indirect command buffer of the sender's (none, or a buffer) is refused
    /// whole; one whose framing breaks is malformed after the records before it.
    #[test]
    fn a_write_naming_no_icb_or_broken_is_refused() {
        let (p, mut dec) = setup();
        let device = p.device();
        let reset = bare(RGFX_ICB_OP_RENDER_RESET, 0);
        for named in [77, 2] {
            assert_eq!(
                dec.decode(&device, &write(named, std::slice::from_ref(&reset)))
                    .unwrap(),
                Err(DecodeError::BadReference)
            );
        }
        // A record whose size passes the message's end.
        let mut broken = bare(RGFX_ICB_OP_RENDER_RESET, 1);
        broken[2] = 64;
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &[reset.clone(), broken]))
                .unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(
            calls(&p, mark),
            vec![
                ("indirectRenderCommandAtIndex:", vec![Arg::U64(0)]),
                ("reset", vec![])
            ]
        );
        // A size of 0 would loop for good: it ends the walk.
        let mut zero = bare(RGFX_ICB_OP_RENDER_RESET, 1);
        zero[2] = 0;
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &[zero])).unwrap(),
            Err(DecodeError::Malformed)
        );
    }

    /// The pipelines an `ICB_WRITE` sets, which it waits to be compiled: each pipeline record's,
    /// nil left out, the walk ending at a broken record.
    #[test]
    fn an_icb_write_names_the_pipelines_it_sets() {
        let records = vec![
            record(RGFX_ICB_OP_RENDER_PIPELINE, 0, &RgfxIcbObject { obj_id: 7 }),
            record(RGFX_ICB_OP_RENDER_PIPELINE, 1, &RgfxIcbObject { obj_id: 0 }),
            bare(RGFX_ICB_OP_RENDER_RESET, 1),
            record(
                RGFX_ICB_OP_COMPUTE_PIPELINE,
                2,
                &RgfxIcbObject { obj_id: 9 },
            ),
        ];
        let message = write(RENDER_ICB, &records);
        assert_eq!(
            crate::graphics::compile::icb_write_pipelines(&message),
            vec![7, 9]
        );
        let mut broken = records;
        broken[2][2] = 200;
        assert_eq!(
            crate::graphics::compile::icb_write_pipelines(&write(RENDER_ICB, &broken)),
            vec![7]
        );
    }
}
