//! Indirect command buffers: `ICB_CREATE`, answered `ICB_STATE` with the platform's
//! `gpuResourceID`, and `ICB_WRITE`, the CPU's encodes into one.
//!
//! The platform's buffer is the one copy of the commands. The guest's CPU encodes reach it as
//! records ([`reims_vgpu_graphics::icb`]) on ring 0, which the guest posts before any command buffer
//! that may execute them, and a kernel's encodes write it on the GPU, through the
//! `gpuResourceID` the guest's argument buffers hold. So a command buffer executes the commands
//! as both sides last wrote them, as Metal's does.
//!
//! Each record is carried out in its own refusal scope: one that names an object the sender does
//! not hold, of the wrong kind, a command past the buffer's size, or that the platform refuses
//! (the Vulkan platform dies where its runtime refuses a setter) is skipped
//! alone, and the message names how many were in one warning. A record whose framing is broken
//! ends the message there.
//!
//! A record that names an object whose posted create the host refused ([`super::refused`]) is
//! dropped, and so is its command's draw or dispatch, as an encoder's draws and dispatches are
//! dropped while a slot of it holds such an object: the command is reset, and its later draws
//! and dispatches are dropped until the guest resets it ([`Dropped`]). Nothing else of the
//! buffer changes, and the command buffers that execute it go on.

use reims_vgpu_graphics::icb::*;
use reims_vgpu_graphics::wire::{
    RGFX_EVT_ICB_STATE, RGFX_ICB_INHERIT_BLEND_COLOR, RGFX_ICB_INHERIT_BUFFERS,
    RGFX_ICB_INHERIT_CULL_MODE, RGFX_ICB_INHERIT_DEPTH_BIAS, RGFX_ICB_INHERIT_DEPTH_CLIP_MODE,
    RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE, RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS,
    RGFX_ICB_INHERIT_FRONT_FACING_WINDING, RGFX_ICB_INHERIT_PIPELINE_STATE,
    RGFX_ICB_INHERIT_SCISSOR_RECTS, RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES,
    RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE, RGFX_ICB_INHERIT_VIEWPORTS, RGFX_ICB_SLOTS_RENDER,
    RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE, RGFX_ICB_SUPPORT_RAY_TRACING,
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RgfxIcbCreate,
    RgfxIcbSlotRun, RgfxIcbSlots, RgfxIcbState, RgfxIcbWrite, RgfxMsgHeader,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::icb::{
    IcbComputeCmd, IcbRenderCmd, IndirectCommandBuffer, IndirectComputeCommand,
    IndirectRenderCommand,
};
use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_vulkan::core::platform::values::IcbDesc;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use reims_vgpu_vulkan::core::raise::RaiseResult;
use reims_vgpu_vulkan::core::sync::Mutex;
use zerocopy::{FromBytes, IntoBytes};

use super::{DecodeError, Entry, GraphicsDecoder, ObjHandle, op_fault, read, refused, refusing};

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

/// Whether `op` is a draw or a dispatch: what a command whose encode named a refused object
/// does not do ([`Dropped`]).
fn acts(op: u16) -> bool {
    (RGFX_ICB_OP_DRAW..=RGFX_ICB_OP_DRAW_INDEXED_PATCHES).contains(&op)
        || op == RGFX_ICB_OP_DISPATCH_THREADGROUPS
        || op == RGFX_ICB_OP_DISPATCH_THREADS
}

/// Whether `op` resets its one command.
fn resets(op: u16) -> bool {
    op == RGFX_ICB_OP_RENDER_RESET || op == RGFX_ICB_OP_COMPUTE_RESET
}

/// The commands of an indirect command buffer whose draws and dispatches are dropped: a CPU
/// encode of each named an object whose posted create the host refused ([`super::refused`]),
/// which the host holds nothing of to bind. That encode is dropped and the command reset, so it
/// draws or dispatches nothing, as an encoder's draws and dispatches are dropped while a slot of it
/// holds such an object; its later draws and dispatches are dropped too, while its other encodes
/// are carried out, until the guest resets the command (a reset record or slot, a reset of a
/// range, a blit's `resetCommandsInBuffer:`) or copies another command over it
/// (`copyIndirectCommandBuffer:`). A command the GPU encodes is not seen here. Empty in every run
/// that refuses nothing, where each record costs one load.
#[derive(Debug, Default)]
pub(crate) struct Dropped {
    /// How many commands `commands` holds, read with no lock.
    count: AtomicUsize,
    commands: Mutex<BTreeSet<u64>>,
}

impl Dropped {
    /// Whether command `index`'s draws and dispatches are dropped.
    fn holds(&self, index: u64) -> bool {
        self.count.load(Ordering::Acquire) != 0 && self.commands.lock().contains(&index)
    }

    /// Command `index`'s draws and dispatches are dropped.
    fn mark(&self, index: u64) {
        let mut commands = self.commands.lock();
        commands.insert(index);
        self.count.store(commands.len(), Ordering::Release);
    }

    /// The commands `len` from `start`, the guest reset them, are carried out again.
    pub(crate) fn clear(&self, start: u64, len: u64) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let end = start.saturating_add(len);
        let mut commands = self.commands.lock();
        commands.retain(|&i| i < start || i >= end);
        self.count.store(commands.len(), Ordering::Release);
    }

    /// The commands `len` from `start` that are dropped.
    fn of(&self, start: u64, len: u64) -> Vec<u64> {
        if self.count.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let end = start.saturating_add(len);
        self.commands.lock().range(start..end).copied().collect()
    }

    /// `len` commands of `source` from `from` are copied over `self`'s from `to`: each copied
    /// command is dropped where its source is, and carried out where it is not.
    pub(crate) fn copied(&self, source: &Dropped, from: u64, to: u64, len: u64) {
        let dropped = source.of(from, len);
        self.clear(to, len);
        for index in dropped {
            self.mark(to.saturating_add(index - from));
        }
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// An `ICB_CREATE`'s answer (`ICB_STATE`): the buffer made, stored under the message's id
    /// before the answer, so an `ICB_WRITE` or a command buffer of its process, which names it
    /// only after the answer, finds it; with its `gpuResourceID`. [`RGFX_STATUS_REFUSED`] with
    /// 0 where the platform makes none, and
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
        Ok(Some(match self.new_icb(device, message, true)? {
            Ok(id) => answer(RGFX_STATUS_OK, id),
            Err(DecodeError::Malformed) => answer(RGFX_STATUS_INVALID_ARGUMENT, 0),
            Err(_) => answer(RGFX_STATUS_REFUSED, 0),
        }))
    }

    /// The buffer of an `ICB_CREATE`, stored under its id; its `gpuResourceID`. `late` for one
    /// made on the ICB worker, after its context's lane handed it over, which a close that came
    /// between makes stored nowhere ([`GraphicsDecoder::store_late`]).
    pub(super) fn new_icb(
        &self,
        device: &P::Device,
        message: &[u8],
        late: bool,
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
        let mut entry = Entry::of(ObjHandle::Icb(icb));
        entry.icb_dropped = Some(Box::default());
        if late {
            self.store_late(m.obj_id, entry);
        } else {
            self.store(m.obj_id, entry);
        }
        Ok(Ok(id))
    }

    /// The indirect command buffer `obj_id` names, where it is one the current context may name,
    /// and its dropped commands ([`Dropped`]).
    fn icb_of(&self, obj_id: u64) -> Option<(&P::Icb, &Dropped)> {
        let entry = self.entry(obj_id)?;
        match (&entry.handle, entry.icb_dropped.as_deref()) {
            (ObjHandle::Icb(icb), Some(dropped)) => Some((icb, dropped)),
            _ => None,
        }
    }

    /// A blit's `resetCommandsInBuffer:` of `len` commands from `start` of the buffer `obj_id`
    /// names, as its command buffer is decoded: the guest's later encodes of them are carried out
    /// ([`Dropped`]).
    pub(super) fn icb_commands_reset(&self, obj_id: u64, start: u64, len: u64) {
        if let Some((_, dropped)) = self.icb_of(obj_id) {
            dropped.clear(start, len);
        }
    }

    /// A blit's `copyIndirectCommandBuffer:` of `len` commands from `from` of `source` to `to` of
    /// `destination`, as its command buffer is decoded: each copied command is dropped where its
    /// source is ([`Dropped::copied`]).
    pub(super) fn icb_commands_copied(
        &self,
        (source, from): (u64, u64),
        (destination, to): (u64, u64),
        len: u64,
    ) {
        if let (Some((_, source)), Some((_, destination))) =
            (self.icb_of(source), self.icb_of(destination))
        {
            destination.copied(source, from, to, len);
        }
    }

    /// An `ICB_WRITE`: each record carried out on the buffer, in order, each refused alone.
    /// Refused as a whole where it names no indirect command buffer of the sender's, and
    /// malformed where a record's framing is broken, after the records before it.
    pub(super) fn icb_write(&self, message: &[u8]) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxIcbWrite>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let Some((icb, dropped)) = self.icb_of(m.obj_id) else {
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
            let record = || self.icb_record(icb, dropped, size, &mut command, &rec, bytes);
            let why = match refusing(record) {
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
                target: "reims_vgpu::graphics", kind:% = why;
                "indirect command buffer {:#x}: {refused} of {records} records not carried out; \
                 the first, op {op:#04x} at command {index}: {why}",
                m.obj_id
            );
        }
        Ok(framing)
    }

    /// An `ICB_SLOTS`: each run's slots taken from the region and carried out on its buffer, each
    /// field as its record would be. A run that names no indirect command buffer of the sender's,
    /// a span not wholly in one block the sender holds, commands past the buffer's size or a slot
    /// shape the host does not know is skipped alone, taking nothing; a field the platform
    /// refuses is skipped alone. One warning names how many were skipped. Malformed where the
    /// runs do not fit the message, after the runs before.
    pub(super) fn icb_slots(&self, message: &[u8]) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxIcbSlots>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let end = (m.header.size as usize).min(message.len());
        let mut tally = Tally::default();
        let mut framing = Ok(());
        let mut at = size_of::<RgfxIcbSlots>();
        for _ in 0..m.run_count {
            let Some(run) = message.get(at..end).and_then(read::<RgfxIcbSlotRun>) else {
                framing = Err(DecodeError::Malformed);
                break;
            };
            at += size_of::<RgfxIcbSlotRun>();
            let done = self.slot_run(&run, &mut tally);
            tally.count(run.icb_obj_id, run.first_command.into(), Ok(done));
        }
        if let Some((icb, index, why)) = &tally.first {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = why;
                "ICB_SLOTS of {} runs (the first on buffer {:#x}): {} of {} runs and fields not \
                 carried out; the first, buffer {icb:#x} at command {index}: {why}",
                m.run_count,
                m.obj_id,
                tally.skipped,
                tally.tried
            );
        }
        Ok(framing)
    }

    /// One run of an `ICB_SLOTS`, refused whole, taking nothing, where its span cannot be read.
    /// Its slots are taken a chunk at a time, so the copy stays small; consecutive commands
    /// whose slots hold a range reset alone are reset in one call.
    fn slot_run(&self, run: &RgfxIcbSlotRun, tally: &mut Tally) -> Result<(), DecodeError> {
        const CHUNK: u64 = 256;
        let Some((icb, dropped)) = self.icb_of(run.icb_obj_id) else {
            return Err(DecodeError::BadReference);
        };
        let kind = run.kind;
        let (a, b) = (u32::from(run.bindings_a), u32::from(run.bindings_b));
        let stride = u64::from(run.slot_bytes);
        let (first, count) = (u64::from(run.first_command), u64::from(run.command_count));
        let shaped =
            slot_bytes(kind, a, b).is_some_and(|least| stride >= least && stride.is_multiple_of(8));
        if !shaped {
            return Err(DecodeError::Malformed);
        }
        let span = stride.checked_mul(count);
        if first.checked_add(count).is_none_or(|e| e > icb.size())
            || span.is_none_or(|len| !self.in_own_block(run.region_offset, len))
        {
            return Err(DecodeError::BadReference);
        }
        let obj = run.icb_obj_id;
        let mut resets: Option<(u64, u64)> = None;
        let mut start = 0;
        while start < count {
            let n = CHUNK.min(count - start);
            // A block released since the check, by a guest racing its own post, refuses the
            // rest of the run.
            let Some(bytes) = self.take_own(run.region_offset + start * stride, stride, n, 2)
            else {
                reset_run(icb, dropped, obj, resets.take(), tally);
                return Err(DecodeError::BadReference);
            };
            for (index, slot) in (first + start..).zip(bytes.chunks_exact(stride as usize)) {
                let (mask, mask_b) = (word(slot, 0), word(slot, 1));
                if mask == 0 && mask_b == 0 {
                    continue;
                }
                if mask == RGFX_ICB_SLOT_RESET_RANGE && mask_b == 0 {
                    resets = match resets {
                        Some((at, len)) if at + len == index => Some((at, len + 1)),
                        before => {
                            reset_run(icb, dropped, obj, before, tally);
                            Some((index, 1))
                        }
                    };
                    continue;
                }
                reset_run(icb, dropped, obj, resets.take(), tally);
                let at = (obj, kind, index);
                self.slot(icb, dropped, at, slot, (a, b), (mask, mask_b), tally);
            }
            start += n;
        }
        reset_run(icb, dropped, obj, resets, tally);
        Ok(())
    }

    /// One command's slot, its bits `masks` taken: the range reset, then each field carried out
    /// on its command ([`slot_fields`]) as its record would be, each refused alone.
    #[allow(clippy::too_many_arguments)] // the slot and what its run says of it
    fn slot(
        &self,
        icb: &P::Icb,
        dropped: &Dropped,
        (obj, kind, index): (u64, u16, u64),
        slot: &[u8],
        bindings: (u32, u32),
        (mask, mask_b): (u64, u64),
        tally: &mut Tally,
    ) {
        if mask & RGFX_ICB_SLOT_RESET_RANGE != 0 {
            reset_run(icb, dropped, obj, Some((index, 1)), tally);
        }
        let mask = mask & !RGFX_ICB_SLOT_RESET_RANGE;
        if mask == 0 && mask_b == 0 {
            return;
        }
        if kind == RGFX_ICB_SLOTS_RENDER {
            match refusing(|| icb.render_command(index)) {
                Ok(Some(c)) => slot_fields(kind, slot, bindings, (mask, mask_b), |field| {
                    let op = field.op();
                    let reset = || c.encode(IcbRenderCmd::Reset);
                    let done = refusing(|| {
                        contain(dropped, index, op, || self.render_field(&c, field), reset)
                    });
                    tally.count(obj, index, done);
                }),
                Ok(None) => tally.count(obj, index, Ok(Err(DecodeError::Refused))),
                Err(raise) => tally.count(obj, index, Err(raise)),
            }
        } else {
            match refusing(|| icb.compute_command(index)) {
                Ok(Some(c)) => slot_fields(kind, slot, bindings, (mask, mask_b), |field| {
                    let op = field.op();
                    let reset = || c.encode(IcbComputeCmd::Reset);
                    let done = refusing(|| {
                        contain(dropped, index, op, || self.compute_field(&c, field), reset)
                    });
                    tally.count(obj, index, done);
                }),
                Ok(None) => tally.count(obj, index, Ok(Err(DecodeError::Refused))),
                Err(raise) => tally.count(obj, index, Err(raise)),
            }
        }
    }

    /// A render slot's `field` on its command: a binding or the pipeline state sent as its record
    /// would send it, with no record made between, and the rest by its record.
    fn render_field(
        &self,
        c: &P::IndirectRenderCommand,
        field: Field<'_>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let cmd = match field {
            Field::Record(op, body) => return self.render_record(c, op, body),
            Field::Pipeline(obj_id) => IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: self.get_render_pipeline(obj_id)?,
            },
            Field::Binding(index, s, strided) => {
                let buffer = self.get_buffer(s.obj_id)?;
                match strided {
                    true => IcbRenderCmd::SetVertexBufferAttributeStride {
                        buffer,
                        offset: s.offset,
                        stride: s.stride,
                        index: index.into(),
                    },
                    false => IcbRenderCmd::SetVertexBuffer {
                        buffer,
                        offset: s.offset,
                        index: index.into(),
                    },
                }
            }
            Field::Second(index, s) => IcbRenderCmd::SetFragmentBuffer {
                buffer: self.get_buffer(s.obj_id)?,
                offset: s.offset,
                index: index.into(),
            },
            Field::Length(..) => return Ok(Err(DecodeError::UnknownCommand)),
        };
        c.encode(cmd)?;
        Ok(Ok(()))
    }

    /// A compute slot's `field` on its command, as [`render_field`](Self::render_field) does.
    fn compute_field(
        &self,
        c: &P::IndirectComputeCommand,
        field: Field<'_>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let cmd = match field {
            Field::Record(op, body) => return self.compute_record(c, op, body),
            Field::Pipeline(obj_id) => IcbComputeCmd::SetComputePipelineState {
                pipeline_state: self.get_compute_pipeline(obj_id)?,
            },
            Field::Binding(index, s, strided) => {
                let buffer = self.get_buffer(s.obj_id)?;
                match strided {
                    true => IcbComputeCmd::SetKernelBufferAttributeStride {
                        buffer,
                        offset: s.offset,
                        stride: s.stride,
                        index: index.into(),
                    },
                    false => IcbComputeCmd::SetKernelBuffer {
                        buffer,
                        offset: s.offset,
                        index: index.into(),
                    },
                }
            }
            Field::Length(index, length) => IcbComputeCmd::SetThreadgroupMemoryLength {
                length: length.into(),
                index: index.into(),
            },
            Field::Second(..) => return Ok(Err(DecodeError::UnknownCommand)),
        };
        c.encode(cmd)?;
        Ok(Ok(()))
    }

    /// One record of an `ICB_WRITE` on `icb` of `size` commands, its command taken from
    /// `command` where the record before it was of the same one.
    fn icb_record(
        &self,
        icb: &P::Icb,
        dropped: &Dropped,
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
                dropped.clear(index, stop - index);
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
            let reset = || c.encode(IcbRenderCmd::Reset);
            contain(
                dropped,
                index,
                op,
                || self.render_record(c, op, bytes),
                reset,
            )
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
            let reset = || c.encode(IcbComputeCmd::Reset);
            contain(
                dropped,
                index,
                op,
                || self.compute_record(c, op, bytes),
                reset,
            )
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

/// What an `ICB_SLOTS` tried and skipped, runs and fields, and the first skipped, for its one
/// warning.
#[derive(Default)]
struct Tally {
    tried: u64,
    skipped: u64,
    first: Option<(u64, u64, String)>,
}

impl Tally {
    /// One run or field of buffer `icb` at command `index`, carried out or skipped as `done` says.
    fn count(&mut self, icb: u64, index: u64, done: RaiseResult<Result<(), DecodeError>>) {
        self.tried += 1;
        let why = match done {
            Ok(Ok(())) => return,
            Ok(Err(e)) => format!("{e:?}"),
            Err(raise) => raise.to_string(),
        };
        self.skipped += 1;
        self.first.get_or_insert((icb, index, why));
    }
}

/// The range reset `run` (its first command and its length), where there is one, in one call:
/// the commands of it are carried out again ([`Dropped`]).
fn reset_run<I: IndirectCommandBuffer<P>, P: Platform>(
    icb: &I,
    dropped: &Dropped,
    obj: u64,
    run: Option<(u64, u64)>,
    tally: &mut Tally,
) {
    if let Some((at, len)) = run {
        tally.count(obj, at, refusing(|| icb.reset_with_range(at, len).map(Ok)));
        dropped.clear(at, len);
    }
}

/// `record`, an encode `op` of command `index`, as [`Dropped`] holds it: a reset of the command
/// carries it out again; a draw or dispatch of a dropped command is dropped; an encode that names
/// an object whose create the host refused is dropped, the command reset (`reset`) and dropped.
fn contain(
    dropped: &Dropped,
    index: u64,
    op: u16,
    record: impl FnOnce() -> RaiseResult<Result<(), DecodeError>>,
    reset: impl FnOnce() -> RaiseResult<()>,
) -> RaiseResult<Result<(), DecodeError>> {
    if resets(op) {
        dropped.clear(index, 1);
    } else if acts(op) && dropped.holds(index) {
        return Ok(Ok(()));
    }
    match record() {
        Err(raise) if refused::is_refused(&raise) => {
            reset()?;
            dropped.mark(index);
            Ok(Ok(()))
        }
        done => done,
    }
}

/// The `k`th `u64` of a slot (its masks are words 0 and 1).
fn word(slot: &[u8], k: usize) -> u64 {
    u64::read_from_prefix(slot.get(k * 8..).unwrap_or_default()).map_or(0, |(w, _)| w)
}

/// A field of a slot, as [`slot_fields`] hands it on: a record's op and body, or a binding or
/// pipeline state taken from the slot as it lies there.
enum Field<'s> {
    /// A field its record carries out: the reset, the stage-in region, the imageblock, the
    /// action and the barrier, and op 0 for a bit the slot's kind does not hold.
    Record(u16, &'s [u8]),
    /// The pipeline state.
    Pipeline(u64),
    /// A binding of the first kind (a vertex or kernel buffer) at its index, and whether it was
    /// set with its attribute stride.
    Binding(u32, RgfxIcbSlotBuffer, bool),
    /// A fragment buffer at its index.
    Second(u32, RgfxIcbSlotBuffer),
    /// A threadgroup memory length at its index.
    Length(u32, u32),
}

impl Field<'_> {
    /// The record op the field is carried out as: its record's, and 0 for a setter's.
    fn op(&self) -> u16 {
        match self {
            Field::Record(op, _) => *op,
            _ => 0,
        }
    }
}

/// Each field a slot's bits `masks` name to `field`, in a fixed order: the command's reset, its
/// pipeline, its bindings of the first kind by index, of the second, the stage-in region and the
/// imageblock (compute), the draw or the dispatch, and the barrier. A bit the slot's kind does
/// not hold, or an action whose op is not one of the kind's draws or dispatches, is op 0, which
/// no record is: the record refuses it.
fn slot_fields(
    kind: u16,
    slot: &[u8],
    (a, b): (u32, u32),
    (mask, mask_b): (u64, u64),
    mut field: impl FnMut(Field<'_>),
) {
    let render = kind == RGFX_ICB_SLOTS_RENDER;
    let fixed = if render {
        size_of::<RgfxIcbRenderSlot>()
    } else {
        size_of::<RgfxIcbComputeSlot>()
    };
    let bits = |n: u32| (1u64 << n) - 1;
    let mut known = RGFX_ICB_SLOT_RESET
        | RGFX_ICB_SLOT_PIPELINE
        | RGFX_ICB_SLOT_ACTION
        | RGFX_ICB_SLOT_BARRIER
        | (bits(a) * RGFX_ICB_SLOT_BUFFER_0);
    if !render {
        known |= RGFX_ICB_SLOT_STAGE_IN | RGFX_ICB_SLOT_IMAGEBLOCK;
    }
    if mask & !known != 0 || mask_b & !bits(b) != 0 {
        field(Field::Record(0, &[]));
    }
    let binding = |i: u32| {
        let at = fixed + i as usize * size_of::<RgfxIcbSlotBuffer>();
        RgfxIcbSlotBuffer::read_from_prefix(slot.get(at..).unwrap_or_default())
            .map_or_else(|_| Default::default(), |(s, _)| s)
    };
    let set = |i: u32| mask & (RGFX_ICB_SLOT_BUFFER_0 << i) != 0;
    if render {
        let Ok((s, _)) = RgfxIcbRenderSlot::read_from_prefix(slot) else {
            return;
        };
        if mask & RGFX_ICB_SLOT_RESET != 0 {
            field(Field::Record(RGFX_ICB_OP_RENDER_RESET, &[]));
        }
        if mask & RGFX_ICB_SLOT_PIPELINE != 0 {
            field(Field::Pipeline(s.pipeline));
        }
        for i in (0..a).filter(|&i| set(i)) {
            field(Field::Binding(i, binding(i), s.strided & (1 << i) != 0));
        }
        for j in (0..b).filter(|&j| mask_b & (1 << j) != 0) {
            field(Field::Second(j, binding(a + j)));
        }
        if mask & RGFX_ICB_SLOT_ACTION != 0 {
            let draws = RGFX_ICB_OP_DRAW..=RGFX_ICB_OP_DRAW_INDEXED_PATCHES;
            let op = if draws.contains(&s.op) { s.op } else { 0 };
            field(Field::Record(op, s.action.as_bytes()));
        }
        if mask & RGFX_ICB_SLOT_BARRIER != 0 {
            field(Field::Record(
                if s.barrier != 0 {
                    RGFX_ICB_OP_RENDER_SET_BARRIER
                } else {
                    RGFX_ICB_OP_RENDER_CLEAR_BARRIER
                },
                &[],
            ));
        }
    } else {
        let Ok((s, _)) = RgfxIcbComputeSlot::read_from_prefix(slot) else {
            return;
        };
        if mask & RGFX_ICB_SLOT_RESET != 0 {
            field(Field::Record(RGFX_ICB_OP_COMPUTE_RESET, &[]));
        }
        if mask & RGFX_ICB_SLOT_PIPELINE != 0 {
            field(Field::Pipeline(s.pipeline));
        }
        for i in (0..a).filter(|&i| set(i)) {
            field(Field::Binding(i, binding(i), s.strided & (1 << i) != 0));
        }
        let lengths = fixed + a as usize * size_of::<RgfxIcbSlotBuffer>();
        for j in (0..b).filter(|&j| mask_b & (1 << j) != 0) {
            let at = lengths + j as usize * 4;
            let length =
                u32::read_from_prefix(slot.get(at..).unwrap_or_default()).map_or(0, |(l, _)| l);
            field(Field::Length(j, length));
        }
        if mask & RGFX_ICB_SLOT_STAGE_IN != 0 {
            field(Field::Record(
                RGFX_ICB_OP_STAGE_IN_REGION,
                s.stage_in.as_bytes(),
            ));
        }
        if mask & RGFX_ICB_SLOT_IMAGEBLOCK != 0 {
            field(Field::Record(
                RGFX_ICB_OP_IMAGEBLOCK,
                s.imageblock.as_bytes(),
            ));
        }
        if mask & RGFX_ICB_SLOT_ACTION != 0 {
            let dispatches = RGFX_ICB_OP_DISPATCH_THREADGROUPS..=RGFX_ICB_OP_DISPATCH_THREADS;
            let op = if dispatches.contains(&s.op) { s.op } else { 0 };
            field(Field::Record(op, s.dispatch.as_bytes()));
        }
        if mask & RGFX_ICB_SLOT_BARRIER != 0 {
            field(Field::Record(
                if s.barrier != 0 {
                    RGFX_ICB_OP_COMPUTE_SET_BARRIER
                } else {
                    RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER
                },
                &[],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::wire::{
        RGFX_CMD_BUFFER_CREATE, RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_WRITE, RgfxBufferCreate,
    };
    use reims_vgpu_vulkan::core::platform::Handle;
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform, Step};
    use zerocopy::{Immutable, IntoBytes};

    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_ICB_SLOTS, RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CONTEXT_OBJ_ID_SHIFT,
        RGFX_CURSOR_SPAN, RGFX_ICB_SLOTS_COMPUTE,
    };

    use super::super::tests::{
        compute_create, pipeline_decoder, pipeline_decoder_over, whole_render_create,
    };
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
        setup_over(Arc::new(Region::new(0x10000).unwrap()))
    }

    /// [`setup`] over `region`.
    fn setup_over(region: Arc<Region>) -> (Arc<FakePlatform>, Dec) {
        let (p, mut dec, _) = pipeline_decoder_over(region);
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
    /// 0, and a short one invalid, storing nothing.
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
                ("resetWithRange:", vec![Arg::U64(0), Arg::U64(COUNT.into())]),
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

    /// The selectors of the commands' encodes since `mark`, and of their buffers' resets of a
    /// range: what a test of [`Dropped`] reads.
    fn encodes(p: &FakePlatform, mark: u64) -> Vec<&'static str> {
        calls(p, mark)
            .into_iter()
            .map(|(sel, _)| sel)
            .filter(|sel| !sel.starts_with("indirect"))
            .collect()
    }

    /// The [`encodes`] an `ICB_WRITE` of `records` into `icb` carries out.
    fn encodes_of(
        p: &FakePlatform,
        dec: &mut Dec,
        icb: u64,
        records: &[Vec<u8>],
    ) -> Vec<&'static str> {
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&p.device(), &write(icb, records)).unwrap(),
            Ok(())
        );
        encodes(p, mark)
    }

    const VB: &str = "setVertexBuffer:offset:atIndex:";
    const FB: &str = "setFragmentBuffer:offset:atIndex:";
    const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:";

    /// An encode of a command that names an object whose create the host refused drops the
    /// command's draws ([`Dropped`]): the encode is dropped and the command reset; its later
    /// setters are carried out and its draws dropped, until the guest resets it (a reset record,
    /// a reset of a range, a blit's reset) or copies a command over it, which a command copied
    /// from a dropped one is not. The other commands, and a dispatch's command of a compute
    /// buffer alike, go on as before.
    #[test]
    fn an_encode_naming_a_refused_object_drops_its_draws_until_a_reset() {
        let (p, mut dec) = setup();
        dec.refuse(99, RGFX_CMD_BUFFER_CREATE);
        let buffer = |op, obj_id, index| {
            record(
                op,
                index,
                &RgfxIcbBuffer {
                    obj_id,
                    offset: 0,
                    index: 0,
                    reserved_14: 0,
                },
            )
        };
        let vb = |obj_id, index| buffer(RGFX_ICB_OP_VERTEX_BUFFER, obj_id, index);
        let shape = draw(4);
        let drawn = |index| record(RGFX_ICB_OP_DRAW, index, &shape);
        let render = |dec: &mut Dec, records: &[Vec<u8>]| encodes_of(&p, dec, RENDER_ICB, records);
        // Command 0 draws; command 1, whose vertex buffer was refused, is reset, keeps its later
        // setter, and draws nothing.
        assert_eq!(
            render(
                &mut dec,
                &[
                    vb(2, 0),
                    drawn(0),
                    vb(99, 1),
                    buffer(RGFX_ICB_OP_FRAGMENT_BUFFER, 3, 1),
                    drawn(1),
                ]
            ),
            [VB, DRAW, "reset", FB]
        );
        // A binding of a buffer the host holds does not end it: the guest's own reset does.
        assert_eq!(render(&mut dec, &[vb(2, 1), drawn(1)]), [VB]);
        assert_eq!(
            render(
                &mut dec,
                &[bare(RGFX_ICB_OP_RENDER_RESET, 1), vb(2, 1), drawn(1)]
            ),
            ["reset", VB, DRAW]
        );
        // So does a reset of a range holding it.
        let range = RgfxIcbRange {
            length: 2,
            reserved_04: 0,
        };
        assert_eq!(
            render(
                &mut dec,
                &[
                    vb(99, 1),
                    drawn(1),
                    record(RGFX_ICB_OP_RENDER_RESET_RANGE, 0, &range),
                    drawn(1),
                ]
            ),
            ["reset", "resetWithRange:", DRAW]
        );
        // And a blit's reset of it, as its command buffer is decoded.
        assert_eq!(render(&mut dec, &[vb(99, 1)]), ["reset"]);
        dec.icb_commands_reset(RENDER_ICB, 1, 1);
        assert_eq!(render(&mut dec, &[drawn(1)]), [DRAW]);
        // A command copied from a dropped one is dropped; one copied from another is not.
        assert_eq!(render(&mut dec, &[vb(99, 2)]), ["reset"]);
        dec.icb_commands_copied((RENDER_ICB, 2), (RENDER_ICB, 5), 1);
        assert_eq!(render(&mut dec, &[drawn(5), drawn(2)]), Vec::<&str>::new());
        dec.icb_commands_copied((RENDER_ICB, 0), (RENDER_ICB, 5), 1);
        assert_eq!(render(&mut dec, &[drawn(5)]), [DRAW]);
        // A compute command's dispatch, as a render command's draw; the other command's
        // dispatch goes on.
        let dispatch = |index| {
            record(
                RGFX_ICB_OP_DISPATCH_THREADS,
                index,
                &RgfxIcbDispatch {
                    grid: [64, 1, 1],
                    threadgroup: [32, 1, 1],
                },
            )
        };
        let records = [
            buffer(RGFX_ICB_OP_KERNEL_BUFFER, 99, 0),
            dispatch(0),
            buffer(RGFX_ICB_OP_KERNEL_BUFFER, 2, 1),
            dispatch(1),
        ];
        assert_eq!(
            encodes_of(&p, &mut dec, COMPUTE_ICB, &records),
            [
                "reset",
                "setKernelBuffer:offset:atIndex:",
                "concurrentDispatchThreads:threadsPerThreadgroup:"
            ]
        );
    }

    /// A slot's binding of an object whose create the host refused drops its command's draw as a
    /// record's does ([`Dropped`]): the command is reset after its pipeline is set, and its
    /// action is dropped; a slot's reset carries the command out again. The slot of another
    /// command draws.
    #[test]
    fn a_slot_naming_a_refused_object_drops_its_draw_until_a_reset() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        dec.refuse(99, RGFX_CMD_BUFFER_CREATE);
        let b0 = RGFX_ICB_SLOT_BUFFER_0;
        let drawing = |reset: bool| RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | b0
                | if reset { RGFX_ICB_SLOT_RESET } else { 0 },
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(4)),
            ..RgfxIcbRenderSlot::default()
        };
        let none = RgfxIcbSlotBuffer::default();
        let refused = [bound(99, 0, 0), none, none];
        let held = [bound(2, 0, 0), none, none];
        let r = SLOTS;
        assert!(region.write(r, &render_slot(&drawing(false), refused)));
        assert!(region.write(r + RENDER_STRIDE, &render_slot(&drawing(false), held)));
        let message = slots(&[run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, r, 0, 2)]);
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let pipeline = "setRenderPipelineState:";
        assert_eq!(encodes(&p, mark), [pipeline, "reset", pipeline, VB, DRAW]);
        // Command 0 again, held this time: still dropped without a reset, drawn after one.
        assert!(region.write(r, &render_slot(&drawing(false), held)));
        let once = slots(&[run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, r, 0, 1)]);
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &once).unwrap(), Ok(()));
        assert_eq!(encodes(&p, mark), [pipeline, VB]);
        assert!(region.write(r, &render_slot(&drawing(true), held)));
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &once).unwrap(), Ok(()));
        assert_eq!(encodes(&p, mark), ["reset", pipeline, VB, DRAW]);
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

    /// Where the slot tests' slots lie: the processes' first block.
    const SLOTS: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
    /// A slot of [`create`]'s render buffer: 2 vertex buffers, 1 fragment buffer.
    const RENDER_STRIDE: u64 = 104 + 24 * 3;
    /// A slot of [`create`]'s compute buffer: 3 kernel buffers, 31 threadgroup memory lengths.
    const COMPUTE_STRIDE: u64 = 88 + 24 * 3 + 128;
    /// Render and compute buffers like [`RENDER_ICB`] and [`COMPUTE_ICB`], which slots write.
    const RENDER_ICB_2: u64 = 60;
    const COMPUTE_ICB_2: u64 = 61;

    /// [`setup`] over a region with room for slots, with a second render and compute buffer of
    /// each shape; and the region.
    fn slot_setup() -> (Arc<FakePlatform>, Dec, Arc<Region>) {
        let size = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let (p, mut dec) = setup_over(Arc::clone(&region));
        let device = p.device();
        for (id, types) in [(RENDER_ICB_2, RENDER_TYPES), (COMPUTE_ICB_2, COMPUTE_TYPES)] {
            let m = create(id, 1, types, 8);
            assert_eq!(dec.decode(&device, m.as_bytes()).unwrap(), Ok(()));
        }
        dec.end_message();
        (p, dec, region)
    }

    /// A slot binding.
    fn bound(obj_id: u64, offset: u64, stride: u64) -> RgfxIcbSlotBuffer {
        RgfxIcbSlotBuffer {
            obj_id,
            offset,
            stride,
        }
    }

    /// A render slot: its fixed part, its 2 vertex buffers and its fragment buffer.
    fn render_slot(fixed: &RgfxIcbRenderSlot, buffers: [RgfxIcbSlotBuffer; 3]) -> Vec<u8> {
        let mut s = fixed.as_bytes().to_vec();
        for b in &buffers {
            s.extend_from_slice(b.as_bytes());
        }
        assert_eq!(s.len() as u64, RENDER_STRIDE);
        s
    }

    /// A compute slot: its fixed part, its 3 kernel buffers and its threadgroup memory lengths
    /// (`lengths`, by index).
    fn compute_slot(
        fixed: &RgfxIcbComputeSlot,
        buffers: [RgfxIcbSlotBuffer; 3],
        lengths: &[(usize, u32)],
    ) -> Vec<u8> {
        let mut s = fixed.as_bytes().to_vec();
        for b in &buffers {
            s.extend_from_slice(b.as_bytes());
        }
        let mut l = [0u32; 32];
        for &(i, len) in lengths {
            l[i] = len;
        }
        s.extend_from_slice(l.as_bytes());
        assert_eq!(s.len() as u64, COMPUTE_STRIDE);
        s
    }

    /// The draw's body as a render slot's action.
    fn action<T: IntoBytes + Immutable>(body: &T) -> [u64; 9] {
        let mut a = [0u64; 9];
        a.as_mut_bytes()[..size_of::<T>()].copy_from_slice(body.as_bytes());
        a
    }

    /// A run of `count` slots of `stride` bytes at `offset` for commands `first..` of `icb`.
    fn run(icb: u64, kind: u16, offset: u64, first: u32, count: u32) -> RgfxIcbSlotRun {
        let render = kind == RGFX_ICB_SLOTS_RENDER;
        RgfxIcbSlotRun {
            icb_obj_id: icb,
            region_offset: offset,
            first_command: first,
            command_count: count,
            slot_bytes: if render {
                RENDER_STRIDE
            } else {
                COMPUTE_STRIDE
            } as u32,
            kind,
            bindings_a: if render { 2 } else { 3 },
            bindings_b: if render { 1 } else { 31 },
        }
    }

    /// An `ICB_SLOTS` of `runs`.
    fn slots(runs: &[RgfxIcbSlotRun]) -> Vec<u8> {
        let len = size_of::<RgfxIcbSlots>() + size_of_val(runs);
        let mut m = RgfxIcbSlots {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_SLOTS,
                size: len as u32,
                seqno: 0,
            },
            obj_id: runs.first().map_or(0, |r| r.icb_obj_id),
            run_count: runs.len() as u32,
            reserved_1c: 0,
        }
        .as_bytes()
        .to_vec();
        for r in runs {
            m.extend_from_slice(r.as_bytes());
        }
        m
    }

    /// The masks of the slot at `offset`.
    fn masks(region: &Region, offset: u64) -> (u64, u64) {
        let b = region.read(offset, 16).unwrap();
        (word(&b, 0), word(&b, 1))
    }

    /// The draw of an `ICB_WRITE` and a render slot's action in the tests.
    const DRAW_INDEXED: RgfxIcbDrawIndexed = RgfxIcbDrawIndexed {
        primitive_type: 3,
        index_count: 6,
        index_type: 1,
        instance_count: 1,
        index_buffer_obj_id: 2,
        index_buffer_offset: 64,
        base_vertex: -4,
        base_instance: 0,
    };

    /// A draw of `vertex_count` vertices.
    fn draw(vertex_count: u32) -> RgfxIcbDraw {
        RgfxIcbDraw {
            primitive_type: 4,
            vertex_start: 1,
            vertex_count,
            instance_count: 2,
            base_instance: 7,
            reserved_14: 0,
        }
    }

    /// Slots drive the setters their records would, field by field, in a fixed order (the reset,
    /// the pipeline, the bindings by index, the stage-in region and imageblock, the action, the
    /// barrier): what records of the same fields in that order drive on a buffer of the same
    /// shape. The take leaves the masks 0, so the same message again carries out nothing.
    #[test]
    fn slots_drive_the_setters_their_records_would() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let patches = RgfxIcbDrawIndexedPatches {
            patches: RgfxIcbDrawPatches {
                number_of_patch_control_points: 4,
                patch_start: 1,
                patch_count: 2,
                instance_count: 3,
                base_instance: 4,
                instance_stride: 12,
                patch_index_buffer_obj_id: 2,
                patch_index_buffer_offset: 8,
                tessellation_factor_buffer_obj_id: 3,
                tessellation_factor_buffer_offset: 48,
            },
            control_point_index_buffer_obj_id: 3,
            control_point_index_buffer_offset: 16,
        };
        let buffer = |obj_id, offset, index| RgfxIcbBuffer {
            obj_id,
            offset,
            index,
            reserved_14: 0,
        };
        let render_records = vec![
            record(
                RGFX_ICB_OP_RENDER_PIPELINE,
                0,
                &RgfxIcbObject {
                    obj_id: RENDER_PIPELINE,
                },
            ),
            record(RGFX_ICB_OP_VERTEX_BUFFER, 0, &buffer(2, 16, 0)),
            record(
                RGFX_ICB_OP_VERTEX_BUFFER_STRIDE,
                0,
                &RgfxIcbBufferStride {
                    obj_id: 0,
                    offset: 8,
                    stride: 24,
                    index: 1,
                    reserved_1c: 0,
                },
            ),
            record(RGFX_ICB_OP_FRAGMENT_BUFFER, 0, &buffer(3, 32, 0)),
            record(RGFX_ICB_OP_DRAW_INDEXED, 0, &DRAW_INDEXED),
            bare(RGFX_ICB_OP_RENDER_SET_BARRIER, 0),
            bare(RGFX_ICB_OP_RENDER_RESET, 1),
            record(
                RGFX_ICB_OP_RENDER_PIPELINE,
                1,
                &RgfxIcbObject {
                    obj_id: RENDER_PIPELINE,
                },
            ),
            record(RGFX_ICB_OP_DRAW_INDEXED_PATCHES, 2, &patches),
            bare(RGFX_ICB_OP_RENDER_CLEAR_BARRIER, 2),
        ];
        let compute_records = vec![
            record(
                RGFX_ICB_OP_COMPUTE_PIPELINE,
                0,
                &RgfxIcbObject {
                    obj_id: COMPUTE_PIPELINE,
                },
            ),
            record(RGFX_ICB_OP_KERNEL_BUFFER, 0, &buffer(2, 8, 0)),
            record(
                RGFX_ICB_OP_KERNEL_BUFFER_STRIDE,
                0,
                &RgfxIcbBufferStride {
                    obj_id: 3,
                    offset: 16,
                    stride: 12,
                    index: 2,
                    reserved_1c: 0,
                },
            ),
            record(
                RGFX_ICB_OP_THREADGROUP_MEMORY,
                0,
                &RgfxIcbThreadgroupMemory {
                    length: 256,
                    index: 1,
                },
            ),
            record(
                RGFX_ICB_OP_THREADGROUP_MEMORY,
                0,
                &RgfxIcbThreadgroupMemory {
                    length: 64,
                    index: 30,
                },
            ),
            record(
                RGFX_ICB_OP_STAGE_IN_REGION,
                0,
                &RgfxIcbRegion {
                    origin: [1, 2, 3],
                    size: [4, 5, 6],
                },
            ),
            record(
                RGFX_ICB_OP_IMAGEBLOCK,
                0,
                &RgfxIcbImageblock {
                    width: 16,
                    height: 8,
                },
            ),
            record(
                RGFX_ICB_OP_DISPATCH_THREADS,
                0,
                &RgfxIcbDispatch {
                    grid: [64, 2, 1],
                    threadgroup: [32, 1, 1],
                },
            ),
            bare(RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER, 0),
            bare(RGFX_ICB_OP_COMPUTE_RESET, 3),
        ];
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &render_records))
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, &write(COMPUTE_ICB, &compute_records))
                .unwrap(),
            Ok(())
        );
        let by_records = calls(&p, mark);

        let r = SLOTS;
        let c = SLOTS + 0x1_0000;
        let b0 = RGFX_ICB_SLOT_BUFFER_0;
        let fixed = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | RGFX_ICB_SLOT_BARRIER
                | b0
                | b0 << 1,
            mask_b: 1,
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW_INDEXED,
            barrier: 1,
            strided: 0b10,
            action: action(&DRAW_INDEXED),
        };
        let buffers = [bound(2, 16, 0), bound(0, 8, 24), bound(3, 32, 0)];
        assert!(region.write(r, &render_slot(&fixed, buffers)));
        let reset_then_pipeline = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_RESET | RGFX_ICB_SLOT_PIPELINE,
            pipeline: RENDER_PIPELINE,
            ..RgfxIcbRenderSlot::default()
        };
        let none = [RgfxIcbSlotBuffer::default(); 3];
        assert!(region.write(r + RENDER_STRIDE, &render_slot(&reset_then_pipeline, none)));
        let patched = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_ACTION | RGFX_ICB_SLOT_BARRIER,
            op: RGFX_ICB_OP_DRAW_INDEXED_PATCHES,
            action: action(&patches),
            ..RgfxIcbRenderSlot::default()
        };
        assert!(region.write(r + 2 * RENDER_STRIDE, &render_slot(&patched, none)));
        let dispatch = RgfxIcbComputeSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | RGFX_ICB_SLOT_BARRIER
                | RGFX_ICB_SLOT_STAGE_IN
                | RGFX_ICB_SLOT_IMAGEBLOCK
                | b0
                | b0 << 2,
            mask_b: 1 << 1 | 1 << 30,
            pipeline: COMPUTE_PIPELINE,
            op: RGFX_ICB_OP_DISPATCH_THREADS,
            barrier: 0,
            strided: 0b100,
            dispatch: RgfxIcbDispatch {
                grid: [64, 2, 1],
                threadgroup: [32, 1, 1],
            },
            stage_in: RgfxIcbRegion {
                origin: [1, 2, 3],
                size: [4, 5, 6],
            },
            imageblock: RgfxIcbImageblock {
                width: 16,
                height: 8,
            },
        };
        let kernels = [bound(2, 8, 0), bound(0, 0, 0), bound(3, 16, 12)];
        let lengths = [(1, 256), (30, 64)];
        assert!(region.write(c, &compute_slot(&dispatch, kernels, &lengths)));
        let reset = RgfxIcbComputeSlot {
            mask: RGFX_ICB_SLOT_RESET,
            ..RgfxIcbComputeSlot::default()
        };
        assert!(region.write(
            c + 3 * COMPUTE_STRIDE,
            &compute_slot(&reset, [RgfxIcbSlotBuffer::default(); 3], &[])
        ));
        let message = slots(&[
            run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, r, 0, 3),
            run(COMPUTE_ICB_2, RGFX_ICB_SLOTS_COMPUTE, c, 0, 4),
        ]);
        assert_eq!(
            crate::graphics::compile::icb_slots_pipelines(&region, &message),
            vec![RENDER_PIPELINE, RENDER_PIPELINE, COMPUTE_PIPELINE]
        );
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(calls(&p, mark), by_records);
        for at in [
            r,
            r + RENDER_STRIDE,
            r + 2 * RENDER_STRIDE,
            c,
            c + 3 * COMPUTE_STRIDE,
        ] {
            assert_eq!(masks(&region, at), (0, 0), "the slot at {at:#x} taken");
        }
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(calls(&p, mark), vec![]);
        assert_eq!(
            crate::graphics::compile::icb_slots_pipelines(&region, &message),
            Vec::<u64>::new()
        );
    }

    /// A run is refused whole, taking nothing, where its span lies in another process's block,
    /// runs past the sender's block or lies in no block, where it names a buffer of another
    /// process's or none, commands past the buffer's size, or a slot shape the host does not
    /// know; the runs around it are carried out, and a field the host cannot carry out is skipped
    /// alone.
    #[test]
    fn a_slot_run_outside_the_senders_block_is_refused_taking_nothing() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let other = SLOTS + 0x8_0000;
        region.grant(0, SLOTS, 0x1_0000);
        region.grant(1, other, 0x1_0000);
        dec.set_context(Some(0));
        let pipeline = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE,
            pipeline: RENDER_PIPELINE,
            ..RgfxIcbRenderSlot::default()
        };
        let none = [RgfxIcbSlotBuffer::default(); 3];
        let last = SLOTS + 0x1_0000 - RENDER_STRIDE;
        let elsewhere = SLOTS + 0x4_0000;
        let good = SLOTS + 0x100;
        for at in [other, last, elsewhere, good] {
            assert!(region.write(at, &render_slot(&pipeline, none)));
        }
        // The good run's second command names a buffer no one holds, then draws.
        let half_bad = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_ACTION | RGFX_ICB_SLOT_BUFFER_0,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(4)),
            ..RgfxIcbRenderSlot::default()
        };
        let stray = [
            bound(99, 0, 0),
            RgfxIcbSlotBuffer::default(),
            RgfxIcbSlotBuffer::default(),
        ];
        assert!(region.write(good + RENDER_STRIDE, &render_slot(&half_bad, stray)));
        let foreign = (1u64 << RGFX_CONTEXT_OBJ_ID_SHIFT) | RENDER_ICB;
        let short = RgfxIcbSlotRun {
            slot_bytes: 100,
            ..run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, good, 0, 1)
        };
        let message = slots(&[
            run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, other, 0, 1),
            run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, last, 0, 2),
            run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, elsewhere, 0, 1),
            run(foreign, RGFX_ICB_SLOTS_RENDER, good, 0, 1),
            run(999, RGFX_ICB_SLOTS_RENDER, good, 0, 1),
            run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, good, 7, 2),
            short,
            run(RENDER_ICB, RGFX_ICB_SLOTS_COMPUTE + 1, good, 0, 1),
            run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, good, 0, 2),
        ]);
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let at = |i| ("indirectRenderCommandAtIndex:", vec![Arg::U64(i)]);
        let pipeline_arg = Arg::Obj(Some(
            dec.require_render_pipeline(RENDER_PIPELINE).unwrap().id(),
        ));
        assert_eq!(
            calls(&p, mark),
            vec![
                at(0),
                ("setRenderPipelineState:", vec![pipeline_arg]),
                at(1),
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
            ]
        );
        for (at, what) in [
            (other, "in another process's block"),
            (last, "past the sender's block"),
            (elsewhere, "in no block"),
        ] {
            assert_eq!(
                masks(&region, at),
                (RGFX_ICB_SLOT_PIPELINE, 0),
                "the slot {what} left as it was"
            );
        }
        assert_eq!(masks(&region, good), (0, 0));
        // A message too short for its runs is malformed after the runs before.
        assert!(region.write(good, &render_slot(&pipeline, none)));
        let mut cut = slots(&[run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, good, 0, 1)]);
        cut[24..28].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            dec.decode(&device, &cut).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(masks(&region, good), (0, 0), "the run before carried out");
    }

    /// A take carries a command's reset out before what was set after it, a range reset alone
    /// or with fields; consecutive commands with a range reset alone are reset in one call.
    #[test]
    fn a_reset_then_a_set_in_one_take_is_carried_out_in_that_order() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let none = [RgfxIcbSlotBuffer::default(); 3];
        let slot = |mask| RgfxIcbRenderSlot {
            mask,
            pipeline: RENDER_PIPELINE,
            ..RgfxIcbRenderSlot::default()
        };
        let range = RGFX_ICB_SLOT_RESET_RANGE;
        let pipeline = RGFX_ICB_SLOT_PIPELINE;
        let masks_of = [
            RGFX_ICB_SLOT_RESET | pipeline,
            range | pipeline,
            range,
            range,
            pipeline,
            range,
            range,
            range,
        ];
        for (i, mask) in (0u64..).zip(masks_of) {
            assert!(region.write(SLOTS + i * RENDER_STRIDE, &render_slot(&slot(mask), none)));
        }
        let mark = p.call_log().mark();
        let message = slots(&[run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 8)]);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let at = |i| ("indirectRenderCommandAtIndex:", vec![Arg::U64(i)]);
        let reset = |i, n| ("resetWithRange:", vec![Arg::U64(i), Arg::U64(n)]);
        let set = (
            "setRenderPipelineState:",
            vec![Arg::Obj(Some(
                dec.require_render_pipeline(RENDER_PIPELINE).unwrap().id(),
            ))],
        );
        assert_eq!(
            calls(&p, mark),
            vec![
                at(0),
                ("reset", vec![]),
                set.clone(),
                reset(1, 1),
                at(1),
                set.clone(),
                reset(2, 2),
                at(4),
                set,
                reset(5, 3),
            ]
        );
        let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB) else {
            panic!("no indirect command buffer");
        };
        for i in [0, 1] {
            let c = icb.render_command(i).unwrap().unwrap();
            assert_ne!(c.pipeline_state_unique_identifier(), 0, "command {i}");
        }
    }

    /// Slots and records reach a buffer in the order their messages come: a guest that switches
    /// to records posts its slots first, and the records' fields, set after, stand.
    #[test]
    fn slots_then_records_are_carried_out_in_their_order() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let slot = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE | RGFX_ICB_SLOT_ACTION,
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(4)),
            ..RgfxIcbRenderSlot::default()
        };
        assert!(region.write(
            SLOTS,
            &render_slot(&slot, [RgfxIcbSlotBuffer::default(); 3])
        ));
        let message = slots(&[run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 1)]);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let records = [record(RGFX_ICB_OP_DRAW, 0, &draw(9))];
        assert_eq!(
            dec.decode(&device, &write(RENDER_ICB, &records)).unwrap(),
            Ok(())
        );
        let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB) else {
            panic!("no indirect command buffer");
        };
        let c = icb.render_command(0).unwrap().unwrap();
        assert_eq!(c.draw_args().unwrap().vertex_count, 9);
        assert_ne!(c.pipeline_state_unique_identifier(), 0);
    }

    /// A take carries out only what the guest set since the last: a GPU-side encode between two
    /// takes stays where the second take sets other fields, as on Metal, where the latest encode
    /// of each field stands.
    #[test]
    fn a_gpu_encode_between_two_takes_stays() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let slot = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE | RGFX_ICB_SLOT_ACTION,
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(4)),
            ..RgfxIcbRenderSlot::default()
        };
        let none = [RgfxIcbSlotBuffer::default(); 3];
        assert!(region.write(SLOTS, &render_slot(&slot, none)));
        let message = slots(&[run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 1)]);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        dec.end_message();
        // The GPU's encode: a kernel writing the command through its gpuResourceID.
        {
            let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB) else {
                panic!("no indirect command buffer");
            };
            let c = icb.render_command(0).unwrap().unwrap();
            c.encode(IcbRenderCmd::DrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 7,
                instance_count: 1,
                base_instance: 0,
            })
            .unwrap();
        }
        // The guest then sets vertex buffer 0 alone: the slot still holds the first draw's
        // fields, but not their bits.
        let mut bytes = region.read(SLOTS, RENDER_STRIDE as usize).unwrap();
        bytes[0..8].copy_from_slice(&RGFX_ICB_SLOT_BUFFER_0.to_le_bytes());
        bytes[104..128].copy_from_slice(bound(2, 0, 0).as_bytes());
        assert!(region.write(SLOTS, &bytes));
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let b2 = buffer(&dec, 2);
        assert_eq!(
            calls(&p, mark),
            vec![
                ("indirectRenderCommandAtIndex:", vec![Arg::U64(0)]),
                (
                    "setVertexBuffer:offset:atIndex:",
                    vec![b2, Arg::U64(0), Arg::U64(0)]
                ),
            ]
        );
        let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB) else {
            panic!("no indirect command buffer");
        };
        let c = icb.render_command(0).unwrap().unwrap();
        assert_eq!(c.draw_args().unwrap().vertex_count, 7);
    }
}
