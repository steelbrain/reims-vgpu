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
    RgfxIcbSlotPagesRun, RgfxIcbSlotRun, RgfxIcbSlots, RgfxIcbState, RgfxIcbWrite, RgfxMsgHeader,
};
use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::icb::{
    IcbComputeCmd, IcbRenderCmd, IndirectCommandBuffer, IndirectComputeCommand,
    IndirectRenderCommand,
};
use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_vulkan::core::platform::values::IcbDesc;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::mem::offset_of;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use reims_vgpu_vulkan::core::raise::{Raise, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;
use zerocopy::FromBytes;

use super::{
    DecodeError, Entry, GraphicsDecoder, ObjHandle, op_fault, read, refused, refused_raise,
    refusing,
};
use reims_vgpu_vulkan::core::raise::refusals;

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
enum Command<'a, P: Platform> {
    None,
    Render(u64, P::IndirectRenderCommand<'a>),
    Compute(u64, P::IndirectComputeCommand<'a>),
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
            tally.count(run.icb_obj_id, run.first_command.into(), carried(Ok(done)));
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

    /// An `ICB_SLOTS_IN_PAGES`: as [`Self::icb_slots`], each run's slots taken in place from the
    /// pages of the pages block it names ([`Self::own_pages`]), where the process's CPU wrote them
    /// through its own cached mapping. A run whose block is no pages block of the sender's, or
    /// whose slots pass its pages, is skipped alone, taking nothing, as is one whose block's view
    /// the platform could not make.
    pub(super) fn icb_slots_in_pages(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxIcbSlots>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let end = (m.header.size as usize).min(message.len());
        let mut tally = Tally::default();
        let mut framing = Ok(());
        let mut at = size_of::<RgfxIcbSlots>();
        for _ in 0..m.run_count {
            let Some(run) = message.get(at..end).and_then(read::<RgfxIcbSlotPagesRun>) else {
                framing = Err(DecodeError::Malformed);
                break;
            };
            at += size_of::<RgfxIcbSlotPagesRun>();
            let done = self.pages_slot_run(device, &run, &mut tally);
            tally.count(run.icb_obj_id, run.first_command.into(), carried(done));
        }
        if let Some((icb, index, why)) = &tally.first {
            log::warn!(
                target: "reims_vgpu::graphics", kind:% = why;
                "ICB_SLOTS_IN_PAGES of {} runs (the first on buffer {:#x}): {} of {} runs and \
                 fields not carried out; the first, buffer {icb:#x} at command {index}: {why}",
                m.run_count,
                m.obj_id,
                tally.skipped,
                tally.tried
            );
        }
        Ok(framing)
    }

    /// The pipelines the slots an `ICB_SLOTS_IN_PAGES` `message` names set (their pipeline bit
    /// set), nil left out, read in place as they stand, as
    /// [`crate::graphics::compile::icb_slots_pipelines`] reads an `ICB_SLOTS`'s: whom the message
    /// waits for. A run whose block is no pages block of the sender's, whose view the platform
    /// could not make, whose slots pass its pages, or of a shape the decoder does not take, names
    /// none.
    pub(crate) fn icb_slots_in_pages_pipelines(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> Vec<u64> {
        let mut named = Vec::new();
        let Some(m) = read::<RgfxIcbSlots>(message) else {
            return named;
        };
        let end = (m.header.size as usize).min(message.len());
        let mut at = size_of::<RgfxIcbSlots>();
        for _ in 0..m.run_count {
            let Some(run) = message.get(at..end).and_then(read::<RgfxIcbSlotPagesRun>) else {
                break;
            };
            at += size_of::<RgfxIcbSlotPagesRun>();
            let stride = u64::from(run.slot_bytes);
            let (a, b) = (u32::from(run.bindings_a), u32::from(run.bindings_b));
            let shaped = slot_bytes(run.kind, a, b)
                .is_some_and(|least| stride >= least && stride.is_multiple_of(8));
            let Ok(Ok(pages)) = self.own_pages(device, run.block_offset, run.block_size) else {
                continue;
            };
            let Some(len) = (stride.checked_mul(u64::from(run.command_count)))
                .filter(|&len| shaped && pages.holds(run.pages_offset, len))
            else {
                continue;
            };
            let first = pages.at().wrapping_add(run.pages_offset);
            for slot in (0..len).step_by(stride as usize) {
                // The mask and the pipeline, the slot's first and third words.
                let word = |k: u64| {
                    // SAFETY: an 8-aligned word of a slot in the block's pages (checked above),
                    // which `pages` keeps mapped; the guest writes it only through atomics.
                    unsafe { AtomicU64::from_ptr(first.wrapping_add(slot + k * 8).as_ptr()) }
                        .load(Ordering::Acquire)
                };
                let (mask, pipeline) = (word(0), word(2));
                if mask & RGFX_ICB_SLOT_PIPELINE != 0 && pipeline != 0 {
                    named.push(pipeline);
                }
            }
        }
        named
    }

    /// One run of an `ICB_SLOTS_IN_PAGES`, as [`Self::slot_run`]: refused whole, taking nothing,
    /// where its block is no pages block of the sender's, or its slots pass the block's pages.
    fn pages_slot_run(
        &self,
        device: &P::Device,
        run: &RgfxIcbSlotPagesRun,
        tally: &mut Tally,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let pages = match self.own_pages(device, run.block_offset, run.block_size)? {
            Ok(pages) => pages,
            Err(why) => return Ok(Err(why)),
        };
        let shape = RgfxIcbSlotRun {
            icb_obj_id: run.icb_obj_id,
            region_offset: run.pages_offset,
            first_command: run.first_command,
            command_count: run.command_count,
            slot_bytes: run.slot_bytes,
            kind: run.kind,
            bindings_a: run.bindings_a,
            bindings_b: run.bindings_b,
        };
        let (offset, stride) = (run.pages_offset, u64::from(run.slot_bytes));
        let first = pages.at().wrapping_add(offset);
        Ok(self.slots_of(
            &shape,
            |len| pages.holds(offset, len),
            // SAFETY: the run's slots lie in the block's pages (`holds`, which `slots_of` checks
            // first), which `pages` keeps mapped: the guest's RAM mapping, which stays mapped as
            // long as the device, or the block's view, which it holds.
            |start, n| unsafe { take_records_at(first.wrapping_add(start * stride), stride, n, 2) },
            tally,
        ))
    }

    /// One run of an `ICB_SLOTS`, refused whole, taking nothing, where its span cannot be read.
    fn slot_run(&self, run: &RgfxIcbSlotRun, tally: &mut Tally) -> Result<(), DecodeError> {
        let (at, stride) = (run.region_offset, u64::from(run.slot_bytes));
        self.slots_of(
            run,
            |len| self.in_own_block(at, len),
            |start, n| self.take_own(at + start * stride, stride, n, 2),
            tally,
        )
    }

    /// The slots of `run`'s commands, of its shape, carried out on its buffer: refused whole,
    /// taking nothing, where its shape is not one the host knows, its commands pass the buffer's
    /// size, or `readable` answers false for its slots' length. `take` takes `n` slots from the
    /// run's `start`th, their masks swapped for 0 ([`Region::take_records`]), or none where they
    /// can no longer be read, which refuses the rest of the run. Its slots are taken a chunk at a
    /// time, so the copy stays small; consecutive commands whose slots hold a range reset alone
    /// are reset in one call.
    ///
    /// [`Region::take_records`]: reims_vgpu_graphics::region::Region::take_records
    fn slots_of(
        &self,
        run: &RgfxIcbSlotRun,
        readable: impl Fn(u64) -> bool,
        take: impl Fn(u64, u64) -> Option<Vec<u8>>,
        tally: &mut Tally,
    ) -> Result<(), DecodeError> {
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
            || span.is_none_or(|len| !readable(len))
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
            let Some(bytes) = take(start, n) else {
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
    /// on its command as its record would be ([`Fields`]), the command taken and its fields sent
    /// in one platform call ([`IndirectCommandBuffer::encode_render_command`]), each refused
    /// alone ([`command_fields`]).
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
        let fields = Fields::of(kind, slot, bindings, (mask, mask_b));
        if kind == RGFX_ICB_SLOTS_RENDER {
            command_fields(
                fields,
                |place| match place {
                    Place::Set(set) => made(self.render_set(set)),
                    Place::Record(op, body) => {
                        carry(dropped, index, op, || self.render_cmd(op, body))
                    }
                },
                || IcbRenderCmd::Reset,
                |cmds| icb.encode_render_command(index, cmds),
                || {
                    let reset = std::iter::once(IcbRenderCmd::Reset);
                    icb.encode_render_command(index, reset).map(drop)
                },
                dropped,
                (obj, index),
                tally,
            );
        } else {
            command_fields(
                fields,
                |place| match place {
                    Place::Set(set) => made(self.compute_set(set)),
                    Place::Record(op, body) => {
                        carry(dropped, index, op, || self.compute_cmd(op, body))
                    }
                },
                || IcbComputeCmd::Reset,
                |cmds| icb.encode_compute_command(index, cmds),
                || {
                    let reset = std::iter::once(IcbComputeCmd::Reset);
                    icb.encode_compute_command(index, reset).map(drop)
                },
                dropped,
                (obj, index),
                tally,
            );
        }
    }

    /// A render slot's setter `set`, its command made as its record would make it, with no
    /// record made between.
    #[inline(always)]
    fn render_set(&self, set: Set) -> Result<IcbRenderCmd<'_, P>, Box<Failure>> {
        Ok(match set {
            Set::Pipeline(obj_id) => IcbRenderCmd::SetRenderPipelineState {
                pipeline_state: raised(self.get_render_pipeline(obj_id))?,
            },
            Set::Binding(index, s, strided) => {
                let buffer = raised(self.get_buffer(s.obj_id))?;
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
            Set::Second(index, s) => IcbRenderCmd::SetFragmentBuffer {
                buffer: raised(self.get_buffer(s.obj_id))?,
                offset: s.offset,
                index: index.into(),
            },
            Set::Length(..) => return Err(Box::new(DecodeError::UnknownCommand.into())),
        })
    }

    /// A compute slot's setter `set`, as [`render_set`](Self::render_set) makes a render slot's.
    #[inline(always)]
    fn compute_set(&self, set: Set) -> Result<IcbComputeCmd<'_, P>, Box<Failure>> {
        Ok(match set {
            Set::Pipeline(obj_id) => IcbComputeCmd::SetComputePipelineState {
                pipeline_state: raised(self.get_compute_pipeline(obj_id))?,
            },
            Set::Binding(index, s, strided) => {
                let buffer = raised(self.get_buffer(s.obj_id))?;
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
            Set::Length(index, length) => IcbComputeCmd::SetThreadgroupMemoryLength {
                length: length.into(),
                index: index.into(),
            },
            Set::Second(..) => return Err(Box::new(DecodeError::UnknownCommand.into())),
        })
    }

    /// One record of an `ICB_WRITE` on `icb` of `size` commands, its command taken from
    /// `command` where the record before it was of the same one.
    fn icb_record<'i>(
        &self,
        icb: &'i P::Icb,
        dropped: &Dropped,
        size: u64,
        command: &mut Command<'i, P>,
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
            uncarried(contain(
                dropped,
                index,
                op,
                || carried(self.render_record(c, op, bytes)),
                reset,
            ))
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
            uncarried(contain(
                dropped,
                index,
                op,
                || carried(self.compute_record(c, op, bytes)),
                reset,
            ))
        } else {
            Ok(Err(DecodeError::UnknownCommand))
        }
    }

    /// A render command's record.
    fn render_record(
        &self,
        c: &P::IndirectRenderCommand<'_>,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        match self.render_cmd(op, bytes)? {
            Ok(cmd) => c.encode(cmd).map(Ok),
            Err(e) => Ok(Err(e)),
        }
    }

    /// The command a render command's record `op` of body `bytes` sends. Inlined always, so a
    /// slot's draw is built where its send takes it ([`command_fields`]).
    #[inline(always)]
    fn render_cmd(
        &self,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<IcbRenderCmd<'_, P>, DecodeError>> {
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
        Ok(Ok(cmd))
    }

    /// A compute command's record.
    fn compute_record(
        &self,
        c: &P::IndirectComputeCommand<'_>,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        match self.compute_cmd(op, bytes)? {
            Ok(cmd) => c.encode(cmd).map(Ok),
            Err(e) => Ok(Err(e)),
        }
    }

    /// The command a compute command's record `op` of body `bytes` sends, as
    /// [`render_cmd`](Self::render_cmd) makes a render command's.
    #[inline(always)]
    fn compute_cmd(
        &self,
        op: u16,
        bytes: &[u8],
    ) -> RaiseResult<Result<IcbComputeCmd<'_, P>, DecodeError>> {
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
        Ok(Ok(cmd))
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
    #[inline]
    fn count(&mut self, icb: u64, index: u64, done: Carried) {
        self.tried += 1;
        if let Err(failure) = done {
            self.skip(icb, index, *failure);
        }
    }

    /// `n` fields carried out.
    #[inline]
    fn carried(&mut self, n: usize) {
        self.tried += n as u64;
    }

    /// One run or field of buffer `icb` at command `index` skipped, for `failure`.
    #[cold]
    #[inline(never)]
    fn skip(&mut self, icb: u64, index: u64, failure: Failure) {
        self.skipped += 1;
        if self.first.is_none() {
            let why = match failure {
                Failure::Decode(e) => format!("{e:?}"),
                Failure::Raise(raise) => raise.to_string(),
            };
            self.first = Some((icb, index, why));
        }
    }
}

/// Why a run or a field was not carried out: the platform's raise, or the decoder's answer.
#[derive(Debug)]
enum Failure {
    Decode(DecodeError),
    Raise(Raise),
}

impl From<DecodeError> for Failure {
    fn from(e: DecodeError) -> Failure {
        Failure::Decode(e)
    }
}

impl From<Raise> for Failure {
    fn from(raise: Raise) -> Failure {
        Failure::Raise(raise)
    }
}

impl Failure {
    /// Whether it names an object whose posted create the host refused ([`refused`]).
    fn refused(&self) -> bool {
        matches!(self, Failure::Raise(raise) if refused::is_refused(raise))
    }
}

/// A field's or a run's carrying out: nothing where it was carried out, and its [`Failure`],
/// boxed, where it was not. So a field carried out, every field of a take but those refused,
/// answers in a register, where its [`RaiseResult`], a [`Raise`] wide, went from frame to frame
/// between the field's encode and its tally, and each copy of it, in loads wider than the stores
/// that wrote it, waited for them to land.
type Carried = Result<(), Box<Failure>>;

/// `done` as [`Carried`].
#[inline]
fn carried(done: RaiseResult<Result<(), DecodeError>>) -> Carried {
    match done {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(Box::new(e.into())),
        Err(raise) => Err(Box::new(raise.into())),
    }
}

/// `done` as the [`RaiseResult`] it was carried as ([`carried`]).
#[inline]
fn uncarried(done: Carried) -> RaiseResult<Result<(), DecodeError>> {
    match done {
        Ok(()) => Ok(Ok(())),
        Err(failure) => match *failure {
            Failure::Decode(e) => Ok(Err(e)),
            Failure::Raise(raise) => Err(raise),
        },
    }
}

/// `got`, its raise as a [`Carried`] failure.
#[inline]
fn raised<T>(got: RaiseResult<T>) -> Result<T, Box<Failure>> {
    got.map_err(|raise| Box::new(raise.into()))
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
        tally.count(
            obj,
            at,
            carried(refusing(|| icb.reset_with_range(at, len).map(Ok))),
        );
        dropped.clear(at, len);
    }
}

/// Each field of a slot, `fields` ([`Fields`]), carried out on command `index` of buffer `obj`,
/// each counted on `tally` and refused alone, as a scope of its own for the command's take and
/// for each field would refuse them: the fields from the first not yet counted, each made into
/// its command by `make` as it is taken ([`Walk`]), handed to `send`, one platform call
/// ([`IndirectCommandBuffer::encode_render_command`]) that takes the command and sends them, in
/// one refusal scope. A field `make` cannot make is settled in the call ([`Walk`]). The field that
/// stopped the call is the one being made or sent then ([`Walk`]): the platform's raise at its
/// send is settled ([`settle`], `reset` resetting the command); a death refuses it alone. The
/// fields after it are carried out in another call, in a scope of its own. A take that answers
/// no command, raises or dies refuses the slot.
///
/// Inlined always, into its one caller of each kind ([`GraphicsDecoder::slot`]): handed over in
/// a call, its closures and the slot's fields were copied into its frame in loads wider than the
/// stores that had just written them, which waited for those to land.
#[inline(always)]
#[allow(clippy::too_many_arguments)] // the slot, its command's calls, and what it is counted on
fn command_fields<'s, Cmd, M: Fn(Place<'s>) -> Made<Cmd>, R: Fn() -> Cmd>(
    fields: Fields<'s>,
    make: M,
    reset_cmd: R,
    send: impl Fn(Walk<'_, 's, M, R>) -> RaiseResult<bool>,
    reset: impl Fn() -> RaiseResult<()>,
    dropped: &Dropped,
    (obj, index): (u64, u64),
    tally: &mut Tally,
) {
    /// Where a call stopped, short of a death.
    enum Stop {
        /// Every field was carried out.
        Done,
        /// The command's take: no command, or its raise.
        Take(Box<Failure>),
        /// The field at a place, and its carrying out.
        At(usize, Carried),
    }
    let total = fields.len();
    // The fields counted so far, from which a call goes on; the place of the field being made
    // or sent, none before the first; the place whose reset a refused object's field sends; and
    // how many fields a call's walk counted as not carried out.
    let mut counted = 0usize;
    let at = Cell::new(None);
    let resetting = Cell::new(None);
    let failed = Cell::new(0usize);
    loop {
        at.set(None);
        resetting.set(None);
        failed.set(0);
        let ran = refusals::refusing(|| {
            let walk = Walk {
                fields: fields.after(counted),
                next: counted,
                make: &make,
                reset: &reset_cmd,
                at: &at,
                resetting: &resetting,
                failed: &failed,
                dropped,
                command: (obj, index),
                tally: &mut *tally,
            };
            match send(walk) {
                Ok(true) => {
                    if resetting.take().is_some() {
                        dropped.mark(index);
                    }
                    Stop::Done
                }
                Ok(false) => Stop::Take(Box::new(DecodeError::Refused.into())),
                Err(raise) => match at.get() {
                    None => Stop::Take(Box::new(raise.into())),
                    // The reset of a field naming a refused object raised: that field's.
                    Some(place) if resetting.get() == Some(place) => {
                        Stop::At(place, Err(Box::new(raise.into())))
                    }
                    Some(place) => {
                        let done = settle(dropped, index, Err(Box::new(raise.into())), &reset);
                        Stop::At(place, done)
                    }
                },
            }
        });
        let place = match ran {
            Ok(Stop::Done) => return tally.carried(total - counted - failed.get()),
            Ok(Stop::Take(failure)) => return tally.count(obj, index, Err(failure)),
            Ok(Stop::At(place, done)) => {
                tally.carried(place - counted - failed.get());
                tally.count(obj, index, done);
                place
            }
            Err(refusal) => {
                tally.count(obj, index, Err(Box::new(refused_raise(&refusal).into())));
                let Some(place) = at.get() else {
                    return;
                };
                tally.carried(place - counted - failed.get());
                place
            }
        };
        counted = place + 1;
        if counted >= total {
            return;
        }
    }
}

/// What a slot's field makes of its command's encode ([`Walk`]): the command to send, none (a
/// dropped command's draw or dispatch, carried out as nothing), or the field's failure.
enum Made<Cmd> {
    Send(Cmd),
    Skip,
    Fail(Box<Failure>),
}

/// A setter's command, or its failure, as [`Made`].
#[inline(always)]
fn made<Cmd>(cmd: Result<Cmd, Box<Failure>>) -> Made<Cmd> {
    match cmd {
        Ok(cmd) => Made::Send(cmd),
        Err(failure) => Made::Fail(failure),
    }
}

/// A record's field `op` of command `index`, as [`Dropped`] holds it, as [`contain`] holds a
/// record: a reset carries the command out again; a draw or dispatch of a dropped command is
/// dropped; else the command `cmd` makes, or its failure.
#[inline(always)]
fn carry<Cmd>(
    dropped: &Dropped,
    index: u64,
    op: u16,
    cmd: impl FnOnce() -> RaiseResult<Result<Cmd, DecodeError>>,
) -> Made<Cmd> {
    if resets(op) {
        dropped.clear(index, 1);
    } else if acts(op) && dropped.holds(index) {
        return Made::Skip;
    }
    match cmd() {
        Ok(Ok(cmd)) => Made::Send(cmd),
        Ok(Err(e)) => Made::Fail(Box::new(e.into())),
        Err(raise) => Made::Fail(Box::new(raise.into())),
    }
}

/// The commands `make` makes of a slot's fields, from place `next` on, in turn, each field
/// settled as its record would be ([`settle`]): a dropped command's draw or dispatch is passed
/// over; a field naming an object whose create the host refused hands out the command's reset
/// (`reset`), noted in `resetting`, and drops the command ([`Dropped::mark`]) once the reset was
/// sent; another field it cannot make is counted on `tally` as not carried out, and on `failed`.
/// `at` holds the place of the field being made, and so of the command last handed out while it
/// is sent: a platform takes the next only once it sent the one before
/// ([`IndirectCommandBuffer::encode_render_command`]).
struct Walk<'w, 's, M, R> {
    fields: Fields<'s>,
    next: usize,
    make: &'w M,
    reset: &'w R,
    at: &'w Cell<Option<usize>>,
    resetting: &'w Cell<Option<usize>>,
    failed: &'w Cell<usize>,
    dropped: &'w Dropped,
    /// The buffer's id and the command's index.
    command: (u64, u64),
    tally: &'w mut Tally,
}

impl<'s, Cmd, M: Fn(Place<'s>) -> Made<Cmd>, R: Fn() -> Cmd> Iterator for Walk<'_, 's, M, R> {
    type Item = Cmd;

    #[inline(always)]
    fn next(&mut self) -> Option<Cmd> {
        let (obj, index) = self.command;
        if self.resetting.take().is_some() {
            self.dropped.mark(index);
        }
        loop {
            let field = self.fields.next()?;
            let place = self.next;
            self.next += 1;
            self.at.set(Some(place));
            match (self.make)(field) {
                Made::Send(cmd) => return Some(cmd),
                Made::Skip => {}
                Made::Fail(failure) => match failure.refused() {
                    true => {
                        self.resetting.set(Some(place));
                        return Some((self.reset)());
                    }
                    false => {
                        self.failed.set(self.failed.get() + 1);
                        self.tally.count(obj, index, Err(failure));
                    }
                },
            }
        }
    }
}

/// `record`, an encode `op` of command `index`, as [`Dropped`] holds it: a reset of the command
/// carries it out again; a draw or dispatch of a dropped command is dropped; an encode that names
/// an object whose create the host refused is dropped, the command reset (`reset`) and dropped
/// ([`settle`]).
#[inline]
fn contain(
    dropped: &Dropped,
    index: u64,
    op: u16,
    record: impl FnOnce() -> Carried,
    reset: impl FnOnce() -> RaiseResult<()>,
) -> Carried {
    if resets(op) {
        dropped.clear(index, 1);
    } else if acts(op) && dropped.holds(index) {
        return Ok(());
    }
    settle(dropped, index, record(), reset)
}

/// `done`, an encode of command `index` carried out, as [`Dropped`] holds it: one that named an
/// object whose create the host refused is dropped, the command reset (`reset`) and dropped.
#[inline]
fn settle(
    dropped: &Dropped,
    index: u64,
    done: Carried,
    reset: impl FnOnce() -> RaiseResult<()>,
) -> Carried {
    match done {
        Err(failure) if failure.refused() => {
            raised(reset())?;
            dropped.mark(index);
            Ok(())
        }
        done => done,
    }
}

/// A copy of the `count` records of `stride` bytes from `first`, each with its first `taken` `u64`
/// words swapped for 0 as it is copied, as [`Region::take_records`] takes them from the region:
/// `None`, taking nothing, where `first` and `stride` are not 8-aligned, or `stride` holds fewer
/// than `taken` words.
///
/// # Safety
/// The `count * stride` bytes from `first` are host memory that stays mapped, readable and
/// writable, while this runs: the guest's, which it may write meanwhile, and only through atomics
/// where the swaps take words.
///
/// [`Region::take_records`]: reims_vgpu_graphics::region::Region::take_records
unsafe fn take_records_at(
    first: HostPtr,
    stride: u64,
    count: u64,
    taken: usize,
) -> Option<Vec<u8>> {
    let len = stride.checked_mul(count)?;
    let words = taken as u64 * 8;
    if !first.addr().is_multiple_of(8) || !stride.is_multiple_of(8) || stride < words {
        return None;
    }
    let mut out = vec![0u8; usize::try_from(len).ok()?];
    for (i, record) in out.chunks_exact_mut(stride as usize).enumerate() {
        let at = first.wrapping_add(i as u64 * stride);
        for w in 0..taken {
            // SAFETY: an 8-aligned word of the records (checked above), which the caller keeps
            // mapped; the guest writes it only through atomics.
            let word = unsafe { AtomicU64::from_ptr(at.wrapping_add(w as u64 * 8).as_ptr()) }
                .swap(0, Ordering::AcqRel);
            record[w * 8..w * 8 + 8].copy_from_slice(&word.to_le_bytes());
        }
        // SAFETY: the record's bytes lie in the records, which the caller keeps mapped; the
        // guest's concurrent writes change only what is copied.
        unsafe {
            std::ptr::copy_nonoverlapping(
                at.wrapping_add(words).as_ptr::<u8>(),
                record[taken * 8..].as_mut_ptr(),
                record.len() - taken * 8,
            );
        }
    }
    Some(out)
}

/// The `k`th `u64` of a slot (its masks are words 0 and 1).
#[inline]
fn word(slot: &[u8], k: usize) -> u64 {
    u64::read_from_prefix(slot.get(k * 8..).unwrap_or_default()).map_or(0, |(w, _)| w)
}

/// A field of a slot, as [`Fields`] hands it on: a record's op and body, or a setter.
#[derive(Clone, Copy)]
enum Place<'s> {
    /// A field its record carries out: the reset, the stage-in region, the imageblock, the
    /// action and the barrier, and op 0 for a bit the slot's kind does not hold.
    Record(u16, &'s [u8]),
    /// A setter ([`Sets`]).
    Set(Set),
}

/// A setter of a slot, taken from the slot as it lies there.
#[derive(Clone, Copy)]
enum Set {
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

/// A slot's setters ([`Sets::each`]): its pipeline, where its bit is set, and its bindings whose
/// bits are, of its `a` of the first kind and of the second. What it holds is worked out from
/// the slot's bits once, where it is made ([`Fields::of`]): worked out at each count and each
/// walk, from a copy of the `Sets`, its two masks were read back as one 2-lane vector for a
/// software popcount, the inner walk's hottest line (measured on ms01).
#[derive(Clone, Copy)]
struct Sets<'s> {
    /// The slot.
    slot: &'s [u8],
    /// Whether the slot is a render command's.
    render: bool,
    pipeline: Option<u64>,
    /// The slot's `strided` bits.
    strided: u32,
    /// The bits of its bindings of the first kind set, by index, and of the second.
    firsts: u64,
    seconds: u64,
    /// The slot's bindings of the first kind, which its bindings of the second follow.
    a: u32,
    /// How many setters it holds.
    len: usize,
}

impl<'s> Sets<'s> {
    /// The setters a slot's bits `(mask, mask_b)` name of its `a` bindings of the first kind and
    /// `b` of the second, `pipeline` its pipeline state's.
    #[inline]
    fn of(
        (slot, render, strided): (&'s [u8], bool, u32),
        (mask, mask_b): (u64, u64),
        (a, b): (u32, u32),
        pipeline: u64,
    ) -> Sets<'s> {
        let bits = |n: u32| (1u64 << n) - 1;
        let pipeline = (mask & RGFX_ICB_SLOT_PIPELINE != 0).then_some(pipeline);
        let firsts = (mask >> RGFX_ICB_SLOT_BUFFER_0.trailing_zeros()) & bits(a);
        let seconds = mask_b & bits(b);
        let len = usize::from(pipeline.is_some())
            + firsts.count_ones() as usize
            + seconds.count_ones() as usize;
        Sets {
            slot,
            render,
            pipeline,
            strided,
            firsts,
            seconds,
            a,
            len,
        }
    }

    /// How many setters it holds.
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    /// Each setter, in the slot's order: the pipeline, the bindings of the first kind by index,
    /// then of the second (a render slot's fragment buffers, a compute slot's threadgroup memory
    /// lengths).
    #[inline]
    fn each(self) -> Each<'s> {
        Each {
            pipeline: self.pipeline,
            firsts: self.firsts,
            seconds: self.seconds,
            sets: self,
        }
    }

    /// The slot's fixed part's size.
    #[inline]
    fn fixed(&self) -> usize {
        if self.render {
            size_of::<RgfxIcbRenderSlot>()
        } else {
            size_of::<RgfxIcbComputeSlot>()
        }
    }

    /// The slot's binding `i`, of the first kind's, then the second's.
    #[inline]
    fn binding(&self, i: u32) -> RgfxIcbSlotBuffer {
        let at = self.fixed() + i as usize * size_of::<RgfxIcbSlotBuffer>();
        RgfxIcbSlotBuffer::read_from_prefix(self.slot.get(at..).unwrap_or_default())
            .map_or_else(|_| Default::default(), |(s, _)| s)
    }

    /// A compute slot's threadgroup memory length `j`.
    #[inline]
    fn length(&self, j: u32) -> u32 {
        let lengths = self.fixed() + self.a as usize * size_of::<RgfxIcbSlotBuffer>();
        let at = lengths + j as usize * 4;
        u32::read_from_prefix(self.slot.get(at..).unwrap_or_default()).map_or(0, |(l, _)| l)
    }
}

/// A slot's setters not yet handed on ([`Sets::each`]): its pipeline, and the bits of its
/// bindings, by index.
#[derive(Clone, Copy)]
struct Each<'s> {
    sets: Sets<'s>,
    pipeline: Option<u64>,
    firsts: u64,
    seconds: u64,
}

impl Iterator for Each<'_> {
    type Item = Set;

    #[inline(always)]
    fn next(&mut self) -> Option<Set> {
        let s = &self.sets;
        if let Some(pipeline) = self.pipeline.take() {
            Some(Set::Pipeline(pipeline))
        } else if self.firsts != 0 {
            let i = self.firsts.trailing_zeros();
            self.firsts &= self.firsts - 1;
            Some(Set::Binding(i, s.binding(i), s.strided & (1 << i) != 0))
        } else if self.seconds != 0 {
            let j = self.seconds.trailing_zeros();
            self.seconds &= self.seconds - 1;
            Some(match s.render {
                true => Set::Second(j, s.binding(s.a + j)),
                false => Set::Length(j, s.length(j)),
            })
        } else {
            None
        }
    }
}

/// The fields a slot's bits name, each a place in a fixed order ([`Fields::of`]): the command's
/// reset, its setters (its pipeline, its bindings of the first kind by index, of the second,
/// [`Sets`]), the stage-in region and the imageblock (compute), the draw or the dispatch, and the
/// barrier. A bit the slot's kind does not hold, or an action whose op is not one of the kind's
/// draws or dispatches, is op 0, which no record is: the record refuses it. The bits are worked
/// out once, where it is made, and each field read from the slot as it is handed on.
#[derive(Clone, Copy)]
struct Fields<'s> {
    slot: &'s [u8],
    render: bool,
    /// The records not yet handed on, by their bits ([`Fields::UNKNOWN`] and on), the setters
    /// coming after the reset's.
    records: u8,
    /// The action's op and the barrier's word.
    op: u16,
    barrier: u16,
    sets: Each<'s>,
    /// How many fields it holds in all.
    len: usize,
}

impl<'s> Fields<'s> {
    /// A bit the slot's kind does not hold, the reset, the stage-in region, the imageblock, the
    /// action and the barrier, in their order.
    const UNKNOWN: u8 = 1 << 0;
    const RESET: u8 = 1 << 1;
    const STAGE_IN: u8 = 1 << 2;
    const IMAGEBLOCK: u8 = 1 << 3;
    const ACTION: u8 = 1 << 4;
    const BARRIER: u8 = 1 << 5;

    /// The fields of a slot `slot` of `kind`, its bits `masks` of its `a` bindings of the first
    /// kind and `b` of the second. A slot too short for its kind's fixed part names the unknown
    /// bits' field alone (the run's shape rules it out).
    #[inline(always)]
    fn of(kind: u16, slot: &'s [u8], (a, b): (u32, u32), (mask, mask_b): (u64, u64)) -> Self {
        let render = kind == RGFX_ICB_SLOTS_RENDER;
        let bits = |n: u32| (1u64 << n) - 1;
        let mut known = RGFX_ICB_SLOT_RESET
            | RGFX_ICB_SLOT_PIPELINE
            | RGFX_ICB_SLOT_ACTION
            | RGFX_ICB_SLOT_BARRIER
            | (bits(a) * RGFX_ICB_SLOT_BUFFER_0);
        if !render {
            known |= RGFX_ICB_SLOT_STAGE_IN | RGFX_ICB_SLOT_IMAGEBLOCK;
        }
        let mut records = 0;
        if mask & !known != 0 || mask_b & !bits(b) != 0 {
            records |= Self::UNKNOWN;
        }
        // The fixed part's words: the two masks, the pipeline, the op and the barrier, the
        // strided bits; the kinds share them.
        let fixed = match render {
            true => size_of::<RgfxIcbRenderSlot>(),
            false => size_of::<RgfxIcbComputeSlot>(),
        };
        if slot.len() < fixed {
            let none = Sets::of((slot, render, 0), (0, 0), (a, b), 0);
            return Fields {
                slot,
                render,
                records,
                op: 0,
                barrier: 0,
                sets: none.each(),
                len: records.count_ones() as usize,
            };
        }
        let half = |at: usize| u16::from_ne_bytes([slot[at], slot[at + 1]]);
        let (pipeline, op, barrier) = (word(slot, 2), half(24), half(26));
        let strided = u32::from_ne_bytes([slot[28], slot[29], slot[30], slot[31]]);
        let sets = Sets::of((slot, render, strided), (mask, mask_b), (a, b), pipeline);
        let fields = [
            (RGFX_ICB_SLOT_RESET, Self::RESET),
            (RGFX_ICB_SLOT_STAGE_IN, Self::STAGE_IN),
            (RGFX_ICB_SLOT_IMAGEBLOCK, Self::IMAGEBLOCK),
            (RGFX_ICB_SLOT_ACTION, Self::ACTION),
            (RGFX_ICB_SLOT_BARRIER, Self::BARRIER),
        ];
        for (bit, field) in fields {
            if mask & bit != 0 {
                records |= field;
            }
        }
        if render {
            records &= !(Self::STAGE_IN | Self::IMAGEBLOCK);
        }
        Fields {
            slot,
            render,
            records,
            op,
            barrier,
            len: records.count_ones() as usize + sets.len(),
            sets: sets.each(),
        }
    }

    /// How many fields it holds in all, those handed on included.
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    /// Its fields after its first `n`.
    #[inline(always)]
    fn after(mut self, n: usize) -> Self {
        for _ in 0..n {
            self.next();
        }
        self
    }

    /// The record of field `bit`.
    #[inline(always)]
    fn record(&self, bit: u8) -> Place<'s> {
        let slot = self.slot;
        let body = |at: usize, len: usize| slot.get(at..at + len).unwrap_or_default();
        let (op, body) = match (bit, self.render) {
            (Self::UNKNOWN, _) => (0, &[][..]),
            (Self::RESET, true) => (RGFX_ICB_OP_RENDER_RESET, &[][..]),
            (Self::RESET, false) => (RGFX_ICB_OP_COMPUTE_RESET, &[][..]),
            (Self::STAGE_IN, _) => (
                RGFX_ICB_OP_STAGE_IN_REGION,
                body(
                    offset_of!(RgfxIcbComputeSlot, stage_in),
                    size_of::<RgfxIcbRegion>(),
                ),
            ),
            (Self::IMAGEBLOCK, _) => (
                RGFX_ICB_OP_IMAGEBLOCK,
                body(
                    offset_of!(RgfxIcbComputeSlot, imageblock),
                    size_of::<RgfxIcbImageblock>(),
                ),
            ),
            (Self::ACTION, true) => {
                let draws = RGFX_ICB_OP_DRAW..=RGFX_ICB_OP_DRAW_INDEXED_PATCHES;
                let op = if draws.contains(&self.op) { self.op } else { 0 };
                let at = offset_of!(RgfxIcbRenderSlot, action);
                (op, body(at, size_of::<[u64; 9]>()))
            }
            (Self::ACTION, false) => {
                let dispatches = RGFX_ICB_OP_DISPATCH_THREADGROUPS..=RGFX_ICB_OP_DISPATCH_THREADS;
                let op = if dispatches.contains(&self.op) {
                    self.op
                } else {
                    0
                };
                let at = offset_of!(RgfxIcbComputeSlot, dispatch);
                (op, body(at, size_of::<RgfxIcbDispatch>()))
            }
            (_, true) => match self.barrier != 0 {
                true => (RGFX_ICB_OP_RENDER_SET_BARRIER, &[][..]),
                false => (RGFX_ICB_OP_RENDER_CLEAR_BARRIER, &[][..]),
            },
            (_, false) => match self.barrier != 0 {
                true => (RGFX_ICB_OP_COMPUTE_SET_BARRIER, &[][..]),
                false => (RGFX_ICB_OP_COMPUTE_CLEAR_BARRIER, &[][..]),
            },
        };
        Place::Record(op, body)
    }
}

impl<'s> Iterator for Fields<'s> {
    type Item = Place<'s>;

    #[inline(always)]
    fn next(&mut self) -> Option<Place<'s>> {
        let first = Self::UNKNOWN | Self::RESET;
        if self.records & first == 0
            && let Some(set) = self.sets.next()
        {
            return Some(Place::Set(set));
        }
        if self.records == 0 {
            return None;
        }
        let bit = self.records & self.records.wrapping_neg();
        self.records &= !bit;
        Some(self.record(bit))
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
        RGFX_CMD_ICB_SLOTS, RGFX_CMD_ICB_SLOTS_IN_PAGES, RGFX_CMD_SURFACE_PAGES,
        RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CONTEXT_OBJ_ID_SHIFT, RGFX_CURSOR_SPAN,
        RGFX_ICB_SLOTS_COMPUTE, RGFX_SURFACE_PAGES_PRIVATE, RgfxPageRun, RgfxSurfacePages,
    };
    use reims_vgpu_vulkan::core::memory::descriptor::{GuestPhysicalRange, MemoryMapDescriptor};
    use reims_vgpu_vulkan::core::memory::map::MemoryMap;
    use reims_vgpu_vulkan::core::testkit::GuestRam;

    use super::super::tests::{
        compute_create, pipeline_decoder, pipeline_decoder_over, whole_render_create,
    };
    use super::*;

    type Dec = GraphicsDecoder<FakePlatform>;
    type Device = <FakePlatform as Platform>::Device;

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

    /// A slot's field the platform dies at (a refusal of the Vulkan platform's, which a death
    /// is) is refused alone: the slot's fields before and after it are carried out, and so are
    /// the next slot's. A death at the take of a slot's command refuses that slot alone.
    #[test]
    fn a_slot_field_the_platform_dies_at_is_refused_alone() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let fixed = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_ACTION | RGFX_ICB_SLOT_BUFFER_0,
            mask_b: 1,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(3)),
            ..RgfxIcbRenderSlot::default()
        };
        let none = RgfxIcbSlotBuffer::default();
        let bindings = [bound(2, 16, 0), none, bound(3, 64, 0)];
        for i in 0..3 {
            assert!(region.write(SLOTS + i * RENDER_STRIDE, &render_slot(&fixed, bindings)));
        }
        let message = slots(&[run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 3)]);
        p.script().push(FB, Step::crash());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(
            encodes(&p, mark),
            [VB, FB, DRAW, VB, FB, DRAW, VB, FB, DRAW]
        );
        let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB_2) else {
            panic!("no indirect command buffer");
        };
        for i in 0..3 {
            let c = icb.render_command(i).unwrap().unwrap();
            assert_eq!(c.draw_args().unwrap().vertex_count, 3, "command {i}");
            assert_ne!(c.vertex_buffer_address(0).unwrap(), 0, "command {i}");
            let fragment = c.fragment_buffer_address(0).unwrap();
            assert_eq!(fragment == 0, i == 0, "command {i}");
        }
        dec.end_message();

        for i in 0..3 {
            assert!(region.write(SLOTS + i * RENDER_STRIDE, &render_slot(&fixed, bindings)));
        }
        let at = "indirectRenderCommandAtIndex:";
        p.script().after(at, 1, Step::crash());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let taken: Vec<_> = calls(&p, mark).into_iter().map(|(sel, _)| sel).collect();
        assert_eq!(taken, [at, VB, FB, DRAW, at, at, VB, FB, DRAW]);
    }

    /// A slot's setters go to the platform in one call, and the one that stops it is refused
    /// alone, as its record would be: a raise, a death, a buffer the sender holds none of, and
    /// a buffer whose create the host refused, which resets the command and drops its draw. The
    /// setters before it stay set, and those after it are sent.
    #[test]
    fn a_slot_setter_that_stops_its_call_is_refused_alone() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        dec.refuse(98, RGFX_CMD_BUFFER_CREATE);
        let fixed = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | RGFX_ICB_SLOT_BUFFER_0
                | (RGFX_ICB_SLOT_BUFFER_0 << 1),
            mask_b: 1,
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW,
            action: action(&draw(3)),
            ..RgfxIcbRenderSlot::default()
        };
        let held = [bound(2, 16, 0), bound(2, 32, 0), bound(3, 64, 0)];
        let pipeline = "setRenderPipelineState:";
        let set = [pipeline, VB, VB, FB, DRAW];
        // Each case: the command, its bindings, what stops the call (the vertex buffer sends let
        // run before it, and its step), what is sent, and which of its vertex buffers, fragment
        // buffer and draw stand.
        type Case<'a> = (
            u64,
            [RgfxIcbSlotBuffer; 3],
            Option<(u64, Step)>,
            &'a [&'a str],
            [bool; 4],
        );
        let raise = || Step::raise("NSInvalidArgumentException", "refused");
        let cases: [Case<'_>; 4] = [
            (0, held, Some((0, raise())), &set, [false, true, true, true]),
            (
                1,
                held,
                Some((1, Step::crash())),
                &set,
                [true, false, true, true],
            ),
            (
                2,
                [bound(99, 0, 0), held[1], held[2]],
                None,
                &[pipeline, VB, FB, DRAW],
                [false, true, true, true],
            ),
            (
                3,
                [held[0], bound(98, 0, 0), held[2]],
                None,
                &[pipeline, VB, "reset", FB],
                [false, false, true, false],
            ),
        ];
        for (index, bindings, stop, sent, stand) in cases {
            assert!(region.write(SLOTS, &render_slot(&fixed, bindings)));
            let message = slots(&[run(
                RENDER_ICB_2,
                RGFX_ICB_SLOTS_RENDER,
                SLOTS,
                index as u32,
                1,
            )]);
            if let Some((skip, step)) = stop {
                p.script().after(VB, skip, step);
            }
            let mark = p.call_log().mark();
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
            assert_eq!(encodes(&p, mark), sent, "command {index}");
            dec.end_message();
            let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB_2) else {
                panic!("no indirect command buffer");
            };
            let c = icb.render_command(index).unwrap().unwrap();
            let standing = [
                c.vertex_buffer_address(0).unwrap() != 0,
                c.vertex_buffer_address(1).unwrap() != 0,
                c.fragment_buffer_address(0).unwrap() != 0,
                c.draw_args().is_some_and(|d| d.vertex_count == 3),
            ];
            assert_eq!(standing, stand, "command {index}");
        }
    }

    /// A slot's field the platform refuses, by a raise or by a death (a refusal of the Vulkan
    /// platform's), is refused alone wherever it lies in the slot, as its record would be: its
    /// first field, a middle one or its last, of a render slot and of a compute slot. The fields
    /// before and after it stand. The slot goes to the platform in one call, which takes the
    /// command and sends its fields, and the fields after the refused one in one more.
    #[test]
    fn a_slot_field_refused_first_middle_or_last_is_refused_alone() {
        let (p, mut dec, region) = slot_setup();
        let device = p.device();
        let b0 = RGFX_ICB_SLOT_BUFFER_0;
        let held = [bound(2, 16, 0), bound(2, 32, 0), bound(3, 64, 0)];
        let steps = || {
            [
                Step::raise("NSInvalidArgumentException", "refused"),
                Step::crash(),
            ]
        };
        // The calls one slot of `fields` drives where the field at `place` is refused.
        let sent = |take: &'static str, fields: &[&'static str], place: usize| {
            let mut sent = vec![take];
            sent.extend_from_slice(&fields[..=place]);
            if place + 1 < fields.len() {
                sent.push(take);
                sent.extend_from_slice(&fields[place + 1..]);
            }
            sent
        };

        let render = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | RGFX_ICB_SLOT_BARRIER
                | b0
                | b0 << 1,
            mask_b: 1,
            pipeline: RENDER_PIPELINE,
            op: RGFX_ICB_OP_DRAW,
            barrier: 1,
            action: action(&draw(3)),
            ..RgfxIcbRenderSlot::default()
        };
        let (pipeline, barrier) = ("setRenderPipelineState:", "setBarrier");
        let fields = [pipeline, VB, VB, FB, DRAW, barrier];
        // Each case: the refused field's place, its selector, and how many of its sends to let
        // run before it.
        let cases = [(0, pipeline, 0), (2, VB, 1), (5, barrier, 0)];
        let mut index = 0u32;
        for (place, sel, skip) in cases {
            for step in steps() {
                assert!(region.write(SLOTS, &render_slot(&render, held)));
                let message = slots(&[run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, SLOTS, index, 1)]);
                p.script().after(sel, skip, step);
                let mark = p.call_log().mark();
                assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
                dec.end_message();
                let got: Vec<_> = calls(&p, mark).into_iter().map(|(sel, _)| sel).collect();
                let take = "indirectRenderCommandAtIndex:";
                assert_eq!(got, sent(take, &fields, place), "place {place}");
                let Some(ObjHandle::Icb(icb)) = dec.named(RENDER_ICB_2) else {
                    panic!("no indirect command buffer");
                };
                let c = icb.render_command(index.into()).unwrap().unwrap();
                let standing = [
                    c.pipeline_state_unique_identifier() != 0,
                    c.vertex_buffer_address(0).unwrap() != 0,
                    c.vertex_buffer_address(1).unwrap() != 0,
                    c.fragment_buffer_address(0).unwrap() != 0,
                    c.draw_args().is_some_and(|d| d.vertex_count == 3),
                    c.has_barrier(),
                ];
                let mut stand = [true; 6];
                stand[place] = false;
                assert_eq!(standing, stand, "place {place}");
                index += 1;
            }
        }

        let compute = RgfxIcbComputeSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
                | RGFX_ICB_SLOT_BARRIER
                | RGFX_ICB_SLOT_STAGE_IN
                | RGFX_ICB_SLOT_IMAGEBLOCK
                | b0
                | b0 << 1,
            mask_b: 1,
            pipeline: COMPUTE_PIPELINE,
            op: RGFX_ICB_OP_DISPATCH_THREADGROUPS,
            barrier: 1,
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
            ..RgfxIcbComputeSlot::default()
        };
        let pipeline = "setComputePipelineState:";
        let kb = "setKernelBuffer:offset:atIndex:";
        let length = "setThreadgroupMemoryLength:atIndex:";
        let fields = [
            pipeline,
            kb,
            kb,
            length,
            "setStageInRegion:",
            "setImageblockWidth:height:",
            "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
            barrier,
        ];
        let cases = [(0, pipeline, 0), (3, length, 0), (7, barrier, 0)];
        let mut index = 0u32;
        for (place, sel, skip) in cases {
            for step in steps() {
                let slot = compute_slot(&compute, held, &[(0, 512)]);
                assert!(region.write(SLOTS, &slot));
                let message = slots(&[run(COMPUTE_ICB_2, RGFX_ICB_SLOTS_COMPUTE, SLOTS, index, 1)]);
                p.script().after(sel, skip, step);
                let mark = p.call_log().mark();
                assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
                dec.end_message();
                let got: Vec<_> = calls(&p, mark).into_iter().map(|(sel, _)| sel).collect();
                let take = "indirectComputeCommandAtIndex:";
                assert_eq!(got, sent(take, &fields, place), "place {place}");
                let Some(ObjHandle::Icb(icb)) = dec.named(COMPUTE_ICB_2) else {
                    panic!("no indirect command buffer");
                };
                let c = icb.compute_command(index.into()).unwrap().unwrap();
                let standing = [
                    c.pipeline_state_unique_identifier() != 0,
                    c.kernel_buffer_address(0).unwrap() != 0,
                    c.kernel_buffer_address(1).unwrap() != 0,
                    c.threadgroup_memory_length(0).unwrap() == 512,
                    c.stage_in_region().size.depth == 6,
                    c.image_block_args().is_some_and(|b| b.width == 16),
                    c.dispatch_threadgroups_args().is_some(),
                    c.has_barrier(),
                ];
                let mut stand = [true; 8];
                stand[place] = false;
                assert_eq!(standing, stand, "place {place}");
                index += 1;
            }
        }
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

    /// The guest RAM's first byte in the tests whose slots lie in a pages block.
    const RAM: u64 = 0x10_0000;
    /// Where those tests' pages blocks lie in the region: past [`SLOTS`]' render and compute
    /// slots, each block's runs and holders' part in 0x2000 bytes.
    const PAGES_BLOCK: u64 = SLOTS + 0x8_0000;

    /// [`slot_setup`] with `pages` host pages of guest RAM from [`RAM`] behind the decoder's
    /// memory map; and the guest RAM, and the host's page.
    fn pages_slot_setup(pages: u64) -> (Arc<FakePlatform>, Dec, Arc<Region>, GuestRam, u64) {
        let (p, mut dec, region) = slot_setup();
        let page = p.vm().page_size();
        let ram = GuestRam::with_range(RAM, pages * page).unwrap();
        p.fake_vm().adopt(&ram);
        let mut desc = MemoryMapDescriptor::new();
        for r in ram.ranges() {
            desc.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.physical().physical_length,
                host_address: r.host(),
            });
        }
        let map = MemoryMap::new(Arc::clone(&p), &desc).unwrap();
        dec.set_guest_memory(map, Arc::new(()));
        (p, dec, region, ram, page)
    }

    /// A `SURFACE_PAGES` of a pages block (`RGFX_SURFACE_PAGES_PRIVATE`) at `block`, of `runs`,
    /// (page, pages) of the guest RAM, over their bytes; its answer.
    fn describe_pages(
        dec: &mut Dec,
        device: &Device,
        block: u64,
        runs: &[(u64, u64)],
        page: u64,
    ) -> u32 {
        let runs: Vec<RgfxPageRun> = runs
            .iter()
            .map(|&(at, n)| RgfxPageRun {
                address: RAM + at * page,
                length: n * page,
            })
            .collect();
        let m = RgfxSurfacePages {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SURFACE_PAGES,
                size: 0,
                seqno: 0,
            },
            context: 0,
            run_count: runs.len() as u32,
            block_offset: block,
            block_size: 0x2000,
            runs_offset: block + 0x1000,
            length: runs.iter().map(|r| r.length).sum(),
        };
        dec.surface_pages_laid_out(device, &m, &runs, RGFX_SURFACE_PAGES_PRIVATE, None)
    }

    /// The guest physical spans of `len` bytes at `offset` in the pages of `runs` (as
    /// [`describe_pages`]), in order.
    fn scatter(runs: &[(u64, u64)], page: u64, offset: u64, len: u64) -> Vec<(u64, u64)> {
        let mut spans = Vec::new();
        let (mut at, end, mut start) = (offset, offset + len, 0);
        for &(first, n) in runs {
            let run_end = start + n * page;
            if at < run_end && at < end {
                let take = run_end.min(end) - at;
                spans.push((RAM + first * page + (at - start), take));
                at += take;
            }
            start = run_end;
        }
        assert_eq!(at, end, "the bytes lie in the runs");
        spans
    }

    /// Writes `bytes` at `offset` in the pages of `runs`.
    fn write_pages(ram: &GuestRam, runs: &[(u64, u64)], page: u64, offset: u64, bytes: &[u8]) {
        let mut from = 0;
        for (pa, n) in scatter(runs, page, offset, bytes.len() as u64) {
            assert!(ram.write(pa, &bytes[from..from + n as usize]));
            from += n as usize;
        }
    }

    /// The masks of the slot at `offset` in the pages of `runs`.
    fn pages_masks(ram: &GuestRam, runs: &[(u64, u64)], page: u64, offset: u64) -> (u64, u64) {
        let mut b = Vec::new();
        for (pa, n) in scatter(runs, page, offset, 16) {
            b.extend(ram.read_vec(pa, n).unwrap());
        }
        (word(&b, 0), word(&b, 1))
    }

    /// A run of an `ICB_SLOTS_IN_PAGES`: [`run`]'s, its slots `offset` bytes into the pages of
    /// the block at `block`.
    fn pages_run(
        icb: u64,
        kind: u16,
        block: u64,
        offset: u64,
        first: u32,
        count: u32,
    ) -> RgfxIcbSlotPagesRun {
        let r = run(icb, kind, 0, first, count);
        RgfxIcbSlotPagesRun {
            icb_obj_id: r.icb_obj_id,
            block_offset: block,
            block_size: 0x2000,
            pages_offset: offset,
            first_command: r.first_command,
            command_count: r.command_count,
            slot_bytes: r.slot_bytes,
            kind: r.kind,
            bindings_a: r.bindings_a,
            bindings_b: r.bindings_b,
        }
    }

    /// An `ICB_SLOTS_IN_PAGES` of `runs`.
    fn pages_slots(runs: &[RgfxIcbSlotPagesRun]) -> Vec<u8> {
        let mut m = slots(&[]);
        let len = m.len() + size_of_val(runs);
        let head = RgfxIcbSlots {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_SLOTS_IN_PAGES,
                size: len as u32,
                seqno: 0,
            },
            obj_id: runs.first().map_or(0, |r| r.icb_obj_id),
            run_count: runs.len() as u32,
            reserved_1c: 0,
        };
        m.copy_from_slice(head.as_bytes());
        for r in runs {
            m.extend_from_slice(r.as_bytes());
        }
        m
    }

    /// The render and compute slots [`slots_in_pages_drive_what_slots_in_the_region_do`] writes:
    /// a render command with every field of its kind, one reset then given a pipeline, and a
    /// compute command with every field of its kind.
    fn slot_bytes_of_each_kind() -> (Vec<u8>, Vec<u8>) {
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
        let mut render = render_slot(&fixed, [bound(2, 16, 0), bound(0, 8, 24), bound(3, 32, 0)]);
        let reset_then_pipeline = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_RESET | RGFX_ICB_SLOT_PIPELINE,
            pipeline: RENDER_PIPELINE,
            ..RgfxIcbRenderSlot::default()
        };
        render.extend(render_slot(
            &reset_then_pipeline,
            [RgfxIcbSlotBuffer::default(); 3],
        ));
        let dispatch = RgfxIcbComputeSlot {
            mask: RGFX_ICB_SLOT_PIPELINE
                | RGFX_ICB_SLOT_ACTION
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
        let compute = compute_slot(&dispatch, kernels, &[(1, 256), (30, 64)]);
        (render, compute)
    }

    /// Slots in a pages block's pages drive the setters the same slots in the region do, read
    /// in place in the guest's RAM where the block is one run, and in its view where it is more,
    /// a slot across two runs among them; the take leaves their masks 0, so the same message
    /// again carries out nothing, and the pipelines they set are what the message waits for until
    /// then.
    #[test]
    fn slots_in_pages_drive_what_slots_in_the_region_do() {
        let (p, mut dec, region, ram, page) = pages_slot_setup(8);
        let device = p.device();
        let (render, compute) = slot_bytes_of_each_kind();
        // One run of two pages, and two runs out of order, the second render slot across them.
        let one = (PAGES_BLOCK, vec![(0, 2)]);
        let two = (PAGES_BLOCK + 0x4000, vec![(5, 1), (3, 2)]);
        let r = page - RENDER_STRIDE - 64;
        let c = 2 * page;
        for (block, runs) in [&one, &two] {
            assert_eq!(
                describe_pages(&mut dec, &device, *block, runs, page),
                RGFX_STATUS_OK
            );
            let r_at = if runs.len() == 1 { 0 } else { r };
            let region_message = {
                assert!(region.write(SLOTS, &render));
                assert!(region.write(SLOTS + 0x1_0000, &compute));
                slots(&[
                    run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 2),
                    run(
                        COMPUTE_ICB_2,
                        RGFX_ICB_SLOTS_COMPUTE,
                        SLOTS + 0x1_0000,
                        0,
                        1,
                    ),
                ])
            };
            let mark = p.call_log().mark();
            assert_eq!(dec.decode(&device, &region_message).unwrap(), Ok(()));
            let by_region = calls(&p, mark);
            assert!(!by_region.is_empty());

            write_pages(&ram, runs, page, r_at, &render);
            let c_at = if runs.len() == 1 {
                RENDER_STRIDE * 2
            } else {
                c
            };
            write_pages(&ram, runs, page, c_at, &compute);
            let message = pages_slots(&[
                pages_run(RENDER_ICB_2, RGFX_ICB_SLOTS_RENDER, *block, r_at, 0, 2),
                pages_run(COMPUTE_ICB_2, RGFX_ICB_SLOTS_COMPUTE, *block, c_at, 0, 1),
            ]);
            assert_eq!(
                dec.icb_slots_in_pages_pipelines(&device, &message),
                vec![RENDER_PIPELINE, RENDER_PIPELINE, COMPUTE_PIPELINE]
            );
            let mark = p.call_log().mark();
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
            assert_eq!(calls(&p, mark), by_region, "the block of {runs:?}");
            // The fake platform's view of several runs reaches the guest's RAM at a sync, as a
            // mapping of the same pages does at once.
            p.fake_vm().sync();
            for at in [r_at, r_at + RENDER_STRIDE, c_at] {
                assert_eq!(
                    pages_masks(&ram, runs, page, at),
                    (0, 0),
                    "the slot at {at:#x} taken"
                );
            }
            let mark = p.call_log().mark();
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
            assert_eq!(calls(&p, mark), vec![]);
            assert_eq!(
                dec.icb_slots_in_pages_pipelines(&device, &message),
                Vec::<u64>::new()
            );
        }
    }

    /// A run of an `ICB_SLOTS_IN_PAGES` is refused whole, taking nothing, where its block is
    /// another process's, no pages block (a block of the region the sender holds, or none), its
    /// slots pass the block's pages, or lie unaligned; the runs around it are carried out.
    #[test]
    fn a_slot_run_outside_the_senders_pages_is_refused_taking_nothing() {
        let (p, mut dec, region, ram, page) = pages_slot_setup(4);
        let device = p.device();
        let runs = [(0, 2)];
        let other = PAGES_BLOCK + 0x4000;
        assert_eq!(
            describe_pages(&mut dec, &device, PAGES_BLOCK, &runs, page),
            RGFX_STATUS_OK
        );
        assert_eq!(
            describe_pages(&mut dec, &device, other, &[(2, 1)], page),
            RGFX_STATUS_OK
        );
        region.grant(0, PAGES_BLOCK, 0x2000);
        region.grant(0, SLOTS, 0x1_0000);
        region.grant(1, other, 0x2000);
        dec.set_context(Some(0));
        let pipeline = RgfxIcbRenderSlot {
            mask: RGFX_ICB_SLOT_PIPELINE,
            pipeline: RENDER_PIPELINE,
            ..RgfxIcbRenderSlot::default()
        };
        let slot = render_slot(&pipeline, [RgfxIcbSlotBuffer::default(); 3]);
        let last = 2 * page - RENDER_STRIDE;
        for at in [0, 8 * RENDER_STRIDE, last] {
            write_pages(&ram, &runs, page, at, &slot);
        }
        assert!(ram.write(RAM + 2 * page, &slot));
        let message = pages_slots(&[
            pages_run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, other, 0, 0, 1),
            pages_run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, SLOTS, 0, 0, 1),
            pages_run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, SLOTS + 0x4_0000, 0, 0, 1),
            pages_run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, PAGES_BLOCK, last, 0, 2),
            pages_run(
                RENDER_ICB,
                RGFX_ICB_SLOTS_RENDER,
                PAGES_BLOCK,
                8 * RENDER_STRIDE + 4,
                0,
                1,
            ),
            pages_run(RENDER_ICB, RGFX_ICB_SLOTS_RENDER, PAGES_BLOCK, 0, 0, 1),
        ]);
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let at = |i| ("indirectRenderCommandAtIndex:", vec![Arg::U64(i)]);
        let pipeline_arg = Arg::Obj(Some(
            dec.require_render_pipeline(RENDER_PIPELINE).unwrap().id(),
        ));
        assert_eq!(
            calls(&p, mark),
            vec![at(0), ("setRenderPipelineState:", vec![pipeline_arg])]
        );
        assert_eq!(pages_masks(&ram, &runs, page, 0), (0, 0));
        for (at, what) in [
            (last, "past the block's pages"),
            (8 * RENDER_STRIDE, "unaligned"),
        ] {
            assert_eq!(
                pages_masks(&ram, &runs, page, at),
                (RGFX_ICB_SLOT_PIPELINE, 0),
                "the slot {what} left as it was"
            );
        }
        assert_eq!(
            pages_masks(&ram, &[(2, 1)], page, 0),
            (RGFX_ICB_SLOT_PIPELINE, 0),
            "the slot in another process's block left as it was"
        );
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

    /// On the rt platform, through the decoder: a command buffer whose first work is a render pass
    /// that executes an indirect command buffer over a range a buffer holds, which a blit of an
    /// earlier command buffer wrote, then a compute dispatch, the command buffer's first, that
    /// reads a texture through a sampler state. The pass draws the range's commands, the kernel
    /// reads each texel, and both command buffers complete.
    ///
    /// rt runs an execution over a range a buffer holds as a count kernel ahead of the pass, which
    /// is then the command buffer's first compute pipeline. The test guards rt 70dfe08320, which
    /// binds rt's set 1 with that kernel: before it, the kernel's bind marked the compute bind
    /// point bound without set 1, and each dispatch after it in the command buffer read its
    /// textures and samplers through an unbound set (at metal2vulkan 5a04298008 the rendering
    /// sample's indirect path lost the device on an RTX 3090 and an RX 7900 XTX, and rt's own test
    /// of the pattern crashed lavapipe). On a device without `VK_EXT_device_generated_commands` the
    /// make answers nil, rt's gap, which the test prints and stops at.
    #[test]
    fn on_vulkan_a_dispatch_after_a_held_range_execution_reads_its_texture_and_sampler() {
        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RgfxBlitOpCopyBufferToBuffer,
        };
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_DISPATCH_THREADS, RGFX_OP_COMPUTE_SET_BUFFERS,
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RGFX_OP_COMPUTE_SET_SAMPLER_STATES,
            RGFX_OP_COMPUTE_SET_TEXTURES, RGFX_OP_COMPUTE_USE_RESOURCES, RgfxComputeBufferBinding,
            RgfxComputeOpDispatchThreads, RgfxComputeOpSetBuffers,
            RgfxComputeOpSetComputePipelineState, RgfxComputeOpSetSamplerStates,
            RgfxComputeOpSetTextures, RgfxComputeOpUseResources, RgfxComputeResourceRef,
            RgfxComputeSamplerBinding, RgfxComputeTextureBinding,
        };
        use reims_vgpu_graphics::ops::render::{
            RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpExecuteCommandsIndirect, RgfxOpSetRenderPipelineState,
        };
        use reims_vgpu_graphics::ops::{
            RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_ENCODER_BEGIN_BLIT,
            RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END,
            RGFX_RESOURCE_KIND_ICB, RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpHeader,
            RgfxRenderPassColorAttachment, RgfxRenderPassDesc,
        };
        use reims_vgpu_graphics::wire::{
            RGFX_CB_OK, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE, RgfxCbComplete,
            RgfxQueueCreate,
        };
        use reims_vgpu_vulkan::RtPlatform;
        use reims_vgpu_vulkan::core::platform::host_range::HostRange;
        use reims_vgpu_vulkan::core::platform::resource::{
            Buffer as _, Library as _, Texture as _,
        };
        use reims_vgpu_vulkan::core::platform::values::{
            ColorAttachmentDesc, FullRenderPipelineDesc, SamplerDesc, TextureDesc,
        };
        use reims_vgpu_vulkan::core::raise::OnFault;

        use crate::platform::ApiPlatform as _;

        const ICB_LIBRARY: &[u8] =
            include_bytes!("../../../../../tests/shaders/icb_indexed.metallib");
        const SAMPLE_LIBRARY: &[u8] =
            include_bytes!("../../../../../tests/shaders/compute_sample.metallib");
        const QUEUE: u64 = 1;
        const ICB: u64 = 2;
        const ARGUMENTS: u64 = 3;
        const POSITIONS: u64 = 4;
        const COLOURS: u64 = 5;
        const INDICES: u64 = 6;
        const RANGES: u64 = 7;
        const STAGING: u64 = 8;
        const OUT: u64 = 9;
        const ENCODE: u64 = 10;
        const SAMPLE: u64 = 11;
        const DRAW: u64 = 12;
        const TARGET: u64 = 13;
        const SAMPLED: u64 = 14;
        const SAMPLER: u64 = 15;
        const ENCODED: u64 = 20;
        const EXECUTED: u64 = 21;

        /// `values`' little-endian bytes.
        fn words(values: &[u32]) -> Vec<u8> {
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        /// `values`' little-endian bytes.
        fn floats(values: &[f32]) -> Vec<u8> {
            values.iter().flat_map(|v| v.to_le_bytes()).collect()
        }

        /// One op: its header, then its body.
        fn op(tag: u16, body: &[u8]) -> Vec<u8> {
            let header = RgfxOpHeader {
                tag,
                flags: 0,
                len: (size_of::<RgfxOpHeader>() + body.len()) as u32,
            };
            [header.as_bytes(), body].concat()
        }

        /// `setComputePipelineState:` of `pipeline_obj_id`.
        fn pipeline(pipeline_obj_id: u64) -> Vec<u8> {
            let set = RgfxComputeOpSetComputePipelineState { pipeline_obj_id };
            op(RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, set.as_bytes())
        }

        /// A buffer's binding at offset 0.
        fn binding(buffer_obj_id: u64) -> RgfxComputeBufferBinding {
            RgfxComputeBufferBinding {
                buffer_obj_id,
                offset: 0,
            }
        }

        /// `setBuffers:offsets:withRange:` from index 0: its head, then `bindings`.
        fn buffers(bindings: &[RgfxComputeBufferBinding]) -> Vec<u8> {
            let head = RgfxComputeOpSetBuffers {
                location: 0,
                length: bindings.len() as u64,
            };
            let body = [head.as_bytes(), bindings.as_bytes()].concat();
            op(RGFX_OP_COMPUTE_SET_BUFFERS, &body)
        }

        /// `dispatchThreads:threadsPerThreadgroup:` of `count` threads in one threadgroup.
        fn threads(count: u64) -> Vec<u8> {
            let dispatch = RgfxComputeOpDispatchThreads {
                grid_width: count,
                grid_height: 1,
                grid_depth: 1,
                threads_width: count,
                threads_height: 1,
                threads_depth: 1,
            };
            op(RGFX_OP_COMPUTE_DISPATCH_THREADS, dispatch.as_bytes())
        }

        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let on = OnFault::Return;
        // A platform of the test's own, whose device goes before the process exits.
        let platform = crate::rt_test_end::platform();
        let device = platform.metal_device(None).unwrap();
        // Indexed draws that inherit the pipeline state, of one vertex and one fragment buffer.
        let indexed = IcbDesc {
            command_types: 2, // MTLIndirectCommandTypeDrawIndexed
            inherit_pipeline_state: true,
            max_vertex_buffer_bind_count: 1,
            max_fragment_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        let made = device.new_indirect_command_buffer(&indexed, 2, 0);
        let Some(icb) = made.expect("never a raise") else {
            assert_eq!(
                device.facts().indirect_tokens,
                None,
                "a device of the extension made no buffer"
            );
            println!("no VK_EXT_device_generated_commands: the make answers nil, rt's gap");
            return;
        };

        // A Shared buffer that holds `bytes`.
        let shared = |bytes: &[u8]| {
            let made = device.new_buffer_of_length(on, bytes.len() as u64, 0);
            let made = made.expect("never a raise").expect("a buffer");
            // SAFETY: the Shared buffer's contents are mapped, `bytes` long, and no command uses
            // the buffer yet.
            unsafe {
                let at = made.contents().as_ptr::<u8>();
                at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
            }
            made
        };
        // Command 0 a green triangle over the target, command 1 a blue one: each three `float4`
        // positions, (-1, -1), (3, -1) and (-1, 3), its three 16-bit indices and its colour.
        let corners = [
            [-1.0f32, -1.0, 0.0, 1.0],
            [3.0, -1.0, 0.0, 1.0],
            [-1.0, 3.0, 0.0, 1.0],
        ];
        let positions = floats([corners; 2].as_flattened().as_flattened());
        let indices: Vec<u8> = (0u16..6).flat_map(u16::to_le_bytes).collect();
        let colours = floats(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);

        let texture = |desc: TextureDesc| {
            let made = device.new_texture(on, &platform.texture_descriptor(&desc));
            made.expect("never a raise").expect("a texture")
        };
        let target = texture(TextureDesc {
            width: 4,
            height: 4,
            usage: 5, // shader read, render target
            ..TextureDesc::default()
        });
        // A row of four texels, of which the kernel's thread `i` reads the `i`th.
        let texels: [[u8; 4]; 4] = [
            [0x10, 0x20, 0x30, 0x40],
            [0x50, 0x60, 0x70, 0x80],
            [0x90, 0xa0, 0xb0, 0xc0],
            [0xd0, 0xe0, 0xf0, 0xff],
        ];
        let sampled = texture(TextureDesc {
            width: 4,
            ..TextureDesc::default()
        });
        let row = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(4, 1, 1));
        let bytes = HostRange::of_slice(texels.as_flattened());
        let replaced = sampled.replace_region(row, 0, 0, bytes, 16, 0);
        replaced.expect("never a raise");
        // `sample_k` reads left of the texture, at `(i + 0.5) / 4 - 1`: repeated, texel `i`.
        let sampler = device.new_sampler_state(&SamplerDesc {
            s_address_mode: Some(2), // MTLSamplerAddressModeRepeat
            ..SamplerDesc::default()
        });
        let sampler = sampler.expect("never a raise").expect("a sampler state");

        let library = |bytes: &[u8]| {
            let made = device.new_library(bytes).expect("never a raise");
            made.expect("a library").expect("a library")
        };
        let icbs = library(ICB_LIBRARY);
        let samples = library(SAMPLE_LIBRARY);
        let function = |library: &<RtPlatform as Platform>::Library, name: &str| {
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function")
        };
        let kernel = |function: &<RtPlatform as Platform>::Function| {
            let made = device.new_compute_pipeline_state_for_function(Some(function));
            made.expect("a compute state").expect("a compute state")
        };
        let encode = kernel(&function(&icbs, "icb_encode"));
        let sample = kernel(&function(&samples, "sample_k"));
        let vertex = function(&icbs, "icb_vertex");
        let fragment = function(&icbs, "icb_fragment");
        let mut desc = FullRenderPipelineDesc::<RtPlatform>::new(&vertex, Some(&fragment));
        desc.colors = vec![(
            0,
            ColorAttachmentDesc {
                pixel_format: 70, // RGBA8Unorm
                ..ColorAttachmentDesc::default()
            },
        )];
        desc.support_indirect_command_buffers = true;
        let draw = device.new_render_pipeline_state_full(&desc);
        let draw = draw.expect("a render state").expect("a render state");

        let mut dec = GraphicsDecoder::<RtPlatform>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
        );
        // Each command buffer one of rt's: a part committed at the pass's end would put the
        // dispatch in a command buffer of its own.
        dec.set_split_after(Some(std::time::Duration::from_secs(3600)));
        let completed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&completed);
        let record = move |m: RgfxCbComplete| sink.lock().push((m.cb_obj_id, m.status));
        dec.set_completion_sink(Arc::new(record));
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: QUEUE,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        let arguments = shared(&icb.gpu_resource_id().to_le_bytes());
        let handles = [
            (ICB, ObjHandle::Icb(icb)),
            (ARGUMENTS, ObjHandle::Buffer(arguments)),
            (POSITIONS, ObjHandle::Buffer(shared(&positions))),
            (COLOURS, ObjHandle::Buffer(shared(&colours))),
            (INDICES, ObjHandle::Buffer(shared(&indices))),
            (RANGES, ObjHandle::Buffer(shared(&words(&[0, 0])))),
            // The range the blit writes: commands 0 and 1.
            (STAGING, ObjHandle::Buffer(shared(&words(&[0, 2])))),
            (OUT, ObjHandle::Buffer(shared(&words(&[0xdead_beef; 4])))),
            (ENCODE, ObjHandle::ComputePipeline(encode)),
            (SAMPLE, ObjHandle::ComputePipeline(sample)),
            (DRAW, ObjHandle::RenderPipeline(draw)),
            (TARGET, ObjHandle::Texture(target)),
            (SAMPLED, ObjHandle::Texture(sampled)),
            (SAMPLER, ObjHandle::Sampler(sampler)),
        ];
        for (obj_id, handle) in handles {
            assert_eq!(dec.insert(obj_id, handle).unwrap(), Ok(()));
        }

        // Decodes the `COMMAND_BUFFER` message of the queue's command buffer `cb_obj_id`, which
        // carries `ops`.
        let commit = |dec: &mut GraphicsDecoder<RtPlatform>, cb_obj_id: u64, ops: &[Vec<u8>]| {
            let ops = ops.concat();
            let m = RgfxCommandBuffer {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_COMMAND_BUFFER,
                    size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                    seqno: cb_obj_id,
                },
                queue_obj_id: QUEUE,
                cb_obj_id,
                op_bytes: ops.len() as u64,
            };
            let message = [m.as_bytes(), ops.as_slice()].concat();
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        };
        // Whether command buffer `cb_obj_id` completes, waiting for it.
        let completes = |cb_obj_id: u64| {
            let done = || completed.lock().iter().any(|&(id, _)| id == cb_obj_id);
            crate::graphics::tests::eventually(done)
        };
        let compute = RgfxComputePassDesc::default();

        // The earlier command buffer: a blit writes the range, then `icb_encode` encodes
        // commands 0 and 1 through the indirect command buffer's `gpuResourceID`.
        let copy = RgfxBlitOpCopyBufferToBuffer {
            source_buffer_obj_id: STAGING,
            destination_buffer_obj_id: RANGES,
            source_offset: 0,
            destination_offset: 0,
            size: 8,
        };
        let scene = [ARGUMENTS, POSITIONS, COLOURS, INDICES].map(binding);
        let used = {
            let head = RgfxComputeOpUseResources { count: 1, usage: 2 };
            let record = RgfxComputeResourceRef {
                kind: RGFX_RESOURCE_KIND_ICB,
                reserved_04: 0,
                obj_id: ICB,
            };
            [head.as_bytes(), record.as_bytes()].concat()
        };
        let encoding = [
            op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            op(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, copy.as_bytes()),
            op(RGFX_OP_ENCODER_END, &[]),
            op(RGFX_OP_ENCODER_BEGIN_COMPUTE, compute.as_bytes()),
            pipeline(ENCODE),
            buffers(&scene),
            op(RGFX_OP_COMPUTE_USE_RESOURCES, &used),
            threads(2),
            op(RGFX_OP_ENCODER_END, &[]),
        ];
        commit(&mut dec, ENCODED, &encoding);
        assert!(
            completes(ENCODED),
            "the encoding command buffer never completed"
        );

        // The command buffer under test: the pass, its first work, clears the target to red and
        // executes the range the blit wrote; then `sample_k`, its first dispatch, samples.
        let pass = RgfxRenderPassDesc {
            color_count: 1,
            ..RgfxRenderPassDesc::default()
        };
        let colour = RgfxRenderPassColorAttachment {
            texture_obj_id: TARGET,
            load_action: 2,  // Clear
            store_action: 1, // Store
            clear_color_bits: [1.0f64.to_bits(), 0, 0, 1.0f64.to_bits()],
            ..RgfxRenderPassColorAttachment::default()
        };
        let begin = [pass.as_bytes(), colour.as_bytes()].concat();
        let state = RgfxOpSetRenderPipelineState {
            pipeline_obj_id: DRAW,
        };
        let held = RgfxOpExecuteCommandsIndirect {
            indirect_command_buffer_obj_id: ICB,
            indirect_range_buffer_obj_id: RANGES,
            indirect_buffer_offset: 0,
        };
        let textures = {
            let head = RgfxComputeOpSetTextures {
                location: 0,
                length: 1,
            };
            let record = RgfxComputeTextureBinding {
                texture_obj_id: SAMPLED,
            };
            [head.as_bytes(), record.as_bytes()].concat()
        };
        let samplers = {
            let head = RgfxComputeOpSetSamplerStates {
                location: 0,
                length: 1,
            };
            let record = RgfxComputeSamplerBinding {
                sampler_obj_id: SAMPLER,
            };
            [head.as_bytes(), record.as_bytes()].concat()
        };
        let executing = [
            op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin),
            op(RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, state.as_bytes()),
            op(RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT, held.as_bytes()),
            op(RGFX_OP_ENCODER_END, &[]),
            op(RGFX_OP_ENCODER_BEGIN_COMPUTE, compute.as_bytes()),
            pipeline(SAMPLE),
            op(RGFX_OP_COMPUTE_SET_TEXTURES, &textures),
            op(RGFX_OP_COMPUTE_SET_SAMPLER_STATES, &samplers),
            buffers(&[binding(OUT)]),
            threads(4),
            op(RGFX_OP_ENCODER_END, &[]),
        ];
        commit(&mut dec, EXECUTED, &executing);
        assert!(
            completes(EXECUTED),
            "the executing command buffer never completed"
        );
        // Neither lost the device.
        assert_eq!(
            *completed.lock(),
            [(ENCODED, RGFX_CB_OK), (EXECUTED, RGFX_CB_OK)]
        );

        // Commands 0 and 1 drawn, the last, blue, on top.
        let Some(ObjHandle::Texture(target)) = dec.object(TARGET) else {
            panic!("no target");
        };
        let mut drawn = [0u8; 64];
        target.read_pixels(&mut drawn, 16);
        let blue: [u8; 64] = std::array::from_fn(|i| [0, 0, 255, 255][i % 4]);
        assert_eq!(drawn, blue);
        let Some(ObjHandle::Buffer(out)) = dec.object(OUT) else {
            panic!("no output buffer");
        };
        // SAFETY: a Shared buffer of 16 bytes, read after the command buffer that wrote it
        // completed.
        let written = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 16) };
        // Thread `i`'s texel `i`, as `pack_float_to_unorm4x8` packs it, little-endian.
        assert_eq!(written, texels.as_flattened());
    }
}
