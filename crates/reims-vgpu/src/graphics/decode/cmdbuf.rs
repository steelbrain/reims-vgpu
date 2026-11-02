//! The command-buffer decoder: it carries out one
//! [`RGFX_CMD_COMMAND_BUFFER`](reims_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
//! message (docs/graphics-kext/design.md, "Stage 3 wire protocol"). It resolves
//! the queue the buffer runs on, opens a `P::CommandBuffer`, walks the op stream
//! that follows the message, and at the end registers the completion path and
//! commits.
//!
//! **The op walk.** Each op is an [`RgfxOpHeader`] then its body. The walk reads
//! the header, checks its `len` (at least a header, a multiple of 8, and within
//! the remaining bytes), and dispatches by the op's group ([`op_group`]): the
//! framing group (encoder begin/end, and the buffer's event signals and waits) it
//! handles here; the render, compute and blit groups it hands to [`super::render`],
//! [`super::compute`] and [`super::blit`], one encoder held open at a time.
//!
//! **Events.** `SIGNAL_EVENT` and `WAIT_EVENT` come between encoders, as Metal's
//! `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` do, and name an event
//! or a shared event. The value a buffer signals a shared event to goes to the guest
//! once the buffer finishes, ahead of its completion record
//! ([`RgfxEventSignalled`](reims_graphics::wire::RgfxEventSignalled)); a buffer that
//! waits is watched, and named in the log where it waits long ([`super::waits`]).
//!
//! **Two kinds of failure.** A message or op that will not parse structurally — a
//! short message, a bad `len`, an op body too short for a header — is a protocol
//! error, reported as a [`DecodeError`] and the work abandoned. A structurally
//! valid stream the host cannot honour — an unknown op, an op with no matching
//! open encoder, a reference to an object the host does not hold — faults the exec
//! ([`op_fault`], [`bad_reference`](super)), since the stream is already running on
//! the GPU's queue; the ring loop's handler turns the fault into a command-buffer
//! error. Either way the buffer is stood in for on its queue by its event ops alone
//! ([`GraphicsDecoder::stand_in_for_failed`]), whose completion posts the buffer's
//! failure in the queue's order, with the shared event values it signals ahead of it.

use std::sync::Arc;

use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE, RGFX_OP_BODY_OFFSET,
    RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_BEGIN_RENDER,
    RGFX_OP_ENCODER_END, RGFX_OP_GROUP_BLIT, RGFX_OP_GROUP_COMPUTE, RGFX_OP_GROUP_FRAMING,
    RGFX_OP_GROUP_RENDER, RGFX_OP_SIGNAL_EVENT, RGFX_OP_WAIT_EVENT,
    RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT, RGFX_PASS_FLAG_RT_ARRAY_LENGTH,
    RGFX_PASS_FLAG_RT_SIZE, RGFX_PASS_FLAG_SAMPLE_POSITIONS, RGFX_PASS_MAX_SAMPLE_POSITIONS,
    RGFX_RENDER_PASS_COLOR_OFFSET, RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpEvent,
    RgfxOpHeader, RgfxRenderPassColorAttachment, RgfxRenderPassDesc, RgfxRenderPassSamplePositions,
    op_group,
};
use reims_graphics::wire::{
    RGFX_EVT_CB_SCHEDULED, RGFX_EVT_EVENT_SIGNALLED, RgfxCbScheduled, RgfxEventSignalled,
    RgfxMsgHeader,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::caps::{DeviceCaps, SAMPLE_COUNTS};
use reims_vgpu_core::platform::command::{
    CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo, ScheduledHandler,
};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder};
use reims_vgpu_core::platform::types::{ClearColor, MtlSamplePosition};
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, ComputePassDesc, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use zerocopy::FromBytes;

use super::super::completion::{cb_complete, cb_failed};
use super::{
    DecodeError, Drain, GraphicsDecoder, ObjHandle, blit, compute, op_fault, read, refusing, render,
};

/// The op stream runs under a live raise handler (the ring loop's), so an encode's
/// raise returns for the handler to catch.
const ON: OnFault = OnFault::Return;

/// What a command buffer's op stream does with events: the shared events it signals, whose values
/// go to the guest once it finishes, and the events it waits for.
#[derive(Default)]
struct EventOps {
    /// Each (shared event, value) the buffer signals, in stream order.
    signals: Vec<(u64, u64)>,
    /// Each (event, value) the buffer waits for, in stream order.
    waits: Vec<(u64, u64)>,
}

/// The event ops of the op stream `ops`, in stream order: whether each signals (else it waits),
/// and its body. The scan stops at an op it cannot step past, as the walk does.
fn event_ops(mut ops: &[u8]) -> Vec<(bool, RgfxOpEvent)> {
    let mut out = Vec::new();
    while let Some(h) = read::<RgfxOpHeader>(ops) {
        let len = h.len as usize;
        if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
            break;
        }
        if matches!(h.tag, RGFX_OP_SIGNAL_EVENT | RGFX_OP_WAIT_EVENT)
            && let Ok((op, _)) =
                RgfxOpEvent::read_from_prefix(&ops[RGFX_OP_BODY_OFFSET as usize..len])
        {
            out.push((h.tag == RGFX_OP_SIGNAL_EVENT, op));
        }
        ops = &ops[len..];
    }
    out
}

/// The encoder the walk currently holds open, one at a time, between an
/// `ENCODER_BEGIN_*` and the matching `ENCODER_END`.
enum OpenEncoder<P: Platform> {
    Render(P::RenderEncoder),
    Compute(P::ComputeEncoder),
    Blit(P::BlitEncoder),
}

/// Whether Metal takes `positions` as a pass's sample positions, where `makes` answers whether
/// the device makes textures of a sample count. Metal's `setSamplePositions:count:` fails an
/// assertion at a count other than 2, 4 or 8 and at a coordinate outside [0, 1), and its render
/// pass validation at a count the device makes no texture of (measured on the M3 Max, macOS 27:
/// 8 there, whatever the attachments' sample count). Either ends the host's process, on the
/// Vulkan platform at the encoder's creation; the bundle sends none of them (its device answers
/// the host's sample counts, and the guest's own descriptor asserts at the rest), so a pass that
/// holds one faults.
fn sample_positions_taken(positions: &[MtlSamplePosition], makes: impl Fn(u32) -> bool) -> bool {
    let count = positions.len();
    matches!(count, 2 | 4 | 8)
        && makes(count as u32)
        && positions
            .iter()
            .all(|p| (0.0..1.0).contains(&p.x) && (0.0..1.0).contains(&p.y))
}

/// Whether a device of `caps` makes textures of `count` samples (`supportsTextureSampleCount:`).
fn makes_sample_count(caps: &DeviceCaps, count: u32) -> bool {
    SAMPLE_COUNTS
        .iter()
        .zip(&caps.sample_counts)
        .any(|(&n, makes)| n == u64::from(count) && matches!(makes, Ok(true)))
}

impl<P: Platform> GraphicsDecoder<P> {
    /// Carries out one `COMMAND_BUFFER` message (`decode`'s
    /// [`RGFX_CMD_COMMAND_BUFFER`](reims_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
    /// arm): opens a command buffer on the named queue, walks the op stream, and
    /// commits with a completion handler, which holds `drain` (its context's) until it has run.
    pub(super) fn decode_command_buffer(
        &self,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The op stream follows the message; `op_bytes` must lie within it.
        let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
        let Some(ops) = (m.op_bytes as usize)
            .checked_add(start)
            .and_then(|end| message.get(start..end))
        else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The queue the buffer runs on; a missing or wrong-kind queue is a
        // message-level bad reference (the op stream has not started).
        let cb = {
            let Some(ObjHandle::CommandQueue(queue)) = self.named(m.queue_obj_id) else {
                return Ok(Err(DecodeError::BadReference));
            };
            match queue.command_buffer(ON)? {
                Some(cb) => cb,
                None => return Ok(Err(DecodeError::Refused)),
            }
        };
        // A structural malformation in the stream stops the walk and is reported;
        // everything else faults the exec and propagates.
        let mut events = EventOps::default();
        if let Err(e) = self.walk_ops(ops, &cb, &mut events)? {
            return Ok(Err(e));
        }
        // Register the scheduled and completion paths and commit. The GPU start and end times
        // are 0 here; the ring loop fills them when it posts the record on the queue's
        // completion ring in a later increment.
        if let Some(handler) = self.scheduled_handler(m.cb_obj_id, m.header.seqno) {
            cb.add_scheduled_handler(ON, handler)?;
        }
        let handler = self.completed_handler(m.cb_obj_id, m.header.seqno, events, drain, false);
        cb.add_completed_handler(ON, handler)?;
        cb.commit(ON)?;
        Ok(Ok(()))
    }

    /// Stands in for the `COMMAND_BUFFER` `message` the decoder could not carry out (refused, or
    /// faulted at an op: a draw with a pipeline the host refused among them), as a buffer Metal
    /// could not run still finishes: a command buffer on its queue holding the buffer's event ops
    /// alone, each wait and signal of an event its process holds, in stream order. Its completion
    /// posts the shared event values signalled and then the buffer's failure ([`cb_failed`]), so
    /// the failure comes in its queue's order, once the buffers committed before it finish, and
    /// nothing waiting for one of its values, on the GPU or in the guest, waits for good.
    /// `Ok(false)` where there is none, and the caller posts the failure at once: a message too
    /// short to name its queue, a queue the process does not hold or that makes no command
    /// buffer, and a context whose work is refused ([`Self::context_failed`]).
    pub(crate) fn stand_in_for_failed(&mut self, message: &[u8]) -> RaiseResult<bool> {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return Ok(false);
        };
        if self.context_failed() {
            return Ok(false);
        }
        // The buffer's context's drain, held until the stand-in's completion handler has run.
        let drain =
            (self.context).map(|context| Arc::clone(self.drains.entry(context).or_default()));
        let Some(ObjHandle::CommandQueue(queue)) = self.named(m.queue_obj_id) else {
            return Ok(false);
        };
        let Some(cb) = queue.command_buffer(ON)? else {
            return Ok(false);
        };
        let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
        let ops = message.get(start..).unwrap_or_default();
        let ops = &ops[..usize::try_from(m.op_bytes).map_or(ops.len(), |n| n.min(ops.len()))];
        let mut events = EventOps::default();
        for (signal, op) in event_ops(ops) {
            let (event, shared) = match self.named(op.event_obj_id) {
                Some(ObjHandle::Event(e)) => (EventRef::Event(e), false),
                Some(ObjHandle::SharedEvent(e)) => (EventRef::Shared(e), true),
                _ => continue,
            };
            if signal {
                cb.encode_signal(Some(event), op.value)?;
                if shared {
                    events.signals.push((op.event_obj_id, op.value));
                }
            } else {
                cb.encode_wait(Some(event), op.value)?;
                events.waits.push((op.event_obj_id, op.value));
            }
        }
        let handler = self.completed_handler(m.cb_obj_id, m.header.seqno, events, drain, true);
        cb.add_completed_handler(ON, handler)?;
        cb.commit(ON)?;
        Ok(true)
    }

    /// The scheduled handler of command buffer `cb_obj_id`, at `seqno`: it posts the buffer's
    /// `CB_SCHEDULED` once the platform has scheduled it, so the guest's command buffer becomes
    /// `Scheduled` and its scheduled handlers and `waitUntilScheduled` go on then, as Metal's do.
    /// `None` without a sink.
    fn scheduled_handler(&self, cb_obj_id: u64, seqno: u64) -> Option<ScheduledHandler> {
        let sink = self.scheduled_sink.clone()?;
        Some(Box::new(move || {
            sink(RgfxCbScheduled {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_CB_SCHEDULED,
                    size: size_of::<RgfxCbScheduled>() as u32,
                    seqno,
                },
                cb_obj_id,
            });
        }))
    }

    /// The completion handler of command buffer `cb_obj_id`, at `seqno`, whose op stream did
    /// `events`: it posts the values the buffer signalled its shared events to, then its
    /// completion, which for a `stand_in` ([`Self::stand_in_for_failed`]) is its failure whatever
    /// the stand-in's outcome, and lets `drain` (its context's) go.
    fn completed_handler(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        events: EventOps,
        drain: Option<Arc<Drain>>,
        stand_in: bool,
    ) -> CompletedHandler {
        let sink = self.completion_sink.clone();
        let signal_sink = self.signal_sink.clone();
        let (failed, context) = (Arc::clone(&self.failed), self.context);
        let signals = events.signals;
        let watched = (!events.waits.is_empty())
            .then(|| self.waits.watch(cb_obj_id, seqno, context, events.waits));
        Box::new(move |info: CompletedInfo| {
            // The platform refused the buffer's completion on its own thread (a death in its
            // read, in the scope this decode carried there): the buffer fails, and so does its
            // context, whose later work is refused until its close.
            if info.is_refused()
                && let Some(context) = context
                && failed.fail(context)
            {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "context {context}'s command buffer {cb_obj_id:#x} (seqno {seqno}) failed \
                     off its op's thread: the context's later work is refused until its close"
                );
            }
            if let Some(waiting) = &watched {
                waiting.finished();
            }
            // The shared events the buffer signalled, ahead of its completion, so the guest
            // has each value before the buffer's handlers run. A buffer that failed may have
            // signalled them before its fault, and a guest waiting for one would wait for
            // good: each is posted whatever the outcome.
            if let Some(signal_sink) = &signal_sink {
                for &(obj_id, value) in &signals {
                    signal_sink(event_signalled(obj_id, value));
                }
            }
            let record = if stand_in {
                cb_failed(cb_obj_id, seqno)
            } else {
                cb_complete(cb_obj_id, seqno, 0, 0, info.error.as_ref())
            };
            if let Some(sink) = &sink {
                sink(record);
            }
            // The buffer's work and its completion are done: a drain of its context may end.
            drop(drain);
        })
    }

    /// Walks the op stream `ops` on the command buffer `cb`. `Ok(Err(..))` for a
    /// structural malformation, `Ok(Ok(()))` when the whole stream was carried out;
    /// a fault propagates as `Err`, and so does an op the platform refuses, which
    /// the lane's refusal scope takes ([`refusing`]). A walk that stops with an
    /// encoder open ends it before the command buffer goes, uncommitted, as Metal
    /// asserts on an encoder released without its end.
    fn walk_ops(
        &self,
        ops: &[u8],
        cb: &P::CommandBuffer,
        events: &mut EventOps,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut encoder: Option<OpenEncoder<P>> = None;
        let walked = refusing(|| self.walk(ops, cb, &mut encoder, events));
        if let Some(open) = encoder.take() {
            end_abandoned(open);
        }
        walked
    }

    /// [`Self::walk_ops`]'s walk, `encoder` the one it holds open.
    fn walk(
        &self,
        mut ops: &[u8],
        cb: &P::CommandBuffer,
        encoder: &mut Option<OpenEncoder<P>>,
        events: &mut EventOps,
    ) -> RaiseResult<Result<(), DecodeError>> {
        while !ops.is_empty() {
            let Some(h) = read::<RgfxOpHeader>(ops) else {
                return Ok(Err(DecodeError::Malformed));
            };
            let len = h.len as usize;
            if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
                return Ok(Err(DecodeError::Malformed));
            }
            let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
            match op_group(h.tag) {
                RGFX_OP_GROUP_FRAMING => self.framing_op(h.tag, body, cb, encoder, events)?,
                RGFX_OP_GROUP_RENDER => {
                    let Some(OpenEncoder::Render(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "render"));
                    };
                    if !render::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                RGFX_OP_GROUP_COMPUTE => {
                    let Some(OpenEncoder::Compute(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "compute"));
                    };
                    if !compute::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                RGFX_OP_GROUP_BLIT => {
                    let Some(OpenEncoder::Blit(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(h.tag, "blit"));
                    };
                    if !blit::decode_op(h.tag, body, self, enc, ON)? {
                        return Err(unknown_op(h.tag));
                    }
                }
                _ => return Err(unknown_op(h.tag)),
            }
            ops = &ops[len..];
        }
        if encoder.is_some() {
            return Err(op_fault("the op stream ended with an encoder still open"));
        }
        Ok(Ok(()))
    }

    /// A framing op: open the encoder its `ENCODER_BEGIN_*` names, end the open
    /// one, or signal or wait for an event between encoders. An already-open
    /// encoder at a begin or at an event op, or no open encoder at an end, faults.
    fn framing_op(
        &self,
        tag: u16,
        body: &[u8],
        cb: &P::CommandBuffer,
        encoder: &mut Option<OpenEncoder<P>>,
        events: &mut EventOps,
    ) -> RaiseResult<()> {
        match tag {
            RGFX_OP_ENCODER_BEGIN_RENDER => {
                already_open(encoder, "ENCODER_BEGIN_RENDER")?;
                let pass = self.build_render_pass(body, |count| {
                    cb.device()
                        .is_some_and(|d| makes_sample_count(&d.caps(), count))
                })?;
                let enc = cb
                    .render_encoder(ON, &pass)?
                    .ok_or_else(|| op_fault("the render encoder was nil"))?;
                *encoder = Some(OpenEncoder::Render(enc));
            }
            RGFX_OP_ENCODER_BEGIN_COMPUTE => {
                already_open(encoder, "ENCODER_BEGIN_COMPUTE")?;
                let desc = compute_pass(body)?;
                let enc = cb
                    .compute_encoder_for_pass(&desc)?
                    .ok_or_else(|| op_fault("the compute encoder was nil"))?;
                *encoder = Some(OpenEncoder::Compute(enc));
            }
            RGFX_OP_ENCODER_BEGIN_BLIT => {
                already_open(encoder, "ENCODER_BEGIN_BLIT")?;
                let enc = cb
                    .blit_encoder(ON)?
                    .ok_or_else(|| op_fault("the blit encoder was nil"))?;
                *encoder = Some(OpenEncoder::Blit(enc));
            }
            RGFX_OP_ENCODER_END => match encoder.take() {
                Some(OpenEncoder::Render(enc)) => enc.encode(ON, RenderCmd::EndEncoding)?,
                Some(OpenEncoder::Compute(enc)) => enc.encode(ON, ComputeCmd::EndEncoding)?,
                Some(OpenEncoder::Blit(enc)) => enc.encode(ON, BlitCmd::EndEncoding)?,
                None => return Err(op_fault("ENCODER_END with no open encoder")),
            },
            RGFX_OP_SIGNAL_EVENT | RGFX_OP_WAIT_EVENT => {
                let signal = tag == RGFX_OP_SIGNAL_EVENT;
                let name = if signal { "SIGNAL_EVENT" } else { "WAIT_EVENT" };
                // Metal's command buffer takes them between encoders alone.
                already_open(encoder, name)?;
                let op = RgfxOpEvent::read_from_prefix(body)
                    .map(|(op, _)| op)
                    .map_err(|_| op_fault(format!("{name}: body too short")))?;
                let (event, shared) = match self.named(op.event_obj_id) {
                    Some(ObjHandle::Event(e)) => (EventRef::Event(e), false),
                    Some(ObjHandle::SharedEvent(e)) => (EventRef::Shared(e), true),
                    _ => return Err(super::bad_reference()),
                };
                if signal {
                    cb.encode_signal(Some(event), op.value)?;
                    if shared {
                        events.signals.push((op.event_obj_id, op.value));
                    }
                } else {
                    cb.encode_wait(Some(event), op.value)?;
                    events.waits.push((op.event_obj_id, op.value));
                }
            }
            _ => return Err(unknown_op(tag)),
        }
        Ok(())
    }

    /// Builds a `P::RenderPassDescriptor` from an `ENCODER_BEGIN_RENDER` op's body
    /// (an [`RgfxRenderPassDesc`] then its `color_count`
    /// [`RgfxRenderPassColorAttachment`] records, and its sample positions where its flag says
    /// so). A body too short faults; a reference to a texture the host does not hold faults
    /// through the resolution API; and so do sample positions Metal does not take
    /// ([`sample_positions_taken`]), of which `makes` answers whether the device makes textures
    /// of their count.
    fn build_render_pass(
        &self,
        body: &[u8],
        makes: impl Fn(u32) -> bool,
    ) -> RaiseResult<P::RenderPassDescriptor> {
        let (desc, _) = RgfxRenderPassDesc::read_from_prefix(body).map_err(|_| {
            op_fault("ENCODER_BEGIN_RENDER: body too short for the pass descriptor")
        })?;
        let pass = self.platform.new_render_pass_descriptor();

        let mut set = PassSet::default();
        if desc.flags & RGFX_PASS_FLAG_RT_ARRAY_LENGTH != 0 {
            set.render_target_array_length = Some(desc.render_target_array_length as u64);
        }
        if desc.flags & RGFX_PASS_FLAG_RT_SIZE != 0 {
            set.render_target_width = Some(desc.render_target_width as u64);
            set.render_target_height = Some(desc.render_target_height as u64);
        }
        if desc.flags & RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT != 0 {
            set.default_raster_sample_count = Some(desc.default_raster_sample_count as u64);
        }
        pass.set_pass(&set);

        let stride = size_of::<RgfxRenderPassColorAttachment>();
        let color_start = RGFX_RENDER_PASS_COLOR_OFFSET as usize;
        for i in 0..desc.color_count as usize {
            let off = color_start + i * stride;
            let a = body
                .get(off..off + stride)
                .and_then(|s| RgfxRenderPassColorAttachment::read_from_bytes(s).ok())
                .ok_or_else(|| {
                    op_fault("ENCODER_BEGIN_RENDER: body too short for a colour attachment")
                })?;
            if a.texture_obj_id == 0 {
                continue; // an absent (nil) attachment sets nothing
            }
            let tex = self.require_texture(a.texture_obj_id)?;
            let resolve = self.get_texture(a.resolve_texture_obj_id)?;
            let aset = ColorAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(a.level as u64),
                slice: Some(a.slice as u64),
                depth_plane: Some(a.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| a.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| a.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| a.resolve_depth_plane as u64),
                load_action: Some(a.load_action as u64),
                store_action: Some(a.store_action as u64),
                store_action_options: Some(a.store_action_options as u64),
                clear_color: Some(clear_color(a.clear_color_bits)),
            };
            pass.set_color_attachment(ON, i as u64, &aset)?;
        }

        // The pass's sample positions, after the colour records.
        if desc.flags & RGFX_PASS_FLAG_SAMPLE_POSITIONS != 0 {
            let off = color_start + desc.color_count as usize * stride;
            let p = body
                .get(off..off + size_of::<RgfxRenderPassSamplePositions>())
                .and_then(|s| RgfxRenderPassSamplePositions::read_from_bytes(s).ok())
                .filter(|p| p.count <= RGFX_PASS_MAX_SAMPLE_POSITIONS)
                .ok_or_else(|| {
                    op_fault("ENCODER_BEGIN_RENDER: no whole sample positions after the colours")
                })?;
            let positions: Vec<MtlSamplePosition> = p.position_bits[..p.count as usize]
                .iter()
                .map(|&[x, y]| MtlSamplePosition {
                    x: f32::from_bits(x),
                    y: f32::from_bits(y),
                })
                .collect();
            if !sample_positions_taken(&positions, &makes) {
                return Err(op_fault(
                    "ENCODER_BEGIN_RENDER: sample positions Metal does not take",
                ));
            }
            pass.set_sample_positions(&positions)?;
        }

        if desc.depth.texture_obj_id != 0 {
            let d = &desc.depth;
            let tex = self.require_texture(d.texture_obj_id)?;
            let resolve = self.get_texture(d.resolve_texture_obj_id)?;
            let dset = DepthAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(d.level as u64),
                slice: Some(d.slice as u64),
                depth_plane: Some(d.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| d.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| d.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| d.resolve_depth_plane as u64),
                load_action: Some(d.load_action as u64),
                store_action: Some(d.store_action as u64),
                store_action_options: Some(d.store_action_options as u64),
                clear_depth: Some(f64::from_bits(d.clear_depth_bits)),
                depth_resolve_filter: Some(d.depth_resolve_filter as u64),
            };
            pass.set_depth_attachment(&dset);
        }

        if desc.stencil.texture_obj_id != 0 {
            let s = &desc.stencil;
            let tex = self.require_texture(s.texture_obj_id)?;
            let resolve = self.get_texture(s.resolve_texture_obj_id)?;
            let sset = StencilAttachmentSet {
                texture: Some(Some(&tex)),
                resolve_texture: resolve.as_ref().map(Some),
                level: Some(s.level as u64),
                slice: Some(s.slice as u64),
                depth_plane: Some(s.depth_plane as u64),
                resolve_level: resolve.as_ref().map(|_| s.resolve_level as u64),
                resolve_slice: resolve.as_ref().map(|_| s.resolve_slice as u64),
                resolve_depth_plane: resolve.as_ref().map(|_| s.resolve_depth_plane as u64),
                load_action: Some(s.load_action as u64),
                store_action: Some(s.store_action as u64),
                store_action_options: Some(s.store_action_options as u64),
                clear_stencil: Some(s.clear_stencil),
                stencil_resolve_filter: Some(s.stencil_resolve_filter as u64),
            };
            pass.set_stencil_attachment(&sset);
        }

        Ok(pass)
    }
}

/// Ends `open`, an encoder a walk left open as it stopped, in a refusal scope of its own: its
/// command buffer is never committed, and what the end answers is not the guest's.
/// The `EVENT_SIGNALLED` for the shared event `obj_id`, which a finished command buffer signalled
/// to `value`.
fn event_signalled(obj_id: u64, value: u64) -> RgfxEventSignalled {
    RgfxEventSignalled {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_EVENT_SIGNALLED,
            size: size_of::<RgfxEventSignalled>() as u32,
            seqno: 0,
        },
        obj_id,
        value,
    }
}

fn end_abandoned<P: Platform>(open: OpenEncoder<P>) {
    let _ = refusing(|| match open {
        OpenEncoder::Render(enc) => enc.encode(ON, RenderCmd::EndEncoding),
        OpenEncoder::Compute(enc) => enc.encode(ON, ComputeCmd::EndEncoding),
        OpenEncoder::Blit(enc) => enc.encode(ON, BlitCmd::EndEncoding),
    });
}

/// A fault: an encoder op whose group has no open encoder.
fn no_encoder(tag: u16, group: &str) -> Raise {
    op_fault(format_args!(
        "a {group} op {tag:#06x} with no open {group} encoder"
    ))
}

/// A fault: an op the host does not decode.
fn unknown_op(tag: u16) -> Raise {
    op_fault(format_args!("unknown op {tag:#06x}"))
}

/// A fault when an encoder is already open at an `ENCODER_BEGIN_*`.
fn already_open<P: Platform>(encoder: &Option<OpenEncoder<P>>, op: &str) -> RaiseResult<()> {
    if encoder.is_some() {
        return Err(op_fault(format_args!("{op} with an encoder already open")));
    }
    Ok(())
}

/// The compute pass descriptor from an `ENCODER_BEGIN_COMPUTE` op's body (an empty
/// body is Metal's default pass).
fn compute_pass(body: &[u8]) -> RaiseResult<ComputePassDesc> {
    if body.is_empty() {
        return Ok(ComputePassDesc::default());
    }
    let (m, _) = RgfxComputePassDesc::read_from_prefix(body)
        .map_err(|_| op_fault("ENCODER_BEGIN_COMPUTE: body too short"))?;
    Ok(ComputePassDesc {
        dispatch_type: (m.flags & RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE != 0)
            .then_some(m.dispatch_type as u64),
    })
}

/// An `MTLClearColor` from its four `f64` bits.
fn clear_color(bits: [u64; 4]) -> ClearColor {
    ClearColor {
        red: f64::from_bits(bits[0]),
        green: f64::from_bits(bits[1]),
        blue: f64::from_bits(bits[2]),
        alpha: f64::from_bits(bits[3]),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES, RGFX_OP_RENDER_DRAW_PRIMITIVES,
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RGFX_OP_RENDER_SET_SCISSOR_RECT, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
        RGFX_OP_RENDER_SET_VIEWPORT, RgfxOpDrawIndexedPrimitives, RgfxOpDrawPrimitives,
        RgfxOpSetFragmentTexture, RgfxOpSetRenderPipelineState, RgfxOpSetScissorRect,
        RgfxOpSetVertexBuffer, RgfxOpSetViewport,
    };
    use reims_graphics::wire::{
        RGFX_CB_OK, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_EVENT_CREATE,
        RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_SET_VALUE,
        RGFX_CMD_TEXTURE_CREATE, RGFX_EVT_CB_COMPLETE, RgfxBufferCreate, RgfxCbComplete,
        RgfxObjectCreate, RgfxQueueCreate, RgfxSharedEventCreate, RgfxSharedEventSetValue,
        RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
    use reims_vgpu_core::platform::values::AttachmentState;
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::*;

    /// One op: its header then its body, `len` the whole op.
    fn op(tag: u16, body: &[u8]) -> Vec<u8> {
        let mut v = RgfxOpHeader {
            tag,
            flags: 0,
            len: (8 + body.len()) as u32,
        }
        .as_bytes()
        .to_vec();
        v.extend_from_slice(body);
        v
    }

    /// A fresh decoder with a queue (id 1), two buffers (2 vertex, 3 index) and two
    /// textures (4 colour attachment, 5 fragment) already created.
    fn setup() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );

        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));

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

        // The textures' resource ids are their object ids, in chunk 0.
        dec.hold_resource_ids(0, 0);
        for id in [4, 5] {
            let t = RgfxTextureCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_TEXTURE_CREATE,
                    size: size_of::<RgfxTextureCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                desc: RgfxTextureDescriptor {
                    pixel_format: 80,
                    texture_type: 2,
                    width: 256,
                    height: 256,
                    depth: 1,
                    mipmap_level_count: 1,
                    array_length: 1,
                    sample_count: 1,
                    usage: 5,
                    ..RgfxTextureDescriptor::default()
                },
                resource_id: id,
                ..RgfxTextureCreate::default()
            };
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        (p, dec)
    }

    /// An `ENCODER_BEGIN_RENDER` body: one colour attachment (texture 4, clear then
    /// store) over a 1920×1080 target.
    fn begin_render_body() -> Vec<u8> {
        let pass = RgfxRenderPassDesc {
            color_count: 1,
            flags: RGFX_PASS_FLAG_RT_SIZE,
            render_target_width: 1920,
            render_target_height: 1080,
            ..RgfxRenderPassDesc::default()
        };
        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: 4,
            load_action: 2,  // Clear
            store_action: 1, // Store
            clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
            ..RgfxRenderPassColorAttachment::default()
        };
        let mut body = pass.as_bytes().to_vec();
        body.extend_from_slice(color.as_bytes());
        body
    }

    /// The reference render ops, as the op stream carries them.
    fn render_ops() -> Vec<u8> {
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 0 }.as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_VERTEX_BUFFER,
            RgfxOpSetVertexBuffer {
                buffer_obj_id: 2,
                offset: 0,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            RgfxOpSetFragmentTexture {
                texture_obj_id: 5,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_VIEWPORT,
            RgfxOpSetViewport {
                origin_x_bits: 0.0f64.to_bits(),
                origin_y_bits: 0.0f64.to_bits(),
                width_bits: 1920.0f64.to_bits(),
                height_bits: 1080.0f64.to_bits(),
                znear_bits: 0.0f64.to_bits(),
                zfar_bits: 1.0f64.to_bits(),
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_SCISSOR_RECT,
            RgfxOpSetScissorRect {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES,
            RgfxOpDrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 6,
                index_type: 1,
                index_buffer_obj_id: 3,
                index_buffer_offset: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops
    }

    /// A `COMMAND_BUFFER` message for queue 1, command buffer `cb_obj_id`, carrying
    /// `ops`.
    fn command_buffer_message(cb_obj_id: u64, seqno: u64, ops: &[u8]) -> Vec<u8> {
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: (RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()) as u32,
                seqno,
            },
            queue_obj_id: 1,
            cb_obj_id,
            op_bytes: ops.len() as u64,
        };
        let mut msg = m.as_bytes().to_vec();
        msg.extend_from_slice(ops);
        msg
    }

    /// A command buffer opens a render encoder for its pass, sends each reference
    /// command in order, ends the encoder, and commits with a completion handler;
    /// completing the buffer produces the completion record.
    #[test]
    fn a_command_buffer_drives_the_encoder() {
        let (p, mut dec) = setup();
        let device = p.device();

        let captured = Arc::new(Mutex::new(Vec::<RgfxCbComplete>::new()));
        let sink = Arc::clone(&captured);
        dec.set_completion_sink(Arc::new(move |record| sink.lock().push(record)));

        let message = command_buffer_message(100, 55, &render_ops());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));

        // The command-buffer and encoder selectors, in order (retains and the
        // descriptor builder's unordered sets filtered out).
        let want = [
            "commandBuffer",
            "renderCommandEncoderWithDescriptor:",
            "setRenderPipelineState:",
            "setVertexBuffer:offset:atIndex:",
            "setFragmentTexture:atIndex:",
            "setViewport:",
            "setScissorRect:",
            "drawPrimitives:vertexStart:vertexCount:",
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
            "endEncoding",
            "addCompletedHandler:",
            "commit",
        ];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| want.contains(s))
            .collect();
        assert_eq!(got, want);

        // Completing the committed buffer runs the handler, which produces the
        // completion record for cb_obj_id 100 at the submission seqno.
        p.world().complete_all(CompleteOn::Caller);
        let recs = captured.lock();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].cb_obj_id, 100);
        assert_eq!(recs[0].header.seqno, 55);
        assert_eq!(recs[0].status, RGFX_CB_OK);
    }

    /// With a scheduled sink, a command buffer gets a scheduled handler before its completed
    /// handler and its commit; once the platform schedules it, its `CB_SCHEDULED` goes to the
    /// guest at its seqno, ahead of its completion record. With none, it gets no scheduled
    /// handler.
    #[test]
    fn a_command_buffer_posts_its_scheduled_record_ahead_of_its_completion() {
        let (p, mut dec) = setup();
        let device = p.device();
        let posted = Arc::new(Mutex::new(Vec::<(u32, u64, u64)>::new()));
        let scheduled = Arc::clone(&posted);
        dec.set_scheduled_sink(Arc::new(move |m| {
            assert_eq!(m.header.size as usize, size_of::<RgfxCbScheduled>());
            scheduled
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        let completions = Arc::clone(&posted);
        dec.set_completion_sink(Arc::new(move |m| {
            completions
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        let mark = p.call_log().mark();
        let message = command_buffer_message(130, 61, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let want = ["addScheduledHandler:", "addCompletedHandler:", "commit"];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| want.contains(s))
            .collect();
        assert_eq!(got, want);
        // The fake schedules a buffer as it is committed.
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SCHEDULED, 130, 61)]);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SCHEDULED, 130, 61),
                (RGFX_EVT_CB_COMPLETE, 130, 61)
            ]
        );

        // No sink: no scheduled handler.
        let (p, mut dec) = setup();
        let device = p.device();
        let mark = p.call_log().mark();
        let message = command_buffer_message(131, 62, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(
            !p.call_log()
                .sels_since(mark)
                .contains(&"addScheduledHandler:")
        );
    }

    /// The render-pass descriptor is built from the wire schema: its colour
    /// attachment carries the wire load and store actions.
    #[test]
    fn a_render_pass_is_built_from_the_wire_schema() {
        let (_p, dec) = setup();
        let pass = dec
            .build_render_pass(&begin_render_body(), |_| true)
            .unwrap();
        assert_eq!(
            pass.color_attachment_state(0).unwrap(),
            AttachmentState {
                load_action: 2,
                store_action: 1,
            }
        );
    }

    /// A pass's sample positions, after its colour records, are set on the platform's pass as
    /// the guest's descriptor holds them; a pass whose flag says so but carries none whole, or
    /// more than Metal's sixteen, faults, and so does one Metal does not take: a count other than
    /// 2, 4 or 8, one the device makes no texture of, or a coordinate outside [0, 1). One without
    /// the flag sets none.
    #[test]
    fn a_render_pass_takes_its_sample_positions() {
        use reims_vgpu_core::testkit::impls::descriptor::PassModel;
        let (p, dec) = setup();
        let mut caps = (*p.device().caps()).clone();
        caps.sample_counts = [Ok(true), Ok(true), Ok(true), Ok(false)];
        assert!(makes_sample_count(&caps, 4));
        assert!(!makes_sample_count(&caps, 8));
        assert!(!makes_sample_count(&caps, 3));
        let mut head = RgfxRenderPassDesc::read_from_prefix(&begin_render_body())
            .unwrap()
            .0;
        head.flags |= RGFX_PASS_FLAG_SAMPLE_POSITIONS;
        let mut positions = RgfxRenderPassSamplePositions {
            count: 2,
            ..RgfxRenderPassSamplePositions::default()
        };
        positions.position_bits[0] = [0.25f32.to_bits(), 0.75f32.to_bits()];
        positions.position_bits[1] = [0.5625f32.to_bits(), 0.125f32.to_bits()];
        let with = |p: &RgfxRenderPassSamplePositions| {
            let mut body = head.as_bytes().to_vec();
            body.extend_from_slice(&begin_render_body()[RGFX_RENDER_PASS_COLOR_OFFSET as usize..]);
            body.extend_from_slice(p.as_bytes());
            body
        };
        let pass = dec.build_render_pass(&with(&positions), |_| true).unwrap();
        let set = pass.with_model(|m: &PassModel| m.sample_positions.clone());
        assert_eq!(
            set,
            Some(vec![
                MtlSamplePosition { x: 0.25, y: 0.75 },
                MtlSamplePosition {
                    x: 0.5625,
                    y: 0.125
                },
            ])
        );
        let plain = dec
            .build_render_pass(&begin_render_body(), |_| true)
            .unwrap();
        assert_eq!(
            plain.with_model(|m: &PassModel| m.sample_positions.clone()),
            None
        );
        let short = with(&positions);
        assert!(
            dec.build_render_pass(&short[..short.len() - 8], |_| true)
                .is_err()
        );
        assert!(
            dec.build_render_pass(&with(&positions), |n| n != 2)
                .is_err()
        );
        for bad in [1.0f32, -0.25, f32::NAN] {
            let mut p = positions;
            p.position_bits[1][0] = bad.to_bits();
            assert!(dec.build_render_pass(&with(&p), |_| true).is_err(), "{bad}");
        }
        for count in [1, 3, 16] {
            positions.count = count;
            assert!(
                dec.build_render_pass(&with(&positions), |_| true).is_err(),
                "{count}"
            );
        }
        positions.count = RGFX_PASS_MAX_SAMPLE_POSITIONS + 1;
        assert!(dec.build_render_pass(&with(&positions), |_| true).is_err());
    }

    /// An unknown op in the stream faults the exec (a raise), after the encoder has
    /// opened.
    #[test]
    fn an_unknown_op_faults() {
        let (p, mut dec) = setup();
        let device = p.device();
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        // A render-group tag (0x1fff) the decoder does not implement.
        ops.extend(op(0x1fff, &[0u8; 8]));
        let message = command_buffer_message(101, 1, &ops);
        assert!(dec.decode(&device, &message).is_err());
    }

    /// A render op with no open encoder faults the exec.
    #[test]
    fn a_render_op_without_an_encoder_faults() {
        let (p, mut dec) = setup();
        let device = p.device();
        let ops = op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        );
        let message = command_buffer_message(102, 1, &ops);
        assert!(dec.decode(&device, &message).is_err());
    }

    /// A command buffer naming a queue the host does not hold is a bad reference,
    /// not a fault; a short message is malformed.
    #[test]
    fn a_missing_queue_and_a_short_message_are_reported() {
        let (p, mut dec) = setup();
        let device = p.device();
        let mut m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
                seqno: 1,
            },
            queue_obj_id: 999,
            cb_obj_id: 1,
            op_bytes: 0,
        }
        .as_bytes()
        .to_vec();
        assert_eq!(
            dec.decode(&device, &m).unwrap(),
            Err(DecodeError::BadReference)
        );
        m.truncate(8);
        assert_eq!(
            dec.decode(&device, &m).unwrap(),
            Err(DecodeError::Malformed)
        );
    }

    /// Creates the event 6 and the shared event 7, which starts at `initial`.
    fn create_events(p: &FakePlatform, dec: &mut GraphicsDecoder<FakePlatform>, initial: u64) {
        let device = p.device();
        let event = RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_EVENT_CREATE,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
        };
        assert_eq!(dec.decode(&device, event.as_bytes()).unwrap(), Ok(()));
        let shared = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 7,
            initial_value: initial,
        };
        assert_eq!(dec.decode(&device, shared.as_bytes()).unwrap(), Ok(()));
    }

    /// An event op's body: the event and the value.
    fn event_op(tag: u16, event_obj_id: u64, value: u64) -> Vec<u8> {
        op(
            tag,
            RgfxOpEvent {
                event_obj_id,
                value,
            }
            .as_bytes(),
        )
    }

    /// A shared event made at a value other than 0 is set to it, after its create.
    #[test]
    fn a_shared_event_starts_at_its_initial_value() {
        let (p, mut dec) = setup();
        let mark = p.call_log().mark();
        create_events(&p, &mut dec, 3);
        let sels = p.call_log().sels_since(mark);
        let made = sels.iter().position(|s| *s == "newSharedEvent").unwrap();
        let set = sels.iter().position(|s| *s == "setSignaledValue:").unwrap();
        assert!(made < set);

        // At 0 nothing is set.
        let (p, mut dec) = setup();
        let mark = p.call_log().mark();
        create_events(&p, &mut dec, 0);
        assert!(!p.call_log().sels_since(mark).contains(&"setSignaledValue:"));
    }

    /// The buffer's event ops are carried between its encoders, in stream order. Once it
    /// finishes, the value it signalled the shared event to goes to the guest ahead of its
    /// completion record; the plain event's signal has no value the guest reads, and is not
    /// posted.
    #[test]
    fn a_command_buffer_signals_and_waits_between_encoders() {
        let (p, mut dec) = setup();
        let device = p.device();
        create_events(&p, &mut dec, 0);

        let posted = Arc::new(Mutex::new(Vec::<(u32, u64, u64)>::new()));
        let signals = Arc::clone(&posted);
        dec.set_signal_sink(Arc::new(move |m| {
            signals.lock().push((m.header.r#type, m.obj_id, m.value));
        }));
        let completions = Arc::clone(&posted);
        dec.set_completion_sink(Arc::new(move |m| {
            completions.lock().push((m.header.r#type, m.cb_obj_id, 0));
        }));

        let mut ops = event_op(RGFX_OP_WAIT_EVENT, 6, 1);
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 9));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 6, 2));
        let mark = p.call_log().mark();
        let message = command_buffer_message(110, 56, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let want = [
            "commandBuffer",
            "encodeWaitForEvent:value:",
            "blitCommandEncoder",
            "endEncoding",
            "encodeSignalEvent:value:",
            "encodeSignalEvent:value:",
            "addCompletedHandler:",
            "commit",
        ];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| want.contains(s))
            .collect();
        assert_eq!(got, want);
        assert!(posted.lock().is_empty());

        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 9),
                (RGFX_EVT_CB_COMPLETE, 110, 0)
            ]
        );
    }

    /// A buffer that faults at an op is stood in for on its queue by one of its event ops alone,
    /// the waits and signals of the events the decoder holds, in stream order: its completion
    /// posts the shared event's value, then the buffer's failure. A buffer naming no queue the
    /// decoder holds has no stand-in.
    #[test]
    fn a_faulted_command_buffer_is_stood_in_for_by_its_event_ops() {
        use reims_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use reims_graphics::wire::RGFX_CB_INTERNAL;
        let (p, mut dec) = setup();
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let posted = Arc::new(Mutex::new(Vec::<(u32, u64, u64)>::new()));
        let signals = Arc::clone(&posted);
        dec.set_signal_sink(Arc::new(move |m| {
            signals.lock().push((m.header.r#type, m.obj_id, m.value));
        }));
        let completions = Arc::clone(&posted);
        dec.set_completion_sink(Arc::new(move |m| {
            completions
                .lock()
                .push((m.header.r#type, m.cb_obj_id, u64::from(m.status)));
        }));

        let mut ops = event_op(RGFX_OP_WAIT_EVENT, 6, 1);
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 9));
        ops.extend(op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        ));
        // Pipeline 99 is nothing the decoder holds: the walk faults here.
        ops.extend(op(
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState {
                pipeline_obj_id: 99,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 99, 3));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 10));
        let message = command_buffer_message(120, 60, &ops);
        assert!(dec.decode(&device, &message).is_err());
        assert!(posted.lock().is_empty());

        let mark = p.call_log().mark();
        assert!(dec.stand_in_for_failed(&message).unwrap());
        let want = [
            "commandBuffer",
            "encodeWaitForEvent:value:",
            "encodeSignalEvent:value:",
            "encodeSignalEvent:value:",
            "addCompletedHandler:",
            "commit",
        ];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| want.contains(s) || s.contains("Encoder"))
            .collect();
        assert_eq!(got, want);
        assert!(posted.lock().is_empty());
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 9),
                (RGFX_EVT_EVENT_SIGNALLED, 7, 10),
                (RGFX_EVT_CB_COMPLETE, 120, u64::from(RGFX_CB_INTERNAL)),
            ]
        );

        let mut elsewhere = command_buffer_message(121, 61, &ops);
        let queue = std::mem::offset_of!(RgfxCommandBuffer, queue_obj_id);
        elsewhere[queue..queue + 8].copy_from_slice(&99u64.to_le_bytes());
        assert!(!dec.stand_in_for_failed(&elsewhere).unwrap());
    }

    /// An event op with an encoder open faults the exec, as does one naming no event.
    #[test]
    fn an_event_op_inside_an_encoder_or_naming_no_event_faults() {
        let (p, mut dec) = setup();
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 1));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        assert!(
            dec.decode(&device, &command_buffer_message(111, 1, &ops))
                .is_err()
        );

        // The buffer 2 is no event; 99 is nothing.
        for named in [2, 99] {
            let ops = event_op(RGFX_OP_WAIT_EVENT, named, 1);
            assert!(
                dec.decode(&device, &command_buffer_message(112, 1, &ops))
                    .is_err()
            );
        }
    }

    /// `SHARED_EVENT_SET_VALUE` sets the shared event it names; naming anything else is a bad
    /// reference.
    #[test]
    fn a_cpu_set_reaches_the_shared_event() {
        let (p, mut dec) = setup();
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let set = |obj_id| RgfxSharedEventSetValue {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_SET_VALUE,
                size: size_of::<RgfxSharedEventSetValue>() as u32,
                seqno: 0,
            },
            obj_id,
            value: 12,
        };
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, set(7).as_bytes()).unwrap(), Ok(()));
        assert!(p.call_log().sels_since(mark).contains(&"setSignaledValue:"));
        for named in [6, 2, 99] {
            assert_eq!(
                dec.decode(&device, set(named).as_bytes()).unwrap(),
                Err(DecodeError::BadReference)
            );
        }
    }
}
