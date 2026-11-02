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
//! **Submission.** A buffer committed by `commitAndWaitUntilSubmitted` asks for its
//! `CB_SUBMITTED` (`SUBMITTED`), at which that call returns in the guest. Metal's returns once the
//! buffer is submitted (measured on the 5300M, the 630 and the M3: about 0.3 ms behind a 100 ms
//! buffer, the buffer then `Committed` or `Scheduled` by a race, and at the signal for a buffer
//! that waits on an event or is behind one on its queue; docs/graphics-kext/dgpu-oracle.md, Q7).
//! The host posts it as soon as it takes a buffer that waits for no event, needs no fill of a
//! device cache and has none before it on its queue that waits for one and is not yet scheduled
//! ([`Held`]), before it walks the buffer's ops ([`Take`]): the device's one decoder carries out
//! each message whole under its lock, so every message the guest sends after it, on any queue, is
//! carried out after the buffer's commit, and the platform orders the guest's later work after it
//! as it orders by commit. A buffer whose walk then faults has its failure at its completion, as
//! Metal reports a command buffer's errors. Any other buffer gets it at its scheduling, once the
//! waits are satisfied: Metal orders a tracked resource across queues behind what it has
//! submitted, not behind a buffer its queue holds (measured on the M3 through the decoder, a
//! later commit on another queue ran first), and the host's fills of device caches are waits
//! too. Either way it goes once, ahead of the buffer's `CB_SCHEDULED` and `CB_COMPLETE`
//! ([`Submitted`]).
//!
//! **A take on the doorbell's thread.** A buffer that gets its `CB_SUBMITTED` at its take may be
//! taken on the thread of the guest's doorbell write instead (`reims_graphics`'s
//! `Device::mmio_write_taking`), where nothing need be waited for there
//! ([`GraphicsDecoder::submits_at_take`]): it is begun there ([`GraphicsDecoder::begin_command_buffer`],
//! which makes its command buffer and fills no device cache), its `CB_SUBMITTED` posted there, and
//! the decoder's lock carried to its queue's service thread, which walks and commits it
//! ([`GraphicsDecoder::finish_command_buffer`]) before the lock goes. A queue's command buffers made
//! and not completed are counted ([`InFlight`]), so the platform's `commandBuffer`, which waits
//! while a queue holds 64, does not wait on that thread.
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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE, RGFX_OP_BODY_OFFSET,
    RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_BEGIN_RENDER,
    RGFX_OP_ENCODER_END, RGFX_OP_GROUP_BLIT, RGFX_OP_GROUP_COMPUTE, RGFX_OP_GROUP_FRAMING,
    RGFX_OP_GROUP_RENDER, RGFX_OP_SIGNAL_EVENT, RGFX_OP_SUBMITTED, RGFX_OP_SURFACE_USE,
    RGFX_OP_WAIT_EVENT, RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT, RGFX_PASS_FLAG_RT_ARRAY_LENGTH,
    RGFX_PASS_FLAG_RT_SIZE, RGFX_PASS_FLAG_SAMPLE_POSITIONS, RGFX_PASS_MAX_SAMPLE_POSITIONS,
    RGFX_RENDER_PASS_COLOR_OFFSET, RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpEvent,
    RgfxOpHeader, RgfxOpSurfaceUse, RgfxRenderPassColorAttachment, RgfxRenderPassDesc,
    RgfxRenderPassSamplePositions, op_group,
};
use reims_graphics::wire::{
    RGFX_EVT_CB_SCHEDULED, RGFX_EVT_CB_SUBMITTED, RGFX_EVT_EVENT_SIGNALLED, RgfxCbScheduled,
    RgfxCbSubmitted, RgfxEventSignalled, RgfxMsgHeader,
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
use reims_vgpu_core::sync::Mutex;
use zerocopy::FromBytes;

use super::super::completion::{cb_complete, cb_failed};
use super::{
    DecodeError, Drain, GraphicsDecoder, ObjHandle, SubmittedSink, blit, compute, op_fault, read,
    refusing, render,
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

/// A command buffer's `CB_SUBMITTED`, posted once: by the first of the host's take of it (for a
/// buffer that waits for nothing), the buffer's scheduled handler and its completed handler, each
/// of which posts it ahead of its own record, so it comes before them on the ring.
pub(crate) struct Submitted {
    sink: SubmittedSink,
    record: RgfxCbSubmitted,
    /// Whether it is posted, held across the post.
    posted: Mutex<bool>,
}

impl Submitted {
    /// Posts the record unless it is posted already.
    pub(crate) fn post(&self) {
        let mut posted = self.posted.lock();
        if !*posted {
            (self.sink)(self.record);
            *posted = true;
        }
    }

    /// Posts the record through `post` unless it is posted already or being posted, waiting for
    /// nothing: `post` posts it waiting for nothing and answers whether it did, and where it did
    /// not, a later [`Self::post`] posts it.
    pub(crate) fn try_post(&self, post: impl FnOnce(&RgfxCbSubmitted) -> bool) {
        let Some(mut posted) = self.posted.try_lock() else {
            return;
        };
        if !*posted {
            *posted = post(&self.record);
        }
    }
}

/// The most command buffers a queue may hold made and not completed for its next to be taken on
/// a doorbell's thread ([`GraphicsDecoder::submits_at_take`]): Metal's `commandBuffer`, and the
/// Vulkan platform's, wait while a queue holds 64, and a buffer leaves the count as its completed
/// handler ends, which may come before the platform lets its place go.
const TAKE_IN_FLIGHT_MAX: usize = 48;

/// A command buffer's place in its queue's count of the host's command buffers made on the queue
/// and not completed, which it leaves when dropped: with its completed handler, once that has run
/// ([`leaving`]), or with the buffer where it is never committed.
struct InFlight(Arc<AtomicUsize>);

impl InFlight {
    /// A place in `count`.
    fn new(count: &Arc<AtomicUsize>) -> InFlight {
        count.fetch_add(1, Ordering::AcqRel);
        InFlight(Arc::clone(count))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `handler`, after which the buffer leaves its queue's count (`in_flight`).
fn leaving(handler: CompletedHandler, in_flight: InFlight) -> CompletedHandler {
    Box::new(move |info: CompletedInfo| {
        handler(info);
        drop(in_flight);
    })
}

/// The op stream of the `COMMAND_BUFFER` `message` whose fixed part is `m`: its `op_bytes`, which
/// must lie within the message.
fn ops_of<'m>(message: &'m [u8], m: &RgfxCommandBuffer) -> Option<&'m [u8]> {
    let start = RGFX_COMMAND_BUFFER_OPS_OFFSET as usize;
    (m.op_bytes as usize)
        .checked_add(start)
        .and_then(|end| message.get(start..end))
}

/// A `COMMAND_BUFFER` begun ([`GraphicsDecoder::begin_command_buffer`]): its command buffer made on
/// its queue and the device caches it uses readied, before its walk
/// ([`GraphicsDecoder::finish_command_buffer`]).
pub(crate) struct Begun<P: Platform> {
    /// The queue it runs on.
    queue_obj_id: u64,
    /// The guest's id of the buffer.
    cb_obj_id: u64,
    /// Its message's seqno.
    seqno: u64,
    /// Its command buffer.
    cb: P::CommandBuffer,
    /// Its place in its queue's count.
    in_flight: InFlight,
    /// The blocks of the device caches it uses.
    used: Vec<u64>,
    /// Its `CB_SUBMITTED`, where it asks for one.
    submitted: Option<Arc<Submitted>>,
    /// Its hold of its queue, where it waits.
    held: Option<Arc<Held>>,
    /// A buffer before it on its queue holds the queue.
    behind: bool,
    /// Its context's drain.
    drain: Option<Arc<Drain>>,
}

impl<P: Platform> Begun<P> {
    /// Its `CB_SUBMITTED`, where it gets it at its take: it asks for one, waits for nothing (no
    /// event, no fill of a device cache), and no buffer before it on its queue holds the queue.
    /// Any other gets it at its scheduling.
    pub(crate) fn submitted_at_take(&self) -> Option<&Arc<Submitted>> {
        (self.submitted.as_ref()).filter(|_| self.held.is_none() && !self.behind)
    }
}

/// Commits `cb`, counted in its queue's hold first where it is `held`.
fn commit_holding<C: CommandBuffer<P>, P: Platform>(
    cb: &C,
    held: Option<&Held>,
) -> RaiseResult<()> {
    if let Some(held) = held {
        held.queue.fetch_add(1, Ordering::AcqRel);
    }
    let committed = cb.commit(ON);
    if committed.is_err()
        && let Some(held) = held
    {
        held.release();
    }
    committed
}

/// A committed command buffer that waits for an event, counted in its queue's hold
/// ([`GraphicsDecoder`]'s `held_queues`) until it is scheduled, or else finishes.
struct Held {
    /// Its queue's count of such buffers not yet scheduled.
    queue: Arc<AtomicUsize>,
    /// Whether it has left the count.
    released: AtomicBool,
}

impl Held {
    /// Takes the buffer out of its queue's count, once.
    fn release(&self) {
        if !self.released.swap(true, Ordering::AcqRel) {
            self.queue.fetch_sub(1, Ordering::AcqRel);
        }
    }
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

/// What the host reads of a command buffer's op stream as it takes the buffer, before its walk:
/// the IOSurface device caches it uses, filled first where the CPU wrote them since, and whether
/// it waits for an event and asks for its `CB_SUBMITTED`, which together decide whether it is
/// submitted at once.
#[derive(Default)]
struct Take {
    /// The `SURFACE_USE` ops, in stream order.
    uses: Vec<RgfxOpSurfaceUse>,
    /// The stream holds a `WAIT_EVENT`, of an event the host holds or not.
    waits: bool,
    /// The stream holds a `SUBMITTED`.
    asks: bool,
}

/// What the host takes of the op stream `ops` ([`Take`]). The scan stops at an op it cannot step
/// past, as the walk does.
fn take(mut ops: &[u8]) -> Take {
    let mut out = Take::default();
    while let Some(h) = read::<RgfxOpHeader>(ops) {
        let len = h.len as usize;
        if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
            break;
        }
        match h.tag {
            RGFX_OP_SURFACE_USE => {
                if let Ok((op, _)) =
                    RgfxOpSurfaceUse::read_from_prefix(&ops[RGFX_OP_BODY_OFFSET as usize..len])
                {
                    out.uses.push(op);
                }
            }
            RGFX_OP_WAIT_EVENT => out.waits = true,
            RGFX_OP_SUBMITTED => out.asks = true,
            _ => {}
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
    /// The IOSurfaces whose device caches it uses (its `SURFACE_USE` ops) are filled first where
    /// they hold an older generation, and it signals its queue's end event for their later copies
    /// ([`super::surface_cache`]). A buffer that asks for its `CB_SUBMITTED` and waits for nothing
    /// gets it before the walk; any other at its scheduling.
    pub(super) fn decode_command_buffer(
        &mut self,
        device: &P::Device,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let begun = match self.begin_command_buffer(device, message, drain, true)? {
            Ok(begun) => begun,
            Err(e) => return Ok(Err(e)),
        };
        // Submitted at once, unless it waits (for an event, or for the host's fill of a device
        // cache) or a buffer before it on its queue does: then at its scheduling. The guest's
        // later messages are carried out after this one's commit, under the decoder's lock.
        if let Some(submitted) = begun.submitted_at_take() {
            submitted.post();
        }
        self.finish_command_buffer(device, message, begun)
    }

    /// Whether the `COMMAND_BUFFER` `message`, of the current context, gets its `CB_SUBMITTED` at
    /// its take ([`Begun::submitted_at_take`]) with nothing waited for on the way, so a doorbell's
    /// thread may begin it, filling no device cache: its context's work is not refused, it names
    /// a command queue of its context, asks for its `CB_SUBMITTED` (with a sink to post it on) and
    /// waits for no event, no buffer before it holds its queue, the device caches it uses need no
    /// fill and no wait for one ([`super::surface_cache`]), and its queue holds fewer than
    /// [`TAKE_IN_FLIGHT_MAX`] command buffers made and not completed, so the platform's
    /// `commandBuffer` does not wait.
    pub(crate) fn submits_at_take(&self, message: &[u8]) -> bool {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return false;
        };
        let Some(ops) = ops_of(message, &m) else {
            return false;
        };
        if self.context_failed()
            || self.submitted_sink.is_none()
            || !matches!(self.named(m.queue_obj_id), Some(ObjHandle::CommandQueue(_)))
        {
            return false;
        }
        let take = take(ops);
        let count = |counts: &HashMap<u64, Arc<AtomicUsize>>| {
            counts
                .get(&m.queue_obj_id)
                .map_or(0, |n| n.load(Ordering::Acquire))
        };
        take.asks
            && !take.waits
            && count(&self.held_queues) == 0
            && count(&self.in_flight) < TAKE_IN_FLIGHT_MAX
            && self.surface_caches.ready(&take.uses)
    }

    /// Begins the `COMMAND_BUFFER` `message`, whose completion handler will hold `drain`: opens a
    /// command buffer on the named queue, readies the device caches its `SURFACE_USE` ops use
    /// (filling each that holds an older generation, where `may_fill`; a doorbell's thread, which
    /// checked [`Self::submits_at_take`], fills none), and makes its `CB_SUBMITTED` and its hold
    /// of its queue ([`Begun`]).
    pub(crate) fn begin_command_buffer(
        &mut self,
        device: &P::Device,
        message: &[u8],
        drain: Option<Arc<Drain>>,
        may_fill: bool,
    ) -> RaiseResult<Result<Begun<P>, DecodeError>> {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The op stream follows the message; `op_bytes` must lie within it.
        let Some(ops) = ops_of(message, &m) else {
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
        let in_flight = InFlight::new(self.in_flight.entry(m.queue_obj_id).or_default());
        let take = take(ops);
        let (used, fill_waits) = self
            .surface_caches
            .before(device, &cb, &take.uses, may_fill)?;
        let submitted = self.submitted(m.cb_obj_id, m.header.seqno, take.asks);
        let (behind, held) = self.hold(m.queue_obj_id, fill_waits || take.waits);
        Ok(Ok(Begun {
            queue_obj_id: m.queue_obj_id,
            cb_obj_id: m.cb_obj_id,
            seqno: m.header.seqno,
            cb,
            in_flight,
            used,
            submitted,
            held,
            behind,
            drain,
        }))
    }

    /// Finishes the `COMMAND_BUFFER` `message` that `begun` began: walks its op stream into its
    /// command buffer, ends its use of the device caches, and commits it with its scheduled and
    /// completed handlers.
    pub(crate) fn finish_command_buffer(
        &mut self,
        device: &P::Device,
        message: &[u8],
        begun: Begun<P>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Begun {
            queue_obj_id,
            cb_obj_id,
            seqno,
            cb,
            in_flight,
            used,
            submitted,
            held,
            drain,
            ..
        } = begun;
        let Some(ops) = read::<RgfxCommandBuffer>(message).and_then(|m| ops_of(message, &m)) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // A structural malformation in the stream stops the walk and is reported;
        // everything else faults the exec and propagates.
        let mut events = EventOps::default();
        if let Err(e) = self.walk_ops(ops, &cb, &mut events)? {
            return Ok(Err(e));
        }
        self.surface_caches
            .after(device, queue_obj_id, &cb, &used)?;
        // Register the scheduled and completion paths and commit. The GPU start and end times
        // are 0 here; the ring loop fills them when it posts the record on the queue's
        // completion ring in a later increment.
        if let Some(handler) =
            self.scheduled_handler(cb_obj_id, seqno, submitted.clone(), held.clone())
        {
            cb.add_scheduled_handler(ON, handler)?;
        }
        let handler = self.completed_handler(
            cb_obj_id,
            seqno,
            events,
            drain,
            false,
            (submitted, held.clone()),
        );
        cb.add_completed_handler(ON, leaving(handler, in_flight))?;
        commit_holding(&cb, held.as_deref())?;
        Ok(Ok(()))
    }

    /// Whether queue `queue_obj_id` is held, a buffer of it that waits for an event not yet
    /// scheduled; and, where the buffer to commit `waits` for an event (the host's fill of a
    /// device cache among them), its own hold of the queue, which keeps it and the queue's later
    /// buffers from being submitted before it is scheduled, as Metal submits a queue's buffers in
    /// order.
    fn hold(&mut self, queue_obj_id: u64, waits: bool) -> (bool, Option<Arc<Held>>) {
        let queue = self.held_queues.entry(queue_obj_id).or_default();
        let behind = queue.load(Ordering::Acquire) > 0;
        let held = waits.then(|| {
            Arc::new(Held {
                queue: Arc::clone(queue),
                released: AtomicBool::new(false),
            })
        });
        (behind, held)
    }

    /// The `CB_SUBMITTED` of command buffer `cb_obj_id`, at `seqno`, where it `asks` for one and
    /// there is a sink.
    fn submitted(&self, cb_obj_id: u64, seqno: u64, asks: bool) -> Option<Arc<Submitted>> {
        let sink = self.submitted_sink.clone().filter(|_| asks)?;
        Some(Arc::new(Submitted {
            sink,
            record: RgfxCbSubmitted {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_CB_SUBMITTED,
                    size: size_of::<RgfxCbSubmitted>() as u32,
                    seqno,
                },
                cb_obj_id,
            },
            posted: Mutex::new(false),
        }))
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
        let in_flight = InFlight::new(self.in_flight.entry(m.queue_obj_id).or_default());
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
        // It holds its queue as the buffer would have.
        let (_, held) = self.hold(m.queue_obj_id, !events.waits.is_empty());
        if let Some(held) = &held {
            let held = Arc::clone(held);
            cb.add_scheduled_handler(ON, Box::new(move || held.release()))?;
        }
        let handler = self.completed_handler(
            m.cb_obj_id,
            m.header.seqno,
            events,
            drain,
            true,
            (None, held.clone()),
        );
        cb.add_completed_handler(ON, leaving(handler, in_flight))?;
        commit_holding(&cb, held.as_deref())?;
        Ok(true)
    }

    /// The scheduled handler of command buffer `cb_obj_id`, at `seqno`: once the platform has
    /// scheduled it, it lets the buffer's queue go where the buffer `held` it, posts its
    /// `submitted` where not yet posted, and then its `CB_SCHEDULED`, so the guest's command
    /// buffer becomes `Scheduled` and its scheduled handlers and `waitUntilScheduled` go on then,
    /// as Metal's do. `None` without a sink, where it holds nothing.
    fn scheduled_handler(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        submitted: Option<Arc<Submitted>>,
        held: Option<Arc<Held>>,
    ) -> Option<ScheduledHandler> {
        let sink = self.scheduled_sink.clone();
        if sink.is_none() && held.is_none() {
            return None;
        }
        Some(Box::new(move || {
            if let Some(held) = &held {
                held.release();
            }
            if let Some(submitted) = &submitted {
                submitted.post();
            }
            let Some(sink) = &sink else {
                return;
            };
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
    /// `events`: where its scheduled handler did not, it lets its queue go where it `held` it and
    /// posts its `submitted`; then it posts the values the buffer signalled its shared events to,
    /// then its completion, which for a `stand_in` ([`Self::stand_in_for_failed`]) is its failure
    /// whatever the stand-in's outcome, and lets `drain` (its context's) go.
    fn completed_handler(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        events: EventOps,
        drain: Option<Arc<Drain>>,
        stand_in: bool,
        (submitted, held): (Option<Arc<Submitted>>, Option<Arc<Held>>),
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
            if let Some(held) = &held {
                held.release();
            }
            if let Some(submitted) = &submitted {
                submitted.post();
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
                let (start, end) = info.gpu_times_ns.unwrap_or((0, 0));
                cb_complete(cb_obj_id, seqno, start, end, info.error.as_ref())
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
            // Carried out before the walk, wherever they lie ([`take`]).
            RGFX_OP_SURFACE_USE | RGFX_OP_SUBMITTED => {}
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
        // The fake answers no GPU times unless set: 0 for each.
        assert_eq!((recs[0].gpu_start_ns, recs[0].gpu_end_ns), (0, 0));
    }

    /// A command buffer's completion record carries the GPU times the platform answers as its
    /// completed handler runs, in nanoseconds of the host's clock, which the guest bundle reports
    /// as `GPUStartTime` and `GPUEndTime`; a failed buffer the platform gives none of carries 0.
    #[test]
    fn a_completion_record_carries_the_platforms_gpu_times() {
        use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
        let (p, mut dec) = setup();
        let device = p.device();
        let captured = Arc::new(Mutex::new(Vec::<RgfxCbComplete>::new()));
        let sink = Arc::clone(&captured);
        dec.set_completion_sink(Arc::new(move |record| sink.lock().push(record)));
        let message = command_buffer_message(140, 70, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let committed = p.world().committed();
        assert_eq!(committed.len(), 1);
        assert!(
            p.world()
                .set_gpu_times(committed[0], 5_000_000_000, 5_002_000_000)
        );
        p.world().complete_all(CompleteOn::Caller);
        let message = command_buffer_message(141, 71, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let failed = p.world().committed();
        assert_eq!(failed.len(), 1);
        let error = NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code: 1,
            description: "Internal Error".to_owned(),
        };
        assert!(p.world().complete(
            failed[0],
            CommandBufferStatus::ERROR,
            Some(error),
            CompleteOn::Caller
        ));
        let recs = captured.lock();
        let times: Vec<_> = recs
            .iter()
            .map(|r| (r.cb_obj_id, r.gpu_start_ns, r.gpu_end_ns))
            .collect();
        assert_eq!(times, [(140, 5_000_000_000, 5_002_000_000), (141, 0, 0)]);
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

    /// A decoder whose submitted, scheduled (where `scheduled`) and completion records go, in the
    /// order posted, to the list it gives: each record's type, buffer and seqno.
    fn posting(scheduled: bool) -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>, Posted) {
        let (p, mut dec) = setup();
        let posted: Posted = Arc::new(Mutex::new(Vec::new()));
        let submitted = Arc::clone(&posted);
        dec.set_submitted_sink(Arc::new(move |m| {
            assert_eq!(m.header.size as usize, size_of::<RgfxCbSubmitted>());
            submitted
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        if scheduled {
            let scheduled = Arc::clone(&posted);
            dec.set_scheduled_sink(Arc::new(move |m| {
                scheduled
                    .lock()
                    .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
            }));
        }
        let completions = Arc::clone(&posted);
        dec.set_completion_sink(Arc::new(move |m| {
            completions
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        (p, dec, posted)
    }

    /// The records a decoder of [`posting`] posted: each one's type, buffer and seqno.
    type Posted = Arc<Mutex<Vec<(u32, u64, u64)>>>;

    /// A command buffer that asks for its `CB_SUBMITTED` gets it once, ahead of its
    /// `CB_SCHEDULED` and `CB_COMPLETE`, at its seqno; one that does not ask gets none.
    #[test]
    fn a_buffer_that_asks_is_submitted_ahead_of_its_other_records() {
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        let mut ops = op(RGFX_OP_SUBMITTED, &[]);
        ops.extend(render_ops());
        assert_eq!(
            dec.decode(&device, &command_buffer_message(140, 70, &ops))
                .unwrap(),
            Ok(())
        );
        // The fake schedules a buffer as it is committed.
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 140, 70),
                (RGFX_EVT_CB_SCHEDULED, 140, 70)
            ]
        );
        let message = command_buffer_message(141, 71, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 140, 70),
                (RGFX_EVT_CB_SCHEDULED, 140, 70),
                (RGFX_EVT_CB_SCHEDULED, 141, 71),
                (RGFX_EVT_CB_COMPLETE, 140, 70),
                (RGFX_EVT_CB_COMPLETE, 141, 71)
            ]
        );
    }

    /// Over a decoder of [`posting`], whose records it keeps posting there: the call-log mark at
    /// which each buffer's `CB_SUBMITTED` was posted, by buffer. Every call logged before the
    /// mark was made before the post.
    fn submitted_at(
        p: &Arc<FakePlatform>,
        dec: &mut GraphicsDecoder<FakePlatform>,
        posted: &Posted,
    ) -> Arc<Mutex<Vec<(u64, u64)>>> {
        let at = Arc::new(Mutex::new(Vec::new()));
        let (log, marks, posted) = (Arc::clone(p), Arc::clone(&at), Arc::clone(posted));
        dec.set_submitted_sink(Arc::new(move |m| {
            marks.lock().push((m.cb_obj_id, log.call_log().mark()));
            posted
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        at
    }

    /// The selectors called from `from` up to the post of buffer `cb_obj_id`'s `CB_SUBMITTED`,
    /// which `at` ([`submitted_at`]) holds.
    fn called_before_submitted(
        p: &FakePlatform,
        at: &Mutex<Vec<(u64, u64)>>,
        from: u64,
        cb_obj_id: u64,
    ) -> Vec<&'static str> {
        let (_, posted) = *at.lock().iter().find(|(cb, _)| *cb == cb_obj_id).unwrap();
        (p.call_log().since(from).into_iter())
            .filter(|c| c.seq < posted)
            .map(|c| c.sel)
            .collect()
    }

    /// With no scheduled sink, a buffer that asks and waits for no event gets no scheduled
    /// handler and is submitted as the host takes it, before it walks any op: its `SUBMITTED`
    /// ends the stream, as the bundle appends it at the commit, and the post comes ahead of the
    /// buffer's first encoder and its commit. One that waits for an event gets a scheduled
    /// handler, which holds its queue until it runs and posts the buffer's `CB_SUBMITTED` there,
    /// at its scheduling, after its commit (the fake schedules a buffer as it is committed).
    #[test]
    fn a_buffer_is_submitted_as_it_is_taken_unless_it_waits() {
        const ENCODER: &str = "renderCommandEncoderWithDescriptor:";
        let (p, mut dec, posted) = posting(false);
        let at = submitted_at(&p, &mut dec, &posted);
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let scheduled = |p: &FakePlatform, mark| {
            p.call_log()
                .sels_since(mark)
                .contains(&"addScheduledHandler:")
        };
        let mut ops = render_ops();
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        let mark = p.call_log().mark();
        let message = command_buffer_message(150, 80, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(!scheduled(&p, mark));
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SUBMITTED, 150, 80)]);
        let before = called_before_submitted(&p, &at, mark, 150);
        assert!(!before.contains(&ENCODER) && !before.contains(&"commit"));
        let all = p.call_log().sels_since(mark);
        assert!(all.contains(&ENCODER) && all.contains(&"commit"));
        let mut ops = event_op(RGFX_OP_WAIT_EVENT, 7, 1);
        ops.extend(render_ops());
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        let mark = p.call_log().mark();
        let message = command_buffer_message(151, 81, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(scheduled(&p, mark));
        let before = called_before_submitted(&p, &at, mark, 151);
        assert!(before.contains(&ENCODER) && before.contains(&"commit"));
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 150, 80),
                (RGFX_EVT_CB_SUBMITTED, 151, 81),
                (RGFX_EVT_CB_COMPLETE, 150, 80),
                (RGFX_EVT_CB_COMPLETE, 151, 81)
            ]
        );
        // Its scheduling let the queue go: a later buffer is submitted as it is taken.
        let mut ops = render_ops();
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        let mark = p.call_log().mark();
        let message = command_buffer_message(152, 82, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(
            posted.lock().last(),
            Some(&(RGFX_EVT_CB_SUBMITTED, 152, 82))
        );
        assert!(!called_before_submitted(&p, &at, mark, 152).contains(&ENCODER));
    }

    /// A buffer that asks and waits for nothing but whose walk then faults is submitted as it was
    /// taken, and fails at its completion, as Metal reports a command buffer's errors: the
    /// stand-in's `CB_COMPLETE`, its failure, follows. One the host cannot take, of a queue it
    /// does not hold, gets no `CB_SUBMITTED`.
    #[test]
    fn a_buffer_whose_walk_faults_is_submitted_then_fails() {
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        let mut ops = op(RGFX_OP_SUBMITTED, &[]);
        // A draw with no open encoder faults the walk.
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        ));
        let message = command_buffer_message(160, 90, &ops);
        assert!(refusing(|| dec.decode(&device, &message)).is_err());
        assert!(dec.stand_in_for_failed(&message).unwrap());
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 160, 90),
                (RGFX_EVT_CB_COMPLETE, 160, 90)
            ]
        );
        let mut message = command_buffer_message(161, 91, &op(RGFX_OP_SUBMITTED, &[]));
        let (mut m, _) = RgfxCommandBuffer::read_from_prefix(&message).unwrap();
        m.queue_obj_id = 0x77;
        message[..size_of::<RgfxCommandBuffer>()].copy_from_slice(m.as_bytes());
        assert_eq!(
            dec.decode(&device, &message).unwrap(),
            Err(DecodeError::BadReference)
        );
        assert_eq!(posted.lock().len(), 2);
    }

    /// A `COMMAND_BUFFER` of queue 1 that draws and asks for its `CB_SUBMITTED`.
    fn asking(cb_obj_id: u64, seqno: u64) -> Vec<u8> {
        let mut ops = render_ops();
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        command_buffer_message(cb_obj_id, seqno, &ops)
    }

    /// A buffer may be taken on a doorbell's thread only where it gets its `CB_SUBMITTED` at its
    /// take and need wait for nothing: not without the ask or a sink to post it on, not where it
    /// waits for an event or a buffer before it holds its queue, not on a queue its context does
    /// not hold, not while its queue holds [`TAKE_IN_FLIGHT_MAX`] command buffers made and not
    /// completed, and not once its context failed.
    #[test]
    fn a_buffer_submits_at_its_take_only_where_it_need_wait_for_nothing() {
        let (_, dec) = setup();
        assert!(!dec.submits_at_take(&asking(170, 100)), "no sink");
        let (p, mut dec, _posted) = posting(false);
        let device = p.device();
        create_events(&p, &mut dec, 0);
        assert!(dec.submits_at_take(&asking(170, 100)));
        let plain = command_buffer_message(171, 101, &render_ops());
        assert!(!dec.submits_at_take(&plain), "no ask");
        let mut waits = event_op(RGFX_OP_WAIT_EVENT, 7, 1);
        waits.extend(op(RGFX_OP_SUBMITTED, &[]));
        let waits = command_buffer_message(172, 102, &waits);
        assert!(!dec.submits_at_take(&waits), "waits for an event");
        // A buffer of queue 1 that waits, not yet scheduled (the fake schedules each at its
        // commit), holds the queue.
        let held = Arc::clone(dec.held_queues.entry(1).or_default());
        held.fetch_add(1, Ordering::AcqRel);
        assert!(
            !dec.submits_at_take(&asking(173, 103)),
            "behind a held buffer"
        );
        held.fetch_sub(1, Ordering::AcqRel);
        assert!(dec.submits_at_take(&asking(173, 103)));
        dec.set_context(Some(9));
        assert!(
            !dec.submits_at_take(&asking(174, 104)),
            "another context's queue"
        );
        dec.set_context(Some(0));
        assert!(dec.submits_at_take(&asking(174, 104)));
        for cb in 0..TAKE_IN_FLIGHT_MAX as u64 {
            let message = command_buffer_message(200 + cb, 200 + cb, &render_ops());
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        }
        assert!(!dec.submits_at_take(&asking(175, 105)), "its queue is full");
        p.world().complete_all(CompleteOn::Caller);
        assert!(dec.submits_at_take(&asking(175, 105)));
        assert!(dec.failed_contexts().fail(0));
        assert!(
            !dec.submits_at_take(&asking(175, 105)),
            "its context failed"
        );
    }

    /// A buffer taken on a doorbell's thread, begun there, with only its command buffer made, and
    /// finished later, makes the calls a decode makes, and posts its `CB_SUBMITTED` once: the
    /// doorbell's thread's post, waiting for nothing, leaves it where the ring is busy, the post
    /// before the walk makes it, and the scheduled and completed handlers make none.
    #[test]
    fn a_buffer_taken_on_a_doorbell_s_thread_is_carried_out_as_one_decoded() {
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &asking(180, 110)).unwrap(), Ok(()));
        let decoded = p.call_log().sels_since(mark);
        p.world().complete_all(CompleteOn::Caller);
        posted.lock().clear();

        let message = asking(181, 111);
        assert!(dec.submits_at_take(&message));
        let mark = p.call_log().mark();
        let drain = dec.command_buffer_drain();
        let begun = dec
            .begin_command_buffer(&device, &message, drain, false)
            .unwrap()
            .unwrap();
        assert_eq!(p.call_log().sels_since(mark), ["commandBuffer"]);
        let submitted = Arc::clone(begun.submitted_at_take().unwrap());
        submitted.try_post(|_| false);
        assert!(posted.lock().is_empty(), "the ring was busy");
        submitted.post();
        submitted.try_post(|_| panic!("posted twice"));
        assert_eq!(
            dec.finish_command_buffer(&device, &message, begun).unwrap(),
            Ok(())
        );
        assert_eq!(p.call_log().sels_since(mark), decoded);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 181, 111),
                (RGFX_EVT_CB_SCHEDULED, 181, 111),
                (RGFX_EVT_CB_COMPLETE, 181, 111)
            ]
        );
    }

    /// A `COMMAND_BUFFER` message for queue `queue_obj_id`, command buffer `cb_obj_id`, carrying
    /// `ops`.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    fn command_buffer_on(queue_obj_id: u64, cb_obj_id: u64, ops: &[u8]) -> Vec<u8> {
        let mut msg = command_buffer_message(cb_obj_id, cb_obj_id, ops);
        let (mut m, _) = RgfxCommandBuffer::read_from_prefix(&msg).unwrap();
        m.queue_obj_id = queue_obj_id;
        msg[..size_of::<RgfxCommandBuffer>()].copy_from_slice(m.as_bytes());
        msg
    }

    /// Through the decoder on `P`: buffer A of queue 1 fills a tracked buffer and asks for its
    /// submission, behind a buffer H of its queue that waits for a shared event the CPU sets
    /// later; as soon as A's `CB_SUBMITTED` comes, buffer B of queue 2 copies the filled buffer
    /// out. Whether A was answered before the CPU let H go, 100 ms on, and the bytes B copied.
    /// Two processes' queues are two queues of the platform's device to the decoder, and an
    /// IOSurface's device cache one tracked buffer to every process that uses it.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_later_commit_on_another_queue<P: crate::platform::ApiPlatform>() -> (bool, Vec<u8>) {
        use std::time::{Duration, Instant};

        use reims_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpFillBuffer,
        };
        use reims_graphics::wire::RgfxSharedEventSetValue;
        use reims_vgpu_core::platform::resource::Buffer as _;
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let submitted = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&submitted);
        dec.set_submitted_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
        let completed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&completed);
        dec.set_completion_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
        for obj_id in [1, 2] {
            let queue = RgfxQueueCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_QUEUE_CREATE,
                    size: size_of::<RgfxQueueCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
                ..RgfxQueueCreate::default()
            };
            assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        }
        for obj_id in [10, 11] {
            let buffer = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
                length: LEN,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        }
        let event = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 7,
            initial_value: 0,
        };
        assert_eq!(dec.decode(&device, event.as_bytes()).unwrap(), Ok(()));
        let hold = command_buffer_on(1, 19, &event_op(RGFX_OP_WAIT_EVENT, 7, 1));
        assert_eq!(dec.decode(&device, &hold).unwrap(), Ok(()));

        let mut ops = op(RGFX_OP_SUBMITTED, &[]);
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        let fill = RgfxBlitOpFillBuffer {
            buffer_obj_id: 10,
            range_location: 0,
            range_length: LEN,
            value: 0x5a,
            ..RgfxBlitOpFillBuffer::default()
        };
        ops.extend(op(RGFX_OP_BLIT_FILL_BUFFER, fill.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let a = command_buffer_on(1, 20, &ops);
        assert_eq!(dec.decode(&device, &a).unwrap(), Ok(()));
        std::thread::sleep(Duration::from_millis(100));
        let early = submitted.lock().contains(&20);
        let set = RgfxSharedEventSetValue {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_SET_VALUE,
                size: size_of::<RgfxSharedEventSetValue>() as u32,
                seqno: 0,
            },
            obj_id: 7,
            value: 1,
        };
        assert_eq!(dec.decode(&device, set.as_bytes()).unwrap(), Ok(()));
        let start = Instant::now();
        while !submitted.lock().contains(&20) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "A was never submitted"
            );
            std::thread::sleep(Duration::from_millis(1));
        }

        let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        let copy = RgfxBlitOpCopyBufferToBuffer {
            source_buffer_obj_id: 10,
            destination_buffer_obj_id: 11,
            source_offset: 0,
            destination_offset: 0,
            size: LEN,
        };
        ops.extend(op(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, copy.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let b = command_buffer_on(2, 21, &ops);
        assert_eq!(dec.decode(&device, &b).unwrap(), Ok(()));
        while !completed.lock().contains(&21) || !completed.lock().contains(&20) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "B never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let Some(ObjHandle::Buffer(out)) = dec.objects.get(&11) else {
            panic!("no buffer 11");
        };
        // SAFETY: a Shared buffer of `LEN` bytes, read after the command buffer that wrote it
        // finished.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), LEN as usize) };
        (early, bytes.to_vec())
    }

    /// On Metal, a command buffer behind one of its queue that waits for an event is not
    /// answered as submitted until that one is scheduled, as Metal's
    /// `commitAndWaitUntilSubmitted` returns only then (the M3, the 5300M and the 630,
    /// docs/graphics-kext/dgpu-oracle.md, Q7): Metal orders a tracked resource across queues
    /// behind work it has submitted, not behind work a queue holds. A buffer committed on another
    /// queue once the answer comes, which reads a tracked buffer the first one writes, sees its
    /// writes (Q6).
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_later_commit_on_another_queue_sees_a_submitted_buffer_s_writes() {
        let (early, bytes) = a_later_commit_on_another_queue::<reims_vgpu_metal::MetalPlatform>();
        assert!(!early, "answered as submitted while its queue was held");
        assert!(bytes.iter().all(|&b| b == 0x5a));
    }

    /// The same on the Vulkan platform
    /// (`on_metal_a_later_commit_on_another_queue_sees_a_submitted_buffer_s_writes`).
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_later_commit_on_another_queue_sees_a_submitted_buffer_s_writes() {
        let (early, bytes) = a_later_commit_on_another_queue::<reims_vgpu_vulkan::VulkanPlatform>();
        assert!(!early, "answered as submitted while its queue was held");
        assert!(bytes.iter().all(|&b| b == 0x5a));
    }

    /// A `BUFFER_CREATE` of `length` bytes, id `obj_id`, at `gpu_address` (0: where the host
    /// places it).
    #[cfg(feature = "vulkan")]
    fn buffer_at(obj_id: u64, gpu_address: u64, length: u64) -> RgfxBufferCreate {
        RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            gpu_address,
            length,
            ..RgfxBufferCreate::default()
        }
    }

    /// A decoder on the Vulkan platform with a queue (1), and, as a process in slot 0 picks
    /// (`RgfxIdSpace`, next fit from the slot's base, a pick of Metal's span of a pin), a
    /// device-placed 1 MiB texture buffer (10) at the slot's first pick, which the host takes no
    /// pin of: the runtime makes its bytes at an address it picks itself.
    #[cfg(feature = "vulkan")]
    fn decoder_with_a_texture_buffer() -> (
        GraphicsDecoder<reims_vgpu_vulkan::VulkanPlatform>,
        <reims_vgpu_vulkan::VulkanPlatform as Platform>::Device,
    ) {
        use crate::platform::ApiPlatform as _;
        use reims_graphics::wire::RGFX_CONTEXT_VA_BASE;
        let platform = reims_vgpu_vulkan::VulkanPlatform::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        dec.hold_resource_ids(0, 0);
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
        let texture = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id: 10,
            gpu_address: RGFX_CONTEXT_VA_BASE,
            resource_id: 1,
            desc: RgfxTextureDescriptor {
                // RGBA8Unorm, MTLTextureTypeTextureBuffer, ShaderRead, Shared.
                pixel_format: 70,
                texture_type: 9,
                width: 1 << 18,
                height: 1,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 1,
                ..RgfxTextureDescriptor::default()
            },
            ..RgfxTextureCreate::default()
        };
        assert_eq!(dec.decode(&device, texture.as_bytes()).unwrap(), Ok(()));
        (dec, device)
    }

    /// A guest's pin meets no range the Vulkan platform's runtime placed for itself: a buffer
    /// pinned at the pick after a texture buffer's is made, though the runtime made the texture
    /// buffer's bytes where it picks.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_pin_meets_no_range_of_a_texture_buffer() {
        use reims_graphics::wire::RGFX_CONTEXT_VA_BASE;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let buffer = buffer_at(11, RGFX_CONTEXT_VA_BASE + 0x8000, 0x1_0000);
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
    }

    /// The host's own ranges lie below every slot's GPU addresses: a buffer the host places
    /// itself lies below them, and, with it, a queue's and a texture buffer's ranges made, every
    /// slot's first pick and the last slot's last pick are made.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_the_host_s_own_ranges_lie_below_every_slot() {
        use reims_graphics::wire::{
            RGFX_CONTEXT_SLOTS, RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_SPAN,
        };
        use reims_vgpu_core::platform::resource::Buffer as _;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let placed = buffer_at(11, 0, 64 << 20);
        assert_eq!(dec.decode(&device, placed.as_bytes()).unwrap(), Ok(()));
        let Some(ObjHandle::Buffer(buffer)) = dec.objects.get(&11) else {
            panic!("no buffer 11");
        };
        assert!(buffer.gpu_address() < RGFX_CONTEXT_VA_BASE);
        // A pick of 64 KiB holds 96 KiB: its four pages, the guard page, rounded up to 32 KiB.
        let slots = u64::from(RGFX_CONTEXT_SLOTS);
        let last = RGFX_CONTEXT_VA_BASE + slots * RGFX_CONTEXT_VA_SPAN - 0x1_8000;
        let picks = (0..slots).map(|s| RGFX_CONTEXT_VA_BASE + s * RGFX_CONTEXT_VA_SPAN);
        for (obj_id, at) in (20..).zip(picks.chain([last])) {
            let pinned = buffer_at(obj_id, at, 0x1_0000);
            assert_eq!(
                dec.decode(&device, pinned.as_bytes()).unwrap(),
                Ok(()),
                "the pin at {at:#x}"
            );
        }
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
