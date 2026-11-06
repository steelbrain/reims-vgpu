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
//! ([`RgfxEventSignalled`]) as the host's event
//! reaches it, where the buffer runs more work after the signal, else once the buffer
//! finishes, and always ahead of its completion record ([`Signals`]); a buffer that
//! waits is watched, and named in the log where it waits long ([`super::waits`]).
//!
//! **Submission.** A buffer committed by `commitAndWaitUntilSubmitted` asks for its
//! `CB_SUBMITTED` (`SUBMITTED`), at which that call returns in the guest. Metal's returns once the
//! buffer is submitted (measured on the 5300M, the 630 and the M3: about 0.3 ms behind a 100 ms
//! buffer, the buffer then `Committed` or `Scheduled` by a race, and at the signal for a buffer
//! that waits on an event or is behind one on its queue; docs/graphics-kext/dgpu-oracle.md, Q7).
//! The host posts it as soon as it takes a buffer that waits for no event, needs no fill of a
//! device cache and has none before it on its queue that waits for one and is not yet scheduled
//! ([`Held`]), before it walks the buffer's ops ([`Take`]); a GPU wait for a buffer of another
//! queue ([`super::surface_order`]) holds back neither its `CB_SUBMITTED` nor its `CB_SCHEDULED`,
//! which then goes at its commit ([`Line`]), as a Mac's order of IOSurface accesses holds back no
//! record (Q16). Its ticket in the commit order is issued first ([`super::sequencer`]), so every
//! command buffer its process sends after it, on any queue, is committed after the buffer's
//! commit, and the platform orders the process's later work after it as it orders by commit;
//! another process's buffers wait for no ticket of its, as they name none of its tracked
//! resources. A buffer whose walk then faults has its failure at its completion, as Metal reports
//! a command buffer's errors. Any other buffer gets it at its scheduling, once the waits are
//! satisfied: Metal orders a tracked resource across queues behind what it has
//! submitted, not behind a buffer its queue holds (measured on the M3 through the decoder, a
//! later commit on another queue ran first), and the host's fills of device caches are waits
//! too. Either way it goes once, ahead of the buffer's `CB_SCHEDULED` and `CB_COMPLETE`
//! ([`Submitted`]).
//!
//! **An answer on the doorbell's thread.** A buffer that gets its `CB_SUBMITTED` at its take may
//! get it on the thread of the guest's doorbell write instead, before its queue's service thread
//! takes it off its ring (`reims_graphics`'s `Device::mmio_write_taking`), where nothing need be
//! waited for or allocated there ([`Decoders::answer_at_doorbell`]): it uses no device cache,
//! and every lock the answer takes is tried. Its turn in its queue's lane and its ticket in the
//! commit order are taken there, in the order its begin would take them, and kept on its queue
//! with its mark in that order ([`Answered`]), then its `CB_SUBMITTED` is posted. Its begin, on
//! the service thread, takes them in place of its own ([`GraphicsDecoder::begin_taken`]), makes
//! its command buffer and hands its walk and commit to the lane as it does any buffer's, and
//! posts no second `CB_SUBMITTED`; so its commit, and every commit it orders, comes where it
//! would had the answer been posted at its begin. One never begun (its queue gone, the device
//! reset) lets them go with its queue's entry. A queue's command buffers made and not completed
//! are counted ([`InFlight`]), and one is answered there only while its queue holds fewer than
//! [`TAKE_IN_FLIGHT_MAX`], so the platform's `commandBuffer`, which waits while a queue holds 64,
//! does not wait at its begin.
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

use std::collections::VecDeque;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use reims_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_VERTEX_BUFFER, RgfxOpDrawPrimitives,
    RgfxOpSetVertexBuffer,
};
use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE, RGFX_OP_BODY_OFFSET,
    RGFX_OP_COMMIT_ORDER, RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE,
    RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_BEGIN_VIDEO, RGFX_OP_ENCODER_END,
    RGFX_OP_GROUP_BLIT, RGFX_OP_GROUP_COMPUTE, RGFX_OP_GROUP_FRAMING, RGFX_OP_GROUP_RENDER,
    RGFX_OP_GROUP_VIDEO, RGFX_OP_SIGNAL_EVENT, RGFX_OP_SUBMITTED, RGFX_OP_SURFACE_ORDER,
    RGFX_OP_SURFACE_USE, RGFX_OP_WAIT_EVENT, RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT,
    RGFX_PASS_FLAG_RT_ARRAY_LENGTH, RGFX_PASS_FLAG_RT_SIZE, RGFX_PASS_FLAG_SAMPLE_POSITIONS,
    RGFX_PASS_FLAG_VISIBILITY_RESULT, RGFX_PASS_MAX_SAMPLE_POSITIONS,
    RGFX_RENDER_PASS_COLOR_OFFSET, RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpEvent,
    RgfxOpHeader, RgfxOpSurfaceUse, RgfxOpVideoPass, RgfxRenderPassColorAttachment,
    RgfxRenderPassDesc, RgfxRenderPassSamplePositions, RgfxRenderPassVisibility,
    RgfxSurfaceOrderWait, op_group,
};
use reims_graphics::wire::{
    RGFX_EVT_CB_SCHEDULED, RGFX_EVT_CB_SUBMITTED, RGFX_EVT_EVENT_SIGNALLED, RgfxCbScheduled,
    RgfxCbSubmitted, RgfxEventSignalled, RgfxMsgHeader,
};
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::caps::{DeviceCaps, SAMPLE_COUNTS};
use reims_vgpu_core::platform::command::{
    CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo, ScheduledHandler,
};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_core::platform::os::clock::{Clock as _, MonoTime};
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder, VertexDraw};
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::platform::types::{ClearColor, MtlSamplePosition};
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, ComputePassDesc, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_core::sync::Mutex;
use zerocopy::FromBytes;

use super::super::completion::{cb_complete, cb_failed};
use super::commit_order::Until;
use super::lane::{Lane, Turn};
use super::sequencer::Ticket;
use super::split::{Parts, SplitHold};
use super::surface_cache::Uses;
use super::surface_order::{Marks, Signalled};
use super::tables::QueueCounts;
use super::{
    DecodeError, Decoders, Drain, GraphicsDecoder, ObjHandle, ScheduledSink, SignalSink,
    SubmittedSink, blit, compute, context_of, op_fault, read, refused, refusing, render, video,
};

/// The op stream runs under a live raise handler (the ring loop's), so an encode's
/// raise returns for the handler to catch.
const ON: OnFault = OnFault::Return;

/// What a command buffer's op stream does with events: the shared events it signals, whose values
/// go to the guest as the GPU reaches them or once it finishes ([`Signals`]), and the events it
/// waits for.
#[derive(Default)]
struct EventOps {
    /// Each (shared event, value) the buffer signals, in stream order.
    signals: Finisher,
    /// Each (event, value) the buffer waits for, in stream order.
    waits: Vec<(u64, u64)>,
}

/// A buffer's [`Signals`], made at its first signal of a shared event and held for its completion
/// handler, which posts what is left of them ([`Signals::finish`]). Dropped without the handler
/// having run (a walk that faulted, which a stand-in replaces; a buffer never committed), it lets
/// their sink go, posting nothing more, so a notification that comes later posts nothing for the
/// buffer, and one never reached holds no ring.
#[derive(Default)]
struct Finisher(Option<Arc<Signals>>);

impl Finisher {
    /// The buffer's signals, made at the first, their values going to `sink`.
    fn signals(&mut self, sink: Option<&SignalSink>) -> &Arc<Signals> {
        self.0.get_or_insert_with(|| {
            Arc::new(Signals(Mutex::new(Posting {
                sink: sink.cloned(),
                signals: Vec::new(),
            })))
        })
    }

    /// The buffer finished: what no notification posted goes now.
    fn finish(&self) {
        if let Some(signals) = &self.0 {
            signals.finish();
        }
    }
}

impl Drop for Finisher {
    fn drop(&mut self) {
        if let Some(signals) = &self.0 {
            signals.0.lock().sink = None;
        }
    }
}

/// The values a command buffer signals its shared events to, each posted to the guest once
/// ([`RgfxEventSignalled`]). A signal the buffer runs more work after (an encoder, or a wait for
/// an event: [`work_after`]) is posted by the platform's notification as the host's event reaches
/// its value
/// ([`notify_listener`](reims_vgpu_core::platform::event::SharedEvent::notify_listener)): where
/// the GPU gets to it in the buffer, as Metal's value and listeners move there (measured:
/// docs/graphics-kext/dgpu-oracle.md, Q18). So a guest thread that waits on the CPU for it before
/// it sets a value the buffer waits for later is released, where a post at the buffer's end would
/// never come. The buffer's completion handler posts every one not yet posted, a signal at the
/// buffer's end among them, ahead of its completion and whatever its outcome, then lets the sink
/// go: a notification that comes later posts nothing. A value already reached when its
/// notification is asked for goes at once: the event is at it, whatever raised it. A signal of a
/// shared event whose create the host refused goes with the completion wherever it lies
/// ([`GraphicsDecoder::note_refused_signal`]).
#[derive(Default)]
struct Signals(Mutex<Posting>);

/// [`Signals`]' state.
#[derive(Default)]
struct Posting {
    /// Where the values go: none without a sink, and none once the completion handler posted.
    sink: Option<SignalSink>,
    /// Each (shared event, value), in stream order, and whether it is posted.
    signals: Vec<(u64, u64, bool)>,
}

impl Signals {
    /// Notes the signal of shared event `obj_id` to `value`: its place in the stream's signals.
    fn push(&self, obj_id: u64, value: u64) -> usize {
        let mut posting = self.0.lock();
        posting.signals.push((obj_id, value, false));
        posting.signals.len() - 1
    }

    /// Whether the values go anywhere, so a notification is worth asking for.
    fn posts(&self) -> bool {
        self.0.lock().sink.is_some()
    }

    /// The signal at `at` was reached: posts it, unless it is posted already or the buffer's
    /// completion handler has run.
    fn reached(&self, at: usize) {
        let mut posting = self.0.lock();
        let Posting { sink, signals } = &mut *posting;
        if let (Some(sink), Some((obj_id, value, posted))) = (sink.as_ref(), signals.get_mut(at))
            && !*posted
        {
            *posted = true;
            sink(event_signalled(*obj_id, *value));
        }
    }

    /// The buffer finished: posts each signal not yet posted, in stream order, and lets the sink
    /// go.
    fn finish(&self) {
        let mut posting = self.0.lock();
        let Some(sink) = posting.sink.take() else {
            return;
        };
        for (obj_id, value, posted) in &mut posting.signals {
            if !*posted {
                *posted = true;
                sink(event_signalled(*obj_id, *value));
            }
        }
    }
}

/// A command buffer's `CB_SUBMITTED`, posted once: by the first of the answer on a doorbell's
/// thread ([`Answered`]) or the host's take of it (for a buffer that waits for nothing), the
/// buffer's submission handler and its completion handler, each of which posts it ahead of its
/// own record, so it comes before them on the ring.
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
}

/// What a command buffer's completion handler posts and lets go where its submission handler has
/// not ([`submission_handler`]): its `CB_SUBMITTED`, its hold of its queue and its `CB_SCHEDULED`.
type Unposted = (
    Option<Arc<Submitted>>,
    Option<Arc<Held>>,
    Option<Arc<Scheduled>>,
);

/// A command buffer's `CB_SCHEDULED`, posted once: by its submission handler, or, for a buffer the
/// platform failed before it took its work, which runs no submission handler, by its completion
/// handler ahead of its completion, so each buffer's comes before its `CB_COMPLETE`.
struct Scheduled {
    sink: ScheduledSink,
    record: RgfxCbScheduled,
    /// Whether it is posted, held across the post.
    posted: Mutex<bool>,
}

impl Scheduled {
    /// Posts the record unless it is posted already.
    fn post(&self) {
        let mut posted = self.posted.lock();
        if !*posted {
            (self.sink)(self.record);
            *posted = true;
        }
    }

    /// Whether the record is posted.
    fn is_posted(&self) -> bool {
        *self.posted.lock()
    }
}

/// A committed command buffer's records in its queue's [`Line`]: its `CB_SUBMITTED`, where it asks
/// for one, and its `CB_SCHEDULED`.
type Records = (Option<Arc<Submitted>>, Arc<Scheduled>);

/// A queue's committed command buffers whose `CB_SCHEDULED` the host may post at their commit,
/// ahead of the platform's submission, and how many of them wait on the GPU for a buffer of another
/// queue ([`super::surface_order`]).
///
/// That wait is an order on the GPU, as a Mac's order of IOSurface accesses across queues and
/// processes is, and a Mac holds back no record for it: a buffer the GPU runs after another
/// process's or queue's 210 ms pass over the same surface returns from
/// `commitAndWaitUntilSubmitted` in 0.05 to 0.81 ms and is scheduled by 0.83 ms on the 5300M (0.57
/// to 4.65 ms on the 630; 0.04 to 0.14 ms on the M3, 1.7 to 4.7 ms in a run's first round), and
/// completes after the pass (docs/graphics-kext/dgpu-oracle.md, Q16). The platform may still hold
/// the buffer until the buffer it waits for finishes (the Vulkan platform holds a wait on a shared
/// event on the host until its value is reached), and the buffers behind it on its queue with it.
/// So such a buffer gets its `CB_SUBMITTED` at its take, as one that waits for nothing does, and
/// its `CB_SCHEDULED` at its commit; so does every buffer of its queue committed while one of them
/// is not yet taken by the platform. A buffer that waits for an event, or for the host's fill of a
/// device cache, or is behind one on its queue, keeps its records at its scheduling ([`Held`]).
///
/// The records of a queue go in its commit order: a buffer whose records go at its commit first
/// posts those of every buffer committed before it on its queue not yet posted, whose order on the
/// queue is fixed and none of which waits for anything by then ([`Line::post_through`]).
#[derive(Default)]
pub(crate) struct Line {
    /// The committed buffers' records not seen posted, oldest first.
    line: Mutex<VecDeque<Records>>,
    /// How many committed buffers of the queue that wait on the GPU for a buffer of another queue
    /// the platform has not yet taken.
    ordered: Arc<AtomicUsize>,
}

impl Line {
    /// How many committed buffers of the queue that wait on the GPU for a buffer of another queue
    /// the platform has not yet taken.
    #[cfg(test)]
    pub(crate) fn ordered(&self) -> usize {
        self.ordered.load(Ordering::Acquire)
    }

    /// Whether a buffer committed now posts its records at its commit, having none of its own
    /// order: one before it on its queue waits on the GPU for a buffer of another queue and is not
    /// yet taken by the platform, which takes this one after it.
    fn behind_an_order(&self) -> bool {
        self.ordered.load(Ordering::Acquire) > 0
    }

    /// The count of the queue's ordered buffers not yet taken by the platform, as a hold that
    /// leaves it once, for a buffer that waits on the GPU for a buffer of another queue.
    fn ordered_hold(&self) -> Arc<Held> {
        Arc::new(Held {
            queue: Arc::clone(&self.ordered),
            released: AtomicBool::new(false),
        })
    }

    /// Notes the records of a buffer just committed, last in the queue's commit order. Those at
    /// the front already posted (by their submission or completion handlers) leave.
    fn committed(&self, submitted: Option<Arc<Submitted>>, scheduled: Arc<Scheduled>) {
        let mut line = self.line.lock();
        while line.front().is_some_and(|(_, s)| s.is_posted()) {
            line.pop_front();
        }
        line.push_back((submitted, scheduled));
    }

    /// Posts, in commit order, the records of every buffer up to and including `last` (one
    /// [`Self::committed`] noted), each `CB_SUBMITTED` ahead of its `CB_SCHEDULED`, and lets them
    /// leave. Posted at most once each, whoever posts them.
    fn post_through(&self, last: &Arc<Scheduled>) {
        let mut line = self.line.lock();
        while let Some((submitted, scheduled)) = line.pop_front() {
            if let Some(submitted) = &submitted {
                submitted.post();
            }
            scheduled.post();
            if Arc::ptr_eq(&scheduled, last) {
                break;
            }
        }
    }
}

/// The most command buffers a queue may hold made and not completed for its next to be answered
/// on a doorbell's thread ([`Decoders::answer_at_doorbell`]): Metal's `commandBuffer`, and the
/// Vulkan platform's, wait while a queue holds 64, and a buffer leaves the count as its completion
/// handler ends, which may come before the platform lets its place go.
const TAKE_IN_FLIGHT_MAX: usize = 48;

/// A command buffer's place in its queue's count of the host's command buffers made on the queue
/// and not completed, which it leaves when dropped: with its completion handler, once that has run
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

/// A command buffer's walk where it may be committed in parts ([`super::split`]).
struct Splitting {
    /// Its context.
    context: u32,
    /// Its queue's count of command buffers made and not completed ([`InFlight`]).
    in_flight: Arc<AtomicUsize>,
    /// When the walk of the part it is walking began.
    since: MonoTime,
    /// Its context's gate, from its first part's commit to its last ([`SplitHold`]).
    hold: Option<SplitHold>,
    /// What its parts before the last found ([`Parts`]).
    parts: Option<Arc<Parts>>,
    /// The platform keeps it whole from now on.
    whole: bool,
}

/// The op stream of the `COMMAND_BUFFER` `message` whose fixed part is `m`: all of the message's
/// ops, which its `op_bytes` must count. The bundle writes a buffer's message as its fixed part
/// and its whole op stream, `op_bytes` the stream's length (`RgfxEncoder::finish`), so a message
/// that disagrees is not one the guest made: carrying out the ops a shorter `op_bytes` covers
/// would drop the rest without a word and answer the buffer completed.
fn ops_of<'m>(message: &'m [u8], m: &RgfxCommandBuffer) -> Option<&'m [u8]> {
    let ops = message.get(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize..)?;
    (u64::try_from(ops.len()).ok()? == m.op_bytes).then_some(ops)
}

/// A `COMMAND_BUFFER` begun ([`GraphicsDecoder::begin_command_buffer`]): its command buffer made on
/// its queue and the device caches it uses readied, before its walk
/// ([`GraphicsDecoder::finish_command_buffer`]).
pub(crate) struct Begun<P: Platform> {
    /// The queue it runs on.
    queue_obj_id: u64,
    /// That queue, where it holds a video encoder and uses a device cache: a decoder's output,
    /// which its client may lock to read each picture, is copied into its pages behind it there
    /// ([`super::surface_cache::Caches::finish`]).
    queue: Option<P::CommandQueue>,
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
    /// Its count among those caches' users, until its commit.
    uses: Option<Uses<P>>,
    /// It holds `SURFACE_ORDER`: it signals its queue's surface timeline as it ends.
    order: bool,
    /// Its queue's marks in the surface order.
    marks: Arc<Marks>,
    /// It waits on the GPU for a buffer of another queue that has not finished.
    waits_for_order: bool,
    /// Its queue's records the host may post at their commit.
    line: Arc<Line>,
    /// Its `CB_SUBMITTED`, where it asks for one.
    submitted: Option<Arc<Submitted>>,
    /// Its hold of its queue, where it waits.
    held: Option<Arc<Held>>,
    /// A buffer before it on its queue holds the queue.
    behind: bool,
    /// Its context's drain.
    drain: Option<Arc<Drain>>,
    /// It holds a video encoder, so it is walked in its turn: its session's work starts as it is
    /// walked, and so in its queue's order.
    ordered: bool,
}

impl<P: Platform> Begun<P> {
    /// It holds a video encoder, so it is walked in its turn.
    pub(crate) fn ordered(&self) -> bool {
        self.ordered
    }

    /// Its `CB_SUBMITTED`, where it gets it at its take: it asks for one, waits for no event and
    /// no fill of a device cache (a GPU wait for a buffer of another queue holds back nothing,
    /// [`Line`]), and no buffer before it on its queue holds the queue. Any other gets it at its
    /// scheduling.
    pub(crate) fn submitted_at_take(&self) -> Option<&Arc<Submitted>> {
        (self.submitted.as_ref()).filter(|_| self.held.is_none() && !self.behind)
    }
}

/// Commits `cb`, which holds its queue where it is `held` (counted since its begin), letting the
/// queue go where the commit fails.
fn commit_holding<C: CommandBuffer<P>, P: Platform>(
    cb: &C,
    held: Option<&Held>,
) -> RaiseResult<()> {
    let committed = cb.commit(ON);
    if committed.is_err()
        && let Some(held) = held
    {
        held.release();
    }
    committed
}

/// A command buffer's place in a count of its queue's: one that waits for an event, counted in its
/// queue's hold ([`QueueCounts`]'s `held`) from its begin until it is scheduled, or else finishes,
/// or goes uncommitted; or one that waits on the GPU for a buffer of another queue, counted in its
/// queue's [`Line`] from its commit until the platform takes it (its submission handler), or else
/// it finishes.
struct Held {
    /// The queue's count of such buffers not yet taken.
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

impl Drop for Held {
    /// A buffer never committed, or whose handlers went unrun, lets its queue go.
    fn drop(&mut self) {
        self.release();
    }
}

/// The submission handler of a command buffer: once the platform has taken its work for the GPU,
/// it lets the buffer's queue go where the buffer `held` it, posts its `submitted` where not yet
/// posted, and then its `scheduled`, so the guest's command buffer becomes `Scheduled` and its
/// scheduled handlers and `waitUntilScheduled` go on then; nothing committed later on its queue
/// runs before it. `None` where it has nothing to do.
fn submission_handler(
    submitted: Option<Arc<Submitted>>,
    held: Option<Arc<Held>>,
    scheduled: Option<Arc<Scheduled>>,
) -> Option<ScheduledHandler> {
    if scheduled.is_none() && held.is_none() {
        return None;
    }
    Some(Box::new(move || {
        if let Some(held) = &held {
            held.release();
        }
        if let Some(submitted) = &submitted {
            submitted.post();
        }
        if let Some(scheduled) = &scheduled {
            scheduled.post();
        }
    }))
}

/// Whether the queue whose counts are `counts` is held, a buffer of it begun before that waits for
/// an event and is not yet scheduled; and, where the buffer begun now `waits` for an event (the
/// host's fill of a device cache among them), its own hold of the queue, counted from now, which
/// keeps it and the queue's later buffers from being submitted before it is scheduled, as Metal
/// submits a queue's buffers in order. It is counted at the buffer's begin, not its commit, so a
/// buffer of the queue begun before this one commits (its lane walks several at once,
/// [`super::lane`]) is behind it, as one begun after its commit is.
fn hold(counts: &QueueCounts, waits: bool) -> (bool, Option<Arc<Held>>) {
    let behind = counts.held.load(Ordering::Acquire) > 0;
    let held = waits.then(|| {
        counts.held.fetch_add(1, Ordering::AcqRel);
        Arc::new(Held {
            queue: Arc::clone(&counts.held),
            released: AtomicBool::new(false),
        })
    });
    (behind, held)
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

/// Whether the op stream `ops`, what follows a signal in a buffer's, holds work the buffer runs
/// after the signal: an encoder, or a wait for an event, which its end comes after. The scan stops
/// at the first such op, which for a signal between encoders is the next one, or at an op it
/// cannot step past, as the walk does.
fn work_after(mut ops: &[u8]) -> bool {
    while let Some(h) = read::<RgfxOpHeader>(ops) {
        let len = h.len as usize;
        if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
            return false;
        }
        if matches!(
            h.tag,
            RGFX_OP_ENCODER_BEGIN_RENDER
                | RGFX_OP_ENCODER_BEGIN_COMPUTE
                | RGFX_OP_ENCODER_BEGIN_BLIT
                | RGFX_OP_ENCODER_BEGIN_VIDEO
                | RGFX_OP_WAIT_EVENT
        ) {
            return true;
        }
        ops = &ops[len..];
    }
    false
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
    /// The buffers of other queues its first `SURFACE_ORDER` names, where it holds one.
    order: Option<Vec<RgfxSurfaceOrderWait>>,
    /// The buffers of its process's other queues its first `COMMIT_ORDER` names, which it is
    /// committed after ([`super::commit_order`]); empty where it holds none.
    commit_after: Vec<RgfxSurfaceOrderWait>,
    /// The stream holds a video encoder, whose session's work starts as it is walked.
    video: bool,
}

/// Whether the op stream `ops` ends in a `SUBMITTED`, where the bundle puts it: appended as the
/// buffer is committed, after every op it encodes (its `SURFACE_USE` and `SURFACE_ORDER` ops go
/// before its first). Only its last 8 bytes are read, so true also where another op's body ends
/// in such bytes, which the scan of the ops then tells apart; a buffer that asks with its
/// `SUBMITTED` elsewhere is answered at its take.
fn ends_in_ask(ops: &[u8]) -> bool {
    let header = size_of::<RgfxOpHeader>();
    (ops.len().checked_sub(header))
        .and_then(|at| read::<RgfxOpHeader>(&ops[at..]))
        .is_some_and(|h| h.tag == RGFX_OP_SUBMITTED && h.len == RGFX_OP_BODY_OFFSET)
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
            RGFX_OP_SURFACE_ORDER if out.order.is_none() => {
                let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
                let (waits, _) = body.as_chunks::<{ size_of::<RgfxSurfaceOrderWait>() }>();
                let waits = (waits.iter())
                    .filter_map(|w| RgfxSurfaceOrderWait::read_from_bytes(w).ok())
                    .collect();
                out.order = Some(waits);
            }
            RGFX_OP_COMMIT_ORDER if out.commit_after.is_empty() => {
                let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
                out.commit_after = Glance::order_waits(body).collect();
            }
            RGFX_OP_ENCODER_BEGIN_VIDEO => out.video = true,
            _ => {}
        }
        ops = &ops[len..];
    }
    out
}

/// Whether the op stream `ops` holds a `SURFACE_ORDER`, as [`take`] finds it.
fn names_order(ops: &[u8]) -> bool {
    take(ops).order.is_some()
}

/// Each op of the op stream `ops`, in stream order: its tag and its body, read in place. It stops
/// at an op it cannot step past, as the walk does.
fn op_bodies(mut ops: &[u8]) -> impl Iterator<Item = (u16, &[u8])> {
    std::iter::from_fn(move || {
        let h = read::<RgfxOpHeader>(ops)?;
        let len = h.len as usize;
        if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
            return None;
        }
        let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
        ops = &ops[len..];
        Some((h.tag, body))
    })
}

/// What an answer on a doorbell's thread reads of a command buffer's op stream, in place and
/// allocating nothing, as [`take`] reads it: whether it asks for its `CB_SUBMITTED` and waits for
/// an event, its first `SURFACE_ORDER`'s and `COMMIT_ORDER`'s bodies, and the blocks its
/// `SURFACE_USE` ops name.
struct Glance<'o> {
    ops: &'o [u8],
    asks: bool,
    waits: bool,
    order: Option<&'o [u8]>,
    commit: Option<&'o [u8]>,
}

impl<'o> Glance<'o> {
    /// What the answer reads of `ops`.
    fn of(ops: &'o [u8]) -> Self {
        let mut glance = Glance {
            ops,
            asks: false,
            waits: false,
            order: None,
            commit: None,
        };
        for (tag, body) in op_bodies(ops) {
            match tag {
                RGFX_OP_WAIT_EVENT => glance.waits = true,
                RGFX_OP_SUBMITTED => glance.asks = true,
                RGFX_OP_SURFACE_ORDER if glance.order.is_none() => glance.order = Some(body),
                RGFX_OP_COMMIT_ORDER if glance.commit.is_none() => glance.commit = Some(body),
                _ => {}
            }
        }
        glance
    }

    /// The blocks the stream's `SURFACE_USE` ops name, in stream order.
    fn used_blocks(&self) -> impl Iterator<Item = u64> + 'o {
        (op_bodies(self.ops))
            .filter(|&(tag, _)| tag == RGFX_OP_SURFACE_USE)
            .filter_map(|(_, body)| RgfxOpSurfaceUse::read_from_prefix(body).ok())
            .map(|(op, _)| op.block_offset)
    }

    /// The buffers of other queues its first `SURFACE_ORDER` (or `COMMIT_ORDER`) names.
    fn order_waits(body: &'o [u8]) -> impl Iterator<Item = RgfxSurfaceOrderWait> + 'o {
        let (waits, _) = body.as_chunks::<{ size_of::<RgfxSurfaceOrderWait>() }>();
        (waits.iter()).filter_map(|w| RgfxSurfaceOrderWait::read_from_bytes(w).ok())
    }
}

/// A command buffer answered `CB_SUBMITTED` on a doorbell's thread before its queue's service
/// thread took it ([`Decoders::answer_at_doorbell`]), kept on its queue until its begin takes it
/// ([`GraphicsDecoder::begin_taken`]): what the answer took for it in the orders its begin keeps.
/// Let go of unbegun (its queue's entry gone, the device reset), its turn passes and its ticket
/// is given back.
pub(crate) struct Answered {
    /// Its message's seqno, and the guest's id of the buffer.
    seqno: u64,
    cb_obj_id: u64,
    /// The device's resets at the answer ([`super::ClosedContexts::resets`]): one from before a
    /// reset is not taken.
    resets: u64,
    /// The commit order's mark at the answer: every ticket issued before its own.
    mark: u64,
    /// Its context's range messages that came before the answer, where any was still to be
    /// carried out ([`super::sequencer::Sequencer::ranges_asked`]).
    ranges: Option<u64>,
    /// Its ticket in the commit order, issued before its `CB_SUBMITTED` was posted.
    ticket: Ticket,
    /// Its turn in its queue's lane, where the queue has one.
    turn: Option<Turn>,
    /// Its `CB_SUBMITTED` was posted; else the queue's ring was busy or full, and its begin posts
    /// it.
    posted: bool,
}

impl<P: Platform> Decoders<P> {
    /// Answers the `COMMAND_BUFFER` `message`, of `context`'s, at the head of its queue's ring,
    /// `CB_SUBMITTED` on a doorbell's thread, before its queue's service thread takes it (the
    /// module doc), where it gets that at its take ([`Begun::submitted_at_take`]) with nothing
    /// waited for or allocated on the way: its context's work is not refused, it names a command
    /// queue of its context, it asks for its `CB_SUBMITTED`, its last op, and waits for no event,
    /// it uses no device cache ([`super::surface_cache`]) and no block whose `SURFACE_PAGES` the
    /// surface worker is still carrying out ([`super::worker::Unsettled`]), which its begin waits
    /// for, waits for no buffer of another queue ([`super::surface_order`]), names in its
    /// `COMMIT_ORDER` only buffers begun ([`super::commit_order`]), no buffer before it holds its
    /// queue, its queue holds fewer than [`TAKE_IN_FLIGHT_MAX`] command buffers made
    /// and not completed and no answered buffer not yet begun, and each lock it takes is free,
    /// each tried. Its turn in its queue's lane and its
    /// ticket in the commit order are taken together, or neither, and kept on its queue
    /// ([`Answered`]), and the answer is posted through `post`, which waits for nothing and answers
    /// whether it posted it. False, with nothing taken, where it is not answered.
    pub(crate) fn answer_at_doorbell(
        &self,
        context: u32,
        message: &[u8],
        post: impl FnOnce(&RgfxCbSubmitted) -> bool,
    ) -> bool {
        let shared = &*self.0;
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return false;
        };
        let Some(ops) = ops_of(message, &m) else {
            return false;
        };
        // A stream that does not end in its ask is left to the take unread, so a buffer that
        // does not ask costs the doorbell no scan of its ops ([`ends_in_ask`]).
        if !ends_in_ask(ops) {
            return false;
        }
        let glance = Glance::of(ops);
        if !glance.asks
            || glance.waits
            || context_of(m.queue_obj_id) != context
            || shared.surface_caches.none_cached(glance.used_blocks()) != Some(true)
            || shared.unsettled.try_any(glance.used_blocks()) != Some(false)
        {
            return false;
        }
        if let Some(body) = glance.order {
            let Some(order) = shared.surface_order.try_lock() else {
                return false;
            };
            if !order.ready(m.queue_obj_id, Glance::order_waits(body)) {
                return false;
            }
        }
        if let Some(body) = glance.commit {
            let Some(order) = shared.surface_order.try_lock() else {
                return false;
            };
            if !order.commit_begun(m.queue_obj_id, Glance::order_waits(body)) {
                return false;
            }
        }
        let Some(ranges) = shared.sequencer.ranges_asked(context, true) else {
            return false;
        };
        // The context's table as it is now, no view taken (a view's taking allocates), with the
        // contexts' lock held, so the context is not let go of here.
        let Some(contexts) = shared.contexts.try_lock() else {
            return false;
        };
        let Some(state) = contexts.get(&context) else {
            return false;
        };
        let answered = state.try_with_live(m.queue_obj_id, |entry| {
            let (ObjHandle::CommandQueue(_), Some(counts)) = (&entry.handle, &entry.queue) else {
                return false;
            };
            if counts.held.load(Ordering::Acquire) != 0
                || counts.in_flight.load(Ordering::Acquire) >= TAKE_IN_FLIGHT_MAX
            {
                return false;
            }
            let Some(mut slot) = counts.answered.try_lock() else {
                return false;
            };
            if slot.is_some() {
                return false;
            }
            let issue = || shared.sequencer.issue_marked(context);
            let (turn, (mark, ticket)) = match &counts.lane {
                Some(lane) => match lane.try_turn_with(issue) {
                    Some((turn, issued)) => (Some(turn), issued),
                    None => return false,
                },
                None => match issue() {
                    Some(issued) => (None, issued),
                    None => return false,
                },
            };
            let record = RgfxCbSubmitted {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_CB_SUBMITTED,
                    size: size_of::<RgfxCbSubmitted>() as u32,
                    seqno: m.header.seqno,
                },
                cb_obj_id: m.cb_obj_id,
            };
            // Its place among the tickets is taken: a buffer that names it may take its own
            // ([`super::commit_order`]).
            counts.order.commit.begin(m.header.seqno);
            let posted = post(&record);
            *slot = Some(Answered {
                seqno: m.header.seqno,
                cb_obj_id: m.cb_obj_id,
                resets: shared.closed.resets(),
                mark,
                ranges,
                ticket,
                turn,
                posted,
            });
            true
        });
        answered == Some(true)
    }
}

/// `handler`, which first lets the buffer's queue's surface timeline know it finished, where it
/// `signalled` it ([`super::surface_order`]), so a buffer decoded once its completion is posted
/// waits for nothing of it.
fn signalling(handler: CompletedHandler, signalled: Option<Signalled>) -> CompletedHandler {
    match signalled {
        None => handler,
        Some(signalled) => Box::new(move |info: CompletedInfo| {
            signalled.finished();
            handler(info);
        }),
    }
}

/// `handler`, a command buffer's submission handler, which first lets the buffer's place among its
/// queue's ordered buffers go where it holds one ([`Line`]); `None` where neither has anything to
/// do.
fn taken_with(
    handler: Option<ScheduledHandler>,
    ordered: Option<Arc<Held>>,
) -> Option<ScheduledHandler> {
    let Some(ordered) = ordered else {
        return handler;
    };
    Some(Box::new(move || {
        ordered.release();
        if let Some(handler) = handler {
            handler();
        }
    }))
}

/// `handler`, a command buffer's completion handler, which first lets the buffer's place among its
/// queue's ordered buffers go where it holds one and its submission handler did not run
/// ([`Line`]).
fn completed_with(handler: CompletedHandler, ordered: Option<Arc<Held>>) -> CompletedHandler {
    match ordered {
        None => handler,
        Some(ordered) => Box::new(move |info: CompletedInfo| {
            ordered.release();
            handler(info);
        }),
    }
}

/// The encoder the walk currently holds open, one at a time, between an
/// `ENCODER_BEGIN_*` and the matching `ENCODER_END`.
enum OpenEncoder<P: Platform> {
    /// A render encoder, and what its ops' checks read of its pass.
    Render(P::RenderEncoder, render::PassFacts),
    /// A compute encoder, and its slots that hold an object whose create the host refused.
    Compute(P::ComputeEncoder, refused::Marks),
    Blit(P::BlitEncoder),
    /// A video encoder: its session's work, placed in the command buffer itself; its hold lets
    /// the session go on the video worker ([`video::SessionHold`]).
    Video(video::SessionHold<P>),
    /// An encoder of the op group, skipped whole: its pass named an object whose create the host
    /// refused (an attachment, a video session). No platform encoder is open, and its ops up to
    /// its end are dropped ([`refused`]).
    Skipped(u16),
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
        &self,
        device: &P::Device,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let begun = match self.begin_command_buffer(device, message, drain)? {
            Ok(begun) => begun,
            Err(e) => return Ok(Err(e)),
        };
        // Submitted at once, unless it waits (for an event, or for the host's fill of a device
        // cache) or a buffer before it on its queue does: then at its scheduling. Its ticket in
        // the commit order is issued first, so a buffer its process sends after the answer is
        // committed after this one ([`super::sequencer`]).
        if let Some(submitted) = begun.submitted_at_take() {
            self.issue_ticket();
            submitted.post();
        }
        self.finish_command_buffer(device, message, begun)
    }

    /// Issues the message's command buffer its ticket in the commit order, before its
    /// `CB_SUBMITTED` is posted at its take ([`super::sequencer`]), where it holds none yet (one
    /// answered on a doorbell's thread holds the answer's).
    fn issue_ticket(&self) {
        if self.ticket.borrow().is_some() {
            return;
        }
        *self.ticket.borrow_mut() = self.shared.sequencer.issue(self.context, false);
    }

    /// The command buffer's turn in the lane of the queue whose counts are `counts`
    /// ([`super::lane`]), where the queue has one.
    fn turn_on(counts: &QueueCounts) -> Option<Turn> {
        counts.lane.as_ref().and_then(|lane| lane.turn(false))
    }

    /// What the answer on a doorbell's thread took for the `COMMAND_BUFFER` `message` of the
    /// queue whose counts are `counts`, where it answered it ([`Answered`]), taken off the queue.
    /// One kept for another buffer, or from before a reset of the device, is let go of: the
    /// answer is of the buffer at the head of its queue's ring, so no buffer of the queue is begun
    /// between it and its begin.
    fn answered_for(&self, counts: &QueueCounts, message: &[u8]) -> Option<Answered> {
        let answered = counts.answered.lock().take()?;
        let m = read::<RgfxCommandBuffer>(message)?;
        let same = m.header.seqno == answered.seqno && m.cb_obj_id == answered.cb_obj_id;
        (same && answered.resets == self.shared.closed.resets()).then_some(answered)
    }

    /// Begins the `COMMAND_BUFFER` `message` a queue's service thread takes off the ring of the
    /// queue `queue`, as [`Self::decode_command_buffer`] carries it out up to its walk, which
    /// [`Self::finish_command_buffer`] does on the queue's lane ([`super::lane`]). Its turn in the
    /// lane is taken first, after its drain and before anything of it can fail, so its commit,
    /// its stand-in's or the failure posted for it comes in the order the queue's buffers were
    /// taken: the turn, the mark and the ticket the answer on a doorbell's thread took for it,
    /// where it was answered there ([`Answered`]). A context closed before the buffer came gets
    /// [`DecodeError::ContextClosed`], with no turn taken; and where it gets its
    /// `CB_SUBMITTED` at its take, that is posted here, once its ticket in the commit order is
    /// issued, unless the answer posted it.
    pub(crate) fn begin_taken(
        &mut self,
        device: &P::Device,
        message: &[u8],
        queue: Option<u64>,
    ) -> RaiseResult<Result<Begun<P>, DecodeError>> {
        self.end_message();
        let drain = self.command_buffer_drain();
        if drain.is_none() && self.context_closed() {
            return Ok(Err(DecodeError::ContextClosed));
        }
        if let Some(counts) = queue.and_then(|queue| self.queue_counts(queue)) {
            match self.answered_for(counts, message) {
                Some(answered) => {
                    self.mark.set(Some(answered.mark));
                    self.ranges.set(Some(answered.ranges));
                    *self.ticket.borrow_mut() = Some(answered.ticket);
                    *self.turn.borrow_mut() = answered.turn;
                    self.posted_at_doorbell.set(answered.posted);
                }
                None => *self.turn.borrow_mut() = Self::turn_on(counts),
            }
            // Its queue's commit marks rise to it, in its turn, whether it is committed, stood in
            // for or fails, however little of its message can be read ([`super::commit_order`]).
            if let Some(h) = read::<RgfxMsgHeader>(message) {
                *self.committed.borrow_mut() = Some((Arc::clone(&counts.order), h.seqno));
            }
        }
        let begun = self.begin_command_buffer(device, message, drain);
        // However far its begin got, a buffer that names it waits for it no more before its own
        // begin, only before its commit ([`super::commit_order`]).
        self.note_begun();
        let begun = match begun? {
            Ok(begun) => begun,
            Err(e) => return Ok(Err(e)),
        };
        if let Some(submitted) = begun.submitted_at_take() {
            self.issue_ticket();
            submitted.post();
        }
        Ok(Ok(begun))
    }

    /// Waits, with no lock held, for the turns in its queue's lane before the command buffer's
    /// (one taken before it), where it has one ([`super::lane`]).
    pub(crate) fn wait_turn(&self) {
        if let Some(turn) = self.turn.borrow().as_ref() {
            turn.wait();
        }
    }

    /// Whether the command buffer's turn in its queue's lane is next ([`Turn::is_next`]), so that
    /// it waits for no buffer of its queue; true where it has no turn.
    pub(crate) fn turn_is_next(&self) -> bool {
        self.turn.borrow().as_ref().is_none_or(Turn::is_next)
    }

    /// Whether the command buffer, begun, would be committed at once, waiting for nothing
    /// ([`Self::wait_to_commit`]): its turn is next, and no buffer of its context answered before
    /// its begin, no range message of its context before it, and no buffer its `COMMIT_ORDER`
    /// names is still to be committed. Waits for nothing: a lock found busy answers false.
    pub(crate) fn commits_at_once(&self) -> bool {
        let named = (self.commit_after.borrow().iter())
            .all(|(marks, seqno)| marks.commit.reached(*seqno, Until::Committed));
        if !self.turn_is_next() || !named {
            return false;
        }
        let sequencer = &self.shared.sequencer;
        let tickets = (self.mark.get())
            .is_none_or(|mark| sequencer.reached(self.context, mark) == Some(true));
        let ranges = match (self.context, self.ranges.get()) {
            (Some(context), Some(Some(asked))) => {
                sequencer.ranges_done(context, asked) == Some(true)
            }
            _ => true,
        };
        tickets && ranges
    }

    /// The lane of the command buffer's queue, where it was given a turn there.
    pub(crate) fn lane(&self) -> Option<Arc<Lane>> {
        self.turn
            .borrow()
            .as_ref()
            .map(|turn| Arc::clone(turn.lane()))
    }

    /// The command buffer's turn in its queue's lane, taken out of the handle, so the caller lets
    /// go of the rest of what the message holds and keeps its place in its queue's order (a
    /// failure it posts after).
    pub(crate) fn take_turn(&mut self) -> Option<Turn> {
        self.turn.get_mut().take()
    }

    /// Begins the `COMMAND_BUFFER` `message`, whose completion handler will hold `drain`: opens a
    /// command buffer on the named queue, readies the device caches its `SURFACE_USE` ops use
    /// (filling each that holds an older generation), has it wait on the GPU for the buffers of
    /// other queues its `SURFACE_ORDER` names that have not finished, and makes its
    /// `CB_SUBMITTED` and, where it waits for an event or a fill, its hold of its queue
    /// ([`Begun`]).
    pub(crate) fn begin_command_buffer(
        &self,
        device: &P::Device,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<Begun<P>, DecodeError>> {
        // The op stream follows the message's fixed part; `op_bytes` must count all of it.
        let parsed = read::<RgfxCommandBuffer>(message)
            .and_then(|m| ops_of(message, &m).map(|ops| (m, ops, take(ops))));
        // The buffers of its process's other queues it is committed after, as its `COMMIT_ORDER`
        // names them ([`super::commit_order`]).
        if let Some((m, _, take)) = &parsed
            && !take.commit_after.is_empty()
        {
            *self.commit_after.borrow_mut() = (self.shared.surface_order.lock())
                .commit_after(m.queue_obj_id, take.commit_after.iter().copied());
        }
        // It is committed after every buffer answered `CB_SUBMITTED` before now, and after its
        // context's range messages that came before now; and, before it takes that place, the
        // buffers it is committed after have taken theirs, so none of them waits for its ticket
        // or a later one ([`super::commit_order`]). An answer on a doorbell's thread took it
        // where they had.
        if self.mark.get().is_none() {
            let after = self.commit_after.borrow();
            if !after.is_empty() {
                let own = (self.committed.borrow().as_ref()).map(|(marks, _)| Arc::clone(marks));
                let cancel = &self.shared.commit_cancel;
                super::commit_order::wait(&after, own.as_deref(), cancel, Until::Begun);
            }
            self.mark.set(Some(self.shared.sequencer.mark()));
        }
        if self.ranges.get().is_none() {
            let asked = (self.context)
                .and_then(|context| self.shared.sequencer.ranges_asked(context, false))
                .flatten();
            self.ranges.set(Some(asked));
        }
        let Some((m, _, take)) = parsed else {
            return Ok(Err(DecodeError::Malformed));
        };
        // The queue the buffer runs on; a missing or wrong-kind queue is a
        // message-level bad reference (the op stream has not started).
        let Some(ObjHandle::CommandQueue(queue)) = self.named(m.queue_obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        let Some(counts) = self.queue_counts(m.queue_obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        let Some(cb) = queue.command_buffer(ON)? else {
            return Ok(Err(DecodeError::Refused));
        };
        super::work::command_buffer(self.context);
        let in_flight = InFlight::new(&counts.in_flight);
        // Its queue's commit marks rise to it, where its begin on a queue's service thread did
        // not note it already: begun now, and committed as the message ends
        // ([`super::commit_order`]).
        if self.committed.borrow().is_none() {
            *self.committed.borrow_mut() = Some((Arc::clone(&counts.order), m.header.seqno));
            self.note_begun();
        }
        // Each block it uses as its `SURFACE_PAGES` left it ([`super::surface_pages`]).
        for u in &take.uses {
            self.settle_surface(u.block_offset);
        }
        let (used, fill_waits, uses) = (self.surface_caches).before(device, &cb, &take.uses)?;
        // The buffers of other queues it runs after, on the GPU, kept for the work the host starts
        // off it (`surface_order`).
        let order_waits = match &take.order {
            Some(waits) => {
                (self.shared.surface_order.lock()).before(device, m.queue_obj_id, &cb, waits)?
            }
            None => Vec::new(),
        };
        let waits_for_order = !order_waits.is_empty();
        *self.order_waits.borrow_mut() = order_waits;
        let submitted = self.submitted(m.cb_obj_id, m.header.seqno, take.asks);
        // Those GPU waits hold back none of its records, nor its queue ([`Line`]).
        let (behind, held) = hold(counts, fill_waits || take.waits);
        Ok(Ok(Begun {
            queue_obj_id: m.queue_obj_id,
            queue: (take.video && uses.is_some()).then(|| queue.clone_ref()),
            cb_obj_id: m.cb_obj_id,
            seqno: m.header.seqno,
            cb,
            in_flight,
            used,
            uses,
            order: take.order.is_some(),
            marks: Arc::clone(&counts.order),
            waits_for_order,
            line: Arc::clone(&counts.line),
            submitted,
            held,
            behind,
            drain,
            ordered: take.video,
        }))
    }

    /// Finishes the `COMMAND_BUFFER` `message` that `begun` began: walks its op stream into its
    /// command buffer, has it signal its queue's surface timeline where it is to
    /// ([`Self::order_signal`]), ends its use of the device caches, and commits it with its
    /// submission and completion handlers.
    pub(crate) fn finish_command_buffer(
        &self,
        device: &P::Device,
        message: &[u8],
        begun: Begun<P>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Begun {
            queue_obj_id,
            queue,
            cb_obj_id,
            seqno,
            cb,
            in_flight,
            used,
            uses,
            order,
            marks,
            waits_for_order,
            line,
            submitted,
            held,
            behind,
            drain,
            ordered,
            ..
        } = begun;
        let Some(ops) = read::<RgfxCommandBuffer>(message).and_then(|m| ops_of(message, &m)) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // A structural malformation in the stream stops the walk and is reported;
        // everything else faults the exec and propagates.
        let mut events = EventOps::default();
        if ordered {
            self.wait_turn();
        }
        // Committed in parts as it is walked, where it may be ([`super::split`]): a buffer of a
        // context whose walk is not ordered, that waits for no event and is behind none that does.
        let mut split = (!ordered && held.is_none() && !behind)
            .then_some(self.context)
            .flatten()
            .map(|context| Splitting {
                context,
                in_flight: Arc::clone(&in_flight.0),
                since: self.shared.platform.clock().now_mono(),
                hold: None,
                parts: None,
                whole: false,
            });
        let mut cb = cb;
        if let Err(e) = self.walk_ops(ops, &mut cb, &mut events, &mut split)? {
            return Ok(Err(e));
        }
        // Its parts' gate, held to its last commit, and their outcome, which its last part's
        // completion makes the buffer's.
        let (parts, split_hold) = split.map_or((None, None), |s| (s.parts, s.hold));
        // After the buffers of its queue taken before it ([`super::lane`]): its signal of its
        // queue's surface timeline is decided in its ring's order ([`Self::order_signal`]).
        self.wait_turn();
        let signalled = self.order_signal(device, queue_obj_id, &marks, seqno, order, &cb)?;
        let signal_committed = signalled.as_ref().map(Signalled::committed);
        // Register the submission and completion paths and commit: no scheduled or completed
        // handler, so the platform posts the buffer's records from the threads that submit it
        // and see it complete, waking no handler thread.
        let scheduled = self.scheduled(cb_obj_id, seqno);
        // Its records go at its commit where the platform may hold it, or a buffer before it on
        // its queue, for a buffer of another queue on the GPU ([`Line`]); else at its scheduling.
        let early = held.is_none() && !behind && (waits_for_order || line.behind_an_order());
        let order_hold = waits_for_order.then(|| line.ordered_hold());
        let noted = scheduled.clone().map(|s| (submitted.clone(), s));
        let submission = submission_handler(submitted.clone(), held.clone(), scheduled.clone());
        let submission = taken_with(submission, order_hold.clone());
        let handler = self.completed_handler(
            cb_obj_id,
            seqno,
            events,
            drain,
            false,
            (submitted, held.clone(), scheduled),
        );
        let handler = signalling(handler, signalled);
        let handler = completed_with(handler, order_hold.clone());
        let handler = match parts {
            Some(parts) => parts.merged(handler),
            None => handler,
        };
        let commit = || {
            if let Some(submission) = submission {
                cb.add_submission_handler(ON, submission)?;
            }
            cb.add_completion_seen_handler(ON, leaving(handler, in_flight))?;
            if let Some(order_hold) = &order_hold {
                order_hold.queue.fetch_add(1, Ordering::AcqRel);
            }
            let committed = commit_holding(&cb, held.as_deref());
            if committed.is_err()
                && let Some(order_hold) = &order_hold
            {
                order_hold.release();
            }
            if committed.is_ok()
                && let Some(signal) = &signal_committed
            {
                signal.note();
            }
            committed
        };
        // After every buffer answered `CB_SUBMITTED` before its begin, and its context's range
        // messages before it ([`Self::wait_to_commit`]); and after the last part of a buffer of
        // its context whose parts are out, where its own are not ([`super::split`]).
        self.wait_to_commit();
        let under_way = match (&split_hold, self.context) {
            (None, Some(context)) => Some(self.shared.split_gate.commit(context)),
            _ => None,
        };
        match uses {
            // Its end event signalled for the caches' later copies under their lock, committed
            // with it let go, still among their users, and then the early copies behind it on its
            // queue ([`super::surface_cache::Caches::finish`]).
            Some(uses) => self.surface_caches.finish(
                device,
                queue_obj_id,
                queue.as_ref(),
                &cb,
                uses,
                &used,
                commit,
            )?,
            None => commit()?,
        }
        // Committed: a buffer behind it in the commit order goes on, and so does its context's
        // next commit and its range messages held while its parts were out.
        self.ticket.borrow_mut().take();
        drop((under_way, split_hold));
        // Last in its queue's order: its records, and those before it, go now where they go at
        // its commit.
        if let Some((submitted, scheduled)) = noted {
            line.committed(submitted, Arc::clone(&scheduled));
            if early {
                line.post_through(&scheduled);
            }
        }
        Ok(Ok(()))
    }

    /// The `CB_SUBMITTED` of command buffer `cb_obj_id`, at `seqno`, where it `asks` for one and
    /// there is a sink: posted already where the answer on a doorbell's thread posted it.
    fn submitted(&self, cb_obj_id: u64, seqno: u64, asks: bool) -> Option<Arc<Submitted>> {
        let sink = self.sinks.submitted.clone().filter(|_| asks)?;
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
            posted: Mutex::new(self.posted_at_doorbell.get()),
        }))
    }

    /// Stands in for the `COMMAND_BUFFER` `message` the decoder could not carry out (refused, or
    /// faulted at an op: one naming an object its process does not hold among them; what names
    /// an object whose create the host refused is dropped instead, [`refused`]), as a buffer Metal
    /// could not run still finishes: a command buffer on its queue holding the buffer's event ops
    /// alone, each wait and signal of an event its process holds, in stream order. Its completion
    /// posts the shared event values signalled and then the buffer's failure ([`cb_failed`]), so
    /// the failure comes in its queue's order, once the buffers committed before it finish, and
    /// nothing waiting for one of its values, on the GPU or in the guest, waits for good.
    /// `Ok(false)` where there is none, and the caller posts the failure at once: a message too
    /// short to name its queue, and a queue the process does not hold or that makes no command
    /// buffer.
    pub(crate) fn stand_in_for_failed(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<bool> {
        let Some(m) = read::<RgfxCommandBuffer>(message) else {
            return Ok(false);
        };
        // All of the message's ops, as the walk takes them ([`ops_of`]), whatever its `op_bytes`
        // read, so a wait or signal past a wrong count still runs.
        let ops = message
            .get(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize..)
            .unwrap_or_default();
        // What other queues' buffers may wait for of it is signalled whatever becomes of it
        // (`surface_order`).
        let names = names_order(ops);
        let (queue_obj_id, seqno) = (m.queue_obj_id, m.header.seqno);
        // The buffer's context's drain, held until the stand-in's completion handler has run.
        let drain = self.command_buffer_drain();
        let Some(ObjHandle::CommandQueue(queue)) = self.named(queue_obj_id) else {
            return Ok(false);
        };
        let Some(counts) = self.queue_counts(queue_obj_id) else {
            return Ok(false);
        };
        let Some(cb) = queue.command_buffer(ON)? else {
            let marks = Some(&*counts.order);
            self.wait_turn();
            self.order_release(device, queue_obj_id, marks, seqno, names, Some(queue))?;
            return Ok(false);
        };
        let in_flight = InFlight::new(&counts.in_flight);
        let mut events = EventOps::default();
        let carried = event_ops(ops);
        // A signal before a wait the stand-in carries is reached before the stand-in's end
        // ([`Signals`]).
        let last_wait = carried.iter().rposition(|(signal, op)| {
            !signal
                && matches!(
                    self.named(op.event_obj_id),
                    Some(ObjHandle::Event(_) | ObjHandle::SharedEvent(_))
                )
        });
        for (at, (signal, op)) in carried.into_iter().enumerate() {
            let (event, shared) = match self.named(op.event_obj_id) {
                Some(ObjHandle::Event(e)) => (EventRef::Event(e), None),
                Some(ObjHandle::SharedEvent(e)) => (EventRef::Shared(e), Some(e)),
                // As the walk takes one whose create the host refused.
                Some(ObjHandle::Refused(r)) => {
                    if signal && r.shared_event() {
                        self.note_refused_signal(&mut events, op);
                    }
                    continue;
                }
                _ => continue,
            };
            if signal {
                cb.encode_signal(Some(event), op.value)?;
                if let Some(shared) = shared {
                    let mid = last_wait.is_some_and(|wait| wait > at);
                    self.note_signal(&mut events, shared, op, mid)?;
                }
            } else {
                cb.encode_wait(Some(event), op.value)?;
                events.waits.push((op.event_obj_id, op.value));
            }
        }
        // It holds its queue as the buffer would have.
        let (_, held) = hold(counts, !events.waits.is_empty());
        if let Some(held) = &held {
            let held = Arc::clone(held);
            cb.add_submission_handler(ON, Box::new(move || held.release()))?;
        }
        // In its queue's order ([`super::lane`]): its signal, and its commit.
        self.wait_turn();
        let signalled =
            self.order_signal(device, queue_obj_id, &counts.order, seqno, names, &cb)?;
        let signal_committed = signalled.as_ref().map(Signalled::committed);
        let handler = self.completed_handler(
            m.cb_obj_id,
            m.header.seqno,
            events,
            drain,
            true,
            (None, held.clone(), None),
        );
        let handler = signalling(handler, signalled);
        cb.add_completion_seen_handler(ON, leaving(handler, in_flight))?;
        // In the commit order where the buffer was, after its context's range messages before it
        // ([`Self::wait_to_commit`]).
        self.wait_to_commit();
        commit_holding(&cb, held.as_deref())?;
        if let Some(signal) = &signal_committed {
            signal.note();
        }
        Ok(true)
    }

    /// Has `cb`, queue `queue_obj_id`'s buffer at `seqno` or its stand-in, signal the queue's
    /// surface timeline as it ends where it `names` the op or a buffer of another queue waits
    /// past what the queue signalled ([`super::surface_order::SurfaceOrder::signal`]); what its
    /// completed handler then calls. One that holds no op and is waited for by none takes no lock
    /// ([`Marks::quiet`]). Called in the buffer's turn in its queue's lane ([`super::lane`]), as
    /// [`Self::order_release`] is, so a queue's buffers note themselves taken in their ring's
    /// order whatever order their walks end in: the order reads a buffer taken past a seqno as
    /// every buffer up to it taken ([`super::surface_order`]).
    fn order_signal(
        &self,
        device: &P::Device,
        queue_obj_id: u64,
        marks: &Marks,
        seqno: u64,
        names: bool,
        cb: &P::CommandBuffer,
    ) -> RaiseResult<Option<Signalled>> {
        if !names && marks.quiet(seqno) {
            return Ok(None);
        }
        (self.shared.surface_order.lock()).signal(device, queue_obj_id, seqno, names, cb)
    }

    /// Queue `queue_obj_id`'s buffer at `seqno`, which `names` the op or not, will not run and
    /// has no stand-in: where a buffer of another queue may wait for it, a command buffer of
    /// `host_queue` (the queue's, where the host holds it) signals the queue's timeline to `seqno`
    /// after the queue's work before it, made once the order's lock is let go of; where it makes
    /// none, the timeline is set on the CPU ([`super::surface_order::Release`]).
    fn order_release(
        &self,
        device: &P::Device,
        queue_obj_id: u64,
        marks: Option<&Marks>,
        seqno: u64,
        names: bool,
        host_queue: Option<&P::CommandQueue>,
    ) -> RaiseResult<()> {
        if !names && marks.is_none_or(|m| m.quiet(seqno)) {
            return Ok(());
        }
        let release =
            (self.shared.surface_order.lock()).release(device, queue_obj_id, seqno, names);
        match release {
            Some(release) => release.signal(host_queue, &self.shared.commit_cancel),
            None => Ok(()),
        }
    }

    /// Waits, with no lock held, until the command buffer may be committed: after the buffers of
    /// its queue taken before it ([`super::lane`]), every buffer of its context answered
    /// `CB_SUBMITTED` before its begin, and its context's range messages that came before its begin
    /// ([`super::sequencer`]); then after the buffers of its process's other queues its
    /// `COMMIT_ORDER` names ([`super::commit_order`]).
    fn wait_to_commit(&self) {
        self.wait_turn();
        let sequencer = &self.shared.sequencer;
        if let Some(mark) = self.mark.get() {
            sequencer.wait_for(self.context, mark);
        }
        if let (Some(context), Some(Some(asked))) = (self.context, self.ranges.get()) {
            sequencer.wait_for_ranges(context, asked);
        }
        let after = std::mem::take(&mut *self.commit_after.borrow_mut());
        if !after.is_empty() {
            let own = (self.committed.borrow().as_ref()).map(|(marks, _)| Arc::clone(marks));
            let cancel = &self.shared.commit_cancel;
            super::commit_order::wait(&after, own.as_deref(), cancel, Until::Committed);
        }
    }

    /// The `CB_SCHEDULED` of command buffer `cb_obj_id`, at `seqno`, where there is a sink.
    fn scheduled(&self, cb_obj_id: u64, seqno: u64) -> Option<Arc<Scheduled>> {
        let sink = self.sinks.scheduled.clone()?;
        Some(Arc::new(Scheduled {
            sink,
            record: RgfxCbScheduled {
                header: RgfxMsgHeader {
                    r#type: RGFX_EVT_CB_SCHEDULED,
                    size: size_of::<RgfxCbScheduled>() as u32,
                    seqno,
                },
                cb_obj_id,
            },
            posted: Mutex::new(false),
        }))
    }

    /// The completion handler of command buffer `cb_obj_id`, at `seqno`, whose op stream did
    /// `events`, which the platform runs on the thread that saw the buffer complete: where its
    /// submission handler did not, it lets its queue go where it `held` it and posts its
    /// `submitted` and its `scheduled`; then it posts the values the buffer signalled its shared
    /// events to that no notification posted ([`Signals::finish`]), then its completion, which
    /// for a `stand_in` ([`Self::stand_in_for_failed`]) is its failure whatever the stand-in's
    /// outcome, and lets `drain` (its context's) go.
    fn completed_handler(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        events: EventOps,
        drain: Option<Arc<Drain>>,
        stand_in: bool,
        (submitted, held, scheduled): Unposted,
    ) -> CompletedHandler {
        let sink = self.sinks.completion.clone();
        let context = self.context;
        let signals = events.signals;
        let watched = (!events.waits.is_empty()).then(|| {
            self.shared
                .waits
                .watch(cb_obj_id, seqno, context, events.waits)
        });
        Box::new(move |info: CompletedInfo| {
            // The platform refused the buffer's completion on its own thread (a death in its
            // read, in the scope this decode carried there, which logs the death at WARN):
            // the buffer fails, as one a GPU error strikes on a Mac does, and its context's later
            // work is carried out, as that process's later buffers run (docs/lessons.md).
            if info.is_refused() {
                log::debug!(
                    target: "reims_vgpu::graphics",
                    "command buffer {cb_obj_id:#x} (seqno {seqno}) failed off its op's thread: \
                     it fails, and its context goes on"
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
            if let Some(scheduled) = &scheduled {
                scheduled.post();
            }
            // The shared events the buffer signalled that no notification posted, ahead of its
            // completion, so the guest has each value before the buffer's handlers run. A
            // buffer that failed may have signalled them before its fault, and a guest waiting
            // for one would wait for good: each is posted whatever the outcome.
            signals.finish();
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

    /// Notes the buffer's signal `op` of the shared event `event` in `events`; where the buffer
    /// runs more work after it (`mid`), the platform posts its value as the event reaches it, on
    /// the device's listener ([`Signals`]).
    fn note_signal(
        &self,
        events: &mut EventOps,
        event: &P::SharedEvent,
        op: RgfxOpEvent,
        mid: bool,
    ) -> RaiseResult<()> {
        let signals = events.signals.signals(self.sinks.signal.as_ref());
        let at = signals.push(op.event_obj_id, op.value);
        if !mid || !signals.posts() {
            return Ok(());
        }
        let signals = Arc::clone(signals);
        let listener = self.shared.listener();
        event.notify_listener(listener, op.value, Box::new(move || signals.reached(at)))
    }

    /// Notes the buffer's signal `op` of a shared event whose create the host refused in
    /// `events`: its value goes to the guest with the buffer's completion, ahead of it
    /// ([`Signals::finish`]), wherever it lies in the stream. The host holds no event a
    /// notification could watch, and the walk runs ahead of the GPU, so a post as the walk reaches
    /// it could come before the buffer's earlier work is done.
    fn note_refused_signal(&self, events: &mut EventOps, op: RgfxOpEvent) {
        let signals = events.signals.signals(self.sinks.signal.as_ref());
        signals.push(op.event_obj_id, op.value);
    }

    /// At an encoder's end, ops left after it: where the part of the buffer walked since the
    /// last took [`Self::split_after`], its queue holds room for another command buffer, the buffer
    /// could be committed now ([`Self::may_commit_now`]) and its context's gate is free or its
    /// own, hands the platform that part ([`CommandBuffer::split`]) and goes on in the command
    /// buffer it answers, `cb` from then on ([`super::split`]). It waits for nothing. Where the
    /// platform keeps the buffer whole, it is walked whole from there.
    fn split_here(
        &self,
        cb: &mut P::CommandBuffer,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<()> {
        let Some(s) = split.as_mut().filter(|s| !s.whole) else {
            return Ok(());
        };
        let clock = self.shared.platform.clock();
        if clock.now_mono().saturating_duration_since(s.since) < self.split_after()
            || s.in_flight.load(Ordering::Acquire) >= TAKE_IN_FLIGHT_MAX
            || !self.may_commit_now()
        {
            return Ok(());
        }
        let hold = match s.hold.take() {
            Some(hold) => hold,
            None => match self.shared.split_gate.try_split(s.context) {
                Some(hold) => hold,
                None => return Ok(()),
            },
        };
        let first = s.parts.is_none();
        let parts = Arc::clone(s.parts.get_or_insert_with(Arc::default));
        // The part's own place among its queue's command buffers, until its completion; the
        // buffer's goes on to its last part.
        let part = parts.part_handler(InFlight::new(&s.in_flight));
        cb.add_completion_seen_handler(ON, part)?;
        match cb.split(ON)? {
            Some(next) => {
                *cb = next;
                s.hold = Some(hold);
                s.since = clock.now_mono();
            }
            None => {
                // Kept whole: the handler just added sees the buffer's own completion, which
                // its outcome then is ([`Parts::merged`]); the gate stays held only where parts
                // are out.
                s.whole = true;
                if !first {
                    s.hold = Some(hold);
                }
            }
        }
        Ok(())
    }

    /// Whether the command buffer could be committed now, waiting for nothing: its turn is next,
    /// every ticket before its begin is given back, its context's range messages before it are
    /// carried out, and the buffers its `COMMIT_ORDER` names are committed
    /// ([`Self::wait_to_commit`]).
    fn may_commit_now(&self) -> bool {
        if !self.turn_is_next() {
            return false;
        }
        let sequencer = &self.shared.sequencer;
        // An order that is busy now counts as not reached: the buffer waits for its next
        // encoder's end, or its commit.
        if let Some(mark) = self.mark.get()
            && sequencer.reached(self.context, mark) != Some(true)
        {
            return false;
        }
        if let (Some(context), Some(Some(asked))) = (self.context, self.ranges.get())
            && sequencer.ranges_done(context, asked) != Some(true)
        {
            return false;
        }
        (self.commit_after.borrow().iter())
            .all(|(marks, seqno)| marks.commit.reached(*seqno, Until::Committed))
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
        cb: &mut P::CommandBuffer,
        events: &mut EventOps,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let mut encoder: Option<OpenEncoder<P>> = None;
        let walked = refusing(|| self.walk(ops, cb, &mut encoder, events, split));
        if let Some(open) = encoder.take() {
            end_abandoned(open);
        }
        walked
    }

    /// [`Self::walk_ops`]'s walk, `encoder` the one it holds open, `split` where it may commit
    /// the buffer in parts ([`Self::split_here`]), each next part then `cb`.
    fn walk(
        &self,
        mut ops: &[u8],
        cb: &mut P::CommandBuffer,
        encoder: &mut Option<OpenEncoder<P>>,
        events: &mut EventOps,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        #[cfg(test)]
        let (mut index, hook) = (0, self.walk_hook());
        while !ops.is_empty() {
            let Some((tag, body, len)) = next_op(ops) else {
                return Ok(Err(DecodeError::Malformed));
            };
            match op_group(tag) {
                RGFX_OP_GROUP_FRAMING => {
                    self.framing_op((tag, body, &ops[len..]), cb, encoder, events)?;
                    if tag == RGFX_OP_ENCODER_END && len < ops.len() {
                        self.split_here(cb, split)?;
                    }
                }
                // An op of an encoder skipped whole ([`OpenEncoder::Skipped`]): dropped.
                group if matches!(encoder.as_ref(), Some(OpenEncoder::Skipped(g)) if *g == group) =>
                    {}
                RGFX_OP_GROUP_RENDER => {
                    let Some(OpenEncoder::Render(enc, pass)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "render"));
                    };
                    // This op and the render ops after it, up to the first of another group,
                    // one run of the encoder's ([`RenderEncoder::encoding`]).
                    match enc.encoding(|| self.render_ops(ops, enc, pass))? {
                        Ok(rest) => ops = rest,
                        Err(e) => return Ok(Err(e)),
                    }
                    continue;
                }
                RGFX_OP_GROUP_COMPUTE => {
                    let Some(OpenEncoder::Compute(enc, marks)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "compute"));
                    };
                    // A dispatch while a slot holds a refused object is dropped ([`refused`]).
                    let dropped =
                        marks.skips() || (marks.drops_draws() && compute::dispatches(tag));
                    if !dropped && !contain(compute::decode_op(tag, body, self, enc, marks, ON))? {
                        return Err(unknown_op(tag));
                    }
                }
                RGFX_OP_GROUP_BLIT => {
                    let Some(OpenEncoder::Blit(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "blit"));
                    };
                    if !contain(blit::decode_op(tag, body, self, enc, ON))? {
                        return Err(unknown_op(tag));
                    }
                }
                RGFX_OP_GROUP_VIDEO => {
                    let Some(OpenEncoder::Video(session)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "video"));
                    };
                    if !contain(video::decode_op(tag, body, self, session.session(), cb))? {
                        return Err(unknown_op(tag));
                    }
                }
                _ => return Err(unknown_op(tag)),
            }
            ops = &ops[len..];
            #[cfg(test)]
            if let Some(hook) = &hook {
                hook(index);
            }
            #[cfg(test)]
            {
                index += 1;
            }
        }
        if encoder.is_some() {
            return Err(op_fault("the op stream ended with an encoder still open"));
        }
        Ok(Ok(()))
    }

    /// The render ops at the start of `ops`, on `enc`, whose pass `pass` describes, up to the
    /// first op of another group or the stream's end: what is left of `ops`, or a malformed op's
    /// error, each as [`Self::walk`] answers it at that op.
    fn render_ops<'o>(
        &self,
        mut ops: &'o [u8],
        enc: &P::RenderEncoder,
        pass: &render::PassFacts,
    ) -> RaiseResult<Result<&'o [u8], DecodeError>> {
        let mut draws = Draws::new();
        while !ops.is_empty() {
            // Draws one after another go to the encoder together; each other op, after them. While
            // a slot holds a refused object, each op goes alone, and a draw is dropped
            // ([`refused`]).
            let refused = pass.refused.drops_draws();
            if !refused && let Some((draw, len)) = self.vertex_draw(ops) {
                draws.push(draw, enc)?;
                ops = &ops[len..];
                continue;
            }
            draws.send(enc)?;
            let Some((tag, body, len)) = next_op(ops) else {
                return Ok(Err(DecodeError::Malformed));
            };
            if op_group(tag) != RGFX_OP_GROUP_RENDER {
                break;
            }
            let dropped = refused && render::draws(tag);
            if !dropped && !contain(render::decode_op(tag, body, self, enc, pass, ON))? {
                return Err(unknown_op(tag));
            }
            ops = &ops[len..];
        }
        draws.send(enc)?;
        Ok(Ok(ops))
    }

    /// The draw at the start of `ops`, and the bytes of its two ops: a `SET_VERTEX_BUFFER` whose
    /// body reads and whose buffer resolves, then a `DRAW_PRIMITIVES` whose body reads, as
    /// [`render::decode_op`] reads each. `None` for anything else, which the walk then carries out
    /// an op at a time, faulting where each op does. The lookup has no effect on what the message
    /// names: a miss is looked up again by the op.
    fn vertex_draw<'s>(&'s self, ops: &[u8]) -> Option<(VertexDraw<'s, P>, usize)> {
        #[cfg(test)]
        if !tests::GATHERS.get() {
            return None;
        }
        let (RGFX_OP_RENDER_SET_VERTEX_BUFFER, body, len) = next_op(ops)? else {
            return None;
        };
        let (bind, _) = RgfxOpSetVertexBuffer::read_from_prefix(body).ok()?;
        let (RGFX_OP_RENDER_DRAW_PRIMITIVES, body, draw_len) = next_op(&ops[len..])? else {
            return None;
        };
        let (draw, _) = RgfxOpDrawPrimitives::read_from_prefix(body).ok()?;
        let buffer = self.get_buffer(bind.buffer_obj_id).ok()?;
        let draw = VertexDraw {
            buffer,
            offset: bind.offset,
            index: bind.index,
            primitive_type: draw.primitive_type,
            vertex_start: draw.vertex_start,
            vertex_count: draw.vertex_count,
        };
        Some((draw, len + draw_len))
    }

    /// A framing op, `tag` with `body` and the stream's `rest` after it: open the encoder its
    /// `ENCODER_BEGIN_*` names, end the open one, or signal or wait for an event between
    /// encoders. An already-open encoder at a begin or at an event op, or no open encoder at an
    /// end, faults.
    fn framing_op(
        &self,
        (tag, body, rest): (u16, &[u8], &[u8]),
        cb: &P::CommandBuffer,
        encoder: &mut Option<OpenEncoder<P>>,
        events: &mut EventOps,
    ) -> RaiseResult<()> {
        match tag {
            RGFX_OP_ENCODER_BEGIN_RENDER => {
                already_open(encoder, "ENCODER_BEGIN_RENDER")?;
                let built = self.build_render_pass(body, |count| {
                    cb.device()
                        .is_some_and(|d| makes_sample_count(&d.caps(), count))
                });
                // A pass that names an object whose create the host refused (an attachment, a
                // resolve texture, its visibility result buffer) is skipped whole ([`refused`]).
                let (pass, facts) = match built {
                    Err(raise) if refused::is_refused(&raise) => {
                        *encoder = Some(OpenEncoder::Skipped(RGFX_OP_GROUP_RENDER));
                        return Ok(());
                    }
                    built => built?,
                };
                let enc = cb
                    .render_encoder(ON, &pass)?
                    .ok_or_else(|| op_fault("the render encoder was nil"))?;
                super::work::render_encoder(self.context);
                *encoder = Some(OpenEncoder::Render(enc, facts));
            }
            RGFX_OP_ENCODER_BEGIN_COMPUTE => {
                already_open(encoder, "ENCODER_BEGIN_COMPUTE")?;
                let desc = compute_pass(body)?;
                let enc = cb
                    .compute_encoder_for_pass(&desc)?
                    .ok_or_else(|| op_fault("the compute encoder was nil"))?;
                *encoder = Some(OpenEncoder::Compute(enc, refused::Marks::default()));
            }
            RGFX_OP_ENCODER_BEGIN_BLIT => {
                already_open(encoder, "ENCODER_BEGIN_BLIT")?;
                let enc = cb
                    .blit_encoder(ON)?
                    .ok_or_else(|| op_fault("the blit encoder was nil"))?;
                *encoder = Some(OpenEncoder::Blit(enc));
            }
            RGFX_OP_ENCODER_BEGIN_VIDEO => {
                already_open(encoder, "ENCODER_BEGIN_VIDEO")?;
                let pass = RgfxOpVideoPass::read_from_prefix(body)
                    .map(|(pass, _)| pass)
                    .map_err(|_| op_fault("ENCODER_BEGIN_VIDEO: body too short"))?;
                // A session whose create the host refused skips the encoder whole ([`refused`]).
                *encoder = Some(match self.require_video_session(pass.session_obj_id) {
                    Ok(session) => OpenEncoder::Video(session),
                    Err(raise) if refused::is_refused(&raise) => {
                        OpenEncoder::Skipped(RGFX_OP_GROUP_VIDEO)
                    }
                    Err(raise) => return Err(raise),
                });
            }
            RGFX_OP_ENCODER_END => match encoder.take() {
                Some(OpenEncoder::Render(enc, _)) => enc.encode(ON, RenderCmd::EndEncoding)?,
                Some(OpenEncoder::Compute(enc, _)) => enc.encode(ON, ComputeCmd::EndEncoding)?,
                Some(OpenEncoder::Blit(enc)) => enc.encode(ON, BlitCmd::EndEncoding)?,
                Some(OpenEncoder::Video(_) | OpenEncoder::Skipped(_)) => {}
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
                    Some(ObjHandle::Event(e)) => (EventRef::Event(e), None),
                    Some(ObjHandle::SharedEvent(e)) => (EventRef::Shared(e), Some(e)),
                    // An event whose create the host refused ([`refused`]): nothing on the host
                    // signals it, so a wait for it is dropped; a signal of a shared one still
                    // reaches the guest's value as the buffer finishes, where its waiters are,
                    // whatever work follows it: with no host event to watch, a post as the walk
                    // reached it could come before the GPU's work before it is done
                    // ([`Self::note_refused_signal`]).
                    Some(ObjHandle::Refused(r)) => {
                        r.used(op.event_obj_id);
                        if signal && r.shared_event() {
                            self.note_refused_signal(events, op);
                        }
                        return Ok(());
                    }
                    _ => return Err(super::bad_reference()),
                };
                if signal {
                    cb.encode_signal(Some(event), op.value)?;
                    if let Some(shared) = shared {
                        self.note_signal(events, shared, op, work_after(rest))?;
                    }
                } else {
                    cb.encode_wait(Some(event), op.value)?;
                    events.waits.push((op.event_obj_id, op.value));
                }
            }
            // Carried out before the walk, wherever they lie ([`take`]).
            RGFX_OP_SURFACE_USE
            | RGFX_OP_SUBMITTED
            | RGFX_OP_SURFACE_ORDER
            | RGFX_OP_COMMIT_ORDER => {}
            _ => return Err(unknown_op(tag)),
        }
        Ok(())
    }

    /// Builds a `P::RenderPassDescriptor` from an `ENCODER_BEGIN_RENDER` op's body
    /// (an [`RgfxRenderPassDesc`] then its `color_count`
    /// [`RgfxRenderPassColorAttachment`] records, then its sample positions and its visibility
    /// result buffer where its flags say so), with what the pass's render ops' checks read of
    /// it. A body too short faults; a reference to a texture or a buffer the host does not
    /// hold faults through the resolution API; and so do sample positions Metal does not take
    /// ([`sample_positions_taken`]), of which `makes` answers whether the device makes textures
    /// of their count, and so does a visibility result type Metal does not define.
    fn build_render_pass(
        &self,
        body: &[u8],
        makes: impl Fn(u32) -> bool,
    ) -> RaiseResult<(P::RenderPassDescriptor, render::PassFacts)> {
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
                texture: Some(Some(tex)),
                resolve_texture: resolve.map(Some),
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

        // The pass's sample positions, after the colour records, then its visibility result
        // buffer.
        let mut off = color_start + desc.color_count as usize * stride;
        if desc.flags & RGFX_PASS_FLAG_SAMPLE_POSITIONS != 0 {
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
            off += size_of::<RgfxRenderPassSamplePositions>();
        }
        let mut facts = render::PassFacts::default();
        if desc.flags & RGFX_PASS_FLAG_VISIBILITY_RESULT != 0 {
            let v = body
                .get(off..off + size_of::<RgfxRenderPassVisibility>())
                .and_then(|s| RgfxRenderPassVisibility::read_from_bytes(s).ok())
                .ok_or_else(|| {
                    op_fault("ENCODER_BEGIN_RENDER: no whole visibility result buffer record")
                })?;
            // `MTLVisibilityResultTypeReset` and `Accumulate`.
            if v.result_type > 1 {
                return Err(op_fault(format_args!(
                    "ENCODER_BEGIN_RENDER: visibility result type {}, which Metal does not define",
                    v.result_type
                )));
            }
            let buffer = self.require_buffer(v.buffer_obj_id)?;
            facts.visibility_length = Some(buffer.length());
            pass.set_pass(&PassSet {
                visibility_result_buffer: Some(Some(buffer)),
                visibility_result_type: Some(u64::from(v.result_type)),
                ..PassSet::default()
            });
        }

        if desc.depth.texture_obj_id != 0 {
            let d = &desc.depth;
            let tex = self.require_texture(d.texture_obj_id)?;
            let resolve = self.get_texture(d.resolve_texture_obj_id)?;
            let dset = DepthAttachmentSet {
                texture: Some(Some(tex)),
                resolve_texture: resolve.map(Some),
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
                texture: Some(Some(tex)),
                resolve_texture: resolve.map(Some),
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

        Ok((pass, facts))
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
        OpenEncoder::Render(enc, _) => enc.encode(ON, RenderCmd::EndEncoding),
        OpenEncoder::Compute(enc, _) => enc.encode(ON, ComputeCmd::EndEncoding),
        OpenEncoder::Blit(enc) => enc.encode(ON, BlitCmd::EndEncoding),
        OpenEncoder::Video(_) | OpenEncoder::Skipped(_) => Ok(()),
    });
}

/// `done`, the outcome of an encoder op: an op that named an object whose create the host refused
/// is dropped, and the walk goes on ([`refused`]); any other outcome as it is.
fn contain(done: RaiseResult<bool>) -> RaiseResult<bool> {
    match done {
        Err(raise) if refused::is_refused(&raise) => Ok(true),
        done => done,
    }
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

/// The op at the start of `ops`: its tag, its body and its whole length, or `None` where its
/// header is cut short, or its length is under the body's offset, no multiple of 8, or past the
/// stream's end, a malformed op.
fn next_op(ops: &[u8]) -> Option<(u16, &[u8], usize)> {
    let h = read::<RgfxOpHeader>(ops)?;
    let len = h.len as usize;
    if len < RGFX_OP_BODY_OFFSET as usize || !len.is_multiple_of(8) || len > ops.len() {
        return None;
    }
    Some((h.tag, &ops[RGFX_OP_BODY_OFFSET as usize..len], len))
}

/// The most draws [`Draws`] holds before it sends them.
pub(super) const DRAWS_HELD: usize = 32;

/// Draws a render run met one after another, not yet sent ([`GraphicsDecoder::render_ops`]):
/// sent together, as each would be alone ([`RenderEncoder::encode_draws`]), before any other op
/// of the run, before the run ends, and whenever [`DRAWS_HELD`] are held.
struct Draws<'a, P: Platform> {
    held: [MaybeUninit<VertexDraw<'a, P>>; DRAWS_HELD],
    /// How many of `held`, from its start, are draws.
    len: usize,
}

impl<'a, P: Platform> Draws<'a, P> {
    fn new() -> Self {
        Draws {
            held: [const { MaybeUninit::uninit() }; DRAWS_HELD],
            len: 0,
        }
    }

    /// Holds `draw`, after sending those held first where they are as many as it holds.
    #[inline]
    fn push(&mut self, draw: VertexDraw<'a, P>, enc: &P::RenderEncoder) -> RaiseResult<()> {
        if self.len == DRAWS_HELD {
            self.send(enc)?;
        }
        self.held[self.len].write(draw);
        self.len += 1;
        Ok(())
    }

    /// Sends the draws held, which it then holds no more.
    #[inline]
    fn send(&mut self, enc: &P::RenderEncoder) -> RaiseResult<()> {
        if self.len == 0 {
            return Ok(());
        }
        let len = std::mem::take(&mut self.len);
        // SAFETY: the first `len` of `held` were written ([`Self::push`]), and `VertexDraw` is
        // `Copy`, so they need no drop.
        let draws = unsafe { std::slice::from_raw_parts(self.held.as_ptr().cast(), len) };
        enc.encode_draws(ON, draws)
    }
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
        RGFX_OP_RENDER_SET_VIEWPORT, RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE,
        RgfxOpDrawIndexedPrimitives, RgfxOpDrawPrimitives, RgfxOpSetFragmentTexture,
        RgfxOpSetRenderPipelineState, RgfxOpSetScissorRect, RgfxOpSetVertexBuffer,
        RgfxOpSetViewport, RgfxOpSetVisibilityResultMode,
    };
    use reims_graphics::ops::video::{
        RGFX_OP_VIDEO_DECODE_H264, RGFX_OP_VIDEO_DECODE_H265, RgfxVideoOpDecodeH264,
        RgfxVideoOpDecodeH265,
    };
    use reims_graphics::wire::{
        RGFX_CB_OK, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_EVENT_CREATE,
        RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SHARED_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_SET_VALUE,
        RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_IOSURFACE, RGFX_CMD_VIDEO_SESSION_CREATE,
        RGFX_EVT_CB_COMPLETE, RGFX_SURFACE_NO_WRITERS, RGFX_VIDEO_CODEC_H264,
        RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RgfxBufferCreate, RgfxCbComplete,
        RgfxObjectCreate, RgfxQueueCreate, RgfxSharedEventCreate, RgfxSharedEventSetValue,
        RgfxTextureCreate, RgfxTextureDescriptor, RgfxTextureFromIOSurface, RgfxVideoSessionCreate,
    };
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::descriptor::RenderPassDescriptor;
    use reims_vgpu_core::platform::values::AttachmentState;
    use reims_vgpu_core::sync::Mutex;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::Step;
    use reims_vgpu_core::testkit::calllog::Arg;
    use reims_vgpu_core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::super::split::SPLIT_AFTER;
    use super::super::{GraphicsVideo, GraphicsVideoSession, VideoDecode, VideoPlane};
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

    thread_local! {
        /// Whether the walk sends draws together ([`GraphicsDecoder::vertex_draw`]); a test of the
        /// walk an op at a time sets it false.
        pub(super) static GATHERS: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }

    /// A draw's two ops: `SET_VERTEX_BUFFER` of `buffer` at `offset`, index 0, then
    /// `DRAW_PRIMITIVES` of `count` triangles' vertices from `start`.
    fn draw_ops(buffer: u64, offset: u64, start: u64) -> Vec<u8> {
        let mut ops = op(
            RGFX_OP_RENDER_SET_VERTEX_BUFFER,
            RgfxOpSetVertexBuffer {
                buffer_obj_id: buffer,
                offset,
                index: 0,
            }
            .as_bytes(),
        );
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: start,
                vertex_count: 3,
            }
            .as_bytes(),
        ));
        ops
    }

    /// A render pass of `draws`, the ops between `ENCODER_BEGIN_RENDER` and `ENCODER_END`.
    fn pass_of(draws: &[u8]) -> Vec<u8> {
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 0 }.as_bytes(),
        ));
        ops.extend_from_slice(draws);
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops
    }

    /// What a fresh decoder does with a command buffer of `ops`, sending its draws together
    /// (`gathers`) or an op at a time, with `script` set on its platform first: its answer, and
    /// each call it made of the platform from the message on, with the call's arguments.
    fn walked(
        ops: &[u8],
        gathers: bool,
        script: &dyn Fn(&FakePlatform),
    ) -> (String, Vec<(&'static str, Vec<Arg>)>) {
        GATHERS.set(gathers);
        let (p, mut dec) = setup();
        script(&p);
        let device = p.device();
        let mark = p.call_log().mark();
        let answer = dec.decode(&device, &command_buffer_message(100, 1, ops));
        p.world().complete_all(CompleteOn::Caller);
        GATHERS.set(true);
        let calls = p.call_log().since(mark).into_iter();
        (
            format!("{answer:?}"),
            calls.map(|c| (c.sel, c.args)).collect(),
        )
    }

    /// Draws sent together reach the platform as they do an op at a time: the same calls, with
    /// the same arguments, in the same order, and the same answer, the walk stopping at the same
    /// op. Five draws whose third names a buffer the context does not hold fault at that op,
    /// after the first two; so do five whose third `setVertexBuffer:offset:atIndex:` the
    /// platform raises at, after its call; and so do a draw whose `DRAW_PRIMITIVES` is too short,
    /// one whose length runs past the stream, a run of draws past what the walk holds, broken by
    /// another op, and a vertex buffer set with no draw after it
    #[test]
    fn draws_sent_together_reach_the_platform_as_they_do_an_op_at_a_time() {
        const SET: &str = "setVertexBuffer:offset:atIndex:";
        const DRAW: &str = "drawPrimitives:vertexStart:vertexCount:";
        let five = |third: Vec<u8>| {
            let mut draws = Vec::new();
            for k in 0..5 {
                match k {
                    2 => draws.extend_from_slice(&third),
                    _ => draws.extend(draw_ops(2, 16 * k, 3 * k)),
                }
            }
            pass_of(&draws)
        };
        let short = {
            let mut ops = draw_ops(2, 32, 6);
            let draw = ops.len() - size_of::<RgfxOpDrawPrimitives>() - 8;
            ops.truncate(draw);
            ops.extend(op(RGFX_OP_RENDER_DRAW_PRIMITIVES, &[0; 8]));
            ops
        };
        let past_the_stream = {
            let mut third = draw_ops(2, 32, 6);
            // The `DRAW_PRIMITIVES` header's length, after the `SET_VERTEX_BUFFER` op.
            let at = 8 + size_of::<RgfxOpSetVertexBuffer>() + 4;
            third[at..at + 4].copy_from_slice(&4096u32.to_ne_bytes());
            five(third)
        };
        let long_run = {
            let mut draws = Vec::new();
            for k in 0..(2 * DRAWS_HELD as u64 + 7) {
                draws.extend(draw_ops(2 + k % 2, 16 * k, 3 * k));
                if k == 40 {
                    draws.extend(op(
                        RGFX_OP_RENDER_SET_VIEWPORT,
                        RgfxOpSetViewport {
                            width_bits: 64.0f64.to_bits(),
                            height_bits: 64.0f64.to_bits(),
                            zfar_bits: 1.0f64.to_bits(),
                            ..RgfxOpSetViewport::default()
                        }
                        .as_bytes(),
                    ));
                }
            }
            pass_of(&draws)
        };
        /// What a case sets on its platform first.
        type Scripted<'s> = &'s dyn Fn(&FakePlatform);
        let raises_at_third: Scripted<'_> = &|p| {
            p.script().after(
                SET,
                2,
                Step::raise("NSInvalidArgumentException", "a test's"),
            )
        };
        let none: Scripted<'_> = &|_| {};
        let cases: [(&str, Vec<u8>, Scripted<'_>, usize); 6] = [
            (
                "a bad reference at the third",
                five(draw_ops(99, 32, 6)),
                none,
                2,
            ),
            (
                "a raise at the third",
                five(draw_ops(2, 32, 6)),
                raises_at_third,
                2,
            ),
            ("a short third", five(short), none, 2),
            ("a third past the stream", past_the_stream, none, 2),
            ("a long run", long_run, none, 2 * DRAWS_HELD + 7),
            ("the reference ops", render_ops(), none, 1),
        ];
        for (case, ops, script, draws) in cases {
            let together = walked(&ops, true, script);
            let alone = walked(&ops, false, script);
            assert_eq!(together, alone, "{case}");
            let drawn = (together.1.iter()).filter(|(sel, _)| *sel == DRAW).count();
            assert_eq!(drawn, draws, "{case}: {together:?}");
        }
    }

    /// A video session's create, of H.264 decode, as object `obj_id`.
    fn video_session_create(obj_id: u64) -> RgfxVideoSessionCreate {
        RgfxVideoSessionCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VIDEO_SESSION_CREATE,
                size: size_of::<RgfxVideoSessionCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            codec: RGFX_VIDEO_CODEC_H264,
            direction: RGFX_VIDEO_DECODE,
        }
    }

    /// A video session's `OBJECT_DESTROY` lets it go on the video worker: the session ends there,
    /// not on the thread that carries the destroy out.
    #[test]
    fn a_video_sessions_destroy_ends_it_on_the_video_worker() {
        let (p, mut dec) = setup();
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        let create = video_session_create(40);
        assert_eq!(dec.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        dec.video_worker().hold();
        assert_eq!(dec.decode(&device, &destroy(40)).unwrap(), Ok(()));
        assert!(seen.ended_on.lock().is_empty());
        dec.video_worker().release();
        dec.video_worker().flush();
        assert_ended_on_the_video_worker(&seen);
    }

    /// An open video encoder's hold of its session, which a walk takes at `ENCODER_BEGIN_VIDEO`,
    /// keeps the session past its `OBJECT_DESTROY`; the hold's drop, the last, then ends it on the
    /// video worker, not on the walk's thread.
    #[test]
    fn a_walks_hold_of_a_destroyed_session_ends_it_on_the_video_worker() {
        let (p, mut dec) = setup();
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        let create = video_session_create(40);
        assert_eq!(dec.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        let hold = dec.require_video_session(40).unwrap();
        assert_eq!(dec.decode(&device, &destroy(40)).unwrap(), Ok(()));
        dec.video_worker().flush();
        assert!(seen.ended_on.lock().is_empty(), "the walk's hold keeps it");
        drop(hold);
        dec.video_worker().flush();
        assert_ended_on_the_video_worker(&seen);
    }

    /// One session ended, on a thread of the video worker's (`reims-vgpu.graphics-video-<i>`).
    fn assert_ended_on_the_video_worker(seen: &VideoSeen) {
        let ended = seen.ended_on.lock();
        let on_worker = |thread: &Option<String>| {
            thread
                .as_deref()
                .and_then(|name| name.strip_prefix("reims-vgpu.graphics-video-"))
                .is_some_and(|i| i.parse::<usize>().is_ok())
        };
        assert!(
            ended.len() == 1 && on_worker(&ended[0]),
            "the session ended on {ended:?}"
        );
    }

    /// A decode a fake host's video was handed: its session's codec, its access unit (offset,
    /// size), its parameter sets and its planes (offset, bytes per row, width, height).
    type SeenDecode = (u32, u64, u64, Vec<u8>, [u64; 4], [u64; 4]);

    /// What a fake host's video was handed: each decode, whether each of its planes was in a
    /// device cache, and the thread each of its sessions ended on.
    #[derive(Default)]
    struct VideoSeen {
        decodes: Mutex<Vec<SeenDecode>>,
        cached: Mutex<Vec<[bool; 2]>>,
        ended_on: Mutex<Vec<Option<String>>>,
    }

    /// A fake host's video of H.264 and HEVC decode, recording into its [`VideoSeen`]; each of
    /// its sessions' decodes answers the next value of the session's event, from 1.
    struct FakeVideo(Arc<VideoSeen>, <FakePlatform as Platform>::Device);

    /// A session of [`FakeVideo`]'s.
    struct FakeSession {
        codec: u32,
        seen: Arc<VideoSeen>,
        event: <FakePlatform as Platform>::SharedEvent,
        decodes: AtomicUsize,
    }

    impl GraphicsVideo<FakePlatform> for FakeVideo {
        fn caps(&self) -> Vec<reims_graphics::wire::RgfxVideoCodecCaps> {
            Vec::new()
        }
        fn session(
            &self,
            codec: u32,
            direction: u32,
        ) -> Option<Arc<dyn GraphicsVideoSession<FakePlatform>>> {
            let codecs = [RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC];
            if !codecs.contains(&codec) || direction != RGFX_VIDEO_DECODE {
                return None;
            }
            Some(Arc::new(FakeSession {
                codec,
                seen: Arc::clone(&self.0),
                event: self.1.new_shared_event()?,
                decodes: AtomicUsize::new(0),
            }))
        }
    }

    impl Drop for FakeSession {
        fn drop(&mut self) {
            let thread = std::thread::current().name().map(str::to_owned);
            self.seen.ended_on.lock().push(thread);
        }
    }

    impl GraphicsVideoSession<FakePlatform> for FakeSession {
        fn event(&self) -> reims_vgpu_core::platform::event::EventRef<'_, FakePlatform> {
            reims_vgpu_core::platform::event::EventRef::Shared(&self.event)
        }

        fn codec(&self) -> u32 {
            self.codec
        }

        fn decode(&self, d: &VideoDecode<'_, FakePlatform>) -> RaiseResult<u64> {
            let plane =
                |p: &VideoPlane<'_, FakePlatform>| [p.offset, p.bytes_per_row, p.width, p.height];
            self.seen.decodes.lock().push((
                self.codec,
                d.offset,
                d.size,
                d.parameter_sets.to_vec(),
                plane(&d.luma),
                plane(&d.chroma),
            ));
            self.seen
                .cached
                .lock()
                .push([d.luma.cached, d.chroma.cached]);
            Ok(self.decodes.fetch_add(1, Ordering::Relaxed) as u64 + 1)
        }
    }

    /// A texture over IOSurface 0x99's plane `plane`, in buffer 6, at `offset` with rows of 64
    /// bytes, `width` by `height` pixels of `pixel_format`.
    fn surface_plane_texture(
        obj_id: u64,
        plane: u32,
        pixel_format: u32,
        offset: u64,
        width: u32,
        height: u32,
    ) -> RgfxTextureFromIOSurface {
        RgfxTextureFromIOSurface {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_FROM_IOSURFACE,
                size: size_of::<RgfxTextureFromIOSurface>() as u32,
                seqno: 0,
            },
            obj_id,
            resource_id: obj_id,
            buffer_obj_id: 6,
            offset,
            bytes_per_row: 64,
            iosurface_id: 0x99,
            plane,
            desc: RgfxTextureDescriptor {
                pixel_format,
                texture_type: 2,
                width,
                height,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 3,
                ..RgfxTextureDescriptor::default()
            },
            writers_offset: RGFX_SURFACE_NO_WRITERS,
        }
    }

    /// A video encoder hands its session each decode with the objects it names resolved: the
    /// access unit's place in its buffer, the parameter sets that follow the op's body, and where
    /// each plane's texture lies in the surface's buffer; the command buffer then waits for the
    /// session's event to reach the value the decode answers, before its commit. A session the
    /// host has no video for is refused.
    #[test]
    fn a_video_encoder_hands_its_session_each_decode() {
        let (p, mut dec) = setup();
        let device = p.device();
        let session = RgfxVideoSessionCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VIDEO_SESSION_CREATE,
                size: size_of::<RgfxVideoSessionCreate>() as u32,
                seqno: 0,
            },
            obj_id: 40,
            codec: RGFX_VIDEO_CODEC_H264,
            direction: RGFX_VIDEO_DECODE,
        };
        assert_eq!(
            dec.decode(&device, session.as_bytes()).unwrap(),
            Err(DecodeError::Refused)
        );
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        assert_eq!(dec.decode(&device, session.as_bytes()).unwrap(), Ok(()));

        // The surface's buffer, and a texture over each of its two planes.
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        for t in [
            surface_plane_texture(7, 0, 10, 32, 64, 16),
            surface_plane_texture(8, 1, 30, 32 + 64 * 16, 32, 8),
        ] {
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }

        let sets = [0u8, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68];
        let mut decode = RgfxVideoOpDecodeH264 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 16,
            bitstream_size: 100,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            parameter_sets_size: sets.len() as u32,
            ..RgfxVideoOpDecodeH264::default()
        }
        .as_bytes()
        .to_vec();
        decode.extend_from_slice(&sets);
        decode.resize(decode.len().next_multiple_of(8), 0);
        let pass = RgfxOpVideoPass { session_obj_id: 40 };
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes());
        ops.extend(op(RGFX_OP_VIDEO_DECODE_H264, &decode));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        assert_eq!(
            dec.decode(&device, &command_buffer_message(100, 1, &ops))
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            *seen.decodes.lock(),
            [(
                RGFX_VIDEO_CODEC_H264,
                16,
                100,
                sets.to_vec(),
                [32, 64, 64, 16],
                [32 + 64 * 16, 64, 32, 8]
            )]
        );
        assert_eq!(*seen.cached.lock(), [[false, false]], "a buffer of its own");
        let calls = p.call_log().calls();
        let sels: Vec<&str> = calls
            .iter()
            .map(|c| c.sel)
            .filter(|&sel| sel == "encodeWaitForEvent:value:" || sel == "commit")
            .collect();
        assert_eq!(sels, ["encodeWaitForEvent:value:", "commit"]);
        let wait = calls
            .iter()
            .find(|c| c.sel == "encodeWaitForEvent:value:")
            .unwrap();
        assert_eq!(wait.args.get(1), Some(&Arg::U64(1)));
    }

    /// A video decode of a command buffer that waits for a buffer of another queue starts once
    /// that buffer has finished: the session's work runs on a queue of its own, which the command
    /// buffer's GPU wait does not hold, so the host waits on the CPU first. A decode whose named
    /// buffer has finished starts at once.
    #[test]
    fn a_video_decode_waits_for_the_buffers_its_command_buffer_runs_after() {
        use reims_graphics::ops::{RGFX_OP_SURFACE_ORDER, RgfxSurfaceOrderWait};

        let (p, mut dec) = setup();
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        let other = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 9,
            ..RgfxQueueCreate::default()
        };
        let session = RgfxVideoSessionCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VIDEO_SESSION_CREATE,
                size: size_of::<RgfxVideoSessionCreate>() as u32,
                seqno: 0,
            },
            obj_id: 40,
            codec: RGFX_VIDEO_CODEC_H264,
            direction: RGFX_VIDEO_DECODE,
        };
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        for m in [other.as_bytes(), session.as_bytes(), buffer.as_bytes()] {
            assert_eq!(dec.decode(&device, m).unwrap(), Ok(()));
        }
        for t in [
            surface_plane_texture(7, 0, 10, 32, 64, 16),
            surface_plane_texture(8, 1, 30, 32 + 64 * 16, 32, 8),
        ] {
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        let order = |waits: &[(u64, u64)]| {
            let mut body = Vec::new();
            for &(queue_obj_id, seqno) in waits {
                let w = RgfxSurfaceOrderWait {
                    queue_obj_id,
                    seqno,
                };
                body.extend_from_slice(w.as_bytes());
            }
            op(RGFX_OP_SURFACE_ORDER, &body)
        };

        // Queue 9's buffer at seqno 5, which reads the surface, committed and not finished.
        let mut read = command_buffer_message(200, 5, &order(&[]));
        let queue = std::mem::offset_of!(RgfxCommandBuffer, queue_obj_id);
        read[queue..queue + 8].copy_from_slice(&9u64.to_le_bytes());
        assert_eq!(dec.decode(&device, &read).unwrap(), Ok(()));

        // Queue 1's decode into the surface names it.
        let decode = RgfxVideoOpDecodeH264 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 16,
            bitstream_size: 100,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            ..RgfxVideoOpDecodeH264::default()
        };
        let pass = RgfxOpVideoPass { session_obj_id: 40 };
        let mut ops = order(&[(9, 5)]);
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes()));
        ops.extend(op(RGFX_OP_VIDEO_DECODE_H264, decode.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let message = command_buffer_message(100, 1, &ops);
        let mut walker = dec.sender(None, super::super::Sinks::default());
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message).unwrap());
            std::thread::sleep(std::time::Duration::from_millis(100));
            assert!(
                seen.decodes.lock().is_empty(),
                "the decode waits for the read"
            );
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(seen.decodes.lock().len(), 1);

        // Named again once finished: no wait.
        let message = command_buffer_message(101, 2, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(seen.decodes.lock().len(), 2);
    }

    /// An HEVC session takes `DECODE_H265` ops as an H.264 one takes `DECODE_H264`, and a decode
    /// of the other codec on a session faults its command buffer, handing the session nothing.
    #[test]
    fn a_decode_goes_to_a_session_of_its_codec() {
        let (p, mut dec) = setup();
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        let session = |obj_id, codec| RgfxVideoSessionCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VIDEO_SESSION_CREATE,
                size: size_of::<RgfxVideoSessionCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            codec,
            direction: RGFX_VIDEO_DECODE,
        };
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            length: 65536,
            ..RgfxBufferCreate::default()
        };
        for m in [
            session(40, RGFX_VIDEO_CODEC_H264).as_bytes(),
            session(41, RGFX_VIDEO_CODEC_HEVC).as_bytes(),
            buffer.as_bytes(),
        ] {
            assert_eq!(dec.decode(&device, m).unwrap(), Ok(()));
        }
        // P010's planes: rows of 64 pixels of two bytes, then pairs of them.
        for t in [
            surface_plane_texture(7, 0, 20, 0, 32, 16),
            surface_plane_texture(8, 1, 60, 64 * 16, 16, 8),
        ] {
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        let sets = [
            0u8, 0, 0, 1, 0x40, 1, 0, 0, 0, 1, 0x42, 1, 0, 0, 0, 1, 0x44, 1,
        ];
        let mut hevc = RgfxVideoOpDecodeH265 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 0,
            bitstream_size: 64,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            parameter_sets_size: sets.len() as u32,
            ..RgfxVideoOpDecodeH265::default()
        }
        .as_bytes()
        .to_vec();
        hevc.extend_from_slice(&sets);
        hevc.resize(hevc.len().next_multiple_of(8), 0);
        let h264 = RgfxVideoOpDecodeH264 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 0,
            bitstream_size: 64,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            ..RgfxVideoOpDecodeH264::default()
        };
        let pass = |session_obj_id, tag, body: &[u8]| {
            let pass = RgfxOpVideoPass { session_obj_id };
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes());
            ops.extend(op(tag, body));
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
            ops
        };
        let ops = pass(41, RGFX_OP_VIDEO_DECODE_H265, &hevc);
        assert_eq!(
            dec.decode(&device, &command_buffer_message(100, 1, &ops))
                .unwrap(),
            Ok(())
        );
        let ops = pass(41, RGFX_OP_VIDEO_DECODE_H264, h264.as_bytes());
        assert!(
            dec.decode(&device, &command_buffer_message(100, 2, &ops))
                .is_err()
        );
        let ops = pass(40, RGFX_OP_VIDEO_DECODE_H265, &hevc);
        assert!(
            dec.decode(&device, &command_buffer_message(100, 3, &ops))
                .is_err()
        );
        assert_eq!(
            *seen.decodes.lock(),
            [(
                RGFX_VIDEO_CODEC_HEVC,
                0,
                64,
                sets.to_vec(),
                [0, 64, 32, 16],
                [64 * 16, 64, 16, 8]
            )]
        );
    }

    /// A rig whose block at `BLOCK` is a cached IOSurface of two pages, with command queue 1, the
    /// bitstream buffer 2, the H.264 session 40 of a fake host's video, buffer 6 over the block
    /// and textures 7 and 8 over its planes; and what the video is handed.
    fn cached_video() -> (super::super::surface_pages::tests::Rig, Arc<VideoSeen>) {
        use super::super::surface_pages::tests::{BLOCK, rig};
        use reims_graphics::wire::RGFX_STATUS_OK;

        let mut r = rig(8);
        let page = r.page;
        assert_eq!(
            r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
            RGFX_STATUS_OK
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
        let bitstream = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 2,
            length: 4096,
            ..RgfxBufferCreate::default()
        };
        let session = RgfxVideoSessionCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_VIDEO_SESSION_CREATE,
                size: size_of::<RgfxVideoSessionCreate>() as u32,
                seqno: 0,
            },
            obj_id: 40,
            codec: RGFX_VIDEO_CODEC_H264,
            direction: RGFX_VIDEO_DECODE,
        };
        let seen = Arc::new(VideoSeen::default());
        r.dec
            .set_video(Arc::new(FakeVideo(Arc::clone(&seen), r.p.device())));
        for m in [queue.as_bytes(), bitstream.as_bytes(), session.as_bytes()] {
            assert_eq!(r.dec.decode(&r.device, m).unwrap(), Ok(()));
        }
        assert_eq!(r.buffer(6, BLOCK, 2 * page), Ok(()));
        r.dec.hold_resource_ids(0, 0);
        for t in [
            surface_plane_texture(7, 0, 10, 32, 64, 16),
            surface_plane_texture(8, 1, 30, 32 + 64 * 16, 32, 8),
        ] {
            assert_eq!(r.dec.decode(&r.device, t.as_bytes()).unwrap(), Ok(()));
        }
        (r, seen)
    }

    /// The `COMMAND_BUFFER` `cb_obj_id` at `seqno` on queue 1 of [`cached_video`]'s rig, which
    /// writes its block at generation 0 and, where it `decodes`, decodes into its planes on
    /// session 40.
    fn writes_cached_video(cb_obj_id: u64, seqno: u64, decodes: bool) -> Vec<u8> {
        use super::super::surface_pages::tests::BLOCK;
        use reims_graphics::ops::{RGFX_OP_SURFACE_USE, RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse};

        let surface_use = RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen: 0,
            flags: RGFX_SURFACE_USE_WRITES,
            reserved_14: 0,
        };
        let mut ops = op(RGFX_OP_SURFACE_USE, surface_use.as_bytes());
        if decodes {
            let decode = RgfxVideoOpDecodeH264 {
                bitstream_buffer_obj_id: 2,
                bitstream_offset: 0,
                bitstream_size: 100,
                luma_texture_obj_id: 7,
                chroma_texture_obj_id: 8,
                ..RgfxVideoOpDecodeH264::default()
            };
            let pass = RgfxOpVideoPass { session_obj_id: 40 };
            ops.extend(op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes()));
            ops.extend(op(RGFX_OP_VIDEO_DECODE_H264, decode.as_bytes()));
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        }
        command_buffer_message(cb_obj_id, seqno, &ops)
    }

    /// A video output whose surface's block has a device cache is handed to the session as in
    /// the cache, which each buffer over the block is, once the cache's fill is done.
    #[test]
    fn a_video_output_in_a_cached_block_is_in_its_cache() {
        let (mut r, seen) = cached_video();
        // The first use fills the cache; the decode's buffer, of the same generation, fills
        // nothing, and its decode finds the fill done.
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, false))
                .unwrap(),
            Ok(())
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(101, 2, true))
                .unwrap(),
            Ok(())
        );
        assert_eq!(*seen.cached.lock(), [[true, true]]);
        assert_eq!(r.dec.caches().counts().fills, 1);
    }

    /// A decoder's output that its client locks to read each picture is copied into its pages
    /// behind the next decode into it, by a command buffer of the host's committed on the
    /// decode's own queue right after it, and the lock's download that follows copies nothing
    /// more, answered once that copy lands. A command buffer that writes the surface and holds no
    /// video encoder has its copy on the host's read-back queue instead, so its own queue is not
    /// held behind it, and its lock too copies nothing more
    #[test]
    fn a_decode_read_back_is_copied_out_behind_the_next_decode() {
        use super::super::surface_pages::tests::{BLOCK, Rig};
        use reims_graphics::ops::{RGFX_OP_SURFACE_USE, RgfxOpSurfaceUse};
        use std::sync::atomic::AtomicBool;

        const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
        /// The source of each copy since `mark`.
        fn copied_since(r: &Rig, mark: u64) -> Vec<u64> {
            (r.p.call_log().since(mark).iter())
                .filter(|c| c.sel == COPY)
                .map(|c| c.args[0].as_obj().map_or(0, |o| o.0))
                .collect()
        }
        /// A `SURFACE_DOWNLOAD` of the block, and whether it was answered.
        fn download(r: &Rig) -> Arc<AtomicBool> {
            let done = Arc::new(AtomicBool::new(false));
            let set = Arc::clone(&done);
            r.dec.surface_download(
                &r.device,
                BLOCK,
                Box::new(move || set.store(true, Ordering::SeqCst)),
                super::super::no_raise(),
            );
            done
        }
        /// Carries out `message`.
        fn run(r: &mut Rig, message: Vec<u8>) {
            assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        }
        let (mut r, _seen) = cached_video();
        let cache = Arg::obj(r.dec.caches().buffer(BLOCK).unwrap())
            .as_obj()
            .unwrap()
            .0;

        // A read fills the cache first, so the decodes find it filled.
        let read = RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen: 0,
            flags: 0,
            reserved_14: 0,
        };
        let read = op(RGFX_OP_SURFACE_USE, read.as_bytes());
        run(&mut r, command_buffer_message(99, 1, &read));
        r.p.world().complete_all(CompleteOn::Caller);
        run(&mut r, writes_cached_video(100, 2, true));
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = download(&r);
        assert_eq!(
            copied_since(&r, mark),
            [cache],
            "the lock downloads the first"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        let mark = r.p.call_log().mark();
        run(&mut r, writes_cached_video(101, 3, true));
        let calls = r.p.call_log().since(mark);
        let commits: Vec<usize> = (calls.iter().enumerate())
            .filter(|(_, c)| c.sel == "commit")
            .map(|(i, _)| i)
            .collect();
        let copy = calls.iter().position(|c| c.sel == COPY);
        assert_eq!(
            copied_since(&r, mark),
            [cache],
            "copied out behind the decode"
        );
        assert!(commits.len() == 2 && copy.is_some_and(|at| commits[0] < at && at < commits[1]));
        let mark = r.p.call_log().mark();
        let done = download(&r);
        assert!(copied_since(&r, mark).is_empty(), "nothing more");
        assert!(!done.load(Ordering::SeqCst), "answered as the copy lands");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));

        let mark = r.p.call_log().mark();
        run(&mut r, writes_cached_video(102, 4, false));
        assert_eq!(
            copied_since(&r, mark),
            [cache],
            "copied out on the read-back queue"
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = download(&r);
        assert!(
            copied_since(&r, mark).is_empty(),
            "its lock copies nothing more"
        );
        assert!(done.load(Ordering::SeqCst));
        let counts = r.dec.caches().counts();
        assert_eq!((counts.early, counts.early_aside), (2, 1));
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
            "addCompletionSeenHandler",
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

    /// With a scheduled sink, a command buffer gets a submission handler before its completion
    /// handler and its commit, and no scheduled or completed handler; once the platform takes its
    /// work, its `CB_SCHEDULED` goes to the guest at its seqno, ahead of its completion record.
    /// With none, it gets no submission handler.
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
        let handlers = [
            "addScheduledHandler:",
            "addCompletedHandler:",
            "addSubmissionHandler",
            "addCompletionSeenHandler",
            "commit",
        ];
        let got: Vec<&str> = p
            .call_log()
            .sels_since(mark)
            .into_iter()
            .filter(|s| handlers.contains(s))
            .collect();
        assert_eq!(
            got,
            ["addSubmissionHandler", "addCompletionSeenHandler", "commit"]
        );
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

        // No sink: no submission handler.
        let (p, mut dec) = setup();
        let device = p.device();
        let mark = p.call_log().mark();
        let message = command_buffer_message(131, 62, &render_ops());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let sels = p.call_log().sels_since(mark);
        assert!(!sels.contains(&"addSubmissionHandler") && !sels.contains(&"addScheduledHandler:"));
    }

    /// A buffer the platform fails before it takes its work runs no submission handler: its
    /// completion handler posts its `CB_SUBMITTED` and `CB_SCHEDULED` ahead of its `CB_COMPLETE`.
    /// One whose submission handler ran has them posted there, once.
    #[test]
    fn a_buffer_never_submitted_is_scheduled_ahead_of_its_completion() {
        use reims_vgpu_core::platform::types::CommandBufferStatus;
        let (_p, dec, posted) = posting(true);
        for ran in [false, true] {
            posted.lock().clear();
            let submitted = dec.submitted(150, 80, true);
            let scheduled = dec.scheduled(150, 80);
            let submission = submission_handler(submitted.clone(), None, scheduled.clone());
            let handler = dec.completed_handler(
                150,
                80,
                EventOps::default(),
                None,
                false,
                (submitted, None, scheduled),
            );
            let submission = submission.unwrap();
            if ran {
                submission();
                assert_eq!(posted.lock().len(), 2, "posted at the submission");
            } else {
                drop(submission);
            }
            handler(CompletedInfo {
                status: CommandBufferStatus::ERROR,
                error: None,
                gpu_times_ns: None,
            });
            assert_eq!(
                *posted.lock(),
                [
                    (RGFX_EVT_CB_SUBMITTED, 150, 80),
                    (RGFX_EVT_CB_SCHEDULED, 150, 80),
                    (RGFX_EVT_CB_COMPLETE, 150, 80)
                ],
                "submission handler ran: {ran}"
            );
        }
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
                .contains(&"addSubmissionHandler")
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
        assert!(dec.stand_in_for_failed(&device, &message).unwrap());
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

    /// Answers `message`, of context 0's, on a doorbell's thread through `dec`'s decoders, each
    /// record it posts pushed onto `posted`, where `posts`, as the queue's ring takes it; whether
    /// it was answered.
    fn answer(
        dec: &GraphicsDecoder<FakePlatform>,
        message: &[u8],
        posted: &Posted,
        posts: bool,
    ) -> bool {
        let posted = Arc::clone(posted);
        dec.decoders().answer_at_doorbell(0, message, move |m| {
            if posts {
                (posted.lock()).push((m.header.r#type, m.cb_obj_id, m.header.seqno));
            }
            posts
        })
    }

    /// What the answer on a doorbell's thread kept on queue 1, let go of; whether there was any.
    fn let_go_of_answer(dec: &GraphicsDecoder<FakePlatform>) -> bool {
        let counts = dec.queue_counts(1).unwrap();
        let answered = counts.answered.lock().take();
        answered.is_some()
    }

    /// A buffer is answered on a doorbell's thread only where it gets its `CB_SUBMITTED` at its
    /// take and need wait for nothing: not without the ask, not where the ask is not its last op,
    /// not where it waits for an event or a
    /// buffer before it holds its queue, not on a queue its context does not hold, not while its
    /// queue holds [`TAKE_IN_FLIGHT_MAX`] command buffers made and not completed or an answered
    /// buffer not yet begun. Where it is, its `CB_SUBMITTED` is posted there, and what the answer
    /// took is kept on its queue; where it is not, nothing is.
    #[test]
    fn a_buffer_is_answered_at_the_doorbell_only_where_it_need_wait_for_nothing() {
        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let answered = |dec: &GraphicsDecoder<FakePlatform>, message: &[u8]| {
            let before = posted.lock().len();
            let yes = answer(dec, message, &posted, true);
            assert_eq!(posted.lock().len(), before + usize::from(yes));
            assert_eq!(
                let_go_of_answer(dec),
                yes,
                "kept on its queue where answered"
            );
            yes
        };
        assert!(answered(&dec, &asking(170, 100)));
        assert_eq!(
            posted.lock().last(),
            Some(&(RGFX_EVT_CB_SUBMITTED, 170, 100))
        );
        let plain = command_buffer_message(171, 101, &render_ops());
        assert!(!answered(&dec, &plain), "no ask");
        // An ask that is not the stream's last op is left to the take, unread; a last op whose
        // body ends in an ask's bytes is read, and tells it apart.
        let mut first = op(RGFX_OP_SUBMITTED, &[]);
        first.extend(render_ops());
        let first = command_buffer_message(178, 108, &first);
        assert!(!answered(&dec, &first), "its ask is not its last op");
        let lookalike = RgfxOpHeader {
            tag: RGFX_OP_SUBMITTED,
            flags: 0,
            len: RGFX_OP_BODY_OFFSET,
        };
        let mut lookalike_ops = render_ops();
        lookalike_ops.extend(op(0x0fff, lookalike.as_bytes()));
        let lookalike = command_buffer_message(179, 109, &lookalike_ops);
        assert!(ends_in_ask(&lookalike_ops), "the bytes read as an ask");
        assert!(
            !answered(&dec, &lookalike),
            "an op's body that ends in an ask's bytes"
        );
        let mut waits = event_op(RGFX_OP_WAIT_EVENT, 7, 1);
        waits.extend(op(RGFX_OP_SUBMITTED, &[]));
        let waits = command_buffer_message(172, 102, &waits);
        assert!(!answered(&dec, &waits), "waits for an event");
        // A buffer of queue 1 that waits, not yet scheduled, holds the queue.
        let held = Arc::clone(&dec.queue_counts(1).unwrap().held);
        held.fetch_add(1, Ordering::AcqRel);
        assert!(!answered(&dec, &asking(173, 103)), "behind a held buffer");
        held.fetch_sub(1, Ordering::AcqRel);
        assert!(answered(&dec, &asking(173, 103)));
        let other = dec
            .decoders()
            .answer_at_doorbell(9, &asking(174, 104), |_| true);
        assert!(!other, "another context's queue");
        for cb in 0..TAKE_IN_FLIGHT_MAX as u64 {
            let message = command_buffer_message(200 + cb, 200 + cb, &render_ops());
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        }
        assert!(!answered(&dec, &asking(175, 105)), "its queue is full");
        p.world().complete_all(CompleteOn::Caller);
        posted.lock().clear();
        assert!(answer(&dec, &asking(175, 105), &posted, true));
        assert!(
            !answer(&dec, &asking(176, 106), &posted, true),
            "an answered buffer not yet begun"
        );
        assert!(let_go_of_answer(&dec));
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SUBMITTED, 175, 105)]);
    }

    /// A buffer that uses a block whose `SURFACE_PAGES` the surface worker is still carrying out
    /// is not answered on a doorbell's thread, with nothing taken, while another block's does not
    /// hold it; once its block is settled it is
    #[test]
    fn a_buffer_using_a_block_still_being_made_is_not_answered_at_the_doorbell() {
        let (p, mut dec, posted) = posting(false);
        create_events(&p, &mut dec, 0);
        const BLOCK: u64 = 0x5_0000;
        let used = RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen: 0,
            flags: 0,
            reserved_14: 0,
        };
        let mut ops = op(RGFX_OP_SURFACE_USE, used.as_bytes());
        ops.extend(render_ops());
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        let message = command_buffer_message(180, 110, &ops);
        let settle = dec.begin_surface_pages(BLOCK);
        assert!(
            !answer(&dec, &message, &posted, true),
            "its block is unsettled"
        );
        assert!(!let_go_of_answer(&dec), "nothing taken");
        drop(settle);
        let other = dec.begin_surface_pages(BLOCK + 0x1_0000);
        assert!(answer(&dec, &message, &posted, true), "another block's");
        assert!(let_go_of_answer(&dec));
        drop(other);
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SUBMITTED, 180, 110)]);
    }

    /// A buffer answered on a doorbell's thread calls nothing of the platform's there, and its
    /// begin takes what the answer took: it makes the calls a decode makes, and gets its
    /// `CB_SUBMITTED` once, the answer's, or its begin's where the answer found the ring busy.
    #[test]
    fn a_buffer_answered_at_the_doorbell_is_carried_out_as_one_decoded() {
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &asking(180, 110)).unwrap(), Ok(()));
        let decoded = p.call_log().sels_since(mark);
        p.world().complete_all(CompleteOn::Caller);
        posted.lock().clear();

        for (cb, seqno, posts) in [(181, 111, true), (182, 112, false)] {
            let message = asking(cb, seqno);
            let mark = p.call_log().mark();
            assert!(answer(&dec, &message, &posted, posts));
            assert!(p.call_log().sels_since(mark).is_empty(), "a platform call");
            let begun = dec
                .begin_taken(&device, &message, Some(1))
                .unwrap()
                .unwrap();
            assert!(!let_go_of_answer(&dec), "the begin took the answer");
            assert_eq!(
                dec.finish_command_buffer(&device, &message, begun).unwrap(),
                Ok(())
            );
            assert_eq!(p.call_log().sels_since(mark), decoded);
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                *posted.lock(),
                [
                    (RGFX_EVT_CB_SUBMITTED, cb, seqno),
                    (RGFX_EVT_CB_SCHEDULED, cb, seqno),
                    (RGFX_EVT_CB_COMPLETE, cb, seqno)
                ],
                "the answer posted it: {posts}"
            );
            posted.lock().clear();
        }
    }

    /// What an answer on a doorbell's thread took is let go of where the buffer is never begun
    /// after it: at the decoder's reset, which closes its context, its ticket given back, so a
    /// buffer begun after the reset is not held up by it; and, where the answer outlived a reset
    /// of the device, at the buffer's begin, which takes it not.
    #[test]
    fn a_reset_between_an_answer_and_its_begin_lets_the_answer_go() {
        let (p, mut dec, posted) = posting(false);
        let sequencer = Arc::clone(&dec.shared.sequencer);
        assert!(answer(&dec, &asking(190, 120), &posted, true));
        assert_eq!(sequencer.out(), 1, "the answer's ticket");
        dec.end_message();
        dec.reset(&p.device());
        assert_eq!(sequencer.out(), 0, "given back at the reset");

        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        let sequencer = Arc::clone(&dec.shared.sequencer);
        assert!(answer(&dec, &asking(191, 121), &posted, true));
        dec.shared.closed.forget_all();
        let begun = dec
            .begin_taken(&device, &asking(191, 121), Some(1))
            .unwrap()
            .unwrap();
        assert!(!let_go_of_answer(&dec), "taken off its queue");
        assert_eq!(
            sequencer.out(),
            1,
            "the begin's own ticket, not the answer's"
        );
        assert_eq!(
            dec.finish_command_buffer(&device, &asking(191, 121), begun)
                .unwrap(),
            Ok(())
        );
        assert_eq!(sequencer.out(), 0);
    }

    /// A buffer answered on a doorbell's thread whose context closes before its begin gives its
    /// ticket in the commit order back as the close takes the context's tables out
    /// ([`GraphicsDecoder::close_tables`]), before the close worker lets go of its objects
    /// (`Closed::let_go`): a commit marked after the ticket that is ordered after every
    /// context's, one of no context, waits for nothing of the closed context's.
    #[test]
    fn a_close_gives_an_unbegun_answer_s_ticket_back_as_it_takes_the_tables() {
        let (p, mut dec, posted) = posting(false);
        let sequencer = Arc::clone(&dec.shared.sequencer);
        assert!(answer(&dec, &asking(192, 122), &posted, true));
        assert_eq!(sequencer.out(), 1, "the answer's ticket");
        let after = sequencer.mark();
        dec.end_message();
        let (dropped, closed) = dec.close_tables(&p.device(), 0);
        assert!(dropped > 0, "the context's objects, its queue among them");
        assert_eq!(
            sequencer.out(),
            0,
            "given back before its objects are let go of"
        );
        sequencer.wait_for(None, after);
        closed.let_go();
        assert_eq!(sequencer.out(), 0);
    }

    /// A command buffer begun after another's answer on a doorbell's thread, on another queue, is
    /// committed after the answered buffer, which is begun only after it: the answer's ticket in
    /// the commit order holds it, as the ticket of one answered at its begin would.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[test]
    fn a_buffer_begun_after_an_answer_is_committed_after_the_answered_buffer() {
        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        let sinks = dec.sinks.clone();
        let mut first = dec.sender(Some(0), sinks.clone());
        let mut second = dec.sender(Some(0), sinks);
        let asked = asking(100, 1);
        assert!(answer(&first, &asked, &posted, true));
        let later = command_buffer_on(6, 200, &render_ops());
        let commits = || {
            p.call_log()
                .calls_to("commit")
                .into_iter()
                .filter_map(|c| c.thread)
                .filter(|t| t == "first" || t == "second")
                .collect::<Vec<_>>()
        };
        std::thread::scope(|scope| {
            let b = std::thread::Builder::new()
                .name("second".into())
                .spawn_scoped(scope, || second.decode(&p.device(), &later).unwrap())
                .unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(commits().is_empty(), "committed before the answered buffer");
            let a = std::thread::Builder::new()
                .name("first".into())
                .spawn_scoped(scope, || {
                    let device = p.device();
                    let begun = first.begin_taken(&device, &asked, Some(1)).unwrap();
                    first.finish_command_buffer(&device, &asked, begun.unwrap())
                })
                .unwrap();
            assert_eq!(a.join().unwrap().unwrap(), Ok(()));
            assert_eq!(b.join().unwrap(), Ok(()));
        });
        assert_eq!(commits(), ["first", "second"]);
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SUBMITTED, 100, 1)]);
        drop(dec);
    }

    /// A buffer answered on a doorbell's thread whose command buffer the platform then refuses at
    /// its begin fails in its turn: its stand-in commits after the buffer begun before it on its
    /// queue, and its failure comes after that buffer's completion.
    #[test]
    fn a_refused_make_after_an_answer_fails_in_turn() {
        use reims_vgpu_core::testkit::script::Step;

        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        let failed = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&failed);
        dec.set_completion_sink(Arc::new(move |m| {
            seen.lock().push((m.cb_obj_id, m.status));
        }));
        let first = command_buffer_message(193, 123, &render_ops());
        let begun = dec.begin_taken(&device, &first, Some(1)).unwrap().unwrap();
        let mut later = dec.decoders().sender(None, dec.sinks.clone());
        assert!(answer(&later, &asking(194, 124), &posted, true));
        p.script().push("commandBuffer", Step::Nil);
        let refused = later
            .begin_taken(&device, &asking(194, 124), Some(1))
            .unwrap();
        assert!(matches!(refused, Err(DecodeError::Refused)));
        let stand_in = {
            let p = Arc::clone(&p);
            move || {
                let stood_in = later.stand_in_for_failed(&p.device(), &asking(194, 124));
                // Its ticket given back after its stand-in's commit, as the lane's job lets go
                // of its decoder.
                drop(later);
                stood_in.map_err(|_| ())
            }
        };
        std::thread::scope(|scope| {
            let stand_in = scope.spawn(stand_in);
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!stand_in.is_finished(), "stood in before its turn");
            assert_eq!(
                dec.finish_command_buffer(&device, &first, begun).unwrap(),
                Ok(())
            );
            dec.end_message();
            assert_eq!(stand_in.join().unwrap(), Ok(true));
        });
        p.world().complete_all(CompleteOn::Caller);
        let failed = failed.lock();
        assert_eq!(failed.iter().map(|f| f.0).collect::<Vec<_>>(), [193, 194]);
        assert_ne!(failed[1].1, RGFX_CB_OK, "the answered buffer failed");
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_SUBMITTED, 194, 124)]);
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
        let Some(ObjHandle::Buffer(out)) = dec.object(11) else {
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

    /// Through the decoder on `P`: a buffer fills a buffer, signals shared event 7 to 1, waits
    /// for 7 to reach 2, and fills it again, as a guest whose CPU waits for the signal before it
    /// sets 2 encodes it. Whether the value 1 reached the guest within 5 s of the commit, before
    /// the buffer completed; then the CPU sets 2, whatever came: whether the buffer completed
    /// within 10 s of that, and whether the value went ahead of its completion, once.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_cpu_wait_for_a_mid_buffer_signal<P: crate::platform::ApiPlatform>() -> (bool, bool, bool) {
        use std::time::{Duration, Instant};

        use reims_graphics::ops::blit::{RGFX_OP_BLIT_FILL_BUFFER, RgfxBlitOpFillBuffer};
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let posted = Arc::new(Mutex::new(Vec::<(u32, u64, u64)>::new()));
        let sink = Arc::clone(&posted);
        dec.set_signal_sink(Arc::new(move |m| {
            sink.lock().push((m.header.r#type, m.obj_id, m.value));
        }));
        let sink = Arc::clone(&posted);
        dec.set_completion_sink(Arc::new(move |m| {
            sink.lock().push((m.header.r#type, m.cb_obj_id, 0));
        }));
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
        let buffer = RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id: 10,
            length: LEN,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
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

        let fill = |value: u8| {
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
            let fill = RgfxBlitOpFillBuffer {
                buffer_obj_id: 10,
                range_location: 0,
                range_length: LEN,
                value,
                ..RgfxBlitOpFillBuffer::default()
            };
            ops.extend(op(RGFX_OP_BLIT_FILL_BUFFER, fill.as_bytes()));
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
            ops
        };
        let mut ops = fill(0x11);
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 1));
        ops.extend(event_op(RGFX_OP_WAIT_EVENT, 7, 2));
        ops.extend(fill(0x22));
        let cb = command_buffer_on(1, 20, &ops);
        assert_eq!(dec.decode(&device, &cb).unwrap(), Ok(()));
        let signalled = (RGFX_EVT_EVENT_SIGNALLED, 7, 1);
        let completed = (RGFX_EVT_CB_COMPLETE, 20, 0);
        let start = Instant::now();
        while !posted.lock().contains(&signalled) && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let early = {
            let posted = posted.lock();
            posted.contains(&signalled) && !posted.contains(&completed)
        };

        let set = RgfxSharedEventSetValue {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_SET_VALUE,
                size: size_of::<RgfxSharedEventSetValue>() as u32,
                seqno: 0,
            },
            obj_id: 7,
            value: 2,
        };
        assert_eq!(dec.decode(&device, set.as_bytes()).unwrap(), Ok(()));
        let start = Instant::now();
        while !posted.lock().contains(&completed) && start.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let posted = posted.lock().clone();
        let finished = posted.contains(&completed);
        (early, finished, posted == [signalled, completed])
    }

    /// On Metal, a guest thread that waits on the CPU for a signal from the middle of a command
    /// buffer, before it sets the value the buffer waits for next, gets the signal while the
    /// buffer runs, as Metal's CPU does (docs/graphics-kext/dgpu-oracle.md, Q18), and the buffer
    /// then completes: the value goes as the host's event reaches it, once, ahead of the buffer's
    /// completion.
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_cpu_wait_for_a_mid_buffer_signal_is_released_before_the_buffer_ends() {
        let (early, finished, once) =
            a_cpu_wait_for_a_mid_buffer_signal::<reims_vgpu_metal::MetalPlatform>();
        assert!(early, "the signal came only at the buffer's end, or never");
        assert!(finished && once);
    }

    /// The same on the Vulkan platform
    /// (`on_metal_a_cpu_wait_for_a_mid_buffer_signal_is_released_before_the_buffer_ends`).
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_cpu_wait_for_a_mid_buffer_signal_is_released_before_the_buffer_ends() {
        let (early, finished, once) =
            a_cpu_wait_for_a_mid_buffer_signal::<reims_vgpu_vulkan::VulkanPlatform>();
        assert!(early, "the signal came only at the buffer's end, or never");
        assert!(finished && once);
    }

    /// Through the decoder on `P`, as the host takes two queues' rings on threads of their own:
    /// buffer A of queue 1 fills a tracked buffer, and buffer B of queue 2, committed plainly
    /// after A by the guest, names A in its `COMMIT_ORDER` and copies the buffer out. B is taken
    /// first, on a thread of its own; A is begun 50 ms on and committed 100 ms after that, and
    /// its message ends, raising its queue's commit mark, once the platform has scheduled it. The
    /// bytes B copied.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_plain_commit_after_a_named_buffer_of_another_queue<P: crate::platform::ApiPlatform>()
    -> Vec<u8> {
        use std::time::{Duration, Instant};

        use reims_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpFillBuffer,
        };
        use reims_vgpu_core::platform::resource::Buffer as _;
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let completed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&completed);
        dec.set_completion_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
        let scheduled = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&scheduled);
        dec.set_scheduled_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
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

        let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
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
        let named = RgfxSurfaceOrderWait {
            queue_obj_id: 1,
            seqno: 20,
        };
        let mut ops = op(RGFX_OP_COMMIT_ORDER, named.as_bytes());
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
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

        let mut first = dec.decoders().sender(None, dec.sinks.clone());
        let mut second = dec.decoders().sender(None, dec.sinks.clone());
        std::thread::scope(|scope| {
            let device = &device;
            let taken_first = scope.spawn(|| {
                let _pool = reims_vgpu_core::rt::Pool::here();
                let begun = second.begin_taken(device, &b, Some(2)).unwrap().unwrap();
                let done = second.finish_command_buffer(device, &b, begun).unwrap();
                second.end_message();
                done
            });
            std::thread::sleep(Duration::from_millis(50));
            let begun = first.begin_taken(device, &a, Some(1)).unwrap().unwrap();
            std::thread::sleep(Duration::from_millis(100));
            assert_eq!(
                first.finish_command_buffer(device, &a, begun).unwrap(),
                Ok(())
            );
            let start = Instant::now();
            while !scheduled.lock().contains(&20) {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "A was never scheduled"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
            first.end_message();
            assert_eq!(taken_first.join().unwrap(), Ok(()));
        });
        let start = Instant::now();
        while !completed.lock().contains(&21) || !completed.lock().contains(&20) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "A or B never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let Some(ObjHandle::Buffer(out)) = dec.object(11) else {
            panic!("no buffer 11");
        };
        // SAFETY: a Shared buffer of `LEN` bytes, read after the command buffer that wrote it
        // finished.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), LEN as usize) };
        bytes.to_vec()
    }

    /// On Metal, a plain commit on another queue that the guest made after a buffer's, and that
    /// names it in its `COMMIT_ORDER`, reads the tracked buffer it writes, though the host took
    /// it first: the host commits it only once the buffer it names is committed, here once Metal
    /// has scheduled that one, and Metal orders a tracked resource across queues behind work it
    /// has submitted (docs/graphics-kext/dgpu-oracle.md, Q6 and Q7). Committed as soon as the
    /// named buffer's `commit` returned, it missed that buffer's writes in 1 of 40 runs of
    /// `graphics::decode`'s tests in one process (measured on the M3 Max; Q6's 0 of 5 ran on an
    /// idle GPU).
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_a_plain_commit_on_another_queue_sees_the_writes_of_the_buffer_it_names() {
        let bytes = a_plain_commit_after_a_named_buffer_of_another_queue::<
            reims_vgpu_metal::MetalPlatform,
        >();
        assert!(
            bytes.iter().all(|&b| b == 0x5a),
            "committed before the buffer it names"
        );
    }

    /// The same on the Vulkan platform
    /// (`on_metal_a_plain_commit_on_another_queue_sees_the_writes_of_the_buffer_it_names`).
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_a_plain_commit_on_another_queue_sees_the_writes_of_the_buffer_it_names() {
        let bytes = a_plain_commit_after_a_named_buffer_of_another_queue::<
            reims_vgpu_vulkan::VulkanPlatform,
        >();
        assert!(
            bytes.iter().all(|&b| b == 0x5a),
            "committed before the buffer it names"
        );
    }

    /// Through the decoder on `P`: buffer A of queue 1, which holds the surface order, waits for
    /// a shared event the CPU sets 100 ms on, then fills a tracked buffer; buffer B of queue 2
    /// names A in its surface order, asks for its submission and copies the buffer out. The
    /// records of B posted by the time its decode returned, whether B was still unfinished 100 ms
    /// on, the order of the completions, and the bytes B copied.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn an_order_behind_a_held_buffer<P: crate::platform::ApiPlatform>()
    -> (Vec<u32>, bool, Vec<u64>, Vec<u8>) {
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
        // B's records, and every completion.
        let records = Arc::new(Mutex::new(Vec::new()));
        let completed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&records);
        dec.set_submitted_sink(Arc::new(move |m| {
            if m.cb_obj_id == 21 {
                sink.lock().push(m.header.r#type);
            }
        }));
        let sink = Arc::clone(&records);
        dec.set_scheduled_sink(Arc::new(move |m| {
            if m.cb_obj_id == 21 {
                sink.lock().push(m.header.r#type);
            }
        }));
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
        let order = |waits: &[RgfxSurfaceOrderWait]| op(RGFX_OP_SURFACE_ORDER, waits.as_bytes());

        let mut ops = order(&[]);
        ops.extend(event_op(RGFX_OP_WAIT_EVENT, 7, 1));
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

        let mut ops = order(&[RgfxSurfaceOrderWait {
            queue_obj_id: 1,
            seqno: 20,
        }]);
        ops.extend(op(RGFX_OP_SUBMITTED, &[]));
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
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
        let posted = records.lock().clone();
        std::thread::sleep(Duration::from_millis(100));
        let waited = !completed.lock().contains(&21);
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
        while !completed.lock().contains(&21) || !completed.lock().contains(&20) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "A or B never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let Some(ObjHandle::Buffer(out)) = dec.object(11) else {
            panic!("no buffer 11");
        };
        // SAFETY: a Shared buffer of `LEN` bytes, read after the command buffer that wrote it
        // finished.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), LEN as usize) };
        let completed = completed.lock().clone();
        (posted, waited, completed, bytes.to_vec())
    }

    /// On Metal, a command buffer that waits on the GPU for a buffer of another queue the
    /// platform holds is answered as submitted and scheduled as its commit returns, as a Mac
    /// answers a buffer it orders after another queue's access to an IOSurface
    /// (docs/graphics-kext/dgpu-oracle.md, Q16), and still runs after that buffer: it completes
    /// after it and copies what it wrote.
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_an_order_behind_a_held_buffer_holds_back_no_record() {
        let (posted, waited, completed, bytes) =
            an_order_behind_a_held_buffer::<reims_vgpu_metal::MetalPlatform>();
        assert_eq!(posted, [RGFX_EVT_CB_SUBMITTED, RGFX_EVT_CB_SCHEDULED]);
        assert!(waited, "B ran before the buffer it names");
        assert_eq!(completed, [20, 21]);
        assert!(bytes.iter().all(|&b| b == 0x5a));
    }

    /// The same on the Vulkan platform, which holds a wait for a shared event's value on the host
    /// until it is reached (`on_metal_an_order_behind_a_held_buffer_holds_back_no_record`). B
    /// still runs after A, copying what A wrote; the order in which the runtime reports the two
    /// queues' completions is its own: on MoltenVK on the M3 Max it reported B's before A's in 20
    /// runs of 20, with B's records held back to its scheduling as with them posted at its commit.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_an_order_behind_a_held_buffer_holds_back_no_record() {
        let (posted, waited, mut completed, bytes) =
            an_order_behind_a_held_buffer::<reims_vgpu_vulkan::VulkanPlatform>();
        assert_eq!(posted, [RGFX_EVT_CB_SUBMITTED, RGFX_EVT_CB_SCHEDULED]);
        assert!(waited, "B ran before the buffer it names");
        completed.sort_unstable();
        assert_eq!(completed, [20, 21]);
        assert!(bytes.iter().all(|&b| b == 0x5a));
    }

    /// Through the decoder on `P`: a pass over a 64-pixel square target with a visibility result
    /// buffer of four words, each of 0xee bytes, whose ops are those of a parallel pass of two
    /// sub-encoders as the bundle joins them, each drawing a square over the target: the first
    /// counts its draw at 16, the second starts at `Disabled` at 0, as the bundle starts a
    /// sub-encoder after one that set a mode, and draws; then `Disabled` at 24 and a draw. The
    /// buffer's words once the command buffer finished.
    #[cfg(any(target_os = "macos", feature = "vulkan"))]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn sub_encoders_after_a_count<P: crate::platform::ApiPlatform>() -> [u64; 4] {
        use std::time::{Duration, Instant};

        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::values::{RenderPipelineDesc, TextureDesc};

        use super::super::draw_cost::fixture;
        const SIDE: u64 = 64;
        const BGRA8: u64 = 80;
        let _pool = reims_vgpu_core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(reims_graphics::region::Region::new(0x10000).unwrap()),
        );
        let completed = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&completed);
        dec.set_completion_sink(Arc::new(move |m| sink.lock().push(m.cb_obj_id)));
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
        // The results, the square's corners and its colour.
        for (obj_id, length) in [(10, 32), (11, 64), (12, 16)] {
            let buffer = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id,
                length,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        }
        let fill = |obj_id: u64, bytes: &[u8]| {
            let Some(ObjHandle::Buffer(b)) = dec.object(obj_id) else {
                panic!("no buffer {obj_id}");
            };
            // SAFETY: a Shared buffer at least `bytes.len()` long, which no command uses yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    b.contents().as_ptr::<u8>(),
                    bytes.len(),
                );
            }
        };
        fill(10, &[0xee; 32]);
        let corners: [[f32; 4]; 4] = [
            [-1.0, -1.0, 0.5, 1.0],
            [1.0, -1.0, 0.5, 1.0],
            [-1.0, 1.0, 0.5, 1.0],
            [1.0, 1.0, 0.5, 1.0],
        ];
        fill(11, corners.as_bytes());
        fill(12, [1.0f32, 0.0, 0.0, 1.0].as_bytes());
        let target = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: SIDE,
            height: SIDE,
            usage: 1 | 4, // ShaderRead | RenderTarget
            ..TextureDesc::default()
        });
        let target = device.new_texture(OnFault::Die, &target).unwrap().unwrap();
        assert_eq!(dec.insert(4, ObjHandle::Texture(target)).unwrap(), Ok(()));
        let pipeline = device
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: None,
                vertex: &fixture::<P>(&device, "render_vbuf"),
                fragment: &fixture::<P>(&device, "render_color"),
                color0_pixel_format: BGRA8,
                blend_over: false,
                resource_index: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(
            dec.insert(5, ObjHandle::RenderPipeline(pipeline)).unwrap(),
            Ok(())
        );

        let pass = RgfxRenderPassDesc {
            color_count: 1,
            flags: RGFX_PASS_FLAG_VISIBILITY_RESULT,
            ..RgfxRenderPassDesc::default()
        };
        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: 4,
            load_action: 2,  // Clear
            store_action: 1, // Store
            ..RgfxRenderPassColorAttachment::default()
        };
        let visibility = RgfxRenderPassVisibility {
            buffer_obj_id: 10,
            result_type: 0,
            reserved_0c: 0,
        };
        let mut begin = pass.as_bytes().to_vec();
        begin.extend_from_slice(color.as_bytes());
        begin.extend_from_slice(visibility.as_bytes());
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin);
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 5 }.as_bytes(),
        ));
        for (buffer_obj_id, index) in [(11, 0), (12, 1)] {
            let bind = RgfxOpSetVertexBuffer {
                buffer_obj_id,
                offset: 0,
                index,
            };
            ops.extend(op(RGFX_OP_RENDER_SET_VERTEX_BUFFER, bind.as_bytes()));
        }
        let draw = RgfxOpDrawPrimitives {
            primitive_type: 4, // TriangleStrip
            vertex_start: 0,
            vertex_count: 4,
        };
        for (mode, offset) in [(2, 16), (0, 0), (0, 24)] {
            let set = RgfxOpSetVisibilityResultMode { mode, offset };
            ops.extend(op(
                RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE,
                set.as_bytes(),
            ));
            ops.extend(op(RGFX_OP_RENDER_DRAW_PRIMITIVES, draw.as_bytes()));
        }
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        assert_eq!(
            dec.decode(&device, &command_buffer_on(1, 20, &ops))
                .unwrap(),
            Ok(())
        );
        let start = Instant::now();
        while !completed.lock().contains(&20) {
            assert!(start.elapsed() < Duration::from_secs(10), "never finished");
            std::thread::sleep(Duration::from_millis(1));
        }
        let Some(ObjHandle::Buffer(results)) = dec.object(10) else {
            panic!("no buffer 10");
        };
        // SAFETY: a Shared buffer of 4 words, read after the command buffer that wrote it finished.
        let words = unsafe { std::slice::from_raw_parts(results.contents().as_ptr::<u64>(), 4) };
        [words[0], words[1], words[2], words[3]]
    }

    /// On Metal, the sub-encoders' draws under `Disabled` write nothing, as every Mac's do: the
    /// count at 16 is the square's 4096 samples, and the words at 0 and 24 keep their bytes. Sent
    /// at those offsets, `Disabled` makes the M3 Max write 0 at 24 (measured), which
    /// [`render::visibility_offset`] keeps it from.
    #[cfg(target_os = "macos")]
    #[test]
    fn on_metal_sub_encoders_after_a_count_write_nothing_under_disabled() {
        let fill = u64::from_ne_bytes([0xee; 8]);
        assert_eq!(
            sub_encoders_after_a_count::<reims_vgpu_metal::MetalPlatform>(),
            [fill, fill, 4096, fill]
        );
    }

    /// The same on the Vulkan platform, whose runtime writes nothing under `Disabled` at any
    /// offset.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_sub_encoders_after_a_count_write_nothing_under_disabled() {
        let fill = u64::from_ne_bytes([0xee; 8]);
        assert_eq!(
            sub_encoders_after_a_count::<reims_vgpu_vulkan::VulkanPlatform>(),
            [fill, fill, 4096, fill]
        );
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

    /// Held by the tests that pin buffers in the contexts' range of GPU addresses on the Vulkan
    /// platform. Its device is the process's, and so are its addresses: under `cargo test`, whose
    /// tests share one process, `on_vulkan_a_pin_meets_no_range_of_a_texture_buffer`'s pin was
    /// refused in 4 of 20 one-process runs of `graphics::decode` (measured on the M3 Max), where
    /// `on_vulkan_the_host_s_own_ranges_lie_below_every_unit`'s pin at the range's base, 96 KiB,
    /// covers its address.
    #[cfg(feature = "vulkan")]
    static CONTEXT_RANGE_PINS: Mutex<()> = Mutex::new(());

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
        let _pins = CONTEXT_RANGE_PINS.lock();
        let _pool = reims_vgpu_core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let buffer = buffer_at(11, RGFX_CONTEXT_VA_BASE + 0x8000, 0x1_0000);
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
    }

    /// The host's own ranges lie below the guest's GPU addresses: a buffer the host places
    /// itself lies below them, and, with it, a queue's and a texture buffer's ranges made, the
    /// first pick of every unit the kext hands the contexts out in and the last unit's last pick
    /// are made.
    #[cfg(feature = "vulkan")]
    #[test]
    fn on_vulkan_the_host_s_own_ranges_lie_below_every_unit() {
        use reims_graphics::wire::{
            RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_END, RGFX_CONTEXT_VA_UNIT, RGFX_CONTEXT_VA_UNITS,
        };
        use reims_vgpu_core::platform::resource::Buffer as _;
        let _pins = CONTEXT_RANGE_PINS.lock();
        let _pool = reims_vgpu_core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let placed = buffer_at(11, 0, 64 << 20);
        assert_eq!(dec.decode(&device, placed.as_bytes()).unwrap(), Ok(()));
        let Some(ObjHandle::Buffer(buffer)) = dec.object(11) else {
            panic!("no buffer 11");
        };
        assert!(buffer.gpu_address() < RGFX_CONTEXT_VA_BASE);
        // A pick of 64 KiB holds 96 KiB: its four pages, the guard page, rounded up to 32 KiB.
        let units = u64::from(RGFX_CONTEXT_VA_UNITS);
        let last = RGFX_CONTEXT_VA_END - 0x1_8000;
        let picks = (0..units).map(|u| RGFX_CONTEXT_VA_BASE + u * RGFX_CONTEXT_VA_UNIT);
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
        let (pass, facts) = dec
            .build_render_pass(&begin_render_body(), |_| true)
            .unwrap();
        assert_eq!(facts.visibility_length, None);
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
        let (pass, _) = dec.build_render_pass(&with(&positions), |_| true).unwrap();
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
        let (plain, _) = dec
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

    /// [`begin_render_body`] with a visibility result record of buffer `buffer_obj_id` and
    /// `result_type` after its colour record, and its sample positions before it where
    /// `positions` is given.
    fn begin_render_with_visibility(
        buffer_obj_id: u64,
        result_type: u32,
        positions: Option<&RgfxRenderPassSamplePositions>,
    ) -> Vec<u8> {
        let plain = begin_render_body();
        let mut head = RgfxRenderPassDesc::read_from_prefix(&plain).unwrap().0;
        head.flags |= RGFX_PASS_FLAG_VISIBILITY_RESULT;
        let mut body = Vec::new();
        if let Some(p) = positions {
            head.flags |= RGFX_PASS_FLAG_SAMPLE_POSITIONS;
            body.extend_from_slice(p.as_bytes());
        }
        let visibility = RgfxRenderPassVisibility {
            buffer_obj_id,
            result_type,
            reserved_0c: 0,
        };
        body.extend_from_slice(visibility.as_bytes());
        let mut out = head.as_bytes().to_vec();
        out.extend_from_slice(&plain[RGFX_RENDER_PASS_COLOR_OFFSET as usize..]);
        out.extend_from_slice(&body);
        out
    }

    /// A pass's visibility result buffer and its type, after its colour records and its sample
    /// positions, are set on the platform's pass, and the pass's ops learn the buffer's length;
    /// a pass whose flag says so but carries no whole record faults, and so does one naming a
    /// buffer the host does not hold, or a type Metal does not define. One without the flag sets
    /// neither.
    #[test]
    fn a_render_pass_takes_its_visibility_result_buffer() {
        use reims_vgpu_core::testkit::impls::descriptor::PassModel;
        let (_p, dec) = setup();
        let Some(ObjHandle::Buffer(buffer)) = dec.object(2) else {
            panic!("no buffer 2");
        };
        let id = buffer.id();
        let mut positions = RgfxRenderPassSamplePositions {
            count: 2,
            ..RgfxRenderPassSamplePositions::default()
        };
        positions.position_bits[0] = [0.25f32.to_bits(), 0.25f32.to_bits()];
        positions.position_bits[1] = [0.75f32.to_bits(), 0.75f32.to_bits()];
        for (result_type, positions) in [(0, None), (1, Some(&positions))] {
            let body = begin_render_with_visibility(2, result_type, positions);
            let (pass, facts) = dec.build_render_pass(&body, |_| true).unwrap();
            assert_eq!(facts.visibility_length, Some(65536));
            let set = pass.with_model(|m: &PassModel| {
                (
                    m.visibility_result_buffer_id(),
                    m.visibility_result_type,
                    m.sample_positions.as_ref().map(Vec::len),
                )
            });
            assert_eq!(
                set,
                (Some(id), Some(u64::from(result_type)), positions.map(|_| 2))
            );
        }
        let (plain, facts) = dec
            .build_render_pass(&begin_render_body(), |_| true)
            .unwrap();
        assert_eq!(facts.visibility_length, None);
        assert_eq!(
            plain.with_model(|m: &PassModel| (
                m.visibility_result_buffer_id(),
                m.visibility_result_type
            )),
            (None, None)
        );
        let whole = begin_render_with_visibility(2, 0, None);
        assert!(
            dec.build_render_pass(&whole[..whole.len() - 4], |_| true)
                .is_err()
        );
        let unheld = begin_render_with_visibility(0x99, 0, None);
        assert!(dec.build_render_pass(&unheld, |_| true).is_err());
        let texture = begin_render_with_visibility(4, 0, None);
        assert!(dec.build_render_pass(&texture, |_| true).is_err());
        let undefined = begin_render_with_visibility(2, 2, None);
        assert!(dec.build_render_pass(&undefined, |_| true).is_err());
    }

    /// A `BUFFER_CREATE` of buffer `obj_id`, `length` bytes.
    fn buffer_of_length(obj_id: u64, length: u64) -> RgfxBufferCreate {
        RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            length,
            ..RgfxBufferCreate::default()
        }
    }

    /// A `SET_VISIBILITY_RESULT_MODE` op of `mode` at `offset`.
    fn visibility_mode(mode: u64, offset: u64) -> Vec<u8> {
        op(
            RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE,
            RgfxOpSetVisibilityResultMode { mode, offset }.as_bytes(),
        )
    }

    /// A pass over a visibility result buffer sends each mode and offset the guest sets, in both
    /// modes, with each draw between them, the last 8 bytes of the buffer among them, through to
    /// `Disabled`, sent at the last of their offsets ([`render::visibility_offset`]); a mode at an
    /// offset whose 8 bytes run past the buffer's end is refused at
    /// its op and fails the command buffer, and so is a mode in a pass with no buffer, where
    /// plain Metal's process ends. `Disabled` is taken at any offset, with a buffer or none.
    #[test]
    fn a_pass_s_visibility_modes_reach_the_platform_within_its_buffer() {
        let (p, mut dec) = setup();
        let device = p.device();
        assert_eq!(
            dec.decode(&device, buffer_of_length(6, 16).as_bytes())
                .unwrap(),
            Ok(())
        );
        let draw = || {
            op(
                RGFX_OP_RENDER_DRAW_PRIMITIVES,
                RgfxOpDrawPrimitives {
                    primitive_type: 3,
                    vertex_start: 0,
                    vertex_count: 3,
                }
                .as_bytes(),
            )
        };
        let pass = |modes: &[(u64, u64)], buffer: Option<u64>| {
            let begin = match buffer {
                Some(id) => begin_render_with_visibility(id, 0, None),
                None => begin_render_body(),
            };
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin);
            for &(mode, offset) in modes {
                ops.extend(visibility_mode(mode, offset));
                ops.extend(draw());
            }
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
            ops
        };
        let sent = |mark| -> Vec<Vec<Arg>> {
            p.call_log()
                .since(mark)
                .into_iter()
                .filter(|c| c.sel == "setVisibilityResultMode:offset:")
                .map(|c| c.args)
                .collect()
        };

        let modes = [(2, 0), (1, 8), (0, 0)];
        let mark = p.call_log().mark();
        let message = command_buffer_message(110, 1, &pass(&modes, Some(6)));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let want: Vec<Vec<Arg>> = [(2, 0), (1, 8), (0, 8)]
            .iter()
            .map(|&(m, o)| vec![Arg::U64(m), Arg::U64(o)])
            .collect();
        assert_eq!(sent(mark), want);
        let sels = p.call_log().sels_since(mark);
        let draws = sels
            .iter()
            .filter(|s| **s == "drawPrimitives:vertexStart:vertexCount:")
            .count();
        assert_eq!(draws, 3);

        for (n, (modes, buffer)) in [
            (&[(2, 8), (1, 16)][..], Some(6)),
            (&[(1, 9)][..], Some(6)),
            (&[(2, u64::MAX - 3)][..], Some(6)),
            (&[(1, 0)][..], None),
            (&[(2, 0)][..], None),
        ]
        .into_iter()
        .enumerate()
        {
            let mark = p.call_log().mark();
            let message =
                command_buffer_message(120 + n as u64, 2 + n as u64, &pass(modes, buffer));
            assert!(
                refusing(|| dec.decode(&device, &message)).is_err(),
                "{modes:?} over {buffer:?}"
            );
            // Every mode before the refused one was sent; the refused one never was.
            assert_eq!(
                sent(mark).len(),
                modes.len() - 1,
                "{modes:?} over {buffer:?}"
            );
            assert!(
                !p.call_log().sels_since(mark).contains(&"commit"),
                "{modes:?} over {buffer:?}"
            );
        }

        let mark = p.call_log().mark();
        let message = command_buffer_message(130, 9, &pass(&[(0, 1000), (0, 3)], None));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(sent(mark).len(), 2);
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
            "addCompletionSeenHandler",
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

    /// A shared event's signal the buffer runs more work after (an encoder, or a wait) goes to the
    /// guest as the host's event reaches its value, before the buffer completes, and not again at
    /// its completion; one at the buffer's end goes at its completion, ahead of it. The CPU's set
    /// stands in for the GPU reaching the signal. So does a stand-in's signal before its wait.
    #[test]
    fn a_signal_before_more_of_its_buffer_goes_as_the_event_reaches_it() {
        use reims_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use reims_graphics::wire::RGFX_CB_INTERNAL;
        use reims_vgpu_core::platform::event::SharedEvent as _;
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
        let reach = |dec: &GraphicsDecoder<FakePlatform>, value: u64| {
            let Some(ObjHandle::SharedEvent(event)) = dec.object(7) else {
                panic!("no shared event 7");
            };
            event.set_signaled_value(value).unwrap();
        };

        // A blit, signal 3, a blit, signal 4, and a wait for 5 after a signal of 5's event to 1
        // (the plain event's signal, posted never).
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 3));
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 4));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 6, 1));
        ops.extend(event_op(RGFX_OP_WAIT_EVENT, 6, 1));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 9));
        let mark = p.call_log().mark();
        let message = command_buffer_message(140, 64, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let asked = p.call_log().sels_since(mark);
        let asked = asked
            .iter()
            .filter(|s| **s == "notifyListener:atValue:block:")
            .count();
        assert_eq!(asked, 2, "the signals to 3 and 4, not the one at the end");
        assert!(posted.lock().is_empty());
        reach(&dec, 3);
        assert_eq!(*posted.lock(), [(RGFX_EVT_EVENT_SIGNALLED, 7, 3)]);
        reach(&dec, 4);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 3),
                (RGFX_EVT_EVENT_SIGNALLED, 7, 4),
                (RGFX_EVT_EVENT_SIGNALLED, 7, 9),
                (RGFX_EVT_CB_COMPLETE, 140, 0),
            ]
        );

        // A value the event is at already goes as the buffer is walked.
        posted.lock().clear();
        let mut ops = event_op(RGFX_OP_SIGNAL_EVENT, 7, 2);
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let message = command_buffer_message(141, 65, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(*posted.lock(), [(RGFX_EVT_EVENT_SIGNALLED, 7, 2)]);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 2),
                (RGFX_EVT_CB_COMPLETE, 141, 0)
            ]
        );

        // A stand-in for a buffer that faults: its signal to 10 before its wait for 11 goes as the
        // event reaches 10, its signal to 12 at its completion.
        posted.lock().clear();
        let mut ops = op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        );
        // Pipeline 99 is nothing the decoder holds: the walk faults here.
        ops.extend(op(
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState {
                pipeline_obj_id: 99,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 10));
        ops.extend(event_op(RGFX_OP_WAIT_EVENT, 7, 11));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 12));
        let message = command_buffer_message(142, 66, &ops);
        assert!(dec.decode(&device, &message).is_err());
        assert!(dec.stand_in_for_failed(&device, &message).unwrap());
        assert!(posted.lock().is_empty());
        reach(&dec, 10);
        assert_eq!(*posted.lock(), [(RGFX_EVT_EVENT_SIGNALLED, 7, 10)]);
        reach(&dec, 11);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 10),
                (RGFX_EVT_EVENT_SIGNALLED, 7, 12),
                (RGFX_EVT_CB_COMPLETE, 142, u64::from(RGFX_CB_INTERNAL)),
            ]
        );
    }

    /// A signal of a shared event whose create the host refused goes to the guest with its buffer's
    /// completion, ahead of it, though the buffer runs more work after it: the host holds no event
    /// a notification could watch, and a post as the walk reaches it could come before the GPU's
    /// work before it is done. Its wait is dropped, and so is everything the platform would carry.
    #[test]
    fn a_refused_shared_event_s_signal_goes_with_its_buffer_s_completion() {
        let (p, mut dec) = setup();
        let device = p.device();
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
        p.script().push("newSharedEvent", Step::Nil);
        let create = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 8,
            initial_value: 0,
        };
        assert!(
            dec.decode_refusing(&device, create.as_bytes())
                .unwrap()
                .is_err()
        );

        let mut ops = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 8, 1));
        ops.extend(event_op(RGFX_OP_WAIT_EVENT, 8, 2));
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let mark = p.call_log().mark();
        let message = command_buffer_message(143, 67, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let sels = p.call_log().sels_since(mark);
        for dropped in [
            "notifyListener:atValue:block:",
            "encodeSignalEvent:value:",
            "encodeWaitForEvent:value:",
        ] {
            assert!(!sels.contains(&dropped), "{dropped}");
        }
        assert!(posted.lock().is_empty(), "posted as the walk reached it");
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 8, 1),
                (RGFX_EVT_CB_COMPLETE, 143, 0)
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
        assert!(dec.stand_in_for_failed(&device, &message).unwrap());
        let want = [
            "commandBuffer",
            "encodeWaitForEvent:value:",
            "encodeSignalEvent:value:",
            "encodeSignalEvent:value:",
            "addCompletionSeenHandler",
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
        assert!(!dec.stand_in_for_failed(&device, &elsewhere).unwrap());
    }

    /// A buffer whose `op_bytes` count fewer than its message's ops is refused as malformed, none
    /// of its ops carried out, and is stood in for by all of its event ops, those past the count
    /// among them, so none is dropped with the ops the count leaves out.
    #[test]
    fn a_buffer_whose_op_bytes_are_short_is_refused_and_stood_in_for_by_all_its_events() {
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

        let mut ops = event_op(RGFX_OP_SIGNAL_EVENT, 7, 4);
        let first = ops.len() as u64;
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 5));
        let mut message = command_buffer_message(130, 70, &ops);
        let op_bytes = std::mem::offset_of!(RgfxCommandBuffer, op_bytes);
        message[op_bytes..op_bytes + 8].copy_from_slice(&first.to_le_bytes());
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &message).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert!(!p.call_log().sels_since(mark).contains(&"commit"));
        assert!(posted.lock().is_empty());

        assert!(dec.stand_in_for_failed(&device, &message).unwrap());
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 4),
                (RGFX_EVT_EVENT_SIGNALLED, 7, 5),
                (RGFX_EVT_CB_COMPLETE, 130, u64::from(RGFX_CB_INTERNAL)),
            ]
        );
    }

    /// A buffer whose commit the platform refuses after its handlers were added (a death at the
    /// commit, which the lane's scope takes, or a raise there) is not carried out, and the process
    /// goes on: its handlers never run, its stand-in posts its signal and its failure once, and the
    /// queue's next buffer is committed and completes.
    #[test]
    fn a_commit_refused_after_its_handlers_were_added_fails_that_buffer_alone() {
        use reims_graphics::wire::{RGFX_CB_INTERNAL, RGFX_CB_OK};
        for (step, death) in [
            (Step::crash(), true),
            (
                Step::raise("NSInternalInconsistencyException", "refused"),
                false,
            ),
        ] {
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

            let mut ops = render_ops();
            ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 4));
            let message = command_buffer_message(160, 90, &ops);
            p.script().push("commit", step);
            let mark = p.call_log().mark();
            let outcome = refusing(|| dec.decode(&device, &message));
            let raise = outcome.expect_err("the commit was refused");
            assert_eq!(
                raise.to_string().contains("refused: platform call commit"),
                death,
                "{raise}"
            );
            let sels = p.call_log().sels_since(mark);
            let seen = sels.iter().position(|&s| s == "addCompletionSeenHandler");
            let commit = sels.iter().position(|&s| s == "commit");
            assert!(seen.is_some() && seen < commit, "{sels:?}");
            dec.end_message();
            assert!(posted.lock().is_empty());

            assert!(dec.stand_in_for_failed(&device, &message).unwrap());
            dec.end_message();
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                *posted.lock(),
                [
                    (RGFX_EVT_EVENT_SIGNALLED, 7, 4),
                    (RGFX_EVT_CB_COMPLETE, 160, u64::from(RGFX_CB_INTERNAL)),
                ]
            );

            posted.lock().clear();
            let next = command_buffer_message(161, 91, &render_ops());
            assert_eq!(dec.decode(&device, &next).unwrap(), Ok(()));
            dec.end_message();
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                *posted.lock(),
                [(RGFX_EVT_CB_COMPLETE, 161, u64::from(RGFX_CB_OK))]
            );
        }
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

    /// A gate the first walk to reach it stops at, after its op `at`, until the test opens it.
    /// Whether a walk took the gate, stopped at it, and whether it is open; and their moves.
    type GateState = Arc<(Mutex<(bool, bool, bool)>, reims_vgpu_core::sync::Condvar)>;

    struct Gate(GateState);

    impl Gate {
        /// The gate, set on the walks of `dec`'s state.
        fn on(dec: &GraphicsDecoder<FakePlatform>, at: usize) -> Gate {
            let state = Arc::new((
                Mutex::new((false, false, false)),
                reims_vgpu_core::sync::Condvar::new(),
            ));
            let hook = Arc::clone(&state);
            dec.set_walk_hook(Some(Arc::new(move |index: usize| {
                let (lock, moved) = &*hook;
                let mut s = lock.lock();
                // (taken by a walk, the walk stopped at it, open)
                if index != at || s.0 {
                    return;
                }
                s.0 = true;
                s.1 = true;
                moved.notify_all();
                while !s.2 {
                    moved.wait(&mut s);
                }
            })));
            Gate(state)
        }

        /// Waits for a walk to stop at the gate.
        fn reached(&self) {
            let (lock, moved) = &*self.0;
            let mut s = lock.lock();
            for _ in 0..3000 {
                if s.1 {
                    return;
                }
                moved.wait_for(&mut s, std::time::Duration::from_millis(1));
            }
            panic!("no walk reached the gate");
        }

        /// Lets the walk stopped at the gate go on.
        fn open(&self) {
            let (lock, moved) = &*self.0;
            lock.lock().2 = true;
            moved.notify_all();
        }
    }

    fn destroy(obj_id: u64) -> Vec<u8> {
        reims_graphics::wire::RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: reims_graphics::wire::RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<reims_graphics::wire::RgfxObjectDestroy>() as u32,
                seqno: 0,
            },
            obj_id,
        }
        .as_bytes()
        .to_vec()
    }

    fn buffer_create(obj_id: u64) -> Vec<u8> {
        RgfxBufferCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_CREATE,
                size: size_of::<RgfxBufferCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            length: 64,
            ..RgfxBufferCreate::default()
        }
        .as_bytes()
        .to_vec()
    }

    /// The gone entries context 0's table keeps for open views.
    fn kept(dec: &GraphicsDecoder<FakePlatform>) -> usize {
        dec.context_state(0).map_or(0, |c| c.gone_len())
    }

    /// A command buffer's walk names objects as its context's table was at its begin, the
    /// changes ring 0 carries out meanwhile unseen: each outcome is the one the buffer had,
    /// carried out whole before them, as under one lock (docs/graphics-kext/design.md, "What a
    /// walk sees"). Destroys mid-walk, of the buffer and textures it names, and an id destroyed,
    /// made again and destroyed again, leave the walk naming the objects it began with, which are
    /// let go of once it ends; the object made after its begin, which it cannot name, is let go of
    /// at its destroy; ring 0 is not held up meanwhile.
    #[test]
    fn a_walk_names_what_its_table_held_at_its_begin() {
        let (p, dec) = setup();
        let device = p.device();
        let original = Arg::obj(match dec.object(2) {
            Some(ObjHandle::Buffer(b)) => b,
            _ => panic!("no buffer 2"),
        });
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut walker = dec.sender(None, super::super::Sinks::default());
        drop(dec);
        let gate = Gate::on(&ring0, 0);
        let message = command_buffer_message(100, 1, &render_ops());
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| {
                let outcome = walker.decode(&p.device(), &message).unwrap();
                walker.end_message();
                outcome
            });
            gate.reached();
            for m in [
                destroy(5),
                destroy(2),
                buffer_create(2),
                destroy(2),
                destroy(3),
            ] {
                assert_eq!(ring0.decode(&device, &m).unwrap(), Ok(()));
            }
            assert_eq!(ring0.object_count(), 2, "the queue and texture 4");
            assert_eq!(
                kept(&ring0),
                3,
                "kept for the walk: 5, 2 and 3, not the 2 made after its begin"
            );
            gate.open();
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(kept(&ring0), 0, "let go of with the walk");
        let bound = p.call_log().calls_to("setVertexBuffer:offset:atIndex:");
        assert_eq!(
            bound.last().unwrap().args[0],
            original,
            "the buffer it began with"
        );
    }

    /// A walk does not name an object made after its begin, as one carried out whole before the
    /// create would not: it fails at the op, as one naming an object destroyed before its begin.
    #[test]
    fn a_walk_names_nothing_made_after_its_begin() {
        let (p, dec) = setup();
        let device = p.device();
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut walker = dec.sender(None, super::super::Sinks::default());
        drop(dec);
        // Destroyed before the begin: the op naming it faults.
        assert_eq!(ring0.decode(&device, &destroy(3)).unwrap(), Ok(()));
        let message = command_buffer_message(100, 1, &render_ops());
        assert!(walker.decode(&device, &message).is_err());
        // Made after the begin: not named.
        let gate = Gate::on(&ring0, 0);
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message).map_err(|_| ()));
            gate.reached();
            let index = RgfxBufferCreate {
                length: 65536,
                ..RgfxBufferCreate::read_from_bytes(&buffer_create(3)).unwrap()
            };
            assert_eq!(ring0.decode(&device, index.as_bytes()).unwrap(), Ok(()));
            gate.open();
            assert!(
                walk.join().unwrap().is_err(),
                "named the buffer made after its begin"
            );
        });
        // A walk begun after the create names it.
        ring0.set_walk_hook(None);
        assert_eq!(walker.decode(&device, &message).unwrap(), Ok(()));
    }

    /// A context's close leaves the decoder no state of it, whatever comes after: a command buffer
    /// handed over after the close (taken off its ring before it) is refused unrun, as a closed
    /// context's, and makes it none; a drain begun after the close (a release the kext posted
    /// behind it) makes none, and ends after the close's own.
    #[test]
    fn a_closed_context_s_late_arrivals_make_no_state() {
        let (p, dec) = setup();
        let device = p.device();
        let ring0 = dec.sender(None, super::super::Sinks::default());
        let mut walker = dec.sender(Some(0), super::super::Sinks::default());
        let message = command_buffer_message(100, 1, &render_ops());
        assert_eq!(walker.decode(&device, &message).unwrap(), Ok(()));
        walker.end_message();
        assert_eq!(p.world().committed().len(), 1);

        let flag = |f: &Arc<AtomicBool>| -> super::super::DrainAnswer {
            let f = Arc::clone(f);
            Box::new(move || f.store(true, Ordering::Release))
        };
        let closed = Arc::new(AtomicBool::new(false));
        let released = Arc::new(AtomicBool::new(false));
        assert_eq!(ring0.drop_context(&device, 0), 5);
        drop(ring0.drain_context(0, true, flag(&closed)));
        assert_eq!(ring0.context_states(), 0);

        drop(ring0.drain_context(0, false, flag(&released)));
        let late = command_buffer_message(101, 2, &render_ops());
        let mut taken = dec.sender(Some(0), super::super::Sinks::default());
        assert!(matches!(
            taken.begin_taken(&device, &late, Some(1)).unwrap(),
            Err(DecodeError::ContextClosed)
        ));
        taken.end_message();
        assert!(matches!(
            taken.decode(&device, &late).unwrap(),
            Err(DecodeError::ContextClosed)
        ));
        taken.end_message();
        assert_eq!(
            ring0.context_states(),
            0,
            "a late arrival made the closed context a state"
        );
        assert_eq!(p.world().committed().len(), 1, "a late buffer ran");

        assert!(!closed.load(Ordering::Acquire) && !released.load(Ordering::Acquire));
        p.world().complete_all(CompleteOn::Caller);
        assert!(closed.load(Ordering::Acquire), "the close was not answered");
        assert!(
            released.load(Ordering::Acquire),
            "the release was not answered"
        );
        assert_eq!(ring0.context_states(), 0);
        drop(device);
    }

    /// A context's close during a walk of its command buffer drops its objects at once, the walk
    /// naming them as at its begin; they are let go of once it ends, and a drain the close begins
    /// is answered once the buffer's completion handler has run.
    #[test]
    fn a_close_during_a_walk_leaves_it_its_objects() {
        let (p, dec) = setup();
        let device = p.device();
        let ring0 = dec.sender(None, super::super::Sinks::default());
        let mut walker = dec.sender(Some(0), super::super::Sinks::default());
        drop(dec);
        let gate = Gate::on(&ring0, 0);
        let message = command_buffer_message(100, 1, &render_ops());
        let answered = Arc::new(AtomicBool::new(false));
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| {
                let outcome = walker.decode(&p.device(), &message).unwrap();
                walker.end_message();
                outcome
            });
            gate.reached();
            assert_eq!(ring0.drop_context(&device, 0), 5);
            assert_eq!(ring0.object_count(), 0);
            let flag = Arc::clone(&answered);
            drop(ring0.drain_context(
                0,
                true,
                Box::new(move || flag.store(true, Ordering::Release)),
            ));
            assert_eq!(
                kept(&ring0),
                0,
                "the closed context's state is the walk's alone"
            );
            gate.open();
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert!(
            !answered.load(Ordering::Acquire),
            "answered before the buffer finished"
        );
        p.world().complete_all(CompleteOn::Caller);
        assert!(answered.load(Ordering::Acquire));
        drop(device);
    }

    /// A command buffer the guest sends after another's `CB_SUBMITTED` is committed after it, on
    /// another queue: its walk goes on, and its commit waits for the first's. A range message is
    /// checked where it comes, waiting for neither (one naming no `Managed` buffer is refused
    /// there, of either context).
    #[test]
    fn a_buffer_sent_after_an_answer_is_committed_after_it() {
        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        let sinks = dec.sinks.clone();
        let mut first = dec.sender(Some(0), sinks.clone());
        let mut second = dec.sender(Some(0), sinks);
        let mut ring0 = dec.sender(Some(0), super::super::Sinks::default());
        let mut other = dec.sender(Some(2), super::super::Sinks::default());
        let gate = Gate::on(&dec, 0);
        drop(dec);
        let range = |obj_id| {
            reims_graphics::wire::RgfxBufferDidModifyRange {
                header: RgfxMsgHeader {
                    r#type: reims_graphics::wire::RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
                    size: size_of::<reims_graphics::wire::RgfxBufferDidModifyRange>() as u32,
                    seqno: 0,
                },
                obj_id,
                location: 0,
                length: 8,
            }
            .as_bytes()
            .to_vec()
        };
        let asked = asking(100, 1);
        let later = command_buffer_on(6, 200, &render_ops());
        let commits = || {
            p.call_log()
                .calls_to("commit")
                .into_iter()
                .filter_map(|c| c.thread)
                .filter(|t| t == "first" || t == "second")
                .collect::<Vec<_>>()
        };
        std::thread::scope(|scope| {
            let a = std::thread::Builder::new()
                .name("first".into())
                .spawn_scoped(scope, || first.decode(&p.device(), &asked).unwrap())
                .unwrap();
            gate.reached();
            assert_eq!(posted.lock().len(), 1, "answered before its walk");
            let b = std::thread::Builder::new()
                .name("second".into())
                .spawn_scoped(scope, || second.decode(&p.device(), &later).unwrap())
                .unwrap();
            assert_eq!(
                ring0.decode(&p.device(), &range(2)).unwrap(),
                Err(DecodeError::BadReference),
                "checked where it came"
            );
            assert_eq!(
                other
                    .decode(
                        &device,
                        &range((2 << reims_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT) | 2)
                    )
                    .unwrap(),
                Err(DecodeError::BadReference)
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(commits().is_empty(), "committed before the answered buffer");
            assert!(!b.is_finished());
            gate.open();
            assert_eq!(a.join().unwrap(), Ok(()));
            assert_eq!(b.join().unwrap(), Ok(()));
        });
        assert_eq!(commits(), ["first", "second"]);
    }

    /// How many times a walk of `ops` calls its hook: once an op but the render ops, whose runs
    /// the walk hands its encoder whole.
    fn hooked(ops: &[u8]) -> usize {
        (op_bodies(ops))
            .filter(|&(tag, _)| op_group(tag) != RGFX_OP_GROUP_RENDER)
            .count()
    }

    /// What a sink of [`splitting`]'s was handed, in order.
    type Sunk<T> = Arc<Mutex<Vec<T>>>;

    /// A decoder that splits command buffers, and what its sinks were handed ([`splitting`]).
    struct Split {
        p: Arc<FakePlatform>,
        dec: GraphicsDecoder<FakePlatform>,
        /// Each `CB_SCHEDULED` and `CB_COMPLETE`: its type, its command buffer and its status.
        posted: Sunk<(u32, u64, u32)>,
        /// Each `CB_COMPLETE` whole.
        records: Sunk<RgfxCbComplete>,
    }

    /// A decoder of [`setup`]'s, its messages of context 0, on a platform that splits command
    /// buffers, whose clock moves on [`SPLIT_AFTER`] after each op a walk carries out, so every
    /// encoder's end may end a part; and the records it posts.
    fn splitting() -> Split {
        let (p, mut dec) = setup();
        dec.set_context(Some(0));
        p.world().split_command_buffers();
        let clock = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |_| {
            clock.virtual_clock().advance(SPLIT_AFTER);
        })));
        let posted = Arc::new(Mutex::new(Vec::new()));
        let scheduled = Arc::clone(&posted);
        dec.set_scheduled_sink(Arc::new(move |m| {
            scheduled
                .lock()
                .push((m.header.r#type, m.cb_obj_id, RGFX_CB_OK));
        }));
        let completions = Arc::clone(&posted);
        let records = Arc::new(Mutex::new(Vec::new()));
        let complete = Arc::clone(&records);
        dec.set_completion_sink(Arc::new(move |m| {
            completions
                .lock()
                .push((m.header.r#type, m.cb_obj_id, m.status));
            complete.lock().push(m);
        }));
        Split {
            p,
            dec,
            posted,
            records,
        }
    }

    /// A command buffer of three render passes walked long is committed in three parts as it is
    /// walked, and is one command buffer to the guest: one `CB_SCHEDULED` and one `CB_COMPLETE`,
    /// after its last part's completion, with the GPU times from its first part's start to its
    /// last part's end, and the error of a part that failed.
    #[test]
    fn a_buffer_walked_long_is_committed_in_parts_with_one_outcome() {
        use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
        let Split {
            p,
            mut dec,
            posted,
            records,
        } = splitting();
        let device = p.device();
        let ops = [render_ops(), render_ops(), render_ops()].concat();
        let message = command_buffer_message(300, 30, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(p.call_log().calls_to("split").len(), 2);
        let parts = p.world().committed();
        assert_eq!(parts.len(), 3, "committed in its parts");
        assert_eq!(
            *posted.lock(),
            [(RGFX_EVT_CB_SCHEDULED, 300, RGFX_CB_OK)],
            "scheduled once, as its last part is"
        );
        for (part, (start, end)) in parts.iter().zip([(10, 20), (25, 40), (45, 60)]) {
            assert!(p.world().set_gpu_times(*part, start, end));
        }
        for part in &parts[..2] {
            assert!(p.world().complete(
                *part,
                CommandBufferStatus::COMPLETED,
                None,
                CompleteOn::Caller
            ));
            assert_eq!(posted.lock().len(), 1, "completed before its last part");
        }
        p.world().complete_all(CompleteOn::Caller);
        let r = records.lock()[0];
        assert_eq!(
            (r.cb_obj_id, r.gpu_start_ns, r.gpu_end_ns, r.status),
            (300, 10, 60, RGFX_CB_OK)
        );
        // A part that fails fails the command buffer, whose later parts complete as they do.
        let message = command_buffer_message(301, 31, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let parts = p.world().committed();
        assert_eq!(parts.len(), 3);
        let error = NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code: 3,
            description: "Page Fault".to_owned(),
        };
        p.world().complete(
            parts[0],
            CommandBufferStatus::COMPLETED,
            None,
            CompleteOn::Caller,
        );
        p.world().complete(
            parts[1],
            CommandBufferStatus::ERROR,
            Some(error),
            CompleteOn::Caller,
        );
        p.world().complete_all(CompleteOn::Caller);
        let completions: Vec<_> = (posted.lock().iter())
            .filter(|(kind, _, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .copied()
            .collect();
        assert_eq!(
            completions,
            [
                (RGFX_EVT_CB_COMPLETE, 300, RGFX_CB_OK),
                (
                    RGFX_EVT_CB_COMPLETE,
                    301,
                    reims_graphics::wire::RGFX_CB_PAGE_FAULT
                )
            ]
        );
    }

    /// A command buffer is committed whole where the platform keeps it whole, and where it could
    /// not be committed when its encoders end: a ticket issued before its begin is out until its
    /// last op.
    #[test]
    fn a_buffer_is_committed_whole_where_its_parts_would_wait() {
        let ops = [render_ops(), render_ops(), render_ops()].concat();
        let (p, mut dec) = setup();
        dec.set_context(Some(0));
        let clock = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |_| {
            clock.virtual_clock().advance(SPLIT_AFTER);
        })));
        let device = p.device();
        let message = command_buffer_message(310, 40, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(p.call_log().calls_to("split").is_empty());
        assert_eq!(
            p.world().committed().len(),
            1,
            "a platform that keeps it whole"
        );

        let Split { p, mut dec, .. } = splitting();
        let device = p.device();
        let ticket = Arc::new(Mutex::new(dec.shared.sequencer.issue(Some(0), false)));
        let last = hooked(&ops) - 1;
        let clock = Arc::clone(&p);
        let held = Arc::clone(&ticket);
        dec.set_walk_hook(Some(Arc::new(move |index| {
            clock.virtual_clock().advance(SPLIT_AFTER);
            if index == last {
                held.lock().take();
            }
        })));
        let message = command_buffer_message(311, 41, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(ticket.lock().is_none());
        assert!(p.call_log().calls_to("split").is_empty());
        assert_eq!(p.world().committed().len(), 1, "its order not yet reached");
    }

    /// While a command buffer's parts are out, a command buffer of another queue of its context
    /// is walked but committed only after its last part; one of another context is not held.
    #[test]
    fn another_queue_s_commit_waits_for_a_buffer_s_last_part() {
        let Split { p, mut dec, .. } = splitting();
        let device = p.device();
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        let ops = [render_ops(), render_ops(), render_ops()].concat();
        let last = hooked(&ops) - 1;
        let gate = Arc::clone(&dec.shared.split_gate);
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (tx, rx) = (Mutex::new(Some(tx)), Mutex::new(rx));
        let clock = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |index| {
            clock.virtual_clock().advance(SPLIT_AFTER);
            // Its first part is out, at its first encoder's end: the other queue's buffer goes;
            // at its last op, that buffer waits at its commit.
            if index == 1 {
                tx.lock().take();
            }
            if index == last {
                for _ in 0..1000 {
                    if gate.waiting(0) == 1 {
                        return;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                panic!("the other queue's buffer never waited at its commit");
            }
        })));
        let mut other = dec.sender(Some(0), super::super::Sinks::default());
        let later = command_buffer_on(6, 320, &render_ops());
        let commits = || {
            (p.call_log().calls().into_iter())
                .filter(|c| c.sel == "commit" || c.sel == "split")
                .filter_map(|c| c.thread)
                .collect::<Vec<_>>()
        };
        std::thread::scope(|scope| {
            let b = std::thread::Builder::new()
                .name("other".into())
                .spawn_scoped(scope, || {
                    let _ = rx.lock().recv();
                    other.decode(&p.device(), &later).unwrap()
                })
                .unwrap();
            let a = std::thread::Builder::new()
                .name("parts".into())
                .spawn_scoped(scope, || {
                    dec.decode(&p.device(), &command_buffer_message(321, 50, &ops))
                        .unwrap()
                })
                .unwrap();
            assert_eq!(a.join().unwrap(), Ok(()));
            assert_eq!(b.join().unwrap(), Ok(()));
        });
        assert_eq!(commits(), ["parts", "parts", "parts", "other"]);
    }

    /// A command buffer whose walk faults after a part has its failure at its stand-in's
    /// completion alone, as one whose walk faults whole; its context's gate is let go.
    #[test]
    fn a_walk_that_faults_after_a_part_fails_at_its_stand_in() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        let mut ops = [render_ops(), render_ops()].concat();
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
        let message = command_buffer_message(330, 60, &ops);
        assert!(refusing(|| dec.decode(&device, &message)).is_err());
        assert_eq!(
            p.call_log().calls_to("split").len(),
            2,
            "its parts committed"
        );
        assert!(dec.stand_in_for_failed(&device, &message).unwrap());
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [(
                RGFX_EVT_CB_COMPLETE,
                330,
                reims_graphics::wire::RGFX_CB_INTERNAL
            )]
        );
        assert!(
            dec.shared.split_gate.try_split(0).is_some(),
            "the gate held after the walk"
        );
    }

    /// A blit encoder's ops, which end a part where the walk may split.
    fn blit_ops() -> Vec<u8> {
        [
            op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            op(RGFX_OP_ENCODER_END, &[]),
        ]
        .concat()
    }

    /// A split after a mid-buffer signal of a shared event leaves the signal in the part before
    /// the cut: the event reaches its value as that part completes, and its value goes to the
    /// guest then, before the buffer's completion, as a whole buffer's goes as the GPU reaches
    /// it. A signal of a shared event whose create the host refused goes with the buffer's
    /// completion, after its last part, never at a part's.
    #[test]
    fn a_split_after_a_mid_buffer_signal_keeps_the_signal_before_the_cut() {
        use reims_vgpu_core::platform::types::CommandBufferStatus;
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let signals = Arc::clone(&posted);
        dec.set_signal_sink(Arc::new(move |m| {
            let value = u32::try_from(m.value).unwrap();
            signals.lock().push((m.header.r#type, m.obj_id, value));
        }));
        p.script().push("newSharedEvent", Step::Nil);
        let create = RgfxSharedEventCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_SHARED_EVENT_CREATE,
                size: size_of::<RgfxSharedEventCreate>() as u32,
                seqno: 0,
            },
            obj_id: 8,
            initial_value: 0,
        };
        assert!(
            dec.decode_refusing(&device, create.as_bytes())
                .unwrap()
                .is_err()
        );
        let mut ops = blit_ops();
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 7, 3));
        ops.extend(event_op(RGFX_OP_SIGNAL_EVENT, 8, 1));
        ops.extend([blit_ops(), blit_ops()].concat());
        let mark = p.call_log().mark();
        let message = command_buffer_message(340, 70, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let parts = p.world().committed();
        assert_eq!(parts.len(), 3, "committed in its parts");
        let signalled: Vec<_> = (p.call_log().since(mark).into_iter())
            .filter(|c| c.sel == "encodeSignalEvent:value:")
            .map(|c| c.obj)
            .collect();
        assert_eq!(
            signalled,
            [parts[1]],
            "the signal in the part before the cut"
        );
        posted.lock().clear();
        let complete = |part| {
            let ok = CommandBufferStatus::COMPLETED;
            assert!(p.world().complete(part, ok, None, CompleteOn::Caller));
        };
        complete(parts[0]);
        assert!(
            posted.lock().is_empty(),
            "posted before the event reached 3"
        );
        complete(parts[1]);
        assert_eq!(*posted.lock(), [(RGFX_EVT_EVENT_SIGNALLED, 7, 3)]);
        complete(parts[2]);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 3),
                (RGFX_EVT_EVENT_SIGNALLED, 8, 1),
                (RGFX_EVT_CB_COMPLETE, 340, RGFX_CB_OK),
            ]
        );
    }

    /// A split after an encoder skipped whole, its pass naming a texture whose create the host
    /// refused: the skipped encoder made nothing, and the buffer's parts complete as one buffer.
    #[test]
    fn a_split_after_a_skipped_encoder_answers_as_the_whole() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        dec.refuse(9, RGFX_CMD_TEXTURE_CREATE);
        // The reference render ops, their pass's colour attachment texture 9.
        let refused = {
            let begin = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
            let mut body = begin_render_body();
            let color = RgfxRenderPassColorAttachment {
                texture_obj_id: 9,
                load_action: 2,
                store_action: 1,
                clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
                ..RgfxRenderPassColorAttachment::default()
            };
            body[size_of::<RgfxRenderPassDesc>()..].copy_from_slice(color.as_bytes());
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &body);
            ops.extend_from_slice(&render_ops()[begin.len()..]);
            ops
        };
        let ops = [render_ops(), refused, render_ops()].concat();
        let mark = p.call_log().mark();
        let message = command_buffer_message(350, 80, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let sels = p.call_log().sels_since(mark);
        let made = (sels.iter())
            .filter(|s| **s == "renderCommandEncoderWithDescriptor:")
            .count();
        assert_eq!(made, 2, "the skipped pass made no encoder");
        assert_eq!(p.call_log().calls_to("split").len(), 2);
        assert_eq!(p.world().committed().len(), 3, "committed in its parts");
        posted.lock().clear();
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_COMPLETE, 350, RGFX_CB_OK)]);
    }
}
