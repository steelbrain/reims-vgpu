//! The command-buffer decoder: it carries out one
//! [`RGFX_CMD_COMMAND_BUFFER`](reims_vgpu_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
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
//! **Submission.** A buffer committed by `commitAndWaitUntilSubmitted` asks for its `CB_SUBMITTED`
//! (`SUBMITTED`), at which that call returns in the guest. Metal's returns once the buffer is
//! submitted (measured on the 5300M, the 630 and the M3: about 0.3 ms behind a 100 ms buffer, the
//! buffer then `Committed` or `Scheduled` by a race, and at the signal for a buffer that waits on
//! an event or is behind one on its queue; docs/graphics-kext/dgpu-oracle.md, Q7). The host posts
//! it as soon as it takes a buffer that waits for no event, follows no deferred copy of a device
//! cache and has none before it on its queue that waits for one and is not yet scheduled
//! ([`Held`]), before it walks the buffer's ops ([`Take`]) and before the fills of device caches
//! its begin makes at once; a GPU wait for a buffer of another queue ([`super::surface_order`])
//! holds back neither its `CB_SUBMITTED` nor its `CB_SCHEDULED`, which then goes at its commit
//! ([`Line`]), as a Mac's order of IOSurface accesses holds back no record (Q16), and nor does a
//! fill made at once, which waits for nothing of the guest's to come (a new surface's first fill,
//! at its first use, takes milliseconds: WindowServer's capture of a window list for Dock waited
//! about 4 ms in `commitAndWaitUntilSubmitted` for one on the 780M host, 11-08). Its ticket in the
//! commit order is issued first ([`super::sequencer`]), so every command buffer its process sends
//! after it, on any queue, is committed after the buffer's commit, and the platform orders the
//! process's later work after it as it orders by commit; another process's buffers wait for no
//! ticket of its, as they name none of its tracked resources. A buffer whose walk then faults has
//! its failure at its completion, as Metal reports a command buffer's errors. Any other buffer gets
//! it at its scheduling, once the waits are satisfied: Metal orders a tracked resource across
//! queues behind what it has submitted, not behind a buffer its queue holds (measured on the M3
//! through the decoder, a later commit on another queue ran first), and a deferred copy of a device
//! cache waits for buffers begun before it, which may wait so. Either way it goes once, ahead of
//! the buffer's `CB_SCHEDULED` and `CB_COMPLETE` ([`Submitted`]).
//!
//! **An answer on the doorbell's thread.** A buffer that gets its `CB_SUBMITTED` at its take may
//! get it on the thread of the guest's doorbell write instead, before its queue's service thread
//! takes it off its ring (`reims_vgpu_graphics`'s `Device::mmio_write_taking`), where nothing need be
//! waited for or allocated there ([`Decoders::answer_at_doorbell`]): it waits for no copy of a
//! device cache it uses, and every lock the answer takes is tried. One whose begin fills a cache
//! is answered at its take, before the fill: a doorbell cannot know the fill will be made at once,
//! as another queue's begin before the take may count a use of the cache, which defers it. Its turn in its queue's lane
//! and its ticket in the commit order are taken there, in the order its begin would take them,
//! and kept on its queue with its mark in that order ([`Answered`]), then its `CB_SUBMITTED` is
//! posted. Its begin, on the service thread, takes them in place of its own
//! ([`GraphicsDecoder::begin_taken`]), makes its command buffer and hands its walk and commit to
//! the lane as it does any buffer's, and posts no second `CB_SUBMITTED`; so its commit, and every
//! commit it orders, comes where it would had the answer been posted at its begin. One never begun
//! (its queue gone, the device reset) lets them go with its queue's entry. A queue's command
//! buffers made and not completed are counted ([`InFlight`]), and one is answered there only while
//! its queue holds fewer than [`TAKE_IN_FLIGHT_MAX`], so the platform's `commandBuffer`, which
//! waits while a queue holds 64, does not wait at its begin.
//!
//! **Scheduling.** A buffer that gets its `CB_SUBMITTED` at its take (it asks for one and waits
//! for nothing) gets its `CB_SCHEDULED` there too, right after it, where no buffer begun before it
//! on its queue owes its own ([`Line::owes`]), so a queue's records keep their order; one answered
//! on a doorbell's thread gets it there where the same holds, and it then needs no submission
//! handler unless it holds something. A buffer that does not ask keeps its `CB_SCHEDULED` to its
//! scheduling: posted at its take, it would need its ticket in the commit order there, as a
//! `CB_SUBMITTED` there does, and a walk held on one of a process's queues would then hold up the
//! process's other queues' commits ([`Begun::scheduled_at_take`]). A thread that waits for it to
//! be scheduled asks for it (`SCHEDULED_ASK`), and where it would have got the record at its take
//! had it asked for its `CB_SUBMITTED`, the ask posts it, its ticket issued first ([`Askable`]):
//! then a held walk holds up the process's other queues' later commits, as after a `CB_SUBMITTED`,
//! only once a thread waits for it. Metal makes such a buffer `Scheduled` within a fraction of a millisecond of its
//! commit, whatever its queue runs on the GPU before it: on the M3 Max (macOS 27) the scheduled
//! handler of a buffer on an idle queue runs about 10 us after `commitAndWaitUntilSubmitted`
//! returns, and behind a 120 ms buffer `waitUntilScheduled` returns in 0.1 ms; commit to the
//! scheduled handler was 0.20 ms on the AMD 5300M and 0.05 ms on Apple's paravirtualized device
//! (11-03, `cws_probe`). Waiting for the platform's submission put it at 0.39 ms on the RTX 5080
//! host (car 12, `submit_probe`), 0.26 ms of it from the take (11-03). Any other buffer gets it at
//! its commit, behind an order on the GPU ([`Line`]), or at its scheduling.
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
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use reims_vgpu_graphics::clock::runtime;
use reims_vgpu_graphics::ops::blit::{
    RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER, RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER,
    RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER, RGFX_OP_BLIT_UPDATE_FENCE, RGFX_OP_BLIT_WAIT_FOR_FENCE,
};
use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE, RGFX_OP_COMPUTE_ENCODE_START_ELSE,
    RGFX_OP_COMPUTE_ENCODE_START_IF, RGFX_OP_COMPUTE_ENCODE_START_WHILE,
    RGFX_OP_COMPUTE_UPDATE_FENCE, RGFX_OP_COMPUTE_WAIT_FOR_FENCE,
};
use reims_vgpu_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER,
    RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
    RGFX_OP_RENDER_UPDATE_FENCE, RGFX_OP_RENDER_USE_RESOURCES, RGFX_OP_RENDER_USE_RESOURCES_STAGES,
    RGFX_OP_RENDER_WAIT_FOR_FENCE, RgfxOpDrawPrimitives, RgfxOpSetVertexBuffer,
};
use reims_vgpu_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_COMPUTE_PASS_FLAG_DISPATCH_TYPE, RGFX_OP_BODY_OFFSET,
    RGFX_OP_COMMIT_ORDER, RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_ENCODER_BEGIN_COMPUTE,
    RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_BEGIN_VIDEO, RGFX_OP_ENCODER_END,
    RGFX_OP_GROUP_BLIT, RGFX_OP_GROUP_COMPUTE, RGFX_OP_GROUP_FRAMING, RGFX_OP_GROUP_RENDER,
    RGFX_OP_GROUP_VIDEO, RGFX_OP_GUEST_TIMES, RGFX_OP_SIGNAL_EVENT, RGFX_OP_SUBMITTED,
    RGFX_OP_SURFACE_ORDER, RGFX_OP_SURFACE_USE, RGFX_OP_UPLOADS_WRITTEN, RGFX_OP_WAIT_EVENT,
    RGFX_PASS_FLAG_DEFAULT_RASTER_SAMPLE_COUNT, RGFX_PASS_FLAG_RATE_MAP,
    RGFX_PASS_FLAG_RT_ARRAY_LENGTH, RGFX_PASS_FLAG_RT_SIZE, RGFX_PASS_FLAG_SAMPLE_POSITIONS,
    RGFX_PASS_FLAG_VISIBILITY_RESULT, RGFX_PASS_MAX_SAMPLE_POSITIONS,
    RGFX_RENDER_PASS_COLOR_OFFSET, RgfxCommandBuffer, RgfxComputePassDesc, RgfxOpEvent,
    RgfxOpHeader, RgfxOpSurfaceUse, RgfxOpVideoPass, RgfxRenderPassColorAttachment,
    RgfxRenderPassDesc, RgfxRenderPassRateMap, RgfxRenderPassSamplePositions,
    RgfxRenderPassVisibility, RgfxSurfaceOrderWait, RgfxUploadWritten, op_group,
};
use reims_vgpu_graphics::wire::{
    RGFX_EVT_CB_SCHEDULED, RGFX_EVT_CB_SUBMITTED, RGFX_EVT_EVENT_SIGNALLED, RgfxCbScheduled,
    RgfxCbSubmitted, RgfxEventSignalled, RgfxMsgHeader,
};
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_vulkan::core::platform::caps::{DeviceCaps, SAMPLE_COUNTS};
use reims_vgpu_vulkan::core::platform::command::{
    CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo, KeptWhole, PartCut,
    ScheduledHandler, WorkNames,
};
use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::event::{EventRef, SharedEvent as _};
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, MonoTime};
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder, VertexDraw};
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::{ClearColor, MtlSamplePosition};
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentSet, ComputePassDesc, DepthAttachmentSet, PassSet, StencilAttachmentSet,
};
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;
use zerocopy::{FromBytes, IntoBytes};

use super::super::completion::{cb_complete, cb_failed, cb_unlanded};
use super::commit_order::Until;
use super::completions::{CompletionTurn, Completions, Post};
use super::helpers::{Hand, Slots};
use super::lane::{Lane, Turn};
use super::sequencer::Ticket;
use super::shares::FanOut;
use super::split::{Estimate, PACE_WAIT_MOST, Parts, Progress, QueueRate, SplitHold};
use super::surface_cache::shapes::ShapeRef;
use super::surface_cache::{Reach, Uses};
use super::surface_order::{AheadEntry, Marks, Signalled};
use super::tables::{FOUND_A_HOLD, QueueCounts};
use super::{
    DecodeError, Decoders, Drain, FinishedSink, GraphicsDecoder, ObjHandle, ScheduledSink,
    SignalSink, SubmittedSink, Then, blit, compute, context_of, op_fault, read, refused, refusing,
    render, video,
};

/// The op stream runs under a live raise handler (the ring loop's), so an encode's
/// raise returns for the handler to catch.
const ON: OnFault = OnFault::Return;

/// How many names of its op stream that its view has not found yet a command buffer's walk finds
/// at its start ([`GraphicsDecoder::find_named`]): every object of a few dozen dispatches or draws,
/// and no more of a longer buffer's, whose walk finds the rest as it reaches them.
const NAMES_AT_ONCE: usize = 128;

/// How many names of its op stream, found already or not, a command buffer's walk reads at its
/// start at most ([`GraphicsDecoder::find_named`]): a render pass names its pipeline and buffers
/// again at each draw, and the read of a long one stops here.
const NAMES_READ: usize = 1024;

/// `MTLRenderStageFragment`: a render encoder's update of the writers' fence covers its fragment
/// stage and every stage before it, its render targets' writes among them.
const RENDER_STAGE_FRAGMENT: u64 = 0x2;

/// What a command buffer's op stream does with events: the shared events it signals, whose values
/// go to the guest as the GPU reaches them or once it finishes ([`Signals`]), the events it
/// waits for, and the decodes of its video encoders, whose sessions' events it waits for too.
#[derive(Default)]
struct EventOps {
    /// Each (shared event, value) the buffer signals, in stream order.
    signals: Finisher,
    /// Each (event, value) the buffer waits for, in stream order.
    waits: Vec<(u64, u64)>,
    /// Whether each of its decodes' pictures landed: the buffer completes failed where one did
    /// not ([`video::Landed`]).
    landings: Vec<Arc<video::Landed>>,
}

/// A buffer's [`Signals`], made at its first signal of a shared event and held for its completion
/// handler, which posts what is left of them ([`Signals::finish`]). Dropped without the handler
/// having run (a walk that faulted, which a stand-in replaces; a buffer never committed), it lets
/// their sink go, posting nothing more, so a notification that comes later posts nothing for the
/// buffer, and one never reached holds no ring.
#[derive(Default)]
struct Finisher(Option<Arc<Signals>>);

impl Finisher {
    /// The buffer's signals, made at the first, their values going to `sink`, and to the other
    /// holders of the events' names through `fan_out`.
    fn signals(&mut self, sink: Option<&SignalSink>, fan_out: &Arc<dyn FanOut>) -> &Arc<Signals> {
        self.0.get_or_insert_with(|| {
            Arc::new(Signals(Mutex::new(Posting {
                sink: sink.cloned(),
                fan_out: Arc::clone(fan_out),
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
/// ([`notify_listener`](reims_vgpu_vulkan::core::platform::event::SharedEvent::notify_listener)): where
/// the GPU gets to it in the buffer, as Metal's value and listeners move there (measured:
/// docs/graphics-kext/dgpu-oracle.md, Q18). So a guest thread that waits on the CPU for it before
/// it sets a value the buffer waits for later is released, where a post at the buffer's end would
/// never come. The buffer's completion handler posts every one not yet posted, a signal at the
/// buffer's end among them, ahead of its completion and whatever its outcome, then lets the sink
/// go: a notification that comes later posts nothing. A value already reached when its
/// notification is asked for goes at once: the event is at it, whatever raised it. A signal of a
/// shared event whose create the host refused goes with the completion wherever it lies
/// ([`GraphicsDecoder::note_refused_signal`]). Each post goes to the other processes that hold
/// the event's name too, as it is posted ([`super::shares`]): whether the event has a name is
/// read then, so a buffer walked before the name was given posts its later values to them.
struct Signals(Mutex<Posting>);

/// [`Signals`]' state.
struct Posting {
    /// Where the values go: none without a sink, and none once the completion handler posted.
    sink: Option<SignalSink>,
    /// Where they go besides: the other holders of an event's name.
    fan_out: Arc<dyn FanOut>,
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
        let Posting {
            sink,
            fan_out,
            signals,
        } = &mut *posting;
        if let (Some(sink), Some((obj_id, value, posted))) = (sink.as_ref(), signals.get_mut(at))
            && !*posted
        {
            *posted = true;
            sink(
                event_signalled(*obj_id, *value),
                fanned(fan_out, *obj_id, *value),
            );
        }
    }

    /// The buffer finished: posts each signal not yet posted, in stream order, and lets the sink
    /// go.
    fn finish(&self) {
        let mut posting = self.0.lock();
        let Some(sink) = posting.sink.take() else {
            return;
        };
        let Posting {
            fan_out, signals, ..
        } = &mut *posting;
        for (obj_id, value, posted) in signals {
            if !*posted {
                *posted = true;
                sink(
                    event_signalled(*obj_id, *value),
                    fanned(fan_out, *obj_id, *value),
                );
            }
        }
    }
}

/// What follows the post of shared event `obj_id`'s `value` to its own holder: its post to the
/// other holders of the event's name through `fan_out`, once the guest's ring holds the holder's
/// own, so no other process has a value before the process that signalled it.
fn fanned(fan_out: &Arc<dyn FanOut>, obj_id: u64, value: u64) -> super::Then {
    let fan_out = Arc::clone(fan_out);
    Box::new(move || fan_out.reached(obj_id, value))
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

/// A command buffer's `CB_SCHEDULED`, posted once: by the first of the answer on a doorbell's
/// thread or the host's take of it, for a buffer that gets its `CB_SUBMITTED` there and has none
/// before it on its queue that owes its own ([`Begun::post_at_take`]); its commit, behind an order on the GPU
/// ([`Line`]); its submission handler; or, for a buffer the platform failed before it took its
/// work, which runs no submission handler, its completion handler ahead of its completion, so each
/// buffer's comes before its `CB_COMPLETE`.
struct Scheduled {
    sink: ScheduledSink,
    record: RgfxCbScheduled,
    /// Whether it is posted, held across the post.
    posted: Mutex<bool>,
    /// Its queue's count of the begun buffers that owe their record ([`Line::owes`]), which it
    /// leaves once it is posted, or as it goes unposted; `None` where it was posted already as it
    /// was made.
    owed: Option<Arc<AtomicUsize>>,
}

impl Scheduled {
    /// Posts the record unless it is posted already, then leaves its queue's count of owed
    /// records: a buffer that finds none owed after it finds its record on the ring before its
    /// own.
    fn post(&self) {
        let mut posted = self.posted.lock();
        if !*posted {
            (self.sink)(self.record);
            *posted = true;
            if let Some(owed) = &self.owed {
                owed.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    /// Whether the record is posted.
    fn is_posted(&self) -> bool {
        *self.posted.lock()
    }
}

impl Drop for Scheduled {
    /// A record never posted (its buffer never committed, or failed before it was taken, its
    /// `CB_COMPLETE` standing for both) is owed no more.
    fn drop(&mut self) {
        if !*self.posted.get_mut()
            && let Some(owed) = &self.owed
        {
            owed.fetch_sub(1, Ordering::AcqRel);
        }
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
/// its `CB_SCHEDULED` there too where it gets that, else at its commit; so does every buffer of
/// its queue committed while one of them is not yet taken by the platform. A buffer that waits for
/// an event, or follows a deferred copy of a device cache, or is behind one on its queue, keeps
/// its records at its scheduling ([`Held`]).
///
/// The records of a queue go in its commit order: a buffer whose records go at its commit first
/// posts those of every buffer committed before it on its queue not yet posted, whose order on the
/// queue is fixed and none of which waits for anything by then ([`Line::post_through`]).
///
/// A buffer that asks for its `CB_SUBMITTED` and waits for nothing, and finds no buffer begun
/// before it on its queue that owes its `CB_SCHEDULED` ([`Self::owes`]), gets its records at its
/// take, or at its doorbell, whatever its queue's order on the GPU ([`Begun::post_at_take`]).
#[derive(Default)]
pub(crate) struct Line {
    /// The committed buffers' records not seen posted, oldest first.
    line: Mutex<VecDeque<Records>>,
    /// The begun buffers whose `CB_SCHEDULED` a waiting thread's ask may post ([`Askable`]),
    /// oldest first, each until its record is posted, or it ends uncommitted.
    asks: Mutex<VecDeque<Arc<Askable>>>,
    /// How many committed buffers of the queue that wait on the GPU for a buffer of another queue
    /// the platform has not yet taken.
    ordered: Arc<AtomicUsize>,
    /// How many begun buffers of the queue owe their `CB_SCHEDULED`: counted from the begin until
    /// the record is posted, or goes unposted ([`Scheduled`]).
    owed: Arc<AtomicUsize>,
}

impl Line {
    /// Whether a buffer begun before now on the queue owes its `CB_SCHEDULED`, so a later
    /// buffer's would come before it.
    pub(crate) fn owes(&self) -> bool {
        self.owed.load(Ordering::Acquire) > 0
    }

    /// How many committed buffers of the queue that wait on the GPU for a buffer of another queue
    /// the platform has not yet taken.
    #[cfg(test)]
    pub(crate) fn ordered(&self) -> usize {
        self.ordered.load(Ordering::Acquire)
    }

    /// The count of the queue's begun buffers that owe their `CB_SCHEDULED` ([`Self::owes`]).
    #[cfg(test)]
    pub(crate) fn owed(&self) -> &AtomicUsize {
        &self.owed
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

    /// Notes `askable`, a buffer just begun whose `CB_SCHEDULED` an ask may post, last in the
    /// queue's order; those at the front whose records went, or that ended uncommitted, leave
    /// first. A queue's records are posted in its order, so its asks close in it, but for a buffer
    /// that ends uncommitted, which leaves once those before it have: an ask of one that closed
    /// does nothing, here or gone ([`Askable::ask`]). Only the front is looked at: the retain over
    /// every open ask, which took each one's lock (and its record's) on every begin, took 0.47 µs
    /// of a Super Resolution buffer's begin on the Iris Xe host (**measured** 2026-11-14, laps of
    /// the begin, gpre m3-max-31335).
    fn askable(&self, askable: Arc<Askable>) {
        let mut asks = self.asks.lock();
        while asks.front().is_some_and(|a| !a.open()) {
            asks.pop_front();
        }
        asks.push_back(askable);
    }

    /// How many asks the queue keeps (for the tests).
    #[cfg(test)]
    fn asks_kept(&self) -> usize {
        self.asks.lock().len()
    }

    /// The buffer at `seqno`, `cb_obj_id`, where an ask may still post its `CB_SCHEDULED`.
    fn asked(&self, seqno: u64, cb_obj_id: u64) -> Option<Arc<Askable>> {
        let asks = self.asks.lock();
        (asks.iter())
            .find(|a| a.seqno == seqno && a.cb_obj_id == cb_obj_id)
            .cloned()
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

/// A begun command buffer whose `CB_SCHEDULED` a thread of its process that waits for it to be
/// scheduled may ask for (`SCHEDULED_ASK`, [`GraphicsDecoder::ask_scheduled`]): one that would get
/// it at its take had it asked for its `CB_SUBMITTED` ([`Begun::askable`]). The ask posts it at
/// once, its ticket in the commit order issued first and given back at its commit, as an answer at
/// its take does: a buffer its process sends once the record is posted is committed after it.
/// So a walk held on one of a process's queues holds up the process's other queues' commits only
/// once a thread has asked to wait for it; until then the buffer keeps its record to its
/// scheduling, holding none of them up.
pub(crate) struct Askable {
    /// Its message's seqno, and the guest's id of the buffer.
    seqno: u64,
    cb_obj_id: u64,
    /// Its context, its ticket's.
    context: Option<u32>,
    /// Where it is.
    state: Mutex<Asking>,
}

/// Where an [`Askable`] buffer is.
enum Asking {
    /// Begun and not yet committed, its record unposted.
    Begun(Arc<Scheduled>),
    /// Asked before its commit: its record posted, its ticket out until its commit. Nothing reads
    /// the ticket: it is held for its drop, which gives it back to the commit order where
    /// [`Askable::end`] leaves this state, at the buffer's commit or as it ends uncommitted.
    Asked { _ticket: Option<Ticket> },
    /// Committed before an ask: an ask posts its record at once, with no ticket, as its commit
    /// already has its place.
    Committed(Arc<Scheduled>),
    /// Ended: asked and committed, or never committed (its stand-in's completion stands for its
    /// record).
    Ended,
}

impl Askable {
    /// Whether an ask may still post its record, or it holds its ticket.
    fn open(&self) -> bool {
        match &*self.state.lock() {
            Asking::Begun(_) | Asking::Asked { .. } => true,
            Asking::Committed(scheduled) => !scheduled.is_posted(),
            Asking::Ended => false,
        }
    }

    /// A thread waits for it to be scheduled: before its commit, its ticket in the commit order
    /// is issued, then its `CB_SCHEDULED` posted; after it, its record is posted. Once asked, or
    /// once it ended, an ask does nothing.
    fn ask(&self, sequencer: &Arc<super::sequencer::Sequencer>) {
        let scheduled = {
            let mut state = self.state.lock();
            match std::mem::replace(&mut *state, Asking::Ended) {
                Asking::Begun(scheduled) => {
                    *state = Asking::Asked {
                        _ticket: sequencer.issue(self.context, false),
                    };
                    scheduled
                }
                Asking::Committed(scheduled) => scheduled,
                other => {
                    *state = other;
                    return;
                }
            }
        };
        scheduled.post();
    }

    /// Its commit is done (`committed`), or it ends uncommitted: a ticket an ask issued is given
    /// back, and an ask from now on posts its record only where it was committed.
    fn end(&self, committed: bool) {
        let mut state = self.state.lock();
        let ended = match std::mem::replace(&mut *state, Asking::Ended) {
            Asking::Begun(scheduled) if committed => Asking::Committed(scheduled),
            _ => Asking::Ended,
        };
        *state = ended;
    }
}

/// The most command buffers a queue may hold made and not completed for its next to be answered
/// on a doorbell's thread ([`Decoders::answer_at_doorbell`]): Metal's `commandBuffer`, and the
/// Vulkan platform's, wait while a queue holds 64, and a buffer leaves the count as its completion
/// handler ends, which may come before the platform lets its place go.
const TAKE_IN_FLIGHT_MAX: usize = 48;

/// A command buffer's place in its queue's count of the host's command buffers made on the queue
/// and not completed, which it leaves when dropped: with its completion handler, once that has run
/// ([`leaving`]), or with the buffer where it is never committed. From its commit on
/// ([`Self::commits`]) it holds a place in the queue's count of those committed too
/// ([`QueueCounts::out`](super::tables::QueueCounts::out)), which it leaves with the other.
struct InFlight {
    count: Arc<AtomicUsize>,
    out: Arc<AtomicUsize>,
    committed: bool,
}

impl InFlight {
    /// A place in `count`, a queue's count of command buffers made and not completed, whose count
    /// of those committed is `out`.
    fn new(count: &Arc<AtomicUsize>, out: &Arc<AtomicUsize>) -> InFlight {
        count.fetch_add(1, Ordering::AcqRel);
        InFlight {
            count: Arc::clone(count),
            out: Arc::clone(out),
            committed: false,
        }
    }

    /// The place, its command buffer about to be committed: in its queue's count of committed
    /// command buffers too, until dropped.
    fn commits(mut self) -> InFlight {
        if !self.committed {
            self.out.fetch_add(1, Ordering::AcqRel);
            self.committed = true;
        }
        self
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.committed {
            self.out.fetch_sub(1, Ordering::AcqRel);
        }
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

/// `handler`, after which the buffer leaves its queue's count (`in_flight`).
fn leaving(handler: CompletedHandler, in_flight: InFlight) -> CompletedHandler {
    Box::new(move |info: CompletedInfo| {
        handler(info);
        drop(in_flight);
    })
}

/// Marks `part`, a command buffer's part before its last about to be committed, to join the next
/// one's completion where its walk's `s` holds no event's wait or signal in it
/// ([`Splitting::events`]): its completion is then observed by nothing but its handler from
/// [`Parts::part_handler`], which notes its outcome for the last part's and lets its place among
/// its queue's command buffers go (the module docs of [`super::split`], "Its parts'
/// completions"). Whether it marked it: `false` where the platform took no mark.
fn joins_next<P: Platform>(part: &P::CommandBuffer, s: &Splitting) -> bool {
    !s.events && part.set_completion_joins_next(true)
}

/// A command buffer's walk where it may be committed in parts ([`super::split`]).
struct Splitting {
    /// Its context.
    context: u32,
    /// Its queue's count of command buffers made and not completed ([`InFlight`]).
    in_flight: Arc<AtomicUsize>,
    /// Its queue's count of those committed ([`QueueCounts::out`](super::tables::QueueCounts::out)).
    out: Arc<AtomicUsize>,
    /// When the walk of the part it is walking began.
    since: MonoTime,
    /// Its context's gate, from its first part's commit to its last ([`SplitHold`]).
    hold: Option<SplitHold>,
    /// What its parts before the last found ([`Parts`]).
    parts: Option<Arc<Parts>>,
    /// The platform keeps it whole from now on.
    whole: bool,
    /// The part being walked into its current command buffer waits for or signals an event: at
    /// its begin, for a cache's fill or another queue's buffer ([`Begun`]'s `waits_at_begin`),
    /// or by an op, so its completion is observed: committed unmarked ([`joins_next`]).
    events: bool,
    /// Its queue's rate of GPU time to walk time, which its parts' completions update.
    rate: Arc<QueueRate>,
    /// Its queue's marks: a wait for its parts ends as the queue ends.
    marks: Arc<Marks>,
    /// Its walk waits for its parts' completions where they run far ahead of the GPU
    /// ([`super::split`], "How far ahead of the GPU"): not once it waited for or signalled an
    /// event by an op, whose part's completion may wait for a buffer that is committed only after
    /// its last part, nor once a buffer of its context waited for its last commit, a wait was
    /// cut short or a part failed.
    paced: bool,
}

/// A chunk of a command buffer dealt to two threads, walked ([`GraphicsDecoder::walk_dealt`]).
struct Walked<P: Platform> {
    /// The command buffer it was walked into, apart from the walk's current one, with its place
    /// among its queue's command buffers (none for the buffer's own, whose place goes on to its
    /// last part); `None` where it was walked into the walk's current one.
    cb: Option<(P::CommandBuffer, Option<InFlight>)>,
    /// The walk's outcome.
    outcome: RaiseResult<Result<(), DecodeError>>,
    /// The blocks the walk saw written through textures over laid-out device caches or their
    /// wide homes ([`GraphicsDecoder::reconcile_laid`]).
    written: Vec<u64>,
    /// The shapes the walk saw written ([`GraphicsDecoder::note_shape`]).
    shapes: Vec<(u64, ShapeRef)>,
    /// The blocks whose caches' buffers the walk named an object over
    /// ([`GraphicsDecoder::note_over_cache`]).
    over: Vec<u64>,
    /// Whether the platform refused a cache's buffer its memory at a use the walk made
    /// ([`GraphicsDecoder::out_of_memory`]).
    out_of_memory: bool,
    /// The first compute and render pipeline states its encoders set, 0 for none
    /// ([`GraphicsDecoder::pipeline_set`]).
    firsts: (u64, u64),
    /// From the hand-over to its walk's start.
    began: Duration,
    /// Its walk.
    walk: Duration,
}

impl<P: Platform> Walked<P> {
    /// Whether the whole chunk was carried out: no fault, and no stop at a malformed op.
    fn walked(&self) -> bool {
        matches!(self.outcome, Ok(Ok(())))
    }
}

/// The part of a walk dealt to two threads that the platform keeps whole by how Metal's GPU
/// fails it from one of its encoders on ([`PartCut::Fails`]), which every later chunk's command
/// buffer is folded into ([`GraphicsDecoder::commit_dealt`]).
enum Kept<P: Platform> {
    /// None: each chunk is committed as a part.
    No,
    /// One apart from the walk's current command buffer.
    Apart(P::CommandBuffer),
    /// The walk's current command buffer.
    Here,
}

/// A walk dealt to two threads, on the lane's thread ([`GraphicsDecoder::walk_dealt`]).
struct Deal<'a, P: Platform> {
    /// Its chunks, each a range of its op stream.
    chunks: &'a [Range<usize>],
    /// Which of them the helper walks ([`helper_chunks`]).
    helper: &'a [bool],
    /// The slots the helper publishes its chunks to.
    slots: &'a Slots<Walked<P>>,
    /// The helper's chunks, handed over.
    hand: &'a Hand,
    /// The command buffers of this thread's chunks after its first, each with its place among
    /// its queue's command buffers, in their order.
    ours: VecDeque<(P::CommandBuffer, InFlight)>,
    /// Its queue's count of command buffers made and not completed ([`InFlight`]).
    in_flight: Arc<AtomicUsize>,
    /// Its queue's count of those committed ([`QueueCounts::out`](super::tables::QueueCounts::out)).
    out: Arc<AtomicUsize>,
    /// This thread's chunks walked and not yet committed or folded, at their index.
    own: Vec<Option<Walked<P>>>,
    /// The next chunk to commit or fold.
    next: usize,
    /// The part kept whole, where one is.
    kept: Kept<P>,
    /// A part was committed.
    parts_out: bool,
    /// The place among its queue's command buffers of the walk's current one, but for the
    /// buffer's own, whose place goes on to its last part.
    place: Option<InFlight>,
    /// When the helper's chunks were handed over.
    handed: MonoTime,
    /// Its queue, for the trace.
    queue: u64,
    /// Its message's seqno, for the trace.
    seqno: u64,
    /// What the trace gives of it.
    stats: DealStats,
}

/// A chunk dealt to the helper ([`GraphicsDecoder::walk_chunks`]): its index among the buffer's,
/// its ops among those handed over, and its command buffer with its place among its queue's.
type Theirs<P> = (
    usize,
    Range<usize>,
    (<P as Platform>::CommandBuffer, InFlight),
);

/// What a walk dealt to two threads took, which the trace gives (`cbpar`).
#[derive(Default)]
struct DealStats {
    /// The helper's chunks were taken back.
    back: bool,
    /// This thread's walk of its chunks.
    walk_here: Duration,
    /// From the hand-over to the helper's walk of its first chunk.
    helper_began: Option<Duration>,
    /// The helper's walk of its chunks taken.
    walk_helper: Duration,
    /// This thread's waits for the helper's chunks.
    waited: Duration,
    /// The helper's chunks this thread waited for.
    waits: u64,
}

/// The op stream of the `COMMAND_BUFFER` `message` whose fixed part is `m`: all of the message's
/// ops, which its `op_bytes` must count. The bundle writes a buffer's message as its fixed part
/// and its whole op stream, `op_bytes` the stream's length (`RgfxEncoder::finish`), so a message
/// that disagrees is not one the guest made: carrying out the ops a shorter `op_bytes` covers
/// would drop the rest without a word and answer the buffer completed.
pub(super) fn ops_of<'m>(message: &'m [u8], m: &RgfxCommandBuffer) -> Option<&'m [u8]> {
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
    /// Its count among the users of the device caches it uses, until its commit.
    uses: Option<Uses<P>>,
    /// A fill of a device cache its begin made found the platform refused the cache's buffer its
    /// memory: it fails with Metal's `MTLCommandBufferErrorOutOfMemory`
    /// ([`GraphicsDecoder::out_of_memory`]).
    out_of_memory: bool,
    /// It holds `SURFACE_ORDER`: it signals its queue's surface timeline as it ends.
    order: bool,
    /// Its queue's marks in the surface order.
    marks: Arc<Marks>,
    /// It waits on the GPU for a buffer of another queue that has not finished.
    waits_for_order: bool,
    /// Its command buffer waits on the GPU for an event from its begin: a fill of a device cache
    /// it uses, or a buffer of another queue ([`Self::waits_for_order`]).
    waits_at_begin: bool,
    /// Its queue's records the host may post at their commit.
    line: Arc<Line>,
    /// Its queue's rate of GPU time to walk time ([`super::split`], "How far ahead of the
    /// GPU").
    rate: Arc<QueueRate>,
    /// Its `CB_SUBMITTED`, where it asks for one.
    submitted: Option<Arc<Submitted>>,
    /// Its `CB_SCHEDULED`, where there is a sink.
    scheduled: Option<Arc<Scheduled>>,
    /// No buffer begun before it on its queue owed its `CB_SCHEDULED` as it was begun.
    first: bool,
    /// Its hold of its queue, where it waits.
    held: Option<Arc<Held>>,
    /// A buffer before it on its queue holds the queue.
    behind: bool,
    /// Its context's drain.
    drain: Option<Arc<Drain>>,
    /// It holds a video encoder, so it is walked in its turn: its session's work starts as it is
    /// walked, and so in its queue's order.
    ordered: bool,
    /// Its queue's turns of their completion posts ([`super::completions`]).
    completions: Arc<Completions>,
    /// The blocks of its decode's output, where it is a decode alone that may post its completion
    /// at its decode ([`ahead_at_decode`]).
    ahead: Option<Vec<u64>>,
    /// It holds a compute encoder ([`Take::computes`]).
    computes: bool,
    /// The names of its work, which the platform keeps for the line of a lost device that loses
    /// it in flight ([`WorkNames`]), its walk noting the firsts of them.
    work_names: Arc<WorkNames>,
}

/// A command buffer's turn of its completion post, taken by the first of its completion handler
/// and its post at its decode ([`video::Landed`]), which posts its record.
type TurnCell = Arc<Mutex<Option<CompletionTurn>>>;

/// What a command buffer's completion handler ends besides its record: its turn of its completion
/// post ([`TurnCell`]), and, where it may post its completion at its decode, its decode's place
/// among those completed ahead of their ends ([`super::surface_order::CompletedAhead`]) and what
/// its end tells once that post is on its ring ([`BothDone`], [`super::AheadSinks::finished`]).
struct Ending<P: Platform> {
    turn: TurnCell,
    entry: Option<AheadEntry<P>>,
    after: Option<(Arc<BothDone>, FinishedSink)>,
}

impl<P: Platform> Ending<P> {
    /// The ending of a command buffer whose turn is `turn`, and that holds no decode completed
    /// ahead.
    fn of(turn: CompletionTurn) -> Self {
        Ending {
            turn: Arc::new(Mutex::new(Some(turn))),
            entry: None,
            after: None,
        }
    }
}

/// What runs once both a command buffer's completion posted at its decode is on its ring and its
/// platform end has come, run by the later of the two ([`GraphicsDecoder::post_at_decode`]): the
/// end's notice to what waits for the buffer, and the release of its context's drain, each after
/// the record, as for a buffer whose handler posts it.
#[derive(Default)]
struct BothDone(Mutex<Option<Option<Then>>>);

impl BothDone {
    /// One of the two came, with what it runs (`None` for the post): runs it and the other's,
    /// where the other came first.
    fn came(&self, then: Option<Then>) {
        let first = {
            let mut state = self.0.lock();
            match state.take() {
                Some(first) => first,
                None => {
                    *state = Some(then);
                    return;
                }
            }
        };
        for then in [first, then].into_iter().flatten() {
            then();
        }
    }
}

/// Whether the op stream `take` read may post its completion at its decode, ahead of its platform
/// end ([`video::Landed`]; the module docs of [`video`], "Completed at the decode"): a decode alone,
/// no other video op, no render, compute or blit work, and no event's wait or signal, so nothing
/// but the decode's picture is the guest's at its end.
fn ahead_at_decode(take: &Take) -> Option<(u64, u64)> {
    let alone = take.video
        && !take.work
        && !take.waits
        && !take.signals
        && take.video_ops == 1
        && take.decodes.len() == 1;
    alone.then(|| take.decodes[0])
}

impl<P: Platform> Begun<P> {
    /// It holds a video encoder, so it is walked in its turn.
    pub(crate) fn ordered(&self) -> bool {
        self.ordered
    }

    /// It holds a compute encoder, whose dispatches its pre-pass reads ([`super::prepass`]).
    pub(crate) fn computes(&self) -> bool {
        self.computes
    }

    /// Its `CB_SUBMITTED`, where it gets it at its take: it asks for one, waits for no event and
    /// follows no deferred copy of a device cache (a GPU wait for a buffer of another queue holds
    /// back nothing, [`Line`], and nor does a fill made at its begin: it gets it before the
    /// fill), and no buffer before it on its queue holds the queue. Any other gets it at its
    /// scheduling.
    pub(crate) fn submitted_at_take(&self) -> Option<&Arc<Submitted>> {
        (self.submitted.as_ref()).filter(|_| self.held.is_none() && !self.behind)
    }

    /// Its `CB_SCHEDULED`, where it gets it at its take: it gets its `CB_SUBMITTED` there
    /// ([`Self::submitted_at_take`]), and no buffer begun before it on its queue owed its own as
    /// it was begun, so the queue's records keep their order. Metal makes a buffer that waits for
    /// nothing `Scheduled` within a fraction of a millisecond of its commit, whatever runs on the
    /// GPU before it (the module doc). Any other gets it at its commit ([`Line`]), or at its
    /// scheduling: one that does not ask would need its ticket in the commit order at its take,
    /// as the `CB_SUBMITTED` there needs, so that another queue's buffer of its process begun
    /// after the record is committed after it, and a walk held on one queue would then hold up
    /// its process's other queues (`a_walk_held_in_a_draw_holds_up_no_other_queue`).
    fn scheduled_at_take(&self) -> Option<&Arc<Scheduled>> {
        (self.scheduled.as_ref()).filter(|_| self.first && self.submitted_at_take().is_some())
    }

    /// Its `CB_SCHEDULED`, where a thread that waits for it to be scheduled may ask for it there
    /// and then (`SCHEDULED_ASK`, [`Askable`]): it would get it at its take had it asked for its
    /// `CB_SUBMITTED` ([`Self::scheduled_at_take`]), and did not ask (one that did got it there,
    /// or has a buffer before it on its queue that owes its own).
    fn askable(&self) -> Option<&Arc<Scheduled>> {
        (self.scheduled.as_ref())
            .filter(|_| self.first && self.submitted.is_none())
            .filter(|_| self.held.is_none() && !self.behind)
    }

    /// Whether it gets a record at its take ([`Self::post_at_take`]), so its ticket in the commit
    /// order is to be issued first: where it gets its `CB_SUBMITTED` there.
    pub(crate) fn posts_at_take(&self) -> bool {
        self.submitted_at_take().is_some()
    }

    /// Posts what it gets at its take: its `CB_SUBMITTED` ([`Self::submitted_at_take`]), unless
    /// an answer on a doorbell's thread posted it, then its `CB_SCHEDULED`
    /// ([`Self::scheduled_at_take`]), unless that answer posted it too.
    pub(crate) fn post_at_take(&self) {
        if let Some(submitted) = self.submitted_at_take() {
            submitted.post();
        }
        if let Some(scheduled) = self.scheduled_at_take() {
            scheduled.post();
        }
    }
}

/// Command buffer `cb_obj_id`'s `CB_SCHEDULED` record, at its message's `seqno`.
fn scheduled_record(cb_obj_id: u64, seqno: u64) -> RgfxCbScheduled {
    RgfxCbScheduled {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_CB_SCHEDULED,
            size: size_of::<RgfxCbScheduled>() as u32,
            seqno,
        },
        cb_obj_id,
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
/// an event and is not yet scheduled; and, where the buffer begun now `waits` for an event (a
/// deferred copy of a device cache among them), its own hold of the queue, counted from now, which
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
    /// The stream holds a `SIGNAL_EVENT`.
    signals: bool,
    /// The stream holds a `SUBMITTED`.
    asks: bool,
    /// The buffers of other queues its first `SURFACE_ORDER` names, where it holds one.
    order: Option<Vec<RgfxSurfaceOrderWait>>,
    /// The buffers of its process's other queues its first `COMMIT_ORDER` names, which it is
    /// committed after ([`super::commit_order`]); empty where it holds none.
    commit_after: Vec<RgfxSurfaceOrderWait>,
    /// The stream holds a video encoder, whose session's work starts as it is walked.
    video: bool,
    /// The stream holds a render, compute or blit encoder, or an op of one.
    work: bool,
    /// The stream holds a compute encoder, whose dispatches its pre-pass reads
    /// ([`super::prepass`]).
    computes: bool,
    /// The stream holds a record of command predication (an `ENCODE_START_*` op), the only ones
    /// the walk sends: where it holds none, its command buffer is told so before any encoder
    /// (`CommandBuffer::holds_no_predication`), and keeps nothing for one.
    predicates: bool,
    /// The luma and chroma textures of each of its decodes, in stream order.
    decodes: Vec<(u64, u64)>,
    /// How many video ops (decodes and encodes) it holds.
    video_ops: usize,
    /// The `Managed` buffers its first `UPLOADS_WRITTEN` names, with the bytes the buffer wrote
    /// in each, which its begin brings to their GPU copies ([`GraphicsDecoder::uploads_written`]);
    /// empty where it holds none.
    written: Vec<RgfxUploadWritten>,
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
            RGFX_OP_SIGNAL_EVENT => out.signals = true,
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
            RGFX_OP_UPLOADS_WRITTEN if out.written.is_empty() => {
                let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
                let (written, _) = body.as_chunks::<{ size_of::<RgfxUploadWritten>() }>();
                out.written = (written.iter())
                    .filter_map(|w| RgfxUploadWritten::read_from_bytes(w).ok())
                    .collect();
            }
            RGFX_OP_ENCODER_BEGIN_VIDEO => out.video = true,
            RGFX_OP_ENCODER_BEGIN_COMPUTE => {
                out.work = true;
                out.computes = true;
            }
            RGFX_OP_ENCODER_BEGIN_RENDER | RGFX_OP_ENCODER_BEGIN_BLIT => out.work = true,
            RGFX_OP_COMPUTE_ENCODE_START_IF
            | RGFX_OP_COMPUTE_ENCODE_START_WHILE
            | RGFX_OP_COMPUTE_ENCODE_START_ELSE
            | RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE => {
                out.work = true;
                out.predicates = true;
            }
            // The device's diagnostic, read as it took the buffer: no work, and nothing taken.
            RGFX_OP_GUEST_TIMES => {}
            tag => match op_group(tag) {
                RGFX_OP_GROUP_RENDER | RGFX_OP_GROUP_COMPUTE | RGFX_OP_GROUP_BLIT => {
                    out.work = true;
                }
                RGFX_OP_GROUP_VIDEO => {
                    let body = &ops[RGFX_OP_BODY_OFFSET as usize..len];
                    out.decodes.extend(video::decode_textures(tag, body));
                    out.video_ops += 1;
                }
                _ => {}
            },
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
pub(super) fn op_bodies(mut ops: &[u8]) -> impl Iterator<Item = (u16, &[u8])> {
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

    /// The stream's `SURFACE_USE` ops, in stream order.
    fn uses(&self) -> impl Iterator<Item = RgfxOpSurfaceUse> + 'o {
        (op_bodies(self.ops))
            .filter(|&(tag, _)| tag == RGFX_OP_SURFACE_USE)
            .filter_map(|(_, body)| RgfxOpSurfaceUse::read_from_prefix(body).ok())
            .map(|(op, _)| op)
    }

    /// The blocks the stream's `SURFACE_USE` ops name, in stream order.
    fn used_blocks(&self) -> impl Iterator<Item = u64> + 'o {
        self.uses().map(|op| op.block_offset)
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
    /// Its `CB_SCHEDULED` was posted after it, no buffer begun before it on its queue owing its
    /// own; else its begin posts it where it may ([`Begun::post_at_take`]).
    scheduled: bool,
}

impl<P: Platform> Decoders<P> {
    /// A `SURFACE_DOWNLOAD` of the block at `block` on a doorbell's thread, the guest's vCPU
    /// (`reims_vgpu_graphics::Host::surface_downloaded`): whether it is carried out already, its
    /// `SURFACE_PAGES` settled ([`super::worker::Unsettled`]) and the caches answering it at once
    /// with nothing to copy and nothing to wait for ([`super::surface_cache`]), as
    /// [`super::GraphicsDecoder::try_surface_download`] would; each lock tried. Allocates and
    /// frees nothing.
    pub(crate) fn downloaded_at_doorbell(&self, block: u64) -> bool {
        let shared = &*self.0;
        shared.unsettled.try_any(std::iter::once(block)) == Some(false)
            && shared.surface_caches.try_downloaded(block)
    }

    /// Answers the `COMMAND_BUFFER` `message`, of `context`'s, at the head of its queue's ring,
    /// `CB_SUBMITTED` on a doorbell's thread, before its queue's service thread takes it (the
    /// module doc), where it gets that at its take ([`Begun::submitted_at_take`]) with nothing
    /// waited for or allocated on the way: its context's work is not refused, it names a command
    /// queue of its context, it asks for its `CB_SUBMITTED`, its last op, and waits for no event,
    /// nor for a copy of a device cache it uses (a fill it needs or one not yet done, or a copy
    /// asked before it that it follows, [`super::surface_cache::Caches::waits_for_nothing`]), it
    /// uses no block whose `SURFACE_PAGES` the surface worker is still carrying out
    /// ([`super::worker::Unsettled`]), which its begin waits for, waits for no buffer of another
    /// queue ([`super::surface_order`]), names in its `COMMIT_ORDER` only buffers begun
    /// ([`super::commit_order`]), no buffer before it holds its queue, its queue holds fewer than
    /// [`TAKE_IN_FLIGHT_MAX`] command buffers made and not completed and no answered buffer not yet
    /// begun, and each lock it takes is free, each tried. Its turn in its queue's lane and its
    /// ticket in the commit order are taken together, or neither, and kept on its queue
    /// ([`Answered`]), and the answer is posted through `post`, which posts a record's bytes,
    /// waits for nothing and answers whether it posted it: its `CB_SUBMITTED`, then, where that
    /// was posted and no buffer begun before it on its queue owes its `CB_SCHEDULED`
    /// ([`Line::owes`]), its `CB_SCHEDULED`, as its take would post it
    /// ([`Begun::post_at_take`]). False, with nothing taken, where it is not answered.
    pub(crate) fn answer_at_doorbell(
        &self,
        context: u32,
        message: &[u8],
        mut post: impl FnMut(&[u8]) -> bool,
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
            || shared.surface_caches.waits_for_nothing(glance.uses()) != Some(true)
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
            let posted = post(record.as_bytes());
            // Every buffer of the queue taken before it is begun, and counted among the owed
            // ([`Line::owes`]) where its record is still to come.
            let scheduled = posted
                && !counts.line.owes()
                && post(scheduled_record(m.cb_obj_id, m.header.seqno).as_bytes());
            *slot = Some(Answered {
                seqno: m.header.seqno,
                cb_obj_id: m.cb_obj_id,
                resets: shared.closed.resets(),
                mark,
                ranges,
                ticket,
                turn,
                posted,
                scheduled,
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
    /// refused (an attachment, a video session), or the platform made no encoder of it
    /// ([`unmade`]). No platform encoder is open, and its ops up to its end are dropped
    /// ([`refused`]).
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
    /// [`RGFX_CMD_COMMAND_BUFFER`](reims_vgpu_graphics::wire::RGFX_CMD_COMMAND_BUFFER)
    /// arm): opens a command buffer on the named queue, walks the op stream, and
    /// commits with a completion handler, which holds `drain` (its context's) until it has run.
    /// The IOSurfaces whose device caches it uses (its `SURFACE_USE` ops) are filled first where
    /// they hold an older generation, and it signals its queue's end event for their later copies
    /// ([`super::surface_cache`]). A buffer that asks for its `CB_SUBMITTED` and waits for nothing
    /// but fills made at its begin gets it before them and the walk, and its `CB_SCHEDULED` there
    /// too where none begun before it on its queue owes its own ([`Begun::post_at_take`]); any
    /// other at its scheduling.
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
        // Submitted and scheduled at once where it asks, unless it waits (for an event, or for a
        // deferred copy of a device cache) or a buffer before it on its queue does: then at its
        // scheduling; and scheduled at its take only where no buffer before it on its queue owes
        // its scheduling. Its begin posted them before its fills where it gets them here.
        // Its ticket in the commit order is issued first, so a buffer its process sends after the
        // answer is committed after this one ([`super::sequencer`]).
        if begun.posts_at_take() {
            self.issue_ticket();
            begun.post_at_take();
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

    /// The command buffer is committed (`committed`), or its message ends with it uncommitted: an
    /// ask of its `CB_SCHEDULED` gives its ticket back, and posts the record from now on only
    /// where it was committed ([`Askable`]).
    pub(super) fn end_ask(&self, committed: bool) {
        if let Some(asked) = self.asked.borrow_mut().take() {
            asked.end(committed);
        }
    }

    /// A `SCHEDULED_ASK` on the ring of queue `queue_obj_id`, taken by its service thread: a
    /// thread of the queue's process waits for the queue's command buffer `cb_obj_id`, at
    /// `cb_seqno`, to be scheduled. Where an ask may post its `CB_SCHEDULED` ([`Askable`]), it
    /// does, once its ticket in the commit order is issued where it is not yet committed; any
    /// other buffer is scheduled as it would have been.
    pub(crate) fn ask_scheduled(&self, queue_obj_id: u64, cb_seqno: u64, cb_obj_id: u64) {
        let Some(counts) = self.queue_counts(queue_obj_id) else {
            return;
        };
        if let Some(askable) = counts.line.asked(cb_seqno, cb_obj_id) {
            askable.ask(&self.shared.sequencer);
        }
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
    /// `CB_SUBMITTED` or its `CB_SCHEDULED` at its take, they are posted here, once its ticket in
    /// the commit order is issued, unless the answer posted them ([`Begun::post_at_take`]). Where
    /// the rest of its work goes to the lane (`to_lane`), the lane's idle thread is woken for it
    /// once its turn is taken ([`Lane::expect`]), so it wakes during the rest of the begin.
    pub(crate) fn begin_taken(
        &mut self,
        device: &P::Device,
        message: &[u8],
        queue: Option<u64>,
        to_lane: bool,
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
                    self.scheduled_at_doorbell.set(answered.scheduled);
                }
                None => *self.turn.borrow_mut() = Self::turn_on(counts),
            }
            if to_lane && let Some(lane) = &counts.lane {
                lane.expect();
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
        if begun.posts_at_take() {
            self.issue_ticket();
            begun.post_at_take();
        }
        Ok(Ok(begun))
    }

    /// Waits, with no lock held, for the turns in its queue's lane before the command buffer's
    /// (one taken before it), where it has one ([`super::lane`]), as a wait of its lane's job
    /// ([`Self::waiting`]).
    pub(crate) fn wait_turn(&self) {
        if let Some(turn) = self.turn.borrow().as_ref() {
            turn.lane().waiting(|| turn.wait());
        }
    }

    /// Runs `wait`, a wait of the command buffer's for what is committed before it, where it has a
    /// turn in its queue's lane as a wait of the lane's job ([`Lane::waiting`]): the lane starts
    /// its next buffer's walk meanwhile.
    fn waiting<R>(&self, wait: impl FnOnce() -> R) -> R {
        match self.lane() {
            Some(lane) => lane.waiting(wait),
            None => wait(),
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

    /// How many idle threads of queue `queue`'s lane a job woke as it came, how many a begin woke
    /// for a job about to come ([`Lane::expect`]), and how many wait for a job; `None` where the
    /// queue has no lane (for the tests).
    #[cfg(test)]
    pub(crate) fn lane_wakes(&self, queue: u64) -> Option<(usize, usize, usize)> {
        let lane = self.queue_counts(queue)?.lane.as_ref()?;
        Some(lane.wakes())
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
    /// `CB_SUBMITTED` and, where it waits for an event or a deferred copy of a cache, its hold of
    /// its queue ([`Begun`]); where it gets its records at its take, they are posted before its
    /// fills, its ticket in the commit order issued first
    /// ([`super::surface_cache::Caches::before_answering`]).
    pub(crate) fn begin_command_buffer(
        &self,
        device: &P::Device,
        message: &[u8],
        drain: Option<Arc<Drain>>,
    ) -> RaiseResult<Result<Begun<P>, DecodeError>> {
        // The op stream follows the message's fixed part; `op_bytes` must count all of it.
        let parsed = read::<RgfxCommandBuffer>(message)
            .and_then(|m| ops_of(message, &m).map(|ops| (m, ops, take(ops))));
        // Past a recovery of the device, its process's two-copy `Managed` textures have their GPU
        // copies filled again first, before anything of the buffer reaches the platform
        // ([`GraphicsDecoder::refill_managed`]).
        if let Some((m, _, _)) = &parsed {
            self.refill_managed(device, context_of(m.queue_obj_id));
        }
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
        // Counted where its process holds an escaped buffer over a copy of its pages, which the
        // runtime brings up to date at its commit ([`super::escape`]).
        self.escaped_copy_commit(context_of(m.queue_obj_id));
        // The bytes it wrote in its `Managed` chunks reach their GPU copies by its commit, before
        // its work, as a range message of each before it would bring them.
        self.uploads_written(&take.written);
        let Some(cb) = queue.command_buffer(ON)? else {
            return Ok(Err(DecodeError::Refused));
        };
        // Its walk sends no record of command predication where its stream holds none, before
        // any encoder of it or of a part cut from it ([`Take::predicates`]).
        if !take.predicates {
            cb.holds_no_predication();
        }
        // Its commits, and its parts', bring up to date its process's escaped copies, which its
        // work may reach through memory with no use naming them ([`super::escape`]).
        if let Some(escapes) = self.escapes(context_of(m.queue_obj_id)) {
            cb.reaches_escaped(&escapes);
        }
        // A lost device names it, and the first pipeline states its encoders set, where it loses
        // it in flight ([`WorkNames`]).
        let work_names = Arc::new(WorkNames::new(m.cb_obj_id));
        cb.names_work(&work_names);
        super::work::command_buffer(self.context);
        let in_flight = InFlight::new(&counts.in_flight, &counts.out);
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
        // What its work reaches of the caches beyond its uses: a decode's new output takes its
        // first fill as met, where the buffer holds no other work ([`Reach`]).
        let reach = Reach {
            context: self.context,
            work: take.work,
            video: take.video,
            decoded: match take.work {
                true => Vec::new(),
                false => self.decoded_whole(&take.decodes),
            },
        };
        let submitted = self.submitted(m.cb_obj_id, m.header.seqno, take.asks);
        // Owed by its queue from now until it is posted, after every buffer begun before it.
        let (scheduled, first) = self
            .scheduled(m.cb_obj_id, m.header.seqno, Some(&counts.line))
            .map_or((None, false), |(s, first)| (Some(s), first));
        // Where it gets its records at its take, they are posted before the fills its uses need,
        // once its ticket in the commit order is issued: a fill the host makes at once waits for
        // nothing of the guest's to come, so it holds back none of them, as a GPU wait for a
        // buffer of another queue does not ([`Begun::submitted_at_take`]). Only a buffer that
        // follows a deferred copy of a cache waits for buffers begun before it, which may wait
        // for the guest; the caches answer no such buffer here.
        let early = submitted.is_some() && !take.waits && counts.held.load(Ordering::Acquire) == 0;
        let mut answer = || {
            self.issue_ticket();
            if let Some(submitted) = &submitted {
                submitted.post();
            }
            if first && let Some(scheduled) = &scheduled {
                scheduled.post();
            }
        };
        let answer = early.then_some(&mut answer as &mut dyn FnMut());
        let super::surface_cache::Begin {
            waits: fill_waits,
            follows,
            out_of_memory,
            uses,
        } = (self.surface_caches).before_answering(device, &cb, &take.uses, &reach, answer)?;
        // The buffers of other queues it runs after, on the GPU, kept for the work the host starts
        // off it (`surface_order`): those its op names, and the decodes completed ahead of their
        // ends whose outputs it uses, which the guest's slots no longer name.
        let kept = &self.shared.completed_ahead;
        let order_waits = if take.order.is_some() || (kept.any() && !take.uses.is_empty()) {
            let now = self.shared.platform.clock().now_mono();
            let blocks = take.uses.iter().map(|u| u.block_offset);
            let ahead = kept.waits_for(m.queue_obj_id, blocks, now);
            if take.order.is_some() || !ahead.is_empty() {
                (self.shared.surface_order.lock()).before(
                    device,
                    m.queue_obj_id,
                    &cb,
                    (take.order.as_deref(), &ahead),
                    kept,
                    now,
                )?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let waits_for_order = !order_waits.is_empty();
        *self.order_waits.borrow_mut() = order_waits;
        // Those GPU waits hold back none of its records, nor its queue ([`Line`]), and nor do the
        // fills of caches made at its begin: only an event's wait, or a deferred copy's, does.
        let (behind, held) = hold(counts, follows || take.waits);
        // A decode alone over device caches may post its completion at its decode.
        let ahead = (ahead_at_decode(&take))
            .filter(|_| uses.is_some() && self.sinks.ahead.is_some())
            .map(|(luma, chroma)| self.decode_blocks(luma, chroma))
            .filter(|blocks| !blocks.is_empty());
        let begun = Begun {
            queue_obj_id: m.queue_obj_id,
            queue: (take.video && uses.is_some()).then(|| queue.clone_ref()),
            cb_obj_id: m.cb_obj_id,
            seqno: m.header.seqno,
            cb,
            in_flight,
            uses,
            out_of_memory,
            order: take.order.is_some(),
            marks: Arc::clone(&counts.order),
            waits_for_order,
            waits_at_begin: fill_waits || waits_for_order,
            line: Arc::clone(&counts.line),
            rate: Arc::clone(&counts.rate),
            submitted,
            scheduled,
            first,
            held,
            behind,
            drain,
            ordered: take.video,
            completions: Arc::clone(&counts.completions),
            ahead,
            computes: take.computes,
            work_names,
        };
        // A thread of its process that waits for it to be scheduled may ask for its record from
        // now, where it would have got it here had it asked for its `CB_SUBMITTED`.
        if let Some(scheduled) = begun.askable() {
            let askable = Arc::new(Askable {
                seqno: m.header.seqno,
                cb_obj_id: m.cb_obj_id,
                context: self.context,
                state: Mutex::new(Asking::Begun(Arc::clone(scheduled))),
            });
            counts.line.askable(Arc::clone(&askable));
            *self.asked.borrow_mut() = Some(askable);
        }
        // cbbegun: queue, seqno: its begin is done, on the thread that began it (its queue's
        // service thread), before its walk there or on its lane (`cenc`)
        reims_vgpu_vulkan::core::trace::event("cbbegun", &[m.queue_obj_id, m.header.seqno]);
        Ok(Ok(begun))
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
            uses,
            out_of_memory,
            order,
            marks,
            waits_for_order,
            waits_at_begin,
            line,
            rate,
            submitted,
            scheduled,
            held,
            behind,
            drain,
            ordered,
            completions,
            ahead,
            work_names,
            ..
        } = begun;
        // Its walk notes the first pipeline states its encoders set in its names
        // ([`Self::pipeline_set`]).
        *self.work_names.borrow_mut() = Some(work_names);
        let Some(ops) = read::<RgfxCommandBuffer>(message).and_then(|m| ops_of(message, &m)) else {
            return Ok(Err(DecodeError::Malformed));
        };
        // cenc: queue, seqno, its op bytes, then its ops, encoders, dispatches, fence updates and
        // fence waits ([`walk_counts`]): its walk's start, on the thread that walks it and commits
        // it, whose `cbcommit`s up to its `cbwalked` are its parts'
        if reims_vgpu_vulkan::core::trace::on() {
            let [n, encoders, dispatches, updates, waits] = walk_counts(ops);
            reims_vgpu_vulkan::core::trace::event(
                "cenc",
                &[
                    queue_obj_id,
                    seqno,
                    ops.len().to_u64(),
                    n,
                    encoders,
                    dispatches,
                    updates,
                    waits,
                ],
            );
        }
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
                in_flight: Arc::clone(&in_flight.count),
                out: Arc::clone(&in_flight.out),
                since: self.shared.platform.clock().now_mono(),
                hold: None,
                parts: None,
                whole: false,
                events: waits_at_begin,
                rate: Arc::clone(&rate),
                marks: Arc::clone(&marks),
                paced: true,
            });
        let mut cb = cb;
        // Dealt to two threads at once, where it may be cut and its queue holds room for a command
        // buffer of each chunk but its first ([`super::split`], "Two threads at once"); in two
        // chunks, one a thread, where work its queue committed before is not yet complete
        // ([`DEALT_BEHIND_WORK`]).
        let cuts = (split.as_ref())
            .map(|s| {
                let room = TAKE_IN_FLIGHT_MAX.saturating_sub(s.in_flight.load(Ordering::Acquire));
                let most = if self.behind_work(&s.out) {
                    DEALT_BEHIND_WORK
                } else {
                    usize::MAX
                };
                (room + 1).min(most)
            })
            .and_then(|most| deal(ops, self.deal_min_bytes(), self.deal_chunk_bytes(), most));
        let walked = match cuts {
            Some(cuts) => self.walk_dealt(
                ops,
                &cuts,
                (queue_obj_id, seqno),
                &mut cb,
                &mut events,
                &mut split,
            )?,
            None => self.walk_ops(ops, &mut cb, &mut events, &mut split)?,
        };
        if let Err(e) = walked {
            return Ok(Err(e));
        }
        // Its parts' gate, held to its last commit, and their outcome, which its last part's
        // completion makes the buffer's.
        let (parts, split_hold) = split.map_or((None, None), |s| (s.parts, s.hold));
        // After the buffers of its queue taken before it ([`super::lane`]): its signal of its
        // queue's surface timeline is decided in its ring's order ([`Self::order_signal`]), and
        // its turn of its completion post taken in its commit order.
        self.wait_turn();
        let turn = completions.turn();
        // A decode alone over device caches posts its completion at its decode where it waits
        // for nothing and is walked whole: it signals its queue's surface timeline, which the
        // buffers of other queues that use its output then wait for, as they would for a buffer
        // they named ([`super::surface_order`], "A decode completed ahead of its end").
        let ahead = ahead.filter(|_| {
            held.is_none()
                && !behind
                && parts.is_none()
                && events.signals.0.is_none()
                && events.waits.is_empty()
                && events.landings.len() == 1
        });
        let names = order || ahead.is_some();
        let signalled = self.order_signal(device, queue_obj_id, &marks, seqno, names, &cb)?;
        let signal_committed = signalled.as_ref().map(Signalled::committed);
        let ahead = ahead.filter(|_| signalled.is_some());
        let landed = ahead
            .as_ref()
            .and_then(|_| events.landings.first().cloned());
        let entry = ahead.map(|blocks| {
            (self.shared.completed_ahead).enter(&self.shared.platform, blocks, queue_obj_id, seqno)
        });
        let after = (self.sinks.ahead.as_ref())
            .filter(|_| landed.is_some())
            .map(|sinks| (Arc::new(BothDone::default()), Arc::clone(&sinks.finished)));
        let ending = Ending {
            turn: Arc::new(Mutex::new(Some(turn))),
            entry,
            after,
        };
        let at_decode = (ending.after.as_ref()).map(|(both, _)| {
            self.post_at_decode(
                cb_obj_id,
                seqno,
                (&ending.turn, both),
                &line,
                scheduled.clone(),
            )
        });
        // Register the submission and completion paths and commit: no scheduled or completed
        // handler, so the platform posts the buffer's records from the threads that submit it
        // and see it complete, waking no handler thread; and none to submit where its records
        // went at its take and it holds nothing.
        // Its records go at its commit where the platform may hold it, or a buffer before it on
        // its queue, for a buffer of another queue on the GPU ([`Line`]); else at its scheduling,
        // where they did not go at its take.
        let early = held.is_none() && !behind && (waits_for_order || line.behind_an_order());
        let order_hold = waits_for_order.then(|| line.ordered_hold());
        let noted = scheduled.clone().map(|s| (submitted.clone(), s));
        let unposted = scheduled.clone().filter(|s| !s.is_posted());
        let submission = submission_handler(submitted.clone(), held.clone(), unposted);
        let submission = taken_with(submission, order_hold.clone());
        let handler = self.completed_handler(
            cb_obj_id,
            seqno,
            events,
            drain,
            false,
            (submitted, held.clone(), scheduled),
            ending,
        );
        let handler = signalling(handler, signalled);
        let handler = completed_with(handler, order_hold.clone());
        let handler = match parts {
            Some(parts) => parts.merged(handler),
            None => handler,
        };
        // A cache's buffer the platform refused its memory at a use the walk made, or at a fill
        // its begin made, fails the buffer, its last part's error its own, as Metal's fails with
        // `MTLCommandBufferErrorOutOfMemory` ([`GraphicsDecoder::out_of_memory`]).
        if self.out_of_memory.take() | out_of_memory {
            cb.fail_out_of_memory();
        }
        self.name_firsts();
        let commit = || {
            if let Some(submission) = submission {
                cb.add_submission_handler(ON, submission)?;
            }
            cb.add_completion_seen_handler(ON, leaving(handler, in_flight.commits()))?;
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
            (None, Some(context)) => Some(self.waiting(|| self.shared.split_gate.commit(context))),
            _ => None,
        };
        match uses {
            // Its end event signalled for the caches' later copies under their lock, committed
            // with it let go, still among their users, and then the early copies behind it on its
            // queue ([`super::surface_cache::Caches::finish`]); the blocks its walk saw written
            // through textures over laid-out caches or their wide homes, the shapes it saw
            // written, and the blocks whose caches' buffers it named objects over told first.
            Some(mut uses) => {
                uses.saw_written(self.laid_written.borrow_mut().drain(..));
                uses.saw_shapes_written(self.shapes_written.borrow_mut().drain(..));
                uses.saw_over_cache(self.over_caches.borrow_mut().drain(..));
                self.surface_caches.finish(
                    device,
                    queue_obj_id,
                    queue.as_ref(),
                    &cb,
                    uses,
                    commit,
                )?
            }
            None => commit()?,
        }
        // Committed: a buffer behind it in the commit order goes on, and so does its context's
        // next commit and its range messages held while its parts were out; so does one sent
        // after an ask's record ([`Askable`]).
        self.ticket.borrow_mut().take();
        self.end_ask(true);
        drop((under_way, split_hold));
        // Last in its queue's order: its records, and those before it, go now where they go at
        // its commit.
        if let Some((submitted, scheduled)) = noted {
            line.committed(submitted, Arc::clone(&scheduled));
            if early {
                line.post_through(&scheduled);
            }
        }
        // Its completion goes at its decode from now, where it goes there: at once where the
        // decode is done already.
        if let (Some(landed), Some(at_decode)) = (landed, at_decode) {
            landed.arm(at_decode);
        }
        // cbwalked: queue, seqno: its walk and its last commit are done (`cenc`)
        reims_vgpu_vulkan::core::trace::event("cbwalked", &[queue_obj_id, seqno]);
        Ok(Ok(()))
    }

    /// The post of command buffer `cb_obj_id`'s completion, at `seqno`, at its decode
    /// ([`video::Landed::arm`]), where it takes its `turn` before its completion handler: the
    /// records of its queue's buffers committed up to it that are not yet posted go first, in
    /// `line`'s order, its own `scheduled` among them, then its `CB_COMPLETE` in its turn, with no
    /// GPU times (0 for each), and failed where its picture failed already, through the sink of
    /// such posts ([`super::AheadSinks::post`]), telling `both` once it is on the ring. Its
    /// completion handler then posts no record ([`Self::completed_handler`]).
    fn post_at_decode(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        (turn, both): (&TurnCell, &Arc<BothDone>),
        line: &Arc<Line>,
        scheduled: Option<Arc<Scheduled>>,
    ) -> video::EarlyPost {
        let turn = Arc::clone(turn);
        let both = Arc::clone(both);
        let line = Arc::clone(line);
        let sink = self.sinks.ahead.as_ref().map(|a| Arc::clone(&a.post));
        let kept = Arc::clone(&self.shared.completed_ahead);
        Box::new(move |landed: Arc<video::Landed>| {
            let Some(turn) = turn.lock().take() else {
                return;
            };
            let failed = landed.post().unwrap_or(false);
            if let Some(scheduled) = &scheduled {
                line.post_through(scheduled);
            }
            let record = match failed {
                true => cb_unlanded(cb_obj_id, seqno, 0, 0),
                false => cb_complete(cb_obj_id, seqno, 0, 0, None),
            };
            kept.posted();
            let post: Post = Box::new(move || match &sink {
                Some(sink) => sink(record, Box::new(move || both.came(None))),
                None => both.came(None),
            });
            turn.finish(Some(post));
        })
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
        // A lost device names the guest's command buffer it stands for, where it loses it in
        // flight ([`WorkNames`]): no encoder of it sets a pipeline state.
        cb.names_work(&Arc::new(WorkNames::new(m.cb_obj_id)));
        let in_flight = InFlight::new(&counts.in_flight, &counts.out);
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
        // In its queue's order ([`super::lane`]): its signal, its turn of its completion post,
        // and its commit.
        self.wait_turn();
        let turn = counts.completions.turn();
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
            Ending::of(turn),
        );
        let handler = signalling(handler, signalled);
        cb.add_completion_seen_handler(ON, leaving(handler, in_flight.commits()))?;
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
    /// `COMMIT_ORDER` names ([`super::commit_order`]); as a wait of its lane's job
    /// ([`Self::waiting`]).
    fn wait_to_commit(&self) {
        self.waiting(|| {
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
        });
    }

    /// The `CB_SCHEDULED` of command buffer `cb_obj_id`, at `seqno`, where there is a sink: posted
    /// already where the answer on a doorbell's thread posted it, else owed by its queue, whose
    /// records `line` keeps, until it is posted ([`Line::owes`]); and whether no buffer begun
    /// before it on that queue owed its own then (true without a `line`).
    fn scheduled(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        line: Option<&Line>,
    ) -> Option<(Arc<Scheduled>, bool)> {
        let sink = self.sinks.scheduled.clone()?;
        let posted = self.scheduled_at_doorbell.get();
        let (owed, first) = match line {
            Some(line) if !posted => {
                let before = line.owed.fetch_add(1, Ordering::AcqRel);
                (Some(Arc::clone(&line.owed)), before == 0)
            }
            _ => (None, true),
        };
        let scheduled = Scheduled {
            sink,
            record: scheduled_record(cb_obj_id, seqno),
            posted: Mutex::new(posted),
            owed,
        };
        Some((Arc::new(scheduled), first))
    }

    /// The completion handler of command buffer `cb_obj_id`, at `seqno`, whose op stream did
    /// `events`, which the platform runs on the thread that saw the buffer complete: where its
    /// submission handler did not, it lets its queue go where it `held` it and posts its
    /// `submitted` and its `scheduled`; then it posts the values the buffer signalled its shared
    /// events to that no notification posted ([`Signals::finish`]), then its completion, which
    /// for a `stand_in` ([`Self::stand_in_for_failed`]) is its failure whatever the stand-in's
    /// outcome, and for a buffer one of whose decodes' pictures did not land is its failure too
    /// ([`cb_unlanded`]), in its turn of its queue's completion posts ([`super::completions`]),
    /// and lets `drain` (its context's) go. Where its post at its decode took its turn first
    /// ([`Self::post_at_decode`]), it posts no record; and its decode leaves those completed ahead
    /// of their ends as it ends (`ending`).
    #[allow(clippy::too_many_arguments)]
    fn completed_handler(
        &self,
        cb_obj_id: u64,
        seqno: u64,
        events: EventOps,
        drain: Option<Arc<Drain>>,
        stand_in: bool,
        (submitted, held, scheduled): Unposted,
        ending: Ending<P>,
    ) -> CompletedHandler {
        let sink = self.sinks.completion.clone();
        let context = self.context;
        let signals = events.signals;
        let landings = events.landings;
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
            let Ending { turn, entry, after } = ending;
            // Posted at its decode: the record is the guest's, or is on its way there in its turn,
            // and its buffer's work is done now. What waits for its end is told once both are.
            let Some(turn) = turn.lock().take() else {
                drop(entry);
                let both = after.as_ref().map(|(both, _)| Arc::clone(both));
                let end: Then = Box::new(move || {
                    if let Some((_, finished)) = &after {
                        finished(seqno);
                    }
                    drop(drain);
                });
                match both {
                    Some(both) => both.came(Some(end)),
                    None => end(),
                }
                return;
            };
            // The GPU times on the device's clock, the one the guest's kext maps the device's
            // times by; none is 0 for each.
            let (start, end) = info.gpu_times_ns.map(runtime::gpu_times).unwrap_or((0, 0));
            // Each decode's outcome is fixed as posted here ([`video::Landed::post`]).
            let unlanded = (landings.iter()).fold(false, |any, landed| {
                landed.post().unwrap_or_else(|| landed.failed()) | any
            });
            let record = if stand_in {
                cb_failed(cb_obj_id, seqno)
            } else if info.error.is_none() && unlanded {
                cb_unlanded(cb_obj_id, seqno, start, end)
            } else {
                cb_complete(cb_obj_id, seqno, start, end, info.error.as_ref())
            };
            if let Some(error) = &info.error {
                log::debug!(
                    target: "reims_vgpu::graphics::completion",
                    "command buffer {cb_obj_id:#x} (seqno {seqno}) failed: status {}, code {} \
                     ({})",
                    record.status,
                    record.error_code,
                    error.description
                );
            }
            // The buffer's work and its completion are done: a drain of its context may end,
            // once the record is on the guest's ring.
            let post: Post = Box::new(move || match &sink {
                Some(sink) => sink(record, Box::new(move || drop(drain))),
                None => drop(drain),
            });
            turn.finish(Some(post));
            drop(entry);
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
        let fan_out: Arc<dyn FanOut> = self.shared.shares.clone();
        let signals = events.signals.signals(self.sinks.signal.as_ref(), &fan_out);
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
        let fan_out: Arc<dyn FanOut> = self.shared.shares.clone();
        let signals = events.signals.signals(self.sinks.signal.as_ref(), &fan_out);
        signals.push(op.event_obj_id, op.value);
    }

    /// Notes the first pipeline states the walk set ([`Self::pipeline_set`]) in its command
    /// buffer's names, before a commit of what it walked: a first noted before stays
    /// ([`WorkNames::note_firsts`]). A dealt walk's chunks note theirs in their order instead
    /// ([`Self::commit_dealt`]).
    fn name_firsts(&self) {
        if let Some(names) = self.work_names.borrow().as_ref() {
            let (compute, render) = self.firsts.get();
            names.note_firsts(compute, render);
        }
    }

    /// At an encoder's end, `rest` the ops left after it: where the part of the buffer walked
    /// since the last is long enough to cut ([`Estimate::cuts`]: [`Self::split_after`] where no
    /// command buffer its queue committed is incomplete, else by its estimated GPU time,
    /// [`Self::behind_work`]), an encoder begins in `rest` ([`encoder_follows`]), its queue holds
    /// room for another command buffer, the buffer could be committed now
    /// ([`Self::may_commit_now`]) and its context's gate is free or its own, hands the platform
    /// that part ([`CommandBuffer::split`]) and goes on in the command buffer it answers, `cb`
    /// from then on ([`super::split`]). It waits for nothing to cut; once it cut, it waits, where
    /// it is paced, while its parts run too far ahead of the GPU ([`Self::pace`]). Where the
    /// platform keeps the buffer whole, it is walked whole from there.
    fn split_here(
        &self,
        cb: &mut P::CommandBuffer,
        split: &mut Option<Splitting>,
        rest: &[u8],
    ) -> RaiseResult<()> {
        let Some(s) = split.as_mut().filter(|s| !s.whole) else {
            return Ok(());
        };
        let clock = self.shared.platform.clock();
        let walked = clock.now_mono().saturating_duration_since(s.since);
        let behind = self.behind_work(&s.out);
        let estimate = (s.parts.as_ref()).map_or_else(
            || s.rate.get().map_or(Estimate::Unknown, Estimate::Rate),
            |parts| parts.estimate(),
        );
        let cuts = if behind {
            estimate.cuts(walked, true)
        } else {
            walked >= self.split_after()
        };
        if !cuts
            || !encoder_follows(rest)
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
        let parts =
            Arc::clone((s.parts).get_or_insert_with(|| Arc::new(Parts::on(Arc::clone(&s.rate)))));
        // The part's own place among its queue's command buffers, until its completion; the
        // buffer's goes on to its last part. A part cut behind incomplete work is committed
        // unmarked, so its completion is seen at its own end, which a paced walk waits for.
        let place = InFlight::new(&s.in_flight, &s.out).commits();
        let cut = (walked, clock.now_mono());
        cb.add_completion_seen_handler(ON, parts.part_handler(place, Some(cut)))?;
        let joins = !behind && joins_next::<P>(cb, s);
        self.name_firsts();
        match cb.split(ON)? {
            Some(next) => {
                *cb = next;
                s.hold = Some(hold);
                s.events = false;
                self.note_part_cut();
                if s.paced && self.paces_here() {
                    s.paced = self.pace(&parts, s.context, &s.marks);
                }
                s.since = clock.now_mono();
            }
            None => {
                // Kept whole: the handler just added sees the buffer's own completion, which
                // its outcome then is ([`Parts::merged`]); the gate stays held only where parts
                // are out. Its last part, unmarked.
                if joins {
                    cb.set_completion_joins_next(false);
                }
                s.whole = true;
                if !first {
                    s.hold = Some(hold);
                }
            }
        }
        Ok(())
    }

    /// Waits, after a cut, while the buffer's parts `parts` run too far ahead of the GPU
    /// ([`Parts::wait_ahead`]), as a wait of its lane's job ([`Self::waiting`]), so the lane's
    /// next job walks meanwhile ([`super::split`], "How far ahead of the GPU"). The wait ends at
    /// once where a reset of the device begins, the buffer's queue ends (`marks`), a buffer of
    /// its context waits for its last commit ([`Self::held_back`]), or [`PACE_WAIT_MOST`] passes
    /// with none of its parts seen complete.
    /// Whether the walk goes on paced: not where the wait was cut short or a part failed.
    fn pace(&self, parts: &Parts, context: u32, marks: &Marks) -> bool {
        let clock = self.shared.platform.clock();
        let began = clock.now_mono();
        let cancel = &self.shared.commit_cancel;
        let reset = cancel.now();
        // Why the wait was cut short: 1 a reset, 2 the queue's end, 3 a buffer of the context
        // waiting, 4 the bound.
        let cut = std::cell::Cell::new(0_u64);
        let progress = std::cell::Cell::new(Progress::from(began));
        let paced = self.waiting(|| {
            parts.wait_ahead(
                || clock.now_mono(),
                |completed| {
                    let now = clock.now_mono();
                    let mut stalled = progress.get();
                    let since = stalled.since(completed, now);
                    progress.set(stalled);
                    let why = if cancel.cut_since(reset) {
                        1
                    } else if marks.commit.has_ended() {
                        2
                    } else if self.held_back(context, marks) {
                        3
                    } else if since >= PACE_WAIT_MOST {
                        4
                    } else {
                        0
                    };
                    cut.set(why);
                    #[cfg(test)]
                    if why == 0 {
                        (self.shared.pace_waits).fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                    why != 0
                },
            )
        });
        if reims_vgpu_vulkan::core::trace::on() {
            let waited = clock.now_mono().saturating_duration_since(began);
            // cbpace: the walk's wait after a cut, in ns, whether it goes on paced, and why the
            // wait was cut short (0 for not, or for a failed part)
            reims_vgpu_vulkan::core::trace::event(
                "cbpace",
                &[
                    u64::try_from(waited.as_nanos()).unwrap_or(u64::MAX),
                    u64::from(paced),
                    cut.get(),
                ],
            );
        }
        paced
    }

    /// Whether a command buffer of `context` waits for the last commit of the buffer of its queue
    /// (`marks`) whose parts are out ([`super::split`], "How far ahead of the GPU"): a commit at
    /// the split's gate or a buffer dealt to two threads to take it
    /// ([`super::split::SplitGate::holds_back`]), a commit of another of the process's queues
    /// that names the queue in the commit order
    /// ([`super::commit_order::CommitMark::has_waiters`]), or one waiting for the context's
    /// tickets or range messages ([`super::sequencer::Sequencer::holds_up`]).
    fn held_back(&self, context: u32, marks: &Marks) -> bool {
        self.shared.split_gate.holds_back(context)
            || marks.commit.has_waiters()
            || self.shared.sequencer.holds_up(context)
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
        self.walk_begins(ops);
        self.walk_range(ops, cb, events, split)
    }

    /// What a command buffer's walk does once, before its first op, whatever ranges it is
    /// walked in: finds the names of its op stream `ops` ([`Self::find_named`]), and forgets
    /// what an earlier walk on the handle noted of laid-out device caches.
    fn walk_begins(&self, ops: &[u8]) {
        #[cfg(test)]
        (self.shared.walks_begun).fetch_add(1, Ordering::Relaxed);
        self.find_named(ops);
        self.laid_uses.borrow_mut().clear();
        self.laid_written.borrow_mut().clear();
        self.shapes_written.borrow_mut().clear();
        self.over_caches.borrow_mut().clear();
        self.out_of_memory.set(false);
        self.encoder_shapes.borrow_mut().clear();
        self.slot_shapes.borrow_mut().clear();
        self.slot_pipeline.set(0);
        self.firsts.set((0, 0));
    }

    /// [`Self::walk_ops`]' walk of `ops`, the whole of a command buffer's op stream or a range
    /// of it that starts at an encoder's begin, after what it does once ([`Self::walk_begins`]).
    fn walk_range(
        &self,
        ops: &[u8],
        cb: &mut P::CommandBuffer,
        events: &mut EventOps,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        // The rt platform's boot diagnostics total a refused draw by its process.
        reims_vgpu_vulkan::rt_platform::diag::walk_context(self.context);
        let mut encoder: Option<OpenEncoder<P>> = None;
        let walked = refusing(|| self.walk(ops, cb, &mut encoder, events, split));
        if let Some(open) = encoder.take() {
            end_abandoned(open);
        }
        walked
    }

    /// [`Self::walk_ops`] for a command buffer whose op stream `ops` is cut at `cuts` into chunks
    /// dealt to two threads at once ([`super::split`], "Two threads at once"), `queue` and `seqno`
    /// its queue and its message's, for the trace: this thread walks the first chunk and those
    /// [`helper_chunks`] leaves it, and a helper ([`super::helpers`]) the others, each thread its
    /// chunks in their order, handed over at once, on a handle of its own
    /// ([`GraphicsDecoder::twin`]), each chunk into a command buffer of its own that the platform
    /// makes on the buffer's queue before any is walked ([`CommandBuffer::part`]), each while the
    /// queue holds room, the first chunk into `cb`; where the platform makes none, or the queue's
    /// room runs out first, the buffer is walked as a whole is. After each of its own
    /// chunks this thread commits every chunk walked, in their order, as a part
    /// ([`CommandBuffer::commit_part`]), waiting for the helper only for the chunk it is to commit
    /// next ([`Self::commit_dealt`]), so the GPU starts on each part while the rest is walked; the
    /// last chunk's command buffer is `cb` from then on, the last part, which its handlers and
    /// records go on. Chunks no helper started by the end of the first are taken back and walked
    /// here, into `cb`, as a whole walk goes on. A walk that faults or stops in a chunk answers as
    /// the first such chunk in their order does, nothing at or after it committed
    /// ([`Self::first_fault`]). Each op is walked once, on one handle or the other.
    fn walk_dealt(
        &self,
        ops: &[u8],
        cuts: &[usize],
        (queue, seqno): (u64, u64),
        cb: &mut P::CommandBuffer,
        events: &mut EventOps,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some((in_flight, out)) =
            (split.as_ref()).map(|s| (Arc::clone(&s.in_flight), Arc::clone(&s.out)))
        else {
            return self.walk_ops(ops, cb, events, split);
        };
        let ends = cuts.iter().copied().chain(std::iter::once(ops.len()));
        let chunks: Vec<Range<usize>> = (std::iter::once(0).chain(cuts.iter().copied()))
            .zip(ends)
            .map(|(start, end)| start..end)
            .collect();
        let n = chunks.len();
        // Every chunk's command buffer but the first's, made now from `cb`, which holds no op of
        // the walk's, each with its place among its queue's command buffers: so the walk makes
        // none while parts of it wait uncommitted, which the queue's room for command buffers
        // would hold up for good where it is full. Each is made only while the queue holds room,
        // as a cut's is ([`Self::split_here`]): [`deal`] took the room there was as it started,
        // and other walks and begins make command buffers on the queue meanwhile.
        let mut parts = Vec::with_capacity(n - 1);
        for _ in 1..n {
            let room = in_flight.load(Ordering::Acquire) < TAKE_IN_FLIGHT_MAX;
            match if room { cb.part(ON)? } else { None } {
                Some(part) => parts.push((part, InFlight::new(&in_flight, &out))),
                None => {
                    drop(parts);
                    return self.walk_ops(ops, cb, events, split);
                }
            }
        }
        let helper = helper_chunks(&chunks);
        let (mut theirs, mut ours) = (Vec::new(), VecDeque::new());
        for (i, part) in (1..n).zip(parts) {
            if helper[i] {
                theirs.push((i, part));
            } else {
                ours.push_back(part);
            }
        }
        self.walk_begins(ops);
        let slots = Slots::new(n);
        let clock = self.shared.platform.clock();
        let handed = clock.now_mono();
        let job = {
            // The helper's chunks, copied: a hand outlives no borrow of the message.
            let mut bytes = Vec::with_capacity(ops.len() / 2 + 1);
            let theirs: Vec<Theirs<P>> = (theirs.into_iter())
                .map(|(i, part)| {
                    let at = bytes.len();
                    bytes.extend_from_slice(&ops[chunks[i].clone()]);
                    (i, at..bytes.len(), part)
                })
                .collect();
            let twin = self.twin();
            let slots = Arc::clone(&slots);
            // Part of this walk's entry, which waits for it ([`crate::entry::run_within`]).
            move || {
                crate::entry::run_within(|_| {
                    let ending = slots.ending();
                    twin.walk_chunks(&bytes, theirs, &slots, handed);
                    drop((twin, ending));
                });
            }
        };
        let hand = self.shared.helpers.hand(Box::new(job));
        #[cfg(test)]
        if self.shared.helper_first.load(Ordering::Relaxed) {
            hand.wait_started();
        }
        let mut deal = Deal {
            chunks: &chunks,
            helper: &helper,
            slots: &slots,
            hand: &hand,
            ours,
            in_flight,
            out,
            own: (0..n).map(|_| None).collect(),
            next: 0,
            kept: Kept::No,
            parts_out: false,
            place: None,
            handed,
            queue,
            seqno,
            stats: DealStats::default(),
        };
        let walked = self.deal_here(ops, &mut deal, cb, events, split);
        // The helper starts no chunk more, and its walk ends before this one's, which it is part
        // of; one it never started is let go of here.
        slots.stop_at(0);
        match hand.take_back() {
            Some(job) => drop(job),
            None => hand.wait(),
        }
        let s = &deal.stats;
        let ns = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        let bytes = |theirs: bool| -> u64 {
            let dealt = chunks.iter().zip(&helper).filter(|(_, h)| **h == theirs);
            dealt.map(|(c, _)| c.len()).sum::<usize>() as u64
        };
        // cbpar: queue, seqno, its chunks, the bytes of this thread's and of the helper's, 1
        // where the helper's were taken back (else 0), this thread's walk of its chunks, the hand
        // to the helper's start, the helper's walk of its chunks, this thread's waits for them and
        // how many it waited for, and the hand to the walk's end, each in ns
        reims_vgpu_vulkan::core::trace::event(
            "cbpar",
            &[
                queue,
                seqno,
                n as u64,
                bytes(false),
                bytes(true),
                u64::from(s.back),
                ns(s.walk_here),
                s.helper_began.map_or(0, ns),
                ns(s.walk_helper),
                ns(s.waited),
                s.waits,
                ns(clock.now_mono().saturating_duration_since(handed)),
            ],
        );
        walked
    }

    /// [`Self::walk_dealt`] on this thread: its chunks walked, and every chunk committed or folded
    /// in their order.
    fn deal_here(
        &self,
        ops: &[u8],
        deal: &mut Deal<'_, P>,
        cb: &mut P::CommandBuffer,
        events: &mut EventOps,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let n = deal.chunks.len();
        let clock = self.shared.platform.clock();
        for k in (0..n).filter(|&k| !deal.helper[k]) {
            let began = clock.now_mono();
            let outcome = self.walk_range(&ops[deal.chunks[k].clone()], cb, events, &mut None);
            let walk = clock.now_mono().saturating_duration_since(began);
            deal.stats.walk_here += walk;
            // No helper took its chunks: they are walked here, as a whole walk goes on from the
            // first chunk's last encoder's end, where it may cut a part; the command buffers made
            // for the later chunks are let go of unused.
            if k == 0
                && let Some(job) = deal.hand.take_back()
            {
                drop(job);
                deal.ours.clear();
                deal.stats.back = true;
                if let Err(e) = outcome? {
                    return Ok(Err(e));
                }
                let rest = &ops[deal.chunks[1].start..];
                self.split_here(cb, split, rest)?;
                return self.walk_range(rest, cb, events, split);
            }
            let walked = Walked {
                cb: None,
                outcome,
                written: std::mem::take(&mut *self.laid_written.borrow_mut()),
                shapes: std::mem::take(&mut *self.shapes_written.borrow_mut()),
                over: std::mem::take(&mut *self.over_caches.borrow_mut()),
                out_of_memory: self.out_of_memory.take(),
                firsts: self.firsts.take(),
                began: began.saturating_duration_since(deal.handed),
                walk,
            };
            if !walked.walked() {
                return self.first_fault(k, walked, deal);
            }
            // This thread's next chunk goes into its own command buffer, `cb` from now on.
            deal.own[k] = Some(match deal.ours.pop_front() {
                Some((part, place)) => {
                    let done = std::mem::replace(cb, part);
                    let done_place = deal.place.replace(place);
                    Walked {
                        cb: Some((done, done_place)),
                        ..walked
                    }
                }
                None => walked,
            });
            let committed = self.commit_dealt(k, false, deal, cb, split)?;
            if let Err(e) = committed {
                return Ok(Err(e));
            }
        }
        self.commit_dealt(n - 1, true, deal, cb, split)
    }

    /// A helper's walk of its chunks of a buffer dealt to two threads ([`Self::walk_dealt`]), on
    /// the twin of the walk's handle: each of `chunks` (its index among the buffer's, its ops in
    /// `bytes`, and its command buffer) walked, and published to its slot. It stops at a chunk it
    /// may not start ([`Slots::starts`]), and after one whose walk faults or stops.
    fn walk_chunks(
        &self,
        bytes: &[u8],
        chunks: Vec<Theirs<P>>,
        slots: &Slots<Walked<P>>,
        handed: MonoTime,
    ) {
        let clock = self.shared.platform.clock();
        for (at, range, (mut cb, place)) in chunks {
            if !slots.starts(at) {
                return;
            }
            let began = clock.now_mono();
            let mut events = EventOps::default();
            let outcome = self.walk_range(&bytes[range], &mut cb, &mut events, &mut None);
            let walked = Walked {
                cb: Some((cb, Some(place))),
                outcome,
                written: std::mem::take(&mut *self.laid_written.borrow_mut()),
                shapes: std::mem::take(&mut *self.shapes_written.borrow_mut()),
                over: std::mem::take(&mut *self.over_caches.borrow_mut()),
                out_of_memory: self.out_of_memory.take(),
                firsts: self.firsts.take(),
                began: began.saturating_duration_since(handed),
                walk: clock.now_mono().saturating_duration_since(began),
            };
            let went_on = walked.walked();
            slots.fill(at, walked);
            if !went_on {
                return;
            }
        }
    }

    /// Commits, as parts, each chunk of a walk dealt to two threads from its next one through
    /// `through`, in their order ([`Self::walk_dealt`]), waiting for the helper's walk of one
    /// only where it is next: where the gate of the buffer's context is held, or the buffer could
    /// be committed now and its gate is free ([`Self::may_commit_now`], [`SplitGate::try_split`]);
    /// else nothing, and the walk goes on, unless `end`, at the walk's end, where it waits to
    /// commit as its commit would ([`Self::wait_to_commit`], [`SplitGate::split`]). The last chunk
    /// is not committed: its command buffer is `cb` from then on, the buffer's last part. A part
    /// the platform keeps whole only by how Metal's GPU fails it from one of its encoders on
    /// ([`PartCut::Fails`]) is kept: every later chunk's command buffer is folded into it in
    /// their order ([`CommandBuffer::fold`]), its encoders running nothing either way, and it is
    /// `cb` at the end, its commit the whole's. One kept whole for a reason of [`KeptWhole`] fails
    /// the buffer ([`kept_whole`]). The first chunk in their order whose walk faulted or stopped
    /// answers for the walk, nothing at or after it committed. What each chunk's walk saw written
    /// of laid-out device caches goes to the buffer's in their order.
    ///
    /// [`SplitGate::try_split`]: super::split::SplitGate::try_split
    /// [`SplitGate::split`]: super::split::SplitGate::split
    fn commit_dealt(
        &self,
        through: usize,
        end: bool,
        deal: &mut Deal<'_, P>,
        cb: &mut P::CommandBuffer,
        split: &mut Option<Splitting>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(s) = split.as_mut() else {
            return Err(op_fault("a dealt command buffer may not be cut"));
        };
        let last = deal.chunks.len() - 1;
        let clock = self.shared.platform.clock();
        while deal.next <= through {
            let j = deal.next;
            // A part goes now where it could be committed now, as a cut does; at the walk's end,
            // once every wait of its commit is met, holding its context's gate then.
            if j < last && matches!(deal.kept, Kept::No) && s.hold.is_none() {
                let now = (self.may_commit_now())
                    .then(|| self.shared.split_gate.try_split(s.context))
                    .flatten();
                s.hold = match now {
                    Some(hold) => Some(hold),
                    None if end => {
                        self.wait_to_commit();
                        Some(self.waiting(|| self.shared.split_gate.split(s.context)))
                    }
                    None => return Ok(Ok(())),
                };
            }
            let (chunk, waited) = match deal.own[j].take() {
                Some(chunk) => (chunk, Duration::ZERO),
                None => {
                    let waiting = clock.now_mono();
                    let (chunk, did) = deal.slots.wait_for(j, || {
                        #[cfg(test)]
                        (self.shared.deal_waits).fetch_add(1, Ordering::Relaxed);
                    });
                    let waited = clock.now_mono().saturating_duration_since(waiting);
                    reims_vgpu_vulkan::core::long_waits::waited("deal-chunk", waited);
                    deal.stats.waited += waited;
                    deal.stats.waits += u64::from(did);
                    let Some(chunk) = chunk else {
                        return Err(op_fault("a dealt chunk was walked by no helper"));
                    };
                    deal.stats.walk_helper += chunk.walk;
                    deal.stats.helper_began = deal.stats.helper_began.or(Some(chunk.began));
                    (chunk, waited)
                }
            };
            let Walked {
                cb: walked_into,
                outcome,
                written,
                shapes,
                over,
                out_of_memory,
                firsts: (compute, render),
                began,
                walk,
            } = chunk;
            if let Err(e) = outcome? {
                return Ok(Err(e));
            }
            deal.next = j + 1;
            if out_of_memory {
                self.out_of_memory.set(true);
            }
            // Its firsts follow those of the chunks before it, before any of it is committed.
            if let Some(names) = self.work_names.borrow().as_ref() {
                names.note_firsts(compute, render);
            }
            self.laid_written.borrow_mut().extend(written);
            self.shapes_written.borrow_mut().extend(shapes);
            let mut noted = self.over_caches.borrow_mut();
            for block in over {
                if !noted.contains(&block) {
                    noted.push(block);
                }
            }
            drop(noted);
            // cbdeal: queue, seqno, the chunk, 1 where the helper walked it (else 0), its op
            // bytes, the hand to its walk's start, its walk, this thread's wait for it, and the
            // hand to now, each in ns
            let ns = |d: Duration| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
            reims_vgpu_vulkan::core::trace::event(
                "cbdeal",
                &[
                    deal.queue,
                    deal.seqno,
                    j as u64,
                    u64::from(deal.helper[j]),
                    deal.chunks[j].len() as u64,
                    ns(began),
                    ns(walk),
                    ns(waited),
                    ns(clock.now_mono().saturating_duration_since(deal.handed)),
                ],
            );
            // Folded into the part kept whole, where one is; else the last part, or a part
            // committed.
            match (std::mem::replace(&mut deal.kept, Kept::No), walked_into) {
                (Kept::Apart(whole), Some((later, _))) => {
                    whole.fold(later);
                    deal.kept = Kept::Apart(whole);
                    continue;
                }
                (Kept::Apart(whole), None) => {
                    let later = std::mem::replace(cb, whole);
                    cb.fold(later);
                    deal.place = None;
                    deal.kept = Kept::Here;
                    continue;
                }
                (Kept::Here, Some((later, _))) => {
                    cb.fold(later);
                    deal.kept = Kept::Here;
                    continue;
                }
                (Kept::Here, None) => {
                    return Err(op_fault(
                        "a dealt chunk was walked into the part kept whole",
                    ));
                }
                (Kept::No, walked_into) if j == last => {
                    // The last part: its place is the buffer's.
                    if let Some((part, _)) = walked_into {
                        *cb = part;
                    }
                    deal.place = None;
                    continue;
                }
                (Kept::No, walked_into) => {
                    // A part, committed: one walked into `cb` is let go of once the last chunk's
                    // command buffer takes its place there. The part's handler holds its place
                    // among its queue's command buffers, or one of its own for the buffer's
                    // first, whose place goes on to its last part.
                    let (part, place) = match walked_into {
                        Some((part, place)) => (Some(part), place),
                        None => (None, deal.place.take()),
                    };
                    let place = (place
                        .unwrap_or_else(|| InFlight::new(&deal.in_flight, &deal.out)))
                    .commits();
                    let parts = Arc::clone(
                        (s.parts).get_or_insert_with(|| Arc::new(Parts::on(Arc::clone(&s.rate)))),
                    );
                    let committing = part.as_ref().unwrap_or(&*cb);
                    committing.add_completion_seen_handler(ON, parts.part_handler(place, None))?;
                    let joins = joins_next::<P>(committing, s);
                    match committing.commit_part(ON)? {
                        PartCut::Cut => {
                            deal.parts_out = true;
                            s.events = false;
                            self.note_part_cut();
                        }
                        PartCut::Fails => {
                            // Kept whole, as a cut the platform refuses keeps it
                            // ([`Self::split_here`]): the handler just added sees its completion,
                            // and the gate stays held only where parts are out. Its commit is
                            // the whole's, the last part's, unmarked.
                            if joins {
                                committing.set_completion_joins_next(false);
                            }
                            s.whole = true;
                            if !deal.parts_out {
                                s.hold = None;
                            }
                            deal.kept = match part {
                                Some(part) => Kept::Apart(part),
                                None => Kept::Here,
                            };
                        }
                        PartCut::Whole(reason) => return Err(kept_whole(reason)),
                    }
                }
            }
        }
        Ok(Ok(()))
    }

    /// The answer of a walk dealt to two threads ([`Self::walk_dealt`]) whose chunk `at`, `fault`,
    /// faulted or stopped: the first such chunk's in their order.
    /// The helper starts no chunk from `at` on, and this thread waits for its walk of each chunk
    /// before it not yet taken; nothing more is committed.
    fn first_fault(
        &self,
        at: usize,
        fault: Walked<P>,
        deal: &Deal<'_, P>,
    ) -> RaiseResult<Result<(), DecodeError>> {
        deal.slots.stop_at(at);
        let before = deal.own.iter().enumerate().take(at).skip(deal.next);
        for (j, _) in before.filter(|(_, own)| own.is_none()) {
            match deal.slots.wait_for(j, || {}).0 {
                Some(chunk) if !chunk.walked() => return chunk.outcome,
                Some(_) => {}
                None => return Err(op_fault("a dealt chunk was walked by no helper")),
            }
        }
        fault.outcome
    }

    /// Finds the first [`NAMES_AT_ONCE`] objects of the current context that the compute and
    /// render ops of the command buffer's op stream `ops` name ([`each_named`]), of the first
    /// [`NAMES_READ`] names, and that its view of the context's table has not found yet, in that
    /// view, where it has one: [`FOUND_A_HOLD`] under each hold of the table's lock
    /// ([`View::find_all`]). The walk then answers them from the view's names, as it would have
    /// found them as it reached them, and takes no lock of the table's for them between the
    /// commands it encodes. A name the view found already is not looked for again, so a draw's
    /// names, which the draws after it name again, are looked for once.
    ///
    /// [`View::find_all`]: super::tables::View::find_all
    fn find_named(&self, ops: &[u8]) {
        let Some(context) = self.context else {
            return;
        };
        let views = self.views.borrow();
        let Some(view) = views.iter().find(|v| v.context().id == context) else {
            return;
        };
        let mut ids = [0u64; FOUND_A_HOLD];
        let (mut n, mut looked, mut read) = (0, 0, 0);
        each_named(ops, |id| {
            read += 1;
            if self.owned(id) && !view.found(id) {
                ids[n] = id;
                n += 1;
                looked += 1;
                if n == FOUND_A_HOLD {
                    view.find_all(ids);
                    n = 0;
                }
            }
            looked < NAMES_AT_ONCE && read < NAMES_READ
        });
        view.find_all(ids[..n].iter().copied());
    }

    /// The writers' fence ([`super::surface_cache::Caches::writes_fence`]) where the encoder being
    /// walked noted a use that may write a laid-out cache or its wide home ([`Self::note_laid`]),
    /// or a shape whose plane has another ([`Self::note_shape`]), for it to update as it ends;
    /// `None` where it noted none.
    fn laid_writes_fence(&self, cb: &P::CommandBuffer) -> RaiseResult<Option<P::Fence>> {
        if !self.laid_uses.borrow().iter().any(|u| u.writes)
            && !(self.surface_caches).shares_plane(&self.encoder_shapes.borrow())
        {
            return Ok(None);
        }
        let Some(device) = cb.device() else {
            return Ok(None);
        };
        self.surface_caches.writes_fence(&device)
    }

    /// The encoder just ended in `cb` wrote, through textures over the planes of laid-out device
    /// caches or over their wide homes, the blocks its uses noted ([`Self::note_laid`]): each
    /// side written is copied into the other in `cb` before the next encoder
    /// ([`super::surface_cache::Caches::reconcile`]), so it reads the write through either, as
    /// the Macs' next encoders read it (docs/graphics-kext/dgpu-oracle.md, Q37). An encoder that
    /// wrote both sides of a block keeps the home's writes in the texels both hold.
    fn reconcile_laid(&self, cb: &P::CommandBuffer) -> RaiseResult<()> {
        let uses = std::mem::take(&mut *self.laid_uses.borrow_mut());
        let mut written: Vec<(u64, bool)> = Vec::new();
        for u in uses.iter().filter(|u| u.writes) {
            match written.iter_mut().find(|(block, _)| *block == u.block) {
                Some((_, from_home)) => *from_home |= u.wide,
                None => written.push((u.block, u.wide)),
            }
        }
        if written.is_empty() {
            return Ok(());
        }
        let both = written.iter().any(|&(block, from_home)| {
            from_home && uses.iter().any(|u| u.block == block && u.writes && !u.wide)
        });
        if both {
            log::warn!(
                target: "reims_vgpu::graphics",
                "an encoder may write an IOSurface through textures over its device cache laid \
                 out at the GPU's pitch and over its wide home at once: the home's writes are kept \
                 in the texels both hold"
            );
        }
        self.laid_written
            .borrow_mut()
            .extend(written.iter().map(|&(block, _)| block));
        self.surface_caches.reconcile(cb, &written)
    }

    /// In the per-shape form, the encoder just ended in `cb` may have written the shapes its uses
    /// noted ([`Self::note_shape`]): each other shape of their planes takes the write in `cb`
    /// before the next encoder ([`super::surface_cache::Caches::reconcile_shapes`]), so it reads
    /// the write through any texture over the plane, as the Macs' next encoders read it
    /// (docs/graphics-kext/dgpu-oracle.md, Q37).
    fn reconcile_shapes(&self, cb: &P::CommandBuffer) -> RaiseResult<()> {
        self.slots_end();
        let written = std::mem::take(&mut *self.encoder_shapes.borrow_mut());
        if written.is_empty() {
            return Ok(());
        }
        if self.surface_caches.reconcile_shapes(cb, &written)? {
            self.out_of_memory.set(true);
        }
        Ok(())
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
                    // An event's wait or signal: the part it goes into is observed.
                    if matches!(tag, RGFX_OP_SIGNAL_EVENT | RGFX_OP_WAIT_EVENT)
                        && let Some(s) = split.as_mut()
                    {
                        s.events = true;
                        s.paced = false;
                    }
                    self.framing_op((tag, body, &ops[len..]), cb, encoder, events)?;
                    if tag == RGFX_OP_ENCODER_END {
                        self.reconcile_laid(cb)?;
                        self.reconcile_shapes(cb)?;
                    }
                    if tag == RGFX_OP_ENCODER_END && len < ops.len() {
                        self.split_here(cb, split, &ops[len..])?;
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
                    // A dispatch while a slot of the encoder's state holds a refused object is
                    // dropped ([`refused`]).
                    let dropped =
                        marks.skips() || (marks.drops_draws() && compute::dispatches(tag));
                    if !dropped {
                        match compute::decode_op(tag, body, self, enc, marks, ON) {
                            Ok(true) => {}
                            Ok(false) => return Err(unknown_op(tag)),
                            Err(raise) => contain(raise)?,
                        }
                    }
                }
                RGFX_OP_GROUP_BLIT => {
                    let Some(OpenEncoder::Blit(enc)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "blit"));
                    };
                    match blit::decode_op(tag, body, self, enc, ON) {
                        Ok(true) => {}
                        Ok(false) => return Err(unknown_op(tag)),
                        Err(raise) => contain(raise)?,
                    }
                }
                RGFX_OP_GROUP_VIDEO => {
                    let Some(OpenEncoder::Video(session)) = encoder.as_ref() else {
                        return Err(no_encoder(tag, "video"));
                    };
                    let landings = &mut events.landings;
                    match video::decode_op(tag, body, self, session.session(), cb, landings) {
                        Ok(true) => {}
                        Ok(false) => return Err(unknown_op(tag)),
                        Err(raise) => contain(raise)?,
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
    ///
    /// Only an op's binding marks a slot that holds a refused object, or ends its mark
    /// ([`refused`]). While no slot of the encoder's state holds one, whether one does is read once
    /// after each op [`render::decode_op`] carries out; from the op that leaves one held, the
    /// pass's ops go on in [`Self::refused_render_ops`].
    fn render_ops<'o>(
        &self,
        mut ops: &'o [u8],
        enc: &P::RenderEncoder,
        pass: &render::PassFacts,
    ) -> RaiseResult<Result<&'o [u8], DecodeError>> {
        if pass.refused.drops_draws() {
            return self.refused_render_ops(ops, enc, pass);
        }
        let mut draws = Draws::new();
        while !ops.is_empty() {
            // Draws one after another go to the encoder together; each other op, after them.
            if let Some((draw, len)) = self.vertex_draw(ops) {
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
            match render::decode_op(tag, body, self, enc, pass, ON) {
                Ok(true) => {}
                Ok(false) => return Err(unknown_op(tag)),
                Err(raise) => contain(raise)?,
            }
            ops = &ops[len..];
            // The draws gathered were sent before the op, so none is left behind.
            if pass.refused.drops_draws() {
                return self.refused_render_ops(ops, enc, pass);
            }
        }
        draws.send(enc)?;
        Ok(Ok(ops))
    }

    /// [`Self::render_ops`] while a slot of the encoder's state may hold a refused object: each op
    /// goes alone while one does, and a draw is dropped. Out of line, as only a pass that binds a
    /// refused pipeline, depth/stencil state or tessellation factor buffer reaches it.
    #[cold]
    #[inline(never)]
    fn refused_render_ops<'o>(
        &self,
        mut ops: &'o [u8],
        enc: &P::RenderEncoder,
        pass: &render::PassFacts,
    ) -> RaiseResult<Result<&'o [u8], DecodeError>> {
        let mut draws = Draws::new();
        // Whether a slot of the encoder's state holds a refused object, read again after each op
        // that may mark one.
        let mut refused = pass.refused.drops_draws();
        while !ops.is_empty() {
            // Draws one after another go to the encoder together; each other op, after them. While
            // a slot of the encoder's state holds a refused object, each op goes alone, and a draw
            // is dropped.
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
            if !(refused && render::draws(tag)) {
                match render::decode_op(tag, body, self, enc, pass, ON) {
                    Ok(true) => {}
                    Ok(false) => return Err(unknown_op(tag)),
                    Err(raise) => contain(raise)?,
                }
                refused = pass.refused.drops_draws();
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
    /// names: a miss is looked up again by the op. Inlined at both of the render walk's loops: out
    /// of line, each op's look went through a call, and its answer through memory.
    #[inline(always)]
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
                self.laid_blit.set(false);
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
                // A pass the platform makes no encoder of is skipped whole ([`unmade`]).
                *encoder = Some(match cb.render_encoder(ON, &pass)? {
                    Some(enc) => {
                        super::work::render_encoder(self.context);
                        OpenEncoder::Render(enc, facts)
                    }
                    None => unmade(RGFX_OP_GROUP_RENDER),
                });
            }
            RGFX_OP_ENCODER_BEGIN_COMPUTE => {
                already_open(encoder, "ENCODER_BEGIN_COMPUTE")?;
                self.laid_blit.set(false);
                let desc = compute_pass(body)?;
                *encoder = Some(match cb.compute_encoder_for_pass(&desc)? {
                    Some(enc) => OpenEncoder::Compute(enc, refused::Marks::default()),
                    None => unmade(RGFX_OP_GROUP_COMPUTE),
                });
            }
            RGFX_OP_ENCODER_BEGIN_BLIT => {
                already_open(encoder, "ENCODER_BEGIN_BLIT")?;
                self.laid_blit.set(true);
                *encoder = Some(match cb.blit_encoder(ON)? {
                    Some(enc) => OpenEncoder::Blit(enc),
                    None => unmade(RGFX_OP_GROUP_BLIT),
                });
            }
            RGFX_OP_ENCODER_BEGIN_VIDEO => {
                already_open(encoder, "ENCODER_BEGIN_VIDEO")?;
                self.laid_blit.set(false);
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
            RGFX_OP_ENCODER_END => {
                // An encoder that may have written a laid-out cache or its wide home updates the
                // writers' fence as it ends, which the copies of its writes into the other side
                // wait for ([`Self::reconcile_laid`]).
                let fence = match encoder {
                    Some(OpenEncoder::Video(_) | OpenEncoder::Skipped(_)) | None => None,
                    Some(_) => self.laid_writes_fence(cb)?,
                };
                let fence = fence.as_ref();
                match encoder.take() {
                    Some(OpenEncoder::Render(enc, _)) => {
                        if let Some(fence) = fence {
                            enc.encode(
                                ON,
                                RenderCmd::UpdateFence {
                                    fence,
                                    stages: RENDER_STAGE_FRAGMENT,
                                },
                            )?;
                        }
                        enc.encode(ON, RenderCmd::EndEncoding)?;
                    }
                    Some(OpenEncoder::Compute(enc, _)) => {
                        if let Some(fence) = fence {
                            enc.encode(ON, ComputeCmd::UpdateFence { fence })?;
                        }
                        enc.encode(ON, ComputeCmd::EndEncoding)?;
                    }
                    Some(OpenEncoder::Blit(enc)) => {
                        if let Some(fence) = fence {
                            enc.encode(ON, BlitCmd::UpdateFence { fence })?;
                        }
                        enc.encode(ON, BlitCmd::EndEncoding)?;
                    }
                    Some(OpenEncoder::Video(_) | OpenEncoder::Skipped(_)) => {}
                    None => return Err(op_fault("ENCODER_END with no open encoder")),
                }
            }
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
            // Carried out before the walk, wherever they lie ([`take`]), the bytes its uploads
            // wrote at its begin; the guest's times are the device's, read as it took the buffer.
            RGFX_OP_SURFACE_USE
            | RGFX_OP_SUBMITTED
            | RGFX_OP_SURFACE_ORDER
            | RGFX_OP_COMMIT_ORDER
            | RGFX_OP_GUEST_TIMES
            | RGFX_OP_UPLOADS_WRITTEN => {}
            _ => return Err(unknown_op(tag)),
        }
        Ok(())
    }

    /// Builds a `P::RenderPassDescriptor` from an `ENCODER_BEGIN_RENDER` op's body
    /// (an [`RgfxRenderPassDesc`] then its `color_count`
    /// [`RgfxRenderPassColorAttachment`] records, then its sample positions, its visibility
    /// result buffer and its rasterization rate map where its flags say so), with what the pass's
    /// render ops' checks read of it. A body too short faults; a reference to a texture or a buffer the host does not
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
        // The rt platform's boot diagnostics count the pass where it takes a view of a video plane
        // image as an attachment, by its shape.
        let mut attached = reims_vgpu_vulkan::rt_platform::diag::PassAttachments::default();

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
            // The pass may write each attachment, and each resolve texture.
            self.texture_written(a.texture_obj_id);
            self.texture_written(a.resolve_texture_obj_id);
            {
                attached.colour(self.video_plane(a.texture_obj_id), false);
                if resolve.is_some() {
                    attached.colour(self.video_plane(a.resolve_texture_obj_id), true);
                }
            }
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
            off += size_of::<RgfxRenderPassVisibility>();
        }
        if desc.flags & RGFX_PASS_FLAG_RATE_MAP != 0 {
            let r = body
                .get(off..off + size_of::<RgfxRenderPassRateMap>())
                .and_then(|s| RgfxRenderPassRateMap::read_from_bytes(s).ok())
                .ok_or_else(|| {
                    op_fault("ENCODER_BEGIN_RENDER: no whole rasterization rate map record")
                })?;
            let map = self.require_rate_map(r.map_obj_id)?;
            pass.set_pass(&PassSet {
                rasterization_rate_map: Some(Some(map)),
                ..PassSet::default()
            });
        }

        if desc.depth.texture_obj_id != 0 {
            let d = &desc.depth;
            let tex = self.require_texture(d.texture_obj_id)?;
            let resolve = self.get_texture(d.resolve_texture_obj_id)?;
            self.texture_written(d.texture_obj_id);
            self.texture_written(d.resolve_texture_obj_id);
            attached.depth_or_stencil(self.video_plane(d.resolve_texture_obj_id));
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
            self.texture_written(s.texture_obj_id);
            self.texture_written(s.resolve_texture_obj_id);
            attached.depth_or_stencil(self.video_plane(s.resolve_texture_obj_id));
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

        reims_vgpu_vulkan::rt_platform::diag::plane_pass(&attached);
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

/// The raise of an encoder op: one that named an object whose create the host refused drops the
/// op, and the walk goes on ([`refused`]); any other is the op's. Each walk matches an op's answer
/// where the op wrote it, and passes only a raise here, out of line: an answer passed on whole to
/// a function (a [`Raise`]'s size) was copied at every op.
#[cold]
#[inline(never)]
fn contain(raise: Raise) -> RaiseResult<()> {
    if refused::is_refused(&raise) {
        Ok(())
    } else {
        Err(raise)
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

/// Whether an encoder begins in `rest`, the ops after an encoder's end, so a part cut there
/// leaves work for the next ([`super::split`], "Where a part is cut"): between encoders a stream
/// holds only framing ops, which put no work on the GPU of their own, an event's signal or wait
/// and what the walk reads at the take, `SUBMITTED` among them, which the bundle puts last.
/// Reads op headers up to the first encoder's begin. An op of another group, and one that does
/// not parse, answer `true`: the walk faults there, after the cut as before.
fn encoder_follows(mut rest: &[u8]) -> bool {
    while !rest.is_empty() {
        let Some((tag, _, len)) = next_op(rest) else {
            return true;
        };
        if op_group(tag) != RGFX_OP_GROUP_FRAMING
            || matches!(
                tag,
                RGFX_OP_ENCODER_BEGIN_RENDER
                    | RGFX_OP_ENCODER_BEGIN_COMPUTE
                    | RGFX_OP_ENCODER_BEGIN_BLIT
                    | RGFX_OP_ENCODER_BEGIN_VIDEO
            )
        {
            return true;
        }
        rest = &rest[len..];
    }
    false
}

/// The degradation a command buffer dealt to two threads counts where the platform keeps it
/// whole at a chunk's end for a reason of [`KeptWhole`] ([`kept_whole`]).
pub(super) const KEPT_WHOLE: &str = "two-ranges-kept-whole";

/// Whether this process logged a buffer kept whole for each reason of [`KeptWhole::ALL`], at the
/// reason's place there: the line is logged the first time alone, the rest counted
/// ([`KEPT_WHOLE`], docs/conventions.md).
static KEPT_WHOLE_LOGGED: [AtomicBool; KeptWhole::ALL.len()] =
    [const { AtomicBool::new(false) }; KeptWhole::ALL.len()];

/// The fault of a command buffer dealt to two threads ([`GraphicsDecoder::walk_dealt`]) that
/// the platform keeps whole at a chunk's end for `reason`, no encoder failing it: its later chunks
/// were walked into other command buffers, which then run nothing, so the buffer fails at its
/// stand-in, where a walk of it whole would not. [`deal`] takes no buffer with an op that sets
/// any [`KeptWhole`] reason, and its tests pin the two lists together; so it is not reached, and
/// where it is, it shows: counted as a degradation ([`KEPT_WHOLE`]) and logged once per reason
/// (docs/conventions.md). It fails the buffer rather than walk its later chunks again.
fn kept_whole(reason: KeptWhole) -> Raise {
    reims_vgpu_vulkan::answer::count_kind(KEPT_WHOLE);
    let at = KeptWhole::ALL
        .iter()
        .position(|&r| r == reason)
        .unwrap_or(0);
    if !KEPT_WHOLE_LOGGED[at].swap(true, Ordering::Relaxed) {
        log::error!(
            target: "reims_vgpu::graphics", kind:% = reason.code();
            "a command buffer dealt to two threads was kept whole at a chunk's end ({}), no \
             encoder failing it: it fails, its later chunks being in other command buffers; the \
             deal's choice of buffers lets in an op that keeps one whole (logged once per \
             reason, counted as {KEPT_WHOLE})",
            reason.code()
        );
    }
    op_fault(format_args!(
        "the platform kept a command buffer dealt to two threads whole at a chunk's end ({}): \
         its later chunks are in other command buffers",
        reason.code()
    ))
}

/// The least op bytes of a command buffer dealt to two threads at once ([`deal`]): one of fewer
/// walks in a fraction of a millisecond, as long as handing chunks over takes on the walk's
/// thread and the helper's wake.
pub(super) const DEAL_MIN_BYTES: usize = 16 * 1024;

/// The least encoders of a command buffer dealt to two threads at once ([`deal`]): the runtime's
/// costs a buffer is cut for come by the encoder (its pass ends, [`super::split`]).
pub(super) const DEAL_MIN_ENCODERS: usize = 8;

/// The op bytes of a chunk dealt to a thread, at least ([`deal`]): each chunk is a part, which
/// the GPU runs while the walk goes on, and the last one's GPU work is left at the walk's end.
/// WindowServer's frame in Launchpad's open view holds about 125 KiB of ops (m3-max-31226),
/// walked in 14.8 ms on one thread (m3-max-31216; an RTX 3090 host, every driver listed,
/// measured): so about 31 chunks of about 0.5 ms of walk each, near the 36 parts its walk on one
/// thread cut of it (m3-max-30753, [`super::split::SPLIT_AFTER`]).
pub(super) const DEAL_CHUNK_BYTES: usize = 4 * 1024;

/// The most chunks a command buffer is dealt in where work its queue committed before is not yet
/// complete as its walk starts: one a thread. Its chunks are cut [`DEAL_CHUNK_BYTES`] apart so the
/// GPU, idle, starts on the first while the rest is walked; behind earlier work the GPU is busy,
/// and each further chunk is one more part, so one more submission to the driver and one more job
/// of its scheduler, which the GPU waits on where it runs out of work ([`super::split`], "How many
/// parts").
const DEALT_BEHIND_WORK: usize = 2;

/// Where a command buffer's op stream `ops` is cut into chunks dealt to two threads at once
/// ([`super::split`], "Two threads at once"): at the end of each encoder that another encoder
/// follows ([`encoder_follows`]), that ends `chunk_bytes` of ops after the last cut or more (a
/// `most`th of the stream, where that is more), into `most` chunks at most; where it holds
/// [`DEAL_MIN_ENCODERS`] encoders and `min_bytes` of ops at least, every one of them a render or
/// blit encoder or an op the walk reads before it starts (`SURFACE_USE`, `SUBMITTED`,
/// `SURFACE_ORDER`, `COMMIT_ORDER`, `GUEST_TIMES`), and none of them a pass of a rasterization
/// rate map, an indirect command buffer's execution, reset, copy or optimization, or a residency
/// of resources (`useResource:`, which may name an indirect command buffer): what each of those
/// does keeps the platform's progress of the command buffer, or commands read at its commit,
/// which keep it whole (the Vulkan platform's `CommandBufferObject::splits`). So no event's
/// signal or wait, compute or video encoder, rate map or indirect command buffer lies in any
/// chunk, every chunk holds an encoder, and the platform cuts the buffer at each chunk's end, or
/// keeps it whole only where it fails from an encoder in it on ([`PartCut`]). `None` otherwise,
/// and for a stream that does not parse, which the walk then faults at. Reads each op's header,
/// and a render pass's flags; a stream shorter than `min_bytes` is not read.
fn deal(ops: &[u8], min_bytes: usize, chunk_bytes: usize, most: usize) -> Option<Vec<usize>> {
    if ops.len() < min_bytes || most < 2 {
        return None;
    }
    let each = chunk_bytes.max(ops.len() / most);
    let (mut at, mut encoders, mut from) = (0, 0, 0);
    // Where the last encoder seen begins: a cut after it has no encoder after it.
    let mut last_begin = 0;
    let mut cuts = Vec::new();
    while at < ops.len() {
        let (tag, body, len) = next_op(&ops[at..])?;
        match tag {
            RGFX_OP_ENCODER_BEGIN_RENDER => {
                let pass = read::<RgfxRenderPassDesc>(body)?;
                if pass.flags & RGFX_PASS_FLAG_RATE_MAP != 0 {
                    return None;
                }
                encoders += 1;
                last_begin = at;
            }
            RGFX_OP_ENCODER_BEGIN_BLIT => {
                encoders += 1;
                last_begin = at;
            }
            RGFX_OP_ENCODER_END => {
                let end = at + len;
                if end < ops.len() && end - from >= each && cuts.len() + 1 < most {
                    cuts.push(end);
                    from = end;
                }
            }
            RGFX_OP_SURFACE_USE
            | RGFX_OP_SUBMITTED
            | RGFX_OP_SURFACE_ORDER
            | RGFX_OP_COMMIT_ORDER
            | RGFX_OP_GUEST_TIMES
            | RGFX_OP_UPLOADS_WRITTEN => {}
            RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER
            | RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT
            | RGFX_OP_RENDER_USE_RESOURCES
            | RGFX_OP_RENDER_USE_RESOURCES_STAGES
            | RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER
            | RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER
            | RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER => return None,
            tag if matches!(op_group(tag), RGFX_OP_GROUP_RENDER | RGFX_OP_GROUP_BLIT) => {}
            _ => return None,
        }
        at += len;
    }
    // The last chunk holds an encoder: a cut after the last one would leave it only the framing
    // ops after it, no work.
    cuts.retain(|&cut| cut <= last_begin);
    (!cuts.is_empty() && encoders >= DEAL_MIN_ENCODERS).then_some(cuts)
}

/// Which of the `chunks` of a command buffer dealt to two threads at once the helper walks
/// ([`super::split`], "Two threads at once"), `true` at each of the helper's. The first chunk is
/// the lane's thread's, walked into the buffer's own command buffer. The others go the longest
/// first (by their op bytes, as far as the stream tells a walk's length), each to the thread
/// dealt fewer bytes so far, the lane's thread on a tie: so the longest chunk starts as the walk
/// starts, on whichever thread is free, and chunks of one length go in turn, the odd ones to the
/// helper. Each thread walks its chunks in their order, and they are committed in their order
/// whichever thread walked them, so only when the walk ends moves.
///
/// Dealt in turn, Unigine Heaven's frame buffer (five chunks of 161, 135, 471, 51 and under 1 KiB)
/// had the lane's thread walk its 471 KiB pass, 12 to 15 ms, only after the 6 ms first chunk,
/// while the helper walked the second and fourth and then waited: the run's 10,022 dealt walks
/// took 15.3 ms from the hand-over to the walk's end at the median and 25.3 ms at the 90th
/// percentile (1920 x 1080, an RX 7900 XTX host, m3-max-32395; measured). From the chunk walks the
/// trace gave, the turns end the walks in 12.6 and 22.7 ms, and this deal in 9.7 and 16.4 ms
/// (modelled). WindowServer's composite of the app's window waits for the app's frame, which its
/// commit order holds until that walk ends.
fn helper_chunks(chunks: &[Range<usize>]) -> Vec<bool> {
    let mut helper = vec![false; chunks.len()];
    let mut longest_first: Vec<usize> = (1..chunks.len()).collect();
    longest_first.sort_by_key(|&i| std::cmp::Reverse(chunks[i].len()));
    let (mut here, mut there) = (chunks.first().map_or(0, |c| c.len()), 0);
    for i in longest_first {
        if there < here {
            helper[i] = true;
            there += chunks[i].len();
        } else {
            here += chunks[i].len();
        }
    }
    helper
}

/// What the trace's `cenc` counts of a command buffer's op stream `ops`, read only while the
/// trace is on: its ops, its encoders (their ends), its dispatches ([`compute::dispatches`]), and
/// its fence updates and fence waits, of every encoder kind.
fn walk_counts(mut ops: &[u8]) -> [u64; 5] {
    let mut counts = [0u64; 5];
    while let Some((tag, _, len)) = next_op(ops) {
        counts[0] += 1;
        match tag {
            RGFX_OP_ENCODER_END => counts[1] += 1,
            RGFX_OP_COMPUTE_UPDATE_FENCE
            | RGFX_OP_RENDER_UPDATE_FENCE
            | RGFX_OP_BLIT_UPDATE_FENCE => {
                counts[3] += 1;
            }
            RGFX_OP_COMPUTE_WAIT_FOR_FENCE
            | RGFX_OP_RENDER_WAIT_FOR_FENCE
            | RGFX_OP_BLIT_WAIT_FOR_FENCE => counts[4] += 1,
            tag if op_group(tag) == RGFX_OP_GROUP_COMPUTE && compute::dispatches(tag) => {
                counts[2] += 1;
            }
            _ => {}
        }
        ops = &ops[len..];
    }
    counts
}

/// Calls `found` with each object id (not nil) the compute and render ops of the op stream `ops`
/// name, read as the walk reads them ([`next_op`], [`compute::names`], [`render::names`]), in
/// stream order, until it answers false, the stream ends or is malformed, or a compute or render
/// op's body is too short for what it names or its tag is one the walk does not decode: where the
/// walk faults, if it gets there.
fn each_named(mut ops: &[u8], mut found: impl FnMut(u64) -> bool) {
    while let Some((tag, body, len)) = next_op(ops) {
        let more = match op_group(tag) {
            RGFX_OP_GROUP_COMPUTE => compute::names(tag, body, &mut found),
            RGFX_OP_GROUP_RENDER => render::names(tag, body, &mut found),
            _ => Some(true),
        };
        if more != Some(true) {
            return;
        }
        ops = &ops[len..];
    }
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

/// The encoder of the op group `group` (render, compute or blit) at its `ENCODER_BEGIN_*` where
/// the platform made none: skipped whole, as a pass that names an object whose create the host
/// refused is ([`OpenEncoder::Skipped`]). The Vulkan platform answers none where its runtime
/// refuses the pass (an attachment or a resolve it builds none of, a gap), and counts the
/// refusal (`rt-refused-<kind>`). Its ops up to its end are dropped and the command buffer goes
/// on, completing as it would have: a Metal app's sends to a nil encoder do nothing and fail no
/// command buffer, and Metal fails none for a pass it renders. The process's first of each group
/// is logged at WARN, the rest not, as the platform counts each refusal (docs/conventions.md).
#[cold]
#[inline(never)]
fn unmade<P: Platform>(group: u16) -> OpenEncoder<P> {
    /// Whether the first skip of each group (render, compute, blit) was logged.
    static LOGGED: [AtomicBool; 3] = [const { AtomicBool::new(false) }; 3];
    let (what, logged) = match group {
        RGFX_OP_GROUP_RENDER => ("render", &LOGGED[0]),
        RGFX_OP_GROUP_COMPUTE => ("compute", &LOGGED[1]),
        _ => ("blit", &LOGGED[2]),
    };
    if !logged.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu::graphics", kind:% = what;
            "the platform made no {what} encoder for a command buffer's pass: the pass is skipped \
             whole and the command buffer goes on (the rest are not logged)"
        );
    }
    OpenEncoder::Skipped(group)
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

    use reims_vgpu_graphics::ops::render::{
        RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES, RGFX_OP_RENDER_DRAW_PRIMITIVES,
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE, RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        RGFX_OP_RENDER_SET_SCISSOR_RECT, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
        RGFX_OP_RENDER_SET_VIEWPORT, RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE,
        RgfxOpDrawIndexedPrimitives, RgfxOpDrawPrimitives, RgfxOpSetFragmentTexture,
        RgfxOpSetRenderPipelineState, RgfxOpSetScissorRect, RgfxOpSetVertexBuffer,
        RgfxOpSetViewport, RgfxOpSetVisibilityResultMode,
    };
    use reims_vgpu_graphics::ops::video::{
        RGFX_OP_VIDEO_DECODE_H264, RGFX_OP_VIDEO_DECODE_H265, RgfxVideoOpDecodeH264,
        RgfxVideoOpDecodeH265,
    };
    use reims_vgpu_graphics::wire::{
        RGFX_CB_INTERNAL, RGFX_CB_OK, RGFX_CMD_BUFFER_CREATE, RGFX_CMD_COMMAND_BUFFER,
        RGFX_CMD_EVENT_CREATE, RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SHARED_EVENT_CREATE,
        RGFX_CMD_SHARED_EVENT_SET_VALUE, RGFX_CMD_TEXTURE_CREATE, RGFX_CMD_TEXTURE_FROM_IOSURFACE,
        RGFX_CMD_VIDEO_SESSION_CREATE, RGFX_EVT_CB_COMPLETE, RGFX_SURFACE_NO_WRITERS,
        RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RgfxBufferCreate,
        RgfxCbComplete, RgfxObjectCreate, RgfxQueueCreate, RgfxSharedEventCreate,
        RgfxSharedEventSetValue, RgfxTextureCreate, RgfxTextureDescriptor,
        RgfxTextureFromIOSurface, RgfxVideoSessionCreate,
    };
    use reims_vgpu_video::core::device_memory::MemoryIdentity;
    use reims_vgpu_vulkan::core::platform::Handle;
    use reims_vgpu_vulkan::core::platform::ObjId;
    use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor;
    use reims_vgpu_vulkan::core::platform::values::AttachmentState;
    use reims_vgpu_vulkan::core::sync::Mutex;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use reims_vgpu_vulkan::core::testkit::HeldCall;
    use reims_vgpu_vulkan::core::testkit::Step;
    use reims_vgpu_vulkan::core::testkit::calllog::Arg;
    use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;
    use zerocopy::IntoBytes;

    use super::super::split::SPLIT_AFTER;
    use super::super::{
        Decoded, GraphicsVideo, GraphicsVideoSession, Landed, VideoDecode, VideoPlane, no_raise,
    };
    use super::*;
    use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;

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
        setup_over(0x10000)
    }

    /// [`setup`], over a region of `size` bytes.
    fn setup_over(size: usize) -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_vgpu_graphics::region::Region::new(size).unwrap()),
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
    /// A plane's memory: its identity, where the plane's buffer lies in it, and its bytes.
    type SeenPages = [Option<(MemoryIdentity, u64, usize)>; 2];

    /// What a fake host's video was handed: each decode, whether each of its planes was in a
    /// device cache, and the thread each of its sessions ended on.
    #[derive(Default)]
    struct VideoSeen {
        decodes: Mutex<Vec<SeenDecode>>,
        cached: Mutex<Vec<[bool; 2]>>,
        /// Each plane's memory: its identity, where the plane's buffer lies in it, and its bytes.
        pages: Mutex<Vec<SeenPages>>,
        output_rows: Mutex<Vec<u32>>,
        ended_on: Mutex<Vec<Option<String>>>,
        /// Each decode's picture fails to land, as its session finds once the decode ran.
        unlanded: std::sync::atomic::AtomicBool,
        /// Each decode answers no picture, as for one the stream's random access skips.
        skipped: std::sync::atomic::AtomicBool,
        /// Each decode's [`Landed`], which a test tells of the decode done ([`Landed::decoded`]).
        landings: Mutex<Vec<Arc<Landed>>>,
        /// Each decode tells of itself done before it answers, as a session whose decode ended
        /// before its command buffer's commit does.
        decoded_at_once: std::sync::atomic::AtomicBool,
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
        fn caps(&self) -> Vec<reims_vgpu_graphics::wire::RgfxVideoCodecCaps> {
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
        fn event(&self) -> reims_vgpu_vulkan::core::platform::event::EventRef<'_, FakePlatform> {
            reims_vgpu_vulkan::core::platform::event::EventRef::Shared(&self.event)
        }

        fn codec(&self) -> u32 {
            self.codec
        }

        fn decode(&self, d: &VideoDecode<'_, FakePlatform>) -> RaiseResult<Decoded> {
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
            let pages = |p: &VideoPlane<'_, FakePlatform>| {
                (p.pages.as_ref()).map(|pages| {
                    (
                        pages.memory.identity(),
                        pages.buffer_offset,
                        pages.memory.length(),
                    )
                })
            };
            self.seen
                .pages
                .lock()
                .push([pages(&d.luma), pages(&d.chroma)]);
            self.seen.output_rows.lock().push(d.output_rows);
            let landed = Arc::new(Landed::default());
            if self.seen.unlanded.load(Ordering::Relaxed) {
                landed.fail();
            }
            if self.seen.decoded_at_once.load(Ordering::Relaxed) {
                landed.decoded();
            }
            self.seen.landings.lock().push(Arc::clone(&landed));
            Ok(Decoded {
                value: self.decodes.fetch_add(1, Ordering::Relaxed) as u64 + 1,
                skipped: self.seen.skipped.load(Ordering::Relaxed),
                landed: Some(landed),
            })
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
    /// access unit's place in its buffer, the parameter sets that follow the op's body, where
    /// each plane's texture lies in the surface's buffer, and the rows the surface holds
    /// (`output_rows`); the command buffer then waits for the
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
            output_rows: 16,
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
        assert!(
            seen.pages.lock()[0].iter().all(Option::is_none),
            "a device buffer is no memory the host's video reaches in place"
        );
        assert_eq!(*seen.output_rows.lock(), [16], "the rows the surface holds");
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

    /// A decode's command buffer completes as its GPU work did where its picture landed, and
    /// failed, Metal's internal error with its code and its GPU times, where it did not (its
    /// planes refused, or a decode the hardware failed once it ran), which the guest's video
    /// driver answers as a decode error; never as a picture decoded.
    #[test]
    fn a_decode_whose_picture_did_not_land_fails_its_command_buffer() {
        let (p, mut dec) = setup();
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
        let captured = Arc::new(Mutex::new(Vec::<RgfxCbComplete>::new()));
        let sink = Arc::clone(&captured);
        dec.set_completion_sink(Arc::new(move |record| sink.lock().push(record)));
        assert_eq!(
            dec.decode(&device, video_session_create(40).as_bytes())
                .unwrap(),
            Ok(())
        );
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
        let decode = RgfxVideoOpDecodeH264 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 16,
            bitstream_size: 100,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            parameter_sets_size: 0,
            output_rows: 16,
        };
        let pass = RgfxOpVideoPass { session_obj_id: 40 };
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes());
        ops.extend(op(RGFX_OP_VIDEO_DECODE_H264, decode.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        for (cb_obj_id, unlanded) in [(100, false), (101, true)] {
            seen.unlanded.store(unlanded, Ordering::Relaxed);
            let message = command_buffer_message(cb_obj_id, cb_obj_id, &ops);
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
            let committed = p.world().committed();
            assert_eq!(committed.len(), 1);
            assert!(p.world().set_gpu_times(committed[0], 7_000, 9_000));
            p.world().complete_all(CompleteOn::Caller);
        }
        let recs: Vec<_> = (captured.lock().iter())
            .map(|r| {
                (
                    r.cb_obj_id,
                    r.status,
                    r.error_code,
                    r.gpu_start_ns,
                    r.gpu_end_ns,
                )
            })
            .collect();
        assert_eq!(
            recs,
            [
                (100, RGFX_CB_OK, 0, 7_000, 9_000),
                (101, RGFX_CB_INTERNAL, 1, 7_000, 9_000)
            ]
        );
    }

    /// A plane of a buffer over host memory in a block of the region reaches the host's video as
    /// the block's memory: the block's bytes, the buffer's place in them, and an identity that
    /// ends at the block's `BLOCK_RELEASE`, one for the block whichever decode names it.
    #[test]
    fn a_video_decode_s_planes_carry_their_block_s_memory() {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN,
            RgfxBufferOverHostMemory,
        };

        let block = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = (block + 0x10_0000 + RGFX_CURSOR_SPAN) as usize;
        let (p, mut dec) = setup_over(size);
        let device = p.device();
        let seen = Arc::new(VideoSeen::default());
        dec.set_video(Arc::new(FakeVideo(Arc::clone(&seen), p.device())));
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
        assert_eq!(dec.decode(&device, session.as_bytes()).unwrap(), Ok(()));
        dec.region.grant(0, block, 0x1_0000);
        let buffer = RgfxBufferOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                size: size_of::<RgfxBufferOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id: 6,
            gpu_address: 0,
            region_offset: block + 0x1000,
            length: 0x8000,
            options: 0,
            reserved_34: 0,
        };
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
        for t in [
            surface_plane_texture(7, 0, 10, 32, 64, 16),
            surface_plane_texture(8, 1, 30, 32 + 64 * 16, 32, 8),
        ] {
            assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));
        }
        let decode = RgfxVideoOpDecodeH264 {
            bitstream_buffer_obj_id: 2,
            bitstream_offset: 16,
            bitstream_size: 100,
            luma_texture_obj_id: 7,
            chroma_texture_obj_id: 8,
            parameter_sets_size: 0,
            output_rows: 0,
        };
        let pass = RgfxOpVideoPass { session_obj_id: 40 };
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_VIDEO, pass.as_bytes());
        ops.extend(op(RGFX_OP_VIDEO_DECODE_H264, decode.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        for seq in [1, 2] {
            let message = command_buffer_message(100 + seq, seq, &ops);
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        }
        let pages = seen.pages.lock().clone();
        assert_eq!(pages.len(), 2);
        let mut keys = Vec::new();
        for planes in &pages {
            for plane in planes {
                let (identity, at, bytes) = plane.clone().expect("the block's memory");
                assert_eq!((at, bytes), (0x1000, 0x1_0000));
                assert!(identity.is_live());
                keys.push(identity.key());
            }
        }
        assert!(
            keys.iter().all(|&k| k == keys[0]),
            "one identity for the block"
        );
        dec.block_released(&device, block, 0, Box::new(|| ()), no_raise());
        assert!(
            pages
                .iter()
                .flatten()
                .flatten()
                .all(|(identity, ..)| !identity.is_live()),
            "the block's release ends it"
        );
    }

    /// Whether a thread waits on the CPU for a buffer's shared event since `mark`, as the host's
    /// work started off a command buffer waits for the buffers of other queues
    /// (`GraphicsDecoder::wait_for_order_on_cpu`): polled until it does, [`PATIENCE`] times at
    /// most.
    ///
    /// [`PATIENCE`]: crate::graphics::tests::PATIENCE
    fn waits_on_the_cpu(p: &FakePlatform, mark: u64) -> bool {
        crate::graphics::tests::eventually(|| {
            (p.call_log().sels_since(mark)).contains(&"waitUntilSignaledValue:timeoutMS:")
        })
    }

    /// A video decode of a command buffer that waits for a buffer of another queue starts once
    /// that buffer has finished: the session's work runs on a queue of its own, which the command
    /// buffer's GPU wait does not hold, so the host waits on the CPU first. A decode whose named
    /// buffer has finished starts at once.
    #[test]
    fn a_video_decode_waits_for_the_buffers_its_command_buffer_runs_after() {
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_ORDER, RgfxSurfaceOrderWait};

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
        let mark = p.call_log().mark();
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message).unwrap());
            assert!(
                waits_on_the_cpu(&p, mark),
                "the decode waits on the CPU for the read"
            );
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
        video_rig(false)
    }

    /// [`cached_video`]'s rig, its block, where `carrying`, deferred carrying its first page, a
    /// page of guest RAM, the other a run of zeros, as a decoder's new output with a page its
    /// process wrote without a lock is (`RGFX_FEATURE_SURFACE_DEFERRED_CARRY`).
    fn video_rig(carrying: bool) -> (super::super::surface_pages::tests::Rig, Arc<VideoSeen>) {
        use super::super::surface_pages::tests::{BLOCK, RAM, rig};
        use reims_vgpu_graphics::wire::{
            RGFX_PAGE_RUN_ZEROS, RGFX_STATUS_OK, RGFX_SURFACE_PAGES_CACHED,
            RGFX_SURFACE_PAGES_DEFERRED, RgfxPageRun,
        };

        let mut r = rig(8);
        let page = r.page;
        if carrying {
            r.dec.set_deferred_budget(8 * page);
            r.dec.set_deferred_carries();
            let runs = [
                RgfxPageRun {
                    address: RAM + page,
                    length: page,
                },
                RgfxPageRun {
                    address: RGFX_PAGE_RUN_ZEROS,
                    length: page,
                },
            ];
            let flags = RGFX_SURFACE_PAGES_CACHED | RGFX_SURFACE_PAGES_DEFERRED;
            assert_eq!(
                r.pages_of_runs(BLOCK, 0, &runs, 2 * page, flags, None),
                RGFX_STATUS_OK
            );
        } else {
            assert_eq!(
                r.pages_cached(BLOCK, &[(1, 2)], 2 * page, true),
                RGFX_STATUS_OK
            );
        }
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
        use reims_vgpu_graphics::ops::{
            RGFX_OP_SURFACE_USE, RGFX_SURFACE_USE_WRITES, RgfxOpSurfaceUse,
        };

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

    /// [`cached_video`]'s rig with its textures 7 and 8 made again, each over the whole of its
    /// plane of a two-plane 4:2:0 surface 64 by 16 pixels (its message's facts): a decoder's
    /// output, whose planes' rows are the surface's bytes 32 to 1568. The block's cache owes its
    /// first fill, its zeros done.
    fn cached_video_whole() -> (super::super::surface_pages::tests::Rig, Arc<VideoSeen>) {
        video_whole(false)
    }

    /// [`cached_video_whole`] of [`video_rig`]'s rig, its block deferred carrying a page where
    /// `carrying`.
    fn video_whole(carrying: bool) -> (super::super::surface_pages::tests::Rig, Arc<VideoSeen>) {
        use reims_vgpu_graphics::wire::{
            RGFX_SURFACE_FACTS_OFFSET, RGFX_YCBCR_MATRIX_ITU_R_709_2, RgfxSurfaceFacts,
        };

        let (mut r, seen) = video_rig(carrying);
        let facts = RgfxSurfaceFacts {
            surface_pixel_format: u32::from_be_bytes(*b"420v"),
            plane_count: 2,
            width: 64,
            height: 16,
            chroma_offset: 32 + 64 * 16,
            chroma_bytes_per_row: 64,
            ycbcr_matrix: RGFX_YCBCR_MATRIX_ITU_R_709_2,
            reserved_24: 0,
        };
        for mut t in [
            surface_plane_texture(7, 0, 10, 32, 64, 16),
            surface_plane_texture(8, 1, 30, 32 + 64 * 16, 32, 8),
        ] {
            let at = RGFX_SURFACE_FACTS_OFFSET as usize;
            t.header.size = (at + size_of::<RgfxSurfaceFacts>()) as u32;
            let mut message = t.as_bytes().to_vec();
            message.resize(at, 0);
            message.extend_from_slice(facts.as_bytes());
            assert_eq!(r.dec.decode(&r.device, &message).unwrap(), Ok(()));
        }
        // The make's zeros done, which a decode waits for on the CPU.
        r.p.world().complete_all(CompleteOn::Caller);
        (r, seen)
    }

    /// The copies into `cache` since `mark`, and the copies out of it: each one's source offset
    /// and size, from the platform's buffer copies.
    fn cache_copies(
        r: &super::super::surface_pages::tests::Rig,
        mark: u64,
        cache: u64,
    ) -> (usize, Vec<(u64, u64)>) {
        const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
        let calls = r.p.call_log().since(mark);
        let id = |a: &Arg| a.as_obj().map_or(0, |o| o.0);
        let into = (calls.iter())
            .filter(|c| c.sel == COPY && id(&c.args[2]) == cache)
            .count();
        let out = (calls.iter())
            .filter(|c| c.sel == COPY && id(&c.args[0]) == cache)
            .map(|c| {
                (
                    c.args[1].as_u64().unwrap_or(0),
                    c.args[4].as_u64().unwrap_or(0),
                )
            })
            .collect();
        (into, out)
    }

    /// The id of [`cached_video`]'s block's cache.
    fn video_cache(r: &super::super::surface_pages::tests::Rig) -> u64 {
        use super::super::surface_pages::tests::BLOCK;
        Arg::obj(r.dec.caches().buffer(BLOCK).unwrap())
            .as_obj()
            .unwrap()
            .0
    }

    /// The `CB_COMPLETE` records a sink took: each one's buffer, status and GPU times.
    type Records = Arc<Mutex<Vec<RgfxCbComplete>>>;

    /// Each record's buffer, status and GPU times.
    fn posted(records: &Records) -> Vec<(u64, u32, u64, u64)> {
        (records.lock().iter())
            .map(|r| (r.cb_obj_id, r.status, r.gpu_start_ns, r.gpu_end_ns))
            .collect()
    }

    /// [`cached_video_whole`]'s rig with queue 9 too, its `CB_COMPLETE` records taken, and the
    /// seqnos of the ends told of buffers whose completions went at their decodes
    /// ([`super::super::AheadSinks::finished`]).
    fn video_posting() -> (
        super::super::surface_pages::tests::Rig,
        Arc<VideoSeen>,
        Records,
        Arc<Mutex<Vec<u64>>>,
    ) {
        let (mut r, seen) = cached_video_whole();
        let records = Records::default();
        let sink = Arc::clone(&records);
        (r.dec).set_completion_sink(Arc::new(move |record| sink.lock().push(record)));
        let ends = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&ends);
        (r.dec).set_finished_sink(Arc::new(move |seqno| sink.lock().push(seqno)));
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 9,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(r.dec.decode(&r.device, queue.as_bytes()).unwrap(), Ok(()));
        (r, seen, records, ends)
    }

    /// `message`, a `COMMAND_BUFFER`, on queue `queue`.
    fn on_queue(mut message: Vec<u8>, queue: u64) -> Vec<u8> {
        let at = std::mem::offset_of!(RgfxCommandBuffer, queue_obj_id);
        message[at..at + 8].copy_from_slice(&queue.to_le_bytes());
        message
    }

    /// The `COMMAND_BUFFER` `cb_obj_id` at `seqno` on queue 9 of [`video_posting`]'s rig that
    /// reads its block and holds nothing else.
    fn reads_cached_video(cb_obj_id: u64, seqno: u64) -> Vec<u8> {
        use super::super::surface_pages::tests::BLOCK;
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_USE, RgfxOpSurfaceUse};

        let surface_use = RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen: 0,
            flags: 0,
            reserved_14: 0,
        };
        let ops = op(RGFX_OP_SURFACE_USE, surface_use.as_bytes());
        on_queue(command_buffer_message(cb_obj_id, seqno, &ops), 9)
    }

    /// Each `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` since `mark`: whether it
    /// signals, its event's id and its value.
    fn event_calls(
        r: &super::super::surface_pages::tests::Rig,
        mark: u64,
    ) -> Vec<(bool, u64, u64)> {
        (r.p.call_log().since(mark).iter())
            .filter_map(|c| {
                let signals = if c.sel == "encodeSignalEvent:value:" {
                    true
                } else if c.sel == "encodeWaitForEvent:value:" {
                    false
                } else {
                    return None;
                };
                let event = c.args.first().and_then(Arg::as_obj).map_or(0, |o| o.0);
                Some((
                    signals,
                    event,
                    c.args.get(1).and_then(Arg::as_u64).unwrap_or(0),
                ))
            })
            .collect()
    }

    /// A command buffer of a decode alone into device caches posts its `CB_COMPLETE` as its
    /// session tells of the decode done, with no GPU times, ahead of its platform end, and posts
    /// no second record at that end. Everything else moves at the end: its context's drain, its
    /// place among its queue's buffers in flight, its decode's place among those kept by their
    /// blocks, and the notice of its end, which the host's presents waiting for it take
    /// ([`super::super::AheadSinks::finished`]).
    #[test]
    fn a_decode_alone_into_caches_completes_at_its_decode_and_ends_at_its_end() {
        use super::super::surface_pages::tests::BLOCK;

        let (r, seen, records, ends) = video_posting();
        let in_flight = |r: &super::super::surface_pages::tests::Rig| {
            let counts = r.dec.queue_counts(1).unwrap();
            counts.in_flight.load(Ordering::Acquire)
        };
        // Walked for context 0, whose drain then holds it.
        let mut walker = r.dec.sender(Some(0), r.dec.sinks.clone());
        assert_eq!(
            walker
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        walker.end_message();
        let drained = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let set = Arc::clone(&drained);
        drop(r.dec.drain_context(
            0,
            false,
            Box::new(move || set.store(true, Ordering::SeqCst)),
        ));
        assert!(posted(&records).is_empty(), "the decode is not done");
        let landed = Arc::clone(&seen.landings.lock()[0]);
        landed.decoded();
        assert_eq!(posted(&records), [(100, RGFX_CB_OK, 0, 0)]);
        assert!(ends.lock().is_empty());
        assert!(!drained.load(Ordering::SeqCst));
        assert_eq!(in_flight(&r), 1);
        assert!(r.dec.shared.completed_ahead.holds(BLOCK));
        assert_eq!(r.dec.shared.completed_ahead.posts(), 1);

        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(posted(&records).len(), 1, "no second record");
        assert_eq!(*ends.lock(), [1]);
        assert!(drained.load(Ordering::SeqCst));
        assert_eq!(in_flight(&r), 0);
        assert!(!r.dec.shared.completed_ahead.holds(BLOCK));
    }

    /// A completion posted at a decode keeps its queue's order: behind a buffer of the queue
    /// committed before it whose record its platform end posts, and posted at the buffer's commit
    /// where its decode was done before it. (The buffer before it holds no op: one that used the
    /// decode's block would hold the decode's start, on the CPU, until its end.)
    #[test]
    fn a_completion_at_a_decode_goes_in_its_queues_order() {
        let (mut r, seen, records, ends) = video_posting();
        assert_eq!(
            r.dec
                .decode(&r.device, &command_buffer_message(100, 1, &[]))
                .unwrap(),
            Ok(())
        );
        let first = r.p.world().committed();
        seen.decoded_at_once.store(true, Ordering::Relaxed);
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(101, 2, true))
                .unwrap(),
            Ok(())
        );
        assert!(posted(&records).is_empty(), "behind buffer 100");
        assert_eq!(seen.landings.lock().len(), 1);
        for id in first {
            r.p.world()
                .complete(id, CommandBufferStatus::COMPLETED, None, CompleteOn::Caller);
        }
        assert_eq!(
            (posted(&records).iter())
                .map(|&(cb, status, ..)| (cb, status))
                .collect::<Vec<_>>(),
            [(100, RGFX_CB_OK), (101, RGFX_CB_OK)]
        );
        assert_eq!(posted(&records)[1], (101, RGFX_CB_OK, 0, 0));
        assert!(ends.lock().is_empty(), "buffer 101's end is to come");
        r.p.world().complete_all(CompleteOn::Caller);
        assert_eq!(posted(&records).len(), 2);
        assert_eq!(*ends.lock(), [2]);
    }

    /// A buffer of another queue that uses a decode's output, begun after the decode's
    /// completion went at its decode, which the guest's slots then name no more, waits on the GPU
    /// for the decode's buffer's end as for a buffer it named: for the value of the decode's
    /// queue's timeline the decode's buffer signals. Counted as a buffer held; one begun after
    /// that end waits for nothing.
    #[test]
    fn a_reader_of_a_decode_completed_ahead_waits_on_the_gpu_for_its_end() {
        let (mut r, seen, records, _) = video_posting();
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        let signals: Vec<_> = (event_calls(&r, mark).into_iter())
            .filter(|&(signals, ..)| signals)
            .collect();
        Arc::clone(&seen.landings.lock()[0]).decoded();
        assert_eq!(posted(&records).len(), 1);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &reads_cached_video(200, 1))
                .unwrap(),
            Ok(())
        );
        let waits = |r: &super::super::surface_pages::tests::Rig, mark| {
            (event_calls(r, mark).into_iter())
                .filter(|&(signals, ..)| !signals)
                .collect::<Vec<_>>()
        };
        let waited = waits(&r, mark);
        assert_eq!(waited.len(), 1, "{waited:?}");
        let (_, event, value) = waited[0];
        assert!(
            signals.contains(&(true, event, value)),
            "the decode's buffer signals {event} to {value} at its end: {signals:?}"
        );
        assert_eq!(r.dec.shared.surface_order.lock().counts().held, 1);

        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &reads_cached_video(201, 2))
                .unwrap(),
            Ok(())
        );
        assert!(waits(&r, mark).is_empty(), "its end has come");
    }

    /// A video decode of another queue into a decode's output, begun after the decode's
    /// completion went at its decode, waits on the CPU for the decode's buffer's end before its
    /// session's work starts, as it would for a buffer it named.
    #[test]
    fn a_decode_after_a_decode_completed_ahead_waits_on_the_cpu_for_its_end() {
        let (mut r, seen, records, _) = video_posting();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        Arc::clone(&seen.landings.lock()[0]).decoded();
        assert_eq!(posted(&records).len(), 1);
        let message = on_queue(writes_cached_video(200, 1, true), 9);
        let mut walker = r.dec.sender(None, super::super::Sinks::default());
        let mark = r.p.call_log().mark();
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&r.p.device(), &message).unwrap());
            assert!(
                waits_on_the_cpu(&r.p, mark),
                "the decode waits on the CPU for buffer 100's end"
            );
            assert_eq!(
                seen.decodes.lock().len(),
                1,
                "it waits for buffer 100's end"
            );
            r.p.world().complete_all(CompleteOn::Caller);
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(seen.decodes.lock().len(), 2);
    }

    /// A lock's download of a decode's output asked after the decode's completion went at its
    /// decode copies the cache out once the decode's buffer has ended: its copy waits on the GPU
    /// for the value the decode's buffer signals at its end, with no wait of its own added.
    #[test]
    fn a_download_after_a_decode_completed_ahead_copies_after_its_end() {
        use super::super::surface_pages::tests::BLOCK;

        let (mut r, seen, records, _) = video_posting();
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        let signals: Vec<_> = (event_calls(&r, mark).into_iter())
            .filter(|&(signals, ..)| signals)
            .collect();
        Arc::clone(&seen.landings.lock()[0]).decoded();
        assert_eq!(posted(&records).len(), 1);
        let mark = r.p.call_log().mark();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let set = Arc::clone(&done);
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            super::super::no_raise(),
        );
        let waits = event_calls(&r, mark);
        assert!(
            (waits.iter()).any(|&(s, event, value)| !s && signals.contains(&(true, event, value))),
            "the copy waits for the decode's buffer's end: {waits:?} {signals:?}"
        );
        assert!(!done.load(Ordering::SeqCst));
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A present of a decode's output asked after the decode's completion went at its decode,
    /// whose writers the guest's slots no longer name, waits for the decode's buffer as for a
    /// writer named: until its end. Counted. (The rig's guest names no writers at all.)
    #[test]
    fn a_present_after_a_decode_completed_ahead_waits_for_its_end() {
        let (mut r, seen, records, _) = video_posting();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        let writers = |r: &super::super::surface_pages::tests::Rig| {
            (r.dec.surface_present(0x99).unwrap().writers.iter())
                .map(|w| (w.queue_obj_id, w.seqno))
                .collect::<Vec<_>>()
        };
        Arc::clone(&seen.landings.lock()[0]).decoded();
        assert_eq!(posted(&records).len(), 1);
        assert_eq!(writers(&r), [(1, 1)]);
        assert_eq!(r.dec.shared.completed_ahead.presents_held(), 1);
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(writers(&r).is_empty());
    }

    /// A blit of another queue that copies from a decode's output, its luma texture's texels and
    /// its buffer's bytes, begun after the decode's completion went at its decode, waits on the
    /// GPU, ahead of its copies, for the value the decode's buffer signals at its end; and the
    /// decode's buffer signals it only behind its wait for its session's event to reach the
    /// decode's value, which the session signals once the picture has landed, its copy or its draw
    /// done (`super::super::super::video_host`'s tests,
    /// `a_picture_landed_behind_its_decode_done_is_posted_then_and_signalled_once_landed`). The
    /// guest names the output's block with the copies, as its driver names every surface a buffer
    /// uses.
    #[test]
    fn a_blit_from_a_decode_completed_ahead_copies_after_its_end() {
        use super::super::surface_pages::tests::BLOCK;
        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpCopyTextureToBuffer,
        };
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_USE, RgfxOpSurfaceUse};

        let (mut r, seen, records, _) = video_posting();
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        let decoding = event_calls(&r, mark);
        let session = (decoding.iter())
            .position(|&(signals, _, value)| !signals && value == 1)
            .expect("the decode's buffer waits for its session's value");
        let signals: Vec<_> = (decoding.iter().enumerate())
            .filter(|&(_, &(signals, ..))| signals)
            .map(|(at, &call)| (at, call))
            .collect();
        assert!(
            !signals.is_empty() && signals.iter().all(|&(at, _)| at > session),
            "its end's signals come behind its session's wait: {decoding:?}"
        );
        Arc::clone(&seen.landings.lock()[0]).decoded();
        assert_eq!(posted(&records).len(), 1);

        let surface_use = RgfxOpSurfaceUse {
            block_offset: BLOCK,
            cpu_gen: 0,
            flags: 0,
            reserved_14: 0,
        };
        let texels = RgfxBlitOpCopyTextureToBuffer {
            source_texture_obj_id: 7,
            destination_buffer_obj_id: 2,
            source_size_width: 64,
            source_size_height: 16,
            source_size_depth: 1,
            destination_bytes_per_row: 64,
            destination_bytes_per_image: 64 * 16,
            ..RgfxBlitOpCopyTextureToBuffer::default()
        };
        let bytes = RgfxBlitOpCopyBufferToBuffer {
            source_buffer_obj_id: 6,
            destination_buffer_obj_id: 2,
            source_offset: 32,
            destination_offset: 2048,
            size: 64 * 16,
        };
        let mut ops = op(RGFX_OP_SURFACE_USE, surface_use.as_bytes());
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        ops.extend(op(RGFX_OP_BLIT_COPY_TEXTURE_TO_BUFFER, texels.as_bytes()));
        ops.extend(op(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, bytes.as_bytes()));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let blit = on_queue(command_buffer_message(200, 1, &ops), 9);
        let mark = r.p.call_log().mark();
        assert_eq!(r.dec.decode(&r.device, &blit).unwrap(), Ok(()));
        let calls = r.p.call_log().since(mark);
        let id = |a: Option<&Arg>| a.and_then(Arg::as_obj).map_or(0, |o| o.0);
        let wait = (calls.iter())
            .position(|c| {
                c.sel == "encodeWaitForEvent:value:"
                    && signals.iter().any(|&(_, (_, event, value))| {
                        id(c.args.first()) == event && c.args.get(1) == Some(&Arg::U64(value))
                    })
            })
            .expect("the blit waits for the decode's buffer's end");
        let Some(ObjHandle::Buffer(into)) = r.dec.object(2) else {
            panic!("no buffer 2");
        };
        let into = Arg::obj(into);
        let copies: Vec<_> = (calls.iter().enumerate())
            .filter(|(_, c)| c.sel.starts_with("copyFrom") && c.args.contains(&into))
            .map(|(at, c)| (at, c.sel.split(':').next().unwrap_or_default()))
            .collect();
        assert!(
            (copies.iter()).any(|&(_, sel)| sel == "copyFromTexture")
                && (copies.iter()).any(|&(_, sel)| sel == "copyFromBuffer"),
            "{copies:?}"
        );
        assert!(
            copies.iter().all(|&(at, _)| at > wait),
            "a copy ahead of its wait: {copies:?}, the wait at {wait}"
        );
        assert_eq!(r.dec.shared.surface_order.lock().counts().held, 1);
    }

    /// A decoder's new output, a command buffer of a decode alone whose textures are the whole of
    /// the surface's two planes, takes no fill of its cache: the cache made as zeros, the decode
    /// waits for those, its picture lands over them, and nothing is copied in from the pages. The
    /// lock's download that follows copies the planes' rows alone into the pages, the bytes 32 to
    /// 1568, so every other byte of the pages, the surface's header among them, stays as the
    /// guest's CPU wrote it
    #[test]
    fn a_decoders_new_output_takes_no_fill_and_copies_out_its_planes_alone() {
        use super::super::surface_pages::tests::BLOCK;

        let (mut r, seen) = cached_video_whole();
        let cache = video_cache(&r);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        assert_eq!(*seen.cached.lock(), [[true, true]], "into the cache");
        let (into, _) = cache_copies(&r, mark, cache);
        assert_eq!(into, 0, "no fill");
        let counts = r.dec.caches().counts();
        assert_eq!(
            (counts.fills, counts.made_fills, counts.decode_firsts),
            (0, 0, 1)
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            super::super::no_raise(),
        );
        let (into, out) = cache_copies(&r, mark, cache);
        assert_eq!((into, out), (0, vec![(32, 1536)]), "the planes' rows alone");
        r.p.world().complete_all(CompleteOn::Caller);
        assert!(done.load(Ordering::SeqCst));
    }

    /// A decoder's new output deferred carrying a page its process wrote without a lock, a
    /// command buffer of a decode alone whose textures are the whole of the surface's two planes,
    /// takes no fill of its cache either: the carried page's first fill is taken as met, as a new
    /// cache's is, nothing copied in, and the cache holds the use's generation, so a later use of
    /// it fills nothing
    #[test]
    fn a_decode_takes_a_carried_fill_as_met() {
        let (mut r, seen) = video_whole(true);
        let cache = video_cache(&r);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        assert_eq!(*seen.cached.lock(), [[true, true]], "into the cache");
        let (into, _) = cache_copies(&r, mark, cache);
        assert_eq!(into, 0, "no fill");
        let counts = r.dec.caches().counts();
        assert_eq!(
            (
                counts.carried,
                counts.fills,
                counts.made_fills,
                counts.decode_firsts
            ),
            (1, 0, 0, 1)
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(101, 2, false))
                .unwrap(),
            Ok(())
        );
        let (into, _) = cache_copies(&r, mark, cache);
        assert_eq!(into, 0, "generation 0 held");
    }

    /// A decoder's new output whose picture fails to land holds the cache's zeros where the
    /// picture did not write, and no byte of the pages or of any other memory: its cache is the
    /// platform's new buffer, zeros as made, which the host fills no further, and nothing was
    /// copied into it. The command buffer completes failed, and the lock's download copies the
    /// planes' rows alone
    #[test]
    fn a_failed_landing_into_a_new_output_leaves_zeros_and_no_host_bytes() {
        use super::super::surface_pages::tests::BLOCK;

        let (mut r, seen) = cached_video_whole();
        let cache = video_cache(&r);
        // No fill of the cache from the rig's start: its zeros are the platform's.
        let fills = (r.p.call_log().calls().iter())
            .filter(|c| c.sel == "fillBuffer:range:value:")
            .filter(|c| c.args[0].as_obj().map(|o| o.0) == Some(cache))
            .count();
        assert_eq!(fills, 0, "the platform's zeros alone");
        seen.unlanded.store(true, Ordering::Relaxed);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let done = Arc::new(AtomicBool::new(false));
        let set = Arc::clone(&done);
        r.dec.surface_download(
            &r.device,
            BLOCK,
            Box::new(move || set.store(true, Ordering::SeqCst)),
            super::super::no_raise(),
        );
        r.p.world().complete_all(CompleteOn::Caller);
        let (into, out) = cache_copies(&r, mark, cache);
        assert_eq!(into, 0, "nothing copied into the cache");
        assert_eq!(out, [(32, 1536)], "the planes' rows alone copied out");
        assert!(done.load(Ordering::SeqCst));
    }

    /// A decode into a new output that answers no picture (one the stream's random access skips)
    /// makes the cache's first fill before anything reads it: in the finish of its command
    /// buffer, which still uses the cache, so a use after it waits for the fill
    #[test]
    fn a_skipped_picture_fills_its_new_output() {
        let (mut r, seen) = cached_video_whole();
        let cache = video_cache(&r);
        seen.skipped.store(true, Ordering::Relaxed);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, true))
                .unwrap(),
            Ok(())
        );
        let (into, _) = cache_copies(&r, mark, cache);
        assert_eq!(into, 1, "the first fill");
        let counts = r.dec.caches().counts();
        assert_eq!(
            (
                counts.made_fills,
                counts.decode_firsts,
                counts.fills_deferred
            ),
            (1, 1, 1)
        );
        // A later decode's picture lands over the fill: nothing more is copied in.
        seen.skipped.store(false, Ordering::Relaxed);
        r.p.world().complete_all(CompleteOn::Caller);
        let mark = r.p.call_log().mark();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(101, 2, true))
                .unwrap(),
            Ok(())
        );
        assert_eq!(cache_copies(&r, mark, cache).0, 0);
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

    /// A decode whose output's device cache is still being filled waits for the fill as long as
    /// it takes, as a Mac's video work waits for its GPU's earlier work on its surface: held past
    /// 2 s on the platform's clock, the wait is named in the log and goes on, and once the fill is
    /// done the picture lands in the cache and the command buffer is taken
    #[test]
    fn a_decode_waits_for_its_outputs_fill_however_long_it_takes() {
        let (mut r, seen) = cached_video();
        // The first use fills the cache, and the fill is held.
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, false))
                .unwrap(),
            Ok(())
        );
        let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
        let message = writes_cached_video(101, 2, true);
        let p = Arc::clone(&r.p);
        let decoders = r.dec.decoders();
        let mut walker = r.dec.sender(None, super::super::Sinks::default());
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message).unwrap());
            // The wait looks, and looks again: so the look after the first one seen began after
            // the platform's clock moved on, and went on.
            let looked_twice = |what: &str| {
                let looks = decoders.cache_wait_looks();
                assert!(
                    crate::graphics::tests::eventually(|| decoders.cache_wait_looks() > looks + 1),
                    "{what}: the wait goes on"
                );
            };
            looked_twice("the decode waits for the fill");
            let named = |lines: &[reims_vgpu_vulkan::core::testkit::logs::Line]| {
                (lines.iter()).any(|(level, _, line)| {
                    *level == log::Level::Warn && line.contains("has not finished after 2s")
                })
            };
            p.virtual_clock().advance(std::time::Duration::from_secs(2));
            looked_twice("2 s on");
            let lines = logs.take();
            assert!(named(&lines), "the wait is named: {lines:?}");
            p.virtual_clock()
                .advance(std::time::Duration::from_secs(60));
            looked_twice("a minute on");
            assert!(
                !walk.is_finished() && seen.decodes.lock().is_empty(),
                "the decode waits for the fill"
            );
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(*seen.cached.lock(), [[true, true]], "into the cache");
    }

    /// A reset of the device ends a decode's wait for its output's fill, as it ends every wait
    /// of the decoder's, so none holds the reset up: the op faults its command buffer, and the
    /// session is handed nothing
    #[test]
    fn a_reset_ends_a_decodes_wait_for_its_outputs_fill() {
        let (mut r, seen) = cached_video();
        assert_eq!(
            r.dec
                .decode(&r.device, &writes_cached_video(100, 1, false))
                .unwrap(),
            Ok(())
        );
        let message = writes_cached_video(101, 2, true);
        let p = Arc::clone(&r.p);
        let decoders = r.dec.decoders();
        let mut walker = r.dec.sender(None, super::super::Sinks::default());
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message));
            let looks = decoders.cache_wait_looks();
            assert!(
                crate::graphics::tests::eventually(|| decoders.cache_wait_looks() > looks + 1),
                "the decode waits for the fill"
            );
            assert!(!walk.is_finished(), "the decode waits for the fill");
            decoders.resetting();
            assert!(walk.join().unwrap().is_err(), "the op faults");
        });
        decoders.reset_done();
        assert!(seen.decodes.lock().is_empty());
        p.world().complete_all(CompleteOn::Caller);
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
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_USE, RgfxOpSurfaceUse};
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

    /// A pass the platform makes no encoder of, render, compute or blit (the Vulkan platform
    /// answers none where its runtime refuses the pass, a gap of its own it counts), is skipped
    /// whole, as one naming an object whose create the host refused is: its ops reach no encoder,
    /// the encoders after it run, and the buffer completes as it would have, `RGFX_CB_OK`, as a
    /// Metal app's sends to a nil encoder do nothing and fail no command buffer.
    #[test]
    fn a_pass_the_platform_makes_no_encoder_of_is_skipped_whole() {
        use reims_vgpu_graphics::ops::blit::{RGFX_OP_BLIT_FILL_BUFFER, RgfxBlitOpFillBuffer};
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RgfxComputeOpDispatchThreadgroups,
        };
        use reims_vgpu_vulkan::core::platform::types::NsErrorText;
        let (p, mut dec) = setup();
        let device = p.device();
        let captured = Arc::new(Mutex::new(Vec::<RgfxCbComplete>::new()));
        let sink = Arc::clone(&captured);
        dec.set_completion_sink(Arc::new(move |record| sink.lock().push(record)));
        let gap = NsErrorText {
            domain: "metal2vulkan-runtime".to_owned(),
            code: 1,
            description: "a pass the runtime renders none of".to_owned(),
        };
        p.script()
            .push("renderCommandEncoderWithDescriptor:", Step::Gap(gap));
        p.script()
            .push("computeCommandEncoderWithDescriptor:", Step::Nil);
        p.script().push("blitCommandEncoder", Step::Nil);
        let dispatch = RgfxComputeOpDispatchThreadgroups {
            threadgroups_width: 1,
            threadgroups_height: 1,
            threadgroups_depth: 1,
            threads_width: 1,
            threads_height: 1,
            threads_depth: 1,
        };
        let compute = [
            op(
                RGFX_OP_ENCODER_BEGIN_COMPUTE,
                RgfxComputePassDesc::default().as_bytes(),
            ),
            op(RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, dispatch.as_bytes()),
            op(RGFX_OP_ENCODER_END, &[]),
        ]
        .concat();
        let fill = RgfxBlitOpFillBuffer {
            buffer_obj_id: 2,
            range_location: 0,
            range_length: 256,
            value: 7,
            ..RgfxBlitOpFillBuffer::default()
        };
        let blit = [
            op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
            op(RGFX_OP_BLIT_FILL_BUFFER, fill.as_bytes()),
            op(RGFX_OP_ENCODER_END, &[]),
        ]
        .concat();
        let pass = [render_ops(), compute, blit].concat();
        let message = command_buffer_message(160, 75, &[pass.clone(), pass].concat());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let sels = p.call_log().sels_since(mark);
        let sent = |sel: &str| sels.iter().filter(|s| **s == sel).count();
        let asked = [
            "renderCommandEncoderWithDescriptor:",
            "computeCommandEncoderWithDescriptor:",
            "blitCommandEncoder",
        ]
        .map(sent);
        assert_eq!(asked, [2, 2, 2], "each encoder asked for at each pass");
        let ran = [
            "drawPrimitives:vertexStart:vertexCount:",
            "dispatchThreadgroups:threadsPerThreadgroup:",
            "fillBuffer:range:value:",
            "endEncoding",
        ]
        .map(sent);
        assert_eq!(ran, [1, 1, 1, 3], "only the encoders made run and end");
        assert_eq!(sent("commit"), 1);
        p.world().complete_all(CompleteOn::Caller);
        let recs = captured.lock();
        let answered: Vec<_> = (recs.iter()).map(|r| (r.cb_obj_id, r.status)).collect();
        assert_eq!(answered, [(160, RGFX_CB_OK)], "completed as it would have");
    }

    /// A command buffer's completion record carries the GPU times the platform answers as its
    /// completed handler runs, in nanoseconds of the host's clock, which the guest bundle reports
    /// as `GPUStartTime` and `GPUEndTime`; a failed buffer the platform gives none of carries 0.
    /// They are the platform's values bit for bit, as the platform's clock is the device's
    /// (`PLATFORM` in `reims_vgpu_graphics::clock::runtime`).
    #[test]
    fn a_completion_record_carries_the_platforms_gpu_times() {
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
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
        use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
        let (_p, dec, posted) = posting(true);
        for ran in [false, true] {
            posted.lock().clear();
            let submitted = dec.submitted(150, 80, true);
            let scheduled = dec.scheduled(150, 80, None).map(|(s, _)| s);
            let submission = submission_handler(submitted.clone(), None, scheduled.clone());
            let handler = dec.completed_handler(
                150,
                80,
                EventOps::default(),
                None,
                false,
                (submitted, None, scheduled),
                Ending::of(Arc::new(Completions::default()).turn()),
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

    /// A queue's asks leave from the front, once closed, at a begin: one that closes behind an
    /// open one (a buffer that ended uncommitted) waits there, and an ask of it posts nothing, as
    /// an ask of one gone does; once those before it close, the next begin lets them all go. An
    /// ask of a buffer committed and posted posts nothing again.
    #[test]
    fn closed_asks_leave_from_the_front() {
        let line = Line::default();
        let sequencer = Arc::new(super::super::sequencer::Sequencer::default());
        let posted = Arc::new(Mutex::new(Vec::new()));
        let sink: ScheduledSink = {
            let posted = Arc::clone(&posted);
            Arc::new(move |r: RgfxCbScheduled| posted.lock().push(r.cb_obj_id))
        };
        // A buffer at `seqno` begun on the queue, its `CB_SCHEDULED` an ask may post.
        let begin = |seqno: u64| {
            let scheduled = Arc::new(Scheduled {
                sink: Arc::clone(&sink),
                record: scheduled_record(100 + seqno, seqno),
                posted: Mutex::new(false),
                owed: None,
            });
            let askable = Arc::new(Askable {
                seqno,
                cb_obj_id: 100 + seqno,
                context: None,
                state: Mutex::new(Asking::Begun(Arc::clone(&scheduled))),
            });
            line.askable(Arc::clone(&askable));
            (askable, scheduled)
        };
        let (first, first_scheduled) = begin(1);
        let (second, _) = begin(2);
        second.end(false);
        let _third = begin(3);
        assert_eq!(line.asks_kept(), 3, "the second waits behind the first");
        line.asked(2, 102).expect("kept").ask(&sequencer);
        assert!(
            posted.lock().is_empty(),
            "an ask of the ended buffer posts nothing"
        );
        first.end(true);
        first_scheduled.post();
        assert_eq!(*posted.lock(), [101]);
        let _fourth = begin(4);
        assert_eq!(
            line.asks_kept(),
            2,
            "the first two left: the third and the fourth"
        );
        assert!(line.asked(1, 101).is_none() && line.asked(2, 102).is_none());
        first.ask(&sequencer);
        assert_eq!(*posted.lock(), [101], "posted once");
    }

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

    /// A buffer that asks and waits for nothing but whose walk then faults is submitted and
    /// scheduled as it was taken, and fails at its completion, as Metal reports a command
    /// buffer's errors: the stand-in's `CB_COMPLETE`, its failure, follows. One the host cannot take, of a queue it
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
                (RGFX_EVT_CB_SCHEDULED, 160, 90),
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
        assert_eq!(posted.lock().len(), 3);
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
        dec.decoders().answer_at_doorbell(0, message, move |bytes| {
            if posts {
                (posted.lock()).push(record_of(bytes));
            }
            posts
        })
    }

    /// The type, buffer and seqno of the `CB_SUBMITTED` or `CB_SCHEDULED` record `bytes`, as an
    /// answer on a doorbell's thread posts it: the two share a layout.
    fn record_of(bytes: &[u8]) -> (u32, u64, u64) {
        let m = RgfxCbScheduled::read_from_bytes(bytes).unwrap();
        (m.header.r#type, m.cb_obj_id, m.header.seqno)
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
    /// buffer not yet begun. Where it is, its `CB_SUBMITTED` and then its `CB_SCHEDULED` are posted
    /// there, no buffer before it owing its own, and what the answer took is kept on its queue;
    /// where it is not, nothing is.
    #[test]
    fn a_buffer_is_answered_at_the_doorbell_only_where_it_need_wait_for_nothing() {
        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        create_events(&p, &mut dec, 0);
        let answered = |dec: &GraphicsDecoder<FakePlatform>, message: &[u8]| {
            let before = posted.lock().len();
            let yes = answer(dec, message, &posted, true);
            assert_eq!(posted.lock().len(), before + 2 * usize::from(yes));
            assert_eq!(
                let_go_of_answer(dec),
                yes,
                "kept on its queue where answered"
            );
            yes
        };
        assert!(answered(&dec, &asking(170, 100)));
        assert!(posted.lock().ends_with(&[
            (RGFX_EVT_CB_SUBMITTED, 170, 100),
            (RGFX_EVT_CB_SCHEDULED, 170, 100)
        ]));
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
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 175, 105),
                (RGFX_EVT_CB_SCHEDULED, 175, 105)
            ]
        );
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
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 180, 110),
                (RGFX_EVT_CB_SCHEDULED, 180, 110)
            ]
        );
    }

    /// A buffer answered on a doorbell's thread calls nothing of the platform's there, and its
    /// begin takes what the answer took: it makes the calls a decode makes, and gets its
    /// `CB_SUBMITTED` and `CB_SCHEDULED` once each, the answer's, or its begin's where the answer
    /// found the ring busy.
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
                .begin_taken(&device, &message, Some(1), false)
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

    /// A buffer that asks for its `CB_SUBMITTED` and waits for nothing gets its `CB_SCHEDULED`
    /// right after it as the host takes it, before its walk and its commit, with no submission
    /// handler, though the platform holds its submission. One that does not ask gets its record
    /// at its scheduling, from its submission handler, and an asking buffer behind it then gets
    /// its `CB_SUBMITTED` at its take and its `CB_SCHEDULED` at its scheduling, after the one
    /// before it. So does one behind a buffer that waits for an event; once nothing before it owes
    /// its record, an asking buffer's goes at its take again.
    #[test]
    fn a_buffer_answered_at_its_take_is_scheduled_there() {
        const SUB: u32 = RGFX_EVT_CB_SUBMITTED;
        const SCHED: u32 = RGFX_EVT_CB_SCHEDULED;
        const ENCODER: &str = "renderCommandEncoderWithDescriptor:";
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        create_events(&p, &mut dec, 0);
        // The call-log mark at each `CB_SCHEDULED`'s post: every call logged before it was made
        // before the post.
        let at = Arc::new(Mutex::new(Vec::new()));
        let (log, marks, sink) = (Arc::clone(&p), Arc::clone(&at), Arc::clone(&posted));
        dec.set_scheduled_sink(Arc::new(move |m| {
            marks.lock().push((m.cb_obj_id, log.call_log().mark()));
            (sink.lock()).push((m.header.r#type, m.cb_obj_id, m.header.seqno));
        }));
        let before_scheduled = |from: u64, cb_obj_id: u64| -> Vec<&'static str> {
            let (_, posted) = *at.lock().iter().find(|(cb, _)| *cb == cb_obj_id).unwrap();
            (p.call_log().since(from).into_iter())
                .filter(|c| c.seq < posted)
                .map(|c| c.sel)
                .collect()
        };
        let mut asks = render_ops();
        asks.extend(op(RGFX_OP_SUBMITTED, &[]));
        p.world().hold_submissions();
        let mark = p.call_log().mark();
        let message = command_buffer_message(161, 91, &asks);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let before = before_scheduled(mark, 161);
        assert!(!before.contains(&ENCODER) && !before.contains(&"commit"));
        let sels = p.call_log().sels_since(mark);
        assert!(sels.contains(&"commit") && !sels.contains(&"addSubmissionHandler"));
        assert_eq!(
            *posted.lock(),
            [(SUB, 161, 91), (SCHED, 161, 91)],
            "at its take, the platform holding it"
        );
        posted.lock().clear();
        for (cb, seqno, ops) in [(160, 90, render_ops()), (165, 95, asks.clone())] {
            let mark = p.call_log().mark();
            let message = command_buffer_message(cb, seqno, &ops);
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
            let sels = p.call_log().sels_since(mark);
            assert!(
                sels.contains(&"addSubmissionHandler"),
                "{cb} has no submission handler"
            );
        }
        assert_eq!(
            *posted.lock(),
            [(SUB, 165, 95)],
            "the plain buffer's record waits for its scheduling, and so the next one's"
        );
        p.world().release_submissions();
        assert_eq!(
            posted.lock()[1..],
            [(SCHED, 160, 90), (SCHED, 165, 95)],
            "each at its scheduling, in their order"
        );
        posted.lock().clear();

        p.world().hold_submissions();
        let mut waits = event_op(RGFX_OP_WAIT_EVENT, 7, 1);
        waits.extend(render_ops());
        let mark = p.call_log().mark();
        let message = command_buffer_message(162, 92, &waits);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(
            p.call_log()
                .sels_since(mark)
                .contains(&"addSubmissionHandler")
        );
        let message = command_buffer_message(163, 93, &asks);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert!(posted.lock().is_empty(), "the wait holds both");
        p.world().release_submissions();
        assert_eq!(
            *posted.lock(),
            [(SCHED, 162, 92), (SUB, 163, 93), (SCHED, 163, 93)],
            "each at its scheduling, in their order"
        );
        let line = Arc::clone(&dec.queue_counts(1).unwrap().line);
        assert!(!line.owes());
        let message = command_buffer_message(164, 94, &asks);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(
            posted.lock()[3..],
            [(SUB, 164, 94), (SCHED, 164, 94)],
            "at its take"
        );
        p.world().complete_all(CompleteOn::Caller);
        assert!(!line.owes());
    }

    /// An answer on a doorbell's thread posts the buffer's `CB_SCHEDULED` after its
    /// `CB_SUBMITTED` where no buffer begun before it on its queue owes its own, and the buffer's
    /// take posts neither again and adds no submission handler. Where one is owed, the answer
    /// posts the `CB_SUBMITTED` alone, and so does the take while it is still owed; the
    /// buffer's `CB_SCHEDULED` then goes at its scheduling.
    #[test]
    fn an_answer_at_the_doorbell_schedules_where_nothing_before_is_owed() {
        const SUB: u32 = RGFX_EVT_CB_SUBMITTED;
        const SCHED: u32 = RGFX_EVT_CB_SCHEDULED;
        let (p, mut dec, posted) = posting(true);
        let device = p.device();
        let message = asking(185, 115);
        assert!(answer(&dec, &message, &posted, true));
        assert_eq!(*posted.lock(), [(SUB, 185, 115), (SCHED, 185, 115)]);
        let mark = p.call_log().mark();
        let begun = dec
            .begin_taken(&device, &message, Some(1), false)
            .unwrap()
            .unwrap();
        assert_eq!(
            dec.finish_command_buffer(&device, &message, begun).unwrap(),
            Ok(())
        );
        dec.end_message();
        assert_eq!(posted.lock().len(), 2, "each posted once");
        assert!(
            !p.call_log()
                .sels_since(mark)
                .contains(&"addSubmissionHandler")
        );

        // A record owed by a buffer before it.
        let line = Arc::clone(&dec.queue_counts(1).unwrap().line);
        line.owed.fetch_add(1, Ordering::AcqRel);
        let message = asking(186, 116);
        assert!(answer(&dec, &message, &posted, true));
        assert_eq!(posted.lock()[2..], [(SUB, 186, 116)]);
        let begun = dec
            .begin_taken(&device, &message, Some(1), false)
            .unwrap()
            .unwrap();
        assert_eq!(posted.lock().len(), 3, "still owed at its take");
        line.owed.fetch_sub(1, Ordering::AcqRel);
        assert_eq!(
            dec.finish_command_buffer(&device, &message, begun).unwrap(),
            Ok(())
        );
        dec.end_message();
        // The fake schedules a buffer as it is committed.
        assert_eq!(posted.lock()[2..], [(SUB, 186, 116), (SCHED, 186, 116)]);
        assert!(!line.owes());
        p.world().complete_all(CompleteOn::Caller);
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
            .begin_taken(&device, &asking(191, 121), Some(1), false)
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
        let sequencer = Arc::clone(&dec.shared.sequencer);
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
            let held = crate::graphics::tests::eventually(|| sequencer.holds_up(0));
            assert!(held, "never waited for the answer's ticket");
            assert!(commits().is_empty(), "committed before the answered buffer");
            let a = std::thread::Builder::new()
                .name("first".into())
                .spawn_scoped(scope, || {
                    let device = p.device();
                    let begun = first.begin_taken(&device, &asked, Some(1), false).unwrap();
                    first.finish_command_buffer(&device, &asked, begun.unwrap())
                })
                .unwrap();
            assert_eq!(a.join().unwrap().unwrap(), Ok(()));
            assert_eq!(b.join().unwrap(), Ok(()));
        });
        assert_eq!(commits(), ["first", "second"]);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 100, 1),
                (RGFX_EVT_CB_SCHEDULED, 100, 1)
            ]
        );
        drop(dec);
    }

    /// A buffer answered on a doorbell's thread whose command buffer the platform then refuses at
    /// its begin fails in its turn: its stand-in commits after the buffer begun before it on its
    /// queue, and its failure comes after that buffer's completion.
    #[test]
    fn a_refused_make_after_an_answer_fails_in_turn() {
        use reims_vgpu_vulkan::core::testkit::script::Step;

        let (p, mut dec, posted) = posting(false);
        let device = p.device();
        let failed = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&failed);
        dec.set_completion_sink(Arc::new(move |m| {
            seen.lock().push((m.cb_obj_id, m.status));
        }));
        let first = command_buffer_message(193, 123, &render_ops());
        let begun = dec
            .begin_taken(&device, &first, Some(1), false)
            .unwrap()
            .unwrap();
        let mut later = dec.decoders().sender(None, dec.sinks.clone());
        assert!(answer(&later, &asking(194, 124), &posted, true));
        p.script().push("commandBuffer", Step::Nil);
        let refused = later
            .begin_taken(&device, &asking(194, 124), Some(1), false)
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
        // The decoder holds no scheduled sink, so the buffer before it owes no record.
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_CB_SUBMITTED, 194, 124),
                (RGFX_EVT_CB_SCHEDULED, 194, 124)
            ]
        );
    }

    /// A `COMMAND_BUFFER` message for queue `queue_obj_id`, command buffer `cb_obj_id`, carrying
    /// `ops`.
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
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_later_commit_on_another_queue<P: crate::platform::ApiPlatform>() -> (bool, Vec<u8>) {
        use std::time::{Duration, Instant};

        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpFillBuffer,
        };
        use reims_vgpu_graphics::wire::RgfxSharedEventSetValue;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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

    /// A command buffer behind one of its queue that waits for an event is not answered as
    /// submitted until that one is scheduled, as Metal's `commitAndWaitUntilSubmitted` returns
    /// only then (the M3, the 5300M and the 630, docs/graphics-kext/dgpu-oracle.md, Q7): Metal
    /// orders a tracked resource across queues behind work it has submitted, not behind work a
    /// queue holds. A buffer committed on another queue once the answer comes, which reads a
    /// tracked buffer the first one writes, sees its writes (Q6).
    #[test]
    fn on_vulkan_a_later_commit_on_another_queue_sees_a_submitted_buffer_s_writes() {
        let (early, bytes) = a_later_commit_on_another_queue::<reims_vgpu_vulkan::RtPlatform>();
        assert!(!early, "answered as submitted while its queue was held");
        assert!(bytes.iter().all(|&b| b == 0x5a));
    }

    /// On the rt platform, through the decoder: process 1's command buffer binds a texture and a
    /// sampler state at the platform's own indices to a compute encoder, behind a wait for a
    /// shared event its name keeps past the process's close. The close's worker lets go of the
    /// two while the buffer waits, giving their indices back. With each class's range three
    /// indices from there, and a spare of each given back after the close, its last owner read by
    /// no buffer, the takes get the last fresh index, then none: the closed process's and the
    /// spare's wait, each given back after the buffer was committed. Once the unname sets the event
    /// past every value and the close's drain is answered, the buffer done, a take gets each of
    /// them, in the order they were given back.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn on_vulkan_rt_s_own_indices_wait_for_a_closed_process_s_running_buffer() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_SAMPLER_STATES, RGFX_OP_COMPUTE_SET_TEXTURES,
            RgfxComputeOpSetSamplerStates, RgfxComputeOpSetTextures, RgfxComputeSamplerBinding,
            RgfxComputeTextureBinding,
        };
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_SHARED_EVENT_NAME, RGFX_CMD_SHARED_EVENT_UNNAME, RGFX_CONTEXT_OBJ_ID_SHIFT,
            RGFX_STATUS_OK, RgfxSharedEventName, RgfxSharedEventUnname,
        };
        use reims_vgpu_vulkan::RtPlatform;
        use reims_vgpu_vulkan::core::platform::resource::{SamplerState as _, Texture as _};
        use reims_vgpu_vulkan::core::platform::values::{SamplerDesc, TextureDesc};
        use reims_vgpu_vulkan::rt_platform::texture::OwnIndices;

        use crate::platform::ApiPlatform as _;

        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        // A platform of the test's own, whose device goes before the process exits.
        let platform = crate::rt_test_end::platform();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<RtPlatform>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
        );
        let obj = |n: u64| (1 << RGFX_CONTEXT_OBJ_ID_SHIFT) | n;
        let header = |r#type: u32, size: usize| RgfxMsgHeader {
            r#type,
            size: size as u32,
            seqno: 0,
        };
        let on = OnFault::Return;
        let small = TextureDesc {
            width: 4,
            height: 4,
            ..TextureDesc::default()
        };
        let desc = platform.texture_descriptor(&small);
        // A texture and a sampler state of the platform's own indices, made on the device, each
        // `None` where no index is left for it.
        let texture = || device.new_texture(on, &desc).expect("never a raise");
        let sampler = || {
            let made = device.new_sampler_state(&SamplerDesc::default());
            made.expect("never a raise")
        };
        let made = || {
            let made = (texture(), sampler());
            (made.0.expect("a texture"), made.1.expect("a sampler state"))
        };
        // Each class's takes of indices given back and give-backs, since the test's start.
        let counts = |own: &OwnIndices| [own.taken_again(), own.given_back()];
        let classes = || [counts(device.indices()), counts(device.sampler_indices())];
        let base = classes();
        let since = || {
            let now = classes();
            [0usize, 1].map(|c| [0usize, 1].map(|k| now[c][k] - base[c][k]))
        };

        // Process 1: its queue, its shared event, named, and the two the buffer binds.
        dec.set_context(Some(1));
        let queue = RgfxQueueCreate {
            header: header(RGFX_CMD_QUEUE_CREATE, size_of::<RgfxQueueCreate>()),
            obj_id: obj(1),
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        let event = RgfxSharedEventCreate {
            header: header(
                RGFX_CMD_SHARED_EVENT_CREATE,
                size_of::<RgfxSharedEventCreate>(),
            ),
            obj_id: obj(3),
            initial_value: 0,
        };
        assert_eq!(dec.decode(&device, event.as_bytes()).unwrap(), Ok(()));
        let named = RgfxSharedEventName {
            header: header(RGFX_CMD_SHARED_EVENT_NAME, size_of::<RgfxSharedEventName>()),
            obj_id: obj(3),
            name: 5,
            listen_queue_obj_id: obj(1),
        };
        let listen: super::super::shares::ListenSink = Arc::new(|_: &RgfxEventSignalled| true);
        assert_eq!(
            dec.name_shared_event(named.as_bytes(), Some(listen)),
            (Some(0), RGFX_STATUS_OK)
        );
        device.indices().leave_fresh(3);
        device.sampler_indices().leave_fresh(3);
        let (spare, read) = (made(), made());
        let spare_at = (spare.0.resource_index(), spare.1.resource_index());
        let read_at = (read.0.resource_index(), read.1.resource_index());
        let (read_texture, read_sampler) = read;
        let texture_in = dec.insert(obj(10), ObjHandle::Texture(read_texture));
        assert_eq!(texture_in.unwrap(), Ok(()));
        let sampler_in = dec.insert(obj(11), ObjHandle::Sampler(read_sampler));
        assert_eq!(sampler_in.unwrap(), Ok(()));
        let mut ops = event_op(RGFX_OP_WAIT_EVENT, obj(3), 1);
        ops.extend(op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        ));
        let mut textures = RgfxComputeOpSetTextures {
            location: 0,
            length: 1,
        }
        .as_bytes()
        .to_vec();
        let binding = RgfxComputeTextureBinding {
            texture_obj_id: obj(10),
        };
        textures.extend_from_slice(binding.as_bytes());
        ops.extend(op(RGFX_OP_COMPUTE_SET_TEXTURES, &textures));
        let mut samplers = RgfxComputeOpSetSamplerStates {
            location: 0,
            length: 1,
        }
        .as_bytes()
        .to_vec();
        let binding = RgfxComputeSamplerBinding {
            sampler_obj_id: obj(11),
        };
        samplers.extend_from_slice(binding.as_bytes());
        ops.extend(op(RGFX_OP_COMPUTE_SET_SAMPLER_STATES, &samplers));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let buffer = command_buffer_on(obj(1), obj(20), &ops);
        assert_eq!(dec.decode(&device, &buffer).unwrap(), Ok(()));
        dec.set_context(None);
        assert_eq!(since(), [[0, 0]; 2]);

        // Its close, as a `CONTEXT_CLOSE` makes it, the buffer waiting: the close worker lets go
        // of the two, giving their indices back, and the drain answers once the buffer is done.
        let (_, closed) = dec.close_tables(&device, 1);
        let drained = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&drained);
        let answer = Box::new(move || flag.store(true, Ordering::Release));
        let drain = dec.drain_context(1, true, answer);
        dec.let_go_later(closed, drain);
        dec.closes_let_go();
        assert_eq!(since(), [[0, 1]; 2]);
        // The spare of each is given back after them, the buffer committed before it too: the
        // takes get the last fresh index, then none.
        drop(spare);
        let fresh = made();
        let (t, s) = (texture(), sampler());
        assert!(
            t.is_none() && s.is_none(),
            "an index taken while a buffer committed before its give-back runs"
        );
        assert!(
            !drained.load(Ordering::Acquire),
            "drained before the buffer ran"
        );
        assert_eq!(since(), [[0, 2]; 2]);

        // The unname sets the event past every value: the buffer runs and completes, and its
        // indices are taken again.
        let unname = RgfxSharedEventUnname {
            header: header(
                RGFX_CMD_SHARED_EVENT_UNNAME,
                size_of::<RgfxSharedEventUnname>(),
            ),
            reserved_10: 0,
            name: 5,
        };
        dec.unname_shared_event(unname.as_bytes());
        let start = Instant::now();
        while !drained.load(Ordering::Acquire) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "the closed process's buffer never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let last = made();
        assert_eq!((last.0.resource_index(), last.1.resource_index()), read_at);
        let again = made();
        assert_eq!(
            (again.0.resource_index(), again.1.resource_index()),
            spare_at
        );
        assert_eq!(since(), [[2, 2]; 2]);
        drop((fresh, again, last));
    }

    /// Through the decoder on `P`: a buffer fills a buffer, signals shared event 7 to 1, waits
    /// for 7 to reach 2, and fills it again, as a guest whose CPU waits for the signal before it
    /// sets 2 encodes it. Whether the value 1 reached the guest before the buffer completed,
    /// which it cannot before 2 is set; then the CPU sets 2, whatever came: whether the buffer
    /// completed within 10 s of that, and whether the value went ahead of its completion, once.
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_cpu_wait_for_a_mid_buffer_signal<P: crate::platform::ApiPlatform>() -> (bool, bool, bool) {
        use std::time::{Duration, Instant};

        use reims_vgpu_graphics::ops::blit::{RGFX_OP_BLIT_FILL_BUFFER, RgfxBlitOpFillBuffer};
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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
        // The buffer completes only once the CPU sets 2 below, so the signal is waited for
        // however long the host takes: one that comes only at the buffer's end never comes here.
        crate::graphics::tests::eventually(|| posted.lock().contains(&signalled));
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

    /// A guest thread that waits on the CPU for a signal from the middle of a command buffer,
    /// before it sets the value the buffer waits for next, gets the signal while the buffer runs,
    /// as Metal's CPU does (docs/graphics-kext/dgpu-oracle.md, Q18), and the buffer then
    /// completes: the value goes as the host's event reaches it, once, ahead of the buffer's
    /// completion.
    #[test]
    fn on_vulkan_a_cpu_wait_for_a_mid_buffer_signal_is_released_before_the_buffer_ends() {
        let (early, finished, once) =
            a_cpu_wait_for_a_mid_buffer_signal::<reims_vgpu_vulkan::RtPlatform>();
        assert!(early, "the signal came only at the buffer's end, or never");
        assert!(finished && once);
    }

    /// Through the decoder on `P`, as the host takes two queues' rings on threads of their own:
    /// buffer A of queue 1 fills a tracked buffer, and buffer B of queue 2, committed plainly
    /// after A by the guest, names A in its `COMMIT_ORDER` and copies the buffer out. B is taken
    /// first, on a thread of its own; A is begun and committed once B waits for A's commit, and
    /// its message ends, raising its queue's commit mark, once the platform has scheduled it. The
    /// bytes B copied.
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn a_plain_commit_after_a_named_buffer_of_another_queue<P: crate::platform::ApiPlatform>()
    -> Vec<u8> {
        use std::time::{Duration, Instant};

        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpFillBuffer,
        };
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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
        let marks = Arc::clone(&dec.queue_counts(1).expect("queue 1").order);
        std::thread::scope(|scope| {
            let device = &device;
            let taken_first = scope.spawn(|| {
                let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
                let begun = second
                    .begin_taken(device, &b, Some(2), false)
                    .unwrap()
                    .unwrap();
                let done = second.finish_command_buffer(device, &b, begun).unwrap();
                second.end_message();
                done
            });
            let waits = crate::graphics::tests::eventually(|| marks.commit.has_waiters());
            assert!(waits, "B never waited for A's commit");
            let begun = first
                .begin_taken(device, &a, Some(1), false)
                .unwrap()
                .unwrap();
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

    /// A plain commit on another queue that the guest made after a buffer's, and that names it in
    /// its `COMMIT_ORDER`, reads the tracked buffer it writes, though the host took it first: the
    /// host commits it only once the buffer it names is committed, here once the platform has
    /// scheduled that one, as Metal orders a tracked resource across queues behind work it has
    /// submitted (docs/graphics-kext/dgpu-oracle.md, Q6 and Q7).
    #[test]
    fn on_vulkan_a_plain_commit_on_another_queue_sees_the_writes_of_the_buffer_it_names() {
        let bytes =
            a_plain_commit_after_a_named_buffer_of_another_queue::<reims_vgpu_vulkan::RtPlatform>();
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
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn an_order_behind_a_held_buffer<P: crate::platform::ApiPlatform>()
    -> (Vec<u32>, bool, Vec<u64>, Vec<u8>) {
        use std::time::{Duration, Instant};

        use reims_vgpu_graphics::ops::blit::{
            RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, RGFX_OP_BLIT_FILL_BUFFER,
            RgfxBlitOpCopyBufferToBuffer, RgfxBlitOpFillBuffer,
        };
        use reims_vgpu_graphics::wire::RgfxSharedEventSetValue;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        const LEN: u64 = 1 << 20;
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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

    /// A command buffer that waits on the GPU for a buffer of another queue the platform holds is
    /// answered as submitted and scheduled as its commit returns, as a Mac answers a buffer it
    /// orders after another queue's access to an IOSurface (docs/graphics-kext/dgpu-oracle.md,
    /// Q16), though the Vulkan platform holds a wait for a shared event's value on the host until
    /// it is reached. B still runs after A, copying what A wrote; the order in which the runtime
    /// reports the two queues' completions is its own: on MoltenVK on the M3 Max it reported B's
    /// before A's in 20 runs of 20, with B's records held back to its scheduling as with them
    /// posted at its commit.
    #[test]
    fn on_vulkan_an_order_behind_a_held_buffer_holds_back_no_record() {
        let (posted, waited, mut completed, bytes) =
            an_order_behind_a_held_buffer::<reims_vgpu_vulkan::RtPlatform>();
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
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn sub_encoders_after_a_count<P: crate::platform::ApiPlatform>() -> [u64; 4] {
        use std::time::{Duration, Instant};

        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::values::{RenderPipelineDesc, TextureDesc};

        use super::super::draw_cost::fixture;
        const SIDE: u64 = 64;
        const BGRA8: u64 = 80;
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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

    /// rt builds `setVisibilityResultMode:offset:` of `Disabled` alone: the sub-encoders' counting
    /// mode at 16 is rt's gap, counted once under rt's code (`rt-refused-gap`), their two
    /// `Disabled` modes are built, and their draws write nothing into the results, every word
    /// keeping its bytes. (Every Mac writes the square's 4096 samples at 16, and nothing at 0 and
    /// 24 under `Disabled`, which [`render::visibility_offset`] keeps the M3 Max from writing 0
    /// at.)
    #[test]
    fn on_vulkan_rt_s_sub_encoders_write_no_count_under_the_counting_mode_it_refuses() {
        let fill = u64::from_ne_bytes([0xee; 8]);
        let codes = ["rt-refused-gap", "rt-unbuilt-render-command"];
        let before = codes.map(crate::platform::counted);
        assert_eq!(
            sub_encoders_after_a_count::<reims_vgpu_vulkan::RtPlatform>(),
            [fill, fill, fill, fill]
        );
        let after = codes.map(crate::platform::counted);
        assert_eq!([after[0] - before[0], after[1] - before[1]], [1, 0]);
    }

    /// A `BUFFER_CREATE` of `length` bytes, id `obj_id`, at `gpu_address` (0: where the host
    /// places it).
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
    /// `on_vulkan_the_host_s_own_ranges_lie_outside_every_unit`'s pin at the range's base, 96 KiB,
    /// covers its address.
    static CONTEXT_RANGE_PINS: Mutex<()> = Mutex::new(());

    /// A decoder on the Vulkan platform with a queue (1), and, as a process in slot 0 picks
    /// (`RgfxIdSpace`, next fit from the slot's base, a pick of Metal's span of a pin), the create
    /// of a device-placed 1 MiB texture buffer (10) at the slot's first pick, which the host takes
    /// no pin of. rt builds no texture buffers: the create is refused, counted under rt's code
    /// (`rt-refused-gap`), so no range of its lies among the pins the tests make after it.
    fn decoder_with_a_texture_buffer() -> (
        GraphicsDecoder<reims_vgpu_vulkan::RtPlatform>,
        <reims_vgpu_vulkan::RtPlatform as Platform>::Device,
    ) {
        use crate::platform::ApiPlatform as _;
        use reims_vgpu_graphics::wire::RGFX_CONTEXT_VA_BASE;
        let platform = reims_vgpu_vulkan::RtPlatform::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
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
        let refused = crate::platform::counted("rt-refused-gap");
        assert_eq!(
            dec.decode(&device, texture.as_bytes()).unwrap(),
            Err(DecodeError::Refused),
            "rt builds no texture buffers"
        );
        assert_eq!(crate::platform::counted("rt-refused-gap") - refused, 1);
        assert!(dec.object(10).is_none());
        (dec, device)
    }

    /// A guest's pin meets no range the Vulkan platform's runtime placed for itself: a buffer
    /// pinned at the pick after a texture buffer's is made, the texture buffer refused (rt builds
    /// none).
    #[test]
    fn on_vulkan_a_pin_meets_no_range_of_a_texture_buffer() {
        use reims_vgpu_graphics::wire::RGFX_CONTEXT_VA_BASE;
        let _pins = CONTEXT_RANGE_PINS.lock();
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let buffer = buffer_at(11, RGFX_CONTEXT_VA_BASE + 0x8000, 0x1_0000);
        assert_eq!(dec.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
    }

    /// The pins of `picks` (address, length), each after a pin at the guest's first pick and a
    /// queue's range made and a texture buffer refused (`decoder_with_a_texture_buffer`): each made,
    /// in no block the rt platform's device knows, at rt's address, counted under rt's code
    /// (`rt-gpu-address-unpinned`), the guest's among them.
    fn pins_in_no_known_block_lie_at_rt_s_own(picks: &[(u64, u64)]) {
        use reims_vgpu_graphics::wire::RGFX_CONTEXT_VA_BASE;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        let _pins = CONTEXT_RANGE_PINS.lock();
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let unpinned = crate::platform::counted("rt-gpu-address-unpinned");
        let guest = buffer_at(11, RGFX_CONTEXT_VA_BASE + 0x8000, 0x1_0000);
        assert_eq!(dec.decode(&device, guest.as_bytes()).unwrap(), Ok(()));
        for (obj_id, &(at, length)) in (20..).zip(picks) {
            let pinned = buffer_at(obj_id, at, length);
            assert_eq!(
                dec.decode(&device, pinned.as_bytes()).unwrap(),
                Ok(()),
                "the pin at {at:#x}"
            );
            let Some(ObjHandle::Buffer(buffer)) = dec.object(obj_id) else {
                panic!("no buffer {obj_id}");
            };
            assert_ne!(
                buffer.gpu_address(),
                at,
                "rt's address for the pin at {at:#x}"
            );
        }
        assert_eq!(
            crate::platform::counted("rt-gpu-address-unpinned") - unpinned,
            1 + picks.len() as u64
        );
    }

    /// The GPU addresses the host gives the Shared pool's blocks where their source has none
    /// (`pool::addresses`, `[2^45, 2^46)`, the testkit's) lie in no block the rt platform's device
    /// knows, whose pool's blocks lie at rt's addresses: pins of a pool block's span there, at the
    /// first picks and at the last, are made at rt's address, counted
    /// ([`pins_in_no_known_block_lie_at_rt_s_own`]).
    #[test]
    fn on_vulkan_pins_at_the_host_s_pool_addresses_lie_at_rt_s_own() {
        use crate::pool::{HOST_BASE, HOST_END};
        // Blocks of 64 KiB and 16 MiB, each a span of its pages and a guard page.
        pins_in_no_known_block_lie_at_rt_s_own(&[
            (HOST_BASE, 0x1_0000),
            (HOST_BASE + 0x1_8000, 16 << 20),
            (HOST_END - 0x1_8000, 0x1_0000),
        ]);
    }

    /// The GPU addresses the host gives the contexts' Private blocks where their source has none
    /// (`pool::addresses`, `[2^44, 2^45)`) lie in no block the rt platform's device knows, as the
    /// pool's do: pins of a block's span there, at the first picks and at the last, and one at
    /// the pool's first pick, are made at rt's address, counted
    /// ([`pins_in_no_known_block_lie_at_rt_s_own`]).
    #[test]
    fn on_vulkan_pins_at_the_host_s_private_addresses_lie_at_rt_s_own() {
        use crate::pool::{HOST_BASE, PRIVATE_BASE, PRIVATE_END};
        // Blocks of 64 KiB and 16 MiB, each a span of its pages and a guard page; the last
        // Private block's span ends where the pool's first starts.
        pins_in_no_known_block_lie_at_rt_s_own(&[
            (PRIVATE_BASE, 0x1_0000),
            (PRIVATE_BASE + 0x1_8000, 16 << 20),
            (PRIVATE_END - 0x1_8000, 0x1_0000),
            (HOST_BASE, 0x1_0000),
        ]);
    }

    /// The GPU addresses the host gives the pages blocks' views where the guest took
    /// `IMPORT_ADDRESSES` (`pool::addresses`, `[2^43, 2^44)`) lie in no block the rt platform's
    /// device knows, as the Private blocks' do: pins of a view's span there, at the first picks
    /// and at the last, and one at the Private blocks' first pick, are made at rt's address,
    /// counted ([`pins_in_no_known_block_lie_at_rt_s_own`]).
    #[test]
    fn on_vulkan_pins_at_the_host_s_import_addresses_lie_at_rt_s_own() {
        use crate::pool::{IMPORT_BASE, IMPORT_END, PRIVATE_BASE};
        // Views of 64 KiB and 16 MiB, each a span of its pages and a guard page; the last view's
        // span ends where the Private blocks' first starts.
        pins_in_no_known_block_lie_at_rt_s_own(&[
            (IMPORT_BASE, 0x1_0000),
            (IMPORT_BASE + 0x1_8000, 16 << 20),
            (IMPORT_END - 0x1_8000, 0x1_0000),
            (PRIVATE_BASE, 0x1_0000),
        ]);
    }

    /// The host's own ranges lie outside the guest's GPU addresses: a buffer the host places
    /// itself meets none of `[RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_END)`, and, with it, a queue's
    /// range made and a texture buffer refused, the first pick of every unit the kext hands the
    /// contexts out in and the last unit's last pick are made. On rt the buffer lies at the
    /// driver's device address, below the range on some drivers and above its end on others
    /// (`reims_vgpu_vulkan::rt_platform::buffer`, "A pinned address").
    #[test]
    fn on_vulkan_the_host_s_own_ranges_lie_outside_every_unit() {
        use reims_vgpu_graphics::wire::{
            RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_END, RGFX_CONTEXT_VA_UNIT, RGFX_CONTEXT_VA_UNITS,
        };
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        let _pins = CONTEXT_RANGE_PINS.lock();
        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let (mut dec, device) = decoder_with_a_texture_buffer();
        let placed = buffer_at(11, 0, 64 << 20);
        assert_eq!(dec.decode(&device, placed.as_bytes()).unwrap(), Ok(()));
        let Some(ObjHandle::Buffer(buffer)) = dec.object(11) else {
            panic!("no buffer 11");
        };
        let (at, end) = (buffer.gpu_address(), buffer.gpu_address() + buffer.length());
        assert!(
            end <= RGFX_CONTEXT_VA_BASE || at >= RGFX_CONTEXT_VA_END,
            "the host's buffer at {at:#x}..{end:#x} meets the contexts' range \
             {RGFX_CONTEXT_VA_BASE:#x}..{RGFX_CONTEXT_VA_END:#x}"
        );
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
        use reims_vgpu_vulkan::core::testkit::impls::descriptor::PassModel;
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
        use reims_vgpu_vulkan::core::testkit::impls::descriptor::PassModel;
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

    /// A pass's rasterization rate map, after its colour records, its sample positions and its
    /// visibility result buffer, is set on the platform's pass; a pass whose flag says so but
    /// carries no whole record faults, and so does one naming no map the host holds.
    #[test]
    fn a_render_pass_takes_its_rasterization_rate_map() {
        use reims_vgpu_vulkan::core::platform::values::RateMapDesc;
        use reims_vgpu_vulkan::core::testkit::impls::descriptor::PassModel;
        let (p, dec) = setup();
        let map = p
            .device()
            .new_rate_map(&p.rate_map_descriptor(&RateMapDesc::default()))
            .unwrap()
            .unwrap();
        let id = map.id();
        dec.insert(0x40, ObjHandle::RateMap(map)).unwrap().unwrap();
        let with = |map_obj_id, visibility: bool| {
            let mut body = if visibility {
                begin_render_with_visibility(2, 0, None)
            } else {
                begin_render_body()
            };
            let mut head = RgfxRenderPassDesc::read_from_prefix(&body).unwrap().0;
            head.flags |= RGFX_PASS_FLAG_RATE_MAP;
            body[..size_of::<RgfxRenderPassDesc>()].copy_from_slice(head.as_bytes());
            body.extend_from_slice(RgfxRenderPassRateMap { map_obj_id }.as_bytes());
            body
        };
        for visibility in [false, true] {
            let (pass, _) = dec
                .build_render_pass(&with(0x40, visibility), |_| true)
                .unwrap();
            assert_eq!(
                pass.with_model(|m: &PassModel| m.rasterization_rate_map_id()),
                Some(id)
            );
        }
        let (plain, _) = dec
            .build_render_pass(&begin_render_body(), |_| true)
            .unwrap();
        assert_eq!(
            plain.with_model(|m: &PassModel| m.rasterization_rate_map_id()),
            None
        );
        let whole = with(0x40, false);
        assert!(
            dec.build_render_pass(&whole[..whole.len() - 4], |_| true)
                .is_err()
        );
        assert!(dec.build_render_pass(&with(0x41, false), |_| true).is_err());
        assert!(dec.build_render_pass(&with(2, false), |_| true).is_err());
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

    /// A buffer whose stream starts with the guest's times (`GUEST_TIMES`) is carried out as the
    /// same buffer without them: the walk steps past the op, which is the device's diagnostic,
    /// whatever its stamps hold, and a stream of the times alone is an empty buffer.
    #[test]
    fn a_buffer_s_guest_times_change_nothing_it_does() {
        use reims_vgpu_graphics::ops::RgfxOpGuestTimes;
        let times = RgfxOpGuestTimes {
            began_ns: u64::MAX,
            committed_ns: 3,
            posted_ns: 0,
        };
        let stamped = |ops: &[u8]| {
            let mut v = op(RGFX_OP_GUEST_TIMES, times.as_bytes());
            v.extend_from_slice(ops);
            v
        };
        let mut blit = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        blit.extend(op(RGFX_OP_ENCODER_END, &[]));
        let mut render = pass_of(&draw_ops(2, 0, 0));
        render.extend(op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]));
        render.extend(op(RGFX_OP_ENCODER_END, &[]));
        for (i, ops) in [Vec::new(), blit, render].into_iter().enumerate() {
            let i = i as u64;
            let (p, mut dec) = setup();
            let device = p.device();
            let mark = p.call_log().mark();
            let plain = command_buffer_message(300 + i, 1, &ops);
            assert_eq!(dec.decode(&device, &plain).unwrap(), Ok(()), "case {i}");
            p.world().complete_all(CompleteOn::Caller);
            let without = p.call_log().sels_since(mark);

            let (p, mut dec) = setup();
            let device = p.device();
            let mark = p.call_log().mark();
            let message = command_buffer_message(300 + i, 1, &stamped(&ops));
            assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()), "case {i}");
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(p.call_log().sels_since(mark), without, "case {i}");
        }
    }

    /// With the trace on, a command buffer's begin's end, its walk's start (with its op bytes and
    /// the counts of its ops, encoders, dispatches and fence updates and waits) and its end after
    /// its commit are each an event of its queue and seqno, in that order, on the thread that
    /// carries it out; a buffer whose walk faults has no end.
    #[test]
    fn a_traced_buffer_marks_its_begin_walk_and_end() {
        use reims_vgpu_graphics::ops::compute::RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS;
        use reims_vgpu_vulkan::core::trace;
        // nextest runs each test in a process of its own, whose trace file is this one
        let path = std::env::temp_dir().join(format!("rvg-cbtrace-{}.txt", std::process::id()));
        assert!(trace::to_file(&path));
        trace::set_clock(|| 5);
        let (p, mut dec) = setup();
        let device = p.device();
        let mut blit = op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]);
        blit.extend(op(RGFX_OP_ENCODER_END, &[]));
        let message = command_buffer_message(310, 7, &blit);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        // The walk faults at the unknown op: the ops after it are counted, never walked.
        let mut faults = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        faults.extend(op(0x1fff, &[0u8; 8]));
        for tag in [
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
            RGFX_OP_COMPUTE_UPDATE_FENCE,
            RGFX_OP_BLIT_WAIT_FOR_FENCE,
            RGFX_OP_RENDER_WAIT_FOR_FENCE,
            RGFX_OP_ENCODER_END,
        ] {
            faults.extend(op(tag, &[0u8; 8]));
        }
        let message = command_buffer_message(311, 8, &faults);
        assert!(dec.decode(&device, &message).is_err());
        trace::flush_before_death();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let events: Vec<(&str, &str)> = text
            .lines()
            .filter_map(|l| {
                let mut fields = l.splitn(3, ' ');
                let (_, thread, rest) = (fields.next()?, fields.next()?, fields.next()?);
                let kind = rest.split(' ').next()?;
                ["cbbegun", "cenc", "cbwalked"]
                    .contains(&kind)
                    .then_some((thread, rest))
            })
            .collect();
        let thread = events[0].0;
        assert!(events.iter().all(|(t, _)| *t == thread), "{events:?}");
        let rests: Vec<&str> = events.iter().map(|(_, rest)| *rest).collect();
        let faulted = format!("cenc 1 8 {} 7 1 1 1 2", faults.len());
        assert_eq!(
            rests,
            [
                "cbbegun 1 7",
                "cenc 1 7 16 2 1 0 0 0",
                "cbwalked 1 7",
                "cbbegun 1 8",
                faulted.as_str()
            ]
        );
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
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use reims_vgpu_graphics::wire::RGFX_CB_INTERNAL;
        use reims_vgpu_vulkan::core::platform::event::SharedEvent as _;
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
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use reims_vgpu_graphics::wire::RGFX_CB_INTERNAL;
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
        use reims_vgpu_graphics::wire::RGFX_CB_INTERNAL;
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
        use reims_vgpu_graphics::wire::{RGFX_CB_INTERNAL, RGFX_CB_OK};
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
    type GateState = Arc<(
        Mutex<(bool, bool, bool)>,
        reims_vgpu_vulkan::core::sync::Condvar,
    )>;

    struct Gate(GateState);

    impl Gate {
        /// The gate, set on the walks of `dec`'s state.
        fn on(dec: &GraphicsDecoder<FakePlatform>, at: usize) -> Gate {
            let state = Arc::new((
                Mutex::new((false, false, false)),
                reims_vgpu_vulkan::core::sync::Condvar::new(),
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
            for _ in 0..crate::graphics::tests::PATIENCE {
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

    /// On the Vulkan platform, a buffer a decoder's table stores is told the guest's id for it
    /// (`super::tables`'s `name_for_loss`): as the guest's destroy lets go of it, its device's
    /// history of released ranges records it by that id, whose bits from 40 are its context, with
    /// the range it had.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: a real-time wait for the release
    fn on_vulkan_a_destroyed_buffer_s_release_names_its_guest_id_and_context() {
        use reims_vgpu_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT;
        use reims_vgpu_vulkan::RtPlatform;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::rt_platform::fault::ObjectKind;

        use crate::platform::ApiPlatform as _;

        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        // A platform of the test's own, whose device goes before the process exits.
        let platform = crate::rt_test_end::platform();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<RtPlatform>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
        );
        let id = (3 << RGFX_CONTEXT_OBJ_ID_SHIFT) | 0x21;
        assert_eq!(dec.decode(&device, &buffer_create(id)).unwrap(), Ok(()));
        let Some(ObjHandle::Buffer(buffer)) = dec.object(id) else {
            panic!("no buffer {id:#x}");
        };
        let range = (buffer.gpu_address(), buffer.length());
        assert_eq!(dec.decode(&device, &destroy(id)).unwrap(), Ok(()));
        let released = || {
            let (kept, _) = device.releases().kept();
            kept.into_iter().find(|release| release.id == id)
        };
        let mut waited = 0;
        while released().is_none() && waited < 1000 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            waited += 1;
        }
        let entry = released().expect("the buffer's release in its device's history");
        assert_eq!(
            (entry.kind, entry.context(), entry.storage),
            (ObjectKind::Buffer, 3, 0)
        );
        assert_eq!((entry.gpu_address, entry.len), range);
    }

    fn destroy(obj_id: u64) -> Vec<u8> {
        reims_vgpu_graphics::wire::RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: reims_vgpu_graphics::wire::RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<reims_vgpu_graphics::wire::RgfxObjectDestroy>() as u32,
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

    /// Queue 1's ring as a removal from its context's table reads it: the highest seqno of a
    /// command buffer published on it, taken or not, which the test sets.
    fn buffers_published(dec: &GraphicsDecoder<FakePlatform>) -> Arc<std::sync::atomic::AtomicU64> {
        let published = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let read = Arc::clone(&published);
        dec.queue_buffers_made(1, Arc::new(move || Some(read.load(Ordering::SeqCst))));
        published
    }

    /// The platform's buffer `obj_id` names in `dec`'s tables, its view of them let go of.
    fn platform_buffer(dec: &mut GraphicsDecoder<FakePlatform>, obj_id: u64) -> ObjId {
        let id = match dec.object(obj_id) {
            Some(ObjHandle::Buffer(b)) => b.id(),
            _ => panic!("no buffer {obj_id}"),
        };
        dec.end_message();
        id
    }

    /// Whether the platform still holds buffer `id`.
    fn holds(p: &FakePlatform, id: ObjId) -> bool {
        p.world().live("MTLBuffer").contains(&id)
    }

    /// A destroy ring 0 carries out while a command buffer its process posted before it on a
    /// queue's ring is not yet committed (here, not yet taken) keeps the object for it, though the
    /// buffer's view, taken after the destroy, does not name it (its work reaches objects through
    /// argument buffers and heaps it names no entry of): the platform lets go of the object once
    /// that buffer is committed, before the buffer completes. The destroy holds up nothing.
    #[test]
    fn a_destroy_waits_for_the_commit_of_a_buffer_posted_before_it() {
        let (p, mut dec) = setup();
        let device = p.device();
        let published = buffers_published(&dec);
        let buffer = platform_buffer(&mut dec, 3);
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut queue = dec.sender(Some(0), super::super::Sinks::default());
        drop(dec);
        // Buffer 7 is on queue 1's ring as ring 0 takes the destroy.
        published.store(7, Ordering::SeqCst);
        assert_eq!(ring0.decode(&device, &destroy(3)).unwrap(), Ok(()));
        ring0.end_message();
        assert_eq!(kept(&ring0), 1, "kept for the buffer posted before it");
        assert!(
            holds(&p, buffer),
            "let go of before the buffer posted first"
        );
        // Its view, taken now, does not name it; its commit lets it go.
        let message = command_buffer_message(100, 7, &[]);
        assert_eq!(queue.decode(&device, &message).unwrap(), Ok(()));
        queue.end_message();
        assert_eq!(kept(&ring0), 0);
        assert!(!holds(&p, buffer), "kept past the buffer's commit");
        p.world().complete_all(CompleteOn::Caller);
        // A destroy with every buffer published committed waits for nothing.
        let other = platform_buffer(&mut ring0, 2);
        assert_eq!(ring0.decode(&device, &destroy(2)).unwrap(), Ok(()));
        assert_eq!(kept(&ring0), 0);
        assert!(!holds(&p, other));
    }

    /// A make behind a destroy that waits for a buffer's commit is carried out at once, as is one
    /// over a live id, whose old object then waits for the same buffer: neither is held, and both
    /// old objects go at the buffer's commit.
    #[test]
    fn a_make_behind_a_waiting_destroy_is_not_held() {
        let (p, mut dec) = setup();
        let device = p.device();
        let published = buffers_published(&dec);
        let destroyed = platform_buffer(&mut dec, 3);
        let replaced = platform_buffer(&mut dec, 2);
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut queue = dec.sender(Some(0), super::super::Sinks::default());
        drop(dec);
        published.store(7, Ordering::SeqCst);
        assert_eq!(ring0.decode(&device, &destroy(3)).unwrap(), Ok(()));
        let objects = ring0.object_count();
        assert_eq!(ring0.decode(&device, &buffer_create(6)).unwrap(), Ok(()));
        assert_eq!(ring0.object_count(), objects + 1, "made at once");
        assert!(ring0.object(6).is_some());
        ring0.end_message();
        // Made again over a live id: the new one at once, the old one kept for the buffer.
        assert_eq!(ring0.decode(&device, &buffer_create(2)).unwrap(), Ok(()));
        ring0.end_message();
        assert_ne!(platform_buffer(&mut ring0, 2), replaced, "the new one");
        assert_eq!(kept(&ring0), 2);
        assert!(holds(&p, destroyed) && holds(&p, replaced));
        let message = command_buffer_message(100, 7, &[]);
        assert_eq!(queue.decode(&device, &message).unwrap(), Ok(()));
        queue.end_message();
        assert_eq!(kept(&ring0), 0);
        assert!(!holds(&p, destroyed) && !holds(&p, replaced));
    }

    /// A command buffer that fails, as one refused at its decode does, still raises its queue's
    /// commit mark as its message ends: a destroy that waits for it waits no more, rather than until
    /// its queue ends.
    #[test]
    fn a_destroy_waits_no_more_for_a_buffer_that_failed() {
        let (p, mut dec) = setup();
        let device = p.device();
        let published = buffers_published(&dec);
        let buffer = platform_buffer(&mut dec, 3);
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut queue = dec.sender(Some(0), super::super::Sinks::default());
        drop(dec);
        published.store(7, Ordering::SeqCst);
        assert_eq!(ring0.decode(&device, &destroy(3)).unwrap(), Ok(()));
        assert!(holds(&p, buffer));
        // It names buffer 9, which its process never made.
        let message = command_buffer_message(100, 7, &draw_ops(9, 0, 0));
        assert!(!matches!(queue.decode(&device, &message), Ok(Ok(()))));
        queue.end_message();
        assert_eq!(kept(&ring0), 0);
        assert!(!holds(&p, buffer), "kept past the failed buffer's end");
    }

    /// A destroy that waits for a buffer of a queue that then ends (its destroy) waits no more:
    /// the queue commits nothing more, and no view is open.
    #[test]
    fn a_destroy_waits_no_more_for_the_buffers_of_a_queue_that_ended() {
        let (p, mut dec) = setup();
        let device = p.device();
        let published = buffers_published(&dec);
        let buffer = platform_buffer(&mut dec, 3);
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        drop(dec);
        published.store(7, Ordering::SeqCst);
        assert_eq!(ring0.decode(&device, &destroy(3)).unwrap(), Ok(()));
        assert!(holds(&p, buffer));
        assert_eq!(ring0.decode(&device, &destroy(1)).unwrap(), Ok(()));
        assert_eq!(kept(&ring0), 0);
        assert!(!holds(&p, buffer), "kept past its queue's end");
    }

    /// A context's close with a command buffer begun and one published behind it, never begun:
    /// the objects go once the begun buffer is done with them (its view closes after its commit),
    /// one its process destroyed before the close among them, which waited for the buffer behind;
    /// none is kept for the buffer that never comes.
    #[test]
    fn a_close_keeps_the_objects_for_the_buffers_in_flight() {
        let (p, mut dec) = setup();
        let device = p.device();
        let published = buffers_published(&dec);
        let destroyed = platform_buffer(&mut dec, 2);
        let named = platform_buffer(&mut dec, 3);
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut queue = dec.sender(Some(0), super::super::Sinks::default());
        drop(dec);
        let message = command_buffer_message(100, 8, &[]);
        let begun = queue
            .begin_taken(&device, &message, Some(1), false)
            .unwrap()
            .unwrap();
        published.store(9, Ordering::SeqCst);
        assert_eq!(ring0.decode(&device, &destroy(2)).unwrap(), Ok(()));
        ring0.end_message();
        let (_, closed) = ring0.close_tables(&device, 0);
        closed.let_go();
        assert!(
            holds(&p, destroyed) && holds(&p, named),
            "let go of under the begun buffer"
        );
        assert_eq!(
            queue
                .finish_command_buffer(&device, &message, begun)
                .unwrap(),
            Ok(())
        );
        queue.end_message();
        assert!(
            !holds(&p, destroyed) && !holds(&p, named),
            "kept past the begun buffer"
        );
    }

    /// The ids [`each_named`] reads off an op stream: its compute ops' in stream order, each run's
    /// records each, nil ids, ops that name nothing and ops of other groups left out.
    #[test]
    fn a_stream_s_compute_names_come_in_order() {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_SET_BUFFERS,
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RGFX_OP_COMPUTE_USE_HEAPS,
            RgfxComputeHeapBinding, RgfxComputeOpDispatchThreadgroups,
            RgfxComputeOpSetComputePipelineState, RgfxComputeOpUseHeaps,
        };
        let mut ops = op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        );
        ops.extend(op(
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState { pipeline_obj_id: 7 }.as_bytes(),
        ));
        ops.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[11, 0, 12])));
        let mut heaps = RgfxComputeOpUseHeaps { count: 1 }.as_bytes().to_vec();
        heaps.extend_from_slice(RgfxComputeHeapBinding { heap_obj_id: 31 }.as_bytes());
        ops.extend(op(RGFX_OP_COMPUTE_USE_HEAPS, &heaps));
        ops.extend(op(
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
            RgfxComputeOpDispatchThreadgroups::default().as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        assert_eq!(named(&ops), [7, 11, 12, 31]);
    }

    /// The ids [`each_named`] reads off a render pass: its render ops' in stream order, each time
    /// an op names one, nil ids, ops that name nothing and the begin's attachments left out.
    #[test]
    fn a_stream_s_render_names_come_in_order() {
        let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 7 }.as_bytes(),
        ));
        for buffer_obj_id in [11, 0, 11] {
            ops.extend(op(
                RGFX_OP_RENDER_SET_VERTEX_BUFFER,
                RgfxOpSetVertexBuffer {
                    buffer_obj_id,
                    offset: 0,
                    index: 0,
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
        }
        ops.extend(op(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            RgfxOpSetFragmentTexture {
                texture_obj_id: 21,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_DRAW_INDEXED_PRIMITIVES,
            RgfxOpDrawIndexedPrimitives {
                primitive_type: 3,
                index_count: 3,
                index_type: 0,
                index_buffer_obj_id: 12,
                index_buffer_offset: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        assert_eq!(named(&ops), [7, 11, 11, 21, 12]);
        let mut unknown = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
        unknown.extend(op(0x1fff, &[0; 8]));
        unknown.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState { pipeline_obj_id: 7 }.as_bytes(),
        ));
        assert_eq!(
            named(&unknown),
            Vec::<u64>::new(),
            "a render op the walk does not decode"
        );
    }

    /// [`each_named`] reads no further than the walk decodes: an op whose head counts more records
    /// than its body holds, whose length overruns the stream or is no multiple of 8, or a compute
    /// tag the walk does not decode ends the reading there; and an answer of false ends it at once.
    #[test]
    fn a_short_or_malformed_stream_ends_the_reading() {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_BUFFERS, RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState,
        };
        let mut counted = op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[11]));
        let mut short = set_buffers(&[12, 13]);
        // The head counts 3 records; the op holds 2.
        short[8..16].copy_from_slice(&3u64.to_le_bytes());
        counted.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &short));
        counted.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[14])));
        assert_eq!(
            named(&counted),
            [11],
            "the second op's records run past its body"
        );
        let mut cut = op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[11]));
        cut.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[12])));
        cut.truncate(cut.len() - 8);
        assert_eq!(named(&cut), [11], "the second op runs past the stream");
        let mut odd = op(
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState { pipeline_obj_id: 7 }.as_bytes(),
        );
        odd[4..8].copy_from_slice(&12u32.to_le_bytes());
        assert_eq!(named(&odd), Vec::<u64>::new());
        let mut unknown = op(0x2fff, &[0; 8]);
        unknown.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[11])));
        assert_eq!(
            named(&unknown),
            Vec::<u64>::new(),
            "an op the walk does not decode"
        );
        let mut first = Vec::new();
        each_named(
            &op(RGFX_OP_COMPUTE_SET_BUFFERS, &set_buffers(&[11, 12, 13])),
            |id| {
                first.push(id);
                false
            },
        );
        assert_eq!(first, [11]);
    }

    /// The ids [`each_named`] reads off `ops`, in order.
    fn named(ops: &[u8]) -> Vec<u64> {
        let mut out = Vec::new();
        each_named(ops, |id| {
            out.push(id);
            true
        });
        out
    }

    /// A `SET_BUFFERS` body binding `ids` from slot 0, each at offset 256.
    fn set_buffers(ids: &[u64]) -> Vec<u8> {
        use reims_vgpu_graphics::ops::compute::{
            RgfxComputeBufferBinding, RgfxComputeOpSetBuffers,
        };
        let mut body = RgfxComputeOpSetBuffers {
            location: 0,
            length: ids.len() as u64,
        }
        .as_bytes()
        .to_vec();
        for &buffer_obj_id in ids {
            let binding = RgfxComputeBufferBinding {
                buffer_obj_id,
                offset: 256,
            };
            body.extend_from_slice(binding.as_bytes());
        }
        body
    }

    /// The names a walk's compute ops give, found as it starts ([`each_named`]), are what its
    /// table held at its begin, as each op would have found them alone: destroys after the walk
    /// found them, of the buffers and the texture it binds, and an id destroyed, made again and
    /// destroyed again, leave it binding the objects it began with, let go of once it ends. One
    /// destroyed before its begin is found by neither, and stays unfound though made again
    /// mid-walk: the op naming it faults.
    #[test]
    fn a_walk_s_compute_names_found_at_its_start_are_its_begin_s() {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_BUFFERS, RGFX_OP_COMPUTE_SET_TEXTURE, RgfxComputeBufferBinding,
            RgfxComputeOpSetBuffers, RgfxComputeOpSetTexture,
        };
        let (p, dec) = setup();
        let device = p.device();
        let buffer = |id| match dec.object(id) {
            Some(ObjHandle::Buffer(b)) => b,
            _ => panic!("no buffer {id}"),
        };
        let bound = Arg::objs(&[Some(buffer(2)), Some(buffer(3))]);
        let textured = Arg::opt(match dec.object(5) {
            Some(ObjHandle::Texture(t)) => Some(t),
            _ => panic!("no texture 5"),
        });
        let mut ring0 = dec.sender(None, super::super::Sinks::default());
        let mut walker = dec.sender(None, super::super::Sinks::default());
        drop(dec);
        let mut ops = op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        );
        let mut binds = RgfxComputeOpSetBuffers {
            location: 0,
            length: 2,
        }
        .as_bytes()
        .to_vec();
        for buffer_obj_id in [2, 3] {
            let b = RgfxComputeBufferBinding {
                buffer_obj_id,
                offset: 0,
            };
            binds.extend_from_slice(b.as_bytes());
        }
        ops.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &binds));
        ops.extend(op(
            RGFX_OP_COMPUTE_SET_TEXTURE,
            RgfxComputeOpSetTexture {
                texture_obj_id: 5,
                index: 0,
            }
            .as_bytes(),
        ));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let message = command_buffer_message(100, 1, &ops);
        // The gate stops the walk after its encoder's begin: past its start, before its binds.
        let gate = Gate::on(&ring0, 0);
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
            assert_eq!(kept(&ring0), 3, "kept for the walk: 5, 2 and 3");
            gate.open();
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(kept(&ring0), 0, "let go of with the walk");
        let log = p.call_log();
        assert_eq!(
            log.calls_to("setBuffers:offsets:withRange:")
                .last()
                .unwrap()
                .args[0],
            bound,
            "the buffers it began with"
        );
        assert_eq!(
            log.calls_to("setTexture:atIndex:").last().unwrap().args[0],
            textured,
            "the texture it began with"
        );
        // Buffer 2 is gone before this walk's begin, and made again while it walks: the walk
        // names neither, and faults at its bind; one begun after the make names it.
        let mut ops = op(
            RGFX_OP_ENCODER_BEGIN_COMPUTE,
            RgfxComputePassDesc::default().as_bytes(),
        );
        let mut bind = RgfxComputeOpSetBuffers {
            location: 0,
            length: 1,
        }
        .as_bytes()
        .to_vec();
        let b = RgfxComputeBufferBinding {
            buffer_obj_id: 2,
            offset: 0,
        };
        bind.extend_from_slice(b.as_bytes());
        ops.extend(op(RGFX_OP_COMPUTE_SET_BUFFERS, &bind));
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
        let message = command_buffer_message(101, 2, &ops);
        ring0.set_walk_hook(None);
        let gate = Gate::on(&ring0, 0);
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| walker.decode(&p.device(), &message).map_err(|_| ()));
            gate.reached();
            assert_eq!(ring0.decode(&device, &buffer_create(2)).unwrap(), Ok(()));
            gate.open();
            assert!(
                walk.join().unwrap().is_err(),
                "named the buffer made after its begin"
            );
        });
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
            taken.begin_taken(&device, &late, Some(1), false).unwrap(),
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
        let sequencer = Arc::clone(&dec.shared.sequencer);
        drop(dec);
        let range = |obj_id| {
            reims_vgpu_graphics::wire::RgfxBufferDidModifyRange {
                header: RgfxMsgHeader {
                    r#type: reims_vgpu_graphics::wire::RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
                    size: size_of::<reims_vgpu_graphics::wire::RgfxBufferDidModifyRange>() as u32,
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
                        &range((2 << reims_vgpu_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT) | 2)
                    )
                    .unwrap(),
                Err(DecodeError::BadReference)
            );
            let held = crate::graphics::tests::eventually(|| sequencer.holds_up(0));
            assert!(held, "never waited for the answer's ticket");
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
        // The cuts themselves: no part completes as these tests walk.
        dec.set_cut_behind_work(true);
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
    /// after its last part's completion, with GPU times from its first part's start that span its
    /// parts' GPU time summed ([`super::super::split`], "Its GPU times"), and the error of a part
    /// that failed.
    #[test]
    fn a_buffer_walked_long_is_committed_in_parts_with_one_outcome() {
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
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
            "scheduled once, at its take"
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
            (300, 10, 50, RGFX_CB_OK),
            "10 + 15 + 15 of GPU time, from the first part's start"
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
                    reims_vgpu_graphics::wire::RGFX_CB_PAGE_FAULT
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
        let committed = p.world().committed();
        assert_eq!(committed.len(), 1, "a platform that keeps it whole");
        assert_eq!(
            p.world().joins_next(committed[0]),
            Some(false),
            "its commit the whole's, unmarked"
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

    /// While a command buffer's parts are out, a command buffer of another queue of its context,
    /// begun before it, is walked but committed only after its last part, waiting at its commit;
    /// one begun after its take waits for its ticket there anyway, as it was scheduled at its take
    /// ([`Begun::post_at_take`]).
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
        let (begun_tx, begun_rx) = std::sync::mpsc::channel::<()>();
        let (out_tx, out_rx) = std::sync::mpsc::channel::<()>();
        let (begun_tx, out_tx, out_rx) = (
            Mutex::new(Some(begun_tx)),
            Mutex::new(Some(out_tx)),
            Mutex::new(out_rx),
        );
        let clock = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |index| {
            clock.virtual_clock().advance(SPLIT_AFTER);
            // The other queue's buffer, begun first, walks on once the first part is out.
            if std::thread::current().name() == Some("other") {
                if index == 0 {
                    begun_tx.lock().take();
                    let _ = out_rx.lock().recv();
                }
                return;
            }
            // Its first part is out, at its first encoder's end: the other queue's buffer goes;
            // at its last op, that buffer waits at its commit.
            if index == 1 {
                out_tx.lock().take();
            }
            if index == last {
                assert!(
                    crate::graphics::tests::eventually(|| gate.waiting(0) == 1),
                    "the other queue's buffer never waited at its commit"
                );
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
                .spawn_scoped(scope, || other.decode(&p.device(), &later).unwrap())
                .unwrap();
            let _ = begun_rx.recv();
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
                reims_vgpu_graphics::wire::RGFX_CB_INTERNAL
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
        use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
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

    /// No part is cut at an encoder's end that only framing ops follow ([`super::super::split`],
    /// "Where a part is cut"): a buffer whose last encoder is followed by a signal of a shared
    /// event and its ask for its submission (`SUBMITTED`, which the bundle puts last on every
    /// buffer of a queue that presents) is committed in a part for each encoder, its last part
    /// holding its last encoder and the signal, rather than in one more part with no work. The
    /// signal still follows the last encoder's work, and goes to the guest ahead of the buffer's
    /// completion; a cut or a deal whose last encoder is followed by an encoder still cuts there.
    #[test]
    fn no_part_is_cut_where_no_encoder_follows() {
        use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
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
        let tail = [
            event_op(RGFX_OP_SIGNAL_EVENT, 7, 2),
            op(RGFX_OP_SUBMITTED, &[]),
        ]
        .concat();
        let ops = [passes(3), tail.clone()].concat();
        let mark = p.call_log().mark();
        let message = command_buffer_message(355, 85, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(p.call_log().calls_to("split").len(), 2, "a cut a pass");
        let parts = p.world().committed();
        assert_eq!(parts.len(), 3, "a part a pass, none of the tail alone");
        let signalled: Vec<_> = (p.call_log().since(mark).into_iter())
            .filter(|c| c.sel == "encodeSignalEvent:value:")
            .map(|c| c.obj)
            .collect();
        assert_eq!(signalled, [parts[2]], "the signal in the last pass's part");
        posted.lock().clear();
        for part in &parts[..2] {
            let ok = CommandBufferStatus::COMPLETED;
            assert!(p.world().complete(*part, ok, None, CompleteOn::Caller));
        }
        assert!(posted.lock().is_empty(), "posted before its last part");
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [
                (RGFX_EVT_EVENT_SIGNALLED, 7, 2),
                (RGFX_EVT_CB_COMPLETE, 355, RGFX_CB_OK),
            ]
        );

        // Dealt to two threads, a chunk an encoder: the last chunk holds the last pass and the
        // ask, so 8 parts, as without the ask.
        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        dec.set_helper_first(true);
        let ops = [passes(8), op(RGFX_OP_SUBMITTED, &[])].concat();
        let mark = p.call_log().mark();
        let message = command_buffer_message(356, 86, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let sels = p.call_log().sels_since(mark);
        assert_eq!(sels.iter().filter(|s| **s == "part").count(), 7);
        assert_eq!(sels.iter().filter(|s| **s == "commitPart").count(), 7);
        assert!(!sels.contains(&"split"), "no cut inside a chunk");
        assert_eq!(p.world().committed().len(), 8, "a part a pass");
        posted.lock().clear();
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_COMPLETE, 356, RGFX_CB_OK)]);

        // Where a deal is cut.
        let pass = render_ops().len();
        let at = |passes: &[usize]| Some(passes.iter().map(|n| n * pass).collect::<Vec<_>>());
        let asked = [passes(8), op(RGFX_OP_SUBMITTED, &[])].concat();
        assert_eq!(
            deal(&asked, 0, 0, 64),
            at(&[1, 2, 3, 4, 5, 6, 7]),
            "no cut after the last encoder"
        );
        assert_eq!(
            deal(&asked, 0, 0, 2),
            at(&[5]),
            "2 chunks, cut at the first encoder's end past half the stream"
        );
        let long = pass_of(
            &(0..64)
                .map(|i| draw_ops(2, 16 * i, i))
                .collect::<Vec<_>>()
                .concat(),
        );
        let last_long = [passes(7), long, op(RGFX_OP_SUBMITTED, &[])].concat();
        assert_eq!(
            deal(&last_long, 0, 0, 2),
            None,
            "2 chunks at most, the first encoder's end past half the stream the last's"
        );

        // What follows an encoder's end.
        assert!(!encoder_follows(&[]));
        assert!(!encoder_follows(&tail));
        assert!(!encoder_follows(&event_op(RGFX_OP_WAIT_EVENT, 7, 2)));
        assert!(encoder_follows(&[tail.clone(), blit_ops()].concat()));
        assert!(encoder_follows(&[tail.clone(), render_ops()].concat()));
        let compute = op(RGFX_OP_ENCODER_BEGIN_COMPUTE, &[]);
        assert!(encoder_follows(&[tail.clone(), compute].concat()));
        let mut short = tail.clone();
        short.truncate(tail.len() - 4);
        assert!(encoder_follows(&short), "a stream that does not parse");
        assert!(
            encoder_follows(
                &render_ops()[op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body()).len()..]
            ),
            "an encoder's op, which the walk faults at"
        );
    }

    /// Whether each of `parts` is marked to join the next one's completion
    /// (`CommandBuffer::set_completion_joins_next`).
    fn marks(p: &FakePlatform, parts: &[ObjId]) -> Vec<bool> {
        (parts.iter())
            .map(|part| p.world().joins_next(*part).expect("a live command buffer"))
            .collect()
    }

    /// A command buffer's parts but its last are marked to join the next one's completion, each
    /// where it waits for and signals no event ([`super::super::split`], "Its parts'
    /// completions"): a part that signals an event, plain or shared, goes unmarked, and so does
    /// the last, which carries the buffer's records. A marked part's completion posts nothing the
    /// guest sees, however late it comes, and a fault in one fails the buffer as before.
    #[test]
    fn a_buffer_s_parts_but_its_last_join_the_next_completion_unless_they_signal() {
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
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
        let ok = CommandBufferStatus::COMPLETED;

        let message = command_buffer_message(390, 120, &passes(4));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let parts = p.world().committed();
        assert_eq!(marks(&p, &parts), [true, true, true, false]);
        posted.lock().clear();
        for part in &parts[..3] {
            assert!(p.world().complete(*part, ok, None, CompleteOn::Caller));
            assert!(
                posted.lock().is_empty(),
                "a marked part's completion posts nothing"
            );
        }
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(*posted.lock(), [(RGFX_EVT_CB_COMPLETE, 390, RGFX_CB_OK)]);

        // A part that signals the plain event 6, and one that signals the shared event 7, which
        // the guest has as the event reaches it: neither is marked.
        let ops = [
            blit_ops(),
            event_op(RGFX_OP_SIGNAL_EVENT, 6, 1),
            blit_ops(),
            event_op(RGFX_OP_SIGNAL_EVENT, 7, 2),
            blit_ops(),
            blit_ops(),
        ]
        .concat();
        let mark = p.call_log().mark();
        let message = command_buffer_message(391, 121, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let parts = p.world().committed();
        let signalled: Vec<_> = (p.call_log().since(mark).into_iter())
            .filter(|c| c.sel == "encodeSignalEvent:value:")
            .map(|c| c.obj)
            .collect();
        assert_eq!(signalled, parts[1..3], "each signal in a part of its own");
        assert_eq!(marks(&p, &parts), [true, false, false, false]);
        posted.lock().clear();
        assert!(p.world().complete(parts[0], ok, None, CompleteOn::Caller));
        assert!(posted.lock().is_empty());
        for part in &parts[1..3] {
            assert!(p.world().complete(*part, ok, None, CompleteOn::Caller));
        }
        assert_eq!(*posted.lock(), [(RGFX_EVT_EVENT_SIGNALLED, 7, 2)]);
        p.world().complete_all(CompleteOn::Caller);
        posted.lock().clear();

        // A marked part fails: the buffer's error, as before.
        let message = command_buffer_message(392, 122, &passes(3));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let parts = p.world().committed();
        assert_eq!(marks(&p, &parts), [true, true, false]);
        posted.lock().clear();
        let error = NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code: 3,
            description: "Page Fault".to_owned(),
        };
        let failed = CommandBufferStatus::ERROR;
        assert!(
            p.world()
                .complete(parts[1], failed, Some(error), CompleteOn::Caller)
        );
        assert!(posted.lock().is_empty());
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            *posted.lock(),
            [(
                RGFX_EVT_CB_COMPLETE,
                392,
                reims_vgpu_graphics::wire::RGFX_CB_PAGE_FAULT
            )]
        );
    }

    /// The first part of a command buffer that waits on the GPU from its begin, for a buffer of
    /// another queue it runs after (its `SURFACE_ORDER`), is not marked to join the next one's
    /// completion; its later parts but the last are.
    #[test]
    fn a_first_part_that_waits_from_its_begin_is_not_marked() {
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_ORDER, RgfxSurfaceOrderWait};
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
        let order = |waits: &[RgfxSurfaceOrderWait]| op(RGFX_OP_SURFACE_ORDER, waits.as_bytes());
        // Queue 6's buffer, committed and not finished.
        let read = command_buffer_on(6, 200, &order(&[]));
        assert_eq!(dec.decode(&device, &read).unwrap(), Ok(()));
        let mut ops = order(&[RgfxSurfaceOrderWait {
            queue_obj_id: 6,
            seqno: 200,
        }]);
        ops.extend(passes(3));
        let mark = p.call_log().mark();
        let message = command_buffer_message(393, 123, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let committed = p.world().committed();
        let parts = &committed[1..];
        let waited: Vec<_> = (p.call_log().since(mark).into_iter())
            .filter(|c| c.sel == "encodeWaitForEvent:value:")
            .map(|c| c.obj)
            .collect();
        assert_eq!(waited, parts[..1], "the wait in its first part");
        assert_eq!(marks(&p, parts), [false, true, false]);
    }

    /// A walk behind a command buffer its queue committed and has not yet seen complete cuts its
    /// parts by their estimated GPU time, and deals a buffer behind one in two chunks, one a
    /// thread ([`super::super::split`], "How many parts"): behind its own first part, with nothing
    /// to estimate its GPU time from, it cuts each [`SPLIT_AFTER`] as with nothing out; with each
    /// part seen complete as the walk goes, it cuts at each encoder's end as before; behind an
    /// earlier buffer of its queue whose walks measured a low rate of GPU time to walk time it
    /// cuts none, and behind one whose walks measured a high one it cuts at each encoder's end;
    /// a part cut behind such work is committed unmarked, so its completion is seen at its own
    /// end; dealt behind one, it is two parts, and dealt with nothing out, a part each chunk. The
    /// guest has each buffer's `CB_COMPLETE` once, after its last part, as before.
    #[test]
    fn a_walk_behind_incomplete_work_cuts_its_parts_by_their_estimated_gpu_time() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        dec.set_cut_behind_work(false);
        let device = p.device();
        let ok = (RGFX_EVT_CB_COMPLETE, RGFX_CB_OK);
        let rate = Arc::clone(&dec.queue_counts(1).expect("queue 1").rate);

        // Behind its own first part, nothing estimated: a part each encoder, its first marked and
        // the others, cut behind it, not.
        let cut = dec.parts_cut();
        let message = command_buffer_message(400, 130, &passes(4));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(dec.parts_cut() - cut, 3, "a part each encoder but the last");
        let parts = p.world().committed();
        assert_eq!(marks(&p, &parts), [true, false, false, false]);
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(answered(&posted).last(), Some(&ok));

        // Each part seen complete as the walk goes: a part each encoder, as before.
        let world = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |_| {
            world.virtual_clock().advance(SPLIT_AFTER);
            world.world().complete_all(CompleteOn::Caller);
        })));
        let cut = dec.parts_cut();
        let message = command_buffer_message(401, 131, &passes(4));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(dec.parts_cut() - cut, 3, "a part each encoder but the last");
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(answered(&posted).last(), Some(&ok));
        let clock = Arc::clone(&p);
        dec.set_walk_hook(Some(Arc::new(move |_| {
            clock.virtual_clock().advance(SPLIT_AFTER);
        })));

        // Behind an earlier buffer of its queue, not yet complete: whole where its queue's walks
        // measured a GPU time a tenth of their walk's, which a part reaches only at 160 ms of walk.
        rate.set(0.1);
        let earlier = command_buffer_message(402, 132, &blit_ops());
        assert_eq!(dec.decode(&device, &earlier).unwrap(), Ok(()));
        let cut = dec.parts_cut();
        let message = command_buffer_message(403, 133, &passes(4));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(dec.parts_cut() - cut, 0, "no part");
        assert_eq!(
            p.world().committed().len(),
            2,
            "the earlier buffer and this one"
        );
        p.world().complete_all(CompleteOn::Caller);
        let completes: Vec<_> = (answered(&posted).into_iter())
            .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .collect();
        assert_eq!(completes, [ok, ok]);

        // Where they measured a GPU time 100 times their walk's, a part reaches 16 ms at 160 µs of
        // walk: a part each encoder, none marked.
        rate.set(100.0);
        let earlier = command_buffer_message(404, 134, &blit_ops());
        assert_eq!(dec.decode(&device, &earlier).unwrap(), Ok(()));
        let cut = dec.parts_cut();
        let message = command_buffer_message(405, 135, &passes(4));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(dec.parts_cut() - cut, 3, "a part each encoder but the last");
        let parts = p.world().committed();
        assert_eq!(marks(&p, &parts[1..]), [false, false, false, false]);
        p.world().complete_all(CompleteOn::Caller);
        let completes: Vec<_> = (answered(&posted).into_iter())
            .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .collect();
        assert_eq!(completes, [ok, ok]);

        // Dealt, a chunk each encoder: behind an earlier buffer, two parts, one a thread; with
        // nothing out, a part each chunk.
        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        dec.set_helper_first(true);
        let earlier = command_buffer_message(406, 136, &blit_ops());
        assert_eq!(dec.decode(&device, &earlier).unwrap(), Ok(()));
        let message = command_buffer_message(407, 137, &passes(8));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(
            p.world().committed().len(),
            3,
            "the earlier buffer and two parts"
        );
        p.world().complete_all(CompleteOn::Caller);
        let message = command_buffer_message(408, 138, &passes(8));
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(p.world().committed().len(), 8, "a part each chunk");
        p.world().complete_all(CompleteOn::Caller);
        let completes: Vec<_> = (answered(&posted).into_iter())
            .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .collect();
        assert_eq!(completes, [ok, ok, ok]);
    }

    /// [`splitting`], its walks paced on the thread that decodes them, as a lane's are.
    fn pacing() -> Split {
        let split = splitting();
        split.dec.set_pace_anywhere(true);
        split
    }

    /// Waits, in real time, until `p` has `n` command buffers or more committed and not
    /// complete, and answers them.
    fn committed_at_least(p: &FakePlatform, n: usize) -> Vec<ObjId> {
        let mut waited = std::time::Duration::ZERO;
        loop {
            let committed = p.world().committed();
            if committed.len() >= n {
                return committed;
            }
            assert!(
                waited < std::time::Duration::from_secs(10),
                "{} committed",
                committed.len()
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
            waited += std::time::Duration::from_millis(1);
        }
    }

    /// Whether a walk of `decoders` waits after a cut for its parts to be seen complete
    /// ([`Decoders::pace_waits`](super::super::Decoders::pace_waits)), polled until one does.
    fn pace_waiting(decoders: &super::super::Decoders<FakePlatform>) -> bool {
        crate::graphics::tests::eventually(|| decoders.pace_waits() > 0)
    }

    /// A paced walk waits after a cut while its parts out run far ahead of the GPU
    /// ([`super::super::split`], "How far ahead of the GPU"): with nothing to estimate their GPU
    /// time from, once two parts or more are out and walked over [`PACE_UNKNOWN_WALK`]; it goes
    /// on as a part is seen complete, and, its parts' GPU time a small share of their walk's, cuts
    /// and commits the rest without waiting again. Its queue keeps the rate measured. The guest
    /// has one `CB_COMPLETE`, after its last part.
    #[test]
    fn a_paced_walk_waits_while_its_parts_run_far_ahead_of_the_gpu() {
        let Split {
            p, mut dec, posted, ..
        } = pacing();
        let device = p.device();
        let rate = Arc::clone(&dec.queue_counts(1).expect("queue 1").rate);
        let decoders = dec.decoders();
        let message = command_buffer_message(410, 140, &passes(8));
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| dec.decode(&device, &message).unwrap());
            let out = committed_at_least(&p, 2);
            assert!(pace_waiting(&decoders), "never waited for its parts");
            let held = p.world().committed();
            assert!(!walk.is_finished(), "waits with {} parts out", held.len());
            assert!(held.len() < 8, "{} parts out", held.len());
            // A tenth of a millisecond of GPU time against at least 250 µs of walk.
            assert!(p.world().set_gpu_times(out[0], 1_000, 101_000));
            assert!(p.world().complete(
                out[0],
                CommandBufferStatus::COMPLETED,
                None,
                CompleteOn::Caller
            ));
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        assert_eq!(p.world().committed().len(), 7, "the rest committed");
        assert!(
            rate.get().is_some_and(|r| r > 0.0 && r < 1.0),
            "{:?}",
            rate.get()
        );
        p.world().complete_all(CompleteOn::Caller);
        let completes: Vec<_> = (answered(&posted).into_iter())
            .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .collect();
        assert_eq!(completes, [(RGFX_EVT_CB_COMPLETE, RGFX_CB_OK)]);
    }

    /// A paced walk whose parts' estimated GPU time stays low waits after a cut while the oldest
    /// of its parts out was committed [`PACE_AHEAD`] ago or more ([`super::super::split`], "How
    /// far ahead of the GPU"): its queue's walks measured a GPU time a hundredth of their walk's,
    /// and its ops walk 4 ms each, so its parts reach that age with a fraction of a millisecond
    /// estimated. It goes on once they are seen complete. The guest has one `CB_COMPLETE`, after
    /// its last part.
    #[test]
    fn a_paced_walk_waits_while_its_oldest_part_out_is_a_refresh_old() {
        use std::sync::atomic::AtomicU64;
        let Split {
            p, mut dec, posted, ..
        } = pacing();
        let device = p.device();
        let step = Arc::new(AtomicU64::new(4_000_000));
        let clock = Arc::clone(&p);
        let steps = Arc::clone(&step);
        dec.set_walk_hook(Some(Arc::new(move |_| {
            let ns = steps.load(std::sync::atomic::Ordering::Relaxed);
            clock.virtual_clock().advance(Duration::from_nanos(ns));
        })));
        dec.queue_counts(1).expect("queue 1").rate.set(0.01);
        let decoders = dec.decoders();
        let message = command_buffer_message(415, 145, &passes(8));
        std::thread::scope(|scope| {
            let walk = scope.spawn(|| dec.decode(&device, &message).unwrap());
            committed_at_least(&p, 2);
            assert!(pace_waiting(&decoders), "never waited for its parts");
            let held = p.world().committed();
            assert!(!walk.is_finished(), "waits with {} parts out", held.len());
            assert!(held.len() < 8, "{} parts out", held.len());
            // The rest walks in no time, so it is one part.
            step.store(0, std::sync::atomic::Ordering::Relaxed);
            for part in held {
                assert!(p.world().set_gpu_times(part, 1_000, 2_000));
                assert!(p.world().complete(
                    part,
                    CommandBufferStatus::COMPLETED,
                    None,
                    CompleteOn::Caller
                ));
            }
            assert_eq!(walk.join().unwrap(), Ok(()));
        });
        p.world().complete_all(CompleteOn::Caller);
        let completes: Vec<_> = (answered(&posted).into_iter())
            .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
            .collect();
        assert_eq!(completes, [(RGFX_EVT_CB_COMPLETE, RGFX_CB_OK)]);
    }

    /// What ends a paced walk's wait at once ([`super::super::split`], "How far ahead of the
    /// GPU").
    #[derive(Clone, Copy, Debug)]
    enum PaceEnd {
        /// A reset of the device begins.
        Reset,
        /// The buffer's queue ends.
        QueueEnd,
        /// A part out fails.
        Fault,
        /// [`PACE_WAIT_MOST`] passes.
        Bound,
    }

    /// A paced walk's wait ends at once where a reset of the device begins, its queue ends, a part
    /// out fails or [`PACE_WAIT_MOST`] passes, none of its parts seen complete; it then walks on
    /// unpaced to its end, cutting and committing the rest as a walk did before it was paced, so
    /// nothing is left half-committed. The guest has one `CB_COMPLETE`, after its last part, with
    /// the failed part's error.
    #[test]
    fn a_paced_walk_s_wait_ends_at_a_reset_its_queue_s_end_a_fault_or_its_bound() {
        use reims_vgpu_vulkan::core::platform::types::NsErrorText;
        for end in [
            PaceEnd::Reset,
            PaceEnd::QueueEnd,
            PaceEnd::Fault,
            PaceEnd::Bound,
        ] {
            let Split {
                p, mut dec, posted, ..
            } = pacing();
            let device = p.device();
            let shared = Arc::clone(&dec.shared);
            let marks = Arc::clone(&dec.queue_counts(1).expect("queue 1").order);
            let decoders = dec.decoders();
            let message = command_buffer_message(420, 150, &passes(8));
            std::thread::scope(|scope| {
                let walk = scope.spawn(|| dec.decode(&device, &message).unwrap());
                let out = committed_at_least(&p, 2);
                assert!(
                    pace_waiting(&decoders),
                    "{end:?}: never waited for its parts"
                );
                assert!(!walk.is_finished(), "{end:?}: waits");
                match end {
                    PaceEnd::Reset => shared.commit_cancel.begin(),
                    PaceEnd::QueueEnd => marks.commit.end(),
                    PaceEnd::Fault => assert!(p.world().complete(
                        out[0],
                        CommandBufferStatus::ERROR,
                        Some(NsErrorText {
                            domain: "MTLCommandBufferErrorDomain".to_owned(),
                            code: 3,
                            description: "Page Fault".to_owned(),
                        }),
                        CompleteOn::Caller
                    )),
                    PaceEnd::Bound => p.virtual_clock().advance(PACE_WAIT_MOST),
                }
                assert_eq!(walk.join().unwrap(), Ok(()), "{end:?}");
            });
            let left = p.world().committed().len();
            let expected = if matches!(end, PaceEnd::Fault) { 7 } else { 8 };
            assert_eq!(left, expected, "{end:?}: every part committed");
            p.world().complete_all(CompleteOn::Caller);
            let status = if matches!(end, PaceEnd::Fault) {
                reims_vgpu_graphics::wire::RGFX_CB_PAGE_FAULT
            } else {
                RGFX_CB_OK
            };
            let completes: Vec<_> = (answered(&posted).into_iter())
                .filter(|(kind, _)| *kind == RGFX_EVT_CB_COMPLETE)
                .collect();
            assert_eq!(completes, [(RGFX_EVT_CB_COMPLETE, status)], "{end:?}");
        }
    }

    /// A walk that waits for an event by an op, or signals one, is never paced
    /// ([`super::super::split`], "How far ahead of the GPU"): a part's completion may wait for a
    /// buffer committed only after the walk's last part (one of its context's, which waits at the
    /// split's gate). One that signals in its first part cuts and commits every part with none
    /// seen complete; a buffer that waits for an event its context's other queue signals is
    /// committed whole, walked with nothing seen complete, and the signalling buffer is committed
    /// after it.
    #[test]
    fn a_walk_that_waits_for_or_signals_an_event_by_an_op_is_not_paced() {
        let Split { p, mut dec, .. } = pacing();
        let device = p.device();
        create_events(&p, &mut dec, 0);
        create_queue(&p, &mut dec, 9);

        // Signals plain event 6 in its first part.
        let mut ops = event_op(RGFX_OP_SIGNAL_EVENT, 6, 1);
        ops.extend(passes(8));
        let message = command_buffer_message(434, 162, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(p.world().committed().len(), 8, "8 parts");
        p.world().complete_all(CompleteOn::Caller);

        // Waits for shared event 7, which queue 9 of its context signals after it.
        let mut ops = event_op(RGFX_OP_WAIT_EVENT, 7, 1);
        ops.extend(passes(8));
        let message = command_buffer_message(436, 163, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(p.world().committed().len(), 1, "whole");
        let mut ops = event_op(RGFX_OP_SIGNAL_EVENT, 7, 1);
        ops.extend(render_ops());
        let signal = command_buffer_on(9, 437, &ops);
        assert_eq!(dec.decode(&device, &signal).unwrap(), Ok(()));
        assert_eq!(
            p.world().committed().len(),
            2,
            "the signal committed after it"
        );
        p.world().complete_all(CompleteOn::Caller);
    }

    /// A walk that waits from its begin for another queue's buffer is paced, and goes on unpaced
    /// once a buffer of its context waits for its last commit ([`super::super::split`], "How far
    /// ahead of the GPU"): the next buffer of that queue, of the same context, waits at the
    /// split's gate, which ends the walk's wait within a slice with none of its parts seen
    /// complete. The walk cuts and commits the rest, and the other queue's buffer is committed
    /// after its last part.
    #[test]
    fn a_paced_walk_goes_on_unpaced_once_a_buffer_of_its_context_waits_for_it() {
        use reims_vgpu_graphics::ops::{RGFX_OP_SURFACE_ORDER, RgfxSurfaceOrderWait};
        let Split { p, mut dec, .. } = pacing();
        let device = p.device();
        create_queue(&p, &mut dec, 9);
        // Queue 9's buffer, committed and not complete, which the walk waits for from its begin.
        let order = |waits: &[RgfxSurfaceOrderWait]| op(RGFX_OP_SURFACE_ORDER, waits.as_bytes());
        let read = command_buffer_on(9, 440, &order(&[]));
        assert_eq!(dec.decode(&device, &read).unwrap(), Ok(()));
        let mut ops = order(&[RgfxSurfaceOrderWait {
            queue_obj_id: 9,
            seqno: 440,
        }]);
        ops.extend(passes(8));
        let message = command_buffer_message(441, 170, &ops);
        let later = command_buffer_on(9, 442, &render_ops());
        let mut other = dec.sender(Some(0), super::super::Sinks::default());
        let commits = || {
            (p.call_log().calls().into_iter())
                .filter(|c| c.sel == "commit" || c.sel == "split")
                .filter_map(|c| c.thread)
                .filter(|t| t == "parts" || t == "other")
                .collect::<Vec<_>>()
        };
        let decoders = dec.decoders();
        std::thread::scope(|scope| {
            let walk = std::thread::Builder::new()
                .name("parts".into())
                .spawn_scoped(scope, || dec.decode(&p.device(), &message).unwrap())
                .unwrap();
            committed_at_least(&p, 3);
            assert!(
                pace_waiting(&decoders),
                "paced from its begin's wait: never waited"
            );
            assert!(!walk.is_finished(), "paced from its begin's wait");
            let out = commits().len();
            assert!(out < 8, "{out} parts out");
            let b = std::thread::Builder::new()
                .name("other".into())
                .spawn_scoped(scope, || other.decode(&p.device(), &later).unwrap())
                .unwrap();
            assert_eq!(walk.join().unwrap(), Ok(()));
            assert_eq!(b.join().unwrap(), Ok(()));
        });
        assert_eq!(
            commits(),
            [vec!["parts".to_owned(); 8], vec!["other".to_owned()]].concat(),
            "every part, none seen complete, then the other queue's buffer"
        );
        p.world().complete_all(CompleteOn::Caller);
    }

    /// Makes queue `obj_id` of `dec`'s context.
    fn create_queue(p: &FakePlatform, dec: &mut GraphicsDecoder<FakePlatform>, obj_id: u64) {
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&p.device(), queue.as_bytes()).unwrap(), Ok(()));
    }

    /// `n` passes of [`render_ops`].
    fn passes(n: usize) -> Vec<u8> {
        (0..n).map(|_| render_ops()).collect::<Vec<_>>().concat()
    }

    /// `n` passes each of one draw that names its pass's index: each pass's calls tell it from
    /// the others'.
    fn numbered(n: u64) -> Vec<u8> {
        (0..n)
            .map(|i| pass_of(&draw_ops(2, 16 * i, i)))
            .collect::<Vec<_>>()
            .concat()
    }

    /// A walk's call: its selector, and its arguments but the objects, which a walk makes anew.
    type Shape = (&'static str, Vec<String>);

    /// The calls a walk made since `mark`, each with whether a helper made it
    /// ([`super::super::helpers`]), and the object its command buffer makes an encoder on: every
    /// call of the platform's but the command buffers' makes, commits, parts and handlers, which a
    /// dealt walk makes otherwise than a whole one.
    fn walk_calls(p: &FakePlatform, mark: u64) -> Vec<(bool, Shape, Option<ObjId>)> {
        const NOT_THE_WALK: [&str; 7] = [
            "commandBuffer",
            "commit",
            "split",
            "part",
            "commitPart",
            "fold",
            "setCompletionJoinsNext",
        ];
        (p.call_log().since(mark).into_iter())
            .filter(|c| !NOT_THE_WALK.contains(&c.sel) && !c.sel.contains("Handler"))
            .map(|c| {
                let args = (c.args.iter())
                    .filter(|a| !matches!(a, Arg::Obj(_)))
                    .map(|a| format!("{a:?}"))
                    .collect();
                let made = c.sel.starts_with("renderCommandEncoder").then_some(c.obj);
                (
                    c.thread.as_deref() == Some("reims-vgpu.graphics-helper"),
                    (c.sel, args),
                    made,
                )
            })
            .collect()
    }

    /// `calls`' chunks, each with the command buffer it was encoded into, in their order: each
    /// thread's calls cut where an encoder is made on another command buffer than the last, and
    /// the chunks of both threads ordered by their command buffers, which the walk makes in the
    /// chunks' order before it walks any, the first chunk's the buffer's own, made before them
    /// ([`GraphicsDecoder::walk_dealt`]). A whole walk's, in its order, where each op of a dealt
    /// buffer was walked once, its chunks in their order ([`shapes`]).
    fn dealt_chunks(calls: &[(bool, Shape, Option<ObjId>)]) -> Vec<(Vec<Shape>, ObjId)> {
        let of = |helper: bool| {
            let mut chunks: Vec<(Vec<Shape>, ObjId)> = Vec::new();
            for (_, shape, made) in calls.iter().filter(|c| c.0 == helper) {
                match made {
                    Some(cb) if chunks.last().is_none_or(|(_, last)| last != cb) => {
                        chunks.push((vec![shape.clone()], *cb));
                    }
                    _ => chunks.last_mut().unwrap().0.push(shape.clone()),
                }
            }
            chunks
        };
        let mut dealt = [of(false), of(true)].concat();
        dealt.sort_by_key(|(_, cb)| *cb);
        dealt
    }

    /// The calls of `chunks`, in their order.
    fn shapes(chunks: &[(Vec<Shape>, ObjId)]) -> Vec<Shape> {
        chunks.iter().flat_map(|(c, _)| c.iter().cloned()).collect()
    }

    /// The calls of a whole walk, `calls`, all made here, in their order.
    fn whole_shapes(calls: &[(bool, Shape, Option<ObjId>)]) -> Vec<Shape> {
        assert!(calls.iter().all(|(helper, ..)| !helper), "walked here");
        calls.iter().map(|(_, shape, _)| shape.clone()).collect()
    }

    /// The command buffers of `chunks`, in their order.
    fn cbs_of(chunks: &[(Vec<Shape>, ObjId)]) -> Vec<ObjId> {
        chunks.iter().map(|(_, cb)| *cb).collect()
    }

    /// Each record [`splitting`]'s sinks were handed since they were last emptied, by its type and
    /// status, emptied.
    fn answered(posted: &Sunk<(u32, u64, u32)>) -> Vec<(u32, u32)> {
        let taken = std::mem::take(&mut *posted.lock());
        taken
            .into_iter()
            .map(|(kind, _, status)| (kind, status))
            .collect()
    }

    /// [`splitting`], its buffers of any length dealt to two threads, a chunk each encoder, the
    /// helper's started before this thread walks its own.
    fn dealing() -> Split {
        let split = splitting();
        split.dec.set_deal_min_bytes(Some(0));
        split.dec.set_deal_chunk_bytes(Some(0));
        split.dec.set_helper_first(true);
        split
    }

    /// Where an op stream is cut into chunks dealt to two threads: at the end of each encoder that
    /// ends the least bytes of a chunk after the last cut, into the most chunks at most, where it
    /// holds 8 render or blit encoders and the least bytes at least; never where it holds an
    /// event's signal or wait, a compute or video encoder, a rasterization rate map's pass, an
    /// indirect command buffer's command or a residency of resources.
    #[test]
    fn deal_cuts_only_streams_of_render_and_blit_encoders() {
        let pass = render_ops().len();
        let eight = passes(8);
        let at = |passes: &[usize]| Some(passes.iter().map(|n| n * pass).collect::<Vec<_>>());
        assert_eq!(
            deal(&eight, 0, 0, 64),
            at(&[1, 2, 3, 4, 5, 6, 7]),
            "each encoder"
        );
        assert_eq!(
            deal(&eight, eight.len(), 2 * pass, 64),
            at(&[2, 4, 6]),
            "chunks of two passes"
        );
        assert_eq!(
            deal(&eight, 0, 3 * pass, 64),
            at(&[3, 6]),
            "the last chunk shorter"
        );
        assert_eq!(deal(&eight, 0, 0, 3), at(&[3, 6]), "3 chunks at most");
        assert_eq!(deal(&eight, 0, 0, 2), at(&[4]), "2 chunks at most");
        assert_eq!(deal(&eight, 0, 0, 1), None, "no room for a second chunk");
        assert_eq!(deal(&eight, 0, 8 * pass, 64), None, "one chunk");
        assert_eq!(
            deal(&eight, eight.len() + 1, 0, 64),
            None,
            "under the least bytes"
        );
        assert_eq!(deal(&passes(7), 0, 0, 64), None, "7 encoders");
        let blits = (0..8).map(|_| blit_ops()).collect::<Vec<_>>().concat();
        assert_eq!(
            deal(&blits, 0, 4 * blit_ops().len(), 64),
            Some(vec![4 * blit_ops().len()]),
            "blit encoders"
        );
        let with = |extra: Vec<u8>| [passes(4), extra, passes(4)].concat();
        let compute = [
            op(RGFX_OP_ENCODER_BEGIN_COMPUTE, &[]),
            op(RGFX_OP_ENCODER_END, &[]),
        ]
        .concat();
        assert_eq!(deal(&with(compute), 0, 0, 64), None, "a compute encoder");
        assert_eq!(
            deal(&with(event_op(RGFX_OP_SIGNAL_EVENT, 7, 3)), 0, 0, 64),
            None,
            "a signal"
        );
        assert_eq!(
            deal(&with(event_op(RGFX_OP_WAIT_EVENT, 7, 3)), 0, 0, 64),
            None,
            "a wait"
        );
        let mut short = eight.clone();
        short.truncate(eight.len() - 4);
        assert_eq!(deal(&short, 0, 0, 64), None, "a stream that does not parse");
    }

    /// The ops of the guest's whose commands keep a command buffer whole for `reason`, each with
    /// the encoder it lies in, which [`deal`] keeps out of a buffer dealt to two threads
    /// ([`kept_whole`]): a match over every reason the platform keeps one whole for
    /// ([`KeptWhole`]), so a new reason does not build until its ops are named here.
    fn keeping_whole(reason: KeptWhole) -> Vec<Vec<u8>> {
        use reims_vgpu_graphics::ops::render::{
            RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER, RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT,
            RGFX_OP_RENDER_USE_RESOURCES, RGFX_OP_RENDER_USE_RESOURCES_STAGES,
        };
        let compute = [
            op(RGFX_OP_ENCODER_BEGIN_COMPUTE, &[]),
            op(RGFX_OP_ENCODER_END, &[]),
        ]
        .concat();
        let render_with = |tag: u16| {
            [
                op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body()),
                op(tag, &[0; 32]),
                op(RGFX_OP_ENCODER_END, &[]),
            ]
            .concat()
        };
        let blit_with = |tag: u16| {
            [
                op(RGFX_OP_ENCODER_BEGIN_BLIT, &[]),
                op(tag, &[0; 32]),
                op(RGFX_OP_ENCODER_END, &[]),
            ]
            .concat()
        };
        let rate_map = {
            let mut body = begin_render_body();
            let flags = RGFX_PASS_FLAG_RT_SIZE | RGFX_PASS_FLAG_RATE_MAP;
            body[4..8].copy_from_slice(&flags.to_le_bytes());
            [
                op(RGFX_OP_ENCODER_BEGIN_RENDER, &body),
                op(RGFX_OP_ENCODER_END, &[]),
            ]
            .concat()
        };
        let executions = [
            render_with(RGFX_OP_RENDER_EXECUTE_COMMANDS_IN_BUFFER),
            render_with(RGFX_OP_RENDER_EXECUTE_COMMANDS_INDIRECT),
            compute.clone(),
        ];
        match reason {
            // The chunks are cut at an encoder's end, and a walk ends each encoder it opens.
            KeptWhole::OpenEncoder => vec![],
            // No op of the guest's has the walk send `setProtectionOptions:` or
            // `encodeSignalEventScheduled:value:`
            // (`deal_keeps_out_every_op_that_keeps_a_buffer_whole`).
            KeptWhole::ProtectionOptions | KeptWhole::ScheduledSignal => vec![],
            KeptWhole::Execution => executions.to_vec(),
            KeptWhole::Progress => [
                vec![rate_map],
                executions.to_vec(),
                vec![
                    render_with(RGFX_OP_RENDER_USE_RESOURCES),
                    render_with(RGFX_OP_RENDER_USE_RESOURCES_STAGES),
                    blit_with(RGFX_OP_BLIT_RESET_COMMANDS_IN_BUFFER),
                    blit_with(RGFX_OP_BLIT_COPY_INDIRECT_COMMAND_BUFFER),
                    blit_with(RGFX_OP_BLIT_OPTIMIZE_INDIRECT_COMMAND_BUFFER),
                ],
            ]
            .concat(),
            KeptWhole::Predication => vec![compute],
            // The walk commits a chunk's command buffer once, as a part or as the whole.
            KeptWhole::Committed => vec![],
        }
    }

    /// Every reason the platform keeps a command buffer whole for, but how its GPU fails it, is
    /// set only by ops [`deal`] keeps out of a buffer dealt to two threads, so [`kept_whole`] is
    /// not reached: each op [`keeping_whole`] names makes a stream it would cut one it does not,
    /// wherever it lies, and no walk of an op sends the two calls whose reasons name no op.
    #[test]
    fn deal_keeps_out_every_op_that_keeps_a_buffer_whole() {
        assert!(deal(&passes(8), 0, 0, 64).is_some());
        for reason in KeptWhole::ALL {
            for ops in keeping_whole(reason) {
                for before in [0, 1, 4, 8] {
                    let stream = [passes(before), ops.clone(), passes(8 - before)].concat();
                    assert_eq!(deal(&stream, 0, 0, 64), None, "{reason:?} after {before}");
                }
            }
        }
        let walk = [
            include_str!("cmdbuf.rs"),
            include_str!("render.rs"),
            include_str!("blit.rs"),
            include_str!("compute.rs"),
            include_str!("video.rs"),
            include_str!("managed.rs"),
            include_str!("refused.rs"),
            include_str!("surface_cache.rs"),
            include_str!("icb.rs"),
            include_str!("rate_map.rs"),
            include_str!("argument.rs"),
            include_str!("mod.rs"),
        ]
        .concat();
        for call in ["set_protection_options", "encode_signal_when_scheduled"] {
            assert!(
                !walk.contains(&format!(".{call}(")),
                "the decoder sends {call}"
            );
        }
    }

    /// A dealt command buffer the platform keeps whole at a chunk's end for a reason of its own
    /// ([`PartCut::Whole`]) fails at its stand-in, counted as a degradation ([`KEPT_WHOLE`]), no
    /// chunk walked again, the chunks before it committed and nothing at or after it, and its
    /// context's gate let go; the line is logged for the first buffer kept whole for its reason
    /// alone, each counted.
    #[test]
    fn a_dealt_buffer_kept_whole_at_a_chunk_s_end_is_counted() {
        let Split {
            p, mut dec, posted, ..
        } = dealing();
        let device = p.device();
        let counted = || {
            (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
                .find(|(code, _)| code == KEPT_WHOLE)
                .map_or(0, |(_, n)| n)
        };
        let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
        let before = counted();
        // Kept whole at chunk 0's end, then at chunk 3's.
        for (n, (seqno, skip)) in [(395, 0), (396, 3)].into_iter().enumerate() {
            let whole = Step::value(PartCut::Whole(KeptWhole::Progress));
            p.script().after("commitPart", skip, whole);
            let begun = dec.walks_begun();
            let mark = p.call_log().mark();
            let message = command_buffer_message(seqno, 120, &numbered(8));
            assert!(refusing(|| dec.decode(&device, &message)).is_err());
            assert_eq!(p.script().pending(), 0, "{seqno}: its script ran");
            assert_eq!(counted(), before + n as u64 + 1, "{seqno}: counted");
            assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
            let dealt = dealt_chunks(&walk_calls(&p, mark));
            assert_eq!(
                p.world().committed(),
                cbs_of(&dealt)[..skip as usize],
                "{seqno}: the chunks before committed"
            );
            assert!(dec.stand_in_for_failed(&device, &message).unwrap());
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                posted.lock().split_off(0),
                [(RGFX_EVT_CB_COMPLETE, seqno, RGFX_CB_INTERNAL)]
            );
            assert!(dec.shared.split_gate.try_split(0).is_some());
        }
        let lines = (logs.take().into_iter())
            .filter(|(level, target, line)| {
                *level == log::Level::Error
                    && target == "reims_vgpu::graphics"
                    && line.contains("was kept whole at a chunk's end (progress)")
            })
            .count();
        assert_eq!(lines, 1, "logged for the first alone");
    }

    /// A dealt command buffer that the platform makes no command buffer for one of its chunks of
    /// ([`CommandBuffer::part`]), each made before any chunk is walked, is walked whole, as one
    /// not dealt: each op once, here, the command buffers made let go of unused, and its parts and
    /// records as a whole walk's. So is one whose queue's room runs out while its chunks' are
    /// made, as other walks and begins make command buffers on it: no make past the room, and no
    /// cut either, its queue full.
    #[test]
    fn a_dealt_buffer_whose_chunk_gets_no_command_buffer_or_room_is_walked_whole() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        let ops = numbered(8);
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &command_buffer_message(375, 105, &ops))
                .unwrap(),
            Ok(())
        );
        let whole = whole_shapes(&walk_calls(&p, mark));
        let parts_whole = p.world().complete_all(CompleteOn::Caller);
        let answered_whole = answered(&posted);

        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        p.script().after("part", 3, Step::Nil);
        let (begun, mark) = (dec.walks_begun(), p.call_log().mark());
        assert_eq!(
            dec.decode(&device, &command_buffer_message(376, 106, &ops))
                .unwrap(),
            Ok(())
        );
        assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
        assert_eq!(
            whole_shapes(&walk_calls(&p, mark)),
            whole,
            "walked whole, here"
        );
        let sels = p.call_log().sels_since(mark);
        assert_eq!(sels.iter().filter(|s| **s == "part").count(), 4);
        assert!(!sels.contains(&"commitPart"), "cut as a whole walk is");
        assert_eq!(
            p.world().complete_all(CompleteOn::Caller),
            parts_whole,
            "its parts"
        );
        assert_eq!(answered(&posted), answered_whole);

        // Its third chunk's make held while other command buffers take the queue's room.
        let in_flight = Arc::clone(&dec.queue_counts(1).unwrap().in_flight);
        let held = HeldCall::new();
        p.script().after("part", 2, Step::Hold(held.clone()));
        let (begun, mark) = (dec.walks_begun(), p.call_log().mark());
        let walked = std::thread::scope(|s| {
            let filler = s.spawn(|| {
                assert!(held.reached(Duration::from_secs(10)), "a make held");
                in_flight.fetch_add(TAKE_IN_FLIGHT_MAX, Ordering::AcqRel);
                held.open();
            });
            let walked = dec.decode(&device, &command_buffer_message(377, 107, &ops));
            filler.join().unwrap();
            walked.unwrap()
        });
        in_flight.fetch_sub(TAKE_IN_FLIGHT_MAX, Ordering::AcqRel);
        assert_eq!(walked, Ok(()));
        assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
        assert_eq!(
            whole_shapes(&walk_calls(&p, mark)),
            whole,
            "walked whole, here"
        );
        let sels = p.call_log().sels_since(mark);
        assert_eq!(sels.iter().filter(|s| **s == "part").count(), 3);
        assert!(
            !sels.contains(&"commitPart") && !sels.contains(&"split"),
            "not cut, its queue full"
        );
        assert_eq!(p.world().complete_all(CompleteOn::Caller), 1, "one commit");
        assert_eq!(answered(&posted), answered_whole);
    }

    /// [`deal`] answers a stream under the least bytes from its length alone, so it reads none of
    /// it: `None` whatever it holds, the ops of a stream it deals at its own length as much as
    /// bytes no walk parses. At the least bytes it reads the stream, and its answer follows what
    /// the stream holds. What that costs is [`deal_costs`]'s measurement.
    #[test]
    fn deal_reads_nothing_of_a_short_stream() {
        let dealt = passes(107);
        let least = dealt.len();
        let unparsed = vec![0xff; least];
        let deal_at =
            |ops: &[u8], min_bytes| deal(ops, min_bytes, DEAL_CHUNK_BYTES, TAKE_IN_FLIGHT_MAX);
        assert!(deal_at(&dealt, least).is_some(), "read: it deals");
        assert_eq!(deal_at(&unparsed, least), None, "read: it does not parse");
        for (ops, what) in [
            (&dealt[..], "it deals"),
            (&unparsed[..], "it does not parse"),
        ] {
            assert_eq!(deal_at(ops, least + 1), None, "a byte short, where {what}");
        }
        assert_eq!(deal_at(&[], least), None, "no ops");
    }

    /// What finding the cuts costs ([`deal`]): nothing measurable on a stream under the least
    /// bytes, which it does not read ([`deal_reads_nothing_of_a_short_stream`]); a read of each
    /// op's header on a long one. Printed: run it by name with `--no-capture`, in a release build.
    #[test]
    #[ignore = "a measurement, run by hand in a release build with --no-capture"]
    #[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
    fn deal_costs() {
        use std::hint::black_box;
        let short = passes(2);
        let long = passes(107);
        let each = |ops: &[u8], runs: u32| {
            let start = std::time::Instant::now();
            for _ in 0..runs {
                black_box(deal(
                    black_box(ops),
                    DEAL_MIN_BYTES,
                    DEAL_CHUNK_BYTES,
                    TAKE_IN_FLIGHT_MAX,
                ));
            }
            start.elapsed().as_nanos() as f64 / f64::from(runs)
        };
        assert!(short.len() < DEAL_MIN_BYTES && long.len() >= DEAL_MIN_BYTES);
        let cuts = deal(&long, DEAL_MIN_BYTES, DEAL_CHUNK_BYTES, TAKE_IN_FLIGHT_MAX);
        assert!(cuts.is_some());
        let (short_ns, long_ns) = (each(&short, 1_000_000), each(&long, 1_000));
        let ops = op_bodies(&long).count();
        eprintln!(
            "deal: {short_ns:.2} ns a stream of {} bytes; {long_ns:.0} ns one of {} bytes, {ops} \
             ops, in {} chunks",
            short.len(),
            long.len(),
            cuts.map_or(0, |c| c.len() + 1)
        );
    }

    /// A command buffer dealt to two threads answers as one walked whole, whichever thread is
    /// ahead: each op walked once, its even chunks here and its odd ones on a helper; each chunk's
    /// command buffer committed as a part in their order, the last chunk's the last; its walk
    /// begun once; one `CB_SCHEDULED` and one `CB_COMPLETE`, after its last part; and the error
    /// of a part that failed.
    #[test]
    fn a_dealt_buffer_answers_as_walked_whole() {
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        let ops = numbered(8);
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &command_buffer_message(360, 90, &ops))
                .unwrap(),
            Ok(())
        );
        let whole = whole_shapes(&walk_calls(&p, mark));
        p.world().complete_all(CompleteOn::Caller);
        let answered_whole = answered(&posted);
        assert_eq!(
            answered_whole,
            [
                (RGFX_EVT_CB_SCHEDULED, RGFX_CB_OK),
                (RGFX_EVT_CB_COMPLETE, RGFX_CB_OK)
            ]
        );

        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        dec.set_helper_first(true);
        // The helper started first, the two walks then in whatever order they run; then the
        // helper's walk held at its first op until this thread waits for one of its chunks.
        for (seqno, held) in [(361, false), (362, true)] {
            let (clock, decoders) = (Arc::clone(&p), dec.decoders());
            let waits = dec.deal_waits();
            dec.set_walk_hook(Some(Arc::new(move |_| {
                let helper = std::thread::current().name() == Some("reims-vgpu.graphics-helper");
                if helper && held {
                    // Bounded: a walk that never waits ends the hold, and the check below fails.
                    crate::graphics::tests::eventually(|| decoders.deal_waits() > waits);
                }
                clock.virtual_clock().advance(SPLIT_AFTER);
            })));
            let (begun, mark) = (dec.walks_begun(), p.call_log().mark());
            assert_eq!(
                dec.decode(&device, &command_buffer_message(seqno, 91, &ops))
                    .unwrap(),
                Ok(())
            );
            assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
            let calls = walk_calls(&p, mark);
            let dealt = dealt_chunks(&calls);
            assert_eq!(dealt.len(), 8, "a chunk a pass");
            assert_eq!(
                shapes(&dealt),
                whole,
                "each op walked once, its chunks in their order"
            );
            assert_eq!(
                calls.iter().filter(|(helper, ..)| *helper).count(),
                calls.len() / 2,
                "the odd chunks on a helper"
            );
            if held {
                assert!(
                    dec.deal_waits() > waits,
                    "this thread waited for the helper"
                );
            }
            let sels = p.call_log().sels_since(mark);
            assert_eq!(sels.iter().filter(|s| **s == "part").count(), 7);
            assert_eq!(sels.iter().filter(|s| **s == "commitPart").count(), 7);
            assert!(!sels.contains(&"split"), "no cut inside a chunk");
            let parts = p.world().committed();
            assert_eq!(parts, cbs_of(&dealt), "each chunk a part, in their order");
            assert_eq!(
                marks(&p, &parts),
                [true, true, true, true, true, true, true, false],
                "every part but the last joins the next one's completion"
            );
            assert_eq!(
                answered(&posted),
                [(RGFX_EVT_CB_SCHEDULED, RGFX_CB_OK)],
                "at its take"
            );
            for part in &parts[..7] {
                let ok = CommandBufferStatus::COMPLETED;
                assert!(p.world().complete(*part, ok, None, CompleteOn::Caller));
            }
            assert!(posted.lock().is_empty(), "completed before its last part");
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(answered(&posted), [(RGFX_EVT_CB_COMPLETE, RGFX_CB_OK)]);
        }

        // A part between the first and the last fails: the buffer's error, as a part's of a
        // whole walk.
        assert_eq!(
            dec.decode(&device, &command_buffer_message(363, 92, &ops))
                .unwrap(),
            Ok(())
        );
        let parts = p.world().committed();
        assert_eq!(parts.len(), 8);
        let error = NsErrorText {
            domain: "MTLCommandBufferErrorDomain".to_owned(),
            code: 3,
            description: "Page Fault".to_owned(),
        };
        let failed = CommandBufferStatus::ERROR;
        assert!(
            p.world()
                .complete(parts[3], failed, Some(error), CompleteOn::Caller)
        );
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            answered(&posted),
            [
                (RGFX_EVT_CB_SCHEDULED, RGFX_CB_OK),
                (
                    RGFX_EVT_CB_COMPLETE,
                    reims_vgpu_graphics::wire::RGFX_CB_PAGE_FAULT
                )
            ]
        );
    }

    /// A command buffer's names for its work, which the platform keeps for the line of a lost
    /// device that loses it in flight ([`WorkNames`]): its object id, and the first compute and
    /// the first render pipeline state its encoders set, in their order, whether it is walked
    /// whole, cut in parts or dealt to two threads, a helper walking its chunks first; each of its
    /// parts holds the same names. A pipeline state set to nil is none, and a buffer that sets
    /// none names none.
    #[test]
    fn a_command_buffer_names_its_first_pipeline_states_in_its_encoders_order() {
        use reims_vgpu_graphics::ops::compute::{
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RgfxComputeOpSetComputePipelineState,
        };
        use reims_vgpu_vulkan::core::platform::resource::Library as _;
        use reims_vgpu_vulkan::core::platform::values::RenderPipelineDesc;
        // Compute pipeline states 70 and 71, and render ones 80 and 81.
        let pipelines = |p: &FakePlatform, dec: &mut GraphicsDecoder<FakePlatform>| {
            let device = p.device();
            let library = device.new_library(&[0; 8]).unwrap().unwrap().unwrap();
            let f = library.new_function(OnFault::Die, "f").unwrap().unwrap();
            for id in [70, 71] {
                let state = device.new_compute_pipeline_state_for_function(Some(&f));
                let state = ObjHandle::ComputePipeline(state.unwrap().unwrap());
                assert_eq!(dec.insert(id, state).unwrap(), Ok(()));
            }
            for id in [80, 81] {
                let state = device.new_render_pipeline_state(&RenderPipelineDesc {
                    label: None,
                    vertex: &f,
                    fragment: &f,
                    color0_pixel_format: 80,
                    resource_index: None,
                });
                let state = ObjHandle::RenderPipeline(state.unwrap().unwrap());
                assert_eq!(dec.insert(id, state).unwrap(), Ok(()));
            }
        };
        let compute = |pipeline_obj_id: u64| {
            let pass = RgfxComputePassDesc::default();
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_COMPUTE, pass.as_bytes());
            let set = RgfxComputeOpSetComputePipelineState { pipeline_obj_id };
            ops.extend(op(
                RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
                set.as_bytes(),
            ));
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
            ops
        };
        let render = |pipeline_obj_id: u64| {
            let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin_render_body());
            let set = RgfxOpSetRenderPipelineState { pipeline_obj_id };
            ops.extend(op(RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, set.as_bytes()));
            ops.extend(draw_ops(2, 0, 0));
            ops.extend(op(RGFX_OP_ENCODER_END, &[]));
            ops
        };
        // The parts committed, the names they share, then completed.
        let named = |p: &FakePlatform| {
            let parts = p.world().committed();
            let names: Vec<Arc<WorkNames>> = (parts.iter())
                .map(|part| p.world().work_names(*part).expect("named"))
                .collect();
            assert!(
                names.iter().all(|n| Arc::ptr_eq(n, &names[0])),
                "its parts share its names"
            );
            p.world().complete_all(CompleteOn::Caller);
            (parts.len(), names[0].cb(), names[0].firsts())
        };
        let ops = [render(0), compute(71), compute(70), render(81), render(80)].concat();

        let (p, mut dec) = setup();
        pipelines(&p, &mut dec);
        let device = p.device();
        let message = command_buffer_message(400, 40, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(named(&p), (1, 400, (71, 81)), "walked whole");
        let message = command_buffer_message(401, 41, &[render(0), render(0)].concat());
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(named(&p), (1, 401, (0, 0)), "none set");

        let Split { p, mut dec, .. } = splitting();
        pipelines(&p, &mut dec);
        let device = p.device();
        let message = command_buffer_message(402, 42, &ops);
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        assert_eq!(named(&p), (5, 402, (71, 81)), "a part an encoder");

        // A chunk a pass, the helper's, the odd ones, walked first: the first render pipeline
        // state set is the second chunk's, though this thread may set the third's first.
        let Split { p, mut dec, .. } = dealing();
        pipelines(&p, &mut dec);
        let device = p.device();
        let mut passes = vec![render(0), render(81), render(80)];
        passes.resize(8, render(0));
        let message = command_buffer_message(403, 43, &passes.concat());
        let mark = p.call_log().mark();
        assert_eq!(dec.decode(&device, &message).unwrap(), Ok(()));
        let helped = walk_calls(&p, mark)
            .iter()
            .filter(|(helper, ..)| *helper)
            .count();
        assert!(helped > 0, "the helper walked chunks of it");
        assert_eq!(named(&p), (8, 403, (0, 81)), "dealt");
    }

    /// A dealt buffer's chunks go to the two threads the longest first, each to the thread dealt
    /// fewer bytes so far, this thread on a tie, the first chunk always this thread's: chunks of
    /// one length in turn, the odd ones to the helper; a longest chunk anywhere after the first
    /// to the helper, which takes the others only while it holds fewer bytes.
    #[test]
    fn helper_chunks_deals_the_longest_first_to_the_thread_dealt_fewer_bytes() {
        let chunks_of = |lens: &[usize]| -> Vec<Range<usize>> {
            let mut at = 0;
            (lens.iter())
                .map(|len| {
                    at += len;
                    at - len..at
                })
                .collect()
        };
        let (f, t) = (false, true);
        for (lens, helper) in [
            (&[4, 4, 4, 4, 4, 4][..], &[f, t, f, t, f, t][..]),
            (&[4, 4, 4, 4, 4], &[f, t, f, t, f]),
            (&[5, 7], &[f, t]),
            // Heaven's frame buffer's chunks, in KiB (m3-max-32395).
            (&[161, 135, 471, 51, 1], &[f, f, t, f, f]),
            (&[10, 10, 10, 100], &[f, f, f, t]),
            (&[100, 10, 10, 10], &[f, t, t, t]),
            (&[30, 10, 50, 40, 20], &[f, f, t, f, t]),
        ] {
            assert_eq!(helper_chunks(&chunks_of(lens)), helper, "{lens:?}");
        }
    }

    /// A dealt buffer whose third pass is far the longest has the helper walk that pass alone,
    /// from the walk's start, and this thread every other chunk: each op walked once, each
    /// chunk a part committed in their order, and the buffer's records as a whole walk's.
    #[test]
    fn a_dealt_buffer_s_longest_chunk_goes_to_the_helper_and_the_order_stays() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        let draws = |pass: u64, n: u64| -> Vec<u8> {
            (0..n).flat_map(|d| draw_ops(2, 16 * pass, d)).collect()
        };
        let long = 2;
        let ops: Vec<u8> = (0..8)
            .map(|i| pass_of(&draws(i, if i == long { 40 } else { 2 })))
            .collect::<Vec<_>>()
            .concat();
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &command_buffer_message(365, 95, &ops))
                .unwrap(),
            Ok(())
        );
        let whole = whole_shapes(&walk_calls(&p, mark));
        p.world().complete_all(CompleteOn::Caller);
        let answered_whole = answered(&posted);

        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        dec.set_helper_first(true);
        let (begun, mark) = (dec.walks_begun(), p.call_log().mark());
        assert_eq!(
            dec.decode(&device, &command_buffer_message(366, 96, &ops))
                .unwrap(),
            Ok(())
        );
        assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
        // Each call to the chunk whose command buffer its thread's last encoder was made on.
        let calls = walk_calls(&p, mark);
        let mut chunk_of: Vec<(ObjId, bool, Vec<Shape>)> = Vec::new();
        let mut last = [None::<usize>, None];
        for (helper, shape, made) in calls {
            let at = match made {
                Some(cb) => {
                    chunk_of.push((cb, helper, Vec::new()));
                    chunk_of.len() - 1
                }
                None => last[usize::from(helper)].expect("a call in an encoder"),
            };
            last[usize::from(helper)] = Some(at);
            chunk_of[at].2.push(shape);
        }
        let parts = p.world().committed();
        assert_eq!(parts.len(), 8, "a chunk a pass, each a part");
        let in_order: Vec<&(ObjId, bool, Vec<Shape>)> = (parts.iter())
            .map(|cb| {
                chunk_of
                    .iter()
                    .find(|c| c.0 == *cb)
                    .expect("a chunk walked")
            })
            .collect();
        assert_eq!(
            in_order.iter().map(|c| c.1).collect::<Vec<_>>(),
            [false, false, true, false, false, false, false, false],
            "the longest chunk the helper's alone"
        );
        assert_eq!(
            in_order
                .iter()
                .flat_map(|c| c.2.iter().cloned())
                .collect::<Vec<_>>(),
            whole,
            "each op walked once, its chunks committed in their order"
        );
        let sels = p.call_log().sels_since(mark);
        assert_eq!(sels.iter().filter(|s| **s == "commitPart").count(), 7);
        assert_eq!(
            answered(&posted),
            [(RGFX_EVT_CB_SCHEDULED, RGFX_CB_OK)],
            "at its take"
        );
        p.world().complete_all(CompleteOn::Caller);
        assert_eq!(
            answered(&posted),
            answered_whole[1..],
            "completed as a whole walk"
        );
    }

    /// The chunks of a dealt buffer no helper has taken by the end of this thread's first are
    /// taken back and walked here, as a whole walk goes on: each op once, the walk begun once,
    /// the command buffers made for the chunks let go of unused, and the buffer's parts and
    /// records as a whole walk's.
    #[test]
    fn a_dealt_buffer_no_helper_takes_is_walked_here() {
        let Split {
            p, mut dec, posted, ..
        } = splitting();
        let device = p.device();
        let ops = numbered(8);
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &command_buffer_message(370, 100, &ops))
                .unwrap(),
            Ok(())
        );
        let whole = whole_shapes(&walk_calls(&p, mark));
        let parts_whole = p.world().complete_all(CompleteOn::Caller);
        let answered_whole = answered(&posted);

        dec.set_deal_min_bytes(Some(0));
        dec.set_deal_chunk_bytes(Some(0));
        let held = dec.shared.helpers.hold_all();
        let (begun, mark) = (dec.walks_begun(), p.call_log().mark());
        assert_eq!(
            dec.decode(&device, &command_buffer_message(371, 101, &ops))
                .unwrap(),
            Ok(())
        );
        drop(held);
        assert_eq!(dec.walks_begun(), begun + 1, "its walk begun once");
        assert_eq!(
            whole_shapes(&walk_calls(&p, mark)),
            whole,
            "each op walked once, in its order"
        );
        let sels = p.call_log().sels_since(mark);
        assert_eq!(
            sels.iter().filter(|s| **s == "part").count(),
            7,
            "every chunk's but the first's"
        );
        assert!(!sels.contains(&"commitPart"), "cut as a whole walk is");
        assert_eq!(
            p.world().complete_all(CompleteOn::Caller),
            parts_whole,
            "its parts"
        );
        assert_eq!(answered(&posted), answered_whole);
    }

    /// A part of a dealt buffer that fails from one of its encoders on keeps the buffer whole
    /// from it, as a whole walk's cut would, whichever thread walked it: every later chunk's
    /// command buffer folded into it in their order, never committed, the parts before it
    /// committed, each op walked once, and the buffer's records and its error as a whole walk's.
    #[test]
    fn a_dealt_part_that_fails_from_an_encoder_on_takes_in_every_later_chunk() {
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsErrorText};
        let Split {
            p,
            mut dec,
            posted,
            records,
        } = dealing();
        let device = p.device();
        let ops = numbered(8);
        dec.set_deal_min_bytes(None);
        let mark = p.call_log().mark();
        assert_eq!(
            dec.decode(&device, &command_buffer_message(379, 109, &ops))
                .unwrap(),
            Ok(())
        );
        let whole = whole_shapes(&walk_calls(&p, mark));
        p.world().complete_all(CompleteOn::Caller);
        posted.lock().clear();
        dec.set_deal_min_bytes(Some(0));
        // The chunk whose part fails (its index among the commits): the first, kept whole with
        // no part out; one of this thread's; one of the helper's; and the last but one, this
        // thread's last, whose command buffer is the walk's.
        for (seqno, failing) in [(380, 0), (381, 2), (382, 3), (383, 6)] {
            p.script()
                .after("commitPart", failing, Step::value(PartCut::Fails));
            let mark = p.call_log().mark();
            assert_eq!(
                dec.decode(&device, &command_buffer_message(seqno, 110, &ops))
                    .unwrap(),
                Ok(())
            );
            let dealt = dealt_chunks(&walk_calls(&p, mark));
            assert_eq!(shapes(&dealt), whole, "{seqno}: each op walked once");
            let cbs = cbs_of(&dealt);
            let sels = p.call_log().sels_since(mark);
            assert_eq!(
                sels.iter().filter(|s| **s == "commitPart").count(),
                failing as usize + 1,
                "{seqno}: no commit tried after the part kept whole"
            );
            let folds = p.call_log().calls_to("fold");
            let folded: Vec<_> = (folds.iter().rev().take(7 - failing as usize).rev())
                .map(|c| c.args[0].as_obj().unwrap())
                .collect();
            assert_eq!(
                folded,
                cbs[failing as usize + 1..],
                "{seqno}: every later chunk folded, in their order"
            );
            let into = folds.iter().rev().take(7 - failing as usize);
            assert!(
                into.into_iter().all(|c| c.obj == cbs[failing as usize]),
                "{seqno}: into the part kept whole"
            );
            let parts = p.world().committed();
            assert_eq!(
                parts,
                cbs[..=failing as usize],
                "{seqno}: the parts before it, then the whole"
            );
            let mut marked = vec![true; failing as usize];
            marked.push(false);
            assert_eq!(
                marks(&p, &parts),
                marked,
                "{seqno}: the parts before it marked, the whole not"
            );
            records.lock().clear();
            let error = NsErrorText {
                domain: "MTLCommandBufferErrorDomain".to_owned(),
                code: 3,
                description: "Page Fault".to_owned(),
            };
            let last = *parts.last().unwrap();
            assert!(p.world().set_gpu_times(last, 10, 30));
            let failed = CommandBufferStatus::ERROR;
            assert!(
                p.world()
                    .complete(last, failed, Some(error), CompleteOn::Caller)
            );
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                answered(&posted),
                [
                    (RGFX_EVT_CB_SCHEDULED, RGFX_CB_OK),
                    (
                        RGFX_EVT_CB_COMPLETE,
                        reims_vgpu_graphics::wire::RGFX_CB_PAGE_FAULT
                    )
                ],
                "{seqno}"
            );
            let r = records.lock()[0];
            if failing == 0 {
                assert_eq!(
                    (r.cb_obj_id, r.gpu_start_ns, r.gpu_end_ns),
                    (seqno, 10, 30),
                    "its one part's GPU times, counted once"
                );
            }
            assert!(
                dec.shared.split_gate.try_split(0).is_some(),
                "{seqno}: the gate let go"
            );
        }
    }

    /// A dealt walk that faults in a chunk has its failure at its stand-in's completion alone, as
    /// one whose walk faults whole: the fault of the first chunk in their order that faults,
    /// whichever thread found its own first, the parts before it committed, nothing at or after
    /// it committed, and its context's gate let go.
    #[test]
    fn a_dealt_walk_that_faults_answers_its_first_faulting_chunk() {
        use reims_vgpu_graphics::ops::blit::RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER;
        let Split {
            p, mut dec, posted, ..
        } = dealing();
        let device = p.device();
        // A render op and a blit op with no encoder open fault the walk, each its own way.
        let render = op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type: 3,
                vertex_start: 0,
                vertex_count: 3,
            }
            .as_bytes(),
        );
        let blit = op(RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER, &[0; 40]);
        // Chunk `i` is what follows pass `i - 1`'s end, and pass `i`: the stream with `bad` at
        // the start of each of `chunks`.
        let with = |faults: &[(u64, &[u8])]| {
            (0..8u64)
                .map(|i| {
                    let bad: Vec<u8> = (faults.iter())
                        .filter(|(at, _)| *at == i)
                        .flat_map(|(_, b)| b.iter().copied())
                        .collect();
                    [bad, pass_of(&draw_ops(2, 16 * i, i))].concat()
                })
                .collect::<Vec<_>>()
                .concat()
        };
        // The stream; the chunk that faults first, and its group: the helper's, this thread's
        // later chunk walked; the helper's, found as this thread found its own later one; this
        // thread's, the helper's later; and this thread's first chunk. A chunk with a fault op is
        // longer than the others, so it is dealt first ([`helper_chunks`]): the longest to the
        // helper, the next to this thread.
        let cases: [(u64, Vec<u8>, usize, &str); 4] = [
            (390, with(&[(3, &render[..])]), 3, "render"),
            (391, with(&[(2, &blit[..]), (5, &render[..])]), 2, "blit"),
            (392, with(&[(3, &render[..]), (4, &blit[..])]), 3, "render"),
            (393, with(&[(0, &blit[..])]), 0, "blit"),
        ];
        for (cb, ops, first, group) in cases {
            assert!(deal(&ops, 0, 0, 64).is_some());
            let mark = p.call_log().mark();
            let message = command_buffer_message(cb, cb - 300, &ops);
            let fault = refusing(|| dec.decode(&device, &message)).unwrap_err();
            let detail = fault.detail().unwrap_or_default().to_owned();
            assert!(
                detail.contains(&format!("a {group} op")),
                "{cb}: the first chunk's fault ({detail})"
            );
            let cbs = cbs_of(&dealt_chunks(&walk_calls(&p, mark)));
            let parts = p.world().committed();
            assert!(
                parts.len() <= first,
                "{cb}: nothing at or after it committed"
            );
            assert_eq!(parts, cbs[..parts.len()], "{cb}: the parts before it");
            assert!(dec.stand_in_for_failed(&device, &message).unwrap());
            p.world().complete_all(CompleteOn::Caller);
            assert_eq!(
                *posted.lock(),
                [(RGFX_EVT_CB_COMPLETE, cb, RGFX_CB_INTERNAL)]
            );
            posted.lock().clear();
            assert!(
                dec.shared.split_gate.try_split(0).is_some(),
                "{cb}: the gate held after the walk"
            );
        }
    }

    /// A dealt walk begun in an entry before the process's exit began ends after it, its helper
    /// starting its chunks once the exit closed the gate and walking all of them, so the entry,
    /// and the exit waiting for it, end: the helper's walk is part of the entry's, with no step of
    /// the exit gate of its own ([`crate::entry::run_within`]). In a child, which exits.
    #[test]
    fn a_dealt_walk_ends_once_the_process_s_exit_began() {
        const TEST: &str =
            "graphics::decode::cmdbuf::tests::a_dealt_walk_ends_once_the_process_s_exit_began";
        const CHILD: &str = "RVG_DEAL_EXIT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(out.status.success(), "{err}");
            assert!(
                err.contains("dealt walk ended: Ok(Ok(())), 4 of 8 chunks on the helper"),
                "{err}"
            );
            return;
        }
        let Split { p, dec, .. } = dealing();
        let in_step = Arc::new(AtomicBool::new(false));
        let walker = {
            let in_step = Arc::clone(&in_step);
            std::thread::Builder::new()
                .name("walker".to_owned())
                .spawn(move || {
                    crate::entry::run(|_| {
                        in_step.store(true, Ordering::SeqCst);
                        while !reims_vgpu_vulkan::exit_gate::closed() {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        let mark = p.call_log().mark();
                        let device = p.device();
                        let mut dec = dec;
                        let message = command_buffer_message(400, 130, &numbered(8));
                        let walked = refusing(|| dec.decode(&device, &message));
                        let calls = walk_calls(&p, mark);
                        let theirs: Vec<_> = calls.iter().filter(|c| c.0).cloned().collect();
                        let theirs = dealt_chunks(&theirs);
                        let _ = std::io::Write::write_fmt(
                            &mut std::io::stderr(),
                            format_args!(
                                "dealt walk ended: {walked:?}, {} of 8 chunks on the helper\n",
                                theirs.len()
                            ),
                        );
                    });
                })
                .unwrap()
        };
        while !in_step.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        reims_vgpu_vulkan::exit_gate::arm();
        drop(walker);
        crate::graphics::tests::host_exit(0);
    }

    /// A stream holds a record of command predication where it holds an `ENCODE_START_*` op, and
    /// only then ([`Take::predicates`]): its command buffer is told before its walk that it holds
    /// none otherwise (`CommandBuffer::holds_no_predication`).
    #[test]
    fn a_stream_predicates_only_where_it_holds_a_start_record() {
        use reims_vgpu_graphics::ops::compute::RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS;
        let begin = op(RGFX_OP_ENCODER_BEGIN_COMPUTE, &[0; 8]);
        let end = op(RGFX_OP_ENCODER_END, &[]);
        let plain = [
            begin.clone(),
            op(RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, &[0; 48]),
            end.clone(),
        ]
        .concat();
        let taken = take(&plain);
        assert!(taken.work && !taken.predicates);
        for (tag, body) in [
            (RGFX_OP_COMPUTE_ENCODE_START_IF, &[0u8; 24][..]),
            (RGFX_OP_COMPUTE_ENCODE_START_WHILE, &[0; 24]),
            (RGFX_OP_COMPUTE_ENCODE_START_ELSE, &[]),
            (RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE, &[]),
        ] {
            let ops = [begin.clone(), op(tag, body), end.clone()].concat();
            let taken = take(&ops);
            assert!(taken.work && taken.predicates, "{tag:#x}");
        }
    }
}
