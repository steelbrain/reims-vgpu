//! The rt platform's command buffers ([`RtCommandBuffer`]): rt's, each recorded as it is encoded
//! and handed to rt's submit thread at its commit.
//!
//! - **Handlers.** rt runs a command buffer's scheduled handlers on its submit thread before the
//!   submission goes to the driver (after it, for one that presents a drawable), one held at
//!   `Committed` for a shared event's value among them, once the submit thread stages it; and its
//!   completed handlers on its completion thread, each kind in the order added. Core's
//!   submission handlers run before its scheduled handlers, and its completion-seen handlers before
//!   its completed handlers, whatever the order they were added in: so the command buffer holds
//!   each kind until its commit, which hands rt one scheduled handler that runs the submission
//!   handlers and then the scheduled ones, and one completed handler that runs the completion-seen
//!   handlers and then the completed ones. A submission handler therefore runs on rt's submit
//!   thread, never inside the commit. A handler added after the commit is Metal's refusal
//!   (its validation stops it): counted (`rt-refused-metal`), and it never runs.
//! - **What a completed handler learns**: rt's status and error then, and its GPU times, in
//!   nanoseconds of rt's host clock (`metal2vulkan_runtime::host_time`); none where either is 0,
//!   as for a command buffer that ended with an error. rt's errors are Metal's
//!   `MTLCommandBufferErrorDomain` codes, as Metal answers its GPU's recovery (measured on the M3
//!   Max, macOS 27.0.1, the bundle's `rgfxCommandBufferError`): of the work a lost device ends,
//!   the hang `Timeout` (2) and its victims `Internal` (1). Metal's hang is the oldest command
//!   buffer not complete: here the earliest submitted of those rt ends as in flight
//!   (`DeviceLost { in_flight: true }`), which is the first of them it ends, as its completion
//!   thread ends the work in flight in its order of submission. It is answered as the hang, once
//!   for the device ([`RtDevice::loss`]); every other command buffer rt ends at the loss or after
//!   (in flight after it, held, not yet submitted or committed later) is a victim, so where none
//!   was in flight every one is. A command buffer rt ended without running (its device quiesced,
//!   its queue abandoned) is `Internal` (1) too. Each command buffer's answer is made as it
//!   completes, by a completed handler its commit adds ahead of core's ([`Failure`]), and kept:
//!   its handlers and `error` read the same one.
//! - **A lost device's line.** rt writes nothing at a loss, and nothing makes the device again:
//!   the first of its command buffers that fails with the device lost writes one line on stderr
//!   for the device, naming its GPU (name, PCI vendor and device IDs), the Vulkan call the device
//!   was first lost at and what it returned (rt's `Device::loss`, set before any command buffer
//!   ends with the device lost: rt's own call, or a window swapchain's that met the loss first,
//!   which the swapchain reports to rt, rt's `Device::report_loss`), and how many of its command
//!   buffers were committed and not complete then, the one failing among them ([`DeviceLoss`]).
//!   Where rt named the call, the fault's lines follow it (`super::fault`): the driver's report
//!   of the fault (rt's `DeviceLoss::fault`, or that rt read none), and each address it names
//!   placed among the decoders' live objects and the blocks the host sees, with the nearest below
//!   and above where none holds it, and among the tables' objects' ranges released before the
//!   loss. Two lines follow, whether rt named the call or not: the command buffers committed and
//!   not complete, the oldest committed first, each with its context, its commit's time and its
//!   encoders' kinds, and what the oldest's encoders name: its render passes' attachments, the
//!   heaps and resources they declare and the buffers they bind (`super::fault`'s docs, "The
//!   command buffers in flight"). Each commit adds its command buffer to its device's list before
//!   rt has it, and its first completed handler takes it off.
//!   Each command buffer that fails with the device lost is counted, `rt-device-lost`, in the
//!   process's degradation counts (docs/conventions.md). Then each command buffer rt ends as in
//!   flight writes a line of its own, naming the work the guest knows it by, as its caller named
//!   it (`CommandBuffer::names_work`): the guest's command buffer, by its object id, and the
//!   object ids of the first compute and the first render pipeline state its encoders set, or
//!   `none`; one the caller named nothing for, the host's own, says so. A command buffer cut in
//!   parts writes one line however many of its parts are lost in flight. The first
//!   [`IN_FLIGHT_NAMED`] write theirs, and the next a line that the rest are counted only: each is
//!   counted, `rt-device-lost-in-flight`. The device is lost from rt's
//!   `Device::loss` on ([`RtDevice::is_lost`]): its swapchains make no Vulkan call on it but their
//!   ends' destroys (`super::swapchain`'s docs, "A lost device").
//!   One its caller failed for a deferred buffer's memory rt refused
//!   (`CommandBuffer::fail_out_of_memory`, `super::buffer`'s "Deferred memory") ends with
//!   `OutOfMemory` (8) where rt gave it no error of its own: its status, once rt's is completed,
//!   is an error, and its error, then, that one. One a dispatch of which its stage's fill skipped
//!   (`super::compute`, [`Kept::fail_internal`]) ends so with `Internal` (1). The first of the
//!   two set is the one it ends with.
//! - **Encoders**: blit encoders ([`super::blit`]), compute encoders ([`super::compute`]), of
//!   the serial dispatch type or the descriptor's (another value is Metal's refusal, nil), and
//!   render encoders ([`super::render`]) of the pass the descriptor keeps
//!   ([`super::descriptor`]): a pass Metal refuses, or rt does, is nil, counted.
//! - **Copies of host pages** (`super::copied`): its encoders note each copy they set or declare
//!   an object over, as Metal makes resident only those, and its commit adds its process's
//!   escaped copies, which the caller names before any encoder (`reaches_escaped`) and its parts
//!   take from it; its commit copies the CPU's writes into those before rt is handed the command
//!   buffer, and where a use of one can write it its first completed handler merges the GPU's
//!   writes back into their pages, before core's completion-seen handlers.
//! - **Handed on** ([`hand_on`]): the completed handlers of a command buffer whose first merges a
//!   copy back do not run on rt's completion thread, which retires every queue's work and would
//!   wait for the merge, but on the platform's queue of handlers handed on
//!   (`reims-vgpu.rt-handed-on`), one at a time, in the order rt's thread handed them on; and so
//!   do those of every command buffer rt completes while handlers handed on have not yet run, so
//!   each queue's completed handlers run in the order rt completes its command buffers, as
//!   before. `waitUntilCompleted` returns once they ran, as rt's does for those it runs; on that
//!   queue, as on rt's thread, it answers at once for a command buffer completed and is rt's
//!   refusal for one not yet completed (`WaitOnOwnThread`), which would wait for itself.
//! - **What it keeps**: the pipeline states its encoders set ([`Kept`]), from their set until it
//!   completes, released after its completed handlers ran. It keeps no texture or sampler state:
//!   an index of the platform's own their release gives back waits for every command buffer
//!   committed before the release, by the device's commit order (`super::texture`'s
//!   `OwnIndices`), which its first completed handler advances as it takes it off the device's
//!   command buffers in flight (`super::fault`'s `Flights`).
//! - **Tiled textures** (`super::tiled`): its encoders note each tiled texture they use whose
//!   bytes are to take, and its commit takes them first, in a command buffer committed ahead of it.
//! - **A signal as it is scheduled** (`encodeSignalEventScheduled:value:`) sets the shared
//!   event's value in the scheduled handler, as Metal signals it once the command buffer is
//!   scheduled, in its order among the scheduled handlers.
//! - **Parts.** A part is an ordinary rt command buffer of the same queue, committed on it: `split`
//!   commits the work so far and answers the next part, `part` makes one uncommitted, and
//!   `commit_part` commits the command buffer as a part. rt's submit thread groups the queue's
//!   submissions itself, so `set_completion_joins_next` keeps the trait's default: never marked.
//!   A command buffer is kept whole ([`KeptWhole`], in its order) while an encoder is open, under
//!   protection options or once an encoder was made under them, once it signals an event as it
//!   is scheduled, and once committed. rt's indirect command buffers are read by the GPU as an
//!   execution runs, whichever part holds it (`super::icb`), and rt builds no rate map, residency
//!   or command predication, so no other reason holds; and it notes no way Metal's GPU fails a
//!   command buffer from an encoder on, so `commit_part` never answers [`PartCut::Fails`] and
//!   `fold`, the Vulkan platform's with nothing noted, lets the later part go. The guest buffer's
//!   status, error, GPU times and handlers are core's, from its parts.
//! - **Protection options** other than 0 are unbuilt ([`Unbuilt::Protection`]): rt protects no
//!   work, and the command buffer runs as one without them.
//! - **The trace** (`core::trace`), where it is on: each command buffer's completion writes the
//!   Vulkan platform's `cbdone`, with its commit's time and rt's submission and GPU times
//!   ([`trace_commit`]), so a traced walk splits a command buffer's wait at its submission's end
//!   and its GPU's start and end; each signal and wait of an event it encodes, as rt takes it,
//!   writes `evsignal` or `evwait`, and a shared event's signal at its scheduling `evsched`, with
//!   the command buffer, as its `cbdone` names it, the event and the value
//!   ([`trace_event_op`]), so a traced walk finds the command buffer that signals the value
//!   another waits for; and commits and completions write rt's metrics, first-use waits and
//!   lowerings among them, at most every 50 ms ([`trace_metrics`]).
//!
//! Every call rt refuses answers as Metal's answers where it fails (nothing encoded, nil, no
//! wait), counted under the refusal's kind ([`rt_refused`]); never a death.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use metal2vulkan_runtime as rt;

use crate::core::platform::command::{
    CommandBuffer, CompletedHandler, CompletedInfo, EscapedCopies, KeptWhole, PartCut,
    ScheduledHandler, WorkNames, run_completed,
};
use crate::core::platform::event::EventRef;
use crate::core::platform::os::Clock as _;
use crate::core::platform::types::{CommandBufferStatus, NsErrorText};
use crate::core::platform::values::ComputePassDesc;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::refusals;
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::rt::{BlockEnv, SerialQueue};
use crate::core::sync::{Condvar, Mutex};
use crate::core::trace;

use super::blit::RtBlitEncoder;
use super::compute::RtComputeEncoder;
use super::device::RtDevice;
use super::diag::{Use, Uses};
use super::render::RtRenderEncoder;
use super::texture::{RtTexture, refused_metal};
use super::tiled::Takes;
use super::unbuilt::{RtRefusal, Unbuilt, refuse, refused_as, rt_refused, rt_refused_create};
use super::{RtObj, RtPlatform};

/// `MTLCommandBufferErrorDomain`, the domain of a failed command buffer's error.
const ERROR_DOMAIN: &str = "MTLCommandBufferErrorDomain";

/// `MTLCommandBufferErrorInternal`.
const ERROR_INTERNAL: i64 = 1;

/// `MTLCommandBufferErrorTimeout`.
const ERROR_TIMEOUT: i64 = 2;

/// The degradation code of a command buffer that fails with its device lost (the module's docs,
/// "A lost device's line").
const DEVICE_LOST: &str = "rt-device-lost";

/// The degradation code of a command buffer rt ends as in flight at its device's loss, counted
/// once however many of its parts it ends so (the module's docs, "A lost device's line").
const DEVICE_LOST_IN_FLIGHT: &str = "rt-device-lost-in-flight";

/// How many command buffers lost in flight on a device write a line naming their work (the
/// module's docs, "A lost device's line"): the first few, of a loss that ends every command buffer
/// in flight, and the next writes that the rest are counted only.
const IN_FLIGHT_NAMED: usize = 8;

/// `MTLCommandBufferErrorOutOfMemory`.
const ERROR_OUT_OF_MEMORY: i64 = 8;

/// A handler as rt runs it.
type Handler = Box<dyn FnOnce(&rt::CommandBuffer) + Send>;

/// The number of the last commit the trace timed ([`trace_commit`]).
static TRACED_COMMITS: AtomicU64 = AtomicU64::new(0);

/// When the trace last wrote rt's metrics ([`trace_metrics`]), on its clock.
static METRICS_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// The least time between two of the trace's `rtmetrics` lines, in nanoseconds: 50 ms.
const METRICS_EVERY: u64 = 50_000_000;

/// Where the trace is on (`core::trace`), a completed handler that writes `cb`'s `cbdone`, which
/// the commit runs before core's: the event and values the trace's readers read for each
/// command buffer; and rt's metrics, where
/// they are due ([`trace_metrics`]). The command buffer is named by its address, and numbered in
/// the order of the commits the trace timed. `cbdone`'s times are rt's (the module's docs, what
/// a completed handler learns), on rt's host clock (`metal2vulkan_runtime::host_time`:
/// `CLOCK_MONOTONIC` on Linux, `CLOCK_MONOTONIC_RAW` on macOS, the performance counter on
/// Windows), which is the trace's own clock wherever the platform installs it
/// (`crate::gpu::timing::trace_clock`): each is moved onto the trace's clock by the two clocks'
/// difference read at the handler, which is then only the time between those two reads; 0 stays
/// 0.
fn trace_commit(cb: &rt::CommandBuffer) -> Option<Handler> {
    if !trace::on() {
        return None;
    }
    let n = TRACED_COMMITS
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1);
    let id = address(cb);
    let at = trace::now();
    trace_metrics(at);
    Some(Box::new(move |made: &rt::CommandBuffer| {
        let host = rt::host_time();
        let now = trace::now();
        let on_trace = |t: u64| -> u64 {
            if t == 0 {
                return 0;
            }
            let moved = i128::from(t) + i128::from(now) - i128::from(host);
            u64::try_from(moved.max(0)).unwrap_or(u64::MAX)
        };
        // cbdone (timed at its handler): command buffer, commit number, then its commit, its
        // submission's start and end, its GPU's start and end, in the trace's nanoseconds
        let times = [
            made.kernel_start_time(),
            made.kernel_end_time(),
            made.gpu_start_time(),
            made.gpu_end_time(),
        ]
        .map(on_trace);
        let [kernel_start, kernel_end, gpu_start, gpu_end] = times;
        trace::event_at(
            now,
            "cbdone",
            &[id, n, at, kernel_start, kernel_end, gpu_start, gpu_end],
        );
        trace_metrics(now);
    }))
}

/// Writes rt's metrics (`rtmetrics`), and more of its counts with its gauges (`rtcounts`), to the
/// trace at `now`, its clock's time, where [`METRICS_EVERY`] has passed since the last write: a
/// commit and a completion call it, so a stall with neither has a line at most that long before
/// it and one at its end.
fn trace_metrics(now: u64) {
    let last = METRICS_WRITTEN.load(Ordering::Relaxed);
    if now.saturating_sub(last) < METRICS_EVERY
        || (METRICS_WRITTEN.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed))
            .is_err()
    {
        return;
    }
    let Some(m) = super::device::metrics() else {
        return;
    };
    // rtmetrics: rt's counts since its device was made: first-use waits, their nanoseconds and
    // the longest; the nanoseconds commits waited for the submission lock and the longest;
    // `VkSubmitInfo2`s and `vkQueueSubmit2` calls; `VkPipeline`s created; lowerings made and
    // lowerings read from the cache directory; submissions made while rt's block-maker thread was
    // inside a `vkFreeMemory`; that thread's `vkAllocateMemory` and `vkFreeMemory` calls, and those
    // of them that waited first for a `vkQueueSubmit2` call inside the driver; the states' main
    // pipelines their finishes made and the finishes' nanoseconds; and the first-use waits for a
    // main pipeline no finish had made, their nanoseconds and the longest. A value added goes at
    // the end, after those a reader of an older trace knows.
    trace::event_at(
        now,
        "rtmetrics",
        &[
            m.first_use_waits,
            m.first_use_wait_nanos,
            m.first_use_wait_max_nanos,
            m.commit_lock_wait_nanos,
            m.commit_lock_wait_max_nanos,
            m.submit_infos,
            m.submissions,
            m.pipelines_created,
            m.translations,
            m.translations_loaded,
            m.submissions_during_frees,
            m.maker_memory_calls,
            m.maker_yields,
            m.finished_pipelines,
            m.finished_pipeline_nanos,
            m.first_use_unfinished_waits,
            m.first_use_unfinished_wait_nanos,
            m.first_use_unfinished_wait_max_nanos,
        ],
    );
    trace::event_at(now, "rtcounts", &counts(&m));
}

/// How many values a trace's `rtcounts` event holds (`counts`).
pub(super) const COUNTS: usize = 24 + super::device::OVER_COPY_METRICS + 7;

/// The values of the `rtcounts` event of `m`, rt's metrics, with rt's gauges as they are now. A
/// value added goes at the end, after those a reader of an older trace knows.
fn counts(m: &rt::Metrics) -> Vec<u64> {
    let wall = crate::os::OsClock::new().now_wall().as_nanos();
    let g = super::device::gauges().unwrap_or_default();
    // rtcounts: the wall clock's nanoseconds since 1970, which line the trace up with a guest's
    // wall clock; then more of rt's counts since its device was made: the takes from its pool,
    // their bytes, their nanoseconds in all and the longest; the host's writes of zeros into
    // blocks, their bytes and nanoseconds, and the bytes zeroed in all; the blocks a GPU fill
    // zeroed and their bytes; new textures' images cleared, their bytes and the barriers after
    // them; textures' allocations of their own, then texture blocks' allocations, each the count
    // and the bytes; the `VkDeviceMemory` allocations made for blocks and their bytes; the waits
    // for the block-maker thread and the longest; the barriers and passes recorded; the
    // over-copies and their bytes, those over an imported block, then each cause's
    // (`device::over_copy_counts`); then rt's gauges now: its live textures, attachment views,
    // held descriptor slots, buffers and `VkDeviceMemory` allocations, then its compiled kernels
    // alive (a state's or only its map's) and the entries of its map of compiled contents,
    // compute and render
    let head = [
        wall,
        m.takes.count,
        m.takes.bytes,
        m.takes.total_nanos,
        m.takes.longest_nanos,
        m.zero_writes.count,
        m.zero_writes.bytes,
        m.zero_writes.total_nanos,
        m.bytes_zeroed,
        m.block_zero_fills,
        m.block_zero_fill_bytes,
        m.texture_clears,
        m.texture_clear_bytes,
        m.texture_clear_barriers,
        m.texture_allocations.count,
        m.texture_allocations.bytes,
        m.texture_block_allocations.count,
        m.texture_block_allocations.bytes,
        m.blocks_made,
        m.block_bytes_made,
        m.maker_waits,
        m.maker_wait_max_nanos,
        m.barriers,
        m.passes,
    ];
    let over = super::device::over_copy_counts(m).map(|(_, count)| count);
    let gauges = [
        g.textures,
        g.attachment_views,
        g.descriptor_slots,
        g.buffers,
        g.blocks,
        g.kernels,
        g.kernel_entries,
    ];
    let counts: Vec<u64> = head.into_iter().chain(over).chain(gauges).collect();
    debug_assert_eq!(counts.len(), COUNTS);
    counts
}

/// Writes to the trace, where it is on, an event op that `cb` encoded and rt took: `kind`, then
/// the command buffer, named as its `cbdone` names it, the event, by its object's address (one
/// shared event's signals and waits name the same), and the value.
fn trace_event_op(kind: &str, cb: &rt::CommandBuffer, event: &rt::Event, value: u64) {
    // evsignal, evwait: command buffer, event, value
    trace::event(kind, &[address(cb), address(event), value]);
}

/// `of`'s address, which names it in the trace.
fn address<T>(of: &T) -> u64 {
    u64::try_from(std::ptr::from_ref(of).addr()).unwrap_or_default()
}

/// The command buffers whose completed handlers rt's completion thread handed on ([`hand_on`])
/// and that have not yet run.
static HANDED: AtomicUsize = AtomicUsize::new(0);

/// The queue that runs the completed handlers handed on ([`hand_on`]), made at the first.
static HANDED_ON: OnceLock<SerialQueue> = OnceLock::new();

thread_local! {
    /// Whether this thread runs a command buffer's completed handlers handed on ([`hand_on`]).
    static RUNS_HANDED: Cell<bool> = const { Cell::new(false) };
}

/// Whether a command buffer's completed handlers handed on ([`hand_on`]) are yet to run, which its
/// wait waits for after rt's.
#[derive(Default)]
struct Handed {
    /// Whether they were handed on and have not yet run.
    pending: Mutex<bool>,
    /// Woken once they ran.
    ran: Condvar,
}

impl Handed {
    /// Returns once no handlers handed on are yet to run.
    fn wait(&self) {
        let mut pending = self.pending.lock();
        while *pending {
            self.ran.wait(&mut pending);
        }
    }
}

/// rt's completed handler that runs `handlers`, a command buffer's (the module's docs, "Handed
/// on"): on rt's completion thread, unless `merges` (the first of them merges a copy back) or
/// handlers handed on are yet to run, when it hands them on to [`HANDED_ON`], `handed` holding
/// them yet to run until they ran. Each run there is a step of the exit gate
/// (`crate::exit_gate`), and parks for good once the process's exit closed it. A run on rt's
/// thread is timed (`super::handler_times`).
fn hand_on(
    handlers: impl FnOnce(&rt::CommandBuffer) + Send + 'static,
    merges: bool,
    handed: Arc<Handed>,
) -> impl FnOnce(&rt::CommandBuffer) + Send + 'static {
    move |made: &rt::CommandBuffer| {
        if !merges && HANDED.load(Ordering::SeqCst) == 0 {
            super::handler_times::time(|| handlers(made));
            return;
        }
        HANDED.fetch_add(1, Ordering::SeqCst);
        *handed.pending.lock() = true;
        let made = made.clone();
        let queue = HANDED_ON.get_or_init(|| {
            SerialQueue::new(&crate::os::OsThreads::new(), "reims-vgpu.rt-handed-on")
        });
        queue.exec_async(move || {
            let Some(step) = crate::exit_gate::step() else {
                crate::exit_gate::stay()
            };
            let was = RUNS_HANDED.replace(true);
            handlers(&made);
            RUNS_HANDED.set(was);
            drop((made, step));
            *handed.pending.lock() = false;
            handed.ran.notify_all();
            HANDED.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

/// Whether rt's command buffer `cb` completed or ended.
fn finished(cb: &rt::CommandBuffer) -> bool {
    matches!(
        cb.status(),
        rt::CommandBufferStatus::Completed | rt::CommandBufferStatus::Error
    )
}

/// `id<MTLCommandBuffer>` on metal2vulkan-runtime.
pub struct RtCommandBuffer {
    /// rt's command buffer.
    rt: rt::CommandBuffer,
    /// rt's queue, which makes its parts.
    queue: rt::CommandQueue,
    /// The device, which `device` answers.
    device: RtObj<RtDevice>,
    /// The handlers until the commit (the module's docs), and what keeps it whole.
    held: Mutex<Held>,
    /// Its encoders made and not yet ended ([`OpenMark`]).
    open: Arc<AtomicUsize>,
    /// What it keeps until it completes.
    kept: Kept,
    /// Its error as Metal's, made as it completes ([`Failure`]).
    failure: Arc<Failure>,
    /// Its process's escaped copies (`super::copied`), which each of its commits syncs, as do
    /// its parts', where the caller named them (`reaches_escaped`).
    escapes: OnceLock<EscapedCopies>,
    /// Whether its completed handlers, handed on, have run ([`hand_on`]).
    handed: Arc<Handed>,
}

/// A command buffer's error as Metal's (the module's docs, "What a completed handler learns"):
/// made at its first read, which a completed handler its commit adds ahead of core's makes, and
/// kept.
struct Failure {
    /// Its error as Metal's, once read.
    error: OnceLock<NsErrorText>,
    /// Its device's loss ([`RtDevice::loss`]).
    loss: Arc<DeviceLoss>,
    /// rt's cause of its device's loss (rt's `Device::loss`), which the loss's line names.
    cause: Box<dyn Fn() -> Option<rt::DeviceLoss> + Send + Sync>,
    /// The guest's names for its work, which its line names where rt ends it as in flight at the
    /// loss, as its caller named them, or the command buffer it was cut from held them
    /// (`names_work`).
    work: OnceLock<Arc<WorkNames>>,
    /// Its number in its device's command buffers committed and not complete
    /// (`super::fault::Flights`), from its commit until its first completed handler runs; 0 for
    /// none.
    flight: AtomicU64,
}

impl Failure {
    /// The answer of rt's `error`, made at the first that is not `None` and kept.
    fn read(&self, error: Option<rt::CommandBufferError>) -> Option<NsErrorText> {
        let error = error?;
        let work = self.work.get().map(Arc::as_ref);
        Some(
            self.error
                .get_or_init(|| ns_error(error, &self.loss, &*self.cause, work))
                .clone(),
        )
    }

    /// Its first completed handler's part: reads its error, then takes it off its device's
    /// command buffers not complete ([`DeviceLoss`]), and their list (`super::fault::Flights`).
    fn completed(&self, error: Option<rt::CommandBufferError>) {
        drop(self.read(error));
        self.loss.open.fetch_sub(1, Ordering::SeqCst);
        let flight = self.flight.swap(0, Ordering::SeqCst);
        self.loss.flights.remove(flight);
    }
}

/// A device's loss as its command buffers see it (the module's docs, "What a completed handler
/// learns" and "A lost device's line"), which each of them keeps.
pub struct DeviceLoss {
    /// The GPU, as the line names it: its name, and its PCI vendor and device IDs.
    gpu: String,
    /// Whether a command buffer in flight at the loss was answered as the hang.
    hung: AtomicBool,
    /// Whether the loss's line was written.
    said: AtomicBool,
    /// How many command buffers rt ended as in flight at the loss came to write their lines
    /// ([`Self::lost_in_flight`]).
    in_flight: AtomicUsize,
    /// Its command buffers committed and not complete: a commit adds one before rt has it (and
    /// takes it off where rt refuses it), and its first completed handler takes it off once its
    /// error is read, so a failing command buffer counts itself.
    open: AtomicI64,
    /// What the loss's fault is placed among (`super::fault`).
    watch: super::fault::FaultWatch,
    /// Its command buffers committed and not complete, which the loss's lines list
    /// (`super::fault`'s docs, "The command buffers in flight"), by their commit order, which the
    /// platform's own indices given back wait on (`super::texture`'s `OwnIndices`).
    flights: Arc<super::fault::Flights>,
}

impl DeviceLoss {
    /// The loss of a device of the GPU `gpu`, before any, its fault placed among nothing.
    pub fn new(gpu: String) -> DeviceLoss {
        DeviceLoss::watching(gpu, super::fault::FaultWatch::default())
    }

    /// The loss of a device of the GPU `gpu`, before any, its fault placed among what `watch`
    /// sees (`super::fault`).
    pub fn watching(gpu: String, watch: super::fault::FaultWatch) -> DeviceLoss {
        DeviceLoss {
            gpu,
            hung: AtomicBool::new(false),
            said: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            open: AtomicI64::new(0),
            watch,
            flights: Arc::default(),
        }
    }

    /// What its fault is placed among (`super::fault`).
    pub(super) fn watch(&self) -> &super::fault::FaultWatch {
        &self.watch
    }

    /// Its command buffers committed and not complete, by their commit order (`super::fault`'s
    /// docs, "The command buffers in flight").
    pub(super) fn flights(&self) -> &Arc<super::fault::Flights> {
        &self.flights
    }

    /// rt ended a command buffer as in flight at the loss, the guest's names for its work `work`
    /// where its caller named them: counted, once however many of its parts it ends so, and, of
    /// the first [`IN_FLIGHT_NAMED`], each writes a line naming them; the next writes that the
    /// rest are counted only (the module's docs, "A lost device's line").
    fn lost_in_flight(&self, work: Option<&WorkNames>) {
        if work.is_some_and(|work| !work.first_told()) {
            return;
        }
        crate::answer::count_kind(DEVICE_LOST_IN_FLIGHT);
        let n = self.in_flight.fetch_add(1, Ordering::SeqCst);
        let line = match (n.cmp(&IN_FLIGHT_NAMED), work) {
            (std::cmp::Ordering::Greater, _) => return,
            (std::cmp::Ordering::Equal, _) => format!(
                "reims-vgpu: more command buffers were lost in flight on the rt platform's \
                 device on {}: the degradation counts give how many in all \
                 ({DEVICE_LOST_IN_FLIGHT})\n",
                self.gpu
            ),
            (std::cmp::Ordering::Less, Some(work)) => {
                let named = |id: u64| match id {
                    0 => "none".to_owned(),
                    id => format!("{id:#x}"),
                };
                let (compute, render) = work.firsts();
                format!(
                    "reims-vgpu: the guest's command buffer {:#x} was lost in flight on the rt \
                     platform's device on {}, its encoders' first compute pipeline state {} and \
                     first render pipeline state {} ({DEVICE_LOST_IN_FLIGHT})\n",
                    work.cb(),
                    self.gpu,
                    named(compute),
                    named(render)
                )
            }
            (std::cmp::Ordering::Less, None) => format!(
                "reims-vgpu: a command buffer of the host's own, which the guest does not \
                 number, was lost in flight on the rt platform's device on {} \
                 ({DEVICE_LOST_IN_FLIGHT})\n",
                self.gpu
            ),
        };
        say_lost(&line);
    }

    /// A command buffer failed with the device lost: counted, and the first writes the loss's
    /// line, with rt's cause, which `cause` reads, and the lines that follow it (the module's
    /// docs, "A lost device's line").
    fn lost(&self, cause: &dyn Fn() -> Option<rt::DeviceLoss>) {
        crate::answer::count_kind(DEVICE_LOST);
        if self.said.swap(true, Ordering::SeqCst) {
            return;
        }
        // The loss's time, as the host has it (`super::fault`'s docs).
        let when = super::fault::wall_now();
        let open = self.open.load(Ordering::SeqCst).max(0);
        let cause = cause();
        let at = match &cause {
            Some(rt::DeviceLoss { call, result, .. }) => format!(" at {call} ({result:?})"),
            None => " with no call rt named".to_owned(),
        };
        say_lost(&format!(
            "reims-vgpu: the rt platform's device on {} was lost{at}, with {open} command buffers \
             committed and not complete: every command buffer fails from here on, and nothing \
             makes the device again ({DEVICE_LOST})\n",
            self.gpu
        ));
        // The driver's report of the fault, its addresses placed (`super::fault`).
        if let Some(cause) = &cause {
            for line in self.watch.lines(&self.gpu, cause.fault.as_deref(), when) {
                say_lost(&line);
            }
        }
        // The command buffers committed and not complete (`super::fault`).
        for line in self.flights.lines(&self.gpu, when) {
            say_lost(&line);
        }
    }
}

/// The lost devices' lines the process wrote, on whichever thread: rt's completion thread runs
/// the completed handler that writes one (`Failure::completed`).
#[cfg(test)]
static SAID_LOST: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Writes a lost device's `line` on stderr, as the platform's other degradation lines
/// (`super::unbuilt`); a test reads them (`SAID_LOST`).
fn say_lost(line: &str) {
    super::unbuilt::say(line);
    #[cfg(test)]
    SAID_LOST.lock().push(line.to_owned());
}

/// A command buffer's handlers until its commit hands them to rt, each kind in the order added,
/// and what keeps it whole but an open encoder.
#[derive(Default)]
struct Held {
    /// Whether the command buffer was committed: a handler added since is refused.
    committed: bool,
    /// Its protection options (`setProtectionOptions:`).
    protection: u64,
    /// Whether an encoder was made under protection options other than 0.
    protected_encoder: bool,
    /// Whether it signals an event as it is scheduled.
    scheduled_signal: bool,
    /// Core's submission handlers.
    submission: Vec<Handler>,
    /// Core's scheduled handlers, and the scheduled signals.
    scheduled: Vec<Handler>,
    /// Core's completion-seen handlers.
    seen: Vec<Handler>,
    /// Core's completed handlers.
    completed: Vec<Handler>,
    /// In a process that records, the recorder's hook of its commit, run first of the handlers
    /// that can tell the guest of its completion ([`crate::recorder::RecCommandBuffer`]).
    recorder: Option<Box<dyn FnOnce() + Send>>,
}

impl crate::recorder::RecCommandBuffer for RtObj<RtCommandBuffer> {
    fn rec_raw(&self) -> crate::recorder::RawCb {
        crate::recorder::RawCb::Rt(self.rt.clone())
    }

    /// Kept for the commit under way, which runs it after the merge of the host pages it copied
    /// and before core's handlers (`commit`).
    fn rec_on_seen(&self, seen: Box<dyn FnOnce() + Send>) {
        self.held.lock().recorder = Some(seen);
    }
}

impl RtCommandBuffer {
    /// The command buffer over rt's `made`, of `queue` and `device`.
    pub fn new(
        made: rt::CommandBuffer,
        queue: rt::CommandQueue,
        device: RtObj<RtDevice>,
    ) -> RtCommandBuffer {
        let rt_device = queue.device().clone();
        let failure = Arc::new(Failure {
            error: OnceLock::new(),
            loss: Arc::clone(device.loss()),
            cause: Box::new(move || rt_device.loss()),
            work: OnceLock::new(),
            flight: AtomicU64::new(0),
        });
        RtCommandBuffer {
            rt: made,
            queue,
            device,
            held: Mutex::new(Held::default()),
            open: Arc::default(),
            kept: Kept::default(),
            failure,
            escapes: OnceLock::new(),
            handed: Arc::default(),
        }
    }

    /// rt's command buffer.
    pub fn rt(&self) -> &rt::CommandBuffer {
        &self.rt
    }

    /// Adds `handler` to the list `list` picks, or refuses it, counted, once the command buffer
    /// was committed. Whether it was added.
    fn hold(
        &self,
        sel: &'static str,
        list: fn(&mut Held) -> &mut Vec<Handler>,
        handler: Handler,
    ) -> bool {
        let mut held = self.held.lock();
        if held.committed {
            drop(held);
            rt_refused(sel, &rt::Error::Refused(rt::Refusal::Committed));
            return false;
        }
        list(&mut held).push(handler);
        true
    }

    /// The mark of an encoder made now, open until it ends ([`OpenMark`]); and an encoder made
    /// under protection options other than 0 keeps the command buffer whole.
    fn opened(&self) -> OpenMark {
        let mut held = self.held.lock();
        if held.protection != 0 {
            held.protected_encoder = true;
        }
        drop(held);
        OpenMark::new(&self.open)
    }

    /// A compute encoder of `dispatch_type`, made by `sel`: nil where rt refuses it, counted.
    fn compute_encoder_of(
        &self,
        sel: &'static str,
        dispatch_type: rt::DispatchType,
    ) -> Option<RtObj<RtComputeEncoder>> {
        match self.rt.compute_command_encoder(dispatch_type) {
            Ok(encoder) => {
                let concurrent = dispatch_type == rt::DispatchType::Concurrent;
                let (mark, kept) = (self.opened(), self.kept.clone());
                let fills = Arc::clone(self.device.stage_fills());
                Some(RtObj::new(RtComputeEncoder::new(
                    encoder, mark, kept, concurrent, fills,
                )))
            }
            Err(error) => {
                rt_refused_create(sel, &error);
                None
            }
        }
    }

    /// Why it may not be cut now: the first of [`KeptWhole::ALL`] that holds (the module's docs).
    fn kept_whole_by(&self) -> Option<KeptWhole> {
        let held = self.held.lock();
        let (protection, scheduled_signal, committed) = (
            held.protection != 0 || held.protected_encoder,
            held.scheduled_signal,
            held.committed,
        );
        drop(held);
        let open = self.open.load(Ordering::SeqCst) != 0;
        let holds = |reason: KeptWhole| match reason {
            KeptWhole::OpenEncoder => open,
            KeptWhole::ProtectionOptions => protection,
            KeptWhole::ScheduledSignal => scheduled_signal,
            KeptWhole::Committed => committed,
            // The GPU reads rt's indirect command buffers as an execution runs, and rt builds no
            // rate map, residency or command predication.
            KeptWhole::Execution | KeptWhole::Progress | KeptWhole::Predication => false,
        };
        KeptWhole::ALL.into_iter().find(|&reason| holds(reason))
    }

    /// The next part, made on its queue: waits, as `commandBuffer` does, while the queue's most
    /// command buffers are made and not yet completed. `None` where rt refuses it (a quiesced
    /// device, a wait for room on rt's own handler thread), counted: then it is kept whole.
    fn next_part(&self) -> Option<RtObj<RtCommandBuffer>> {
        match self.queue.command_buffer() {
            Ok(made) => {
                let part = RtCommandBuffer::new(made, self.queue.clone(), self.device.clone_ref());
                if let Some(escapes) = self.escapes.get() {
                    let _ = part.escapes.set(escapes.clone());
                }
                if let Some(work) = self.failure.work.get() {
                    let _ = part.failure.work.set(Arc::clone(work));
                }
                Some(RtObj::new(part))
            }
            Err(error) => {
                rt_refused("commandBuffer", &error);
                None
            }
        }
    }
}

/// What a command buffer keeps from its encoders' sets until it completes: the objects whose drop
/// gives back what the work recorded with them reads, by their identity, each once. A pipeline
/// state's drop gives back its sampler indices (`super::pipeline`), which rt's recorded work reads
/// while rt keeps its own state. The commit hands them to a completed handler, which drops them
/// after core's; a command buffer never committed drops them with itself.
/// It holds too its uses of textures over an imported block, for the boot diagnostics
/// (`super::diag`), which its commit counts, the tiled textures it used whose bytes its commit
/// takes (`super::tiled`), whether its caller failed it for a deferred buffer's memory
/// ([`Failed`]), the copies of host pages it used, which its commit and completion sync
/// (`super::copied`), and its encoders' names, which its commit takes for its device's
/// list of command buffers in flight (`super::fault`'s `Encoded`).
#[derive(Clone, Default)]
pub struct Kept(
    Arc<Mutex<HashMap<ObjId, Box<dyn Send>>>>,
    Uses,
    Takes,
    Failed,
    super::copied::Used,
    super::fault::Encoded,
);

/// The code of Metal's error a command buffer was failed with, 0 for none, the first set kept: by
/// its caller for a deferred buffer's memory rt refused (`CommandBuffer::fail_out_of_memory`), or
/// by a dispatch of it that its stage's fill skipped ([`Kept::fail_internal`]) (the module's docs,
/// "What a completed handler learns").
#[derive(Clone, Default)]
struct Failed(Arc<AtomicI64>);

impl Failed {
    /// The code the command buffer was failed with; `None` where it was not.
    fn code(&self) -> Option<i64> {
        Some(self.0.load(Ordering::Acquire)).filter(|&code| code != 0)
    }

    /// Whether the command buffer was failed.
    fn holds(&self) -> bool {
        self.code().is_some()
    }

    /// Fails the command buffer with `code`, unless it was failed already.
    fn fail(&self, code: i64) {
        let _ = self
            .0
            .compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
    }
}

impl Kept {
    /// Its uses of textures over an imported block (`super::diag`).
    pub fn uses(&self) -> &Uses {
        &self.1
    }

    /// The tiled textures it used whose bytes are to take (`super::tiled`).
    pub fn takes(&self) -> &Takes {
        &self.2
    }

    /// The copies of host pages it used, to sync at its commit and completion (`super::copied`).
    pub fn copies(&self) -> &super::copied::Used {
        &self.4
    }

    /// Its encoders' names, each joined as its encoder ends (`super::fault`'s `Naming`).
    pub fn encoded(&self) -> &super::fault::Encoded {
        &self.5
    }

    /// An encoder of it used `texture` as `how` says: a use for the boot diagnostics
    /// ([`Uses::add`]), of a tiled texture whose bytes are to take, a take at its commit
    /// ([`Takes::used`]), of a texture over a copy, the copy's use (`super::copied::Used`), one
    /// that can write it unless the texture is a blit's source or is bound to a shader's slot and
    /// no shader writes it (`RtTexture::shader_writes`).
    pub fn used(&self, texture: &RtObj<RtTexture>, how: Use) {
        self.1.add(texture, how);
        self.2.used(texture);
        let writes = match how {
            Use::Sampled => texture.shader_writes(),
            Use::Blitted => false,
            Use::Rendered | Use::BlittedInto => true,
        };
        self.4.add(texture.copied(), writes);
    }

    /// Fails the command buffer with `Internal` (1), unless it was failed already: a dispatch of
    /// it that its stage's fill skipped (`super::compute`; the module's docs, "What a completed
    /// handler learns").
    pub fn fail_internal(&self) {
        self.3.fail(ERROR_INTERNAL);
    }

    /// Keeps `object`, whose identity is `id`, unless an object of that identity is kept.
    pub fn keep(&self, id: ObjId, object: Box<dyn Send>) {
        self.0.lock().entry(id).or_insert(object);
    }

    /// What is kept, taken: none is kept since.
    fn take(&self) -> HashMap<ObjId, Box<dyn Send>> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// One encoder of a command buffer, open from its make until it ends or goes, rt ending an
/// encoder let go of open: the command buffer counts its open encoders, which keep it whole
/// ([`KeptWhole::OpenEncoder`]).
pub struct OpenMark {
    /// The command buffer's count of open encoders.
    open: Arc<AtomicUsize>,
    /// Whether this one ended.
    ended: AtomicBool,
}

impl OpenMark {
    /// One more open encoder of `open`.
    fn new(open: &Arc<AtomicUsize>) -> OpenMark {
        open.fetch_add(1, Ordering::SeqCst);
        OpenMark {
            open: Arc::clone(open),
            ended: AtomicBool::new(false),
        }
    }

    /// The encoder ended: once, however often it is called.
    pub fn end(&self) {
        if !self.ended.swap(true, Ordering::SeqCst) {
            self.open.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

impl Drop for OpenMark {
    fn drop(&mut self) {
        self.end();
    }
}

/// rt's handler that runs core's scheduled or submission handler `block`.
fn scheduled(block: ScheduledHandler) -> Handler {
    #[cfg(test)]
    let slot = crate::core::testkit::capture::current_slot();
    Box::new(move |_made: &rt::CommandBuffer| {
        // A callout on rt's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        #[cfg(test)]
        crate::core::testkit::capture::with_slot(slot, block);
        #[cfg(not(test))]
        block();
    })
}

/// rt's handler that runs core's completed or completion-seen handler `block` with what the
/// command buffer answers then ([`completed_info`]), its error `failure`'s, failed as `failed`
/// says. Added inside a refusal scope (the graphics lane's), the read and `block` run in one on
/// rt's thread too ([`run_completed`]).
fn completed(block: CompletedHandler, failure: Arc<Failure>, failed: Failed) -> Handler {
    #[cfg(test)]
    let slot = crate::core::testkit::capture::current_slot();
    let carried = refusals::carry();
    Box::new(move |made: &rt::CommandBuffer| {
        // A callout on rt's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        let run = || run_completed(carried, || completed_info(made, &failure, &failed), block);
        #[cfg(test)]
        crate::core::testkit::capture::with_slot(slot, run);
        #[cfg(not(test))]
        run();
    })
}

/// What a completed handler learns of `made`, its error `failure`'s, failed as `failed` says (the
/// module's docs).
fn completed_info(made: &rt::CommandBuffer, failure: &Failure, failed: &Failed) -> CompletedInfo {
    let (start, end) = (made.gpu_start_time(), made.gpu_end_time());
    CompletedInfo {
        status: status_of(made, failed),
        error: error_of(made, failure, failed),
        gpu_times_ns: (start != 0 && end >= start).then_some((start, end)),
    }
}

/// `made`'s status as Metal's, an error once completed where `failed` holds (the module's docs,
/// "What a completed handler learns").
fn status_of(made: &rt::CommandBuffer, failed: &Failed) -> CommandBufferStatus {
    match made.status() {
        rt::CommandBufferStatus::Completed if failed.holds() => CommandBufferStatus::ERROR,
        other => status(other),
    }
}

/// `made`'s error as Metal's: rt's, as `failure` answers it, else, once completed, the one
/// `failed` holds (the module's docs, "What a completed handler learns").
fn error_of(made: &rt::CommandBuffer, failure: &Failure, failed: &Failed) -> Option<NsErrorText> {
    if let Some(error) = failure.read(made.error()) {
        return Some(error);
    }
    let completed = matches!(made.status(), rt::CommandBufferStatus::Completed);
    let code = failed.code().filter(|_| completed)?;
    let description = match code {
        ERROR_OUT_OF_MEMORY => {
            "a buffer the command buffer used was refused its memory at that use"
        }
        _ => "a dispatch of the command buffer was skipped: its stage's fill failed or breached",
    };
    Some(NsErrorText {
        domain: ERROR_DOMAIN.to_owned(),
        code,
        description: description.to_owned(),
    })
}

/// rt's status as Metal's raw `MTLCommandBufferStatus`.
fn status(status: rt::CommandBufferStatus) -> CommandBufferStatus {
    match status {
        rt::CommandBufferStatus::NotEnqueued => CommandBufferStatus::NOT_ENQUEUED,
        rt::CommandBufferStatus::Enqueued => CommandBufferStatus::ENQUEUED,
        rt::CommandBufferStatus::Committed => CommandBufferStatus::COMMITTED,
        rt::CommandBufferStatus::Scheduled => CommandBufferStatus::SCHEDULED,
        rt::CommandBufferStatus::Completed => CommandBufferStatus::COMPLETED,
        rt::CommandBufferStatus::Error => CommandBufferStatus::ERROR,
    }
}

/// rt's error as Metal's `NSError` (the module's docs): `Timeout` for the first command buffer
/// lost in flight on the device whose loss is `loss`, `Internal` for any other; a command buffer
/// that fails with its device lost tells `loss`, with rt's cause, which `cause` reads
/// ([`DeviceLoss::lost`]), and one lost in flight the guest's names for its work, `work`, where
/// its caller named them ([`DeviceLoss::lost_in_flight`]).
fn ns_error(
    error: rt::CommandBufferError,
    loss: &DeviceLoss,
    cause: &dyn Fn() -> Option<rt::DeviceLoss>,
    work: Option<&WorkNames>,
) -> NsErrorText {
    if let rt::CommandBufferError::DeviceLost { in_flight } = error {
        loss.lost(cause);
        if in_flight {
            loss.lost_in_flight(work);
        }
    }
    let code = match error {
        rt::CommandBufferError::DeviceLost { in_flight: true }
            if !loss.hung.swap(true, Ordering::SeqCst) =>
        {
            ERROR_TIMEOUT
        }
        _ => ERROR_INTERNAL,
    };
    NsErrorText {
        domain: ERROR_DOMAIN.to_owned(),
        code,
        description: format!("metal2vulkan-runtime's command buffer failed: {error:?}"),
    }
}

/// rt's event of `event`; `None`, counted as Metal's refusal at `sel`, for nil.
fn rt_event<'a>(
    sel: &'static str,
    event: Option<EventRef<'a, RtPlatform>>,
) -> Option<&'a rt::Event> {
    match event {
        Some(EventRef::Event(event)) => Some(event),
        Some(EventRef::Shared(event)) => Some(event.as_event()),
        None => {
            refused_as(sel, RtRefusal::Refused, &"a nil event");
            None
        }
    }
}

impl CommandBuffer<RtPlatform> for RtObj<RtCommandBuffer> {
    /// Held until the commit, behind the completion-seen handlers (the module's docs).
    fn add_completed_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        let _ = on;
        let handler = completed(block, Arc::clone(&self.failure), self.kept.3.clone());
        self.hold("addCompletedHandler:", |h| &mut h.completed, handler);
        Ok(())
    }

    /// Held until the commit, behind the submission handlers (the module's docs).
    fn add_scheduled_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        let _ = on;
        let handler = scheduled(block);
        self.hold("addScheduledHandler:", |h| &mut h.scheduled, handler);
        Ok(())
    }

    /// Run first of the scheduled handlers, on rt's submit thread as the command buffer is
    /// scheduled: rt fixes its place in its queue's order at the commit.
    fn add_submission_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        let _ = on;
        let handler = scheduled(block);
        self.hold("addSubmissionHandler", |h| &mut h.submission, handler);
        Ok(())
    }

    /// Run first of the completed handlers, on rt's completion thread.
    fn add_completion_seen_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        let _ = on;
        let handler = completed(block, Arc::clone(&self.failure), self.kept.3.clone());
        self.hold("addCompletionSeenHandler", |h| &mut h.seen, handler);
        Ok(())
    }

    /// rt's blit encoder; nil where rt refuses it (another encoder open, the command buffer
    /// committed), counted.
    fn blit_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::BlitEncoder>> {
        let recorded_call = crate::recorder::rec_call!("blitCommandEncoder", self);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::BlitEncoder>> {
                let _ = on;
                Ok(match self.rt.blit_command_encoder() {
                    Ok(encoder) => {
                        let kept = self.kept.clone();
                        let blit = super::fault::EncoderNames::new(super::fault::EncoderKind::Blit);
                        kept.encoded().push(blit);
                        Some(RtObj::new(RtBlitEncoder::new(encoder, self.opened(), kept)))
                    }
                    Err(error) => {
                        rt_refused_create("blitCommandEncoder", &error);
                        None
                    }
                })
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Hands rt the held handlers (the module's docs), then commits, the copies of host pages it
    /// used and its process's escaped copies copied in first and, those a use of which can write
    /// them, merged back first of the completed handlers (`super::copied`), which are then handed
    /// on ([`hand_on`]), then the bytes of the tiled textures it used taken (`super::tiled`). A
    /// second commit is Metal's refusal, counted; a commit rt refuses is counted, and its
    /// handlers never run. In a process that records, the recorder writes the memory it reaches
    /// that changed before rt has it ([`crate::recorder::before_commit`]), and takes what its GPU
    /// work wrote after the merge of the host pages it copied, before core's handlers can tell
    /// the guest ([`crate::recorder::completion_seen`]).
    fn commit(&self, on: OnFault) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("commit", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            let sel = "commit";
            if self.held.lock().committed {
                rt_refused(sel, &rt::Error::Refused(rt::Refusal::Committed));
                return Ok(());
            }
            // Before the handlers are taken: the recorder's hook joins them (`rec_on_seen`).
            crate::recorder::before_commit(&recorded_call, self);
            let (mut submission, scheduled, mut seen, completed, recorder) = {
                let mut held = self.held.lock();
                if held.committed {
                    drop(held);
                    rt_refused(sel, &rt::Error::Refused(rt::Refusal::Committed));
                    return Ok(());
                }
                held.committed = true;
                let take = std::mem::take::<Vec<Handler>>;
                (
                    take(&mut held.submission),
                    take(&mut held.scheduled),
                    take(&mut held.seen),
                    take(&mut held.completed),
                    held.recorder.take(),
                )
            };
            submission.extend(scheduled);
            seen.extend(completed);
            // The recorder's hook goes after the merge, the error and the trace's cbdone, before
            // core's handlers.
            if let Some(hook) = recorder {
                seen.insert(0, Box::new(move |_made: &rt::CommandBuffer| hook()));
            }
            // The trace's cbdone goes after its error and its merge, before core's handlers.
            if let Some(done) = trace_commit(&self.rt) {
                seen.insert(0, done);
            }
            // Its error is made before core's handlers run, as rt completes it, so a lost
            // device's hang is the first it ends in flight (the module's docs).
            let failure = Arc::clone(&self.failure);
            seen.insert(
                0,
                Box::new(move |made: &rt::CommandBuffer| failure.completed(made.error())),
            );
            if let Some(escapes) = self.escapes.get() {
                self.kept.copies().escaped(escapes);
            }
            let copies = self.kept.copies().take();
            let merges = super::copied::writes(&copies);
            if let Some(merge) = super::copied::merging(&copies) {
                seen.insert(0, Box::new(merge));
            }
            self.kept.uses().submitted();
            let kept = self.kept.take();
            if !kept.is_empty() {
                seen.push(Box::new(move |_made: &rt::CommandBuffer| drop(kept)));
            }
            // Its encoders' names, for its entry in the device's command buffers in flight.
            let encoders = self.kept.encoded().take();
            let run = |handlers: Vec<Handler>| {
                move |made: &rt::CommandBuffer| {
                    handlers.into_iter().for_each(|handler| handler(made))
                }
            };
            let committed = || -> Result<(), rt::Error> {
                if !submission.is_empty() {
                    self.rt.add_scheduled_handler(run(submission))?;
                }
                let handed = Arc::clone(&self.handed);
                self.rt
                    .add_completed_handler(hand_on(run(seen), merges, handed))?;
                super::copied::copy_in(&copies);
                self.kept.takes().take(&self.device);
                // Not complete until its first completed handler runs, which a refusal never runs.
                let loss = &self.failure.loss;
                let number = loss.flights.add(super::fault::Flight {
                    cb: self.failure.work.get().map(|work| work.cb()),
                    committed: super::fault::wall_now(),
                    encoders,
                });
                self.failure.flight.store(number, Ordering::SeqCst);
                loss.open.fetch_add(1, Ordering::SeqCst);
                let committed = self.rt.commit();
                if committed.is_err() {
                    loss.open.fetch_sub(1, Ordering::SeqCst);
                    let flight = self.failure.flight.swap(0, Ordering::SeqCst);
                    loss.flights.remove(flight);
                }
                committed
            };
            if let Err(error) = committed() {
                rt_refused(sel, &error);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A serial compute encoder: nil where rt refuses it, counted.
    fn compute_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
        let recorded_call = crate::recorder::rec_call!("computeCommandEncoder", self);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
                let _ = on;
                let serial = rt::DispatchType::Serial;
                Ok(self.compute_encoder_of("computeCommandEncoder", serial))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn device(&self) -> Option<<RtPlatform as Platform>::Device> {
        Some(self.device.clone_ref())
    }

    /// rt's signal; nothing, counted, for a nil event or one rt refuses (an encoder open, the
    /// command buffer committed).
    fn encode_signal(
        &self,
        event: Option<EventRef<'_, RtPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let recorded_call =
            crate::recorder::rec_call!("encodeSignalEvent:value:", self, event, value);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let sel = "encodeSignalEvent:value:";
            if let Some(event) = rt_event(sel, event) {
                match self.rt.encode_signal_event(event, value) {
                    Ok(()) => trace_event_op("evsignal", &self.rt, event, value),
                    Err(error) => {
                        rt_refused(sel, &error);
                    }
                }
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The event's value set in the scheduled handler (the module's docs); nothing, counted, for
    /// nil.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&<RtPlatform as Platform>::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(
            "encodeSignalEventScheduled:value:",
            self,
            signal_event_scheduled,
            value
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let sel = "encodeSignalEventScheduled:value:";
            let Some(event) = signal_event_scheduled else {
                refused_as(sel, RtRefusal::Refused, &"a nil event");
                return Ok(());
            };
            let traced = address(event.as_event());
            let event = rt::SharedEvent::clone(event);
            let signal = move |_made: &rt::CommandBuffer| event.set_signaled_value(value);
            if self.hold(sel, |h| &mut h.scheduled, Box::new(signal)) {
                self.held.lock().scheduled_signal = true;
                // evsched: command buffer, event, value (as evsignal's, [`trace_event_op`])
                trace::event("evsched", &[address(&self.rt), traced, value]);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's wait; nothing, counted, for a nil event or one rt refuses.
    fn encode_wait(&self, event: Option<EventRef<'_, RtPlatform>>, value: u64) -> RaiseResult<()> {
        let recorded_call =
            crate::recorder::rec_call!("encodeWaitForEvent:value:", self, event, value);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let sel = "encodeWaitForEvent:value:";
            if let Some(event) = rt_event(sel, event) {
                match self.rt.encode_wait_for_event(event, value) {
                    Ok(()) => trace_event_op("evwait", &self.rt, event, value),
                    Err(error) => {
                        rt_refused(sel, &error);
                    }
                }
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's error as Metal's, the one its completed handlers learn, or the caller's failure of it
    /// (the module's docs).
    fn error(&self) -> Option<NsErrorText> {
        error_of(&self.rt, &self.failure, &self.kept.3)
    }

    /// rt's render encoder of the pass the descriptor keeps: nil where Metal refuses the pass (an
    /// action Metal has none of, a level or slice past 32 bits) or rt refuses it (another encoder
    /// open, the command buffer committed, an attachment rt renders into none of), counted.
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &<RtPlatform as Platform>::RenderPassDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::RenderEncoder>> {
        let recorded_call = crate::recorder::rec_call!(
            "renderCommandEncoderWithDescriptor:",
            self,
            render_pass_descriptor
        );
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::RenderEncoder>> {
                let _ = on;
                let sel = "renderCommandEncoderWithDescriptor:";
                let pass = match render_pass_descriptor.rt_pass() {
                    Ok(pass) => pass,
                    Err(why) => return Ok(refused_metal(sel, &why)),
                };
                Ok(match self.rt.render_command_encoder(&pass) {
                    Ok(encoder) => {
                        let (mark, kept) = (self.opened(), self.kept.clone());
                        let targets = render_pass_descriptor.targets();
                        for target in &targets {
                            kept.uses().add_made(&target.made, Use::Rendered);
                        }
                        let blocks = targets.iter().filter_map(|t| t.made.block());
                        super::copied::store_outs(sel, blocks);
                        render_pass_descriptor.note_copies(kept.copies());
                        for tiled in render_pass_descriptor.tiled() {
                            kept.takes().used_tiled(&tiled);
                        }
                        Some(RtObj::new(RtRenderEncoder::new(
                            encoder, mark, kept, targets,
                        )))
                    }
                    Err(error) => {
                        let what = render_pass_descriptor.detail(&pass);
                        super::unbuilt::refusal_detail(sel, &error, Some(&what));
                        rt_refused_create(sel, &error);
                        None
                    }
                })
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Nothing for 0; other options are unbuilt ([`Unbuilt::Protection`]), counted, and change
    /// nothing the GPU runs. They keep the command buffer whole ([`KeptWhole::ProtectionOptions`]).
    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("setProtectionOptions:", self, options);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            if options != 0 {
                refuse(Unbuilt::Protection, "setProtectionOptions:");
            }
            self.held.lock().protection = options;
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's status as Metal's, or the caller's failure of it (the module's docs). In a process
    /// that records, a final status, which rt sets before it runs the command buffer's handlers,
    /// is answered once the recorder took what its GPU work wrote
    /// ([`crate::recorder::completion_seen`]).
    fn status(&self) -> CommandBufferStatus {
        let status = status_of(&self.rt, &self.kept.3);
        if (status == CommandBufferStatus::COMPLETED || status == CommandBufferStatus::ERROR)
            && let Some(id) = crate::recorder::completion_key(self)
        {
            crate::recorder::completion_seen(id);
        }
        status
    }

    /// Ends the command buffer with `OutOfMemory` (the module's docs, "What a completed handler
    /// learns").
    fn fail_out_of_memory(&self) {
        self.kept.3.fail(ERROR_OUT_OF_MEMORY);
    }

    /// rt's wait, then the wait for its completed handlers handed on ([`hand_on`]); none,
    /// counted, where rt refuses it (in a scheduled handler, or on rt's completion thread for a
    /// command buffer not yet completed, which it would never see), and so on the queue of
    /// handlers handed on, as on rt's thread (the module's docs, "Handed on"). In a process that
    /// records, it returns once the recorder took what the command buffer's GPU work wrote
    /// ([`crate::recorder::completion_seen`]).
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("waitUntilCompleted", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            let handing = RUNS_HANDED.get();
            if handing && !finished(&self.rt) {
                let own = rt::Error::Refused(rt::Refusal::WaitOnOwnThread);
                rt_refused("waitUntilCompleted", &own);
                return Ok(());
            }
            match self.rt.wait_until_completed() {
                Ok(_) => {
                    if !handing {
                        self.handed.wait();
                    }
                    if let Some(id) = crate::recorder::completion_key(self) {
                        crate::recorder::completion_seen(id);
                    }
                }
                Err(error) => {
                    rt_refused("waitUntilCompleted", &error);
                }
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's wait; none, counted, where rt refuses it, as for
    /// [`wait_until_completed`](CommandBuffer::wait_until_completed).
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("waitUntilScheduled", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            if let Err(error) = self.rt.wait_until_scheduled() {
                rt_refused("waitUntilScheduled", &error);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A compute encoder of the descriptor's dispatch type, serial where it sets none: nil for a
    /// value no `MTLDispatchType` is, Metal's refusal, and where rt refuses it, counted.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
        let recorded_call =
            crate::recorder::rec_call!("computeCommandEncoderWithDescriptor:", self, desc);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
                let sel = "computeCommandEncoderWithDescriptor:";
                let dispatch_type = match desc.dispatch_type {
                    None | Some(0) => rt::DispatchType::Serial,
                    Some(1) => rt::DispatchType::Concurrent,
                    Some(other) => {
                        return Ok(refused_metal(sel, &format!("MTLDispatchType {other}")));
                    }
                };
                Ok(self.compute_encoder_of(sel, dispatch_type))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The work so far committed as a part, and the next part, where it may be cut (the
    /// module's docs); `None` otherwise, or where rt refuses the next part, counted, when it is
    /// kept whole.
    fn split(&self, on: OnFault) -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
        let recorded_call = crate::recorder::rec_call!("m2v.split", self);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
                if self.kept_whole_by().is_some() {
                    return Ok(None);
                }
                let Some(next) = self.next_part() else {
                    return Ok(None);
                };
                self.commit(on)?;
                Ok(Some(next))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A part made now and not committed, where it may be cut (the module's docs); `None`
    /// otherwise, or where rt refuses it, counted.
    fn part(&self, on: OnFault) -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
        let recorded_call = crate::recorder::rec_call!("m2v.part", self);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
                let _ = on;
                if self.kept_whole_by().is_some() {
                    return Ok(None);
                }
                Ok(self.next_part())
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Committed as a part where it may be cut ([`PartCut::Cut`]); else nothing, and why it is
    /// kept whole. Never [`PartCut::Fails`] (the module's docs).
    fn commit_part(&self, on: OnFault) -> RaiseResult<PartCut> {
        let mut recorded_call = crate::recorder::rec_call!("m2v.commitPart", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<PartCut> {
            if let Some(reason) = self.kept_whole_by() {
                return Ok(PartCut::Whole(reason));
            }
            self.commit(on)?;
            Ok(PartCut::Cut)
        });
        // Whether it was cut, which a player follows: `"Cut"` alone commits.
        if let Ok(cut) = &recorded_answer {
            crate::recorder::answer(&mut recorded_call, "cut", || {
                crate::recorder::text_value(cut)
            });
        }
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The Vulkan platform's fold with nothing noted: rt notes no way Metal's GPU fails a command
    /// buffer from an encoder on, so `commit_part` never answers [`PartCut::Fails`] and no caller
    /// folds. Lets `later` go uncommitted.
    fn fold(&self, later: <RtPlatform as Platform>::CommandBuffer) {
        let recorded_call = crate::recorder::rec_call!("m2v.fold", self, later);
        crate::recorder::run(|| {
            drop(later);
        });
        crate::recorder::rec_done!(recorded_call, ());
    }

    /// Keeps `escapes`, its process's escaped copies, which each of its commits syncs, and each
    /// part's cut from it (`super::copied`, the module's docs); the first named alone is kept.
    fn reaches_escaped(&self, escapes: &EscapedCopies) {
        let _ = self.escapes.set(escapes.clone());
    }

    /// Keeps `names`, the guest's names for its work, which its line names where rt ends it, or a
    /// part cut from it, as in flight at its device's loss (the module's docs, "A lost device's
    /// line"); the first named alone is kept.
    fn names_work(&self, names: &Arc<WorkNames>) {
        let _ = self.failure.work.set(Arc::clone(names));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;

    use super::*;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::CommandQueue as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::event::SharedEvent as _;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::NsRange;
    use crate::rt_platform::{RtBuffer, RtQueue};

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// rt's errors are Metal's, in Metal's command buffer domain, as Metal answers its GPU's
    /// recovery: of a lost device's work, the first command buffer lost in flight the hang,
    /// `Timeout` (2), once for the device, and every other its victim, `Internal` (1), as is a
    /// command buffer ended unrun; each command buffer keeps the answer of its first read. rt's
    /// statuses are Metal's values.
    #[test]
    fn rt_s_errors_and_statuses_are_metal_s() {
        use rt::CommandBufferError::{DeviceLost, NotRun};
        let codes = |errors: &[rt::CommandBufferError]| {
            let loss = DeviceLoss::new("a GPU".to_owned());
            (errors.iter())
                .map(|&e| ns_error(e, &loss, &|| None, None))
                .inspect(|e| assert_eq!(e.domain, ERROR_DOMAIN))
                .map(|e| e.code)
                .collect::<Vec<_>>()
        };
        let in_flight = DeviceLost { in_flight: true };
        let held = DeviceLost { in_flight: false };
        assert_eq!(codes(&[in_flight, in_flight, held]), [2, 1, 1]);
        assert_eq!(codes(&[held, NotRun, in_flight, in_flight]), [1, 1, 2, 1]);
        assert_eq!(codes(&[NotRun, held]), [1, 1]);
        // A command buffer keeps its answer; the device's next one lost in flight is a victim.
        let loss = Arc::new(DeviceLoss::new("a GPU".to_owned()));
        let failure = |loss: &Arc<DeviceLoss>| Failure {
            error: OnceLock::new(),
            loss: Arc::clone(loss),
            cause: Box::new(|| None),
            work: OnceLock::new(),
            flight: AtomicU64::new(0),
        };
        let (first, second) = (failure(&loss), failure(&loss));
        assert_eq!(first.read(None), None);
        let reads = [first.read(Some(in_flight)), second.read(Some(in_flight))];
        assert_eq!(reads.map(|e| e.map(|e| e.code)), [Some(2), Some(1)]);
        assert_eq!(first.read(Some(in_flight)).map(|e| e.code), Some(2));
        assert_eq!(second.read(Some(in_flight)).map(|e| e.code), Some(1));
        let statuses = [
            rt::CommandBufferStatus::NotEnqueued,
            rt::CommandBufferStatus::Enqueued,
            rt::CommandBufferStatus::Committed,
            rt::CommandBufferStatus::Scheduled,
            rt::CommandBufferStatus::Completed,
            rt::CommandBufferStatus::Error,
        ];
        assert_eq!(statuses.map(|s| status(s).0), [0, 1, 2, 3, 4, 5]);
    }

    /// A lost device writes one line, at the first of its command buffers that fails with it lost,
    /// naming its GPU, the Vulkan call rt names as its loss and what it returned, and how many of
    /// its command buffers were committed and not complete then, the failing one among them; each
    /// command buffer that fails so is counted, once however often its error is read (the
    /// module's docs, "A lost device's line"). The fault's lines follow it, here the one that rt
    /// read no report (`super::fault`). Where rt names no call, the line says so, and no fault's
    /// line follows.
    #[test]
    fn a_lost_device_writes_one_line_and_counts_each_command_buffer() {
        use rt::CommandBufferError::{DeviceLost, NotRun};
        let said = |gpu: &str| said_lost(&format!(" on {gpu} was lost "));
        let before = counted(DEVICE_LOST);
        let loss = Arc::new(DeviceLoss::new("A GPU (1234:5678)".to_owned()));
        let failure = || Failure {
            error: OnceLock::new(),
            loss: Arc::clone(&loss),
            cause: Box::new(|| {
                Some(rt::DeviceLoss {
                    call: "vkQueueSubmit2",
                    result: ash::vk::Result::ERROR_DEVICE_LOST,
                    fault: None,
                })
            }),
            work: OnceLock::new(),
            flight: AtomicU64::new(0),
        };
        let (first, second, third) = (failure(), failure(), failure());
        // Three committed (`RtCommandBuffer::commit`'s count).
        loss.open.fetch_add(3, Ordering::SeqCst);
        // A command buffer ended unrun is no loss.
        third.completed(Some(NotRun));
        assert_eq!(said("A GPU (1234:5678)"), Vec::<String>::new());
        assert_eq!(counted(DEVICE_LOST), before);
        first.completed(Some(DeviceLost { in_flight: true }));
        second.completed(Some(DeviceLost { in_flight: false }));
        // Read again, each keeps its answer and is not counted again.
        let reads = [first.read(Some(NotRun)), second.read(Some(NotRun))];
        assert_eq!(reads.map(|e| e.map(|e| e.code)), [Some(2), Some(1)]);
        assert_eq!(
            said("A GPU (1234:5678)"),
            [
                "reims-vgpu: the rt platform's device on A GPU (1234:5678) was lost at \
                 vkQueueSubmit2 (ERROR_DEVICE_LOST), with 2 command buffers committed and not \
                 complete: every command buffer fails from here on, and nothing makes the device \
                 again (rt-device-lost)\n"
            ]
        );
        assert_eq!(
            said_lost(" on A GPU (1234:5678) faulted: "),
            [
                "reims-vgpu: the rt platform's device on A GPU (1234:5678) faulted: rt read no \
                 report of the fault (the device has no VK_EXT_device_fault, the read failed, or a \
                 window's swapchain reported the loss)\n"
            ]
        );
        assert_eq!(counted(DEVICE_LOST), before + 2);
        assert_eq!(loss.open.load(Ordering::SeqCst), 0);
        // Another device, whose loss rt names no call for.
        let other = Arc::new(DeviceLoss::new("B GPU (1234:9abc)".to_owned()));
        other.open.fetch_add(1, Ordering::SeqCst);
        let unnamed = Failure {
            error: OnceLock::new(),
            loss: Arc::clone(&other),
            cause: Box::new(|| None),
            work: OnceLock::new(),
            flight: AtomicU64::new(0),
        };
        unnamed.completed(Some(DeviceLost { in_flight: true }));
        assert_eq!(
            said("B GPU (1234:9abc)"),
            [
                "reims-vgpu: the rt platform's device on B GPU (1234:9abc) was lost with no call \
                 rt named, with 1 command buffers committed and not complete: every command \
                 buffer fails from here on, and nothing makes the device again (rt-device-lost)\n"
            ]
        );
        assert_eq!(
            said_lost(" on B GPU (1234:9abc) faulted: "),
            Vec::<String>::new()
        );
    }

    /// Each command buffer rt ends as in flight at its device's loss writes a line after the
    /// device's, naming the guest's command buffer and the first compute and render pipeline
    /// states its encoders set, `none` for a kind it set none of, as its caller named them; one cut
    /// in parts, which share its names, writes one however many of them are lost in flight, and
    /// one its caller named nothing for says it is the host's own. One rt ends held, not in
    /// flight, writes none. Of the first [`IN_FLIGHT_NAMED`] each writes its line, the next that
    /// the rest are counted only, and each is counted once (the module's docs, "A lost device's
    /// line").
    #[test]
    fn each_command_buffer_lost_in_flight_names_the_guest_s_work() {
        use rt::CommandBufferError::DeviceLost;
        let gpu = "C GPU (1234:def0)";
        let before = counted(DEVICE_LOST_IN_FLIGHT);
        let loss = Arc::new(DeviceLoss::new(gpu.to_owned()));
        let failure = |work: Option<&Arc<WorkNames>>| {
            loss.open.fetch_add(1, Ordering::SeqCst);
            let failure = Failure {
                error: OnceLock::new(),
                loss: Arc::clone(&loss),
                cause: Box::new(|| None),
                work: OnceLock::new(),
                flight: AtomicU64::new(0),
            };
            if let Some(work) = work {
                let _ = failure.work.set(Arc::clone(work));
            }
            failure
        };
        let lost = |failure: Failure, in_flight: bool| {
            failure.completed(Some(DeviceLost { in_flight }));
        };
        let in_flight = || {
            (said_lost(&format!(" device on {gpu}")).into_iter())
                .filter(|line| line.contains(" in flight "))
                .collect::<Vec<_>>()
        };
        // Its encoders set compute pipeline states 0xb, then 0xc, and a render one, 0xd, after.
        let both = Arc::new(WorkNames::new(0x2a));
        both.note_firsts(0xb, 0);
        both.note_firsts(0xc, 0xd);
        assert_eq!(both.firsts(), (0xb, 0xd));
        // Its two parts lost in flight, one held, one that set neither, and one of the host's.
        lost(failure(Some(&both)), true);
        lost(failure(Some(&both)), true);
        lost(failure(Some(&Arc::new(WorkNames::new(0x2c)))), false);
        lost(failure(Some(&Arc::new(WorkNames::new(0x2b)))), true);
        lost(failure(None), true);
        let named = [
            "reims-vgpu: the guest's command buffer 0x2a was lost in flight on the rt platform's \
             device on C GPU (1234:def0), its encoders' first compute pipeline state 0xb and first \
             render pipeline state 0xd (rt-device-lost-in-flight)\n",
            "reims-vgpu: the guest's command buffer 0x2b was lost in flight on the rt platform's \
             device on C GPU (1234:def0), its encoders' first compute pipeline state none and \
             first render pipeline state none (rt-device-lost-in-flight)\n",
            "reims-vgpu: a command buffer of the host's own, which the guest does not number, was \
             lost in flight on the rt platform's device on C GPU (1234:def0) \
             (rt-device-lost-in-flight)\n",
        ];
        assert_eq!(in_flight(), named);
        assert_eq!(counted(DEVICE_LOST_IN_FLIGHT), before + 3);
        // Seven more: the first five write theirs, the sixth that the rest are counted only.
        for cb in 0x30..0x37 {
            lost(failure(Some(&Arc::new(WorkNames::new(cb)))), true);
        }
        let lines = in_flight();
        assert_eq!(lines.len(), IN_FLIGHT_NAMED + 1, "{lines:#?}");
        assert_eq!(lines[..3], named);
        assert!(lines[7].contains("command buffer 0x34 was lost in flight"));
        assert_eq!(
            lines[8],
            "reims-vgpu: more command buffers were lost in flight on the rt platform's device on \
             C GPU (1234:def0): the degradation counts give how many in all \
             (rt-device-lost-in-flight)\n"
        );
        assert_eq!(counted(DEVICE_LOST_IN_FLIGHT), before + 10);
    }

    /// The lost devices' lines the process wrote that hold `part`.
    fn said_lost(part: &str) -> Vec<String> {
        (SAID_LOST.lock().iter())
            .filter(|line| line.contains(part))
            .cloned()
            .collect()
    }

    /// On the GPU the test holds, a loss reported to rt (rt's `Device::report_loss`, as a window's
    /// swapchain reports one its own Vulkan call meets first) is the device's: it is lost, a
    /// command buffer committed after it ends with Metal's `MTLCommandBufferErrorInternal` (1),
    /// its status an error, as its completed handler is told, and the device's line names the
    /// reported call and result (the module's docs, "A lost device's line").
    #[test]
    fn a_loss_reported_to_rt_is_the_device_s_and_its_line_names_the_call_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let reported = " was lost at vkQueuePresentKHR (ERROR_DEVICE_LOST), with ";
        assert!(!device.is_lost());
        device.rt().report_loss(rt::DeviceLoss {
            call: "vkQueuePresentKHR",
            result: ash::vk::Result::ERROR_DEVICE_LOST,
            fault: None,
        });
        assert!(device.is_lost(), "the reported loss is the device's");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let told = Arc::new(Mutex::new(None));
        let tell = Arc::clone(&told);
        let see: CompletedHandler = Box::new(move |info: CompletedInfo| {
            *tell.lock() = Some((info.status, info.error));
        });
        assert_eq!(made.add_completed_handler(on, see), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        let answered = |e: Option<NsErrorText>| e.map(|e| (e.domain, e.code));
        let failed = (
            CommandBufferStatus::ERROR,
            Some((ERROR_DOMAIN.to_owned(), ERROR_INTERNAL)),
        );
        assert_eq!((made.status(), answered(made.error())), failed);
        let told = told.lock().take().expect("the completed handler ran");
        assert_eq!((told.0, answered(told.1)), failed);
        let lines = said_lost(reported);
        assert_eq!(lines.len(), 1, "{lines:?}");
    }

    /// On the GPU the test holds, a command buffer committed and not complete is on its device's
    /// list of those in flight, by the guest's id and its context, with its encoders' kinds in
    /// order and the buffers they bind by guest id; once it completes, it is off the list.
    #[test]
    fn a_command_buffer_in_flight_is_listed_with_its_context_and_encoders_on_vulkan() {
        use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let buffer = device.new_buffer_of_length(on, 64, 0);
        let buffer = buffer.expect("never a raise").expect("a buffer");
        super::super::fault::name_buffer(&buffer, 0x500_0000_0041);
        let shared = device.new_shared_event().expect("a shared event");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        made.names_work(&Arc::new(WorkNames::new(0x500_0000_0009)));
        assert_eq!(made.encode_wait(Some(EventRef::Shared(&shared)), 1), Ok(()));
        let blit = made.blit_encoder(on).expect("never a raise");
        let blit = blit.expect("a blit encoder");
        let fill = BlitCmd::FillBuffer {
            buffer: &buffer,
            range: NsRange::new(0, 16),
            value: 1,
        };
        assert_eq!(blit.encode(on, fill), Ok(()));
        assert_eq!(blit.encode(on, BlitCmd::EndEncoding), Ok(()));
        let compute = made.compute_encoder(on).expect("never a raise");
        let compute = compute.expect("a compute encoder");
        let bind = ComputeCmd::SetBuffers {
            buffers: &[Some(&buffer), None],
            offsets: &[0, 0],
            range: NsRange::new(0, 2),
        };
        assert_eq!(compute.encode(on, bind), Ok(()));
        assert_eq!(compute.encode(on, ComputeCmd::EndEncoding), Ok(()));
        assert_eq!(made.commit(on), Ok(()));

        let flights = &device.loss().flights;
        let gpu = "GPU (1234:def0)";
        let lines = flights.lines(gpu, super::super::fault::wall_now());
        assert_eq!(lines.len(), 2, "{lines:#?}");
        assert!(
            lines[0].starts_with(
                "reims-vgpu: the rt platform's device on GPU (1234:def0), at its loss, had 1 \
                 command buffers committed and not complete, the oldest committed first: the \
                 guest's command buffer 0x50000000009 of context 5, committed at "
            ),
            "{}",
            lines[0]
        );
        assert!(
            lines[0].ends_with(" s before the loss), its encoders (2): blit, compute\n"),
            "{}",
            lines[0]
        );
        assert_eq!(
            lines[1],
            "reims-vgpu: the oldest command buffer committed and not complete at the loss of the \
             rt platform's device on GPU (1234:def0), the guest's command buffer 0x50000000009 of \
             context 5: no render passes; the heaps its encoders declare (useHeap:): none; the \
             buffers they declare (useResource:): none; the textures they declare \
             (useResource:): none; the buffers they bind: 0x50000000041\n"
        );
        assert_eq!(shared.set_signaled_value(1), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        assert_eq!(made.status(), CommandBufferStatus::COMPLETED);
        let mut waited = 0;
        while !flights.is_empty() && waited < 1000 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            waited += 1;
        }
        assert!(flights.is_empty(), "{:#?}", flights.lines(gpu, 0));
    }

    /// On the GPU the test holds: a command buffer its caller failed for a deferred buffer's
    /// memory ends with Metal's `MTLCommandBufferErrorOutOfMemory` (8) once rt completes it, its
    /// status an error, as its completed handlers are told; one not failed completes with none
    #[test]
    fn a_command_buffer_failed_out_of_memory_ends_with_metal_s_error_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        for failed in [false, true] {
            let made = queue.command_buffer(on).expect("never a raise");
            let made = made.expect("a command buffer");
            let told = Arc::new(Mutex::new(None));
            let tell = Arc::clone(&told);
            let see: CompletedHandler = Box::new(move |info: CompletedInfo| {
                *tell.lock() = Some((info.status, info.error));
            });
            assert_eq!(made.add_completed_handler(on, see), Ok(()));
            if failed {
                made.fail_out_of_memory();
            }
            assert_eq!(made.commit(on), Ok(()));
            assert_eq!(made.wait_until_completed(on), Ok(()));
            let (status, error) = match failed {
                true => (
                    CommandBufferStatus::ERROR,
                    Some((ERROR_DOMAIN.to_owned(), ERROR_OUT_OF_MEMORY)),
                ),
                false => (CommandBufferStatus::COMPLETED, None),
            };
            let answered = |e: Option<NsErrorText>| e.map(|e| (e.domain, e.code));
            assert_eq!(
                (made.status(), answered(made.error())),
                (status, error.clone())
            );
            let told = told.lock().take().expect("the completed handler ran");
            assert_eq!((told.0, answered(told.1)), (status, error));
        }
    }

    /// On the GPU the test holds: a command buffer a dispatch of which its stage's fill skipped
    /// ends with Metal's `MTLCommandBufferErrorInternal` (1) once rt completes it, its status an
    /// error; the first failure set is the one it ends with, though its caller fails it for a
    /// deferred buffer's memory after
    #[test]
    fn a_command_buffer_a_skipped_dispatch_failed_ends_with_internal_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        made.kept.fail_internal();
        made.fail_out_of_memory();
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        let failed = (
            CommandBufferStatus::ERROR,
            Some((ERROR_DOMAIN.to_owned(), ERROR_INTERNAL)),
        );
        let answered = |e: Option<NsErrorText>| e.map(|e| (e.domain, e.code));
        assert_eq!((made.status(), answered(made.error())), failed);
    }

    /// Command buffers on the GPU the test holds: handlers in core's order whatever the order
    /// added, each completed handler told the status and GPU times; a handler added after the
    /// commit, and a second commit, refused and counted; kept whole once committed, and while it
    /// signals as scheduled; signals and waits on an
    /// event and a shared event across queues; a signal as scheduled set before the scheduled
    /// handlers added after it; nil events refused and counted; compute and render encoders
    /// unbuilt, nil and counted; protection options other than 0 counted; a wait in a scheduled
    /// handler refused, counted, and returning.
    #[test]
    fn command_buffers_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = |queue: &RtObj<RtQueue>| {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let metal = RtRefusal::Refused.code();

        // Handlers.
        let first = made(&queue);
        assert_eq!(first.status(), CommandBufferStatus::NOT_ENQUEUED);
        assert!(first.device().is_some_and(|d| d.id() == device.id()));
        let ran = Arc::new(Mutex::new(Vec::new()));
        let note = |name: &'static str| -> ScheduledHandler {
            let ran = Arc::clone(&ran);
            Box::new(move || ran.lock().push((name, None)))
        };
        let see = |name: &'static str| -> CompletedHandler {
            let ran = Arc::clone(&ran);
            Box::new(move |info: CompletedInfo| ran.lock().push((name, Some(info))))
        };
        assert_eq!(first.add_completed_handler(on, see("completed")), Ok(()));
        assert_eq!(first.add_scheduled_handler(on, note("scheduled")), Ok(()));
        assert_eq!(first.add_completion_seen_handler(on, see("seen")), Ok(()));
        assert_eq!(first.add_submission_handler(on, note("submission")), Ok(()));
        let whole = |why| -> RaiseResult<PartCut> { Ok(PartCut::Whole(why)) };
        let before = counted(metal);
        assert_eq!(first.commit(on), Ok(()));
        assert_eq!(first.add_completed_handler(on, see("late")), Ok(()));
        assert_eq!(first.commit(on), Ok(()));
        assert_eq!(counted(metal), before + 2);
        assert_eq!(first.wait_until_scheduled(), Ok(()));
        assert_eq!(first.wait_until_completed(on), Ok(()));
        assert_eq!(first.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(first.error(), None);
        assert_eq!(first.commit_part(on), whole(KeptWhole::Committed));
        let order = ran.lock();
        let names: Vec<&str> = order.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["submission", "scheduled", "seen", "completed"]);
        for (_, info) in order.iter().skip(2) {
            let info = info.as_ref().expect("a completed handler's info");
            assert_eq!(info.status, CommandBufferStatus::COMPLETED);
            assert_eq!(info.error, None);
            let (start, end) = info.gpu_times_ns.expect("GPU times");
            assert!(start != 0 && start <= end, "{start} {end}");
        }
        drop(order);

        // Events: the waits, on another queue, committed first.
        let event = device.new_event(on).expect("never a raise");
        let event = event.expect("an event");
        let shared = device.new_shared_event().expect("a shared event");
        let other = device.new_command_queue().expect("another command queue");
        let waits = made(&other);
        assert_eq!(waits.encode_wait(Some(EventRef::Event(&event)), 1), Ok(()));
        assert_eq!(
            waits.encode_wait(Some(EventRef::Shared(&shared)), 7),
            Ok(())
        );
        assert_eq!(waits.commit(on), Ok(()));
        let signals = made(&queue);
        assert_eq!(
            signals.encode_signal(Some(EventRef::Event(&event)), 1),
            Ok(())
        );
        assert_eq!(
            signals.encode_signal(Some(EventRef::Shared(&shared)), 5),
            Ok(())
        );
        assert_eq!(signals.commit(on), Ok(()));
        assert_eq!(signals.wait_until_completed(on), Ok(()));
        assert_eq!(signals.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(shared.signaled_value(), 5);
        assert_ne!(waits.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(shared.set_signaled_value(7), Ok(()));
        assert_eq!(waits.wait_until_completed(on), Ok(()));
        assert_eq!(waits.status(), CommandBufferStatus::COMPLETED);

        // A signal as scheduled, before the scheduled handler added after it.
        let at_scheduled = device.new_shared_event().expect("a shared event");
        let scheduled = made(&queue);
        assert_eq!(
            scheduled.encode_signal_when_scheduled(Some(&at_scheduled), 9),
            Ok(())
        );
        let saw = Arc::new(AtomicU64::new(0));
        let (seen, read) = (Arc::clone(&saw), at_scheduled.clone_ref());
        let look = move || seen.store(read.signaled_value(), Ordering::Release);
        assert_eq!(scheduled.add_scheduled_handler(on, Box::new(look)), Ok(()));
        assert_eq!(scheduled.commit_part(on), whole(KeptWhole::ScheduledSignal));
        assert_eq!(scheduled.commit(on), Ok(()));
        assert_eq!(scheduled.wait_until_completed(on), Ok(()));
        assert_eq!(saw.load(Ordering::Acquire), 9);

        // Nil events.
        let before = counted(metal);
        let nil = made(&queue);
        assert_eq!(nil.encode_signal(None, 1), Ok(()));
        assert_eq!(nil.encode_wait(None, 1), Ok(()));
        assert_eq!(nil.encode_signal_when_scheduled(None, 1), Ok(()));
        assert_eq!(counted(metal), before + 3);

        // A pass of no attachment at Metal's default sample count of 0, which Metal refuses, and
        // unbuilt protection options; the command buffer still runs.
        let refused = counted(metal);
        let protection = counted(Unbuilt::Protection.code());
        let pass = platform.new_render_pass_descriptor();
        assert!(matches!(nil.render_encoder(on, &pass), Ok(None)));
        assert_eq!(counted(metal), refused + 1);
        assert_eq!(nil.set_protection_options(on, 0), Ok(()));
        assert_eq!(counted(Unbuilt::Protection.code()), protection);
        assert_eq!(nil.set_protection_options(on, 1), Ok(()));
        assert_eq!(counted(Unbuilt::Protection.code()), protection + 1);
        assert_eq!(nil.commit(on), Ok(()));
        assert_eq!(nil.wait_until_completed(on), Ok(()));
        assert_eq!(nil.status(), CommandBufferStatus::COMPLETED);

        // A wait in a scheduled handler: rt's refusal on a rule of its own.
        let (ours, before) = (RtRefusal::Ours.code(), counted(metal));
        let ours_before = counted(ours);
        let waiter = made(&queue);
        let returned = Arc::new(AtomicBool::new(false));
        let (done, me) = (Arc::clone(&returned), waiter.clone_ref());
        let wait = move || {
            let waited = me.wait_until_completed(OnFault::Return);
            done.store(waited.is_ok(), Ordering::Release);
        };
        assert_eq!(waiter.add_scheduled_handler(on, Box::new(wait)), Ok(()));
        assert_eq!(waiter.commit(on), Ok(()));
        assert_eq!(waiter.wait_until_completed(on), Ok(()));
        assert!(returned.load(Ordering::Acquire));
        assert_eq!(counted(ours), ours_before + 1);
        assert_eq!(counted(metal), before);
    }

    /// Parts on the GPU the test holds: kept whole while an encoder is open; cut once it ended,
    /// each part an rt command buffer of the same queue, run in their order (a fill, a copy of it
    /// in the part made before the fill's commit, and a copy of that in the part `split`
    /// answered); never marked to join the next; an encoder let go of unended no longer open;
    /// kept whole under protection options, and once an encoder was made under them.
    #[test]
    fn parts_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let made = || {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let buffer = device.new_buffer_of_length(on, 48, 0);
        let buffer = buffer.expect("never a raise").expect("a buffer");
        let whole = |why| -> RaiseResult<PartCut> { Ok(PartCut::Whole(why)) };
        let encode = |made: &RtObj<RtCommandBuffer>, cmd: BlitCmd<'_, RtPlatform>| {
            let encoder = made.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            assert_eq!(encoder.encode(on, cmd), Ok(()));
            encoder
        };
        let fill = || BlitCmd::FillBuffer {
            buffer: &buffer,
            range: NsRange::new(0, 16),
            value: 0x5a,
        };
        let copy = |from: u64, to: u64| BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&buffer),
            source_offset: from,
            destination_buffer: Some(&buffer),
            destination_offset: to,
            size: 16,
        };
        let completed = CommandBufferStatus::COMPLETED;

        // Three parts.
        let first = made();
        let encoder = encode(&first, fill());
        assert!(matches!(first.part(on), Ok(None)));
        assert!(matches!(first.split(on), Ok(None)));
        assert_eq!(first.commit_part(on), whole(KeptWhole::OpenEncoder));
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        let second = first.part(on).expect("never a raise").expect("a part");
        let encoder = encode(&second, copy(0, 16));
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert!(!first.set_completion_joins_next(true));
        assert_eq!(first.commit_part(on), Ok(PartCut::Cut));
        assert_eq!(first.commit_part(on), whole(KeptWhole::Committed));
        let third = second
            .split(on)
            .expect("never a raise")
            .expect("the next part");
        assert_eq!(second.commit_part(on), whole(KeptWhole::Committed));
        let encoder = encode(&third, copy(16, 32));
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(third.commit(on), Ok(()));
        assert_eq!(third.wait_until_completed(on), Ok(()));
        let statuses = [&first, &second, &third].map(|part| part.status());
        assert_eq!(statuses, [completed; 3]);
        // SAFETY: a Shared buffer's contents are mapped, 48 bytes long.
        let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), 48) };
        assert_eq!(bytes, [0x5a; 48]);

        // An encoder let go of unended: rt ends it.
        let let_go = made();
        drop(encode(&let_go, fill()));
        assert_eq!(let_go.commit_part(on), Ok(PartCut::Cut));
        assert_eq!(let_go.wait_until_completed(on), Ok(()));
        assert_eq!(let_go.status(), completed);

        // Protection options.
        let protected = made();
        assert_eq!(protected.set_protection_options(on, 1), Ok(()));
        assert_eq!(
            protected.commit_part(on),
            whole(KeptWhole::ProtectionOptions)
        );
        assert_eq!(protected.set_protection_options(on, 0), Ok(()));
        assert!(matches!(protected.part(on), Ok(Some(_))));
        assert_eq!(protected.set_protection_options(on, 1), Ok(()));
        let encoder = encode(&protected, fill());
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(protected.set_protection_options(on, 0), Ok(()));
        assert_eq!(
            protected.commit_part(on),
            whole(KeptWhole::ProtectionOptions)
        );
        assert_eq!(protected.commit(on), Ok(()));
        assert_eq!(protected.wait_until_completed(on), Ok(()));
        assert_eq!(protected.status(), completed);
    }

    /// On the GPU the test holds, work committed on any command queue once a command buffer was
    /// scheduled runs after that command buffer, which is what reims-vgpu's present path rests on
    /// when a window's draw gives the guest its surfaces back at its scheduling
    /// (`DrawnPresents::after`): rt submits every command queue's work, an unshared queue's as a
    /// plain one's, to the device's one `VkQueue` from one thread in commit order, and ends each
    /// segment with a global memory barrier (rt's README, "Queues, command buffers and events").
    /// A command buffer of an unshared queue, the window sink's kind, copies a Shared buffer out
    /// behind 2 GiB of copies within a Private one. Once its scheduled handler has run, the test's
    /// thread commits fills of that Shared buffer on a plain queue and on another unshared one.
    /// The copy reads none of them. The command buffer presents nothing, so it is scheduled as it
    /// goes to the driver only while its queue has nothing in flight (that README section), as
    /// here; a window's draw presents a drawable, and is never held.
    #[test]
    fn work_committed_after_a_scheduling_runs_after_it_on_vulkan() {
        use crate::core::num::ToUsize as _;
        const HALF: u64 = 64 << 20;
        const READ: u64 = 4096;
        fn copy<'a>(
            (from, from_at): (&'a RtObj<RtBuffer>, u64),
            (to, to_at): (&'a RtObj<RtBuffer>, u64),
            size: u64,
        ) -> BlitCmd<'a, RtPlatform> {
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: from_at,
                destination_buffer: Some(to),
                destination_offset: to_at,
                size,
            }
        }
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let made = |queue: &RtObj<RtQueue>| {
            let made = queue.command_buffer(on).expect("never a raise");
            made.expect("a command buffer")
        };
        let buffer = |length: u64, options: u64| {
            let made = device.new_buffer_of_length(on, length, options);
            made.expect("never a raise").expect("a buffer")
        };
        let encode = |made: &RtObj<RtCommandBuffer>, cmds: Vec<BlitCmd<'_, RtPlatform>>| {
            let encoder = made.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            for cmd in cmds {
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        };
        let (slow, read, out) = (buffer(2 * HALF, 0x20), buffer(READ, 0), buffer(READ, 0));
        // SAFETY: a Shared buffer's contents are mapped, `READ` bytes long.
        let bytes = |buffer: &RtObj<RtBuffer>| unsafe {
            std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), READ.to_usize()).to_vec()
        };
        // SAFETY: as `bytes`'s; no command buffer uses `read` yet.
        unsafe { std::ptr::write_bytes(read.contents().as_ptr::<u8>(), 0x11, READ.to_usize()) };

        let sink = device
            .new_command_queue_unshared()
            .expect("an unshared queue");
        let reader = made(&sink);
        let mut cmds: Vec<_> = (0..32u64)
            .map(|n| copy((&slow, n % 2 * HALF), (&slow, (n + 1) % 2 * HALF), HALF))
            .collect();
        cmds.push(copy((&read, 0), (&out, 0), READ));
        encode(&reader, cmds);
        let (scheduled, signal) = std::sync::mpsc::channel();
        let note = Box::new(move || scheduled.send(()).expect("the test waits"));
        assert_eq!(reader.add_scheduled_handler(on, note), Ok(()));
        assert_eq!(reader.commit(on), Ok(()));
        let patience = std::time::Duration::from_secs(10);
        signal.recv_timeout(patience).expect("the reader scheduled");

        let plain = device.new_command_queue().expect("a queue");
        let other = device
            .new_command_queue_unshared()
            .expect("an unshared queue");
        let fills = [(&plain, 0, 0xee), (&other, READ / 2, 0xdd)].map(|(queue, at, value)| {
            let fill = made(queue);
            let range = NsRange::new(at, READ / 2);
            encode(
                &fill,
                vec![BlitCmd::FillBuffer {
                    buffer: &read,
                    range,
                    value,
                }],
            );
            assert_eq!(fill.commit(on), Ok(()));
            fill
        });
        for each in fills.iter().chain([&reader]) {
            assert_eq!(each.wait_until_completed(on), Ok(()));
            assert_eq!(each.status(), CommandBufferStatus::COMPLETED);
        }
        let half = READ.to_usize() / 2;
        let filled = [vec![0xee; half], vec![0xdd; half]].concat();
        assert_eq!(bytes(&read), filled, "the fills ran");
        let unfilled = vec![0x11; READ.to_usize()];
        assert_eq!(bytes(&out), unfilled, "the copy read before the fills");
    }
}
