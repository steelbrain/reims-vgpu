//! The rt platform's command buffers ([`RtCommandBuffer`]): rt's, each recorded as it is encoded
//! and handed to rt's submit thread at its commit.
//!
//! - **Handlers.** rt runs a command buffer's scheduled handlers on its submit thread before the
//!   submission goes to the driver, and its completed handlers on its completion thread, each
//!   kind in the order added. Core's submission handlers run before its scheduled handlers, and
//!   its completion-seen handlers before its completed handlers, whatever the order they were
//!   added in: so the command buffer holds each kind until its commit, which hands rt one
//!   scheduled handler that runs the submission handlers and then the scheduled ones, and one
//!   completed handler that runs the completion-seen handlers and then the completed ones. A
//!   submission handler therefore runs on rt's submit thread, never inside the commit. A handler
//!   added after the commit is Metal's refusal (its validation stops it): counted
//!   (`rt-refused-metal`), and it never runs.
//! - **What a completed handler learns**: rt's status and error then, and its GPU times, in
//!   nanoseconds of rt's host clock (`metal2vulkan_rt::host_time`); none where either is 0,
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
//!   for the device, naming its GPU (name, PCI vendor and device IDs) and how many of its command
//!   buffers were committed and not complete then, the one failing among them ([`DeviceLoss`]).
//!   Each command buffer that fails with the device lost is counted, `rt-device-lost`, in the
//!   process's degradation counts (docs/conventions.md).
//!   One its caller failed for a deferred buffer's memory rt refused
//!   (`CommandBuffer::fail_out_of_memory`, `super::buffer`'s "Deferred memory") ends with
//!   `OutOfMemory` (8) where rt gave it no error of its own: its status, once rt's is completed,
//!   is an error, and its error, then, that one.
//! - **Encoders**: blit encoders ([`super::blit`]), compute encoders ([`super::compute`]), of
//!   the serial dispatch type or the descriptor's (another value is Metal's refusal, nil), and
//!   render encoders ([`super::render`]) of the pass the descriptor keeps
//!   ([`super::descriptor`]): a pass Metal refuses, or rt does, is nil, counted.
//! - **Copies of host pages** (`super::copied`): its encoders note each copy they set or declare
//!   an object over, as Metal makes resident only those; its commit copies the CPU's writes into
//!   those before rt is handed the command buffer, and where it used one its first completed
//!   handler merges the GPU's writes back into their pages, before core's completion-seen
//!   handlers.
//! - **What it keeps**: the pipeline states its encoders set ([`Kept`]), from their set until it
//!   completes, released after its completed handlers ran.
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
//!   is scheduled, and once committed. rt builds no indirect command buffer, rate map, residency
//!   or command predication, so no other reason holds; and it notes no way Metal's GPU fails a
//!   command buffer from an encoder on, so `commit_part` never answers [`PartCut::Fails`] and
//!   `fold`, the Vulkan platform's with nothing noted, lets the later part go. The guest buffer's
//!   status, error, GPU times and handlers are core's, from its parts.
//! - **Protection options** other than 0 are unbuilt ([`Unbuilt::Protection`]): rt protects no
//!   work, and the command buffer runs as one without them.
//!
//! Every call rt refuses answers as Metal's answers where it fails (nothing encoded, nil, no
//! wait), counted under the refusal's kind ([`rt_refused`]); never a death.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use metal2vulkan_rt as rt;

use crate::core::platform::command::{
    CommandBuffer, CompletedHandler, CompletedInfo, KeptWhole, PartCut, ScheduledHandler,
    run_completed,
};
use crate::core::platform::event::EventRef;
use crate::core::platform::types::{CommandBufferStatus, NsErrorText};
use crate::core::platform::values::ComputePassDesc;
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::refusals;
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::rt::BlockEnv;
use crate::core::sync::Mutex;

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

/// `MTLCommandBufferErrorOutOfMemory`.
const ERROR_OUT_OF_MEMORY: i64 = 8;

/// A handler as rt runs it.
type Handler = Box<dyn FnOnce(&rt::CommandBuffer) + Send>;

/// `id<MTLCommandBuffer>` on metal2vulkan-rt.
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
}

/// A command buffer's error as Metal's (the module's docs, "What a completed handler learns"):
/// made at its first read, which a completed handler its commit adds ahead of core's makes, and
/// kept.
struct Failure {
    /// Its error as Metal's, once read.
    error: OnceLock<NsErrorText>,
    /// Its device's loss ([`RtDevice::loss`]).
    loss: Arc<DeviceLoss>,
}

impl Failure {
    /// The answer of rt's `error`, made at the first that is not `None` and kept.
    fn read(&self, error: Option<rt::CommandBufferError>) -> Option<NsErrorText> {
        let error = error?;
        Some(
            self.error
                .get_or_init(|| ns_error(error, &self.loss))
                .clone(),
        )
    }

    /// Its first completed handler's part: reads its error, then takes it off its device's
    /// command buffers not complete ([`DeviceLoss`]).
    fn completed(&self, error: Option<rt::CommandBufferError>) {
        drop(self.read(error));
        self.loss.open.fetch_sub(1, Ordering::SeqCst);
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
    /// Its command buffers committed and not complete: a commit adds one before rt has it (and
    /// takes it off where rt refuses it), and its first completed handler takes it off once its
    /// error is read, so a failing command buffer counts itself.
    open: AtomicI64,
}

impl DeviceLoss {
    /// The loss of a device of the GPU `gpu`, before any.
    pub fn new(gpu: String) -> DeviceLoss {
        DeviceLoss {
            gpu,
            hung: AtomicBool::new(false),
            said: AtomicBool::new(false),
            open: AtomicI64::new(0),
        }
    }

    /// A command buffer failed with the device lost: counted, and the first writes the loss's
    /// line (the module's docs, "A lost device's line").
    fn lost(&self) {
        crate::answer::count_kind(DEVICE_LOST);
        if self.said.swap(true, Ordering::SeqCst) {
            return;
        }
        let open = self.open.load(Ordering::SeqCst).max(0);
        say_lost(&format!(
            "reims-vgpu: the rt platform's device on {} was lost, with {open} command buffers \
             committed and not complete: every command buffer fails from here on, and nothing \
             makes the device again ({DEVICE_LOST})\n",
            self.gpu
        ));
    }
}

#[cfg(test)]
thread_local! {
    /// The lost devices' lines this thread wrote.
    static SAID_LOST: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Writes a lost device's `line` on stderr, as the platform's other degradation lines
/// (`super::unbuilt`); a test reads those its thread wrote.
fn say_lost(line: &str) {
    super::unbuilt::say(line);
    #[cfg(test)]
    SAID_LOST.with(|said| said.borrow_mut().push(line.to_owned()));
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
}

impl RtCommandBuffer {
    /// The command buffer over rt's `made`, of `queue` and `device`.
    pub fn new(
        made: rt::CommandBuffer,
        queue: rt::CommandQueue,
        device: RtObj<RtDevice>,
    ) -> RtCommandBuffer {
        let failure = Arc::new(Failure {
            error: OnceLock::new(),
            loss: Arc::clone(device.loss()),
        });
        RtCommandBuffer {
            rt: made,
            queue,
            device,
            held: Mutex::new(Held::default()),
            open: Arc::default(),
            kept: Kept::default(),
            failure,
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
                Some(RtObj::new(RtComputeEncoder::new(
                    encoder, mark, kept, concurrent,
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
            // rt builds no indirect command buffer, rate map, residency or command predication.
            KeptWhole::Execution | KeptWhole::Progress | KeptWhole::Predication => false,
        };
        KeptWhole::ALL.into_iter().find(|&reason| holds(reason))
    }

    /// The next part, made on its queue: waits, as `commandBuffer` does, while the queue's most
    /// command buffers are made and not yet completed. `None` where rt refuses it (a quiesced
    /// device, a wait for room on rt's own handler thread), counted: then it is kept whole.
    fn next_part(&self) -> Option<RtObj<RtCommandBuffer>> {
        match self.queue.command_buffer() {
            Ok(made) => Some(RtObj::new(RtCommandBuffer::new(
                made,
                self.queue.clone(),
                self.device.clone_ref(),
            ))),
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
/// ([`Failed`]), and the copies of host pages it used, which its commit and completion sync
/// (`super::copied`).
#[derive(Clone, Default)]
pub struct Kept(
    Arc<Mutex<HashMap<ObjId, Box<dyn Send>>>>,
    Uses,
    Takes,
    Failed,
    super::copied::Used,
);

/// Whether a command buffer's caller failed it for a deferred buffer's memory rt refused
/// (`CommandBuffer::fail_out_of_memory`; the module's docs, "What a completed handler learns").
#[derive(Clone, Default)]
struct Failed(Arc<AtomicBool>);

impl Failed {
    /// Whether the command buffer was failed so.
    fn holds(&self) -> bool {
        self.0.load(Ordering::Acquire)
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

    /// An encoder of it used `texture` as `how` says: a use for the boot diagnostics
    /// ([`Uses::add`]), of a tiled texture whose bytes are to take, a take at its commit
    /// ([`Takes::used`]), and of a texture over a copy, the copy's use (`super::copied::Used`).
    pub fn used(&self, texture: &RtObj<RtTexture>, how: Use) {
        self.1.add(texture, how);
        self.2.used(texture);
        self.4.add(texture.copied());
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

/// `made`'s error as Metal's: rt's, as `failure` answers it, else `OutOfMemory` once completed
/// where `failed` holds (the module's docs, "What a completed handler learns").
fn error_of(made: &rt::CommandBuffer, failure: &Failure, failed: &Failed) -> Option<NsErrorText> {
    if let Some(error) = failure.read(made.error()) {
        return Some(error);
    }
    let completed = matches!(made.status(), rt::CommandBufferStatus::Completed);
    (completed && failed.holds()).then(|| NsErrorText {
        domain: ERROR_DOMAIN.to_owned(),
        code: ERROR_OUT_OF_MEMORY,
        description: "a buffer the command buffer used was refused its memory at that use"
            .to_owned(),
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
/// that fails with its device lost tells `loss` ([`DeviceLoss::lost`]).
fn ns_error(error: rt::CommandBufferError, loss: &DeviceLoss) -> NsErrorText {
    if let rt::CommandBufferError::DeviceLost { .. } = error {
        loss.lost();
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
        description: format!("metal2vulkan-rt's command buffer failed: {error:?}"),
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
    /// used copied in first and merged back first of the completed handlers (`super::copied`),
    /// then the bytes of the tiled textures it used taken (`super::tiled`). A second commit is
    /// Metal's refusal, counted; a commit rt refuses is counted, and its handlers never run.
    fn commit(&self, on: OnFault) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("commit", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            let sel = "commit";
            let (mut submission, scheduled, mut seen, completed) = {
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
                )
            };
            submission.extend(scheduled);
            seen.extend(completed);
            // Its error is made before core's handlers run, as rt completes it, so a lost
            // device's hang is the first it ends in flight (the module's docs).
            let failure = Arc::clone(&self.failure);
            seen.insert(
                0,
                Box::new(move |made: &rt::CommandBuffer| failure.completed(made.error())),
            );
            let copies = self.kept.copies().take();
            if let Some(merge) = super::copied::merging(&copies) {
                seen.insert(0, Box::new(merge));
            }
            self.kept.uses().submitted();
            let kept = self.kept.take();
            if !kept.is_empty() {
                seen.push(Box::new(move |_made: &rt::CommandBuffer| drop(kept)));
            }
            let run = |handlers: Vec<Handler>| {
                move |made: &rt::CommandBuffer| {
                    handlers.into_iter().for_each(|handler| handler(made))
                }
            };
            let committed = || -> Result<(), rt::Error> {
                if !submission.is_empty() {
                    self.rt.add_scheduled_handler(run(submission))?;
                }
                self.rt.add_completed_handler(run(seen))?;
                super::copied::copy_in(&copies);
                self.kept.takes().take(&self.device);
                // Not complete until its first completed handler runs, which a refusal never runs.
                let loss = &self.failure.loss;
                loss.open.fetch_add(1, Ordering::SeqCst);
                let committed = self.rt.commit();
                if committed.is_err() {
                    loss.open.fetch_sub(1, Ordering::SeqCst);
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
            if let Some(event) = rt_event(sel, event)
                && let Err(error) = self.rt.encode_signal_event(event, value)
            {
                rt_refused(sel, &error);
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
            let event = rt::SharedEvent::clone(event);
            let signal = move |_made: &rt::CommandBuffer| event.set_signaled_value(value);
            if self.hold(sel, |h| &mut h.scheduled, Box::new(signal)) {
                self.held.lock().scheduled_signal = true;
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
            if let Some(event) = rt_event(sel, event)
                && let Err(error) = self.rt.encode_wait_for_event(event, value)
            {
                rt_refused(sel, &error);
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

    /// rt's status as Metal's, or the caller's failure of it (the module's docs).
    fn status(&self) -> CommandBufferStatus {
        status_of(&self.rt, &self.kept.3)
    }

    /// Ends the command buffer with `OutOfMemory` (the module's docs, "What a completed handler
    /// learns").
    fn fail_out_of_memory(&self) {
        self.kept.3.0.store(true, Ordering::Release);
    }

    /// rt's wait; none, counted, where rt refuses it (in a scheduled handler, or on rt's
    /// completion thread for a command buffer not yet completed, which it would never see).
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("waitUntilCompleted", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            if let Err(error) = self.rt.wait_until_completed() {
                rt_refused("waitUntilCompleted", &error);
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
    use crate::rt_platform::RtQueue;

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
                .map(|&e| ns_error(e, &loss))
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
    /// naming its GPU and how many of its command buffers were committed and not complete then,
    /// the failing one among them; each command buffer that fails so is counted, once however
    /// often its error is read (the module's docs, "A lost device's line").
    #[test]
    fn a_lost_device_writes_one_line_and_counts_each_command_buffer() {
        use rt::CommandBufferError::{DeviceLost, NotRun};
        let said = || SAID_LOST.with(|said| said.borrow().clone());
        let before = counted(DEVICE_LOST);
        let loss = Arc::new(DeviceLoss::new("A GPU (1234:5678)".to_owned()));
        let failure = || Failure {
            error: OnceLock::new(),
            loss: Arc::clone(&loss),
        };
        let (first, second, third) = (failure(), failure(), failure());
        // Three committed (`RtCommandBuffer::commit`'s count).
        loss.open.fetch_add(3, Ordering::SeqCst);
        // A command buffer ended unrun is no loss.
        third.completed(Some(NotRun));
        assert_eq!(said(), Vec::<String>::new());
        assert_eq!(counted(DEVICE_LOST), before);
        first.completed(Some(DeviceLost { in_flight: true }));
        second.completed(Some(DeviceLost { in_flight: false }));
        // Read again, each keeps its answer and is not counted again.
        let reads = [first.read(Some(NotRun)), second.read(Some(NotRun))];
        assert_eq!(reads.map(|e| e.map(|e| e.code)), [Some(2), Some(1)]);
        assert_eq!(
            said(),
            [
                "reims-vgpu: the rt platform's device on A GPU (1234:5678) was lost, with 2 command \
                 buffers committed and not complete: every command buffer fails from here on, and \
                 nothing makes the device again (rt-device-lost)\n"
            ]
        );
        assert_eq!(counted(DEVICE_LOST), before + 2);
        assert_eq!(loss.open.load(Ordering::SeqCst), 0);
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

        // A pass of no attachment, which rt renders none of, and unbuilt protection options; the
        // command buffer still runs.
        let gaps = counted(RtRefusal::Gap.code());
        let protection = counted(Unbuilt::Protection.code());
        let pass = platform.new_render_pass_descriptor();
        assert!(matches!(nil.render_encoder(on, &pass), Ok(None)));
        assert_eq!(counted(RtRefusal::Gap.code()), gaps + 1);
        assert_eq!(nil.set_protection_options(on, 0), Ok(()));
        assert_eq!(counted(Unbuilt::Protection.code()), protection);
        assert_eq!(nil.set_protection_options(on, 1), Ok(()));
        assert_eq!(counted(Unbuilt::Protection.code()), protection + 1);
        assert_eq!(nil.commit(on), Ok(()));
        assert_eq!(nil.wait_until_completed(on), Ok(()));
        assert_eq!(nil.status(), CommandBufferStatus::COMPLETED);

        // A wait in a scheduled handler.
        let before = counted(metal);
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
        assert_eq!(counted(metal), before + 1);
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
}
