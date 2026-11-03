//! `id<MTLCommandQueue>` and `id<MTLCommandBuffer>`: core's `CommandQueue` and `CommandBuffer`
//! over the runtime's.
//!
//! - **Handlers** run on the runtime's handler threads, in the queue's order. A completed
//!   handler reads its buffer's `status` and `error` when it runs, as on the Metal platform, and
//!   runs marked as ours, so a panic in it aborts. Its read of a failure that dies (below) is the
//!   code of the isolated task that added it, if one did: once that task ended, nothing dies
//!   there, as core's handlers of the task run none ([`completed_handler`]).
//! - **`encodeSignalEventScheduled:value:`** is the runtime's: the event's value is set as the
//!   buffer is scheduled, before its status reads `Scheduled` and before `waitUntilScheduled`
//!   returns, as Metal's; sent once the buffer is committed, it sets nothing, as Metal's, and once
//!   it completed it dies, where Metal's ends the process.
//! - **A nil event**, which the runtime's calls cannot take, dies at the send.
//! - **An open encoder** kills the command buffer's sends that Metal asserts at while one is
//!   ("with uncommitted encoder"): a new encoder, an event, `setProtectionOptions:`, `commit`.
//! - **Where a command buffer is** ([`Progress`]) is kept for what changes once the GPU has run
//!   it: a rate map's reset is taken again once the passes that took it completed, the indirect
//!   command buffers its commands use keep their numbers until then, and what is kept beside
//!   their commands moves once its blits ran ([`crate::gpu::icb`]). A completed handler of this
//!   platform's keeps it, added the first time an encoder needs it, so before the handlers core
//!   adds once it has encoded.
//! - **The records of an indirect command buffer an execution runs** are checked by the runtime
//!   as they stand at commit, or, where the GPU may have written them, once the work before the
//!   execution completed. A refusal at commit dies there naming the runtime's refusal, and says
//!   it is of the records where the runtime names an execution's ([`committing`]). The later one
//!   is a host step's, which fails the command buffer with `CommandBufferError::Refused`, as does
//!   one of an indirect dispatch's arguments: Metal runs those commands and fails no command
//!   buffer so, and it dies naming the runtime's refusal (`CommandBuffer::refusal`), never a
//!   failure the guest could see. It dies as the command buffer completes, before any handler of
//!   core's learns of it, at the selector of its first execution of an indirect command buffer
//!   ([`Ending`]), and at `commit` where its status or error is read first ([`failure`]). An
//!   isolated task's command buffer that completes after the task ended dies at neither, as
//!   core's handlers of the task run none then.
//! - **An execution's commands that decide a death** (a draw of points under a state whose points
//!   Metal sizes none of; a command whose state can encode indirect commands in an encoder that
//!   made resident for writing a buffer whose ID is not the runtime's) are read at the commit, as
//!   the runtime runs them, and die there naming the execution and the command; an execution of a
//!   buffer whose records the GPU may have written is asked at the runtime's host step instead,
//!   which reads them once the work before it completed, and dies there, at `commit`, at a
//!   command that decides a death ([`executions`]). An encoder the command buffer's failure runs
//!   nothing of is not asked.
//! - **A draw's or dispatch's pipeline the driver did not make**, which the runtime makes while
//!   the command goes on and reports as the command buffer's failure
//!   (`CommandBufferError::PipelineCreation`), dies
//!   naming the driver's error ([`failed_pipeline`]) before anything of the failure reaches
//!   core: at `commit` where its status or error is read first, which a completed handler's
//!   does before it runs core's block ([`failure`]), and at `waitUntilCompleted` as the wait
//!   returns, as the runtime runs the completed handlers before its waiters return. Metal makes
//!   a state's pipelines as it makes the state, and fails no command buffer so.
//! - **An indexed indirect draw's indices past the command buffer's index pool**
//!   (`CommandBufferError::IndexPoolOverflow`) die there too ([`pool_overflow`]): the runtime
//!   checks such a draw's indices, or rewrites a strip's that restarts at another index than the
//!   all-ones one, into a pool before the render pass, a sixty-fourth of the device's largest
//!   device-local heap, and draws nothing of a draw past it, where Metal
//!   draws it.
//! - **A sample through a cubic sampler an argument buffer holds**
//!   (`CommandBufferError::CubicArgumentSampler`) dies there too ([`cubic_argument_sampler`]):
//!   the runtime does not filter bicubically through an argument buffer's sampler, and fails a
//!   command buffer whose function reached one, where Metal samples and fails none. A function
//!   that never reaches one runs as Metal's.
//! - **A command whose buffer over the guest's pages was rebound before its command buffer was
//!   submitted, in an encoder the runtime cannot encode again**
//!   (`CommandBufferError::EncodedUseRebound`) dies there too ([`encoded_use_rebound`]): the
//!   runtime resolves a command's use of a buffer over the caller's host pages as the command is
//!   encoded, and where the buffer is rebound (a backing replaced, `crate::gpu::backing`) before
//!   the command buffer's first submission, it encodes the compute, blit and render encoders that
//!   named it again from their calls, so they reach the new pages as Metal's do (a call refused at
//!   its encoding is refused again, and the calls after it run), and a resource state encoder takes
//!   no pages (an indirect mapping update reads its buffer as it runs). It fails only an encoder it
//!   cannot encode again: a compute encoder of the runtime's command predication that shares its
//!   unit, holds a loop or leaves work out (no record of ours reaches the runtime's predication:
//!   this platform's is its own, [`crate::gpu::predication`]); a render pass that executes an
//!   indirect command buffer, which the runtime records later, or whose end captured its draws'
//!   tile reach; and one a call of which answers otherwise than at its encoding. An encoder whose
//!   dispatch, draw or tile dispatch had its pipeline still being made at the commit is encoded
//!   again as any other, its pipeline's making left to the submission: the replace waits for no
//!   making. Each submission holding such an encoder runs nothing and the command
//!   buffer fails, where Metal resolves the use as it submits and runs the command over the pages
//!   the buffer lies in then. The runtime announces the first such failure (`EncodedUseRebound`,
//!   [`crate::answer::DIES`]) on the thread that makes the submission, where it dies first, ending
//!   the process.
//! - **Every other failure of the runtime's own**, where Metal fails no command buffer, dies
//!   there too, naming it ([`answered`]): a mesh draw's grid the device cannot launch where Metal
//!   draws it (`MeshGridPastLimits`, which dies as unanswered); an emulated mesh draw past the
//!   command buffer's mesh records pool (`MeshRecordsOverflow`), or an emulated tessellation draw
//!   past the same pool (`TessellationRecordsOverflow`, an indirect patch draw on a device without
//!   `tessellationShader`), which it draws nothing of; a Vulkan failure of a submission other
//!   than a lost device or memory (`Internal`); and any other the runtime gives, whatever origin
//!   it states, until one is reviewed here. Metal's own failures, the fourteen the runtime states as Metal's (`FailureOrigin::Metal`) and this
//!   platform lists, reach core as Metal's: those this platform commits a command buffer with as
//!   Metal's GPU fails it (below), a lost device, memory, an address fault, a GPU hang
//!   (`GpuHang`, code 2) and a victim of the GPU's recovery (`GpuRecoveryVictim`, code 1) at a
//!   draw of primitive type 9 or of an object grid of more than 2^29 threadgroups, which Metal's
//!   GPU discards (the runtime fails a direct draw's command buffer from its encoder on, as
//!   Metal's, and an indirect one's where its arguments pass reads the grid, the rest of its work
//!   running), a queue Metal stopped serving after its second such fault or hang
//!   (`AccessRevoked`, code 4), a render pass without attachments past 16384 or with a
//!   `defaultRasterSampleCount` Metal fails, a detached buffer's use, an encoder made under
//!   protection options (`ProtectedEncoder`, code 16 or 1), a unit of the runtime's command
//!   predication left with a block open (`PredicationOpenAtUnitEnd`, code 1), a memoryless
//!   texture's use by a command (`InvalidResource`, code 9, which the runtime fails the Metal
//!   submission holding it with, from its first encoder), and the driver's relay of one of
//!   Metal's codes. No protection option and no record of predication reaches
//!   the runtime here (below), so neither of those two fails a command buffer yet.
//! - **A failure of the GPU's work that the driver relayed** (`CommandBufferError::Execution`, and
//!   `PageFault` and `DeviceLost` of the same kind): MoltenVK runs the runtime's batches as Metal
//!   command buffers, and the runtime gives a failed one's Metal code to the command buffer whose
//!   batch it was, the device kept, as Metal keeps it. Core sees that code as Metal's failure of
//!   the command buffer where it is one of Metal's ([`METAL_CODES`]); any other dies where the
//!   failure is first read, as a pipeline's does ([`relayed`]). Metal's description of the
//!   failure, as the driver's message gave it, goes beside the code in the error's description,
//!   which core's log lines of a failed command buffer print, and in the death's line
//!   ([`command_buffer_error`]).
//! - **A wait the runtime refuses** as one that would not return hangs where Metal's never
//!   returns, and dies where Metal's answer is not measured, by the cause the runtime names
//!   ([`refused_wait`]).
//! - **Command predication**'s units of compute encoders end at a blit or render encoder, an
//!   event and the commit ([`predication::boundary`]).
//! - **How Metal's GPU fails a command buffer for what its encoders did** is injected at its
//!   commit ([`rt::CommandBuffer::commit_failed`]): Metal's code, the first encoder that runs
//!   nothing, every one after it running nothing too, and whether its event signals fire
//!   ([`Failing`]). Measured on the M3 Max (macOS 26.7), `setProtectionOptions:` is kept here,
//!   as the runtime has no protected content ([`CommandBufferObject::made_encoder`]). An encoder
//!   made under options other than 0 runs nothing, nor does any after it, and fails the command
//!   buffer with code 16 ("Protection Violation") below 0x10 and `MTLCommandBufferErrorInternal`
//!   ("Invalid Input") from 0x10 up, none of its signals firing; options from 0x10 up fail it so
//!   with no encoder made under them too, every encoder running. An encoder made before the
//!   options runs. Under 1, 3 and 7 a compute or render encoder's creation, and a blit encoder's
//!   first command, die, as Metal ends the process there. A protection failure reaches no later
//!   command buffer: one committed on its queue while it is in flight runs, on Metal (measured
//!   alike) as here.
//!
//!   A render encoder's draw of primitive type 9, at which Metal's GPU hangs, faults or fails a
//!   victim of its recovery, the runtime fails the command buffer at as Metal's GPU does
//!   (`crate::gpu::encode_render`): a command buffer of a hang's queue committed while it is in
//!   flight runs nothing and completes without an error, as Metal's
//!   (`rt::DrawPrimitiveType::Nine`; measured on the M3 Max under macOS 26.7 and 27.0.1,
//!   `metal_runs_after_a_type_9_hang_here`, `later_command_buffers_of_a_hang_run_as_metals`),
//!   one of a fault's or a victim's runs, and the queue's second hang or fault to complete
//!   revokes it: a command buffer committed on it after that fails at its commit with
//!   `AccessRevoked` (4), running nothing and signalling nothing, as Metal's
//!   (`type_9_faults_revoke_queues_as_metals`). Such a draw is told to the command buffer
//!   ([`CommandBufferObject::fails_at_type_9`]): one that fails another way too, by its own
//!   encoders, its protection options, a use of a detached object at its commit, or another
//!   type-9 draw's other failure, dies at the commit, as do two different ways at once of any
//!   kind, which nothing measured.
//! - **A submission that uses a detached buffer or heap** fails with
//!   `MTLCommandBufferErrorOutOfMemory` (8), whose description is Metal's ("Insufficient
//!   Memory"), as Metal fails it. Metal splits a command buffer into submissions, and decides
//!   each as it submits it, once the event waits before it are met (measured on the M3 Max,
//!   macOS 26.7, `tests/detach_backing.rs`):
//!   - every render pass is a submission of its own;
//!   - a run of consecutive blit encoders is one, and so is a run of consecutive compute
//!     encoders;
//!   - an event wait or signal ends a run.
//!
//!   The runtime splits a command buffer so, and checks each submission as it submits it, once
//!   its event waits are met ([`backing::submission_check`]): a wait another command buffer's
//!   GPU signal meets, once that signal landed, as Metal checks it once the signal happened. One
//!   that uses a detached object, or one released detached since the command buffer was made,
//!   runs nothing; every other submission runs, the event signals fire, the scheduled and
//!   completed handlers run, and the command buffer completes in its queue's order with status
//!   `Error`. The use can be a binding, a `useResource:` or `useHeap:`, a blit's operand, a
//!   command of an indirect command buffer it executes, or a texture over the buffer, a view of
//!   that texture, or a resource placed in the heap, as the runtime lists a submission's uses.
//!   This platform differs from Metal where:
//!   - the object is replaced after the command was encoded and before its command buffer's
//!     first submission (committed after the replace, or committed before it and held behind an
//!     event wait), in an encoder the runtime cannot encode again (the failure above): Metal
//!     runs the submission over the new pages. The runtime resolved the command's use as it
//!     encoded it, and fails the command buffer (`CommandBufferError::EncodedUseRebound`), which
//!     dies ([`encoded_use_rebound`]); a compute, blit or render encoder it encodes again
//!     reaches the new pages, as Metal's. The runtime decides each submission so as it submits
//!     it, after the check: one the check fails, a use detached then, keeps code 8, as Metal's,
//!     and a later one rebound so dies all the same, the command buffer keeping that code;
//!   - a submission that executes an indirect command buffer the GPU may write is checked again
//!     once the runtime recorded what its commands reach, and failing then, only what had not
//!     run yet runs nothing. Metal decides it once, before any of it runs.
//!
//!   A command that reaches a detached object only through an address it loads names nothing the
//!   check sees, so its submission runs, as Metal's does, reading 0 and dropping the write, on
//!   every host: the runtime's detach leaves the object in no memory, replaced or never replaced,
//!   and out of what such a command reaches until a command buffer uses it after a replace
//!   (`backing`'s docs).
//!
//!   A texture over a buffer or placed in a heap, and a buffer placed in a heap, hold the buffer
//!   or heap, as Metal's do, so a detached one is released only with the last of them, and a use
//!   through one fails until then ([`crate::gpu::resource::Under`]).
//!
//!   A command buffer Metal's GPU fails for what its encoders did is decided at its commit, with
//!   no check: one that uses a detached object then dies there, as one failing two ways does.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use metal2vulkan::metal::DispatchType;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::command::{
    CommandBuffer, CommandQueue, CompletedHandler, CompletedInfo, ScheduledHandler, run_completed,
};
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
use reims_vgpu_core::platform::values::ComputePassDesc;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::refusals::{self, Carried};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::tasks::isolation::TaskLife;

use crate::answer::{
    Policy, hang, metal_aborts, plain, plain_at, refusal, refused, send, unrepresentable,
    unstopped_get,
};
use crate::gpu::backing::{self, Opened};
use crate::gpu::desc::VulkanRenderPassDescriptor;
use crate::gpu::device::VulkanDevice;
use crate::gpu::encode_blit::{EncoderObject, VulkanBlitEncoder};
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::encode_render::VulkanRenderEncoder;
use crate::gpu::event::VulkanSharedEvent;
use crate::gpu::icb::VulkanIcb;
use crate::gpu::metal_space::{self, FirstUse};
use crate::gpu::predication::{self, Predication};
use crate::gpu::timing;
use crate::platform::{Obj, VulkanPlatform};

pub(crate) mod executions;

/// A runtime command queue, with the device it came from.
#[derive(Debug)]
pub struct QueueObject {
    /// The runtime's queue.
    pub raw: rt::CommandQueue,
    /// The device that made it.
    pub device: VulkanDevice,
}

impl QueueObject {
    /// The queue of `raw`, made by `device`.
    pub(crate) fn new(raw: rt::CommandQueue, device: VulkanDevice) -> QueueObject {
        QueueObject { raw, device }
    }
}

/// `id<MTLCommandQueue>`.
pub type VulkanCommandQueue = Obj<QueueObject>;

/// A runtime command buffer, with the device it came from.
#[derive(Debug)]
pub struct CommandBufferObject {
    /// The runtime's command buffer.
    pub raw: rt::CommandBuffer,
    /// The device of its queue.
    pub device: VulkanDevice,
    /// How many of its encoders are open: `setProtectionOptions:`, which the runtime answers
    /// whatever is open, dies while one is.
    pub open_encoders: Arc<AtomicUsize>,
    /// Its progress, once an encoder needed it ([`CommandBufferObject::progress`]).
    progress: Mutex<Option<Arc<Progress>>>,
    /// Its command predication.
    pub(crate) predication: Arc<Predication>,
    /// Its place among the releases of detached objects until it commits, which its submission
    /// check then holds ([`backing::submission_check`]).
    opened: Mutex<Option<Opened>>,
    /// How many runtime encoders it has made: each encoder of Metal's, or a part of one where
    /// [`predication`] ends a compute encoder's runtime encoder and makes another. As no two are
    /// open at once, each one's count before it is its place among the encoders the runtime
    /// ended, which [`rt::CommitFailure::encoders`] counts.
    made: AtomicUsize,
    /// How Metal's GPU fails it for what its encoders did ([`Failing`]), which its commit fails
    /// it with ([`CommandBufferObject::failure`]).
    failing: Mutex<Vec<Failing>>,
    /// How Metal's GPU fails it at its draws of primitive type 9, which the runtime fails it
    /// with itself ([`CommandBufferObject::fails_at_type_9`]).
    type_9: Mutex<Vec<Failing>>,
    /// The protection options `setProtectionOptions:` set, which the runtime does not hold.
    protection: Mutex<Protection>,
    /// Whether it carries the submission handler that takes Metal's allocation for its work for
    /// the tiler ([`tiler_handler`]), added at its first such work while the process has not
    /// taken it ([`CommandBufferObject::tiler_work`]).
    tiler: AtomicBool,
    /// The executions of indirect command buffers whose commands its commit reads, each with its
    /// encoder's place ([`CommandBufferObject::checked_at_commit`]).
    executions: Mutex<Vec<(usize, executions::Execution)>>,
}

/// The submission handler of a command buffer that holds work for Metal's tiler
/// ([`CommandBufferObject::tiler_work`]): Metal's allocation for that work. Metal takes its own
/// allocation for the first such work in the process as it submits it to the GPU
/// ([`FirstUse::TilerSubmission`]; `vaprobe rendersubmit`, plain Metal, M3 Max: none of it at the
/// commit of a render pass behind an event wait, all of it by the command buffer's completion).
/// The runtime runs the handler as the command buffer's last batch reaches its queue, on the
/// thread that submits it, before the command buffer reads scheduled, before
/// `waitUntilScheduled` returns and before any scheduled or completed handler runs. A command
/// buffer encoded once the process took the allocation carries none: the allocation is held for
/// good, and a submission handler makes the runtime run the command buffer's transition to
/// scheduled on a handler thread, woken for it, where it runs it on its scheduler thread as it
/// submits the command buffer (`rt::WorkCounts::scheduled_where_submitted`). Every command
/// buffer this platform commits carries a submission check ([`backing::submission_check`]), which
/// the runtime runs on its scheduler thread, so that thread submits it, once its waits are met
/// and the work ahead of it on its queue is submitted: after `commit` returns, as Metal submits
/// on a thread of its own and tells the program only through scheduled (the runtime's
/// measurement on an M4). A range placed between the commit and the submission lies by the threads'
/// timing, on Metal as here. A command buffer the runtime fails before its submission takes
/// nothing.
fn tiler_handler() -> rt::Handler {
    Box::new(|_| metal_space::first_use(FirstUse::TilerSubmission))
}

/// `MTLCommandBufferErrorInternal`, which Metal's GPU fails a command buffer with under
/// protection options with a bit from 0x10 up ("Invalid Input").
pub(crate) const INVALID_INPUT: u64 = 1;

/// The error code Metal's GPU fails a command buffer with where an encoder was made under
/// protection options below 0x10 ("Protection Violation"), which no public
/// `MTLCommandBufferError` names.
pub(crate) const PROTECTION_VIOLATION: u64 = 16;

/// The codes Metal fails a command buffer with: each `MTLCommandBufferError` but `None`, and
/// [`PROTECTION_VIOLATION`].
pub(crate) const METAL_CODES: [u64; 11] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, PROTECTION_VIOLATION];

/// A way Metal's GPU fails a command buffer for what one of its encoders did: with `code`, from
/// the encoder at `from` on, its event signals firing where `signals` says (the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Failing {
    /// The `MTLCommandBufferError` code.
    pub code: u64,
    /// The first encoder that runs nothing, in the order the runtime ended them.
    pub from: usize,
    /// Whether its event signals fire.
    pub signals: bool,
}

/// What `setProtectionOptions:` left on a command buffer.
#[derive(Debug, Default, Clone, Copy)]
struct Protection {
    /// The options set last: 0 until one is set.
    options: u64,
    /// Whether an encoder was made under options other than 0.
    used: bool,
}

/// Whether protection options end the process where an encoder is made under them: 1, 3 and 7,
/// at a compute or render encoder's creation and a blit encoder's first command (measured on
/// the M3 Max: SIGABRT, no message).
fn aborts_encoders(options: u64) -> bool {
    matches!(options, 1 | 3 | 7)
}

impl CommandBufferObject {
    /// Work for Metal's tiler encoded by `sel`, `first`: a render encoder made, or a blit's reset
    /// or copy of an indirect command buffer. Metal's own allocations for it the first time, and
    /// the command buffer's submission a first use too, which its submission handler takes, added
    /// here while the process has not taken it ([`tiler_handler`]). Nothing here stops a thread
    /// of a task that ended: a render encoder's runtime encoder is not in its handle yet
    /// (`crate::gpu::encode_render`'s `make`), and dropped it would abandon the command buffer.
    pub(crate) fn tiler_work(&self, sel: &'static str, first: FirstUse) {
        metal_space::first_use(first);
        if !metal_space::taken(FirstUse::TilerSubmission)
            && !self.tiler.swap(true, Ordering::SeqCst)
        {
            unstopped_get(sel, || self.raw.add_submission_handler(tiler_handler()));
        }
    }

    /// Dies at `sel`, the creation of a compute or render encoder, under protection options that
    /// end the process there ([`aborts_encoders`]).
    pub(crate) fn protected_creation(&self, sel: &'static str) {
        let options = plain(|| self.protection.lock().options);
        if aborts_encoders(options) {
            metal_aborts(
                sel,
                &format!("an encoder made under protection options {options}"),
            );
        }
    }

    /// A runtime encoder it made, for an encoder of Metal's (`whole`) or a part of one: its place
    /// among the encoders the runtime ends. An encoder of Metal's made under protection options
    /// other than 0 fails the command buffer from it, with
    /// [`PROTECTION_VIOLATION`] below 0x10 and [`INVALID_INPUT`] from there up, its event
    /// signals firing none; under options that end the process at its first command (a blit
    /// encoder's, [`aborts_encoders`]) it answers them, which that command dies at. Nothing here
    /// stops a thread of a task that ended: the encoder is not in its handle yet, and dropped
    /// it would abandon the command buffer.
    pub(crate) fn made_encoder(&self, whole: bool) -> (usize, Option<u64>) {
        let at = self.made.fetch_add(1, Ordering::SeqCst);
        if !whole {
            return (at, None);
        }
        let options = {
            let mut protection = self.protection.lock();
            if protection.options != 0 {
                protection.used = true;
            }
            protection.options
        };
        if options == 0 {
            return (at, None);
        }
        if aborts_encoders(options) {
            return (at, Some(options));
        }
        self.fail(Failing {
            code: protection_code(options),
            from: at,
            signals: false,
        });
        (at, None)
    }

    /// Records `failing`, which its commit fails it with ([`CommandBufferObject::failure`]).
    pub(crate) fn fail(&self, failing: Failing) {
        self.failing.lock().push(failing);
    }

    /// Records `failing`, how a draw of primitive type 9 in it fails it, as Metal's GPU does,
    /// which the runtime fails it with itself (`crate::gpu::encode_render`'s `nine`): its
    /// commit dies where it fails another way too ([`CommandBufferObject::failure`]).
    pub(crate) fn fails_at_type_9(&self, failing: Failing) {
        plain(|| self.type_9.lock().push(failing));
    }

    /// `execution`, a command of the encoder it made last, is checked at its commit, where its
    /// commands' records stand as the runtime runs them ([`executions`]).
    pub(crate) fn checked_at_commit(&self, execution: executions::Execution) {
        let at = self.made.load(Ordering::SeqCst).saturating_sub(1);
        plain(|| self.executions.lock().push((at, execution)));
    }

    /// Checks at its commit the executions [`CommandBufferObject::checked_at_commit`] took, in
    /// their order, but those of an encoder `failed` runs nothing of; each whose records it read
    /// for a death is held as read until it ends ([`executions::reads`]).
    fn check_executions(&self, failed: Option<&rt::FailedEncoders>, progress: Option<&Progress>) {
        let taken = std::mem::take(&mut *self.executions.lock());
        let runs = |at: usize| match failed {
            Some(rt::FailedEncoders::From(from)) => at < *from,
            Some(rt::FailedEncoders::Set(set)) => !set.contains(&at),
            _ => true,
        };
        let read: Vec<_> = taken
            .into_iter()
            .filter(|(at, _)| runs(*at))
            .filter_map(|(_, execution)| execution.check())
            .collect();
        if let Some(progress) = progress {
            for icb in &read {
                executions::reads(icb, progress);
            }
        }
    }

    /// How Metal's GPU fails it, at its commit (`sel`): what its encoders did ([`Failing`]),
    /// and protection options with a bit from 0x10 up, which fail it with [`INVALID_INPUT`]
    /// though no encoder was made under them (measured), all of its encoders running; `None`
    /// where nothing does, or only its draws of primitive type 9, which the runtime fails it at
    /// ([`CommandBufferObject::fails_at_type_9`]). Where either fails it, a use of a detached
    /// object at the commit (`detached`, asked only then) is another way. Two different ways at
    /// once, which nothing measured, die. A use of a detached object alone fails only its
    /// submissions ([`backing::submission_check`]).
    fn failure(
        &self,
        sel: &'static str,
        detached: impl FnOnce() -> bool,
    ) -> Option<rt::CommitFailure> {
        let mut ways = plain(|| self.failing.lock().clone());
        let options = plain(|| self.protection.lock().options);
        if protection_code(options) == INVALID_INPUT {
            ways.push(Failing {
                code: INVALID_INPUT,
                from: self.made.load(Ordering::SeqCst),
                signals: false,
            });
        }
        let own = ways.len();
        ways.extend(plain(|| self.type_9.lock().clone()));
        let first = *ways.first()?;
        if detached() {
            ways.push(Failing {
                code: backing::DETACHED_USE,
                from: 0,
                signals: true,
            });
        }
        if ways
            .iter()
            .any(|w| (w.code, w.signals) != (first.code, first.signals))
        {
            unrepresentable(
                sel,
                &format!("a command buffer Metal's GPU fails in more than one way: {ways:?}"),
            );
        }
        ways.truncate(own);
        let first = *ways.first()?;
        let mut failure = rt::CommitFailure::new(first.code);
        let from = ways.iter().map(|w| w.from).min().unwrap_or(first.from);
        failure.encoders = rt::FailedEncoders::From(from);
        failure.signals = first.signals;
        Some(failure)
    }

    /// Its progress ([`Progress`]), for `sel`, an encoder's command: made the first time one asks,
    /// with the completed handler that ends it, which then comes before every handler core adds
    /// once it has encoded. An encoder's command comes before the commit, which refuses an open
    /// encoder, so the handler's refusal after the commit never comes.
    pub(crate) fn progress(&self, sel: &'static str) -> Arc<Progress> {
        let mut kept = plain(|| self.progress.lock());
        if let Some(progress) = &*kept {
            return Arc::clone(progress);
        }
        let progress = Arc::new(Progress::default());
        let ending = Ending {
            progress: Some(Arc::clone(&progress)),
            task: TaskLife::current(),
            carried: refusals::carry(),
            #[cfg(test)]
            slot: reims_vgpu_core::testkit::capture::current_slot(),
        };
        let handler: rt::Handler = Box::new(move |cb: &rt::CommandBuffer| ending.run(cb));
        plain(|| self.raw.add_completed_handler(handler)).unwrap_or_else(|e| refused(sel, &e));
        *kept = Some(Arc::clone(&progress));
        progress
    }
}

/// Where a command buffer is, as [`Progress`] sees it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Stage {
    /// Not committed yet.
    #[default]
    Encoding,
    /// Committed, not yet complete: what the GPU has run of it is unknown.
    Committed,
    /// Completed: the GPU ran it all.
    Completed,
    /// Failed: what the GPU ran of it is unknown, for good.
    Failed,
    /// Gone without a commit: the GPU ran none of it.
    Dropped,
}

/// What this platform keeps of one command buffer until the GPU has run it: its [`Stage`], what
/// its commands use until then, and what changes once it completed or failed.
#[derive(Default)]
pub struct Progress {
    state: Mutex<ProgressState>,
}

impl std::fmt::Debug for Progress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Progress")
            .field("stage", &self.stage())
            .finish_non_exhaustive()
    }
}

/// A command buffer's [`Progress`], under its lock.
#[derive(Default)]
struct ProgressState {
    stage: Stage,
    /// What its commands use, held until it completes, fails or goes, as Metal's command buffer
    /// retains them.
    held: Vec<Box<dyn std::any::Any + Send + Sync>>,
    /// What changes once it completed, in the order its commands were encoded.
    changes: Vec<Box<dyn FnOnce() + Send>>,
    /// What changes once it failed, in the order its commands were encoded.
    failures: Vec<Box<dyn FnOnce() + Send>>,
    /// The indirect command buffers its commands execute, with the executions' selectors, held as
    /// `held` is.
    executions: Vec<(&'static str, VulkanIcb)>,
}

impl Progress {
    /// Where the command buffer is now.
    pub(crate) fn stage(&self) -> Stage {
        plain(|| self.state.lock().stage)
    }

    /// Whether the GPU may still run some of it: it is not committed, or committed and not
    /// complete.
    pub(crate) fn running(&self) -> bool {
        matches!(self.stage(), Stage::Encoding | Stage::Committed)
    }

    /// `object` is held until the command buffer completes, fails or goes.
    pub(crate) fn hold(&self, object: impl std::any::Any + Send + Sync) {
        plain(|| self.state.lock().held.push(Box::new(object)));
    }

    /// An execution `sel` of `icb`, which is held until the command buffer completes, fails or
    /// goes.
    pub(crate) fn executes(&self, sel: &'static str, icb: VulkanIcb) {
        plain(|| self.state.lock().executions.push((sel, icb)));
    }

    /// The selector of its first execution of an indirect command buffer, if its commands
    /// execute one.
    fn first_execution(&self) -> Option<&'static str> {
        self.state.lock().executions.first().map(|(sel, _)| *sel)
    }

    /// `change` runs once the command buffer completed, after the changes added before it and
    /// before its stage is set; it is dropped unrun when the command buffer fails or goes.
    pub(crate) fn at_completion(&self, change: impl FnOnce() + Send + 'static) {
        plain(|| self.state.lock().changes.push(Box::new(change)));
    }

    /// `change` runs once the command buffer failed, as [`Progress::at_completion`]'s once it
    /// completed; it is dropped unrun when the command buffer completes or goes.
    pub(crate) fn at_failure(&self, change: impl FnOnce() + Send + 'static) {
        plain(|| self.state.lock().failures.push(Box::new(change)));
    }

    /// The commit begins: no thread stop can come between this and the runtime's commit, so a
    /// command buffer never committed never shows as committed.
    fn committing(&self) {
        let mut state = self.state.lock();
        if state.stage == Stage::Encoding {
            state.stage = Stage::Committed;
        }
    }

    /// The command buffer ends at `stage`: its changes run if it completed, its failures' if it
    /// failed, what it held goes, then the stage is set. It runs on the runtime's thread, or
    /// where the command buffer goes, and sends nothing, so it checks no task's end. Each change
    /// runs, and each object held is let go of, in the scope `carried` names, the one the
    /// command that made the progress carried (the graphics lane's): a death in one is refused
    /// there and logged, and the rest run and go, and the stage is set, all the same. Outside
    /// every scope each runs and goes as it always did, in the same order, and a death in one
    /// ends the process.
    fn end(&self, stage: Stage, carried: Carried) {
        let (changes, failures, held, executions) = {
            let mut state = self.state.lock();
            (
                std::mem::take(&mut state.changes),
                std::mem::take(&mut state.failures),
                std::mem::take(&mut state.held),
                std::mem::take(&mut state.executions),
            )
        };
        let ran = match stage {
            Stage::Completed => {
                drop(failures);
                changes
            }
            Stage::Failed => {
                drop(changes);
                failures
            }
            Stage::Encoding | Stage::Committed | Stage::Dropped => {
                drop((changes, failures));
                Vec::new()
            }
        };
        let refused = |what: &str, refusal: refusals::Refusal| {
            log::warn!(
                target: "reims_vgpu::refused",
                "a command buffer's {what} as it ended was refused: {refusal}"
            );
        };
        for change in ran {
            if let Err(refusal) = carried.run(change) {
                refused("change", refusal);
            }
        }
        for object in held {
            if let Err(refusal) = carried.run(move || drop(object)) {
                refused("release of what its commands used", refusal);
            }
        }
        for execution in executions {
            if let Err(refusal) = carried.run(move || drop(execution)) {
                refused("release of an indirect command buffer it executed", refusal);
            }
        }
        self.state.lock().stage = stage;
    }
}

/// The completed handler that ends a command buffer's [`Progress`]; dropped unrun, as the runtime
/// drops the handlers of a command buffer that goes uncommitted, it ends it as gone.
struct Ending {
    /// The progress it ends, until it has.
    progress: Option<Arc<Progress>>,
    /// The isolated task whose code made the progress, if one's did: the command buffer is the
    /// task's, and its death in the handler is the task's code ([`as_task`]).
    task: Option<Arc<TaskLife>>,
    /// The refusal scope of the op whose command made the progress (the graphics lane's), which
    /// the death in the handler refuses instead of ending the process; none outside every scope.
    /// The command buffer's own completed handlers then see its refused step as their failure.
    carried: Carried,
    /// The test's death slot, which a death in the handler lands in.
    #[cfg(test)]
    slot: Option<Arc<reims_vgpu_core::testkit::capture::DeathSlot>>,
}

impl Ending {
    /// The handler, on the runtime's handler thread, marked ours: the command buffer completed or
    /// failed.
    fn run(mut self, cb: &rt::CommandBuffer) {
        let _env = BlockEnv::foreign();
        let stage = if cb.status() == rt::CommandBufferStatus::Error {
            Stage::Failed
        } else {
            Stage::Completed
        };
        let die = |sel| {
            refused_step(sel, cb);
        };
        let refused = cb.error() == Some(rt::CommandBufferError::Refused);
        self.end(stage, refused.then_some(die));
    }

    /// Ends the progress at `stage`, after `refused`'s death at the selector of its first
    /// execution of an indirect command buffer where a host step refused the command buffer and
    /// its commands execute one. The death is the task's code, so none comes once the task ended;
    /// in the scope the progress carried, it is refused there, logged, and the progress ends.
    fn end(&mut self, stage: Stage, refused: Option<impl Fn(&'static str)>) {
        let Some(progress) = self.progress.take() else {
            return;
        };
        let task = self.task.take();
        let carried = self.carried;
        let end = || {
            if let Some(refused) = refused
                && let Some(sel) = progress.first_execution()
                && let Err(refusal) = carried.run(|| as_task(task.as_ref(), || refused(sel)))
            {
                log::warn!(
                    target: "reims_vgpu::refused",
                    "a refused command buffer's execution of an indirect command buffer: {refusal}"
                );
            }
            progress.end(stage, carried);
        };
        #[cfg(test)]
        reims_vgpu_core::testkit::capture::with_slot(self.slot.take(), end);
        #[cfg(not(test))]
        end();
    }
}

/// Runs `f` as the code of `task`, the isolated task a command buffer is of (none outside
/// isolation), as core carries its completed handlers (`TaskLife::carry`): not at all once the
/// task ended, so that a task's command buffer that ends after it kills nothing. A platform death
/// in `f` still ends the process, as on the Metal platform a crash inside Metal does.
fn as_task(task: Option<&Arc<TaskLife>>, f: impl FnOnce()) {
    match task {
        Some(life) => {
            let _ = life.run(f);
        }
        None => f(),
    }
}

/// Runs the commit `call`: a refusal of the records of an indirect command buffer the command
/// buffer executes ([`of_records`]), which the runtime checks as they stand at commit, dies at
/// the commit, saying so and naming the runtime's refusal. Any other refusal is the caller's to
/// decide, which dies at the commit naming the runtime's refusal too ([`send`]): of the command
/// buffer's state, of the heaps' or function tables' entries, or of what else the runtime records
/// then.
fn committing(call: impl FnOnce() -> Result<(), rt::Error>) -> Result<(), rt::Error> {
    match call() {
        Err(e) if of_records(&e) => fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: "commit",
            },
            &format!(
                "the Vulkan runtime refused the records of an indirect command buffer the \
                 command buffer executes, which it checks as they stand at commit: {e}"
            ),
        ),
        made => made,
    }
}

/// Whether `e`, a refusal of a commit, is of the records of an indirect command buffer the
/// command buffer executes: the runtime names the method a refusal is of by its Metal selector,
/// and those of the records by an execution's (`execute_commands_in_buffer` and its indirect
/// form). A refusal it gives no method (a runtime gap, a compile) is named alone.
fn of_records(e: &rt::Error) -> bool {
    match e {
        rt::Error::Invalid(invalid) => invalid.call.starts_with("execute_commands_in_buffer"),
        _ => false,
    }
}

impl Drop for Ending {
    fn drop(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.end(Stage::Dropped, self.carried);
        }
    }
}

/// Runs `call`, the command buffer's `sel`, as [`send`] does, but for an encoder still open:
/// Metal asserts there ("with uncommitted encoder": `exec_open_encoder_at_commit`,
/// `render_segments_open_twice`), so it dies whatever handler is live.
fn encoding<T>(
    sel: &'static str,
    policy: Policy,
    call: impl FnOnce() -> Result<T, rt::Error>,
) -> RaiseResult<T> {
    send(sel, policy, || match call() {
        Err(rt::Error::Invalid(invalid)) if matches!(invalid.problem, rt::Problem::EncoderOpen) => {
            uncommitted_encoder(sel)
        }
        made => made,
    })
}

/// The death of `sel`, `encodeSignalEventScheduled:value:`, sent once its command buffer
/// completed or failed, where Metal's process ends with a segmentation fault.
fn signal_after_completion(sel: &'static str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::OutOfBounds,
            call: sel,
        },
        "Metal ends the process with a segmentation fault at a scheduled signal of a command \
         buffer that completed",
    )
}

/// The death of `sel` while an encoder of its command buffer is open, as Metal asserts.
fn uncommitted_encoder(sel: &'static str) -> ! {
    metal_aborts(
        sel,
        "a command buffer with an open encoder (\"with uncommitted encoder\")",
    )
}

/// Runs `call`, the wait `sel` for a command buffer (or for a new one of a queue), as [`send`]
/// does with `Dies`, but for a wait the runtime refuses as one that would not return
/// (`WaitWouldDeadlock`), decided by [`refused_wait`].
fn waiting<T>(sel: &'static str, call: impl FnOnce() -> Result<T, rt::Error>) -> RaiseResult<T> {
    send(sel, Policy::Dies, || match call() {
        Err(rt::Error::WaitWouldDeadlock(cause)) => {
            match refused_wait(cause, rt::process_exiting) {
                RefusedWait::Hangs => hang(),
                RefusedWait::Unmeasured => unmeasured_wait(sel, cause),
            }
        }
        made => made,
    })
}

/// The death of the wait `sel` the runtime refused for `cause`, as [`RefusedWait::Unmeasured`].
fn unmeasured_wait(sel: &'static str, cause: rt::DeadlockCause) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        &format!(
            "the Vulkan runtime refused the wait, as {cause}, where Metal's answer is not \
             measured"
        ),
    )
}

/// What a wait the runtime refuses does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RefusedWait {
    /// It hangs, as Metal's never returns there: this thread parks for good ([`hang`]).
    Hangs,
    /// It dies: Metal's answer there is not measured.
    Unmeasured,
}

/// What a wait the runtime refuses for `cause` does, the process `exiting` or not:
/// - a handler's wait for its own command buffer or a later one of its queue, and a new command
///   buffer in the queue's own handler while all 64 are held (`OwnQueue`): Metal's never
///   returns there (measured), and a hang stays a hang;
/// - any wait for a command buffer not yet finished once the process is exiting (`Exiting`), on
///   any thread: Metal's goes on waiting, and the exit ends the thread with the process, so the
///   thread parks (the library registers no exit handler that waits);
/// - a handler's wait for another queue's command buffer that would leave the device's handler
///   threads none free (`HandlerThreadCap`), the 512th such wait: Metal ran 511 at once, and at
///   the 512th its process's dispatch threads ran out and it never returned (measured on the M3
///   Max), so a hang stays a hang. Metal's other handlers stop there too, where the runtime's go
///   on, and Metal counts the process's threads, the runtime each device's;
/// - a handler's wait for another queue's command buffer that would close a cycle of handlers'
///   waits (`Cycle`), and any cause the runtime adds: Metal's answer is not measured, so it
///   dies, unless the process is exiting, when it parks as an exiting process's wait does. Only
///   these ask whether the process is exiting.
fn refused_wait(cause: rt::DeadlockCause, exiting: impl FnOnce() -> bool) -> RefusedWait {
    match cause {
        rt::DeadlockCause::OwnQueue
        | rt::DeadlockCause::HandlerThreadCap
        | rt::DeadlockCause::Exiting => RefusedWait::Hangs,
        _ if exiting() => RefusedWait::Hangs,
        _ => RefusedWait::Unmeasured,
    }
}

/// `id<MTLCommandBuffer>`.
pub type VulkanCommandBuffer = Obj<CommandBufferObject>;

/// The code Metal's GPU fails a command buffer with where an encoder was made under protection
/// `options`, not 0: [`INVALID_INPUT`] where a bit from 0x10 up is set, else
/// [`PROTECTION_VIOLATION`] (measured on the M3 Max over every option below 0x10 but 1, 3 and 7,
/// and 0x10 to 0x100, 0x11, 0x13, `u32::MAX`, `1 << 63`, `u64::MAX - 1` and `u64::MAX`). 0 for
/// none.
fn protection_code(options: u64) -> u64 {
    match options {
        0 => 0,
        o if o >= 0x10 => INVALID_INPUT,
        _ => PROTECTION_VIOLATION,
    }
}

impl CommandQueue<VulkanPlatform> for VulkanCommandQueue {
    /// Blocks while the queue holds 64 uncompleted command buffers, as Metal's does, and hangs
    /// where none can complete first ([`waiting`]).
    fn command_buffer(&self, _on: OnFault) -> RaiseResult<Option<VulkanCommandBuffer>> {
        let raw = waiting("commandBuffer", || self.raw.command_buffer())?;
        let predication = Predication::new(&self.device);
        Ok(Some(Obj::new(CommandBufferObject {
            raw,
            device: self.device.clone_ref(),
            open_encoders: Arc::default(),
            progress: Mutex::new(None),
            predication,
            opened: Mutex::new(Some(Opened::new())),
            made: AtomicUsize::new(0),
            failing: Mutex::new(Vec::new()),
            type_9: Mutex::new(Vec::new()),
            protection: Mutex::new(Protection::default()),
            tiler: AtomicBool::new(false),
            executions: Mutex::new(Vec::new()),
        })))
    }

    fn set_label(&self, label: &str) {
        plain(|| self.raw.set_label(label));
    }
}

/// Why `cb` failed, as the runtime says: a host step's refusal dies at `commit`, whose commands
/// it recorded, naming it, as Metal runs the command the runtime refused; so does a draw's
/// pipeline the driver did not make ([`failed_pipeline`]), and every other failure that is none
/// of Metal's ([`answered`]).
fn failure(cb: &rt::CommandBuffer) -> Option<rt::CommandBufferError> {
    let e = cb.error()?;
    match e {
        rt::CommandBufferError::Refused => refused_step("commit", cb),
        rt::CommandBufferError::PipelineCreation(_) => failed_pipeline("commit", cb),
        e => answered("commit", e, Some(cb)),
    }
    Some(e)
}

/// Dies at `sel`, where a command buffer's failure `e` is read first, unless `e` is one of Metal's
/// failures, which core then sees as Metal's: a failure this platform committed the command buffer
/// with as Metal's GPU fails it ([`Failing`], a detached object's use), a lost device's, memory's,
/// an address fault, a GPU hang and a victim of the GPU's recovery, a revoked queue's, a render
/// pass without attachments past 16384, a detached buffer's use, an encoder made under protection
/// options, a unit of command predication left with a block open, a render pass without attachments
/// whose `defaultRasterSampleCount`'s low 8 bits are 16 to 127, a memoryless texture's use by a
/// command, each with Metal's code, and one the driver relayed with one of Metal's codes
/// ([`relayed`]). Those are the fourteen variants the
/// runtime states as Metal's (`rt::FailureOrigin::Metal`), each listed here, so that one the
/// runtime adds later, whatever origin it states, dies until it is reviewed ([`unanswered`]). Every
/// other is the runtime's own (`rt::FailureOrigin::Runtime`), where Metal fails no command buffer,
/// and dies naming it: an indirect draw's indices the index pool could not hold
/// ([`pool_overflow`]), a sample through a cubic sampler an argument buffer holds
/// ([`cubic_argument_sampler`]), a command's buffer rebound before the command buffer was submitted
/// ([`encoded_use_rebound`]), an emulated mesh or tessellation draw past the records pool
/// ([`mesh_records_overflow`], [`tessellation_records_overflow`]), a Vulkan failure of a
/// submission ([`vulkan_failure`]), and any other ([`unanswered`]). A host step's
/// refusal and a draw's pipeline die before, where `cb` names them ([`refused_step`],
/// [`failed_pipeline`]). A relayed failure's death names Metal's description of it, which `cb`, the
/// command buffer that failed, keeps.
fn answered(sel: &'static str, e: rt::CommandBufferError, cb: Option<&rt::CommandBuffer>) {
    match e {
        rt::CommandBufferError::Injected(_)
        | rt::CommandBufferError::DeviceLost
        | rt::CommandBufferError::OutOfMemory
        | rt::CommandBufferError::PageFault
        | rt::CommandBufferError::GpuHang
        | rt::CommandBufferError::GpuRecoveryVictim
        | rt::CommandBufferError::AccessRevoked
        | rt::CommandBufferError::RenderTargetSize
        | rt::CommandBufferError::DetachedBacking
        | rt::CommandBufferError::ProtectedEncoder(_)
        | rt::CommandBufferError::PredicationOpenAtUnitEnd { .. }
        | rt::CommandBufferError::DefaultRasterSampleCount
        | rt::CommandBufferError::InvalidResource => {}
        rt::CommandBufferError::Execution(code) => {
            relayed(sel, code, cb.and_then(rt::CommandBuffer::error_description))
        }
        rt::CommandBufferError::IndexPoolOverflow => pool_overflow(sel),
        rt::CommandBufferError::CubicArgumentSampler => cubic_argument_sampler(sel),
        rt::CommandBufferError::EncodedUseRebound => encoded_use_rebound(sel),
        rt::CommandBufferError::MeshRecordsOverflow(overflow) => {
            mesh_records_overflow(sel, overflow)
        }
        rt::CommandBufferError::TessellationRecordsOverflow(overflow) => {
            tessellation_records_overflow(sel, overflow)
        }
        rt::CommandBufferError::Internal(code) => vulkan_failure(sel, code),
        e => unanswered(sel, e),
    }
}

/// The death at `sel` of a command buffer an emulated mesh draw failed, naming the first such
/// draw: its records did not fit the rest of the command buffer's mesh records pool, which is
/// the device's budget, or its vertices passed a `u32`, and the runtime drew nothing of it,
/// where Metal draws it and fails no command buffer.
fn mesh_records_overflow(sel: &'static str, overflow: rt::MeshRecordsOverflow) -> ! {
    unrepresentable(
        sel,
        &format!(
            "mesh draw {} of the command buffer ({} object and {} mesh threadgroups), whose \
             records passed its mesh records pool of {} bytes or whose vertices passed a u32, \
             which the runtime drew nothing of, failing the command buffer, where Metal draws it",
            overflow.draw,
            overflow.object_threadgroups,
            overflow.mesh_threadgroups,
            overflow.budget
        ),
    )
}

/// The death at `sel` of a command buffer an emulated tessellation draw failed (an indirect patch
/// draw on a device without `tessellationShader`), naming the first such draw: its records, one
/// per patch and instance, did not fit the rest of the command buffer's records pool, which
/// emulated mesh draws share and which is the device's budget, and the runtime drew nothing of it,
/// where Metal draws it and fails no command buffer.
fn tessellation_records_overflow(
    sel: &'static str,
    overflow: rt::TessellationRecordsOverflow,
) -> ! {
    unrepresentable(
        sel,
        &format!(
            "tessellation draw {} of the command buffer ({} patch records), whose records passed \
             its records pool of {} bytes, which the runtime drew nothing of, failing the command \
             buffer, where Metal draws it",
            overflow.draw, overflow.records, overflow.budget
        ),
    )
}

/// The death at `sel` of a command buffer whose submission Vulkan failed with `code`, other than
/// a lost device or memory (or that the runtime failed so for its own broken invariant, as
/// `VK_ERROR_UNKNOWN`): no failure of Metal's answers it.
fn vulkan_failure(sel: &'static str, code: rt::VkResultCode) -> ! {
    unrepresentable(
        sel,
        &format!(
            "a submission Vulkan failed with VkResult {}, which no failure of Metal's command \
             buffers answers",
            code.0
        ),
    )
}

/// The death at `sel` of a command buffer the runtime failed in a way nothing here matched to
/// a failure of Metal's: one the runtime adds after this platform was written, of either origin
/// it states, until it is reviewed here.
fn unanswered(sel: &'static str, e: rt::CommandBufferError) -> ! {
    unrepresentable(
        sel,
        &format!(
            "the runtime's failure of the command buffer ({e:?}), which nothing here matched to \
             a failure of Metal's"
        ),
    )
}

/// The death at `sel` of a command buffer a function failed by sampling, gathering or querying
/// through a sampler an argument buffer holds whose min or mag filter is cubic: the runtime
/// does not filter bicubically there, and fails the command buffer where its function reached
/// one, where Metal samples it and fails no command buffer.
fn cubic_argument_sampler(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a sample through a cubic sampler an argument buffer holds, which the runtime does not \
         filter bicubically there and fails the command buffer of, where Metal samples it",
    )
}

/// The death at `sel` of a command buffer the runtime failed as it was first submitted, as a
/// command of it named a buffer over the guest's pages that was rebound after the command was
/// encoded, in an encoder the runtime could not encode again (the module docs): the runtime
/// resolved the use as the command was encoded, and ran nothing of each submission holding such
/// a command, where Metal resolves it as it submits the command buffer and runs the command over
/// the pages the buffer lies in then.
fn encoded_use_rebound(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "a command naming a buffer over the guest's pages that was rebound after the command was \
         encoded and before its command buffer was submitted, in an encoder the runtime could \
         not encode again, whose use it resolved as it was encoded and ran nothing of, failing \
         the command buffer, where Metal runs it over the pages the buffer lies in as it submits",
    )
}

/// The death at `sel` of a command buffer an indexed indirect draw failed: the runtime keeps
/// such a draw's indices, checked or rewritten by a pass before the render pass, in the command
/// buffer's index pool, a sixty-fourth of the device's largest device-local heap, and draws
/// nothing of one past it, where Metal draws the draw and fails no command buffer.
fn pool_overflow(sel: &'static str) -> ! {
    unrepresentable(
        sel,
        "the indices of an indexed indirect draw past the command buffer's index pool, which the \
         runtime drew nothing of where Metal draws them",
    )
}

/// Dies at `sel` unless `code`, a failure of the GPU's work the driver relayed, is one of
/// Metal's codes ([`METAL_CODES`]), which core then sees as Metal's failure of the command
/// buffer. The death's line names Metal's `description` of the failure, where the driver's
/// message gave one ([`relayed_line`]).
fn relayed(sel: &'static str, code: u64, description: Option<String>) {
    if !METAL_CODES.contains(&code) {
        unrepresentable(sel, &relayed_line(code, description.as_deref()));
    }
}

/// The line of a relayed failure's death ([`relayed`]): its code, and Metal's `description` of
/// it where the driver's message gave one, not empty.
fn relayed_line(code: u64, description: Option<&str>) -> String {
    let line = format!(
        "a failure of the GPU's work the driver relayed with code {code}, which no failure of \
         Metal's command buffers has"
    );
    match description.filter(|d| !d.is_empty()) {
        Some(description) => format!("{line} (Metal's description: {description})"),
        None => line,
    }
}

/// The death at `sel` of `cb`, which failed as the driver did not make the `VkPipeline` of a draw
/// or a dispatch it recorded, naming the driver's error and the command, by its encoder's place
/// among the command buffer's ended encoders and its order there: the runtime makes a draw's or
/// a dispatch's pipeline while the command goes on, where Metal makes a state's pipelines as it
/// makes the state, and fails no command buffer so.
fn failed_pipeline(sel: &'static str, cb: &rt::CommandBuffer) -> ! {
    match cb.failed_pipeline() {
        Some(failed) => {
            let (command, encoder) = match failed.state {
                rt::FailedState::Compute(_) => ("dispatch", "compute"),
                _ => ("draw", "render"),
            };
            fatal_with_detail(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: sel,
                },
                &format!(
                    "{} (the pipeline of {command} {} of the command buffer's {encoder} encoder \
                     {})",
                    refusal(&failed.error),
                    failed.draw,
                    failed.encoder
                ),
            )
        }
        None => unrepresentable(
            sel,
            "a draw's or dispatch's pipeline the runtime did not keep, which failed",
        ),
    }
}

/// The death at `sel` of `cb`, which a host step refused, naming the refusal.
fn refused_step(sel: &'static str, cb: &rt::CommandBuffer) -> ! {
    match cb.refusal() {
        Some(refusal) => refused(sel, &refusal),
        None => unrepresentable(sel, "a host step's refusal the runtime did not keep"),
    }
}

/// The `NSError` of a failed command buffer: Metal's domain and code for the runtime's error,
/// and Metal's description of a use of a detached buffer or heap. A failure the driver relayed
/// (`Execution`, and the `PageFault` and `DeviceLost` the runtime maps codes 3 and 11 to) names
/// the runtime's error and, beside it, Metal's `relayed` description of the failure, as the
/// driver's message gave it (`rt::CommandBuffer::error_description`), where not empty. Core
/// prints the description in its log lines of a failed command buffer; nothing else reads it.
fn command_buffer_error(e: rt::CommandBufferError, relayed: Option<String>) -> NsErrorText {
    let description = match (e, relayed.filter(|d| !d.is_empty())) {
        (rt::CommandBufferError::Injected(backing::DETACHED_USE), _) => {
            "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".to_owned()
        }
        (e, Some(metal)) => {
            format!(
                "the Vulkan runtime's command buffer failed: {e:?} (Metal's description: {metal})"
            )
        }
        (e, None) => format!("the Vulkan runtime's command buffer failed: {e:?}"),
    };
    NsErrorText {
        domain: "MTLCommandBufferErrorDomain".to_owned(),
        code: i64::try_from(e.metal_code()).unwrap_or(i64::MAX),
        description,
    }
}

/// The `NSError` of `cb`, once it failed with one of Metal's failures ([`failure`],
/// [`command_buffer_error`]).
fn failure_error(cb: &rt::CommandBuffer) -> Option<NsErrorText> {
    let e = failure(cb)?;
    Some(command_buffer_error(e, cb.error_description()))
}

/// The runtime's handler of core's completed handler `block`, added by the code of `task`, the
/// isolated task whose code runs, if one's does: it hands `block` the command buffer's `status`
/// and `error` as they are then, after `read`, its read of the failure, which dies for a failure
/// that is none of Metal's ([`failure`]). The read is the task's code ([`as_task`]), as the
/// death in [`Ending`] is: once the task ended, core's handlers of it run none
/// (`TaskLife::carry`), so none reads the failure, and nothing dies; `block` is called all the
/// same, with the failure as its code says ([`command_buffer_error`]), and decides itself
/// whether it runs. Added inside a refusal scope (the graphics lane's), the read and `block` run
/// in one on the runtime's thread too ([`run_completed`]): the read's death fails the command
/// buffer to `block` ([`CompletedInfo::refused`]) and the process goes on.
fn completed_handler(
    block: CompletedHandler,
    task: Option<Arc<TaskLife>>,
    read: impl FnOnce(&rt::CommandBuffer) + Send + 'static,
) -> rt::Handler {
    #[cfg(test)]
    let slot = reims_vgpu_core::testkit::capture::current_slot();
    let carried = refusals::carry();
    Box::new(move |cb: &rt::CommandBuffer| {
        // A callout on the runtime's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        let run = || {
            run_completed(
                carried,
                || {
                    as_task(task.as_ref(), || read(cb));
                    let error = cb
                        .error()
                        .map(|e| command_buffer_error(e, cb.error_description()));
                    let gpu_times_ns = match (cb.gpu_start_time(), cb.gpu_end_time()) {
                        (Ok(start), Ok(end)) => CompletedInfo::gpu_times(start, end),
                        _ => None,
                    };
                    CompletedInfo {
                        status: CommandBufferStatus(cb.status().raw()),
                        error,
                        gpu_times_ns,
                    }
                },
                block,
            );
        };
        #[cfg(test)]
        reims_vgpu_core::testkit::capture::with_slot(slot, run);
        #[cfg(not(test))]
        run();
    })
}

/// The runtime's handler running `block` once: a scheduled handler, on the runtime's handler
/// thread as the command buffer becomes `Scheduled`, or a submission handler, on the thread that
/// took its last batch.
fn scheduled_handler(block: ScheduledHandler) -> rt::Handler {
    #[cfg(test)]
    let slot = reims_vgpu_core::testkit::capture::current_slot();
    Box::new(move |_cb: &rt::CommandBuffer| {
        // A callout on the runtime's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        #[cfg(test)]
        reims_vgpu_core::testkit::capture::with_slot(slot, block);
        #[cfg(not(test))]
        block();
    })
}

/// The runtime's form of an event core passes; a nil one dies at `sel`.
fn event<'a>(sel: &'static str, event: Option<EventRef<'a, VulkanPlatform>>) -> rt::EventRef<'a> {
    match event {
        Some(EventRef::Event(e)) => rt::EventRef::Event(e),
        Some(EventRef::Shared(e)) => rt::EventRef::Shared(e),
        None => unrepresentable(sel, "a nil event"),
    }
}

impl CommandBuffer<VulkanPlatform> for VulkanCommandBuffer {
    /// Refused once the buffer is committed, as Metal's validation refuses it. A failure that is
    /// none of Metal's dies as the handler reads it, unless the isolated task whose code added
    /// the handler ended ([`completed_handler`]).
    fn add_completed_handler(&self, _on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        let handler = completed_handler(block, TaskLife::current(), |cb| {
            failure(cb);
        });
        send("addCompletedHandler:", Policy::Dies, || {
            self.raw.add_completed_handler(handler)
        })
    }

    /// Refused once the buffer is committed, as Metal's validation refuses it. The runtime runs
    /// it once every wait of the buffer is satisfied and its last batch is submitted, before its
    /// completed handlers.
    fn add_scheduled_handler(&self, _on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        let handler = scheduled_handler(block);
        send("addScheduledHandler:", Policy::Dies, || {
            self.raw.add_scheduled_handler(handler)
        })
    }

    /// The runtime's submission handler: run on the thread that took the buffer's last batch for
    /// its `VkQueue`, before it is scheduled, and never for a buffer that fails before then.
    /// Refused once the buffer is committed.
    fn add_submission_handler(&self, _on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        let handler = scheduled_handler(block);
        send("addSubmissionHandler", Policy::Dies, || {
            self.raw.add_submission_handler(handler)
        })
    }

    /// The runtime's completion-seen handler: run on the thread that saw the buffer complete,
    /// before its completed handlers, a failure read as a completed handler reads it
    /// ([`completed_handler`]). Refused once the buffer is committed.
    fn add_completion_seen_handler(
        &self,
        _on: OnFault,
        block: CompletedHandler,
    ) -> RaiseResult<()> {
        let handler = completed_handler(block, TaskLife::current(), |cb| {
            failure(cb);
        });
        send("addCompletionSeenHandler", Policy::Dies, || {
            self.raw.add_completion_seen_handler(handler)
        })
    }

    /// Refused while another encoder is open and after the commit, as Metal's validation
    /// refuses them.
    fn blit_encoder(&self, _on: OnFault) -> RaiseResult<Option<VulkanBlitEncoder>> {
        let raw = encoding("blitCommandEncoder", Policy::Dies, || {
            predication::boundary(&self.predication, "blitCommandEncoder");
            self.raw.blit_command_encoder().map(Box::new)
        })?;
        Ok(Some(Obj::new(EncoderObject::new(raw, self))))
    }

    /// Refused while an encoder is open or after a commit, as Metal's validation refuses it.
    /// Every refusal dies naming the runtime's, and one of the records of an indirect command
    /// buffer it executes says so ([`committing`]). Its progress is committed past the send's
    /// check of the task's end, where a thread of an ended task stops with the command buffer
    /// never committed. One that Metal's GPU fails for what its encoders did is committed failed
    /// ([`CommandBufferObject::failure`]); any other is committed with its submission check,
    /// which fails a submission that uses a detached buffer or heap (module docs). A slow commit
    /// and a late completion are debug lines ([`timing::commit`]). One that holds work for
    /// Metal's tiler takes Metal's allocation for it as the runtime submits it
    /// ([`tiler_handler`]). One committed while a command buffer of its queue that Metal's GPU
    /// hangs at is in flight runs nothing and completes without an error, as on Metal, as the
    /// runtime commits it (the module docs). It is committed through the runtime's
    /// `commit_and_submit`, which submits on this thread the ready batches that need nothing of
    /// the scheduler's thread, and runs their submission handlers here, inside the call; a batch
    /// with a submission check, which every command buffer here carries but one committed failed,
    /// is the scheduler's thread's, as under `commit`.
    fn commit(&self, _on: OnFault) -> RaiseResult<()> {
        // The guest's GPU work, which keeps the GPU awake ([`crate::awake`]).
        self.device.awake.touch();
        let progress = self.progress.lock().clone();
        encoding("commit", Policy::Dies, || {
            predication::boundary(&self.predication, "commit");
            let opened = self.opened.lock().take();
            let detached = || {
                opened
                    .as_ref()
                    .is_some_and(|opened| backing::uses_detached(&self.raw, opened))
            };
            let failure = self.failure("commit", detached);
            self.check_executions(failure.as_ref().map(|f| &f.encoders), progress.as_deref());
            let check = match (&failure, opened) {
                (None, Some(opened)) => Some(backing::submission_check(opened.commit())),
                _ => None,
            };
            if let Some(progress) = &progress {
                progress.committing();
            }
            committing(|| {
                timing::commit(&self.raw, || match (failure, check) {
                    (Some(failure), _) => self.raw.commit_failed(failure),
                    (None, Some(check)) => self
                        .raw
                        .set_submission_check(check)
                        .and_then(|()| self.raw.commit_and_submit()),
                    (None, None) => self.raw.commit_and_submit(),
                })
            })
        })
    }

    /// A serial encoder, refused as `blitCommandEncoder` is; under protection options 1, 3 and 7
    /// it dies ([`CommandBufferObject::protected_creation`]). A first use of Metal's device
    /// ([`FirstUse::ComputeEncoder`]).
    fn compute_encoder(&self, _on: OnFault) -> RaiseResult<Option<VulkanComputeEncoder>> {
        let raw = encoding("computeCommandEncoder", Policy::Dies, || {
            self.protected_creation("computeCommandEncoder");
            self.raw.compute_command_encoder()
        })?;
        metal_space::first_use(FirstUse::ComputeEncoder);
        Ok(Some(predication::encoder(raw, self)))
    }

    fn device(&self) -> Option<VulkanDevice> {
        Some(self.device.clone_ref())
    }

    fn encode_signal(
        &self,
        event: Option<EventRef<'_, VulkanPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeSignalEvent:value:";
        // evsig: command buffer, event, value
        trace_event("evsig", &self.raw, event.as_ref(), value);
        encoding(sel, Policy::Dies, || {
            let event = self::event(sel, event);
            predication::boundary(&self.predication, sel);
            self.raw.encode_signal_event(event, value)
        })
    }

    /// The runtime's: the event's value becomes at least `value` as the command buffer is
    /// scheduled, before its status reads `Scheduled` and before `waitUntilScheduled` returns, as
    /// Metal's (the runtime's measurement on an M3 Max under macOS 27.0.1). Sent once the command
    /// buffer is committed it sets nothing, as Metal's; once it completed, it dies, where Metal's
    /// ends the process with a segmentation fault
    /// (`a_scheduled_signal_sent_after_the_commit_sets_nothing_as_metals`, measured on the M3 Max
    /// under macOS 27.0.1 after a `Completed` and an `Error` completion): at the runtime's refusal
    /// of one that completed or failed (`ScheduledSignalAfterCompletion`,
    /// [`signal_after_completion`]). While an encoder is open it dies, as Metal asserts. Core
    /// sends it before the commit alone.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&VulkanSharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeSignalEventScheduled:value:";
        let traced = signal_event_scheduled.map_or(0, event_id);
        // evsched: command buffer, event, value
        trace_event_id("evsched", &self.raw, traced, value);
        encoding(sel, Policy::Dies, || {
            let event =
                signal_event_scheduled.unwrap_or_else(|| unrepresentable(sel, "a nil event"));
            match self.raw.encode_signal_event_scheduled(event, value) {
                Err(rt::Error::Invalid(invalid))
                    if matches!(invalid.problem, rt::Problem::ScheduledSignalAfterCompletion) =>
                {
                    signal_after_completion(sel)
                }
                made => made,
            }
        })
    }

    fn encode_wait(
        &self,
        event: Option<EventRef<'_, VulkanPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeWaitForEvent:value:";
        // evwait: command buffer, event, value
        trace_event("evwait", &self.raw, event.as_ref(), value);
        encoding(sel, Policy::Dies, || {
            let event = self::event(sel, event);
            predication::boundary(&self.predication, sel);
            self.raw.encode_wait_for_event(event, value)
        })
    }

    fn error(&self) -> Option<NsErrorText> {
        plain(|| failure_error(&self.raw))
    }

    /// An encoder of the pass, nil where Metal answers nil (`encode_render::open`).
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &VulkanRenderPassDescriptor,
    ) -> RaiseResult<Option<VulkanRenderEncoder>> {
        let sel = "renderCommandEncoderWithDescriptor:";
        plain_at(sel, || predication::boundary(&self.predication, sel));
        crate::gpu::encode_render::open(self, on, render_pass_descriptor)
    }

    /// Kept here, as the runtime has no protected content: the encoders made under them decide
    /// how the command buffer fails ([`CommandBufferObject::made_encoder`]). While an encoder is
    /// open it dies, as Metal asserts (`exec_protection_envelope_open_encoder`), and after an
    /// encoder went without `endEncoding`, naming the runtime's refusal (`EncoderAbandoned`),
    /// where plain Metal asserts as the encoder goes; so do other
    /// options once an encoder was made under some, or after options from 0x10 up, which
    /// nothing measured.
    fn set_protection_options(&self, _on: OnFault, options: u64) -> RaiseResult<()> {
        let sel = "setProtectionOptions:";
        if self.open_encoders.load(Ordering::SeqCst) != 0 {
            uncommitted_encoder(sel);
        }
        send(sel, Policy::Dies, || {
            self.raw
                .set_protection_options(rt::DEFAULT_PROTECTION_OPTIONS)
        })?;
        let was = plain(|| *self.protection.lock());
        let kept = was.used || protection_code(was.options) == INVALID_INPUT;
        if kept && options != was.options {
            unrepresentable(
                sel,
                &format!(
                    "protection options {options:#x} after {:#x}, which nothing measured",
                    was.options
                ),
            );
        }
        plain(|| self.protection.lock().options = options);
        Ok(())
    }

    /// Scheduled or completed once its submission handler ran ([`tiler_handler`]).
    fn status(&self) -> CommandBufferStatus {
        let status = plain(|| {
            let status = self.raw.status();
            failure(&self.raw);
            status
        });
        CommandBufferStatus(status.raw())
    }

    /// Blocks. A wait the runtime knows would never return (a handler's for its own buffer)
    /// hangs, as Metal's does, and one whose answer on Metal is not measured dies ([`waiting`]).
    /// A command buffer a draw's pipeline failed, or a host step's refusal, or any failure
    /// that is none of Metal's, dies as the wait returns ([`failed_pipeline`],
    /// [`refused_step`], [`answered`]), where no completed handler read its failure first.
    fn wait_until_completed(&self, _on: OnFault) -> RaiseResult<()> {
        const SEL: &str = "waitUntilCompleted";
        waiting(SEL, || self.raw.wait_until_completed())?;
        match plain(|| self.raw.error()) {
            Some(rt::CommandBufferError::PipelineCreation(_)) => failed_pipeline(SEL, &self.raw),
            Some(rt::CommandBufferError::Refused) => refused_step(SEL, &self.raw),
            Some(e) => answered(SEL, e, Some(&self.raw)),
            None => {}
        }
        Ok(())
    }

    /// Blocks, and hangs or dies, as `wait_until_completed` does. It returns once the command
    /// buffer's submission handler ran ([`tiler_handler`]).
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        waiting("waitUntilScheduled", || self.raw.wait_until_scheduled())
    }

    /// The pass's dispatch type, serial when unset as Metal's descriptor's is. Metal makes an
    /// encoder of any type, and runs every type but concurrent as serial (measured, macOS 26.7,
    /// M3 Max): an encoder of type 2, 3, `0x80000000` or `0xffffffff` orders 8 dependent
    /// dispatches over 4M threads as a serial one does, where a concurrent one's race, and skips
    /// a nil entry of `useResources:count:usage:`, where a concurrent one ends the process. So
    /// such a type is serial here; a type past 32 bits, which the guest's record cannot hold
    /// and nothing measured, dies at `setDispatchType:`.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<VulkanComputeEncoder>> {
        let dispatch_type = plain_at("setDispatchType:", || match desc.dispatch_type {
            None => DispatchType::Serial,
            Some(raw) => match u32::try_from(raw) {
                Ok(raw) => DispatchType::try_from(raw).unwrap_or(DispatchType::Serial),
                Err(_) => unrepresentable(
                    "setDispatchType:",
                    &format!("dispatch type {raw:#x}, past 32 bits, which nothing measured"),
                ),
            },
        });
        let sel = "computeCommandEncoderWithDescriptor:";
        let raw = encoding(sel, Policy::Dies, || {
            self.protected_creation(sel);
            self.raw
                .compute_command_encoder_with_dispatch_type(dispatch_type)
        })?;
        metal_space::first_use(FirstUse::ComputeEncoder);
        Ok(Some(predication::encoder(raw, self)))
    }
}

/// An event's number in the trace: its object's address.
pub(crate) fn event_id<T: Send + Sync + 'static>(event: &Obj<T>) -> u64 {
    u64::try_from(std::ptr::from_ref::<T>(event).addr()).unwrap_or_default()
}

/// A traced encode of an event: its command buffer, the event and the value.
fn trace_event(
    kind: &str,
    cb: &rt::CommandBuffer,
    event: Option<&EventRef<'_, VulkanPlatform>>,
    value: u64,
) {
    if !reims_vgpu_core::trace::on() {
        return;
    }
    let id = match event {
        Some(EventRef::Shared(e)) => event_id(*e),
        Some(EventRef::Event(e)) => event_id(*e),
        None => 0,
    };
    trace_event_id(kind, cb, id, value);
}

fn trace_event_id(kind: &str, cb: &rt::CommandBuffer, event: u64, value: u64) {
    reims_vgpu_core::trace::event(kind, &[crate::gpu::timing::id(cb), event, value]);
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// A fresh buffer of the queue's device
    fn buffer(queue: &VulkanCommandQueue) -> VulkanCommandBuffer {
        queue.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// commandBuffer makes a fresh buffer, not enqueued, of the queue's device; setLabel: sets
    /// the queue's label
    #[test]
    fn queue_command_buffer_and_label() {
        let q = dev().new_command_queue().unwrap();
        let (a, b) = (buffer(&q), buffer(&q));
        assert_ne!(a.id(), b.id());
        assert_eq!(a.status(), CommandBufferStatus::NOT_ENQUEUED);
        assert_eq!(a.device().unwrap().id(), q.device.id());
        q.set_label("reims-vgpu.exec");
        assert_eq!(q.raw.label().as_deref(), Some("reims-vgpu.exec"));
    }

    /// An empty buffer commits and completes: its completed handler sees Completed and no
    /// error, after the scheduled signal set its event; a handler added after the commit, and a
    /// second commit, die
    #[test]
    fn commit_completes_and_runs_handlers() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let scheduled = d.new_shared_event().unwrap();
        cb.encode_signal_when_scheduled(Some(&scheduled), 5)
            .unwrap();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(
            OnFault::Return,
            Box::new(move |info| tx.send(info).unwrap()),
        )
        .unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let info = rx.recv().unwrap();
        assert_eq!(info.status, CommandBufferStatus::COMPLETED);
        assert_eq!(info.error, None);
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(cb.error(), None);
        assert_eq!(scheduled.signaled_value(), 5);
        // A handler after the commit, and a second commit, die: Metal raises at neither.
        let late = catch_death(|| {
            let _ = cb.add_completed_handler(OnFault::Return, Box::new(|_| {}));
        });
        assert_eq!(late.unwrap_err().file, "addCompletedHandler:");
        for on in [OnFault::Return, OnFault::Die] {
            let site = catch_death(|| {
                let _ = cb.commit(on);
            })
            .unwrap_err();
            assert_eq!(site.file, "commit");
        }
    }

    /// `encodeSignalEventScheduled:value:` sets the event's value as its command buffer is
    /// scheduled, behind an event wait once the wait is met, and, sent after the commit while the
    /// command buffer waits, sets nothing, as Metal's: the value does not rise, and nothing dies.
    /// Sent once the command buffer completed, or once it failed (here under protection options
    /// from 0x10, with error code 1), it dies at the runtime's refusal, where Metal's ends the
    /// process with a segmentation fault
    #[cfg(target_os = "macos")]
    #[test]
    fn a_scheduled_signal_sent_after_the_commit_sets_nothing_as_metals() {
        use reims_vgpu_core::platform::Platform;
        use reims_vgpu_metal::MetalPlatform;

        /// Whether the value reached the scheduled signal's 1 and the late one's 7, once the
        /// command buffer completed.
        fn reached<P: Platform>(d: &P::Device) -> ([bool; 2], P::CommandBuffer) {
            let q = d.new_command_queue().unwrap();
            let (event, held) = (d.new_shared_event().unwrap(), d.new_shared_event().unwrap());
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&held)), 1).unwrap();
            cb.encode_signal_when_scheduled(Some(&event), 1).unwrap();
            cb.commit(OnFault::Return).unwrap();
            cb.encode_signal_when_scheduled(Some(&event), 7).unwrap();
            held.set_signaled_value(1).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            let values = [1, 7].map(|value| event.wait_until_signaled_value(value, 0));
            (values, cb)
        }
        let (vulkan, cb) = reached::<VulkanPlatform>(&dev());
        assert_eq!(vulkan, [true, false]);
        let event = dev().new_shared_event().unwrap();
        let site = catch_death(|| cb.encode_signal_when_scheduled(Some(&event), 3)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::OutOfBounds, "encodeSignalEventScheduled:value:")
        );
        let failed = buffer(&dev().new_command_queue().unwrap());
        failed
            .set_protection_options(OnFault::Return, 0x10)
            .unwrap();
        assert_eq!(completed(&failed), (5, Some(1)));
        let site =
            catch_death(|| failed.encode_signal_when_scheduled(Some(&event), 3)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::OutOfBounds, "encodeSignalEventScheduled:value:")
        );
        if let Some(md) = MetalPlatform::system_default_device() {
            assert_eq!(reached::<MetalPlatform>(&md).0, vulkan);
        }
    }

    /// A command buffer whose first command waits for an unsignalled shared event is not
    /// scheduled until the CPU signals it: 100 ms after its commit it still reads committed and
    /// its scheduled handler has not run, which runs once the event is signalled, before its
    /// completed handler, the blit behind the wait run. Metal on this host does the same (the
    /// comparison below), and so does every Mac measured: an AMD Radeon Pro 5300M and an Intel
    /// UHD 630 under macOS 26.7.1 schedule such a buffer at the signal
    /// (docs/graphics-kext/dgpu-oracle.md, Q7). The graphics lane posts `CB_SCHEDULED` from that
    /// handler, so the guest's buffer reads scheduled no earlier
    #[test]
    fn a_command_buffer_waiting_for_an_event_is_scheduled_at_its_signal() {
        use reims_vgpu_core::platform::Platform;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::NsRange;

        /// On `d`: the buffer's status 100 ms after its commit, then the order its scheduled
        /// and completed handlers ran in around the CPU's signal, and the first byte its blit
        /// filled.
        fn held<P: Platform>(d: &P::Device) -> (CommandBufferStatus, Vec<&'static str>, u8) {
            let q = d.new_command_queue().unwrap();
            let event = d.new_shared_event().unwrap();
            let b = d
                .new_buffer_of_length(OnFault::Return, 16, 0)
                .unwrap()
                .unwrap();
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for cmd in [
                BlitCmd::FillBuffer {
                    buffer: &b,
                    range: NsRange::new(0, 16),
                    value: 0x5a,
                },
                BlitCmd::EndEncoding,
            ] {
                blit.encode(OnFault::Return, cmd).unwrap();
            }
            let seen = Arc::new(Mutex::new(Vec::new()));
            let at = Arc::clone(&seen);
            cb.add_scheduled_handler(
                OnFault::Return,
                Box::new(move || at.lock().push("scheduled")),
            )
            .unwrap();
            let (at, (done, completed)) = (Arc::clone(&seen), mpsc::channel());
            let handler: CompletedHandler = Box::new(move |_| {
                at.lock().push("completed");
                done.send(()).unwrap();
            });
            cb.add_completed_handler(OnFault::Return, handler).unwrap();
            cb.commit(OnFault::Return).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
            let status = cb.status();
            seen.lock().push("signal");
            event.set_signaled_value(1).unwrap();
            completed
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            // SAFETY: a Shared buffer of 16 bytes, read after its command buffer completed.
            let first = unsafe { *b.contents().as_ptr::<u8>() };
            let order = seen.lock().clone();
            (status, order, first)
        }
        let vulkan = held::<VulkanPlatform>(&dev());
        assert_eq!(
            vulkan,
            (
                CommandBufferStatus::COMMITTED,
                vec!["signal", "scheduled", "completed"],
                0x5a
            )
        );
        #[cfg(target_os = "macos")]
        if let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() {
            assert_eq!(held::<reims_vgpu_metal::MetalPlatform>(&md), vulkan);
        }
    }

    /// A completed handler learns its buffer's GPU times, in nanoseconds of the host's clock: a
    /// run of a blit fill that ends no earlier than it starts and lasts under a second, as
    /// Metal's on this host
    #[test]
    fn a_completed_handler_learns_the_gpu_times() {
        use reims_vgpu_core::platform::Platform;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::types::NsRange;

        /// On `d`: the GPU times a blit fill's completed handler learned.
        fn learned<P: Platform>(d: &P::Device) -> Option<(u64, u64)> {
            let q = d.new_command_queue().unwrap();
            let b = d
                .new_buffer_of_length(OnFault::Return, 16, 0)
                .unwrap()
                .unwrap();
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            for cmd in [
                BlitCmd::FillBuffer {
                    buffer: &b,
                    range: NsRange::new(0, 16),
                    value: 0x5a,
                },
                BlitCmd::EndEncoding,
            ] {
                blit.encode(OnFault::Return, cmd).unwrap();
            }
            let (done, completed) = mpsc::channel();
            let handler: CompletedHandler = Box::new(move |info| {
                done.send(info.gpu_times_ns).unwrap();
            });
            cb.add_completed_handler(OnFault::Return, handler).unwrap();
            cb.commit(OnFault::Return).unwrap();
            completed
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap()
        }
        let plausible = |times: Option<(u64, u64)>| {
            matches!(times, Some((start, end)) if start > 0 && end >= start
                && end - start < 1_000_000_000)
        };
        let vulkan = learned::<VulkanPlatform>(&dev());
        assert!(plausible(vulkan), "{vulkan:?}");
        #[cfg(target_os = "macos")]
        if let Some(md) = reims_vgpu_metal::MetalPlatform::system_default_device() {
            let metal = learned::<reims_vgpu_metal::MetalPlatform>(&md);
            assert!(plausible(metal), "{metal:?}");
        }
    }

    /// Metal takes its own allocation for a command buffer's first work for its tiler as it
    /// submits that work (`vaprobe rendersubmit`): behind an event wait, once the wait is met,
    /// not at the commit
    #[test]
    fn tiler_work_behind_a_wait_is_taken_once_the_wait_is_met() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work("test", FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
    }

    /// With no event wait and nothing ahead of it on its queue, a command buffer's first work
    /// for the tiler is taken as the runtime submits it, after its commit and before it reads
    /// scheduled, as Metal submits it on a thread of its own
    #[test]
    fn tiler_work_with_no_wait_is_taken_as_it_is_submitted() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        cb.tiler_work("test", FirstUse::TilerBlit);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// A command buffer's first work for the tiler, the process's first, adds its submission
    /// handler without stopping a thread of an isolated task that ended: a render encoder's
    /// runtime encoder is not in its handle yet as its work is taken (`encode_render`'s `make`)
    #[test]
    fn an_ended_tasks_work_for_the_tiler_stops_nothing() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        assert!(!metal_space::taken(FirstUse::TilerSubmission));
        let sel = "renderCommandEncoderWithDescriptor:";
        let life = TaskLife::new();
        let went_on = life.run(|| {
            life.exit();
            cb.tiler_work(sel, FirstUse::RenderEncoder);
        });
        assert!(went_on.is_some());
        assert!(cb.tiler.load(Ordering::SeqCst));
    }

    /// Once the process took Metal's allocation for its work for the tiler, a command buffer
    /// with such work carries no submission handler for it: the runtime runs its transition to
    /// scheduled on its scheduler thread as it submits it, with no handler thread woken for it,
    /// as it runs one with no work for the tiler
    #[test]
    fn tiler_work_once_taken_is_scheduled_where_submitted() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        cb.tiler_work("test", FirstUse::TilerBlit);
        assert!(cb.tiler.load(Ordering::SeqCst));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::taken(FirstUse::TilerSubmission));
        for tiler in [true, false] {
            let q = d.new_command_queue().unwrap();
            let cb = buffer(&q);
            if tiler {
                cb.tiler_work("test", FirstUse::TilerBlit);
            }
            assert!(!cb.tiler.load(Ordering::SeqCst));
            let before = d.raw.work_counts().scheduled_where_submitted;
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            assert_eq!(d.raw.work_counts().scheduled_where_submitted, before + 1);
        }
    }

    /// A render encoder made through the platform's `renderCommandEncoderWithDescriptor:` (a
    /// render pass that clears and stores a texture) is work for the tiler: the process's first
    /// is taken as the runtime submits its command buffer, before it reads scheduled
    #[test]
    fn a_render_encoders_tiler_work_is_taken_as_it_is_submitted() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        use reims_vgpu_core::platform::values::{ColorAttachmentSet, TextureDesc};
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let p = crate::platform::tests::platform();
        let target = d
            .new_texture(
                OnFault::Die,
                &p.texture_descriptor(&TextureDesc {
                    pixel_format: 80,
                    width: 16,
                    height: 16,
                    usage: 4,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                }),
            )
            .unwrap()
            .unwrap();
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(2),
            store_action: Some(1),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let cb = buffer(&q);
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
        assert!(metal_space::process().took(FirstUse::RenderEncoder));
        assert!(cb.tiler.load(Ordering::SeqCst));
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// A wait met before the commit holds nothing back: the runtime submits the command buffer
    /// at once, and its work for the tiler is taken before it reads scheduled
    #[test]
    fn tiler_work_behind_a_met_wait_is_taken_as_it_is_submitted() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        event.set_signaled_value(1).unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work("test", FirstUse::TilerBlit);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// An event wait after the work holds the command buffer's submission back until the wait
    /// is met, and its work for the tiler with it
    #[test]
    fn tiler_work_before_a_wait_is_taken_at_the_submission() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.tiler_work("test", FirstUse::TilerBlit);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
    }

    /// A command buffer with no wait of its own, committed behind one on its queue whose wait
    /// holds it, takes its work for the tiler as the runtime submits it behind the earlier one
    #[test]
    fn tiler_work_behind_a_waiting_command_buffer_is_taken_at_its_submission() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let first = buffer(&q);
        first
            .encode_wait(Some(EventRef::Shared(&event)), 1)
            .unwrap();
        first.commit(OnFault::Return).unwrap();
        let cb = buffer(&q);
        cb.tiler_work("test", FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        first.wait_until_completed(OnFault::Return).unwrap();
    }

    /// No call finds a command buffer scheduled before its work for the tiler is taken: `status`
    /// answers committed while its wait holds it, and scheduled or completed only with the
    /// allocation taken, as the runtime runs the submission handler first
    #[test]
    fn tiler_work_is_taken_before_the_command_buffer_reads_scheduled() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = buffer(&q);
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.tiler_work("test", FirstUse::TilerBlit);
        cb.commit(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMMITTED);
        assert!(!metal_space::process().took(FirstUse::TilerSubmission));
        event.set_signaled_value(1).unwrap();
        // At most 10 s: the event's value is met, so the runtime submits it at once.
        let mut polls = 0;
        while cb.status() == CommandBufferStatus::COMMITTED {
            polls += 1;
            assert!(
                polls < 10_000,
                "the command buffer was not scheduled within 10 s"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(metal_space::process().took(FirstUse::TilerSubmission));
        cb.wait_until_completed(OnFault::Return).unwrap();
    }

    /// A buffer signals an event when it completes, and a later buffer waits for it; both
    /// kinds of event are taken, and a nil event dies
    #[test]
    fn events_signal_and_wait() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let shared = d.new_shared_event().unwrap();
        let event = d.new_event(OnFault::Return).unwrap().unwrap();
        let first = buffer(&q);
        first
            .encode_signal(Some(EventRef::Shared(&shared)), 1)
            .unwrap();
        first
            .encode_signal(Some(EventRef::Event(&event)), 1)
            .unwrap();
        let second = buffer(&q);
        second
            .encode_wait(Some(EventRef::Event(&event)), 1)
            .unwrap();
        first.commit(OnFault::Return).unwrap();
        second.commit(OnFault::Return).unwrap();
        second.wait_until_completed(OnFault::Return).unwrap();
        assert!(shared.wait_until_signaled_value(1, 1000));
        let cb = buffer(&q);
        let site = catch_death(|| cb.encode_wait(None, 1)).unwrap_err();
        assert_eq!(site.file, "encodeWaitForEvent:value:");
    }

    /// A completed handler's wait for its own command buffer, which Metal never returns from,
    /// hangs: the handler neither returns nor dies
    #[test]
    fn a_handlers_wait_for_its_own_buffer_hangs() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        let own = cb.clone_ref();
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(
            OnFault::Return,
            Box::new(move |_| {
                tx.send("waiting").unwrap();
                let waited = own.wait_until_completed(OnFault::Return);
                tx.send(if waited.is_ok() { "returned" } else { "raised" })
                    .unwrap();
            }),
        )
        .unwrap();
        cb.commit(OnFault::Return).unwrap();
        assert_eq!(rx.recv().unwrap(), "waiting");
        // Still blocked: no answer, and the handler still holds its sender (a death would have
        // dropped it).
        let after = rx.recv_timeout(std::time::Duration::from_millis(300));
        assert_eq!(after, Err(mpsc::RecvTimeoutError::Timeout));
    }

    /// A wait the runtime refuses hangs where Metal's never returns, a handler's wait in its own
    /// queue or past the device's handler threads, and any once the process is exiting, and dies
    /// for the causes where Metal's answer is not measured while it is not; only those ask
    /// whether the process is exiting. A live process is not exiting
    #[test]
    fn refused_waits_hang_only_where_metal_does() {
        use rt::DeadlockCause::{Cycle, Exiting, HandlerThreadCap, OwnQueue};
        let asked = &std::cell::Cell::new(0);
        let exiting = |answer| {
            move || {
                asked.set(asked.get() + 1);
                answer
            }
        };
        for answer in [false, true] {
            for cause in [OwnQueue, HandlerThreadCap, Exiting] {
                assert_eq!(refused_wait(cause, exiting(answer)), RefusedWait::Hangs);
            }
        }
        assert_eq!(asked.get(), 0);
        assert_eq!(refused_wait(Cycle, exiting(false)), RefusedWait::Unmeasured);
        assert_eq!(refused_wait(Cycle, exiting(true)), RefusedWait::Hangs);
        assert_eq!(asked.get(), 2);
        assert!(!rt::process_exiting());
    }

    /// Two queues' completed handlers that wait for each other's command buffer close a cycle the
    /// runtime refuses: the wait dies at its selector, as Metal's answer there is not measured
    #[test]
    fn a_cycle_of_handlers_waits_dies() {
        use reims_vgpu_core::raise::death::DeathSite;
        use reims_vgpu_core::testkit::capture::expect_death;

        let site = DeathSite {
            kind: DeathKind::Abort,
            file: "waitUntilCompleted",
            line: 0,
        };
        // The process's device, made before `expect_death`'s clock starts, which times the cycle
        // alone: making it compiles what the runtime's shared caches hold, which can take longer
        // than the clock allows.
        let d = dev();
        expect_death(site, move || {
            let (qa, qb) = (
                d.new_command_queue().unwrap(),
                d.new_command_queue().unwrap(),
            );
            let event = d.new_shared_event().unwrap();
            let (a, b) = (buffer(&qa), buffer(&qb));
            b.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            let (to_a, to_b) = (a.clone_ref(), b.clone_ref());
            let (tx, rx) = mpsc::channel();
            a.add_completed_handler(
                OnFault::Return,
                Box::new(move |_| {
                    tx.send(()).unwrap();
                    let _ = to_b.wait_until_completed(OnFault::Return);
                }),
            )
            .unwrap();
            b.add_completed_handler(
                OnFault::Return,
                Box::new(move |_| {
                    let _ = to_a.wait_until_completed(OnFault::Return);
                }),
            )
            .unwrap();
            a.commit(OnFault::Return).unwrap();
            b.commit(OnFault::Return).unwrap();
            rx.recv().unwrap();
            event.set_signaled_value(1).unwrap();
            a.wait_until_completed(OnFault::Return).unwrap();
        });
    }

    /// Runs `body` in a child copy of this test binary, its deaths ending it as in production,
    /// and answers how the child ended: what an exit does is the process's, which only a child
    /// can show.
    fn in_child(test: &str, body: impl FnOnce()) -> std::process::ExitStatus {
        if std::env::var("RVG_CHILD").as_deref() == Ok(test) {
            reims_vgpu_core::testkit::capture::set_capture_enabled(false);
            body();
            std::process::exit(0);
        }
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env("RVG_CHILD", test)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
    }

    /// An exit while a completed handler waits for another queue's command buffer, which the
    /// runtime refuses once the process is exiting, leaves the handler parked, as Metal's wait
    /// never returns there: the process exits as asked, and dies at nothing
    #[test]
    fn an_exit_parks_a_handlers_wait_for_another_queue() {
        /// An exit handler, registered before the runtime's, that keeps the process past the
        /// runtime's refusal of the blocked wait (within 50 ms of its exit handler).
        extern "C" fn linger() {
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        unsafe extern "C" {
            fn atexit(f: extern "C" fn()) -> std::ffi::c_int;
        }
        let status = in_child(
            "gpu::command::tests::an_exit_parks_a_handlers_wait_for_another_queue",
            || {
                // SAFETY: `linger` takes nothing and never unwinds.
                assert_eq!(unsafe { atexit(linger) }, 0);
                let d = dev();
                let (exec_queue, paging_queue) = (
                    d.new_command_queue().unwrap(),
                    d.new_command_queue().unwrap(),
                );
                let never = d.new_shared_event().unwrap();
                let exec = buffer(&exec_queue);
                exec.encode_wait(Some(EventRef::Shared(&never)), 1).unwrap();
                exec.commit(OnFault::Return).unwrap();
                let paging = buffer(&paging_queue);
                let waited = exec.clone_ref();
                let (tx, rx) = mpsc::channel();
                paging
                    .add_completed_handler(
                        OnFault::Return,
                        Box::new(move |_| {
                            tx.send(()).unwrap();
                            let _ = waited.wait_until_completed(OnFault::Die);
                        }),
                    )
                    .unwrap();
                paging.commit(OnFault::Return).unwrap();
                rx.recv().unwrap();
                std::thread::sleep(std::time::Duration::from_millis(100));
            },
        );
        assert_eq!(status.code(), Some(0), "{status:?}");
    }

    /// A host step's refusal of a command buffer that executes an indirect command buffer dies
    /// as the command buffer ends, at the execution's selector, outside isolation and while the
    /// command buffer's task runs; once the task ended, as core's handlers of it run none, it
    /// dies at nothing, and the command buffer's progress ends all the same
    #[test]
    fn a_refused_command_buffer_of_an_ended_task_dies_at_nothing() {
        use reims_vgpu_core::platform::values::IcbDesc;

        let d = dev();
        let desc = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        let icb = d.new_indirect_command_buffer(&desc, 1, 0).unwrap().unwrap();
        let sel = "executeCommandsInBuffer:withRange:";
        let ending = |task| {
            let progress = Arc::new(Progress::default());
            progress.executes(sel, icb.clone_ref());
            let ending = Ending {
                progress: Some(Arc::clone(&progress)),
                task,
                carried: refusals::carry(),
                slot: None,
            };
            (progress, ending)
        };
        let refused = |sel| {
            unrepresentable(sel, "a host step's refusal");
        };
        let life = TaskLife::new();
        for task in [None, Some(Arc::clone(&life))] {
            let (_, mut e) = ending(task);
            let site = catch_death(|| e.end(Stage::Failed, Some(refused))).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        }
        let (progress, mut e) = ending(Some(Arc::clone(&life)));
        assert!(life.exit());
        e.end(Stage::Failed, Some(refused));
        assert_eq!(progress.stage(), Stage::Failed);
        // Made by an op of the graphics lane's, in its refusal scope: the death is refused on the
        // handler's thread, and the progress ends all the same.
        let (progress, mut e) = refusals::refusing(|| ending(None)).unwrap();
        std::thread::Builder::new()
            .name("the runtime's progress".into())
            .spawn(move || e.end(Stage::Failed, Some(refused)))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(progress.stage(), Stage::Failed);
    }

    /// A command buffer's progress ends on the runtime's thread in the scope its command carried
    /// (the graphics lane's): a change that dies at the completion and an object whose release
    /// dies are refused there, the other changes run, the other objects go, and the stage is
    /// set. Outside every scope (the paravirt device's), the same release dies, as before
    #[test]
    fn a_progress_ends_in_the_scope_its_command_carried() {
        /// An object whose release dies, as the platform's release of what a command used can.
        struct DiesAtRelease;
        impl Drop for DiesAtRelease {
            fn drop(&mut self) {
                unrepresentable("release", "a release none of Metal's");
            }
        }
        let ending = |progress: &Arc<Progress>| Ending {
            progress: Some(Arc::clone(progress)),
            task: None,
            carried: refusals::carry(),
            slot: None,
        };
        let (ran, gone) = (Arc::new(AtomicUsize::new(0)), Arc::new(()));
        let progress = Arc::new(Progress::default());
        progress.at_completion(|| unrepresentable("commit", "a change none of Metal's"));
        let after = Arc::clone(&ran);
        progress.at_completion(move || {
            after.fetch_add(1, Ordering::SeqCst);
        });
        progress.hold(DiesAtRelease);
        progress.hold(Arc::clone(&gone));
        let mut e = refusals::refusing(|| ending(&progress)).unwrap();
        std::thread::Builder::new()
            .name("the runtime's progress".into())
            .spawn(move || e.end(Stage::Completed, None::<fn(&'static str)>))
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(progress.stage(), Stage::Completed);
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "the change after the refused one"
        );
        assert_eq!(
            Arc::strong_count(&gone),
            1,
            "the object held after the refused one"
        );
        // Outside every scope the release dies.
        let progress = Arc::new(Progress::default());
        progress.hold(DiesAtRelease);
        let mut e = ending(&progress);
        let site = catch_death(|| e.end(Stage::Completed, None::<fn(&'static str)>)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "release"));
    }

    /// Core's completed handler added inside a refusal scope (the graphics lane's) reads a failure
    /// that dies in that scope on the runtime's thread: the handler is handed the command buffer
    /// failed by the refusal, and the process goes on. Added outside every scope (the paravirt
    /// device's), the same read dies, as before
    #[test]
    fn a_completed_handlers_read_in_a_scope_fails_its_command_buffer() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let handler = |seen: mpsc::Sender<CompletedInfo>| {
            let block: CompletedHandler = Box::new(move |info| seen.send(info).unwrap());
            completed_handler(block, None, |_| {
                unrepresentable("commit", "a failure none of Metal's");
            })
        };
        let (tx, rx) = mpsc::channel();
        let h = refusals::refusing(|| handler(tx)).unwrap();
        let raw = cb.raw.clone();
        std::thread::Builder::new()
            .name("the runtime's completion".into())
            .spawn(move || h(&raw))
            .unwrap()
            .join()
            .unwrap();
        let info = rx.recv().unwrap();
        assert!(info.is_refused(), "{info:?}");
        assert_eq!(info.status, CommandBufferStatus::ERROR);
        let description = info.error.unwrap().description;
        assert!(description.contains("commit"), "{description}");
        // Outside every scope the read dies, and the handler is not run.
        let (tx, rx) = mpsc::channel();
        let h = handler(tx);
        let site = catch_death(|| h(&cb.raw)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
        assert!(rx.try_recv().is_err());
    }

    /// Core's completed handler reads a failure that dies as the code of the isolated task that
    /// added it: outside isolation and while the task runs it dies, and once the task ended it
    /// dies at nothing, the handler still handed the command buffer's status
    #[test]
    fn a_completed_handlers_read_of_an_ended_tasks_failure_dies_at_nothing() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let handler = |task, seen: mpsc::Sender<CompletedInfo>| {
            let block: CompletedHandler = Box::new(move |info| seen.send(info).unwrap());
            completed_handler(block, task, |_| {
                unrepresentable("commit", "a failure none of Metal's");
            })
        };
        let life = TaskLife::new();
        for task in [None, Some(Arc::clone(&life))] {
            let (tx, rx) = mpsc::channel();
            let h = handler(task, tx);
            let site = catch_death(|| h(&cb.raw)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
            assert!(rx.try_recv().is_err());
        }
        // Added by the task's code, as core adds it: the task is the handler's.
        let (tx, rx) = mpsc::channel();
        let h = life.run(|| handler(TaskLife::current(), tx)).unwrap();
        assert!(life.exit());
        h(&cb.raw);
        assert_eq!(rx.recv().unwrap().status, CommandBufferStatus::COMPLETED);
    }

    /// A thread of an isolated task that ended stops at a command buffer's send before anything
    /// of it dies or takes effect: an event send of a nil event, an encoder's creation, and a
    /// commit whose unit of compute encoders holds an open block of command predication, each
    /// of which dies while the task runs, stop, and the command buffer stays uncommitted
    #[test]
    fn an_ended_tasks_sends_stop_before_they_die() {
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        let condition = crate::gpu::resource::tests::shared(16);
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let start = ComputeCmd::EncodeStartIf {
            buffer: Some(&condition),
            offset: 0,
            comparison: 0,
            reference_value: 0,
        };
        e.encode(OnFault::Return, start).unwrap();
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let sends: [(&str, &dyn Fn()); 6] = [
            ("encodeSignalEvent:value:", &|| {
                let _ = cb.encode_signal(None, 1);
            }),
            ("encodeSignalEventScheduled:value:", &|| {
                let _ = cb.encode_signal_when_scheduled(None, 1);
            }),
            ("encodeWaitForEvent:value:", &|| {
                let _ = cb.encode_wait(None, 1);
            }),
            ("blitCommandEncoder", &|| {
                let _ = cb.blit_encoder(OnFault::Return);
            }),
            ("renderCommandEncoderWithDescriptor:", &|| {
                use reims_vgpu_core::platform::Platform as _;
                let pass = crate::platform::tests::platform().render_pass_descriptor();
                let _ = cb.render_encoder(OnFault::Return, &pass);
            }),
            ("commit", &|| {
                let _ = cb.commit(OnFault::Return);
            }),
        ];
        for (sel, send) in sends {
            let site = catch_death(send).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
            let life = TaskLife::new();
            let stopped = life.run(|| {
                life.exit();
                send();
            });
            assert!(stopped.is_none(), "{sel}");
        }
        assert_eq!(cb.status(), CommandBufferStatus::NOT_ENQUEUED);
    }

    /// A thread of an isolated task that ended stops at commit before the runtime's commit, and
    /// the command buffer's progress stays uncommitted: what it uses is not held as the GPU's
    #[test]
    fn an_ended_tasks_commit_leaves_its_command_buffer_uncommitted() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        let progress = cb.progress("test");
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            cb.commit(OnFault::Return)
        });
        assert!(stopped.is_none());
        assert_eq!(progress.stage(), Stage::Encoding);
        assert_eq!(cb.status(), CommandBufferStatus::NOT_ENQUEUED);
    }

    /// A failure of the GPU's work the driver relayed reaches core with Metal's code where its
    /// code is one of Metal's, and dies where it is none
    #[test]
    fn relayed_failures_take_metals_codes_or_die() {
        for code in METAL_CODES {
            relayed("commit", code, None);
            let e = command_buffer_error(rt::CommandBufferError::Execution(code), None);
            assert_eq!(
                (e.domain.as_str(), e.code),
                ("MTLCommandBufferErrorDomain", i64::try_from(code).unwrap())
            );
        }
        for code in [0, 5, 6, 13, 15, 17, u64::MAX] {
            let site = catch_death(|| relayed("waitUntilCompleted", code, Some("x".to_owned())))
                .unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, "waitUntilCompleted"),
                "{code}"
            );
        }
    }

    /// A relayed failure's description and death line carry Metal's description beside the
    /// runtime's error, as the driver's message gave it (MoltenVK 1.4.1's forms), and the error
    /// alone where the message gave none; a detached object's use keeps Metal's own description
    #[test]
    fn relayed_failures_name_metals_description() {
        use rt::CommandBufferError as E;
        let metal = "Discarded (victim of GPU error/recovery)";
        let e = command_buffer_error(E::Execution(9), Some(metal.to_owned()));
        assert_eq!(
            (e.code, e.description.as_str()),
            (
                9,
                "the Vulkan runtime's command buffer failed: Execution(9) (Metal's description: \
                 Discarded (victim of GPU error/recovery))"
            )
        );
        let fault =
            command_buffer_error(E::PageFault, Some("Caused GPU Address Fault Error".into()));
        assert_eq!(
            (fault.code, fault.description.as_str()),
            (
                3,
                "the Vulkan runtime's command buffer failed: PageFault (Metal's description: \
                 Caused GPU Address Fault Error)"
            )
        );
        for none in [None, Some(String::new())] {
            let e = command_buffer_error(E::Execution(9), none);
            assert_eq!(
                e.description,
                "the Vulkan runtime's command buffer failed: Execution(9)"
            );
        }
        let detached = command_buffer_error(E::Injected(backing::DETACHED_USE), Some(metal.into()));
        assert_eq!(
            detached.description,
            "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        );
        assert!(relayed_line(13, Some(metal)).ends_with(
            "with code 13, which no failure of Metal's command buffers has (Metal's description: \
             Discarded (victim of GPU error/recovery))"
        ));
        assert!(
            relayed_line(13, Some("")).ends_with("which no failure of Metal's command buffers has")
        );
    }

    /// A command buffer's failure that is one of Metal's reaches core; every failure of the
    /// runtime's own dies at the selector that reads it first, at the commit's and at the wait, its
    /// line naming it: a Vulkan failure of a submission, one nothing here matched to Metal's, and
    /// an indirect draw's indices past the index pool. The fourteen this platform lists as Metal's are those the runtime states as
    /// Metal's, and the others it names its own: a change of either at a new pin fails here, for
    /// review
    #[test]
    fn failures_metal_has_none_of_die_named() {
        use rt::CommandBufferError as E;
        let metals = [
            E::Injected(8),
            E::DeviceLost,
            E::OutOfMemory,
            E::PageFault,
            E::GpuHang,
            E::GpuRecoveryVictim,
            E::AccessRevoked,
            E::RenderTargetSize,
            E::DetachedBacking,
            E::ProtectedEncoder(1),
            E::PredicationOpenAtUnitEnd { record: 0 },
            E::DefaultRasterSampleCount,
            E::InvalidResource,
            E::Execution(1),
        ];
        for e in metals {
            assert_eq!(e.origin(), rt::FailureOrigin::Metal, "{e:?}");
            answered("commit", e, None);
        }
        let named = [
            (E::Internal(rt::VkResultCode(-13)), "VkResult -13"),
            (E::IndexPoolOverflow, "index pool"),
        ];
        for (e, _) in named {
            assert_eq!(e.origin(), rt::FailureOrigin::Runtime, "{e:?}");
        }
        for sel in ["commit", "waitUntilCompleted"] {
            for (e, names) in named {
                *crate::answer::LAST_LINE.lock() = None;
                let site = catch_death(|| answered(sel, e, None)).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
                assert!(line.contains(names), "{e:?}: {line}");
            }
        }
    }

    /// A refusal of a commit dies at the commit naming the runtime's refusal; only one the
    /// runtime names an execution's, of the records of an indirect command buffer (a command's
    /// pipeline state whose colour formats the pass does not hold, which the runtime checks as
    /// it replays the records), says so
    #[test]
    fn a_commit_blames_the_records_only_for_their_refusal() {
        use rt::{Invalid, Problem, RuntimeGap};

        let records = |call| {
            rt::Error::Invalid(Invalid::new(
                call,
                Problem::Draw(rt::DrawProblem::AttachmentFormat(rt::Attachment::Color(0))),
            ))
        };
        for call in [
            "execute_commands_in_buffer",
            "execute_commands_in_buffer_indirect",
        ] {
            assert!(of_records(&records(call)));
            let site = catch_death(|| committing(|| Err(records(call)))).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, "commit"));
        }
        for other in [
            rt::Error::Invalid(Invalid::new("commit", Problem::AlreadyCommitted)),
            rt::Error::NotImplemented(RuntimeGap::SparseHeaps),
            rt::Error::DeviceLost,
        ] {
            assert!(!of_records(&other));
            assert_eq!(committing(|| Err(other.clone())), Err(other));
        }
    }

    /// A Shared buffer of 16 zero bytes, and a blit encoder of `cb` that fills it with `value`
    /// and ends.
    fn filled(cb: &VulkanCommandBuffer, value: u8) -> crate::gpu::resource::VulkanBuffer {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::types::NsRange;
        let b = crate::gpu::resource::tests::shared(16);
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let range = NsRange::new(0, 16);
        let fill = BlitCmd::FillBuffer {
            buffer: &b,
            range,
            value,
        };
        blit.encode(OnFault::Return, fill).unwrap();
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        b
    }

    /// The 16 bytes of a Shared buffer.
    fn sixteen(b: &crate::gpu::resource::VulkanBuffer) -> Vec<u8> {
        use reims_vgpu_core::platform::resource::Buffer as _;
        // SAFETY: a Shared buffer of 16 bytes, read after its command buffer completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), 16) }.to_vec()
    }

    /// `cb` committed and completed: its status and its error's code.
    fn completed(cb: &VulkanCommandBuffer) -> (u64, Option<i64>) {
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        (cb.status().0, cb.error().map(|e| e.code))
    }

    /// Protection options fail a command buffer as Metal's GPU does: an encoder made under
    /// options below 0x10 runs nothing and fails it with code 16, one made before them runs,
    /// and none of its event signals fires; options from 0x10 up fail it with code 1, with an
    /// encoder or without; options below 0x10 with no encoder under them, or 0 again before
    /// one, complete it
    #[test]
    fn protection_options_fail_command_buffers_as_metals_gpu_does() {
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let event = d.new_shared_event().unwrap();
        let before = filled(&cb, 0x5a);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        let under = filled(&cb, 0x6b);
        let signal = Some(EventRef::Shared(&event));
        cb.encode_signal(signal, 7).unwrap();
        assert_eq!(completed(&cb), (5, Some(16)));
        assert_eq!(
            (sixteen(&before), sixteen(&under)),
            (vec![0x5a; 16], vec![0; 16])
        );
        assert_eq!(event.signaled_value(), 0);
        for options in [0x10, u64::MAX] {
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            assert_eq!(completed(&cb), (5, Some(1)), "{options:#x}, no encoder");
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            let under = filled(&cb, 0x6b);
            assert_eq!(completed(&cb), (5, Some(1)), "{options:#x}");
            assert_eq!(sixteen(&under), vec![0; 16]);
        }
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        assert_eq!(completed(&cb), (4, None), "options 2, no encoder");
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        cb.set_protection_options(OnFault::Return, 0).unwrap();
        let after = filled(&cb, 0x7c);
        assert_eq!(completed(&cb), (4, None), "options 2, then 0");
        assert_eq!(sixteen(&after), vec![0x7c; 16]);
    }

    /// Under protection options 1, 3 and 7 a compute or render encoder's creation dies, and so
    /// does a blit encoder's first command, as Metal ends the process there; with no encoder the
    /// command buffer completes. Other options once an encoder was made under some, or after
    /// options from 0x10 up, die
    #[test]
    fn protection_options_that_end_the_process_die() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        let q = dev().new_command_queue().unwrap();
        for options in [1, 3, 7] {
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            let site = catch_death(|| cb.compute_encoder(OnFault::Return)).unwrap_err();
            assert_eq!(site.file, "computeCommandEncoder");
            let pass = crate::platform::tests::platform().render_pass_descriptor();
            let site = catch_death(|| cb.render_encoder(OnFault::Return, &pass)).unwrap_err();
            assert_eq!(site.file, "renderCommandEncoderWithDescriptor:");
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            let end = || blit.encode(OnFault::Return, BlitCmd::EndEncoding);
            assert_eq!(catch_death(end).unwrap_err().file, "endEncoding");
            let cb = buffer(&q);
            cb.set_protection_options(OnFault::Return, options).unwrap();
            assert_eq!(completed(&cb), (4, None), "options {options}, no encoder");
        }
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        filled(&cb, 0x11);
        cb.set_protection_options(OnFault::Return, 2).unwrap();
        let site = catch_death(|| cb.set_protection_options(OnFault::Return, 0)).unwrap_err();
        assert_eq!(site.file, "setProtectionOptions:");
        let cb = buffer(&q);
        cb.set_protection_options(OnFault::Return, 0x10).unwrap();
        cb.set_protection_options(OnFault::Return, 0x10).unwrap();
        let site = catch_death(|| cb.set_protection_options(OnFault::Return, 0)).unwrap_err();
        assert_eq!(site.file, "setProtectionOptions:");
    }

    /// A command buffer Metal's GPU fails in one way fails so at its commit, from the first
    /// encoder that fails; one that only uses a detached object is left to its submissions, and
    /// one only its draws of primitive type 9 fail, of one kind, to the runtime. One it would
    /// fail in two, a use of a detached object and a failure of its own or a draw of type 9's,
    /// or a draw of type 9's and another of its own or of another kind, dies at its commit, as
    /// nothing measured it
    #[test]
    fn a_command_buffer_failing_two_ways_dies_at_its_commit() {
        let q = dev().new_command_queue().unwrap();
        let cb = buffer(&q);
        assert!(
            cb.failure("commit", || true).is_none(),
            "a detached use alone"
        );
        for from in [3, 1] {
            cb.fail(Failing {
                code: PROTECTION_VIOLATION,
                from,
                signals: false,
            });
        }
        let failure = cb.failure("commit", || false).unwrap();
        assert_eq!(
            (failure.code, failure.encoders, failure.signals),
            (PROTECTION_VIOLATION, rt::FailedEncoders::From(1), false)
        );
        let site = catch_death(|| cb.failure("commit", || true)).unwrap_err();
        assert_eq!(site.file, "commit");
        let hang = Failing {
            code: rt::CommandBufferError::GpuHang.metal_code(),
            from: 2,
            signals: true,
        };
        cb.fails_at_type_9(hang);
        let site = catch_death(|| cb.failure("commit", || false)).unwrap_err();
        assert_eq!(site.file, "commit");
        let cb = buffer(&q);
        for from in [2, 4] {
            cb.fails_at_type_9(Failing { from, ..hang });
        }
        assert!(cb.failure("commit", || false).is_none(), "type 9 alone");
        let site = catch_death(|| cb.failure("commit", || true)).unwrap_err();
        assert_eq!(site.file, "commit");
        cb.fails_at_type_9(Failing {
            code: rt::CommandBufferError::PageFault.metal_code(),
            ..hang
        });
        let site = catch_death(|| cb.failure("commit", || false)).unwrap_err();
        assert_eq!(site.file, "commit");
    }

    /// A command buffer the runtime failed for a command's buffer rebound before its first
    /// submission (`CommandBufferError::EncodedUseRebound`) dies at the selector where its
    /// failure is read, at the commit's (a completed handler's, its status or error) and at the
    /// wait, its line naming the rebind and where Metal runs the command
    #[test]
    fn an_encoded_use_rebound_dies_naming_the_rebind() {
        for sel in ["commit", "waitUntilCompleted"] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| {
                encoded_use_rebound(sel);
            })
            .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains("rebound after the command was encoded"),
                "{line}"
            );
            assert!(line.ends_with("the buffer lies in as it submits"), "{line}");
        }
    }

    /// While an encoder is open, what Metal asserts at dies whatever the handler: another
    /// encoder, an event, setProtectionOptions:, commit; once it ends, each is taken. An encoder
    /// that goes without endEncoding is open no more, but abandons its command buffer, whose
    /// setProtectionOptions: then dies, naming the runtime's refusal (`EncoderAbandoned`), where
    /// plain Metal asserts as the encoder goes (the runtime's statement)
    #[test]
    fn an_open_encoder_kills_what_metal_asserts_at() {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        let d = dev();
        let q = d.new_command_queue().unwrap();
        let cb = buffer(&q);
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let event = d.new_shared_event().unwrap();
        let dies = |sel: &str, call: &dyn Fn()| {
            let site = catch_death(call).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        };
        dies("computeCommandEncoder", &|| {
            let _ = cb.compute_encoder(OnFault::Return);
        });
        dies("encodeSignalEvent:value:", &|| {
            let _ = cb.encode_signal(Some(EventRef::Shared(&event)), 1);
        });
        dies("setProtectionOptions:", &|| {
            let _ = cb.set_protection_options(OnFault::Return, 0);
        });
        dies("commit", &|| {
            let _ = cb.commit(OnFault::Return);
        });
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.set_protection_options(OnFault::Return, 0).unwrap();
        cb.encode_signal(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // An encoder that goes without endEncoding is open no more, and abandons the buffer.
        let cb = buffer(&q);
        drop(cb.blit_encoder(OnFault::Return).unwrap().unwrap());
        *crate::answer::LAST_LINE.lock() = None;
        dies("setProtectionOptions:", &|| {
            let _ = cb.set_protection_options(OnFault::Return, 0);
        });
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        assert!(line.contains("EncoderAbandoned"), "{line}");
    }
}
