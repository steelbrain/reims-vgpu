//! Platform trait: `CommandQueue`, `CommandBuffer`: queue and command-buffer selectors.

use super::event::EventRef;
use super::types::{CommandBufferStatus, NsErrorText};
use super::values::ComputePassDesc;
use super::{Handle, Platform};
use crate::core::raise::{OnFault, RaiseResult};

/// What a completed handler learns of its buffer: core's handlers read `status`, `error` and the
/// GPU times of the buffer they were added to, on Metal's completion thread. The platform reads
/// them when the handler runs; no getter has an effect.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompletedInfo {
    /// `-[MTLCommandBuffer status]` when the handler runs.
    pub status: CommandBufferStatus,
    /// `-[MTLCommandBuffer error]` when the handler runs; `None` for nil.
    pub error: Option<NsErrorText>,
    /// `-[MTLCommandBuffer GPUStartTime]` and `GPUEndTime` when the handler runs, in nanoseconds
    /// of the platform's host clock ([`CompletedInfo::gpu_times`]); `None` where the platform
    /// answers none. Only the graphics lane reads them, into its guest's `CB_COMPLETE`.
    pub gpu_times_ns: Option<(u64, u64)>,
}

/// What [`CommandBuffer::commit_part`] made of a command buffer at the end of a range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartCut {
    /// Committed as a part: the rest goes on in the command buffer [`CommandBuffer::part`] made.
    Cut,
    /// Not committed: it fails from one of its encoders on, every one after it running nothing.
    Fails,
    /// Not committed: it may not be cut, for the reason given, and runs every encoder encoded
    /// into it.
    Whole(KeptWhole),
}

/// Why a command buffer may not be cut into parts, but for how Metal's GPU fails it from an
/// encoder on ([`PartCut::Fails`]), as the Vulkan platform keeps it whole ([`PartCut::Whole`];
/// `gpu/command.rs`'s `CommandBufferObject::splits`): each set by the commands named, so a
/// caller that walks a buffer's chunks on two threads at once takes none whose commands set one
/// (`reims-vgpu`'s `decode/cmdbuf.rs`, `deal`, whose tests pin the two lists together).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeptWhole {
    /// An encoder is open.
    OpenEncoder,
    /// Protection options are set, or an encoder was made under them (`setProtectionOptions:`).
    ProtectionOptions,
    /// An indirect command buffer's execution, whose commands its commit reads
    /// (`executeCommandsInBuffer:…`, render or compute).
    Execution,
    /// Its progress is kept to its completion: a pass of a rasterization rate map, and an
    /// indirect command buffer's execution, reset or copy, or residency (`useResource:`), in a
    /// render, compute or blit encoder.
    Progress,
    /// A block of command predication is open, or a unit's record emptied it (compute
    /// encoders).
    Predication,
    /// An event is signalled as it is scheduled (`encodeSignalEventScheduled:value:`).
    ScheduledSignal,
    /// It was committed.
    Committed,
}

impl KeptWhole {
    /// Every reason, in the order the Vulkan platform asks them.
    pub const ALL: [KeptWhole; 7] = [
        KeptWhole::OpenEncoder,
        KeptWhole::ProtectionOptions,
        KeptWhole::Execution,
        KeptWhole::Progress,
        KeptWhole::Predication,
        KeptWhole::ScheduledSignal,
        KeptWhole::Committed,
    ];

    /// Its name in the log.
    pub fn code(self) -> &'static str {
        match self {
            KeptWhole::OpenEncoder => "open-encoder",
            KeptWhole::ProtectionOptions => "protection-options",
            KeptWhole::Execution => "execution",
            KeptWhole::Progress => "progress",
            KeptWhole::Predication => "predication",
            KeptWhole::ScheduledSignal => "scheduled-signal",
            KeptWhole::Committed => "committed",
        }
    }
}

/// The domain of the error a command buffer completes with where the platform's read of it, as
/// its completed handler ran, died in the refusal scope the handler's registration carried
/// ([`crate::core::raise::refusals::carry`]): the platform failed it there, where outside every scope
/// the death ends the process. Not Metal's: only the graphics lane's handlers see it.
pub const REFUSED_DOMAIN: &str = "reims-vgpu.refused";

/// `MTLCommandBufferErrorInternal`, the code of a refused command buffer's error.
const ERROR_INTERNAL: i64 = 1;

thread_local! {
    /// How deep this thread is in platform commits that may run submission handlers inside the
    /// call ([`committing`]).
    static COMMITTING: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Runs `commit`, a platform's commit that may run the submission handlers of any command buffer
/// of the process inside the call, on this thread (the Vulkan platform's runtime submits there the
/// ready batches committed before it, its own among them), marked so a handler run inside it can
/// tell ([`inside_commit`]). A platform whose submission handlers run only on its own threads
/// (Metal's) marks none.
pub fn committing<R>(commit: impl FnOnce() -> R) -> R {
    /// Leaves the mark on drop, an unwinding commit's included.
    struct Leave;
    impl Drop for Leave {
        fn drop(&mut self) {
            COMMITTING.with(|c| c.set(c.get() - 1));
        }
    }
    COMMITTING.with(|c| c.set(c.get() + 1));
    let _leave = Leave;
    commit()
}

/// Whether this thread is inside a platform commit that may run submission handlers
/// ([`committing`]): a submission handler that runs where this is false runs on the platform's
/// own thread.
pub fn inside_commit() -> bool {
    COMMITTING.with(|c| c.get() > 0)
}

impl CompletedInfo {
    /// What a completed handler learns of a command buffer whose read the platform refused
    /// ([`REFUSED_DOMAIN`]): status `Error`, and an internal error naming the refusal.
    pub fn refused(refusal: &crate::core::raise::refusals::Refusal) -> CompletedInfo {
        CompletedInfo {
            status: CommandBufferStatus::ERROR,
            error: Some(NsErrorText {
                domain: REFUSED_DOMAIN.to_owned(),
                code: ERROR_INTERNAL,
                description: refusal.to_string(),
            }),
            gpu_times_ns: None,
        }
    }

    /// [`CompletedInfo::gpu_times_ns`] of a buffer whose `GPUStartTime` and `GPUEndTime` are
    /// `start` and `end`, in seconds of host time: `None` where either is 0 (Metal's answer for a
    /// buffer the GPU did not run) or not finite, or the run ends before it starts.
    pub fn gpu_times(start: f64, end: f64) -> Option<(u64, u64)> {
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a finite positive f64 to u64 rounds, saturating at u64::MAX"
        )]
        let ns = |t: f64| (t.is_finite() && t > 0.0).then(|| (t * 1e9).round() as u64);
        let (start, end) = (ns(start)?, ns(end)?);
        (end >= start).then_some((start, end))
    }

    /// Whether the platform refused the command buffer's read ([`CompletedInfo::refused`]).
    pub fn is_refused(&self) -> bool {
        self.error
            .as_ref()
            .is_some_and(|e| e.domain == REFUSED_DOMAIN)
    }
}

/// Runs completed handler `block` with what `read` finds of its command buffer, in the scope
/// `carried` names, the one the handler's registration carried ([`crate::core::raise::refusals::carry`]):
/// a death in `read` hands `block` [`CompletedInfo::refused`] instead, and a death in `block` ends
/// only `block`, each logged. Outside every scope `read` and `block` run plainly, one after the
/// other, as they always did.
pub fn run_completed(
    carried: crate::core::raise::refusals::Carried,
    read: impl FnOnce() -> CompletedInfo,
    block: CompletedHandler,
) {
    let info = carried.run(read).unwrap_or_else(|refusal| {
        log::warn!(
            target: "reims_vgpu::refused", kind:% = refusal;
            "a command buffer's completion was refused, and the command buffer failed: {refusal}"
        );
        CompletedInfo::refused(&refusal)
    });
    if let Err(refusal) = carried.run(|| block(info)) {
        log::warn!(
            target: "reims_vgpu::refused", kind:% = refusal;
            "a command buffer's completed handler was refused: {refusal}"
        );
    }
}

/// A completed handler (`addCompletedHandler:`'s block). Runs once, on a Metal thread (4.2 item 6).
/// Where it is added inside a refusal scope, the platform runs its read of the buffer and the
/// handler in that scope ([`crate::core::raise::refusals::Carried`]): a death in the read hands the
/// handler [`CompletedInfo::refused`], and a death in the handler ends only the handler.
pub type CompletedHandler = Box<dyn FnOnce(CompletedInfo) + Send>;

/// A scheduled handler (`addScheduledHandler:`'s block): runs once, on a platform thread, once
/// the command buffer is scheduled (every wait of it satisfied and its work submitted), before
/// its completed handlers.
pub type ScheduledHandler = Box<dyn FnOnce() + Send>;

/// `id<MTLCommandQueue>`: the device's `reims-vgpu.blit-up`, `reims-vgpu.blit-down` and
/// `reims-vgpu.exec`, and the service's queues.
/// Core never nil-checks one.
pub trait CommandQueue<P: Platform>: Handle {
    /// sel: commandBuffer — core never checks for nil; later sends
    /// go to nil (`nil_send!`).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn command_buffer(&self, on: OnFault) -> RaiseResult<Option<P::CommandBuffer>>;
    /// sel: setLabel: — (`reims-vgpu.blit-up`, `reims-vgpu.blit-down`,
    /// `reims-vgpu.exec`). No path reaches it under a handler: a raise inside it dies at the send.
    fn set_label(&self, label: &str);
}

/// A command buffer's completion as a sync file, which another device waits for on its GPU
/// ([`CommandBuffer::export_completion_sync_file`]): an open descriptor on Linux and macOS.
#[cfg(unix)]
pub type SyncFile = std::os::fd::OwnedFd;
/// On Windows the runtime exports no sync file, so the type has no values, and a caller always
/// waits for a command buffer on the host.
#[cfg(windows)]
pub type SyncFile = std::convert::Infallible;

/// `id<MTLCommandBuffer>` (`MTLCommandBufferSPI` included).
pub trait CommandBuffer<P: Platform>: Handle {
    /// sel: addCompletedHandler:
    /// The handler runs on a Metal thread after completion or error; a thread with no handler
    /// (6.2).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn add_completed_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()>;
    /// sel: addScheduledHandler:
    /// The handler runs on a platform thread once the buffer is scheduled, before its
    /// completed handlers. Reached both under a handler and with none: `on` says which this
    /// call is.
    fn add_scheduled_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()>;
    /// The platform's submission point, no Metal call: the handler runs once the platform has
    /// taken the buffer's work for the GPU, its place in its queue's order fixed, on the thread
    /// that took it, before the buffer is scheduled and before it can complete; a buffer that
    /// fails before then may run none. It must be short, and wait for no GPU work. Metal tells of
    /// no point before scheduling, so its scheduled handler stands for it there. Reached both
    /// under a handler and with none: `on` says which this call is.
    fn add_submission_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()>;
    /// The platform's completion as seen, no Metal call: the handler runs once the buffer
    /// completed or failed, with what a completed handler is given, on the platform's thread
    /// that saw it, before the buffer's completed handlers, in its queue's commit order; a buffer
    /// with such handlers and no completed handler wakes no handler thread. It must be short and
    /// never block. Metal tells of a completion only on its completed handlers' thread, so a
    /// completed handler stands for it there. Reached both under a handler and with none: `on`
    /// says which this call is.
    fn add_completion_seen_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()>;
    /// sel: blitCommandEncoder — May return nil, which core never
    /// checks (the blit decoder's init); later sends to it are no-ops (`nil_send!`).
    ///
    /// Reached both under a handler and with none: `on` says which this call is.
    fn blit_encoder(&self, on: OnFault) -> RaiseResult<Option<P::BlitEncoder>>;
    /// sel: commit
    /// Reached both under a handler and with none: `on` says which this call is.
    fn commit(&self, on: OnFault) -> RaiseResult<()>;
    /// sel: computeCommandEncoder
    /// Reached both under a handler and with none: `on` says which this call is.
    fn compute_encoder(&self, on: OnFault) -> RaiseResult<Option<P::ComputeEncoder>>;
    /// sel: device
    fn device(&self) -> Option<P::Device>;
    /// sel: encodeSignalEvent:value:
    /// Signals `value` when the buffer completes.
    fn encode_signal(&self, event: Option<EventRef<'_, P>>, value: u64) -> RaiseResult<()>;
    /// sel: encodeSignalEventScheduled:value: SPI
    /// Undeclared anywhere; runtime encoding `v32@0:8@16Q24`, so `value` is `u64`
    /// (`metal/tests/spi_encodings.rs`). Signals `value` when the buffer is scheduled.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&P::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()>;
    /// sel: encodeWaitForEvent:value:
    fn encode_wait(&self, event: Option<EventRef<'_, P>>, value: u64) -> RaiseResult<()>;
    /// sel: error — `None` for nil.
    fn error(&self) -> Option<NsErrorText>;
    /// sel: renderCommandEncoderWithDescriptor:
    /// Core throws on nil. aborts: Metal's validation on some descriptors (a pass with no render
    /// target, measured).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &P::RenderPassDescriptor,
    ) -> RaiseResult<Option<P::RenderEncoder>>;
    /// sel: setProtectionOptions: SPI — (the type-5 envelope),
    /// Declared on `MTLCommandBufferSPI` (`uint64_t`).
    /// Reached both under a handler and with none: `on` says which this call is.
    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()>;
    /// sel: status — Raw `MTLCommandBufferStatus`.
    fn status(&self) -> CommandBufferStatus;
    /// sel: waitUntilCompleted
    /// Blocks.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()>;
    /// sel: waitUntilScheduled — Blocks.
    fn wait_until_scheduled(&self) -> RaiseResult<()>;
    /// sel: computeCommandEncoderWithDescriptor: — from a compute pass descriptor made from
    /// `desc`.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<P::ComputeEncoder>>;
    /// The platform's own, no Metal call: hands the GPU the work encoded so far, with no encoder
    /// open, as a part of the command buffer committed on its queue now, and answers the command
    /// buffer of the same queue that goes on from it, which the caller encodes the rest into and
    /// commits as the last part. The parts run in their order on the queue, as the encoders of
    /// one command buffer do; each part completes on its own, and the caller makes of them the
    /// one command buffer's outcome: its handlers go on the last part, every part's failure is
    /// the command buffer's. `None` where the platform keeps the command buffer whole, which it
    /// then is, unchanged: the default, as Metal's commit starts the GPU's work at once; and where
    /// a part would answer otherwise than the whole, as the platform's docs list. Reached both under a handler and with none: `on`
    /// says which this call is.
    fn split(&self, on: OnFault) -> RaiseResult<Option<P::CommandBuffer>> {
        let _ = on;
        Ok(None)
    }
    /// The platform's own, no Metal call: a command buffer of the same queue, made now and not
    /// committed, for the caller to encode a later range of this one's work into, on this thread
    /// or another while this one's is encoded, and to commit after this one is committed as a
    /// part ([`Self::commit_part`]), as a part itself or as the last. `None` where this one may not be cut
    /// now ([`Self::split`]'s refusals). Its make waits as `split`'s does; the caller leaves
    /// room.
    fn part(&self, on: OnFault) -> RaiseResult<Option<P::CommandBuffer>>;
    /// The platform's own, no Metal call, at the end of the range encoded into this one before
    /// the one [`Self::part`] made, no encoder open: commits this one as a part, as
    /// [`Self::split`] does, where it may be cut ([`PartCut::Cut`]); else commits nothing, and
    /// answers [`PartCut::Fails`] where it fails from one of its encoders on, every encoder
    /// after it running nothing, as Metal's GPU fails a command buffer for what its encoders did
    /// (the caller then folds the later one into it, [`Self::fold`], and commits it as the
    /// whole), and [`PartCut::Whole`] with its reason otherwise ([`KeptWhole`]).
    fn commit_part(&self, on: OnFault) -> RaiseResult<PartCut>;
    /// The platform's own, no Metal call: this command buffer, which fails from one of its
    /// encoders on ([`PartCut::Fails`]), takes in what `later`, a command buffer [`Self::part`]
    /// made for a later range and encoded, noted of how Metal's GPU fails it, as though its
    /// encoders had been this one's, after this one's own; `later` is let go of uncommitted. Its
    /// encoders run nothing in either: they follow the one this fails from.
    fn fold(&self, later: P::CommandBuffer);
    /// The platform's own, no Metal call, before the commit: `true` marks the buffer's completion
    /// as joining the next one's of its queue, `false` takes the mark off. A marked buffer's work
    /// may go to the GPU in one submission with the buffers of its queue committed right behind
    /// it, the first one not marked closing it: its GPU work, its order on the queue and its GPU
    /// times are unchanged, but it is seen complete, its handlers run and the events it signals
    /// reach their values only once the last buffer of that submission completed. So a caller
    /// marks only a buffer whose completion nothing observes before a later one's: a part of a
    /// command buffer but its last that signals and waits for no event (`reims-vgpu`'s
    /// `decode/cmdbuf.rs`). Whether the platform took the mark: `false`, the default, where it
    /// sends each buffer on its own, as Metal sees each complete at its own end.
    fn set_completion_joins_next(&self, joins: bool) -> bool {
        let _ = joins;
        false
    }
    /// The platform's own, no Metal call, before the commit: the command buffer ends with
    /// Metal's `MTLCommandBufferErrorOutOfMemory` (8) in `MTLCommandBufferErrorDomain`, its status
    /// an error, whatever its work does, unless it ends with an error of its own: a command of it
    /// named a buffer made deferred whose memory the platform refused at that use
    /// ([`Buffer::make_memory`](super::resource::Buffer::make_memory)), and was dropped. Nothing
    /// by default, for a platform that defers no buffer's memory.
    fn fail_out_of_memory(&self) {}
    /// The platform's own, no Metal call: asked before the commit, the buffer's completion is
    /// exported as a sync file once it is committed ([`Self::export_completion_sync_file`]).
    /// `false` where the platform exports none, the default: the caller waits for the buffer on
    /// the host.
    fn add_completion_sync_file(&self) -> bool {
        false
    }
    /// The platform's own, no Metal call: after the commit, the buffer's completion as a sync
    /// file, asked for before it ([`Self::add_completion_sync_file`]), which a device of another
    /// driver waits for on its GPU ([`Self::encode_wait_for_sync_file`]): `Some(Some(file))`;
    /// `Some(None)` where the buffer has finished, so there is nothing to wait for (its
    /// [`Self::status`] tells how it ended); `None` where the platform refuses it, the default:
    /// the caller waits for the buffer on the host.
    fn export_completion_sync_file(&self) -> Option<Option<SyncFile>> {
        None
    }
    /// The platform's own, no Metal call: before any encoder, the buffer's work waits on its GPU
    /// for the sync file it is given, another device's completion
    /// ([`Self::export_completion_sync_file`]); the buffer's device takes the file. `false`, the
    /// file closed, where the platform cannot, the default: the caller waits for that completion
    /// on the host.
    fn encode_wait_for_sync_file(&self, _file: SyncFile) -> bool {
        false
    }
    /// The platform's own, no Metal call: before any encoder, the caller's word that no
    /// compute encoder of the buffer, nor of the parts cut from it ([`Self::split`],
    /// [`Self::part`]), is sent a command predication record (`encodeStartIf:...`,
    /// `encodeStartElse`, `encodeStartWhile:...`, `encodeStartDoWhile`, or an End of one), so the
    /// platform keeps nothing for one. A record sent after it dies. Nothing by default: the
    /// platform then keeps what a record would need, as before.
    fn holds_no_predication(&self) {}
}
