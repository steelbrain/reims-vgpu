//! Platform trait: `CommandQueue`, `CommandBuffer`: queue and command-buffer selectors.

use super::event::EventRef;
use super::types::{CommandBufferStatus, NsErrorText};
use super::values::ComputePassDesc;
use super::{Handle, Platform};
use crate::raise::{OnFault, RaiseResult};

/// What a completed handler learns of its buffer: core's handlers read `status` and `error` of the
/// buffer they were added to, on Metal's completion thread. The platform reads both when the
/// handler runs; neither getter has an effect.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompletedInfo {
    /// `-[MTLCommandBuffer status]` when the handler runs.
    pub status: CommandBufferStatus,
    /// `-[MTLCommandBuffer error]` when the handler runs; `None` for nil.
    pub error: Option<NsErrorText>,
}

/// The domain of the error a command buffer completes with where the platform's read of it, as
/// its completed handler ran, died in the refusal scope the handler's registration carried
/// ([`crate::raise::refusals::carry`]): the platform failed it there, where outside every scope
/// the death ends the process. Not Metal's: only the graphics lane's handlers see it.
pub const REFUSED_DOMAIN: &str = "reims-vgpu.refused";

/// `MTLCommandBufferErrorInternal`, the code of a refused command buffer's error.
const ERROR_INTERNAL: i64 = 1;

impl CompletedInfo {
    /// What a completed handler learns of a command buffer whose read the platform refused
    /// ([`REFUSED_DOMAIN`]): status `Error`, and an internal error naming the refusal.
    pub fn refused(refusal: &crate::raise::refusals::Refusal) -> CompletedInfo {
        CompletedInfo {
            status: CommandBufferStatus::ERROR,
            error: Some(NsErrorText {
                domain: REFUSED_DOMAIN.to_owned(),
                code: ERROR_INTERNAL,
                description: refusal.to_string(),
            }),
        }
    }

    /// Whether the platform refused the command buffer's read ([`CompletedInfo::refused`]).
    pub fn is_refused(&self) -> bool {
        self.error
            .as_ref()
            .is_some_and(|e| e.domain == REFUSED_DOMAIN)
    }
}

/// Runs completed handler `block` with what `read` finds of its command buffer, in the scope
/// `carried` names, the one the handler's registration carried ([`crate::raise::refusals::carry`]):
/// a death in `read` hands `block` [`CompletedInfo::refused`] instead, and a death in `block` ends
/// only `block`, each logged. Outside every scope `read` and `block` run plainly, one after the
/// other, as they always did.
pub fn run_completed(
    carried: crate::raise::refusals::Carried,
    read: impl FnOnce() -> CompletedInfo,
    block: CompletedHandler,
) {
    let info = carried.run(read).unwrap_or_else(|refusal| {
        log::warn!(
            target: "reims_vgpu::refused",
            "a command buffer's completion was refused, and the command buffer failed: {refusal}"
        );
        CompletedInfo::refused(&refusal)
    });
    if let Err(refusal) = carried.run(|| block(info)) {
        log::warn!(
            target: "reims_vgpu::refused",
            "a command buffer's completed handler was refused: {refusal}"
        );
    }
}

/// A completed handler (`addCompletedHandler:`'s block). Runs once, on a Metal thread (4.2 item 6).
/// Where it is added inside a refusal scope, the platform runs its read of the buffer and the
/// handler in that scope ([`crate::raise::refusals::Carried`]): a death in the read hands the
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
    /// Core throws on nil. aborts: Metal's validation on some descriptors (render_pass_no_target,
    /// testing.md 7.4).
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
}
