//! The command queue and command buffer: core's `CommandQueue` and `CommandBuffer`, one typed
//! call each, run through the message layer.

use std::ptr::NonNull;

use block2::RcBlock;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CompletedHandler, CompletedInfo, ScheduledHandler};
use reims_vgpu_core::platform::event::EventRef;
use reims_vgpu_core::platform::types::{CommandBufferStatus, NsErrorText};
use reims_vgpu_core::platform::values::ComputePassDesc;
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::gpu::event;
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj, error_text};
use crate::spi::methods::CommandBufferSpi;

/// `id<MTLCommandQueue>`.
pub type MetalCommandQueue = Obj<ProtocolObject<dyn MTLCommandQueue>>;

impl reims_vgpu_core::platform::command::CommandQueue<MetalPlatform> for MetalCommandQueue {
    /// `commandBuffer`: a `+0` buffer, retained. An optimized build elides Metal's own
    /// autorelease of it, so an uncommitted buffer goes at the handle's drop; in a dev build that
    /// autorelease stays in the thread's pool until the pool pops.
    fn command_buffer(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::CommandBuffer>> {
        msg::send("commandBuffer", Policy::On(on), || self.commandBuffer())
            .map(|b| b.map(Obj::from_retained))
    }
    /// `setLabel:` with a new `NSString` of `label`, released after the call.
    fn set_label(&self, label: &str) {
        let text = msg::new_string(label);
        msg::get("setLabel:", || self.setLabel(Some(&text)));
    }
}

/// What a completed handler learns of `cb`: its `status`, `error` and GPU times, read when the
/// handler runs.
fn completed_info(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> CompletedInfo {
    let status = msg::get("status", || cb.status());
    let error = msg::get("error", || cb.error());
    let start = msg::get("GPUStartTime", || cb.GPUStartTime());
    let end = msg::get("GPUEndTime", || cb.GPUEndTime());
    CompletedInfo {
        status: CommandBufferStatus(status.0.to_u64()),
        error: error.as_deref().map(error_text),
        gpu_times_ns: CompletedInfo::gpu_times(start, end),
    }
}

/// `id<MTLCommandBuffer>`.
pub type MetalCommandBuffer = Obj<ProtocolObject<dyn MTLCommandBuffer>>;

impl reims_vgpu_core::platform::command::CommandBuffer<MetalPlatform> for MetalCommandBuffer {
    /// The block reads the buffer's `status` and `error` when Metal calls it, on Metal's completion
    /// thread, and runs `block` once with them; `block` is dropped there, when it returns. Added
    /// inside a refusal scope (the graphics lane's), the read and `block` run in one there too
    /// (`run_completed`): a death in the read fails the command buffer to `block`, and the process
    /// goes on.
    fn add_completed_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        let carried = reims_vgpu_core::raise::refusals::carry();
        let once = Mutex::new(Some(block));
        let handler = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // A foreign callout: marked ours, so a panic here aborts before it unwinds into Metal.
            let _env = reims_vgpu_core::rt::BlockEnv::foreign();
            // Metal calls a completed handler once; a second call would find nothing to run.
            let Some(block) = once.lock().take() else {
                return;
            };
            // SAFETY: Metal passes the live command buffer the handler was added to.
            let cb = unsafe { cb.as_ref() };
            reims_vgpu_core::platform::command::run_completed(
                carried,
                || completed_info(cb),
                block,
            );
        });
        let handler = RcBlock::as_ptr(&handler);
        // SAFETY: a live block; Metal copies it.
        msg::send("addCompletedHandler:", Policy::On(on), || unsafe {
            self.addCompletedHandler(handler)
        })
    }
    /// The block runs `block` once when Metal calls it, on Metal's thread, as the buffer is
    /// scheduled; `block` is dropped there, when it returns.
    fn add_scheduled_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        let once = Mutex::new(Some(block));
        let handler = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            // A foreign callout: marked ours, so a panic here aborts before it unwinds into Metal.
            let _env = reims_vgpu_core::rt::BlockEnv::foreign();
            // Metal calls a scheduled handler once; a second call would find nothing to run.
            if let Some(block) = once.lock().take() {
                block();
            }
        });
        let handler = RcBlock::as_ptr(&handler);
        // SAFETY: a live block; Metal copies it.
        msg::send("addScheduledHandler:", Policy::On(on), || unsafe {
            self.addScheduledHandler(handler)
        })
    }
    /// Metal tells of no submission before scheduling: a scheduled handler.
    fn add_submission_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        self.add_scheduled_handler(on, block)
    }
    /// Metal tells of a completion only on its completed handlers' thread: a completed handler.
    fn add_completion_seen_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        self.add_completed_handler(on, block)
    }
    /// `blitCommandEncoder`: a `+0` encoder, retained.
    fn blit_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::BlitEncoder>> {
        msg::send("blitCommandEncoder", Policy::On(on), || {
            self.blitCommandEncoder()
        })
        .map(|e| e.map(Obj::from_retained))
    }
    fn commit(&self, on: OnFault) -> RaiseResult<()> {
        msg::send("commit", Policy::On(on), || self.get().commit())
    }
    /// `computeCommandEncoder`: a `+0` encoder, retained.
    fn compute_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::ComputeEncoder>> {
        msg::send("computeCommandEncoder", Policy::On(on), || {
            self.computeCommandEncoder()
        })
        .map(|e| e.map(Obj::from_retained))
    }
    /// `device`: a `+0` device, retained.
    fn device(&self) -> Option<<MetalPlatform as Platform>::Device> {
        Some(Obj::from_retained(msg::get("device", || {
            self.get().device()
        })))
    }
    /// A nil event is sent as it is.
    fn encode_signal(
        &self,
        event: Option<EventRef<'_, MetalPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let event = event.map(self::event);
        msg::send("encodeSignalEvent:value:", Policy::Caught, || {
            CommandBufferSpi::encodeSignalEvent_value(self.get(), event, value)
        })
    }
    /// `encodeSignalEventScheduled:value:` (SPI).
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&<MetalPlatform as Platform>::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        let event = signal_event_scheduled.map(|e| e.get());
        msg::send("encodeSignalEventScheduled:value:", Policy::Caught, || {
            self.encodeSignalEventScheduled_value(event, value)
        })
    }
    /// A nil event is sent as it is.
    fn encode_wait(
        &self,
        event: Option<EventRef<'_, MetalPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let event = event.map(self::event);
        msg::send("encodeWaitForEvent:value:", Policy::Caught, || {
            CommandBufferSpi::encodeWaitForEvent_value(self.get(), event, value)
        })
    }
    fn error(&self) -> Option<NsErrorText> {
        msg::get("error", || self.get().error())
            .as_deref()
            .map(error_text)
    }
    /// `renderCommandEncoderWithDescriptor:`: a `+0` encoder, retained, marked when its pass is
    /// over a map that is not immutable ([`crate::gpu::rate_map`]).
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &<MetalPlatform as Platform>::RenderPassDescriptor,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::RenderEncoder>> {
        let sel = "renderCommandEncoderWithDescriptor:";
        let encoder = msg::send(sel, Policy::On(on), || {
            self.renderCommandEncoderWithDescriptor(render_pass_descriptor)
        })?
        .map(Obj::from_retained);
        if let Some(e) = &encoder {
            crate::gpu::rate_map::mark_pass(e, render_pass_descriptor);
        }
        Ok(encoder)
    }
    /// `setProtectionOptions:` (SPI).
    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()> {
        msg::send("setProtectionOptions:", Policy::On(on), || {
            self.setProtectionOptions(options)
        })
    }
    /// The raw `MTLCommandBufferStatus`.
    fn status(&self) -> CommandBufferStatus {
        CommandBufferStatus(msg::get("status", || self.get().status()).0.to_u64())
    }
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()> {
        msg::send("waitUntilCompleted", Policy::On(on), || {
            self.waitUntilCompleted()
        })
    }
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        msg::send("waitUntilScheduled", Policy::Caught, || {
            self.waitUntilScheduled()
        })
    }
    /// `[MTLComputePassDescriptor new]`, made to hold the value, then
    /// `computeCommandEncoderWithDescriptor:`.
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::ComputeEncoder>> {
        let d = crate::platform::class_new(objc2_metal::MTLComputePassDescriptor::new);
        crate::gpu::desc::apply_compute_pass(desc, &d);
        compute_encoder_with_descriptor(self, &d)
    }
}

/// `computeCommandEncoderWithDescriptor:`: a `+0` encoder, retained.
fn compute_encoder_with_descriptor(
    buffer: &MetalCommandBuffer,
    compute_pass_descriptor: &crate::gpu::desc::MetalComputePassDescriptor,
) -> RaiseResult<Option<<MetalPlatform as Platform>::ComputeEncoder>> {
    let sel = "computeCommandEncoderWithDescriptor:";
    msg::send(sel, Policy::Caught, || {
        buffer.computeCommandEncoderWithDescriptor(compute_pass_descriptor)
    })
    .map(|e| e.map(Obj::from_retained))
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use objc2_metal::{MTLCommandEncoder as _, MTLSharedEvent as _};
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::sync::Mutex;

    use super::*;
    use crate::gpu::resource::testkit::*;

    /// commandBuffer makes a fresh buffer (+0, retained by the handle); setLabel: sets the queue's
    /// label
    #[test]
    fn queue_command_buffer_and_label() {
        let k = kit();
        let a = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let b = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        assert_ne!(a.id(), b.id());
        assert_eq!(a.status(), CommandBufferStatus::NOT_ENQUEUED);
        k.queue.set_label("reims-vgpu.exec");
        assert_eq!(
            k.queue.get().label().unwrap().to_string(),
            "reims-vgpu.exec"
        );
    }

    /// blitCommandEncoder, computeCommandEncoder, computeCommandEncoderWithDescriptor:,
    /// renderCommandEncoderWithDescriptor: each give an encoder (ended before the next); device is
    /// the queue's device
    #[test]
    fn encoders_and_device() {
        let k = kit();
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        blit.get().endEncoding();
        let compute = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        compute.get().endEncoding();
        let compute = cb
            .compute_encoder_for_pass(&ComputePassDesc::default())
            .unwrap()
            .unwrap();
        compute.get().endEncoding();
        let render = cb
            .render_encoder(OnFault::Return, &k.render_pass)
            .unwrap()
            .unwrap();
        render.get().endEncoding();
        assert_eq!(cb.device().unwrap().id(), k.device.id());
    }

    /// A compute pass value gives an encoder, with or without its dispatch type
    #[test]
    fn compute_encoder_from_a_value() {
        let k = kit();
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        for dispatch_type in [None, Some(1)] {
            let compute = cb
                .compute_encoder_for_pass(&ComputePassDesc { dispatch_type })
                .unwrap()
                .unwrap();
            compute.get().endEncoding();
        }
    }

    /// Commit, waitUntilScheduled, waitUntilCompleted, status and error on a buffer that completes;
    /// encodeSignalEvent:value: (an MTLEvent and an MTLSharedEvent), encodeWaitForEvent:value:,
    /// encodeSignalEventScheduled:value: (the undeclared SPI) and setProtectionOptions: are encoded
    /// and take effect
    #[test]
    fn commit_signal_and_wait() {
        let k = kit();
        let event = k.device.new_event(OnFault::Return).unwrap().unwrap();
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.set_protection_options(OnFault::Return, 0).unwrap();
        cb.encode_signal(Some(EventRef::Event(&event)), 1).unwrap();
        cb.encode_wait(Some(EventRef::Event(&event)), 1).unwrap();
        cb.encode_signal_when_scheduled(Some(&k.shared_event), 5)
            .unwrap();
        cb.encode_signal(Some(EventRef::Shared(&k.shared_event)), 7)
            .unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_scheduled().unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(cb.error(), None);
        assert_eq!(k.shared_event.get().signaledValue(), 7);
    }

    /// addCompletedHandler: runs the handler once, on Metal's thread, with the buffer's status and
    /// error read when it runs
    #[test]
    fn completed_handler_runs_once_with_status() {
        let k = kit();
        let cb = k.queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let calls = Arc::new(AtomicU32::new(0));
        let seen = Arc::new(Mutex::new(None));
        let (c, s) = (calls.clone(), seen.clone());
        cb.add_completed_handler(
            OnFault::Return,
            Box::new(move |info| {
                c.fetch_add(1, Ordering::SeqCst);
                *s.lock() = Some(info);
            }),
        )
        .unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // Metal runs completed handlers after waitUntilCompleted returns, on its own thread.
        for _ in 0..500 {
            if calls.load(Ordering::SeqCst) > 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let info = seen.lock().take().unwrap();
        assert_eq!(info.status, CommandBufferStatus::COMPLETED);
        assert_eq!(info.error, None);
    }

    /// completed_info reads status and error of the buffer Metal passes (a buffer never committed:
    /// NotEnqueued, no error)
    #[test]
    fn completed_info_reads_the_buffer() {
        let k = kit();
        let cb = k.queue.command_buffer(OnFault::Die).unwrap().unwrap();
        let info = completed_info(cb.get());
        assert_eq!(info.status, CommandBufferStatus::NOT_ENQUEUED);
        assert_eq!(info.error, None);
    }
}
