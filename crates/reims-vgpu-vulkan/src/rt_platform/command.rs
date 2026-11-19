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
//!   `MTLCommandBufferErrorDomain` codes: a lost device `DeviceRemoved` (11), and a command
//!   buffer rt ended without running (its device quiesced, its queue abandoned) `Internal` (1).
//! - **Encoders**: blit encoders ([`super::blit`]). Compute and render encoders are unbuilt
//!   ([`Unbuilt::Encoder`]): nil.
//! - **A signal as it is scheduled** (`encodeSignalEventScheduled:value:`) sets the shared
//!   event's value in the scheduled handler, as Metal signals it once the command buffer is
//!   scheduled, in its order among the scheduled handlers.
//! - **Parts**: none. `split` keeps the trait's default and `part` answers nil, so each guest
//!   command buffer is one rt command buffer, committed at the guest's commit; `commit_part`
//!   answers the command buffer whole ([`PartCut::Whole`]), and `fold` is unreachable, since
//!   nothing is cut. Neither is counted.
//! - **Protection options** other than 0 are unbuilt ([`Unbuilt::Protection`]): rt protects no
//!   work, and the command buffer runs as one without them.
//!
//! Every call rt refuses answers as Metal's answers where it fails (nothing encoded, nil, no
//! wait), counted under the refusal's kind ([`rt_refused`]); never a death.

use metal2vulkan_rt as rt;

use crate::core::platform::command::{
    CommandBuffer, CompletedHandler, CompletedInfo, KeptWhole, PartCut, ScheduledHandler,
    run_completed,
};
use crate::core::platform::event::EventRef;
use crate::core::platform::types::{CommandBufferStatus, NsErrorText};
use crate::core::platform::values::ComputePassDesc;
use crate::core::platform::{Handle, Platform};
use crate::core::raise::refusals;
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::rt::BlockEnv;
use crate::core::sync::Mutex;

use super::blit::RtBlitEncoder;
use super::device::RtDevice;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse, refuse_create, refused_as, rt_refused, rt_refused_create,
};
use super::{RtObj, RtPlatform};

/// `MTLCommandBufferErrorDomain`, the domain of a failed command buffer's error.
const ERROR_DOMAIN: &str = "MTLCommandBufferErrorDomain";

/// `MTLCommandBufferErrorInternal`.
const ERROR_INTERNAL: i64 = 1;

/// `MTLCommandBufferErrorDeviceRemoved`.
const ERROR_DEVICE_REMOVED: i64 = 11;

/// A handler as rt runs it.
type Handler = Box<dyn FnOnce(&rt::CommandBuffer) + Send>;

/// `id<MTLCommandBuffer>` on metal2vulkan-rt.
pub struct RtCommandBuffer {
    /// rt's command buffer.
    rt: rt::CommandBuffer,
    /// The device, which `device` answers.
    device: RtObj<RtDevice>,
    /// The handlers until the commit (the module's docs).
    held: Mutex<Held>,
}

/// A command buffer's handlers until its commit hands them to rt, each kind in the order added.
#[derive(Default)]
struct Held {
    /// Whether the command buffer was committed: a handler added since is refused.
    committed: bool,
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
    /// The command buffer over rt's `made`, of `device`.
    pub fn new(made: rt::CommandBuffer, device: RtObj<RtDevice>) -> RtCommandBuffer {
        RtCommandBuffer {
            rt: made,
            device,
            held: Mutex::new(Held::default()),
        }
    }

    /// rt's command buffer.
    pub fn rt(&self) -> &rt::CommandBuffer {
        &self.rt
    }

    /// Adds `handler` to the list `list` picks, or refuses it, counted, once the command buffer
    /// was committed.
    fn hold(&self, sel: &'static str, list: fn(&mut Held) -> &mut Vec<Handler>, handler: Handler) {
        let mut held = self.held.lock();
        if held.committed {
            drop(held);
            rt_refused(sel, &rt::Error::Refused(rt::Refusal::Committed));
            return;
        }
        list(&mut held).push(handler);
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
/// command buffer answers then ([`completed_info`]). Added inside a refusal scope (the graphics
/// lane's), the read and `block` run in one on rt's thread too ([`run_completed`]).
fn completed(block: CompletedHandler) -> Handler {
    #[cfg(test)]
    let slot = crate::core::testkit::capture::current_slot();
    let carried = refusals::carry();
    Box::new(move |made: &rt::CommandBuffer| {
        // A callout on rt's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        let run = || run_completed(carried, || completed_info(made), block);
        #[cfg(test)]
        crate::core::testkit::capture::with_slot(slot, run);
        #[cfg(not(test))]
        run();
    })
}

/// What a completed handler learns of `made` (the module's docs).
fn completed_info(made: &rt::CommandBuffer) -> CompletedInfo {
    let (start, end) = (made.gpu_start_time(), made.gpu_end_time());
    CompletedInfo {
        status: status(made.status()),
        error: made.error().map(ns_error),
        gpu_times_ns: (start != 0 && end >= start).then_some((start, end)),
    }
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

/// rt's error as Metal's `NSError` (the module's docs).
fn ns_error(error: rt::CommandBufferError) -> NsErrorText {
    let code = match error {
        rt::CommandBufferError::DeviceLost { .. } => ERROR_DEVICE_REMOVED,
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
        let handler = completed(block);
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
    fn add_completion_seen_handler(
        &self,
        on: OnFault,
        block: CompletedHandler,
    ) -> RaiseResult<()> {
        let _ = on;
        let handler = completed(block);
        self.hold("addCompletionSeenHandler", |h| &mut h.seen, handler);
        Ok(())
    }

    /// rt's blit encoder; nil where rt refuses it (another encoder open, the command buffer
    /// committed), counted.
    fn blit_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::BlitEncoder>> {
        let _ = on;
        Ok(match self.rt.blit_command_encoder() {
            Ok(encoder) => Some(RtObj::new(RtBlitEncoder::new(encoder))),
            Err(error) => {
                rt_refused_create("blitCommandEncoder", &error);
                None
            }
        })
    }

    /// Hands rt the held handlers (the module's docs), then commits. A second commit is Metal's
    /// refusal, counted; a commit rt refuses is counted, and its handlers never run.
    fn commit(&self, on: OnFault) -> RaiseResult<()> {
        let _ = on;
        let sel = "commit";
        let held = {
            let mut held = self.held.lock();
            if held.committed {
                drop(held);
                rt_refused(sel, &rt::Error::Refused(rt::Refusal::Committed));
                return Ok(());
            }
            let taken = std::mem::take(&mut *held);
            held.committed = true;
            taken
        };
        let Held {
            mut submission,
            scheduled,
            mut seen,
            completed,
            ..
        } = held;
        submission.extend(scheduled);
        seen.extend(completed);
        let run = |handlers: Vec<Handler>| {
            move |made: &rt::CommandBuffer| handlers.into_iter().for_each(|handler| handler(made))
        };
        let committed = || -> Result<(), rt::Error> {
            if !submission.is_empty() {
                self.rt.add_scheduled_handler(run(submission))?;
            }
            if !seen.is_empty() {
                self.rt.add_completed_handler(run(seen))?;
            }
            self.rt.commit()
        };
        if let Err(error) = committed() {
            rt_refused(sel, &error);
        }
        Ok(())
    }

    /// Unbuilt: nil ([`Unbuilt::Encoder`]).
    fn compute_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
        let _ = on;
        refuse_create(Unbuilt::Encoder, "computeCommandEncoder");
        Ok(None)
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
        let sel = "encodeSignalEvent:value:";
        if let Some(event) = rt_event(sel, event)
            && let Err(error) = self.rt.encode_signal_event(event, value)
        {
            rt_refused(sel, &error);
        }
        Ok(())
    }

    /// The event's value set in the scheduled handler (the module's docs); nothing, counted, for
    /// nil.
    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&<RtPlatform as Platform>::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeSignalEventScheduled:value:";
        let Some(event) = signal_event_scheduled else {
            refused_as(sel, RtRefusal::Refused, &"a nil event");
            return Ok(());
        };
        let event = rt::SharedEvent::clone(event);
        let signal = move |_made: &rt::CommandBuffer| event.set_signaled_value(value);
        self.hold(sel, |h| &mut h.scheduled, Box::new(signal));
        Ok(())
    }

    /// rt's wait; nothing, counted, for a nil event or one rt refuses.
    fn encode_wait(
        &self,
        event: Option<EventRef<'_, RtPlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let sel = "encodeWaitForEvent:value:";
        if let Some(event) = rt_event(sel, event)
            && let Err(error) = self.rt.encode_wait_for_event(event, value)
        {
            rt_refused(sel, &error);
        }
        Ok(())
    }

    /// rt's error as Metal's (the module's docs).
    fn error(&self) -> Option<NsErrorText> {
        self.rt.error().map(ns_error)
    }

    /// Unbuilt: nil ([`Unbuilt::Encoder`]), which core faults on.
    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &<RtPlatform as Platform>::RenderPassDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::RenderEncoder>> {
        let _ = (on, render_pass_descriptor);
        refuse_create(Unbuilt::Encoder, "renderCommandEncoderWithDescriptor:");
        Ok(None)
    }

    /// Nothing for 0; other options are unbuilt ([`Unbuilt::Protection`]), counted, and change
    /// nothing.
    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()> {
        let _ = on;
        if options != 0 {
            refuse(Unbuilt::Protection, "setProtectionOptions:");
        }
        Ok(())
    }

    fn status(&self) -> CommandBufferStatus {
        status(self.rt.status())
    }

    /// rt's wait; none, counted, where rt refuses it (in a scheduled handler, or on rt's
    /// completion thread for a command buffer not yet completed, which it would never see).
    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()> {
        let _ = on;
        if let Err(error) = self.rt.wait_until_completed() {
            rt_refused("waitUntilCompleted", &error);
        }
        Ok(())
    }

    /// rt's wait; none, counted, where rt refuses it, as for
    /// [`wait_until_completed`](CommandBuffer::wait_until_completed).
    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        if let Err(error) = self.rt.wait_until_scheduled() {
            rt_refused("waitUntilScheduled", &error);
        }
        Ok(())
    }

    /// Unbuilt: nil ([`Unbuilt::Encoder`]).
    fn compute_encoder_for_pass(
        &self,
        desc: &ComputePassDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::ComputeEncoder>> {
        let _ = desc;
        refuse_create(Unbuilt::Encoder, "computeCommandEncoderWithDescriptor:");
        Ok(None)
    }

    /// Nil: each guest command buffer is one rt command buffer (the module's docs).
    fn part(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
        let _ = on;
        Ok(None)
    }

    /// Commits nothing, and answers the command buffer whole: rt keeps a command buffer's
    /// progress to its completion ([`KeptWhole::Progress`]), or [`KeptWhole::Committed`] once
    /// committed. No caller reaches it: [`part`](CommandBuffer::part) answers nil.
    fn commit_part(&self, on: OnFault) -> RaiseResult<PartCut> {
        let _ = on;
        let committed = self.held.lock().committed;
        let reason = if committed {
            KeptWhole::Committed
        } else {
            KeptWhole::Progress
        };
        Ok(PartCut::Whole(reason))
    }

    /// Unreachable: [`part`](CommandBuffer::part) answers nil, so nothing is cut and no caller
    /// folds. Lets `later` go uncommitted.
    fn fold(&self, later: <RtPlatform as Platform>::CommandBuffer) {
        drop(later);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;
    use crate::core::platform::command::CommandQueue as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::event::SharedEvent as _;
    use crate::rt_platform::RtQueue;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// rt's errors are Metal's, in Metal's command buffer domain: a lost device `DeviceRemoved`
    /// (11), a command buffer ended unrun `Internal` (1); rt's statuses are Metal's values.
    #[test]
    fn rt_s_errors_and_statuses_are_metal_s() {
        let lost = ns_error(rt::CommandBufferError::DeviceLost { in_flight: true });
        assert_eq!((lost.domain.as_str(), lost.code), (ERROR_DOMAIN, 11));
        let unrun = ns_error(rt::CommandBufferError::NotRun);
        assert_eq!((unrun.domain.as_str(), unrun.code), (ERROR_DOMAIN, 1));
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

    /// Command buffers on the GPU the test holds: handlers in core's order whatever the order
    /// added, each completed handler told the status and GPU times; a handler added after the
    /// commit, and a second commit, refused and counted; kept whole; signals and waits on an
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
        assert!(matches!(first.part(on), Ok(None)));
        let whole = |why| -> RaiseResult<PartCut> { Ok(PartCut::Whole(why)) };
        assert_eq!(first.commit_part(on), whole(KeptWhole::Progress));
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
        assert_eq!(waits.encode_wait(Some(EventRef::Shared(&shared)), 7), Ok(()));
        assert_eq!(waits.commit(on), Ok(()));
        let signals = made(&queue);
        assert_eq!(signals.encode_signal(Some(EventRef::Event(&event)), 1), Ok(()));
        assert_eq!(signals.encode_signal(Some(EventRef::Shared(&shared)), 5), Ok(()));
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
        assert_eq!(scheduled.encode_signal_when_scheduled(Some(&at_scheduled), 9), Ok(()));
        let saw = Arc::new(AtomicU64::new(0));
        let (seen, read) = (Arc::clone(&saw), at_scheduled.clone_ref());
        let look = move || seen.store(read.signaled_value(), Ordering::Release);
        assert_eq!(scheduled.add_scheduled_handler(on, Box::new(look)), Ok(()));
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

        // Unbuilt encoders and protection options; the command buffer still runs.
        let encoders = counted(Unbuilt::Encoder.code());
        let protection = counted(Unbuilt::Protection.code());
        assert!(matches!(nil.compute_encoder(on), Ok(None)));
        let pass = platform.new_render_pass_descriptor();
        assert!(matches!(nil.render_encoder(on, &pass), Ok(None)));
        let desc = ComputePassDesc::default();
        assert!(matches!(nil.compute_encoder_for_pass(&desc), Ok(None)));
        assert_eq!(counted(Unbuilt::Encoder.code()), encoders + 3);
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
}
