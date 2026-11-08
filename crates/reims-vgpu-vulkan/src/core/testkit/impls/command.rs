//! Fake: the `platform/command.rs` traits over the fake models (architecture 13.2).
//!
//! The queue's and command buffer's ordered calls log their selector and arguments
//! (`fake::calllog`) and consult the script (`fake::script`) before the model; `status`, `error`
//! and `device` are free and read the model.

use super::super::calllog::Arg;
use super::super::objects::{CommandBufferModel, Fk, Model, completion_read, kind};
use super::super::platform::FakePlatform;
use super::fatal;
use crate::core::platform::command::*;
use crate::core::platform::types::{CommandBufferStatus, NsErrorText};
use crate::core::platform::{EventRef, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::testkit::script::Policy;

impl CommandQueue<FakePlatform> for Fk<kind::CommandQueue> {
    fn command_buffer(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::CommandBuffer>> {
        self.make(Policy::On(on), "commandBuffer", vec![], |w| {
            Some(w.new_obj::<kind::CommandBuffer>(Model::CommandBuffer(
                CommandBufferModel::default(),
            )))
        })
    }

    fn set_label(&self, label: &str) {
        fatal(self.act(Policy::Fatal, "setLabel:", vec![Arg::Str(label.to_owned())]));
    }
}

impl CommandBuffer<FakePlatform> for Fk<kind::CommandBuffer> {
    /// As the Vulkan platform does: the read of the buffer and `block` run in the refusal scope the
    /// handler was added in, where it was added in one (`run_completed`), so a read that dies
    /// ([`World::complete_dying_read`]) hands `block` the buffer refused there.
    ///
    /// [`World::complete_dying_read`]: super::super::objects::World::complete_dying_read
    fn add_completed_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "addCompletedHandler:",
            vec![Arg::Block(true)],
        )?;
        let carried = crate::core::raise::refusals::carry();
        let block: CompletedHandler = Box::new(move |info| {
            run_completed(carried, || completion_read(info), block);
        });
        if let Model::CommandBuffer(m) = &mut self.state().model {
            m.handlers.push(block);
        }
        Ok(())
    }

    /// Run at commit, on the committing thread, before the buffer can complete
    /// ([`commit_obj`](super::super::objects::commit_obj)).
    fn add_scheduled_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "addScheduledHandler:",
            vec![Arg::Block(true)],
        )?;
        if let Model::CommandBuffer(m) = &mut self.state().model {
            m.scheduled.push(block);
        }
        Ok(())
    }

    /// Run at commit, on the committing thread, before its scheduled handlers, as the fake takes
    /// a buffer's work as it is committed ([`commit_obj`](super::super::objects::commit_obj)).
    fn add_submission_handler(&self, on: OnFault, block: ScheduledHandler) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "addSubmissionHandler",
            vec![Arg::Block(true)],
        )?;
        if let Model::CommandBuffer(m) = &mut self.state().model {
            m.submission.push(block);
        }
        Ok(())
    }

    /// Run where the buffer completes, before its completed handlers, the read and `block` in
    /// the refusal scope it was added in, as a completed handler's are
    /// ([`complete_obj`](super::super::objects::complete_obj)).
    fn add_completion_seen_handler(&self, on: OnFault, block: CompletedHandler) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "addCompletionSeenHandler",
            vec![Arg::Block(true)],
        )?;
        let carried = crate::core::raise::refusals::carry();
        let block: CompletedHandler = Box::new(move |info| {
            run_completed(carried, || completion_read(info), block);
        });
        if let Model::CommandBuffer(m) = &mut self.state().model {
            m.seen.push(block);
        }
        Ok(())
    }

    fn blit_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::BlitEncoder>> {
        self.make(Policy::On(on), "blitCommandEncoder", vec![], |w| {
            Some(w.new_obj::<kind::BlitEncoder>(Model::None))
        })
    }

    /// Its submission handlers, and those of the buffers the world takes at this commit
    /// ([`World::release_submissions_at_next_commit`]), run inside the call, marked so
    /// ([`crate::core::platform::command::committing`]), as the Vulkan platform's runtime runs them.
    ///
    /// [`World::release_submissions_at_next_commit`]: crate::core::testkit::objects::World::release_submissions_at_next_commit
    fn commit(&self, on: OnFault) -> RaiseResult<()> {
        self.act(Policy::On(on), "commit", vec![])?;
        crate::core::platform::command::committing(|| {
            super::super::objects::commit_obj(self.obj())
        });
        Ok(())
    }

    fn compute_encoder(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::ComputeEncoder>> {
        self.make(Policy::On(on), "computeCommandEncoder", vec![], |w| {
            Some(w.new_obj::<kind::ComputeEncoder>(Model::None))
        })
    }

    fn device(&self) -> Option<<FakePlatform as Platform>::Device> {
        self.device_handle()
    }

    /// `None`, logging nothing, until the world splits command buffers
    /// ([`World::split_command_buffers`]); then logged, the buffer committed as at `commit`, and a
    /// new command buffer answered.
    ///
    /// [`World::split_command_buffers`]: super::super::objects::World::split_command_buffers
    fn split(&self, on: OnFault) -> RaiseResult<Option<<FakePlatform as Platform>::CommandBuffer>> {
        if !self.world().splits() {
            return Ok(None);
        }
        let next = self.make(Policy::On(on), "split", vec![], |w| {
            Some(w.new_obj::<kind::CommandBuffer>(Model::CommandBuffer(
                CommandBufferModel::default(),
            )))
        })?;
        if next.is_some() {
            super::super::objects::commit_obj(self.obj());
        }
        Ok(next)
    }

    fn encode_signal(
        &self,
        event: Option<EventRef<'_, FakePlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        let ev = event.map(|e| match e {
            EventRef::Event(x) => x.erase(),
            EventRef::Shared(x) => x.erase(),
        });
        self.act(
            Policy::Caught,
            "encodeSignalEvent:value:",
            vec![Arg::Obj(event.map(|e| e.id())), Arg::U64(value)],
        )?;
        if let (Some(ev), Model::CommandBuffer(m)) = (ev, &mut self.state().model) {
            m.signal_on_complete.push((ev, value));
        }
        Ok(())
    }

    fn encode_signal_when_scheduled(
        &self,
        signal_event_scheduled: Option<&<FakePlatform as Platform>::SharedEvent>,
        value: u64,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "encodeSignalEventScheduled:value:",
            vec![Arg::opt(signal_event_scheduled), Arg::U64(value)],
        )?;
        if let (Some(ev), Model::CommandBuffer(m)) = (
            signal_event_scheduled.map(Fk::erase),
            &mut self.state().model,
        ) {
            m.signal_on_schedule.push((ev, value));
        }
        Ok(())
    }

    fn encode_wait(
        &self,
        event: Option<EventRef<'_, FakePlatform>>,
        value: u64,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "encodeWaitForEvent:value:",
            vec![Arg::Obj(event.map(|e| e.id())), Arg::U64(value)],
        )
    }

    fn error(&self) -> Option<NsErrorText> {
        match &self.state().model {
            Model::CommandBuffer(m) => m.error.clone(),
            _ => None,
        }
    }

    fn render_encoder(
        &self,
        on: OnFault,
        render_pass_descriptor: &<FakePlatform as Platform>::RenderPassDescriptor,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::RenderEncoder>> {
        self.make(
            Policy::On(on),
            "renderCommandEncoderWithDescriptor:",
            vec![Arg::obj(render_pass_descriptor)],
            |w| Some(w.new_obj::<kind::RenderEncoder>(Model::None)),
        )
    }

    fn set_protection_options(&self, on: OnFault, options: u64) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "setProtectionOptions:",
            vec![Arg::U64(options)],
        )
    }

    fn status(&self) -> CommandBufferStatus {
        match &self.state().model {
            Model::CommandBuffer(m) => m.status,
            _ => CommandBufferStatus::default(),
        }
    }

    fn wait_until_completed(&self, on: OnFault) -> RaiseResult<()> {
        self.act(Policy::On(on), "waitUntilCompleted", vec![])?;
        super::super::objects::wait_status(self.obj(), CommandBufferStatus::COMPLETED);
        Ok(())
    }

    fn wait_until_scheduled(&self) -> RaiseResult<()> {
        self.act(Policy::Caught, "waitUntilScheduled", vec![])?;
        super::super::objects::wait_status(self.obj(), CommandBufferStatus::SCHEDULED);
        Ok(())
    }

    /// A new encoder, with the value in the log.
    fn compute_encoder_for_pass(
        &self,
        desc: &crate::core::platform::values::ComputePassDesc,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::ComputeEncoder>> {
        self.make(
            Policy::Caught,
            "computeCommandEncoderWithDescriptor:",
            vec![Arg::val(desc)],
            |w| Some(w.new_obj::<kind::ComputeEncoder>(Model::None)),
        )
    }
}
