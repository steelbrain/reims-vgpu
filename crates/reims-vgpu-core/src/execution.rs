//! Backend-independent immutable execution IR and executor port.
//!
//! A submitted command buffer is an ordered, owned value. Decoders and
//! mutable encoder accumulators do not cross this boundary: draw and compute
//! payloads are already prepared, blits carry resolved resource identities and
//! checked ranges, and resource-state commands carry the exact resource
//! lifetime they may update. [`SubmissionContext`] retains the command stream's
//! complete segment and resource-participation envelope beside those commands.

use crate::{ContentStamp, ResolvedBlit, ResolvedResourceState, SubmissionContext};
use reims_vgpu_protocol::SubmissionIdentity;

/// One fully owned command in a resolved command buffer.
#[derive(Debug)]
pub enum ResolvedCommand<Draw, Compute> {
    Draw(Draw),
    Compute(Compute),
    Blit(Box<ResolvedBlit>),
    ResourceState(ResolvedResourceState),
}

impl<Draw, Compute> ResolvedCommand<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw(_) => ExecutionKind::Draw,
            Self::Compute(_) => ExecutionKind::Compute,
            Self::Blit(_) => ExecutionKind::Blit,
            Self::ResourceState(_) => ExecutionKind::ResourceState,
        }
    }
}

/// Ordered commands from one semantic command-buffer boundary.
#[derive(Debug)]
pub struct ResolvedCommandBuffer<Draw, Compute> {
    commands: Box<[ResolvedCommand<Draw, Compute>]>,
}

impl<Draw, Compute> ResolvedCommandBuffer<Draw, Compute> {
    pub fn new(commands: impl Into<Box<[ResolvedCommand<Draw, Compute>]>>) -> Self {
        Self {
            commands: commands.into(),
        }
    }

    pub fn single(command: ResolvedCommand<Draw, Compute>) -> Self {
        Self::new(vec![command])
    }

    pub fn commands(&self) -> &[ResolvedCommand<Draw, Compute>] {
        &self.commands
    }

    pub fn into_commands(self) -> Box<[ResolvedCommand<Draw, Compute>]> {
        self.commands
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

/// One immutable submission: protocol context and its ordered command buffer.
#[derive(Debug)]
pub struct ResolvedSubmission<Draw, Compute> {
    pub context: SubmissionContext,
    pub command_buffer: ResolvedCommandBuffer<Draw, Compute>,
}

impl<Draw, Compute> ResolvedSubmission<Draw, Compute> {
    pub fn single(context: SubmissionContext, command: ResolvedCommand<Draw, Compute>) -> Self {
        Self {
            context,
            command_buffer: ResolvedCommandBuffer::single(command),
        }
    }
}

/// Semantic operation class used to validate executor completions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionKind {
    Draw,
    Compute,
    Blit,
    ResourceState,
}

impl ExecutionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Draw => "draw",
            Self::Compute => "compute",
            Self::Blit => "blit",
            Self::ResourceState => "resource_state",
        }
    }
}

/// A successful resolved blit's semantic effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlitCompletion {
    /// The destination version whose bytes were written, when the command has
    /// a destination. A no-op command completes with `None`.
    pub written: Option<ContentStamp>,
}

/// A resource-state command accepted at its ordered point in the submission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceStateCompletion {
    pub update: ResolvedResourceState,
}

/// One command handler's output and any persistent GPU replicas it created.
#[derive(Debug)]
pub struct CommandExecution<Output> {
    pub output: Output,
    pub gpu_materialized: std::sync::Arc<[ContentStamp]>,
}

impl<Output> CommandExecution<Output> {
    pub fn new(
        output: Output,
        gpu_materialized: impl Into<std::sync::Arc<[ContentStamp]>>,
    ) -> Self {
        Self {
            output,
            gpu_materialized: gpu_materialized.into(),
        }
    }

    pub fn without_gpu_materialization(output: Output) -> Self {
        Self::new(output, std::sync::Arc::from([]))
    }
}

/// Operation-specific result carried by a completion fact.
#[derive(Debug)]
pub enum ExecutionOutput<Draw, Compute> {
    Draw(Draw),
    Compute(Compute),
    Blit(BlitCompletion),
    ResourceState(ResourceStateCompletion),
}

impl<Draw, Compute> ExecutionOutput<Draw, Compute> {
    pub const fn kind(&self) -> ExecutionKind {
        match self {
            Self::Draw(_) => ExecutionKind::Draw,
            Self::Compute(_) => ExecutionKind::Compute,
            Self::Blit(_) => ExecutionKind::Blit,
            Self::ResourceState(_) => ExecutionKind::ResourceState,
        }
    }
}

/// Immutable completion returned through the same port as its submission.
#[derive(Debug)]
pub struct ExecutionCompletion<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
    /// Current semantic versions materialized as persistent GPU replicas.
    pub gpu_materialized: std::sync::Arc<[ContentStamp]>,
}

/// Completion shape for an ordered resolved command buffer.
pub type ResolvedExecutionCompletion<Draw, Compute> =
    ExecutionCompletion<Box<[ExecutionOutput<Draw, Compute>]>>;

/// Validated completion identity paired with its operation-specific output.
#[derive(Debug)]
pub struct ExecutionReceipt<Output> {
    pub submission: SubmissionIdentity,
    pub output: Output,
    pub gpu_materialized: std::sync::Arc<[ContentStamp]>,
}

/// The core-owned submission/completion boundary implemented by an executor.
pub trait ExecutionPort: std::fmt::Debug + Send + Sync {
    type Submission;
    type Completion;
    type Error;

    fn execute(&self, submission: Self::Submission) -> Result<Self::Completion, Self::Error>;
}

/// Execute every command in order and build one immutable completion.
///
/// The caller supplies capability handlers, but not ordering, identity, output
/// assembly, or materialization de-duplication. This is the shared execution
/// seam for a Vulkan-only handler and the composition handler which also owns
/// guest-memory blits and core state transitions.
pub fn execute_resolved_submission<Draw, Compute, DrawOutput, ComputeOutput, Error>(
    submission: ResolvedSubmission<Draw, Compute>,
    mut draw: impl FnMut(&SubmissionContext, Draw) -> Result<CommandExecution<DrawOutput>, Error>,
    mut compute: impl FnMut(
        &SubmissionContext,
        Compute,
    ) -> Result<CommandExecution<ComputeOutput>, Error>,
    mut blit: impl FnMut(
        &SubmissionContext,
        ResolvedBlit,
    ) -> Result<CommandExecution<BlitCompletion>, Error>,
    mut resource_state: impl FnMut(
        &SubmissionContext,
        ResolvedResourceState,
    ) -> Result<CommandExecution<ResourceStateCompletion>, Error>,
) -> Result<ResolvedExecutionCompletion<DrawOutput, ComputeOutput>, Error> {
    let identity = submission.context.identity;
    let mut outputs = Vec::with_capacity(submission.command_buffer.commands().len());
    let mut materialized = std::collections::BTreeSet::new();
    for command in submission.command_buffer.into_commands().into_vec() {
        let execution = match command {
            ResolvedCommand::Draw(command) => {
                let execution = draw(&submission.context, command)?;
                CommandExecution::new(
                    ExecutionOutput::Draw(execution.output),
                    execution.gpu_materialized,
                )
            }
            ResolvedCommand::Compute(command) => {
                let execution = compute(&submission.context, command)?;
                CommandExecution::new(
                    ExecutionOutput::Compute(execution.output),
                    execution.gpu_materialized,
                )
            }
            ResolvedCommand::Blit(command) => {
                let execution = blit(&submission.context, *command)?;
                CommandExecution::new(
                    ExecutionOutput::Blit(execution.output),
                    execution.gpu_materialized,
                )
            }
            ResolvedCommand::ResourceState(command) => {
                let execution = resource_state(&submission.context, command)?;
                CommandExecution::new(
                    ExecutionOutput::ResourceState(execution.output),
                    execution.gpu_materialized,
                )
            }
        };
        materialized.extend(execution.gpu_materialized.iter().copied());
        outputs.push(execution.output);
    }
    Ok(ExecutionCompletion {
        submission: identity,
        output: outputs.into_boxed_slice(),
        gpu_materialized: materialized.into_iter().collect::<Vec<_>>().into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{
        ByteLength, ContentVersion, GuestVirtualAddress, ResourceId, SubmissionId, TaskId,
    };

    fn range(index: u32) -> crate::ResolvedBufferRange {
        crate::ResolvedBufferRange {
            content: ContentStamp {
                resource: ResourceId::new(index, 1),
                version: ContentVersion::new(2),
            },
            address: GuestVirtualAddress::new(u64::from(index) << 12),
            length: ByteLength::new(16),
        }
    }

    #[test]
    fn the_owned_envelope_keeps_order_context_and_operation_kinds_together() {
        let context = SubmissionContext::standalone(7);
        let submission: ResolvedSubmission<u32, u32> = ResolvedSubmission {
            context: context.clone(),
            command_buffer: ResolvedCommandBuffer::new(vec![
                ResolvedCommand::Draw(41),
                ResolvedCommand::Blit(Box::new(ResolvedBlit::Copy {
                    source: range(1),
                    destination: range(2),
                })),
                ResolvedCommand::Compute(43),
            ]),
        };

        assert_eq!(submission.context, context);
        assert_eq!(
            submission
                .command_buffer
                .commands()
                .iter()
                .map(ResolvedCommand::kind)
                .collect::<Vec<_>>(),
            vec![
                ExecutionKind::Draw,
                ExecutionKind::Blit,
                ExecutionKind::Compute
            ]
        );
    }

    #[test]
    fn completion_identity_is_separate_from_the_ordered_outputs() {
        let completion = ExecutionCompletion {
            submission: SubmissionIdentity {
                id: SubmissionId::new(9),
                task: TaskId::new(3),
            },
            output: vec![ExecutionOutput::<u32, ()>::Draw(17)].into_boxed_slice(),
            gpu_materialized: std::sync::Arc::from([]),
        };

        assert_eq!(completion.output[0].kind(), ExecutionKind::Draw);
        assert_eq!(completion.submission.id, SubmissionId::new(9));
    }

    #[test]
    fn resource_state_is_a_command_not_mutable_submission_metadata() {
        let update = ResolvedResourceState {
            resource: None,
            mappings: vec![reims_vgpu_protocol::SurfaceId::new(5)].into_boxed_slice(),
            ops: reims_vgpu_protocol::ResourceValidityOps::PAGE_ON,
        };
        let buffer: ResolvedCommandBuffer<(), ()> =
            ResolvedCommandBuffer::single(ResolvedCommand::ResourceState(update));
        assert_eq!(buffer.commands()[0].kind(), ExecutionKind::ResourceState);
    }

    #[test]
    fn one_dispatcher_owns_order_outputs_and_materialization_deduplication() {
        use std::cell::RefCell;

        let context = SubmissionContext::standalone(7);
        let first = range(1).content;
        let second = range(2).content;
        let submission = ResolvedSubmission {
            context: context.clone(),
            command_buffer: ResolvedCommandBuffer::new(vec![
                ResolvedCommand::Draw(1),
                ResolvedCommand::Compute(2),
                ResolvedCommand::Draw(3),
            ]),
        };
        let order = RefCell::new(Vec::new());
        let completion = execute_resolved_submission(
            submission,
            |seen, draw| {
                assert_eq!(seen, &context);
                order.borrow_mut().push(draw);
                Ok::<_, ()>(CommandExecution::new(draw * 10, vec![first]))
            },
            |seen, compute| {
                assert_eq!(seen, &context);
                order.borrow_mut().push(compute);
                Ok::<_, ()>(CommandExecution::new(compute * 10, vec![first, second]))
            },
            |_, _| unreachable!(),
            |_, _| unreachable!(),
        )
        .unwrap();

        assert_eq!(*order.borrow(), [1, 2, 3]);
        assert_eq!(completion.submission, context.identity);
        assert_eq!(completion.gpu_materialized.as_ref(), &[first, second]);
        assert!(matches!(completion.output[0], ExecutionOutput::Draw(10)));
        assert!(matches!(completion.output[1], ExecutionOutput::Compute(20)));
        assert!(matches!(completion.output[2], ExecutionOutput::Draw(30)));
    }
}
