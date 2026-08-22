//! Compute segment sequencing for control-flow and indirect-command records.
//!
//! Vulkan execution for these record families is not implemented. The segment
//! latches the first typed refusal so later dispatches cannot escape a failed
//! sequencing region.

use crate::runtime::Device;

use crate::runtime::compute_exec::{ComputeAccum, ComputeStatus};
use crate::runtime::decode::compute::{Command as ComputeCommand, Kind};
use crate::runtime::host::{HostMemory, HostOps};

/// Latched reason that blocks later dispatches in the same compute segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequencingBlock {
    ControlFlow,
    IndirectCommandBuffer,
}

pub struct ComputeSession {
    pub control_depth: i32,
}

impl ComputeSession {
    pub fn open(dispatch_type: u32) -> Result<Self, ComputeStatus> {
        {
            let _ = dispatch_type;
            Err(ComputeStatus::Unsupported("compute_session_unimplemented"))
        }
    }

    pub fn encode_control<M: HostMemory + HostOps>(
        &mut self,
        state: &Device,
        host: &M,
        task_id: u32,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        {
            let _ = (state, host, task_id, cmd);
            ComputeStatus::Unsupported("compute_control_flow_unimplemented")
        }
    }

    pub fn encode_icb<M: HostMemory + HostOps>(
        &mut self,
        state: &mut Device,
        host: &mut M,
        task_id: u32,
        cmd: &ComputeCommand,
        acc: &ComputeAccum,
    ) -> ComputeStatus {
        if cmd.indirect_command_buffer_ref == 0 {
            return ComputeStatus::MissingBuffer("compute_icb_ref_zero");
        }
        {
            let _ = (state, host, task_id, cmd, acc);
            ComputeStatus::Unsupported("compute_icb_execute_unimplemented")
        }
    }

    /// Finish a sequencing session. No GPU session exists until these record
    /// families gain Vulkan execution.
    pub fn finish<M: HostMemory + HostOps>(
        self,
        host: &mut M,
        state: &mut Device,
        task_id: u32,
    ) -> ComputeStatus {
        {
            let _ = (host, state, task_id);
            ComputeStatus::Ok
        }
    }
}

/// The mutable state of one `SEGMENT_TYPE_COMPUTE` segment.
///
/// These three share a single lifetime: they come into existence when the
/// segment opens, every record in the segment reads and mutates them together,
/// and the session commits when the segment ends. Passing them as one value
/// keeps that lifetime visible at each call site.
#[derive(Default)]
pub struct ComputeSegment {
    /// Pipeline / bind state accumulated across the segment's records.
    pub acc: ComputeAccum,
    /// Multi-record encoder, opened on demand by the first control-flow or ICB
    /// record and committed at segment end.
    pub session: Option<ComputeSession>,
    /// Latched sequencing failure; once set it refuses later dispatches.
    pub block: Option<SequencingBlock>,
}

pub fn ensure_session(
    session: &mut Option<ComputeSession>,
    dispatch_type: u32,
) -> Result<&mut ComputeSession, ComputeStatus> {
    if session.is_none() {
        *session = Some(ComputeSession::open(dispatch_type)?);
    }
    Ok(session.as_mut().unwrap())
}

pub fn apply_sequencing<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut ComputeSegment,
) -> ComputeStatus {
    if seg.block.is_some() {
        return ComputeStatus::Unsupported("sequencing_block_active");
    }
    match cmd.kind {
        Kind::ControlStartDoWhile
        | Kind::ControlEndDoWhile
        | Kind::ControlStartWhile
        | Kind::ControlEndWhile
        | Kind::ControlStartIf
        | Kind::ControlStartElse
        | Kind::ControlEndIf => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::ControlFlow);
                    return e;
                }
            };
            let st = sess.encode_control(state, host, task_id, cmd);
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::ControlFlow);
            }
            st
        }
        Kind::ExecuteCommandsInBuffer | Kind::ExecuteCommandsInBufferIndirect => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::IndirectCommandBuffer);
                    return e;
                }
            };
            let st = sess.encode_icb(state, host, task_id, cmd, &seg.acc);
            // Latch only on failure so successful materialize+execute does not
            // block later dispatches in the segment.
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::IndirectCommandBuffer);
            }
            st
        }
        _ => ComputeStatus::Unsupported("sequencing_unknown_kind"),
    }
}

/// Finish an open session at compute-segment end (no-op if none).
pub fn finish_session<M: HostMemory + HostOps>(
    session: &mut Option<ComputeSession>,
    state: &mut Device,
    host: &mut M,
    task_id: u32,
) -> Option<ComputeStatus> {
    session.take().map(|s| s.finish(host, state, task_id))
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};

    use crate::runtime::host::FakeHost;

    #[test]
    fn icb_latches_sequencing_block() {
        let mut host = FakeHost::new();
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut seg = ComputeSegment::default();
        let cmd = ComputeCommand {
            kind: Kind::ExecuteCommandsInBuffer,
            indirect_command_buffer_ref: 1,
            ..ComputeCommand::default()
        };
        let st = apply_sequencing(&mut state, &mut host, 1, &cmd, &mut seg);
        // Missing list entry → MissingBuffer; latches sequencing block.
        assert!(
            matches!(
                st,
                ComputeStatus::MissingBuffer(_) | ComputeStatus::Unsupported(_)
            ),
            "unexpected {st:?}"
        );
        assert_eq!(seg.block, Some(SequencingBlock::IndirectCommandBuffer));
        if let Some(s) = seg.session.take() {
            let _ = s.finish(&mut host, &mut state, 1);
        }
    }
}
