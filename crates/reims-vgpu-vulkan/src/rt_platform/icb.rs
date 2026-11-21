//! The rt platform's indirect command buffers ([`RtIcb`]): rt's, of indexed draw commands that
//! draw with the executing encoder's pipeline state and bind their own buffers, which a kernel or
//! a vertex function encodes on the GPU through the buffer's `gpuResourceID`.
//!
//! - **Its make** (`newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`, [`make`]) is
//!   rt's, of the descriptor's command types, `inheritPipelineState`, `inheritBuffers` and vertex,
//!   fragment and kernel buffer bind counts, each count past 32 bits handed to rt as `u32::MAX`,
//!   which rt refuses as past its own bound. The descriptor's other properties have no field of
//!   rt's: the commands draw with the executing encoder's render states, whatever the `inherit*`
//!   properties say, as rt documents an M4's commands do. The options are not read: rt keeps the
//!   records in Private memory, which no CPU encode reaches. Each of rt's refusals answers nil,
//!   counted under its kind (`rt-refused-gap` for `Gap::IndirectCommandBuffers`, its kind named in
//!   the boot diagnostics, `IndirectCommandBuffers-<kind>`, and in the first line of each kind):
//!   `Extension` on a device without `VK_EXT_device_generated_commands` (an Intel Iris Xe, the UHD
//!   630 on Windows, a Mac host's Vulkan), `CommandTypes` for types other than indexed draws alone,
//!   `CommandPipelines` for commands that set their own pipeline state, `InheritedBuffers`,
//!   `BindCount` past 31 vertex or fragment buffers, and `Capacity` for no commands or more than
//!   the device runs in one execution. A descriptor of a resource index (`setResourceIndex:`) is
//!   unbuilt ([`Unbuilt::Icb`]): nil, counted.
//! - **Its answers.** `gpuResourceID` is rt's, the device address of the records' header: the
//!   guest's argument encoder writes it into an argument buffer's `command_buffer` member, and a
//!   shader's encodes reach the records through it. `size` is rt's, the command count asked;
//!   `resourceIndex` is 0 (rt gives the buffer no index); `uniqueIdentifier` is one of the pipeline
//!   states' process-wide count from 1 (`super::pipeline`).
//! - **Its commands** are the GPU's to encode. A render encoder's executions are rt's, of a range
//!   the CPU gives and of one a buffer holds (`super::render`); a blit's reset and optimize are
//!   rt's, and so is a synchronize of the buffer, which records nothing (`super::blit`); a
//!   `useResource:` of it is rt's, the command buffer keeping its records until it retires
//!   (`super::compute`'s `declared`). Every buffer of rt's holds draws alone, so a compute
//!   encoder's execution of one is Metal's refusal, counted (`rt-refused-metal`).
//! - **The commands' slots.** An execution of a pipeline state carries each buffer slot its
//!   vertex and fragment functions read below the buffer's bind counts as a slot word of rt's
//!   push range, which the two functions share, 256 bytes on every device: a command of more
//!   slots than it holds is rt's refusal (`IndirectCommandBuffers-DrawInput`, `rt-refused-gap`).
//!   It holds 25 of a state whose vertex function reads all but one of them and whose fragment
//!   function reads one, of which a vertex function of 28 slots keeps its first 24 (measured on
//!   an RTX 3090 and a Radeon RX 7900 XTX). Each slot word also takes a token of the device's,
//!   and three more (the sequence, the index buffer and the draw), at most its
//!   `maxIndirectCommandsTokenCount`: 16 on lavapipe, 32 on NVIDIA's driver and 128 on RADV, so
//!   the push range binds first but on lavapipe, which refuses a command of more than 13 slots
//!   (`IndirectCommandBuffers-Tokens`). Metal has no query of either, and the guest is told
//!   nothing of them: a buffer's bind counts are Metal's, at most 31 each. Either refusal is at
//!   the execution: it draws nothing, and the encoder's later commands draw. The first line of
//!   each kind names the buffer's bind counts and the device's tokens ([`ExecutionDetail`]): which
//!   slots the state's functions read is rt's to know. A state made without
//!   `supportIndirectCommandBuffers` is refused there too
//!   (`IndirectCommandBuffers-UnflaggedState`), and so is one that reads what else a command
//!   cannot carry (`IndirectCommandBuffers-DrawInput`).
//! - **Unbuilt** ([`Unbuilt::Icb`]), each counted: commands the CPU encodes
//!   (`indirectRenderCommandAtIndex:`, `indirectComputeCommandAtIndex:`), which answer nil, so
//!   each of the guest's records of a CPU encode is refused; a CPU `resetWithRange:`, which resets
//!   nothing; and a blit's copy between buffers (`copyIndirectCommandBuffer:...`), which encodes
//!   nothing.

use std::ops::Range;

use metal2vulkan_runtime as rt;

use crate::core::platform::Platform;
use crate::core::platform::icb::IndirectCommandBuffer;
use crate::core::platform::types::NsRange;
use crate::core::platform::values::IcbDesc;
use crate::core::raise::RaiseResult;

use super::blit::Stop;
use super::pipeline::next_identifier;
use super::unbuilt::{Unbuilt, refuse, refuse_create, rt_refused_create};
use super::{RtDevice, RtObj, RtPlatform};

/// The creator's selector.
const MAKE: &str = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";

/// `id<MTLIndirectCommandBuffer>` on metal2vulkan-runtime: rt's buffer.
pub struct RtIcb {
    rt: rt::IndirectCommandBuffer,
    unique_identifier: u64,
    /// What a refused execution's first line names of it ([`ExecutionDetail`]).
    detail: ExecutionDetail,
}

impl RtIcb {
    /// rt's buffer.
    pub fn rt(&self) -> &rt::IndirectCommandBuffer {
        &self.rt
    }

    /// What an execution of it asks of rt, for the process's first line of each kind of rt's
    /// refusals of one.
    pub(super) fn detail(&self) -> &ExecutionDetail {
        &self.detail
    }
}

/// What an execution asked of rt, for the process's first line of each kind of rt's refusals of
/// it (`super::unbuilt::refusal_detail`): its buffer's vertex and fragment buffer bind counts as
/// rt took them, the most slot words one of its commands takes, and the device's
/// `maxIndirectCommandsTokenCount`. Which slots below the bind counts the state's functions read,
/// the slot words a command takes, and which of them its push range holds are rt's to know, which
/// no call of rt's answers here, so the line says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExecutionDetail {
    /// The buffer's vertex and fragment buffer bind counts.
    pub(super) binds: (u32, u32),
    /// The device's `maxIndirectCommandsTokenCount`, rt's floor's.
    pub(super) tokens: Option<u32>,
}

impl std::fmt::Display for ExecutionDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (vertex, fragment) = self.binds;
        write!(
            f,
            "an execution of a buffer of {vertex} vertex and {fragment} fragment buffer slots a \
             command, at most {} slot words a command (which of them its state's functions read \
             is not known here), ",
            vertex.saturating_add(fragment)
        )?;
        match self.tokens {
            Some(tokens) => write!(
                f,
                "on a device of {tokens} indirect command tokens, {} slot words a command",
                tokens.saturating_sub(3)
            ),
            None => f.write_str("on a device of no indirect command tokens"),
        }
    }
}

/// A count of the guest's as rt takes it: one past 32 bits as `u32::MAX`, which rt refuses as
/// past its own bound.
fn count(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The commands of `range` as rt's: one past 32 bits is past every buffer's commands, Metal's
/// refusal.
pub(super) fn commands(range: NsRange) -> Result<Range<u32>, Stop> {
    let end = range.location.checked_add(range.length);
    let end = end.and_then(|end| u32::try_from(end).ok());
    match (u32::try_from(range.location), end) {
        (Ok(start), Some(end)) => Ok(start..end),
        _ => Err(Stop::Metal("a range of commands past 32 bits")),
    }
}

/// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` of `desc` and `max_count` on
/// `device` (the module's docs); nil where rt refuses it, or of a resource index, counted.
pub(super) fn make(device: &RtDevice, desc: &IcbDesc, max_count: u64) -> Option<RtObj<RtIcb>> {
    if desc.resource_index.is_some() {
        refuse_create(Unbuilt::Icb, MAKE);
        return None;
    }
    let mut d = rt::IndirectCommandBufferDescriptor::new();
    d.command_types = rt::IndirectCommandType::from_bits(desc.command_types);
    d.inherit_pipeline_state = desc.inherit_pipeline_state;
    d.inherit_buffers = desc.inherit_buffers;
    d.max_vertex_buffer_bind_count = count(desc.max_vertex_buffer_bind_count);
    d.max_fragment_buffer_bind_count = count(desc.max_fragment_buffer_bind_count);
    d.max_kernel_buffer_bind_count = count(desc.max_kernel_buffer_bind_count);
    let detail = ExecutionDetail {
        binds: (
            d.max_vertex_buffer_bind_count,
            d.max_fragment_buffer_bind_count,
        ),
        tokens: device.facts().indirect_tokens,
    };
    let made = device
        .rt()
        .new_indirect_command_buffer(&d, count(max_count));
    match made {
        Ok(made) => Some(RtObj::new(RtIcb {
            rt: made,
            unique_identifier: next_identifier(),
            detail,
        })),
        Err(error) => {
            rt_refused_create(MAKE, &error);
            None
        }
    }
}

impl IndirectCommandBuffer<RtPlatform> for RtObj<RtIcb> {
    /// rt's: the records' header's device address.
    fn gpu_resource_id(&self) -> u64 {
        crate::recorder::plain_resource_id(self, || self.rt.gpu_resource_id())
    }

    /// Unbuilt: nil ([`Unbuilt::Icb`]).
    fn compute_command(
        &self,
        _command_index: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::IndirectComputeCommand<'_>>> {
        refuse(Unbuilt::Icb, "indirectComputeCommandAtIndex:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::Icb`]).
    fn render_command(
        &self,
        _command_index: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::IndirectRenderCommand<'_>>> {
        refuse(Unbuilt::Icb, "indirectRenderCommandAtIndex:");
        Ok(None)
    }

    /// Unbuilt: resets nothing ([`Unbuilt::Icb`]).
    fn reset_with_range(&self, _location: u64, _length: u64) -> RaiseResult<()> {
        refuse(Unbuilt::Icb, "resetWithRange:");
        Ok(())
    }

    /// 0: rt gives the buffer no index.
    fn resource_index(&self) -> u64 {
        0
    }

    /// rt's: the command count asked.
    fn size(&self) -> u64 {
        u64::from(self.rt.size())
    }

    /// One of the pipeline states' process-wide count from 1.
    fn unique_identifier(&self) -> u64 {
        self.unique_identifier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range of commands is rt's as it is; one whose start or end is past 32 bits is Metal's
    /// refusal.
    #[test]
    fn a_range_of_commands_past_32_bits_is_metal_s_refusal() {
        assert!(matches!(commands(NsRange::new(1, 2)), Ok(range) if range == (1..3)));
        let past = [
            NsRange::new(u64::from(u32::MAX), 1),
            NsRange::new(1 << 32, 0),
            NsRange::new(u64::MAX, 2),
        ];
        for range in past {
            assert!(matches!(commands(range), Err(Stop::Metal(_))), "{range:?}");
        }
    }

    /// A refused execution's detail names its buffer's bind counts, the most slot words a command
    /// takes, and the device's tokens and the slot words they leave a command, or that the
    /// device has none.
    #[test]
    fn a_refused_execution_s_detail_names_the_bind_counts_and_the_device_s_tokens() {
        let detail = ExecutionDetail {
            binds: (29, 1),
            tokens: Some(32),
        };
        assert_eq!(
            detail.to_string(),
            "an execution of a buffer of 29 vertex and 1 fragment buffer slots a command, at most \
             30 slot words a command (which of them its state's functions read is not known \
             here), on a device of 32 indirect command tokens, 29 slot words a command"
        );
        let detail = ExecutionDetail {
            binds: (2, 0),
            tokens: None,
        };
        assert_eq!(
            detail.to_string(),
            "an execution of a buffer of 2 vertex and 0 fragment buffer slots a command, at most 2 \
             slot words a command (which of them its state's functions read is not known here), on \
             a device of no indirect command tokens"
        );
    }
}
