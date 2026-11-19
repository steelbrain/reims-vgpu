//! The rt platform's command queue ([`RtQueue`]): rt's queue, whose `commandBuffer` makes rt's
//! command buffers ([`super::command`]).

use metal2vulkan_rt as rt;

use crate::core::platform::command::CommandQueue;
use crate::core::platform::{Handle, Platform};
use crate::core::raise::{OnFault, RaiseResult};

use super::command::RtCommandBuffer;
use super::device::RtDevice;
use super::unbuilt::rt_refused_create;
use super::{RtObj, RtPlatform};

/// `id<MTLCommandQueue>` on metal2vulkan-rt.
pub struct RtQueue {
    /// rt's queue.
    rt: rt::CommandQueue,
    /// The device, which its command buffers answer.
    device: RtObj<RtDevice>,
}

impl RtQueue {
    /// The queue over rt's `queue` of `device`.
    pub fn new(queue: rt::CommandQueue, device: RtObj<RtDevice>) -> RtQueue {
        RtQueue { rt: queue, device }
    }

    /// rt's queue.
    pub fn rt(&self) -> &rt::CommandQueue {
        &self.rt
    }
}

impl CommandQueue<RtPlatform> for RtObj<RtQueue> {
    /// rt's command buffer, which waits as Metal's does while the queue's most command buffers
    /// are made and not yet completed; nil where rt refuses it, counted under the refusal's kind
    /// (a wait for room on rt's own handler thread, a quiesced device).
    fn command_buffer(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
        let _ = on;
        Ok(match self.rt.command_buffer() {
            Ok(made) => Some(RtObj::new(RtCommandBuffer::new(
                made,
                self.rt.clone(),
                self.device.clone_ref(),
            ))),
            Err(error) => {
                rt_refused_create("commandBuffer", &error);
                None
            }
        })
    }

    /// Nothing: a label is a diagnostic, and nothing reads it back.
    fn set_label(&self, label: &str) {
        let _ = label;
    }
}
