//! The rt platform's command queue ([`RtQueue`]): rt's queue. Its command buffers are unbuilt
//! ([`Unbuilt::CommandBuffer`]), so `commandBuffer` is refused, nil and counted, until they are.

use metal2vulkan_rt as rt;

use crate::core::platform::Platform;
use crate::core::platform::command::CommandQueue;
use crate::core::raise::{OnFault, RaiseResult};

use super::unbuilt::{Unbuilt, refuse_create};
use super::{RtObj, RtPlatform};

/// `id<MTLCommandQueue>` on metal2vulkan-rt.
pub struct RtQueue {
    /// rt's queue.
    rt: rt::CommandQueue,
}

impl RtQueue {
    /// The queue over rt's `queue`.
    pub fn new(queue: rt::CommandQueue) -> RtQueue {
        RtQueue { rt: queue }
    }

    /// rt's queue.
    pub fn rt(&self) -> &rt::CommandQueue {
        &self.rt
    }
}

impl CommandQueue<RtPlatform> for RtObj<RtQueue> {
    /// Unbuilt: nil ([`Unbuilt::CommandBuffer`]).
    fn command_buffer(
        &self,
        on: OnFault,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::CommandBuffer>> {
        let _ = on;
        refuse_create(Unbuilt::CommandBuffer, "commandBuffer");
        Ok(None)
    }

    /// Nothing: a label is a diagnostic, and nothing reads it back.
    fn set_label(&self, label: &str) {
        let _ = label;
    }
}
