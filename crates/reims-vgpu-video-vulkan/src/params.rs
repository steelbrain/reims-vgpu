//! A session parameters object (`VkVideoSessionParametersKHR`), of either
//! codec: made from its codec's create info, destroyed as it goes. What it
//! holds is the codec's (`h264/params.rs`, `h265/params.rs`).

use std::sync::Arc;

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};

/// A session parameters object, destroyed as it goes.
pub(crate) struct Parameters {
    device: Arc<VideoDevice>,
    handle: vk::VideoSessionParametersKHR,
}

impl Parameters {
    /// The object `info` describes, its codec's create info chained.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        info: &vk::VideoSessionParametersCreateInfoKHR<'_>,
    ) -> Result<Self, Error> {
        let mut handle = vk::VideoSessionParametersKHR::null();
        // SAFETY: the session `info` names is the device's; `info`, its
        // chain and the std structures with what they point at live through
        // the call.
        unsafe {
            (device.video().fp().create_video_session_parameters_khr)(
                device.device().handle(),
                info,
                std::ptr::null(),
                &mut handle,
            )
        }
        .named("vkCreateVideoSessionParametersKHR")?;
        Ok(Self {
            device: Arc::clone(device),
            handle,
        })
    }

    /// The object's handle.
    pub(crate) fn handle(&self) -> vk::VideoSessionParametersKHR {
        self.handle
    }
}

impl Drop for Parameters {
    fn drop(&mut self) {
        // SAFETY: the object is the device's, and the work that used it is
        // done: its holder waits for that in the timeline.
        unsafe {
            (self
                .device
                .video()
                .fp()
                .destroy_video_session_parameters_khr)(
                self.device.device().handle(),
                self.handle,
                std::ptr::null(),
            );
        }
    }
}
