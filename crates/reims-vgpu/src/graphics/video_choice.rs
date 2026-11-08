//! Which of the host's video arms the reims-graphics device's video runs on, on the Vulkan
//! platform ([`graphics_video`]): the first that takes the runtime device, in this order. Each
//! that declines says why in one line at info, and costs no more than asking: a GPU that is not
//! NVIDIA's loads no CUDA.
//!
//! 1. NVDEC, where the device is an NVIDIA GPU that CUDA opens (Linux: `super::nvidia`).
//! 2. VA-API, on the device's render node, whatever its vendor (Linux: `super::vaapi`).
//! 3. Vulkan Video, on the device's video-decode queue (`super::vulkan::video`).

use std::sync::Arc;

use reims_vgpu_video_core::runtime::Runtime;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;

use super::decode::GraphicsVideo;

/// The hardware video of `device`, from the first arm that takes it (module docs), the host
/// APIs' threads and waits `runtime`'s; `None` where none does.
pub(crate) fn graphics_video(
    device: &VulkanDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<VulkanPlatform>>> {
    #[cfg(target_os = "linux")]
    if let Some(video) = super::nvidia::open(device, &runtime) {
        return Some(video);
    }
    #[cfg(target_os = "linux")]
    if let Some(video) = super::vaapi::open_on(device, Arc::clone(&runtime)) {
        return Some(video);
    }
    let _ = runtime;
    super::vulkan::graphics_video(device)
}
