//! Which of the host's video arms the reims-graphics device's video runs on
//! ([`rt_graphics_video`]): the first that takes the platform device, in this order. Each that declines says why in one line at info, and costs no
//! more than asking: a GPU that is not NVIDIA's loads no CUDA.
//!
//! 1. NVDEC, where the device is an NVIDIA GPU that CUDA opens (Linux: `super::nvidia`).
//! 2. VA-API, on the device's render node, whatever its vendor (Linux: `super::vaapi`), or on
//!    Windows on its DXGI adapter, over Mesa's VA-on-D3D12 driver (`super::vaapi::open_on_rt`),
//!    by the LUID Vulkan answers for the device, where each picture lands by the CPU's copy.
//!
//! VA-API decodes alone: its encoders read a surface the GPU copied a frame
//! into, which rt's platform does not release to the driver (`super::vaapi::PlatformRules`), and
//! it decodes a reference picture into a surface of its own, not into its output's cache
//! (`super::vaapi::CachedReferences`). NVENC's input is released to CUDA after each frame's copy
//! (`super::nvidia::encode::fill`).

use std::sync::Arc;

use reims_vgpu_video::core::runtime::Runtime;

use super::decode::GraphicsVideo;

#[cfg(test)]
#[cfg(target_os = "linux")]
pub(crate) mod tested_gpu;

/// The rt platform's device.
type RtDevice = reims_vgpu_vulkan::rt_platform::RtObj<reims_vgpu_vulkan::rt_platform::RtDevice>;

/// The hardware video of the rt platform's `device`, from the first arm that takes it (module
/// docs), the host APIs' threads and waits `runtime`'s; `None` where none does.
pub(crate) fn rt_graphics_video(
    device: &RtDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<reims_vgpu_vulkan::RtPlatform>>> {
    #[cfg(target_os = "linux")]
    if let Some(video) = super::nvidia::open::<reims_vgpu_vulkan::RtPlatform>(
        device,
        rt_nvidia_gpu(device),
        &runtime,
    ) {
        return Some(video);
    }
    #[cfg(any(target_os = "linux", windows))]
    if let Some(video) = super::vaapi::open_on_rt(device, Arc::clone(&runtime)) {
        return Some(video);
    }
    let _ = (device, runtime);
    None
}

/// What the NVIDIA arm reads of the rt platform's `device`'s GPU.
#[cfg(target_os = "linux")]
pub(crate) fn rt_nvidia_gpu(device: &RtDevice) -> super::nvidia::NvidiaGpu {
    let facts = device.facts();
    super::nvidia::NvidiaGpu {
        registry_id: facts.registry_id,
        vendor_id: facts.vendor_id,
        pci: device.host_gpu().pci,
    }
}
