//! Which of the host's video arms the reims-graphics device's video runs on ([`graphics_video`]
//! on the Vulkan platform, `rt_graphics_video` on the rt platform): the first that takes the
//! platform device, in this order. Each that declines says why in one line at info, and costs no
//! more than asking: a GPU that is not NVIDIA's loads no CUDA.
//!
//! 1. NVDEC, where the device is an NVIDIA GPU that CUDA opens (Linux: `super::nvidia`).
//! 2. VA-API, on the device's render node, whatever its vendor (Linux: `super::vaapi`), or on
//!    Windows on its DXGI adapter, over Mesa's VA-on-D3D12 driver (`super::vaapi`). The rt
//!    platform has no VA-API arm yet.
//!
//! On the rt platform NVDEC decodes alone: NVENC reads a buffer the GPU copied a frame into,
//! which rt's platform does not release to CUDA yet (`super::nvidia::open`'s `encodes`).

use std::sync::Arc;

use reims_vgpu_video::core::runtime::Runtime;
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
    if let Some(video) =
        super::nvidia::open::<VulkanPlatform>(device, nvidia_gpu(device), &runtime, true)
    {
        return Some(video);
    }
    #[cfg(any(target_os = "linux", windows))]
    if let Some(video) = super::vaapi::open_on(device, Arc::clone(&runtime)) {
        return Some(video);
    }
    let _ = (device, runtime);
    None
}

/// What the NVIDIA arm reads of the Vulkan platform's `device`'s GPU.
#[cfg(target_os = "linux")]
pub(crate) fn nvidia_gpu(device: &VulkanDevice) -> super::nvidia::NvidiaGpu {
    let properties = device.raw.properties();
    super::nvidia::NvidiaGpu {
        registry_id: device.registry_id,
        vendor_id: properties.vendor_id,
        pci: properties.pci_bus,
    }
}

/// The rt platform's device.
#[cfg(feature = "rt")]
type RtDevice = reims_vgpu_vulkan::rt_platform::RtObj<reims_vgpu_vulkan::rt_platform::RtDevice>;

/// The hardware video of the rt platform's `device`, from the first arm that takes it (module
/// docs), the host APIs' threads and waits `runtime`'s; `None` where none does.
#[cfg(feature = "rt")]
pub(crate) fn rt_graphics_video(
    device: &RtDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<reims_vgpu_vulkan::RtPlatform>>> {
    #[cfg(target_os = "linux")]
    if let Some(video) = super::nvidia::open::<reims_vgpu_vulkan::RtPlatform>(
        device,
        rt_nvidia_gpu(device),
        &runtime,
        false,
    ) {
        return Some(video);
    }
    let _ = (device, runtime);
    None
}

/// What the NVIDIA arm reads of the rt platform's `device`'s GPU.
#[cfg(all(feature = "rt", target_os = "linux"))]
pub(crate) fn rt_nvidia_gpu(device: &RtDevice) -> super::nvidia::NvidiaGpu {
    let facts = device.facts();
    super::nvidia::NvidiaGpu {
        registry_id: facts.registry_id,
        vendor_id: facts.vendor_id,
        pci: device.host_gpu().pci,
    }
}
