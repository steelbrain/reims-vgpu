//! The Vulkan platform's device of the GPU a test of a video arm tests, on a host of one GPU or
//! several ([`device_for`]). The product runs on a device the first arm that takes it (`super`'s
//! order): NVDEC where the device is an NVIDIA GPU, else VA-API on the device's render node. So a
//! test of NVDEC or NVENC takes a device of an NVIDIA GPU, and a test of VA-API a device of a GPU
//! that is not NVIDIA's and has a render node. That is the process's default device where its GPU
//! is the arm's, as the guest's video runs on that device; else the device of the first other GPU
//! that is, by render node, made as the product makes a device of the compositor's GPU: by the
//! registry ID `reims_vgpu_vulkan::device_choice::compositor_gpu` answers for the node's numbers,
//! through `Platform::open_device`. A host with no such GPU fails the test, naming each GPU seen.
//! The test says on stderr which GPU it took.
//!
//! Where a device selection variable lists one GPU alone (`MESA_VK_DEVICE_SELECT` with its `!`),
//! that GPU is the default device and the only one seen. Where none is set, the default device is
//! the library's pick, which need not be the arm's: on a host of a Radeon 780M and an RTX 3090
//! the pick was the 3090, and a VA-API encode test that took the default device opened NVIDIA's
//! VA driver, which encodes nothing, and failed (**measured**, 2026-11-20, m3-max-58078); on a
//! host of an Iris Xe and an RTX 3090 the pick was the Iris Xe, where an NVDEC test finds no
//! NVDEC.

use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::platform::Platform as _;
use reims_vgpu_vulkan::device_choice::{CompositorGpu, compositor_gpu};
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::host_gpus::render_node_of;

/// NVIDIA's PCI vendor ID, by which the NVIDIA arm takes a GPU (`nvidia`'s `NVIDIA_VENDOR`).
const NVIDIA_VENDOR: u32 = 0x10de;

/// The arm a test tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arm {
    /// NVDEC and NVENC, on an NVIDIA GPU.
    Nvidia,
    /// VA-API, on the render node of a GPU that is not NVIDIA's.
    VaApi,
}

impl Arm {
    /// Whether the product runs this arm on `device` (the module doc).
    fn runs_on(self, device: &VulkanDevice) -> bool {
        let nvidia = super::nvidia_gpu(device).vendor_id == NVIDIA_VENDOR;
        match self {
            Arm::Nvidia => nvidia,
            Arm::VaApi => !nvidia && render_node_of(&device.raw).is_some(),
        }
    }
}

/// The device of `platform`'s on the GPU a test of `arm` tests (the module doc), said on stderr;
/// panics naming each GPU seen where no GPU of the host's is the arm's.
pub(crate) fn device_for(platform: &VulkanPlatform, arm: Arm) -> VulkanDevice {
    let default = VulkanPlatform::system_default_device().expect("a Vulkan device");
    let mut seen = vec![gpu_name(&default)];
    let found = if arm.runs_on(&default) {
        Some(default)
    } else {
        let own = VulkanPlatform::default_device_registry_id();
        (render_nodes().into_iter())
            .filter_map(|numbers| match compositor_gpu(numbers, own) {
                CompositorGpu::Other(id) => Some(id),
                CompositorGpu::Own => None,
            })
            .filter_map(|id| platform.open_device(id))
            .find(|device| {
                seen.push(gpu_name(device));
                arm.runs_on(device)
            })
    };
    let Some(device) = found else {
        panic!(
            "no GPU of the host's runs {arm:?}: seen {}",
            seen.join(", ")
        );
    };
    eprintln!("tested on {} for {arm:?}", gpu_name(&device));
    device
}

/// `device`'s GPU, by its name and its render node, as a test's lines name it.
pub(crate) fn gpu_name(device: &VulkanDevice) -> String {
    let name = &device.raw.properties().name;
    match render_node_of(&device.raw) {
        Some(node) => format!("{name} ({})", node.display()),
        None => format!("{name} (no render node)"),
    }
}

/// The DRM numbers of each render node sysfs lists (`/sys/class/drm/renderD128/dev` reads
/// `226:128`), in their order.
fn render_nodes() -> Vec<(u32, u32)> {
    let listed = std::fs::read_dir("/sys/class/drm").into_iter().flatten();
    let mut nodes: Vec<(u32, u32)> = (listed.flatten())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
        .filter_map(|entry| {
            let numbers = std::fs::read_to_string(entry.path().join("dev")).ok()?;
            let (major, minor) = numbers.trim().split_once(':')?;
            Some((major.parse().ok()?, minor.parse().ok()?))
        })
        .collect();
    nodes.sort_unstable();
    nodes
}
