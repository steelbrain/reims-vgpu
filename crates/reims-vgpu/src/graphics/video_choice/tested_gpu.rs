//! The device of the GPU a test of a video arm tests, on a host of one GPU or several. The product
//! runs on a device the first arm that takes it (`super`'s order): NVDEC where the device is an
//! NVIDIA GPU, else VA-API on the device's render node. So a test of NVDEC or NVENC runs on a
//! device of an NVIDIA GPU, and a test of VA-API on a device of a GPU that is not NVIDIA's and has
//! a render node ([`Arm`]). The test says on stderr which GPU it runs on.
//!
//! **On the rt platform** ([`check_rt`]) the process makes one device, the library's pick, which
//! only the host's user's selection variable steers: the one GPU, else a discrete GPU first. Where
//! the arm does not run on it, the test fails, naming the device and the variable that picks a GPU
//! of the host's it runs on, with that GPU's IDs (`MESA_VK_DEVICE_SELECT=1002:1900` for the
//! Radeon 780M of a host of it and an RTX 3090, where the pick is the 3090), or saying that no GPU
//! of the host's runs the arm.
//!
//! **On the Vulkan platform** ([`device_for`]) a test takes the process's default device where its
//! GPU is the arm's, as the guest's video runs on that device; else the device of the first other
//! GPU that is, by render node, made as the product makes a device of the compositor's GPU: by the
//! registry ID `reims_vgpu_vulkan::device_choice::compositor_gpu` answers for the node's numbers,
//! through `Platform::open_device`. A host with no such GPU fails the test, naming each GPU seen.
//! Where a device selection variable lists one GPU alone (`MESA_VK_DEVICE_SELECT` with its `!`),
//! that GPU is the default device and the only one seen. Where none is set, the default device is
//! the library's pick, which need not be the arm's: on a host of a Radeon 780M and an RTX 3090
//! the pick was the 3090, and a VA-API encode test that took the default device opened NVIDIA's
//! VA driver, which encodes nothing, and failed (**measured**, 2026-11-20, m3-max-58078).

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
    /// Whether the product runs this arm on the Vulkan platform's `device` (the module doc).
    fn runs_on(self, device: &VulkanDevice) -> bool {
        let vendor_id = super::nvidia_gpu(device).vendor_id;
        self.runs_on_gpu(vendor_id, render_node_of(&device.raw).is_some())
    }

    /// Whether the product runs this arm on a GPU of PCI vendor `vendor_id`, which has a render
    /// node where `render_node` (the module doc).
    fn runs_on_gpu(self, vendor_id: u32, render_node: bool) -> bool {
        let nvidia = vendor_id == NVIDIA_VENDOR;
        match self {
            Arm::Nvidia => nvidia,
            Arm::VaApi => !nvidia && render_node,
        }
    }
}

/// Checks that `arm` runs on the rt platform's `device`, the process's one device, and says so on
/// stderr; panics where it does not, naming the device and the selection variable that picks a
/// GPU of the host's that `arm` runs on, or saying that none does (the module doc).
#[cfg(feature = "rt")]
pub(crate) fn check_rt(device: &super::RtDevice, arm: Arm) {
    let gpu = device.host_gpu();
    let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
    let name = &device.facts().name;
    let name = match &node {
        Some(node) => format!("{name} ({})", node.display()),
        None => format!("{name} (no render node)"),
    };
    if arm.runs_on_gpu(super::rt_nvidia_gpu(device).vendor_id, node.is_some()) {
        eprintln!("tested on {name} for {arm:?}");
        return;
    }
    let other = (render_nodes().into_iter())
        .filter_map(|node| Some((node.ids?, node.path)))
        .find(|&((vendor, _), _)| arm.runs_on_gpu(vendor, true));
    match other {
        Some(((vendor, id), path)) => panic!(
            "the rt platform's one device, {name}, runs no {arm:?}: set \
             MESA_VK_DEVICE_SELECT={vendor:04x}:{id:04x} to pick the GPU of {}",
            path.display()
        ),
        None => panic!(
            "the rt platform's one device, {name}, runs no {arm:?}, nor does any other GPU of \
             the host's"
        ),
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
            .filter_map(|node| match compositor_gpu(node.numbers, own) {
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

/// A render node sysfs lists.
struct RenderNode {
    /// Its major and minor numbers (`/sys/class/drm/renderD128/dev` reads `226:128`).
    numbers: (u32, u32),
    /// Its path in `/dev/dri`.
    path: std::path::PathBuf,
    /// Its GPU's PCI vendor and device IDs (`device/vendor` and `device/device` read `0x1002` and
    /// `0x1900`); `None` where sysfs lists none.
    ids: Option<(u32, u32)>,
}

/// Each render node sysfs lists, in the order of their numbers.
fn render_nodes() -> Vec<RenderNode> {
    let read = |path: std::path::PathBuf| std::fs::read_to_string(path).ok();
    let hex = |text: String| u32::from_str_radix(text.trim().trim_start_matches("0x"), 16).ok();
    let listed = std::fs::read_dir("/sys/class/drm").into_iter().flatten();
    let mut nodes: Vec<RenderNode> = (listed.flatten())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("renderD"))
        .filter_map(|entry| {
            let numbers = read(entry.path().join("dev"))?;
            let (major, minor) = numbers.trim().split_once(':')?;
            let vendor = read(entry.path().join("device/vendor")).and_then(hex);
            let device = read(entry.path().join("device/device")).and_then(hex);
            Some(RenderNode {
                numbers: (major.parse().ok()?, minor.parse().ok()?),
                path: std::path::Path::new("/dev/dri").join(entry.file_name()),
                ids: vendor.zip(device),
            })
        })
        .collect();
    nodes.sort_unstable_by_key(|node| node.numbers);
    nodes
}
