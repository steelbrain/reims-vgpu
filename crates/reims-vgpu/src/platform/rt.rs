//! `RtPlatform` as an `ApiPlatform`, under the `rt` feature: the platform over metal2vulkan-rt
//! (`reims_vgpu_vulkan::rt_platform`), built piece by piece. A piece not built yet is a counted,
//! typed refusal there. A graphics device's start makes the process's rt device and answers the
//! guest's device info from it; the guest's work is refused at each unbuilt piece it reaches.

use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::platform::Platform;

use super::*;

impl ApiPlatform for RtPlatform {
    fn process() -> Arc<RtPlatform> {
        crate::singleton::rt_platform()
    }

    /// The process's default device, or the device whose registry ID is `id`
    /// (`Platform::open_device`).
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => self.system_default_device(),
            Some(id) => self.open_device(id),
        }
    }

    /// Never reached: no texture exists until the textures' piece is built.
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        let _ = (dst, bytes_per_row);
        texture.unreachable()
    }
}
