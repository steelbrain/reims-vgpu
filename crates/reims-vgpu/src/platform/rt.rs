//! `RtPlatform` as an `ApiPlatform`, under the `rt` feature: the platform over metal2vulkan-rt
//! (`reims_vgpu_vulkan::rt_platform`), built piece by piece. A piece not built yet is a counted,
//! typed refusal there. A graphics device's start makes the process's rt device and answers the
//! guest's device info from it; the guest's work is refused at each unbuilt piece it reaches.

use reims_vgpu_vulkan::RtPlatform;
use reims_vgpu_vulkan::core::platform::Platform;

use super::*;

impl ApiPlatform for RtPlatform {
    /// rt keeps no texture over a device cache's buffer: each is a view of its shape's image
    /// (`graphics::decode::surface_shapes`), and every surface is cached.
    const SURFACE_SHAPES: bool = true;

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

    /// The whole texture, through rt's `getBytes:` (`RtTexture::read_pixels`); `dst` is left as
    /// it was where rt refuses it, counted.
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        texture.read_pixels(dst, bytes_per_row);
    }
}
