//! `RtPlatform` as an `ApiPlatform`, under the `rt` feature: the platform over metal2vulkan-rt
//! (`reims_vgpu_vulkan::rt_platform`), built piece by piece. A piece not built yet is a counted,
//! typed refusal there. A graphics device's start makes the process's rt device and answers the
//! guest's device info from it; the guest's work is refused at each unbuilt piece it reaches.

use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::{RtPlatform, rt_platform};

use super::*;

impl ApiPlatform for RtPlatform {
    /// rt keeps no texture over a device cache's buffer: each is a view of its shape's image
    /// (`graphics::decode::surface_shapes`), and every surface is cached.
    const SURFACE_SHAPES: bool = true;

    /// rt places a pinned buffer or heap at its pin only inside a block it knows, and counts
    /// each other one (`rt-gpu-address-unpinned`), which the decoder counts again by kind.
    const PINS_IN_BLOCKS: bool = true;

    /// Each of rt's imports of the host's memory is a Vulkan allocation of its own while it lives
    /// (`reims_vgpu_vulkan::rt_platform::buffer`, "Imports"), which a driver may charge each
    /// submission for: a surface's cache copies its pages through the host's CPU instead.
    const STAGED_SURFACE_COPIES: bool = true;

    fn process() -> Arc<RtPlatform> {
        crate::singleton::rt_platform()
    }

    /// rt's blocks (`pool::RtBlocks`), each known to `device` from its take until just before it
    /// goes back (`KnownBlocks`): the buffers and heaps the guest pins inside its pool's blocks
    /// and its Private blocks are ranges of them, at the guest's addresses.
    fn block_source(
        &self,
        device: &<Self as Platform>::Device,
    ) -> Option<Arc<dyn crate::pool::BlockSource>> {
        let known = Box::new(device.known_blocks().clone());
        Some(Arc::new(crate::pool::RtBlocks::new(
            device.rt().clone(),
            known,
        )))
    }

    /// rt's import of a pages block's view ([`import_view`]).
    fn view_imports(&self) -> Option<ViewImports<RtPlatform>> {
        Some(Arc::new(import_view))
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

    /// rt's image for views (`RtObj::<RtDevice>::new_texture_for_views`, metal2vulkan-rt's
    /// `Device::new_texture_for_views`): it takes no index, and the guest's textures over the
    /// shape are its views at their own. Nil where rt refuses it, counted.
    fn shape_image(
        &self,
        device: &<Self as Platform>::Device,
        desc: &TextureDesc,
    ) -> Option<<Self as Platform>::Texture> {
        device.new_texture_for_views(desc)
    }
}

/// rt's import of the `length` bytes of a pages block's view at `first`, as a no-copy buffer's is
/// made and counted, known to `device` from now (`rt_platform::import_known`): its GPU address,
/// and what forgets it. A buffer over the view pinned at the address plus its offset is that range
/// of it, at the guest's address. rt runs `then` once it freed the import.
fn import_view(
    device: &<RtPlatform as Platform>::Device,
    first: HostPtr,
    length: u64,
    then: Box<dyn FnOnce() + Send>,
) -> Option<(u64, Box<dyn FnOnce() + Send>)> {
    let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
    let (block, offset) = rt_platform::import_known(device, sel, first, length, then)?;
    let (known, id) = (device.known_blocks().clone(), block.id);
    let forget: Box<dyn FnOnce() + Send> = Box::new(move || drop(known.forget(id)));
    Some((block.gpu_address.saturating_add(offset), forget))
}
