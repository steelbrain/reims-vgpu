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

    /// rt places a shape's image in a Private block of its own, whose bytes the shape's first
    /// fill writes before anything reads them, rather than clearing an image of its own
    /// ([`ApiPlatform::placed_shape_image`]).
    const PLACED_SHAPE_IMAGES: bool = true;

    /// rt places a pinned buffer or heap at its pin only inside a block it knows, and counts
    /// each other one (`rt-gpu-address-unpinned`), which the decoder counts again by kind.
    const PINS_IN_BLOCKS: bool = true;

    /// Each of rt's imports of the host's memory is a Vulkan allocation of its own while it lives
    /// (`reims_vgpu_vulkan::rt_platform::buffer`, "Imports"), which a driver may charge each
    /// submission for: a surface's cache copies its pages through the host's CPU instead.
    const STAGED_SURFACE_COPIES: bool = true;

    /// rt keeps no view of an IOSurface's pages: a surface is a device image whose bytes the
    /// device cache's copies move at the guest's events (lock, unlock and the GPU's work that
    /// uses it), and the views and imports of guest RAM serve an app's own no-copy memory alone.
    const SURFACE_VIEWS: bool = false;

    fn process() -> Arc<RtPlatform> {
        crate::singleton::rt_platform()
    }

    /// The files behind the map's ranges, registered while the device lives
    /// (`super::vulkan::lease_guest_ram`): on Linux the views of a pages block's guest pages, an
    /// app's no-copy memory, are mapped from them, and where the GPU takes no pointer to those
    /// pages, the device imports a dma-buf of them (`rt_platform::import_known`).
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        super::vulkan::lease_guest_ram(map)
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

    /// The platform's let-go of the process's rt device (`RtPlatform::let_go_of_device`): where
    /// nothing beyond the platform held it, rt tears it down on the calling thread before this
    /// returns, so that no device of rt's is live as the process exits.
    fn let_go_of_devices(&self, bound: std::time::Duration) -> Result<(), String> {
        self.let_go_of_device(bound)
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

    /// rt's image for views placed in a Private block of its own
    /// (`RtObj::<RtDevice>::new_placed_texture_for_views`), its bytes the block's as they are.
    /// `None` where rt places none, counted.
    fn placed_shape_image(
        &self,
        device: &<Self as Platform>::Device,
        desc: &TextureDesc,
    ) -> Option<<Self as Platform>::Texture> {
        device.new_placed_texture_for_views(desc)
    }

    /// The first of the host's video arms that takes the device (`graphics::video_choice`): on
    /// Linux, NVDEC's decodes on an NVIDIA GPU, then VA-API's on the GPU's render node; on
    /// Windows, VA-API's on the GPU's DXGI adapter; none on macOS.
    fn graphics_video(
        &self,
        device: &<Self as Platform>::Device,
        runtime: Arc<dyn reims_vgpu_video::core::runtime::Runtime>,
    ) -> Option<Arc<dyn crate::GraphicsVideo<Self>>> {
        crate::graphics::rt_graphics_video(device, runtime)
    }
}

/// rt's import of the `length` bytes of a pages block's view at `first`, as a no-copy buffer's is
/// made and counted, known to `device` from now (`rt_platform::import_known`), a copy of the pages
/// whose GPU writes are dropped where the block's process maps them read-only (`read_only`): its
/// GPU address, and what forgets it. A buffer over the view pinned at the address plus its offset
/// is that range of it, at the guest's address. rt runs `then` once it freed the import.
fn import_view(
    device: &<RtPlatform as Platform>::Device,
    first: HostPtr,
    length: u64,
    read_only: bool,
    then: Box<dyn FnOnce() + Send>,
) -> Option<(u64, Box<dyn FnOnce() + Send>)> {
    let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
    let imported = rt_platform::import_known(device, sel, first, length, read_only, then);
    let (block, offset) = imported?;
    let (known, id) = (device.known_blocks().clone(), block.id);
    let forget: Box<dyn FnOnce() + Send> = Box::new(move || drop(known.forget(id)));
    Some((block.gpu_address.saturating_add(offset), forget))
}
