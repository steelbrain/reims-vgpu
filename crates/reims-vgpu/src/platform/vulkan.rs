//! `VulkanPlatform` as an `ApiPlatform` (`cfg(vulkan_platform)`): the runtime's devices, on macOS
//! over MoltenVK, with the `vulkan` feature, and on Linux natively, always.

use reims_vgpu_core::memory::dirty::DirtyLog;
use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::os::GuestRamFile;

use super::*;

#[cfg(target_os = "linux")]
mod lent_memory;

/// The platform over metal2vulkan's runtime (architecture 2.2): the runtime's devices, and on
/// macOS windows whose frames Metal copies from the runtime's (`window::window_sink`).
impl ApiPlatform for VulkanPlatform {
    fn process() -> Arc<VulkanPlatform> {
        crate::singleton::vulkan_platform()
    }

    /// The runtime's system default device (`MTLCreateSystemDefaultDevice()`), or the device
    /// whose registry ID, as the platform answers it, is `id` (`Platform::open_device`): the
    /// default device answers Metal's `registryID` of the same GPU, which is what a host
    /// names it by, and any other device the runtime's own ID.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => VulkanPlatform::system_default_device(),
            Some(id) => self.open_device(id),
        }
    }

    /// On Linux a shared mapping of a memfd of the library's
    /// (`reims_vgpu_vulkan::os::RegionMemory`): RADV takes no pointer to it and imports dma-bufs
    /// of its pages, which spares each of the device's submissions the walk of every page amdgpu
    /// makes of a userptr import; a device that takes the pointer imports it as before. None on
    /// macOS.
    fn graphics_region(&self, len: usize) -> Option<reims_graphics::Region> {
        let memory = reims_vgpu_vulkan::os::RegionMemory::new(u64::try_from(len).ok()?)?;
        let base = memory.base();
        // SAFETY: the memory's `len` bytes from its page-aligned first byte are fresh zeros of a
        // mapping of its own, read and write, which nothing else reaches and which it unmaps only
        // as it drops, after the region's last use.
        unsafe { reims_graphics::Region::over(base, len, Box::new(memory)) }
    }

    /// The whole texture, through the runtime's `get_bytes`
    /// (`reims_vgpu_vulkan::api::read_pixels`).
    fn read_pixels(
        &self,
        texture: &<Self as Platform>::Texture,
        dst: &mut [u8],
        bytes_per_row: usize,
    ) {
        reims_vgpu_vulkan::api::read_pixels(texture, dst, bytes_per_row);
    }

    /// The files behind the map's ranges, registered while the device lives
    /// (`reims_vgpu_vulkan::os::lease_guest_memory`): on Linux the platform makes its views of the
    /// guest's pages of them. A range without a file is not registered, and a view of its pages
    /// is refused as a failed remap is. On macOS nothing is registered, since Mach makes a view
    /// by remapping addresses. Each range carries its guest physical address, and with the
    /// device's dirty log the platform finds the guest pages behind an import and the log they
    /// stamp (`reims_vgpu_vulkan::os::dirty`).
    fn guest_memory(
        &self,
        map: &MemoryMap,
        dirty: Option<Arc<DirtyLog>>,
    ) -> Box<dyn Any + Send + Sync> {
        let ranges = map.ranges().iter().zip(map.files());
        let files = ranges.filter_map(|(range, file)| {
            let file = file.as_ref()?;
            Some(GuestRamFile {
                host: range.host_address.addr().to_u64(),
                guest: range.physical_address,
                length: range.physical_length,
                fd: Arc::clone(&file.fd),
                offset: file.offset,
            })
        });
        Box::new(reims_vgpu_vulkan::os::lease_guest_memory(files, dirty))
    }

    /// Each plane the surface was made with (`reims_vgpu_vulkan::os::plane_layout`): on Linux
    /// the stand-in's record of its spec, on macOS IOSurface's per-plane getters.
    fn surface_planes(
        &self,
        surface: &<Self as Platform>::Surface,
    ) -> Vec<crate::video::SurfacePlane> {
        use reims_vgpu_core::platform::os::surface::Surface as _;
        (0..surface.plane_count())
            .map_while(|i| reims_vgpu_vulkan::os::plane_layout(surface, i))
            .map(|p| crate::video::SurfacePlane {
                offset: p.offset,
                width: p.width,
                height: p.height,
                bytes_per_element: p.bytes_per_element,
                bytes_per_row: p.bytes_per_row,
            })
            .collect()
    }

    /// Vulkan Video on the device's video-decode queue (`graphics::vulkan::video`).
    fn graphics_video(
        &self,
        device: &<Self as Platform>::Device,
    ) -> Option<Arc<dyn crate::GraphicsVideo<Self>>> {
        crate::graphics::vulkan_graphics_video(device)
    }

    fn lend_surface_bytes(
        &self,
        surface: &<Self as Platform>::Surface,
        write: &mut dyn FnMut(&mut [u8]),
    ) -> Result<(), &'static str> {
        reims_vgpu_vulkan::os::lend_surface_bytes(surface, write)
    }

    fn lend_surface_bytes_for_reading(
        &self,
        surface: &<Self as Platform>::Surface,
        read: &mut dyn FnMut(&[u8]),
    ) -> Result<(), &'static str> {
        reims_vgpu_vulkan::os::lend_surface_bytes_for_reading(surface, read)
    }

    /// On Linux, the bytes `lend_surface_bytes` lends, as a dma-buf of their pages made by
    /// udmabuf and as the surface's view of them, with an identity attached to the surface
    /// (`platform/vulkan/lent_memory.rs`). On macOS none: there the Metal platform lends a
    /// surface's IOSurface to VideoToolbox (`platform/metal/lent_memory.rs`).
    #[cfg(target_os = "linux")]
    fn surface_device_memory(
        &self,
        surface: &<Self as Platform>::Surface,
    ) -> Result<Box<dyn DeviceMemory>, &'static str> {
        lent_memory::lent_memory(surface).map(|lent| Box::new(lent) as Box<dyn DeviceMemory>)
    }
}
