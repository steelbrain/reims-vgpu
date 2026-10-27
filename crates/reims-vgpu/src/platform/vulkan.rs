//! `VulkanPlatform` as an `ApiPlatform` (feature `vulkan`): the runtime's devices, on macOS over
//! MoltenVK and on Linux natively.

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

    /// The texels of a texture made on an IOSurface's plane, in the surface's memory
    /// (`reims_vgpu_vulkan::api::read_in_place`).
    fn read_in_place(
        &self,
        texture: &<Self as Platform>::Texture,
        read: &mut dyn FnMut(*const u8, usize, usize),
    ) -> bool {
        reims_vgpu_vulkan::api::read_in_place(texture, read)
    }

    /// The files behind the map's ranges, registered while the device lives
    /// (`reims_vgpu_vulkan::os::lease_guest_memory`): on Linux the platform makes its views of the
    /// guest's pages of them. A range without a file is not registered, and a view of its pages
    /// is refused as a failed remap is. On macOS nothing is registered, since Mach makes a view
    /// by remapping addresses.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
        let ranges = map.ranges().iter().zip(map.files());
        let files = ranges.filter_map(|(range, file)| {
            let file = file.as_ref()?;
            Some(GuestRamFile {
                host: range.host_address.addr().to_u64(),
                length: range.physical_length,
                fd: Arc::clone(&file.fd),
                offset: file.offset,
            })
        });
        Box::new(reims_vgpu_vulkan::os::lease_guest_memory(files))
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
