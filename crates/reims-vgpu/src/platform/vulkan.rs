//! `VulkanPlatform` as an `ApiPlatform`: the runtime's devices, on macOS over MoltenVK, and on
//! Linux natively.

use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::os::GuestRamFile;

use super::*;

#[cfg(target_os = "linux")]
mod dma_buf;
#[cfg(target_os = "linux")]
pub(crate) use dma_buf::host_dma_buf;

/// The platform over metal2vulkan's runtime (architecture 2.2): the runtime's devices.
impl ApiPlatform for VulkanPlatform {
    fn process() -> Arc<VulkanPlatform> {
        crate::singleton::platform()
    }

    /// The process's default device (`MTLCreateSystemDefaultDevice()`, the GPU the platform's
    /// choice of device picks where the host has more than one), or the device
    /// whose registry ID, as the platform answers it, is `id` (`Platform::open_device`): the
    /// default device answers Metal's `registryID` of the same GPU, which is what a host
    /// names it by, and any other device the runtime's own ID.
    fn metal_device(&self, registry_id: Option<u64>) -> Option<<Self as Platform>::Device> {
        match registry_id {
            None => VulkanPlatform::system_default_device(),
            Some(id) => self.open_device(id),
        }
    }

    /// Memory of the library's own, which it frees through the platform's VM as the region
    /// drops, so the unmap waits until the runtime has freed every import of its pages: the
    /// region's last reference can go in a no-copy buffer's deallocator, which the runtime runs
    /// before it frees the buffer's import. On Linux a shared mapping of a memfd
    /// (`reims_vgpu_vulkan::os::RegionMemory`): RADV takes no pointer to it and imports dma-bufs
    /// of its pages, which spares each of the device's submissions the walk of every page amdgpu
    /// makes of a userptr import; a device that takes the pointer imports it as before. Where the
    /// host gives no such mapping, and on macOS, private memory
    /// (`reims_vgpu_vulkan::os::PrivateMemory`), which every GPU imports by its pointer.
    fn graphics_region(&self, len: usize) -> Option<reims_vgpu_graphics::region::Region> {
        let length = u64::try_from(len).ok()?;
        if let Some(memory) = reims_vgpu_vulkan::os::RegionMemory::new(length) {
            let base = memory.base();
            // SAFETY: the memory's `len` bytes from its page-aligned first byte are fresh zeros of
            // a mapping of its own, read and write, which nothing else reaches and which it unmaps
            // only as it drops, after the region's last use.
            return unsafe {
                reims_vgpu_graphics::region::Region::over(base, len, Box::new(memory))
            };
        }
        let memory = reims_vgpu_vulkan::os::PrivateMemory::new(length)?;
        let base = memory.base();
        // SAFETY: as above: the memory's own fresh zeros, read and write, from its page-aligned
        // first byte, deallocated only as it drops.
        unsafe { reims_vgpu_graphics::region::Region::over(base, len, Box::new(memory)) }
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
    /// by remapping addresses.
    fn guest_memory(&self, map: &MemoryMap) -> Box<dyn Any + Send + Sync> {
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
        Box::new(reims_vgpu_vulkan::os::lease_guest_memory(files))
    }

    /// The first of the host's video arms that takes the device (`graphics::video_choice`): on
    /// Linux, NVDEC on an NVIDIA GPU, else VA-API on the device's render node; none on macOS.
    fn graphics_video(
        &self,
        device: &<Self as Platform>::Device,
        runtime: Arc<dyn reims_vgpu_video::core::runtime::Runtime>,
    ) -> Option<Arc<dyn crate::GraphicsVideo<Self>>> {
        crate::graphics::vulkan_graphics_video(device, runtime)
    }
}
