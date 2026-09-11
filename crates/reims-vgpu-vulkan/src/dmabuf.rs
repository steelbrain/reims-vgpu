//! Capability admission for the experimental Linux DMA-BUF buffer transport.
use ash::vk;
use std::ffi::CStr;
pub const EXTENSIONS: [&CStr; 3] = [
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::ext::queue_family_foreign::NAME,
];
pub const USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw(),
);
/// # Safety
/// `pd` belongs to `instance`; Vulkan 1.2 baseline has been established.
pub unsafe fn supported(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    extensions: &[vk::ExtensionProperties],
) -> bool {
    if !EXTENSIONS.iter().all(|name| {
        extensions
            .iter()
            .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == *name)
    }) {
        return false;
    }
    let mut caps = vk::ExternalBufferProperties::default();
    unsafe {
        instance.get_physical_device_external_buffer_properties(
            pd,
            &vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(USAGE)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            &mut caps,
        )
    };
    caps.external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
}
