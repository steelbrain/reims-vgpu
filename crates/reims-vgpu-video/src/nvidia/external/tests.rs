//! [`ImportedBuffer`] on the NVIDIA host: a buffer past the first byte of a Vulkan device's
//! allocation dedicated to it, exported as an opaque file descriptor and imported dedicated, lands
//! where it lies (module docs, "Mapped from the allocation's first byte").

use std::ffi::CStr;
use std::os::fd::{FromRawFd as _, OwnedFd};
use std::sync::{Arc, OnceLock};

use ash::vk;

use super::*;
use crate::nvidia::cuda::ffi::{CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST, CUDA_MEMCPY2D};
use crate::nvidia::cuda::{GpuSelector, Pinned, Stream};
use crate::nvidia::wait;

/// The exported buffer's bytes.
const BYTES: u64 = 1 << 20;

/// The buffers imported from its allocation, each `(offset, length)`: 224 bytes in, as a device
/// cache's first plane lies past CoreVideo's 32 bytes, to the buffer's last byte; and a page in,
/// short of it.
const PARTS: [(u64, u64); 2] = [(224, BYTES - 224), (4096, 64 << 10)];

/// What a byte reads as before any copy lands: neither the zeros nor the pattern the test writes.
const UNWRITTEN: u8 = 0xee;

/// A buffer of the exported memory, past its allocation's first byte, imported dedicated as an
/// exporter's dedicated allocation must be: written through its own mapping, it lands at its
/// offset in the allocation, read through a mapping of the whole, with every other byte as it was.
/// CUDA maps a dedicated import's buffers only from offset 0 (module docs), so a mapping from the
/// buffer's offset fails the import.
#[test]
#[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
fn a_buffer_past_a_dedicated_allocations_first_byte_lands_at_its_offset() {
    let gpu = Arc::new(Gpu::open(&GpuSelector::First).expect("an NVIDIA GPU"));
    let exporter = Exporter::on(&gpu);
    let size = exporter.size;
    let bytes = usize::try_from(size).expect("the allocation's bytes");
    let whole = ImportedBuffer::import(&gpu, exporter.fd(), size, 0, size, true)
        .expect("the allocation imported whole");
    let stream = Stream::new(&gpu).expect("a CUDA stream");
    let mut host = Pinned::new(&gpu, bytes).expect("page-locked memory");
    for (offset, length) in PARTS {
        let part = ImportedBuffer::import(&gpu, exporter.fd(), size, offset, length, true)
            .unwrap_or_else(|e| panic!("a buffer of {length} bytes at {offset}: {e}"));
        let (start, end) = (offset as usize, (offset + length) as usize);
        assert_eq!(part.length(), end - start);
        host.bytes_mut().fill(0);
        upload(&gpu, &stream, &host, whole.address_at(0), bytes);
        let pattern: Vec<u8> = (0..end - start).map(|i| (i % 251) as u8 + 1).collect();
        host.bytes_mut()[..pattern.len()].copy_from_slice(&pattern);
        upload(&gpu, &stream, &host, part.address_at(0), pattern.len());
        host.bytes_mut().fill(UNWRITTEN);
        download(&gpu, &stream, whole.address_at(0), &mut host, bytes);
        let read = host.bytes();
        let stray = |range: std::ops::Range<usize>| range.clone().find(|&at| read[at] != 0);
        assert_eq!(stray(0..start), None, "a byte before the buffer at {offset} was written");
        assert_eq!(stray(end..bytes), None, "a byte past the buffer at {offset} was written");
        let wrong = (start..end).find(|&at| read[at] != pattern[at - start]);
        assert_eq!(wrong, None, "the buffer at {offset} landed elsewhere");
    }
}

/// The Vulkan loader and an instance, made at the first use and kept until the process ends,
/// never destroyed (`vaapi::va::spin`'s `vulkan` says why).
fn vulkan() -> &'static (ash::Entry, ash::Instance) {
    static VULKAN: OnceLock<(ash::Entry, ash::Instance)> = OnceLock::new();
    VULKAN.get_or_init(|| {
        // SAFETY: loads the system's Vulkan loader, kept open for the process.
        let entry = unsafe { ash::Entry::load() }.expect("a Vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: a valid create info, for the call.
        let instance = unsafe { entry.create_instance(&info, None) }.expect("a Vulkan instance");
        (entry, instance)
    })
}

/// A Vulkan device of a CUDA device's GPU, and a buffer of [`BYTES`] bytes in device-local
/// memory bound to an allocation dedicated to it, exportable as an opaque file descriptor.
struct Exporter {
    device: ash::Device,
    fds: ash::khr::external_memory_fd::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// The allocation's bytes.
    size: u64,
}

impl Exporter {
    fn on(gpu: &Gpu) -> Self {
        let (_, instance) = vulkan();
        let (domain, bus, device) = gpu.info().pci;
        let pci = (domain as u32, bus as u32, device as u32);
        // SAFETY: the instance's own calls, with valid out-pointers.
        let physicals = unsafe { instance.enumerate_physical_devices() }.expect("Vulkan's GPUs");
        let has = |physical, name: &CStr| {
            // SAFETY: a physical device of the instance.
            unsafe { instance.enumerate_device_extension_properties(physical) }
                .unwrap_or_default()
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let physical = physicals
            .into_iter()
            .find(|&physical| {
                if !has(physical, ash::ext::pci_bus_info::NAME) {
                    return false;
                }
                let mut bus = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
                let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut bus);
                // SAFETY: a physical device of the instance, and a chain of valid structures.
                unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
                (bus.pci_domain, bus.pci_bus, bus.pci_device) == pci
            })
            .unwrap_or_else(|| panic!("no Vulkan device at the CUDA device's PCI address {pci:?}"));
        assert!(
            has(physical, ash::khr::external_memory_fd::NAME),
            "the GPU's Vulkan device exports no opaque file descriptor"
        );
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&priorities)];
        let extensions = [ash::khr::external_memory_fd::NAME.as_ptr()];
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions);
        // SAFETY: a valid create info, for the call; destroyed in the drop.
        let device =
            unsafe { instance.create_device(physical, &info, None) }.expect("vkCreateDevice");
        let opaque = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(opaque);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(BYTES)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        // SAFETY: a valid create info, for the call; destroyed in the drop.
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.expect("vkCreateBuffer");
        // SAFETY: a buffer of the device.
        let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
        // SAFETY: a physical device of the instance.
        let types = unsafe { instance.get_physical_device_memory_properties(physical) };
        let index = (0..types.memory_type_count)
            .find(|&index| {
                let flags = types.memory_types[index as usize].property_flags;
                needs.memory_type_bits & (1 << index) != 0
                    && flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .expect("a device-local memory type the buffer takes");
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(opaque);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(index)
            .push_next(&mut dedicated)
            .push_next(&mut export);
        // SAFETY: a valid allocate info, dedicated to the device's buffer; freed in the drop.
        let memory = unsafe { device.allocate_memory(&allocate, None) }.expect("vkAllocateMemory");
        // SAFETY: the buffer's dedicated allocation, of its size, in a type it takes.
        unsafe { device.bind_buffer_memory(buffer, memory, 0) }.expect("vkBindBufferMemory");
        let fds = ash::khr::external_memory_fd::Device::new(instance, &device);
        Self {
            device,
            fds,
            buffer,
            memory,
            size: needs.size,
        }
    }

    /// A new opaque file descriptor of the allocation.
    fn fd(&self) -> OwnedFd {
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(self.memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        // SAFETY: memory of the device, allocated exportable as an opaque file descriptor.
        let raw = unsafe { self.fds.get_memory_fd(&info) }.expect("vkGetMemoryFdKHR");
        // SAFETY: a descriptor the call made, which the caller now owns.
        unsafe { OwnedFd::from_raw_fd(raw) }
    }
}

impl Drop for Exporter {
    fn drop(&mut self) {
        // SAFETY: the device's own objects, unused by any GPU work, each destroyed once, the
        // device last.
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
            self.device.destroy_device(None);
        }
    }
}

/// A copy of `bytes` bytes as one row, the source and destination to be named.
fn row(bytes: usize) -> CUDA_MEMCPY2D {
    CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: CU_MEMORYTYPE_HOST,
        srcHost: std::ptr::null(),
        srcDevice: 0,
        srcArray: std::ptr::null_mut(),
        srcPitch: bytes,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(),
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: bytes,
        WidthInBytes: bytes,
        Height: 1,
    }
}

/// The first `bytes` of `host` copied to `to`, waited for.
fn upload(gpu: &Arc<Gpu>, stream: &Stream, host: &Pinned, to: CUdeviceptr, bytes: usize) {
    assert!(bytes <= host.bytes().len());
    let copy = CUDA_MEMCPY2D {
        srcHost: host.bytes().as_ptr().cast(),
        dstDevice: to,
        ..row(bytes)
    };
    run(gpu, stream, &copy);
}

/// `bytes` bytes from `from` copied into the first of `host`, waited for.
fn download(gpu: &Arc<Gpu>, stream: &Stream, from: CUdeviceptr, host: &mut Pinned, bytes: usize) {
    assert!(bytes <= host.bytes().len());
    let copy = CUDA_MEMCPY2D {
        srcMemoryType: CU_MEMORYTYPE_DEVICE,
        srcDevice: from,
        dstMemoryType: CU_MEMORYTYPE_HOST,
        dstHost: host.bytes_mut().as_mut_ptr().cast(),
        ..row(bytes)
    };
    run(gpu, stream, &copy);
}

/// `copy` made on `stream`, and waited for, as long as the module waits for the GPU at most
/// (`wait::deadline`).
fn run(gpu: &Arc<Gpu>, stream: &Stream, copy: &CUDA_MEMCPY2D) {
    let cuda = gpu.cuda();
    {
        let _current = gpu.push().expect("the GPU's context");
        // SAFETY: the copy's host side is page-locked memory the caller holds until the copy is
        // waited for below, and its device side a mapping of an import the caller holds, each
        // as long as the copy; the stream is live, on the context current here.
        let result = unsafe { (cuda.f.cuMemcpy2DAsync_v2)(copy, stream.raw()) };
        cuda.check("cuMemcpy2DAsync", result).expect("a copy queued");
    }
    let done = wait::poll(wait::deadline(), || {
        stream.is_done().expect("the copy").then_some(())
    });
    assert!(done.is_some(), "a copy of {} bytes outlasted the GPU's wait", copy.WidthInBytes);
}
