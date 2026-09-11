use ash::{vk, Entry};
use serde_json::json;
use std::ffi::{CStr, CString};

#[repr(C)]
struct Export {
    memfd: u32,
    flags: u32,
    offset: u64,
    size: u64,
}

fn main() {
    unsafe {
        run();
    }
}
unsafe fn run() {
    let entry = Entry::load().unwrap();
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )
        .unwrap();
    let gpu = instance
        .enumerate_physical_devices()
        .unwrap()
        .into_iter()
        .find(|&p| {
            let x = instance.get_physical_device_properties(p);
            x.vendor_id == 0x1002 && x.device_id == 0x1586
        })
        .expect("Radeon 8060S not found");
    let props = instance.get_physical_device_properties(gpu);
    println!(
        "{}",
        json!({"device":CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy(),"test":"gpu_read_write_and_page_order"})
    );
    let needed = [
        ash::khr::external_memory_fd::NAME,
        ash::ext::external_memory_dma_buf::NAME,
        ash::ext::queue_family_foreign::NAME,
    ];
    let available = instance.enumerate_device_extension_properties(gpu).unwrap();
    for name in needed {
        assert!(
            available
                .iter()
                .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name),
            "missing {name:?}"
        );
    }
    let handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    let usage = vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::UNIFORM_BUFFER
        | vk::BufferUsageFlags::VERTEX_BUFFER
        | vk::BufferUsageFlags::INDEX_BUFFER;
    let mut caps = vk::ExternalBufferProperties::default();
    instance.get_physical_device_external_buffer_properties(
        gpu,
        &vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(handle),
        &mut caps,
    );
    println!(
        "{}",
        json!({"external_features":format!("{:?}",caps.external_memory_properties.external_memory_features)})
    );
    assert!(caps
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE));
    let family = instance
        .get_physical_device_queue_family_properties(gpu)
        .iter()
        .position(|p| p.queue_count > 0 && p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .unwrap() as u32;
    let priority = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priority)];
    let extensions: Vec<_> = needed.iter().map(|x| x.as_ptr()).collect();
    let device = instance
        .create_device(
            gpu,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&extensions),
            None,
        )
        .unwrap();
    let ext = ash::khr::external_memory_fd::Device::new(&instance, &device);
    let path = CString::new("/dev/udmabuf").unwrap();
    let exporter = libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
    assert!(exporter >= 0, "{}", std::io::Error::last_os_error());
    assert_eq!(libc::sysconf(libc::_SC_PAGESIZE), 4096);
    let iterations: usize = std::env::args()
        .nth(1)
        .unwrap_or("1".into())
        .parse()
        .unwrap();
    assert!((1..=20).contains(&iterations));
    let memory_types = instance.get_physical_device_memory_properties(gpu);
    let pool = device
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )
        .unwrap();
    let cmd = device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1),
        )
        .unwrap()[0];
    let queue = device.get_device_queue(family, 0);
    for iteration in 0..iterations {
        for (size, scattered) in [
            (65536usize, false),
            (1048576, false),
            (16777216, false),
            (65536, true),
            (1048576, true),
        ] {
            let pages = size / 4096;
            let backing_size = if scattered { size * 2 } else { size };
            // Unique, reordered offsets with gaps; physical contiguity is not assumed.
            let order: Vec<usize> = (0..pages)
                .map(|i| {
                    if scattered {
                        (i * 3 + 5) % (pages * 2)
                    } else {
                        i
                    }
                })
                .collect();
            let mut inverse = vec![None; backing_size / 4096];
            for (packed, &original) in order.iter().enumerate() {
                assert!(inverse[original].replace(packed).is_none());
            }
            let name = CString::new("isolated-gpu-dmabuf-test").unwrap();
            let fd = libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING);
            assert!(fd >= 0);
            assert_eq!(libc::ftruncate(fd, backing_size as i64), 0);
            assert_eq!(libc::fcntl(fd, libc::F_ADD_SEALS, libc::F_SEAL_SHRINK), 0);
            let original = map(fd, backing_size);
            for i in 0..backing_size / 4 {
                original.add(i).write_volatile(pattern(i, iteration, 0));
            }
            let dma = if scattered {
                // Header followed by 24-byte udmabuf_create_item entries; u64 storage provides alignment.
                let mut list = vec![0u64; 1 + pages * 3];
                let header = list.as_mut_ptr().cast::<u32>();
                header.write(1);
                header.add(1).write(pages as u32);
                for (i, &page) in order.iter().enumerate() {
                    list[1 + i * 3] = fd as u64;
                    list[2 + i * 3] = (page * 4096) as u64;
                    list[3 + i * 3] = 4096;
                }
                libc::ioctl(exporter, 0x40087543u64 as libc::c_ulong, list.as_ptr())
            } else {
                let export = Export {
                    memfd: fd as u32,
                    flags: 1,
                    offset: 0,
                    size: size as u64,
                };
                libc::ioctl(exporter, 0x40187542u64 as libc::c_ulong, &export)
            };
            assert!(dma >= 0, "export: {}", std::io::Error::last_os_error());
            let packed = map(dma, size);
            sync(dma, 3);
            // Initialize through the dma-buf map within the CPU cache-maintenance interval.
            for i in 0..size / 4 {
                packed.add(i).write_volatile(pattern(
                    order[i / 1024] * 1024 + i % 1024,
                    iteration,
                    1,
                ));
            }
            sync(dma, 7);
            let mut hp = vk::MemoryFdPropertiesKHR::default();
            ext.get_memory_fd_properties(handle, dma, &mut hp).unwrap();
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size as u64)
                        .usage(usage)
                        .push_next(&mut external),
                    None,
                )
                .unwrap();
            let req = device.get_buffer_memory_requirements(buffer);
            assert!(req.size <= size as u64);
            let bits = req.memory_type_bits & hp.memory_type_bits;
            assert_ne!(bits, 0);
            // Keep the original descriptor for CPU synchronization; Vulkan owns the duplicate on success.
            let imported_fd = libc::fcntl(dma, libc::F_DUPFD_CLOEXEC, 0);
            assert!(imported_fd >= 0);
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(handle)
                .fd(imported_fd);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let allocation = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(size as u64)
                        .memory_type_index(bits.trailing_zeros())
                        .push_next(&mut import)
                        .push_next(&mut dedicated),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(buffer, allocation, 0).unwrap();
            let staging = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size((size * 2) as u64)
                        .usage(
                            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
                        ),
                    None,
                )
                .unwrap();
            let sr = device.get_buffer_memory_requirements(staging);
            let mt = (0..memory_types.memory_type_count)
                .find(|&i| {
                    sr.memory_type_bits & (1 << i) != 0
                        && memory_types.memory_types[i as usize]
                            .property_flags
                            .contains(
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                })
                .expect("coherent staging type");
            let sm = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(sr.size)
                        .memory_type_index(mt),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(staging, sm, 0).unwrap();
            let sp = device
                .map_memory(sm, 0, sr.size, vk::MemoryMapFlags::empty())
                .unwrap()
                .cast::<u32>();
            for i in 0..size / 4 {
                sp.add(i).write_volatile(0xdeadbeef);
                sp.add(size / 4 + i)
                    .write_volatile(pattern(i, iteration, 2));
            }
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let acquire = vk::BufferMemoryBarrier::default()
                .buffer(buffer)
                .offset(0)
                .size(size as u64)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(family)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[acquire],
                &[],
            );
            let upload = vk::BufferMemoryBarrier::default()
                .buffer(staging)
                .offset(size as u64)
                .size(size as u64)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[upload],
                &[],
            );
            device.cmd_copy_buffer(
                cmd,
                buffer,
                staging,
                &[vk::BufferCopy::default().size(size as u64)],
            );
            // Finish reads before overwriting the imported payload.
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            );
            device.cmd_copy_buffer(
                cmd,
                staging,
                buffer,
                &[vk::BufferCopy::default()
                    .src_offset(size as u64)
                    .size(size as u64)],
            );
            let release = vk::BufferMemoryBarrier::default()
                .buffer(buffer)
                .offset(0)
                .size(size as u64)
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[release],
                &[],
            );
            let readback = vk::BufferMemoryBarrier::default()
                .buffer(staging)
                .offset(0)
                .size(size as u64)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &[readback],
                &[],
            );
            device.end_command_buffer(cmd).unwrap();
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            let commands = [cmd];
            device
                .queue_submit(
                    queue,
                    &[vk::SubmitInfo::default().command_buffers(&commands)],
                    fence,
                )
                .unwrap();
            device
                .wait_for_fences(&[fence], true, 5_000_000_000)
                .expect("GPU completion within 5 seconds");
            for i in 0..size / 4 {
                assert_eq!(
                    sp.add(i).read_volatile(),
                    pattern(order[i / 1024] * 1024 + i % 1024, iteration, 1),
                    "GPU read mismatch at {i}"
                );
            }
            sync(dma, 3);
            for i in 0..size / 4 {
                assert_eq!(
                    packed.add(i).read_volatile(),
                    pattern(i, iteration, 2),
                    "GPU write mismatch at {i}"
                );
            }
            for i in 0..backing_size / 4 {
                let expected = match inverse[i / 1024] {
                    Some(p) => pattern(p * 1024 + i % 1024, iteration, 2),
                    None => pattern(i, iteration, 0),
                };
                assert_eq!(
                    original.add(i).read_volatile(),
                    expected,
                    "Original mapping/guard mismatch at {i}"
                );
            }
            sync(dma, 7);
            device.destroy_fence(fence, None);
            device.unmap_memory(sm);
            device.destroy_buffer(staging, None);
            device.free_memory(sm, None);
            device.destroy_buffer(buffer, None);
            device.free_memory(allocation, None);
            assert_eq!(libc::munmap(packed.cast(), size), 0);
            assert_eq!(libc::munmap(original.cast(), backing_size), 0);
            assert_eq!(libc::close(dma), 0);
            assert_eq!(libc::close(fd), 0);
            assert_no_test_memfd_refs();
            println!(
                "{}",
                json!({"retired_memfd_refs":0,"iteration":iteration,"size":size,"scattered":scattered,"list_entries":if scattered {pages} else {1},"gpu_read":"pass","gpu_write":"pass","original_mapping_and_guards":"pass"})
            );
        }
    }
    device.destroy_command_pool(pool, None);
    libc::close(exporter);
    device.destroy_device(None);
    instance.destroy_instance(None);
}

fn pattern(word: usize, iteration: usize, phase: u32) -> u32 {
    (word as u32).wrapping_mul(0x9e3779b9).rotate_left(7)
        ^ (iteration as u32).wrapping_mul(0x85ebca6b)
        ^ phase.wrapping_mul(0xc2b2ae35)
}
unsafe fn map(fd: i32, size: usize) -> *mut u32 {
    let p = libc::mmap(
        std::ptr::null_mut(),
        size,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    assert_ne!(
        p,
        libc::MAP_FAILED,
        "mmap: {}",
        std::io::Error::last_os_error()
    );
    p.cast()
}
unsafe fn sync(fd: i32, flags: u64) {
    loop {
        let r = libc::ioctl(fd, 0x40086200u64 as libc::c_ulong, &flags);
        if r == 0 {
            return;
        }
        let error = std::io::Error::last_os_error();
        if matches!(error.raw_os_error(), Some(libc::EINTR) | Some(libc::EAGAIN)) {
            continue;
        }
        panic!("DMA_BUF_IOCTL_SYNC: {error}");
    }
}

fn assert_no_test_memfd_refs() {
    let label = "isolated-gpu-dmabuf-test";
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
    assert!(
        !maps.contains(label),
        "test memfd mapping retained after retirement"
    );
    for fd in std::fs::read_dir("/proc/self/fd").unwrap() {
        if let Ok(target) = std::fs::read_link(fd.unwrap().path()) {
            assert!(
                !target.to_string_lossy().contains(label),
                "test memfd descriptor retained after retirement"
            );
        }
    }
}
