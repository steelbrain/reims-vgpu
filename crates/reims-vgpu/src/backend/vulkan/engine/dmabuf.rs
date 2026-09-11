//! Experimental packed-buffer fallback with explicit, serialized ownership.
use super::{
    context::DeviceContext,
    host_ram::{HostRamDecline, ImportedHostRam},
};
use ash::vk;
use reims_vgpu_memory::dmabuf::SharedDmaBuf;
use std::os::fd::{AsRawFd, IntoRawFd};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, Weak,
};

#[derive(Debug)]
pub(crate) struct Lease {
    backing: Arc<SharedDmaBuf>,
    buffer: vk::Buffer,
    retired: AtomicBool,
    quarantined: AtomicBool,
    alias: (usize, usize),
}
impl Lease {
    pub fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }
    pub fn quarantined(&self) -> bool {
        self.quarantined.load(Ordering::Acquire)
    }
}
#[derive(Default)]
pub(crate) struct Rail {
    live: Mutex<Vec<Weak<Lease>>>,
    quarantine: Mutex<Vec<Arc<Lease>>>,
    poisoned: AtomicBool,
}
impl Rail {
    fn snapshot(&self) -> Vec<Arc<Lease>> {
        let mut registry = self.live.lock().unwrap();
        let mut live = Vec::new();
        registry.retain(|weak| {
            if let Some(lease) = weak
                .upgrade()
                .filter(|l| !l.retired.load(Ordering::Acquire))
            {
                live.push(lease);
                true
            } else {
                false
            }
        });
        live
    }
    pub fn has_live(&self) -> bool {
        self.poisoned.load(Ordering::Acquire) || !self.snapshot().is_empty()
    }
    /// The Vulkan device has been destroyed, so even unconfirmed submissions
    /// can no longer touch these aliases. Driver-owned temp objects died with it.
    pub fn device_destroyed(&self) {
        for lease in self.quarantine.lock().unwrap().drain(..) {
            super::release_host_alias(lease.alias);
        }
    }
    /// Submit under the engine's queue lock. Completion is deliberately not
    /// published here: the caller signals guest completion after CPU sync.
    pub unsafe fn submit(
        &self,
        ctx: &DeviceContext,
        commands: &[vk::CommandBuffer],
    ) -> Result<(), vk::Result> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        let live = self.snapshot();
        for lease in &live {
            sync(&lease.backing, false)?;
            sync(&lease.backing, true)?;
        }
        let pool = unsafe {
            ctx.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gq),
                None,
            )
        }?;
        let mut unconfirmed = false;
        let result = (|| {
            let barriers = unsafe {
                ctx.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .command_buffer_count(2),
                )
            }?;
            for (index, acquire) in [(0, true), (1, false)] {
                let cmd = barriers[index];
                unsafe {
                    ctx.device.begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                }?;
                let bb: Vec<_> = live
                    .iter()
                    .map(|l| {
                        vk::BufferMemoryBarrier::default()
                            .buffer(l.buffer)
                            .offset(0)
                            .size(l.backing.len())
                            .src_queue_family_index(if acquire {
                                vk::QUEUE_FAMILY_FOREIGN_EXT
                            } else {
                                ctx.gq
                            })
                            .dst_queue_family_index(if acquire {
                                ctx.gq
                            } else {
                                vk::QUEUE_FAMILY_FOREIGN_EXT
                            })
                            .src_access_mask(if acquire {
                                vk::AccessFlags::empty()
                            } else {
                                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                            })
                            .dst_access_mask(if acquire {
                                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
                            } else {
                                vk::AccessFlags::empty()
                            })
                    })
                    .collect();
                unsafe {
                    ctx.device.cmd_pipeline_barrier(
                        cmd,
                        if acquire {
                            vk::PipelineStageFlags::TOP_OF_PIPE
                        } else {
                            vk::PipelineStageFlags::ALL_COMMANDS
                        },
                        if acquire {
                            vk::PipelineStageFlags::ALL_COMMANDS
                        } else {
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE
                        },
                        vk::DependencyFlags::empty(),
                        &[],
                        &bb,
                        &[],
                    )
                };
                unsafe { ctx.device.end_command_buffer(cmd) }?;
            }
            let fence = unsafe {
                ctx.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            }?;
            let mut cbs = Vec::with_capacity(commands.len() + 2);
            cbs.push(barriers[0]);
            cbs.extend_from_slice(commands);
            cbs.push(barriers[1]);
            let submitted = unsafe {
                ctx.device.queue_submit(
                    ctx.queue(),
                    &[vk::SubmitInfo::default().command_buffers(&cbs)],
                    fence,
                )
            };
            let result = match submitted {
                Ok(()) => unsafe { ctx.device.wait_for_fences(&[fence], true, u64::MAX) },
                Err(e) => Err(e),
            };
            if result.is_err() {
                // A failed wait is not proof of completion. Retain backing and
                // leave temp Vulkan objects to device destruction, never free
                // them while an unconfirmed submission might still use them.
                unconfirmed = true;
                self.poisoned.store(true, Ordering::Release);
                for lease in &live {
                    lease.quarantined.store(true, Ordering::Release);
                }
                self.quarantine.lock().unwrap().extend(live.iter().cloned());
                crate::observe::fail(
                    "dmabuf_submission_unconfirmed reason=dmabuf_submission_unconfirmed",
                );
                return Err(vk::Result::ERROR_DEVICE_LOST);
            }
            unsafe { ctx.device.destroy_fence(fence, None) };
            for lease in &live {
                sync(&lease.backing, false)?;
                sync(&lease.backing, true)?;
            }
            Ok(())
        })();
        if !unconfirmed {
            unsafe { ctx.device.destroy_command_pool(pool, None) };
        }
        result
    }
}
fn sync(backing: &SharedDmaBuf, end: bool) -> Result<(), vk::Result> {
    backing.sync_cpu(end).map_err(|errno| {
        crate::observe::fail(format!(
            "dmabuf_cpu_sync_failed reason=dmabuf_cpu_sync_failed errno={errno}"
        ));
        vk::Result::ERROR_DEVICE_LOST
    })
}

pub(crate) unsafe fn import(
    ctx: &DeviceContext,
    guest: &crate::runtime::guest_ram::GuestRamImport,
) -> Result<ImportedHostRam, HostRamDecline> {
    let loader = ctx
        .dmabuf_loader
        .as_ref()
        .ok_or(HostRamDecline::DmaBufUnavailable)?;
    let backing = guest.dmabuf().ok_or(HostRamDecline::DmaBufUnavailable)?;
    if guest.gpa_base().is_some() || backing.len() != guest.len() {
        return Err(HostRamDecline::DmaBufUnavailable);
    }
    let size = backing.len();
    let mut props = vk::MemoryFdPropertiesKHR::default();
    unsafe {
        loader.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            backing.fd().as_raw_fd(),
            &mut props,
        )
    }
    .map_err(|result| HostRamDecline::DmaBufImport { result })?;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let buffer = unsafe {
        ctx.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(reims_vgpu_vulkan::dmabuf::USAGE)
                .push_next(&mut external),
            None,
        )
    }
    .map_err(|result| HostRamDecline::CreateBuffer { result })?;
    let result = (|| {
        let req = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if req.size > size {
            return Err(HostRamDecline::TooSmall {
                required: req.size,
                available: size,
            });
        }
        let request = ctx
            .caps
            .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload);
        let pick = reims_vgpu_vulkan::memory::select_memory_type(
            &ctx.memory_properties,
            req.memory_type_bits & props.memory_type_bits,
            &request,
            size,
            ctx.caps.max_allocation_size,
        )
        .map_err(|_| HostRamDecline::DmaBufUnavailable)?;
        let fd = backing
            .fd()
            .try_clone()
            .map_err(|_| HostRamDecline::DmaBufUnavailable)?;
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let memory = unsafe {
            ctx.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(size)
                    .memory_type_index(pick.index)
                    .push_next(&mut import)
                    .push_next(&mut dedicated),
                None,
            )
        }
        .map_err(|result| HostRamDecline::DmaBufImport { result })?;
        let _transferred = fd.into_raw_fd();
        if let Err(result) = unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe { ctx.device.free_memory(memory, None) };
            return Err(HostRamDecline::BindBuffer { result });
        }
        let lease = Arc::new(Lease {
            backing: Arc::clone(backing),
            buffer,
            retired: AtomicBool::new(false),
            quarantined: AtomicBool::new(false),
            alias: (guest.host_base(), size as usize),
        });
        ctx.dmabuf_rail
            .live
            .lock()
            .unwrap()
            .push(Arc::downgrade(&lease));
        crate::observe::off(format!(
            "dmabuf_import id={} bytes={size} mtype={} ok=true serialized=true",
            guest.id().get(),
            pick.index
        ));
        Ok(ImportedHostRam {
            buffer,
            memory,
            memory_type_index: pick.index,
            size,
            alias: Some((guest.host_base(), size as usize)),
            dmabuf: Some(lease),
        })
    })();
    if result.is_err() {
        unsafe { ctx.device.destroy_buffer(buffer, None) };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_memory::{dmabuf::SharedPage, GuestRamImport};
    use std::os::fd::{FromRawFd, OwnedFd};

    #[test]
    #[ignore = "Z13 regression: requires shared host-pointer refusal and accessible udmabuf"]
    fn packed_dma_buffers_preserve_data_order_and_retire() {
        unsafe {
            let mut engine = super::super::lock_engine();
            let super::super::EngineState {
                ref mut owner,
                ref counters,
                ..
            } = &mut *engine;
            let ctx = owner
                .ensure(counters)
                .expect("Vulkan device required for integration test");
            assert!(
                ctx.dmabuf_loader.is_some(),
                "DMA-BUF capability required for this test"
            );
            for iteration in 0..3u32 {
                for scattered in [false, true] {
                    const SIZE: usize = 65536;
                    let total = if scattered { SIZE * 2 } else { SIZE };
                    let raw = libc::memfd_create(
                        c"reims-dmabuf-test".as_ptr(),
                        libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                    );
                    assert!(raw >= 0);
                    let fd = OwnedFd::from_raw_fd(raw);
                    assert_eq!(libc::ftruncate(fd.as_raw_fd(), total as i64), 0);
                    assert_eq!(
                        libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
                        0
                    );
                    let original = libc::mmap(
                        std::ptr::null_mut(),
                        total,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        fd.as_raw_fd(),
                        0,
                    );
                    assert_ne!(original, libc::MAP_FAILED);
                    std::ptr::write_bytes(original, 0xa7, total);
                    let order: Vec<usize> = (0..SIZE / 4096)
                        .map(|i| {
                            if scattered {
                                (3 * i + 5) % (total / 4096)
                            } else {
                                i
                            }
                        })
                        .collect();
                    let pages: Vec<_> = order
                        .iter()
                        .map(|i| SharedPage {
                            fd: fd.as_raw_fd(),
                            reserved: 0,
                            offset: (i * 4096) as u64,
                        })
                        .collect();
                    let backing = Arc::new(SharedDmaBuf::export(&pages, 4096).unwrap());
                    let packed = libc::mmap(
                        std::ptr::null_mut(),
                        SIZE,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED,
                        backing.fd().as_raw_fd(),
                        0,
                    );
                    assert_ne!(packed, libc::MAP_FAILED);
                    backing.sync_cpu(false).unwrap();
                    for i in 0..SIZE / 4 {
                        packed
                            .cast::<u32>()
                            .add(i)
                            .write_volatile((i as u32) ^ iteration);
                    }
                    backing.sync_cpu(true).unwrap();
                    let mut guest =
                        GuestRamImport::new_host_allocation(packed as usize, SIZE as u64, 4096)
                            .unwrap();
                    guest.attach_dmabuf(Arc::clone(&backing)).unwrap();
                    let mut imports = super::super::host_ram::HostRamImports::default();
                    let guest = Arc::new(guest);
                    let reference = reims_vgpu_memory::GuestRef::new(
                        Arc::clone(&guest),
                        guest.slice(0, SIZE as u64).unwrap(),
                    )
                    .unwrap();
                    let bound = imports.bind(ctx, &reference).unwrap();
                    assert!(
                        ctx.dmabuf_rail.has_live(),
                        "actual host-pointer refusal must take the new DMA-BUF path"
                    );
                    // The fixture owns its mmap; suppress HostOps alias-release routing only.
                    let staging = ctx
                        .device
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size(SIZE as u64)
                                .usage(vk::BufferUsageFlags::TRANSFER_DST),
                            None,
                        )
                        .unwrap();
                    let req = ctx.device.get_buffer_memory_requirements(staging);
                    let mt = (0..ctx.memory_properties.memory_type_count)
                        .find(|&i| {
                            req.memory_type_bits & (1 << i) != 0
                                && ctx.memory_properties.memory_types[i as usize]
                                    .property_flags
                                    .contains(
                                        vk::MemoryPropertyFlags::HOST_VISIBLE
                                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                                    )
                        })
                        .unwrap();
                    let memory = ctx
                        .device
                        .allocate_memory(
                            &vk::MemoryAllocateInfo::default()
                                .allocation_size(req.size)
                                .memory_type_index(mt),
                            None,
                        )
                        .unwrap();
                    ctx.device.bind_buffer_memory(staging, memory, 0).unwrap();
                    let cpu = ctx
                        .device
                        .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
                        .unwrap()
                        .cast::<u32>();
                    let pool = ctx
                        .device
                        .create_command_pool(
                            &vk::CommandPoolCreateInfo::default().queue_family_index(ctx.gq),
                            None,
                        )
                        .unwrap();
                    let cmd = ctx
                        .device
                        .allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_pool(pool)
                                .command_buffer_count(1),
                        )
                        .unwrap()[0];
                    ctx.device
                        .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                        .unwrap();
                    ctx.device.cmd_copy_buffer(
                        cmd,
                        bound.buffer,
                        staging,
                        &[vk::BufferCopy::default()
                            .src_offset(bound.offset)
                            .size(SIZE as u64)],
                    );
                    ctx.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[],
                    );
                    ctx.device.cmd_fill_buffer(
                        cmd,
                        bound.buffer,
                        bound.offset,
                        SIZE as u64,
                        0x12340000 | iteration,
                    );
                    let readback = vk::BufferMemoryBarrier::default()
                        .buffer(staging)
                        .size(SIZE as u64)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ);
                    ctx.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::HOST,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[readback],
                        &[],
                    );
                    ctx.device.end_command_buffer(cmd).unwrap();
                    let fence = ctx
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .unwrap();
                    ctx.submit_guest_work(&[cmd], fence).unwrap();
                    ctx.device
                        .wait_for_fences(&[fence], true, 5_000_000_000)
                        .unwrap();
                    for i in 0..SIZE / 4 {
                        assert_eq!(cpu.add(i).read_volatile(), (i as u32) ^ iteration);
                    }
                    backing.sync_cpu(false).unwrap();
                    for p in 0..total / 4096 {
                        let expected = if order.contains(&p) {
                            0x12340000 | iteration
                        } else {
                            0xa7a7a7a7
                        };
                        for word in 0..1024 {
                            assert_eq!(
                                original.cast::<u32>().add(p * 1024 + word).read_volatile(),
                                expected
                            );
                        }
                    }
                    backing.sync_cpu(true).unwrap();
                    ctx.device.destroy_fence(fence, None);
                    ctx.device.destroy_command_pool(pool, None);
                    ctx.device.unmap_memory(memory);
                    ctx.device.destroy_buffer(staging, None);
                    ctx.device.free_memory(memory, None);
                    guest.retire();
                    let super::super::host_ram::ParentRetire::Ready(mut allocation) =
                        imports.retire(guest.id())
                    else {
                        panic!("no child or pending work");
                    };
                    allocation.alias = None;
                    allocation.destroy(&ctx.device);
                    assert!(
                        !ctx.dmabuf_rail.has_live(),
                        "retired resource must not appear in later submissions"
                    );
                    assert!(
                        imports.bind(ctx, &reference).is_err(),
                        "retired identity cannot resurrect"
                    );
                    assert_eq!(libc::munmap(packed, SIZE), 0);
                    assert_eq!(libc::munmap(original, total), 0);
                }
            }
        }
    }
}
