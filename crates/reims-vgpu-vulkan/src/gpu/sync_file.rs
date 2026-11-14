//! A command buffer's completion as a sync file, and a wait for one on another device's GPU: the
//! platform's own `CommandBuffer::add_completion_sync_file`, `export_completion_sync_file` and
//! `encode_wait_for_sync_file`, which order the window's mirror between the runtime's GPU and the
//! compositor's (reims-vgpu's `graphics::mirror`).
//!
//! - **The export** is the runtime's (`rt::CommandBuffer::add_completion_sync_file` and
//!   `export_completion_sync_file`): a binary semaphore its command buffer's last batch signals,
//!   exported as `VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT`. A device that exports none
//!   (MoltenVK among them) refuses it (`RuntimeGap::SyncFileExport`), as any refusal is: the caller
//!   waits for the command buffer on the host.
//! - **The wait.** A binary semaphore of the waiting command buffer's device, made for the one
//!   wait, takes the file as a temporary import (`vkImportSemaphoreFdKHR` with
//!   `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`, which a sync file's import requires), and the command
//!   buffer waits for it before its work (`rt::CommandBuffer::encode_wait_for_vk_semaphore`).
//!   The semaphore goes once the command buffer has completed, or with the command buffer let go
//!   uncommitted (its completed handler holds it), destroyed through the generation of the device
//!   it was made on, never where that generation is parked (`crate::generation`). Refused, the
//!   file closed, where the device does not enable `VK_KHR_external_semaphore_fd`, its GPU imports
//!   no sync file (`vkGetPhysicalDeviceExternalSemaphoreProperties`), or a step fails.

use std::os::fd::{FromRawFd as _, IntoRawFd as _, OwnedFd};

use ash::vk;
use metal2vulkan_runtime as rt;

/// The semaphore of one wait, destroyed as it goes (the module doc).
struct Waited {
    /// The waiting command buffer's device: the generation the semaphore was made on.
    generation: rt::Device,
    /// That generation's `VkDevice`.
    handle: vk::Device,
    /// The semaphore.
    semaphore: vk::Semaphore,
}

impl Drop for Waited {
    fn drop(&mut self) {
        if self.generation.is_parked(self.handle) {
            return;
        }
        // SAFETY: the semaphore this generation made, which nothing waits for any more: its
        // command buffer completed, or went uncommitted; the handles are used while the
        // generation lives.
        unsafe {
            self.generation
                .raw()
                .device
                .destroy_semaphore(self.semaphore, None);
        }
    }
}

/// Whether the GPU of `handles` imports a sync file into a binary semaphore.
fn imports_sync_files(handles: &rt::DeviceHandles<'_>) -> bool {
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
    let mut properties = vk::ExternalSemaphoreProperties::default();
    // SAFETY: the physical device of the live instance; Vulkan 1.1's query, under the runtime's
    // floor of 1.2.
    unsafe {
        handles
            .instance
            .get_physical_device_external_semaphore_properties(
                handles.physical_device,
                &info,
                &mut properties,
            );
    }
    (properties.external_semaphore_features).contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE)
}

/// `command_buffer`, before any encoder, waits on its GPU for the sync file `file` (the module
/// doc): `false`, the file closed, where it is refused.
pub(crate) fn encode_wait(command_buffer: &rt::CommandBuffer, file: OwnedFd) -> bool {
    let generation = command_buffer.device().clone();
    if !generation.enables_extension(ash::khr::external_semaphore_fd::NAME) {
        return false;
    }
    // SAFETY: the handles are used while `generation` lives, here and in `Waited`.
    let handles = unsafe { generation.raw() };
    if !imports_sync_files(&handles) {
        return false;
    }
    let fns = ash::khr::external_semaphore_fd::Device::new(handles.instance, handles.device);
    let handle = handles.device.handle();
    // SAFETY: the live device.
    let Ok(semaphore) =
        (unsafe { (handles.device).create_semaphore(&vk::SemaphoreCreateInfo::default(), None) })
    else {
        return false;
    };
    let waited = Waited {
        generation: generation.clone(),
        handle,
        semaphore,
    };
    // The command buffer's completed handler holds the semaphore from here: it goes once the
    // command buffer completed, or with the command buffer let go uncommitted.
    if (command_buffer.add_completed_handler(Box::new(move |_| drop(waited)))).is_err() {
        return false;
    }
    let fd = file.into_raw_fd();
    let import = vk::ImportSemaphoreFdInfoKHR::default()
        .semaphore(semaphore)
        .flags(vk::SemaphoreImportFlags::TEMPORARY)
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
        .fd(fd);
    // SAFETY: a binary semaphore of the device, with no wait pending; an import that succeeds
    // takes the descriptor.
    if unsafe { fns.import_semaphore_fd(&import) }.is_err() {
        // SAFETY: the import failed, so the descriptor is still this call's.
        drop(unsafe { OwnedFd::from_raw_fd(fd) });
        return false;
    }
    // SAFETY: the semaphore is a binary semaphore of the command buffer's device, whose signal
    // the import gave it before the commit; no other wait consumes it; the command buffer's
    // completed handler keeps it until the command buffer completed.
    unsafe { command_buffer.encode_wait_for_vk_semaphore(semaphore) }.is_ok()
}

#[cfg(test)]
mod tests {
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::{CommandBufferStatus, NsRange};
    use crate::core::raise::OnFault;
    use crate::gpu::device::tests::dev;

    /// A command buffer's completion as a sync file, which a command buffer of another queue
    /// waits for on the GPU: where the runtime exports none (`RuntimeGap::SyncFileExport`,
    /// MoltenVK) the ask answers `false` and the export `None`, and the caller waits on the host;
    /// where it exports one, the waiting command buffer's copy reads what the first one filled
    #[test]
    fn a_completion_is_waited_for_on_the_gpu_or_refused() {
        let d = dev();
        let (q, r) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        let filled = d
            .new_buffer_of_length(OnFault::Return, 4096, 0)
            .unwrap()
            .unwrap();
        let copied = d
            .new_buffer_of_length(OnFault::Return, 4096, 0)
            .unwrap()
            .unwrap();
        let first = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = first.blit_encoder(OnFault::Return).unwrap().unwrap();
        blit.encode(
            OnFault::Return,
            BlitCmd::FillBuffer {
                buffer: &filled,
                range: NsRange {
                    location: 0,
                    length: 4096,
                },
                value: 0xab,
            },
        )
        .unwrap();
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        let asked = first.add_completion_sync_file();
        first.commit(OnFault::Return).unwrap();
        let exported = first.export_completion_sync_file();
        if !asked {
            assert!(exported.is_none(), "no sync file was asked for");
            first.wait_until_completed(OnFault::Return).unwrap();
            assert_eq!(first.status(), CommandBufferStatus::COMPLETED);
            return;
        }
        let second = r.command_buffer(OnFault::Return).unwrap().unwrap();
        match exported.expect("an asked sync file is exported") {
            Some(file) => assert!(second.encode_wait_for_sync_file(file), "the wait"),
            None => assert_eq!(first.status(), CommandBufferStatus::COMPLETED),
        }
        let blit = second.blit_encoder(OnFault::Return).unwrap().unwrap();
        blit.encode(
            OnFault::Return,
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&filled),
                source_offset: 0,
                destination_buffer: Some(&copied),
                destination_offset: 0,
                size: 4096,
            },
        )
        .unwrap();
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        second.commit(OnFault::Return).unwrap();
        second.wait_until_completed(OnFault::Return).unwrap();
        first.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(second.status(), CommandBufferStatus::COMPLETED);
        assert_eq!(first.status(), CommandBufferStatus::COMPLETED);
        // SAFETY: a shared buffer of 4096 bytes, which no command buffer writes any more.
        let bytes = unsafe { std::slice::from_raw_parts(copied.contents().as_ptr::<u8>(), 4096) };
        assert!(
            bytes.iter().all(|&b| b == 0xab),
            "the copy came after the fill"
        );
    }
}
