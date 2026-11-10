//! Windows OS services for the ported Apple paravirtual GPU.
//! The same IOSurface metadata and lifetime implementation is shared with Linux; backing
//! pages use Windows section views, never Linux file descriptors or copied guest RAM.

mod clock;
mod failure;
mod import;
mod mapping;
mod reservation;
#[path = "standin.rs"]
mod surface;
mod threads;
mod vm;

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use std::ffi::CStr;
use std::ptr::NonNull;
use std::sync::Arc;

pub use clock::WindowsClock as OsClock;
pub use surface::{StandInSurface as OsSurface, StandInSurfaces as OsSurfaces};
pub(crate) use surface::{
    base_views_pages, own_alias, plane_at, plane_shape, recorded_mut, surface_memory_hold,
    ycbcr_matrix,
};
pub use threads::WindowsThreads as OsThreads;
pub use vm::WindowsVm as OsVm;
pub(crate) use vm::{OwnMemory, allocate_aliasable};

/// Windows uses Win32 WSI, selected before a native window is created.
pub(crate) const WINDOW_SURFACE_EXTENSIONS: [&CStr; 1] = [ash::khr::win32_surface::NAME];
pub(crate) const WINDOW_DEVICE_EXTENSIONS: [&CStr; 0] = [];

pub(crate) fn open_loader() -> Result<ash::Entry, ash::LoadingError> {
    // SAFETY: ash retains the loaded DLL while its entry points are in use.
    unsafe { ash::Entry::load() }
}

/// Vulkan owns an acquired Win32 swapchain image; no Metal drawable fetch is needed.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Drawables {}
impl Drawables {
    pub(crate) fn load(_device: &rt::Device) -> Option<Self> {
        None
    }
    pub(crate) unsafe fn fetch(&self, _image: ash::vk::Image) -> bool {
        match *self {}
    }
}

/// Windows has no autorelease pool or per-process Unix descriptor limit to install.
pub fn install_process_services() {}
/// The actual Vulkan device supplies identity and capabilities; no fabricated Metal identity.
pub fn default_registry_id() -> Option<u64> {
    None
}
/// Windows exposes no portable GPU core count; Vulkan still supplies device capabilities.
pub fn default_gpu_core_count() -> Option<u64> {
    None
}

/// The device owns its RAM registrations until all GPU use has ended.
#[derive(Debug)]
pub struct GuestMemoryLease {
    ids: Vec<u64>,
}
impl Drop for GuestMemoryLease {
    fn drop(&mut self) {
        for &id in &self.ids {
            vm::unregister_guest(id);
        }
    }
}

/// Register the device's live shared RAM files without changing their mappings or bytes.
pub fn lease_guest_memory(
    ranges: impl IntoIterator<Item = super::GuestRamFile>,
    _dirty: Option<Arc<reims_vgpu_core::memory::dirty::DirtyLog>>,
) -> GuestMemoryLease {
    let ids = ranges
        .into_iter()
        .map(|range| {
            vm::register_guest(
                range.host as usize,
                range.length as usize,
                &range.fd,
                range.offset as usize,
            )
            .unwrap_or_else(|error| {
                failure::failed(error, "Windows could not register the guest RAM section")
            })
        })
        .collect();
    GuestMemoryLease { ids }
}

pub(crate) use import::{ImportView, TWINS_READ_ONLY, import_view, shown_at};
/// The caller keeps the bytes mapped until the import's deallocator/keep-alive is released.
pub(crate) unsafe fn host_memory(at: NonNull<u8>, _logged: bool) -> rt::HostMemory {
    // SAFETY: the caller supplies HostMemory's complete memory lifetime contract.
    unsafe { rt::HostMemory::new(at) }
}
pub(crate) fn dma_buf_refusal() -> Option<String> {
    None
}
pub(crate) fn allocate_zeros(length: u64) -> Result<HostAddr, KernReturn> {
    allocate_aliasable(length)
}

/// Win32 MEMORYSTATUSEX; IOSurface allocations are bounded by actual physical RAM.
#[repr(C)]
struct MemoryStatus {
    size: u32,
    load: u32,
    total_physical: u64,
    available_physical: u64,
    total_pagefile: u64,
    available_pagefile: u64,
    total_virtual: u64,
    available_virtual: u64,
    available_extended: u64,
}
#[link(name = "kernel32")]
unsafe extern "system" {
    fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
}
fn physical_memory() -> u64 {
    let mut status = MemoryStatus {
        size: size_of::<MemoryStatus>() as u32,
        load: 0,
        total_physical: 0,
        available_physical: 0,
        total_pagefile: 0,
        available_pagefile: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended: 0,
    };
    // SAFETY: status has the documented Win32 layout and its initialized size field.
    assert_ne!(
        unsafe { GlobalMemoryStatusEx(&mut status) },
        0,
        "physical RAM query failed"
    );
    status.total_physical
}
