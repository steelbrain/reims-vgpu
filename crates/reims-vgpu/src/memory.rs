//! [`MemoryMap`]: the guest's RAM as the host maps it, which a graphics device is made over
//! ([`GraphicsConfig::memory_map`](crate::GraphicsConfig::memory_map)), with the file behind each
//! range where the host names one, and the platform's lease on it ([`GuestMemoryLease`]).

use std::any::Any;
use std::sync::Arc;

use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::memory::descriptor::{
    GuestPhysicalRange, MemoryMapDescriptor as CoreMemoryMap,
};
pub use reims_vgpu_vulkan::os::HostFile;

/// Guest RAM and the host memory behind it, in order, with the file behind each range where the
/// host names one. The ranges are copied into the device at its creation; nothing is checked
/// until then.
#[derive(Clone, Default, Debug)]
pub struct MemoryMap {
    ranges: Vec<GuestPhysicalRange>,
    /// One entry per range, in the ranges' order.
    files: Vec<Option<MemoryFile>>,
}

/// The file behind one range of a [`MemoryMap`]: the range's host memory is a shared mapping
/// (`MAP_SHARED`) of `file` from byte `offset` on Linux and macOS, and on Windows a view
/// (`MapViewOfFile`) of the pagefile-backed section `file` names from that byte.
#[derive(Clone, Debug)]
pub struct MemoryFile {
    /// The file, held open while any map or device holds it: a descriptor on Linux and macOS
    /// ([`HostFile`]), a section's handle on Windows.
    pub file: Arc<HostFile>,
    /// The file offset of the range's first byte, a multiple of the host's page size.
    pub offset: u64,
}

impl MemoryMap {
    /// An empty map.
    pub fn new() -> MemoryMap {
        MemoryMap::default()
    }

    /// Appends one range with no file, as given (no check, no merge).
    ///
    /// # Safety
    /// As [`MemoryMap::push_with_file`].
    pub unsafe fn push(&mut self, guest_physical_address: u64, length: u64, host: *mut u8) {
        // SAFETY: the caller's contract.
        unsafe { self.push_with_file(guest_physical_address, length, host, None) }
    }

    /// Appends one range and the file behind it, if any, as given (no check, no merge).
    ///
    /// # Safety
    /// `host .. host + length` is host memory (`mach_vm_allocate`d or equivalent) that stays
    /// mapped, and is not unmapped, mapped anew or discarded, while any graphics device made over
    /// the map lives and after it, for as long as the platform's imports of it may live: on the
    /// Vulkan platform, whose runtime frees them on a thread of its own, until the process exits.
    /// The device remaps it and reads and writes the guest's memory through it. With `file`, that
    /// memory is a shared mapping (on Windows a view) of `file.file` from `file.offset`, and the
    /// file holds at least `length` bytes from there.
    pub unsafe fn push_with_file(
        &mut self,
        guest_physical_address: u64,
        length: u64,
        host: *mut u8,
        file: Option<MemoryFile>,
    ) {
        self.ranges.push(GuestPhysicalRange {
            physical_address: guest_physical_address,
            physical_length: length,
            host_address: HostPtr::from_ptr(host),
        });
        self.files.push(file);
    }

    /// The ranges, in order.
    pub fn ranges(&self) -> &[GuestPhysicalRange] {
        &self.ranges
    }

    /// The file behind each range, in the ranges' order: `None` for a range with no file.
    pub fn files(&self) -> &[Option<MemoryFile>] {
        &self.files
    }

    /// Core's descriptor of the map.
    pub(crate) fn to_core(&self) -> CoreMemoryMap {
        let mut m = CoreMemoryMap::new();
        for r in &self.ranges {
            m.add_range(*r);
        }
        m
    }
}

/// The platform's lease on the guest's RAM for one graphics device
/// ([`ApiPlatform::guest_memory`](crate::ApiPlatform::guest_memory)), held by the views of the
/// surfaces' pages made through the map.
pub(crate) type GuestMemoryLease = Arc<dyn Any + Send + Sync>;
