//! The guest's RAM a device holds on Windows: each range of it registered in the backing table
//! (`crate::os::files`) as a view of the pagefile-backed section the host names, so views of the
//! guest's pages map that section ([`super::vm`]), and handed whole to the devices that take it
//! (`crate::os`'s `host_ram`), while the device lives.

use std::sync::Arc;

use crate::os::GuestRamFile;
use crate::os::files::{Backing, Registration};

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these sections.
#[derive(Debug)]
pub struct GuestMemoryLease {
    /// The RAM as the devices that take it whole are handed it (`crate::os`'s `host_ram`), let go
    /// of first.
    _ram: Arc<crate::os::host_ram::LeasedRam>,
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a section behind in the backing table
/// (`crate::os::files`'s `Registration::guest_ram`): counted where the same range was registered
/// already over the same section handle (`crate::os::files`'s module docs).
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let ranges: Vec<GuestRamFile> = ranges.into_iter().collect();
    let ram: Vec<(u64, u64)> = ranges.iter().map(|r| (r.host, r.length)).collect();
    let registrations = ranges
        .into_iter()
        .map(|r| {
            let backing = Backing {
                file: r.file,
                offset: r.offset,
            };
            Registration::guest_ram(r.host, r.length, backing)
        })
        .collect();
    GuestMemoryLease {
        _ram: crate::os::host_ram::lease(&ram),
        _registrations: registrations,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::core::num::{ToU64 as _, ToUsize as _};
    use crate::core::platform::os::vm::{HostAddr, HostVm as _, KernReturn};
    use crate::os::files::pieces;

    use super::super::WindowsVm;
    use super::super::vm::{page_size, section};

    /// The guest's RAM is registered while the lease is held: a view of its pages shows them both
    /// ways; the lease's drop forgets it
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        use std::os::windows::io::AsRawHandle as _;
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        let p = page_size();
        let file = section(4 * p).unwrap();
        // The host's own mapping of its RAM, as QEMU maps it.
        // SAFETY: a whole view of a section the test holds open.
        let ram = unsafe { MapViewOfFile(file.as_raw_handle(), FILE_MAP_WRITE, 0, 0, 0) };
        let host = ram.Value.addr().to_u64();
        assert_ne!(host, 0);
        let lease = lease_guest_memory([GuestRamFile {
            host,
            guest: 0,
            length: 4 * p,
            file: Arc::clone(&file),
            offset: 0,
        }]);
        assert!(pieces(host, 4 * p).is_some());
        let v = WindowsVm::new();
        let view = v.allocate_anywhere(p).unwrap();
        let guest = HostAddr::from_addr((host + 2 * p).to_usize());
        assert_eq!(v.remap_shared_overwrite(view, guest, p), Ok(view));
        // SAFETY: the guest's page, mapped writable by the host, and the view of it.
        unsafe {
            guest.as_ptr::<u8>().write_volatile(0x61);
            assert_eq!(view.as_ptr::<u8>().read_volatile(), 0x61);
        }
        assert_eq!(v.deallocate(guest, p), KernReturn(4), "registered");
        assert_eq!(v.deallocate(view, p), KernReturn::SUCCESS);
        drop(lease);
        assert!(pieces(host, p).is_none());
        // SAFETY: the host's mapping, unmapped once.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: ram.Value }) };
    }
}
