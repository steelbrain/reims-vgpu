//! The guest's RAM a device holds on Linux: each range of it registered in the backing table
//! (`crate::os::files`) as a shared mapping of the file the host names, and handed whole to the
//! devices that take it (`crate::os`'s `host_ram`), while the device lives.

use std::sync::Arc;

use crate::os::GuestRamFile;
use crate::os::files::{Backing, Registration};

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these files.
#[derive(Debug)]
pub struct GuestMemoryLease {
    /// The RAM as the devices that take it whole are handed it (`crate::os`'s `host_ram`), let go of
    /// first.
    _ram: Arc<crate::os::host_ram::LeasedRam>,
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a file behind in the backing table
/// (`crate::os::files`'s `Registration::guest_ram`): counted where the same range was registered
/// already over the same descriptor (`crate::os::files`'s module docs). Warns where the views'
/// budget of mappings is short of the guest's RAM (the `budget` module).
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let ranges: Vec<GuestRamFile> = ranges.into_iter().collect();
    let ram: Vec<(u64, u64)> = ranges.iter().map(|r| (r.host, r.length)).collect();
    let guest_ram = ranges
        .iter()
        .map(|r| r.host.saturating_add(r.length) - r.host)
        .fold(0u64, u64::saturating_add);
    let registrations: Vec<Registration> = ranges
        .into_iter()
        .map(|r| {
            let backing = Backing {
                file: r.file,
                offset: r.offset,
            };
            Registration::guest_ram(r.host, r.length, backing)
        })
        .collect();
    super::budget::warn_if_short(guest_ram);
    GuestMemoryLease {
        _ram: crate::os::host_ram::lease(&ram),
        _registrations: registrations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::files::pieces;
    use crate::os::files::tests::{file, shape};

    /// A lease registers each range of the guest's RAM at its file offset, until it drops
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        let f = file();
        let base = 0x7d00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            guest: offset,
            length,
            file: Arc::clone(&f),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x2000, 0x4000)]);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x1000).unwrap()),
            [(0, 0x1000, 0x1000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x11_000, 0x1000).unwrap()),
            [(0, 0x1000, 0x5000)]
        );
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
    }
}
