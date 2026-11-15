//! The memory map's sections on Windows (`super`): a range's host memory there is a view of a
//! pagefile-backed section (`CreateFileMapping` of `INVALID_HANDLE_VALUE`), which the create
//! checks and duplicates as it does a file on Linux and macOS. It uses nothing of the crate's, so
//! a scratch crate that includes this file checks it alone.

use std::ffi::c_void;
use std::num::NonZeroU64;
use std::os::windows::io::{FromRawHandle as _, OwnedHandle};

use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// The system's page size and allocation granularity (`GetSystemInfo`, which does not fail):
/// 4 KiB and 64 KiB on x86-64.
fn system_info() -> (u64, u64) {
    // SAFETY: an all-zero `SYSTEM_INFO` is a valid value of the plain struct for the call to fill.
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    // SAFETY: `info` is a live struct to write.
    unsafe { GetSystemInfo(&mut info) };
    (
        u64::from(info.dwPageSize),
        u64::from(info.dwAllocationGranularity),
    )
}

/// The host's page size: 4 KiB on x86-64.
pub(super) fn host_page_size() -> NonZeroU64 {
    NonZeroU64::new(system_info().0).unwrap_or(NonZeroU64::MIN)
}

/// Whether the section `handle` names holds `length` bytes from `offset` that the library may map
/// for reading and writing: a view of them maps (`MapViewOfFile` with `FILE_MAP_WRITE`, from
/// `offset` rounded down to the allocation granularity, which a view's offset must be a multiple
/// of), and is unmapped at once. The view reserves address space only and touches no page. The
/// system's refusal otherwise: of a handle that names no section, of a section the handle may not
/// map so, or of one holding fewer bytes.
pub(super) fn section_holds(handle: *mut c_void, offset: u64, length: u64) -> std::io::Result<()> {
    let granule = system_info().1.max(1);
    let from = offset - offset % granule;
    let bytes = (offset - from)
        .checked_add(length)
        .and_then(|b| usize::try_from(b).ok())
        .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let high = u32::try_from(from >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(from & 0xffff_ffff).unwrap_or(0);
    // SAFETY: a view at an address the system picks of whatever `handle` names, or a failure.
    let view = unsafe { MapViewOfFile(handle, FILE_MAP_WRITE, high, low, bytes) };
    if view.Value.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the view just made, which nothing else knows.
    unsafe { UnmapViewOfFile(view) };
    Ok(())
}

/// A duplicate of `handle` in this process, with the access it has and not inherited by child
/// processes (`DuplicateHandle`), which the map holds.
pub(super) fn duplicate(handle: *mut c_void) -> std::io::Result<OwnedHandle> {
    let mut copy: HANDLE = std::ptr::null_mut();
    // SAFETY: duplicates whatever `handle` names within this process, or fails.
    let copied = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            handle,
            GetCurrentProcess(),
            &mut copy,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    };
    if copied == 0 || copy.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: the duplicate is new, and nothing else owns it.
    Ok(unsafe { OwnedHandle::from_raw_handle(copy) })
}

#[cfg(test)]
mod tests {
    use std::os::windows::io::AsRawHandle as _;

    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT,
    };

    use super::*;

    /// A new pagefile-backed section of `bytes`, committed, with the page protection `protect`.
    fn section(bytes: u64, protect: u32) -> OwnedHandle {
        let high = u32::try_from(bytes >> 32).unwrap();
        let low = u32::try_from(bytes & 0xffff_ffff).unwrap();
        // SAFETY: a new unnamed section of the paging file, with no security attributes.
        let made = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                protect | SEC_COMMIT,
                high,
                low,
                std::ptr::null(),
            )
        };
        assert!(!made.is_null(), "{}", std::io::Error::last_os_error());
        // SAFETY: the section just made, owned once.
        unsafe { OwnedHandle::from_raw_handle(made) }
    }

    /// A section holds the bytes it was made with from any page, past the allocation granularity
    /// too, and none past its end; a read-only section, and a handle of none, hold nothing the
    /// library may write
    #[test]
    fn a_section_holds_its_bytes_writable() {
        let p = host_page_size().get();
        let s = section(32 * p, PAGE_READWRITE);
        let h = s.as_raw_handle();
        assert!(section_holds(h, 0, 32 * p).is_ok());
        assert!(section_holds(h, 17 * p, 15 * p).is_ok());
        assert!(
            section_holds(h, 17 * p, 16 * p).is_err(),
            "a page past its end"
        );
        assert!(section_holds(h, 33 * p, 0).is_err(), "from past its end");
        let read_only = section(4 * p, PAGE_READONLY);
        assert!(section_holds(read_only.as_raw_handle(), 0, 4 * p).is_err());
        assert!(section_holds(std::ptr::null_mut(), 0, p).is_err());
    }

    /// A duplicate names the same section, and outlives the host's own handle; a handle of
    /// nothing is not duplicated
    #[test]
    fn a_duplicate_outlives_the_hosts_handle() {
        let p = host_page_size().get();
        let s = section(4 * p, PAGE_READWRITE);
        let copy = duplicate(s.as_raw_handle()).unwrap();
        drop(s);
        assert!(section_holds(copy.as_raw_handle(), 0, 4 * p).is_ok());
        assert!(duplicate(std::ptr::null_mut()).is_err());
    }
}
