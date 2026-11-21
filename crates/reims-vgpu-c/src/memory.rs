//! The guest's RAM as the host maps it, which the graphics device's config names
//! (`RvgGraphicsDeviceConfig.memory_map`): [`RvgMemoryMap`], its [`RvgMemoryRange`]s, and the
//! [`RvgMemoryRangeFile`] behind each range: a file on Linux and macOS, and on Windows a
//! pagefile-backed section, checked and duplicated there by `windows.rs`.

use std::ffi::c_void;
#[cfg(not(windows))]
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
#[cfg(not(windows))]
use std::os::fd::FromRawFd;
use std::sync::Arc;

use reims_vgpu::{HostFile, MemoryFile, MemoryMap};

use crate::status::{RVG_ERR_INVALID_ARGUMENT, RvgStatus};

#[cfg(windows)]
mod windows;

/// One range of guest physical memory and the host memory behind it.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RvgMemoryRange {
    pub physical_address: u64,
    pub physical_length: u64,
    /// `physical_length` bytes, mapped (`mach_vm_allocate`d or equivalent), and not unmapped,
    /// mapped anew or discarded, until the process exits: the host GPU's imports of them, which
    /// the device makes, can outlive the device's release.
    pub host_address: *mut c_void,
}

/// The guest's RAM, as the host maps it. The ranges are copied at the create call, in order.
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RvgMemoryMap {
    pub ranges: *const RvgMemoryRange,
    pub range_count: usize,
}

/// The file behind one range of a memory map: the range's host memory is a shared mapping
/// (MAP_SHARED) of `fd` from byte `offset`. Plain data.
#[cfg(not(windows))]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvgMemoryRangeFile {
    /// An open file descriptor, which the library duplicates during the create (the host may
    /// close its own afterwards), or -1 for a range with no file.
    pub fd: i32,
    /// 0; other values are refused.
    pub flags: u32,
    /// The file offset of the range's first byte, a multiple of the host's page size.
    pub offset: u64,
}

#[cfg(not(windows))]
const _: () = assert!(size_of::<RvgMemoryRangeFile>() == 16);

/// The section behind one range of a memory map, on Windows: the range's host memory is a view
/// (MapViewOfFile) of the pagefile-backed section `handle` names (CreateFileMapping of
/// INVALID_HANDLE_VALUE), from byte `offset`. Plain data.
#[cfg(windows)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RvgMemoryRangeFile {
    /// A handle to the section, which the library may map for reading and writing
    /// (FILE_MAP_WRITE) and duplicates during the create (the host may close its own
    /// afterwards), or NULL for a range with no section.
    pub handle: *mut c_void,
    /// 0; other values are refused.
    pub flags: u32,
    /// The section offset of the range's first byte, a multiple of the host's page size.
    pub offset: u64,
}

#[cfg(windows)]
const _: () = assert!(size_of::<RvgMemoryRangeFile>() == 24);

/// The Rust API's map of a host's `RvgMemoryMap`, with the files behind its ranges
/// (`RvgGraphicsDeviceConfig.memory_files`, NULL for none; sections on Windows).
/// `Err(RVG_ERR_INVALID_ARGUMENT)`, in this order: ranges that are NULL but counted; files
/// without a map; a file entry [`file_refusal`] refuses, the first range's first; a file that
/// cannot be duplicated. Each
/// file is duplicated only once every entry has passed, so a refusal before then touches no
/// file. A refusal of the files logs a line saying which.
///
/// # Safety
/// `map` is NULL or a live map whose `ranges` hold `range_count` ranges; `files` is NULL or
/// holds `range_count` entries.
pub(crate) unsafe fn memory_map(
    map: *const RvgMemoryMap,
    files: *const RvgMemoryRangeFile,
) -> Result<Option<MemoryMap>, RvgStatus> {
    // SAFETY: the caller's contract.
    let Some(map) = (unsafe { map.as_ref() }) else {
        if !files.is_null() {
            log::warn!("device not created: memory files without a memory map");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
        return Ok(None);
    };
    let ranges: &[RvgMemoryRange] = if map.range_count == 0 {
        &[]
    } else if map.ranges.is_null() {
        return Err(RVG_ERR_INVALID_ARGUMENT);
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(map.ranges, map.range_count) }
    };
    let files: &[RvgMemoryRangeFile] = if files.is_null() || ranges.is_empty() {
        &[]
    } else {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts(files, ranges.len()) }
    };
    let page = host_page_size();
    for (i, (r, f)) in ranges.iter().zip(files).enumerate() {
        if let Some(why) = file_refusal(r, f, page) {
            log::warn!("device not created: the file of memory range {i} {why}");
            return Err(RVG_ERR_INVALID_ARGUMENT);
        }
    }
    let mut m = MemoryMap::new();
    for (i, r) in ranges.iter().enumerate() {
        let file = match files.get(i).filter(|f| f.names_a_file()) {
            Some(f) => match f.duplicate() {
                Ok(file) => Some(MemoryFile {
                    file: Arc::new(file),
                    offset: f.offset,
                }),
                Err(e) => {
                    log::warn!(
                        "device not created: the file of memory range {i}, {}, was not \
                         duplicated: {e}",
                        f.name()
                    );
                    return Err(RVG_ERR_INVALID_ARGUMENT);
                }
            },
            None => None,
        };
        // SAFETY: the header's contract on `host_address`: mapped, and not unmapped or mapped
        // anew, until the process exits, and, with a file, a shared mapping of it from `offset`.
        unsafe {
            m.push_with_file(
                r.physical_address,
                r.physical_length,
                r.host_address.cast(),
                file,
            );
        }
    }
    Ok(Some(m))
}

/// Why the file entry `f` of range `r` is refused, as the end of a log line, or `None`: an `fd`
/// below -1, `flags` other than 0, and for an `fd` of 0 or more, a file `fstat` cannot read, one
/// holding fewer than `offset + physical_length` bytes, or an `offset` that is not a multiple of
/// `page`, in that order.
#[cfg(not(windows))]
fn file_refusal(r: &RvgMemoryRange, f: &RvgMemoryRangeFile, page: NonZeroU64) -> Option<String> {
    if f.fd < -1 {
        return Some(format!("has fd {}, below -1", f.fd));
    }
    if f.flags != 0 {
        return Some(format!("has flags {:#x}, not 0", f.flags));
    }
    if f.fd == -1 {
        return None;
    }
    let size = match file_size(f.fd) {
        Ok(size) => size,
        Err(e) => return Some(format!("has fd {}, which fstat cannot read: {e}", f.fd)),
    };
    if f.offset
        .checked_add(r.physical_length)
        .is_none_or(|end| size < end)
    {
        return Some(format!(
            "holds {size:#x} bytes, fewer than its offset {:#x} and the range's {:#x}",
            f.offset, r.physical_length
        ));
    }
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    None
}

/// Why the section entry `f` of range `r` is refused on Windows, as the end of a log line, or
/// `None`: `flags` other than 0, and for a `handle` that is not NULL, one of which no view the
/// library may write maps `offset + physical_length` bytes (a handle that names no section, a
/// section it may not map so, or one holding fewer bytes: `windows::section_holds`), or an
/// `offset` that is not a multiple of `page`, in that order.
#[cfg(windows)]
fn file_refusal(r: &RvgMemoryRange, f: &RvgMemoryRangeFile, page: NonZeroU64) -> Option<String> {
    if f.flags != 0 {
        return Some(format!("has flags {:#x}, not 0", f.flags));
    }
    if f.handle.is_null() {
        return None;
    }
    if let Err(e) = windows::section_holds(f.handle, f.offset, r.physical_length) {
        return Some(format!(
            "has handle {:p}, of which no writable view of its offset {:#x} and the range's \
             {:#x} bytes maps: {e}",
            f.handle, f.offset, r.physical_length
        ));
    }
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    None
}

#[cfg(not(windows))]
impl RvgMemoryRangeFile {
    /// Whether the entry names a file: an `fd` of 0 or more ([`file_refusal`] refuses one below
    /// -1).
    fn names_a_file(&self) -> bool {
        self.fd >= 0
    }

    /// The file, as a log line names it.
    fn name(&self) -> String {
        format!("fd {}", self.fd)
    }

    /// A duplicate of the file, which the map holds ([`duplicate`]).
    fn duplicate(&self) -> std::io::Result<HostFile> {
        duplicate(self.fd)
    }
}

#[cfg(windows)]
impl RvgMemoryRangeFile {
    /// Whether the entry names a section: a `handle` that is not NULL.
    fn names_a_file(&self) -> bool {
        !self.handle.is_null()
    }

    /// The section, as a log line names it.
    fn name(&self) -> String {
        format!("handle {:p}", self.handle)
    }

    /// A duplicate of the section's handle, which the map holds (`windows::duplicate`).
    fn duplicate(&self) -> std::io::Result<HostFile> {
        windows::duplicate(self.handle)
    }
}

/// The host's page size on Windows (`windows::host_page_size`): 4 KiB on x86-64.
#[cfg(windows)]
fn host_page_size() -> NonZeroU64 {
    windows::host_page_size()
}

/// The host's page size (`sysconf(_SC_PAGESIZE)`, which does not fail): 16 KiB on arm64 macOS,
/// 4 KiB on x86-64.
#[cfg(not(windows))]
fn host_page_size() -> NonZeroU64 {
    // SAFETY: a query of a constant; it reads and writes no memory of the caller's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page)
        .ok()
        .and_then(NonZeroU64::new)
        .unwrap_or(NonZeroU64::MIN)
}

/// The size of the file `fd` names (`fstat`).
#[cfg(not(windows))]
fn file_size(fd: i32) -> std::io::Result<u64> {
    let mut st = MaybeUninit::<libc::stat>::zeroed();
    // SAFETY: `st` is a writable `struct stat`; an `fd` that names no open file is an error the
    // call returns.
    if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fstat` succeeded, so it filled `st`; it was zeroed before.
    let st = unsafe { st.assume_init() };
    Ok(u64::try_from(st.st_size).unwrap_or(0))
}

/// A duplicate of `fd`, close-on-exec (`fcntl(F_DUPFD_CLOEXEC)`), which the map holds.
#[cfg(not(windows))]
fn duplicate(fd: i32) -> std::io::Result<HostFile> {
    // SAFETY: `fcntl` reads and writes no memory of the caller's; an `fd` that names no open
    // file is an error the call returns.
    let new = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a descriptor this call opened, owned by nothing else.
    Ok(unsafe { HostFile::from_raw_fd(new) })
}
