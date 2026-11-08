//! The guest's RAM as the host maps it, which the graphics device's config names
//! (`RvgGraphicsDeviceConfig.memory_map`): [`RvgMemoryMap`], its [`RvgMemoryRange`]s, and the
//! [`RvgMemoryRangeFile`] behind each range.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::Arc;

use reims_vgpu::{MemoryFile, MemoryMap};

use crate::status::{RVG_ERR_INVALID_ARGUMENT, RvgStatus};

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
    /// `physical_length` bytes, mapped (`mach_vm_allocate`d or equivalent) for as long as any
    /// device made over the map lives.
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

const _: () = assert!(size_of::<RvgMemoryRangeFile>() == 16);

/// The Rust API's map of a host's `RvgMemoryMap`, with the files behind its ranges
/// (`RvgGraphicsDeviceConfig.memory_files`, NULL for none). `Err(RVG_ERR_INVALID_ARGUMENT)`, in
/// this order: ranges that are NULL but counted; files without a map; a file entry
/// [`file_refusal`] refuses, the first range's first; a file that cannot be duplicated. Each
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
        let file = match files.get(i) {
            Some(f) if f.fd >= 0 => match duplicate(f.fd) {
                Ok(fd) => Some(MemoryFile {
                    fd: Arc::new(fd),
                    offset: f.offset,
                }),
                Err(e) => {
                    log::warn!(
                        "device not created: the file of memory range {i}, fd {}, was not \
                         duplicated: {e}",
                        f.fd
                    );
                    return Err(RVG_ERR_INVALID_ARGUMENT);
                }
            },
            _ => None,
        };
        // SAFETY: the header's contract on `host_address`: mapped while any device made over
        // the map lives, and, with a file, a shared mapping of it from `offset`.
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

/// The host's page size (`sysconf(_SC_PAGESIZE)`, which does not fail): 16 KiB on arm64 macOS,
/// 4 KiB on x86-64.
fn host_page_size() -> NonZeroU64 {
    // SAFETY: a query of a constant; it reads and writes no memory of the caller's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page)
        .ok()
        .and_then(NonZeroU64::new)
        .unwrap_or(NonZeroU64::MIN)
}

/// The size of the file `fd` names (`fstat`).
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
fn duplicate(fd: i32) -> std::io::Result<OwnedFd> {
    // SAFETY: `fcntl` reads and writes no memory of the caller's; an `fd` that names no open
    // file is an error the call returns.
    let new = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a descriptor this call opened, owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(new) })
}
