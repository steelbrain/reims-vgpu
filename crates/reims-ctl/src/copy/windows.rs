//! [`super`]'s file system calls on Windows.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// The errors of a refused clone between two volumes: ERROR_NOT_SAME_DEVICE.
pub(super) const APART: &[i32] = &[17];

/// The errors of a refused clone on a file system that makes none: ERROR_INVALID_FUNCTION and
/// ERROR_NOT_SUPPORTED.
pub(super) const UNSUPPORTED: &[i32] = &[1, 50];

/// `dst`, made empty.
pub(super) fn create(_from: &File, dst: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(dst)
}

/// Makes `dst` a clone of `src`: none on Windows yet, which refuses as a file system that makes
/// none.
pub(super) fn clone(_src: &Path, _dst: &Path) -> io::Result<()> {
    Err(io::Error::from_raw_os_error(UNSUPPORTED[1]))
}

/// The parts of `f`, `len` bytes long, that hold data: all of it.
pub(super) fn data_ranges(_f: &File, len: u64) -> Vec<(u64, u64)> {
    if len == 0 { Vec::new() } else { vec![(0, len)] }
}

/// The type of the file system `dir` is on: NTFS, ReFS, exFAT.
pub(super) fn file_system(dir: &Path) -> Option<String> {
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, GetVolumeInformationByHandleW,
    };
    // A directory opens with backup semantics, for no access but its attributes.
    let d = OpenOptions::new()
        .access_mode(0)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(dir)
        .ok()?;
    let mut name = [0u16; 64];
    // SAFETY: d is open for the call, name holds its length in UTF-16 units, and the other
    // buffers are left out.
    let ok = unsafe {
        GetVolumeInformationByHandleW(
            d.as_raw_handle(),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            name.as_mut_ptr(),
            name.len() as u32,
        )
    } != 0;
    let len = name.iter().position(|&c| c == 0)?;
    ok.then(|| String::from_utf16_lossy(&name[..len]))
}
