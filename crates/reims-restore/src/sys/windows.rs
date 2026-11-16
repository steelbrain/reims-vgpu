//! [`super`] on Windows, where reims-ctl makes no arm64 VM (its host.rs), so no restore runs:
//! these answer as reims-ctl's code that the restore shares with x86-64 VMs needs them.

use anyhow::{Context, Result};
use std::os::windows::ffi::OsStrExt as _;
use std::path::Path;
use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

/// The space free to this user on the volume `dir`, which exists, is on.
pub fn free_space(dir: &Path) -> Result<u64> {
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut free = 0u64;
    // SAFETY: a NUL-terminated path, and one out-pointer the call fills; the others are null.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("GetDiskFreeSpaceExW {}", dir.display()));
    }
    Ok(free)
}

/// The space the file `m` describes takes on its volume, counted as its length: a sparse file's
/// holes count as written, so a check of room for its copy asks for more than the copy takes.
pub fn allocated(m: &std::fs::Metadata) -> u64 {
    m.len()
}
