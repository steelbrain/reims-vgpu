//! [`super`] on macOS and Linux.

use anyhow::{Context, Result};
use std::os::unix::fs::MetadataExt as _;
use std::path::Path;

/// The space free to this user on the file system `dir`, which exists, is on.
pub fn free_space(dir: &Path) -> Result<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes())?;
    // SAFETY: statvfs fills the struct it is given, from a NUL-terminated path.
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("statvfs {}", dir.display()));
    }
    #[allow(clippy::useless_conversion)]
    Ok(u64::from(s.f_bavail) * u64::from(s.f_frsize))
}

/// The space the file `m` describes takes on its file system: its 512-byte blocks, fewer than
/// its length where it is sparse.
pub fn allocated(m: &std::fs::Metadata) -> u64 {
    m.blocks() * 512
}
