//! [`super`]'s file system calls on macOS and Linux.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

/// The errors of a refused reflink between two file systems.
pub(super) const APART: &[i32] = &[libc::EXDEV];

/// The errors of a refused reflink on a file system that makes none.
pub(super) const UNSUPPORTED: &[i32] =
    &[libc::EOPNOTSUPP, libc::ENOTSUP, libc::ENOTTY, libc::ENOSYS];

/// `dst`, made empty, with the permissions of `from`, as `cp` makes its copy.
pub(super) fn create(from: &File, dst: &Path) -> io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(from.metadata()?.mode() & 0o7777)
        .open(dst)
}

/// Makes `dst` a reflink of `src`, or returns the file system's refusal.
#[cfg(target_os = "linux")]
pub(super) fn clone(src: &Path, dst: &Path) -> io::Result<()> {
    let from = File::open(src)?;
    let to = create(&from, dst)?;
    // SAFETY: FICLONE takes the source's descriptor; both are open for the call.
    if unsafe { libc::ioctl(to.as_raw_fd(), libc::FICLONE, from.as_raw_fd()) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Makes `dst` a clone of `src`, or returns the file system's refusal.
#[cfg(target_os = "macos")]
pub(super) fn clone(src: &Path, dst: &Path) -> io::Result<()> {
    let (s, d) = (c_path(src)?, c_path(dst)?);
    // SAFETY: both are NUL-terminated paths, alive for the call.
    if unsafe { libc::clonefile(s.as_ptr(), d.as_ptr(), 0) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(target_os = "macos")]
fn c_path(p: &Path) -> io::Result<std::ffi::CString> {
    use std::os::unix::ffi::OsStrExt;
    std::ffi::CString::new(p.as_os_str().as_bytes()).map_err(io::Error::other)
}

/// The parts of `f`, `len` bytes long, that hold data, as `[start, end)`: all of it where the file
/// system tells no holes.
pub(super) fn data_ranges(f: &File, len: u64) -> Vec<(u64, u64)> {
    let seek = |at: u64, whence| {
        // SAFETY: lseek on an open descriptor.
        let r = unsafe { libc::lseek(f.as_raw_fd(), at as libc::off_t, whence) };
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(r as u64)
        }
    };
    let mut ranges = Vec::new();
    let mut at = 0;
    while at < len {
        let start = match seek(at, libc::SEEK_DATA) {
            Ok(s) => s,
            // No data after `at`.
            Err(e) if e.raw_os_error() == Some(libc::ENXIO) => break,
            Err(_) => at,
        };
        let end = seek(start, libc::SEEK_HOLE).unwrap_or(len).min(len);
        if end <= start {
            break;
        }
        ranges.push((start, end));
        at = end;
    }
    ranges
}

/// The type of the file system `dir` is on: btrfs, ext4, tmpfs.
#[cfg(target_os = "linux")]
pub(super) fn file_system(dir: &Path) -> Option<String> {
    let mounts = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    super::mount_type(&mounts, &std::fs::canonicalize(dir).ok()?)
}

/// The type of the file system `dir` is on: apfs, hfs, exfat.
#[cfg(target_os = "macos")]
pub(super) fn file_system(dir: &Path) -> Option<String> {
    let p = c_path(dir).ok()?;
    // SAFETY: statfs fills the struct it is given; all zeros is a valid one.
    let mut s: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statfs(p.as_ptr(), &mut s) } != 0 {
        return None;
    }
    // SAFETY: f_fstypename is NUL-terminated.
    let name = unsafe { std::ffi::CStr::from_ptr(s.f_fstypename.as_ptr()) };
    Some(name.to_string_lossy().into_owned())
}
