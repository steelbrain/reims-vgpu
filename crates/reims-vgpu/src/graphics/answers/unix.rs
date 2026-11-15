//! The build of a file on Linux and macOS ([`file_build`]), which names the build of the library
//! that runs (`super::library_build`).

use std::path::Path;

/// The file at `path`, as a build: its path with every link resolved (each of reims-ctl's QEMU
/// installs is a directory of its own, named for its time and commit, which `qemu/current` links
/// to), its device and inode, its size, and its modification and change times to the nanosecond
/// the filesystem keeps. A file written again in place differs in its change time, which no copy
/// or archive keeps; one renamed over it differs in its inode too.
pub(super) fn file_build(path: &Path) -> Option<Vec<u8>> {
    use std::os::unix::fs::MetadataExt;
    let path = std::fs::canonicalize(path).ok()?;
    let meta = std::fs::metadata(&path).ok()?;
    let mut build = path.as_os_str().as_encoded_bytes().to_vec();
    for word in [meta.dev(), meta.ino(), meta.len()] {
        build.extend_from_slice(&word.to_le_bytes());
    }
    for word in [
        meta.mtime(),
        meta.mtime_nsec(),
        meta.ctime(),
        meta.ctime_nsec(),
    ] {
        build.extend_from_slice(&word.to_le_bytes());
    }
    Some(build)
}

/// A symbolic link at `link` to `original`, for the tests.
#[cfg(test)]
pub(super) fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(original, link)
}
