//! The build of a file on Windows ([`file_build`]), which names the build of the library that runs
//! (`super::library_build`).

use std::fs::File;
use std::os::windows::io::AsRawHandle as _;
use std::path::Path;

use windows_sys::Win32::Storage::FileSystem::{
    FILE_BASIC_INFO, FILE_ID_INFO, FILE_INFO_BY_HANDLE_CLASS, FileBasicInfo, FileIdInfo,
    GetFileInformationByHandleEx,
};

/// The file at `path`, as a build: its path with every link resolved, its volume's serial number
/// and its file ID (128 bits, the two halves), its size, and its creation, last write and change
/// times in the 100 ns the filesystem keeps: seven words after the path, as on Unix. A file
/// written again in place differs in its change time; one renamed over it differs in its ID too.
pub(super) fn file_build(path: &Path) -> Option<Vec<u8>> {
    let path = std::fs::canonicalize(path).ok()?;
    let file = File::open(&path).ok()?;
    let len = file.metadata().ok()?.len();
    // SAFETY: each record is its class's, plain data.
    let (id, times) = unsafe {
        (
            information::<FILE_ID_INFO>(&file, FileIdInfo)?,
            information::<FILE_BASIC_INFO>(&file, FileBasicInfo)?,
        )
    };
    let (low, high) = id.FileId.Identifier.split_at(8);
    let half = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap_or_default());
    let mut build = path.as_os_str().as_encoded_bytes().to_vec();
    for word in [id.VolumeSerialNumber, half(low), half(high), len] {
        build.extend_from_slice(&word.to_le_bytes());
    }
    for word in [times.CreationTime, times.LastWriteTime, times.ChangeTime] {
        build.extend_from_slice(&word.to_le_bytes());
    }
    Some(build)
}

/// The open `file`'s information of `class`, whose record is a `T`; `None` where Windows gives
/// none.
///
/// # Safety
///
/// `T` is `class`'s record, plain data that any bytes Windows writes make a valid one.
unsafe fn information<T: Default>(file: &File, class: FILE_INFO_BY_HANDLE_CLASS) -> Option<T> {
    let mut info = T::default();
    let size = u32::try_from(size_of::<T>()).ok()?;
    // SAFETY: the open file's handle, and a record of `size` bytes, `class`'s own as the caller
    // promises, to fill.
    let got = unsafe {
        GetFileInformationByHandleEx(file.as_raw_handle(), class, (&raw mut info).cast(), size)
    };
    (got != 0).then_some(info)
}

/// A symbolic link at `link` to the file `original`, for the tests (Windows makes one for a
/// process without the privilege where Developer Mode is on).
#[cfg(test)]
pub(super) fn symlink(original: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(original, link)
}
