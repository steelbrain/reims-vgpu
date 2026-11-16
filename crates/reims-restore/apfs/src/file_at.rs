//! Reads and writes at an offset of a file, which reims-apfs and reims-restore take files by:
//! [`FileAt`], std's `FileExt` on Unix (`pread`, `pwrite`), and the same four methods on Windows
//! over std's `seek_read` and `seek_write` (`windows.rs`), which also move the file's cursor
//! there, so a file read at offsets is never read from its cursor too.

#[cfg(unix)]
pub use std::os::unix::fs::FileExt as FileAt;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::FileAt;
