//! What the restore asks the host's OS where Unix (`unix.rs`) and Windows (`windows.rs`) differ:
//! the space free on a file system, and the space a file takes on it. Positioned reads and writes
//! are reims-apfs's [`reims_apfs::file_at::FileAt`].

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;
