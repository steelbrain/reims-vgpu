//! OS-specific RAM file ownership, behind the unchanged descriptor-based C ABI.
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;
#[cfg(unix)]
pub(super) use unix::{duplicate, file_size, host_page_size};
#[cfg(windows)]
pub(super) use windows::{duplicate, file_size, host_page_size};
