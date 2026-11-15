//! What reims-ctl does through the host's OS where macOS and Linux (`unix`) and Windows
//! (`windows`) differ: file modes and positioned reads, the local time, the user's private
//! directories, child processes, and the terminal. Each module gives the same items, which the
//! rest of reims-ctl calls as `crate::os::...`.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::*;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::*;

/// A way to end a process, by its pid, with what a stop says once it has ended (`ENDS`).
pub type End = (fn(i32), &'static str);

/// A local date and time, by the host's clock and time zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalTime {
    pub year: i32,
    /// 1 to 12.
    pub month: u32,
    /// 1 to 31.
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}
