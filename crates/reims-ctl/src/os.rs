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

/// Whether the host has a sound output device and a recording device (`sound_devices`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SoundDevices {
    pub output: bool,
    pub input: bool,
}

/// A way to end a process, by its pid, with what a stop says once it has ended (`ENDS`).
pub type End = (fn(i32), &'static str);

/// Whether `wait`, how long a stop waits for QEMU to end after QMP's `quit` before it ends it
/// (`QUIT_WAIT`, then `ENDS`), follows `measured`, the longest QEMU was measured to take to end
/// on the OS (`QEMU_EXIT_MEASURED`): about twice it, and 10 s at least.
pub const fn follows_the_measured_exit(
    wait: std::time::Duration,
    measured: std::time::Duration,
) -> bool {
    let twice = 2 * measured.as_millis();
    let want = if twice > 10_000 { twice } else { 10_000 };
    wait.as_millis() >= want && wait.as_millis() <= want + 5_000
}

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
