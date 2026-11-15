//! The `crate::os` items on Windows.

use anyhow::{Context, Result};
use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};
use windows_sys::Win32::Foundation::SYSTEMTIME;

/// What a write to a closed pipe fails with: ERROR_NO_DATA, "The pipe is being closed".
pub const CLOSED_PIPE: i32 = 232;

/// Reads and writes at an offset of a file, as Unix's `FileExt` does them; on Windows each moves
/// the file's cursor, which reims-ctl does not read where it uses these.
pub trait FileExt {
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    fn write_all_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
}

impl FileExt for std::fs::File {
    fn read_exact_at(&self, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_read(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    fn write_all_at(&self, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_write(buf, offset) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

/// Nothing: Windows keeps no Unix permissions, and runs a program by its name.
pub fn set_mode(_p: &Path, _mode: u32) -> io::Result<()> {
    Ok(())
}

/// Lets the user write `p`: clears its read-only attribute.
pub fn make_writable(p: &Path) -> io::Result<()> {
    let mut perm = std::fs::metadata(p)?.permissions();
    #[allow(
        clippy::permissions_set_readonly_false,
        reason = "Windows' read-only attribute, not Unix's mode bits"
    )]
    perm.set_readonly(false);
    std::fs::set_permissions(p, perm)
}

fn system_time(local: bool) -> SYSTEMTIME {
    use windows_sys::Win32::System::SystemInformation::{GetLocalTime, GetSystemTime};
    // SAFETY: an all-zero SYSTEMTIME is valid, and each call fills the one it is given.
    unsafe {
        let mut t: SYSTEMTIME = std::mem::zeroed();
        if local {
            GetLocalTime(&mut t);
        } else {
            GetSystemTime(&mut t);
        }
        t
    }
}

fn from_system_time(t: &SYSTEMTIME) -> super::LocalTime {
    super::LocalTime {
        year: i32::from(t.wYear),
        month: u32::from(t.wMonth),
        day: u32::from(t.wDay),
        hour: u32::from(t.wHour),
        minute: u32::from(t.wMinute),
        second: u32::from(t.wSecond),
    }
}

/// Seconds since 1970 of `t`, a date and time of the proleptic Gregorian calendar.
fn epoch_secs(t: &super::LocalTime) -> i64 {
    // Days from civil, after Howard Hinnant's algorithm.
    let (m, d) = (i64::from(t.month), i64::from(t.day));
    let y = i64::from(t.year) - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    days * 86_400 + i64::from(t.hour) * 3600 + i64::from(t.minute) * 60 + i64::from(t.second)
}

/// The local date and time now.
pub fn local_now() -> super::LocalTime {
    from_system_time(&system_time(true))
}

/// The local time now in `fmt`, whose `%Y`, `%m`, `%d`, `%H`, `%M`, `%S`, `%z` and `%%` are
/// strftime(3)'s.
pub fn now_stamp(fmt: &str) -> String {
    let local = local_now();
    let utc = from_system_time(&system_time(false));
    // The two clocks are read a moment apart: the zone's offset, in whole minutes, is their
    // difference rounded.
    let offset = (epoch_secs(&local) - epoch_secs(&utc) + 30).div_euclid(60);
    stamp(fmt, &local, offset)
}

/// `fmt` written for `t`, a local time `offset` minutes ahead of UTC.
fn stamp(fmt: &str, t: &super::LocalTime, offset: i64) -> String {
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&format!("{:04}", t.year)),
            Some('m') => out.push_str(&format!("{:02}", t.month)),
            Some('d') => out.push_str(&format!("{:02}", t.day)),
            Some('H') => out.push_str(&format!("{:02}", t.hour)),
            Some('M') => out.push_str(&format!("{:02}", t.minute)),
            Some('S') => out.push_str(&format!("{:02}", t.second)),
            Some('z') => {
                let sign = if offset < 0 { '-' } else { '+' };
                let m = offset.abs();
                out.push_str(&format!("{sign}{:02}{:02}", m / 60, m % 60));
            }
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// The host's RAM in MiB, where the host says.
pub fn memory_mib() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: an all-zero MEMORYSTATUSEX with its length set is what the call fills.
    unsafe {
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (GlobalMemoryStatusEx(&mut m) != 0).then_some(m.ullTotalPhys / (1024 * 1024))
    }
}

/// Makes `d`, a directory for QMP sockets outside the VMs' own. It is in the user's own
/// temporary directory, whose access list keeps other users out, and the directory takes it.
pub fn private_dir(d: &Path) -> Result<()> {
    std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))
}

/// Makes the directory `d`, which must not exist; in the user's own temporary directory, it
/// takes that directory's access list.
pub fn create_private_dir(d: &Path) -> io::Result<()> {
    std::fs::create_dir(d)
}

/// Whether the file `m` describes is this user's: any in the user's own temporary directory,
/// where reims-ctl asks.
pub fn owned_by_me(_m: &std::fs::Metadata) -> bool {
    true
}

/// The group that owns the file `m` describes: none on Windows.
pub fn gid(_m: &std::fs::Metadata) -> Option<u32> {
    None
}

/// Whether this process may open the file `p` to read and write it.
pub fn may_read_write(p: &Path) -> bool {
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(p)
        .is_ok()
}

/// A process, open for what it was opened for until it drops.
struct Process(windows_sys::Win32::Foundation::HANDLE);

impl Process {
    fn open(pid: i32, access: u32) -> io::Result<Process> {
        use windows_sys::Win32::System::Threading::OpenProcess;
        if pid <= 0 {
            return Err(io::Error::from_raw_os_error(ERROR_INVALID_PARAMETER));
        }
        // SAFETY: OpenProcess has no preconditions; a null handle is its failure.
        let h = unsafe { OpenProcess(access, 0, pid as u32) };
        if h.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(Process(h))
        }
    }

    /// Whether it has ended, for a process opened with `PROCESS_SYNCHRONIZE`.
    fn ended(&self) -> bool {
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        use windows_sys::Win32::System::Threading::WaitForSingleObject;
        // SAFETY: the handle is open; a wait of 0 ms only reads its state.
        unsafe { WaitForSingleObject(self.0, 0) == WAIT_OBJECT_0 }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // SAFETY: the handle is this Process's own, closed once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

/// What OpenProcess fails with for a pid no process has: ERROR_INVALID_PARAMETER.
const ERROR_INVALID_PARAMETER: i32 = 87;

/// What OpenProcess fails with for a process this user may not open: ERROR_ACCESS_DENIED.
const ERROR_ACCESS_DENIED: i32 = 5;

/// Whether process `pid` runs: it exists and has not ended, though it may be another user's. A
/// process that ended is kept, with its pid, while any handle to it is open.
pub fn alive(pid: i32) -> bool {
    use windows_sys::Win32::System::Threading::{
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
    };
    match Process::open(pid, PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE) {
        Ok(p) => !p.ended(),
        Err(e) => e.raw_os_error() == Some(ERROR_ACCESS_DENIED),
    }
}

/// Whether no process `pid` runs: none has the pid, or the one that has it has ended.
pub fn gone(pid: i32) -> bool {
    use windows_sys::Win32::System::Threading::PROCESS_SYNCHRONIZE;
    match Process::open(pid, PROCESS_SYNCHRONIZE) {
        Ok(p) => p.ended(),
        Err(e) => e.raw_os_error() == Some(ERROR_INVALID_PARAMETER),
    }
}

/// Process `pid`'s command line. Not on Windows yet.
pub fn cmdline(_pid: i32) -> Option<Vec<u8>> {
    None
}

/// Whether process `pid` may be a QEMU. Not on Windows yet.
pub fn may_be_qemu(_pid: i32) -> bool {
    false
}

/// Runs `cmd` detached. Not on Windows yet.
pub fn spawn_detached(_cmd: Command) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "reims-ctl starts no QEMU on Windows yet",
    ))
}

/// The ways a stop ends a QEMU that QMP's `quit` did not: none on Windows yet.
pub const ENDS: &[super::End] = &[];

/// Runs `cmd` to its end, then ends this process with its exit code: Windows replaces no
/// running program with another. Returns only why `cmd` could not run.
pub fn exec(mut cmd: Command) -> io::Error {
    match cmd.status() {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => e,
    }
}

/// Runs `cmd` in the console to its end.
pub fn run_in_terminal(mut cmd: Command) -> io::Result<ExitStatus> {
    cmd.status()
}

/// Discards what the console's input holds that was not read yet.
pub fn flush_input() {
    use windows_sys::Win32::System::Console::{
        FlushConsoleInputBuffer, GetStdHandle, STD_INPUT_HANDLE,
    };
    // SAFETY: GetStdHandle has no preconditions, and FlushConsoleInputBuffer fails on a handle
    // that is no console's.
    unsafe { FlushConsoleInputBuffer(GetStdHandle(STD_INPUT_HANDLE)) };
}

/// Reads what the terminal sends. Not on Windows yet: nothing.
pub fn read_terminal(_buf: &mut [u8], _timeout: std::time::Duration) -> io::Result<usize> {
    Ok(0)
}

/// A connection to a Unix socket, as QMP's: none on Windows yet, which refuses each connect.
pub struct UnixStream(std::convert::Infallible);

impl UnixStream {
    pub fn connect(_path: &Path) -> io::Result<UnixStream> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "reims-ctl speaks no QMP on Windows yet",
        ))
    }

    pub fn set_read_timeout(&self, _t: Option<std::time::Duration>) -> io::Result<()> {
        match self.0 {}
    }

    pub fn set_write_timeout(&self, _t: Option<std::time::Duration>) -> io::Result<()> {
        match self.0 {}
    }

    pub fn try_clone(&self) -> io::Result<UnixStream> {
        match self.0 {}
    }
}

impl io::Read for UnixStream {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        match self.0 {}
    }
}

impl io::Write for UnixStream {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        match self.0 {}
    }

    fn flush(&mut self) -> io::Result<()> {
        match self.0 {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stamp is strftime's for its fields, the offset as `+hhmm`, and the seconds since 1970
    /// count the calendar's days.
    #[test]
    fn a_stamp_writes_the_fields_strftime_writes() {
        let t = super::super::LocalTime {
            year: 2026,
            month: 3,
            day: 9,
            hour: 7,
            minute: 5,
            second: 4,
        };
        assert_eq!(
            stamp("%Y-%m-%d %H:%M:%S %z", &t, -480),
            "2026-03-09 07:05:04 -0800"
        );
        assert_eq!(stamp("%Y%m%d-%H%M%S", &t, 330), "20260309-070504");
        assert_eq!(stamp("%z %% %q", &t, 345), "+0545 % %q");
        assert_eq!(epoch_secs(&t), 1_773_039_904);
        let epoch = super::super::LocalTime {
            year: 1970,
            month: 1,
            day: 1,
            hour: 0,
            minute: 0,
            second: 0,
        };
        assert_eq!(epoch_secs(&epoch), 0);
    }
}
