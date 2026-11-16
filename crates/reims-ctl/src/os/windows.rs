//! The `crate::os` items on Windows.

use anyhow::{Context, Result};
use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};
use windows_sys::Win32::Foundation::SYSTEMTIME;

mod socket;
#[cfg(test)]
pub use socket::UnixListener;
pub use socket::UnixStream;

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

/// Process `pid`'s command line, as the process was given it, in the encoding of
/// `OsStr::as_encoded_bytes`; None for a process whose command line can't be read: one of
/// another user's, or one that has ended.
pub fn cmdline(pid: i32) -> Option<Vec<u8>> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Wdk::System::Threading::{
        NtQueryInformationProcess, ProcessCommandLineInformation,
    };
    use windows_sys::Win32::Foundation::UNICODE_STRING;
    use windows_sys::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
    let p = Process::open(pid, PROCESS_QUERY_LIMITED_INFORMATION).ok()?;
    // The size first, which the query fails with, then the line: a UNICODE_STRING and the text
    // it points to after it, in a buffer aligned for the string's pointer.
    let mut len = 0u32;
    // SAFETY: a query into no buffer, which writes only `len`.
    unsafe {
        NtQueryInformationProcess(
            p.0,
            ProcessCommandLineInformation,
            std::ptr::null_mut(),
            0,
            &mut len,
        )
    };
    if (len as usize) < std::mem::size_of::<UNICODE_STRING>() {
        return None;
    }
    let mut buf = vec![0u64; (len as usize).div_ceil(8)];
    // SAFETY: buf holds `len` bytes or more, aligned for a UNICODE_STRING.
    let status = unsafe {
        NtQueryInformationProcess(
            p.0,
            ProcessCommandLineInformation,
            buf.as_mut_ptr().cast(),
            (buf.len() * 8) as u32,
            &mut len,
        )
    };
    if status < 0 {
        return None;
    }
    // SAFETY: the query wrote a UNICODE_STRING at buf's start, whose text, `Length` bytes, is in
    // buf after it.
    let line = unsafe {
        let s = &*buf.as_ptr().cast::<UNICODE_STRING>();
        if s.Buffer.is_null() {
            return None;
        }
        std::slice::from_raw_parts(s.Buffer, usize::from(s.Length) / 2)
    };
    Some(std::ffi::OsString::from_wide(line).into_encoded_bytes())
}

/// Whether process `pid` may be a QEMU: its program's name starts `qemu-system`, as each of
/// QEMU's system emulators' does (`qemu-system-x86_64.exe`). Windows gives a pid to a new process
/// soon after its last owner ended, so a pid file a QEMU left can name another program.
pub fn may_be_qemu(pid: i32) -> bool {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Threading::{
        PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
    };
    let Ok(p) = Process::open(pid, PROCESS_QUERY_LIMITED_INFORMATION) else {
        return false;
    };
    let mut buf = vec![0u16; 32_768];
    let mut size = buf.len() as u32;
    // SAFETY: buf holds `size` UTF-16 units, which the call sets to the path's length.
    if unsafe { QueryFullProcessImageNameW(p.0, PROCESS_NAME_WIN32, buf.as_mut_ptr(), &mut size) }
        == 0
    {
        return false;
    }
    let image = std::path::PathBuf::from(std::ffi::OsString::from_wide(&buf[..size as usize]));
    image
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.to_ascii_lowercase().starts_with("qemu-system"))
}

/// Runs `cmd` detached, and returns once it runs: with no console (`DETACHED_PROCESS`), so
/// closing the terminal does not end it, in a process group of its own, and out of this
/// process's job where the job lets it go (`CREATE_BREAKAWAY_FROM_JOB`), as one an SSH login
/// runs in, which ends its processes when the login does. Windows keeps nothing of a process
/// that ended for its parent to collect. `cmd` is given its own standard handles: this
/// process's are not inherited ([`without_inherited_stdio`]).
pub fn spawn_detached(mut cmd: Command) -> io::Result<()> {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::{
        CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS,
    };
    let flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP;
    cmd.creation_flags(flags | CREATE_BREAKAWAY_FROM_JOB);
    without_inherited_stdio(|| match cmd.spawn() {
        // A job that keeps its processes refuses the breakaway.
        Err(e) if e.raw_os_error() == Some(ERROR_ACCESS_DENIED) => {
            cmd.creation_flags(flags);
            cmd.spawn().map(drop)
        }
        r => r.map(drop),
    })
}

/// Runs `f` with this process's standard handles made uninheritable, then puts them back as they
/// were. A new process inherits every inheritable handle of its parent, whatever standard handles
/// it is given, so a detached QEMU would hold the pipe `reims-ctl start | tee` writes to, and
/// the reader would wait for the VM to end.
fn without_inherited_stdio<T>(f: impl FnOnce() -> T) -> T {
    use windows_sys::Win32::Foundation::{
        GetHandleInformation, HANDLE_FLAG_INHERIT, INVALID_HANDLE_VALUE, SetHandleInformation,
    };
    use windows_sys::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    let mut inheritable = Vec::new();
    for id in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
        // SAFETY: the process's own standard handles, which stay open; flags are read into a u32.
        unsafe {
            let h = GetStdHandle(id);
            let mut flags = 0u32;
            if !h.is_null()
                && h != INVALID_HANDLE_VALUE
                && GetHandleInformation(h, &mut flags) != 0
                && flags & HANDLE_FLAG_INHERIT != 0
                && SetHandleInformation(h, HANDLE_FLAG_INHERIT, 0) != 0
            {
                inheritable.push(h);
            }
        }
    }
    let r = f();
    for h in inheritable {
        // SAFETY: as above.
        unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) };
    }
    r
}

/// The ways a stop ends a QEMU that QMP's `quit` did not: TerminateProcess. Windows has no
/// signal that asks a process with no console to end.
pub const ENDS: &[super::End] = &[(terminate, "killed (TerminateProcess)")];

fn terminate(pid: i32) {
    use windows_sys::Win32::System::Threading::{PROCESS_TERMINATE, TerminateProcess};
    if let Ok(p) = Process::open(pid, PROCESS_TERMINATE) {
        // SAFETY: the handle is open with PROCESS_TERMINATE; pid is a QEMU's, checked by
        // runtime::pid.
        unsafe { TerminateProcess(p.0, 1) };
    }
}

/// Runs `cmd` to its end in the console ([`run_in_terminal`]), then ends this process with its
/// exit code: Windows replaces no running program with another. Returns only why `cmd` could
/// not run.
pub fn exec(cmd: Command) -> io::Error {
    match run_in_terminal(cmd) {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => e,
    }
}

/// The command that runs the install script `script`, scripts/install-ctl.sh: `shell`, the
/// MSYS2 bash the install ran in, by its path (a `bash` on PATH can be WSL's), as a login shell
/// of MSYS2's CLANG64, whose PATH has CLANG64's clang, in the directory it starts in
/// (`CHERE_INVOKING`). The script's path has `/` for `\`: the script finds its checkout with
/// `dirname`, which splits at `/` alone.
pub fn install_script(shell: &Path, script: &Path) -> Result<Command> {
    if shell.as_os_str().is_empty() {
        anyhow::bail!(
            "the install record names no MSYS2 bash to run {} in: install again with \
             scripts\\install.ps1",
            script.display()
        );
    }
    let mut c = Command::new(shell);
    c.arg("-l")
        .arg(script.to_string_lossy().replace('\\', "/"))
        .env("MSYSTEM", "CLANG64")
        .env("CHERE_INVOKING", "1");
    no_glob(&mut c);
    Ok(c)
}

/// Keeps `c`, where it is an MSYS2 program (MSYS2's git, its bash), from expanding wildcards
/// and braces in its arguments, which MSYS2's runtime does for a program a Windows program
/// starts, as a shell would: `@{u}` reached MSYS2's git as `@u`. `noglob` in `MSYS` turns that
/// off; a Windows program ignores it.
pub fn no_glob(c: &mut Command) {
    let mut msys = std::env::var("MSYS").unwrap_or_default();
    if !msys.split_whitespace().any(|w| w == "noglob") {
        if !msys.is_empty() {
            msys.push(' ');
        }
        msys.push_str("noglob");
    }
    c.env("MSYS", msys);
}

/// Runs `cmd` in the console to its end. Ctrl+C and Ctrl+Break reach every process attached to
/// the console, so this one takes them as handled until `cmd` exits, and `cmd` gets them as it
/// would from a shell: a handler is not inherited, as ignoring them would be.
pub fn run_in_terminal(mut cmd: Command) -> io::Result<ExitStatus> {
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    unsafe extern "system" fn handled(ctrl: u32) -> windows_sys::core::BOOL {
        (ctrl == CTRL_C_EVENT || ctrl == CTRL_BREAK_EVENT).into()
    }
    // SAFETY: handled is a function for the process's life, removed again below.
    unsafe { SetConsoleCtrlHandler(Some(handled), 1) };
    let status = cmd.status();
    // SAFETY: as above.
    unsafe { SetConsoleCtrlHandler(Some(handled), 0) };
    status
}

/// Whether this process's window station is shown on a desktop (`WSF_VISIBLE`): an interactive
/// sign-in's, where a window can open, and not a service's or an SSH login's.
pub fn desktop_shown() -> bool {
    use windows_sys::Win32::System::StationsAndDesktops::{
        GetProcessWindowStation, GetUserObjectInformationW, UOI_FLAGS, USEROBJECTFLAGS,
    };
    /// winuser.h's flag of a window station a user sees.
    const WSF_VISIBLE: u32 = 1;
    // SAFETY: the process's window station handle is not to be closed; the call fills a
    // USEROBJECTFLAGS of the size given.
    unsafe {
        let station = GetProcessWindowStation();
        if station.is_null() {
            return false;
        }
        let mut flags: USEROBJECTFLAGS = std::mem::zeroed();
        let mut needed = 0u32;
        GetUserObjectInformationW(
            station,
            UOI_FLAGS,
            (&mut flags as *mut USEROBJECTFLAGS).cast(),
            std::mem::size_of::<USEROBJECTFLAGS>() as u32,
            &mut needed,
        ) != 0
            && flags.dwFlags & WSF_VISIBLE != 0
    }
}

/// A registry key, open to read until it drops.
struct Key(windows_sys::Win32::System::Registry::HKEY);

impl Key {
    /// `path` under `HKEY_LOCAL_MACHINE`.
    fn local_machine(path: &str) -> Option<Key> {
        use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, KEY_READ, RegOpenKeyExW};
        let path = wide(path);
        let mut key = std::ptr::null_mut();
        // SAFETY: path is NUL-terminated, and the call fills `key`.
        let r = unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, path.as_ptr(), 0, KEY_READ, &mut key) };
        (r == 0).then_some(Key(key))
    }

    /// The names of its subkeys.
    fn subkeys(&self) -> Vec<String> {
        use windows_sys::Win32::System::Registry::RegEnumKeyExW;
        let mut names = Vec::new();
        for i in 0.. {
            let mut name = [0u16; 256];
            let mut len = name.len() as u32;
            // SAFETY: name holds `len` UTF-16 units; the rest is left out.
            let r = unsafe {
                RegEnumKeyExW(
                    self.0,
                    i,
                    name.as_mut_ptr(),
                    &mut len,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if r != 0 {
                break;
            }
            names.push(String::from_utf16_lossy(&name[..len as usize]));
        }
        names
    }

    /// The strings of the value `value` of its subkey `sub`, a string or a list of strings.
    fn strings(&self, sub: &str, value: &str) -> Vec<String> {
        use windows_sys::Win32::System::Registry::{
            RRF_RT_REG_MULTI_SZ, RRF_RT_REG_SZ, RegGetValueW,
        };
        let (sub, value) = (wide(sub), wide(value));
        let mut data = vec![0u16; 4096];
        let mut size = (data.len() * 2) as u32;
        // SAFETY: sub and value are NUL-terminated; data holds `size` bytes.
        let r = unsafe {
            RegGetValueW(
                self.0,
                sub.as_ptr(),
                value.as_ptr(),
                RRF_RT_REG_SZ | RRF_RT_REG_MULTI_SZ,
                std::ptr::null_mut(),
                data.as_mut_ptr().cast(),
                &mut size,
            )
        };
        if r != 0 {
            return Vec::new();
        }
        data.truncate(size as usize / 2);
        data.split(|&c| c == 0)
            .filter(|s| !s.is_empty())
            .map(String::from_utf16_lossy)
            .collect()
    }

    /// Its values that are numbers, each by its name.
    fn numbers(&self) -> Vec<(String, u32)> {
        use windows_sys::Win32::System::Registry::{REG_DWORD, RegEnumValueW};
        let mut values = Vec::new();
        for i in 0.. {
            let mut name = vec![0u16; 1024];
            let mut len = name.len() as u32;
            let (mut kind, mut data, mut size) = (0u32, 0u32, 4u32);
            // SAFETY: name holds `len` UTF-16 units and data `size` bytes.
            let r = unsafe {
                RegEnumValueW(
                    self.0,
                    i,
                    name.as_mut_ptr(),
                    &mut len,
                    std::ptr::null(),
                    &mut kind,
                    (&mut data as *mut u32).cast(),
                    &mut size,
                )
            };
            // ERROR_MORE_DATA: a value too big for a number, which is no number.
            if r == 234 {
                continue;
            }
            if r != 0 {
                break;
            }
            if kind == REG_DWORD {
                values.push((String::from_utf16_lossy(&name[..len as usize]), data));
            }
        }
        values
    }
}

impl Drop for Key {
    fn drop(&mut self) {
        // SAFETY: the key is this Key's own, closed once.
        unsafe { windows_sys::Win32::System::Registry::RegCloseKey(self.0) };
    }
}

/// `s` in UTF-16, NUL-terminated.
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

/// The Vulkan driver manifests (ICD JSON files) the Vulkan loader reads on Windows, in its order,
/// each once: the display adapters' (`VulkanDriverName` in each one's key of the display device
/// class, where a driver package installs it), then those listed under
/// `HKLM\SOFTWARE\Khronos\Vulkan\Drivers` (each a value named for its path, enabled at 0). A
/// manifest whose file is gone, as an adapter removed leaves its key, is left out.
pub fn vulkan_driver_files() -> Vec<std::path::PathBuf> {
    const DISPLAY_CLASS: &str =
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";
    let mut files = Vec::new();
    if let Some(class) = Key::local_machine(DISPLAY_CLASS) {
        for adapter in class.subkeys() {
            files.extend(class.strings(&adapter, "VulkanDriverName"));
        }
    }
    if let Some(drivers) = Key::local_machine(r"SOFTWARE\Khronos\Vulkan\Drivers") {
        files.extend(
            drivers
                .numbers()
                .into_iter()
                .filter(|(_, enabled)| *enabled == 0)
                .map(|(path, _)| path),
        );
    }
    let mut seen = Vec::new();
    for f in files.into_iter().map(std::path::PathBuf::from) {
        if f.is_file() && !seen.contains(&f) {
            seen.push(f);
        }
    }
    seen
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

/// While it lives, the console gives what the terminal sends as the terminal sent it, escape
/// sequences and all (`ENABLE_VIRTUAL_TERMINAL_INPUT`), for [`read_terminal`]; where it drops,
/// the console's mode as it found it, in which crossterm reads the console's key records. It
/// does nothing where stdin is no console.
pub struct TerminalAnswers(Option<(windows_sys::Win32::Foundation::HANDLE, u32)>);

/// [`TerminalAnswers`], from now.
pub fn terminal_answers() -> TerminalAnswers {
    use windows_sys::Win32::System::Console::{
        ENABLE_VIRTUAL_TERMINAL_INPUT, GetConsoleMode, GetStdHandle, STD_INPUT_HANDLE,
        SetConsoleMode,
    };
    let mut mode = 0;
    // SAFETY: GetStdHandle has no preconditions; the mode calls fail on a handle that is no
    // console's.
    let set = unsafe {
        let h = GetStdHandle(STD_INPUT_HANDLE);
        (GetConsoleMode(h, &mut mode) != 0
            && SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_INPUT) != 0)
            .then_some((h, mode))
    };
    TerminalAnswers(set)
}

impl Drop for TerminalAnswers {
    fn drop(&mut self) {
        if let Some((h, mode)) = self.0 {
            // SAFETY: h is stdin's console handle, which the process keeps.
            unsafe { windows_sys::Win32::System::Console::SetConsoleMode(h, mode) };
        }
    }
}

/// Reads what the terminal sends into `buf`, waiting at most `timeout` for it: the characters of
/// the console's key-press records, in UTF-8, its other records (focus, the mouse, the window's
/// size) skipped. Ok(0) where it sent none in that time, or stdin is no console. Under
/// [`terminal_answers`] the console gives the terminal's escape sequences as their characters.
pub fn read_terminal(buf: &mut [u8], timeout: std::time::Duration) -> io::Result<usize> {
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, INPUT_RECORD, KEY_EVENT, ReadConsoleInputW, STD_INPUT_HANDLE,
    };
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let mut mode = 0;
    // SAFETY: GetStdHandle has no preconditions; GetConsoleMode fails on a handle that is no
    // console's.
    let h = unsafe { GetStdHandle(STD_INPUT_HANDLE) };
    if unsafe { GetConsoleMode(h, &mut mode) } == 0 {
        return Ok(0);
    }
    let deadline = std::time::Instant::now() + timeout;
    let mut records = [INPUT_RECORD::default(); 64];
    // A record's character is one UTF-16 unit, at most three bytes of UTF-8. A character of two
    // units split between two reads is lost; the terminal's answers are ASCII.
    let room = (buf.len() / 3).clamp(1, records.len()) as u32;
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        // SAFETY: h is stdin's console handle.
        match unsafe { WaitForSingleObject(h, left.as_millis().clamp(1, 1 << 30) as u32) } {
            WAIT_OBJECT_0 => {}
            WAIT_TIMEOUT => return Ok(0),
            _ => return Err(io::Error::last_os_error()),
        }
        let mut got = 0u32;
        // SAFETY: records holds at least `room` records, the most the call writes; the wait
        // above saw one queued, so it does not block.
        if unsafe { ReadConsoleInputW(h, records.as_mut_ptr(), room, &mut got) } == 0 {
            return Err(io::Error::last_os_error());
        }
        let units: Vec<u16> = records[..got as usize]
            .iter()
            .filter(|r| u32::from(r.EventType) == KEY_EVENT)
            .filter_map(|r| {
                // SAFETY: a KEY_EVENT record holds a key event, whose character is UTF-16.
                let (down, c) = unsafe {
                    (
                        r.Event.KeyEvent.bKeyDown,
                        r.Event.KeyEvent.uChar.UnicodeChar,
                    )
                };
                (down != 0 && c != 0).then_some(c)
            })
            .collect();
        if !units.is_empty() {
            let s = String::from_utf16_lossy(&units);
            let n = s.len().min(buf.len());
            buf[..n].copy_from_slice(&s.as_bytes()[..n]);
            return Ok(n);
        }
        if left.is_zero() {
            return Ok(0);
        }
    }
}

/// The space the file `m` describes takes on its volume, counted as its length: a sparse file's
/// holes count as written.
pub fn allocated(m: &std::fs::Metadata) -> u64 {
    m.len()
}

/// Whether `a` and `b`, which exist, are on one volume: the same volume path
/// (`GetVolumePathNameW`, which a folder a volume is mounted on answers too).
pub fn same_file_system(a: &Path, b: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetVolumePathNameW;
    let volume = |p: &Path| -> Option<String> {
        let wide: Vec<u16> = p.as_os_str().encode_wide().chain([0]).collect();
        let mut root = vec![0u16; 1024];
        // SAFETY: wide is NUL-terminated, and root holds its length in UTF-16 units.
        if unsafe { GetVolumePathNameW(wide.as_ptr(), root.as_mut_ptr(), root.len() as u32) } == 0 {
            return None;
        }
        let n = root.iter().position(|&c| c == 0).unwrap_or(root.len());
        Some(String::from_utf16_lossy(&root[..n]).to_lowercase())
    };
    matches!((volume(a), volume(b)), (Some(x), Some(y)) if x == y)
}

/// The flag [`set_at_interrupt`] gave, for its console handler.
static INTERRUPT_FLAG: std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicBool> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// From now, Ctrl+C and Ctrl+Break set `flag`, and end the process no more: the console's
/// handler takes them as handled, as Unix's set_at_interrupt takes SIGINT and SIGTERM.
pub fn set_at_interrupt(flag: &'static std::sync::atomic::AtomicBool) {
    use std::sync::atomic::Ordering;
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    unsafe extern "system" fn stop(ctrl: u32) -> windows_sys::core::BOOL {
        let ours = ctrl == CTRL_C_EVENT || ctrl == CTRL_BREAK_EVENT;
        let flag = INTERRUPT_FLAG.load(Ordering::SeqCst);
        // SAFETY: a null pointer, or the &'static flag set_at_interrupt stored.
        if let (true, Some(flag)) = (ours, unsafe { flag.as_ref() }) {
            flag.store(true, Ordering::SeqCst);
        }
        ours.into()
    }
    INTERRUPT_FLAG.store(std::ptr::from_ref(flag).cast_mut(), Ordering::SeqCst);
    // SAFETY: stop is a function for the process's life.
    unsafe { SetConsoleCtrlHandler(Some(stop), 1) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the stdio test runs as reims-ctl: starts a detached process that lives about 5 s,
    /// as a start does QEMU, and returns.
    #[test]
    fn start_a_detached_process() {
        if std::env::var_os("REIMS_CTL_TEST_DETACH").is_some() {
            let mut sleeper = Command::new("cmd");
            sleeper
                .args(["/c", "ping -n 6 127.0.0.1"])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            spawn_detached(sleeper).unwrap();
        }
    }

    /// A detached process holds no handle of its starter's standard output: a reader of the pipe
    /// that output is, as `reims-ctl start NAME | tee`'s tee, sees its end once the starter ends,
    /// not once the detached process does.
    #[test]
    fn a_detached_process_holds_no_pipe_of_its_starters_output() {
        let t = std::time::Instant::now();
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "os::windows::tests::start_a_detached_process"])
            .env("REIMS_CTL_TEST_DETACH", "1")
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("1 passed"),
            "{out:?}"
        );
        assert!(
            t.elapsed() < std::time::Duration::from_secs(3),
            "{:?}",
            t.elapsed()
        );
    }

    /// An update runs the install script in the MSYS2 bash the install recorded, as a CLANG64
    /// login shell, the script's path with `/` for `\`, and with no wildcard expansion; a record
    /// that names no bash stops it, saying what to do.
    #[test]
    fn the_install_script_runs_in_the_recorded_msys2_bash() {
        let c = install_script(
            Path::new("C:\\msys64\\usr\\bin\\bash.exe"),
            Path::new("C:\\Users\\u\\reims-vgpu\\scripts\\install-ctl.sh"),
        )
        .unwrap();
        assert_eq!(c.get_program(), "C:\\msys64\\usr\\bin\\bash.exe");
        let args: Vec<_> = c.get_args().collect();
        assert_eq!(args, ["-l", "C:/Users/u/reims-vgpu/scripts/install-ctl.sh"]);
        let env = |k: &str| {
            c.get_envs()
                .find(|(n, _)| *n == k)
                .and_then(|(_, v)| v)
                .map(|v| v.to_string_lossy().into_owned())
        };
        assert_eq!(env("MSYSTEM").as_deref(), Some("CLANG64"));
        assert_eq!(env("CHERE_INVOKING").as_deref(), Some("1"));
        assert!(env("MSYS").unwrap().split(' ').any(|w| w == "noglob"));
        let e = install_script(Path::new(""), Path::new("C:\\r\\scripts\\install-ctl.sh"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("install again with scripts\\install.ps1"), "{e}");
    }

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

    /// The argument that has [`console_child`] run, with the file it writes what it read to.
    const CONSOLE_CHILD: &str = "reims-ctl-console-child=";

    /// What [`a_terminals_answer_reaches_read_terminal_as_sent`] runs in a pseudoconsole: asks
    /// the terminal its device attributes (DA1) under [`terminal_answers`], as `view::term`
    /// asks it, and writes what [`read_terminal`] gives back into the file its argument names;
    /// in a run of the tests, nothing.
    #[test]
    fn console_child() {
        use std::io::Write;
        let Some(out) =
            std::env::args().find_map(|a| a.strip_prefix(CONSOLE_CHILD).map(String::from))
        else {
            return;
        };
        let _answers = terminal_answers();
        let mut stdout = std::io::stdout();
        stdout
            .write_all(b"\x1b[c")
            .and_then(|_| stdout.flush())
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut got = Vec::new();
        while !da1_answer(&got) {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            let mut buf = [0u8; 256];
            match read_terminal(&mut buf, left).unwrap() {
                0 => break,
                n => got.extend_from_slice(&buf[..n]),
            }
        }
        std::fs::write(out, &got).unwrap();
    }

    /// Whether `b` holds an answer to DA1: `ESC [ ?`, its attributes, `c`.
    fn da1_answer(b: &[u8]) -> bool {
        b.windows(3)
            .position(|w| w == b"\x1b[?")
            .is_some_and(|i| b[i..].contains(&b'c'))
    }

    /// The answer a terminal sends to a query reaches [`read_terminal`] as the terminal sent it,
    /// escape sequence and all, under [`terminal_answers`]: in a pseudoconsole, whose console
    /// answers the device attributes query itself or passes it on to this test, which answers as
    /// a terminal does.
    #[test]
    fn a_terminals_answer_reaches_read_terminal_as_sent() {
        use std::io::{Read, Write};
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Console::{
            COORD, ClosePseudoConsole, CreatePseudoConsole, HPCON,
        };
        use windows_sys::Win32::System::Threading::{
            CreateProcessW, DeleteProcThreadAttributeList, EXTENDED_STARTUPINFO_PRESENT,
            GetExitCodeProcess, InitializeProcThreadAttributeList,
            PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, PROCESS_INFORMATION, STARTF_USESTDHANDLES,
            STARTUPINFOEXW, TerminateProcess, UpdateProcThreadAttribute, WaitForSingleObject,
        };
        let out = std::env::temp_dir().join(format!("reims-ctl-console-{}", std::process::id()));
        let (in_read, mut in_write) = std::io::pipe().unwrap();
        let (mut out_read, out_write) = std::io::pipe().unwrap();
        let mut pc: HPCON = 0;
        // SAFETY: both pipe ends are open; the console duplicates them.
        let hr = unsafe {
            CreatePseudoConsole(
                COORD { X: 80, Y: 25 },
                in_read.as_raw_handle(),
                out_write.as_raw_handle(),
                0,
                &mut pc,
            )
        };
        assert_eq!(hr, 0, "CreatePseudoConsole");
        drop((in_read, out_write));
        // The terminal: takes all the console writes, and answers each DA1 it passes on.
        let terminal = std::thread::spawn(move || {
            let count = |b: &[u8], q: &[u8]| b.windows(q.len()).filter(|w| *w == q).count();
            let (mut seen, mut answered, mut chunk) = (Vec::new(), 0, [0u8; 4096]);
            while let Ok(n @ 1..) = out_read.read(&mut chunk) {
                seen.extend_from_slice(&chunk[..n]);
                let asked = count(&seen, b"\x1b[c") + count(&seen, b"\x1b[0c");
                for _ in answered..asked {
                    let _ = in_write.write_all(b"\x1b[?62;4c");
                }
                answered = asked;
            }
            seen
        });
        let mut size = 0usize;
        // SAFETY: a size query, with no list.
        unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut size) };
        let mut list = vec![0usize; size.div_ceil(size_of::<usize>())];
        let attrs = list.as_mut_ptr().cast();
        // SAFETY: list holds `size` bytes, aligned for the pointers the list holds; the
        // pseudoconsole's attribute is its handle itself.
        unsafe {
            assert_ne!(InitializeProcThreadAttributeList(attrs, 1, 0, &mut size), 0);
            assert_ne!(
                UpdateProcThreadAttribute(
                    attrs,
                    0,
                    PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE as usize,
                    pc as *const std::ffi::c_void,
                    size_of::<HPCON>(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                ),
                0
            );
        }
        let mut si = STARTUPINFOEXW::default();
        si.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        // The child's standard handles are its console's, not this process's, which a test
        // harness may have made pipes.
        si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        si.StartupInfo.hStdInput = INVALID_HANDLE_VALUE;
        si.StartupInfo.hStdOutput = INVALID_HANDLE_VALUE;
        si.StartupInfo.hStdError = INVALID_HANDLE_VALUE;
        si.lpAttributeList = attrs;
        let exe = std::env::current_exe().unwrap();
        let mut line = wide(&format!(
            "\"{}\" --exact os::windows::tests::console_child \"{CONSOLE_CHILD}{}\"",
            exe.display(),
            out.display()
        ));
        let mut pi = PROCESS_INFORMATION::default();
        // SAFETY: line is NUL-terminated and writable; si and its list live past the call.
        let ok = unsafe {
            CreateProcessW(
                std::ptr::null(),
                line.as_mut_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                EXTENDED_STARTUPINFO_PRESENT,
                std::ptr::null(),
                std::ptr::null(),
                &si.StartupInfo,
                &mut pi,
            )
        };
        assert_ne!(ok, 0, "CreateProcessW: {}", io::Error::last_os_error());
        let mut code = 0u32;
        // SAFETY: pi's handles are the child's, closed once here; the list and the console are
        // this test's, ended once the child has.
        let waited = unsafe {
            let waited = WaitForSingleObject(pi.hProcess, 30_000);
            if waited != WAIT_OBJECT_0 {
                TerminateProcess(pi.hProcess, 1);
            }
            GetExitCodeProcess(pi.hProcess, &mut code);
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
            DeleteProcThreadAttributeList(attrs);
            ClosePseudoConsole(pc);
            waited
        };
        let seen = terminal.join().unwrap();
        let got = std::fs::read(&out).unwrap_or_default();
        let _ = std::fs::remove_file(&out);
        let console = String::from_utf8_lossy(&seen);
        assert_eq!(
            (waited, code),
            (WAIT_OBJECT_0, 0),
            "the console wrote {console:?}"
        );
        assert!(
            da1_answer(&got),
            "read {:?}; the console wrote {console:?}",
            String::from_utf8_lossy(&got)
        );
    }
}
