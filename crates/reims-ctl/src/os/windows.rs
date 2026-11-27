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
    memory_status().map(|m| m.ullTotalPhys / (1024 * 1024))
}

/// The memory a new process can commit now, in MiB, where the host says: what is left of
/// Windows' commit limit, the host's RAM and page file together (`ullAvailPageFile`).
pub fn memory_room_mib() -> Option<u64> {
    memory_status().map(|m| m.ullAvailPageFile / (1024 * 1024))
}

/// The host's memory now, as `GlobalMemoryStatusEx` says it.
fn memory_status() -> Option<windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: an all-zero MEMORYSTATUSEX with its length set is what the call fills.
    unsafe {
        let mut m: MEMORYSTATUSEX = std::mem::zeroed();
        m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (GlobalMemoryStatusEx(&mut m) != 0).then_some(m)
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

/// When the file `m` describes last changed, as Unix's ctime tells: none, as std's metadata on
/// Windows gives no such time.
pub fn changed(_m: &std::fs::Metadata) -> Option<std::time::SystemTime> {
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

/// Runs `cmd` detached, its standard input empty and its output and errors into `log`, and
/// returns once it runs: with no console (`DETACHED_PROCESS`), so closing the terminal does not
/// end it, in a process group of its own, and out of this process's job where the job lets it go
/// (`CREATE_BREAKAWAY_FROM_JOB`), as one an SSH login runs in, which ends its processes when the
/// login does. Windows keeps nothing of a process that ended for its parent to collect.
///
/// It inherits those standard handles and no other (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`). A new
/// process otherwise inherits every inheritable handle of its parent, so a QEMU would hold the
/// pipe `reims-ctl start | tee` writes to, or any handle reims-ctl's own starter left it
/// inheritable, and their readers would wait for the VM to end. std's `Command` takes no such
/// list, so the process is made here from `cmd`'s program, arguments, environment changes and
/// directory ([`command_line`], [`environment`]); its standard I/O settings are not read, nor an
/// `env_clear`, which reims-ctl does not use.
pub fn spawn_detached(cmd: Command, log: std::fs::File) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
    use windows_sys::Win32::System::Threading::{
        CREATE_BREAKAWAY_FROM_JOB, CREATE_NEW_PROCESS_GROUP, DETACHED_PROCESS,
    };
    let null = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("NUL")?;
    // The handles a process is given must be inheritable; these two close once it runs.
    for h in [null.as_raw_handle(), log.as_raw_handle()] {
        // SAFETY: both files are open.
        if unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
            return Err(io::Error::last_os_error());
        }
    }
    let line = command_line(&cmd)?;
    let env = environment(&cmd)?;
    let dir = cmd.get_current_dir().map(|d| wide_os(d.as_os_str()));
    let flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP;
    let create = |flags| {
        create_process(
            line.clone(),
            env.as_deref(),
            dir.as_deref(),
            flags,
            null.as_raw_handle(),
            log.as_raw_handle(),
        )
    };
    match create(flags | CREATE_BREAKAWAY_FROM_JOB) {
        // A job that keeps its processes refuses the breakaway.
        Err(e) if e.raw_os_error() == Some(ERROR_ACCESS_DENIED) => create(flags),
        r => r,
    }
}

/// Makes the process `line` runs, with `flags`, the environment block `env` (this process's
/// where None) and the directory `dir` (this process's where None), its standard input `stdin`
/// and its output and errors `out`, which are the only handles it inherits.
fn create_process(
    mut line: Vec<u16>,
    env: Option<&[u16]>,
    dir: Option<&[u16]>,
    flags: u32,
    stdin: windows_sys::Win32::Foundation::HANDLE,
    out: windows_sys::Win32::Foundation::HANDLE,
) -> io::Result<()> {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::{
        CREATE_UNICODE_ENVIRONMENT, CreateProcessW, DeleteProcThreadAttributeList,
        EXTENDED_STARTUPINFO_PRESENT, InitializeProcThreadAttributeList,
        LPPROC_THREAD_ATTRIBUTE_LIST, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, PROCESS_INFORMATION,
        STARTF_USESTDHANDLES, STARTUPINFOEXW, UpdateProcThreadAttribute,
    };
    /// An attribute list, deleted where it drops; its memory is the caller's.
    struct List(LPPROC_THREAD_ATTRIBUTE_LIST);
    impl Drop for List {
        fn drop(&mut self) {
            // SAFETY: a list InitializeProcThreadAttributeList made, deleted once.
            unsafe { DeleteProcThreadAttributeList(self.0) };
        }
    }
    // The list holds the handles' address: the array lives past CreateProcessW.
    let handles: Vec<HANDLE> = if stdin == out {
        vec![stdin]
    } else {
        vec![stdin, out]
    };
    let mut size = 0usize;
    // SAFETY: a size query, with no list.
    unsafe { InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut size) };
    let mut memory = vec![0usize; size.div_ceil(size_of::<usize>())];
    let attrs: LPPROC_THREAD_ATTRIBUTE_LIST = memory.as_mut_ptr().cast();
    // SAFETY: memory holds `size` bytes, aligned for the pointers a list holds.
    if unsafe { InitializeProcThreadAttributeList(attrs, 1, 0, &mut size) } == 0 {
        return Err(io::Error::last_os_error());
    }
    let list = List(attrs);
    // SAFETY: the list is initialized with room for one attribute; the handle list's value is
    // the array of handles, which outlives the list's use.
    let updated = unsafe {
        UpdateProcThreadAttribute(
            list.0,
            0,
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
            handles.as_ptr().cast(),
            size_of_val(handles.as_slice()),
            std::ptr::null_mut(),
            std::ptr::null(),
        )
    };
    if updated == 0 {
        return Err(io::Error::last_os_error());
    }
    let mut si = STARTUPINFOEXW::default();
    si.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = stdin;
    si.StartupInfo.hStdOutput = out;
    si.StartupInfo.hStdError = out;
    si.lpAttributeList = list.0;
    let mut pi = PROCESS_INFORMATION::default();
    // SAFETY: line is NUL-terminated and writable; env is a Unicode environment block and dir
    // NUL-terminated where given; si and its list, and the handles it names, live past the call.
    let ok = unsafe {
        CreateProcessW(
            std::ptr::null(),
            line.as_mut_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            1,
            flags | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            env.map_or(std::ptr::null(), |e| e.as_ptr().cast()),
            dir.map_or(std::ptr::null(), <[u16]>::as_ptr),
            &si.StartupInfo,
            &mut pi,
        )
    };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the new process's handles, this function's own, closed once.
    unsafe {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    drop(list);
    Ok(())
}

/// `cmd`'s program and arguments as a command line, NUL-terminated, written as std's `Command`
/// writes one: the program in quotes, and an argument in quotes where it is empty or holds a space
/// or a tab, the backslashes before each of its quotes and before the closing quote doubled and
/// each quote escaped, so the C runtime's parser, QEMU's and Rust's, reads back the arguments
/// given. The program is found as CreateProcessW finds it: by its path, as reims-ctl gives
/// QEMU's.
fn command_line(cmd: &Command) -> io::Result<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;
    const QUOTE: u16 = b'"' as u16;
    const BACKSLASH: u16 = b'\\' as u16;
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidInput, what.to_string());
    let program = cmd.get_program();
    if program.encode_wide().any(|c| c == QUOTE || c == 0) {
        return Err(invalid("a program's path with a quote or a NUL"));
    }
    let mut line = vec![QUOTE];
    line.extend(program.encode_wide());
    line.push(QUOTE);
    for arg in cmd.get_args() {
        line.push(u16::from(b' '));
        let quote = arg.is_empty()
            || arg
                .encode_wide()
                .any(|c| c == u16::from(b' ') || c == u16::from(b'\t'));
        if quote {
            line.push(QUOTE);
        }
        let mut backslashes = 0;
        for c in arg.encode_wide() {
            match c {
                0 => return Err(invalid("an argument with a NUL")),
                BACKSLASH => backslashes += 1,
                _ => {
                    if c == QUOTE {
                        line.extend(std::iter::repeat_n(BACKSLASH, backslashes + 1));
                    }
                    backslashes = 0;
                }
            }
            line.push(c);
        }
        if quote {
            line.extend(std::iter::repeat_n(BACKSLASH, backslashes));
            line.push(QUOTE);
        }
    }
    line.push(0);
    Ok(line)
}

/// The environment block `cmd` runs with, for CreateProcessW's `CREATE_UNICODE_ENVIRONMENT`:
/// None, this process's own, where `cmd` changes none; else this process's with `cmd`'s changes,
/// each name once whatever its case, as Windows matches names, in the order of their names in
/// capitals, as Windows sorts a block.
fn environment(cmd: &Command) -> io::Result<Option<Vec<u16>>> {
    use std::ffi::{OsStr, OsString};
    use std::os::windows::ffi::OsStrExt;
    if cmd.get_envs().len() == 0 {
        return Ok(None);
    }
    let key = |k: &OsStr| k.to_string_lossy().to_uppercase();
    let mut env: std::collections::BTreeMap<String, (OsString, OsString)> = std::env::vars_os()
        .map(|(k, v)| (key(&k), (k, v)))
        .collect();
    for (k, v) in cmd.get_envs() {
        match v {
            Some(v) => env.insert(key(k), (k.to_owned(), v.to_owned())),
            None => env.remove(&key(k)),
        };
    }
    let mut block = Vec::new();
    for (k, v) in env.values() {
        let start = block.len();
        block.extend(k.encode_wide());
        block.push(u16::from(b'='));
        block.extend(v.encode_wide());
        if block[start..].contains(&0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "an environment variable with a NUL",
            ));
        }
        block.push(0);
    }
    if block.is_empty() {
        block.push(0);
    }
    block.push(0);
    Ok(Some(block))
}

/// `s` in UTF-16, NUL-terminated.
fn wide_os(s: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    s.encode_wide().chain([0]).collect()
}

/// How long a stop waits for QEMU to end after QMP's `quit` before it ends it (`ENDS`): about
/// twice [`QEMU_EXIT_MEASURED`] ([`super::follows_the_measured_exit`]).
pub const QUIT_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// The longest QEMU took to end on Windows, from the guest's shutdown to the process's end.
/// **Measured** on an Intel UHD Graphics 630, Windows 11, 2026-11-20, three boots of a macOS
/// 26.7.2 guest stopped after Safari ran a WebGL page: 2.4, 4.9 and 28.1 s. In the 28.1 s, QEMU
/// called `ExitProcess` 1.4 s after the shutdown, and its process ended 26.7 s after that. A stop
/// that gave QEMU 15 s after `quit` (10 s, then 5 s after `TerminateProcess`) ended in "did not
/// exit", with QEMU gone 2 s later.
pub const QEMU_EXIT_MEASURED: std::time::Duration = std::time::Duration::from_millis(28_100);

const _: () = assert!(super::follows_the_measured_exit(
    QUIT_WAIT,
    QEMU_EXIT_MEASURED
));

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

/// Whether the Windows Hypervisor Platform answers that a hypervisor runs, as QEMU's WHPX asks it
/// (`WHvGetCapability`, `WHvCapabilityCodeHypervisorPresent`); None where the platform's DLL,
/// WinHvPlatform.dll, is missing, as it is until the feature is turned on. The DLL is loaded
/// here, not linked: a binary that imports it does not start without it.
pub fn whpx_present() -> Option<bool> {
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
    };
    type GetCapability =
        unsafe extern "system" fn(u32, *mut core::ffi::c_void, u32, *mut u32) -> i32;
    let name: Vec<u16> = "WinHvPlatform.dll\0".encode_utf16().collect();
    // SAFETY: a NUL-terminated name; the module stays loaded for the process's life.
    let module = unsafe {
        LoadLibraryExW(
            name.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if module.is_null() {
        return None;
    }
    // SAFETY: a NUL-terminated name in a loaded module.
    let f = unsafe { GetProcAddress(module, c"WHvGetCapability".as_ptr().cast()) }?;
    // SAFETY: WinHvPlatform.h's signature of WHvGetCapability.
    let get: GetCapability = unsafe { std::mem::transmute(f) };
    // WHV_CAPABILITY, a union whose first member, HypervisorPresent, is a BOOL; the buffer has
    // room for the union's largest member.
    let mut buf = [0u8; 256];
    let mut written = 0u32;
    // SAFETY: the buffer and its size, and a u32 for what was written.
    let hr = unsafe { get(0, buf.as_mut_ptr().cast(), buf.len() as u32, &mut written) };
    Some(hr >= 0 && u32::from_ne_bytes([buf[0], buf[1], buf[2], buf[3]]) != 0)
}

/// Whether Windows has a sound output device and a recording device, as waveOut and waveIn count
/// them. QEMU's DirectSound opens the default of each as it starts.
pub fn sound_devices() -> super::SoundDevices {
    use windows_sys::Win32::Media::Audio::{waveInGetNumDevs, waveOutGetNumDevs};
    // SAFETY: neither takes an argument.
    unsafe {
        super::SoundDevices {
            output: waveOutGetNumDevs() > 0,
            input: waveInGetNumDevs() > 0,
        }
    }
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

/// Removes the reims-ctl and reims-util that scripts/install-ctl.sh set aside beside this
/// reims-ctl as it installed new ones over them, where no process runs them any more
/// (`update::remove_set_aside`).
pub fn remove_set_aside_programs() {
    if let Some(bin) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        crate::update::remove_set_aside(&bin);
    }
}

/// What the interface draws with in the output mode of the console's screen buffer: line ends
/// and escape sequences acted on (`ENABLE_PROCESSED_OUTPUT`, `ENABLE_VIRTUAL_TERMINAL_PROCESSING`)
/// and lines wrapped at its edge (`ENABLE_WRAP_AT_EOL_OUTPUT`). The console wrote reims-ctl's line
/// ends and escape sequences as characters after an update's install script in a Windows
/// Terminal, and wrote past the line's end over its last column, as it does with these off.
pub const DRAWS_WITH: u32 = windows_sys::Win32::System::Console::ENABLE_PROCESSED_OUTPUT
    | windows_sys::Win32::System::Console::ENABLE_WRAP_AT_EOL_OUTPUT
    | windows_sys::Win32::System::Console::ENABLE_VIRTUAL_TERMINAL_PROCESSING;

/// The console this process is attached to, by `name`, `CONIN$` for its input or `CONOUT$` for
/// its screen buffer, whatever its standard handles are, with its mode; None where it has no
/// console. The handle is the caller's to close.
fn open_console(name: &str) -> Option<(windows_sys::Win32::Foundation::HANDLE, u32)> {
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Console::GetConsoleMode;
    let name = wide(name);
    // SAFETY: name is NUL-terminated; GetConsoleMode fails on a handle that is no console's, which
    // is closed here.
    unsafe {
        let h = CreateFileW(
            name.as_ptr(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if h == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut mode = 0;
        if GetConsoleMode(h, &mut mode) == 0 {
            CloseHandle(h);
            return None;
        }
        Some((h, mode))
    }
}

/// The console's modes, its input's and its screen buffer's output's, as [`keep_console_modes`]
/// found them, which it gives back to the console as it drops: a program run on the console can
/// end leaving modes of its own ([`DRAWS_WITH`]).
pub struct ConsoleModes(Vec<(windows_sys::Win32::Foundation::HANDLE, u32)>);

/// [`ConsoleModes`], from now; none where this process has no console.
pub fn keep_console_modes() -> ConsoleModes {
    ConsoleModes(
        ["CONIN$", "CONOUT$"]
            .into_iter()
            .filter_map(open_console)
            .collect(),
    )
}

impl Drop for ConsoleModes {
    fn drop(&mut self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Console::SetConsoleMode;
        for &(h, mode) in &self.0 {
            // SAFETY: h is a console handle open_console opened, closed once here.
            unsafe {
                SetConsoleMode(h, mode);
                CloseHandle(h);
            }
        }
    }
}

/// Turns on in the output mode of the console's screen buffer what the interface draws with
/// ([`DRAWS_WITH`]), which a program run on the console before may have left off; its other
/// flags stay. Nothing where this process has no console.
pub fn console_for_drawing() {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Console::SetConsoleMode;
    if let Some((h, mode)) = open_console("CONOUT$") {
        // SAFETY: h is the console handle open_console opened, closed once here.
        unsafe {
            SetConsoleMode(h, mode | DRAWS_WITH);
            CloseHandle(h);
        }
    }
}

/// Runs `cmd` in the console to its end. Ctrl+C and Ctrl+Break reach every process attached to
/// the console, so this one takes them as handled until `cmd` exits, and `cmd` gets them as it
/// would from a shell: a handler is not inherited, as ignoring them would be. The console gets
/// back the modes it had before `cmd` ([`keep_console_modes`]).
pub fn run_in_terminal(mut cmd: Command) -> io::Result<ExitStatus> {
    use windows_sys::Win32::System::Console::{
        CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler,
    };
    unsafe extern "system" fn handled(ctrl: u32) -> windows_sys::core::BOOL {
        (ctrl == CTRL_C_EVENT || ctrl == CTRL_BREAK_EVENT).into()
    }
    // SAFETY: handled is a function for the process's life, removed again below.
    unsafe { SetConsoleCtrlHandler(Some(handled), 1) };
    let modes = keep_console_modes();
    let status = cmd.status();
    drop(modes);
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

/// Whether another process holds the file `p` open so that it can be neither deleted nor
/// replaced: opened without FILE_SHARE_DELETE, as QEMU opens a drive's file
/// (block/file-win32.c), so that an open of it for delete access fails with a sharing violation.
/// A file that does not exist is held by no one.
pub fn held_against_delete(p: &Path) -> bool {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Foundation::ERROR_SHARING_VIOLATION;
    use windows_sys::Win32::Storage::FileSystem::{
        DELETE, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
    };
    let open = std::fs::OpenOptions::new()
        .access_mode(DELETE)
        .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
        .open(p);
    matches!(open, Err(e) if e.raw_os_error() == Some(ERROR_SHARING_VIOLATION as i32))
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

    /// A process that lives about `secs` seconds: cmd.exe's ping of this host.
    fn sleeper(secs: u32) -> Command {
        let cmd = std::env::var_os("ComSpec").unwrap_or_else(|| "cmd.exe".into());
        let mut c = Command::new(cmd);
        c.args(["/c", &format!("ping -n {} 127.0.0.1", secs + 1)]);
        c
    }

    /// The null device, as a detached process's log.
    fn null_log() -> std::fs::File {
        std::fs::OpenOptions::new().write(true).open("NUL").unwrap()
    }

    /// What the stdio test runs as reims-ctl: starts a detached process that lives about 5 s,
    /// as a start does QEMU, and returns.
    #[test]
    fn start_a_detached_process() {
        if std::env::var_os("REIMS_CTL_TEST_DETACH").is_some() {
            spawn_detached(sleeper(5), null_log()).unwrap();
        }
    }

    /// A detached process inherits no handle but its standard ones: the reader of a pipe whose
    /// write end its starter holds inheritable, as a starter's own starter can leave one, sees
    /// the pipe's end once the starter closes its end, while the detached process still runs.
    #[test]
    fn a_detached_process_inherits_no_other_handle() {
        use std::io::Read;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
        let (mut reader, writer) = std::io::pipe().unwrap();
        // SAFETY: the write end is open.
        let set = unsafe {
            SetHandleInformation(
                writer.as_raw_handle(),
                HANDLE_FLAG_INHERIT,
                HANDLE_FLAG_INHERIT,
            )
        };
        assert_ne!(set, 0, "{}", io::Error::last_os_error());
        spawn_detached(sleeper(10), null_log()).unwrap();
        drop(writer);
        let t = std::time::Instant::now();
        let mut got = Vec::new();
        reader.read_to_end(&mut got).unwrap();
        assert!(
            t.elapsed() < std::time::Duration::from_secs(5),
            "the pipe stayed open {:?}",
            t.elapsed()
        );
    }

    /// The variable that has [`echo_child`] write what it was given into the file it names.
    const ECHO: &str = "REIMS_CTL_TEST_ECHO";

    /// What [`a_detached_process_runs_with_its_commands_arguments_environment_and_directory`]
    /// runs detached: writes its arguments, a line each, then `REIMS_CTL_TEST_ECHO_VALUE`,
    /// whether `USERPROFILE` is set, and its directory into the file `REIMS_CTL_TEST_ECHO`
    /// names, through a file beside it, renamed once written; in a run of the tests, nothing.
    #[test]
    fn echo_child() {
        let Some(out) = std::env::var_os(ECHO) else {
            return;
        };
        let mut text: Vec<String> = std::env::args().skip(1).collect();
        text.push(std::env::var("REIMS_CTL_TEST_ECHO_VALUE").unwrap_or_default());
        text.push(std::env::var_os("USERPROFILE").is_some().to_string());
        text.push(std::env::current_dir().unwrap().display().to_string());
        let out = std::path::PathBuf::from(out);
        let part = out.with_extension("part");
        std::fs::write(&part, text.join("\n")).unwrap();
        std::fs::rename(&part, &out).unwrap();
    }

    /// A detached process runs with its command's arguments as given, spaces, quotes and
    /// backslashes in them included, its command's environment changes on this process's
    /// environment, a name in any case, in its command's directory, and with its output in the
    /// log.
    #[test]
    fn a_detached_process_runs_with_its_commands_arguments_environment_and_directory() {
        let dir = std::env::temp_dir().join(format!("reims-ctl-echo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let work = dir.join("work dir");
        std::fs::create_dir_all(&work).unwrap();
        let out = dir.join("echo.txt");
        let log_path = dir.join("echo.log");
        let given = [
            "plain",
            "two words",
            "",
            "a \"quoted\" word",
            r"C:\a\b\",
            r#"back\"slash"#,
            "tab\there",
        ];
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "os::windows::tests::echo_child"])
            .args(given)
            .env(ECHO, &out)
            .env("Reims_Ctl_Test_Echo_Value", "a value, with spaces")
            .env_remove("userprofile")
            .current_dir(&work);
        let log = std::fs::File::create(&log_path).unwrap();
        spawn_detached(c, log).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let done = || {
            std::fs::read_to_string(&log_path).is_ok_and(|l| l.contains("test result"))
                && out.exists()
        };
        while !done() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let text = std::fs::read_to_string(&out).unwrap_or_default();
        let log = std::fs::read_to_string(&log_path).unwrap_or_default();
        let _ = std::fs::remove_dir_all(&dir);
        let mut want: Vec<String> = ["--exact", "os::windows::tests::echo_child"]
            .into_iter()
            .chain(given)
            .map(String::from)
            .collect();
        want.extend([
            "a value, with spaces".to_string(),
            "false".to_string(),
            work.display().to_string(),
        ]);
        assert_eq!(text.split('\n').collect::<Vec<_>>(), want, "log: {log}");
        assert!(log.contains("1 passed"), "{log}");
    }

    /// A command line quotes the program, and an argument only where it is empty or holds a space
    /// or a tab, escaping its quotes and the backslashes before them and before its closing
    /// quote, as std's `Command` writes them.
    #[test]
    fn a_command_line_quotes_as_std_does() {
        let mut c = Command::new(r"C:\q\qemu-system-x86_64.exe");
        c.args([
            "-name",
            "",
            "a b",
            r"C:\x\",
            r"C:\x y\",
            "say \"hi\"",
            r#"a\"b"#,
        ]);
        let line = command_line(&c).unwrap();
        assert_eq!(line.last(), Some(&0));
        assert_eq!(
            String::from_utf16(&line[..line.len() - 1]).unwrap(),
            r#""C:\q\qemu-system-x86_64.exe" -name "" "a b" C:\x\ "C:\x y\\" "say \"hi\"" a\\\"b"#
        );
        let mut bad = Command::new("q");
        bad.arg("a\0b");
        assert_eq!(
            command_line(&bad).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
    }

    /// A command with no environment changes runs in this process's environment; with changes,
    /// its block holds this process's variables with them, each name once whatever its case,
    /// sorted by name in capitals, and ends in two NULs.
    #[test]
    fn an_environment_block_has_each_name_once() {
        assert_eq!(environment(&Command::new("q")).unwrap(), None);
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut c = Command::new("q");
        c.env("path", &path)
            .env("REIMS_CTL_TEST_NEW", "x")
            .env_remove("UserProfile");
        let block = environment(&c).unwrap().unwrap();
        assert_eq!(&block[block.len() - 2..], &[0, 0]);
        let entries: Vec<String> = block[..block.len() - 1]
            .split(|&u| u == 0)
            .filter(|e| !e.is_empty())
            .map(String::from_utf16_lossy)
            .collect();
        let names: Vec<String> = entries
            .iter()
            .map(|e| {
                let at = e[1..].find('=').map_or(e.len(), |i| i + 1);
                e[..at].to_uppercase()
            })
            .collect();
        assert_eq!(
            names.iter().filter(|n| *n == "PATH").count(),
            1,
            "{names:?}"
        );
        assert!(entries.contains(&format!("path={}", path.to_string_lossy())));
        assert!(entries.contains(&"REIMS_CTL_TEST_NEW=x".to_string()));
        assert!(!names.contains(&"USERPROFILE".to_string()), "{names:?}");
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
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

    /// Runs this binary's test `test` in a pseudoconsole of 80 by 25, as a terminal runs a
    /// program, with `arg` its argument, and waits at most `wait` for it to end, ending it then.
    /// Returns whether it ended in that time (`WAIT_OBJECT_0`), its exit code, and what the
    /// console wrote to the terminal, which answers each device attributes query (DA1) the
    /// console passes on, as a terminal does.
    fn in_pseudoconsole(test: &str, arg: &str, wait: std::time::Duration) -> (u32, u32, Vec<u8>) {
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
        let mut line = wide(&format!("\"{}\" --exact {test} \"{arg}\"", exe.display()));
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
            let waited = WaitForSingleObject(pi.hProcess, wait.as_millis() as u32);
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
        (waited, code, terminal.join().unwrap())
    }

    /// The answer a terminal sends to a query reaches [`read_terminal`] as the terminal sent it,
    /// escape sequence and all, under [`terminal_answers`]: in a pseudoconsole, whose console
    /// answers the device attributes query itself or passes it on to this test, which answers as
    /// a terminal does.
    #[test]
    fn a_terminals_answer_reaches_read_terminal_as_sent() {
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        let out = std::env::temp_dir().join(format!("reims-ctl-console-{}", std::process::id()));
        let (waited, code, seen) = in_pseudoconsole(
            "os::windows::tests::console_child",
            &format!("{CONSOLE_CHILD}{}", out.display()),
            std::time::Duration::from_secs(30),
        );
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

    /// The console's input mode and the output mode of its screen buffer, as this process's
    /// standard input and output name them: `IN OUT`, in hex.
    fn console_modes() -> String {
        use windows_sys::Win32::System::Console::{
            GetConsoleMode, GetStdHandle, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
        };
        let mode = |which| {
            let mut mode = 0;
            // SAFETY: GetStdHandle has no preconditions; GetConsoleMode fails on a handle that
            // is no console's.
            match unsafe { GetConsoleMode(GetStdHandle(which), &mut mode) } {
                0 => "-".to_string(),
                _ => format!("{mode:x}"),
            }
        };
        format!("{} {}", mode(STD_INPUT_HANDLE), mode(STD_OUTPUT_HANDLE))
    }

    /// The argument that has [`console_breaker`] change the console's modes.
    const BREAKER: &str = "reims-ctl-console-breaker";

    /// What the console-mode tests run as a program that ends leaving the console's modes
    /// changed: the output mode with [`DRAWS_WITH`] off, as the console was left after an
    /// update's install script in a Windows Terminal, and the input mode with line input, echo
    /// and Ctrl+C's processing off; in a run of the tests, nothing.
    #[test]
    fn console_breaker() {
        use windows_sys::Win32::System::Console::{
            ENABLE_ECHO_INPUT, ENABLE_LINE_INPUT, ENABLE_PROCESSED_INPUT, GetConsoleMode,
            GetStdHandle, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, SetConsoleMode,
        };
        if !std::env::args().any(|a| a == BREAKER) {
            return;
        }
        let off = ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT;
        let (mut input, mut output) = (0, 0);
        // SAFETY: GetStdHandle has no preconditions; the mode calls fail on a handle that is no
        // console's, which the asserts catch.
        unsafe {
            let (i, o) = (
                GetStdHandle(STD_INPUT_HANDLE),
                GetStdHandle(STD_OUTPUT_HANDLE),
            );
            assert_ne!(GetConsoleMode(i, &mut input), 0);
            assert_ne!(GetConsoleMode(o, &mut output), 0);
            assert_ne!(SetConsoleMode(i, input & !off), 0);
            assert_ne!(SetConsoleMode(o, output & !DRAWS_WITH), 0);
        }
    }

    /// [`console_breaker`], as a program.
    fn breaker() -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "os::windows::tests::console_breaker", BREAKER]);
        c
    }

    /// The argument that has [`modes_child`] run, followed by how and `=` and its file.
    const MODES_CHILD: &str = "reims-ctl-modes-child:";

    /// What the console-mode tests run in a pseudoconsole: [`breaker`] as reims-ctl runs a
    /// program, by how its argument names: `update`, through `update::run_on_console`, as an
    /// update runs the install script; `terminal`, through [`run_in_terminal`], as the interface
    /// runs less; `drawing`, alone, then [`console_for_drawing`], as the interface starts. Then
    /// the breaker alone, which leaves the modes changed. It writes the console's modes before,
    /// after the first run and after the second, a line each ([`console_modes`]), and whether the
    /// runs succeeded, into the file its argument names; in a run of the tests, nothing.
    #[test]
    fn modes_child() {
        let Some(arg) =
            std::env::args().find_map(|a| a.strip_prefix(MODES_CHILD).map(String::from))
        else {
            return;
        };
        let (how, out) = arg.split_once('=').unwrap();
        let before = console_modes();
        let first = match how {
            "update" => crate::update::run_on_console(&mut breaker()),
            "terminal" => run_in_terminal(breaker()),
            _ => {
                let s = breaker().status();
                console_for_drawing();
                s
            }
        };
        let after = console_modes();
        let second = breaker().status();
        let left = console_modes();
        let ran = |r: io::Result<ExitStatus>| r.is_ok_and(|s| s.success());
        let text = format!("{before}\n{after}\n{left}\n{} {}", ran(first), ran(second));
        std::fs::write(out, text).unwrap();
    }

    /// The console's modes as [`modes_child`] found them, run in a pseudoconsole as `how` names:
    /// before, after what it ran, and after the breaker alone, each input's then output's. Fails
    /// where a run failed, or where the breaker alone left the output mode with any of
    /// [`DRAWS_WITH`] or the input mode as it was, which would make the tests pass whatever
    /// reims-ctl does.
    fn modes_around(how: &str) -> [(u32, u32); 3] {
        use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
        let out =
            std::env::temp_dir().join(format!("reims-ctl-modes-{how}-{}", std::process::id()));
        let (waited, code, seen) = in_pseudoconsole(
            "os::windows::tests::modes_child",
            &format!("{MODES_CHILD}{how}={}", out.display()),
            std::time::Duration::from_secs(30),
        );
        let got = std::fs::read_to_string(&out).unwrap_or_default();
        let _ = std::fs::remove_file(&out);
        let console = String::from_utf8_lossy(&seen);
        assert_eq!(
            (waited, code),
            (WAIT_OBJECT_0, 0),
            "the console wrote {console:?}"
        );
        let lines: Vec<&str> = got.lines().collect();
        assert_eq!(
            lines.get(3),
            Some(&"true true"),
            "the child wrote {got:?}; the console wrote {console:?}"
        );
        let modes = |line: &str| {
            let hex = |m: &str| u32::from_str_radix(m, 16).ok();
            line.split_once(' ')
                .and_then(|(i, o)| Some((hex(i)?, hex(o)?)))
                .unwrap_or_else(|| panic!("no console modes in {line:?}"))
        };
        let m = [modes(lines[0]), modes(lines[1]), modes(lines[2])];
        assert_eq!(m[2].1 & DRAWS_WITH, 0, "the breaker left {:x?}", m[2]);
        assert_ne!(
            m[2].0, m[0].0,
            "the breaker left the input mode {:x}",
            m[0].0
        );
        m
    }

    /// An update gives the console back the modes it had before the program it ran there,
    /// before reims-ctl writes anything or starts the new reims-ctl: after an update's install
    /// script, a Windows Terminal's console wrote reims-ctl's line ends and escape sequences as
    /// characters.
    #[test]
    fn what_an_update_runs_leaves_the_console_in_the_modes_it_found() {
        let [before, after, _] = modes_around("update");
        assert_eq!(after, before, "the console's modes after, then before");
    }

    /// A program run in the terminal, as the interface runs less, leaves the console in the
    /// modes it found, in which the interface draws again.
    #[test]
    fn a_program_run_in_the_terminal_leaves_the_console_in_the_modes_it_found() {
        let [before, after, _] = modes_around("terminal");
        assert_eq!(after, before, "the console's modes after, then before");
    }

    /// The interface draws with line ends and escape sequences acted on and lines wrapped,
    /// whatever a program run before left off.
    #[test]
    fn the_interface_starts_drawing_with_what_a_program_left_off() {
        let [_, after, _] = modes_around("drawing");
        assert_eq!(
            after.1 & DRAWS_WITH,
            DRAWS_WITH,
            "the output mode {:x}",
            after.1
        );
    }
}
