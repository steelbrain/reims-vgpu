//! The `crate::os` items on macOS and Linux.

use anyhow::{Context, Result, bail};
use std::io;
use std::path::Path;
use std::process::{Command, ExitStatus};

pub use std::os::unix::fs::FileExt;
#[cfg(test)]
pub use std::os::unix::net::UnixListener;
pub use std::os::unix::net::UnixStream;

/// What a write to a closed pipe fails with: EPIPE.
pub const CLOSED_PIPE: i32 = libc::EPIPE;

/// Sets `p`'s permissions to `mode`.
pub fn set_mode(p: &Path, mode: u32) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode))
}

/// Lets the user read and write `p`, its other permissions kept.
pub fn make_writable(p: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(p)?.permissions();
    perm.set_mode(perm.mode() | 0o600);
    std::fs::set_permissions(p, perm)
}

/// The local time now, as strftime(3) writes it in `fmt`.
pub fn now_stamp(fmt: &str) -> String {
    // SAFETY: time and localtime_r fill the structs given; strftime writes at most buf's size.
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = [0u8; 64];
        let f = std::ffi::CString::new(fmt).unwrap();
        let n = libc::strftime(buf.as_mut_ptr().cast(), buf.len(), f.as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

/// The local date and time now.
pub fn local_now() -> super::LocalTime {
    // SAFETY: as in now_stamp.
    let tm = unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        tm
    };
    super::LocalTime {
        year: tm.tm_year + 1900,
        month: (tm.tm_mon + 1) as u32,
        day: tm.tm_mday as u32,
        hour: tm.tm_hour as u32,
        minute: tm.tm_min as u32,
        second: tm.tm_sec as u32,
    }
}

/// The host's RAM in MiB, where the host says.
#[cfg(target_os = "macos")]
pub fn memory_mib() -> Option<u64> {
    let mut v: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: hw.memsize is a u64.
    if unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut v as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } == 0
    {
        return Some(v / (1024 * 1024));
    }
    None
}

/// The host's RAM in MiB, where the host says.
#[cfg(not(target_os = "macos"))]
pub fn memory_mib() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map(|kib| kib / 1024)
}

/// The memory a new process can take now, in MiB, where the host says: `MemAvailable` in Linux's
/// `/proc/meminfo`. A Mac says nothing.
pub fn memory_room_mib() -> Option<u64> {
    mem_available_mib(Path::new("/proc/meminfo"))
}

/// `MemAvailable` in `file`, as `/proc/meminfo` gives it, in MiB; None where it isn't there.
fn mem_available_mib(file: &Path) -> Option<u64> {
    std::fs::read_to_string(file)
        .ok()?
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .map(|kib| kib / 1024)
}

/// The user's id.
pub fn uid() -> u32 {
    // SAFETY: getuid has no preconditions.
    unsafe { libc::getuid() }
}

/// The group that owns the file `m` describes.
pub fn gid(m: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    Some(m.gid())
}

/// When the inode of the file `m` describes last changed (its ctime): any write to the file, or
/// change of its times, sets it to the present, and no program can set it otherwise.
pub fn changed(m: &std::fs::Metadata) -> Option<std::time::SystemTime> {
    use std::os::unix::fs::MetadataExt;
    let secs = u64::try_from(m.ctime()).ok()?;
    let nanos = u32::try_from(m.ctime_nsec()).ok()?;
    Some(std::time::UNIX_EPOCH + std::time::Duration::new(secs, nanos))
}

/// Makes `d`, a directory for QMP sockets outside the VMs' own, the user's alone (0700) where it
/// is new, and refuses one that is not a directory, another user owns, or others may write into:
/// whoever could write there could put a socket of theirs in QMP's place and drive the VM. Some
/// of them are under `/tmp`, which every user writes into (`paths::short_runtime_dir`).
pub fn private_dir(d: &Path) -> Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(d)
        .with_context(|| format!("create {}", d.display()))?;
    let m = std::fs::symlink_metadata(d).with_context(|| format!("read {}", d.display()))?;
    if !m.is_dir() || m.uid() != uid() || m.mode() & 0o022 != 0 {
        bail!(
            "{} is not a directory of this user's that others cannot write into, so it holds no \
             QMP socket: remove it, or make it so",
            d.display()
        );
    }
    Ok(())
}

/// Makes the directory `d`, which must not exist, the user's alone (0700).
pub fn create_private_dir(d: &Path) -> io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(d)
}

/// Whether the file `m` describes is this user's.
pub fn owned_by_me(m: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    m.uid() == uid()
}

/// Whether this process may open the file `p` to read and write it.
pub fn may_read_write(p: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(f) = std::ffi::CString::new(p.as_os_str().as_bytes()) else {
        return false;
    };
    // SAFETY: f is a NUL-terminated path; access only reads it.
    unsafe { libc::access(f.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
}

/// Whether process `pid` runs: it exists, though it may be another user's.
pub fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only checks the pid.
    let r = unsafe { libc::kill(pid, 0) };
    r == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Whether no process `pid` exists.
pub fn gone(pid: i32) -> bool {
    // SAFETY: signal 0 sends nothing; it asks only whether the process exists.
    let r = unsafe { libc::kill(pid, 0) };
    r == -1 && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

/// Process `pid`'s command line, its arguments each ended by a NUL; None for a process that
/// has none to read.
#[cfg(target_os = "linux")]
pub fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let b = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    // A zombie's command line is empty.
    (!b.is_empty()).then_some(b)
}

/// Process `pid`'s command line, as `KERN_PROCARGS2` gives it: its argument count, its
/// executable's path, and its arguments, each ended by a NUL.
#[cfg(target_os = "macos")]
pub fn cmdline(pid: i32) -> Option<Vec<u8>> {
    let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
    let mut size: libc::size_t = 0;
    // SAFETY: a size query; mib names a valid sysctl.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    let mut buf = vec![0u8; size];
    // SAFETY: buf holds size bytes.
    if unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            3,
            buf.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    } != 0
    {
        return None;
    }
    buf.truncate(size);
    Some(buf)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn cmdline(_pid: i32) -> Option<Vec<u8>> {
    None
}

/// Whether process `pid` may be the QEMU that writes the pid file its command line names: any
/// process here, whose program [`cmdline`] does not tell apart.
pub fn may_be_qemu(_pid: i32) -> bool {
    true
}

/// Runs `cmd` detached, its standard input empty and its output and errors into `log`, and
/// returns once it runs: in a session of its own, so closing the terminal does not end it, and
/// reparented away from this process by a double fork, so a process that stays open never holds
/// it as a zombie once it exits.
pub fn spawn_detached(mut cmd: Command, log: std::fs::File) -> io::Result<()> {
    use std::os::unix::process::CommandExt;
    cmd.stdin(std::process::Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // SAFETY: only async-signal-safe calls between fork and exec: setsid, fork, _exit.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(io::Error::last_os_error());
            }
            match libc::fork() {
                -1 => Err(io::Error::last_os_error()),
                0 => Ok(()),
                _ => libc::_exit(0),
            }
        });
    }
    let mut child = cmd.spawn()?;
    let _ = child.wait();
    Ok(())
}

/// How long a stop waits for QEMU to end after QMP's `quit` before it ends it (`ENDS`): the 10 s
/// floor, which covers twice [`QEMU_EXIT_MEASURED`] ([`super::follows_the_measured_exit`]).
pub const QUIT_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// The longest QEMU took to end on Linux, from the guest's shutdown to the process's end.
/// **Measured** on a Radeon 780M, 2026-11-20: 0.44 s, a macOS 26.7.2 guest stopped after Safari
/// ran a WebGL page. Not measured on macOS hosts.
pub const QEMU_EXIT_MEASURED: std::time::Duration = std::time::Duration::from_millis(440);

const _: () = assert!(super::follows_the_measured_exit(
    QUIT_WAIT,
    QEMU_EXIT_MEASURED
));

/// The ways a stop ends a QEMU that QMP's `quit` did not, in order, each with what the stop says
/// once QEMU is gone: SIGTERM, then SIGKILL.
pub const ENDS: &[super::End] = &[
    (sigterm, "powered off (SIGTERM)"),
    (sigkill, "killed (SIGKILL)"),
];

fn sigterm(pid: i32) {
    // SAFETY: kill only sends the signal; pid is a QEMU's, checked by runtime::pid.
    unsafe { libc::kill(pid, libc::SIGTERM) };
}

fn sigkill(pid: i32) {
    // SAFETY: as in sigterm.
    unsafe { libc::kill(pid, libc::SIGKILL) };
}

/// Runs `cmd` in this process's place; returns only why it could not.
pub fn exec(mut cmd: Command) -> io::Error {
    use std::os::unix::process::CommandExt;
    cmd.exec()
}

/// The command that runs the install script `script`, scripts/install-ctl.sh: `bash`, from PATH,
/// as its `#!/usr/bin/env bash` would. The install records no shell here.
pub fn install_script(_shell: &Path, script: &Path) -> Result<Command> {
    let mut c = Command::new("bash");
    c.arg(script);
    Ok(c)
}

/// Both: the backend [`crate::qemu::sound_routing`] routes by the host's devices, DirectSound, is
/// Windows's.
pub fn sound_devices() -> super::SoundDevices {
    super::SoundDevices {
        output: true,
        input: true,
    }
}

/// Nothing: on Windows this keeps an MSYS2 program from expanding wildcards in the arguments
/// reims-ctl gives it, which a Unix program never does itself.
pub fn no_glob(_c: &mut Command) {}

/// Nothing: macOS and Linux replace a running program's file like any other, and the installer
/// sets no program aside here (`update::remove_set_aside`).
pub fn remove_set_aside_programs() {}

/// Runs `cmd` in the terminal to its end, as system(3) runs a command: Ctrl+C and Ctrl+\ reach
/// the terminal's whole foreground process group, so this process ignores them until `cmd`
/// exits, and `cmd` gets them as it would from a shell.
pub fn run_in_terminal(mut cmd: Command) -> io::Result<ExitStatus> {
    use std::os::unix::process::CommandExt;
    // SAFETY: only signal, which is async-signal-safe, between fork and exec.
    unsafe {
        cmd.pre_exec(|| {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            Ok(())
        });
    }
    // SAFETY: dispositions set to ignore, then back to what they were.
    let (int, quit) = unsafe {
        (
            libc::signal(libc::SIGINT, libc::SIG_IGN),
            libc::signal(libc::SIGQUIT, libc::SIG_IGN),
        )
    };
    let status = cmd.status();
    unsafe {
        libc::signal(libc::SIGINT, int);
        libc::signal(libc::SIGQUIT, quit);
    }
    status
}

/// Discards what the terminal sent on stdin that was not read yet.
pub fn flush_input() {
    // SAFETY: tcflush only discards stdin's queued input.
    unsafe { libc::tcflush(0, libc::TCIFLUSH) };
}

/// While it lives, [`read_terminal`] reads what the terminal sends as the terminal sent it, as
/// stdin always gives it here.
pub struct TerminalAnswers;

/// [`TerminalAnswers`], from now.
pub fn terminal_answers() -> TerminalAnswers {
    TerminalAnswers
}

/// Reads what the terminal on stdin sends into `buf`, waiting at most `timeout` for it: Ok(0)
/// where it sent nothing in that time, or stdin ended or could not be read; the wait's error,
/// [`io::ErrorKind::Interrupted`] where a signal cut it short.
pub fn read_terminal(buf: &mut [u8], timeout: std::time::Duration) -> io::Result<usize> {
    let mut pfd = libc::pollfd {
        fd: 0,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one pollfd, for stdin.
    let n = unsafe { libc::poll(&mut pfd, 1, timeout.as_millis().max(1) as libc::c_int) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    if n == 0 {
        return Ok(0);
    }
    // SAFETY: buf holds its length in bytes.
    let r = unsafe { libc::read(0, buf.as_mut_ptr().cast(), buf.len()) };
    Ok(r.max(0) as usize)
}

/// The space the file `m` describes takes on its file system: its 512-byte blocks, fewer than
/// its length where it is sparse.
pub fn allocated(m: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt as _;
    m.blocks() * 512
}

/// Whether `a` and `b`, which exist, are on one file system: the same device.
pub fn same_file_system(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev(),
        _ => false,
    }
}

/// Whether another process holds the file `p` open so that it can be neither deleted nor
/// replaced: never here, where a file's name is removed or replaced while processes hold it
/// open, and they keep the file they opened.
pub fn held_against_delete(_p: &Path) -> bool {
    false
}

/// The flag [`set_at_interrupt`] gave, for its signal handler.
static INTERRUPT_FLAG: std::sync::atomic::AtomicPtr<std::sync::atomic::AtomicBool> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// From now, Ctrl-C (SIGINT) and SIGTERM set `flag`, and end the process no more.
pub fn set_at_interrupt(flag: &'static std::sync::atomic::AtomicBool) {
    use std::sync::atomic::Ordering;
    extern "C" fn stop(_: libc::c_int) {
        // An atomic load and store are all a signal handler may do here.
        let flag = INTERRUPT_FLAG.load(Ordering::SeqCst);
        // SAFETY: a null pointer, or the &'static flag set_at_interrupt stored.
        if let Some(flag) = unsafe { flag.as_ref() } {
            flag.store(true, Ordering::SeqCst);
        }
    }
    INTERRUPT_FLAG.store(std::ptr::from_ref(flag).cast_mut(), Ordering::SeqCst);
    let handler = stop as extern "C" fn(libc::c_int) as libc::sighandler_t;
    // SAFETY: the handler only loads and stores atomics.
    unsafe {
        libc::signal(libc::SIGINT, handler);
        libc::signal(libc::SIGTERM, handler);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The memory a Linux host has available is read from `MemAvailable` as the kernel writes it.
    #[test]
    fn available_memory_is_read_from_meminfo() {
        let env = crate::testenv::Env::new("meminfo");
        let file = env.root.join("meminfo");
        assert_eq!(mem_available_mib(&file), None);
        std::fs::write(
            &file,
            "MemTotal:       32000000 kB\nMemFree:         1000000 kB\nMemAvailable:    3145728 kB\n",
        )
        .unwrap();
        assert_eq!(mem_available_mib(&file), Some(3072));
    }
}
