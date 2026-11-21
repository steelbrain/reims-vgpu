//! A helper's watch on Linux: its footprint from `/proc`, and `PR_SET_PDEATHSIG`, which ends it
//! with the thread that started it.

use std::io;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command};

pub(super) use super::unix::{host_memory, mask_broken_pipes};

/// Has the child the helper runs in ask for `SIGKILL` at its parent thread's end: the helper's
/// own thread, which lives until the process has ended, or the whole library's process.
pub(super) fn prepare(command: &mut Command) {
    let parent = libc::pid_t::try_from(std::process::id()).unwrap_or(0);
    let signal = libc::c_ulong::from(libc::SIGKILL.unsigned_abs());
    // SAFETY: the closure runs in the child between its fork and its exec, and calls only prctl
    // and getppid, which are async-signal-safe; its errors allocate nothing.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, signal) != 0 {
                return Err(io::Error::last_os_error());
            }
            // The library's process ended before the signal was asked for, so it never comes.
            if libc::getppid() != parent {
                return Err(io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// The helper's end comes with its parent's (`prepare`): nothing of the helper's is needed.
pub(super) fn watch_parent() -> io::Result<()> {
    Ok(())
}

/// A running helper's footprint, from its `/proc/<pid>/status`.
#[derive(Debug)]
pub(super) struct Watched {
    status: PathBuf,
}

impl Watched {
    pub(super) fn new(child: &Child) -> io::Result<Watched> {
        Ok(Watched {
            status: PathBuf::from(format!("/proc/{}/status", child.id())),
        })
    }

    /// Its resident and swapped memory, in bytes (`VmRSS` and `VmSwap`): a process that has ended
    /// and not been reaped has neither, 0.
    pub(super) fn footprint(&self) -> io::Result<u64> {
        Ok(footprint_of(&std::fs::read_to_string(&self.status)?))
    }
}

/// The resident and swapped memory a `/proc/<pid>/status` names, in bytes.
fn footprint_of(status: &str) -> u64 {
    kib(status, "VmRSS:")
        .saturating_add(kib(status, "VmSwap:"))
        .saturating_mul(1024)
}

/// The value of `field`, in KiB, in a `/proc/<pid>/status`; 0 where it is missing.
fn kib(status: &str, field: &str) -> u64 {
    status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|value| value.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

/// Whether the process `pid` runs: it has a `/proc` entry, and is not a zombie.
#[cfg(test)]
pub(super) fn alive(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        // The state follows the command's name, which ends at the line's last ')'.
        !stat
            .rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
    })
}

#[cfg(test)]
mod tests {
    use super::footprint_of;

    /// A helper's footprint counts what it has swapped out, as zram holds it, beside what is
    /// resident.
    #[test]
    fn a_footprint_counts_resident_and_swapped_memory() {
        let status = "Name:\thelper\nVmPeak:\t 9000000 kB\nVmRSS:\t  204800 kB\nRssAnon:\t  1 kB\n\
                      VmSwap:\t  102400 kB\n";
        assert_eq!(footprint_of(status), (204_800 + 102_400) * 1024);
        assert_eq!(footprint_of("Name:\thelper\nState:\tZ (zombie)\n"), 0);
    }
}
