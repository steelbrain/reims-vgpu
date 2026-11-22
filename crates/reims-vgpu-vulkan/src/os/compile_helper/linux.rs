//! A helper's watch on Linux: its footprint from `/proc`, `PR_SET_PDEATHSIG`, which ends it with
//! the thread that started it, and the host's free memory from `/proc/meminfo` and the memory caps
//! of the process's cgroups.

use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

pub(super) use super::unix::{
    host_cpus, host_memory, mask_broken_pipes, no_core_files, unresident,
};

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
    kib_of(status, field).unwrap_or(0)
}

/// The value of `field`, in KiB, in a `/proc` file of `field: value kB` lines.
fn kib_of(text: &str, field: &str) -> Option<u64> {
    text.lines()
        .find_map(|line| line.strip_prefix(field))
        .and_then(|value| value.trim().trim_end_matches("kB").trim().parse().ok())
}

/// Where the cgroup v2 hierarchy is mounted.
const CGROUPS: &str = "/sys/fs/cgroup";

/// What the host can give this process now without swapping, in bytes: `MemAvailable` of
/// `/proc/meminfo`, and no more than the least headroom of the memory-capped cgroups the process
/// is in ([`cgroup_room`]), whose cap the kernel enforces by killing; `None` where `/proc/meminfo`
/// does not say.
pub(super) fn free_memory() -> Option<u64> {
    let available = available_of(&std::fs::read_to_string("/proc/meminfo").ok()?)?;
    let own = std::fs::read_to_string("/proc/self/cgroup").ok();
    Some(match own.as_deref().and_then(own_cgroup) {
        Some(path) => available.min(cgroup_room(Path::new(CGROUPS), path)),
        None => available,
    })
}

/// The process's cgroup in the v2 hierarchy that a `/proc/self/cgroup` names (its `0::` line).
fn own_cgroup(text: &str) -> Option<&str> {
    text.lines().find_map(|line| line.strip_prefix("0::"))
}

/// The least headroom ([`headroom_of`]) of the memory-capped cgroups from `path`, under the
/// hierarchy's mount `root`, up to `root` itself; `u64::MAX` where none is capped.
fn cgroup_room(root: &Path, path: &str) -> u64 {
    let mut room = u64::MAX;
    let mut dir = root.join(path.trim_start_matches('/'));
    loop {
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).ok();
        if let (Some(max), Some(current), Some(stat)) = (
            read("memory.max"),
            read("memory.current"),
            read("memory.stat"),
        ) && let Some(level) = headroom_of(&max, &current, &stat)
        {
            room = room.min(level);
        }
        if dir == root || !dir.starts_with(root) || !dir.pop() {
            return room;
        }
    }
}

/// A cgroup's headroom, in bytes, from its `memory.max`, `memory.current` and `memory.stat`: its
/// cap less what it holds, where the inactive page cache (`inactive_file`), which the kernel
/// reclaims first, counts as free; `None` where it has no cap (`max`).
fn headroom_of(max: &str, current: &str, stat: &str) -> Option<u64> {
    let max: u64 = max.trim().parse().ok()?;
    let current: u64 = current.trim().parse().ok()?;
    let inactive_file = (stat.lines())
        .find_map(|line| line.strip_prefix("inactive_file "))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
    Some(max.saturating_sub(current.saturating_sub(inactive_file)))
}

/// The `MemAvailable` a `/proc/meminfo` names, in bytes.
fn available_of(meminfo: &str) -> Option<u64> {
    kib_of(meminfo, "MemAvailable:")?.checked_mul(1024)
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
    use super::{available_of, cgroup_room, footprint_of, headroom_of, own_cgroup};

    /// A helper's footprint counts what it has swapped out, as zram holds it, beside what is
    /// resident.
    #[test]
    fn a_footprint_counts_resident_and_swapped_memory() {
        let status = "Name:\thelper\nVmPeak:\t 9000000 kB\nVmRSS:\t  204800 kB\nRssAnon:\t  1 kB\n\
                      VmSwap:\t  102400 kB\n";
        assert_eq!(footprint_of(status), (204_800 + 102_400) * 1024);
        assert_eq!(footprint_of("Name:\thelper\nState:\tZ (zombie)\n"), 0);
    }

    /// The host's free memory is `MemAvailable`, which counts reclaimable caches beside the free
    /// pages; a meminfo without it says nothing.
    #[test]
    fn the_hosts_free_memory_is_mem_available() {
        let meminfo = "MemTotal:       32768000 kB\nMemFree:         1024000 kB\n\
                       MemAvailable:   19660800 kB\nBuffers:          102400 kB\n";
        assert_eq!(available_of(meminfo), Some(19_660_800 * 1024));
        assert_eq!(available_of("MemTotal:       32768000 kB\n"), None);
    }

    /// A cgroup's headroom is its cap less what it holds, with the inactive page cache free; one
    /// with no cap has none to say.
    #[test]
    fn a_cgroups_headroom_is_its_cap_less_what_it_holds_but_inactive_cache() {
        let stat = "anon 6442450944\nfile 2147483648\nactive_file 1073741824\n\
                    inactive_file 536870912\n";
        let gib = 1u64 << 30;
        assert_eq!(
            headroom_of("25769803776\n", "17179869184\n", stat),
            Some(24 * gib - 16 * gib + gib / 2)
        );
        assert_eq!(headroom_of("max\n", "17179869184\n", stat), None);
        assert_eq!(headroom_of("1073741824\n", "2147483648\n", ""), Some(0));
    }

    /// The process's cgroup is the v2 line of `/proc/self/cgroup`, and the room under cgroups is
    /// the least headroom of the capped ones from it up to the hierarchy's root.
    #[test]
    fn the_room_under_cgroups_is_the_least_headroom_up_to_the_root() {
        assert_eq!(
            own_cgroup("1:name=systemd:/x\n0::/user.slice/job.scope\n"),
            Some("/user.slice/job.scope")
        );
        assert_eq!(own_cgroup("4:memory:/x\n"), None);
        let pid = std::process::id();
        let root = std::env::temp_dir().join(format!("rvg-cgroups-{pid}"));
        let slice = root.join("user.slice");
        let job = slice.join("job.scope");
        std::fs::create_dir_all(&job).unwrap();
        let write = |dir: &std::path::Path, max: &str, current: &str| {
            std::fs::write(dir.join("memory.max"), max).unwrap();
            std::fs::write(dir.join("memory.current"), current).unwrap();
            std::fs::write(dir.join("memory.stat"), "inactive_file 0\n").unwrap();
        };
        assert_eq!(cgroup_room(&root, "/user.slice/job.scope"), u64::MAX);
        write(&job, "max\n", "100\n");
        write(&slice, "1000\n", "400\n");
        assert_eq!(cgroup_room(&root, "/user.slice/job.scope"), 600);
        write(&job, "500\n", "300\n");
        assert_eq!(cgroup_room(&root, "/user.slice/job.scope"), 200);
        assert_eq!(cgroup_room(&root, "/user.slice"), 600);
        std::fs::remove_dir_all(&root).unwrap();
    }
}
