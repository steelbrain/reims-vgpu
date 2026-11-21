//! A helper's watch on macOS: its physical footprint from `proc_pid_rusage`, the helper's watch
//! of its parent's exit, as macOS has no signal at a parent's death, and the host's free memory
//! from `host_statistics64`.

use std::io;
use std::process::{Child, Command};
use std::sync::OnceLock;

pub(super) use super::unix::{host_cpus, host_memory, mask_broken_pipes, unresident};

/// What the host can give now, in bytes: its free pages less the speculative ones, which the
/// file-backed count holds, its file-backed pages and its purgeable pages (`host_statistics64`'s
/// `HOST_VM_INFO64`); `None` where it does not say.
pub(super) fn free_memory() -> Option<u64> {
    // The host port, asked once: each ask adds a reference to it.
    static HOST: OnceLock<mach2::port::mach_port_t> = OnceLock::new();
    // SAFETY: mach_host_self takes nothing.
    let host = *HOST.get_or_init(|| unsafe { mach2::mach_init::mach_host_self() });
    // SAFETY: a vm_statistics64 is plain data, which zero bytes make a valid one.
    let mut stats: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: `stats` is a vm_statistics64, the flavor HOST_VM_INFO64 fills, of `count` integers.
    let read = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &mut count,
        )
    };
    if read != libc::KERN_SUCCESS {
        return None;
    }
    let pages = u64::from(stats.free_count)
        .saturating_sub(u64::from(stats.speculative_count))
        .saturating_add(u64::from(stats.external_page_count))
        .saturating_add(u64::from(stats.purgeable_count));
    pages.checked_mul(super::unix::page_size()?)
}

/// Nothing of the child's: the helper watches its parent itself ([`watch_parent`]).
pub(super) fn prepare(_command: &mut Command) {}

/// Ends the helper at once when its parent, the library's process, exits: a thread of the
/// helper's waits for the parent's `NOTE_EXIT` on a kqueue. A parent that has already exited ends
/// the helper here.
pub(super) fn watch_parent() -> io::Result<()> {
    // SAFETY: getppid takes nothing.
    let parent = unsafe { libc::getppid() };
    // launchd adopted the helper: its parent has already exited.
    if parent == 1 {
        exit_now();
    }
    // SAFETY: kqueue takes nothing; its descriptor stays the watch's for the process's life.
    let queue = unsafe { libc::kqueue() };
    if queue < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a kevent is plain data, which zero bytes make a valid one.
    let mut change: libc::kevent = unsafe { std::mem::zeroed() };
    change.ident = usize::try_from(parent).map_err(io::Error::other)?;
    change.filter = libc::EVFILT_PROC;
    change.flags = libc::EV_ADD | libc::EV_ONESHOT;
    change.fflags = libc::NOTE_EXIT;
    // SAFETY: one change from a valid kevent, and no events asked for.
    let added =
        unsafe { libc::kevent(queue, &change, 1, std::ptr::null_mut(), 0, std::ptr::null()) };
    if added < 0 {
        let e = io::Error::last_os_error();
        // The parent exited between the two calls.
        if e.raw_os_error() == Some(libc::ESRCH) {
            exit_now();
        }
        return Err(e);
    }
    std::thread::Builder::new()
        .name("reims-vgpu.compile-helper-parent".to_owned())
        .spawn(move || {
            // SAFETY: as `change` above.
            let mut event: libc::kevent = unsafe { std::mem::zeroed() };
            loop {
                // SAFETY: no changes, and room for one event in `event`.
                let n = unsafe {
                    libc::kevent(queue, std::ptr::null(), 0, &mut event, 1, std::ptr::null())
                };
                let interrupted =
                    n < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted;
                // The parent's exit, or a watch that can no longer see it: either way the helper
                // ends, as no one would end it later.
                if n != 0 && !interrupted {
                    exit_now();
                }
            }
        })?;
    Ok(())
}

/// Ends the helper's process at once, with no atexit handler: a driver's destructors must not run
/// while its compiler may still be working on another thread.
fn exit_now() -> ! {
    // SAFETY: _exit ends the process; it takes no pointer.
    unsafe { libc::_exit(1) }
}

/// A running helper's physical footprint.
#[derive(Debug)]
pub(super) struct Watched {
    pid: libc::pid_t,
}

impl Watched {
    pub(super) fn new(child: &Child) -> io::Result<Watched> {
        Ok(Watched {
            pid: libc::pid_t::try_from(child.id()).map_err(io::Error::other)?,
        })
    }

    /// Its physical footprint, in bytes (`ri_phys_footprint`).
    pub(super) fn footprint(&self) -> io::Result<u64> {
        // SAFETY: a rusage_info_v2 is plain data, which zero bytes make a valid one.
        let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a rusage_info_v2, the flavor RUSAGE_INFO_V2 fills.
        let read = unsafe {
            libc::proc_pid_rusage(self.pid, libc::RUSAGE_INFO_V2, (&raw mut info).cast())
        };
        if read != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(info.ri_phys_footprint)
    }
}

/// Whether the process `pid` exists.
#[cfg(test)]
pub(super) fn alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    // SAFETY: a null signal sends nothing; it only looks the process up.
    unsafe { libc::kill(pid, 0) == 0 }
}
