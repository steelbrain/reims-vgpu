//! What Linux and macOS share of a helper's watch: the host's memory, and the talk's broken
//! pipes.

/// The host's physical memory, in bytes: its pages times their size; `None` where the system does
/// not say.
pub(super) fn host_memory() -> Option<u64> {
    // SAFETY: sysconf reads a constant of the system's, and takes no pointer.
    let (pages, size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    u64::try_from(pages)
        .ok()?
        .checked_mul(u64::try_from(size).ok()?)
}

/// Blocks `SIGPIPE` on the calling thread, the talk's, so its write to a helper that has ended
/// fails with `EPIPE`, and the signal, pending on the thread alone, goes with it: the host's
/// process need not ignore it.
pub(super) fn mask_broken_pipes() {
    // SAFETY: the set is initialised by sigemptyset before sigaddset and pthread_sigmask read it,
    // and the old mask is not asked for.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGPIPE);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
}
