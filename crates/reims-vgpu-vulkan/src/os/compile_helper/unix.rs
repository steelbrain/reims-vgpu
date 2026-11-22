//! What Linux and macOS share of a helper's watch: the host's memory and CPUs, what of the guest's
//! RAM is not resident, the talk's broken pipes, and the helper's core-file limit.

use std::io;

use crate::core::num::{ToU64 as _, ToUsize as _};

/// The most of a range `mincore` is asked of at once: 1 GiB.
const MINCORE_CHUNK: u64 = 1 << 30;

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

/// The host's page size, in bytes; `None` where the system does not say.
pub(super) fn page_size() -> Option<u64> {
    // SAFETY: sysconf reads a constant of the system's, and takes no pointer.
    u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) })
        .ok()
        .filter(|&page| page > 0)
}

/// The host's logical CPUs online, whatever the process's affinity; `None` where the system does
/// not say.
pub(super) fn host_cpus() -> Option<u64> {
    // SAFETY: sysconf reads a value of the system's, and takes no pointer.
    u64::try_from(unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) }).ok()
}

/// How many bytes of `ranges` (each a host address and a length) are not resident in the host's
/// memory, by `mincore`, page by page: what the guest takes as it touches them. A range it cannot
/// read counts as resident, as does every range where the page size is not said.
pub(super) fn unresident(ranges: &[(u64, u64)]) -> u64 {
    let Some(page) = page_size() else {
        return 0;
    };
    let mut pages = Vec::new();
    let mut out = 0_u64;
    for &(host, length) in ranges {
        let end = host.saturating_add(length);
        // mincore takes a page-aligned start.
        let mut at = host - host % page;
        while at < end {
            let len = (end - at).min(MINCORE_CHUNK);
            pages.clear();
            pages.resize(len.div_ceil(page).to_usize(), 0_u8);
            let address = std::ptr::with_exposed_provenance_mut::<libc::c_void>(at.to_usize());
            // SAFETY: mincore only reads the residency of the pages from `address` for `len`
            // bytes, writing one byte a page into `pages`, which holds one a page; an unmapped
            // range fails with ENOMEM and writes nothing that counts.
            let read = unsafe { libc::mincore(address, len.to_usize(), pages.as_mut_ptr().cast()) };
            if read == 0 {
                let not = pages.iter().filter(|&&page| page & 1 == 0).count();
                out = out.saturating_add(not.to_u64().saturating_mul(page));
            }
            at += len;
        }
    }
    out
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

/// Sets the calling process's core-file limit (`RLIMIT_CORE`) to 0, its soft limit and its hard
/// one, so the process cannot raise it again: its crash writes no core file, and a `core_pattern`
/// pipe that honours the limit keeps none (`systemd-coredump` keeps no core of a process whose
/// limit is below a page).
pub(super) fn no_core_files() -> io::Result<()> {
    let none = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: setrlimit reads the one rlimit it is given; lowering a limit needs no privilege.
    if unsafe { libc::setrlimit(libc::RLIMIT_CORE, &none) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
