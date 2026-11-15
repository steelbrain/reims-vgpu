//! The region's own memory on Linux and macOS: a private anonymous mapping.

/// `len` bytes of private anonymous memory, zeros, each page committed as it is first touched;
/// `None` where the host refuses the mapping.
pub(super) fn map(len: usize) -> Option<*mut u8> {
    // SAFETY: a fresh anonymous mapping, at an address the kernel picks.
    let base = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    (base != libc::MAP_FAILED).then_some(base.cast())
}

/// Unmaps what [`map`] made.
///
/// # Safety
///
/// `base` and `len` are a mapping [`map`] made, which nothing reaches after this.
pub(super) unsafe fn unmap(base: *mut u8, len: usize) {
    // SAFETY: the caller's mapping.
    unsafe { libc::munmap(base.cast(), len) };
}

/// The process's peak resident size in bytes: `ru_maxrss` is in bytes on macOS and in KiB on
/// Linux.
#[cfg(test)]
pub(super) fn peak_resident() -> u64 {
    // SAFETY: getrusage writes the struct it is given.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) }, 0);
    let maxrss = u64::try_from(usage.ru_maxrss).unwrap();
    if cfg!(target_os = "macos") {
        maxrss
    } else {
        maxrss * 1024
    }
}
