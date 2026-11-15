//! The region's own memory on Windows: committed private memory (`VirtualAlloc`). Its pages are
//! zeros, and each takes a physical page only as it is first touched, as Linux's and macOS's
//! anonymous mappings do; Windows charges the whole length to the system's commit limit at once.

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

/// `len` bytes of committed private memory, zeros, each page made resident as it is first
/// touched; `None` where the host refuses it.
pub(super) fn map(len: usize) -> Option<*mut u8> {
    // SAFETY: a fresh allocation, at an address the system picks.
    let base = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            len,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    (!base.is_null()).then_some(base.cast())
}

/// Frees what [`map`] made.
///
/// # Safety
///
/// `base` is an allocation [`map`] made, which nothing reaches after this.
pub(super) unsafe fn unmap(base: *mut u8, _len: usize) {
    // SAFETY: the caller's allocation, released whole (a release takes a length of 0).
    unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
}

/// The process's peak working set in bytes.
#[cfg(test)]
pub(super) fn peak_resident() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: the counters are plain data the call fills, given their size.
    let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    let size = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS>()).unwrap();
    counters.cb = size;
    assert_ne!(
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) },
        0
    );
    u64::try_from(counters.PeakWorkingSetSize).unwrap()
}
