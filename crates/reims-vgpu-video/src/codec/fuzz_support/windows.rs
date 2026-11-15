//! The fuzzers' harness on Windows: the CPU time from `GetThreadTimes`
//! and `GetProcessTimes`, and no abort report.

use std::time::Duration;

use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, GetProcessTimes, GetThreadTimes,
};

use super::Accounting;

/// [`Accounting::cpu_time`]: the kernel and user times Windows keeps for
/// the calling thread (`GetThreadTimes`) or for the process
/// (`GetProcessTimes`), together, which it counts in 100 ns units.
pub(super) fn cpu_time(accounting: Accounting) -> Duration {
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let [mut made, mut ended, mut kernel, mut user] = [zero; 4];
    // SAFETY: the calling thread's or process's pseudo-handle, and four
    // FILETIMEs the call writes, and nothing else.
    let read = unsafe {
        match accounting {
            Accounting::Thread => GetThreadTimes(
                GetCurrentThread(),
                &mut made,
                &mut ended,
                &mut kernel,
                &mut user,
            ),
            Accounting::Process => GetProcessTimes(
                GetCurrentProcess(),
                &mut made,
                &mut ended,
                &mut kernel,
                &mut user,
            ),
        }
    };
    assert_ne!(
        read,
        0,
        "the {accounting:?} CPU time could not be read: {}",
        std::io::Error::last_os_error()
    );
    let units =
        |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    Duration::from_nanos(
        units(kernel)
            .saturating_add(units(user))
            .saturating_mul(100),
    )
}

/// Nothing: on Windows, abort ends the process with a fast fail
/// (`__fastfail`), and a thread's stack overflow, once Rust's handler has
/// printed the thread, with the exception unhandled; neither raises a
/// signal a handler could catch. An input that takes the process past
/// [`HARD_LIMIT`](super::HARD_LIMIT) is still named, by the allocator
/// before it aborts.
pub(super) fn install_abort_report() {}
