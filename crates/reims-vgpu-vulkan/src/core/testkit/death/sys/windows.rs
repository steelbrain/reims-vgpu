//! Windows's calls the harness uses, under the names the Unix modules give theirs where they mean
//! the same. A death's record carries the Unix number of the signal that would have ended the
//! child there ([`SIGABRT`] and the rest), so a report reads the same on every OS.
//!
//! - **An abort** is a fast fail on Windows (`std::process::abort` is `__fastfail`), which no
//!   handler of the process sees: core's death marks itself on the dying thread before it aborts
//!   (`testkit::death::dying`), and the child's exit status, `STATUS_STACK_BUFFER_OVERRUN`, is the
//!   death of [`SIGABRT`] ([`exit_signal`]).
//! - **A fault** reaches the process's unhandled-exception filter on the faulting thread, once no
//!   handler took it ([`arm`]): the filter marks the death and lets the exception end the process
//!   as it would have, with the system's error dialog turned off.

use std::ffi::c_int;
use std::fs::File;
use std::os::windows::io::AsRawHandle as _;

use windows_sys::Win32::Foundation::{CloseHandle, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    EXCEPTION_POINTERS, SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SetErrorMode,
    SetUnhandledExceptionFilter,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, GetCurrentThreadId, GetThreadDescription, TerminateProcess,
};

pub const SIGILL: c_int = 4;
pub const SIGTRAP: c_int = 5;
pub const SIGABRT: c_int = 6;
pub const SIGBUS: c_int = 7;
pub const SIGSEGV: c_int = 11;

/// The exception filter's answer that lets the exception go on to end the process.
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// The Unix signal a child ended by the exception or status `code` would have died of, where the
/// harness marks one: a fast fail (an abort), an access or paging fault, the stack's overflow, an
/// illegal or privileged instruction, a breakpoint.
fn signal_of(code: u32) -> Option<c_int> {
    match code {
        0xC000_0409 => Some(SIGABRT),
        0xC000_0005 | 0xC000_00FD => Some(SIGSEGV),
        0xC000_0006 => Some(SIGBUS),
        0xC000_001D | 0xC000_0096 => Some(SIGILL),
        0x8000_0003 => Some(SIGTRAP),
        _ => None,
    }
}

/// The signal a child's exit status says it died of ([`signal_of`] its exit code), as a Unix
/// status's `signal()` says it; `None` for an exit.
pub fn exit_signal(status: std::process::ExitStatus) -> Option<c_int> {
    status
        .code()
        .and_then(|code| signal_of(code.cast_unsigned()))
}

/// The calling thread's system-wide id (`GetCurrentThreadId`).
pub fn current_thread_id() -> u64 {
    // SAFETY: no precondition.
    u64::from(unsafe { GetCurrentThreadId() })
}

/// The calling thread's name (`GetThreadDescription`), as UTF-8 in `name`, cut at its end; the
/// length written.
pub fn thread_name(name: &mut [u8]) -> usize {
    let mut wide: *mut u16 = std::ptr::null_mut();
    // SAFETY: the running thread's pseudo-handle, and a live out-parameter.
    let ok = unsafe { GetThreadDescription(GetCurrentThread(), &mut wide) };
    if ok < 0 || wide.is_null() {
        return 0;
    }
    let mut units = Vec::new();
    loop {
        // SAFETY: the system's NUL-terminated string, read up to and including its NUL.
        let unit = unsafe { wide.add(units.len()).read() };
        if unit == 0 {
            break;
        }
        units.push(unit);
    }
    let mut len = 0;
    let mut buf = [0u8; 4];
    for c in char::decode_utf16(units).map(|c| c.unwrap_or('?')) {
        let bytes = c.encode_utf8(&mut buf).as_bytes();
        let Some(dst) = name.get_mut(len..len + bytes.len()) else {
            break;
        };
        dst.copy_from_slice(bytes);
        len += bytes.len();
    }
    // SAFETY: the string `GetThreadDescription` allocated, freed once.
    unsafe { LocalFree(wide.cast()) };
    len
}

/// Ends the process at once with `status`, running nothing of its own (`TerminateProcess` of
/// itself), as `_exit` does on Unix.
pub fn _exit(status: c_int) -> ! {
    // SAFETY: the current process's pseudo-handle; the call does not return.
    unsafe { TerminateProcess(GetCurrentProcess(), status.cast_unsigned()) };
    loop {
        std::thread::park();
    }
}

/// Nothing on Windows: the filter runs on the faulting thread's own stack, which Windows keeps a
/// guarantee of for an overflow's handling.
pub fn widen_alternate_stack() {}

/// A shared, writable view of `file`'s first `len` bytes (`CreateFileMapping`, `MapViewOfFile`),
/// whose writes reach the file after the process ends however it ends, or null.
pub fn map_shared(file: &File, len: usize) -> *mut u8 {
    // SAFETY: a mapping of the open file at its current size, with no security attributes or name.
    let mapping = unsafe {
        CreateFileMappingW(
            file.as_raw_handle(),
            std::ptr::null(),
            PAGE_READWRITE,
            0,
            0,
            std::ptr::null(),
        )
    };
    if mapping.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: a view of the mapping just made, at an address the system picks; it keeps the
    // mapping alive once the handle closes, and is never unmapped.
    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, len) };
    // SAFETY: the mapping's handle, closed once.
    unsafe { CloseHandle(mapping) };
    view.Value.cast()
}

/// What [`arm`]'s filter calls with the signal of a fault that ends the child.
static ON_DEATH: std::sync::OnceLock<fn(c_int)> = std::sync::OnceLock::new();

/// The process's unhandled-exception filter in a child: marks the death of a fault the harness
/// marks ([`signal_of`]) and lets the exception go on.
unsafe extern "system" fn on_unhandled(info: *const EXCEPTION_POINTERS) -> i32 {
    // SAFETY: the system's record of the exception, live while the filter runs.
    let code = unsafe { info.as_ref().and_then(|i| i.ExceptionRecord.as_ref()) }
        .map(|r| r.ExceptionCode.cast_unsigned());
    if let (Some(sig), Some(on_death)) = (code.and_then(signal_of), ON_DEATH.get()) {
        on_death(sig);
    }
    EXCEPTION_CONTINUE_SEARCH
}

/// Installs the child's unhandled-exception filter, which calls `on_death` on the faulting thread
/// with the signal of a fault that ends the child, and turns off the system's error dialogs, so a
/// fault ends the child at once.
pub fn arm(on_death: fn(c_int)) {
    let _ = ON_DEATH.set(on_death);
    // SAFETY: process-wide settings; the filter is a function of the right type that lives as long
    // as the process.
    unsafe {
        SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
        SetUnhandledExceptionFilter(Some(on_unhandled));
    }
}
