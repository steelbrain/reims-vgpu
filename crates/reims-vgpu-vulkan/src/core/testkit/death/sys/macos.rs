//! Darwin's C library: the calls and constants the harness uses, the same on arm64 and x86-64.

use std::ffi::{c_char, c_int, c_void};

pub const PROT_READ: c_int = 1;
pub const PROT_WRITE: c_int = 2;
pub const MAP_SHARED: c_int = 1;
pub const MAP_FIXED: c_int = 0x10;
pub const SIGILL: c_int = 4;
pub const SIGTRAP: c_int = 5;
pub const SIGABRT: c_int = 6;
pub const SIGBUS: c_int = 10;
pub const SIGSEGV: c_int = 11;
pub const NSIG: usize = 32;
pub const SA_ONSTACK: c_int = 0x0001;
pub const SA_SIGINFO: c_int = 0x0040;
pub const SIG_DFL: usize = 0;
pub const SIG_IGN: usize = 1;

/// `struct sigaction`'s handler union.
#[repr(C)]
#[derive(Clone, Copy)]
pub union Handler {
    /// `sa_sigaction`, with `SA_SIGINFO`.
    pub action: extern "C" fn(c_int, *mut c_void, *mut c_void),
    /// `sa_handler`, without it.
    pub handler: extern "C" fn(c_int),
    /// `SIG_DFL`, `SIG_IGN`, or a handler's address.
    pub raw: usize,
}

/// Darwin's `struct sigaction` as `sigaction(2)` takes it: the handler, `sigset_t` (32
/// bits), the flags.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SigAction {
    pub handler: Handler,
    pub mask: u32,
    pub flags: c_int,
}

impl SigAction {
    /// The action `handler` with `flags` and no signal masked.
    pub const fn new(handler: Handler, flags: c_int) -> SigAction {
        SigAction {
            handler,
            mask: 0,
            flags,
        }
    }
}

unsafe extern "C" {
    pub fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    pub safe fn getpagesize() -> c_int;
    pub fn sigaction(sig: c_int, act: *const SigAction, old: *mut SigAction) -> c_int;
    pub safe fn raise(sig: c_int) -> c_int;
    pub safe fn pthread_self() -> *mut c_void;
    pub fn pthread_threadid_np(thread: *mut c_void, id: *mut u64) -> c_int;
    pub fn pthread_getname_np(thread: *mut c_void, name: *mut c_char, len: usize) -> c_int;
    pub safe fn _exit(status: c_int) -> !;
}

/// The calling thread's system-wide id (`pthread_threadid_np`), without allocating.
pub fn current_thread_id() -> u64 {
    let mut tid = 0u64;
    // SAFETY: `pthread_self()` is the calling thread, and the call writes one `u64`.
    unsafe { pthread_threadid_np(pthread_self(), &raw mut tid) };
    tid
}

/// Leaves the calling thread's alternate signal stack as std made it: Darwin's `SIGSTKSZ`,
/// 128 KiB, holds the harness's handler, std's stack-overflow handler it chains to, and the
/// handler of the `SIGABRT` that one raises, nested.
pub fn widen_alternate_stack() {}
