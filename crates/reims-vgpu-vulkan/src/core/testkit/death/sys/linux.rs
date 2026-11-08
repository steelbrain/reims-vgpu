//! glibc on Linux: the calls and constants the harness uses (x86-64's values).

use std::ffi::{c_char, c_int, c_void};

pub const PROT_READ: c_int = 1;
pub const PROT_WRITE: c_int = 2;
pub const MAP_SHARED: c_int = 1;
pub const MAP_PRIVATE: c_int = 2;
pub const MAP_FIXED: c_int = 0x10;
pub const MAP_ANONYMOUS: c_int = 0x20;
pub const SIGILL: c_int = 4;
pub const SIGTRAP: c_int = 5;
pub const SIGABRT: c_int = 6;
pub const SIGBUS: c_int = 7;
pub const SIGSEGV: c_int = 11;
/// glibc's `_NSIG`: signals 1 to 64.
pub const NSIG: usize = 65;
pub const SA_ONSTACK: c_int = 0x0800_0000;
pub const SA_SIGINFO: c_int = 0x0004;
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

/// glibc's `struct sigaction` as `sigaction(2)` takes it: the handler, `sigset_t` (1024 bits),
/// the flags, and the restorer, which glibc sets itself.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SigAction {
    pub handler: Handler,
    pub mask: [u64; 16],
    pub flags: c_int,
    pub restorer: usize,
}

impl SigAction {
    /// The action `handler` with `flags` and no signal masked.
    pub const fn new(handler: Handler, flags: c_int) -> SigAction {
        SigAction {
            handler,
            mask: [0; 16],
            flags,
            restorer: 0,
        }
    }
}

/// glibc's `stack_t`.
#[repr(C)]
pub struct StackT {
    pub sp: *mut c_void,
    pub flags: c_int,
    pub size: usize,
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
    /// `pthread_t` is an `unsigned long`, passed as a pointer-sized value.
    pub safe fn pthread_self() -> *mut c_void;
    pub fn pthread_getname_np(thread: *mut c_void, name: *mut c_char, len: usize) -> c_int;
    pub safe fn _exit(status: c_int) -> !;
    safe fn gettid() -> c_int;
    fn sigaltstack(stack: *const StackT, old: *mut StackT) -> c_int;
}

/// The calling thread's system-wide id (`gettid`), without allocating.
pub fn current_thread_id() -> u64 {
    u64::from(gettid().cast_unsigned())
}

/// The alternate signal stack [`widen_alternate_stack`] gives a thread: 128 KiB, Darwin's
/// `SIGSTKSZ`.
const ALTERNATE_STACK: usize = 0x2_0000;

/// Gives the calling thread an alternate signal stack of [`ALTERNATE_STACK`] bytes in place of
/// std's. std sizes its stack by glibc's `SIGSTKSZ`, 8 KiB, which the harness's handler overflows
/// when it chains to std's stack-overflow handler: that one aborts, and the handler of the
/// `SIGABRT` it raises runs nested on the same stack, each signal frame with the CPU's whole
/// register state. The process then dies of `SIGSEGV` instead of std's `SIGABRT`. std's own
/// stack stays mapped until it frees it at the thread's exit; this one is never unmapped.
pub fn widen_alternate_stack() {
    // SAFETY: a fresh private anonymous mapping at an address the kernel picks.
    let sp = unsafe {
        mmap(
            std::ptr::null_mut(),
            ALTERNATE_STACK,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if sp.addr() == usize::MAX || sp.is_null() {
        return;
    }
    let stack = StackT {
        sp,
        flags: 0,
        size: ALTERNATE_STACK,
    };
    // SAFETY: `stack` names the mapping above, which is never unmapped; the old stack is not
    // asked for.
    let _ = unsafe { sigaltstack(&raw const stack, std::ptr::null_mut()) };
}
