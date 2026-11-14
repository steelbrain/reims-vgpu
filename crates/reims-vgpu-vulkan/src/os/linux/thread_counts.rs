//! The calling thread's user-space instructions and cycles, which the cost tests count their
//! steps in: a busy host leaves them as they are, where CPU time moves run to run.

use std::cell::Cell;

thread_local! {
    static FDS: Cell<Option<[i32; 2]>> = const { Cell::new(None) };
}

/// `struct perf_event_attr` as far as the fields set here, at the size of its first version.
#[repr(C)]
#[derive(Default)]
struct Attr {
    kind: u32,
    size: u32,
    config: u64,
    sample: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup: u32,
    bp_type: u32,
    config1: u64,
}

/// A counting perf event of this thread's, of the hardware event `config`, in user space.
fn open(config: u64) -> i32 {
    let attr = Attr {
        kind: 0, // PERF_TYPE_HARDWARE
        size: size_of::<Attr>() as u32,
        config,
        // exclude_kernel (bit 5) and exclude_hv (bit 6).
        flags: (1 << 5) | (1 << 6),
        ..Attr::default()
    };
    // SAFETY: `attr` is a perf_event_attr of the size it names; pid 0 and cpu -1 count this
    // thread on any CPU.
    unsafe { libc::syscall(libc::SYS_perf_event_open, &attr, 0, -1, -1, 0) as i32 }
}

/// This thread's instructions and cycles so far, from two perf events of its own opened at its
/// first call; 0 where the host gives none.
pub(crate) fn now() -> [u64; 2] {
    let fds = FDS.with(|f| match f.get() {
        Some(fds) => fds,
        None => {
            let fds = [open(1), open(0)]; // INSTRUCTIONS, CPU_CYCLES
            f.set(Some(fds));
            fds
        }
    });
    fds.map(|fd| {
        let mut v = 0u64;
        // SAFETY: a counting event's read writes one `u64`.
        let n = unsafe { libc::read(fd, (&raw mut v).cast(), 8) };
        if fd < 0 || n != 8 { 0 } else { v }
    })
}
