//! The scheduling hints a thread that walks the guest's command buffers gives Linux
//! ([`hint_walk_thread`]): the kernel still places it, on any CPU it may run on.
//!
//! A walk thread of the graphics device's lanes walks a command buffer's ops one after another, so
//! wherever the GPU keeps up its speed is the frame rate; it gives itself these hints as it starts
//! (`Threads::give_walk_hints`). Linux offers an unprivileged thread hints about itself, none of
//! which binds it to a CPU:
//! - **A utilization floor** (`sched_setattr`'s `SCHED_FLAG_UTIL_CLAMP_MIN`, kernel docs
//!   "Utilization Clamping"): the thread asks to be treated as needing the CPU's full capacity,
//!   1024, while it runs, the most the floor can ask, as macOS's user-interactive QoS is the most
//!   a QoS asks. The kernel reads it in two places (`kernel/sched/fair.c`, read from source):
//!   - **Placement**, where the kernel runs capacity-aware scheduling, on CPUs of unequal
//!     capacity (`intel_pstate` turns it on for a hybrid processor without SMT, in its active
//!     mode too). A CPU fits the thread only where its capacity, less any thermal or frequency
//!     pressure, reaches the floor (`util_fits_cpu`), so a floor of 1024 fits only the
//!     highest-capacity CPUs, unpressured; every other CPU, the lesser performance cores among
//!     them, is a lesser fit. On a wake the kernel takes the first idle CPU the thread fits, else
//!     the idle CPU of the most capacity, else the CPU it would have taken anyway
//!     (`select_idle_capacity`). While the thread runs on a CPU it does not fit, the load
//!     balancer may move it to a CPU of more capacity with spare time (misfit migration), never
//!     to one of less, and never off a CPU of the most capacity it may use.
//!   - **Clocking**, under the `schedutil` governor: the CPU it runs on is clocked up while it
//!     is runnable. Under `intel_pstate`'s or `amd-pstate`'s own P-state selection (active mode)
//!     the CPU picks its own clock, and this one is lost.
//!
//!   On a hybrid processor with SMT, and on CPUs of one type, every CPU has the same capacity and
//!   placement reads nothing. Neither is a bound: where no CPU it fits is free, it runs where the
//!   kernel puts it.
//! - **What is not given.** A lower nice needs `CAP_SYS_NICE` or `RLIMIT_NICE`, and where the host
//!   sets one (CachyOS's `ananicy-cpp` gives QEMU nice 9) a lower one would undo its policy; so
//!   the thread keeps its policy and nice. EEVDF's own slice (`sched_runtime`) is a latency hint
//!   the kernel docs give no throughput for, and is left as it is. Affinity is never set: the
//!   kernel balances the threads of every process over every CPU.
//!
//! A thread inherits its creator's floor, so every other thread the library starts on Linux, a
//! queue's worker or a display window's thread, drops it as it starts ([`drop_inherited_hints`]):
//! a walk thread may be the one that restarts an idle queue's worker, and so may the runtime's
//! completion threads, which give themselves the same floor. `SCHED_FLAG_RESET_ON_FORK`
//! would not do it: `sched_setattr` takes the flag only with a policy, and the floor's request
//! keeps the thread's (`SCHED_FLAG_KEEP_POLICY`). The thread asks for the kernel's own default
//! floor (`sched_util_min` -1), which leaves a cgroup's `cpu.uclamp.min` as it applies to every
//! thread; a floor the host set on QEMU's own threads is not kept.
//!
//! Measured, as an unprivileged user, at nice 0 and 9: the kernel takes the floor and reads it
//! back on CachyOS's 7.2.9 and 7.2.8 kernels (i9-13900H; Core Ultra 9 275HX) and Ubuntu's 7.0.0
//! (Ryzen 7 7840HS), all built with `CONFIG_UCLAMP_TASK`, with the user's cgroup's
//! `cpu.uclamp.min` at 0 and `max` at `max` and `kernel.sched_util_clamp_min` at 1024. All three
//! run their CPUs' own P-state selection (`intel_pstate` and `amd-pstate-epp` in active mode).
//! Only the 275HX, a hybrid without SMT, runs capacity-aware scheduling (its `cpu_capacity` 1024,
//! 991 and 747, so its performance cores of 991 are a lesser fit); the i9-13900H, a hybrid with
//! SMT, reads 1024 on every CPU, and there a thread that ran WindowServer's commands one after
//! another spent 43-61% of a Finder drag on the efficiency cores with the floor or without it. A kernel
//! without `CONFIG_UCLAMP_TASK` refuses the call, and the thread runs as it would have. The first
//! hint's outcome is logged once; a dropped floor is not logged.

use std::sync::atomic::{AtomicBool, Ordering};

/// `SCHED_FLAG_KEEP_POLICY`: the thread keeps its policy.
const KEEP_POLICY: u64 = 0x08;
/// `SCHED_FLAG_KEEP_PARAMS`: the thread keeps its nice and its slice.
const KEEP_PARAMS: u64 = 0x10;
/// `SCHED_FLAG_UTIL_CLAMP_MIN`: the request sets the utilization floor.
const UTIL_CLAMP_MIN: u64 = 0x20;
/// The floor asked for: the CPU's full capacity (`SCHED_CAPACITY_SCALE`).
const FULL_CAPACITY: u32 = 1024;
/// A clamp of -1: the kernel's own default for the thread's policy, as a thread that asked for
/// none has.
const KERNEL_DEFAULT: u32 = u32::MAX;

/// `struct sched_attr` as `sched_setattr` takes it, with the utilization clamps
/// (`SCHED_ATTR_SIZE_VER1`, 56 bytes).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct SchedAttr {
    size: u32,
    sched_policy: u32,
    sched_flags: u64,
    sched_nice: i32,
    sched_priority: u32,
    sched_runtime: u64,
    sched_deadline: u64,
    sched_period: u64,
    sched_util_min: u32,
    sched_util_max: u32,
}

/// Gives the calling thread, a walk thread, its scheduling hints (the module docs), keeping its
/// policy, nice and CPUs. The first call logs what the kernel took.
pub(crate) fn hint_walk_thread() {
    let line = match set_util_floor(FULL_CAPACITY) {
        Ok(()) => format!(
            "walk threads ask the scheduler for a utilization floor of {FULL_CAPACITY} of 1024"
        ),
        Err(e) => {
            format!("walk threads give the scheduler no hint: sched_setattr refused it ({e})")
        }
    };
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("{line}");
    }
}

/// Gives the calling thread, one the library starts, the kernel's
/// default utilization floor in place of the one it inherited (the module docs), keeping its
/// policy, nice and CPUs. A kernel that refuses it leaves the thread as it started; one without
/// `CONFIG_UCLAMP_TASK` gave it no floor to inherit.
pub(crate) fn drop_inherited_hints() {
    let _ = set_util_floor(KERNEL_DEFAULT);
}

/// The request that sets the calling thread's utilization floor to `floor`, leaving its policy,
/// nice and slice as they are.
fn util_floor_request(floor: u32) -> SchedAttr {
    SchedAttr {
        size: u32::try_from(size_of::<SchedAttr>()).unwrap_or(u32::MAX),
        sched_flags: KEEP_POLICY | KEEP_PARAMS | UTIL_CLAMP_MIN,
        sched_util_min: floor,
        ..SchedAttr::default()
    }
}

/// Sets the calling thread's utilization floor to `floor` ([`util_floor_request`]).
fn set_util_floor(floor: u32) -> std::io::Result<()> {
    let request = util_floor_request(floor);
    // SAFETY: the calling thread's attributes, from a live `sched_attr` of the size it names.
    let rc = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, std::ptr::from_ref(&request), 0) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::core::platform::os::Threads;

    use super::super::LinuxThreads;
    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    /// The calling thread's attributes, as `sched_getattr` reads them.
    fn read_back() -> SchedAttr {
        let mut attr = SchedAttr::default();
        // SAFETY: the calling thread's attributes, into a live `sched_attr` of the size given.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_sched_getattr,
                0,
                std::ptr::from_mut(&mut attr),
                size_of::<SchedAttr>(),
                0,
            )
        };
        assert_eq!(rc, 0, "{}", std::io::Error::last_os_error());
        attr
    }

    /// The floor a walk thread reads back: the full capacity, or, on a kernel without
    /// `CONFIG_UCLAMP_TASK` (which has no `sched_util_clamp_min` sysctl), which refuses the hint,
    /// none.
    fn walk_floor() -> u32 {
        if std::path::Path::new("/proc/sys/kernel/sched_util_clamp_min").exists() {
            FULL_CAPACITY
        } else {
            0
        }
    }

    /// The request is the 56-byte `sched_attr` that sets the floor alone
    #[test]
    fn the_request_sets_only_the_floor() {
        assert_eq!(size_of::<SchedAttr>(), 56);
        for floor in [FULL_CAPACITY, KERNEL_DEFAULT] {
            let request = util_floor_request(floor);
            assert_eq!(request.size, 56);
            assert_eq!(
                request.sched_flags,
                KEEP_POLICY | KEEP_PARAMS | UTIL_CLAMP_MIN
            );
            assert_eq!(request.sched_util_min, floor);
        }
    }

    /// A walk thread runs with the floor, its policy, nice and CPUs its creator's
    #[test]
    fn a_walk_thread_keeps_its_policy_nice_and_cpus() {
        let creator = read_back();
        // SAFETY: a zeroed set, filled by the kernel for the calling thread.
        let mut creator_cpus: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        // SAFETY: the calling thread's mask, into a set of the size given.
        let rc =
            unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut creator_cpus) };
        assert_eq!(rc, 0);
        let (tx, rx) = mpsc::channel();
        LinuxThreads::new().spawn(
            "reims-vgpu.walk",
            512 * 1024,
            Box::new(move || {
                LinuxThreads::new().give_walk_hints();
                // SAFETY: a zeroed set, filled by the kernel for the calling thread.
                let mut cpus: libc::cpu_set_t = unsafe { std::mem::zeroed() };
                // SAFETY: the calling thread's mask, into a set of the size given.
                let rc =
                    unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut cpus) };
                assert_eq!(rc, 0);
                // SAFETY: two live sets.
                let same = unsafe { libc::CPU_EQUAL(&cpus, &creator_cpus) };
                tx.send((read_back(), same)).unwrap();
            }),
        );
        let (attr, same_cpus) = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(attr.sched_util_min, walk_floor());
        assert_eq!(attr.sched_policy, creator.sched_policy);
        assert_eq!(attr.sched_nice, creator.sched_nice);
        assert_eq!(attr.sched_runtime, creator.sched_runtime);
        assert!(same_cpus);
    }

    /// A queue's worker and a named thread, each started from a walk thread, run with the
    /// kernel's default floor and their creator's policy and nice, while the walk thread keeps
    /// its floor
    #[test]
    fn a_thread_started_from_a_walk_thread_drops_its_floor() {
        let (tx, rx) = mpsc::channel();
        LinuxThreads::new().spawn(
            "reims-vgpu.walk",
            512 * 1024,
            Box::new(move || {
                LinuxThreads::new().give_walk_hints();
                let walk = read_back();
                let (worker_tx, plain_tx) = (tx.clone(), tx.clone());
                LinuxThreads::new().spawn_worker(
                    "reims-vgpu.worker",
                    Box::new(move || worker_tx.send(("worker", read_back())).unwrap()),
                );
                LinuxThreads::new().spawn(
                    "reims-vgpu.ui",
                    512 * 1024,
                    Box::new(move || plain_tx.send(("plain", read_back())).unwrap()),
                );
                tx.send(("walk", walk)).unwrap();
            }),
        );
        let mut seen: Vec<(&str, SchedAttr)> =
            (0..3).map(|_| rx.recv_timeout(WAIT).unwrap()).collect();
        seen.sort_by_key(|(who, _)| *who);
        let [(_, plain), (_, walk), (_, worker)] = seen[..] else {
            unreachable!()
        };
        assert_eq!(walk.sched_util_min, walk_floor());
        for attr in [plain, worker] {
            assert_eq!(attr.sched_util_min, 0);
            assert_eq!(attr.sched_policy, walk.sched_policy);
            assert_eq!(attr.sched_nice, walk.sched_nice);
        }
    }
}
