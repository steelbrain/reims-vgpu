//! The scheduling hint a thread that walks the guest's command buffers gives macOS
//! ([`hint_walk_thread`]): the user-initiated QoS class.
//!
//! A walk thread of the graphics device's lanes walks a command buffer's ops one after another, so
//! wherever the GPU keeps up its speed is the frame rate; it gives itself this hint as it starts
//! (`Threads::give_walk_hints`). A thread asks macOS for a QoS class of its own
//! (`pthread_set_qos_class_self_np`, `<pthread/qos.h>`), which the scheduler weighs where threads
//! contend for the CPU: the user-initiated class runs above the default class, the one a build or
//! any other process the user starts on the host runs at, and below the user-interactive class of
//! the host's own interface (`<sys/qos.h>`, read from the SDK). It is the class QEMU's HVF vCPU
//! threads ask for (`accel/hvf/hvf-accel-ops.c`), whose guest's commands the walks carry out: a
//! guest beside a host build keeps the CPU its frames take, and a busy guest, or several, cannot
//! take the CPU from the host's desktop.
//!
//! - **What is not given.** The relative priority within the class stays 0, the class's own; no
//!   policy or priority is set through `pthread_setschedparam`, which would take the thread out of
//!   the QoS classes for good; no CPU is chosen.
//! - **The threads the library starts** (a queue's worker, a named thread) run at the default
//!   class whatever their creator's: `pthread_create` gives a thread whose attributes name no
//!   class `QOS_CLASS_DEFAULT` (`<sys/qos.h>`; the tests read it back), so a walk thread passes
//!   its class to none of them and nothing drops it, where Linux's floor is inherited and dropped
//!   (`os/linux/hints.rs`). A lane's thread gives itself the hint as it starts. The runtime's
//!   threads and MoltenVK's are theirs.
//!
//! The first hint's outcome is logged once.

use std::sync::atomic::{AtomicBool, Ordering};

/// The class a walk thread asks for: user-initiated, QEMU's HVF vCPU threads' own.
const WALK_CLASS: libc::qos_class_t = libc::qos_class_t::QOS_CLASS_USER_INITIATED;

/// Gives the calling thread, a walk thread, its scheduling hint (the module docs), at the class's
/// own relative priority. The first call logs what macOS took.
pub(crate) fn hint_walk_thread() {
    // SAFETY: sets the calling thread's own class; it takes no pointer.
    let rc = unsafe { libc::pthread_set_qos_class_self_np(WALK_CLASS, 0) };
    let line = match rc {
        0 => "walk threads run at the user-initiated QoS class".to_owned(),
        e => format!(
            "walk threads give the scheduler no hint: pthread_set_qos_class_self_np refused it \
             ({})",
            std::io::Error::from_raw_os_error(e)
        ),
    };
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use crate::core::platform::os::Threads;

    use super::super::threads::MacThreads;

    const WAIT: Duration = Duration::from_secs(10);

    /// `QOS_CLASS_USER_INITIATED`, `QOS_CLASS_DEFAULT` (`<sys/qos.h>`).
    const USER_INITIATED: u32 = 0x19;
    const DEFAULT: u32 = 0x15;

    /// The calling thread's QoS class and relative priority, as `pthread_get_qos_class_np` reads
    /// them.
    fn read_back() -> (u32, libc::c_int) {
        let mut class = 0u32;
        let mut priority: libc::c_int = 1;
        // SAFETY: the calling thread, and two live outputs: a `qos_class_t` is a `u32`.
        let rc = unsafe {
            libc::pthread_get_qos_class_np(
                libc::pthread_self(),
                (&raw mut class).cast::<libc::qos_class_t>(),
                &raw mut priority,
            )
        };
        assert_eq!(rc, 0, "{}", std::io::Error::from_raw_os_error(rc));
        (class, priority)
    }

    /// A thread the library starts runs at the default class until it gives itself the walk
    /// hint, then at the user-initiated class, at its relative priority 0
    #[test]
    fn a_walk_thread_runs_at_the_user_initiated_class() {
        let (tx, rx) = mpsc::channel();
        MacThreads::new().spawn(
            "reims-vgpu.walk",
            512 * 1024,
            Box::new(move || {
                let before = read_back();
                MacThreads::new().give_walk_hints();
                tx.send((before, read_back())).unwrap();
            }),
        );
        let (before, after) = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(before, (DEFAULT, 0));
        assert_eq!(after, (USER_INITIATED, 0));
    }

    /// A queue's worker and a named thread, each started from a walk thread, run at the default
    /// class, while the walk thread keeps its own
    #[test]
    fn a_thread_started_from_a_walk_thread_runs_at_the_default_class() {
        let (tx, rx) = mpsc::channel();
        MacThreads::new().spawn(
            "reims-vgpu.walk",
            512 * 1024,
            Box::new(move || {
                MacThreads::new().give_walk_hints();
                let (worker_tx, plain_tx) = (tx.clone(), tx.clone());
                MacThreads::new().spawn_worker(
                    "reims-vgpu.worker",
                    Box::new(move || worker_tx.send(("worker", read_back())).unwrap()),
                );
                MacThreads::new().spawn(
                    "reims-vgpu.ui",
                    512 * 1024,
                    Box::new(move || plain_tx.send(("plain", read_back())).unwrap()),
                );
                tx.send(("walk", read_back())).unwrap();
            }),
        );
        let mut seen: Vec<(&str, (u32, libc::c_int))> =
            (0..3).map(|_| rx.recv_timeout(WAIT).unwrap()).collect();
        seen.sort_by_key(|(who, _)| *who);
        assert_eq!(
            seen,
            [
                ("plain", (DEFAULT, 0)),
                ("walk", (USER_INITIATED, 0)),
                ("worker", (DEFAULT, 0)),
            ]
        );
    }
}
