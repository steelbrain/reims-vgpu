//! `Threads` on Linux: `std::thread`s, detached, with the stack and name core asks for. Every
//! thread started here drops the hints it inherited from the thread that started it; a thread that
//! walks the guest's command buffers, which the platform does not start, gives itself the walk
//! hints ([`Threads::give_walk_hints`], [`super::hints`]).
//!
//! No Objective-C runs on Linux, so the reason macOS starts plain pthreads (D35: no unwind
//! handler at a thread's base for a foreign exception's search phase to find) does not apply. A
//! panic on one of these threads is the library's, and its panic hook aborts the process before
//! anything unwinds. A thread's name is a diagnostic; Linux keeps 15 bytes of it, so a longer one
//! is cut there, on a character boundary.

use std::sync::Arc;

use crate::core::platform::os::{Threads, TimerClock};
use crate::core::raise::death::{DeathId, DeathKind, fatal};

use super::clock::LinuxClock;

/// The bytes of a thread's name Linux keeps (`TASK_COMM_LEN` less its NUL).
const NAME_BYTES: usize = 15;

/// `name`, cut to the bytes Linux keeps.
fn kept_name(name: &str) -> &str {
    let mut end = name.len().min(NAME_BYTES);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

/// A detached thread named `name` with `stack_size` bytes of stack that runs `body`. A refused
/// thread ends the process.
fn spawn_thread(name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    let made = std::thread::Builder::new()
        .name(kept_name(name).to_owned())
        .stack_size(stack_size)
        .spawn(body);
    if made.is_err() {
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "pthread_create",
        })
    }
}

/// The process's threads, and its clock for timers.
#[derive(Debug, Default, Clone, Copy)]
pub struct LinuxThreads;

impl LinuxThreads {
    /// The process's Threads.
    pub fn new() -> LinuxThreads {
        LinuxThreads
    }
}

impl Threads for LinuxThreads {
    /// A thread that drops the hints it inherited ([`super::hints::drop_inherited_hints`]): a
    /// display window's.
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(
            name,
            stack_size,
            Box::new(move || {
                super::hints::drop_inherited_hints();
                body();
            }),
        );
    }

    /// A worker named after its queue, with a GCD worker's 512 KiB of stack, as on macOS. It drops
    /// the hints it inherited ([`super::hints::drop_inherited_hints`]): a walk thread, or one of
    /// the runtime's threads, may be the one that starts it.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(
            label,
            crate::core::rt::queue::WORKER_STACK,
            Box::new(move || {
                super::hints::drop_inherited_hints();
                body();
            }),
        );
    }

    /// The walk hints ([`super::hints::hint_walk_thread`]), for a thread the platform did not
    /// start.
    fn give_walk_hints(&self) {
        super::hints::hint_walk_thread();
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(*self)
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(LinuxClock::new()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    /// The running thread's name, as Linux keeps it.
    fn current_name() -> String {
        let mut buf = [0 as libc::c_char; 64];
        // SAFETY: a live buffer of the length given.
        let rc =
            unsafe { libc::pthread_getname_np(libc::pthread_self(), buf.as_mut_ptr(), buf.len()) };
        assert_eq!(rc, 0);
        // SAFETY: pthread_getname_np NUL-terminates within the buffer.
        unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    /// A spawned thread is named, cut to the 15 bytes Linux keeps, and has at least the stack
    /// asked for
    #[test]
    fn a_thread_is_named_and_sized() {
        let (tx, rx) = mpsc::channel();
        LinuxThreads::new().spawn(
            "reims-vgpu.test",
            512 * 1024,
            Box::new(move || {
                let mut attr = std::mem::MaybeUninit::<libc::pthread_attr_t>::uninit();
                let mut size = 0usize;
                // SAFETY: the running thread's attributes, read and destroyed here.
                unsafe {
                    libc::pthread_getattr_np(libc::pthread_self(), attr.as_mut_ptr());
                    libc::pthread_attr_getstacksize(attr.as_ptr(), &mut size);
                    libc::pthread_attr_destroy(attr.as_mut_ptr());
                }
                tx.send((current_name(), size)).unwrap();
            }),
        );
        let (name, size) = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(name, "reims-vgpu.test");
        assert!(size >= 512 * 1024, "{size}");
        assert_eq!(kept_name("reims-vgpu.display-present"), "reims-vgpu.disp");
        assert_eq!(kept_name("reims-vgpu.é1234"), "reims-vgpu.é12");
        assert_eq!(kept_name("reims-vgpu.ééé"), "reims-vgpu.éé");
    }

    /// A queue's worker is named after the queue, cut to the bytes Linux keeps
    #[test]
    fn a_worker_is_named_after_its_queue() {
        let (tx, rx) = mpsc::channel();
        LinuxThreads::new().spawn_worker(
            "reims-vgpu.ui",
            Box::new(move || tx.send(current_name()).unwrap()),
        );
        assert_eq!(rx.recv_timeout(WAIT).unwrap(), "reims-vgpu.ui");
    }
}
