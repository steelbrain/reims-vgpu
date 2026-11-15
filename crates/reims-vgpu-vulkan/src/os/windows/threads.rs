//! `Threads` on Windows: `std::thread`s, detached, with the stack and name core asks for. A thread
//! that walks the guest's command buffers, which the platform does not start, gives itself the
//! walk hint ([`Threads::give_walk_hints`], [`super::hints`]).
//!
//! No Objective-C runs on Windows, so the reason macOS starts plain threads (D35: no unwind
//! handler at a thread's base for a foreign exception's search phase to find) does not apply. A
//! panic on one of these threads is the library's, and its panic hook aborts the process before
//! anything unwinds. A thread's name is a diagnostic, its description (`SetThreadDescription`,
//! which `std` sets from the name), kept whole. The stack size is the thread's reserve, which
//! `std` asks `CreateThread` for (`STACK_SIZE_PARAM_IS_A_RESERVATION`).

use std::sync::Arc;

use crate::core::platform::os::{Threads, TimerClock};
use crate::core::raise::death::{DeathId, DeathKind, fatal};

use super::clock::WindowsClock;

/// A detached thread named `name` with `stack_size` bytes of stack that runs `body`. A refused
/// thread ends the process.
fn spawn_thread(name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    let made = std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack_size)
        .spawn(body);
    if made.is_err() {
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "CreateThread",
        })
    }
}

/// The process's threads, and its clock for timers.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsThreads;

impl WindowsThreads {
    /// The process's Threads.
    pub fn new() -> WindowsThreads {
        WindowsThreads
    }
}

impl Threads for WindowsThreads {
    /// A thread of its own: a display window's.
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(name, stack_size, body);
    }

    /// A worker named after its queue, with a GCD worker's 512 KiB of stack, as on macOS.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(label, crate::core::rt::queue::WORKER_STACK, body);
    }

    /// The walk hint ([`super::hints::hint_walk_thread`]), for a thread the platform did not
    /// start.
    fn give_walk_hints(&self) {
        super::hints::hint_walk_thread();
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(*self)
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(WindowsClock::new()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetCurrentThreadStackLimits, GetThreadDescription,
    };

    use super::*;

    const WAIT: Duration = Duration::from_secs(10);

    /// The running thread's description.
    fn current_name() -> String {
        let mut text: windows_sys::core::PWSTR = std::ptr::null_mut();
        // SAFETY: the pseudo-handle of the running thread, and a live out-parameter.
        let hr = unsafe { GetThreadDescription(GetCurrentThread(), &mut text) };
        assert!(hr >= 0, "GetThreadDescription: {hr:#x}");
        // SAFETY: a NUL-terminated wide string the call allocated.
        let len = (0..).take_while(|&i| unsafe { *text.add(i) } != 0).count();
        // SAFETY: `len` wide characters before the NUL.
        let name = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, len) });
        // SAFETY: the string the call allocated with LocalAlloc, freed once.
        unsafe { LocalFree(text.cast()) };
        name
    }

    /// A spawned thread is named in full and has at least the stack asked for
    #[test]
    fn a_thread_is_named_and_sized() {
        let (tx, rx) = mpsc::channel();
        WindowsThreads::new().spawn(
            "reims-vgpu.display-present",
            512 * 1024,
            Box::new(move || {
                let (mut low, mut high) = (0usize, 0usize);
                // SAFETY: live out-parameters for the running thread's stack.
                unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
                tx.send((current_name(), high - low)).unwrap();
            }),
        );
        let (name, size) = rx.recv_timeout(WAIT).unwrap();
        assert_eq!(name, "reims-vgpu.display-present");
        assert!(size >= 512 * 1024, "{size}");
    }

    /// A queue's worker is named after the queue
    #[test]
    fn a_worker_is_named_after_its_queue() {
        let (tx, rx) = mpsc::channel();
        WindowsThreads::new().spawn_worker(
            "reims-vgpu.ui",
            Box::new(move || tx.send(current_name()).unwrap()),
        );
        assert_eq!(rx.recv_timeout(WAIT).unwrap(), "reims-vgpu.ui");
    }
}
