//! Windows thread creation. There is no pthread scheduling state or Objective-C runtime.
//! Threads use Windows' normal priority so rendering cannot starve the host's desktop or WSL2.

use super::clock::WindowsClock;
use reims_vgpu_core::platform::os::{Threads, TimerClock};
use std::sync::Arc;

/// Detached threads whose names and stacks are provided by core.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsThreads;

impl WindowsThreads {
    /// FIFO threads use the OS's normal scheduling policy.
    pub fn new() -> Self {
        Self
    }
    /// UI threads use the same policy; no inherited Linux scheduler hints exist here.
    pub fn without_hints() -> Self {
        Self
    }
}

impl Threads for WindowsThreads {
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        std::thread::Builder::new()
            .name(name.to_owned())
            .stack_size(stack_size)
            .spawn(body)
            .unwrap_or_else(|error| {
                super::failure::failed(error, "Windows failed to create the requested GPU thread")
            });
    }

    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        self.spawn(label, reims_vgpu_core::rt::queue::WORKER_STACK, body);
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(*self)
    }
    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(WindowsClock::new()))
    }
}
