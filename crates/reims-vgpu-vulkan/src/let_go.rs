//! The platform's let-go of its GPU devices ([`let_go_of_devices`]): what the C API's release
//! of the process's last graphics device makes, and what a test that holds a device to its end
//! makes before it returns.
//!
//! - **Why.** The platform keeps each device it adopts for the process, as Metal's device lives
//!   for it (`gpu::device`'s `DEVICES`), with what it made of it for itself. On NVIDIA's
//!   proprietary driver a process that exits with a runtime device live can end with `SIGSEGV`
//!   in the driver's own exit handler, which tears the driver's SPIR-V compiler down while a
//!   worker thread of the driver's, started from the thread that made the device, still runs in
//!   it. Measured by the runtime's maintainers on an RTX 3090 (620.70.06), 8 processes at once,
//!   each keeping a device whose check of textures over host pages had no record yet: 6 of 80
//!   exits right after the check ended faulted, 0 of 80 with the device dropped first. Vulkan has
//!   no call that waits for that worker, so the remedy is the drop (the runtime's HOWTO, section
//!   8).
//! - **What goes.** Every device adopted, and what the platform keeps of them: the libraries of
//!   its own shaders, its no-draw and predication pipelines, the device the sampler mempool
//!   follows, the device whose answers describe formats, the devices handed the guest's RAM, and
//!   each device's own threads ([`DeviceThreads`] and its keeping awake, `crate::awake`), stopped
//!   and joined. Where that was a runtime device's last handle, the runtime tears it down on the
//!   calling thread, its own threads stopped and waited for, before the let-go returns.
//! - **Only once nothing else holds one.** The let-go waits for every handle beyond those to go
//!   (an object, a command buffer still running, a thread that holds one), looking every
//!   [`POLL`], for at most the bound it is given. Where one outlives it, nothing goes, and the
//!   answer names each device still held and by how many handles.
//! - **After it,** a device the platform adopts is a new runtime device, whose IDs start from
//!   new pools: no value of the old one reaches it.
//! - **Once the process's exit began** (the exit gate closed), nothing goes: the runtime leaves
//!   its devices to the exit then, and a thread this would join may be parked for good.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::core::platform::os::clock::{Clock as _, Deadline, MonoTime};
use crate::core::sync::{Condvar, Mutex};

/// How often the let-go looks at the handles held of the devices it waits for.
pub const POLL: Duration = Duration::from_millis(1);

/// Lets go of the platform's GPU devices (the module docs), once nothing beyond the platform's
/// own keeps holds one: waits at most `bound` for that. `Ok` with how many devices it let go of;
/// `Err` naming each device still held where the wait ended first, nothing let go.
pub fn let_go_of_devices(bound: Duration) -> Result<usize, String> {
    if crate::exit_gate::closed() {
        return Ok(0);
    }
    let clock = crate::os::OsClock::new();
    let deadline = clock.now_mono().saturating_add(bound);
    let (devices, libraries) = loop {
        match crate::gpu::device::take_if_only_kept() {
            Ok(taken) => break taken,
            Err(held) if clock.now_mono() >= deadline => return Err(held),
            Err(_) => std::thread::sleep(POLL),
        }
    };
    let let_go = devices.len();
    // What holds a runtime device's handle and none of the platform's device objects first, then
    // the libraries, which hold the objects, and the objects last: the last handle of each
    // runtime device goes with its object, here.
    drop(crate::gpu::rate_map::take_no_draws());
    drop(crate::gpu::predication::take_pipelines());
    drop(libraries);
    drop(crate::gpu::metal_space::take_sampler_ids());
    drop(crate::platform::take_answering_device());
    #[cfg(target_os = "linux")]
    drop(crate::os::take_ram_devices());
    drop(devices);
    log::debug!(
        target: "reims_vgpu_vulkan::let_go",
        "let go of {let_go} GPU devices"
    );
    Ok(let_go)
}

/// A stop the device's threads wait on between their steps ([`DeviceThreads`]).
#[derive(Debug, Default)]
pub(crate) struct Stop {
    /// Whether the threads are to end.
    stopped: Mutex<bool>,
    /// Raised as they are told to.
    told: Condvar,
}

impl Stop {
    /// Waits until `until` or the stop, whichever comes first: whether the threads are to end.
    pub(crate) fn wait_until(&self, until: MonoTime) -> bool {
        let clock = crate::os::OsClock::new();
        let mut stopped = self.stopped.lock();
        while !*stopped {
            if clock
                .wait_until(&self.told, &mut stopped, Deadline::Mono(until))
                .timed_out()
            {
                break;
            }
        }
        *stopped
    }

    /// Whether the threads are to end.
    pub(crate) fn stopped(&self) -> bool {
        *self.stopped.lock()
    }

    /// Tells the threads to end.
    fn tell(&self) {
        *self.stopped.lock() = true;
        self.told.notify_all();
    }
}

/// The threads of the library's that hold a handle of a device's runtime device beside its
/// keeping awake (its work log, `crate::work_log`, and its warm pass, `crate::records`): told to
/// end and joined as the device goes, so that none holds the runtime device past it.
#[derive(Debug, Default)]
pub(crate) struct DeviceThreads {
    /// What the threads wait on.
    pub(crate) stop: Arc<Stop>,
    /// The threads, until [`DeviceThreads::end`] joined them.
    threads: Vec<JoinHandle<()>>,
}

impl DeviceThreads {
    /// Keeps `thread`, a thread that waits on [`Self::stop`], for the device's end.
    pub(crate) fn keep(&mut self, thread: Option<JoinHandle<()>>) {
        self.threads.extend(thread);
    }

    /// Tells the threads to end and waits for them; once the process's exit closed the gate,
    /// waits for none, as one may be parked for good.
    pub(crate) fn end(&mut self) {
        self.stop.tell();
        for thread in self.threads.drain(..) {
            if !crate::exit_gate::closed() {
                let _ = thread.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stop ends a wait for a far deadline at once, and every wait after it; a wait with no
    /// stop ends at its deadline
    #[test]
    fn a_stop_ends_the_threads_waits() {
        let clock = crate::os::OsClock::new();
        let mut threads = DeviceThreads::default();
        let stop = Arc::clone(&threads.stop);
        assert!(!stop.wait_until(clock.now_mono().saturating_add(Duration::from_millis(5))));
        let waiter = std::thread::Builder::new().spawn(move || {
            stop.wait_until(clock.now_mono().saturating_add(Duration::from_secs(3600)))
        });
        threads.keep(std::thread::Builder::new().spawn(|| ()).ok());
        threads.end();
        assert!(waiter.unwrap().join().unwrap());
        assert!(threads.stop.stopped());
        assert!(threads.threads.is_empty());
    }
}
