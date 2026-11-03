//! Keeps the GPU out of its runtime suspend while the guest draws, as a Mac's GPU that drives a
//! display stays up between frames.
//!
//! - **Why.** A host's runtime power management suspends a GPU that has had no work for its
//!   autosuspend delay, and the next work waits for its resume. Measured on the RTX 5080 Laptop
//!   GPU (NVIDIA 615.71.09, open kernel module, `DynamicPowerManagement=3`, `power/control`
//!   `auto`, the host's stock policy): its `power/runtime_status` went from `active` to
//!   `suspended` about 10 s after its last work. An `nvidia-smi` query then took 992 ms, against
//!   18 ms awake, and WindowServer's frame at each minute's turn, which follows a static screen,
//!   waited 954 ms for the buffer it makes first. The same make took under 5 ms while the guest
//!   drew. Awake and idle, at P8 with a 180 MHz graphics clock, the board drew 7.2 W (the median
//!   of `nvidia-smi`'s `power.draw` each second for 56 s: a mean of 8.35 W, 7.55 W after the first
//!   20 s). Suspended, the board is powered off, and `nvidia-smi` reads it only by resuming it.
//! - **The guest's work.** Every command buffer the device commits counts
//!   ([`Awake::touch`], from the commit). A window's present counts as the frame's command buffer
//!   it follows, and so does a guest's.
//! - **The keeping.** For [`AWAKE_FOR`] after the last of that work, the thread
//!   `reims-vgpu.gpu-awake` looks every half [`period`]. At each look that finds no commit since
//!   the look before, it submits one command buffer of a timestamp write (empty where the queue's
//!   family writes no timestamps), so the GPU is never a whole period without work. Past
//!   [`AWAKE_FOR`] it parks until the next commit, and the GPU suspends as the host's policy has
//!   it: the first frame after a longer break pays the resume once.
//! - **The period** is half the device's autosuspend delay, read where sysfs gives it for the
//!   device's PCI address (`power/autosuspend_delay_ms`, `VK_EXT_pci_bus_info`'s address), else
//!   half of [`ASSUMED_DELAY`]. The RTX 5080 Laptop GPU's file answers EIO, so its period is
//!   2.5 s; amdgpu writes its delay there.
//! - **On every GPU.** No driver or GPU kind is told apart: a GPU without runtime suspend, or
//!   one that drives the host's display, gets one small command buffer per look while the guest
//!   is idle, and none while it draws.
//! - **Nothing the guest sees.** The thread's objects are raw Vulkan objects of its own, made at
//!   its first submission: a command pool and its one command buffer, a query pool of one
//!   timestamp, and a fence; and a runtime queue of its own, whose `VkQueue` it borrows for each
//!   submission (`rt::CommandQueue::with_vk_queue`). No resource of the runtime's is made, so no
//!   GPU address, ID, stamp or count the guest reads changes. The runtime counts each submission
//!   as the program's GPU work, so a texture's host shadow read before it is read again after.
//! - **A failed submission** (a lost device) ends the thread, with one warning, and the GPU
//!   suspends as the host's policy has it.
//! - **Devices the process adopts** ([`crate::gpu::device::adopt`]) have the thread. A device
//!   made for a test alone has [`Awake::default`], which counts and keeps nothing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::Thread;
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::clock::{Clock as _, Deadline, MonoTime};

/// How long after the guest's last GPU work the GPU is kept awake.
pub(crate) const AWAKE_FOR: Duration = Duration::from_secs(5 * 60);

/// The autosuspend delay assumed where sysfs gives none for the device.
pub(crate) const ASSUMED_DELAY: Duration = Duration::from_secs(5);

/// The shortest period, for a delay of a few milliseconds.
const LEAST_PERIOD: Duration = Duration::from_millis(100);

/// The log target of the thread's lines.
const TARGET: &str = "reims_vgpu_vulkan::awake";

/// A device's keeping awake (the module docs): what its commits count, and its thread.
#[derive(Debug, Default)]
pub struct Awake(Option<Arc<Shared>>);

/// What the device's commits and its thread share.
#[derive(Debug, Default)]
struct Shared {
    /// How many command buffers the device committed.
    work: AtomicU64,
    /// Whether the thread is parked, or about to park, until the next commit.
    parked: AtomicBool,
    /// The thread, as it set itself.
    thread: OnceLock<Thread>,
}

impl Awake {
    /// Starts `device`'s thread (the module docs), parked until the device's first commit.
    pub(crate) fn start(device: &rt::Device) -> Awake {
        let shared = Arc::new(Shared::default());
        let period = period(delay_text(device).as_deref());
        let (device, kept) = (device.clone(), Arc::clone(&shared));
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.gpu-awake".into())
            .spawn(move || keep(&device, &kept, period));
        match spawned {
            Ok(_) => Awake(Some(shared)),
            Err(e) => {
                log::warn!(target: TARGET, "the GPU is not kept awake: no thread ({e})");
                Awake(None)
            }
        }
    }

    /// The device committed a command buffer: counted, and the thread woken where it is parked.
    pub(crate) fn touch(&self) {
        let Some(shared) = &self.0 else {
            return;
        };
        // Sequentially consistent with the thread's store of `parked` and its read of the count
        // after it: either the thread reads this commit's count and does not park, or this read
        // finds it parked, or about to, and wakes it (`park` keeps a wake that comes first).
        shared.work.fetch_add(1, Ordering::SeqCst);
        if shared.parked.load(Ordering::SeqCst)
            && let Some(thread) = shared.thread.get()
        {
            thread.unpark();
        }
    }
}

/// The longest the thread lets the GPU go without work, for the device's autosuspend delay as
/// sysfs writes it (`delay_ms`, milliseconds): half the delay, at least [`LEAST_PERIOD`]; half of
/// [`ASSUMED_DELAY`] where there is no file, it cannot be read, or it holds no positive delay
/// (`-1`, which turns autosuspend off).
pub(crate) fn period(delay_ms: Option<&str>) -> Duration {
    let delay = delay_ms
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map_or(ASSUMED_DELAY, Duration::from_millis);
    (delay / 2).max(LEAST_PERIOD)
}

/// The text of `device`'s `power/autosuspend_delay_ms` in sysfs, where it has a PCI address and
/// the file can be read.
fn delay_text(device: &rt::Device) -> Option<String> {
    let (domain, bus, slot, function) = device.properties().pci_bus?;
    let path = format!(
        "/sys/bus/pci/devices/{domain:04x}:{bus:02x}:{slot:02x}.{function:x}/power/autosuspend_delay_ms"
    );
    std::fs::read_to_string(path).ok()
}

/// What a look of the thread does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Look {
    /// The device committed since the look before: no submission.
    Busy,
    /// No commit since the look before, and the last within [`AWAKE_FOR`]: one submission.
    Touch,
    /// No commit within [`AWAKE_FOR`], or none yet: park until the next.
    Rest,
}

/// The thread's record of the guest's work: the count of commits it saw last, and the look that
/// saw it change.
#[derive(Debug, Default)]
pub(crate) struct Keep {
    seen: u64,
    last: Option<MonoTime>,
}

impl Keep {
    /// The look at `now`, the device's count of commits being `work`.
    pub(crate) fn look(&mut self, work: u64, now: MonoTime) -> Look {
        if work != self.seen {
            self.seen = work;
            self.last = Some(now);
            return Look::Busy;
        }
        match self.last {
            Some(at) if now.saturating_duration_since(at) < AWAKE_FOR => Look::Touch,
            _ => Look::Rest,
        }
    }

    /// The count of commits the last look saw.
    pub(crate) fn seen(&self) -> u64 {
        self.seen
    }
}

/// The thread: looks every half `period` (the module docs) until a submission fails.
fn keep(device: &rt::Device, shared: &Shared, period: Duration) {
    let _ = shared.thread.set(std::thread::current());
    log::debug!(
        target: TARGET,
        "the GPU is kept awake {AWAKE_FOR:?} after the last commit, at most {period:?} without work"
    );
    let clock = crate::os::OsClock::new();
    let mut keep = Keep::default();
    let mut toucher: Option<Toucher> = None;
    loop {
        let now = clock.now_mono();
        match keep.look(shared.work.load(Ordering::SeqCst), now) {
            Look::Busy => {}
            Look::Touch => {
                let touched = match &mut toucher {
                    Some(toucher) => toucher.touch(),
                    None => Toucher::new(device).and_then(|made| toucher.insert(made).touch()),
                };
                if let Err(e) = touched {
                    log::warn!(target: TARGET, "the GPU is no longer kept awake: {e}");
                    return;
                }
            }
            Look::Rest => {
                shared.parked.store(true, Ordering::SeqCst);
                while shared.work.load(Ordering::SeqCst) == keep.seen() {
                    std::thread::park();
                }
                shared.parked.store(false, Ordering::SeqCst);
                continue;
            }
        }
        clock.sleep_until(Deadline::Mono(now.saturating_add(period / 2)));
    }
}

/// How long the toucher's end waits for its last submission.
const LAST_WAIT: Duration = Duration::from_secs(1);

/// The thread's submission (the module docs): its queue and its raw objects.
struct Toucher {
    device: rt::Device,
    queue: rt::CommandQueue,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    timestamps: vk::QueryPool,
    fence: vk::Fence,
    /// Whether the fence has a submission of the buffer to signal.
    pending: bool,
}

impl Toucher {
    /// The objects on `device`, the buffer recorded once: a reset of the one timestamp and its
    /// write, where the queue's family writes timestamps.
    fn new(device: &rt::Device) -> Result<Toucher, String> {
        let queue = device
            .new_command_queue()
            .map_err(|e| format!("no queue ({e:?})"))?;
        // SAFETY: the closure keeps no handle and calls nothing.
        let family = unsafe { queue.with_vk_queue(|_, family| family) };
        // SAFETY: the handles are used while `device`, which this object holds, lives; the
        // objects made of them are this object's own, destroyed at its end.
        let handles = unsafe { device.raw() };
        let (vk_device, instance) = (handles.device, handles.instance);
        // SAFETY: the physical device is the device's own.
        let families = unsafe {
            instance.get_physical_device_queue_family_properties(handles.physical_device)
        };
        let timestamped = usize::try_from(family)
            .ok()
            .and_then(|at| families.get(at))
            .is_some_and(|f| f.timestamp_valid_bits > 0);
        let mut toucher = Toucher {
            device: device.clone(),
            queue,
            pool: vk::CommandPool::null(),
            buffer: vk::CommandBuffer::null(),
            timestamps: vk::QueryPool::null(),
            fence: vk::Fence::null(),
            pending: false,
        };
        // The device's commands with the runtime's loader word put back, as the loader's own and
        // a layer's need (`rt::Device::with_loader_word`). An object made before a refusal is
        // destroyed by the toucher's end.
        device
            .with_loader_word(|| toucher.make(vk_device, family, timestamped))
            .map_err(|e| format!("its objects were refused ({e})"))?;
        Ok(toucher)
    }

    /// Makes the objects on `vk_device`, the toucher's device, for a queue of `family`, and
    /// records the buffer: a reset of the one timestamp and its write where `timestamped`.
    fn make(
        &mut self,
        vk_device: &ash::Device,
        family: u32,
        timestamped: bool,
    ) -> Result<(), vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        // SAFETY: the infos are valid, and every object is this toucher's.
        unsafe {
            self.pool = vk_device.create_command_pool(&pool_info, None)?;
            let buffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.buffer = vk_device.allocate_command_buffers(&buffer_info)?[0];
            if timestamped {
                let query_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(1);
                self.timestamps = vk_device.create_query_pool(&query_info, None)?;
            }
            self.fence = vk_device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let begin = vk::CommandBufferBeginInfo::default();
            vk_device.begin_command_buffer(self.buffer, &begin)?;
            if timestamped {
                vk_device.cmd_reset_query_pool(self.buffer, self.timestamps, 0, 1);
                let stage = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                vk_device.cmd_write_timestamp(self.buffer, stage, self.timestamps, 0);
            }
            vk_device.end_command_buffer(self.buffer)
        }
    }

    /// Submits the buffer, where its last submission completed; where it is still pending, the
    /// GPU has work ahead of it on its queue, and nothing is submitted.
    fn touch(&mut self) -> Result<(), String> {
        // SAFETY: the handles are used while the device lives.
        let vk_device = unsafe { self.device.raw() }.device;
        let (buffer, fence) = (self.buffer, self.fence);
        if self.pending {
            // SAFETY: the fence is this toucher's.
            let done = self.device.with_loader_word(|| unsafe {
                let done = vk_device.get_fence_status(fence)?;
                if done {
                    vk_device.reset_fences(&[fence])?;
                }
                Ok::<bool, vk::Result>(done)
            });
            match done {
                Ok(true) => self.pending = false,
                Ok(false) => return Ok(()),
                Err(e) => return Err(format!("its fence answered {e}")),
            }
        }
        let buffers = [buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
        // SAFETY: the submission is the closure's only call, keeps no handle and waits for
        // nothing; the buffer is executable and not pending (its fence signalled), and the fence
        // is unsignalled.
        let submitted = unsafe {
            self.queue
                .with_vk_queue(|queue, _| vk_device.queue_submit(queue, &submit, fence))
        };
        submitted.map_err(|e| format!("its submission was refused ({e})"))?;
        self.pending = true;
        Ok(())
    }
}

impl Drop for Toucher {
    fn drop(&mut self) {
        // SAFETY: the handles are used while the device lives.
        let vk_device = unsafe { self.device.raw() }.device;
        let pending = self.pending;
        let (pool, timestamps, fence) = (self.pool, self.timestamps, self.fence);
        self.device.with_loader_word(|| {
            // SAFETY: the objects are this toucher's; its last submission completed or is given
            // up on after the wait, on a device that answers no more.
            unsafe {
                if pending {
                    let nanos = u64::try_from(LAST_WAIT.as_nanos()).unwrap_or(u64::MAX);
                    let _ = vk_device.wait_for_fences(&[fence], true, nanos);
                }
                if fence != vk::Fence::null() {
                    vk_device.destroy_fence(fence, None);
                }
                if timestamps != vk::QueryPool::null() {
                    vk_device.destroy_query_pool(timestamps, None);
                }
                if pool != vk::CommandPool::null() {
                    vk_device.destroy_command_pool(pool, None);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// The time `secs` seconds after the clock's start.
    fn at(secs: f64) -> MonoTime {
        MonoTime::from_nanos((secs * 1e9) as u64)
    }

    /// No commit yet: the thread parks.
    #[test]
    fn a_device_never_committed_to_rests() {
        let mut keep = Keep::default();
        assert_eq!(keep.look(0, at(0.0)), Look::Rest);
        assert_eq!(keep.look(0, at(600.0)), Look::Rest);
    }

    /// While commits come between looks, nothing is submitted; at each look after the last, one
    /// submission, until [`AWAKE_FOR`] after the look that saw it, when the thread parks; a commit
    /// after that starts the looks again, for another [`AWAKE_FOR`].
    #[test]
    fn it_stops_awake_for_after_the_last_work_and_starts_again_on_new_work() {
        let step = period(None) / 2;
        let mut keep = Keep::default();
        let mut now = at(10.0);
        let mut last = now;
        let mut work = 0;
        for _ in 0..4 {
            work += 1;
            assert_eq!(keep.look(work, now), Look::Busy);
            last = now;
            now = now.saturating_add(step);
        }
        while now.saturating_duration_since(last) < AWAKE_FOR {
            assert_eq!(keep.look(work, now), Look::Touch, "at {now:?}");
            now = now.saturating_add(step);
        }
        assert_eq!(keep.look(work, last.saturating_add(AWAKE_FOR)), Look::Rest);
        assert_eq!(keep.look(work, now.saturating_add(600 * S)), Look::Rest);
        assert_eq!(keep.seen(), work);

        let again = now.saturating_add(900 * S);
        work += 1;
        assert_eq!(keep.look(work, again), Look::Busy);
        assert_eq!(keep.look(work, again.saturating_add(step)), Look::Touch);
        let near_end = again.saturating_add(AWAKE_FOR - step);
        assert_eq!(keep.look(work, near_end), Look::Touch);
        assert_eq!(keep.look(work, again.saturating_add(AWAKE_FOR)), Look::Rest);
    }

    /// A look every half period keeps the GPU under a period without work: a commit right after
    /// one look shows at the next, and the look after submits.
    #[test]
    fn the_gpu_goes_under_a_period_without_work() {
        let period = period(None);
        let step = period / 2;
        let mut keep = Keep::default();
        let first = at(1.0);
        assert_eq!(keep.look(1, first), Look::Busy);
        let commit = first.saturating_add(Duration::from_millis(1));
        assert_eq!(keep.look(2, first.saturating_add(step)), Look::Busy);
        let touch = first.saturating_add(2 * step);
        assert_eq!(keep.look(2, touch), Look::Touch);
        assert!(touch.saturating_duration_since(commit) < period);
    }

    /// The period is half sysfs's delay, at least [`LEAST_PERIOD`], else half the assumed one.
    #[test]
    fn the_period_is_half_the_autosuspend_delay() {
        assert_eq!(period(Some("5000\n")), Duration::from_millis(2500));
        assert_eq!(period(Some("10000")), Duration::from_secs(5));
        assert_eq!(period(Some("20\n")), LEAST_PERIOD);
        assert_eq!(period(None), ASSUMED_DELAY / 2);
        assert_eq!(period(Some("")), ASSUMED_DELAY / 2);
        assert_eq!(period(Some("-1\n")), ASSUMED_DELAY / 2);
        assert_eq!(period(Some("0\n")), ASSUMED_DELAY / 2);
    }

    /// A device's `Awake` of a test counts nothing and wakes nothing.
    #[test]
    fn a_default_awake_keeps_nothing() {
        let awake = Awake::default();
        awake.touch();
        assert!(awake.0.is_none());
    }
}
