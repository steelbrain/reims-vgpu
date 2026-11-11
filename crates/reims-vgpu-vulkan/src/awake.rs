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
//!   drew. Kept awake, the minute's make took 0.34 to 0.48 ms.
//! - **What it costs.** Awake and idle, at P8 with a 180 MHz graphics clock, the board drew 7.2 W
//!   with no process on it (the median of `nvidia-smi`'s `power.draw` read each second for 56 s,
//!   the reads keeping it awake: a mean of 8.35 W, 7.55 W after the first 20 s), and 9.25 W with a
//!   guest booted, idle between two of WindowServer's minute frames, and kept awake by this
//!   module (the median of the same reads each second for 50 s). Suspended, the board is powered
//!   off, and `nvidia-smi` reads it only by resuming it.
//! - **The guest's work.** Every command buffer the device commits counts
//!   ([`Awake::touch`], from the commit). A window's present counts as the frame's command buffer
//!   it follows, and so does a guest's.
//! - **The keeping.** For [`AWAKE_FOR`] after the last of that work, the thread
//!   `reims-vgpu.gpu-awake` looks at the device's count of commits every [`look`]. At each look
//!   that finds no commit since the look before, it submits one command buffer of a timestamp
//!   write (empty where the queue's family writes no timestamps), so the GPU goes at most two
//!   looks without work. Past [`AWAKE_FOR`] it parks until the next commit, and the GPU suspends
//!   as the host's policy has it: the first frame after a longer break pays the resume once.
//! - **How often it looks.** Every quarter of the device's autosuspend delay where sysfs gives
//!   one for the device's PCI address (`power/autosuspend_delay_ms`, `VK_EXT_pci_bus_info`'s
//!   address): the kernel's autosuspend counts from the driver's last sign of work, so half the
//!   delay without work never reaches it. Elsewhere every [`UNREAD_LOOK`], which holds the RTX
//!   5080 Laptop GPU: its file answers EIO, and its driver suspends it by an idle detection of its
//!   own, which the submission's kind does not move and its spacing does. Measured with a test
//!   program holding one device and 512 MiB of its memory, reading `runtime_status` every 100 ms:
//!   one submission every 1 s or every 1.25 s, a timestamp write, an empty command buffer or a
//!   4 KiB fill alike, and the GPU suspended every 42 to 47 s, each time for about 1.2 s until the
//!   next submission resumed it (a guest's frame in that window waits for the resume); one every
//!   500 ms, and it did not suspend in 400 s; every 250 ms, not in 60 s.
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
//! - **A recovered device.** The objects are of the generation of the device they were made on
//!   (`crate::generation`): where the runtime has made its `VkDevice` again since, a touch first
//!   destroys them on the old one and makes them, with a new queue, on the new one; a touch whose
//!   queue a recovery moved as it was submitted submits nothing.
//! - **At the process's exit** the threads make no Vulkan call. The driver's destructors run
//!   then, on the exiting thread, while every other thread goes on, and a call into what they
//!   freed ends the process. Measured on the RTX 5080 Laptop GPU (NVIDIA 615.71.09), before the
//!   gate, the workspace's tests logged 79 segfaults of this thread in a six-minute run, in
//!   processes whose tests had passed, at an instruction pointer equal to the faulting address
//!   (`0xcaf0003c` to `0xcaf00082`, 14 bytes apart, or 0). Each Vulkan step of a thread (its
//!   toucher's making, a touch, the toucher's end) is a step of the library's exit gate
//!   ([`crate::exit_gate`]), whose hook waits out a step in progress and refuses the next: a
//!   thread refused keeps its toucher, undestroyed, and parks until the process ends. The thread
//!   registers the hook as it starts, after its device was made, and again after each of its
//!   first [`FIRST_TOUCHES`] touches, after what the driver registered at the toucher's first
//!   calls.
//! - **Devices the process adopts** ([`crate::gpu::device::adopt`]) have the thread. A device
//!   made for a test alone has [`Awake::default`], which counts and keeps nothing.
//! - **The device's end** ([`Awake::stop`], as the platform lets go of its devices,
//!   [`crate::let_go`]) stops the thread at its next look or at once where it waits, its toucher
//!   destroyed in a step of its own, and waits for it: the thread holds a handle of the runtime
//!   device, which then goes before the device's own.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::Thread;
use std::time::Duration;

use crate::core::platform::os::clock::{Clock as _, MonoTime};
use crate::generation::Generation;
use ash::vk;
use metal2vulkan_runtime as rt;

/// How long after the guest's last GPU work the GPU is kept awake.
pub(crate) const AWAKE_FOR: Duration = Duration::from_secs(5 * 60);

/// How often the thread looks where sysfs gives no autosuspend delay for the device: two looks,
/// the longest the GPU goes without work, are the 500 ms spacing that held the RTX 5080 Laptop
/// GPU awake (the module docs).
pub(crate) const UNREAD_LOOK: Duration = Duration::from_millis(250);

/// The shortest time between looks, for a delay of a few milliseconds.
const LEAST_LOOK: Duration = Duration::from_millis(50);

/// The log target of the thread's lines.
const TARGET: &str = "reims_vgpu_vulkan::awake";

/// Ends a thread's Vulkan steps for good, the process exiting: its toucher is never destroyed (the
/// process's end frees it) and the thread parks until the process ends, keeping its device.
fn stay(toucher: Option<Toucher>) -> ! {
    std::mem::forget(toucher);
    crate::exit_gate::stay()
}

/// A device's keeping awake (the module docs): what its commits count, and its thread.
#[derive(Debug, Default)]
pub struct Awake {
    /// What the commits and the thread share; `None` where the thread did not start.
    shared: Option<Arc<Shared>>,
    /// The thread, until [`Awake::stop`] joined it.
    thread: Option<std::thread::JoinHandle<()>>,
}

/// What the device's commits and its thread share.
#[derive(Debug, Default)]
struct Shared {
    /// How many command buffers the device committed.
    work: AtomicU64,
    /// Whether the thread is parked, or about to park, until the next commit.
    parked: AtomicBool,
    /// The thread, as it set itself.
    thread: OnceLock<Thread>,
    /// Whether the thread is to end ([`Awake::stop`]).
    stop: AtomicBool,
}

impl Awake {
    /// Starts `device`'s thread (the module docs), parked until the device's first commit.
    pub(crate) fn start(device: &rt::Device) -> Awake {
        Awake::looking(device, look(delay_text(device).as_deref()))
    }

    /// Starts `device`'s thread, looking every `look`, parked until the device's first commit.
    fn looking(device: &rt::Device, look: Duration) -> Awake {
        crate::exit_gate::arm();
        let shared = Arc::new(Shared::default());
        let (device, kept) = (device.clone(), Arc::clone(&shared));
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.gpu-awake".into())
            .spawn(move || keep(&device, &kept, look));
        match spawned {
            Ok(thread) => Awake {
                shared: Some(shared),
                thread: Some(thread),
            },
            Err(e) => {
                log::warn!(target: TARGET, "the GPU is not kept awake: no thread ({e})");
                Awake::default()
            }
        }
    }

    /// Ends the thread and waits for it (the module docs): once this returns, it holds nothing of
    /// the device. Once the process's exit closed the gate, the thread may be parked for good, and
    /// nothing waits for it.
    pub(crate) fn stop(&mut self) {
        let Some(shared) = &self.shared else {
            return;
        };
        shared.stop.store(true, Ordering::SeqCst);
        let Some(thread) = self.thread.take() else {
            return;
        };
        thread.thread().unpark();
        if !crate::exit_gate::closed() {
            let _ = thread.join();
        }
    }

    /// The device committed a command buffer: counted, and the thread woken where it is parked.
    pub(crate) fn touch(&self) {
        let Some(shared) = &self.shared else {
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

/// How often the thread looks, for the device's autosuspend delay as sysfs writes it (`delay_ms`,
/// milliseconds): every quarter of the delay, at least [`LEAST_LOOK`] apart; every
/// [`UNREAD_LOOK`] where there is no file, it cannot be read, or it holds no positive delay (`-1`,
/// which turns autosuspend off).
pub(crate) fn look(delay_ms: Option<&str>) -> Duration {
    delay_ms
        .and_then(|text| text.trim().parse::<u64>().ok())
        .filter(|&ms| ms > 0)
        .map_or(UNREAD_LOOK, |ms| {
            (Duration::from_millis(ms) / 4).max(LEAST_LOOK)
        })
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

/// The thread: looks every `look` (the module docs) until a submission fails or the device ends
/// ([`Awake::stop`]).
fn keep(device: &rt::Device, shared: &Shared, look: Duration) {
    let _ = shared.thread.set(std::thread::current());
    log::debug!(
        target: TARGET,
        "the GPU is kept awake {AWAKE_FOR:?} after the last commit, looked at every {look:?}"
    );
    let clock = crate::os::OsClock::new();
    let mut keep = Keep::default();
    let mut toucher: Option<Toucher> = None;
    let mut touches = 0_u32;
    loop {
        if shared.stop.load(Ordering::SeqCst) {
            let Some(held) = crate::exit_gate::step() else {
                stay(toucher);
            };
            drop(toucher.take());
            drop(held);
            return;
        }
        let now = clock.now_mono();
        match keep.look(shared.work.load(Ordering::SeqCst), now) {
            Look::Busy => {}
            Look::Touch => {
                let Some(held) = crate::exit_gate::step() else {
                    stay(toucher);
                };
                let touched = match &mut toucher {
                    Some(toucher) => toucher.touch(),
                    None => Toucher::new(device).and_then(|made| toucher.insert(made).touch()),
                };
                if let Err(e) = touched {
                    log::warn!(target: TARGET, kind:% = e; "the GPU is no longer kept awake: {e}");
                    drop(toucher.take());
                    drop(held);
                    return;
                }
                drop(held);
                if touches < FIRST_TOUCHES {
                    touches += 1;
                    crate::exit_gate::arm();
                }
            }
            Look::Rest => {
                shared.parked.store(true, Ordering::SeqCst);
                while shared.work.load(Ordering::SeqCst) == keep.seen()
                    && !shared.stop.load(Ordering::SeqCst)
                {
                    std::thread::park();
                }
                shared.parked.store(false, Ordering::SeqCst);
                continue;
            }
        }
        // Until the next look, or the device's end, which unparks the thread.
        let next = now.saturating_add(look);
        while !shared.stop.load(Ordering::SeqCst) {
            let left = next.saturating_duration_since(clock.now_mono());
            if left.is_zero() {
                break;
            }
            std::thread::park_timeout(left);
        }
    }
}

/// The touches after each of which a thread registers the exit gate's hook again: the first makes the
/// toucher and submits, and the next two reach its fence's status and reset.
const FIRST_TOUCHES: u32 = 3;

/// How long the toucher's end waits for its last submission.
const LAST_WAIT: Duration = Duration::from_secs(1);

/// The thread's submission (the module docs): its queue and its raw objects, on the generation
/// of the device they were made on.
struct Toucher {
    device: rt::Device,
    queue: rt::CommandQueue,
    generation: Generation,
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
        let generation = Generation::serving(&queue)?;
        let family = generation.family();
        let (vk_device, timestamped) = {
            // SAFETY: the handles are used while the generation, which this object holds, lives;
            // the objects made of them are this object's own, destroyed at its end.
            let handles = unsafe { generation.raw() };
            // SAFETY: the physical device is the device's own.
            let families = unsafe {
                (handles.instance)
                    .get_physical_device_queue_family_properties(handles.physical_device)
            };
            let timestamped = usize::try_from(family)
                .ok()
                .and_then(|at| families.get(at))
                .is_some_and(|f| f.timestamp_valid_bits > 0);
            (handles.device.clone(), timestamped)
        };
        let made_on = generation.device().clone();
        let mut toucher = Toucher {
            device: device.clone(),
            queue,
            generation,
            pool: vk::CommandPool::null(),
            buffer: vk::CommandBuffer::null(),
            timestamps: vk::QueryPool::null(),
            fence: vk::Fence::null(),
            pending: false,
        };
        // The device's commands with the runtime's loader word put back, as the loader's own and
        // a layer's need (`rt::Device::with_loader_word`). An object made before a refusal is
        // destroyed by the toucher's end.
        made_on
            .with_loader_word(|| toucher.make(&vk_device, family, timestamped))
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
    /// GPU has work ahead of it on its queue, and nothing is submitted. Where the device
    /// recovered since the objects were made, they are made again on its new `VkDevice` first.
    fn touch(&mut self) -> Result<(), String> {
        if !self.generation.serves(&self.queue) {
            // The old objects go once the new ones are made, on the `VkDevice` they were made on.
            *self = Toucher::new(&self.device)
                .map_err(|e| format!("after the device's recovery: {e}"))?;
            log::debug!(
                target: TARGET,
                "the toucher's objects were made again on the device's new VkDevice"
            );
        }
        let made_on = self.generation.device().clone();
        // SAFETY: the handles are used while the generation, which `made_on` holds, lives.
        let vk_device = unsafe { made_on.raw() }.device;
        let (buffer, fence) = (self.buffer, self.fence);
        if self.pending {
            // SAFETY: the fence is this toucher's.
            let done = made_on.with_loader_word(|| unsafe {
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
        // is unsignalled. The queue is the one the generation was taken with.
        let submitted = unsafe {
            self.generation.with_queue(&self.queue, |queue| {
                vk_device.queue_submit(queue, &submit, fence)
            })
        };
        // A recovery moved the queue as it was submitted to: the next touch makes the objects
        // again.
        let Some(submitted) = submitted else {
            return Ok(());
        };
        submitted.map_err(|e| format!("its submission was refused ({e})"))?;
        self.pending = true;
        Ok(())
    }
}

impl Drop for Toucher {
    fn drop(&mut self) {
        let made_on = self.generation.device().clone();
        // SAFETY: the handles are used while the generation, which `made_on` holds, lives: its
        // `VkDevice`, on which every object here was made, whether or not it still serves.
        let vk_device = unsafe { made_on.raw() }.device;
        let pending = self.pending;
        let (pool, timestamps, fence) = (self.pool, self.timestamps, self.fence);
        made_on.with_loader_word(|| {
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
        let step = look(None);
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

    /// The GPU goes under two looks without work: a commit right after one look shows at the
    /// next, and the look after submits.
    #[test]
    fn the_gpu_goes_under_two_looks_without_work() {
        let step = look(None);
        let mut keep = Keep::default();
        let first = at(1.0);
        assert_eq!(keep.look(1, first), Look::Busy);
        let commit = first.saturating_add(Duration::from_millis(1));
        assert_eq!(keep.look(2, first.saturating_add(step)), Look::Busy);
        let touch = first.saturating_add(2 * step);
        assert_eq!(keep.look(2, touch), Look::Touch);
        assert!(touch.saturating_duration_since(commit) < 2 * step);
    }

    /// The thread looks every quarter of sysfs's delay, at least [`LEAST_LOOK`] apart, else every
    /// [`UNREAD_LOOK`].
    #[test]
    fn it_looks_every_quarter_of_the_autosuspend_delay() {
        assert_eq!(look(Some("5000\n")), Duration::from_millis(1250));
        assert_eq!(look(Some("10000")), Duration::from_millis(2500));
        assert_eq!(look(Some("20\n")), LEAST_LOOK);
        assert_eq!(look(None), UNREAD_LOOK);
        assert_eq!(look(Some("")), UNREAD_LOOK);
        assert_eq!(look(Some("-1\n")), UNREAD_LOOK);
        assert_eq!(look(Some("0\n")), UNREAD_LOOK);
    }

    /// The submission runs on the device: its fence signals, and the buffer, recorded once, is
    /// submitted again after it.
    #[test]
    fn the_touch_runs_on_the_gpu_and_runs_again() {
        let device = crate::gpu::device::tests::dev();
        let mut toucher = Toucher::new(&device.raw).unwrap();
        // SAFETY: the handles are used while the device lives.
        let vk_device = unsafe { device.raw.raw() }.device;
        for _ in 0..3 {
            toucher.touch().unwrap();
            assert!(toucher.pending);
            // SAFETY: the fence is the toucher's.
            let waited = device.raw.with_loader_word(|| unsafe {
                vk_device.wait_for_fences(&[toucher.fence], true, 5_000_000_000)
            });
            assert_eq!(waited, Ok(()));
        }
    }

    /// A touch after the device's recovery runs on its new `VkDevice`: the toucher's objects are
    /// made again there, and its fence signals.
    #[test]
    fn a_touch_after_the_devices_recovery_runs_on_its_new_device() {
        // A device of its own, as the recovery moves every user of the device.
        let device = rt::Device::system_default().expect("a runtime device");
        let mut toucher = Toucher::new(&device).unwrap();
        let wait = |toucher: &Toucher| {
            // SAFETY: the handles are used while the toucher's generation lives.
            let vk_device = unsafe { toucher.generation.raw() }.device;
            // SAFETY: the fence is the toucher's, on its generation's device.
            let waited = toucher.generation.device().with_loader_word(|| unsafe {
                vk_device.wait_for_fences(&[toucher.fence], true, 5_000_000_000)
            });
            assert_eq!(waited, Ok(()));
        };
        toucher.touch().unwrap();
        assert!(toucher.pending);
        wait(&toucher);
        // SAFETY: only the handle is read, while the generation lives.
        let first = unsafe { toucher.generation.raw() }.device.handle();

        device.recover_for_test().expect("the recovery");
        for _ in 0..2 {
            toucher.touch().unwrap();
            assert!(toucher.pending);
            wait(&toucher);
        }
        // SAFETY: as above.
        let second = unsafe { toucher.generation.raw() }.device.handle();
        assert_ne!(first, second, "the objects are the new VkDevice's");
        assert!(toucher.generation.serves(&toucher.queue));
    }

    /// A process that exits amid its thread's touches ends with its exit's status: the exit
    /// gate's hook stops the touches before the driver's destructors run (the module docs, "At
    /// the process's exit"). Each child looks every millisecond, so its thread touches the GPU
    /// throughout its exit.
    #[test]
    fn an_exit_amid_touches_ends_with_its_status() {
        const TEST: &str = "awake::tests::an_exit_amid_touches_ends_with_its_status";
        if std::env::var("RVG_CHILD").as_deref() == Ok(TEST) {
            let device = crate::gpu::device::tests::dev();
            let awake = Awake::looking(&device.raw, Duration::from_millis(1));
            awake.touch();
            std::thread::sleep(Duration::from_millis(50));
            std::process::exit(0);
        }
        for run in 0..8 {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
                .env("RVG_CHILD", TEST)
                .output()
                .unwrap();
            assert_eq!(out.status.code(), Some(0), "run {run}: {out:?}");
        }
    }

    /// A device's `Awake` of a test counts nothing and wakes nothing.
    #[test]
    fn a_default_awake_keeps_nothing() {
        let mut awake = Awake::default();
        awake.touch();
        assert!(awake.shared.is_none());
        awake.stop();
    }

    /// The device's end stops its thread and waits for it, at once wherever the thread is: parked
    /// for the first commit, waiting an hour for its next look after one, or touching every
    /// millisecond with a toucher made, which it destroys as it ends (the module docs).
    #[test]
    fn a_stop_ends_the_thread_and_waits_for_it() {
        let device = crate::gpu::device::tests::dev();
        let hour = Duration::from_secs(3600);
        for (look, touched) in [
            (hour, false),
            (hour, true),
            (Duration::from_millis(1), true),
        ] {
            let mut awake = Awake::looking(&device.raw, look);
            if touched {
                awake.touch();
                std::thread::sleep(Duration::from_millis(50));
            }
            let (done, stopped) = std::sync::mpsc::channel();
            let stopping = std::thread::Builder::new().spawn(move || {
                awake.stop();
                done.send(awake.thread.is_none()).unwrap();
            });
            stopping.unwrap();
            let ended = stopped.recv_timeout(Duration::from_secs(10));
            assert_eq!(
                ended,
                Ok(true),
                "looking every {look:?}, touched: {touched}"
            );
        }
    }
}
