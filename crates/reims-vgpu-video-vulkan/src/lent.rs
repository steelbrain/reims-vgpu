//! Copies out on a queue another submitter lends ([`LentQueue`]). Its own work runs on that
//! queue too, so a copy that waited there on the GPU for its decode would hold that work back
//! behind the decode: each copy is instead submitted from a thread of the device's once the host
//! sees the work it waits for done, with no wait of its own.

use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use ash::vk;

use crate::error::{Error, Named};

/// How long the copier waits for a copy's waits at a time, logging each time it waited this long.
const WAIT_STEP: Duration = Duration::from_secs(5);

/// A queue of the device's that another submitter also submits to, which it lends the decoders
/// for each of their copies out ([`crate::VideoDevice::from_raw_copying_on`]): a queue of a family
/// whose objects the decoders' outputs are, such as a buffer of the submitter's own that a
/// picture lands in.
pub trait LentQueue: Send + Sync {
    /// The queue's family, which copies (`TRANSFER`, `GRAPHICS` or `COMPUTE`).
    fn family(&self) -> u32;
    /// Calls `submit` with the queue, under the lock every submission to it takes. `submit`
    /// submits once, keeps nothing, and waits for nothing on the host.
    fn with_queue(&self, submit: &mut dyn FnMut(vk::Queue));
}

/// The commands a device submits with, which the copier submits with too.
#[derive(Clone)]
pub(crate) struct Submitter {
    pub(crate) device: ash::Device,
    /// `VK_KHR_synchronization2`'s commands, on an instance below Vulkan 1.3.
    pub(crate) synchronization2: Option<ash::khr::synchronization2::Device>,
}

impl Submitter {
    /// `vkQueueSubmit2`, or its `VK_KHR_synchronization2` alias.
    ///
    /// # Safety
    ///
    /// `queue` is the device's and held by the caller for the call; the submissions' command
    /// buffers are recorded and their semaphores, and `fence` where given, the device's.
    pub(crate) unsafe fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), Error> {
        match &self.synchronization2 {
            // SAFETY: as the caller promises.
            Some(khr) => unsafe { khr.queue_submit2(queue, submits, fence) },
            // SAFETY: as the caller promises.
            None => unsafe { self.device.queue_submit2(queue, submits, fence) },
        }
        .named("vkQueueSubmit2")
    }
}

/// One submission, read out of its `VkSubmitInfo2`.
struct Job {
    /// Timeline semaphores, and the values the host waits for each to reach.
    waits: Vec<(vk::Semaphore, u64)>,
    commands: Vec<vk::CommandBuffer>,
    /// Timeline semaphores, and the values the submission signals.
    signals: Vec<(vk::Semaphore, u64)>,
    fence: vk::Fence,
}

/// A lent queue, and the thread that submits the copies to it.
pub(crate) struct LentCopies {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
}

impl LentCopies {
    /// The copier of `lent`, on a thread of its own.
    pub(crate) fn new(lent: Box<dyn LentQueue>, submitter: Submitter) -> Result<Self, Error> {
        let lent: Arc<dyn LentQueue> = Arc::from(lent);
        let (jobs, taken) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("rvg-video-copies".into())
            .spawn(move || run(&taken, lent.as_ref(), &submitter))
            .map_err(|error| Error::Unsupported(format!("no thread for the copies: {error}")))?;
        Ok(Self {
            jobs: Some(jobs),
            thread: Some(thread),
        })
    }

    /// Hands `submits` to the copier, which submits each once the host sees its waits reached.
    ///
    /// # Safety
    ///
    /// As for [`Submitter::queue_submit2`], but for the queue; each wait is of a timeline
    /// semaphore, to a value a submission made or to be made before the copier's signals it; the
    /// command buffers, semaphores and fence live until the submissions' signals are reached.
    pub(crate) unsafe fn submit(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), Error> {
        let Some(jobs) = &self.jobs else {
            return Err(Error::Unsupported("the copier has ended".into()));
        };
        for (at, s) in submits.iter().enumerate() {
            // SAFETY: each count and pointer pair is the caller's, valid through the call; so
            // for the two below.
            let waits = unsafe { parts(s.p_wait_semaphore_infos, s.wait_semaphore_info_count) };
            // SAFETY: as above.
            let commands = unsafe { parts(s.p_command_buffer_infos, s.command_buffer_info_count) };
            // SAFETY: as above.
            let signals =
                unsafe { parts(s.p_signal_semaphore_infos, s.signal_semaphore_info_count) };
            let job = Job {
                waits: waits.iter().map(|w| (w.semaphore, w.value)).collect(),
                commands: commands.iter().map(|c| c.command_buffer).collect(),
                signals: signals.iter().map(|w| (w.semaphore, w.value)).collect(),
                fence: if at + 1 == submits.len() {
                    fence
                } else {
                    vk::Fence::null()
                },
            };
            jobs.send(job)
                .map_err(|_| Error::Unsupported("the copier has ended".into()))?;
        }
        Ok(())
    }
}

impl Drop for LentCopies {
    fn drop(&mut self) {
        // The copier submits what it holds, then ends.
        drop(self.jobs.take());
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            log::error!("the video copier ended in a panic");
        }
    }
}

/// `count` items from `items`; none for a count of 0.
///
/// # Safety
///
/// Where `count` is not 0, `items` points at `count` of them, valid for the borrow.
unsafe fn parts<'a, T>(items: *const T, count: u32) -> &'a [T] {
    if count == 0 || items.is_null() {
        return &[];
    }
    // SAFETY: as the caller promises.
    unsafe { std::slice::from_raw_parts(items, count as usize) }
}

/// The copier: each job once its waits are reached, in order.
fn run(taken: &mpsc::Receiver<Job>, lent: &dyn LentQueue, submitter: &Submitter) {
    let device = &submitter.device;
    for job in taken {
        if !wait(device, &job.waits) {
            continue;
        }
        let commands: Vec<_> = (job.commands.iter())
            .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
            .collect();
        let signals: Vec<_> = (job.signals.iter())
            .map(|&(semaphore, value)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(semaphore)
                    .value(value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let submits = [vk::SubmitInfo2::default()
            .command_buffer_infos(&commands)
            .signal_semaphore_infos(&signals)];
        let mut submitted = Err(Error::Unsupported("the lent queue lent nothing".into()));
        lent.with_queue(&mut |queue| {
            // SAFETY: the queue is the device's, held under the lender's lock for the call; the
            // job's command buffers are recorded and its semaphores and fence the device's
            // (`LentCopies::submit`'s word).
            submitted = unsafe { submitter.queue_submit2(queue, &submits, job.fence) };
        });
        let Err(error) = submitted else {
            continue;
        };
        log::error!("a picture's copy out on a lent queue was not submitted ({error})");
        // Its signals are raised from the host, so what waits for them goes on.
        for &(semaphore, value) in &job.signals {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(semaphore)
                .value(value);
            // SAFETY: the semaphore is the device's, a timeline semaphore, below `value`.
            if let Err(error) = unsafe { device.signal_semaphore(&info) } {
                log::error!("a copy's signal from the host: {error}");
            }
        }
    }
}

/// Waits on the host for each of `waits`; false, logged, where the device can no longer reach
/// them.
fn wait(device: &ash::Device, waits: &[(vk::Semaphore, u64)]) -> bool {
    if waits.is_empty() {
        return true;
    }
    let (semaphores, values): (Vec<_>, Vec<_>) = waits.iter().copied().unzip();
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    let step = u64::try_from(WAIT_STEP.as_nanos()).unwrap_or(u64::MAX);
    loop {
        // SAFETY: the semaphores are the device's timeline semaphores, alive until the job's
        // signals are reached (`LentCopies::submit`'s word).
        match unsafe { device.wait_semaphores(&info, step) } {
            Ok(()) => return true,
            Err(vk::Result::TIMEOUT) => {
                log::warn!("a picture's copy out has waited {WAIT_STEP:?} more for its decode");
            }
            Err(error) => {
                log::error!("a picture's copy out could not wait for its decode ({error})");
                return false;
            }
        }
    }
}
