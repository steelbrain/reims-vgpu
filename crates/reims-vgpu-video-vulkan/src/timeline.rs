//! A decoder's timeline semaphore, which each decode submission signals
//! with the next value, and what waits on it before it is let go: Vulkan
//! objects that GPU work still in flight uses are destroyed only once the
//! semaphore has passed that work's value.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use ash::vk;
use reims_vgpu_video_core::sync::Mutex;

use crate::device::VideoDevice;
use crate::error::{Error, Named};

/// How long a teardown waits for the GPU's work before it gives up and
/// leaks what that work uses, rather than destroy it under the GPU.
const TEARDOWN_WAIT: Duration = Duration::from_secs(10);

/// A decoder's timeline: its semaphore, the last value submitted, and what
/// waits for the GPU to pass a value before it drops.
pub(crate) struct Timeline {
    device: Arc<VideoDevice>,
    semaphore: vk::Semaphore,
    /// The value of the last submission made: the semaphore reaches it
    /// once every one is done.
    submitted: AtomicU64,
    /// Each with the value the GPU must pass before it drops.
    retired: Mutex<Vec<(u64, Box<dyn Send>)>>,
}

impl Timeline {
    pub(crate) fn new(device: &Arc<VideoDevice>) -> Result<Arc<Self>, Error> {
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: `info` and its chain live through the call.
        let semaphore =
            unsafe { device.device().create_semaphore(&info, None) }.named("vkCreateSemaphore")?;
        Ok(Arc::new(Self {
            device: Arc::clone(device),
            semaphore,
            submitted: AtomicU64::new(0),
            retired: Mutex::new(Vec::new()),
        }))
    }

    /// The semaphore.
    pub(crate) fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }

    /// The value the next submission signals.
    pub(crate) fn next(&self) -> u64 {
        self.submitted.load(Ordering::Acquire) + 1
    }

    /// Records that a submission signalling `value` was made.
    pub(crate) fn submitted(&self, value: u64) {
        self.submitted.store(value, Ordering::Release);
    }

    /// The value of the last submission made.
    pub(crate) fn last_submitted(&self) -> u64 {
        self.submitted.load(Ordering::Acquire)
    }

    /// The value the GPU has reached; 0 where the device cannot say, as
    /// after it is lost, so that nothing is let go on its word.
    pub(crate) fn completed(&self) -> u64 {
        // SAFETY: the semaphore is the device's.
        unsafe {
            self.device
                .device()
                .get_semaphore_counter_value(self.semaphore)
        }
        .unwrap_or(0)
    }

    /// Waits for the GPU to reach `value`, for `timeout` at most.
    pub(crate) fn wait(&self, value: u64, timeout: Duration) -> Result<(), Error> {
        let semaphores = [self.semaphore];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let nanos = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: `info` and its arrays live through the call.
        match unsafe { self.device.device().wait_semaphores(&info, nanos) } {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => Err(Error::Timeout),
            Err(result) => Err(Error::Vulkan("vkWaitSemaphores", result)),
        }
    }

    /// Drops `item` once the GPU has reached `value`: now, where it has.
    pub(crate) fn retire(&self, value: u64, item: Box<dyn Send>) {
        if value <= self.completed() {
            drop(item);
        } else {
            self.retired.lock().push((value, item));
        }
    }

    /// Drops what waited for values the GPU has reached.
    pub(crate) fn collect(&self) {
        let completed = self.completed();
        let done: Vec<Box<dyn Send>> = {
            let mut retired = self.retired.lock();
            let (done, waiting) = std::mem::take(&mut *retired)
                .into_iter()
                .partition::<Vec<_>, _>(|(value, _)| *value <= completed);
            *retired = waiting;
            done.into_iter().map(|(_, item)| item).collect()
        };
        drop(done);
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        let retired = std::mem::take(&mut *self.retired.lock());
        match self.wait(self.last_submitted(), TEARDOWN_WAIT) {
            Ok(()) => {
                drop(retired);
                // SAFETY: the semaphore is the device's, and every
                // submission that signals it is done.
                unsafe { self.device.device().destroy_semaphore(self.semaphore, None) };
            }
            Err(error) => {
                // Destroying them under the GPU would be worse than
                // leaking them.
                log::error!(
                    "a decoder's GPU work did not finish ({error}): leaking {} objects it uses",
                    retired.len()
                );
                std::mem::forget(retired);
            }
        }
    }
}
