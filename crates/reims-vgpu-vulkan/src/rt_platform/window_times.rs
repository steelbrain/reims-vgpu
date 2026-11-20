//! A window's own times, for the host's summary of each ten seconds (reims-vgpu's
//! `graphics::summary`). Each kind is counted in full, every call or command buffer of it, with
//! its count, its time in all and the longest ([`take`]), and written as the summary's `window:`
//! part ([`line`]):
//! - **`vkAcquireNextImageKHR`**: the CPU time inside each call of a window's swapchain
//!   (`super::swapchain`), from the call's start, holding the window system's lock, to its
//!   return. A take of an image makes one call or more, each waiting up to 1 ms
//!   (`crate::swapchain`'s `take_image`).
//! - **`vkQueuePresentKHR`**: the CPU time inside each call, from its start, holding the window
//!   system's lock and rt's lock of its presenting queue, to its return; and apart, each
//!   present's wait for that lock of rt's (`metal2vulkan_runtime::Device::with_present_queue`),
//!   from the window system's lock taken to the call's start. Where the device's queue family has
//!   one queue, that lock is the one every submission of rt's takes.
//! - **Frames on the GPU**: each window frame's command buffer (reims-vgpu's
//!   `graphics::rt_window`: the display drawn into the swapchain's image, committed on rt's
//!   queue), from its GPU start to its GPU end as its completed handler learns them
//!   ([`CompletedInfo::gpu_times_ns`]), the span of rt's two timestamps of it. One that ended
//!   with an error has no GPU times, and is not counted.
//!
//! The CPU times are on the trace's clock ([`crate::core::trace::now`]). The trace
//! (`REIMS_VGPU_TRACE`) times the same spans one by one where it is on: `winacquire` each take's
//! call, `winpresent` each present's, its lock waits with the window system's, and `cbdone` each
//! command buffer's GPU start and end, the window's among rt's. These counts give their sums
//! without it.
//!
//! A diagnostic: nothing is counted until the summary turns counting on ([`count`]), where the log
//! names its target; off, each call costs one relaxed load. A kind is three atomics, added to with
//! no lock; a take swaps each for 0, so a call counted while a take reads them can fall into
//! either period, its count in one and its time in the other. Which lines the host writes is free
//! under the contract.
//!
//! [`CompletedInfo::gpu_times_ns`]: crate::core::platform::command::CompletedInfo::gpu_times_ns

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

/// Whether the times are counted ([`count`]).
static ON: AtomicBool = AtomicBool::new(false);

/// The `vkAcquireNextImageKHR` calls.
static ACQUIRES: Counter = Counter::new();

/// The `vkQueuePresentKHR` calls.
static PRESENTS: Counter = Counter::new();

/// The presents' waits for rt's lock of its presenting queue.
static PRESENT_LOCK_WAITS: Counter = Counter::new();

/// The window frames' command buffers on the GPU.
static FRAMES_ON_GPU: Counter = Counter::new();

/// What one kind counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Spans {
    /// How many.
    pub count: u64,
    /// Their time in all.
    pub total: Duration,
    /// The longest.
    pub longest: Duration,
}

/// What a period counted of each kind (the module doc), since the [`take`] before.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Period {
    /// The `vkAcquireNextImageKHR` calls.
    pub acquires: Spans,
    /// The `vkQueuePresentKHR` calls.
    pub presents: Spans,
    /// The presents' waits for rt's lock of its presenting queue, one a present.
    pub present_lock_waits: Spans,
    /// The window frames' command buffers on the GPU.
    pub frames_on_gpu: Spans,
}

/// One kind's count, nanoseconds in all and longest, since the take before.
struct Counter {
    count: AtomicU64,
    nanos: AtomicU64,
    longest: AtomicU64,
}

impl Counter {
    const fn new() -> Counter {
        Counter {
            count: AtomicU64::new(0),
            nanos: AtomicU64::new(0),
            longest: AtomicU64::new(0),
        }
    }

    /// One more, of `nanos`.
    fn add(&self, nanos: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.nanos.fetch_add(nanos, Ordering::Relaxed);
        self.longest.fetch_max(nanos, Ordering::Relaxed);
    }

    /// What it counted, counting from none again.
    fn take(&self) -> Spans {
        Spans {
            count: self.count.swap(0, Ordering::Relaxed),
            total: Duration::from_nanos(self.nanos.swap(0, Ordering::Relaxed)),
            longest: Duration::from_nanos(self.longest.swap(0, Ordering::Relaxed)),
        }
    }
}

/// Turns counting on, for the rest of the process.
pub fn count() {
    ON.store(true, Ordering::Relaxed);
}

/// Whether the times are counted: a caller reads its clock only then.
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// A `vkAcquireNextImageKHR` call took `nanos`: counted where counting is on.
pub(crate) fn acquired(nanos: u64) {
    if on() {
        ACQUIRES.add(nanos);
    }
}

/// A present waited `lock_wait` nanoseconds for rt's lock of its presenting queue, then its
/// `vkQueuePresentKHR` call took `call`: counted where counting is on.
pub(crate) fn presented(lock_wait: u64, call: u64) {
    if on() {
        PRESENT_LOCK_WAITS.add(lock_wait);
        PRESENTS.add(call);
    }
}

/// A window frame's command buffer completed with `gpu_times_ns`, its GPU start and end
/// ([`CompletedInfo::gpu_times_ns`]): counted where it has them and counting is on.
///
/// [`CompletedInfo::gpu_times_ns`]: crate::core::platform::command::CompletedInfo::gpu_times_ns
pub fn frame_ran(gpu_times_ns: Option<(u64, u64)>) {
    if let Some((start, end)) = gpu_times_ns
        && on()
    {
        FRAMES_ON_GPU.add(end.saturating_sub(start));
    }
}

/// What was counted since the take before, of each kind; each is counted from none again.
pub fn take() -> Period {
    Period {
        acquires: ACQUIRES.take(),
        presents: PRESENTS.take(),
        present_lock_waits: PRESENT_LOCK_WAITS.take(),
        frames_on_gpu: FRAMES_ON_GPU.take(),
    }
}

/// The summary's part of `period`: `window: ...`, each kind by name with its count, its time in
/// all and the longest, the lock waits without their count, which is the presents'.
pub fn line(period: &Period) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let spans = |name: &str, s: &Spans| {
        format!(
            "{name} {} in {:.1} ms, the longest {:.1} ms",
            s.count,
            ms(s.total),
            ms(s.longest)
        )
    };
    let waits = &period.present_lock_waits;
    format!(
        "window: {}; {}; presenting-queue lock waits {:.1} ms, the longest {:.1} ms; {}",
        spans("vkAcquireNextImageKHR", &period.acquires),
        spans("vkQueuePresentKHR", &period.presents),
        ms(waits.total),
        ms(waits.longest),
        spans("frames on the GPU", &period.frames_on_gpu),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off, nothing is counted; on, every call of each kind is, short ones included, with its
    /// count, time in all and longest; a frame without GPU times is not; a take answers them
    /// once.
    #[test]
    fn a_windows_times_are_counted_in_full_once_on() {
        acquired(1_000_000);
        presented(1_000, 2_000_000);
        frame_ran(Some((10, 300_010)));
        assert_eq!(take(), Period::default());
        count();
        acquired(1_000_000);
        acquired(50_000);
        presented(1_000, 2_000_000);
        presented(400_000, 500_000);
        frame_ran(Some((10, 300_010)));
        frame_ran(Some((5_000, 105_000)));
        frame_ran(None);
        let spans = |count, total, longest| Spans {
            count,
            total: Duration::from_nanos(total),
            longest: Duration::from_nanos(longest),
        };
        assert_eq!(
            take(),
            Period {
                acquires: spans(2, 1_050_000, 1_000_000),
                presents: spans(2, 2_500_000, 2_000_000),
                present_lock_waits: spans(2, 401_000, 400_000),
                frames_on_gpu: spans(2, 400_000, 300_000),
            }
        );
        assert_eq!(take(), Period::default());
    }

    /// The part names each kind with its count, time in all and longest, in milliseconds to a
    /// tenth, the lock waits without a count; a period with none says 0 of each.
    #[test]
    fn the_window_part_names_each_kind() {
        let spans = |count, total_us, longest_us| Spans {
            count,
            total: Duration::from_micros(total_us),
            longest: Duration::from_micros(longest_us),
        };
        let period = Period {
            acquires: spans(412, 3_140, 920),
            presents: spans(238, 41_600, 2_260),
            present_lock_waits: spans(238, 380, 160),
            frames_on_gpu: spans(238, 30_940, 310),
        };
        assert_eq!(
            line(&period),
            "window: vkAcquireNextImageKHR 412 in 3.1 ms, the longest 0.9 ms; vkQueuePresentKHR \
             238 in 41.6 ms, the longest 2.3 ms; presenting-queue lock waits 0.4 ms, the longest \
             0.2 ms; frames on the GPU 238 in 30.9 ms, the longest 0.3 ms"
        );
        assert_eq!(
            line(&Period::default()),
            "window: vkAcquireNextImageKHR 0 in 0.0 ms, the longest 0.0 ms; vkQueuePresentKHR 0 \
             in 0.0 ms, the longest 0.0 ms; presenting-queue lock waits 0.0 ms, the longest 0.0 \
             ms; frames on the GPU 0 in 0.0 ms, the longest 0.0 ms"
        );
    }
}
