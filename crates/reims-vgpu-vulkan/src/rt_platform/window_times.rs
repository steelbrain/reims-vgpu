//! A window's own times, for the host's summary of each ten seconds (reims-vgpu's
//! `graphics::summary`). Each kind is counted in full, every call or command buffer of it, with
//! its count, its time in all and the longest ([`take`]), and written as the summary's `window:`
//! part ([`line()`]):
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
//! - **`vkCreateSwapchainKHR`**: the CPU time inside each call of a window's swapchain, made or
//!   refused, holding the window system's lock: a window makes one at its first draw and at each
//!   remake (a new size, a swapchain out of date), and none once rt's device is lost
//!   (`super::swapchain`'s docs, "A lost device").
//!
//! The CPU times are on the trace's clock ([`crate::core::trace::now`]). The trace
//! (`REIMS_VGPU_TRACE`) times the same spans one by one where it is on: `winacquire` each take's
//! call, `winpresent` each present's, its lock waits with the window system's, and `cbdone` each
//! command buffer's GPU start and end, the window's among rt's. These counts give their sums
//! without it.
//!
//! A diagnostic: nothing is counted until the summary turns counting on ([`count`]), where the log
//! names its target, or reims-vgpu's trip line does ([`count_trip`]), where the log names its own;
//! off, each call costs one relaxed load. The two count apart, each in a set of its own that only
//! its take empties ([`take`], [`take_trip`]), so the summary's ten seconds stay whole beside the
//! trip line's seconds. A kind is three atomics, added to with no lock; a take swaps each for 0,
//! so a call counted while a take reads them can fall into either period, its count in one and its
//! time in the other. Which lines the host writes is free under the contract.
//!
//! [`CompletedInfo::gpu_times_ns`]: crate::core::platform::command::CompletedInfo::gpu_times_ns

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

/// Which sets count: [`SUMMARY_ON`] ([`count`]) and [`TRIP_ON`] ([`count_trip`]).
static ON: AtomicU8 = AtomicU8::new(0);

/// [`ON`]'s bit for the summary's set, [`SUMMARY`].
const SUMMARY_ON: u8 = 1;

/// [`ON`]'s bit for the trip line's set, [`TRIP`].
const TRIP_ON: u8 = 2;

/// The summary's set ([`take`]).
static SUMMARY: Kinds = Kinds::new();

/// The trip line's set ([`take_trip`]).
static TRIP: Kinds = Kinds::new();

/// A set's counts, one of each kind.
struct Kinds {
    /// The `vkAcquireNextImageKHR` calls.
    acquires: Counter,
    /// The `vkQueuePresentKHR` calls.
    presents: Counter,
    /// The presents' waits for rt's lock of its presenting queue.
    present_lock_waits: Counter,
    /// The window frames' command buffers on the GPU.
    frames_on_gpu: Counter,
    /// The `vkCreateSwapchainKHR` calls.
    creates: Counter,
}

impl Kinds {
    /// A set that has counted none.
    const fn new() -> Kinds {
        Kinds {
            acquires: Counter::new(),
            presents: Counter::new(),
            present_lock_waits: Counter::new(),
            frames_on_gpu: Counter::new(),
            creates: Counter::new(),
        }
    }

    /// What the set counted since its take before, of each kind; each is counted from none again.
    fn take(&self) -> Period {
        Period {
            acquires: self.acquires.take(),
            presents: self.presents.take(),
            present_lock_waits: self.present_lock_waits.take(),
            frames_on_gpu: self.frames_on_gpu.take(),
            creates: self.creates.take(),
        }
    }
}

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
    /// The `vkCreateSwapchainKHR` calls.
    pub creates: Spans,
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

/// Turns the summary's counting on, for the rest of the process.
pub fn count() {
    ON.fetch_or(SUMMARY_ON, Ordering::Relaxed);
}

/// Turns the trip line's counting on, for the rest of the process.
pub fn count_trip() {
    ON.fetch_or(TRIP_ON, Ordering::Relaxed);
}

/// Whether the times are counted, in either set: a caller reads its clock only then.
pub fn on() -> bool {
    ON.load(Ordering::Relaxed) != 0
}

/// `add` to each set that counts.
fn counted(add: impl Fn(&Kinds)) {
    let on = ON.load(Ordering::Relaxed);
    if on & SUMMARY_ON != 0 {
        add(&SUMMARY);
    }
    if on & TRIP_ON != 0 {
        add(&TRIP);
    }
}

/// A `vkAcquireNextImageKHR` call took `nanos`: counted where counting is on.
pub(crate) fn acquired(nanos: u64) {
    counted(|kinds| kinds.acquires.add(nanos));
}

/// A present waited `lock_wait` nanoseconds for rt's lock of its presenting queue, then its
/// `vkQueuePresentKHR` call took `call`: counted where counting is on.
pub(crate) fn presented(lock_wait: u64, call: u64) {
    counted(|kinds| {
        kinds.present_lock_waits.add(lock_wait);
        kinds.presents.add(call);
    });
}

/// A `vkCreateSwapchainKHR` call took `nanos`: counted where counting is on.
pub(crate) fn created(nanos: u64) {
    counted(|kinds| kinds.creates.add(nanos));
}

/// A window frame's command buffer completed with `gpu_times_ns`, its GPU start and end
/// ([`CompletedInfo::gpu_times_ns`]): counted where it has them and counting is on.
///
/// [`CompletedInfo::gpu_times_ns`]: crate::core::platform::command::CompletedInfo::gpu_times_ns
pub fn frame_ran(gpu_times_ns: Option<(u64, u64)>) {
    if let Some((start, end)) = gpu_times_ns {
        counted(|kinds| kinds.frames_on_gpu.add(end.saturating_sub(start)));
    }
}

/// What the summary's set counted since its take before, of each kind; each is counted from none
/// again.
pub fn take() -> Period {
    SUMMARY.take()
}

/// What the trip line's set counted since its take before, of each kind; each is counted from
/// none again.
pub fn take_trip() -> Period {
    TRIP.take()
}

/// The summary's part of `period`: `window: ...`, each kind by name with its count, its time in
/// all and the longest, the lock waits without their count, which is the presents'; the creates
/// last, after the parts written before they were counted.
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
        "window: {}; {}; presenting-queue lock waits {:.1} ms, the longest {:.1} ms; {}; {}",
        spans("vkAcquireNextImageKHR", &period.acquires),
        spans("vkQueuePresentKHR", &period.presents),
        ms(waits.total),
        ms(waits.longest),
        spans("frames on the GPU", &period.frames_on_gpu),
        spans("vkCreateSwapchainKHR", &period.creates),
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
        created(700_000);
        assert_eq!(take(), Period::default());
        count();
        acquired(1_000_000);
        acquired(50_000);
        presented(1_000, 2_000_000);
        presented(400_000, 500_000);
        frame_ran(Some((10, 300_010)));
        frame_ran(Some((5_000, 105_000)));
        frame_ran(None);
        created(700_000);
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
                creates: spans(1, 700_000, 700_000),
            }
        );
        assert_eq!(take(), Period::default());
    }

    /// The trip line's set counts apart from the summary's: each counts only once on, and a take
    /// of one leaves the other's counts whole
    #[test]
    fn the_trip_line_s_times_are_counted_apart_from_the_summary_s() {
        count_trip();
        assert!(on());
        acquired(1_000);
        presented(10, 2_000);
        let spans = |count, total, longest| Spans {
            count,
            total: Duration::from_nanos(total),
            longest: Duration::from_nanos(longest),
        };
        assert_eq!(take(), Period::default(), "the summary's set is not on");
        count();
        frame_ran(Some((100, 400)));
        assert_eq!(
            take(),
            Period {
                frames_on_gpu: spans(1, 300, 300),
                ..Period::default()
            }
        );
        assert_eq!(
            take_trip(),
            Period {
                acquires: spans(1, 1_000, 1_000),
                presents: spans(1, 2_000, 2_000),
                present_lock_waits: spans(1, 10, 10),
                frames_on_gpu: spans(1, 300, 300),
                ..Period::default()
            }
        );
        assert_eq!(take_trip(), Period::default());
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
            creates: spans(2, 1_480, 910),
        };
        assert_eq!(
            line(&period),
            "window: vkAcquireNextImageKHR 412 in 3.1 ms, the longest 0.9 ms; vkQueuePresentKHR \
             238 in 41.6 ms, the longest 2.3 ms; presenting-queue lock waits 0.4 ms, the longest \
             0.2 ms; frames on the GPU 238 in 30.9 ms, the longest 0.3 ms; vkCreateSwapchainKHR \
             2 in 1.5 ms, the longest 0.9 ms"
        );
        assert_eq!(
            line(&Period::default()),
            "window: vkAcquireNextImageKHR 0 in 0.0 ms, the longest 0.0 ms; vkQueuePresentKHR 0 \
             in 0.0 ms, the longest 0.0 ms; presenting-queue lock waits 0.0 ms, the longest 0.0 \
             ms; frames on the GPU 0 in 0.0 ms, the longest 0.0 ms; vkCreateSwapchainKHR 0 in 0.0 \
             ms, the longest 0.0 ms"
        );
    }
}
