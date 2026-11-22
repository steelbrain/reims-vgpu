//! The host's summary of each ten seconds in its log, for the walks that time a guest's frames.
//! Every [`PERIOD`], while the log names [`TARGET`] at debug, the summary writes up to seven
//! lines of what the period held. Each line starts with its stamp: the host's wall time
//! (`CLOCK_REALTIME`, seconds since the Unix epoch to the millisecond) and the trace's clock in
//! nanoseconds (`reims_vgpu_vulkan::core::trace::now`, rt's host clock, which `cbdone` and the
//! trace's other events count: `CLOCK_MONOTONIC` on Linux), so a reader joins the lines to its own
//! steps by the wall time, and to a trace with no conversion on any host:
//!
//! - **Presents**: the presents answered, and how many a second. Then the time from each one's
//!   take to its show, from its show to its answer, and from its take to its answer, at the 50th,
//!   90th and 99th percentiles by nearest rank ([`answered`], which `present`'s
//!   `PresentAnswers::refreshed` calls, and `PresentAnswers::answered_at_show` for a display with
//!   no window to draw it).
//! - **Window**: the window's own times (`reims_vgpu_vulkan::rt_platform::window_times`): the CPU
//!   time inside its swapchain's `vkAcquireNextImageKHR` and `vkQueuePresentKHR` calls, its
//!   presents' waits for rt's lock of the presenting queue, its frames' command buffers' GPU
//!   time, and the CPU time inside its `vkCreateSwapchainKHR` calls, each with its count, its time
//!   in all and the longest (the waits with no count: one a present).
//! - **rt**: rt's part (`reims_vgpu_vulkan::rt_platform::summary`), once its device is made.
//! - **Managed**: the ranges of `Managed` buffers the host carried out, the app's range messages
//!   and their bytes apart from the upload chunks of `set*Bytes:` data that command buffers' own
//!   streams name, their bytes and those buffers ([`managed`], whose doc says what each counts).
//! - **Waits**: our waits of 5 ms or more, by name, each with its count, its time in all and the
//!   longest (`reims_vgpu_vulkan::core::long_waits`, whose doc names them).
//! - **Kept**: what the decoders hold for the guest now, by holder, and the platform's textures
//!   and buffers no guest id names; and the IOSurfaces' device caches holding the most, one by
//!   one ([`kept`], whose doc says what each counts), where a decoder lives.
//!
//! - **Asked for by name**: [`start`] turns the summary on, and the long waits', the window's
//!   times' and the `Managed` ranges' counting, only where `REIMS_VGPU_LOG` names the target
//!   itself at debug as the first decoder is made (`reims_vgpu_vulkan::core::logging::names`), as
//!   `decode::work` does, and makes its thread then.
//! - **Off**, a present's answer costs one relaxed load, and so do a wait, a window's call and a
//!   `Managed` range.
//!
//! A diagnostic: which lines the host writes is free under the contract.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_vulkan::core::long_waits::{self, Waits};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, Deadline, MonoTime, WallTime};
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::rt_platform::{self, window_times};

pub(crate) mod kept;
pub(crate) mod managed;

/// The log target of the lines, at debug, which the filter names to start them ([`start`]).
pub(crate) const TARGET: &str = "reims_vgpu::summary";

/// How often the thread writes its lines.
pub(crate) const PERIOD: Duration = Duration::from_secs(10);

/// Whether the summary is on ([`start`]).
static ON: AtomicBool = AtomicBool::new(false);

/// The presents answered since the last line: each one's time from its take to its show, from its
/// show to its answer, and from its take to its answer.
static PRESENTS: Mutex<Vec<[Duration; 3]>> = Mutex::new(Vec::new());

/// Whether the summary is on.
pub(crate) fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// A present taken at `taken` and shown at `shown` is answered at `answered`, on the platform's
/// monotonic clock: counted in the next line where the summary is on.
pub(crate) fn answered(taken: MonoTime, shown: MonoTime, answered: MonoTime) {
    if !on() {
        return;
    }
    PRESENTS.lock().push([
        shown.saturating_duration_since(taken),
        answered.saturating_duration_since(shown),
        answered.saturating_duration_since(taken),
    ]);
}

/// Turns the summary's count of presents on, with no thread to write its lines (for the tests,
/// each in a process of its own under nextest).
#[cfg(test)]
pub(crate) fn count_presents() {
    ON.store(true, Ordering::Relaxed);
}

/// The presents counted since the last take (for the tests).
#[cfg(test)]
pub(crate) fn take_presents() -> Vec<[Duration; 3]> {
    std::mem::take(&mut *PRESENTS.lock())
}

/// Turns the summary on and starts the thread that writes its lines (the module docs), once,
/// where the filter names the target at debug and its lines are written, its ticks and stamps on
/// `platform`'s clock.
pub(crate) fn start<P: Platform>(platform: &Arc<P>) {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if !(reims_vgpu_vulkan::core::logging::names(TARGET, log::Level::Debug)
            && log::log_enabled!(target: TARGET, log::Level::Debug))
        {
            return;
        }
        let platform = Arc::clone(platform);
        let made = std::thread::Builder::new()
            .name("reims-vgpu-summary".into())
            .spawn(move || {
                let clock = platform.clock();
                let mut rt = reims_vgpu_vulkan::rt_platform::summary::Part::default();
                let mut since = clock.now_mono();
                loop {
                    clock.sleep_until(Deadline::Mono(since.saturating_add(PERIOD)));
                    let now = clock.now_mono();
                    let over = now.saturating_duration_since(since);
                    since = now;
                    let stamp = stamp(clock.now_wall(), reims_vgpu_vulkan::core::trace::now());
                    let mut presents = std::mem::take(&mut *PRESENTS.lock());
                    let rt_live = rt_platform::summary::live_textures_and_buffers();
                    let [kept_part, largest_part] = kept::take(rt_live);
                    let parts = [
                        Some(presents_part(over, &mut presents)),
                        Some(window_times::line(&window_times::take())),
                        rt.take(),
                        Some(managed::line(&managed::take())),
                        Some(waits_part(&long_waits::take())),
                        kept_part,
                        largest_part,
                    ];
                    for part in parts.into_iter().flatten() {
                        log::debug!(
                            target: TARGET,
                            "{stamp} over {:.1} s: {part}",
                            over.as_secs_f64()
                        );
                    }
                }
            });
        if made.is_ok() {
            long_waits::count();
            window_times::count();
            managed::count();
            ON.store(true, Ordering::Relaxed);
        }
    });
}

/// A line's stamp at `wall`, with `trace`, the trace's clock in nanoseconds.
fn stamp(wall: WallTime, trace: u64) -> String {
    let ms = wall.as_nanos() / 1_000_000;
    format!("summary at {}.{:03} trace {trace}", ms / 1_000, ms % 1_000)
}

/// The presents' part of a line `over` that time, of `presents`, each one's times as
/// [`PRESENTS`] keeps them (sorted here).
fn presents_part(over: Duration, presents: &mut [[Duration; 3]]) -> String {
    let n = presents.len();
    let mut part = format!(
        "presents: {n} answered, {:.1} a second",
        n as f64 / over.as_secs_f64().max(1e-9)
    );
    if n == 0 {
        return part;
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    // The nearest rank of percentile `p` of `n`: its 1-based rank is p * n / 100 rounded up.
    let rank = |p: usize| (p * n).div_ceil(100).max(1) - 1;
    for (i, name) in ["take to show", "show to answer", "take to answer"]
        .into_iter()
        .enumerate()
    {
        presents.sort_unstable_by_key(|times| times[i]);
        let at = |p: usize| ms(presents[rank(p)][i]);
        let _ = write!(
            part,
            "; {name} p50 {:.1} p90 {:.1} p99 {:.1} ms",
            at(50),
            at(90),
            at(99)
        );
    }
    part
}

/// The waits' part of a line, of `waits` by name.
fn waits_part(waits: &[(&'static str, Waits)]) -> String {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let by_name: Vec<String> = (waits.iter())
        .map(|(name, w)| {
            format!(
                "{name} {} in {:.1} ms, the longest {:.1} ms",
                w.count,
                ms(w.total),
                ms(w.longest)
            )
        })
        .collect();
    if by_name.is_empty() {
        "waits of 5 ms or more: none".to_owned()
    } else {
        format!("waits of 5 ms or more: {}", by_name.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stamp gives the wall time to the millisecond, and the trace's clock.
    #[test]
    fn a_stamp_gives_the_wall_time_to_the_millisecond_and_the_trace_clock() {
        let wall = WallTime::from_nanos(1_795_190_400_123_456_789);
        assert_eq!(stamp(wall, 98_765), "summary at 1795190400.123 trace 98765");
        assert_eq!(
            stamp(WallTime::from_nanos(5_000_000), 0),
            "summary at 0.005 trace 0"
        );
    }

    /// The presents' part counts them, a second, and gives each time's percentiles by nearest
    /// rank; with none, it says so alone.
    #[test]
    fn the_presents_part_gives_each_times_percentiles_by_nearest_rank() {
        let ms = Duration::from_millis;
        assert_eq!(
            presents_part(PERIOD, &mut []),
            "presents: 0 answered, 0.0 a second"
        );
        // Takes to shows of 1 to 100 ms, shows to answers of 100 down to 1 ms.
        let mut presents: Vec<[Duration; 3]> =
            (1..=100).map(|i| [ms(i), ms(101 - i), ms(101)]).collect();
        assert_eq!(
            presents_part(PERIOD, &mut presents),
            "presents: 100 answered, 10.0 a second; take to show p50 50.0 p90 90.0 p99 99.0 ms; \
             show to answer p50 50.0 p90 90.0 p99 99.0 ms; take to answer p50 101.0 p90 101.0 \
             p99 101.0 ms"
        );
        let mut one = [[ms(4), ms(8), ms(12)]];
        assert_eq!(
            presents_part(PERIOD, &mut one),
            "presents: 1 answered, 0.1 a second; take to show p50 4.0 p90 4.0 p99 4.0 ms; show \
             to answer p50 8.0 p90 8.0 p99 8.0 ms; take to answer p50 12.0 p90 12.0 p99 12.0 ms"
        );
    }

    /// The waits' part names each wait with its count, time and longest, or says there were none.
    #[test]
    fn the_waits_part_names_each_wait() {
        assert_eq!(waits_part(&[]), "waits of 5 ms or more: none");
        let w = Waits {
            count: 2,
            total: Duration::from_micros(15_500),
            longest: Duration::from_millis(9),
        };
        assert_eq!(
            waits_part(&[("lane-turn", w), ("pool-take", w)]),
            "waits of 5 ms or more: lane-turn 2 in 15.5 ms, the longest 9.0 ms; pool-take 2 in \
             15.5 ms, the longest 9.0 ms"
        );
    }
}
