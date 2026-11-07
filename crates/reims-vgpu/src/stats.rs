//! Frame statistics, a diagnostic. With `REIMS_VGPU_STATS` on (`1`, `true`, `yes` or `on`), each
//! display gets a [`DisplayStats`], and the library's thread `reims-vgpu.stats` writes one line
//! per live display to stderr every second, by port:
//!
//! ```text
//! reims-vgpu stats: port 0: guest 59.9/s, max gap 17.2 ms; window 60.0/s, max gap 16.9 ms; window GPU 0.41 ms avg, 0.97 ms max
//! ```
//!
//! - *guest*: the frames the guest presented, as the display's new-frame callback reports them,
//!   per second, and the longest time between two of them.
//! - *window*, for a display made with a window: the frames its sink drew, per second, and the
//!   longest time between two.
//! - *window GPU*: the GPU time of the sink's draws, as the sink measures it, the mean and the
//!   longest; `-` while the sink has reported none. The Vulkan platform's sink measures it (its
//!   command buffers' GPU end less their start, as the runtime answers them), on either OS; the
//!   Metal platform's does not.
//!
//! A rate is over the time since the display's last line. A gap still open when the line is
//! written counts too, so a guest that stops presenting shows a gap that grows by a second each
//! line; `-` until the first frame.
//!
//! The inputs are three hooks. [`DisplayStats::guest_present`] is called by the closure
//! [`observe`] wraps around the display's new-frame callback (for a window display, the window's
//! tee). A window's sink counts each frame it commits with [`DisplayStats::window_present`], and
//! the Vulkan platform's sink reports each one's GPU time with [`DisplayStats::window_gpu_time`],
//! through
//! the stats [`of_display`] finds for its display (`window::SinkDisplay`). Each reads the platform's monotonic clock and updates atomics: no lock,
//! no wait, nothing the guest or the host sees. With the switch off, no display has stats,
//! nothing is wrapped and no thread is made.
//!
//! The lines go to stderr directly, not through the log, so that the switch alone shows them. A
//! failed write is ignored.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use reims_vgpu_core::display::Display as CoreDisplay;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::os::{Deadline, MonoTime, Threads, TimerClock};
use reims_vgpu_core::sync::{Condvar, Mutex};

use crate::NewFrameFn;

/// The switch.
const STATS_VAR: &str = "REIMS_VGPU_STATS";

/// The time between two reports.
const PERIOD: Duration = Duration::from_secs(1);

/// A [`Presents`] field's value before the first frame.
const NONE: u64 = u64::MAX;

/// Whether the environment turns the statistics on, read once per process. A value that is not
/// UTF-8 counts as unset.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(STATS_VAR).is_ok_and(|v| switch(&v)))
}

/// Whether a switch's value turns it on.
fn switch(value: &str) -> bool {
    let value = value.trim();
    ["1", "true", "yes", "on"]
        .iter()
        .any(|on| value.eq_ignore_ascii_case(on))
}

/// The frames of one kind: how many since the last report, when the last was, and the longest
/// time between two since the last report (nanoseconds on the monotonic clock).
struct Presents {
    count: AtomicU64,
    last: AtomicU64,
    max_gap: AtomicU64,
}

impl Presents {
    fn new() -> Presents {
        Presents {
            count: AtomicU64::new(0),
            last: AtomicU64::new(NONE),
            max_gap: AtomicU64::new(0),
        }
    }

    /// A frame at `now`.
    fn record(&self, now: u64) {
        self.count.fetch_add(1, Relaxed);
        let last = self.last.swap(now, Relaxed);
        if last != NONE {
            self.max_gap.fetch_max(now.saturating_sub(last), Relaxed);
        }
    }

    /// The frames since the last report, and the longest gap among them and the one still open
    /// at `now` (`None` before the first frame); starts the next period.
    fn take(&self, now: u64) -> (u64, Option<u64>) {
        let count = self.count.swap(0, Relaxed);
        let ended = self.max_gap.swap(0, Relaxed);
        let last = self.last.load(Relaxed);
        let gap = (last != NONE).then(|| ended.max(now.saturating_sub(last)));
        (count, gap)
    }
}

/// The GPU times of the window's draws since the last report: how many, their sum and the
/// longest (nanoseconds).
#[derive(Default)]
struct GpuTimes {
    count: AtomicU64,
    total: AtomicU64,
    max: AtomicU64,
}

/// One display's statistics (module doc).
pub(crate) struct DisplayStats {
    /// The display's port.
    port: u64,
    /// Whether the display was made with a window.
    window: bool,
    /// The platform's clock.
    clock: Arc<dyn TimerClock>,
    /// The guest's presents.
    guest: Presents,
    /// The window's draws.
    drawn: Presents,
    /// The window's GPU times.
    gpu: GpuTimes,
    /// When the period of the next report started.
    since: AtomicU64,
}

impl DisplayStats {
    /// The stats of the display on `port`, from `now` on `clock`.
    fn new(clock: Arc<dyn TimerClock>, port: u64, window: bool) -> DisplayStats {
        let since = AtomicU64::new(mono_now(&*clock));
        DisplayStats {
            port,
            window,
            clock,
            guest: Presents::new(),
            drawn: Presents::new(),
            gpu: GpuTimes::default(),
            since,
        }
    }

    /// The guest presented a frame.
    pub(crate) fn guest_present(&self) {
        self.guest.record(mono_now(&*self.clock));
    }

    /// The window's sink drew a frame.
    pub(crate) fn window_present(&self) {
        self.drawn.record(mono_now(&*self.clock));
    }

    /// One of the window's draws took `gpu` of GPU time.
    #[cfg_attr(
        not(any(test, vulkan_platform)),
        expect(
            dead_code,
            reason = "only the Vulkan platform's sink measures its frames' GPU time"
        )
    )]
    pub(crate) fn window_gpu_time(&self, gpu: Duration) {
        let ns = u64::try_from(gpu.as_nanos()).unwrap_or(u64::MAX);
        self.gpu.count.fetch_add(1, Relaxed);
        self.gpu.total.fetch_add(ns, Relaxed);
        self.gpu.max.fetch_max(ns, Relaxed);
    }

    /// The line for the period up to now, which starts the next.
    fn report(&self) -> String {
        let now = mono_now(&*self.clock);
        let since = self.since.swap(now, Relaxed);
        let seconds = Duration::from_nanos(now.saturating_sub(since)).as_secs_f64();
        let mut line = format!("reims-vgpu stats: port {}: ", self.port);
        presents(&mut line, "guest", self.guest.take(now), seconds);
        if self.window {
            line.push_str("; ");
            presents(&mut line, "window", self.drawn.take(now), seconds);
            line.push_str("; window GPU ");
            let count = self.gpu.count.swap(0, Relaxed);
            let total = self.gpu.total.swap(0, Relaxed);
            let max = self.gpu.max.swap(0, Relaxed);
            match total.checked_div(count) {
                None => line.push('-'),
                Some(mean) => {
                    let _ = write!(
                        line,
                        "{:.2} ms avg, {:.2} ms max",
                        millis(mean),
                        millis(max)
                    );
                }
            }
        }
        line
    }
}

/// `kind`'s rate over `seconds` and its longest gap, onto `line`.
fn presents(line: &mut String, kind: &str, (count, gap): (u64, Option<u64>), seconds: f64) {
    let rate = if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    };
    let _ = write!(line, "{kind} {rate:.1}/s, max gap ");
    match gap {
        Some(gap) => {
            let _ = write!(line, "{:.1} ms", millis(gap));
        }
        None => line.push('-'),
    }
}

/// Nanoseconds as milliseconds.
fn millis(ns: u64) -> f64 {
    Duration::from_nanos(ns).as_secs_f64() * 1000.0
}

/// Nanoseconds now on `clock`'s monotonic clock.
fn mono_now(clock: &dyn TimerClock) -> u64 {
    clock.now_on(Deadline::Mono(MonoTime::from_nanos(0)))
}

/// With the statistics on, the stats of a new display on `port`, made with a window or not, and
/// `new_frame` wrapped to count the guest's presents before it runs: `None`, with `new_frame`
/// untouched, when they are off, when there is no new-frame callback to count them by, or when
/// the platform has no clock. The callback stays present or absent as it was, which the guest
/// reads.
pub(crate) fn observe(
    threads: &dyn Threads,
    port: u64,
    window: bool,
    new_frame: &mut Option<NewFrameFn>,
) -> Option<Arc<DisplayStats>> {
    if !on() {
        return None;
    }
    wrap(threads.timer_clock(), port, window, new_frame)
}

/// [`observe`] with the statistics on, on `clock`.
fn wrap(
    clock: Option<Arc<dyn TimerClock>>,
    port: u64,
    window: bool,
    new_frame: &mut Option<NewFrameFn>,
) -> Option<Arc<DisplayStats>> {
    let clock = clock?;
    let inner = new_frame.take()?;
    let stats = Arc::new(DisplayStats::new(clock, port, window));
    *new_frame = Some(counting(&stats, inner));
    Some(stats)
}

/// `inner`, after counting each call as one of the guest's presents in `stats`.
fn counting(stats: &Arc<DisplayStats>, inner: NewFrameFn) -> NewFrameFn {
    let stats = Arc::clone(stats);
    Arc::new(move || {
        stats.guest_present();
        inner();
    })
}

/// The stats of each display made, by its core display's address, until they end.
struct Registry {
    displays: Vec<(usize, Weak<DisplayStats>)>,
    /// Whether the reporting thread runs.
    reporting: bool,
}

/// The process's registry.
fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(Registry {
            displays: Vec::new(),
            reporting: false,
        })
    })
}

/// A core display's address, its key in the registry.
fn key<P: Platform>(display: &Weak<CoreDisplay<P>>) -> usize {
    display.as_ptr().cast::<()>().addr()
}

/// Registers `stats` as `display`'s, once the display exists, for [`of_display`] and the
/// reports; starts the reporting thread at the first. Entries whose stats have ended, or that
/// another display at the same address left, are dropped here.
pub(crate) fn register<P: Platform>(stats: &Arc<DisplayStats>, display: &Weak<CoreDisplay<P>>) {
    let mut registry = registry().lock();
    registry.add(key(display), stats);
    if !registry.reporting {
        registry.reporting = start(Arc::clone(&stats.clock));
    }
}

impl Registry {
    /// `stats` at `key`, in place of any entry there and of the ended ones.
    fn add(&mut self, key: usize, stats: &Arc<DisplayStats>) {
        self.displays
            .retain(|(k, s)| *k != key && s.strong_count() > 0);
        self.displays.push((key, Arc::downgrade(stats)));
    }

    /// The stats at `key`, while they live.
    fn find(&self, key: usize) -> Option<Arc<DisplayStats>> {
        self.displays
            .iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, s)| s.upgrade())
    }

    /// The live stats, by port; drops the ended ones.
    fn live(&mut self) -> Vec<Arc<DisplayStats>> {
        self.displays.retain(|(_, s)| s.strong_count() > 0);
        let mut live: Vec<Arc<DisplayStats>> = self
            .displays
            .iter()
            .filter_map(|(_, s)| s.upgrade())
            .collect();
        live.sort_by_key(|s| s.port);
        live
    }
}

/// The stats of `display`, for its window's sink: `None` with the statistics off.
pub(crate) fn of_display<P: Platform>(display: &Weak<CoreDisplay<P>>) -> Option<Arc<DisplayStats>> {
    registry().lock().find(key(display))
}

/// The lines of `displays`, each ending in a newline. The stats are released after the
/// registry's lock, which the caller no longer holds.
fn lines(displays: &[Arc<DisplayStats>]) -> String {
    let mut lines = String::new();
    for stats in displays {
        lines.push_str(&stats.report());
        lines.push('\n');
    }
    lines
}

/// Starts the reporting thread, which waits on `clock`: whether it started. One that did not is
/// tried again at the next display.
fn start(clock: Arc<dyn TimerClock>) -> bool {
    let started = std::thread::Builder::new()
        .name("reims-vgpu.stats".into())
        .spawn(move || report_forever(&*clock));
    if let Err(e) = &started {
        log::warn!("frame statistics: no reporting thread: {e}");
    }
    started.is_ok()
}

/// Every [`PERIOD`] on `clock`, the lines of every live display, written to stderr in one write.
/// A late wake skips to the next point of the schedule.
fn report_forever(clock: &dyn TimerClock) -> ! {
    let lock = Mutex::new(false);
    let cv = Condvar::new();
    let step = u64::try_from(PERIOD.as_nanos()).unwrap_or(u64::MAX);
    let mut next = mono_now(clock).saturating_add(step);
    loop {
        {
            let mut guard = lock.lock();
            while mono_now(clock) < next {
                clock.wait_until(&cv, &mut guard, Deadline::Mono(MonoTime::from_nanos(next)));
            }
        }
        let live = registry().lock().live();
        let lines = lines(&live);
        drop(live);
        if !lines.is_empty() {
            let _ = std::io::stderr().lock().write_all(lines.as_bytes());
        }
        let now = mono_now(clock);
        next = next.saturating_add(step);
        if next <= now {
            next = now.saturating_add(step);
        }
    }
}

#[cfg(test)]
mod tests;
