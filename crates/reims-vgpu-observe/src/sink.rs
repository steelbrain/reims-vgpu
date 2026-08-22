//! Temporary bring-up log for draw/scanout research. Append-only `/tmp/reims-vgpu-draw.log`.
//!
//! Verbose lines: `REIMS_VGPU_DRAW_LOG=1` only — full-frame logging otherwise stalls the
//! guest compositor. Failures always append (lightweight, fail-visible).
//!
//! ## Offline offline-analysis prefixes (`/tmp/reims-vgpu-fail.log`, always-on)
//!
//! | Prefix | Meaning |
//! | --- | --- |
//! | `OFF present_txn` | Display present packet (`op` 6/7/8, channel, surface id, task) |
//! | `OFF present_black` | max_rgb==0 after capture (console will stay black) |
//! | `OFF present_paint` | HostAction paint / Unchanged |
//! | `OFF host_cache_store` | Discrete-GPU host surface cache write |
//! | `OFF host_cache_evict` | Cache drop (unmap/delete) |
//! | `OFF m2v_store` | metal2vulkan Store to IOSurface texture/type-4 mid (incl. is_front) |
//! | `OFF m2v_store_gva` | metal2vulkan Store to type-2/3 GVA |
//! | `OFF m2v_load_seed` | Load seed path (host_cache vs missing) |
//! | `OFF load_seed_black` | Deduplicated zero-RGB Load seed preserved by protocol provenance |
//! | `OFF linear_sample` | Display-sized type-2/3 sample provenance + content census |
//! | `OFF sampled_branch_census` | Cumulative per-branch sampled-resolution counts:bytes, every 256 |
//! | `OFF sample_alpha_mask` | Deduplicated zero-RGB/nonzero-alpha sample census; alpha is preserved |
//! | `linear_sample_miss` | Display-sized type-2/3 sample failed, with descriptor identity |
//! | `OFF linear_coverage_gap` | Typed stage-in/shader-evaluated coverage check rejected full-display ownership |
//! | `import_content` | Resident-to-guest Store census; display rows include exact changed/R↔B-swapped pixel counts |
//! | `linux_m2v_resources` | Per-draw resource census; `fixed_gap=[...]` names decoded fixed state absent from the Vulkan request |
//! | `linux_m2v_timing` | always-on stage µs: load/m2v/setup/engine/composite + total |
//! | `OFF display_clear` | Clear-only stream Store into a display-sized mid |
//! | `OFF rt_resolve` | Color RT lookup (type-4/5/11 → mapping_id) display-sized |
//! | `OFF front_wb` | note_front_buffer_writeback latch / post-boundary skip |
//! | `OFF blit` | Product blit path enter/result (buffer↔texture) |
//!
//! **rgb_nz / max_rgb** on OFF lines count **pixels with max(B,G,R) > 0** (BGRA).
//! Byte-wise `nonzero_stats` still counts alpha=255 as nonzero — do not use that
//! alone to claim the screen is not black.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(test, feature = "test-fixtures"))]
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::Mutex,
};

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: AtomicBool = AtomicBool::new(false);
#[cfg(any(test, feature = "test-fixtures"))]
static FAIL_FILE: Mutex<Option<File>> = Mutex::new(None);
#[cfg(any(test, feature = "test-fixtures"))]
static DRAW_FILE: Mutex<Option<File>> = Mutex::new(None);

/// Which always-on sink a line targets.
#[derive(Clone, Copy)]
enum Sink {
    Fail,
    Draw,
}

pub fn enabled() -> bool {
    if !INIT.swap(true, Ordering::Relaxed) {
        // Through the shared parse, and read once. Nothing is emitted for an
        // unrecognized value: this is the emit path itself, so a report from
        // here would recurse into the sink that is being asked whether it is
        // enabled. Such a value reads as off, which is what every
        // non-affirmative value already did.
        let on = matches!(
            std::env::var("REIMS_VGPU_DRAW_LOG").as_deref(),
            Ok("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
        );
        ENABLED.store(on, Ordering::Relaxed);
    }
    ENABLED.load(Ordering::Relaxed)
}

/// Whether verbose draw logging (`REIMS_VGPU_DRAW_LOG=1`) is active. Lets always-on
/// paths skip building expensive *diagnostic-only* detail (e.g. per-peer
/// full-frame rescans for a log field) on a normal boot without losing the
/// always-on line itself.
pub fn draw_log_enabled() -> bool {
    enabled()
}

/// Milliseconds since the first log line of this process. Appended as a
/// trailing `t=<ms>` field so cross-boot phase timing (first present, desktop
/// settle, tranche bursts) is measurable from the logs alone. Trailing — not a
/// prefix — so `awk '{print $1}'` line-class censuses keep working.
///
/// `pub(crate)` so always-on rate proxies (e.g. display-signal cadence) can
/// window their counters on the same process-monotonic clock that stamps every
/// line — no second time base to reconcile against `t=`.
pub fn elapsed_ms() -> u128 {
    T0.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis()
}

/// The same process-monotonic clock in microseconds.
///
/// Millisecond resolution cannot express every cadence we owe the guest: a
/// 120 Hz frame is 8333 µs, and rounding it to 8 ms delivers 125 Hz. Paths that
/// pace something the guest measures need this one; `t=` stamps stay in
/// milliseconds so line-class censuses keep working.
pub fn elapsed_us() -> u64 {
    T0.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_micros() as u64
}

static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// Sink paths. Test runs write per-process files instead of the product
/// `/tmp/reims-vgpu-fail.log`: `cargo test` runs on the same machine as live product
/// boots, and both appending to one shared file interleaves test fixture
/// lines (synthetic device resets, malformed-packet fail_events,
/// deferred_flush_lost probes) into live A/B evidence — indistinguishable
/// from real device failures when reading the log offline. Unit-test builds
/// isolate via `cfg(test)`; integration-test binaries (no `cfg(test)` on the
/// lib) must call [`redirect_logs_for_tests`] before the first log line.
static FAIL_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
static DRAW_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();

fn test_path(kind: &str) -> String {
    format!("/tmp/reims-vgpu-{kind}-test-{}.log", std::process::id())
}

pub fn fail_log_path() -> &'static str {
    #[cfg(any(test, feature = "test-fixtures"))]
    return FAIL_PATH.get_or_init(|| test_path("fail"));
    #[cfg(not(any(test, feature = "test-fixtures")))]
    FAIL_PATH.get_or_init(|| "/tmp/reims-vgpu-fail.log".to_string())
}

pub fn draw_log_path() -> &'static str {
    #[cfg(any(test, feature = "test-fixtures"))]
    return DRAW_PATH.get_or_init(|| test_path("draw"));
    #[cfg(not(any(test, feature = "test-fixtures")))]
    DRAW_PATH.get_or_init(|| "/tmp/reims-vgpu-draw.log".to_string())
}

/// Test-harness support: point the always-on sinks at per-process files so a
/// test run never contaminates a concurrent live boot's logs. For integration
/// test binaries, where `cfg(test)` does not apply to the lib; call once
/// before anything logs. No effect on a sink that already resolved its path.
pub fn redirect_logs_for_tests() {
    let _ = FAIL_PATH.set(test_path("fail"));
    let _ = DRAW_PATH.set(test_path("draw"));
}

/// Synchronous single-line append (unit-test builds only). Worker + MMIO proxy
/// lines may arrive concurrently; keep each record on one physical line so
/// failure evidence never merges into another event.
#[cfg(any(test, feature = "test-fixtures"))]
fn append_sync(file: &Mutex<Option<File>>, path: &str, msg: &str, t: u128) {
    let mut file = file.lock().unwrap_or_else(|e| e.into_inner());
    if file.is_none() {
        *file = OpenOptions::new().create(true).append(true).open(path).ok();
    }
    let Some(f) = file.as_mut() else {
        return;
    };
    if writeln!(f, "{msg} t={t}").is_err() {
        // A later record gets one fresh open attempt after a failed write.
        *file = None;
    }
}

/// Emit one always-on line to `sink`, timestamped with the process-monotonic
/// clock at the call site (so `t=` reflects when the event happened, not when
/// the background writer drains it).
///
/// Product builds hand the formatted line to a background writer thread
/// ([`writer`]) so the doorbell / worker vCPU never pays a `write(2)` syscall
/// or contends the file lock — full-frame GPU boots emit ~200k lines/boot and
/// the synchronous path serialized them on the guest's critical path. Unit-test
/// builds stay synchronous: many tests write a line then immediately
/// `read_to_string` the sink and assert on it.
fn emit(sink: Sink, msg: &str) {
    let t = elapsed_ms();
    #[cfg(any(test, feature = "test-fixtures"))]
    {
        if matches!(sink, Sink::Fail) {
            if let Some(buf) = CAPTURED.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                buf.push(msg.to_string());
            }
        }
        let (file, path) = match sink {
            Sink::Fail => (&FAIL_FILE, fail_log_path()),
            Sink::Draw => (&DRAW_FILE, draw_log_path()),
        };
        append_sync(file, path, msg, t);
    }
    #[cfg(not(any(test, feature = "test-fixtures")))]
    writer::enqueue(sink, format!("{msg} t={t}"));
}

/// Flood self-detector (regression guard). A per-event line wrongly routed to
/// the always-on sink (the `type5_view_zc`/`InvalidateResources` class that
/// buried the curated fail view under ~130k lines/boot) fires per bind/op — far
/// above any legitimate always-on rate. This watches the **always-on** stream in
/// the background writer thread (zero producer cost) and emits ONE
/// `log_flood_detected` line per window per runaway prefix, so a regression that
/// reintroduces a flood is named on the very boot it lands instead of silently
/// drowning real failures. Legitimate always-on lines are self-clocked windowed
/// summaries (`drain_duty`, `store_routes`) well under the threshold.
#[cfg(any(test, not(feature = "test-fixtures")))]
const FLOOD_WINDOW_MS: u128 = 1000;
#[cfg(any(test, not(feature = "test-fixtures")))]
const FLOOD_THRESHOLD_PER_WINDOW: u64 = 1000;

/// The flood-accounting key for an always-on line: its slug — the first
/// whitespace token, skipping a leading `OFF ` marker. Groups a runaway line by
/// kind (`type5_view_zc`, `map_family`, …) so the warning names the culprit.
#[cfg(any(test, not(feature = "test-fixtures")))]
fn flood_key(line: &str) -> &str {
    let slug = line.strip_prefix("OFF ").unwrap_or(line);
    slug.split(' ').next().unwrap_or(slug)
}

/// Windowed per-prefix counter for the always-on stream. Pure + always compiled
/// so the threshold/keying is unit-tested without a background thread.
#[cfg(any(test, not(feature = "test-fixtures")))]
struct FloodWindow {
    counts: std::collections::HashMap<String, u64>,
    window_start_ms: u128,
}

#[cfg(any(test, not(feature = "test-fixtures")))]
impl FloodWindow {
    fn new(now: u128) -> Self {
        Self {
            counts: std::collections::HashMap::new(),
            window_start_ms: now,
        }
    }

    /// Record one always-on line. When the ~1 s window closes, returns the
    /// prefixes that exceeded the flood threshold (sorted desc by count for a
    /// stable warning order) and opens a fresh window; otherwise returns empty.
    fn note(&mut self, line: &str, now: u128) -> Vec<(String, u64)> {
        *self.counts.entry(flood_key(line).to_string()).or_insert(0) += 1;
        if now.saturating_sub(self.window_start_ms) < FLOOD_WINDOW_MS {
            return Vec::new();
        }
        let mut flooders: Vec<(String, u64)> = self
            .counts
            .drain()
            .filter(|(_, c)| *c >= FLOOD_THRESHOLD_PER_WINDOW)
            .collect();
        flooders.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        self.window_start_ms = now;
        flooders
    }
}

/// Background log writer (product builds). A single thread owns both sink files
/// behind buffered writers; producers only push a formatted line onto an mpsc
/// channel. The thread batch-drains (block on one, then greedily take the rest)
/// and flushes after each batch, so failure visibility trails real time by at
/// most one drain cycle while the hot path stays syscall-free.
#[cfg(not(any(test, feature = "test-fixtures")))]
mod writer {
    use super::{draw_log_path, fail_log_path, Sink};
    use std::io::{BufWriter, Write};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::OnceLock;

    enum Msg {
        Fail(String),
        Draw(String),
    }

    // `Sender<T>: Sync` (std, since 1.72), so producers share one sender with no
    // lock — the hot path is a lock-free channel send.
    static SENDER: OnceLock<Sender<Msg>> = OnceLock::new();

    fn sender() -> &'static Sender<Msg> {
        SENDER.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel::<Msg>();
            // Resolve sink paths on the spawning thread (honors any prior
            // `redirect_logs_for_tests`); the writer owns the file handles.
            let fail_path = fail_log_path().to_string();
            let draw_path = draw_log_path().to_string();
            let _ = std::thread::Builder::new()
                .name("reims-vgpu-drawlog".to_string())
                .spawn(move || writer_loop(rx, fail_path, draw_path));
            tx
        })
    }

    fn open(path: &str) -> Option<BufWriter<std::fs::File>> {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(BufWriter::new)
    }

    fn writer_loop(rx: Receiver<Msg>, fail_path: String, draw_path: String) {
        let mut fail = open(&fail_path);
        let mut draw = open(&draw_path);
        let mut flood = super::FloodWindow::new(super::elapsed_ms());
        // Block for the next line, then greedily drain everything already
        // queued before a single flush — one syscall amortizes a whole burst.
        while let Ok(first) = rx.recv() {
            write_watched(&mut fail, &mut draw, &mut flood, first);
            while let Ok(m) = rx.try_recv() {
                write_watched(&mut fail, &mut draw, &mut flood, m);
            }
            if let Some(w) = fail.as_mut() {
                let _ = w.flush();
            }
            if let Some(w) = draw.as_mut() {
                let _ = w.flush();
            }
        }
    }

    /// Write one line, and for the always-on (Fail) sink feed the flood
    /// self-detector — a runaway prefix gets one named `log_flood_detected`
    /// warning per window, written straight to the fail file (not re-queued, so
    /// it never self-counts). All in the writer thread: no producer-side cost.
    fn write_watched(
        fail: &mut Option<BufWriter<std::fs::File>>,
        draw: &mut Option<BufWriter<std::fs::File>>,
        flood: &mut super::FloodWindow,
        m: Msg,
    ) {
        if let Msg::Fail(s) = &m {
            let flooders = flood.note(s, super::elapsed_ms());
            if let Some(w) = fail.as_mut() {
                for (prefix, count) in flooders {
                    let _ = writeln!(
                        w,
                        "log_flood_detected prefix={prefix} count={count} window_ms={} threshold={} t={}",
                        super::FLOOD_WINDOW_MS,
                        super::FLOOD_THRESHOLD_PER_WINDOW,
                        super::elapsed_ms()
                    );
                }
            }
        }
        write_msg(fail, draw, m);
    }

    fn write_msg(
        fail: &mut Option<BufWriter<std::fs::File>>,
        draw: &mut Option<BufWriter<std::fs::File>>,
        m: Msg,
    ) {
        let (w, line) = match m {
            Msg::Fail(s) => (fail.as_mut(), s),
            Msg::Draw(s) => (draw.as_mut(), s),
        };
        if let Some(w) = w {
            let _ = w.write_all(line.as_bytes());
            let _ = w.write_all(b"\n");
        }
    }

    /// Push one already-timestamped line to the background writer. Lock-free and
    /// never blocks on I/O — a bare channel send.
    pub(super) fn enqueue(sink: Sink, line: String) {
        let msg = match sink {
            Sink::Fail => Msg::Fail(line),
            Sink::Draw => Msg::Draw(line),
        };
        let _ = sender().send(msg);
    }
}

/// Test-only in-memory copy of the always-on stream, armed by [`FailCapture`].
#[cfg(any(test, feature = "test-fixtures"))]
static CAPTURED: std::sync::Mutex<Option<Vec<String>>> = std::sync::Mutex::new(None);

/// Records every always-on ([`fail`] / [`off`]) line emitted while it is alive.
///
/// The always-on stream is the project's primary evidence, and until this
/// existed nothing could assert on it: a probe could name a field after the
/// quantity its check compared while printing a different one, and no test
/// could tell. That is not hypothetical — the compute flush's
/// `map_generation_drift` printed a *content* generation in a field called
/// `gen`, next to the `map_generation` it had actually compared, and the
/// mismatch was read off a live boot as a generation that had gone backwards.
///
/// Relies on the crate's serial test convention (`--test-threads=1`); a second
/// capture armed concurrently would see the other test's lines.
#[cfg(any(test, feature = "test-fixtures"))]
pub struct FailCapture;

#[cfg(any(test, feature = "test-fixtures"))]
impl FailCapture {
    /// Arm the capture, and drop every dedup latch first.
    ///
    /// The latches outlive individual tests — one process runs the whole suite —
    /// so without this a test's emitter can decide it has already said its line
    /// on behalf of a test that ran earlier and picked the same discriminant.
    // A code span, not a link: `forget_all_latches` is `#[cfg(test)]` and
    // rustdoc never documents a `cfg(test)` item, so a link to it cannot
    // resolve on any arm and would read as rot in the intra-doc pass.
    /// See `super::emit::forget_all_latches` for what that costs and why the
    /// clearing belongs here rather than at each fixture.
    pub fn start() -> Self {
        super::emit::forget_all_latches();
        Self::arm()
    }

    /// Arm a further window in the *same* latched run, keeping what the windows
    /// before it claimed.
    ///
    /// For the one shape [`FailCapture::start`] cannot serve: a test whose
    /// subject is the dedup itself. Those walk a sequence of distinct events in
    /// separate windows so each assertion names the line it is about, and then
    /// open a last window, repeat every event, and require silence. Under
    /// `start` that last window would clear the very latches it is checking and
    /// see every line again, so the test would fail while the latch worked.
    ///
    /// Use this only when the latch state carried in is what is being asserted.
    /// It reintroduces, deliberately and locally, exactly the cross-test
    /// coupling `start` exists to remove — so a test that reaches for it is
    /// claiming the earlier windows are its own, and it must open the sequence
    /// with a `start`.
    pub fn resume() -> Self {
        Self::arm()
    }

    fn arm() -> Self {
        *CAPTURED.lock().unwrap_or_else(|p| p.into_inner()) = Some(Vec::new());
        Self
    }

    /// Every always-on line emitted since `start`, in order.
    pub fn lines(&self) -> Vec<String> {
        CAPTURED
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_default()
    }

    /// The one line whose first whitespace token is `slug`. Panics unless
    /// exactly one matched — "no line" and "several lines" are both reasons a
    /// downstream assertion would otherwise pass or fail for the wrong reason.
    pub fn one(&self, slug: &str) -> String {
        let hits: Vec<String> = self
            .lines()
            .into_iter()
            .filter(|l| l.split_whitespace().next() == Some(slug))
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "expected exactly one `{slug}` line, got {hits:?} (all: {:?})",
            self.lines()
        );
        hits.into_iter().next().unwrap_or_default()
    }
}

#[cfg(any(test, feature = "test-fixtures"))]
impl Drop for FailCapture {
    fn drop(&mut self) {
        *CAPTURED.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

pub fn line(msg: impl AsRef<str>) {
    if !enabled() {
        return;
    }
    emit(Sink::Draw, msg.as_ref());
}

/// Always-on fail-visible line (writeback / backend / missing resource / offline OFF).
pub fn fail(msg: impl AsRef<str>) {
    emit(Sink::Fail, msg.as_ref());
    if enabled() {
        emit(Sink::Draw, msg.as_ref());
    }
}

/// Always-on offline analysis line (prefix `OFF `). Same sink as [`fail`].
#[inline]
pub fn off(msg: impl AsRef<str>) {
    fail(format!("OFF {}", msg.as_ref()));
}

/// Count nonzero **bytes** and max sample in a tightly packed image buffer.
///
/// Note: solid black with alpha=255 has nz == byte_len (every A channel). Prefer
/// [`bgra_rgb_stats`] when diagnosing visible content vs QMP black.
pub fn nonzero_stats(buf: &[u8]) -> (usize, u8) {
    let mut nz = 0usize;
    let mut max = 0u8;
    for &b in buf {
        if b != 0 {
            nz += 1;
        }
        if b > max {
            max = b;
        }
    }
    (nz, max)
}

/// Visible-content stats for tight BGRA8: rgb_nz = pixels with max(B,G,R)>0,
/// max_rgb = max of B/G/R, px0 = first pixel BGRA.
pub fn bgra_rgb_stats(bgra: &[u8]) -> (usize, u8, [u8; 4]) {
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if bgra.len() >= 4 {
        [bgra[0], bgra[1], bgra[2], bgra[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in bgra.chunks_exact(4) {
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (rgb_nz, max_rgb, px0)
}

/// Fused present-capture stats for tight BGRA8: one pass yielding what
/// [`nonzero_stats`] and [`bgra_rgb_stats`] compute separately —
/// `(byte_nz, byte_max, rgb_nz, max_rgb, px0)`. The present drain path scans
/// the full 8 MiB frame on every present while holding the device lock; folding
/// the two per-pixel passes into one halves that measure-only lock-hold (a
/// direct win on present cadence / boot convergence). Byte-exact with the two
/// separate functions: `byte_nz`/`byte_max` count all four channels,
/// `rgb_nz`/`max_rgb` the low three.
///
/// On x86_64 this dispatches to an SSE2 vectorized kernel (16 bytes/iteration,
/// ~11× the scalar loop measured at opt-level 2); SSE2 is baseline for the
/// arch so no runtime feature detection is needed. The scalar body remains the
/// `bgra_present_stats_byte_exact_with_sse2` unit asserts against.
#[inline]
pub fn bgra_present_stats(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86_64 baseline ABI, always available.
        unsafe { bgra_present_stats_sse2(bgra) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bgra_present_stats_scalar(bgra)
    }
}

/// Scalar reference for [`bgra_present_stats`] — the byte-exact definition the
/// SSE2 kernel matches. Kept out of the x86_64 hot path but used verbatim on
/// other arches and as the unit-test oracle.
pub fn bgra_present_stats_scalar(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    let mut byte_nz = 0usize;
    let mut byte_max = 0u8;
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if bgra.len() >= 4 {
        [bgra[0], bgra[1], bgra[2], bgra[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in bgra.chunks_exact(4) {
        // Per-byte nonzero/max across all four channels (== nonzero_stats over
        // a length that is a multiple of 4; the present frame always is).
        for &b in px {
            if b != 0 {
                byte_nz += 1;
            }
            if b > byte_max {
                byte_max = b;
            }
        }
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (byte_nz, byte_max, rgb_nz, max_rgb, px0)
}

/// SSE2 kernel for [`bgra_present_stats`], byte-exact with
/// [`bgra_present_stats_scalar`]. Processes 16 bytes (4 BGRA pixels) per
/// iteration: `pmaxub` for the running byte/rgb maxima, `pcmpeqb` + `psadbw`
/// to count zero bytes (`byte_nz = len − zeros`), and a per-u32-lane
/// `pcmpeqd` on alpha-masked pixels to count fully-black-rgb pixels
/// (`rgb_nz = pixels − rgb_zeros`). The 8-bit zero accumulator is flushed via
/// `psadbw` every 255 iterations so it cannot overflow.
///
/// # Safety
/// Requires SSE2, which is guaranteed on every x86_64 target.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn bgra_present_stats_sse2(bgra: &[u8]) -> (usize, u8, usize, u8, [u8; 4]) {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is guaranteed on x86_64; all loads/stores below are bounded
    // by `n = len & !15` (full 16-byte blocks) with a scalar tail for the rest.
    unsafe {
        let px0 = if bgra.len() >= 4 {
            [bgra[0], bgra[1], bgra[2], bgra[3]]
        } else {
            [0, 0, 0, 0]
        };
        let n = bgra.len() & !15;
        let mut byte_max = 0u8;
        let mut max_rgb = 0u8;
        let zero = _mm_setzero_si128();
        // Keep the low three bytes (B,G,R) of each little-endian pixel, drop alpha.
        let rgb_mask = _mm_set1_epi32(0x00FF_FFFFu32 as i32);
        let mut vmax = zero; // running max over all four channels
        let mut vmax_rgb = zero; // running max over B/G/R only
        let mut vzero_bytes = zero; // 64-bit lanes: total zero-byte count
        let mut vzero_rgb = zero; // 32-bit lanes: fully-black-rgb pixel count
        let mut ptr = bgra.as_ptr();
        let mut rem = n;
        while rem > 0 {
            // Bound the 8-bit zero-byte accumulator to ≤255 before flushing.
            let block = rem.min(255 * 16);
            let mut inner_zero = zero;
            let mut b = 0usize;
            while b < block {
                let v = _mm_loadu_si128(ptr as *const __m128i);
                vmax = _mm_max_epu8(vmax, v);
                let zmask = _mm_cmpeq_epi8(v, zero); // 0xFF per zero byte
                inner_zero = _mm_sub_epi8(inner_zero, zmask); // +1 per zero byte
                let vr = _mm_and_si128(v, rgb_mask);
                vmax_rgb = _mm_max_epu8(vmax_rgb, vr);
                let rgb_eq = _mm_cmpeq_epi32(vr, zero); // 0xFFFFFFFF per black-rgb px
                vzero_rgb = _mm_sub_epi32(vzero_rgb, rgb_eq);
                ptr = ptr.add(16);
                b += 16;
            }
            vzero_bytes = _mm_add_epi64(vzero_bytes, _mm_sad_epu8(inner_zero, zero));
            rem -= block;
        }
        let mut lanes = [0u8; 16];
        _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, vmax);
        for &x in &lanes {
            byte_max = byte_max.max(x);
        }
        _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, vmax_rgb);
        for &x in &lanes {
            max_rgb = max_rgb.max(x);
        }
        let mut z64 = [0u64; 2];
        _mm_storeu_si128(z64.as_mut_ptr() as *mut __m128i, vzero_bytes);
        let mut byte_nz = n - (z64[0] + z64[1]) as usize;
        let mut z32 = [0u32; 4];
        _mm_storeu_si128(z32.as_mut_ptr() as *mut __m128i, vzero_rgb);
        let rgb_zeros = (z32[0] + z32[1] + z32[2] + z32[3]) as usize;
        let mut rgb_nz = n / 4 - rgb_zeros;
        // Scalar tail (frame length is a multiple of 4 but not always of 16).
        for px in bgra[n..].chunks_exact(4) {
            for &b in px {
                if b != 0 {
                    byte_nz += 1;
                }
                byte_max = byte_max.max(b);
            }
            let m = px[0].max(px[1]).max(px[2]);
            if m != 0 {
                rgb_nz += 1;
            }
            max_rgb = max_rgb.max(m);
        }
        (byte_nz, byte_max, rgb_nz, max_rgb, px0)
    }
}

/// Same for tight RGBA8 (m2v encode output).
pub fn rgba_rgb_stats(rgba: &[u8]) -> (usize, u8, [u8; 4]) {
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let px0 = if rgba.len() >= 4 {
        [rgba[0], rgba[1], rgba[2], rgba[3]]
    } else {
        [0, 0, 0, 0]
    };
    for px in rgba.chunks_exact(4) {
        let m = px[0].max(px[1]).max(px[2]);
        if m != 0 {
            rgb_nz += 1;
        }
        if m > max_rgb {
            max_rgb = m;
        }
    }
    (rgb_nz, max_rgb, px0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn flood_key_is_the_slug_skipping_the_off_marker() {
        assert_eq!(
            flood_key("OFF type5_view_zc ref=355 sid=62 view=1920x1080 t=1"),
            "type5_view_zc"
        );
        assert_eq!(
            flood_key("map_family op=InvalidateResources ch=3 t=1"),
            "map_family"
        );
        // A bare slug with no fields still keys on itself.
        assert_eq!(flood_key("present_converge"), "present_converge");
    }

    #[test]
    fn flood_window_names_only_over_threshold_prefixes_once_per_window() {
        let mut fw = FloodWindow::new(0);
        // A runaway prefix (over threshold) alongside a quiet one, all inside the
        // window → nothing reported until the window closes.
        for _ in 0..FLOOD_THRESHOLD_PER_WINDOW {
            assert!(fw.note("OFF hot_line a=1 t=1", 10).is_empty());
        }
        assert!(fw.note("OFF quiet_line b=2 t=1", 10).is_empty());

        // A note past the window boundary closes it: only the over-threshold
        // prefix is named (the +1 in this closing window still counts).
        let flooders = fw.note("OFF hot_line a=1 t=1", FLOOD_WINDOW_MS);
        assert_eq!(flooders.len(), 1, "only the runaway prefix is reported");
        assert_eq!(flooders[0].0, "hot_line");
        assert!(flooders[0].1 >= FLOOD_THRESHOLD_PER_WINDOW);

        // Window reset: the quiet prefix never trips it across a fresh window.
        assert!(fw
            .note("OFF quiet_line b=2 t=1", FLOOD_WINDOW_MS + 10)
            .is_empty());
        let none = fw.note("OFF quiet_line b=2 t=1", 2 * FLOOD_WINDOW_MS + 20);
        assert!(none.is_empty(), "a quiet prefix never floods");
    }

    #[test]
    fn bgra_present_stats_is_byte_exact_with_separate_scans() {
        // Mixed content: black, alpha-only, colored, saturated — the classes
        // the present proxies distinguish.
        let frame: Vec<u8> = vec![
            0, 0, 0, 0, // fully black
            0, 0, 0, 255, // alpha-only (rgb-empty, byte-nonzero)
            10, 20, 30, 40, // colored
            255, 255, 255, 255, // saturated
            0, 200, 0, 128, // green only
        ];
        let (nz, maxb) = nonzero_stats(&frame);
        let (rgb_nz, max_rgb, px0) = bgra_rgb_stats(&frame);
        let fused = bgra_present_stats(&frame);
        assert_eq!(
            fused,
            (nz, maxb, rgb_nz, max_rgb, px0),
            "fused present stats must equal the two separate scans byte-for-byte"
        );
        // Sanity: alpha-only pixel counts as byte-nonzero but not rgb-nonzero.
        assert_eq!(rgb_nz, 3, "black + alpha-only are rgb-empty");
        assert_eq!(nz, 2 + 3 + 4 + 2, "nonzero bytes across all four channels");
        // The dispatched entry must equal the scalar reference on this arch too.
        assert_eq!(fused, bgra_present_stats_scalar(&frame));
    }

    #[test]
    fn bgra_present_stats_byte_exact_with_sse2() {
        // Exercise the SSE2 kernel against the scalar reference over sizes that
        // hit the full-block path, the 255-iteration accumulator flush, the
        // sub-16-byte scalar tail, and the short/empty guards — with content
        // covering every class (black, alpha-only, single-channel, saturated).
        for &pixels in &[0usize, 1, 3, 4, 16, 17, 255 * 4, 255 * 4 + 5, 1920 * 1080] {
            let mut frame = vec![0u8; pixels * 4];
            for (i, b) in frame.iter_mut().enumerate() {
                // Deterministic pseudo-content with black runs and saturation.
                let v = (i.wrapping_mul(2_654_435_761) >> 11) & 0xff;
                *b = if i % 7 == 0 { 0 } else { v as u8 };
            }
            let want = bgra_present_stats_scalar(&frame);
            let got = bgra_present_stats(&frame);
            assert_eq!(got, want, "SSE2 kernel diverged at pixels={pixels}");
        }
        // All-black and all-saturated corner cases.
        assert_eq!(
            bgra_present_stats(&[0u8; 64]),
            bgra_present_stats_scalar(&[0u8; 64])
        );
        assert_eq!(
            bgra_present_stats(&[255u8; 64]),
            bgra_present_stats_scalar(&[255u8; 64])
        );
    }

    #[test]
    fn append_reuses_the_open_file_handle() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = format!("/tmp/reims-vgpu-draw-log-{nonce}.log");
        let moved = format!("{path}.moved");
        let file = Mutex::new(None);

        append_sync(&file, &path, "first", 0);
        fs::rename(&path, &moved).expect("rename open log");
        append_sync(&file, &path, "second", 0);

        assert!(!std::path::Path::new(&path).exists());
        let body = fs::read_to_string(&moved).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("first t="));
        assert!(lines[1].starts_with("second t="));
        fs::remove_file(moved).unwrap();
    }

    #[test]
    fn fail_writes_apv_fail_log() {
        let path = fail_log_path();
        let marker = format!("draw_log_selftest_{}", std::process::id());
        fail(&marker);
        let body = fs::read_to_string(path).expect("fail log readable");
        assert!(
            body.lines().any(|l| l.contains(&marker)),
            "fail() must append to the fail log"
        );
        assert_ne!(
            path, "/tmp/reims-vgpu-fail.log",
            "test builds must not share the product fail log — a cargo test \
             run concurrent with a live boot corrupts A/B evidence"
        );
    }

    #[test]
    fn bgra_rgb_stats_black_alpha_full_is_zero_rgb_nz() {
        // Solid black opaque: every A byte is 255 → byte nz is full, rgb_nz must be 0.
        let mut bgra = vec![0u8; 16];
        for px in bgra.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let (byte_nz, _) = nonzero_stats(&bgra);
        assert_eq!(byte_nz, 4); // four alpha bytes
        let (rgb_nz, max_rgb, px0) = bgra_rgb_stats(&bgra);
        assert_eq!(rgb_nz, 0);
        assert_eq!(max_rgb, 0);
        assert_eq!(px0, [0, 0, 0, 255]);
    }

    #[test]
    fn bgra_rgb_stats_gray_counts_pixels() {
        let mut bgra = vec![0u8; 8];
        bgra[0] = 100;
        bgra[1] = 100;
        bgra[2] = 100;
        bgra[3] = 255;
        bgra[7] = 255;
        let (rgb_nz, max_rgb, _) = bgra_rgb_stats(&bgra);
        assert_eq!(rgb_nz, 1);
        assert_eq!(max_rgb, 100);
    }
}
