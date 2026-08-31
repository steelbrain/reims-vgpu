//! Temporary bring-up log for metal/scanout (research). Append-only `/tmp/reims-vgpu-draw.log`.
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
//! | `OFF m2v_store` | metal2vulkan Store to type-11/type-4 mid (incl. is_front) |
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
#[cfg(any(test, feature = "testing"))]
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::Mutex,
};

static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: AtomicBool = AtomicBool::new(false);
#[cfg(any(test, feature = "testing"))]
static FAIL_FILE: Mutex<Option<File>> = Mutex::new(None);
#[cfg(any(test, feature = "testing"))]
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
        let on = reims_vgpu_env::switch(reims_vgpu_env::DRAW_LOG) == reims_vgpu_env::Switch::On;
        ENABLED.store(on, Ordering::Relaxed);
    }
    ENABLED.load(Ordering::Relaxed)
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
    #[cfg(any(test, feature = "testing"))]
    return FAIL_PATH.get_or_init(|| test_path("fail"));
    #[cfg(not(any(test, feature = "testing")))]
    FAIL_PATH.get_or_init(|| "/tmp/reims-vgpu-fail.log".to_string())
}

pub fn draw_log_path() -> &'static str {
    #[cfg(any(test, feature = "testing"))]
    return DRAW_PATH.get_or_init(|| test_path("draw"));
    #[cfg(not(any(test, feature = "testing")))]
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
#[cfg(any(test, feature = "testing"))]
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
    #[cfg(any(test, feature = "testing"))]
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
    #[cfg(not(any(test, feature = "testing")))]
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
///
/// Everything in this block down to `writer_beat_due` belongs to the background
/// writer, so it is compiled when that writer is — and for this crate's own
/// tests of it. A *consumer's* test build turns on `testing`, which replaces
/// the writer with a synchronous append, and in that one configuration this is
/// genuinely dead. Saying so in a `cfg` is cheaper and more honest than an
/// `allow(dead_code)` that would also hide a real one.
#[cfg(any(test, not(feature = "testing")))]
const FLOOD_WINDOW_MS: u128 = 1000;
#[cfg(any(test, not(feature = "testing")))]
const FLOOD_THRESHOLD_PER_WINDOW: u64 = 1000;

/// The flood-accounting key for an always-on line: its slug — the first
/// whitespace token, skipping a leading `OFF ` marker. Groups a runaway line by
/// kind (`type5_view_zc`, `map_family`, …) so the warning names the culprit.
#[cfg(any(test, not(feature = "testing")))]
fn flood_key(line: &str) -> &str {
    let slug = line.strip_prefix("OFF ").unwrap_or(line);
    slug.split(' ').next().unwrap_or(slug)
}

/// Windowed per-prefix counter for the always-on stream. Pure, so the
/// threshold and the keying are unit-tested without a background thread.
#[cfg(any(test, not(feature = "testing")))]
struct FloodWindow {
    counts: std::collections::HashMap<String, u64>,
    window_start_ms: u128,
}

#[cfg(any(test, not(feature = "testing")))]
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

/// How often the writer thread reports on itself. One second, the same census
/// interval every other levels line in this device shares, so a `log_writer`
/// row reads against the `store_routes` and `drain_duty` rows beside it.
#[cfg(any(test, not(feature = "testing")))]
const WRITER_BEAT_MS: u64 = 1_000;

/// Whether the writer should emit its heartbeat now.
///
/// A pure function of `(last, now)` for the same reason
/// `released_pages::claim_census_interval` is: a rate gate written inline
/// against the clock can only be checked by a boot. Unlike that one this has a
/// single caller on a single thread, so it needs no atomic — but it does need
/// to be checkable, because the whole value of the beat is its cadence.
#[cfg(any(test, not(feature = "testing")))]
fn writer_beat_due(last_ms: u128, now_ms: u128) -> bool {
    now_ms.saturating_sub(last_ms) >= u128::from(WRITER_BEAT_MS)
}

/// Background log writer (product builds). A single thread owns both sink files
/// behind buffered writers; producers only push a formatted line onto an mpsc
/// channel. The thread batch-drains (block on one, then greedily take the rest)
/// and flushes after each batch, so failure visibility trails real time by at
/// most one drain cycle while the hot path stays syscall-free.
///
/// # Why it reports on itself
///
/// Every line is stamped at **enqueue**, so a log that ends at `t=29 682` while
/// the process lived to 38 s has two readings and the file cannot tell them
/// apart: the device stopped emitting, or this thread fell behind and its
/// backlog died with the process. That is not a hypothetical distinction —
/// those eight seconds are the eight seconds before a guest kernel panic on the
/// host-pointer-import rail, so which reading is right decides whether any
/// instrument can see that defect at all.
///
/// So the writer emits a `log_writer` row of its own, stamped with its own
/// clock at write time and carrying the exact queue depth. It is written
/// directly rather than enqueued — a heartbeat that queued behind the backlog it
/// is measuring would report the backlog's clock, not its own. Read it as:
///
/// - beat present, `queued` small, then the file ends → the device stopped
///   emitting, and the last producer line is where it stopped;
/// - beat present, `queued` climbing → this thread is behind, and everything
///   after the last written line was lost with the process.
///
/// The wait is a timeout rather than a block, so an idle device still beats. A
/// silent writer and a silent device must not look alike.
#[cfg(not(any(test, feature = "testing")))]
mod writer {
    use super::{draw_log_path, fail_log_path, Sink};
    use std::io::{BufWriter, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
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

    /// Messages enqueued and not yet written. `mpsc` has no depth of its own, so
    /// the two ends keep it between them: one relaxed add beside a channel send
    /// the hot path was already paying for, and one relaxed subtract on the
    /// writer. Exact, and it is the discriminator the heartbeat exists to carry.
    static QUEUED: AtomicU64 = AtomicU64::new(0);

    fn writer_loop(rx: Receiver<Msg>, fail_path: String, draw_path: String) {
        let mut fail = open(&fail_path);
        let mut draw = open(&draw_path);
        let mut flood = super::FloodWindow::new(super::elapsed_ms());
        let mut last_beat_ms = super::elapsed_ms();
        let mut wrote_since_beat = 0u64;
        // Wait for the next line with a timeout, then greedily drain everything
        // already queued before a single flush — one syscall amortizes a whole
        // burst. The timeout is what lets an idle device still beat.
        loop {
            match rx.recv_timeout(std::time::Duration::from_millis(super::WRITER_BEAT_MS)) {
                Ok(first) => {
                    write_watched(&mut fail, &mut draw, &mut flood, first);
                    wrote_since_beat += 1;
                    while let Ok(m) = rx.try_recv() {
                        write_watched(&mut fail, &mut draw, &mut flood, m);
                        wrote_since_beat += 1;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                // Every producer is gone, which for this process means shutdown.
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
            let now = super::elapsed_ms();
            if super::writer_beat_due(last_beat_ms, now) {
                if let Some(w) = fail.as_mut() {
                    let _ = writeln!(
                        w,
                        "OFF log_writer wrote={wrote_since_beat} queued={} beat_ms={} t={now}",
                        QUEUED.load(Ordering::Relaxed),
                        now.saturating_sub(last_beat_ms),
                    );
                }
                last_beat_ms = now;
                wrote_since_beat = 0;
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
        QUEUED.fetch_sub(1, Ordering::Relaxed);
    }

    /// Push one already-timestamped line to the background writer. Lock-free and
    /// never blocks on I/O — a bare channel send.
    pub(super) fn enqueue(sink: Sink, line: String) {
        let msg = match sink {
            Sink::Fail => Msg::Fail(line),
            Sink::Draw => Msg::Draw(line),
        };
        QUEUED.fetch_add(1, Ordering::Relaxed);
        if sender().send(msg).is_err() {
            // No writer to reach it, so it is not queued either.
            QUEUED.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Test-only in-memory copy of the always-on stream, armed by [`FailCapture`].
#[cfg(any(test, feature = "testing"))]
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
#[cfg(any(test, feature = "testing"))]
pub struct FailCapture;

#[cfg(any(test, feature = "testing"))]
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

#[cfg(any(test, feature = "testing"))]
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

/// A verbose line whose *text* is expensive to build.
///
/// [`line`] already drops its argument when verbose logging is off, but the
/// caller has paid for the `format!` by the time it is called. A hot path that
/// wants to skip that cost used to ask whether the sink was open and branch, which
/// put a query about observability state inside a product path — the one shape
/// `AGENTS.md` and this module both rule out, because a path that can read the
/// log's state is a path that could come to depend on it.
///
/// Taking the text as a closure moves the question back inside the sink. The
/// caller states what it would say; whether saying it is worth building is not
/// its decision to make.
pub fn verbose(build: impl FnOnce() -> String) {
    if !enabled() {
        return;
    }
    emit(Sink::Draw, &build());
}

/// Run a diagnostic-only block, and only when verbose logging is on.
///
/// [`verbose`] covers a line whose text is expensive. This covers the wider
/// case: a block that scans a frame for its byte statistics, walks a guest page
/// table to describe a mapping, or emits several lines at once. Those cost more
/// than a `format!` and none of them may run on a normal boot.
///
/// The closure returns nothing, which is the whole point. A caller that asked
/// [`enabled`] and branched held a `bool` it could store, thread into a
/// decision, or read a second time — and observability that a product path can
/// read is observability a product path can come to depend on. Handing the work
/// *to* the sink instead lets nothing escape the block: whatever it computes is
/// for the log and dies there.
pub fn when_verbose(diagnose: impl FnOnce()) {
    if !enabled() {
        return;
    }
    diagnose();
}

/// Always-on fail-visible line (writeback / Metal / missing resource / offline OFF).
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
/// reference on other targets (arm backend-metal build) and the oracle the
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

/// Same for tight RGBA8 (m2v encode output), plus the mean.
///
/// # Why a count of non-zero pixels was not enough
///
/// `rgb_nz` counts pixels with `max(R,G,B) > 0`, and that separates a surface
/// with content from one that is black. It cannot separate a surface with
/// content from one that is uniformly **white**, because both saturate it: a
/// 1920x1080 store of live desktop and a 1920x1080 store of flat `0xff` both
/// report about 2 073 600. The defect this rail is read for is a plane that
/// comes up uniform white, so the one distinction the record could not make was
/// the one being looked for -- ten labelled boots were scored on `rgb_nz` and
/// the white and painted populations overlapped completely.
///
/// The mean makes it. Flat white is 255, and this rail's painted desktop
/// measures near 100 in the same units, so the two are far apart and the
/// comparison needs no threshold chosen in advance. It is the mean of the same
/// per-pixel `max(R,G,B)` the other two fields are computed from, in the same
/// pass, so the three cannot describe different pixels.
///
/// Accumulated in `u64`: a 4K surface has more pixels than `u32` can sum at 255
/// each, and a saturating `u32` would report the brightest surfaces as darker
/// than they are -- silently, and only for the largest ones.
pub fn rgba_rgb_stats(rgba: &[u8]) -> RgbaRgbStats {
    let mut rgb_nz = 0usize;
    let mut max_rgb = 0u8;
    let mut sum = 0u64;
    let mut count = 0u64;
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
        sum += u64::from(m);
        count += 1;
    }
    RgbaRgbStats {
        rgb_nz,
        max_rgb,
        px0,
        // Zero pixels means no evidence about brightness, and 0 is the only
        // answer that does not invent one. `rgb_nz` is 0 beside it, so a reader
        // can tell an empty buffer from a black one.
        //
        // The quotient cannot exceed 255 -- every term summed is a `u8` and
        // there are `count` of them -- so the narrowing is exact rather than
        // saturating, and stating it as a `u8::try_from` keeps that an
        // assertion instead of a silent truncation if the accumulation ever
        // changes shape.
        mean_rgb: sum
            .checked_div(count)
            .and_then(|m| u8::try_from(m).ok())
            .unwrap_or(0),
    }
}

/// What one tight RGBA8 buffer looks like, as the four numbers that travel
/// together.
///
/// A tuple grew to four members and the third and fourth were both plausible
/// as either position at a call site. Named fields make the emit sites read as
/// what they print.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbaRgbStats {
    /// Pixels with `max(R,G,B) > 0`.
    pub rgb_nz: usize,
    /// Largest `max(R,G,B)` over the buffer.
    pub max_rgb: u8,
    /// First pixel, RGBA.
    pub px0: [u8; 4],
    /// Mean `max(R,G,B)`; 255 is flat white and this rail's desktop is near 100.
    pub mean_rgb: u8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Force the sink open or closed for one test and put it back.
    ///
    /// `ENABLED` is read once per process from the environment, so a test that
    /// wants the other arm has to set it directly. Serialized against itself:
    /// the suite runs `--test-threads=1`, and this makes the pairing explicit
    /// rather than relying on that.
    fn with_sink<R>(open: bool, body: impl FnOnce() -> R) -> R {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        redirect_logs_for_tests();
        let was_init = INIT.swap(true, Ordering::Relaxed);
        let was_enabled = ENABLED.swap(open, Ordering::Relaxed);
        let out = body();
        ENABLED.store(was_enabled, Ordering::Relaxed);
        INIT.store(was_init, Ordering::Relaxed);
        out
    }

    /// A diagnostic block does not run while the sink is closed.
    ///
    /// This is the property every caller gave up by asking whether logging was
    /// on and branching itself: the answer stayed on the product path's side of
    /// the call, and the work it guarded — a frame-wide byte scan, a guest
    /// page-table walk, a per-binding `format!` — stayed the product path's to
    /// skip, or to forget to skip. Here the sink owns both halves, and a caller
    /// has nothing to get wrong because it is never told the answer.
    #[test]
    fn a_diagnostic_block_does_not_run_while_the_sink_is_closed() {
        with_sink(false, || {
            let mut ran = 0u32;
            when_verbose(|| ran += 1);
            verbose(|| {
                ran += 1;
                "unreachable_line".to_owned()
            });
            assert_eq!(ran, 0, "a closed sink ran a diagnostic block");
        });
    }

    /// And does run while it is open, so the gate is a gate and not a deletion.
    #[test]
    fn a_diagnostic_block_runs_while_the_sink_is_open() {
        with_sink(true, || {
            let mut ran = 0u32;
            when_verbose(|| ran += 1);
            verbose(|| {
                ran += 1;
                "verbose_probe_line".to_owned()
            });
            assert_eq!(ran, 2, "an open sink skipped a diagnostic block");
        });
    }

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

    /// The writer's heartbeat is a census line, not a per-batch one.
    ///
    /// It is emitted from the writer's drain loop, which on a driven boot wakes
    /// hundreds of times a second, so without this gate the beat would be a
    /// flood in the file it exists to make readable — the same failure
    /// `released_pages::note_levels` was built to avoid. A second's worth of
    /// wake-ups yields one line.
    #[test]
    fn a_seconds_worth_of_writer_wakeups_beats_once() {
        let base = 5_000u128;
        assert!(
            !writer_beat_due(base, base),
            "no time has passed, so nothing is due"
        );
        let due = (1..1000)
            .filter(|i| writer_beat_due(base, base + i))
            .count();
        assert_eq!(due, 0, "a wake-up inside the interval must not beat");
        assert!(
            writer_beat_due(base, base + u128::from(WRITER_BEAT_MS)),
            "the wake-up that reaches the next interval beats"
        );
    }

    /// An idle writer still beats, and it beats once per interval rather than
    /// once per interval it slept through.
    ///
    /// This is the reading the whole line exists for: a device that has stopped
    /// emitting must look different from a writer that has stopped writing, and
    /// it only can if the beat survives silence.
    #[test]
    fn a_long_silence_is_still_due_exactly_once_per_interval() {
        let base = 0u128;
        assert!(writer_beat_due(base, base + 10_000), "silence is still due");
        // The caller advances `last` to `now`, so the interval after a long
        // silence is measured from the beat, not from the silence's start.
        assert!(!writer_beat_due(10_000, 10_500));
        assert!(writer_beat_due(10_000, 11_000));
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

    /// A flat white surface and a painted one are the two this record has to
    /// tell apart, and the non-zero count cannot.
    ///
    /// This is the measurement written down: ten labelled boots of the
    /// blank-field defect were scored on `rgb_nz` and the white and painted
    /// populations overlapped completely, because both saturate a count of
    /// pixels with any colour in them. The mean separates them by more than a
    /// hundred, so no threshold has to be chosen in advance.
    #[test]
    fn a_flat_white_surface_is_distinguishable_from_a_painted_one_only_by_the_mean() {
        let white = vec![0xff_u8; 4 * 64];
        let mut painted = vec![0u8; 4 * 64];
        for (i, px) in painted.chunks_exact_mut(4).enumerate() {
            // Values in this rail's painted range, varied so the buffer is not
            // itself flat -- a flat grey would make the test pass for the wrong
            // reason.
            px[0] = 90 + (i as u8 % 21);
            px[1] = px[0];
            px[2] = px[0];
            px[3] = 0xff;
        }

        let w = rgba_rgb_stats(&white);
        let p = rgba_rgb_stats(&painted);

        assert_eq!(
            w.rgb_nz, p.rgb_nz,
            "the non-zero count cannot separate these, which is why the mean exists"
        );
        assert_eq!(w.mean_rgb, 255, "flat white is the top of the range");
        assert!(
            p.mean_rgb < 130,
            "a painted desktop must land far below white: {}",
            p.mean_rgb
        );
    }

    /// An empty buffer reports no brightness rather than inventing one.
    #[test]
    fn an_empty_buffer_reports_a_zero_mean_beside_a_zero_count() {
        let stats = rgba_rgb_stats(&[]);
        assert_eq!(stats.mean_rgb, 0);
        assert_eq!(stats.rgb_nz, 0);
        assert_eq!(stats.px0, [0, 0, 0, 0]);
    }
}
