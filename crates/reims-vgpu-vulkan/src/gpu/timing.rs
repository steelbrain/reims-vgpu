//! Debug lines that time where a guest waits on the runtime's compiles, under this module's
//! target: `REIMS_VGPU_LOG=warn,reims_vgpu_vulkan::gpu::timing=debug`. They are diagnostics: no
//! line changes what the guest or the host sees, and with the target's debug lines off nothing
//! here reads a clock or adds a handler.
//!
//! Where the runtime (metal2vulkan-runtime) pays for a pipeline state, as its sources say:
//! - **At the creation**, synchronously on the calling thread: each stage's translation into
//!   SPIR-V, unless the runtime's memo, the creation's binary archives or its shader cache on disk
//!   holds it, then the variant's shader modules and layouts. The cache on disk is this
//!   platform's `~/.cache/reims-vgpu-vulkan` ([`crate::platform`]), under a directory of each
//!   runtime build: a new build of the runtime starts from an empty one. [`creation`] times each
//!   creator: render and compute states, from a descriptor or a function, the display's present
//!   pipelines, the predication step's pipeline, libraries and function lookups.
//! - **At a state's first use**: a render state's `VkPipeline` of each graphics key (the pass's
//!   formats and samples, the draw's topology, the encoder state the pipeline takes statically)
//!   is made by the first draw that needs it, a compute state's of each region specialization by
//!   the first dispatch, inside that command's runtime call on the encoding thread, where MoltenVK
//!   makes the Metal pipeline under it. [`encoder_call`] writes an encoder command that takes
//!   [`SLOW_CALL`] or more, naming the pipeline state set last.
//! - **At the commit**: the commands of the indirect command buffers a command buffer executes
//!   are recorded then, on the committing thread, making the `VkPipeline`s they need; [`commit`]
//!   writes a commit that takes [`SLOW_CALL`] or more. Those whose records the GPU may write are
//!   recorded later, by a host step on the runtime's scheduler once the work before them
//!   completed, which delays the command buffer's completion: [`commit`] also writes a command
//!   buffer that completes [`LATE_COMPLETION`] or more after its commit, with the times the
//!   runtime kept of it. Its GPU times are the host's ([`crate::platform`]'s device options): it
//!   "began" when its submission ended, or when its queue's command buffer before it "ended",
//!   where later, and "ended" when the runtime saw it complete.
//! - **Every completion**: [`commit`] adds each command buffer's times to a summary of its queue's
//!   (queues of one label are one), and the first completion [`SUMMARY_SPAN`] or more after the
//!   summary's first writes a line of each queue, then starts the next: how many completed, and
//!   the median, 95th percentile and most of each leg from the commit to the handler, in the
//!   times above. A queue's command buffers do not overlap in them, so its GPU legs' sum is how
//!   long it held the GPU, its completions' detection included.
//!
//! **Measured** on the M3 Max (MoltenVK, runtime 60b8cadc9), a render state of the suite's
//! `tri_vert` and `tri_frag` with one BGRA8 target, then three draws: with an empty shader cache
//! the creation took 43 ms and the first draw 10 ms; in a second process over the cache the first
//! wrote, 27 ms and 3 ms; a second state of the same descriptor in the same process, 8 ms and
//! 0.3 ms. Every later draw took 0.1 ms, and the commits under 1 ms.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use metal2vulkan::pipeline::{MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::os::{Clock as _, MonoTime};

use crate::os::OsClock;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_core::trace;

/// An encoder command or a commit that takes this long or longer is written.
pub const SLOW_CALL: Duration = Duration::from_millis(5);

/// A command buffer that completes this long or longer after its commit is written.
pub const LATE_COMPLETION: Duration = Duration::from_millis(500);

/// A completion this long or longer after the summary's first writes the summary ([`commit`]).
pub const SUMMARY_SPAN: Duration = Duration::from_secs(1);

/// The commits a line named so far: each line's command buffer is `#n`, its commit's number.
static COMMITS: AtomicU64 = AtomicU64::new(0);

/// The command buffers completed since the summary's first, of each queue.
static SUMMARY: Mutex<Summary> = Mutex::new(Summary::new());

/// Whether this module's debug lines are written.
pub fn on() -> bool {
    log::log_enabled!(log::Level::Debug)
}

/// The monotonic clock now: the platform's ([`OsClock`]).
fn now() -> MonoTime {
    OsClock::new().now_mono()
}

/// The time since `start` on [`now`]'s clock.
fn elapsed_since(start: MonoTime) -> Duration {
    now().saturating_duration_since(start)
}

/// `d` in milliseconds.
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// A stage's function as a line names it: its name, quoted, or a compiled function's.
pub fn stage(source: &StageSource) -> String {
    match source {
        StageSource::Function(f) => format!("{:?}", f.function.info().name),
        StageSource::Compiled(_) => "a compiled function".to_owned(),
    }
}

/// A render descriptor's stages as a line names them.
pub fn render_stages(d: &RenderPipelineDescriptor) -> String {
    match &d.fragment {
        Some(fragment) => format!("vertex {}, fragment {}", stage(&d.vertex), stage(fragment)),
        None => format!("vertex {}, no fragment function", stage(&d.vertex)),
    }
}

/// A mesh render descriptor's stages as a line names them.
pub fn mesh_stages(d: &MeshPipelineDescriptor) -> String {
    let object = d
        .object
        .as_ref()
        .map_or("no object function".to_owned(), |o| {
            format!("object {}", stage(o))
        });
    let fragment = d
        .fragment
        .as_ref()
        .map_or("no fragment function".to_owned(), |f| {
            format!("fragment {}", stage(f))
        });
    format!("{object}, mesh {}, {fragment}", stage(&d.mesh))
}

/// A made state as a line names it, by its `uniqueIdentifier`.
pub fn state(unique_identifier: u64) -> String {
    format!("made uniqueIdentifier {unique_identifier}")
}

/// A creator's answer `made` as its line says it: `object`'s words of what it made, nil, or the
/// `NSError`'s description.
pub fn answer<T>(
    made: &Result<Option<T>, NsErrorText>,
    object: impl FnOnce(&T) -> String,
) -> String {
    match made {
        Ok(Some(made)) => object(made),
        Ok(None) => "nil".to_owned(),
        Err(e) => format!("refused: {}", e.description),
    }
}

/// [`answer`] of a creator whose caller may catch a raise.
pub fn raised_answer<T>(
    made: &RaiseResult<Result<Option<T>, NsErrorText>>,
    object: impl FnOnce(&T) -> String,
) -> String {
    made.as_ref()
        .map_or_else(|_| "raised".to_owned(), |made| answer(made, object))
}

/// Runs the creator `make` of `sel`, and when the lines are on, writes its wall time on the
/// calling thread, with `what` it made of and the `outcome` (each read only for the line), and
/// `later`, what the state leaves to its first use.
pub fn creation<T>(
    sel: &str,
    later: &str,
    make: impl FnOnce() -> T,
    what: impl FnOnce() -> String,
    outcome: impl FnOnce(&T) -> String,
) -> T {
    if !on() {
        return make();
    }
    let start = now();
    let made = make();
    let elapsed = elapsed_since(start);
    log::debug!(
        "{sel}: {:.1} ms, synchronous on the calling thread{later}: {}; {}",
        ms(elapsed),
        outcome(&made),
        what(),
    );
    made
}

/// What a render state leaves to its first use, as a creation line says it.
pub const RENDER_LATER: &str = " (its VkPipelines are made at the draws that first need them)";

/// What a compute state leaves to its first use, as a creation line says it.
pub const DISPATCH_LATER: &str =
    " (its VkPipelines are made at the dispatches that first need them)";

/// Runs the encoder command `sel`'s runtime call `call`, and when the lines are on and it takes
/// [`SLOW_CALL`] or more, writes its wall time on the encoding thread, naming the
/// `uniqueIdentifier` of the pipeline state the encoder set last, if any (0 is none).
#[inline]
pub fn encoder_call<T>(sel: &str, pipeline: &AtomicU64, call: impl FnOnce() -> T) -> T {
    if !on() {
        return call();
    }
    timed_encoder_call(sel, pipeline, call)
}

/// [`encoder_call`] with the lines on: out of line, as they are a diagnostic most runs leave
/// off.
#[cold]
#[inline(never)]
fn timed_encoder_call<T>(sel: &str, pipeline: &AtomicU64, call: impl FnOnce() -> T) -> T {
    let start = now();
    let out = call();
    let elapsed = elapsed_since(start);
    if elapsed >= SLOW_CALL {
        let pipeline = match pipeline.load(Ordering::Relaxed) {
            0 => String::new(),
            id => format!(
                ", under the pipeline state of uniqueIdentifier {id} (a state makes a \
                 VkPipeline at the command that first needs it)"
            ),
        };
        log::debug!(
            "{sel}: {:.1} ms on the encoding thread{pipeline}",
            ms(elapsed)
        );
    }
    out
}

/// Runs `commit`, `cb`'s, and when the lines are on, writes it if it takes [`SLOW_CALL`] or more,
/// and has a completed handler of `cb`, added last, add the command buffer's times to its queue's
/// summary (the module documentation) and write the command buffer if it completes
/// [`LATE_COMPLETION`] or more after the commit began. A handler the runtime refuses (the command buffer was committed
/// already, which the commit then refuses too) is no line.
pub fn commit<T>(cb: &rt::CommandBuffer, commit: impl FnOnce() -> T) -> T {
    let traced = trace::on();
    if !on() && !traced {
        return commit();
    }
    let n = COMMITS.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
    let queue = cb.command_queue().label().unwrap_or_default();
    let start = now();
    let host_start = rt::host_time();
    let late = queue.clone();
    if traced {
        // cbcommit: command buffer, commit number, queue label
        trace::event_text("cbcommit", &[id(cb), n], &queue);
    }
    let _ = cb.add_completed_handler(Box::new(move |cb: &rt::CommandBuffer| {
        let elapsed = elapsed_since(start);
        let times = Times {
            commit: host_start,
            kernel_start: cb.kernel_start_time(),
            kernel_end: cb.kernel_end_time(),
            gpu_start: cb.gpu_start_time().unwrap_or(0.0),
            gpu_end: cb.gpu_end_time().unwrap_or(0.0),
            handler: rt::host_time(),
        };
        if traced {
            // cbdone (timed at its handler): command buffer, commit number, then its commit,
            // submission start and end, GPU start and end, in the trace's nanoseconds
            trace::event_at(
                nanos(times.handler),
                "cbdone",
                &[
                    id(cb),
                    n,
                    nanos(times.commit),
                    nanos(times.kernel_start),
                    nanos(times.kernel_end),
                    nanos(times.gpu_start),
                    nanos(times.gpu_end),
                ],
            );
            work_counts(cb);
        }
        if !on() {
            return;
        }
        let lines = SUMMARY.lock().add(&late, &times);
        for line in lines {
            log::debug!("{line}");
        }
        if elapsed < LATE_COMPLETION {
            return;
        }
        let since = |t: f64| {
            if t <= 0.0 {
                "none".to_owned()
            } else {
                format!("{:.1} ms", (t - host_start) * 1000.0)
            }
        };
        let gpu = |t: Result<f64, rt::Error>| t.map_or_else(|_| "none".to_owned(), since);
        log::debug!(
            "command buffer #{n} of queue {late:?} completed {:.1} ms after its commit began \
             (status {:?}); from the commit: submission began {}, ended {}; GPU began {}, \
             ended {}",
            ms(elapsed),
            cb.status(),
            since(cb.kernel_start_time()),
            since(cb.kernel_end_time()),
            gpu(cb.gpu_start_time()),
            gpu(cb.gpu_end_time()),
        );
    }));
    let out = commit();
    if traced {
        // committed: command buffer: its commit returned
        trace::event("committed", &[id(cb)]);
    }
    let elapsed = elapsed_since(start);
    if elapsed >= SLOW_CALL {
        log::debug!(
            "commit: {:.1} ms on the committing thread, command buffer #{n} of queue {queue:?} \
             (its indirect command buffers' commands are recorded here)",
            ms(elapsed),
        );
    }
    out
}

/// A command buffer's number in the trace: its address.
pub fn id(cb: &rt::CommandBuffer) -> u64 {
    u64::try_from(std::ptr::from_ref(cb).addr()).unwrap_or_default()
}

/// Seconds of the runtime's host clock, in the trace's nanoseconds (its clock is the runtime's,
/// [`trace_clock`]); 0 stays 0.
fn nanos(seconds: f64) -> u64 {
    if seconds <= 0.0 {
        return 0;
    }
    std::time::Duration::try_from_secs_f64(seconds)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// The trace's clock: the runtime's host clock, in nanoseconds, so a command buffer's times and
/// the trace's events are of one clock.
pub fn trace_clock() -> u64 {
    nanos(rt::host_time())
}

/// The draws encoded while the trace is on, which [`work_counts`] writes.
static DRAWS: AtomicU64 = AtomicU64::new(0);

/// Counts a draw for the trace ([`work_counts`]).
pub fn count_draw() {
    if trace::on() {
        DRAWS.fetch_add(1, Ordering::Relaxed);
    }
}

/// The runtime's counts of what its commit path did, written to the trace at most every 100 ms,
/// and the draws encoded so far.
fn work_counts(cb: &rt::CommandBuffer) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let t = trace::now();
    let last = LAST.load(Ordering::Relaxed);
    if t.saturating_sub(last) < 100_000_000
        || LAST
            .compare_exchange(last, t, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    let c = cb.command_queue().device().work_counts();
    // wc: the counts named in the order written, then the draws
    trace::event(
        "wc",
        &[
            c.timeline_waits,
            c.command_pools_made,
            c.live_tables_uploaded,
            c.pipelines_pooled,
            c.variant_pipelines_made,
            c.host_step_sampler_sets_made,
            c.ahead_render_passes_made,
            c.storage_forms_taken,
            c.addressable_buffers_made,
            c.dispatch_pipelines_pooled,
            c.creation_pipelines_made,
            c.kept_backings_made,
            c.gpu_pool_blocks_made,
            c.gpu_pool_blocks_released,
            c.gpu_pool_leases_peak,
            DRAWS.load(Ordering::Relaxed),
        ],
    );
}

/// The host times of a command buffer, in seconds of the runtime's `host_time` clock, each 0
/// where the runtime kept none: its commit began, the runtime's kernel start and end, GPU start and
/// end ([`crate::platform`]'s device options), and its last completed handler ran.
#[derive(Clone, Copy, Debug, Default)]
struct Times {
    commit: f64,
    kernel_start: f64,
    kernel_end: f64,
    gpu_start: f64,
    gpu_end: f64,
    handler: f64,
}

/// The legs a summary line names, in the order [`Times::legs`] answers them.
const LEGS: [&str; 6] = [
    "commit to submission",
    "submitting",
    "behind the queue",
    "GPU until seen complete",
    "seen complete to the handler",
    "commit to the handler",
];

/// The leg of [`LEGS`] whose sum a summary line writes: the GPU's.
const GPU_LEG: usize = 3;

impl Times {
    /// The legs in seconds, as [`LEGS`] names them: from the commit to the kernel start, to the
    /// kernel end, to the GPU start (the queue's command buffer before it still held the GPU), to
    /// the GPU end, to the handler, and from the commit to the handler. None where a time is
    /// missing.
    fn legs(&self) -> Option<[f64; 6]> {
        let t = [
            self.commit,
            self.kernel_start,
            self.kernel_end,
            self.gpu_start,
            self.gpu_end,
            self.handler,
        ];
        if t.iter().any(|&t| t <= 0.0) {
            return None;
        }
        Some([
            t[1] - t[0],
            t[2] - t[1],
            t[3] - t[2],
            t[4] - t[3],
            t[5] - t[4],
            t[5] - t[0],
        ])
    }
}

/// The command buffers of one queue a summary holds.
#[derive(Default)]
struct QueueLegs {
    /// The command buffers completed.
    completed: u64,
    /// Those of them with a time missing.
    untimed: u64,
    /// Each leg's values, in seconds, as [`LEGS`] names them.
    legs: [Vec<f64>; 6],
}

/// The command buffers completed since the first a summary holds, by their queue's label.
struct Summary {
    /// The handler time of its first completion, 0 before.
    first: f64,
    queues: BTreeMap<String, QueueLegs>,
}

impl Summary {
    const fn new() -> Self {
        Self {
            first: 0.0,
            queues: BTreeMap::new(),
        }
    }

    /// Adds a command buffer of the queue labelled `queue` that completed with `times`. When its
    /// handler ran [`SUMMARY_SPAN`] or more after the summary's first, answers a line of each
    /// queue, ordered by label, and starts over; otherwise none.
    fn add(&mut self, queue: &str, times: &Times) -> Vec<String> {
        let held = self.queues.entry(queue.to_owned()).or_default();
        held.completed += 1;
        match times.legs() {
            Some(legs) => {
                for (values, leg) in held.legs.iter_mut().zip(legs) {
                    values.push(leg);
                }
            }
            None => held.untimed += 1,
        }
        if times.handler <= 0.0 {
            return Vec::new();
        }
        if self.first <= 0.0 {
            self.first = times.handler;
        }
        let span = times.handler - self.first;
        if span < SUMMARY_SPAN.as_secs_f64() {
            return Vec::new();
        }
        let queues = std::mem::take(&mut self.queues);
        self.first = 0.0;
        queues
            .into_iter()
            .map(|(label, held)| summary_line(&label, span, held))
            .collect()
    }
}

/// The summary line of the queue labelled `label`, which held `held` over `span` seconds.
fn summary_line(label: &str, span: f64, mut held: QueueLegs) -> String {
    let mut line = format!(
        "summary of queue {label:?} over {span:.2} s: {} command buffers completed, {} without \
         the runtime's times",
        held.completed, held.untimed
    );
    if held.untimed == held.completed {
        return line;
    }
    let gpu: f64 = held.legs[GPU_LEG].iter().sum();
    line.push_str("; ms p50/p95/max of");
    for (i, (name, values)) in LEGS.iter().zip(&mut held.legs).enumerate() {
        values.sort_unstable_by(f64::total_cmp);
        let at = |percent: usize| values[(values.len() - 1) * percent / 100] * 1000.0;
        let sep = if i == 0 { " " } else { ", " };
        line.push_str(&format!(
            "{sep}{name} {:.2}/{:.2}/{:.2}",
            at(50),
            at(95),
            at(100)
        ));
    }
    line.push_str(&format!("; GPU legs sum to {:.1} ms", gpu * 1000.0));
    line
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;

    use log::LevelFilter;
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::sync::Mutex;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// The marker that makes a run of [`slow_work_is_a_debug_line`] the child.
    const CHILD: &str = "REIMS_VGPU_VULKAN_TIMING_CHILD";

    /// The lines of this module a hook took.
    type Seen = Arc<Mutex<Vec<String>>>;

    /// Installs a hook taking this module's debug lines, in this process for good.
    fn hook() -> Seen {
        let seen = Seen::default();
        let lines = Arc::clone(&seen);
        let installed = reims_vgpu_core::logging::set_hook(
            LevelFilter::Debug,
            Box::new(move |_, target, message| {
                if target == module_path!().trim_end_matches("::tests") {
                    lines.lock().push(message.to_owned());
                }
            }),
        );
        assert!(installed);
        seen
    }

    /// With this module's debug lines on, a creation, a slow encoder command and a command buffer
    /// completing late are each a line, and a fast command is none. The hook is the process's, so
    /// the test runs in a child of its own.
    #[test]
    fn slow_work_is_a_debug_line() {
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "gpu::timing::tests::slow_work_is_a_debug_line",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .env_remove(reims_vgpu_core::logging::ENV_VAR)
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            return;
        }
        assert!(!on(), "no line is written before a logger takes them");
        let seen = hook();
        assert!(on());

        let made = creation(
            "make:",
            RENDER_LATER,
            || 5,
            || "of it".to_owned(),
            |n| format!("made {n}"),
        );
        assert_eq!(made, 5);
        let pipeline = AtomicU64::new(7);
        encoder_call("fast:", &pipeline, || ());
        encoder_call("slow:", &pipeline, || {
            std::thread::sleep(SLOW_CALL + Duration::from_millis(1));
        });
        {
            let lines = seen.lock();
            assert_eq!(lines.len(), 2, "{lines:?}");
            assert!(lines[0].starts_with("make:: "), "{}", lines[0]);
            assert!(
                lines[0].ends_with(&format!(
                    " ms, synchronous on the calling thread{RENDER_LATER}: made 5; of it"
                )),
                "{}",
                lines[0]
            );
            assert!(lines[1].starts_with("slow:: "), "{}", lines[1]);
            assert!(
                lines[1].contains(
                    " ms on the encoding thread, under the pipeline state of \
                                   uniqueIdentifier 7 "
                ),
                "{}",
                lines[1]
            );
        }

        // A command buffer waiting on an event the host signals past LATE_COMPLETION.
        let d = dev();
        let q = d.new_command_queue().unwrap();
        q.set_label("reims-vgpu.timing-test");
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        cb.commit(OnFault::Return).unwrap();
        std::thread::sleep(LATE_COMPLETION + Duration::from_millis(50));
        event.set_signaled_value(1).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        // The completion's line is the last handler's, on the runtime's thread.
        let deadline = now().saturating_add(Duration::from_secs(10));
        let late = loop {
            let found = seen
                .lock()
                .iter()
                .find(|l| l.starts_with("command buffer #"))
                .cloned();
            if let Some(line) = found {
                break line;
            }
            assert!(now() < deadline, "no completion line");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(
            late.starts_with("command buffer #1 of queue \"reims-vgpu.timing-test\" completed "),
            "{late}"
        );
    }

    /// A summary answers no line until a completion [`SUMMARY_SPAN`] or more after its first,
    /// then one of each queue by label, with each leg's median, 95th percentile and most, and its
    /// GPU legs' sum, and starts over.
    #[test]
    fn summary_writes_each_queue_once_a_span() {
        let times = |at: f64, gpu_ms: f64| Times {
            commit: at,
            kernel_start: at + 0.001,
            kernel_end: at + 0.0015,
            gpu_start: at + 0.002,
            gpu_end: at + 0.002 + gpu_ms / 1000.0,
            handler: at + 0.0025 + gpu_ms / 1000.0,
        };
        let untimed = Times {
            commit: 100.4,
            handler: 100.5,
            ..Times::default()
        };
        let mut summary = Summary::new();
        assert!(summary.add("b", &times(100.0, 4.0)).is_empty());
        assert!(summary.add("a", &untimed).is_empty());
        let lines = summary.add("b", &times(101.0, 8.0));
        assert_eq!(
            lines,
            [
                "summary of queue \"a\" over 1.00 s: 1 command buffers completed, 1 without the \
                 runtime's times",
                "summary of queue \"b\" over 1.00 s: 2 command buffers completed, 0 without the \
                 runtime's times; ms p50/p95/max of commit to submission 1.00/1.00/1.00, \
                 submitting 0.50/0.50/0.50, behind the queue 0.50/0.50/0.50, GPU until seen \
                 complete 4.00/4.00/8.00, seen complete to the handler 0.50/0.50/0.50, commit to \
                 the handler 6.50/6.50/10.50; GPU legs sum to 12.0 ms",
            ]
        );
        assert!(summary.queues.is_empty());
        assert!(summary.add("b", &times(102.0, 4.0)).is_empty());
        assert!(summary.add("b", &times(102.5, 4.0)).is_empty());
    }

    /// Prints, in microseconds, how long a command buffer of one 16-byte blit fill and a signal
    /// of a shared event takes from its commit to its completed handler, with the legs of
    /// [`LEGS`] between, and how long the commit holds the committing thread: through the
    /// platform as core drives it (a blit encoder, a completed handler, the signal, the commit),
    /// and through the runtime alone on the same device, 200 of each, alternating, after 20 to
    /// warm up. The difference is what the platform adds between the commit and core's handler.
    /// Measured on the M3 Max (MoltenVK, a dev build, runtime a62e8a365f), at the median: 278 us
    /// from the commit to the handler through the platform and 269 us through the runtime, of
    /// which the commit to the submission 29 us and 20 us (the platform's submission check, on
    /// the runtime's scheduler thread) and the GPU until seen complete 221 us and 222 us; the
    /// commit held its thread 13 us and 10 us; one handler thread woken a command buffer, and
    /// every transition to scheduled run where submitted, either way.
    #[test]
    #[ignore = "a measurement, run by hand: prints the latencies"]
    fn commit_to_handler_latency() {
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::types::NsRange;

        let d = dev();
        let q = d.new_command_queue().unwrap();
        let raw_q = d.raw.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let buffer = crate::gpu::resource::tests::shared(16);
        let times = |commit: f64, cb: &rt::CommandBuffer| Times {
            commit,
            kernel_start: cb.kernel_start_time(),
            kernel_end: cb.kernel_end_time(),
            gpu_start: cb.gpu_start_time().unwrap_or(0.0),
            gpu_end: cb.gpu_end_time().unwrap_or(0.0),
            handler: rt::host_time(),
        };
        /// A path's name, each command buffer's times and how long its commit held its thread,
        /// and the runtime's counts below, summed.
        type Run = (&'static str, Vec<(Times, Duration)>, [u64; 3]);
        let mut runs: [Run; 2] = [
            ("platform", Vec::new(), [0; 3]),
            ("runtime", Vec::new(), [0; 3]),
        ];
        // The runtime's counts of handler threads woken, command buffer waiters woken and
        // transitions to `Scheduled` run where submitted.
        let counts = || {
            let c = d.raw.work_counts();
            [
                c.wakes_handler_threads,
                c.wakes_command_buffer_waiters,
                c.scheduled_where_submitted,
            ]
        };
        let mut value = 0;
        for round in 0..220 {
            for (platform, (_, seen, counted)) in runs.iter_mut().enumerate() {
                let before = counts();
                value += 1;
                let (tx, rx) = std::sync::mpsc::channel();
                let held = if platform == 0 {
                    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
                    let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
                    let fill = BlitCmd::FillBuffer {
                        buffer: &buffer,
                        range: NsRange::new(0, 16),
                        value: 7,
                    };
                    blit.encode(OnFault::Return, fill).unwrap();
                    blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
                    let raw = cb.raw.clone();
                    let start = rt::host_time();
                    cb.add_completed_handler(
                        OnFault::Return,
                        Box::new(move |_| tx.send(times(start, &raw)).unwrap()),
                    )
                    .unwrap();
                    cb.encode_signal(Some(EventRef::Shared(&event)), value)
                        .unwrap();
                    let at = now();
                    cb.commit(OnFault::Return).unwrap();
                    let held = elapsed_since(at);
                    cb.wait_until_completed(OnFault::Return).unwrap();
                    held
                } else {
                    let cb = raw_q.command_buffer().unwrap();
                    let mut e = cb.blit_command_encoder().unwrap();
                    e.fill_buffer(&buffer.raw, 0..16, 7).unwrap();
                    e.end_encoding().unwrap();
                    let start = rt::host_time();
                    cb.add_completed_handler(Box::new(move |cb: &rt::CommandBuffer| {
                        tx.send(times(start, cb)).unwrap();
                    }))
                    .unwrap();
                    cb.encode_signal_event(rt::EventRef::Shared(&event), value)
                        .unwrap();
                    let at = now();
                    cb.commit().unwrap();
                    let held = elapsed_since(at);
                    cb.wait_until_completed().unwrap();
                    held
                };
                let t = rx.recv().unwrap();
                if round >= 20 {
                    seen.push((t, held));
                    let after = counts();
                    for (n, (after, before)) in
                        counted.iter_mut().zip(after.into_iter().zip(before))
                    {
                        *n += after - before;
                    }
                }
            }
        }
        for (name, seen, [handler_wakes, waiter_wakes, scheduled_here]) in runs {
            let us = |s: f64| format!("{:.0}", s * 1e6);
            let at = |v: &[f64], percent: usize| v[(v.len() - 1) * percent / 100];
            let mut held: Vec<f64> = seen.iter().map(|(_, d)| d.as_secs_f64()).collect();
            held.sort_unstable_by(f64::total_cmp);
            println!(
                "{name}: {} command buffers, the commit held its thread {} us at the median, \
                 {} us p90; {handler_wakes} handler threads and {waiter_wakes} command buffer \
                 waiters woken, {scheduled_here} scheduled where submitted",
                seen.len(),
                us(at(&held, 50)),
                us(at(&held, 90))
            );
            let legs: Vec<[f64; 6]> = seen.iter().filter_map(|(t, _)| t.legs()).collect();
            for (i, leg) in LEGS.iter().enumerate() {
                let mut v: Vec<f64> = legs.iter().map(|l| l[i]).collect();
                v.sort_unstable_by(f64::total_cmp);
                println!(
                    "  {leg}: p10 {} us, median {} us, p90 {} us",
                    us(at(&v, 10)),
                    us(at(&v, 50)),
                    us(at(&v, 90))
                );
            }
        }
    }

    /// Prints the time from a command buffer's commit to its completion, in microseconds, on the
    /// runtime's default device made with and without `gpu_times_from_host`, over command
    /// buffers of one 256-byte blit fill each, 25 on one device then 25 on the other, for 40
    /// rounds after one to warm up: what writing a command buffer's two GPU timestamps costs
    /// ([`crate::platform`]'s device options). On MoltenVK 1.4.1 both devices answer GPU times
    /// from the host (the runtime's `TimestampsWaitForEveryQueue`), so the two lines measure the
    /// same path there.
    #[test]
    #[ignore = "a measurement, run by hand: prints the latencies"]
    fn gpu_times_from_host_latency() {
        let mut from_host = rt::DeviceOptions::default();
        from_host.gpu_times_from_host = true;
        let mut runs: Vec<_> = [
            ("GPU timestamps", rt::DeviceOptions::default()),
            ("from the host", from_host),
        ]
        .into_iter()
        .map(|(name, options)| {
            let d = rt::Device::system_default_with(&options).unwrap();
            let q = d.new_command_queue().unwrap();
            let b = d.new_buffer(4096, rt::ResourceOptions::default()).unwrap();
            (name, q, b, Vec::new())
        })
        .collect();
        for round in 0..41 {
            for (_, q, b, latencies) in &mut runs {
                for _ in 0..25 {
                    let cb = q.command_buffer().unwrap();
                    let mut e = cb.blit_command_encoder().unwrap();
                    e.fill_buffer(b, 0..256, 7).unwrap();
                    e.end_encoding().unwrap();
                    let start = now();
                    cb.commit().unwrap();
                    cb.wait_until_completed().unwrap();
                    if round > 0 {
                        latencies.push(elapsed_since(start).as_micros());
                    }
                }
            }
        }
        for (name, _, _, v) in &mut runs {
            v.sort_unstable();
            let at = |percent: usize| v[(v.len() - 1) * percent / 100];
            println!(
                "{name}: {} command buffers, commit to completion p10 {} us, median {} us, \
                 p90 {} us",
                v.len(),
                at(10),
                at(50),
                at(90)
            );
        }
    }
}
