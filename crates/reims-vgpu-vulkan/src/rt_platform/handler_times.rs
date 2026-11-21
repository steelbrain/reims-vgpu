//! The time our completed handler holds rt's completion thread: rt runs a command buffer's
//! completed handlers on that thread, one after another, and `super::command`'s `hand_on` runs the
//! platform's there inline unless it hands them on (`super::command`'s docs, "Handed on"). While
//! one runs, no other command buffer of the device completes. A diagnostic: which lines the host
//! writes is free under the contract.
//!
//! - **What is kept.** Each inline run ([`time`]): how many ran, those over 1 ms and over 16 ms,
//!   the longest, and a histogram of their times from which the line reads its p50 and p99, each
//!   the top of its bucket, an eighth of an octave wide, or the longest where that is less. The
//!   handlers handed on to `reims-vgpu.rt-handed-on` are not timed: they run off rt's thread.
//! - **The longest one's releases.** What the platform's objects' drops let go of on the thread
//!   while a run lasts ([`released`]), by kind: buffers (`buffer`, of their length), heaps
//!   (`heap`, of their size), textures by how they were made (`texture-<made as>`, as the boot
//!   diagnostics name it, of their levels' bytes as rt lays them out times their samples:
//!   [`texture_bytes`]), pipeline states (`pipeline-state`) and samplers (`sampler`), of no
//!   bytes. The run that is the longest so far keeps its own.
//! - **Cheap and always on.** A run reads rt's host clock (`rt::host_time`) twice and adds to
//!   relaxed atomics; it takes a lock only when it is the longest so far. A drop off a run reads
//!   a thread-local and computes nothing.
//! - **The line**, `completed handlers on rt's completion thread: N run, p50 X ms, p99 Y ms, max
//!   Z ms, over 1 ms A, over 16 ms B; the longest's releases: KIND N (B bytes), ...` (or `none`),
//!   cumulative since the process began ([`line()`]): with the boot diagnostics at the device's end
//!   (`super::diag::write`), and every 5 s, where it changed, in the host's log at debug under
//!   `reims_vgpu::rt_gauges` (`super::gauge_log`).

use std::cell::RefCell;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use metal2vulkan_runtime as rt;

use crate::core::sync::Mutex;

/// The sub-buckets of an octave, as a power of two: 8.
const SUB: u32 = 3;

/// The histogram's buckets: one for each nanosecond count below `1 << SUB`, then `1 << SUB` for
/// each octave above.
const BUCKETS: usize = ((64 - SUB as usize) + 1) << SUB;

/// The kinds a run's releases keep apart, more than the kinds there are.
const KINDS: usize = 16;

/// One nanosecond count's bucket.
fn bucket(nanos: u64) -> usize {
    if nanos < 1 << SUB {
        return nanos as usize;
    }
    let octave = 63 - nanos.leading_zeros();
    let sub = (nanos >> (octave - SUB)) & ((1 << SUB) - 1);
    (((octave - SUB + 1) as usize) << SUB) | sub as usize
}

/// The most nanoseconds `bucket` holds.
fn top(bucket: usize) -> u64 {
    if bucket < 1 << SUB {
        return bucket as u64;
    }
    let octave = (bucket >> SUB) as u32 + SUB - 1;
    let sub = (bucket & ((1 << SUB) - 1)) as u64;
    let low = (1u64 << octave) | (sub << (octave - SUB));
    low + ((1u64 << (octave - SUB)) - 1)
}

/// What a run's drops let go of, by kind: each kind, how many, and their bytes; an empty kind
/// ends them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Releases([(&'static str, u64, u64); KINDS]);

impl Releases {
    /// None yet.
    const NONE: Releases = Releases([("", 0, 0); KINDS]);

    /// Adds one release of `kind`, of `bytes`; the last kind takes any past the room, as `other`.
    fn add(&mut self, kind: &'static str, bytes: u64) {
        let at = (self.0.iter())
            .position(|&(k, _, _)| k == kind || k.is_empty())
            .unwrap_or(KINDS - 1);
        let entry = &mut self.0[at];
        if entry.0 != kind {
            entry.0 = if entry.0.is_empty() { kind } else { "other" };
        }
        entry.1 += 1;
        entry.2 = entry.2.saturating_add(bytes);
    }

    /// The kinds, those of the most bytes first, then the most releases.
    fn kinds(&self) -> Vec<(&'static str, u64, u64)> {
        let mut kinds: Vec<_> = (self.0.iter())
            .filter(|e| !e.0.is_empty())
            .copied()
            .collect();
        kinds.sort_by_key(|&(kind, n, bytes)| (Reverse(bytes), Reverse(n), kind));
        kinds
    }
}

thread_local! {
    /// The releases of the run on this thread, while one runs ([`time`]).
    static TALLY: RefCell<Option<Releases>> = const { RefCell::new(None) };
}

/// Counts a release of `kind` and its bytes (the module's docs) where a run on this thread is
/// timed; `bytes` is called only then.
pub(super) fn released(kind: &'static str, bytes: impl FnOnce() -> u64) {
    let on = TALLY.try_with(|t| t.try_borrow().is_ok_and(|t| t.is_some()));
    if !on.unwrap_or(false) {
        return;
    }
    let bytes = bytes();
    let _ = TALLY.try_with(|t| {
        if let Ok(mut t) = t.try_borrow_mut()
            && let Some(t) = t.as_mut()
        {
            t.add(kind, bytes);
        }
    });
}

/// The kind of a texture made as `made_as` (`super::diag::Made::made_as`).
pub(super) fn texture_kind(made_as: &str) -> &'static str {
    match made_as {
        "ordinary" => "texture-ordinary",
        "surface-image" => "texture-surface-image",
        "view" => "texture-view",
        "surface-view" => "texture-surface-view",
        "over-buffer" => "texture-over-buffer",
        "tiled" => "texture-tiled",
        "in-heap" => "texture-in-heap",
        "drawable" => "texture-drawable",
        _ => "texture",
    }
}

/// The bytes of a texture of `d` on `device`: each level's as rt lays it out
/// (`Device::texture_level_info`), over its slices (a cube's six faces each), times its samples.
pub(super) fn texture_bytes(device: &rt::Device, d: &rt::TextureDescriptor) -> u64 {
    let faces = match d.texture_type {
        rt::TextureType::Cube | rt::TextureType::CubeArray => 6,
        _ => 1,
    };
    let slices = u64::from(d.array_length.max(1)) * faces;
    let level = |level: u32| {
        let query = rt::TextureLevelQuery {
            pixel_format: u64::from(d.pixel_format.as_raw()),
            width: u64::from(d.width),
            height: u64::from(d.height),
            depth: u64::from(d.depth),
            array_length: slices,
            level: u64::from(level),
            planes: 0,
        };
        device.texture_level_info(&query).bytes
    };
    let mut bytes = 0u64;
    for at in 0..d.mipmap_level_count.max(1) {
        bytes = bytes.saturating_add(level(at));
    }
    bytes.saturating_mul(u64::from(d.sample_count.max(1)))
}

/// The runs' times (the module's docs).
pub(super) struct Times {
    count: AtomicU64,
    over_1_ms: AtomicU64,
    over_16_ms: AtomicU64,
    max: AtomicU64,
    buckets: [AtomicU64; BUCKETS],
    /// The longest run's nanoseconds and its releases.
    longest: Mutex<(u64, Releases)>,
}

/// The process's runs.
static TIMES: Times = Times::new();

impl Times {
    /// No runs yet.
    const fn new() -> Times {
        Times {
            count: AtomicU64::new(0),
            over_1_ms: AtomicU64::new(0),
            over_16_ms: AtomicU64::new(0),
            max: AtomicU64::new(0),
            buckets: [const { AtomicU64::new(0) }; BUCKETS],
            longest: Mutex::new((0, Releases::NONE)),
        }
    }

    /// Runs `handlers`, timed, with their releases on this thread.
    fn time(&self, handlers: impl FnOnce()) {
        let was = TALLY.try_with(|t| t.replace(Some(Releases::NONE)));
        let start = rt::host_time();
        handlers();
        let nanos = rt::host_time().saturating_sub(start);
        let tally = TALLY.try_with(|t| t.replace(was.ok().flatten()));
        self.add(nanos, tally.ok().flatten().unwrap_or(Releases::NONE));
    }

    /// Counts a run of `nanos` that let go of `releases`.
    fn add(&self, nanos: u64, releases: Releases) {
        self.count.fetch_add(1, Relaxed);
        if nanos > 1_000_000 {
            self.over_1_ms.fetch_add(1, Relaxed);
        }
        if nanos > 16_000_000 {
            self.over_16_ms.fetch_add(1, Relaxed);
        }
        self.buckets[bucket(nanos)].fetch_add(1, Relaxed);
        if self.max.fetch_max(nanos, Relaxed) < nanos {
            let mut longest = self.longest.lock();
            if longest.0 < nanos {
                *longest = (nanos, releases);
            }
        }
    }

    /// The nanoseconds at or under which `percent` of the runs took, to the top of its bucket.
    fn percentile(&self, counts: &[u64], percent: u64) -> u64 {
        let all: u64 = counts.iter().sum();
        let rank = (all * percent).div_ceil(100).max(1);
        let mut seen = 0;
        for (at, n) in counts.iter().enumerate() {
            seen += n;
            if seen >= rank {
                return top(at).min(self.max.load(Relaxed));
            }
        }
        0
    }

    /// The line (the module's docs).
    fn line(&self) -> String {
        let counts: Vec<u64> = (self.buckets.iter()).map(|b| b.load(Relaxed)).collect();
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let (_, releases) = *self.longest.lock();
        let kinds: Vec<String> = (releases.kinds().iter())
            .map(|(kind, n, bytes)| format!("{kind} {n} ({bytes} bytes)"))
            .collect();
        format!(
            "completed handlers on rt's completion thread: {} run, p50 {:.3} ms, p99 {:.3} ms, \
             max {:.3} ms, over 1 ms {}, over 16 ms {}; the longest's releases: {}",
            self.count.load(Relaxed),
            ms(self.percentile(&counts, 50)),
            ms(self.percentile(&counts, 99)),
            ms(self.max.load(Relaxed)),
            self.over_1_ms.load(Relaxed),
            self.over_16_ms.load(Relaxed),
            if kinds.is_empty() {
                "none".to_owned()
            } else {
                kinds.join(", ")
            }
        )
    }
}

/// Runs `handlers`, the completed handlers `super::command`'s `hand_on` runs inline on rt's
/// completion thread, timed with what they let go of (the module's docs).
pub(super) fn time(handlers: impl FnOnce()) {
    TIMES.time(handlers);
}

/// The process's line (the module's docs).
pub(super) fn line() -> String {
    TIMES.line()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each nanosecond count falls in a bucket whose top is at least it and within an eighth of
    /// it, the buckets in order.
    #[test]
    fn a_time_s_bucket_holds_it_within_an_eighth() {
        let mut last = 0;
        for nanos in (0..4096).chain([1_000_000, 16_000_000, 152_500_000, u64::MAX / 3, u64::MAX]) {
            let b = bucket(nanos);
            assert!(b < BUCKETS, "{nanos}");
            assert!(b >= last, "{nanos}");
            last = b;
            assert!(top(b) >= nanos, "{nanos}");
            assert!(
                top(b) - nanos <= nanos / 8,
                "{nanos} in {b} tops at {}",
                top(b)
            );
        }
        assert_eq!(bucket(u64::MAX), BUCKETS - 1);
        assert_eq!(top(BUCKETS - 1), u64::MAX);
    }

    /// The line holds the runs, their p50 and p99 to the top of their bucket (40 us's tops at
    /// 40.959 us, 3 ms's at 3.146 ms), the longest, those over 1 ms and over 16 ms, and the
    /// longest's releases by kind, the most bytes first.
    #[test]
    fn the_completed_handlers_line_says_their_times_and_the_longest_s_releases() {
        let times = Times::new();
        assert_eq!(
            times.line(),
            "completed handlers on rt's completion thread: 0 run, p50 0.000 ms, p99 0.000 ms, \
             max 0.000 ms, over 1 ms 0, over 16 ms 0; the longest's releases: none"
        );
        for _ in 0..97 {
            times.add(40_000, Releases::NONE);
        }
        let mut two = Releases::NONE;
        two.add("pipeline-state", 0);
        times.add(2_000_000, two);
        let mut long = Releases::NONE;
        long.add("buffer", 65_536);
        long.add("texture-ordinary", 8_294_400);
        long.add("buffer", 4096);
        long.add("sampler", 0);
        times.add(18_250_000, long);
        times.add(3_000_000, two);
        assert_eq!(
            times.line(),
            "completed handlers on rt's completion thread: 100 run, p50 0.041 ms, p99 3.146 ms, \
             max 18.250 ms, over 1 ms 3, over 16 ms 1; the longest's releases: texture-ordinary \
             1 (8294400 bytes), buffer 2 (69632 bytes), sampler 1 (0 bytes)"
        );
    }

    /// A timed run counts what is let go of on its thread while it runs, and nothing after; a
    /// release off a run is not counted.
    #[test]
    fn a_timed_run_counts_the_releases_on_its_thread() {
        let times = Times::new();
        released("heap", || unreachable!("no run is timed"));
        times.time(|| {
            released("heap", || 1 << 20);
            released("buffer", || 256);
            std::thread::scope(|s| {
                s.spawn(|| released("buffer", || unreachable!("another thread")));
            });
        });
        released("heap", || unreachable!("the run ended"));
        let (_, releases) = *times.longest.lock();
        assert_eq!(releases.kinds(), [("heap", 1, 1 << 20), ("buffer", 1, 256)]);
        assert_eq!(times.count.load(Relaxed), 1);
    }

    /// More kinds than the room keeps fold into the last, named `other`.
    #[test]
    fn kinds_past_the_room_count_as_other() {
        const NAMES: [&str; KINDS + 2] = [
            "k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11", "k12", "k13",
            "k14", "k15", "k16", "k17",
        ];
        let mut r = Releases::NONE;
        for name in NAMES {
            r.add(name, 1);
        }
        assert_eq!(r.0[KINDS - 1], ("other", 3, 3));
        assert_eq!(r.0[KINDS - 2], ("k14", 1, 1));
    }
}
