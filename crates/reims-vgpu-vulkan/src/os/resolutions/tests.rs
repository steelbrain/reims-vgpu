#![expect(
    clippy::single_range_in_vec_init,
    reason = "a slice of one range is the argument the log takes"
)]

use std::collections::BTreeMap;
use std::sync::Barrier;

use reims_vgpu_core::memory::dirty::{DirtyLog, PAGE_SIZE};
use reims_vgpu_core::num::ToU64 as _;

use super::*;

const P: u64 = PAGE_SIZE;

/// The guest page behind page `i` of the tests' imports: scattered, no two neighbours following
/// on.
fn guest_page(i: u64) -> u64 {
    (i * 7919) % (1 << 18)
}

/// `range` of the tests' import as the backing table resolves it: a segment a page, each cut to
/// the range, counted in `asked`.
fn resolve(range: &Range<u64>, asked: &AtomicU64) -> Vec<Segment> {
    asked.fetch_add(1, Ordering::Relaxed);
    let mut segments = Vec::new();
    let mut at = range.start;
    while at < range.end {
        let page = at / P;
        let end = ((page + 1) * P).min(range.end);
        segments.push(Segment {
            offset: at,
            length: end - at,
            guest: Some(guest_page(page) * P + (at - page * P)),
        });
        at = end;
    }
    segments
}

/// A range is resolved once however often it is asked, and a range inside a kept one is cut from
/// it: the same answers as resolving each
#[test]
fn a_range_is_resolved_once_and_ranges_inside_it_cut() {
    let asked = AtomicU64::new(0);
    let r = Resolutions::default();
    let ask = |range: Range<u64>| -> Vec<Segment> {
        r.segments(&range, |range| resolve(range, &asked)).to_vec()
    };
    let direct = |range: Range<u64>| resolve(&range, &AtomicU64::new(0));
    assert_eq!(ask(0..8 * P), direct(0..8 * P));
    assert_eq!(ask(0..8 * P), direct(0..8 * P));
    assert_eq!(asked.load(Ordering::Relaxed), 1);
    assert_eq!(ask(P + 8..3 * P - 8), direct(P + 8..3 * P - 8));
    assert_eq!(ask(7 * P..8 * P), direct(7 * P..8 * P));
    assert_eq!(asked.load(Ordering::Relaxed), 1);
    // A range past the kept one is resolved, and holds it from then on.
    assert_eq!(ask(0..16 * P), direct(0..16 * P));
    assert_eq!(ask(0..8 * P), direct(0..8 * P));
    assert_eq!(ask(9 * P..10 * P), direct(9 * P..10 * P));
    assert_eq!(asked.load(Ordering::Relaxed), 2);
    assert_eq!(r.resolutions(), 2);
}

/// Past [`KEPT`] ranges the oldest goes, and is resolved again when asked again
#[test]
fn the_oldest_range_goes_past_kept() {
    let asked = AtomicU64::new(0);
    let r = Resolutions::default();
    let ask = |i: u64| {
        r.segments(&(i * 2 * P..(i * 2 + 1) * P), |range| {
            resolve(range, &asked)
        });
    };
    for i in 0..=KEPT.to_u64() {
        ask(i);
    }
    assert_eq!(asked.load(Ordering::Relaxed), KEPT.to_u64() + 1);
    ask(KEPT.to_u64());
    assert_eq!(asked.load(Ordering::Relaxed), KEPT.to_u64() + 1);
    ask(0);
    assert_eq!(asked.load(Ordering::Relaxed), KEPT.to_u64() + 2);
}

/// A kept range's segments cut to a range inside it: the first and last pieces cut, their guest
/// addresses moved by as much, a piece in no RAM kept as none
#[test]
fn cut_segments() {
    let kept = [
        Segment {
            offset: 0,
            length: 2 * P,
            guest: Some(0x10_0000 - 2 * P),
        },
        Segment {
            offset: 2 * P,
            length: P,
            guest: None,
        },
        Segment {
            offset: 3 * P,
            length: 4 * P,
            guest: Some(0x1_0000_0000),
        },
    ];
    let range = |r: Range<u64>| -> Vec<(u64, u64, Option<u64>)> {
        (cut(&kept, &r).into_iter())
            .map(|s| (s.offset, s.length, s.guest))
            .collect()
    };
    assert_eq!(
        range(P..4 * P + 8),
        vec![
            (P, P, Some(0x10_0000 - P)),
            (2 * P, P, None),
            (3 * P, P + 8, Some(0x1_0000_0000)),
        ]
    );
    assert_eq!(
        range(5 * P..6 * P),
        vec![(5 * P, P, Some(0x1_0000_0000 + 2 * P))]
    );
    assert_eq!(range(2 * P + 1..2 * P + 2), vec![(2 * P + 1, 1, None)]);
    assert_eq!(range(7 * P..8 * P), vec![]);
}

/// The pages of the measurement's surface: 3840x2400 texels of 4 bytes, over 4 KiB pages.
const SURFACE_PAGES: u64 = 3840 * 2400 * 4 / P;
/// The guest's RAM the surface lies scattered in ([`guest_page`]).
const RAM: u64 = (1 << 18) * P;
/// The host addresses of the RAM's first byte and of the view of the surface's pages.
const HOST_RAM: u64 = 0x7f00_0000_0000;
const VIEW: u64 = 0x7e00_0000_0000;

/// The backing table and the dirty log's registry as the Linux resolver asks them
/// (`os/linux/files.rs`'s `pieces`, then `os/linux/dirty.rs`'s `resolved`): each behind a lock
/// that every reporting thread takes on a miss, the table's runs in a map, and each piece holding
/// its file.
struct Backing {
    table: Mutex<Table>,
    /// The RAM's registration for the log: its file offsets, and the guest physical address of
    /// the first.
    registry: Mutex<(Range<u64>, u64)>,
}

struct Table {
    /// The view's runs, a page each: the host address of each, and its file offset.
    runs: BTreeMap<u64, u64>,
    /// The RAM's registration: its host addresses, the file offset of the first, and the file.
    ram: (Range<u64>, u64),
    file: Arc<u64>,
}

impl Backing {
    fn new() -> Backing {
        let runs = (0..SURFACE_PAGES)
            .map(|i| (VIEW + i * P, guest_page(i) * P))
            .collect();
        Backing {
            table: Mutex::new(Table {
                runs,
                ram: (HOST_RAM..HOST_RAM + RAM, 0),
                file: Arc::new(0),
            }),
            registry: Mutex::new((0..RAM, 0)),
        }
    }

    /// The bytes `range` of an import at host address `base`, resolved.
    fn resolve(&self, base: u64, range: &Range<u64>) -> Vec<Segment> {
        let (first, end) = (base + range.start, base + range.end);
        // Each piece: its host address, length, file offset and file.
        let mut pieces: Vec<(u64, u64, u64, Arc<u64>)> = Vec::new();
        {
            let table = self.table.lock();
            let mut at = first;
            while at < end {
                let run = (table.runs.range(..=at).next_back())
                    .filter(|&(&start, _)| at < start + P)
                    .map(|(&start, &offset)| (offset + (at - start), (start + P).min(end)));
                let (offset, to) = match run {
                    Some(found) => found,
                    None if table.ram.0.contains(&at) => (
                        table.ram.1 + (at - table.ram.0.start),
                        table.ram.0.end.min(end),
                    ),
                    None => break,
                };
                pieces.push((at, to - at, offset, Arc::clone(&table.file)));
                at = to;
            }
        }
        let registry = self.registry.lock();
        (pieces.into_iter())
            .map(|(at, length, offset, _file)| Segment {
                offset: at - base,
                length,
                guest: (registry.0.contains(&offset))
                    .then(|| registry.1 + (offset - registry.0.start)),
            })
            .collect()
    }
}

/// The guest physical runs of `segments`, as the Linux log's `guest_runs` takes them.
fn guest_runs(segments: &[Segment]) -> impl Iterator<Item = Range<u64>> + '_ {
    (segments.iter()).filter_map(|s| s.guest.map(|g| g..g + s.length))
}

/// This thread's CPU time in nanoseconds and the instructions it retired, each a count that only
/// grows; instructions are 0 where the host gives none.
#[cfg(target_os = "macos")]
fn counts() -> [u64; 2] {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock: u32) -> u64;
        fn thread_selfcounts(kind: i32, buf: *mut u64, nbytes: usize) -> i32;
    }
    const CLOCK_THREAD_CPUTIME_ID: u32 = 16;
    let mut out = [0u64; 2];
    // SAFETY: kind 1 writes two `u64`s, the thread's instructions and cycles, into `out`.
    let r = unsafe { thread_selfcounts(1, out.as_mut_ptr(), size_of_val(&out)) };
    // SAFETY: a clock macOS names.
    let ns = unsafe { clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) };
    [ns, if r == 0 { out[0] } else { 0 }]
}

/// This thread's CPU time in nanoseconds, and no instructions.
#[cfg(target_os = "linux")]
fn counts() -> [u64; 2] {
    let mut at = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `at` is a timespec, which `clock_gettime` writes.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) },
        0
    );
    let ns = u64::try_from(at.tv_sec).unwrap() * 1_000_000_000 + u64::try_from(at.tv_nsec).unwrap();
    [ns, 0]
}

/// What the runtime's reports of a command buffer's in-place writes over a 3840x2400 surface in
/// scattered 4 KiB guest pages cost, through the resolutions of its two logs on the threads that
/// report them at once (the commit's, a completion turn's and the completion's): each report
/// asks the surface's own import for its bytes once, and the guest's RAM, whose offsets are host
/// addresses, once for each run of the RAM the surface's pages are, stamping the pages each
/// answer names in a log that watches them. A measurement, run by hand in a release build (module
/// docs); `RESOLUTIONS_THREADS` (3) and `RESOLUTIONS_REPORTS` (100) set its size.
#[test]
#[ignore = "a measurement, run by hand in a release build"]
#[expect(
    clippy::disallowed_methods,
    reason = "a measurement: real time is what it reports"
)]
fn reports_cost() {
    let number = |name: &str, default: u64| {
        (std::env::var(name).ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let threads = number("RESOLUTIONS_THREADS", 3);
    let reports = number("RESOLUTIONS_REPORTS", 100);
    let backing = Backing::new();
    let log = DirtyLog::new(&[0..RAM], Box::new(|| {}));
    let pages: Vec<Range<u64>> = (0..SURFACE_PAGES)
        .map(|i| guest_page(i) * P..(guest_page(i) + 1) * P)
        .collect();
    log.watch(&pages);
    // The runs of the RAM the surface's pages are, in their order: no two follow on.
    let window: Vec<Range<u64>> = (pages.iter())
        .map(|g| HOST_RAM + g.start..HOST_RAM + g.end)
        .collect();
    let own = Resolutions::default();
    let ram = Resolutions::default();
    let report = || {
        let bytes = 0..SURFACE_PAGES * P;
        let segments = own.segments(&bytes, |r| backing.resolve(VIEW, r));
        log.wrote_runs(guest_runs(&segments));
        for run in &window {
            let segments = ram.segments(run, |r| backing.resolve(0, r));
            log.wrote_runs(guest_runs(&segments));
        }
    };
    // One report first, as the commit's resolves the surface's own bytes before the others ask.
    report();
    let resolved = own.resolutions() + ram.resolutions();
    let start = Barrier::new(usize::try_from(threads).unwrap());
    let figures: Vec<(f64, [u64; 2])> = std::thread::scope(|s| {
        let runs: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    start.wait();
                    let (at, before) = (std::time::Instant::now(), counts());
                    for _ in 0..reports {
                        report();
                    }
                    let after = counts();
                    (
                        at.elapsed().as_secs_f64(),
                        [after[0] - before[0], after[1] - before[1]],
                    )
                })
            })
            .collect();
        runs.into_iter().map(|r| r.join().unwrap()).collect()
    });
    let calls = 1 + window.len();
    let slowest = figures.iter().map(|f| f.0).fold(0.0, f64::max);
    let mean =
        |i: usize| figures.iter().map(|f| f.1[i] as f64).sum::<f64>() / (threads * reports) as f64;
    let (ns, instructions) = (mean(0), mean(1));
    let resolutions = (own.resolutions() + ram.resolutions() - resolved) as f64;
    println!(
        "reports_cost: {threads} threads x {reports} reports of {calls} calls: a report {:.3} ms \
         (the slowest thread's), {:.3} ms CPU, {:.0} instructions, {:.0} resolutions; a call \
         {:.1} ns CPU, {:.0} instructions",
        slowest * 1e3 / reports as f64,
        ns / 1e6,
        instructions,
        resolutions / (threads * reports) as f64,
        ns / calls as f64,
        instructions / calls as f64,
    );
}
