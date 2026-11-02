#![expect(
    clippy::single_range_in_vec_init,
    reason = "a slice of one range is the argument the log takes"
)]

use std::sync::atomic::AtomicUsize;

use metal2vulkan_runtime::DirtyPages as _;
use reims_vgpu_core::memory::dirty::host_wrote;

use super::*;
use crate::os::linux::lease_guest_memory;
use crate::os::linux::vm::memfd;

const P: u64 = PAGE_SIZE;
const W1: NonZeroU64 = NonZeroU64::new(1).unwrap();
const W2: NonZeroU64 = NonZeroU64::new(2).unwrap();

/// A log over guest 0..1 MiB and 4 GiB..4 GiB + 1 MiB, whose host counts the times it was told a
/// stamp is wanted.
fn log() -> (Arc<DirtyLog>, Arc<AtomicUsize>) {
    let told = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&told);
    let log = Arc::new(DirtyLog::new(
        &[0..0x10_0000, 0x1_0000_0000..0x1_0010_0000],
        Box::new(move || {
            t.fetch_add(1, Ordering::SeqCst);
        }),
    ));
    (log, told)
}

/// A harvest the host runs at once: the watched pages of `written` were written.
fn harvest(log: &DirtyLog, written: &[Range<u64>]) -> DirtyStamp {
    let mut watched = Vec::new();
    let stamp = log.begin(&mut watched).unwrap();
    let mut words = Vec::new();
    for run in &watched {
        let mut g = run.start;
        while g < run.end {
            let mut word = 0u64;
            for i in 0..64 {
                let page = g + i * P;
                if written.iter().any(|w| w.start < page + P && page < w.end) {
                    word |= 1 << i;
                }
            }
            words.push(word);
            g += GRANULE;
        }
    }
    assert!(log.end(stamp, &words));
    stamp
}

/// The guest's RAM as two ranges of one file, contiguous at fake host addresses (the backing
/// table only records them) and in the file, apart in the guest: guest 0..1 MiB at `base`, and
/// guest 4 GiB on at `base + 1 MiB`, from file offset 1 MiB.
fn ram(base: u64, log: Option<Arc<DirtyLog>>) -> super::super::GuestMemoryLease {
    let f = memfd(0x20_0000).unwrap();
    let range = |host, guest, offset| GuestRamFile {
        host,
        guest,
        length: 0x10_0000,
        fd: Arc::clone(&f),
        offset,
    };
    lease_guest_memory(
        [
            range(base, 0, 0),
            range(base + 0x10_0000, 0x1_0000_0000, 0x10_0000),
        ],
        log,
    )
}

fn changed(d: &ImportDirty, since: DirtyStamp) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    d.changed(rt::DirtyStamp(since), &mut out);
    out
}

fn changed_except(d: &ImportDirty, since: DirtyStamp, w: NonZeroU64) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    d.changed_except(rt::DirtyStamp(since), w, &mut out);
    out
}

/// An import has answers only over the guest's RAM of a device with a log
#[test]
fn an_import_has_answers_only_over_logged_ram() {
    let base = 0x7e00_0000_0000;
    let _plain = ram(base, None);
    assert!(ImportDirty::new(base + P).is_none(), "no log");
    let (log, _) = log();
    let base = 0x7e10_0000_0000;
    let _lease = ram(base, Some(Arc::clone(&log)));
    assert!(ImportDirty::new(base + P).is_some());
    assert!(ImportDirty::new(base + 0x40_0000).is_none(), "past the RAM");
}

/// A watch resolves to the guest's pages, split where the guest's RAM is apart, and answers in
/// the import's offsets, within the live watches only
#[test]
fn a_watch_answers_in_the_imports_offsets() {
    let base = 0x7e20_0000_0000;
    let (log, _) = log();
    let _lease = ram(base, Some(Arc::clone(&log)));
    // From guest 1 MiB - 8 KiB: two pages below the hole, then guest 4 GiB on.
    let d = ImportDirty::new(base + 0x10_0000 - 2 * P).unwrap();
    assert_eq!(changed(&d, 0), vec![], "nothing watched");
    let watch = d.watch(0..4 * P);
    assert_eq!(log.counts().watched_granules, 2);
    let s1 = harvest(&log, &[]);
    assert_eq!(changed(&d, s1), vec![]);
    let s2 = harvest(
        &log,
        &[0x10_0000 - P..0x10_0000, 0x1_0000_0000..0x1_0000_0000 + P],
    );
    assert_eq!(changed(&d, s1), vec![P..3 * P]);
    assert_eq!(changed(&d, s2), vec![]);
    drop(watch);
    assert_eq!(log.counts().watched_granules, 0);
    assert_eq!(changed(&d, s1), vec![], "no live watch");
}

/// Two watches of one import answer once for the bytes both cover
#[test]
fn overlapping_watches_answer_once() {
    let base = 0x7e30_0000_0000;
    let (log, _) = log();
    let _lease = ram(base, Some(Arc::clone(&log)));
    let d = ImportDirty::new(base).unwrap();
    let _a = d.watch(0..4 * P);
    let _b = d.watch(2 * P..8 * P);
    let s1 = harvest(&log, &[]);
    harvest(&log, &[0..8 * P]);
    assert_eq!(changed(&d, s1), vec![0..8 * P]);
}

/// The runtime's writes stamp the guest's pages: a named writer's do not come back to it, and do
/// to every other asker and import over the same pages
#[test]
fn the_runtimes_writes_and_writers() {
    let base = 0x7e40_0000_0000;
    let (log, _) = log();
    let _lease = ram(base, Some(Arc::clone(&log)));
    let a = ImportDirty::new(base).unwrap();
    let b = ImportDirty::new(base + P).unwrap();
    let _wa = a.watch(0..4 * P);
    let _wb = b.watch(0..2 * P);
    let s1 = harvest(&log, &[]);
    a.wrote_as(P..2 * P, W1);
    assert_eq!(changed_except(&a, s1, W1), vec![]);
    assert_eq!(changed_except(&a, s1, W2), vec![P..2 * P]);
    assert_eq!(changed_except(&b, s1, W2), vec![0..P], "b's offsets");
    a.wrote(3 * P..3 * P + 1);
    assert_eq!(changed_except(&a, s1, W1), vec![3 * P..4 * P]);
}

/// core's writes through a host address stamp the guest's pages behind it
#[test]
fn core_writes_reach_the_log() {
    let base = 0x7e50_0000_0000;
    let (log, _) = log();
    let _lease = ram(base, Some(Arc::clone(&log)));
    let d = ImportDirty::new(base).unwrap();
    let _w = d.watch(0..4 * P);
    let s1 = harvest(&log, &[]);
    host_wrote(base + 2 * P + 8, 8);
    assert_eq!(changed(&d, s1), vec![2 * P..3 * P]);
}

/// A request asks the host once per stamp, and a wait for it returns once the harvest ends
#[test]
fn requests_and_waits_go_to_the_log() {
    let base = 0x7e60_0000_0000;
    let (log, told) = log();
    let _lease = ram(base, Some(Arc::clone(&log)));
    let d = ImportDirty::new(base).unwrap();
    let s = d.request();
    assert_eq!(d.request(), s);
    assert_eq!(told.load(Ordering::SeqCst), 1);
    harvest(&log, &[]);
    assert!(d.wait(s));
    log.close();
    assert!(!d.wait(rt::DirtyStamp(s.0 + 1)));
}

/// The writers' switch reads `0` as off and anything else, or nothing, as on
#[test]
fn the_writers_switch() {
    assert!(writers_on(None));
    assert!(writers_on(Some(std::ffi::OsStr::new("1"))));
    assert!(!writers_on(Some(std::ffi::OsStr::new("0"))));
}
