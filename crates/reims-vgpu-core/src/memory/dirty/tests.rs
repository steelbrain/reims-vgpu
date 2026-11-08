#![expect(
    clippy::single_range_in_vec_init,
    reason = "a slice of one range is the argument the log takes"
)]

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use super::*;
use crate::testkit::clock::VirtualClock;

const P: u64 = PAGE_SIZE;
const W1: NonZeroU64 = NonZeroU64::new(1).unwrap();
const W2: NonZeroU64 = NonZeroU64::new(2).unwrap();

/// A log over 64 MiB of RAM at 0 and 64 MiB at 4 GiB, whose host counts the times it was told a
/// stamp is wanted.
fn counted() -> (DirtyLog, Arc<AtomicUsize>) {
    let told = Arc::new(AtomicUsize::new(0));
    let t = Arc::clone(&told);
    let log = DirtyLog::new(
        &[0..0x400_0000, 0x1_0000_0000..0x1_0400_0000],
        Box::new(move || {
            t.fetch_add(1, Ordering::SeqCst);
        }),
    );
    (log, told)
}

/// One run of guest RAM from `guest`, as an import of `length` bytes sees it from offset 0.
fn import(guest: u64, length: u64) -> Vec<Segment> {
    vec![Segment {
        offset: 0,
        length,
        guest: Some(guest),
    }]
}

fn changed(log: &DirtyLog, segments: &[Segment], since: DirtyStamp) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    log.changed(segments, since, &mut out);
    out
}

fn changed_except(
    log: &DirtyLog,
    segments: &[Segment],
    since: DirtyStamp,
    writer: NonZeroU64,
) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    log.changed_except(segments, since, writer, &mut out);
    out
}

/// The bitmap words of `watched` with the pages of `written` (guest physical) set.
fn words(watched: &[Range<u64>], written: &[Range<u64>]) -> Vec<u64> {
    let mut out = Vec::new();
    for run in watched {
        let mut g = run.start;
        while g < run.end {
            let mut word = 0u64;
            for i in 0..64 {
                let page = g + i * P;
                if written.iter().any(|w| w.start < page + P && page < w.end) {
                    word |= 1 << i;
                }
            }
            out.push(word);
            g += GRANULE;
        }
    }
    out
}

/// A harvest the host runs at once: the watched pages of `written` were written.
fn harvest(log: &DirtyLog, written: &[Range<u64>]) -> DirtyStamp {
    let mut watched = Vec::new();
    let stamp = log.begin(&mut watched).expect("no harvest in flight");
    assert!(log.end(stamp, &words(&watched, written)));
    stamp
}

/// A request answers the next harvest's stamp and tells the host once per stamp
#[test]
fn a_request_answers_the_next_harvest_and_tells_the_host_once() {
    let (log, told) = counted();
    assert_eq!(log.request(), 1);
    assert_eq!(log.request(), 1);
    assert_eq!(told.load(Ordering::SeqCst), 1);
    let mut watched = Vec::new();
    assert_eq!(log.begin(&mut watched), Some(1));
    assert_eq!(log.begin(&mut watched), None, "one harvest at a time");
    assert_eq!(log.request(), 2);
    assert_eq!(told.load(Ordering::SeqCst), 2);
    assert!(!log.end(2, &[]), "not the harvest in flight");
    assert!(log.end(1, &[]));
    assert_eq!(log.published(), 1);
    assert_eq!(log.request(), 2);
    assert_eq!(told.load(Ordering::SeqCst), 2);
}

/// The watched runs are whole granules, joined, and only the watched ones
#[test]
fn begin_gives_the_watched_granules_joined() {
    let (log, _) = counted();
    log.watch(&[P..2 * P, GRANULE + 5 * P..GRANULE + 6 * P]);
    log.watch(&[0x1_0000_0000 + 3 * GRANULE..0x1_0000_0000 + 3 * GRANULE + 1]);
    let mut watched = Vec::new();
    log.begin(&mut watched).unwrap();
    assert_eq!(
        watched,
        vec![
            0..2 * GRANULE,
            0x1_0000_0000 + 3 * GRANULE..0x1_0000_0000 + 4 * GRANULE
        ]
    );
    assert_eq!(log.counts().watched_granules, 3);
    assert!(!log.end(1, &[0, 0]), "three words are owed");
    assert!(log.end(1, &[0, 0, 0]));
}

/// A watch's pages read as written before it; then only the pages a harvest saw written
#[test]
fn a_harvest_stamps_only_the_written_pages() {
    let (log, _) = counted();
    let imp = import(0x10_0000, 16 * P);
    log.watch(&[0x10_0000..0x10_0000 + 16 * P]);
    assert_eq!(changed(&log, &imp, 0), vec![0..16 * P]);
    let s1 = harvest(&log, &[]);
    assert_eq!(
        changed(&log, &imp, 0),
        vec![0..16 * P],
        "the watch began before 1"
    );
    assert_eq!(changed(&log, &imp, s1), vec![]);
    let s2 = harvest(
        &log,
        &[
            0x10_0000 + 2 * P..0x10_0000 + 4 * P,
            0x10_0000 + 9 * P..0x10_0000 + 9 * P + 1,
        ],
    );
    assert_eq!(changed(&log, &imp, s1), vec![2 * P..4 * P, 9 * P..10 * P]);
    assert_eq!(changed(&log, &imp, s2), vec![]);
}

/// A whole granule written takes one run, and a word's runs of bits each take theirs
#[test]
fn whole_words_and_runs_of_bits() {
    let (log, _) = counted();
    log.watch(&[0..2 * GRANULE]);
    let s1 = harvest(&log, &[]);
    let mut watched = Vec::new();
    let s2 = log.begin(&mut watched).unwrap();
    assert!(log.end(s2, &[!0, 0b1011 << 60]));
    assert_eq!(
        changed(&log, &import(0, 2 * GRANULE), s1),
        vec![
            0..GRANULE,
            GRANULE + 60 * P..GRANULE + 62 * P,
            GRANULE + 63 * P..2 * GRANULE
        ]
    );
}

/// Bytes outside the RAM, pages of no watch and segments of no guest RAM read as written
#[test]
fn what_is_not_watched_reads_as_written() {
    let (log, _) = counted();
    log.watch(&[0..GRANULE]);
    let s = harvest(&log, &[]);
    let segments = [
        Segment {
            offset: 0,
            length: 2 * P,
            guest: Some(0),
        },
        Segment {
            offset: 2 * P,
            length: P,
            guest: None,
        },
        Segment {
            offset: 3 * P,
            length: P,
            guest: Some(GRANULE),
        },
        Segment {
            offset: 4 * P,
            length: P,
            guest: Some(0x8000_0000),
        },
        Segment {
            offset: 5 * P,
            length: P,
            guest: Some(P),
        },
    ];
    assert_eq!(changed(&log, &segments, s), vec![2 * P..5 * P]);
}

/// An unaligned import answers in its own offsets, cut to its bytes
#[test]
fn an_unaligned_import_answers_in_its_offsets() {
    let (log, _) = counted();
    log.watch(&[0..GRANULE]);
    let s1 = harvest(&log, &[]);
    let s2 = harvest(&log, &[3 * P..4 * P]);
    assert!(s2 > s1);
    // From guest 0x2100: offset 0 is 0x2100, page 3 is 0x3000..0x4000 = 0xf00..0x1f00.
    let seg = [Segment {
        offset: 0x40,
        length: 0x3000,
        guest: Some(0x2100),
    }];
    assert_eq!(changed(&log, &seg, s1), vec![0x40 + 0xf00..0x40 + 0x1f00]);
}

/// The end of the last watch of a granule drops its stamps, and a new watch stamps it again
#[test]
fn unwatch_drops_the_stamps() {
    let (log, _) = counted();
    let imp = import(0, P);
    log.watch(&[0..P]);
    log.watch(&[0..P]);
    let s = harvest(&log, &[]);
    log.unwatch(&[0..P]);
    assert_eq!(changed(&log, &imp, s), vec![], "one watch is left");
    log.unwatch(&[0..P]);
    assert_eq!(changed(&log, &imp, s), vec![0..P]);
    assert_eq!(log.counts().watched_granules, 0);
    log.watch(&[0..P]);
    assert_eq!(
        changed(&log, &imp, s),
        vec![0..P],
        "stamped with the next harvest"
    );
    let s2 = harvest(&log, &[]);
    assert_eq!(changed(&log, &imp, s2), vec![]);
}

/// The library's own writes stamp the next harvest's number, and show after any earlier stamp
#[test]
fn own_writes_stamp_the_next_harvest() {
    let (log, _) = counted();
    let imp = import(0, 4 * P);
    log.watch(&[0..4 * P]);
    let s1 = harvest(&log, &[]);
    log.wrote(P + 10..P + 20);
    assert_eq!(changed(&log, &imp, s1), vec![P..2 * P]);
    let s2 = harvest(&log, &[]);
    assert_eq!(changed(&log, &imp, s1), vec![P..2 * P]);
    assert_eq!(changed(&log, &imp, s2), vec![]);
    // A write while a harvest is in flight is after it.
    let mut watched = Vec::new();
    let s3 = log.begin(&mut watched).unwrap();
    log.wrote(2 * P..3 * P);
    assert!(log.end(s3, &words(&watched, &[])));
    assert_eq!(changed(&log, &imp, s3), vec![2 * P..3 * P]);
}

/// A write that lasts reads as written at every stamp until it ends, then by its stamp
#[test]
fn a_lasting_write_reads_as_written_until_it_ends() {
    let (log, _) = counted();
    let imp = import(0, 4 * P);
    log.watch(&[0..4 * P]);
    let s1 = harvest(&log, &[]);
    let id = log.begin_write(P..2 * P);
    let s2 = harvest(&log, &[]);
    assert_eq!(changed(&log, &imp, s2), vec![P..2 * P]);
    log.end_write(id);
    assert_eq!(
        changed(&log, &imp, s2),
        vec![P..2 * P],
        "stamped after the last harvest"
    );
    let s3 = harvest(&log, &[]);
    assert_eq!(changed(&log, &imp, s3), vec![]);
    assert!(s1 < s2);
}

/// A named writer's writes do not come back to it, and come back to everyone else
#[test]
fn a_named_write_is_left_out_only_for_its_writer() {
    let (log, _) = counted();
    let imp = import(0, 4 * P);
    log.watch(&[0..4 * P]);
    let s1 = harvest(&log, &[]);
    log.wrote_as(0..2 * P, W1);
    assert_eq!(changed_except(&log, &imp, s1, W1), vec![]);
    assert_eq!(changed_except(&log, &imp, s1, W2), vec![0..2 * P]);
    assert_eq!(changed(&log, &imp, s1), vec![0..2 * P]);
    // The guest writes one of the pages after: it shows to the writer.
    let s2 = harvest(&log, &[P..2 * P]);
    assert_eq!(changed_except(&log, &imp, s1, W1), vec![P..2 * P]);
    // Another writer takes a page over: the first writer sees it, the second does not.
    log.wrote_as(0..P, W2);
    assert_eq!(changed_except(&log, &imp, s2, W1), vec![0..P]);
    assert_eq!(changed_except(&log, &imp, s2, W2), vec![]);
}

/// A harvest that ends after a named write stamped past it still shows to the writer
#[test]
fn a_guest_write_under_a_later_named_write_shows_to_the_writer() {
    let (log, _) = counted();
    let imp = import(0, P);
    log.watch(&[0..P]);
    let s1 = harvest(&log, &[]);
    let mut watched = Vec::new();
    let s2 = log.begin(&mut watched).unwrap();
    // Stamped s2 + 1, before the guest's s2 arrives.
    log.wrote_as(0..P, W1);
    assert!(log.end(s2, &words(&watched, &[0..P])));
    assert_eq!(changed_except(&log, &imp, s1, W1), vec![0..P]);
    let s3 = harvest(&log, &[]);
    assert_eq!(changed_except(&log, &imp, s3, W1), vec![]);
    // An unnamed own write shows to every writer.
    log.wrote(0..P);
    assert_eq!(changed_except(&log, &imp, s3, W1), vec![0..P]);
}

/// A batch of writes stamps each run as its own write would, named or not
#[test]
fn a_batch_stamps_as_its_runs_would() {
    let (log, _) = counted();
    let imp = import(0, 8 * P);
    log.watch(&[0..8 * P]);
    let s1 = harvest(&log, &[]);
    log.wrote_runs([P + 8..2 * P, 3 * P..3 * P + 1, 6 * P..8 * P]);
    assert_eq!(
        changed(&log, &imp, s1),
        vec![P..2 * P, 3 * P..4 * P, 6 * P..8 * P]
    );
    let s2 = harvest(&log, &[]);
    log.wrote_runs_as([0..P, 2 * P..3 * P], W1);
    assert_eq!(changed_except(&log, &imp, s2, W1), vec![]);
    assert_eq!(changed_except(&log, &imp, s2, W2), vec![0..P, 2 * P..3 * P]);
    // An unnamed batch over the named pages raises their other writes.
    let s3 = harvest(&log, &[]);
    log.wrote_runs([0..P, 4 * P..5 * P]);
    assert_eq!(changed_except(&log, &imp, s3, W1), vec![0..P, 4 * P..5 * P]);
    log.wrote_runs([]);
    assert_eq!(changed(&log, &imp, s3), vec![0..P, 4 * P..5 * P]);
}

/// A batch of writes over chunks with side tables takes the named lock once, not once a run
#[test]
fn a_batch_takes_the_named_lock_once() {
    let (log, _) = counted();
    // Four pages in each of two chunks, both with side tables.
    let second = CHUNK_PAGES.to_u64() * P;
    let imp = vec![
        Segment {
            offset: 0,
            length: 4 * P,
            guest: Some(0),
        },
        Segment {
            offset: 4 * P,
            length: 4 * P,
            guest: Some(second),
        },
    ];
    log.watch(&[0..4 * P, second..second + 4 * P]);
    log.wrote_as(0..P, W1);
    log.wrote_as(second..second + P, W1);
    let s1 = harvest(&log, &[]);
    let takes = |log: &DirtyLog| log.named_takes.load(Ordering::Relaxed);
    let all = vec![0..P, 2 * P..3 * P, 4 * P..5 * P, 7 * P..8 * P];
    let runs = [
        0..P,
        2 * P..3 * P,
        second..second + P,
        second + 3 * P..second + 4 * P,
    ];
    let before = takes(&log);
    log.wrote_runs(runs.clone());
    assert_eq!(takes(&log) - before, 1, "one take for the batch");
    assert_eq!(changed_except(&log, &imp, s1, W1), all);
    let s2 = harvest(&log, &[]);
    let before = takes(&log);
    log.wrote_runs_as(runs, W2);
    assert_eq!(takes(&log) - before, 1, "one take for the named batch");
    assert_eq!(changed_except(&log, &imp, s2, W2), vec![]);
    assert_eq!(changed_except(&log, &imp, s2, W1), all);
    // A batch over chunks without side tables takes no lock.
    let (plain, _) = counted();
    plain.watch(&[0..4 * P]);
    plain.wrote_runs([0..P, 2 * P..3 * P]);
    assert_eq!(takes(&plain), 0);
}

/// A wait answers at once once published, and false at once while closed or retired
#[test]
fn waits_refuse_at_once_while_closed_and_after_the_retire() {
    let (log, told) = counted();
    let clock = VirtualClock::new();
    let s = log.request();
    let mut watched = Vec::new();
    assert_eq!(log.begin(&mut watched), Some(s));
    assert!(log.end(s, &[]));
    assert!(log.wait(s, &clock));
    log.close();
    log.close();
    assert!(!log.wait(s + 1, &clock));
    log.reopen();
    assert!(!log.wait(s + 1, &clock), "one close is left");
    log.reopen();
    log.retire();
    assert!(!log.wait(s + 1, &clock));
    assert!(log.wait(s, &clock), "published stays published");
    let before = told.load(Ordering::SeqCst);
    let _ = log.request();
    let _ = log.wait(s + 5, &clock);
    assert_eq!(
        told.load(Ordering::SeqCst),
        before,
        "no call after the retire"
    );
    assert_eq!(log.counts().refused, 4);
}

/// A wait that times out stalls the log until the next publication
#[test]
fn a_timed_out_wait_stalls_until_the_next_publication() {
    let (log, _) = counted();
    let clock = VirtualClock::new();
    let s = log.request();
    assert!(!log.wait_for(s, &clock, Duration::ZERO));
    assert_eq!(log.counts().stalls, 1);
    assert!(!log.wait(s, &clock), "stalled: false at once");
    assert_eq!(log.counts().refused, 1);
    let mut watched = Vec::new();
    assert_eq!(log.begin(&mut watched), Some(s));
    assert!(log.end(s, &[]));
    assert!(log.wait(s, &clock));
    let s2 = log.request();
    let t = std::thread::scope(|scope| {
        let waiter = scope.spawn(|| log.wait_for(s2, &clock, Duration::from_secs(3600)));
        let mut watched = Vec::new();
        let b = log.begin(&mut watched);
        assert_eq!(b, Some(s2));
        assert!(log.end(s2, &[]));
        waiter.join().unwrap()
    });
    assert!(t, "a publication wakes the wait");
}

/// Two RAM ranges that share a 2 MiB chunk keep apart, and the bytes between them read written
#[test]
fn ranges_sharing_a_chunk() {
    let log = DirtyLog::new(&[0..0x10_0000, 0x18_0000..0x40_0000], Box::new(|| {}));
    log.watch(&[0..0x40_0000]);
    let s1 = harvest(&log, &[]);
    let s = harvest(&log, &[0x18_0000..0x18_1000]);
    let imp = import(0, 0x40_0000);
    assert_eq!(
        changed(&log, &imp, s1),
        vec![0x10_0000..0x18_1000],
        "the hole, and the written page"
    );
    assert_eq!(changed(&log, &imp, s), vec![0x10_0000..0x18_0000]);
}

/// The page stamps the queries read so far.
fn pages_read(log: &DirtyLog) -> u64 {
    log.pages_read.load(Ordering::Relaxed)
}

/// One segment of one page for each guest page of `pages`, in order: an import over scattered
/// guest pages, as a copy past the views' gate sees them.
fn scattered(pages: &[u64]) -> Vec<Segment> {
    (pages.iter().enumerate())
        .map(|(i, &page)| Segment {
            offset: i.to_u64() * P,
            length: P,
            guest: Some(page * P),
        })
        .collect()
}

/// A granule written after `since` by no one is passed over: none of its pages is read, one
/// written after it has its pages read and its written page shown, and it is passed over again
/// from the write's own stamp
#[test]
fn a_quiet_granule_reads_no_page() {
    let (log, _) = counted();
    let imp = import(0, 4 * GRANULE);
    log.watch(&[0..4 * GRANULE]);
    let s1 = harvest(&log, &[]);
    let before = pages_read(&log);
    assert_eq!(changed(&log, &imp, s1), vec![]);
    assert_eq!(pages_read(&log) - before, 0, "four quiet granules");
    let s2 = harvest(&log, &[2 * GRANULE + 5 * P..2 * GRANULE + 6 * P]);
    let before = pages_read(&log);
    assert_eq!(
        changed(&log, &imp, s1),
        vec![2 * GRANULE + 5 * P..2 * GRANULE + 6 * P]
    );
    assert_eq!(pages_read(&log) - before, 64, "the written granule alone");
    let before = pages_read(&log);
    assert_eq!(changed(&log, &imp, s2), vec![]);
    assert_eq!(
        pages_read(&log) - before,
        0,
        "quiet again from the write's own stamp"
    );
    // A granule no watch holds has no peak: its pages are read, and read as written.
    assert_eq!(changed(&log, &import(4 * GRANULE, P), s2), vec![0..P]);
}

/// A write on either side of a granule boundary shows, whether the segment holding it ends at
/// the boundary, starts at it, or spans it, and the quiet side of each stays passed over
#[test]
fn a_write_at_a_granule_boundary_shows() {
    let (log, _) = counted();
    // Pages 63 and 64 (the last of granule 0, the first of granule 1), then 127, 128 and 191:
    // scattered, out of guest order.
    let imp = scattered(&[64, 63, 191, 128, 127]);
    log.watch(&[0..4 * GRANULE]);
    let s1 = harvest(&log, &[]);
    log.wrote(64 * P..64 * P + 1);
    assert_eq!(
        changed(&log, &imp, s1),
        vec![0..P],
        "page 64, the first segment"
    );
    let s2 = harvest(&log, &[63 * P..64 * P]);
    assert_eq!(changed(&log, &imp, s1), vec![0..2 * P]);
    assert_eq!(changed(&log, &imp, s2), vec![]);
    log.wrote(127 * P..129 * P);
    assert_eq!(changed(&log, &imp, s2), vec![3 * P..5 * P]);
    // One segment across the boundary of granules 1 and 2, written at its far side's first page.
    let across = import(120 * P, 16 * P);
    let s3 = harvest(&log, &[]);
    let before = pages_read(&log);
    log.wrote(128 * P..128 * P + 8);
    assert_eq!(changed(&log, &across, s3), vec![8 * P..9 * P]);
    assert_eq!(
        pages_read(&log) - before,
        8,
        "granule 1's eight pages passed over"
    );
}

/// The writers' answers are unchanged past the peaks: a granule only a writer wrote is read and
/// left out for it, another writer's write in it shows, and a lasting write shows in a granule
/// whose peak is quiet
#[test]
fn named_and_lasting_writes_past_the_peaks() {
    let (log, _) = counted();
    let imp = import(0, 2 * GRANULE);
    log.watch(&[0..2 * GRANULE]);
    let s1 = harvest(&log, &[]);
    log.wrote_as(3 * P..4 * P, W1);
    assert_eq!(changed_except(&log, &imp, s1, W1), vec![]);
    assert_eq!(changed_except(&log, &imp, s1, W2), vec![3 * P..4 * P]);
    log.wrote_as(5 * P..6 * P, W2);
    assert_eq!(changed_except(&log, &imp, s1, W1), vec![5 * P..6 * P]);
    let s2 = harvest(&log, &[]);
    let before = pages_read(&log);
    assert_eq!(changed_except(&log, &imp, s2, W1), vec![]);
    assert_eq!(pages_read(&log) - before, 0, "both granules quiet since s2");
    let id = log.begin_write(GRANULE + 7 * P..GRANULE + 8 * P);
    assert_eq!(
        changed(&log, &imp, s2),
        vec![GRANULE + 7 * P..GRANULE + 8 * P],
        "a lasting write in a quiet granule"
    );
    log.end_write(id);
    assert_eq!(
        changed(&log, &imp, s2),
        vec![GRANULE + 7 * P..GRANULE + 8 * P]
    );
}

/// A granule watched again after its last watch ended is quiet again once a harvest passes its
/// fresh stamp: its peak comes down with its stamps
#[test]
fn a_rewatched_granule_comes_quiet() {
    let (log, _) = counted();
    let imp = import(0, GRANULE);
    log.watch(&[0..GRANULE]);
    harvest(&log, &[0..P]);
    log.unwatch(&[0..GRANULE]);
    let s = harvest(&log, &[]);
    assert_eq!(
        changed(&log, &imp, s),
        vec![0..GRANULE],
        "not watched: written"
    );
    log.watch(&[0..GRANULE]);
    assert_eq!(
        changed(&log, &imp, s),
        vec![0..GRANULE],
        "fresh: written at s"
    );
    let s2 = harvest(&log, &[]);
    let before = pages_read(&log);
    assert_eq!(changed(&log, &imp, s2), vec![]);
    assert_eq!(pages_read(&log) - before, 0);
    log.wrote(9 * P..10 * P);
    assert_eq!(changed(&log, &imp, s2), vec![9 * P..10 * P]);
}

/// Beside a harvester and a second writer on threads of their own, every host write a query
/// follows shows in it, for any stamp from before the write: a passed-over granule never hides
/// one
#[test]
fn a_query_beside_harvests_sees_every_write_before_it() {
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    const GRANULES: u64 = 16;
    /// Stops the other threads however the scope ends, a failed assertion included.
    struct Stop<'a>(&'a AtomicBool);
    impl Drop for Stop<'_> {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let (log, _) = counted();
    let log = &log;
    // Every fourth page of 16 granules, as scattered segments.
    let pages: Vec<u64> = (0..GRANULES * 64).step_by(4).collect();
    let imp = scattered(&pages);
    log.watch(&[0..GRANULES * GRANULE]);
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let _stop = Stop(&done);
        let (tx, rx) = mpsc::channel::<(DirtyStamp, usize)>();
        // The harvester: each harvest writes a page of its own choosing, or none.
        scope.spawn(|| {
            let mut n = 0u64;
            while !done.load(Ordering::SeqCst) {
                let page = (n * 37) % (GRANULES * 64);
                let mut watched = Vec::new();
                if let Some(stamp) = log.begin(&mut watched) {
                    let written = if n % 3 == 0 {
                        vec![]
                    } else {
                        vec![page * P..page * P + P]
                    };
                    assert!(log.end(stamp, &words(&watched, &written)));
                }
                n += 1;
            }
        });
        // A second writer, named, over pages the query does not ask about.
        scope.spawn(|| {
            let mut n = 0u64;
            while !done.load(Ordering::SeqCst) {
                let page = ((n * 13) % (GRANULES * 16)) * 4 + 1;
                log.wrote_as(page * P..page * P + P, W2);
                n += 1;
            }
        });
        let count = pages.len();
        let written = pages.clone();
        scope.spawn(move || {
            for k in 0..20_000usize {
                let i = (k * 7919) % count;
                let since = log.started.load(Ordering::SeqCst);
                log.wrote(written[i] * P..written[i] * P + 1);
                tx.send((since, i)).unwrap();
            }
        });
        let mut checked = 0;
        for (since, i) in rx {
            let shown = changed(log, &imp, since);
            let at = i.to_u64() * P;
            assert!(
                shown.iter().any(|r| r.start <= at && at < r.end),
                "page {} written after stamp {since} not shown",
                pages[i]
            );
            checked += 1;
        }
        assert_eq!(checked, 20_000);
    });
}
