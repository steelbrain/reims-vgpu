//! Which guest page frames this device has written, for the whole boot.
//!
//! # The question this exists to answer
//!
//! Twelve guest kernel panics have been recorded on this project whose victims
//! are unrelated subsystems — an apfs btree node, an ifnet function pointer, a
//! HID driver's heap element, a malloc small-zone free list — several of them
//! filled with `0xffffffffffffffff`, which is what opaque white BGRA looks like
//! to a reader who is not expecting pixels. The standing reading is that some
//! write of this device's landed at an address it did not own. **That reading
//! has never been more than a shape match**, and it must not be quoted as an
//! attribution.
//!
//! That account used to be cited from the root `AGENTS.md`, which no longer
//! carries findings; it is restated here because this module exists to settle
//! it, and a reader who meets the citation and not the finding cannot.
//!
//! It stayed that way because nothing this device emitted could be compared
//! against what a panic actually names. XNU's `pmap_page_protect` panic prints a
//! **guest physical page number** (`pn=0x46b53b`), and this device knew its own
//! write destinations only as transient locals. "Did we write there?" was not a
//! hard question — it was an unasked one.
//!
//! This is the set that answers it: one bit per guest frame, set by every rail
//! that puts bytes into guest RAM, accumulated for the life of the boot and
//! dumped to the fail log as run-length spans. A panic's `pn` is then a lookup.
//!
//! # Read a hit and a miss differently
//!
//! They are not symmetric and a scorer must not treat them as such.
//!
//! A **miss** is strong: this device demonstrably never wrote that frame, so
//! whatever corrupted it was not these write rails. That exonerates.
//!
//! A **hit** is evidence proportional to the footprint's density. A boot that
//! wrote 34 000 distinct frames of a 16 GiB guest has touched 0.8 % of it, so an
//! unrelated victim lands inside by chance about one time in 125. That is
//! informative and it is not proof: the device is *supposed* to write those
//! frames, and one it legitimately owned a moment ago may be one the guest has
//! since freed. `pages` is on every summary line precisely so a reader can
//! compute that ratio rather than assume it.
//!
//! # Frames are 4 KiB regardless of the guest's page size
//!
//! [`FRAME_SHIFT`] is fixed rather than taking the device's `page_shift`, for
//! two reasons. It removes a `page_shift` parameter from every hook, several of
//! which sit at layers that have no business knowing the guest's page geometry
//! (`gpa_map`, the QEMU host shim). And 4 KiB is at least as fine as any guest
//! page this project supports — arm64's 16 KiB page marks four frames and stays
//! exact — so nothing is rounded up into a frame no byte reached.
//!
//! # Whole boot, not a window
//!
//! A panic reports the state of memory, not the time it was corrupted, and the
//! damaging write can predate it by minutes — the malloc free-list class is
//! discovered by a *later* allocation, not by the write that broke it. A
//! footprint that forgot would answer a question nobody is asking.
//!
//! # No silent cap
//!
//! The bit array is fixed and covers frames below [`MAX_FRAME`]. A mark above
//! that is counted in `dropped` and reported on every summary line, because a
//! footprint that quietly failed to record a write produces exactly the "miss"
//! that reads as an exoneration.
//!
//! # The address is the discriminator; the payload is not
//!
//! A companion census used to sample one write in 64 and score the longest run
//! of `0xff` bytes in it, on the theory that a device which rarely writes white
//! would make a white victim a sharp signal. A two-phase boot — 300 s of a page
//! with no white, then 300 s of an overwhelmingly white one — answered it: the
//! `0xff`-run rate tracked the guest's own content by about 99x, and the longest
//! run reached 4 961 bytes, longer than either poisoned `kalloc` element the
//! panic reports name. So this device does write those runs, whenever the guest
//! paints white, and a victim full of `0xff` is no more likely to be ours for
//! being full of `0xff`. The payload carries no information the frame set does
//! not, and the census that measured it was deleted rather than left sampling
//! every rail's payload forever to re-derive its own negative result.
//!
//! # This records where writes went; it does not adjudicate them
//!
//! A second companion — a write-after-retire detector — kept a parallel bit set
//! of frames the guest had said were no longer a surface's, and raised an alarm
//! when a write landed in one. It is gone, for three reasons that compound:
//!
//! - **It could not attribute its own findings.** Only the mapping rail's hits
//!   were ever a claim about this device; a raw-GVA write into a page some other
//!   surface used to own is ordinary guest page recycling with no event that
//!   could have cleared the bit. Its one live outing read 12 432 hits on
//!   essentially a single frame and was recorded as UNATTRIBUTED.
//! - **On the pathway that can be measured it never ran.** A 25 s driven
//!   x86/PCI Safari boot reported `retire_scans=0` over all 73 census samples,
//!   as did a 600 s boot before it — the same reading that had already once been
//!   traced to a structurally unreachable delete path and "repaired".
//! - **It was the most expensive thing in this module.** Excluding an aliased
//!   page needs every other live mapping's page list, so each Unmap built a
//!   `HashSet` of every mapped GPA in the device, on the drain worker that
//!   `drain_duty` shows at 0.93-0.99.
//!
//! A guest lifetime transition retires its page plan and mapping import. A raw
//! physical-page match is not a second ownership oracle: shared storage and
//! recycled pages make that inference ambiguous, so this module records writes
//! and does not turn their addresses into product behavior.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Frames are 4 KiB. See the module header — this is deliberately not the
/// guest's page size.
pub const FRAME_SHIFT: u32 = 12;

/// Frames the set can represent: 16 Mi of them, so 64 GiB of guest-physical
/// space, against a rig that boots a 16 GiB guest — the PCI hole and any high
/// BAR aperture sit well inside it. Costs one bit each: 2 MiB, once per process.
pub const MAX_FRAME: u64 = 16 * 1024 * 1024;

const WORDS: usize = (MAX_FRAME / 64) as usize;

/// Emit the run-length dump no more often than this. The summary line is
/// per-census (once a second); the runs are the expensive part.
const DUMP_INTERVAL_MS: u64 = 30_000;

/// Runs per `guest_write_footprint_runs` line. Keeps a line to a width a human
/// can read while keeping the part count low enough that reassembly is obvious.
const RUNS_PER_LINE: usize = 48;

struct Footprint {
    bits: Box<[AtomicU64]>,
    /// Frames whose bit went 0 → 1. Maintained incrementally so the summary
    /// costs no scan; the dump recomputes runs by scanning, which is why the
    /// dump is rate-limited and the summary is not.
    pages: AtomicU64,
    /// Marks for a frame at or above [`MAX_FRAME`]. Reported, never swallowed.
    dropped: AtomicU64,
    last_dump_ms: AtomicU64,
    last_dump_pages: AtomicU64,
    dump_seq: AtomicUsize,
    /// The runs the previous dump reported, so this one can report only what it
    /// adds. The set is monotone — a bit goes 0 → 1 and never back — so every
    /// dump after the first restates almost all of the one before it: a measured
    /// 1 239 of 1 243 spans on one boot and 1 724 of 1 733 on another, which put
    /// the reprint at roughly 15% of every byte in the failure log. That log is
    /// the only ground truth for what the protocol actually exercises, so the
    /// space matters.
    last_dump_runs: std::sync::Mutex<Vec<(u64, u64)>>,
}

impl Footprint {
    fn new() -> Self {
        let mut bits = Vec::with_capacity(WORDS);
        bits.resize_with(WORDS, || AtomicU64::new(0));
        Self {
            bits: bits.into_boxed_slice(),
            pages: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            last_dump_ms: AtomicU64::new(0),
            last_dump_pages: AtomicU64::new(u64::MAX),
            dump_seq: AtomicUsize::new(0),
            last_dump_runs: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn mark_range(&self, first: u64, last: u64) {
        if first < MAX_FRAME {
            let bounded_last = last.min(MAX_FRAME - 1);
            let first_word = first / 64;
            let last_word = bounded_last / 64;
            for word in first_word..=last_word {
                let lo = if word == first_word { first % 64 } else { 0 };
                let hi = if word == last_word {
                    bounded_last % 64 + 1
                } else {
                    64
                };
                let width = hi - lo;
                let mask = if width == 64 {
                    u64::MAX
                } else {
                    ((1u64 << width) - 1) << lo
                };
                // A contiguous range updates each bitmap word once. The
                // returned old word also names every 0 -> 1 transition, so the
                // distinct-page level remains exact without one atomic pair per
                // 4 KiB frame.
                let prev = self.bits[word as usize].fetch_or(mask, Ordering::Relaxed);
                let added = (mask & !prev).count_ones();
                if added != 0 {
                    self.pages.fetch_add(u64::from(added), Ordering::Relaxed);
                }
            }
        }
        let dropped_first = first.max(MAX_FRAME);
        if dropped_first <= last {
            self.dropped
                .fetch_add(last - dropped_first + 1, Ordering::Relaxed);
        }
    }

    fn get(&self, frame: u64) -> bool {
        if frame >= MAX_FRAME {
            return false;
        }
        let word = (frame / 64) as usize;
        self.bits[word].load(Ordering::Relaxed) & (1u64 << (frame % 64)) != 0
    }

    /// Inclusive `[start, end]` frame runs, ascending.
    fn runs(&self) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = Vec::new();
        for (w, cell) in self.bits.iter().enumerate() {
            let mut word = cell.load(Ordering::Relaxed);
            if word == 0 {
                continue;
            }
            let base = (w as u64) * 64;
            loop {
                let lo = word.trailing_zeros() as u64;
                // The run inside this word ends at the first clear bit at or
                // above `lo`. When the word is set through bit 63 there is no
                // such bit, and `trailing_zeros` of the resulting zero is 64 —
                // a length measured from bit 0, not from `lo`. Clamping to what
                // remains of the word is the difference between reporting
                // frames 60..=63 and claiming 60..=123, sixty frames this
                // device never wrote.
                let len = u64::from((!word >> lo).trailing_zeros()).min(64 - lo);
                let (s, e) = (base + lo, base + lo + len - 1);
                match out.last_mut() {
                    // Runs are found per 64-bit word, so a span crossing a word
                    // boundary arrives as two adjacent runs. Rejoin them, or the
                    // dump reports a fragmentation that is an artefact of the
                    // container rather than a fact about the device.
                    Some(last) if last.1 + 1 == s => last.1 = e,
                    _ => out.push((s, e)),
                }
                if lo + len >= 64 {
                    break;
                }
                word &= !(((1u64 << len) - 1) << lo);
                if word == 0 {
                    break;
                }
            }
        }
        out
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    fn reset(&self) {
        for cell in self.bits.iter() {
            cell.store(0, Ordering::Relaxed);
        }
        self.pages.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
        self.last_dump_ms.store(0, Ordering::Relaxed);
        self.last_dump_pages.store(u64::MAX, Ordering::Relaxed);
        self.dump_seq.store(0, Ordering::Relaxed);
        if let Ok(mut prev) = self.last_dump_runs.lock() {
            prev.clear();
        }
    }
}

/// The parts of `now` that `prev` did not already cover.
///
/// Both lists are sorted, disjoint and inclusive-ended, which is what
/// [`Footprint::runs`] produces. The result is expressed in the same form, so a
/// reader reassembling a boot's footprint takes the union of every dump's runs
/// and gets exactly what a full dump would have said.
///
/// A plain set-difference of the two run *lists* would be wrong: a single new
/// frame between two existing runs merges them, so the merged run is "new" as a
/// run while almost all of its frames are not. The difference has to be taken
/// over the covered space, which is what this does.
fn runs_added(now: &[(u64, u64)], prev: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    let mut i = 0usize;
    for &(start, end) in now {
        let mut cursor = start;
        // `prev` is sorted, and `now` is walked in order, so the index only ever
        // moves forward across the whole call.
        while i < prev.len() && prev[i].1 < cursor {
            i += 1;
        }
        let mut j = i;
        while j < prev.len() && prev[j].0 <= end {
            let (ps, pe) = prev[j];
            if ps > cursor {
                out.push((cursor, ps - 1));
            }
            cursor = cursor.max(pe.saturating_add(1));
            if cursor > end {
                break;
            }
            j += 1;
        }
        if cursor <= end {
            out.push((cursor, end));
        }
    }
    out
}

static FOOTPRINT: std::sync::LazyLock<Footprint> = std::sync::LazyLock::new(Footprint::new);

/// bytes it put in each.
pub fn note_written_range(gpa: u64, len: u64) {
    if len == 0 {
        return;
    }
    let first = gpa >> FRAME_SHIFT;
    let last = gpa.saturating_add(len - 1) >> FRAME_SHIFT;
    FOOTPRINT.mark_range(first, last);
}

/// Whether this device has written the frame containing `gpa` at any point in
/// this boot. The scorer's question, exposed so a test can ask it directly.
pub fn wrote_gpa(gpa: u64) -> bool {
    FOOTPRINT.get(gpa >> FRAME_SHIFT)
}

/// Distinct frames written, and marks discarded for being at or above
/// [`MAX_FRAME`].
pub fn counts() -> (u64, u64) {
    (
        FOOTPRINT.pages.load(Ordering::Relaxed),
        FOOTPRINT.dropped.load(Ordering::Relaxed),
    )
}

/// The per-census summary line, and — at most every [`DUMP_INTERVAL_MS`], and
/// only when the set has grown since the last one — the run-length dump.
///
/// Returns the lines rather than emitting them, so the caller keeps the choice
/// of sink and a test can read them without a log fixture.
pub fn census_lines(now_ms: u64) -> Vec<String> {
    let fp = &*FOOTPRINT;
    let (pages, dropped) = counts();
    let kib = (pages << FRAME_SHIFT) / 1024;
    // Levels, not per-interval: running totals for the boot, unlike
    // `store_routes`. Summing them across census lines multiplies by the
    // cadence, which is the error `AGENTS.md` describes for the opposite
    // mistake — a per-window series read as a boot total. Both directions are
    // wrong and neither is visible in the number, which is why the line says
    // which kind it is.
    let mut out = vec![format!(
        "guest_write_footprint pages={pages} kib={kib} dropped={dropped} \
         frame_shift={FRAME_SHIFT} (levels, not per-interval)"
    )];

    let last_ms = fp.last_dump_ms.load(Ordering::Relaxed);
    let last_pages = fp.last_dump_pages.load(Ordering::Relaxed);
    let due = last_pages == u64::MAX || now_ms.saturating_sub(last_ms) >= DUMP_INTERVAL_MS;
    if !due || pages == last_pages {
        return out;
    }
    fp.last_dump_ms.store(now_ms, Ordering::Relaxed);
    fp.last_dump_pages.store(pages, Ordering::Relaxed);
    let seq = fp.dump_seq.fetch_add(1, Ordering::Relaxed);

    let runs = fp.runs();
    // Report what this dump adds, not the whole set again. `seq` orders the
    // deltas and the summary line above carries the absolute `pages` level, so a
    // reader can both reassemble the footprint (union the dumps in `seq` order)
    // and check the reassembly against `pages` without any dump restating what
    // an earlier one already said.
    let added = match fp.last_dump_runs.lock() {
        Ok(mut prev) => {
            let added = runs_added(&runs, &prev);
            *prev = runs;
            added
        }
        // A poisoned lock means a previous dump panicked mid-update, so the
        // recorded set cannot be trusted to be a subset. Report everything —
        // over-reporting costs log space; under-reporting loses frames.
        Err(_) => runs,
    };
    let parts = added.len().div_ceil(RUNS_PER_LINE).max(1);
    for (i, chunk) in added.chunks(RUNS_PER_LINE).enumerate() {
        let spans: Vec<String> = chunk
            .iter()
            .map(|(a, b)| format!("{a:#x}-{b:#x}"))
            .collect();
        out.push(format!(
            "guest_write_footprint_runs seq={seq} part={}/{parts} added={} {}",
            i + 1,
            added.len(),
            spans.join(" ")
        ));
    }
    out
}

/// One test at a time over the process-global set, cleared on entry.
///
/// The set is deliberately global — it is a property of the boot, not of any
/// object — so a test asserting "this rail marked frame X" is only meaningful
/// if no other test is marking concurrently. `--test-threads=1` is the project
/// rule and this does not depend on it: a global whose correctness rests on the
/// runner's flags breaks for whoever runs a single test by name.
#[cfg(any(test, feature = "test-fixtures"))]
static TEST_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the set exclusively for a test and clear it. Held for the caller's scope.
#[cfg(any(test, feature = "test-fixtures"))]
pub fn exclusive_for_tests() -> std::sync::MutexGuard<'static, ()> {
    let g = TEST_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    FOOTPRINT.reset();
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::exclusive_for_tests as fresh;

    #[test]
    fn a_written_range_marks_every_frame_it_touches_including_partial_ends() {
        let _g = fresh();
        // Starts mid-frame and ends mid-frame: 0x1800..=0x37ff is three frames,
        // not the one the start address names.
        note_written_range(0x1800, 0x2000);
        assert!(wrote_gpa(0x1000), "the partial first frame counts");
        assert!(wrote_gpa(0x2000));
        assert!(wrote_gpa(0x3000), "the partial last frame counts");
        assert!(!wrote_gpa(0x0));
        assert!(!wrote_gpa(0x4000));
        assert_eq!(counts().0, 3);
    }

    #[test]
    fn a_zero_length_write_marks_nothing() {
        let _g = fresh();
        // Without the guard, `first..=last` with last == first claims a frame no
        // byte reached — inflating the footprint, which weakens every later hit.
        note_written_range(0x9000, 0);
        assert_eq!(counts(), (0, 0));
        assert!(!wrote_gpa(0x9000));
    }

    #[test]
    fn marking_the_same_frame_twice_counts_it_once() {
        let _g = fresh();
        note_written_range(0x5000, 0x1000);
        note_written_range(0x5000, 0x1000);
        note_written_range(0x5fff, 1);
        assert_eq!(counts().0, 1, "distinct frames, not marks");
    }

    #[test]
    fn a_scatter_list_does_not_claim_the_frames_between_its_pages() {
        let _g = fresh();
        // A fragmented surface's page list. A range over its hull would mark
        // 0x2000..0x8000 as well, which is memory belonging to someone else —
        // and every one of those frames would then read as a hit.
        for gpa in [0x1000u64, 0x9000] {
            note_written_range(gpa, 0x1000);
        }
        assert_eq!(counts().0, 2);
        assert!(!wrote_gpa(0x5000), "the gap is not ours to claim");
    }

    #[test]
    fn an_arm64_page_marks_its_four_frames_exactly() {
        let _g = fresh();
        note_written_range(0x4000, 1 << 14);
        assert_eq!(counts().0, 4, "16 KiB is four 4 KiB frames");
        for f in 4..8u64 {
            assert!(wrote_gpa(f << 12));
        }
        assert!(!wrote_gpa(0x8000), "and not the fifth");
    }

    #[test]
    fn a_frame_past_the_end_of_the_set_is_dropped_loudly_and_never_reads_back_as_written() {
        let _g = fresh();
        let past = MAX_FRAME << FRAME_SHIFT;
        note_written_range(past, 0x1000);
        assert_eq!(counts(), (0, 1), "counted as dropped, not as a page");
        assert!(
            !wrote_gpa(past),
            "an unrecorded write must not answer `true`; a false hit invents evidence"
        );
        let line = &census_lines(0)[0];
        assert!(
            line.contains("dropped=1"),
            "a dropped mark has to reach the log, or the miss it causes reads as \
             an exoneration: {line}"
        );
    }

    #[test]
    fn every_out_of_range_frame_is_counted_when_one_range_crosses_the_bound() {
        let _g = fresh();
        let start = (MAX_FRAME - 1) << FRAME_SHIFT;
        note_written_range(start, 4 << FRAME_SHIFT);
        assert_eq!(counts(), (1, 3));
        assert!(wrote_gpa(start));
    }

    #[test]
    fn runs_rejoin_across_word_boundaries_and_report_each_gap() {
        let _g = fresh();
        // 60..=70 crosses the 64-bit word boundary, which the per-word scan
        // finds as 60..=63 and 64..=70. Reported unjoined, the dump would claim
        // a fragmentation the device never produced.
        for frame in 60u64..=70 {
            note_written_range(frame << FRAME_SHIFT, 1);
        }
        note_written_range(200 << FRAME_SHIFT, 1);
        assert_eq!(FOOTPRINT.runs(), vec![(60, 70), (200, 200)]);
    }

    #[test]
    fn a_word_set_end_to_end_is_one_run_and_does_not_spin() {
        let _g = fresh();
        // `len == 64` is the case where the shift clearing the consumed bits
        // would be undefined. A wrong guard here hangs the census thread rather
        // than reporting a wrong number, which is the worse failure.
        note_written_range(0, 128 << FRAME_SHIFT);
        assert_eq!(FOOTPRINT.runs(), vec![(0, 127)]);
        assert_eq!(counts().0, 128);
    }

    #[test]
    fn a_run_ending_at_the_top_bit_of_a_word_terminates() {
        let _g = fresh();
        // Sets bits 32..=63 of word 0 and nothing in word 1: the scan must stop
        // at the end of the word rather than shifting past it.
        note_written_range(32 << FRAME_SHIFT, 32 << FRAME_SHIFT);
        assert_eq!(FOOTPRINT.runs(), vec![(32, 63)]);
    }

    #[test]
    fn the_dump_is_rate_limited_but_the_summary_is_not() {
        let _g = fresh();
        note_written_range(0x1000, 0x1000);
        let first = census_lines(0);
        assert!(
            first
                .iter()
                .any(|l| l.starts_with("guest_write_footprint_runs")),
            "the first census must carry a dump, or a panic in the first 30 s has \
             nothing to be scored against: {first:?}"
        );

        note_written_range(0x9000, 0x1000);
        let soon = census_lines(1_000);
        assert_eq!(soon.len(), 1, "summary only inside the interval: {soon:?}");
        assert!(soon[0].contains("pages=2"), "{}", soon[0]);

        let later = census_lines(DUMP_INTERVAL_MS);
        assert!(
            later.iter().any(|l| l.contains("0x9-0x9")),
            "the growth must appear once the interval elapses: {later:?}"
        );
    }

    #[test]
    fn a_dump_is_skipped_when_the_set_did_not_grow() {
        let _g = fresh();
        note_written_range(0x1000, 0x1000);
        let _ = census_lines(0);
        // The same frame again leaves the set unchanged, so re-emitting an
        // identical run list every 30 s would be pure log volume.
        note_written_range(0x1000, 0x1000);
        let idle = census_lines(10 * DUMP_INTERVAL_MS);
        assert_eq!(idle.len(), 1, "{idle:?}");
    }

    #[test]
    fn every_run_of_a_dump_is_reachable_from_the_part_lines_alone() {
        let _g = fresh();
        // More runs than fit on one line, so reassembly is what is under test.
        let n = RUNS_PER_LINE as u64 * 2 + 5;
        for i in 0..n {
            note_written_range((i * 4) << FRAME_SHIFT, 1);
        }
        let lines = census_lines(0);
        let parts: Vec<&String> = lines
            .iter()
            .filter(|l| l.starts_with("guest_write_footprint_runs"))
            .collect();
        assert!(parts.len() > 2, "expected several parts: {}", parts.len());
        let seen: usize = parts
            .iter()
            .map(|line| {
                line.split_whitespace()
                    .filter(|t| t.starts_with("0x") && t.contains('-'))
                    .count()
            })
            .sum();
        assert_eq!(
            seen, n as usize,
            "the chunks must sum to the whole set, or a scorer reassembling them \
             reports a smaller footprint than the device has"
        );
    }

    /// A second dump reports what it adds, not the whole set again.
    ///
    /// The set only ever grows, so a dump that restates it costs the log the
    /// entire history every time. What makes the delta safe is that the union of
    /// the dumps still reconstructs the footprint — asserted here by taking that
    /// union and comparing it to what a single full dump of the same bits says.
    #[test]
    fn a_later_dump_reports_only_what_it_adds() {
        let _g = fresh();
        note_written_range(0x1000, 0x1000);
        note_written_range(0x9000, 0x1000);
        let first = census_lines(0);
        let first_spans = spans_of(&first);
        assert_eq!(first_spans, vec![(1, 1), (9, 9)]);

        // A frame between the two, plus one beyond them. The frame between
        // merges 0x1 and 0x9 into one run once 0x2..=0x8 fill in, so a naive
        // diff of run *lists* would re-report frames 1 and 9.
        for f in 2..=8u64 {
            note_written_range(f << FRAME_SHIFT, 1);
        }
        note_written_range(0x20 << FRAME_SHIFT, 1);
        let second = census_lines(DUMP_INTERVAL_MS);
        let second_spans = spans_of(&second);
        assert_eq!(
            second_spans,
            vec![(2, 8), (0x20, 0x20)],
            "frames 1 and 9 were already reported and must not appear again"
        );

        // The union of both dumps is the whole footprint.
        let mut union: Vec<u64> = first_spans
            .iter()
            .chain(second_spans.iter())
            .flat_map(|&(a, b)| a..=b)
            .collect();
        union.sort_unstable();
        union.dedup();
        let expected: Vec<u64> = (1..=9).chain(std::iter::once(0x20)).collect();
        assert_eq!(union, expected);
    }

    /// `runs_added` works over covered frames, not over run identity.
    #[test]
    fn a_merge_only_reports_the_frames_that_caused_it() {
        // Two runs joined by one frame: only the joining frame is new.
        assert_eq!(runs_added(&[(0, 10)], &[(0, 4), (6, 10)]), vec![(5, 5)]);
        // A run that grew at both ends.
        assert_eq!(runs_added(&[(0, 10)], &[(3, 6)]), vec![(0, 2), (7, 10)]);
        // Nothing new at all.
        assert!(runs_added(&[(0, 4), (6, 10)], &[(0, 4), (6, 10)]).is_empty());
        // No prior dump: everything is new.
        assert_eq!(runs_added(&[(2, 3)], &[]), vec![(2, 3)]);
        // A prior run entirely below and entirely above the new one.
        assert_eq!(
            runs_added(&[(10, 12)], &[(0, 2), (10, 10), (20, 22)]),
            vec![(11, 12)]
        );
    }

    /// The `(start, end)` frame pairs named by a census's dump lines, in order.
    fn spans_of(lines: &[String]) -> Vec<(u64, u64)> {
        lines
            .iter()
            .filter(|l| l.starts_with("guest_write_footprint_runs"))
            .flat_map(|l| {
                l.split_whitespace()
                    .filter(|t| t.starts_with("0x") && t.contains('-'))
                    .map(|t| {
                        let (a, b) = t.split_once('-').expect("span");
                        (
                            u64::from_str_radix(a.trim_start_matches("0x"), 16).expect("start"),
                            u64::from_str_radix(b.trim_start_matches("0x"), 16).expect("end"),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}
