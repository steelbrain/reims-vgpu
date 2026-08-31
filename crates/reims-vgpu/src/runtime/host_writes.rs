//! Which guest pages this device has written, and when.
//!
//! The hypervisor's dirty bitmap witnesses guest CPU stores and nothing else, so
//! a host-side copy vouched for by "the guest has not written since I looked" can
//! still be stale — because *we* wrote. This is the missing half of that witness.
//!
//! # Why pages and not mappings
//!
//! Three candidate rules were scored against a full content fold before this one
//! was built, each by its own census counter. Those counters are gone with the
//! rules they scored — [`crate::runtime::gather_witness`] takes only the page-exact answer
//! now — so the names below are what the readings were called at the time and are
//! not greppable in a current log.
//!
//! A per-mapping count was measured first and it leaks. One driven boot read
//! fifteen binds where the sampled window's own mapping had not been written, the
//! guest had not written, and the bytes moved anyway. Guest pages are reachable
//! under more than one mapping id, so "mapping 12 was not written" is not "these pages were not
//! written", and a cache keyed on the former serves stale pixels fifteen times a
//! minute.
//!
//! The same boot read zero counterexamples for a *global* count, which moves for
//! every write anywhere — because it moves for every write including the ones a
//! narrower rule fails to attribute. Sound, and it invalidates a texture because
//! an unrelated scanout was composited.
//!
//! # Where it stands
//!
//! Once every writer records here — which took a second pass, because
//! `map_fresh_span_within`'s callers write through a raw alias and were invisible
//! to a hand-picked list of call sites — a driven boot reads **zero** binds where
//! the page-exact rule vouched and the bytes had moved, alongside zero for both
//! wider rules. Of the binds where the guest was quiet and the bytes were
//! identical, this rule serves 93 %; the rest are windows whose page set had just
//! moved.
//!
//! That measurement is what the fold is still there for. It runs on one bind in
//! [`crate::runtime::gather_witness::AUDIT_STRIDE`] rather than all of them, and its
//! counterexample cell is `gw_audit_unsound`: a standing alarm on the rule this
//! module exists to make sound, rather than the per-bind decision it began as.
//!
//! What that licenses is a cache over the zero-copy sampled gathers, valid iff
//! the hypervisor's guest generation has not moved **and** this says the pages
//! were not written. Neither half is sufficient alone and the measurements above
//! are what say so, rather than an argument that they ought to be.
//!
//! Built and measured live on a driven x86/PCI boot: **5852 gathers skipped
//! against 4167 taken, 14.25 GB not read against 4.56 GB read — 75.8 % of the
//! rail's bytes gone** — with all three unsound cells still zero and a Wikipedia
//! page rendering correctly under scroll.
//!
//! # Shape
//!
//! A per-page last-write epoch in lazily allocated chunks. One chunk occupies
//! one minimum supported host page; a contiguous surface therefore hashes its
//! chunk once and indexes page cells directly instead of hashing every GPA.
//! `wrote_any_since` has **no horizon**: however long ago the reader last
//! looked, the answer is exact.
//!
//! It was a ring of the last 64 writes, walked back to the reader's mark, until
//! 2026-08-09. That shape was chosen because "between two binds of the same
//! window (~8 ms apart) there is usually no host write to compare against", and
//! that premise does not survive compositing. A driven macos-26 boot put the
//! number on it — of 22 710 asks, **16 402** were quiet, **6 294** were refused
//! because the ring had forgotten, and **14** were a write that had really
//! landed in the window. Each refusal cost a full re-gather, and they were 17.2
//! GB of guest memory re-read per boot.
//!
//! The map was run as a shadow first, deciding nothing and only reporting
//! whether it agreed, and swapped in once six rails read zero in the one
//! direction that loses a frame. Cutting the refusals also let
//! [`crate::runtime::gather_witness`]'s content audit run on macos-26 for the
//! first time — it needs 64 consecutive *vouched* binds of one window — and it
//! agrees ~950 times across three boots. That audit, not a second copy of this
//! predicate, is the standing alarm here.
//!
//! Everything still fails closed. A write that cannot name its pages answers
//! "assume written". Named page epochs have no horizon and are never cleared.

/// Why [`HostWrites::wrote_any_since`] could not call a window quiet.
///
/// Causes used to share one `true`, and only `Overlap` means the bytes under the
/// reader's pages actually moved. `Unnamed` is the fail-closed answer when a
/// writer loses its allocation identity. Keeping the two distinct says whether
/// a workload really overlaps a cached window or whether a writer failed to
/// name its pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostWriteVerdict {
    /// Nothing this device recorded touched these pages.
    Quiet,
    /// A recorded write names one of these pages — the bytes moved.
    Overlap,
    /// A writer that could not say which pages it landed in, so every reader
    /// older than it must assume its own.
    Unnamed,
}

impl HostWriteVerdict {
    /// True for everything except [`Self::Quiet`] — the sense the caller needs
    /// when it is deciding whether to vouch, spelled once so a new variant
    /// cannot be silently read as quiet.
    pub fn wrote(self) -> bool {
        !matches!(self, Self::Quiet)
    }

    /// Census route naming this verdict, for the witness that reports it.
    pub fn route(self) -> &'static str {
        match self {
            Self::Quiet => "gw_hw_quiet",
            Self::Overlap => "gw_hw_overlap",
            Self::Unnamed => "gw_hw_unnamed",
        }
    }
}

/// A per-page last-write record: the answer [`HostWrites`] gives.
///
/// # It replaced a ring, and the number that retired the ring was fourteen
///
/// The module's `# Shape` section has the reading. Of 22 710 asks in a driven
/// macos-26 boot, 16 402 were quiet, 6 294 were refused because the 64-entry
/// ring had forgotten, and **14** were a write that had really landed in the
/// window. Each refusal cost a full re-gather; together they were 17.2 GB of
/// guest memory re-read per boot, and cutting them took the rail from 17.9 to
/// 23.6 frames a second.
///
/// # Fail-closed
///
/// A write that cannot name its pages sets [`Self::unnamed_at`], and every
/// reader older than it is refused. Named pages are never shed: forgetting one
/// would answer `Quiet` for a page that was written, which is the direction that
/// loses a frame.
///
/// # What watches it
///
/// Not a second copy of this predicate. [`crate::runtime::gather_witness`]'s
/// content audit folds the window's bytes across consecutive vouched binds and
/// compares them, which is the rule checked against the pixels rather than
/// against another implementation of itself. On macos-26 that audit had **never
/// run** before this record — it needs 64 consecutive vouched binds and the
/// ring's refusal rate meant the run never happened — and it now completes ~950
/// times across three boots with `gw_audit_unsound` at zero.
#[derive(Debug, Default)]
struct PageEpochs {
    /// Last epoch at which this device wrote each guest page. The page number
    /// itself selects a chunk and a cell; only populated chunks are allocated.
    chunks: std::collections::HashMap<u64, Box<EpochChunk>>,
    /// Newest write that could not name its pages. Fail-closed for any reader
    /// older than it, exactly as the ring's `Unknown` entry is.
    unnamed_at: u64,
    /// Pages currently armed: released by the guest and not mapped again.
    /// Unbounded on purpose — its size is the guest's, and a cap here would be
    /// a cap on what can be observed.
    armed: u64,
    /// Findings waiting to be drained. See [`RELEASED_REPORT_CAP`].
    hits: Vec<ReleasedWrite>,
    /// Findings the queue could not hold.
    hits_dropped: u64,
    /// Writes that named no pages while at least one page was armed. Such a
    /// write cannot be attributed to a page, so it is neither a finding nor a
    /// clean sheet, and saying so is the difference between a quiet instrument
    /// and a blind one.
    unnamed_while_armed: u64,
}

/// One minimum supported guest page per allocation. This is derived from the
/// project's page geometry rather than from a workload size.
const EPOCH_CHUNK_BYTES: usize = 1usize << crate::model::PAGE_SHIFT_X86;
const EPOCHS_PER_CHUNK: usize = EPOCH_CHUNK_BYTES / std::mem::size_of::<u64>();
const _: () = assert!(EPOCHS_PER_CHUNK.is_power_of_two());

#[derive(Debug)]
struct EpochChunk {
    /// Epoch that covers every cell. A later partial write lives in `cells`;
    /// readers take whichever is newer.
    all_at: u64,
    cells: [u64; EPOCHS_PER_CHUNK],
    /// Release epoch per page, or 0 for a page the guest has not taken back.
    ///
    /// Lazily allocated, because the pages a guest releases are a small part of
    /// the pages this device writes and an unarmed chunk should not pay for the
    /// array. See [`PageEpochs::arm`] for why the marker lives in this cell
    /// rather than in a watch of its own.
    released: Option<Box<[u64; EPOCHS_PER_CHUNK]>>,
}

impl Default for EpochChunk {
    fn default() -> Self {
        Self {
            all_at: 0,
            cells: [0; EPOCHS_PER_CHUNK],
            released: None,
        }
    }
}

/// A write this device recorded against a page the guest had taken back.
///
/// Both epochs travel with it: `released_at` is the write census epoch current
/// when the guest released the page, `wrote_at` the epoch of the write that
/// landed on it. Their difference is how many writes this device recorded in
/// between, which is the only interval that means anything here — wall time
/// would say when the report was drained, not when the ordering broke.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleasedWrite {
    /// Guest-physical address of the page, at this guest's page geometry.
    pub gpa: u64,
    /// Write census epoch current when the guest released the page.
    pub released_at: u64,
    /// Write census epoch of the write that landed on it afterwards.
    pub wrote_at: u64,
}

/// How many findings the report queue holds between drains.
///
/// This bounds the **derived alarm queue**, not the watch: the armed-page
/// population has no bound, and a page that reports is disarmed, so one page
/// can occupy this at most once. The queue is drained on every drain tranche.
/// What does not fit is counted — see [`HostWrites::dropped_released_writes`] —
/// because a defect that fires faster than the drain is itself a reading.
const RELEASED_REPORT_CAP: usize = 64;

/// Everything a hit needs that is not in the chunk: where to put it, and the
/// counters it moves. Passed alongside the chunk so the borrow checker can see
/// that the chunk and the queue are disjoint parts of [`PageEpochs`].
struct HitSink<'a> {
    armed: &'a mut u64,
    hits: &'a mut Vec<ReleasedWrite>,
    hits_dropped: &'a mut u64,
    page_shift: u32,
}

impl HitSink<'_> {
    /// Report and disarm every armed page in `chunk[slot..slot + take]`.
    ///
    /// Disarming is what makes one late write one finding: the page has now
    /// been written and keeping it armed would re-report the same defect for
    /// every later write to it.
    fn check(&mut self, chunk: &mut EpochChunk, chunk_key: u64, slot: usize, take: usize, at: u64) {
        let Some(released) = chunk.released.as_deref_mut() else {
            return;
        };
        for (offset, cell) in released[slot..slot + take].iter_mut().enumerate() {
            if *cell == 0 {
                continue;
            }
            let released_at = std::mem::replace(cell, 0);
            *self.armed = self.armed.saturating_sub(1);
            let page = chunk_key * EPOCHS_PER_CHUNK as u64 + (slot + offset) as u64;
            if self.hits.len() < RELEASED_REPORT_CAP {
                self.hits.push(ReleasedWrite {
                    gpa: page << self.page_shift,
                    released_at,
                    wrote_at: at,
                });
            } else {
                *self.hits_dropped += 1;
            }
        }
    }
}

impl PageEpochs {
    fn note_page_range(&mut self, mut page: u64, mut count: usize, epoch: u64, page_shift: u32) {
        let Self {
            chunks,
            armed,
            hits,
            hits_dropped,
            ..
        } = self;
        let mut sink = HitSink {
            armed,
            hits,
            hits_dropped,
            page_shift,
        };
        while count != 0 {
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
            let take = count.min(EPOCHS_PER_CHUNK - slot);
            let chunk = chunks
                .entry(chunk_key)
                .or_insert_with(|| Box::new(EpochChunk::default()));
            if slot == 0 && take == EPOCHS_PER_CHUNK {
                chunk.all_at = epoch;
            } else {
                chunk.cells[slot..slot + take].fill(epoch);
            }
            sink.check(chunk, chunk_key, slot, take, epoch);
            page += take as u64;
            count -= take;
        }
    }

    /// Record that the guest has taken `page` back.
    ///
    /// The marker lives in the same chunk the write census already touches, so
    /// a write finds it without a second lookup and without a sweep. That is
    /// the whole difference from the watch this replaced: a sweep costs the
    /// watched population per drain tranche, which is why that watch had a cap,
    /// and a cap on an instrument decides what it is able to see.
    ///
    /// A page released twice without an intervening map keeps its **first**
    /// release epoch: the question is whether anything was written since the
    /// guest stopped wanting us there, and re-stamping it would forgive a write
    /// that had already happened.
    fn arm(&mut self, page: u64, epoch: u64) {
        let chunk_key = page / EPOCHS_PER_CHUNK as u64;
        let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
        let chunk = self
            .chunks
            .entry(chunk_key)
            .or_insert_with(|| Box::new(EpochChunk::default()));
        let released = chunk
            .released
            .get_or_insert_with(|| Box::new([0; EPOCHS_PER_CHUNK]));
        if released[slot] != 0 {
            return;
        }
        // Epoch 0 is "never written", so an arm at epoch 0 would be
        // indistinguishable from an unarmed cell. Arm at 1 instead: no write
        // has happened yet, so no write can be older than it.
        released[slot] = epoch.max(1);
        self.armed += 1;
    }

    /// The guest has mapped `page` again, so writing to it is legitimate.
    fn disarm(&mut self, page: u64) {
        let chunk_key = page / EPOCHS_PER_CHUNK as u64;
        let Some(chunk) = self.chunks.get_mut(&chunk_key) else {
            return;
        };
        let Some(released) = chunk.released.as_deref_mut() else {
            return;
        };
        let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
        if std::mem::replace(&mut released[slot], 0) != 0 {
            self.armed = self.armed.saturating_sub(1);
        }
    }

    fn note_pages<I>(&mut self, pages: I, epoch: u64, page_shift: u32)
    where
        I: IntoIterator<Item = u64>,
    {
        let Self {
            chunks,
            armed,
            hits,
            hits_dropped,
            ..
        } = self;
        let mut sink = HitSink {
            armed,
            hits,
            hits_dropped,
            page_shift,
        };
        let mut pages = pages.into_iter().peekable();
        while let Some(gpa) = pages.next() {
            let page = gpa >> page_shift;
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let chunk = chunks
                .entry(chunk_key)
                .or_insert_with(|| Box::new(EpochChunk::default()));
            let slot = (page % EPOCHS_PER_CHUNK as u64) as usize;
            chunk.cells[slot] = epoch;
            sink.check(chunk, chunk_key, slot, 1, epoch);
            while pages
                .peek()
                .is_some_and(|&next| (next >> page_shift) / EPOCHS_PER_CHUNK as u64 == chunk_key)
            {
                let gpa = pages.next().expect("peeked page");
                let slot = ((gpa >> page_shift) % EPOCHS_PER_CHUNK as u64) as usize;
                chunk.cells[slot] = epoch;
                sink.check(chunk, chunk_key, slot, 1, epoch);
            }
        }
    }

    fn note_unknown(&mut self, epoch: u64) {
        self.unnamed_at = epoch;
        if self.armed != 0 {
            self.unnamed_while_armed += 1;
        }
    }

    /// The verdict this map would give, in the ring's own vocabulary.
    fn verdict(&self, since: u64, pages: &[u64], page_shift: u32) -> HostWriteVerdict {
        if since < self.unnamed_at {
            return HostWriteVerdict::Unnamed;
        }
        let mut first = 0usize;
        while first < pages.len() {
            let page = pages[first] >> page_shift;
            let chunk_key = page / EPOCHS_PER_CHUNK as u64;
            let mut end = first + 1;
            while end < pages.len()
                && (pages[end] >> page_shift) / EPOCHS_PER_CHUNK as u64 == chunk_key
            {
                end += 1;
            }
            if let Some(chunk) = self.chunks.get(&chunk_key) {
                if chunk.all_at > since {
                    return HostWriteVerdict::Overlap;
                }
                for &gpa in &pages[first..end] {
                    let slot = ((gpa >> page_shift) % EPOCHS_PER_CHUNK as u64) as usize;
                    if chunk.cells[slot] > since {
                        return HostWriteVerdict::Overlap;
                    }
                }
            }
            first = end;
        }
        HostWriteVerdict::Quiet
    }

    #[cfg(test)]
    fn recorded_pages(&self) -> usize {
        self.chunks
            .values()
            .map(|chunk| {
                if chunk.all_at != 0 {
                    EPOCHS_PER_CHUNK
                } else {
                    chunk.cells.iter().filter(|&&epoch| epoch != 0).count()
                }
            })
            .sum()
    }
}

/// Which guest pages this device has written, and when.
#[derive(Debug)]
pub struct HostWrites {
    /// Monotonic stamp; a reader records the value current at its own read and
    /// asks later whether anything newer touched its pages. Never 0 once any
    /// write has happened, so 0 is usable as "never looked".
    epoch: u64,
    /// Page addresses are normalized at the geometry of this guest, so arm64's
    /// 16 KiB pages occupy one cell rather than four sparse 4 KiB cells.
    page_shift: u32,
    /// The record itself. See [`PageEpochs`] for why it is page-keyed and what
    /// watches it.
    pages: PageEpochs,
}

impl Default for HostWrites {
    fn default() -> Self {
        Self::new(crate::model::PAGE_SHIFT_X86)
    }
}

impl HostWrites {
    pub fn new(page_shift: u32) -> Self {
        Self {
            epoch: 0,
            page_shift,
            pages: PageEpochs::default(),
        }
    }

    /// The stamp a reader records beside a copy it has just taken.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Record a write covering `pages`, which the caller resolved from
    /// `mapping_id`'s page list.
    ///
    /// `None` means the caller could not name every page, which is recorded as
    /// an unnamed write rather than as a partial one — naming some of a write's
    /// pages and not the rest would leave the unnamed ones readable as `Quiet`.
    ///
    /// The mapping id is not stored. It was, while the record was a ring that
    /// resolved the page list again at read time and had to refuse once the
    /// mapping moved; capturing the pages here is what retired that refusal.
    pub fn note_mapping(&mut self, pages: Option<&[u64]>) {
        self.epoch = self.epoch.wrapping_add(1);
        match pages {
            Some(p) => self
                .pages
                .note_pages(p.iter().copied(), self.epoch, self.page_shift),
            None => self.pages.note_unknown(self.epoch),
        }
    }

    /// Record a write covering exactly `pages` (page-aligned guest addresses).
    pub fn note_pages(&mut self, pages: Vec<u64>) {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.note_pages(pages, self.epoch, self.page_shift);
    }

    /// Record an already-resolved page iterator without materializing a second
    /// allocation. The caller retains ownership of the allocation identity;
    /// this type owns only its page-exact epochs.
    pub fn note_page_iter<I>(&mut self, pages: I)
    where
        I: IntoIterator<Item = u64>,
    {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.note_pages(pages, self.epoch, self.page_shift);
    }

    /// Record the exact page runs retained with an admitted guest allocation.
    /// The run partition was derived once with the resource, so a repeated
    /// Store updates slices rather than rebuilding adjacency page by page.
    pub fn note_footprint(&mut self, footprint: &crate::runtime::guest_ram::GuestPageFootprint) {
        self.epoch = self.epoch.wrapping_add(1);
        if footprint.page_size() != (1u64 << self.page_shift) {
            self.pages.note_unknown(self.epoch);
            return;
        }
        for run in footprint.runs() {
            let first_page = footprint.pages()[run.start] >> self.page_shift;
            self.pages.note_page_range(
                first_page,
                run.end - run.start,
                self.epoch,
                self.page_shift,
            );
        }
    }

    /// Record a write whose pages are not known. Invalidates every reader older
    /// than it.
    pub fn note_unknown(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.note_unknown(self.epoch);
    }

    /// Has this device written any of `pages` since `since`, and if so on what
    /// grounds?
    ///
    /// `since` is a value previously returned by [`Self::epoch`]. Everything it
    /// cannot decide answers as written. Only [`HostWriteVerdict::Overlap`]
    /// says a recorded write actually covers one of `pages`;
    /// [`HostWriteVerdict::Unnamed`] says a writer could not name its pages.
    ///
    /// One chunk lookup per contiguous run and one direct cell read per page;
    /// there is nothing to walk back to. The ring this replaced took
    /// `&DeviceState` to resolve a mapping-named write at read time; the pages
    /// are captured when the write happens now, so no state is needed and a
    /// mapping re-pointed afterwards cannot make the answer wrong.
    pub fn wrote_any_since(&self, since: u64, pages: &[u64]) -> HostWriteVerdict {
        self.pages.verdict(since, pages, self.page_shift)
    }

    /// Record that the guest has taken `gpa` back, so a write to it from here
    /// on is a write this device was told not to make.
    ///
    /// See [`crate::runtime::released_pages`] for what a finding means. The
    /// address is page-aligned at this guest's geometry by the same shift the
    /// write census uses, so an arm and a write cannot disagree about which
    /// cell they mean.
    pub fn release_page(&mut self, gpa: u64) {
        self.pages.arm(gpa >> self.page_shift, self.epoch);
    }

    /// The guest has mapped `gpa` again, so writing to it is legitimate.
    pub fn remap_page(&mut self, gpa: u64) {
        self.pages.disarm(gpa >> self.page_shift);
    }

    /// Take the findings recorded since the last drain.
    pub fn take_released_writes(&mut self) -> Vec<ReleasedWrite> {
        std::mem::take(&mut self.pages.hits)
    }

    /// How many pages are armed: released by the guest and not mapped again.
    pub fn armed_pages(&self) -> u64 {
        self.pages.armed
    }

    /// Findings the report queue could not hold. Non-zero means the readings
    /// name fewer pages than were written after release.
    pub fn dropped_released_writes(&self) -> u64 {
        self.pages.hits_dropped
    }

    /// Writes that named no pages while something was armed, and so could
    /// neither implicate nor clear it.
    pub fn unnamed_writes_while_armed(&self) -> u64 {
        self.pages.unnamed_while_armed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 4096;

    /// The record rules a window in and out by page, and a reader that already
    /// saw a write is not invalidated by it.
    #[test]
    fn a_write_to_other_pages_leaves_this_window_quiet() {
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_pages(vec![9 * P, 10 * P]);
        assert_eq!(
            w.wrote_any_since(mark, &[3 * P, 4 * P]),
            HostWriteVerdict::Quiet
        );
        assert_eq!(
            w.wrote_any_since(mark, &[4 * P, 10 * P]),
            HostWriteVerdict::Overlap
        );
    }

    /// The reader asks about writes *after* its own mark, so the write it
    /// already accounted for must not invalidate it forever.
    #[test]
    fn a_write_the_reader_already_saw_does_not_answer_for_a_later_mark() {
        let mut w = HostWrites::default();
        w.note_pages(vec![4 * P]);
        let after = w.epoch();
        assert_eq!(w.wrote_any_since(after, &[4 * P]), HostWriteVerdict::Quiet);
        w.note_pages(vec![4 * P]);
        assert_eq!(
            w.wrote_any_since(after, &[4 * P]),
            HostWriteVerdict::Overlap
        );
    }

    /// A writer that could not name its pages must invalidate everything older.
    /// Reading it as quiet is the wrong-frame direction.
    #[test]
    fn a_write_that_named_no_pages_invalidates_every_older_reader() {
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_unknown();
        assert_eq!(
            w.wrote_any_since(mark, &[999 * P]),
            HostWriteVerdict::Unnamed,
            "a writer that named no pages must be reported as unnamed and not as \
             a write that landed in this window"
        );
        assert_eq!(
            w.wrote_any_since(w.epoch(), &[999 * P]),
            HostWriteVerdict::Quiet
        );
    }

    /// **There is no horizon.** A reader whose mark is arbitrarily old still
    /// gets an exact answer — this is the whole of what replacing the ring
    /// bought, and the ring answered `Aged` here 6 294 times a driven macos-26
    /// boot.
    #[test]
    fn a_reader_older_than_any_number_of_writes_is_still_answered_exactly() {
        let mut w = HostWrites::default();
        let stale = w.epoch();
        for i in 0..4096u64 {
            w.note_pages(vec![(100 + i) * P]);
        }
        assert_eq!(
            w.wrote_any_since(stale, &[3 * P]),
            HostWriteVerdict::Quiet,
            "four thousand writes to other pages do not touch this window"
        );
        assert_eq!(
            w.wrote_any_since(stale, &[101 * P]),
            HostWriteVerdict::Overlap,
            "and a write that did land here is not forgotten behind the later ones"
        );
    }

    /// A compositor re-Storing one surface every frame updates one entry, so the
    /// record does not grow with the frame rate. It was the ring's `supersede`
    /// that used to carry this; here it is what a map does anyway.
    #[test]
    fn a_write_repeated_every_frame_occupies_one_entry() {
        let mut w = HostWrites::default();
        for _ in 0..10_000 {
            w.note_pages(vec![4 * P, 5 * P]);
        }
        assert_eq!(w.pages.recorded_pages(), 2);
        assert_eq!(
            w.wrote_any_since(w.epoch(), &[4 * P]),
            HostWriteVerdict::Quiet
        );
    }

    /// Chunks stay exact across their boundary and never impose a record
    /// horizon. A write on the far side cannot displace an older page.
    #[test]
    fn chunk_boundaries_preserve_every_older_page() {
        let mut w = HostWrites::default();
        let mark = w.epoch();
        let far = (EPOCHS_PER_CHUNK as u64 * 4096) + 3 * P;
        w.note_pages(vec![3 * P, far]);
        assert_eq!(w.wrote_any_since(mark, &[3 * P]), HostWriteVerdict::Overlap);
        assert_eq!(w.wrote_any_since(mark, &[far]), HostWriteVerdict::Overlap);
        assert_eq!(w.pages.recorded_pages(), 2);
    }

    #[test]
    fn a_retained_full_chunk_is_one_epoch_then_partial_writes_stay_page_exact() {
        let pages: std::sync::Arc<[u64]> = (0..EPOCHS_PER_CHUNK as u64)
            .map(|page| page * P)
            .collect::<Vec<_>>()
            .into();
        let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(pages, P)
            .expect("one contiguous allocation chunk");
        let mut w = HostWrites::default();
        let before = w.epoch();
        w.note_footprint(&footprint);
        let after_full = w.epoch();
        assert_eq!(w.pages.recorded_pages(), EPOCHS_PER_CHUNK);
        assert_eq!(
            w.wrote_any_since(before, &[(EPOCHS_PER_CHUNK as u64 - 1) * P]),
            HostWriteVerdict::Overlap
        );
        assert_eq!(
            w.wrote_any_since(after_full, &[7 * P]),
            HostWriteVerdict::Quiet
        );

        w.note_pages(vec![7 * P]);
        assert_eq!(
            w.wrote_any_since(after_full, &[7 * P]),
            HostWriteVerdict::Overlap
        );
        assert_eq!(
            w.wrote_any_since(after_full, &[8 * P]),
            HostWriteVerdict::Quiet,
            "a later partial write must not refresh the chunk-wide epoch"
        );
    }

    #[test]
    fn a_retained_footprint_with_other_page_geometry_fails_closed() {
        let pages: std::sync::Arc<[u64]> = [0x4000].into();
        let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(pages, 1 << 14)
            .expect("arm64 footprint");
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_footprint(&footprint);
        assert_eq!(
            w.wrote_any_since(mark, &[0x9000]),
            HostWriteVerdict::Unnamed
        );
    }

    /// The page index follows the guest geometry carried by `DeviceState`.
    /// Arm64's four-times-wider pages remain two records here, not eight sparse
    /// x86-sized cells that merely happen to give the same verdict.
    #[test]
    fn arm64_pages_use_arm64_geometry() {
        const ARM_PAGE: u64 = 1 << crate::model::PAGE_SHIFT_ARM64E;
        let mut w = HostWrites::new(crate::model::PAGE_SHIFT_ARM64E);
        let mark = w.epoch();
        w.note_pages(vec![3 * ARM_PAGE, 4 * ARM_PAGE]);
        assert_eq!(w.pages.recorded_pages(), 2);
        assert_eq!(
            w.wrote_any_since(mark, &[4 * ARM_PAGE]),
            HostWriteVerdict::Overlap
        );
        assert_eq!(
            w.wrote_any_since(mark, &[5 * ARM_PAGE]),
            HostWriteVerdict::Quiet
        );
    }

    /// A mapping re-pointed after the write is answered exactly, because the
    /// pages were captured when the write happened.
    ///
    /// The ring resolved a mapping-named write at *read* time and had to answer
    /// `Unresolvable` once the mapping moved — a correct refusal, but a refusal,
    /// and each one cost a re-gather. Page 7 was written and still reads
    /// `Overlap`, page 3 never was and reads `Quiet`, whatever the mapping
    /// points at now.
    #[test]
    fn a_mapping_re_pointed_after_the_write_is_still_answered_by_the_pages_it_wrote() {
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_mapping(Some(&[7 * P]));
        assert_eq!(w.wrote_any_since(mark, &[7 * P]), HostWriteVerdict::Overlap);
        assert_eq!(w.wrote_any_since(mark, &[8 * P]), HostWriteVerdict::Quiet);
        // Whatever the mapping is re-pointed at, this write's pages do not move.
        w.note_mapping(Some(&[8 * P]));
        assert_eq!(w.wrote_any_since(mark, &[7 * P]), HostWriteVerdict::Overlap);
        assert_eq!(w.wrote_any_since(mark, &[3 * P]), HostWriteVerdict::Quiet);
    }

    /// A writer that cannot name every page of its mapping must not have the
    /// pages it *could* name mistaken for the whole write.
    #[test]
    fn a_mapping_write_that_cannot_name_all_its_pages_invalidates_everything_older() {
        let mut w = HostWrites::default();
        let mark = w.epoch();
        w.note_mapping(None);
        assert_eq!(w.wrote_any_since(mark, &[3 * P]), HostWriteVerdict::Unnamed);
    }

    /// Only `Quiet` is a permission. A new variant must not be readable as one.
    #[test]
    fn every_verdict_but_quiet_counts_as_written() {
        for v in [HostWriteVerdict::Overlap, HostWriteVerdict::Unnamed] {
            assert!(v.wrote(), "{v:?}");
        }
        assert!(!HostWriteVerdict::Quiet.wrote());
    }
}
