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
//! A per-page last-write epoch. `wrote_any_since` is one lookup per page the
//! reader names, and it has **no horizon**: however long ago the reader last
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
//! Everything still fails closed. A write that cannot name its pages, and a
//! reader older than the one event that can clear this record, both answer
//! "assume written".

/// Why [`HostWrites::wrote_any_since`] could not call a window quiet.
///
/// Causes used to share one `true`, and only the first of them means the bytes
/// under the reader's pages actually moved. The others are this type's
/// fail-closed rule firing — a correct answer to "can you rule this out", and a
/// very different thing to report. A boot that reads mostly `Unnamed` says its
/// writers are not naming their pages; one that reads any `Forgotten` at all
/// says [`PageEpochs::PAGES`] is too small for the write rate; one that reads
/// mostly `Overlap` says the device really is writing the windows it samples,
/// and only then is there nothing to reclaim here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostWriteVerdict {
    /// Nothing this device recorded touched these pages.
    Quiet,
    /// A recorded write names one of these pages — the bytes moved.
    Overlap,
    /// A writer that could not say which pages it landed in, so every reader
    /// older than it must assume its own.
    Unnamed,
    /// The record was cleared for reaching its bound and no longer holds the
    /// writes this reader is asking about.
    ///
    /// The last remnant of the ring's horizon, and the only one left: it can now
    /// fire only on a whole-record reset rather than on every ask that reaches
    /// past 64 writes. Six rails read it **zero**. A non-zero reading is an
    /// alarm on the bound, not a normal cost.
    Forgotten,
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
            Self::Forgotten => "gw_hw_forgotten",
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
/// # Fail-closed, in the two ways that are left
///
/// A write that cannot name its pages sets [`Self::unnamed_at`], and every
/// reader older than it is refused. Reaching [`Self::PAGES`] clears the whole
/// map and sets [`Self::reset_at`], and every reader older than *that* is
/// refused. Shedding entries one page at a time would answer `Quiet` for a page
/// that was written, which is the direction that loses a frame, so it is not
/// done.
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
#[derive(Default, Debug)]
struct PageEpochs {
    /// Last epoch at which this device wrote each page it has ever written.
    last_write: std::collections::HashMap<u64, u64>,
    /// Newest write that could not name its pages. Fail-closed for any reader
    /// older than it, exactly as the ring's `Unknown` entry is.
    unnamed_at: u64,
    /// Newest epoch at which the map was cleared for reaching [`Self::PAGES`].
    ///
    /// Clearing is the only sound way to shed entries: forgetting *one* page
    /// would answer `Quiet` for a page that was written, which is the direction
    /// that loses frames. So the whole map goes and every reader older than the
    /// reset is refused, which is the ring's own failure mode confined to a rare
    /// event instead of a per-ask one.
    reset_at: u64,
}

impl PageEpochs {
    /// The most pages the map will hold before it resets.
    ///
    /// A 1080p scanout is ~2 000 pages, so this is a few hundred distinct
    /// full-screen surfaces' worth. Sized to make `hw_shadow_reset` a rare event
    /// rather than to fit a measurement — if it is not rare, that counter says
    /// so and the shadow's own verdict degrades to the ring's.
    const PAGES: usize = 1 << 20;

    fn note_pages(&mut self, pages: &[u64], epoch: u64) {
        if self.last_write.len().saturating_add(pages.len()) > Self::PAGES {
            self.last_write.clear();
            self.reset_at = epoch;
            crate::runtime::drain::note_store_route("hw_shadow_reset");
        }
        for &p in pages {
            self.last_write.insert(p, epoch);
        }
    }

    fn note_unknown(&mut self, epoch: u64) {
        self.unnamed_at = epoch;
    }

    /// The verdict this map would give, in the ring's own vocabulary.
    fn verdict(&self, since: u64, pages: &[u64]) -> HostWriteVerdict {
        if since < self.unnamed_at {
            return HostWriteVerdict::Unnamed;
        }
        if since < self.reset_at {
            return HostWriteVerdict::Forgotten;
        }
        if pages
            .iter()
            .any(|p| self.last_write.get(p).is_some_and(|&e| e > since))
        {
            return HostWriteVerdict::Overlap;
        }
        HostWriteVerdict::Quiet
    }
}

/// Which guest pages this device has written, and when.
#[derive(Default, Debug)]
pub struct HostWrites {
    /// Monotonic stamp; a reader records the value current at its own read and
    /// asks later whether anything newer touched its pages. Never 0 once any
    /// write has happened, so 0 is usable as "never looked".
    epoch: u64,
    /// The record itself. See [`PageEpochs`] for why it is page-keyed and what
    /// watches it.
    pages: PageEpochs,
}

impl HostWrites {
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
            Some(p) => self.pages.note_pages(p, self.epoch),
            None => self.pages.note_unknown(self.epoch),
        }
    }

    /// Record a write covering exactly `pages` (page-aligned guest addresses).
    pub fn note_pages(&mut self, pages: Vec<u64>) {
        self.epoch = self.epoch.wrapping_add(1);
        self.pages.note_pages(&pages, self.epoch);
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
    /// cannot decide answers as written, and names which of the two
    /// undecidables it was: a write that named no pages, or a record cleared for
    /// reaching its bound. Only [`HostWriteVerdict::Overlap`] says a recorded
    /// write actually covers one of `pages`.
    ///
    /// One lookup per page, and no walk — there is nothing to walk back to. The
    /// ring this replaced took `&DeviceState` to resolve a mapping-named write
    /// at read time; the pages are captured when the write happens now, so no
    /// state is needed and a mapping re-pointed afterwards cannot make the
    /// answer wrong.
    pub fn wrote_any_since(&self, since: u64, pages: &[u64]) -> HostWriteVerdict {
        self.pages.verdict(since, pages)
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
        assert_eq!(w.wrote_any_since(mark, &[3 * P, 4 * P]), HostWriteVerdict::Quiet);
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
        assert_eq!(w.wrote_any_since(after, &[4 * P]), HostWriteVerdict::Overlap);
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
        assert_eq!(w.wrote_any_since(w.epoch(), &[999 * P]), HostWriteVerdict::Quiet);
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
        assert_eq!(w.pages.last_write.len(), 2);
        assert_eq!(w.wrote_any_since(w.epoch(), &[4 * P]), HostWriteVerdict::Quiet);
    }

    /// Shedding entries one page at a time would answer `Quiet` for a page that
    /// was written, which is the direction that loses a frame. The map clears
    /// whole and refuses everything older instead.
    #[test]
    fn a_full_record_clears_whole_and_refuses_every_older_reader() {
        let mut s = PageEpochs::default();
        let pages: Vec<u64> = (0..PageEpochs::PAGES as u64).map(|i| i * P).collect();
        s.note_pages(&pages, 1);
        assert_eq!(s.verdict(0, &[3 * P]), HostWriteVerdict::Overlap);
        s.note_pages(&[u64::from(u32::MAX) * P], 2);
        assert_eq!(s.verdict(0, &[3 * P]), HostWriteVerdict::Forgotten);
        assert_eq!(s.verdict(2, &[3 * P]), HostWriteVerdict::Quiet);
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
        for v in [
            HostWriteVerdict::Overlap,
            HostWriteVerdict::Unnamed,
            HostWriteVerdict::Forgotten,
        ] {
            assert!(v.wrote(), "{v:?}");
        }
        assert!(!HostWriteVerdict::Quiet.wrote());
    }
}
