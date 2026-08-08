//! Maximal GPA-contiguous stretches of a page window, as pure arithmetic.
//!
//! A guest window arrives as an ordered list of page GPAs, and the consumers —
//! host-pointer imports, GPU gathers, packed views — all want the same
//! reduction: the maximal stretches where consecutive entries ascend by
//! exactly one page. What differs per consumer is what happens to a stretch
//! (import it, copy it, refuse the window), which is device policy and stays
//! with the device. Nothing here maps, binds or emits.

use alloc::vec::Vec;
use core::ops::Range;

/// One GPA-contiguous stretch of a window, carrying the byte sub-range of it
/// the request covers.
///
/// `pages` indexes the caller's window list; `start_offset` is the byte offset
/// into the stretch's first page where the covered bytes begin (non-zero only
/// for the window's first stretch), and `len` how many bytes of the request
/// the stretch carries. The caller turns `pages` into whatever its rail binds
/// — a host-pointer import, a copy source — and adds `start_offset` to the
/// base it obtains.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowRun {
    pub pages: Range<usize>,
    pub start_offset: u64,
    pub len: u64,
}

/// Cut `window` into maximal GPA-contiguous stretches covering `span` bytes
/// from `head_off` into the first page.
///
/// `None` if the window runs out before `span` is met, or if `head_off`
/// reaches past the first page — a partial cover would hand the consumer a
/// short buffer, which is a wrong frame rather than a slow one. A zero `span`
/// is covered by no stretches.
pub fn coalesce_window(
    window: &[u64],
    page: u64,
    head_off: u64,
    span: u64,
) -> Option<Vec<WindowRun>> {
    if page == 0 || head_off >= page {
        return None;
    }
    let mut runs: Vec<WindowRun> = Vec::new();
    let mut consumed = 0u64;
    let mut i = 0usize;
    while i < window.len() && consumed < span {
        let mut j = i + 1;
        while j < window.len() && window[j] == window[i] + ((j - i) as u64) * page {
            j += 1;
        }
        let start_offset = if i == 0 { head_off } else { 0 };
        let avail = ((j - i) as u64) * page - start_offset;
        let len = avail.min(span - consumed);
        runs.push(WindowRun {
            pages: i..j,
            start_offset,
            len,
        });
        consumed += len;
        i = j;
    }
    (consumed == span).then_some(runs)
}

/// Maximal packed-contig runs in a page-GPA list, as index ranges.
///
/// Each run is a half-open index range `[start, end)` into `gpas` where
/// `gpas[i+1] == gpas[i] + page_size`. Callers multi-import one run at a time.
pub fn contig_page_runs(gpas: &[u64], page_size: u64) -> Vec<Range<usize>> {
    if gpas.is_empty() || page_size == 0 {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut start = 0usize;
    for i in 1..gpas.len() {
        if gpas[i] != gpas[i - 1].wrapping_add(page_size) {
            runs.push(start..i);
            start = i;
        }
    }
    runs.push(start..gpas.len());
    runs
}

/// How many runs [`contig_page_runs`] would return, without building them.
///
/// The packed-view pre-check and the lines that report a fragmented decline
/// both want only the count, and on this rail the fragmented answer is the
/// common one: a compositor mapping of 2040 pages in 511 runs is asked hundreds
/// of times a second, and materializing a 511-element `Vec` to read its `len()`
/// was the entire cost of the check. Counting the breaks is the same traversal
/// with no allocation.
pub fn contig_run_count(gpas: &[u64], page_size: u64) -> usize {
    if gpas.is_empty() || page_size == 0 {
        return 0;
    }
    1 + (1..gpas.len())
        .filter(|&i| gpas[i] != gpas[i - 1].wrapping_add(page_size))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const PAGE: u64 = 0x1000;

    /// The two run forms agree: the byte-carrying coalescer cuts the window at
    /// exactly the breaks the index form reports.
    #[test]
    fn the_byte_form_cuts_where_the_index_form_cuts() {
        let window = [
            0x10000, 0x11000, // one stretch of two
            0x20000, // alone
            0x40000, 0x41000, 0x42000, // three
        ];
        let idx = contig_page_runs(&window, PAGE);
        assert_eq!(idx, vec![0..2, 2..3, 3..6]);
        assert_eq!(contig_run_count(&window, PAGE), 3);
        let runs = coalesce_window(&window, PAGE, 0, 6 * PAGE).unwrap();
        assert_eq!(
            runs.iter().map(|r| r.pages.clone()).collect::<Vec<_>>(),
            idx
        );
    }

    /// The head offset is charged to the first stretch only, and the byte
    /// lengths total the span.
    #[test]
    fn the_head_offset_belongs_to_the_first_stretch_alone() {
        let window = [0x10000, 0x11000, 0x20000];
        let span = 2 * PAGE + 0x800 - 0x40;
        let runs = coalesce_window(&window, PAGE, 0x40, span).unwrap();
        assert_eq!(runs[0].start_offset, 0x40);
        assert_eq!(runs[0].len, 2 * PAGE - 0x40);
        assert_eq!(runs[1].start_offset, 0);
        assert_eq!(runs[1].len, 0x800);
        assert_eq!(runs.iter().map(|r| r.len).sum::<u64>(), span);
    }

    /// A window shorter than the span is a refusal, not a short cover.
    #[test]
    fn a_window_that_runs_out_refuses_rather_than_covering_less() {
        let window = [0x10000, 0x20000];
        assert!(coalesce_window(&window, PAGE, 0, 2 * PAGE + 1).is_none());
        assert!(coalesce_window(&window, PAGE, 0x40, 2 * PAGE).is_none());
        assert!(coalesce_window(&window, PAGE, 0, 2 * PAGE).is_some());
    }

    /// Degenerate geometry refuses: a zero page size cannot tile anything and
    /// a head offset past the first page names a byte outside the window.
    #[test]
    fn degenerate_geometry_is_refused() {
        let window = [0x10000];
        assert!(coalesce_window(&window, 0, 0, 1).is_none());
        assert!(coalesce_window(&window, PAGE, PAGE, 1).is_none());
        // A zero span needs no stretches at all.
        assert_eq!(coalesce_window(&window, PAGE, 0, 0).unwrap(), vec![]);
        assert_eq!(coalesce_window(&[], PAGE, 0, 0).unwrap(), vec![]);
        assert!(coalesce_window(&[], PAGE, 0, 1).is_none());
    }

    /// The index forms answer empty inputs with empty answers.
    #[test]
    fn index_runs_of_nothing_are_nothing() {
        assert!(contig_page_runs(&[], PAGE).is_empty());
        assert_eq!(contig_run_count(&[], PAGE), 0);
        assert!(contig_page_runs(&[0x1000], 0).is_empty());
        assert_eq!(contig_run_count(&[0x1000], 0), 0);
        assert_eq!(contig_page_runs(&[0x1000], PAGE), vec![0..1]);
        assert_eq!(contig_run_count(&[0x1000], PAGE), 1);
    }
}
