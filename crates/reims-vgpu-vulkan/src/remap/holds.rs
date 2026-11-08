//! The host pages runtime imports hold, and the unmaps that wait for them ([`super::UnmapHold`]):
//! the bookkeeping alone, which makes no call (`super`'s docs, "Unmaps an import waits for").

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::ops::Range;

/// The live holds and the unmaps waiting for them, each a range of whole pages. A leaf lock
/// guards it: nothing is called under it.
#[derive(Debug)]
pub(super) struct Holds {
    /// The next hold's number.
    next: u64,
    /// Each live hold's pages, by their first byte and the hold's number, to their end.
    held: BTreeMap<(u64, u64), u64>,
    /// How many live holds took each length in bytes: the longest bounds how far below a range a
    /// live hold that meets it starts ([`Holds::meets`]), and falls again as such holds go.
    lengths: BTreeMap<u64, usize>,
    /// The unmaps that wait, no two sharing a page: each made whole once no hold meets it.
    waiting: Vec<Range<u64>>,
    /// The unmaps released holds are making now, taken out of [`Self::waiting`]: an overwrite
    /// that meets one waits until it is made (`super::UnmapHold`'s drop).
    unmapping: Vec<Range<u64>>,
}

impl Holds {
    /// None.
    pub(super) const fn new() -> Holds {
        Holds {
            next: 0,
            held: BTreeMap::new(),
            lengths: BTreeMap::new(),
            waiting: Vec::new(),
            unmapping: Vec::new(),
        }
    }

    /// A hold on `pages`: its number.
    pub(super) fn hold(&mut self, pages: Range<u64>) -> u64 {
        let id = self.next;
        self.next = self.next.wrapping_add(1);
        *(self.lengths)
            .entry(pages.end.saturating_sub(pages.start))
            .or_default() += 1;
        self.held.insert((pages.start, id), pages.end);
        id
    }

    /// The most bytes a live hold took.
    fn longest(&self) -> u64 {
        self.lengths
            .last_key_value()
            .map_or(0, |(&length, _)| length)
    }

    /// Whether a live hold meets `pages`: one that starts no further below them than the longest
    /// live hold's length.
    fn meets(&self, pages: &Range<u64>) -> bool {
        let from = pages.start.saturating_sub(self.longest());
        (self.held.range((from, 0)..(pages.end, 0)))
            .any(|(&(first, _), &end)| first < pages.end && pages.start < end)
    }

    /// `pages` show other pages from now on (an overwrite, or an unmap made now): what waited
    /// to be unmapped there no longer does.
    pub(super) fn replaced(&mut self, pages: &Range<u64>) {
        if self.waiting.is_empty() || pages.is_empty() {
            return;
        }
        let mut kept = Vec::with_capacity(self.waiting.len());
        for waiting in self.waiting.drain(..) {
            if waiting.end <= pages.start || pages.end <= waiting.start {
                kept.push(waiting);
                continue;
            }
            if waiting.start < pages.start {
                kept.push(waiting.start..pages.start);
            }
            if pages.end < waiting.end {
                kept.push(pages.end..waiting.end);
            }
        }
        self.waiting = kept;
    }

    /// An unmap of `pages`: `true` where a live hold meets them, and the unmap waits, whole, for
    /// the last hold that meets it; `false` where the caller makes it now.
    pub(super) fn unmap(&mut self, pages: Range<u64>) -> bool {
        self.replaced(&pages);
        if pages.is_empty() || !self.meets(&pages) {
            return false;
        }
        self.waiting.push(pages);
        true
    }

    /// Whether any unmap waits.
    pub(super) fn unmaps_wait(&self) -> bool {
        !self.waiting.is_empty()
    }

    /// Whether an unmap of a page of `pages` waits, or is being made.
    #[cfg(test)]
    pub(super) fn waits(&self, pages: &Range<u64>) -> bool {
        (self.waiting.iter()).any(|waiting| waiting.start < pages.end && pages.start < waiting.end)
            || self.unmapping_meets(pages)
    }

    /// `unmaps`, which [`Self::release`] answered, are being made from now on.
    pub(super) fn unmapping(&mut self, unmaps: &[Range<u64>]) {
        self.unmapping.extend_from_slice(unmaps);
    }

    /// `unmaps`, which [`Self::unmapping`] took, are made.
    pub(super) fn unmapped(&mut self, unmaps: &[Range<u64>]) {
        self.unmapping.retain(|making| !unmaps.contains(making));
    }

    /// Whether an unmap being made meets `pages`.
    pub(super) fn unmapping_meets(&self, pages: &Range<u64>) -> bool {
        (self.unmapping.iter()).any(|making| making.start < pages.end && pages.start < making.end)
    }

    /// The hold `id` on the pages from `first` goes: the unmaps no live hold meets any more,
    /// which the caller makes.
    pub(super) fn release(&mut self, first: u64, id: u64) -> Vec<Range<u64>> {
        if let Some(end) = self.held.remove(&(first, id))
            && let Entry::Occupied(mut held) = self.lengths.entry(end.saturating_sub(first))
        {
            *held.get_mut() -= 1;
            if *held.get() == 0 {
                held.remove();
            }
        }
        if self.waiting.is_empty() {
            return Vec::new();
        }
        let (free, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.waiting)
            .into_iter()
            .partition(|waiting| !self.meets(waiting));
        self.waiting = kept;
        free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 0x4000;

    /// `n` pages from page `at`.
    fn pages(at: u64, n: u64) -> Range<u64> {
        at * PAGE..(at + n) * PAGE
    }

    /// An unmap no hold meets is made now; one a hold meets waits, whole, until the last hold
    /// that meets it goes, and is made once
    #[test]
    fn an_unmap_waits_for_every_hold_that_meets_it() {
        let mut h = Holds::new();
        assert!(!h.unmap(pages(10, 2)), "no hold");
        let a = h.hold(pages(10, 2));
        let b = h.hold(pages(11, 4));
        assert!(!h.unmap(pages(20, 1)), "pages no hold meets");
        assert!(
            h.unmap(pages(8, 4)),
            "pages the first hold meets, with others"
        );
        assert_eq!(
            h.release(10 * PAGE, a),
            Vec::<Range<u64>>::new(),
            "b meets it"
        );
        assert_eq!(h.release(11 * PAGE, b), vec![pages(8, 4)]);
        assert_eq!(
            h.release(11 * PAGE, b),
            Vec::<Range<u64>>::new(),
            "made once"
        );
    }

    /// Two holds on the same pages each keep an unmap of them waiting
    #[test]
    fn holds_on_the_same_pages_count_apart() {
        let mut h = Holds::new();
        let a = h.hold(pages(4, 1));
        let b = h.hold(pages(4, 1));
        assert!(h.unmap(pages(4, 1)));
        assert!(h.release(4 * PAGE, b).is_empty());
        assert_eq!(h.release(4 * PAGE, a), vec![pages(4, 1)]);
    }

    /// A long hold far below a range still meets it
    #[test]
    fn a_long_hold_meets_pages_far_into_it() {
        let mut h = Holds::new();
        let long = h.hold(pages(0, 1000));
        let _short = h.hold(pages(2000, 1));
        assert!(h.unmap(pages(999, 1)));
        assert!(!h.unmap(pages(1000, 1)), "past its end");
        assert_eq!(h.release(0, long), vec![pages(999, 1)]);
    }

    /// A long hold, once released, no longer widens the search below a range: the longest live
    /// hold bounds it, whichever went before
    #[test]
    fn the_longest_hold_is_the_longest_that_lives() {
        let mut h = Holds::new();
        let long = h.hold(pages(0, 1000));
        let again = h.hold(pages(5000, 1000));
        let short = h.hold(pages(2000, 2));
        assert_eq!(h.longest(), 1000 * PAGE);
        assert!(h.release(0, long).is_empty());
        assert_eq!(h.longest(), 1000 * PAGE, "another as long lives");
        assert!(h.release(5000 * PAGE, again).is_empty());
        assert_eq!(h.longest(), 2 * PAGE);
        assert!(!h.unmap(pages(999, 1)), "the long hold went");
        assert!(h.unmap(pages(2001, 1)));
        assert_eq!(h.release(2000 * PAGE, short), vec![pages(2001, 1)]);
        assert_eq!(h.longest(), 0);
        assert!(h.lengths.is_empty());
    }

    /// Whether a live hold meets pages is what a walk of every live hold answers, over holds
    /// taken and released at random, of lengths from a page to many, the same pages held twice
    /// among them
    #[test]
    fn holds_meet_pages_as_a_walk_of_every_live_hold_does() {
        let mut seed = 0x2545_f491_u64;
        let mut next = move |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        let mut h = Holds::new();
        let mut live: Vec<(Range<u64>, u64)> = Vec::new();
        for step in 0..20_000 {
            if live.is_empty() || next(3) != 0 {
                let length = if next(50) == 0 {
                    1 + next(400)
                } else {
                    1 + next(4)
                };
                let held = pages(next(600), length);
                let id = h.hold(held.clone());
                live.push((held, id));
            } else {
                let (held, id) =
                    live.swap_remove(usize::try_from(next(live.len() as u64)).unwrap());
                h.release(held.start, id);
            }
            let asked = pages(next(700), 1 + next(8));
            let walked =
                (live.iter()).any(|(held, _)| held.start < asked.end && asked.start < held.end);
            assert_eq!(h.meets(&asked), walked, "step {step}: {asked:x?}");
            let longest = live.iter().map(|(held, _)| held.end - held.start).max();
            assert_eq!(h.longest(), longest.unwrap_or(0), "step {step}");
        }
    }

    /// An unmap a release answered meets pages while it is being made, and no more once made;
    /// those of another release stay
    #[test]
    fn a_released_unmap_meets_pages_until_it_is_made() {
        let mut h = Holds::new();
        let (a, b) = (h.hold(pages(0, 4)), h.hold(pages(10, 2)));
        assert!(h.unmap(pages(0, 4)) && h.unmap(pages(10, 2)));
        let first = h.release(0, a);
        assert_eq!(first, vec![pages(0, 4)]);
        h.unmapping(&first);
        let second = h.release(10 * PAGE, b);
        h.unmapping(&second);
        assert!(h.unmapping_meets(&pages(3, 2)) && h.waits(&pages(3, 1)));
        assert!(!h.unmapping_meets(&pages(4, 6)), "between them");
        assert!(!h.unmaps_wait(), "none waits: both are being made");
        h.unmapped(&first);
        assert!(!h.unmapping_meets(&pages(0, 4)) && !h.waits(&pages(0, 4)));
        assert!(
            h.unmapping_meets(&pages(11, 1)),
            "the other is still being made"
        );
        h.unmapped(&second);
        assert!(!h.unmapping_meets(&pages(0, 20)));
    }

    /// Pages mapped anew, or unmapped now, under a waiting unmap leave it: what of it they cover
    /// is never unmapped later, and the rest still waits
    #[test]
    fn pages_replaced_under_a_waiting_unmap_leave_it() {
        let mut h = Holds::new();
        let a = h.hold(pages(0, 8));
        assert!(h.unmap(pages(0, 8)));
        h.replaced(&pages(2, 2));
        assert!(h.unmap(pages(6, 1)), "waits again, alone");
        assert_eq!(
            h.release(0, a),
            vec![pages(0, 2), pages(4, 2), pages(7, 1), pages(6, 1)]
        );
    }
}
