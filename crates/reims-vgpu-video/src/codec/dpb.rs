//! What H.264's and HEVC's decoded picture buffers share: their pictures,
//! the ones among them waiting for output, and the "bumping" process that
//! outputs them, which is the same process in both standards (H.264
//! C.4.5.3, H.265 C.5.2.4): the waiting picture with the smallest order
//! count is output, and its buffer emptied when it is no longer a
//! reference.
//!
//! When to bump is each standard's own, and each decoder says it in its
//! own terms: H.264's C.4.4 and C.4.5 (`h264::decoder`), HEVC's C.5.2.2
//! and C.5.2.3 (`hevc::decoder`). How pictures are marked, and the lists
//! made of them, are each standard's `dpb` module's.

/// A picture as a decoded picture buffer stores it.
pub(crate) trait Stored {
    /// What the picture's output emits.
    type Output;

    /// The order count the bumping process outputs by: H.264's
    /// `PicOrderCnt()` of a frame, HEVC's `PicOrderCntVal`.
    fn order_count(&self) -> i32;

    /// Whether it is marked "used for reference", short-term or long-term.
    fn is_reference(&self) -> bool;

    /// Its output, held while it is "needed for output".
    fn pending(&self) -> Option<&Self::Output>;

    /// Takes its output: it is no longer needed for output.
    fn take_pending(&mut self) -> Option<Self::Output>;
}

/// A decoded picture buffer: its pictures, in the order they were stored.
pub(crate) struct Dpb<E> {
    pub(crate) entries: Vec<E>,
}

impl<E: Stored> Dpb<E> {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Empties every buffer, outputting nothing.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    /// The pictures held, for reference or for output.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The pictures waiting for output.
    pub(crate) fn pending_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.pending().is_some())
            .count()
    }

    /// The smallest order count of the pictures waiting for output.
    pub(crate) fn min_pending_poc(&self) -> Option<i32> {
        self.entries
            .iter()
            .filter(|e| e.pending().is_some())
            .map(Stored::order_count)
            .min()
    }

    /// Empties the buffers of the pictures that are neither references nor
    /// waiting for output.
    pub(crate) fn remove_unused(&mut self) {
        self.entries
            .retain(|e| e.is_reference() || e.pending().is_some());
    }

    /// The bumping process: outputs the waiting picture with the smallest
    /// order count, the first stored of those that tie, and empties its
    /// buffer when it is no longer a reference.
    pub(crate) fn bump(&mut self) -> Option<E::Output> {
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.pending().is_some())
            .min_by_key(|(_, e)| e.order_count())
            .map(|(i, _)| i)?;
        let entry = self.entries.get_mut(index)?;
        let output = entry.take_pending();
        if !entry.is_reference() {
            self.entries.remove(index);
        }
        output
    }

    /// Bumps while `more` holds of the buffer and a picture waits, each
    /// output into `outputs`.
    pub(crate) fn bump_while(
        &mut self,
        outputs: &mut Vec<E::Output>,
        more: impl Fn(&Self) -> bool,
    ) {
        while more(self) {
            let Some(output) = self.bump() else {
                break;
            };
            outputs.push(output);
        }
    }

    /// Bumps until nothing waits for output.
    pub(crate) fn bump_all(&mut self) -> Vec<E::Output> {
        let mut outputs = Vec::new();
        self.bump_while(&mut outputs, |_| true);
        outputs
    }

    /// Drops every waiting output without emitting it. The buffers stay
    /// until they are emptied.
    pub(crate) fn discard_pending(&mut self) {
        for entry in &mut self.entries {
            entry.take_pending();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture of an order count, a reference or not, waiting with its
    /// own number or not.
    struct Picture(i32, bool, Option<u32>);

    impl Stored for Picture {
        type Output = u32;

        fn order_count(&self) -> i32 {
            self.0
        }

        fn is_reference(&self) -> bool {
            self.1
        }

        fn pending(&self) -> Option<&u32> {
            self.2.as_ref()
        }

        fn take_pending(&mut self) -> Option<u32> {
            self.2.take()
        }
    }

    fn dpb(entries: Vec<Picture>) -> Dpb<Picture> {
        Dpb { entries }
    }

    #[test]
    fn bumping_outputs_the_smallest_order_count_first_and_keeps_references() {
        let mut d = dpb(vec![
            Picture(4, true, Some(0)),
            Picture(2, false, Some(1)),
            Picture(6, true, Some(2)),
            Picture(0, true, None),
        ]);
        assert_eq!(d.min_pending_poc(), Some(2));
        assert_eq!(d.bump(), Some(1));
        assert_eq!(d.len(), 3);
        assert_eq!(d.bump_all(), [0, 2]);
        assert_eq!((d.len(), d.pending_count()), (3, 0));
        assert_eq!(d.bump(), None);
    }

    #[test]
    fn ties_go_out_in_the_order_stored() {
        let mut d = dpb(vec![Picture(3, false, Some(0)), Picture(3, false, Some(1))]);
        assert_eq!(d.bump_all(), [0, 1]);
        assert!(d.is_empty());
    }

    #[test]
    fn bumping_stops_when_its_condition_does_or_nothing_waits() {
        let mut d = dpb(vec![
            Picture(1, false, Some(0)),
            Picture(2, false, Some(1)),
            Picture(3, false, Some(2)),
        ]);
        let mut out = Vec::new();
        d.bump_while(&mut out, |d| d.pending_count() > 1);
        assert_eq!(out, [0, 1]);
        d.bump_while(&mut out, |_| true);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn discarded_outputs_leave_unused_buffers_to_be_emptied() {
        let mut d = dpb(vec![Picture(0, false, Some(0)), Picture(1, true, Some(1))]);
        d.discard_pending();
        assert_eq!((d.len(), d.pending_count()), (2, 0));
        d.remove_unused();
        assert_eq!(d.len(), 1);
    }
}
