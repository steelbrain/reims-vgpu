//! The decoded picture buffer, for frames: reference marking (8.2.5),
//! and reference list initialisation (8.2.4.2) and modification
//! (8.2.4.3). The bumping process of output order (C.4.5.3) is the one
//! HEVC's DPB shares ([`crate::codec::dpb`]).

use std::sync::Arc;

use super::Error;
use super::slice::{ListModification, MemoryManagement, RefPicMarking};
use crate::codec::dpb::Stored;

/// How a frame is marked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Marking {
    Unused,
    ShortTerm,
    /// With its `LongTermFrameIdx`.
    LongTerm(u32),
}

/// A frame in the DPB.
pub(crate) struct Entry<P, O> {
    /// The decoded frame; `None` for a frame the gap process of 8.2.5.2
    /// inferred, which has no samples.
    pub(crate) picture: Option<Arc<P>>,
    pub(crate) frame_num: u32,
    /// `FrameNumWrap` (8-27), as of the picture being decoded.
    pub(crate) frame_num_wrap: i32,
    pub(crate) top: i32,
    pub(crate) bottom: i32,
    pub(crate) marking: Marking,
    /// What to emit when the frame is output; `Some` while it is "needed
    /// for output".
    pub(crate) pending: Option<O>,
}

impl<P, O> Entry<P, O> {
    /// `PicOrderCnt()` of the frame (8-1).
    pub(crate) fn poc(&self) -> i32 {
        self.top.min(self.bottom)
    }

    fn is_short_term(&self) -> bool {
        self.marking == Marking::ShortTerm
    }

    fn long_term_idx(&self) -> Option<u32> {
        match self.marking {
            Marking::LongTerm(idx) => Some(idx),
            _ => None,
        }
    }
}

impl<P, O> Stored for Entry<P, O> {
    type Output = O;

    fn order_count(&self) -> i32 {
        self.poc()
    }

    fn is_reference(&self) -> bool {
        self.marking != Marking::Unused
    }

    fn pending(&self) -> Option<&O> {
        self.pending.as_ref()
    }

    fn take_pending(&mut self) -> Option<O> {
        self.pending.take()
    }
}

/// The DPB: reference frames, and in display order the frames waiting
/// for output. `O` is what the output of a frame emits.
pub(crate) type Dpb<P, O> = crate::codec::dpb::Dpb<Entry<P, O>>;

/// What marking made of the current picture.
pub(crate) struct CurrentMarking {
    pub(crate) marking: Marking,
    /// Whether operation 5 ran.
    pub(crate) unmark_all: bool,
}

impl<P, O> Dpb<P, O> {
    /// 8.2.4.1 for frames: `FrameNumWrap` relative to `frame_num` (8-27).
    pub(crate) fn update_frame_num_wrap(
        &mut self,
        frame_num: u32,
        max_frame_num: u32,
    ) -> Result<(), Error> {
        for e in &mut self.entries {
            let wrap = if e.frame_num > frame_num {
                i64::from(e.frame_num).checked_sub(i64::from(max_frame_num))
            } else {
                Some(i64::from(e.frame_num))
            };
            e.frame_num_wrap = wrap
                .and_then(|wrap| i32::try_from(wrap).ok())
                .ok_or(Error::Overflow("FrameNumWrap"))?;
        }
        Ok(())
    }

    fn reference_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_reference()).count()
    }

    /// Unmarks the short-term frame with the smallest `FrameNumWrap`.
    fn unmark_oldest_short_term(&mut self) -> bool {
        let oldest = self
            .entries
            .iter_mut()
            .filter(|e| e.is_short_term())
            .min_by_key(|e| e.frame_num_wrap);
        match oldest {
            Some(e) => {
                e.marking = Marking::Unused;
                true
            }
            None => false,
        }
    }

    /// The sliding window (8.2.5.3), before a new short-term frame joins
    /// a DPB that holds `max(max_num_ref_frames, 1)` references.
    pub(crate) fn sliding_window(&mut self, max_num_ref_frames: u8) {
        let max = usize::from(max_num_ref_frames.max(1));
        while self.reference_count() >= max {
            if !self.unmark_oldest_short_term() {
                break;
            }
        }
    }

    /// Marks references after the current picture is decoded (8.2.5.1),
    /// and says how the current picture is to be marked. `frame_num` is
    /// its `CurrPicNum`.
    pub(crate) fn mark(
        &mut self,
        idr: bool,
        marking: &RefPicMarking,
        frame_num: u32,
        max_frame_num: u32,
        max_num_ref_frames: u8,
    ) -> Result<CurrentMarking, Error> {
        let mut current = Marking::ShortTerm;
        let mut unmark_all = false;
        if idr {
            for e in &mut self.entries {
                e.marking = Marking::Unused;
            }
            if marking.long_term_reference_flag {
                current = Marking::LongTerm(0);
            }
        } else if let Some(ops) = &marking.adaptive {
            for &op in ops {
                match op {
                    MemoryManagement::UnmarkShortTerm(diff) => {
                        let pic_num = pic_num_x(frame_num, diff)?;
                        if let Some(e) = self
                            .entries
                            .iter_mut()
                            .find(|e| e.is_short_term() && i64::from(e.frame_num_wrap) == pic_num)
                        {
                            e.marking = Marking::Unused;
                        }
                    }
                    MemoryManagement::UnmarkLongTerm(long_term_pic_num) => {
                        if let Some(e) = self
                            .entries
                            .iter_mut()
                            .find(|e| e.long_term_idx() == Some(long_term_pic_num))
                        {
                            e.marking = Marking::Unused;
                        }
                    }
                    MemoryManagement::ShortTermToLongTerm(diff, idx) => {
                        let pic_num = pic_num_x(frame_num, diff)?;
                        let target = self.entries.iter().position(|e| {
                            e.is_short_term() && i64::from(e.frame_num_wrap) == pic_num
                        });
                        if let Some(target) = target {
                            for (i, e) in self.entries.iter_mut().enumerate() {
                                if i == target {
                                    e.marking = Marking::LongTerm(idx);
                                } else if e.long_term_idx() == Some(idx) {
                                    e.marking = Marking::Unused;
                                }
                            }
                        }
                    }
                    MemoryManagement::MaxLongTermFrameIdx(plus1) => {
                        // MaxLongTermFrameIdx: None is "no long-term frame
                        // indices". Nothing but this operation reads it:
                        // operations 3 and 6 are the stream's to keep
                        // within it (7.4.3.3).
                        let max = plus1.checked_sub(1);
                        for e in &mut self.entries {
                            if let Some(idx) = e.long_term_idx()
                                && max.is_none_or(|max| idx > max)
                            {
                                e.marking = Marking::Unused;
                            }
                        }
                    }
                    MemoryManagement::UnmarkAll => {
                        for e in &mut self.entries {
                            e.marking = Marking::Unused;
                        }
                        unmark_all = true;
                    }
                    MemoryManagement::CurrentToLongTerm(idx) => {
                        for e in &mut self.entries {
                            if e.long_term_idx() == Some(idx) {
                                e.marking = Marking::Unused;
                            }
                        }
                        current = Marking::LongTerm(idx);
                    }
                }
            }
        } else {
            self.update_frame_num_wrap(frame_num, max_frame_num)?;
            self.sliding_window(max_num_ref_frames);
        }
        // A stream that keeps more references than max_num_ref_frames
        // breaks 7.4.3.3; the oldest short-term frames go, so that the
        // DPB stays within the size the surfaces were made for, the
        // current frame among them.
        let max = usize::from(max_num_ref_frames.max(1));
        while self.reference_count() >= max {
            if !self.unmark_oldest_short_term() {
                break;
            }
        }
        Ok(CurrentMarking {
            marking: current,
            unmark_all,
        })
    }

    /// The indices of the entries `keep` keeps, in the order `key` sorts
    /// them, ties in the DPB's order.
    fn indices_by<K: Ord>(
        &self,
        keep: impl Fn(&Entry<P, O>) -> bool,
        key: impl Fn(&Entry<P, O>) -> K,
    ) -> Vec<usize> {
        let mut found: Vec<(usize, &Entry<P, O>)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| keep(e))
            .collect();
        found.sort_by_key(|(_, e)| key(e));
        found.into_iter().map(|(i, _)| i).collect()
    }

    fn long_terms(&self) -> Vec<usize> {
        self.indices_by(|e| e.long_term_idx().is_some(), Entry::long_term_idx)
    }

    /// The initial list of a P or SP frame (8.2.4.2.1): short-term frames
    /// by descending `PicNum`, then long-term by ascending
    /// `LongTermPicNum`.
    pub(crate) fn initial_list_p(&self) -> Vec<usize> {
        let mut list = self.indices_by(Entry::is_short_term, |e| {
            std::cmp::Reverse(e.frame_num_wrap)
        });
        list.extend(self.long_terms());
        list
    }

    /// The initial lists of a B frame (8.2.4.2.3), around the current
    /// frame's `PicOrderCnt`. Inferred frames have no order count, and
    /// are left out.
    pub(crate) fn initial_lists_b(&self, poc: i32) -> [Vec<usize>; 2] {
        // The short-term frames with samples, by ascending order count.
        let mut existing: Vec<(usize, i32)> = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.is_short_term() && e.picture.is_some())
            .map(|(i, e)| (i, e.poc()))
            .collect();
        existing.sort_by_key(|&(_, frame_poc)| frame_poc);
        let before: Vec<usize> = existing
            .iter()
            .rev()
            .filter(|&&(_, frame_poc)| frame_poc < poc)
            .map(|&(i, _)| i)
            .collect();
        let after: Vec<usize> = existing
            .iter()
            .filter(|&&(_, frame_poc)| frame_poc > poc)
            .map(|&(i, _)| i)
            .collect();
        let long = self.long_terms();
        let mut l0 = before.clone();
        l0.extend(&after);
        l0.extend(&long);
        let mut l1 = after;
        l1.extend(&before);
        l1.extend(&long);
        if l1.len() > 1 && l0 == l1 {
            l1.swap(0, 1);
        }
        [l0, l1]
    }

    /// Truncates or pads an initial list to `num_active` entries and
    /// applies the slice's modifications (8.2.4.3) for a frame whose
    /// `CurrPicNum` is `frame_num`.
    pub(crate) fn modify_list(
        &self,
        initial: &[usize],
        ops: &[ListModification],
        num_active: usize,
        frame_num: u32,
        max_frame_num: u32,
    ) -> Result<Vec<Option<usize>>, Error> {
        // num_active entries of the initial list, padded with "no
        // reference", and one more, which (8-37) and (8-38) shift into.
        let slots = num_active
            .checked_add(1)
            .ok_or(Error::Overflow("num_ref_idx_lX_active_minus1 + 2"))?;
        let mut list: Vec<Option<usize>> =
            (0..num_active).map(|i| initial.get(i).copied()).collect();
        list.push(None);
        // CurrPicNum, and MaxPicNum, which is MaxFrameNum for frames
        // (7.4.3).
        let curr = i64::from(frame_num);
        let max = i64::from(max_frame_num);
        let overflow = Error::Overflow("picNumLXNoWrap");
        let mut pred = curr;
        let mut ref_idx = 0usize;
        for &op in ops {
            if ref_idx >= num_active {
                return Err(Error::OutOfRange("ref_pic_list_modification"));
            }
            let target = match op {
                ListModification::ShortTermSubtract(diff)
                | ListModification::ShortTermAdd(diff) => {
                    let abs_diff = i64::from(diff).checked_add(1).ok_or(overflow.clone())?;
                    if abs_diff > max {
                        return Err(Error::OutOfRange("abs_diff_pic_num_minus1"));
                    }
                    // (8-34) and (8-35): the prediction moves by
                    // abs_diff_pic_num and comes back into 0 to
                    // MaxPicNum − 1 by the standard's own wrap.
                    let moved = if matches!(op, ListModification::ShortTermSubtract(_)) {
                        pred.checked_sub(abs_diff)
                    } else {
                        pred.checked_add(abs_diff)
                    };
                    let no_wrap = moved
                        .and_then(|moved| {
                            if moved < 0 {
                                moved.checked_add(max)
                            } else if moved >= max {
                                moved.checked_sub(max)
                            } else {
                                Some(moved)
                            }
                        })
                        .ok_or(overflow.clone())?;
                    pred = no_wrap;
                    // (8-36).
                    let pic_num = if no_wrap > curr {
                        no_wrap.checked_sub(max).ok_or(overflow.clone())?
                    } else {
                        no_wrap
                    };
                    self.entries
                        .iter()
                        .position(|e| e.is_short_term() && i64::from(e.frame_num_wrap) == pic_num)
                        .ok_or(Error::MissingReference)?
                }
                ListModification::LongTerm(long_term_pic_num) => self
                    .entries
                    .iter()
                    .position(|e| e.long_term_idx() == Some(long_term_pic_num))
                    .ok_or(Error::MissingReference)?,
            };
            // (8-37) and (8-38): insert at ref_idx, then drop the later
            // copy of the same frame.
            list.insert(ref_idx, Some(target));
            list.truncate(slots);
            ref_idx = ref_idx.checked_add(1).ok_or(Error::Overflow("refIdxLX"))?;
            let mut later = list.split_off(ref_idx);
            later.retain(|&entry| entry != Some(target));
            list.extend(later);
            list.resize(slots, None);
        }
        list.truncate(num_active);
        Ok(list)
    }
}

/// `picNumX` (8-39).
fn pic_num_x(curr_pic_num: u32, difference_of_pic_nums_minus1: u32) -> Result<i64, Error> {
    i64::from(difference_of_pic_nums_minus1)
        .checked_add(1)
        .and_then(|difference| i64::from(curr_pic_num).checked_sub(difference))
        .ok_or(Error::Overflow("picNumX"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(frame_num: u32, poc: i32, marking: Marking) -> Entry<(), u32> {
        Entry {
            picture: Some(Arc::new(())),
            frame_num,
            frame_num_wrap: i32::try_from(frame_num).unwrap_or(0),
            top: poc,
            bottom: poc,
            marking,
            pending: None,
        }
    }

    fn dpb(entries: Vec<Entry<(), u32>>) -> Dpb<(), u32> {
        Dpb { entries }
    }

    #[test]
    fn p_lists_put_recent_short_terms_first_then_long_terms() {
        let d = dpb(vec![
            frame(1, 2, Marking::ShortTerm),
            frame(3, 6, Marking::ShortTerm),
            frame(0, 0, Marking::LongTerm(1)),
            frame(2, 4, Marking::ShortTerm),
            frame(5, 9, Marking::LongTerm(0)),
        ]);
        assert_eq!(d.initial_list_p(), [1, 3, 0, 4, 2]);
    }

    #[test]
    fn b_lists_split_at_the_current_order_count() {
        let d = dpb(vec![
            frame(0, 0, Marking::ShortTerm),
            frame(1, 8, Marking::ShortTerm),
            frame(2, 4, Marking::ShortTerm),
            frame(3, 12, Marking::ShortTerm),
        ]);
        let [l0, l1] = d.initial_lists_b(6);
        assert_eq!(l0, [2, 0, 1, 3]);
        assert_eq!(l1, [1, 3, 2, 0]);
    }

    #[test]
    fn identical_b_lists_swap_the_first_two_of_list_1() {
        let d = dpb(vec![
            frame(0, 0, Marking::ShortTerm),
            frame(1, 4, Marking::ShortTerm),
        ]);
        let [l0, l1] = d.initial_lists_b(8);
        assert_eq!(l0, [1, 0]);
        assert_eq!(l1, [0, 1]);
    }

    #[test]
    fn the_sliding_window_drops_the_smallest_frame_num_wrap() -> Result<(), Error> {
        let mut d = dpb(vec![
            frame(14, 0, Marking::ShortTerm),
            frame(15, 2, Marking::ShortTerm),
            frame(0, 4, Marking::ShortTerm),
        ]);
        d.update_frame_num_wrap(1, 16)?;
        d.sliding_window(3);
        assert_eq!(d.entries[0].marking, Marking::Unused);
        assert_eq!(d.entries[1].marking, Marking::ShortTerm);
        assert_eq!(d.entries[2].marking, Marking::ShortTerm);
        Ok(())
    }

    #[test]
    fn mmco_ops_unmark_and_convert_by_pic_num() -> Result<(), Error> {
        let mut d = dpb(vec![
            frame(1, 2, Marking::ShortTerm),
            frame(2, 4, Marking::ShortTerm),
            frame(3, 6, Marking::ShortTerm),
        ]);
        let marking = RefPicMarking {
            adaptive: Some(vec![
                // CurrPicNum 4: picNumX 3 is unmarked, 2 goes long-term 0.
                MemoryManagement::UnmarkShortTerm(0),
                MemoryManagement::MaxLongTermFrameIdx(1),
                MemoryManagement::ShortTermToLongTerm(1, 0),
            ]),
            ..RefPicMarking::default()
        };
        let current = d.mark(false, &marking, 4, 16, 4)?;
        assert_eq!(current.marking, Marking::ShortTerm);
        assert_eq!(d.entries[0].marking, Marking::ShortTerm);
        assert_eq!(d.entries[1].marking, Marking::LongTerm(0));
        assert_eq!(d.entries[2].marking, Marking::Unused);
        Ok(())
    }

    #[test]
    fn mmco_6_takes_the_index_from_another_frame() -> Result<(), Error> {
        let mut d = dpb(vec![frame(1, 2, Marking::LongTerm(0))]);
        let marking = RefPicMarking {
            adaptive: Some(vec![MemoryManagement::CurrentToLongTerm(0)]),
            ..RefPicMarking::default()
        };
        let current = d.mark(false, &marking, 2, 16, 4)?;
        assert_eq!(current.marking, Marking::LongTerm(0));
        assert_eq!(d.entries[0].marking, Marking::Unused);
        Ok(())
    }

    #[test]
    fn mmco_4_with_zero_unmarks_every_long_term() -> Result<(), Error> {
        let mut d = dpb(vec![
            frame(1, 2, Marking::LongTerm(0)),
            frame(2, 4, Marking::LongTerm(1)),
            frame(3, 6, Marking::ShortTerm),
        ]);
        let marking = RefPicMarking {
            adaptive: Some(vec![MemoryManagement::MaxLongTermFrameIdx(0)]),
            ..RefPicMarking::default()
        };
        d.mark(false, &marking, 4, 16, 4)?;
        assert_eq!(d.entries[0].marking, Marking::Unused);
        assert_eq!(d.entries[1].marking, Marking::Unused);
        assert_eq!(d.entries[2].marking, Marking::ShortTerm);
        Ok(())
    }

    #[test]
    fn mmco_5_unmarks_everything() -> Result<(), Error> {
        let mut d = dpb(vec![
            frame(1, 2, Marking::LongTerm(0)),
            frame(2, 4, Marking::ShortTerm),
        ]);
        let marking = RefPicMarking {
            adaptive: Some(vec![MemoryManagement::UnmarkAll]),
            ..RefPicMarking::default()
        };
        let current = d.mark(false, &marking, 3, 16, 4)?;
        assert!(current.unmark_all);
        assert!(d.entries.iter().all(|e| e.marking == Marking::Unused));
        Ok(())
    }

    #[test]
    fn list_modification_moves_named_frames_to_the_front() -> Result<(), Error> {
        let d = dpb(vec![
            frame(1, 2, Marking::ShortTerm),
            frame(2, 4, Marking::ShortTerm),
            frame(3, 6, Marking::ShortTerm),
            frame(0, 0, Marking::LongTerm(0)),
        ]);
        let initial = d.initial_list_p();
        assert_eq!(initial, [2, 1, 0, 3]);
        // CurrPicNum 4: subtract 3 -> PicNum 1; then long-term 0; then add
        // 1 from 1 -> PicNum 2.
        let ops = [
            ListModification::ShortTermSubtract(2),
            ListModification::LongTerm(0),
            ListModification::ShortTermAdd(0),
        ];
        let list = d.modify_list(&initial, &ops, 4, 4, 16)?;
        assert_eq!(list, [Some(0), Some(3), Some(1), Some(2)]);
        Ok(())
    }

    #[test]
    fn list_modification_wraps_pic_nums() -> Result<(), Error> {
        let mut d = dpb(vec![
            frame(15, 0, Marking::ShortTerm),
            frame(0, 2, Marking::ShortTerm),
        ]);
        d.update_frame_num_wrap(1, 16)?;
        // CurrPicNum 1: subtract 2 -> -1, which is frame_num 15.
        let list = d.modify_list(
            &d.initial_list_p(),
            &[ListModification::ShortTermSubtract(1)],
            2,
            1,
            16,
        )?;
        assert_eq!(list, [Some(0), Some(1)]);
        Ok(())
    }

    #[test]
    fn short_lists_pad_with_no_reference() -> Result<(), Error> {
        let d = dpb(vec![frame(0, 0, Marking::ShortTerm)]);
        let list = d.modify_list(&d.initial_list_p(), &[], 3, 1, 16)?;
        assert_eq!(list, [Some(0), None, None]);
        Ok(())
    }

    #[test]
    fn bumping_outputs_frames_by_order_count_and_keeps_references() {
        let mut d = dpb(vec![
            frame(0, 4, Marking::ShortTerm),
            frame(1, 2, Marking::Unused),
            frame(2, 6, Marking::ShortTerm),
        ]);
        for (i, e) in d.entries.iter_mut().enumerate() {
            e.pending = Some(u32::try_from(i).unwrap_or(0));
        }
        assert_eq!(d.bump_all(), [1, 0, 2]);
        assert_eq!(d.len(), 2);
    }
}
