//! Each frame's picture type, references and DPB slot, and so the order
//! frames are encoded in. Plain data, planned with no GPU; both codecs'
//! encoders take it.
//!
//! Frames come in display order. IDR pictures fall where the codec crate's
//! [`KeyFrames`] places them: the first frame, a forced one, the first past
//! the IDR period, and the first whose order would pass its bound. Between
//! them, with `b_frames` above 0, up to that many frames wait as B
//! pictures for the next frame, which becomes the run's anchor and is
//! encoded before them: `I0 B1 B2 P3` is encoded `I0 P3 B1 B2`. An anchor is
//! an I picture where the intra period has passed since the last I or IDR
//! picture, else a P picture. B pictures still waiting when an IDR picture
//! comes, or at a flush, end their run as x264 ends one: the last becomes
//! the anchor, so that none predicts across the IDR picture.
//!
//! **References.** Anchors (IDR, I and P pictures) are references; B
//! pictures are not. The DPB holds at most `max_references` of them: once
//! it is full, an anchor drops the oldest as it is kept, H.264's sliding
//! window (8.2.5.3), which H.265's reference picture sets follow too. A P
//! picture predicts from the most recent `references`, most recent first; a
//! B picture from the nearest anchor before it (list 0) and the nearest
//! after it (list 1). Both are each list's default order (H.264 8.2.4.2,
//! H.265 8.3.4), so no list is modified. With B pictures the DPB holds two
//! references at least, so that a B picture's anchors are both there.
//!
//! **Slots.** Each picture is set up in the least DPB slot no reference
//! holds, which a reference keeps until it is dropped: `max_references + 1`
//! slots, the picture being encoded's among them.
//!
//! **Numbers.** Each IDR picture's `idr_pic_id` is one more than the last
//! one's. H.264's `frame_num` is 0 at an IDR picture, and one more than the
//! last reference's, modulo `MaxFrameNum`, at every other (7.4.3): B
//! pictures share theirs with the anchor after them.

use reims_vgpu_video_codec::encode::gop::KeyFrames;

/// When IDR and I pictures come, how many B pictures run between
/// references, and how many references a P picture predicts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GopConfig {
    /// At most this many frames from one IDR picture to the next; none
    /// for an IDR picture only at the first frame and where one is forced.
    pub idr_period: Option<u32>,
    /// At most this many frames from one I or IDR picture to the next I
    /// picture; none for no I pictures but the IDR pictures.
    pub intra_period: Option<u32>,
    /// Consecutive B pictures between references, where the profile and
    /// the device have B pictures; 0 for none.
    pub b_frames: u32,
    /// The references a P picture predicts from, the most recent: 1 at
    /// least, and at most what the device takes.
    pub references: u32,
}

impl Default for GopConfig {
    /// An IDR picture every 60 frames, no other I picture, no B picture,
    /// one reference.
    fn default() -> Self {
        Self {
            idr_period: Some(60),
            intra_period: None,
            b_frames: 0,
            references: 1,
        }
    }
}

/// A picture's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PictureKind {
    /// An IDR picture: decoding can start here.
    Idr,
    /// An I picture that is not IDR: later pictures may still predict
    /// across it.
    I,
    /// A P picture.
    P,
    /// A B picture, which nothing predicts from.
    B,
}

/// A reference picture, as the pictures that predict from it name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reference {
    /// Its number among the references, from 0 in encode order.
    pub(crate) number: u64,
    /// The DPB slot it is kept in.
    pub(crate) slot: u32,
    /// Its place in display order from its IDR picture.
    pub(crate) order: u32,
    /// H.264's `frame_num`.
    pub(crate) frame_num: u32,
    /// Its type: IDR, I or P.
    pub(crate) kind: PictureKind,
}

/// A frame with its place in the stream, in encode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Planned<F> {
    /// The frame, as it was given.
    pub(crate) frame: F,
    /// Its type.
    pub(crate) kind: PictureKind,
    /// Its place in display order from its IDR picture: what the order
    /// counts are made from.
    pub(crate) order: u32,
    /// H.264's `frame_num`.
    pub(crate) frame_num: u32,
    /// The last IDR picture's `idr_pic_id`: this one's, for an IDR
    /// picture.
    pub(crate) idr_pic_id: u16,
    /// Its number among the references, where it is one.
    pub(crate) reference: Option<u64>,
    /// The DPB slot it is set up in.
    pub(crate) setup_slot: u32,
    /// List 0, in its order: what a P or B picture predicts from.
    pub(crate) list0: Vec<Reference>,
    /// List 1, for a B picture.
    pub(crate) list1: Vec<Reference>,
    /// Every reference the DPB holds as it is encoded, oldest first: its
    /// lists' and those kept for later pictures.
    pub(crate) dpb: Vec<Reference>,
}

/// What the planner keeps to, the device's limits applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limits {
    /// References list 0 of a P picture holds.
    pub(crate) p_l0: u32,
    /// References the DPB holds: `p_l0`, and 2 at least with B pictures.
    pub(crate) max_references: u32,
    /// Consecutive B pictures.
    pub(crate) b_frames: u32,
    /// `log2(MaxFrameNum)`.
    pub(crate) log2_max_frame_num: u8,
}

/// Ours: the most B pictures in a run, each a frame held back.
pub const MAX_B_FRAMES: u32 = 7;

impl Limits {
    /// `config` within a device's limits: B pictures where `b_pictures`
    /// says the profile and device have them, at most [`MAX_B_FRAMES`]; the
    /// references at most `max_p_l0`, and a DPB of them and one more slot
    /// within `max_slots`.
    pub(crate) fn new(
        config: &GopConfig,
        b_pictures: bool,
        max_p_l0: u32,
        max_slots: u32,
        log2_max_frame_num: u8,
    ) -> Self {
        let b_frames = if b_pictures {
            config.b_frames.min(MAX_B_FRAMES)
        } else {
            0
        };
        let room = max_slots.saturating_sub(1).max(1);
        let p_l0 = config.references.clamp(1, max_p_l0.max(1)).min(room);
        let wanted = if b_frames > 0 { p_l0.max(2) } else { p_l0 };
        let max_references = wanted.min(room);
        // Without room for both anchors, no B picture.
        let b_frames = if max_references < 2 { 0 } else { b_frames };
        Self {
            p_l0,
            max_references,
            b_frames,
            log2_max_frame_num,
        }
    }

    /// The DPB slots a session needs: one a reference, and one for the
    /// picture being encoded.
    pub(crate) fn slots(&self) -> u32 {
        self.max_references + 1
    }
}

/// Plans frames as they come.
pub(crate) struct Planner<F> {
    limits: Limits,
    intra_period: Option<u32>,
    keys: KeyFrames,
    /// Frames waiting to be B pictures, with their order.
    waiting: Vec<(F, u32)>,
    /// The references held, oldest first.
    dpb: Vec<Reference>,
    next_number: u64,
    /// The last reference's `frame_num`.
    prev_ref_frame_num: u32,
    /// The last IDR picture's `idr_pic_id`.
    idr_pic_id: u16,
    /// Whether an IDR picture has come yet.
    started: bool,
    /// The order of the last I or IDR picture.
    last_intra: u32,
    /// The next frame is an IDR picture, whatever comes.
    restart: bool,
}

impl<F> Planner<F> {
    /// A planner of `config`'s pattern within `limits`, before its first
    /// frame.
    pub(crate) fn new(config: &GopConfig, limits: Limits) -> Self {
        Self {
            limits,
            intra_period: config.intra_period,
            keys: KeyFrames::new(config.idr_period, None),
            waiting: Vec::new(),
            dpb: Vec::new(),
            next_number: 0,
            prev_ref_frame_num: 0,
            idr_pic_id: 0,
            started: false,
            last_intra: 0,
            restart: false,
        }
    }

    /// Makes the next frame an IDR picture, as after a picture that
    /// failed: the references later pictures would predict from may hold
    /// nothing.
    pub(crate) fn restart(&mut self) {
        self.restart = true;
    }

    /// Takes the next frame in display order, with its time stamp and
    /// whether it must be an IDR picture, and returns the frames that can
    /// be encoded now, in encode order.
    pub(crate) fn push(&mut self, frame: F, pts: i64, force_idr: bool) -> Vec<Planned<F>> {
        let force = force_idr || std::mem::take(&mut self.restart);
        let order = self.keys.place(pts, force);
        if order == 0 {
            let mut out = self.close_run();
            out.push(self.idr(frame));
            return out;
        }
        let anchors = self.waiting.len() >= usize::try_from(self.limits.b_frames).unwrap_or(0);
        self.waiting.push((frame, order));
        if anchors {
            self.close_run()
        } else {
            Vec::new()
        }
    }

    /// Returns every frame still waiting, planned, in encode order.
    pub(crate) fn flush(&mut self) -> Vec<Planned<F>> {
        self.close_run()
    }

    /// The least slot no reference holds.
    fn free_slot(&self) -> u32 {
        (0..self.limits.slots())
            .find(|slot| !self.dpb.iter().any(|r| r.slot == *slot))
            .unwrap_or(0)
    }

    fn max_frame_num(&self) -> u32 {
        1 << self.limits.log2_max_frame_num.clamp(4, 16)
    }

    fn idr(&mut self, frame: F) -> Planned<F> {
        self.dpb.clear();
        if self.started {
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
        }
        self.started = true;
        self.prev_ref_frame_num = 0;
        self.last_intra = 0;
        self.keep(frame, PictureKind::Idr, 0, 0, Vec::new())
    }

    /// Plans reference `frame` as `kind`, at `order` and `frame_num`,
    /// predicting from `list0`, and keeps it in the DPB.
    fn keep(
        &mut self,
        frame: F,
        kind: PictureKind,
        order: u32,
        frame_num: u32,
        list0: Vec<Reference>,
    ) -> Planned<F> {
        let number = self.next_number;
        // Numbers only tell apart the few references a DPB holds.
        self.next_number = self.next_number.wrapping_add(1);
        let reference = Reference {
            number,
            slot: self.free_slot(),
            order,
            frame_num,
            kind,
        };
        let planned = Planned {
            frame,
            kind,
            order,
            frame_num,
            idr_pic_id: self.idr_pic_id,
            reference: Some(number),
            setup_slot: reference.slot,
            list0,
            list1: Vec::new(),
            dpb: self.dpb.clone(),
        };
        let max = usize::try_from(self.limits.max_references.max(1)).unwrap_or(1);
        while self.dpb.len() >= max {
            self.dpb.remove(0);
        }
        self.dpb.push(reference);
        planned
    }

    /// Ends a run of waiting frames: the last becomes its anchor, then the
    /// others are B pictures between the reference before and it.
    fn close_run(&mut self) -> Vec<Planned<F>> {
        let mut out = Vec::with_capacity(self.waiting.len());
        let Some((last, order)) = self.waiting.pop() else {
            return out;
        };
        let frame_num = (self.prev_ref_frame_num + 1) % self.max_frame_num();
        self.prev_ref_frame_num = frame_num;
        let intra = self
            .intra_period
            .is_some_and(|period| order.saturating_sub(self.last_intra) >= period.max(1));
        let anchor = if intra {
            self.last_intra = order;
            self.keep(last, PictureKind::I, order, frame_num, Vec::new())
        } else {
            let p_l0 = usize::try_from(self.limits.p_l0).unwrap_or(1);
            let list0 = self.dpb.iter().rev().take(p_l0).copied().collect();
            self.keep(last, PictureKind::P, order, frame_num, list0)
        };
        out.push(anchor);
        let waiting = std::mem::take(&mut self.waiting);
        let b_frame_num = (self.prev_ref_frame_num + 1) % self.max_frame_num();
        for (frame, order) in waiting {
            let (list0, list1) = b_lists(&self.dpb, order);
            out.push(Planned {
                frame,
                kind: PictureKind::B,
                order,
                frame_num: b_frame_num,
                idr_pic_id: self.idr_pic_id,
                reference: None,
                setup_slot: self.free_slot(),
                list0: list0.into_iter().take(1).collect(),
                list1: list1.into_iter().take(1).collect(),
                dpb: self.dpb.clone(),
            });
        }
        out
    }
}

/// The default lists of a B picture at `order` (H.264 8.2.4.2.3, H.265
/// 8.3.4 with every reference current): list 0 the references before it,
/// nearest first, then those after it, nearest first; list 1 the other
/// way about.
fn b_lists(dpb: &[Reference], order: u32) -> (Vec<Reference>, Vec<Reference>) {
    let mut before: Vec<Reference> = dpb.iter().filter(|r| r.order < order).copied().collect();
    let mut after: Vec<Reference> = dpb.iter().filter(|r| r.order > order).copied().collect();
    before.sort_by_key(|r| std::cmp::Reverse(r.order));
    after.sort_by_key(|r| r.order);
    let list0 = before.iter().chain(&after).copied().collect();
    let list1 = after.iter().chain(&before).copied().collect();
    (list0, list1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(references: u32, b_frames: u32) -> Limits {
        let config = GopConfig {
            references,
            b_frames,
            ..GopConfig::default()
        };
        Limits::new(&config, true, 7, 17, 8)
    }

    fn plan(config: GopConfig, limits: Limits, frames: u32, forced: &[u32]) -> Vec<Planned<u32>> {
        let mut planner = Planner::new(&config, limits);
        let mut out = Vec::new();
        for n in 0..frames {
            out.extend(planner.push(n, i64::from(n) * 3000, forced.contains(&n)));
        }
        out.extend(planner.flush());
        out
    }

    fn types(planned: &[Planned<u32>]) -> String {
        planned
            .iter()
            .map(|p| {
                let t = match p.kind {
                    PictureKind::Idr => "IDR",
                    PictureKind::I => "I",
                    PictureKind::P => "P",
                    PictureKind::B => "B",
                };
                format!("{t}{}", p.frame)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn numbers(list: &[Reference]) -> Vec<u32> {
        list.iter().map(|r| r.order).collect()
    }

    /// Checks what every plan must hold: each picture's slot is held by
    /// no reference of its DPB, each list names references of its DPB, the
    /// DPB stays within its size, a B picture's lists hold an anchor
    /// before it and one after it, and the slots and DPB follow from the
    /// pictures before.
    fn check(planned: &[Planned<u32>], limits: Limits) {
        let mut held: Vec<Reference> = Vec::new();
        let mut last_ref_frame_num = 0;
        for p in planned {
            if p.kind == PictureKind::Idr {
                held.clear();
            }
            assert_eq!(p.dpb, held, "{} {:?}", p.frame, p.kind);
            assert!(p.dpb.len() <= limits.max_references as usize);
            assert!(p.setup_slot < limits.slots());
            assert!(!p.dpb.iter().any(|r| r.slot == p.setup_slot));
            for r in p.list0.iter().chain(&p.list1) {
                assert!(
                    p.dpb.contains(r),
                    "{} predicts from a dropped reference",
                    p.frame
                );
            }
            let max_frame_num = 1 << limits.log2_max_frame_num;
            match p.kind {
                PictureKind::Idr => {
                    assert_eq!((p.order, p.frame_num), (0, 0));
                    assert!(p.list0.is_empty() && p.list1.is_empty());
                }
                PictureKind::I => assert!(p.list0.is_empty() && p.list1.is_empty()),
                PictureKind::P => {
                    assert!(!p.list0.is_empty() && p.list1.is_empty());
                    assert!(p.list0.len() <= limits.p_l0 as usize);
                    // Most recent first: each before the last in encode
                    // order.
                    let numbers: Vec<u64> = p.list0.iter().map(|r| r.number).collect();
                    assert!(numbers.windows(2).all(|w| w[0] > w[1]));
                }
                PictureKind::B => {
                    assert_eq!(p.list0.len(), 1);
                    assert_eq!(p.list1.len(), 1);
                    assert!(p.list0[0].order < p.order && p.list1[0].order > p.order);
                    assert!(p.reference.is_none());
                }
            }
            if p.kind != PictureKind::Idr {
                assert_eq!(p.frame_num, (last_ref_frame_num + 1) % max_frame_num);
            }
            if let Some(number) = p.reference {
                last_ref_frame_num = p.frame_num;
                if held.len() >= limits.max_references as usize {
                    held.remove(0);
                }
                held.push(Reference {
                    number,
                    slot: p.setup_slot,
                    order: p.order,
                    frame_num: p.frame_num,
                    kind: p.kind,
                });
            }
        }
    }

    #[test]
    fn p_only_streams_keep_display_order() {
        let config = GopConfig {
            idr_period: Some(4),
            ..GopConfig::default()
        };
        let limits = limits(1, 0);
        let planned = plan(config, limits, 10, &[]);
        assert_eq!(types(&planned), "IDR0 P1 P2 P3 IDR4 P5 P6 P7 IDR8 P9");
        let frame_nums: Vec<u32> = planned.iter().map(|p| p.frame_num).collect();
        assert_eq!(frame_nums, [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]);
        let ids: Vec<u16> = planned.iter().map(|p| p.idr_pic_id).collect();
        assert_eq!(ids, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
        // One reference, two slots used in turn.
        let slots: Vec<u32> = planned.iter().map(|p| p.setup_slot).collect();
        assert_eq!(slots, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!(numbers(&planned[2].list0), [1]);
        check(&planned, limits);
    }

    #[test]
    fn b_pictures_wait_for_their_anchor_and_share_its_frame_num() {
        let config = GopConfig {
            idr_period: Some(8),
            b_frames: 2,
            ..GopConfig::default()
        };
        let limits = limits(1, 2);
        assert_eq!(limits.max_references, 2);
        let planned = plan(config, limits, 11, &[]);
        assert_eq!(types(&planned), "IDR0 P3 B1 B2 P6 B4 B5 P7 IDR8 P10 B9");
        let frame_nums: Vec<u32> = planned.iter().map(|p| p.frame_num).collect();
        assert_eq!(frame_nums, [0, 1, 2, 2, 2, 3, 3, 3, 0, 1, 2]);
        let b1 = &planned[2];
        assert_eq!((numbers(&b1.list0), numbers(&b1.list1)), (vec![0], vec![3]));
        assert_eq!(numbers(&planned[4].list0), [3]);
        // P7 closed the run before the IDR picture: it predicts from P6.
        assert_eq!(numbers(&planned[7].list0), [6]);
        check(&planned, limits);
    }

    #[test]
    fn p_pictures_predict_from_the_most_recent_references() {
        let config = GopConfig {
            idr_period: None,
            references: 3,
            ..GopConfig::default()
        };
        let limits = limits(3, 0);
        let planned = plan(config, limits, 8, &[]);
        assert_eq!(numbers(&planned[1].list0), [0]);
        assert_eq!(numbers(&planned[3].list0), [2, 1, 0]);
        assert_eq!(numbers(&planned[7].list0), [6, 5, 4]);
        // The sliding window drops the oldest: 4 slots in turn.
        let slots: Vec<u32> = planned.iter().map(|p| p.setup_slot).collect();
        assert_eq!(slots, [0, 1, 2, 3, 0, 1, 2, 3]);
        check(&planned, limits);
    }

    #[test]
    fn i_pictures_come_each_intra_period_between_idr_pictures() {
        let config = GopConfig {
            idr_period: Some(12),
            intra_period: Some(4),
            b_frames: 1,
            references: 2,
        };
        let limits = limits(2, 1);
        let planned = plan(config, limits, 14, &[]);
        assert_eq!(
            types(&planned),
            "IDR0 P2 B1 I4 B3 P6 B5 I8 B7 P10 B9 P11 IDR12 P13"
        );
        // A B picture before an I picture predicts from it (an open GOP).
        assert_eq!(numbers(&planned[4].list1), [4]);
        check(&planned, limits);
    }

    #[test]
    fn a_forced_idr_picture_closes_the_run_before_it() {
        let config = GopConfig {
            idr_period: None,
            b_frames: 3,
            references: 1,
            ..GopConfig::default()
        };
        let limits = limits(1, 3);
        let planned = plan(config, limits, 10, &[6]);
        assert_eq!(types(&planned), "IDR0 P4 B1 B2 B3 P5 IDR6 P9 B7 B8");
        assert_eq!(planned[6].idr_pic_id, 1);
        check(&planned, limits);
    }

    #[test]
    fn a_restart_makes_the_next_frame_an_idr_picture() {
        let config = GopConfig {
            idr_period: None,
            ..GopConfig::default()
        };
        let limits = limits(1, 0);
        let mut planner = Planner::new(&config, limits);
        let mut planned = Vec::new();
        for n in 0..3 {
            planned.extend(planner.push(n, i64::from(n), false));
        }
        planner.restart();
        for n in 3..5 {
            planned.extend(planner.push(n, i64::from(n), false));
        }
        assert_eq!(types(&planned), "IDR0 P1 P2 IDR3 P4");
        check(&planned, limits);
    }

    #[test]
    fn frame_numbers_wrap_and_lists_stay_most_recent_first() {
        let config = GopConfig {
            idr_period: None,
            b_frames: 2,
            references: 4,
            ..GopConfig::default()
        };
        let mut limits = limits(4, 2);
        limits.log2_max_frame_num = 4;
        let planned = plan(config, limits, 70, &[]);
        let frame_nums: Vec<u32> = planned.iter().map(|p| p.frame_num).collect();
        assert!(frame_nums.iter().all(|&n| n < 16));
        assert!(frame_nums.windows(2).any(|w| w[0] == 15 && w[1] == 0));
        check(&planned, limits);
    }

    #[test]
    fn limits_follow_the_device() {
        let config = GopConfig {
            idr_period: None,
            intra_period: None,
            b_frames: 12,
            references: 9,
        };
        // The 5080's H.264: 7 references in list 0, 17 slots.
        let l = Limits::new(&config, true, 7, 17, 8);
        assert_eq!((l.p_l0, l.max_references, l.b_frames), (7, 7, MAX_B_FRAMES));
        // No B pictures in the profile or on the device.
        let l = Limits::new(&config, false, 7, 17, 8);
        assert_eq!(l.b_frames, 0);
        // B pictures need two references; a DPB of two slots has room for
        // one.
        let one = GopConfig {
            references: 1,
            b_frames: 2,
            ..config
        };
        assert_eq!(Limits::new(&one, true, 7, 17, 8).max_references, 2);
        let tight = Limits::new(&one, true, 7, 2, 8);
        assert_eq!((tight.max_references, tight.b_frames), (1, 0));
        assert_eq!(tight.slots(), 2);
    }

    /// Patterns of every shape, each picture's plan checked against a
    /// model of the DPB and its slots.
    #[test]
    fn every_pattern_keeps_its_references_in_their_slots() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            u32::try_from(seed % u64::from(n)).unwrap_or(0)
        };
        for _ in 0..300 {
            let config = GopConfig {
                idr_period: [None, Some(1), Some(7), Some(30)][next(4) as usize],
                intra_period: [None, Some(1), Some(3), Some(10)][next(4) as usize],
                b_frames: next(5),
                references: 1 + next(7),
            };
            let limits = Limits::new(&config, next(4) != 0, 1 + next(7), 2 + next(16), 4);
            let frames = 1 + next(120);
            let forced: Vec<u32> = (0..next(4)).map(|_| next(frames)).collect();
            let planned = plan(config, limits, frames, &forced);
            assert_eq!(planned.len(), frames as usize);
            let mut shown: Vec<u32> = planned.iter().map(|p| p.frame).collect();
            shown.sort_unstable();
            assert_eq!(shown, (0..frames).collect::<Vec<_>>());
            for &f in &forced {
                let p = planned.iter().find(|p| p.frame == f).unwrap();
                assert_eq!(p.kind, PictureKind::Idr, "forced frame {f} of {config:?}");
            }
            check(&planned, limits);
        }
    }
}
