//! Each frame's picture type and references, and so the order frames are
//! encoded in, and the decode time stamps of the samples that come out.
//!
//! Frames come in display order. The first frame, a forced key frame, the
//! first frame past the key-frame interval, and the first whose order would
//! pass [`MAX_ORDER`] are IDR pictures ([`KeyFrames`]). Between them, with
//! `b_frames` above 0, up to that many frames wait as B pictures for the
//! next frame, which becomes a P picture and is encoded before them:
//! `I0 B1 B2 P3` is encoded `I0 P3 B1 B2`. A B picture
//! predicts from the reference before it (list 0) and the one after it
//! (list 1), and nothing predicts from it; a P picture predicts from the
//! reference before it. B pictures still waiting when an IDR picture comes,
//! or at a flush, end their run as x264 ends one: the last becomes a P
//! picture, so that none predicts across the IDR picture.

use std::collections::VecDeque;

use super::FrameType;
use crate::codec::num::ToUsize;

/// The highest order a picture is given, its place in display order from
/// its IDR picture: the frame that would take a higher one is an IDR
/// picture instead ([`KeyFrames`]), so that a GOP is at most 2^30 frames.
///
/// The standards hold a picture's order counts, and the counts a decoder
/// derives from their LSBs, within 32 bits:
/// - H.264 8.2.1: `TopFieldOrderCnt`, `BottomFieldOrderCnt` and
///   `PicOrderCntMsb` within -2^31 to 2^31 - 1. A frame's counts are twice
///   its order ([`super::encoder`]), so the order is at most 2^30 - 1, and
///   the count 2^31 - 2.
/// - HEVC 8.3.1: `PicOrderCntVal` within -2^31 to 2^31 - 1. It is the
///   order, so the order is at most 2^31 - 1.
///
/// The bound is H.264's, the tighter, for both: one rule for every
/// encoder that places its key frames by [`KeyFrames`]. It also keeps a GOP
/// shorter than the one NVENC is configured with where it runs B pictures,
/// 2^31 - 1 frames (the NVIDIA crate's `nvenc::encoder::configure`).
///
/// Reordering moves no count past it. A picture's order is its place in
/// display order from the IDR picture, a B picture's too: above the
/// reference before it and below the one after it. The frames waiting as B
/// pictures when a frame would pass the bound have lower orders than that
/// frame, and their run is closed with a P picture before the IDR picture,
/// as before any IDR picture.
pub const MAX_ORDER: u32 = (1 << 30) - 1;

/// Where IDR pictures fall, frame by frame in display order: at the first
/// frame, at a forced key frame, at the first frame past the key-frame
/// interval in frames or in time stamp ticks, and at the first whose order
/// would pass [`MAX_ORDER`]. The GOP planner places VA-API's key frames by
/// it, and the NVIDIA backend NVENC's.
#[derive(Clone, Debug)]
pub struct KeyFrames {
    /// At most this many frames from one IDR picture to the next.
    interval: Option<u32>,
    /// At most this many ticks from one IDR picture to the next.
    interval_duration: Option<i64>,
    /// The order the next frame gets if it is not an IDR picture: the
    /// frames since the last IDR picture, it included.
    next_order: u32,
    /// The last IDR picture's time stamp; none before the first frame.
    key_pts: Option<i64>,
}

impl KeyFrames {
    /// Key frames at most `interval` frames or `interval_duration` ticks
    /// apart, where either is set, before the first frame.
    pub fn new(interval: Option<u32>, interval_duration: Option<i64>) -> Self {
        Self {
            interval,
            interval_duration,
            next_order: 0,
            key_pts: None,
        }
    }

    /// Changes the interval, from the next frame on.
    pub fn set_interval(&mut self, frames: Option<u32>, duration: Option<i64>) {
        self.interval = frames;
        self.interval_duration = duration;
    }

    /// Places the next frame in display order, with its time stamp and
    /// whether a key frame is forced there, and returns its order: 0 for
    /// an IDR picture.
    pub fn place(&mut self, pts: i64, force: bool) -> u32 {
        if self.is_key(pts, force) {
            self.next_order = 1;
            self.key_pts = Some(pts);
            return 0;
        }
        let order = self.next_order;
        // At most MAX_ORDER, so the add never saturates: `is_key` makes
        // the frame past it an IDR picture.
        self.next_order = order.saturating_add(1);
        order
    }

    fn is_key(&self, pts: i64, force: bool) -> bool {
        let Some(key_pts) = self.key_pts else {
            return true;
        };
        let by_count = self
            .interval
            .is_some_and(|interval| self.next_order >= interval.max(1));
        let by_time = self
            .interval_duration
            .is_some_and(|duration| pts.saturating_sub(key_pts) >= duration.max(1));
        force || by_count || by_time || self.next_order > MAX_ORDER
    }

    /// Ours, for tests: the next frame that is not an IDR picture gets
    /// `order`, at least 1, as if that many frames had followed the last
    /// IDR picture.
    #[cfg(test)]
    pub(crate) fn skip_to(&mut self, order: u32) {
        self.next_order = order;
    }
}

/// When key frames come, and how many B pictures run between references.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GopConfig {
    /// At most this many frames from one IDR picture to the next.
    pub(crate) key_frame_interval: Option<u32>,
    /// At most this many ticks from one IDR picture to the next.
    pub(crate) key_frame_interval_duration: Option<i64>,
    /// Consecutive B pictures; 0 for none.
    pub(crate) b_frames: u32,
}

/// A frame with its place in the stream, in encode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Planned<F> {
    /// The frame, as it was given.
    pub(crate) frame: F,
    /// Whether it is coded as an IDR, P or B picture.
    pub(crate) frame_type: FrameType,
    /// Frames since the last IDR picture in display order, the IDR picture
    /// being 0: what the order counts are made from. At most [`MAX_ORDER`].
    pub(crate) order: u32,
    /// For IDR and P pictures, its number among the references: they are
    /// numbered from 0 in encode order. B pictures are not references.
    pub(crate) reference: Option<u64>,
    /// The reference list 0 predicts from: the one before it.
    pub(crate) forward: Option<u64>,
    /// The reference list 1 predicts from: the one after it (B only).
    pub(crate) backward: Option<u64>,
}

/// Plans frames as they come.
#[derive(Debug)]
pub(crate) struct GopPlanner<F> {
    /// Consecutive B pictures; 0 for none.
    b_frames: u32,
    keys: KeyFrames,
    /// Frames waiting to be B pictures, with their order.
    waiting: Vec<(F, u32)>,
    /// The last reference's number, none before the first frame.
    last_reference: Option<u64>,
    next_reference: u64,
}

impl<F> GopPlanner<F> {
    /// A planner of `config`'s pattern, before its first frame.
    pub(crate) fn new(config: GopConfig) -> Self {
        Self {
            b_frames: config.b_frames,
            keys: KeyFrames::new(
                config.key_frame_interval,
                config.key_frame_interval_duration,
            ),
            waiting: Vec::new(),
            last_reference: None,
            next_reference: 0,
        }
    }

    /// Changes when key frames come, from the next frame on.
    pub(crate) fn set_key_frame_interval(&mut self, frames: Option<u32>, duration: Option<i64>) {
        self.keys.set_interval(frames, duration);
    }

    /// Ours, for tests: [`KeyFrames::skip_to`].
    #[cfg(test)]
    pub(crate) fn skip_to(&mut self, order: u32) {
        self.keys.skip_to(order);
    }

    fn take_reference(&mut self) -> u64 {
        let number = self.next_reference;
        // Numbers only tell apart the few references a DPB holds, which a
        // wrap past 2^64 pictures would keep apart too.
        self.next_reference = self.next_reference.wrapping_add(1);
        number
    }

    /// Takes the next frame in display order, with its time stamp, and
    /// returns the frames that can be encoded now, in encode order.
    pub(crate) fn push(&mut self, frame: F, pts: i64, force_key_frame: bool) -> Vec<Planned<F>> {
        let order = self.keys.place(pts, force_key_frame);
        if order == 0 {
            let mut out = self.close_run();
            let reference = self.take_reference();
            out.push(Planned {
                frame,
                frame_type: FrameType::Idr,
                order: 0,
                reference: Some(reference),
                forward: None,
                backward: None,
            });
            self.last_reference = Some(reference);
            return out;
        }
        let anchors = self.waiting.len() >= self.b_frames.to_usize();
        self.waiting.push((frame, order));
        if !anchors {
            return Vec::new();
        }
        // The frame is the run's last: its P picture.
        self.close_run()
    }

    /// Encodes `frame` as a P picture, then the waiting frames as B
    /// pictures between the reference before and it.
    fn anchor_run(&mut self, frame: F, order: u32, out: &mut Vec<Planned<F>>) {
        let forward = self.last_reference;
        let reference = self.take_reference();
        out.push(Planned {
            frame,
            frame_type: FrameType::P,
            order,
            reference: Some(reference),
            forward,
            backward: None,
        });
        for (frame, order) in self.waiting.drain(..) {
            out.push(Planned {
                frame,
                frame_type: FrameType::B,
                order,
                reference: None,
                forward,
                backward: Some(reference),
            });
        }
        self.last_reference = Some(reference);
    }

    /// Ends a run of waiting frames: the last becomes a P picture.
    fn close_run(&mut self) -> Vec<Planned<F>> {
        let mut out = Vec::with_capacity(self.waiting.len());
        let Some((last, order)) = self.waiting.pop() else {
            return out;
        };
        self.anchor_run(last, order, &mut out);
        out
    }

    /// Returns every frame still waiting, planned, in encode order.
    pub(crate) fn flush(&mut self) -> Vec<Planned<F>> {
        self.close_run()
    }
}

/// Decode time stamps for samples in decode order, from the frames' time
/// stamps in the order they were sent, as Apple's encoders give them: the
/// `k`th sample's DTS is the PTS of the `k`th frame sent. Apple's host
/// gives its encoder VideoToolbox's queue of the PTSes sent, and AVE's
/// H.264 and HEVC encoders emitted frames 0 4 2 1 3 with DTSes 0 1 2 3 4
/// frames. So with B pictures a reference's DTS is before its PTS and a B
/// picture's after it, and the DTSes increase. Without B pictures DTS is
/// PTS.
///
/// A sample comes out only after its own frame, and the frames before it
/// in decode order, were sent, so there is always a PTS to take.
#[derive(Debug, Default)]
pub struct DecodeTimes {
    /// PTSes of the frames sent, in the order sent, not yet used as a DTS.
    pts: VecDeque<i64>,
}

impl DecodeTimes {
    /// No frame sent yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// A frame was sent, with this PTS.
    pub fn push(&mut self, pts: i64) {
        self.pts.push_back(pts);
    }

    /// The DTS of the next sample out: the oldest PTS sent and not yet
    /// used. A sample of no frame sent, which the encoders never make, gets
    /// 0.
    pub fn next_dts(&mut self) -> i64 {
        self.pts.pop_front().unwrap_or(0)
    }

    /// The DTS the next sample out will have, which [`Self::next_dts`]
    /// then takes.
    pub(crate) fn peek(&self) -> i64 {
        self.pts.front().copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(config: GopConfig, frames: u32, forced: &[u32]) -> Vec<Planned<u32>> {
        let mut planner = GopPlanner::new(config);
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
                let t = match p.frame_type {
                    FrameType::Idr => 'I',
                    FrameType::P => 'P',
                    FrameType::B => 'B',
                };
                format!("{t}{}", p.frame)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn p_only_streams_keep_display_order() {
        let config = GopConfig {
            key_frame_interval: Some(4),
            ..GopConfig::default()
        };
        let planned = plan(config, 10, &[]);
        assert_eq!(types(&planned), "I0 P1 P2 P3 I4 P5 P6 P7 I8 P9");
        assert_eq!(planned[5].order, 1);
        assert_eq!(planned[5].forward, planned[4].reference);
    }

    #[test]
    fn b_pictures_wait_for_the_next_reference() {
        let config = GopConfig {
            key_frame_interval: Some(8),
            b_frames: 2,
            ..GopConfig::default()
        };
        let planned = plan(config, 11, &[]);
        assert_eq!(types(&planned), "I0 P3 B1 B2 P6 B4 B5 P7 I8 P10 B9");
        let b1 = &planned[2];
        assert_eq!(b1.order, 1);
        assert_eq!(b1.forward, planned[0].reference);
        assert_eq!(b1.backward, planned[1].reference);
        assert_eq!(b1.reference, None);
        // P7 closed the run before the IDR picture: it predicts from P6.
        assert_eq!(planned[7].forward, planned[4].reference);
    }

    #[test]
    fn forced_key_frames_and_durations_start_new_gops() {
        let config = GopConfig {
            key_frame_interval: None,
            key_frame_interval_duration: Some(5 * 3000),
            b_frames: 1,
        };
        let planned = plan(config, 9, &[3]);
        assert_eq!(types(&planned), "I0 P2 B1 I3 P5 B4 P7 B6 I8");
        assert_eq!(planned[4].order, 2);
    }

    #[test]
    fn a_frame_past_the_highest_order_is_an_idr_picture() {
        let mut keys = KeyFrames::new(None, None);
        assert_eq!(keys.place(0, false), 0);
        keys.skip_to(MAX_ORDER - 1);
        let orders: Vec<u32> = (1..5).map(|n| keys.place(n * 3000, false)).collect();
        // 2^30 - 2 and 2^30 - 1, then an IDR picture where 2^30 would come.
        assert_eq!(orders, [MAX_ORDER - 1, MAX_ORDER, 0, 1]);
        // The interval runs from the new IDR picture.
        keys.set_interval(Some(3), None);
        let orders: Vec<u32> = (5..9).map(|n| keys.place(n * 3000, false)).collect();
        assert_eq!(orders, [2, 0, 1, 2]);
    }

    #[test]
    fn b_pictures_before_the_highest_order_end_their_run() {
        let config = GopConfig {
            b_frames: 2,
            ..GopConfig::default()
        };
        let mut planner = GopPlanner::new(config);
        let mut planned = planner.push(0, 0, false);
        planner.skip_to(MAX_ORDER - 3);
        for n in 1..7 {
            planned.extend(planner.push(n, i64::from(n) * 3000, false));
        }
        planned.extend(planner.flush());
        // Frame 4 takes the highest order, and waits as a B picture; frame
        // 5 would pass it, so it is an IDR picture, and frame 4 ends its
        // run as a P picture before it.
        assert_eq!(types(&planned), "I0 P3 B1 B2 P4 I5 P6");
        let orders: Vec<u32> = planned.iter().map(|p| p.order).collect();
        let m = MAX_ORDER;
        assert_eq!(orders, [0, m - 1, m - 3, m - 2, m, 0, 1]);
        // The B pictures sit between their references, whose orders are
        // above and below theirs.
        assert_eq!(planned[2].forward, planned[0].reference);
        assert_eq!(planned[2].backward, planned[1].reference);
        assert_eq!(planned[4].forward, planned[1].reference);
        assert_eq!(planned[6].forward, planned[5].reference);
    }

    #[test]
    fn decode_times_are_the_ptses_sent_in_order() {
        let config = GopConfig {
            b_frames: 2,
            ..GopConfig::default()
        };
        let mut planner = GopPlanner::new(config);
        let mut times = DecodeTimes::new();
        let mut out = Vec::new();
        for n in 0..7_u32 {
            let pts = i64::from(n) * 3000;
            times.push(pts);
            for planned in planner.push(n, pts, false) {
                out.push((i64::from(planned.frame) * 3000, times.next_dts()));
            }
        }
        for planned in planner.flush() {
            out.push((i64::from(planned.frame) * 3000, times.next_dts()));
        }
        // I0 P3 B1 B2 P6 B4 B5: each sample's DTS is the PTS of the frame
        // sent in its place, so the B pictures' are after their PTSes.
        let sent: Vec<i64> = (0..7).map(|n| n * 3000).collect();
        let dts: Vec<i64> = out.iter().map(|&(_, dts)| dts).collect();
        assert_eq!(dts, sent);
        let pts: Vec<i64> = out.iter().map(|&(pts, _)| pts / 3000).collect();
        assert_eq!(pts, [0, 3, 1, 2, 6, 4, 5]);
        assert_eq!(out[1], (9000, 3000));
        assert_eq!(out[3], (6000, 9000));
    }
}
