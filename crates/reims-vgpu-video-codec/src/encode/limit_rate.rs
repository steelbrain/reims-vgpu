//! Ours: the rate control of a stream held to a hard limit, for a driver
//! whose own rate control does not keep the HRD buffer it is given. The
//! encoder core codes each picture at a constant QP that this chooses:
//! - no second of decode time holds more than the limit: the buffer and a
//!   second at the peak ([`super::EncoderSettings::buffer_bits`]). That is
//!   the limit as VideoToolbox's header defines it, over "any contiguous
//!   segment of that duration (in decode time)", held over the pictures
//!   themselves rather than through a buffer;
//! - the pictures until the next key frame share what the average allows,
//!   less what the pictures so far took over it, each by its type's
//!   complexity: the bits it took last at its QP, as if each 6 QP halved
//!   them;
//! - a picture predicted to take more than 85% of what its second has
//!   left is given the QP that fits, and one that still takes more
//!   than is left is coded again at a higher QP ([`LimitRate::retry`]).
//!
//! It knows nothing of the hardware, so the encoder core drives it and its
//! tests run anywhere.

use std::collections::VecDeque;

use super::FrameType;

/// What the I, P and B pictures' QPs are offset by from one base: I 2
/// below, B 2 above.
const OFFSET: PerType<f64> = PerType {
    idr: -2.0,
    p: 0.0,
    b: 2.0,
};
/// The QPs pictures are given: none under 10, which no limit needs, and
/// none over 51, the largest `SliceQpY` at any bit depth (H.264 7.4.3,
/// HEVC 7.4.7.1).
const MIN_QP: f64 = 10.0;
const MAX_QP: f64 = 51.0;
/// The share of what its second has left that a picture is planned to
/// take at most, for the error of its prediction.
const MARGIN: f64 = 0.85;
/// How far one picture's base QP moves from the last picture's, whatever
/// their types, so that the quality does not jump. A key frame follows the
/// P pictures before it, not the key frame a second back, which kept the
/// first seconds' key frames below their P pictures' quality (VA-API on
/// an RX 7900 XTX, 2026-10-24).
const MAX_STEP: f64 = 4.0;
/// How much of a P or B picture's last complexity its next prediction
/// keeps.
const KEEP: f64 = 0.5;
/// A starting guess of what a picture of each type costs before one is
/// coded, in bits per pixel at QP 30, P pictures a tenth of a key frame
/// and B pictures three fiftieths. The first key frame's own bits replace
/// the key frame's, and its share of the first second follows from the P
/// pictures'. Of a sixth, a tenth and a twentieth, a tenth gave a mean
/// PSNR within 0.15 dB of the best in each of 12 runs, the suite's six
/// limit cases and six more limits on vtencode's frames, and a worst
/// frame within 0.35 dB of the best in 11 (VA-API on an RX 7900 XTX,
/// 2026-10-24). A twentieth
/// lost up to 1.4 dB of the texture's worst frames, and a sixth up to 6 dB
/// of vtencode's. A key frame that costs more than its second has left is
/// coded again.
const PRIOR_BITS_PER_PIXEL: PerType<f64> = PerType {
    idr: 0.6,
    p: 0.06,
    b: 0.036,
};
const PRIOR_QP: f64 = 30.0;

/// A value for each picture type.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PerType<T> {
    idr: T,
    p: T,
    b: T,
}

impl<T: Copy> PerType<T> {
    fn get(&self, frame_type: FrameType) -> T {
        match frame_type {
            FrameType::Idr => self.idr,
            FrameType::P => self.p,
            FrameType::B => self.b,
        }
    }

    fn get_mut(&mut self, frame_type: FrameType) -> &mut T {
        match frame_type {
            FrameType::Idr => &mut self.idr,
            FrameType::P => &mut self.p,
            FrameType::B => &mut self.b,
        }
    }

    fn map(self, f: impl Fn(T) -> T) -> Self {
        Self {
            idr: f(self.idr),
            p: f(self.p),
            b: f(self.b),
        }
    }
}

/// `2^(-qp/6)`: how bits scale with QP.
fn scale(qp: f64) -> f64 {
    (-qp / 6.0).exp2()
}

/// The rate control's state (module doc).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LimitRate {
    /// The most bits any second of decode time holds.
    limit: f64,
    /// The average aimed at, in bits a second.
    average: f64,
    frame_rate: f64,
    /// Ticks a second of the decode times.
    timescale: i64,
    /// The pictures kept in the last second, by decode time, and their
    /// bits.
    recent: VecDeque<(i64, f64)>,
    /// The bits the average allowed the pictures so far and they did not
    /// take: negative where they took more. It is held within a second of
    /// the average either way, so that a stretch of easy or hard pictures
    /// moves the pictures after it for a second at most.
    credit: f64,
    /// Each type's bits at QP 0, as its last picture's bits and QP give it.
    complexity: PerType<f64>,
    /// Whether a picture of each type has been coded.
    seen: PerType<bool>,
    /// The last picture's QP less its type's offset.
    last_base: Option<f64>,
}

impl LimitRate {
    /// A rate control for pictures of `pixels` at `frame_rate`, their
    /// decode times on a clock of `timescale` ticks a second, aiming at
    /// `average` bits a second with at most `limit` bits in any second.
    #[expect(
        clippy::as_conversions,
        reason = "a size in pixels and a limit in bits as floats, as the rate control's arithmetic is"
    )]
    pub(crate) fn new(
        pixels: u64,
        frame_rate: f64,
        timescale: u32,
        average: u32,
        limit: u64,
    ) -> Self {
        let pixels = pixels as f64;
        let limit = (limit as f64).max(1.0);
        Self {
            limit,
            average: f64::from(average.max(1)).min(limit),
            frame_rate: if frame_rate > 0.0 { frame_rate } else { 30.0 },
            timescale: i64::from(timescale.max(1)),
            recent: VecDeque::new(),
            credit: 0.0,
            complexity: PRIOR_BITS_PER_PIXEL.map(|bpp| bpp * pixels / scale(PRIOR_QP)),
            seen: PerType {
                idr: false,
                p: false,
                b: false,
            },
            last_base: None,
        }
    }

    /// Takes a new average and limit, from the next picture.
    #[expect(
        clippy::as_conversions,
        reason = "a limit in bits as a float, as the rate control's arithmetic is"
    )]
    pub(crate) fn set_rates(&mut self, average: u32, limit: u64) {
        self.limit = (limit as f64).max(1.0);
        self.average = f64::from(average.max(1)).min(self.limit);
    }

    /// Takes a new frame rate, from the next picture.
    pub(crate) fn set_frame_rate(&mut self, frame_rate: f64) {
        if frame_rate > 0.0 {
            self.frame_rate = frame_rate;
        }
    }

    /// The bits a picture decoded at `dts` may take: the limit, less the
    /// pictures kept in the second that ends with it.
    pub(crate) fn allowance(&self, dts: i64) -> f64 {
        let start = dts.saturating_sub(self.timescale);
        let held: f64 = (self.recent.iter())
            .filter(|&&(at, _)| at > start)
            .map(|&(_, bits)| bits)
            .sum();
        self.limit - held
    }

    /// The bits a picture of `frame_type` takes at `qp`, as predicted.
    fn predicted(&self, frame_type: FrameType, qp: f64) -> f64 {
        self.complexity.get(frame_type) * scale(qp)
    }

    /// The QP that makes a picture of `frame_type` take `bits`, as
    /// predicted.
    fn qp_for(&self, frame_type: FrameType, bits: f64) -> f64 {
        -6.0 * (bits.max(1.0) / self.complexity.get(frame_type)).log2()
    }

    /// The QP of the next picture, of `frame_type`, decoded at `dts`, with
    /// `remaining` pictures to the next key frame counting this one, or
    /// none where no key frame is planned.
    pub(crate) fn plan(&self, frame_type: FrameType, remaining: Option<u32>, dts: i64) -> u8 {
        let offset = OFFSET.get(frame_type);
        // The horizon: to the next key frame, or a second where none is
        // planned.
        let frames = match remaining {
            Some(n) => f64::from(n.max(1)),
            None => self.frame_rate.round().max(1.0),
        };
        let share = self.average * frames / self.frame_rate;
        let budget = (share + self.credit).max(share / 4.0);
        // This picture, then P pictures, at one base QP.
        let weight = self.predicted(frame_type, offset) + (frames - 1.0) * self.complexity.p;
        let mut qp = -6.0 * (budget / weight).log2() + offset;
        if let Some(last) = self.last_base {
            qp = qp.clamp(last + offset - MAX_STEP, last + offset + MAX_STEP);
        }
        // Never more than its second has left, whatever the steps.
        qp = qp.max(self.qp_for(frame_type, self.allowance(dts) * MARGIN));
        whole_qp(qp)
    }

    /// After a picture of `frame_type` decoded at `dts` and coded at `qp`
    /// took `bits`, more than [`LimitRate::allowance`]: the QP to code it
    /// again at, or none at the largest. Its type's complexity is taken
    /// from this try.
    pub(crate) fn retry(
        &mut self,
        frame_type: FrameType,
        qp: u8,
        bits: f64,
        dts: i64,
    ) -> Option<u8> {
        if f64::from(qp) >= MAX_QP {
            return None;
        }
        *self.complexity.get_mut(frame_type) = bits.max(1.0) / scale(f64::from(qp));
        let fit = self.qp_for(frame_type, self.allowance(dts) * MARGIN);
        Some(whole_qp(fit.max(f64::from(qp) + 1.0)))
    }

    /// Takes a picture of `frame_type` decoded at `dts` and coded at `qp`
    /// into `bits`.
    pub(crate) fn coded(&mut self, frame_type: FrameType, qp: u8, bits: f64, dts: i64) {
        let qp = f64::from(qp);
        let complexity = bits.max(1.0) / scale(qp);
        let seen = self.seen.get(frame_type);
        let kept = self.complexity.get_mut(frame_type);
        *kept = if seen && frame_type != FrameType::Idr {
            KEEP * *kept + (1.0 - KEEP) * complexity
        } else {
            complexity
        };
        if !seen && frame_type == FrameType::Idr && !self.seen.p {
            // Until a P picture is coded, it is assumed to cost a tenth of
            // the key frame, as the priors say.
            let prior = PRIOR_BITS_PER_PIXEL;
            self.complexity.p = complexity * prior.p / prior.idr;
            self.complexity.b = complexity * prior.b / prior.idr;
        }
        *self.seen.get_mut(frame_type) = true;
        self.last_base = Some(qp - OFFSET.get(frame_type));
        let start = dts.saturating_sub(self.timescale);
        while self.recent.front().is_some_and(|&(at, _)| at <= start) {
            self.recent.pop_front();
        }
        self.recent.push_back((dts, bits));
        self.credit = (self.credit + self.average / self.frame_rate - bits)
            .clamp(-self.average, self.average);
    }
}

/// A QP as the pictures are given it: rounded up, within 10 to 51.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the clamp holds it to 10 to 51, and a NaN is taken to 10 first"
)]
fn whole_qp(qp: f64) -> u8 {
    // A NaN passes a clamp.
    let qp = if qp.is_nan() {
        MIN_QP
    } else {
        qp.clamp(MIN_QP, MAX_QP)
    };
    qp.ceil() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ticks a frame at 30 frames a second on a 90 kHz clock.
    const TICKS: i64 = 3000;

    /// A picture's bits at `qp`, by `complexity` at QP 0.
    fn bits_at(complexity: f64, qp: u8) -> f64 {
        complexity * scale(f64::from(qp))
    }

    /// The bits and QPs of `frames` pictures coded as the encoder core
    /// codes them, a key frame every `interval`: each picture planned, coded
    /// again while it takes more than its second has left, and kept. Key
    /// frames cost `key` at QP 0 and P pictures a fortieth of it, but with
    /// `spikes`, every 16th P picture costs four fifths of a key frame, as
    /// vtencode's chroma drift makes it when it wraps.
    fn run(
        rate: &mut LimitRate,
        frames: u32,
        interval: u32,
        key: f64,
        spikes: bool,
    ) -> Vec<(f64, u8)> {
        let mut out = Vec::new();
        for n in 0..frames {
            let dts = i64::from(n) * TICKS;
            let since = n % interval;
            let (frame_type, complexity) = match since {
                0 => (FrameType::Idr, key),
                s if spikes && s % 16 == 0 => (FrameType::P, key * 0.8),
                _ => (FrameType::P, key / 40.0),
            };
            let mut qp = rate.plan(frame_type, Some(interval - since), dts);
            let mut bits = bits_at(complexity, qp);
            while bits > rate.allowance(dts) {
                match rate.retry(frame_type, qp, bits, dts) {
                    Some(next) => qp = next,
                    None => break,
                }
                bits = bits_at(complexity, qp);
            }
            rate.coded(frame_type, qp, bits, dts);
            out.push((bits, qp));
        }
        out
    }

    /// The most bits in any 30 pictures in a row: a second.
    fn busiest(bits: &[(f64, u8)]) -> f64 {
        bits.windows(30)
            .map(|w| w.iter().map(|(b, _)| b).sum())
            .fold(0.0, f64::max)
    }

    /// The bits a second over 10 seconds of pictures.
    fn per_second(bits: &[(f64, u8)]) -> f64 {
        bits.iter().map(|(b, _)| b).sum::<f64>() / 10.0
    }

    #[test]
    fn no_second_takes_more_than_the_limit() {
        // 720p under a limit of 775 kbit, with an average of two thirds of
        // it, with key frames that cost 1.2 Mbit
        // at QP 26 and spikes as costly.
        let (average, limit) = (516_666, 775_000);
        let mut rate = LimitRate::new(1280 * 720, 30.0, 90_000, average, limit);
        let bits = run(&mut rate, 300, 30, 1.2e6 / scale(26.0), true);
        assert!(busiest(&bits) <= 775_000.0, "{}", busiest(&bits));
        // Near the average, and over it by a twentieth at most.
        let got = per_second(&bits);
        assert!(got > 0.8 * 516_666.0 && got <= 1.05 * 516_666.0, "{got}");
    }

    #[test]
    fn key_frames_take_what_their_second_has_left() {
        // Key frames 40 times a P picture: once the first two have stepped
        // up to it, each takes more than a third of the limit, all that a
        // buffer of half a second at a peak of two thirds of it holds.
        let (average, limit) = (516_666, 775_000);
        let mut rate = LimitRate::new(1280 * 720, 30.0, 90_000, average, limit);
        let bits = run(&mut rate, 300, 30, 1.2e6 / scale(26.0), false);
        assert!(busiest(&bits) <= 775_000.0);
        let keys: Vec<f64> = bits.iter().step_by(30).skip(2).map(|&(b, _)| b).collect();
        assert!(keys.iter().all(|&b| b > 775_000.0 / 3.0), "{keys:?}");
    }

    #[test]
    fn an_easy_content_stays_at_its_average() {
        // An average of a third of the limit on pictures that reach it at
        // about QP 23: the limit is never near, and the average is kept to
        // within a tenth.
        let mut rate = LimitRate::new(1280 * 720, 30.0, 90_000, 1_000_000, 3_000_000);
        let bits = run(&mut rate, 300, 60, 0.6e6 / scale(26.0), false);
        let got = per_second(&bits);
        assert!((got - 1e6).abs() < 1e5, "{got}");
    }

    #[test]
    fn a_retry_raises_the_qp_until_the_largest() {
        let mut rate = LimitRate::new(1280 * 720, 30.0, 90_000, 500_000, 750_000);
        let first = rate.plan(FrameType::Idr, Some(30), 0);
        // Ten times the limit: the next QP fits it, as predicted.
        let next = rate.retry(FrameType::Idr, first, 7.5e6, 0);
        assert!(next.is_some_and(|qp| qp >= first + 6));
        // Just over: one QP more at least.
        let over = rate.retry(FrameType::Idr, 40, 750_001.0, 0);
        assert_eq!(over.map(|qp| qp > 40), Some(true));
        assert_eq!(rate.retry(FrameType::Idr, 51, 1e9, 0), None);
    }

    #[test]
    fn a_picture_holds_its_second_and_no_more() {
        let mut rate = LimitRate::new(640 * 360, 30.0, 90_000, 300_000, 450_000);
        assert_eq!(rate.allowance(0), 450_000.0);
        rate.coded(FrameType::Idr, 30, 100_000.0, 0);
        assert_eq!(rate.allowance(TICKS), 350_000.0);
        rate.coded(FrameType::P, 30, 50_000.0, TICKS);
        assert_eq!(rate.allowance(29 * TICKS), 300_000.0);
        // A second after the key frame, its second is over.
        assert_eq!(rate.allowance(30 * TICKS), 400_000.0);
        assert_eq!(rate.allowance(31 * TICKS), 450_000.0);
        // A lower limit holds from the next picture.
        rate.set_rates(100_000, 150_000);
        assert_eq!(rate.allowance(2 * TICKS), 0.0);
    }

    #[test]
    fn a_lower_limit_raises_the_qp() {
        let mut high = LimitRate::new(1280 * 720, 30.0, 90_000, 2_000_000, 3_000_000);
        let mut low = high.clone();
        low.set_rates(200_000, 300_000);
        for rate in [&mut high, &mut low] {
            rate.coded(FrameType::Idr, 30, 80_000.0, 0);
        }
        let (low, high) = (
            low.plan(FrameType::P, Some(29), TICKS),
            high.plan(FrameType::P, Some(29), TICKS),
        );
        assert!(low > high, "{low} <= {high}");
    }
}
