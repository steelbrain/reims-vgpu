//! Encoding's portable half: the settings an encoder is made with, the
//! samples it gives out, and what it writes itself, which is everything
//! above the slice data (ITU-T H.264 08/2021, ITU-T H.265 09/2023, by
//! clause):
//!
//! - the parameter sets and slice headers, so that the VUI is ours to set,
//!   and the `avcC` and `hvcC` configuration records built from them;
//! - each picture's type, references and order counts ([`gop`]), and so
//!   the decode order, decode time stamps and sync flags of its samples;
//! - the level, from the size, rate and bitrate, where none is asked for.
//!
//! The slice data is the hardware's: the VA-API backend's encoder hands
//! the driver these headers packed ([`encoder`]), and turns its output
//! into [`EncodedSample`]s. NVENC writes its own headers, and takes the B
//! pictures' limits, the key frames ([`gop::KeyFrames`]) and the decode
//! times ([`gop::DecodeTimes`]) from here.

pub(crate) mod bits;
pub mod encoder;
pub mod gop;
pub mod h264;
pub mod hevc;
pub mod limit_rate;
pub mod nal;

use std::num::NonZeroU32;
use std::sync::Arc;

/// A codec the encoders write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264, `avc1`.
    H264,
    /// HEVC, `hvc1`.
    Hevc,
}

/// A profile the encoders write, which fixes the codec and bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// H.264 Constrained Baseline: `profile_idc` 66 with
    /// `constraint_set0_flag` and `constraint_set1_flag`, so that Baseline
    /// and Main decoders take it. CAVLC, no B slices.
    H264ConstrainedBaseline,
    /// H.264 Main, `profile_idc` 77.
    H264Main,
    /// H.264 High, `profile_idc` 100, with the 8x8 transform.
    H264High,
    /// HEVC Main, `general_profile_idc` 1: 8-bit 4:2:0.
    HevcMain,
    /// HEVC Main 10, `general_profile_idc` 2: 10-bit 4:2:0.
    HevcMain10,
}

impl Profile {
    /// The codec.
    pub const fn codec(self) -> Codec {
        match self {
            Self::H264ConstrainedBaseline | Self::H264Main | Self::H264High => Codec::H264,
            Self::HevcMain | Self::HevcMain10 => Codec::Hevc,
        }
    }

    /// The bitstream's profile number: `profile_idc` (H.264) or
    /// `general_profile_idc` (HEVC).
    pub const fn profile_idc(self) -> u8 {
        match self {
            Self::H264ConstrainedBaseline => 66,
            Self::H264Main => 77,
            Self::H264High => 100,
            Self::HevcMain => 1,
            Self::HevcMain10 => 2,
        }
    }

    /// Bits a sample.
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::HevcMain10 => 10,
            _ => 8,
        }
    }

    /// The input format the profile takes.
    pub const fn input_format(self) -> InputFormat {
        match self {
            Self::HevcMain10 => InputFormat::P010,
            _ => InputFormat::Nv12,
        }
    }

    /// Whether the profile has B slices.
    pub const fn has_b_slices(self) -> bool {
        matches!(self, Self::H264Main | Self::H264High)
    }
}

/// How the frames handed to an encoder are laid out: 4:2:0, a luma plane
/// and an interleaved Cb/Cr plane half as tall.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputFormat {
    /// 8-bit samples, one byte each (CoreVideo's `420v` and `420f`).
    Nv12,
    /// 10-bit samples in the high bits of little-endian 16-bit words
    /// (CoreVideo's `x420` and `xf20`).
    P010,
}

impl InputFormat {
    /// The bytes one sample takes.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12 => 1,
            Self::P010 => 2,
        }
    }
}

/// Frames a second, as a fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameRate {
    /// Frames.
    pub numerator: u32,
    /// Per this many seconds.
    pub denominator: u32,
}

impl FrameRate {
    /// `numerator / denominator` frames a second.
    pub const fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// The rate as a float, 0 when the denominator is 0.
    pub fn as_f64(self) -> f64 {
        if self.denominator == 0 {
            0.0
        } else {
            f64::from(self.numerator) / f64::from(self.denominator)
        }
    }

    /// The rate in lowest terms with both parts under 65536, as VA-API's
    /// frame rate parameter and the VUI's 32-bit fields take it.
    pub fn reduced(self) -> Self {
        // The gcd is 0 only where both are, which are in lowest terms.
        let (mut numerator, mut denominator) =
            match NonZeroU32::new(gcd(self.numerator, self.denominator)) {
                Some(gcd) => (self.numerator / gcd, self.denominator / gcd),
                None => (self.numerator, self.denominator),
            };
        while numerator > 0xffff || denominator > 0xffff {
            numerator = numerator.div_ceil(2);
            denominator = denominator.div_ceil(2);
        }
        Self {
            numerator,
            denominator,
        }
    }
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while let Some(t) = a.checked_rem(b) {
        a = b;
        b = t;
    }
    a
}

/// How the bitrate is controlled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RateControl {
    /// A fixed quantiser for each picture type, 0 to 51 (8-bit; HEVC Main
    /// 10 takes the same range, as VA-API does).
    ConstantQp {
        /// For I pictures.
        intra: u8,
        /// For P pictures.
        predicted: u8,
        /// For B pictures.
        bidirectional: u8,
    },
    /// A constant bitrate.
    ConstantBitRate {
        /// The rate.
        bits_per_second: u32,
    },
    /// A variable bitrate around an average, under a peak.
    VariableBitRate {
        /// The average.
        average_bits_per_second: u32,
        /// The peak, at least the average.
        peak_bits_per_second: u32,
    },
    /// A variable bitrate that aims at a quality, under a peak (VA-API's
    /// QVBR).
    QualityVariableBitRate {
        /// The average.
        average_bits_per_second: u32,
        /// The peak, at least the average.
        peak_bits_per_second: u32,
        /// The quality factor, 1 (best) to 51.
        quality: u8,
    },
}

impl RateControl {
    /// The highest rate the stream may reach, for the level and the HRD
    /// buffer; `None` for a constant quantiser.
    pub const fn peak_bits_per_second(self) -> Option<u32> {
        match self {
            Self::ConstantQp { .. } => None,
            Self::ConstantBitRate { bits_per_second } => Some(bits_per_second),
            Self::VariableBitRate {
                peak_bits_per_second,
                ..
            }
            | Self::QualityVariableBitRate {
                peak_bits_per_second,
                ..
            } => Some(peak_bits_per_second),
        }
    }
}

/// The colour code points of ITU-T H.273, as the VUI carries them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColourDescription {
    /// `colour_primaries`: 1 BT.709, 9 BT.2020, 12 P3-D65, ...
    pub primaries: u8,
    /// `transfer_characteristics`: 1 BT.709, 16 PQ, 18 HLG, ...
    pub transfer: u8,
    /// `matrix_coefficients`: 1 BT.709, 6 BT.601, 9 BT.2020 NCL, ...
    pub matrix: u8,
}

/// H.264's entropy coder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Entropy {
    /// CAVLC, the only one Constrained Baseline has.
    Cavlc,
    /// CABAC, Main and High's default.
    Cabac,
}

/// Ours: the HRD buffer's length where nothing asks for another, a
/// second at the peak, as libva's users size it.
pub(crate) const SECOND_BUFFER_MILLISECONDS: u32 = 1000;

/// What an encoder is made with. [`EncoderSettings::new`] gives the
/// defaults; the encoder checks the rest against the driver when it is
/// made.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderSettings {
    /// The profile, and so the codec and bit depth.
    pub profile: Profile,
    /// The picture's width, even.
    pub width: u32,
    /// The picture's height, even.
    pub height: u32,
    /// Whether samples use the full range (0 to 255) rather than the video
    /// range (16 to 235): the VUI's `video_full_range_flag`.
    pub full_range: bool,
    /// The colour code points for the VUI; none leaves them unspecified.
    pub colour: Option<ColourDescription>,
    /// The sample aspect ratio, horizontal to vertical; none is
    /// unspecified.
    pub sample_aspect_ratio: Option<(u16, u16)>,
    /// The expected frame rate: the VUI's timing, and what rate control
    /// budgets by.
    pub frame_rate: FrameRate,
    /// Ticks a second of the time stamps and durations frames carry.
    pub timescale: u32,
    /// How the bitrate is controlled.
    pub rate_control: RateControl,
    /// How long the HRD (VBV) buffer holds at the peak rate, in
    /// milliseconds: [`SECOND_BUFFER_MILLISECONDS`], or less where the
    /// stream is held to a hard limit. It starts three quarters full.
    pub buffer_milliseconds: u32,
    /// Whether the peak and the buffer are a hard limit, as
    /// `DataRateLimits` is: no second of the stream may hold more than the
    /// buffer and a second at the peak. Where the driver's own rate control
    /// does not keep the buffer ([`encoder::DriverLimits::keeps_buffer`]),
    /// the encoder picks each picture's QP itself ([`limit_rate`]).
    pub(crate) hard_limit: bool,
    /// At most this many frames from one key frame to the next; none for
    /// no limit (a key frame only at the start and when one is forced).
    pub key_frame_interval: Option<u32>,
    /// At most this long, in ticks, from one key frame to the next.
    pub key_frame_interval_duration: Option<i64>,
    /// Whether frames may be encoded out of display order: B pictures,
    /// where the profile and driver have them.
    pub allow_frame_reordering: bool,
    /// At most this many frames held back before a sample comes out; none
    /// for no limit. B pictures hold frames back, so 0 forbids them.
    pub max_frame_delay: Option<u32>,
    /// Consecutive B pictures between references when reordering is
    /// allowed: none for the encoder's choice.
    pub b_frames: Option<u32>,
    /// The level as the bitstream numbers it (`level_idc`, or HEVC's
    /// `general_level_idc`); none to choose the lowest that fits.
    pub level: Option<u8>,
    /// H.264's entropy coder; none for the profile's default.
    pub entropy: Option<Entropy>,
    /// The quality level on VA-API's scale, 1 the best and slowest, which
    /// NVENC's presets are mapped from; none for the driver's default.
    pub quality_level: Option<u32>,
}

impl EncoderSettings {
    /// Settings for `profile` at `width` by `height`, 30 frames a second
    /// on a 90 kHz clock, at a variable 5 Mbit/s with a 7.5 Mbit/s peak, a
    /// key frame every 2 seconds, frames reordered where the profile can.
    pub fn new(profile: Profile, width: u32, height: u32) -> Self {
        Self {
            profile,
            width,
            height,
            full_range: false,
            colour: None,
            sample_aspect_ratio: None,
            frame_rate: FrameRate::new(30, 1),
            timescale: 90_000,
            rate_control: RateControl::VariableBitRate {
                average_bits_per_second: 5_000_000,
                peak_bits_per_second: 7_500_000,
            },
            buffer_milliseconds: SECOND_BUFFER_MILLISECONDS,
            hard_limit: false,
            key_frame_interval: Some(60),
            key_frame_interval_duration: None,
            allow_frame_reordering: true,
            max_frame_delay: None,
            b_frames: None,
            level: None,
            entropy: None,
            quality_level: None,
        }
    }

    /// The HRD buffer at `peak` bits a second, in bits, at least one, and
    /// how full it starts: three quarters, as libva's users start it.
    pub fn buffer_bits(&self, peak: u32) -> (u32, u32) {
        // A size past a u32, as a product past a u64 would be, is a u32's
        // largest.
        let size = u64::from(peak)
            .checked_mul(u64::from(self.buffer_milliseconds))
            .and_then(|bits| u32::try_from(bits / 1000).ok())
            .unwrap_or(u32::MAX)
            .max(1);
        // A quarter of a u32 three times fits one: this never saturates.
        (size, (size / 4).saturating_mul(3))
    }

    /// Holds the stream to its peak and buffer as a hard limit where `hard`, as
    /// `DataRateLimits` does; to a peak alone otherwise.
    pub fn set_hard_limit(&mut self, hard: bool) {
        self.hard_limit = hard;
    }

    /// The input format frames come in.
    pub fn input_format(&self) -> InputFormat {
        self.profile.input_format()
    }

    /// The entropy coder H.264 uses: CAVLC for Constrained Baseline,
    /// else what was asked, CABAC by default.
    pub fn h264_entropy(&self) -> Entropy {
        match (self.profile, self.entropy) {
            (Profile::H264ConstrainedBaseline, _) | (_, Some(Entropy::Cavlc)) => Entropy::Cavlc,
            _ => Entropy::Cabac,
        }
    }
}

/// One plane of a frame in host memory.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    /// The plane's bytes, from its first row.
    pub data: &'a [u8],
    /// Bytes from one row to the next.
    pub bytes_per_row: usize,
}

/// A frame to encode, in display order.
#[derive(Clone, Copy, Debug)]
pub struct InputFrame<'a> {
    /// The luma plane, then the interleaved chroma plane, in the
    /// settings' input format.
    pub planes: [Plane<'a>; 2],
    /// The presentation time stamp, in the settings' ticks; each frame's
    /// later than the last.
    pub pts: i64,
    /// The frame's duration in ticks, or 0 when unknown.
    pub duration: i64,
    /// Whether this frame must be a key frame.
    pub force_key_frame: bool,
    /// The caller's tag for the frame, given back with its sample.
    pub id: u64,
}

/// A picture's type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameType {
    /// An IDR picture: a key frame, where decoding can start.
    Idr,
    /// A P picture.
    P,
    /// A B picture, which nothing predicts from.
    B,
}

/// One encoded frame, in decode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSample {
    /// The tag of the frame it encodes.
    pub id: u64,
    /// Its NAL units, each behind a four-byte big-endian length, without
    /// the parameter sets, which are in the configuration record.
    pub data: Vec<u8>,
    /// The presentation time stamp, as the frame came.
    pub pts: i64,
    /// The decode time stamp: increasing, the PTS of the frame sent in
    /// this sample's place, as Apple's encoders give it ([`gop::DecodeTimes`]),
    /// so a B picture's is after its PTS.
    pub dts: i64,
    /// The duration, as the frame came.
    pub duration: i64,
    /// The picture's type.
    pub frame_type: FrameType,
    /// Whether decoding can start here: an IDR picture.
    pub is_sync: bool,
    /// Whether it predicts from other frames.
    pub depends_on_others: bool,
    /// Whether other frames may predict from it (a reference picture).
    pub is_depended_on: bool,
    /// The configuration record (`avcC` or `hvcC`), on the first sample and
    /// on the first after the parameter sets change; none otherwise.
    pub configuration: Option<Arc<[u8]>>,
}

/// A frame an encoder took and has no sample for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FailedSample<E> {
    /// The tag of the frame.
    pub id: u64,
    /// Its picture's error; `None` for a picture never coded, lost with an
    /// earlier picture of its run, which failed.
    pub error: Option<E>,
}

/// What an encoder gives back for each frame it took, in decode order: its
/// sample, or why it has none. Each frame's result is its own, so one
/// failure loses no sample coded with it (core's `EncodedFrame::error`).
pub type Coded<E> = Result<EncodedSample, FailedSample<E>>;

/// Why settings cannot be encoded: what is out of range or missing, in
/// words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

impl Unsupported {
    /// A value worked out from the settings or from what the hardware
    /// wrote, named, past what its type holds.
    pub(crate) fn overflow(name: &str) -> Self {
        Self(format!("{name} overflows"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rates_reduce_to_16_bits() {
        assert_eq!(
            FrameRate::new(60_000, 2002).reduced(),
            FrameRate::new(30_000, 1001)
        );
        assert_eq!(FrameRate::new(25, 1).reduced(), FrameRate::new(25, 1));
        let big = FrameRate::new(1_000_000, 33_333).reduced();
        assert!(big.numerator <= 0xffff && big.denominator <= 0xffff);
        assert!((big.as_f64() - 30.0).abs() < 0.01);
    }

    #[test]
    fn baseline_takes_cavlc_whatever_is_asked() {
        let mut settings = EncoderSettings::new(Profile::H264ConstrainedBaseline, 64, 64);
        settings.entropy = Some(Entropy::Cabac);
        assert_eq!(settings.h264_entropy(), Entropy::Cavlc);
        settings.profile = Profile::H264High;
        assert_eq!(settings.h264_entropy(), Entropy::Cabac);
    }
}
