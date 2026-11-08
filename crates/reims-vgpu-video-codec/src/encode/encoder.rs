//! The encoder above the accelerator: it checks the settings against the
//! driver, picks the coded size, level and number of B pictures, writes
//! the parameter sets and configuration record, plans each frame
//! ([`super::gop`]), numbers it, writes its slice header, and turns what
//! the accelerator encodes into [`EncodedSample`]s in decode order. The
//! accelerator ([`Accelerator`]) uploads frames and runs the hardware;
//! VA-API's is the VA-API backend's.

use std::collections::VecDeque;
use std::sync::Arc;

use super::gop::{DecodeTimes, GopConfig, GopPlanner, Planned};
use super::limit_rate::LimitRate;
use super::{
    Codec, Coded, EncodedSample, EncoderSettings, FailedSample, FrameRate, FrameType, InputFrame,
    Profile, RateControl, Unsupported, h264, hevc, nal,
};
use crate::num::ToUsize;

/// B pictures between references when the settings leave it to the
/// encoder: one, which holds one frame back.
pub const DEFAULT_B_FRAMES: u32 = 1;
/// The most B pictures in a run: each holds a frame back.
pub const MAX_B_FRAMES: u32 = 4;
/// The QP of the PPS, which slices of a bitrate-controlled stream start
/// from.
const PPS_QP: u8 = 26;
/// `MaxNumMergeCand` the HEVC slice headers say.
const MAX_NUM_MERGE_CAND: u8 = 5;
/// Ours: how many times a picture held to a hard limit is coded at most,
/// each try after the first at a higher QP ([`LimitRate::retry`]), before
/// it is kept at whatever it took.
const LIMIT_TRIES: u32 = 4;

/// What the driver allows, as the encoder needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverLimits {
    /// Whether the driver takes a constant QP (`VA_RC_CQP`).
    pub constant_qp: bool,
    /// Whether it takes a constant bitrate (`VA_RC_CBR`).
    pub constant_bit_rate: bool,
    /// Whether it takes a variable bitrate (`VA_RC_VBR`).
    pub variable_bit_rate: bool,
    /// Whether it takes a quality-defined variable bitrate (`VA_RC_QVBR`).
    pub quality_variable_bit_rate: bool,
    /// Whether its own rate control keeps the HRD buffer it is given.
    /// Where it does not, a stream held to a hard limit
    /// ([`EncoderSettings::hard_limit`]) is coded at a constant QP that
    /// ours picks for each picture ([`super::limit_rate`]).
    pub keeps_buffer: bool,
    /// References each list takes.
    pub max_l0: u32,
    /// References list 1 takes: 0 where the driver has no B pictures.
    pub max_l1: u32,
    /// The surface sizes the encoder takes.
    pub min_width: u32,
    /// The shortest surface it encodes.
    pub min_height: u32,
    /// The widest surface it encodes.
    pub max_width: u32,
    /// The tallest surface it encodes.
    pub max_height: u32,
    /// HEVC's tools, resolved from what the driver allows and requires.
    pub hevc_tools: Option<hevc::Tools>,
}

/// The parameter sets of a stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Headers {
    /// H.264's.
    H264 {
        /// The sequence parameter set.
        sps: h264::Sps,
        /// The picture parameter set.
        pps: h264::Pps,
    },
    /// HEVC's: the VPS is made from the SPS.
    Hevc {
        /// The video and sequence parameter sets.
        sps: hevc::Sps,
        /// The picture parameter set.
        pps: hevc::Pps,
    },
}

/// What a packed header is to the driver (va.h, `VAEncPackedHeaderType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedKind {
    /// The sequence's parameter sets (`VAEncPackedHeaderSequence`).
    Sequence,
    /// A picture's parameter set (`VAEncPackedHeaderPicture`).
    Picture,
    /// A slice header (`VAEncPackedHeaderSlice`).
    Slice,
}

/// A packed header: NAL units in Annex B form, escaped, and their length
/// in bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedUnit {
    /// What the driver is to take it as.
    pub kind: PackedKind,
    /// The bytes: a start code, then the escaped NAL unit.
    pub data: Vec<u8>,
    /// Its length in bits, the start code and any emulation prevention bytes included.
    pub bits: usize,
}

impl PackedUnit {
    /// A whole NAL unit behind a start code.
    fn whole(kind: PackedKind, unit: &[u8]) -> Result<Self, Unsupported> {
        let data = [&nal::START_CODE, unit].concat();
        let bits = (data.len().checked_mul(8))
            .ok_or_else(|| Unsupported::overflow("a packed unit's length"))?;
        Ok(Self { kind, data, bits })
    }

    /// A slice header of `bits` bits in `raw` (`header` bytes of NAL unit
    /// header, then the rest), escaped behind a start code; the length
    /// counts the start code and the emulation prevention bytes added.
    fn slice(raw: &[u8], header: usize, bits: usize) -> Result<Self, Unsupported> {
        let (header, rbsp) = raw.split_at_checked(header).unwrap_or((raw, &[]));
        let data = [&nal::START_CODE, &nal::escape(header, rbsp)[..]].concat();
        // The start code's and the emulation prevention bytes' bits on top
        // of the header's.
        let bits = (data.len().checked_sub(raw.len()))
            .and_then(|added| added.checked_mul(8))
            .and_then(|added| added.checked_add(bits))
            .ok_or_else(|| Unsupported::overflow("a packed slice header's length"))?;
        Ok(Self {
            kind: PackedKind::Slice,
            data,
            bits,
        })
    }
}

/// What the accelerator is configured with, once.
#[derive(Debug)]
pub struct StreamParams<'a> {
    /// The guest's settings for the stream.
    pub settings: &'a EncoderSettings,
    /// The size encoded, before the SPS crops it to the settings' size.
    pub coded_width: u32,
    /// The coded height, a whole number of macroblocks or coding blocks.
    pub coded_height: u32,
    /// Consecutive B pictures the stream has, 0 for none.
    pub b_frames: u32,
    /// References the DPB holds.
    pub max_references: u32,
    /// Frames from one IDR picture to the next, 0 for no limit.
    pub intra_period: u32,
    /// Whether the encoder picks each picture's QP, which its slice header
    /// says ([`super::limit_rate`]): the driver is then configured for a
    /// constant QP, and given no rate control or HRD, whatever the
    /// settings' rate control.
    pub picture_qp: bool,
}

/// A reference picture, as the pictures that predict from it name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReferencePicture {
    /// Its number among the references ([`Planned::reference`]).
    pub number: u64,
    /// Its order count: H.264's `TopFieldOrderCnt`, or HEVC's POC.
    pub poc: i32,
    /// H.264's `frame_num`.
    pub frame_num: u32,
}

/// One picture for the accelerator to encode.
#[derive(Debug)]
pub struct PictureParams<'a> {
    /// The guest's settings, as they are for this picture.
    pub settings: &'a EncoderSettings,
    /// The stream's parameter sets.
    pub headers: &'a Headers,
    /// Whether it is an IDR, P or B picture.
    pub frame_type: FrameType,
    /// Its order count.
    pub poc: i32,
    /// H.264's `frame_num`.
    pub frame_num: u32,
    /// Its number if it is a reference.
    pub reference: Option<u64>,
    /// The reference list 0 predicts from, for a P or B picture.
    pub list0: Option<ReferencePicture>,
    /// The reference list 1 predicts from, for a B picture.
    pub list1: Option<ReferencePicture>,
    /// The references held before it, oldest first.
    pub dpb: &'a [ReferencePicture],
    /// The references held after it, by number.
    pub retained: &'a [u64],
    /// The parameter sets to send before the slice: on IDR pictures.
    pub parameter_sets: &'a [PackedUnit],
    /// The slice header, packed.
    pub slice_header: &'a PackedUnit,
    /// The H.264 slice header, for an H.264 stream.
    pub h264_slice: Option<&'a h264::SliceHeader>,
    /// The HEVC slice header, for an HEVC stream.
    pub hevc_slice: Option<&'a hevc::SliceHeader>,
    /// Whether rate control, frame rate and HRD are to be sent: on the
    /// first picture, and after they change.
    pub rate_control_changed: bool,
}

/// Uploads frames and encodes pictures on the hardware.
pub trait Accelerator {
    /// An uploaded frame, such as a VA surface.
    type Input;
    /// The accelerator's error, which carries the encoder's too.
    type Error: From<Unsupported>;

    /// Prepares for the stream, before the first frame.
    fn configure(&mut self, stream: &StreamParams<'_>) -> Result<(), Self::Error>;

    /// Copies a frame out of host memory.
    fn upload(&mut self, frame: &InputFrame<'_>) -> Result<Self::Input, Self::Error>;

    /// Encodes one picture and returns what the hardware wrote: an Annex B
    /// byte stream of its NAL units, parameter sets included. The picture
    /// is not the stream's until [`Accelerator::keep`]: before it, the
    /// same picture may be encoded again in its place, at another QP.
    fn encode(
        &mut self,
        input: &Self::Input,
        picture: &PictureParams<'_>,
    ) -> Result<Vec<u8>, Self::Error>;

    /// Takes the picture last encoded as the stream's: it becomes a
    /// reference if it is one, and the references it does not retain are
    /// let go.
    fn keep(&mut self, picture: &PictureParams<'_>);
}

/// A frame waiting to be encoded.
#[derive(Debug)]
struct Pending<I> {
    input: I,
    pts: i64,
    duration: i64,
    id: u64,
}

/// What a picture's slice header is made from.
#[derive(Clone, Copy)]
struct SliceAt {
    frame_type: FrameType,
    order: i32,
    frame_num: u32,
    idr_pic_id: u32,
    list0: Option<ReferencePicture>,
    list1: Option<ReferencePicture>,
    qp: u8,
}

/// An H.264 or HEVC encoder over an [`Accelerator`].
pub struct Encoder<A: Accelerator> {
    accelerator: A,
    settings: EncoderSettings,
    headers: Headers,
    parameter_sets: Vec<PackedUnit>,
    coded_size: (u32, u32),
    b_frames: u32,
    max_references: usize,
    /// The configuration record, and the parameter set units in it.
    record: Arc<[u8]>,
    record_units: Vec<Vec<u8>>,
    /// Whether the record goes out with the next sample.
    record_due: bool,
    planner: GopPlanner<Pending<A::Input>>,
    times: DecodeTimes,
    dpb: VecDeque<ReferencePicture>,
    /// H.264's `frame_num` for the next picture, and the next IDR
    /// picture's `idr_pic_id`.
    frame_num: u32,
    idr_pic_id: u32,
    rate_control_changed: bool,
    last_pts: Option<i64>,
    /// A picture failed: the next frame is coded as an IDR picture, since
    /// the references the stream would predict from are gone.
    restart: bool,
    /// Our rate control, where the stream is held to a hard limit that the
    /// driver's own does not keep ([`DriverLimits::keeps_buffer`]).
    limit_rate: Option<LimitRate>,
    /// Pictures kept since the last IDR picture, counting it.
    since_key: u32,
}

/// `value` rounded up to a multiple of `to`, a power of two; none past a
/// u32.
fn align(value: u32, to: u32) -> Option<u32> {
    value.checked_next_multiple_of(to)
}

fn check(condition: bool, why: impl FnOnce() -> String) -> Result<(), Unsupported> {
    if condition {
        Ok(())
    } else {
        Err(Unsupported(why()))
    }
}

/// Checks what settings ask for, the driver aside.
fn check_settings(settings: &EncoderSettings) -> Result<(), Unsupported> {
    let s = settings;
    check(s.width > 0 && s.height > 0, || "an empty picture".into())?;
    check(
        s.width.is_multiple_of(2) && s.height.is_multiple_of(2),
        || format!("{}x{}: 4:2:0 needs an even size", s.width, s.height),
    )?;
    check(
        s.frame_rate.numerator > 0 && s.frame_rate.denominator > 0,
        || "a frame rate of 0".into(),
    )?;
    check(s.timescale > 0, || "a timescale of 0".into())?;
    match s.rate_control {
        RateControl::ConstantQp {
            intra,
            predicted,
            bidirectional,
        } => check(intra.max(predicted).max(bidirectional) <= 51, || {
            "a QP above 51".into()
        }),
        RateControl::ConstantBitRate { bits_per_second } => {
            check(bits_per_second > 0, || "a bitrate of 0".into())
        }
        RateControl::VariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
        }
        | RateControl::QualityVariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
            ..
        } => check(
            average_bits_per_second > 0 && peak_bits_per_second >= average_bits_per_second,
            || "a bitrate of 0, or a peak under the average".into(),
        ),
    }
}

/// The average and the peak a rate control aims at, in bits a second;
/// none for a constant QP.
fn rates(rate_control: RateControl) -> Option<(u32, u32)> {
    match rate_control {
        RateControl::ConstantQp { .. } => None,
        RateControl::ConstantBitRate { bits_per_second } => {
            Some((bits_per_second, bits_per_second))
        }
        RateControl::VariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
        }
        | RateControl::QualityVariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
            ..
        } => Some((average_bits_per_second, peak_bits_per_second)),
    }
}

/// Whether ours picks each picture's QP: where the stream is held to a
/// hard limit by a rate, the driver's rate control does not keep it, and
/// the driver takes a constant QP.
fn picks_qp(settings: &EncoderSettings, limits: &DriverLimits) -> bool {
    settings.hard_limit
        && !limits.keeps_buffer
        && limits.constant_qp
        && rates(settings.rate_control).is_some()
}

/// Our rate control for a stream of `settings` at `coded_size`, where it
/// picks each picture's QP ([`picks_qp`]).
fn limit_rate(
    settings: &EncoderSettings,
    limits: &DriverLimits,
    coded_size: (u32, u32),
) -> Result<Option<LimitRate>, Unsupported> {
    if !picks_qp(settings, limits) {
        return Ok(None);
    }
    let Some((average, peak)) = rates(settings.rate_control) else {
        return Ok(None);
    };
    let pixels = u64::from(coded_size.0)
        .checked_mul(u64::from(coded_size.1))
        .ok_or_else(|| Unsupported::overflow("the coded size's pixels"))?;
    Ok(Some(LimitRate::new(
        pixels,
        settings.frame_rate.as_f64(),
        settings.timescale,
        average,
        hard_limit(settings, peak)?,
    )))
}

/// The bits a stream held to a hard limit may put in any second: its
/// buffer and a second at its peak.
fn hard_limit(settings: &EncoderSettings, peak: u32) -> Result<u64, Unsupported> {
    u64::from(settings.buffer_bits(peak).0)
        .checked_add(u64::from(peak))
        .ok_or_else(|| Unsupported::overflow("the hard limit"))
}

fn rate_control_supported(rate_control: RateControl, limits: &DriverLimits) -> bool {
    match rate_control {
        RateControl::ConstantQp { .. } => limits.constant_qp,
        RateControl::ConstantBitRate { .. } => limits.constant_bit_rate,
        RateControl::VariableBitRate { .. } => limits.variable_bit_rate,
        RateControl::QualityVariableBitRate { .. } => limits.quality_variable_bit_rate,
    }
}

/// `num_units_in_tick` and `time_scale` for a frame rate, `ticks_per_frame`
/// ticks a frame (2 for H.264's VUI, 1 for HEVC's).
fn timing(rate: FrameRate, ticks_per_frame: u32) -> Option<(u32, u32)> {
    let rate = rate.reduced();
    Some((
        rate.denominator,
        rate.numerator.checked_mul(ticks_per_frame)?,
    ))
}

impl<A: Accelerator> Encoder<A> {
    /// An encoder for `settings` on `accelerator`, within what the driver
    /// allows.
    pub fn new(
        mut accelerator: A,
        settings: EncoderSettings,
        limits: &DriverLimits,
    ) -> Result<Self, A::Error> {
        check_settings(&settings)?;
        let s = &settings;
        // Our rate control takes the driver's constant QP alone.
        let ours = picks_qp(s, limits);
        check(
            ours || rate_control_supported(s.rate_control, limits),
            || format!("the driver lacks {:?}", s.rate_control),
        )?;
        check(
            s.width <= limits.max_width && s.height <= limits.max_height,
            || {
                format!(
                    "{}x{} is over the driver's {}x{}",
                    s.width, s.height, limits.max_width, limits.max_height
                )
            },
        )?;
        let reorder = s.allow_frame_reordering
            && s.profile.has_b_slices()
            && limits.max_l1 >= 1
            && limits.max_l0 >= 1;
        let b_frames = if reorder {
            s.b_frames
                .unwrap_or(DEFAULT_B_FRAMES)
                .min(s.max_frame_delay.unwrap_or(u32::MAX))
                .min(MAX_B_FRAMES)
        } else {
            0
        };
        // The references, and with the picture being coded, the pictures
        // the DPB holds.
        let (max_references, dpb_pictures): (u32, u32) = if b_frames > 0 { (2, 3) } else { (1, 2) };
        let peak = s.rate_control.peak_bits_per_second();
        let fps = s.frame_rate.as_f64();
        let intra_period = s.key_frame_interval.unwrap_or(0);
        // The coded size: the settings' raised to the driver's least and
        // rounded up to `to`, and what the SPS crops off it.
        let coded = |to: u32| {
            let size = |value: u32, least: u32| {
                let coded = align(value.max(least), to)?;
                Some((coded, coded.checked_sub(value)?))
            };
            size(s.width, limits.min_width)
                .zip(size(s.height, limits.min_height))
                .ok_or_else(|| Unsupported::overflow("the coded size"))
        };
        let (headers, coded_size) = match s.profile.codec() {
            Codec::H264 => {
                let ((coded_width, crop_right), (coded_height, crop_bottom)) = coded(16)?;
                let needs = h264::LevelNeeds {
                    profile: s.profile,
                    width_in_mbs: coded_width / 16,
                    height_in_mbs: coded_height / 16,
                    frames_per_second: fps,
                    bits_per_second: peak,
                    dpb_frames: max_references,
                };
                let level_idc = match s.level {
                    Some(level) => {
                        check(h264::level_fits(level, &needs), || {
                            format!("H.264 level_idc {level} does not fit {needs:?}")
                        })?;
                        level
                    }
                    None => h264::choose_level(&needs)?,
                };
                let sps = h264::Sps {
                    profile: s.profile,
                    level_idc,
                    max_num_ref_frames: u8::try_from(max_references).unwrap_or(1),
                    width_in_mbs: coded_width / 16,
                    height_in_mbs: coded_height / 16,
                    crop_right,
                    crop_bottom,
                    vui: h264::Vui {
                        sample_aspect_ratio: s.sample_aspect_ratio,
                        full_range: s.full_range,
                        colour: s.colour,
                        timing: timing(s.frame_rate, 2),
                        max_num_reorder_frames: u32::from(b_frames > 0),
                        max_dec_frame_buffering: max_references,
                    },
                };
                let pps = h264::Pps {
                    profile: s.profile,
                    entropy: s.h264_entropy(),
                    pic_init_qp: PPS_QP,
                    transform_8x8_mode_flag: s.profile == Profile::H264High,
                };
                (Headers::H264 { sps, pps }, (coded_width, coded_height))
            }
            Codec::Hevc => {
                let tools = limits
                    .hevc_tools
                    .ok_or_else(|| Unsupported("the driver reports no HEVC tools".into()))?;
                let min_cb = 1u32
                    .checked_shl(u32::from(tools.log2_min_cb_size))
                    .ok_or_else(|| Unsupported::overflow("the least coding block"))?;
                let ((coded_width, crop_right), (coded_height, crop_bottom)) = coded(min_cb)?;
                let needs = hevc::LevelNeeds {
                    width: coded_width,
                    height: coded_height,
                    frames_per_second: fps,
                    bits_per_second: peak,
                    dpb_pictures,
                };
                let level_idc = match s.level {
                    Some(level) => {
                        check(hevc::level_fits(level, &needs), || {
                            format!("HEVC general_level_idc {level} does not fit {needs:?}")
                        })?;
                        level
                    }
                    None => hevc::choose_level(&needs)?,
                };
                let mut tools = tools;
                // cu_qp_delta lets rate control change QP inside a
                // picture; a constant QP has no use for it, nor our rate
                // control, which picks one a picture.
                if ours || matches!(s.rate_control, RateControl::ConstantQp { .. }) {
                    tools.cu_qp_delta = tools.cu_qp_delta && tools.cu_qp_delta_required;
                }
                let sps = hevc::Sps {
                    profile: s.profile,
                    level_idc,
                    width: coded_width,
                    height: coded_height,
                    crop_right,
                    crop_bottom,
                    max_dec_pic_buffering_minus1: max_references,
                    max_num_reorder_pics: u32::from(b_frames > 0),
                    tools,
                    vui: hevc::Vui {
                        sample_aspect_ratio: s.sample_aspect_ratio,
                        full_range: s.full_range,
                        colour: s.colour,
                        timing: timing(s.frame_rate, 1),
                    },
                };
                let pps = hevc::Pps {
                    init_qp: PPS_QP,
                    tools,
                };
                (Headers::Hevc { sps, pps }, (coded_width, coded_height))
            }
        };
        let limit_rate = limit_rate(s, limits, coded_size)?;
        let (parameter_sets, record_units) = parameter_sets(&headers)?;
        let record = record_of(&headers, &record_units)
            .ok_or_else(|| Unsupported("our own SPS did not read back".into()))?;
        accelerator.configure(&StreamParams {
            settings: &settings,
            coded_width: coded_size.0,
            coded_height: coded_size.1,
            b_frames,
            max_references,
            intra_period,
            picture_qp: limit_rate.is_some(),
        })?;
        let planner = GopPlanner::new(GopConfig {
            key_frame_interval: settings.key_frame_interval,
            key_frame_interval_duration: settings.key_frame_interval_duration,
            b_frames,
        });
        Ok(Self {
            accelerator,
            settings,
            headers,
            parameter_sets,
            coded_size,
            b_frames,
            max_references: max_references.to_usize(),
            record: record.into(),
            record_units,
            record_due: true,
            planner,
            times: DecodeTimes::new(),
            dpb: VecDeque::new(),
            frame_num: 0,
            idr_pic_id: 0,
            rate_control_changed: true,
            last_pts: None,
            restart: false,
            limit_rate,
            since_key: 0,
        })
    }

    /// The settings, as changed since.
    pub fn settings(&self) -> &EncoderSettings {
        &self.settings
    }

    /// The configuration record (`avcC` or `hvcC`) of the parameter sets
    /// in use.
    pub fn configuration(&self) -> Arc<[u8]> {
        self.record.clone()
    }

    /// The size the hardware encodes, before cropping.
    pub fn coded_size(&self) -> (u32, u32) {
        self.coded_size
    }

    /// B pictures in a run; 0 when frames keep their order.
    pub fn b_frames(&self) -> u32 {
        self.b_frames
    }

    /// The level the SPS says.
    pub fn level_idc(&self) -> u8 {
        match &self.headers {
            Headers::H264 { sps, .. } => sps.level_idc,
            Headers::Hevc { sps, .. } => sps.level_idc,
        }
    }

    /// The accelerator.
    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    /// The accelerator, to change how it works.
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }

    /// Encodes one frame and returns each frame that comes out, in decode
    /// order, with its own result: none while it waits as a B picture, or
    /// several when it ends a run. `Err` is a frame not taken: one whose
    /// time stamp does not follow the last, or that does not upload.
    pub fn encode(&mut self, frame: &InputFrame<'_>) -> Result<Vec<Coded<A::Error>>, A::Error> {
        if let Some(last) = self.last_pts {
            check(frame.pts > last, || {
                format!("PTS {} does not follow {last}", frame.pts)
            })?;
        }
        let input = self.accelerator.upload(frame)?;
        self.last_pts = Some(frame.pts);
        self.times.push(frame.pts);
        let pending = Pending {
            input,
            pts: frame.pts,
            duration: frame.duration,
            id: frame.id,
        };
        let restart = core::mem::take(&mut self.restart);
        let planned = self
            .planner
            .push(pending, frame.pts, frame.force_key_frame || restart);
        Ok(self.encode_all(planned))
    }

    /// Encodes the frames still waiting and returns each with its own
    /// result. Frames may follow: the next starts a new run.
    pub fn flush(&mut self) -> Vec<Coded<A::Error>> {
        let planned = self.planner.flush();
        self.encode_all(planned)
    }

    /// Encodes the planned pictures in order, each with its own result.
    /// Those coded before one that fails keep their samples. The failed
    /// one has its error, and the rest of the run, which may predict from
    /// it, are lost with it, never coded. Then the references are dropped,
    /// each lost picture's decode time stamp is taken so the next samples'
    /// stay in step, and the next frame is coded as an IDR picture, so that
    /// one failed picture does not fail every picture to the next key
    /// frame.
    fn encode_all(&mut self, planned: Vec<Planned<Pending<A::Input>>>) -> Vec<Coded<A::Error>> {
        let mut coded = Vec::with_capacity(planned.len());
        let mut failed = false;
        for picture in planned {
            let id = picture.frame.id;
            if failed {
                self.times.next_dts();
                coded.push(Err(FailedSample { id, error: None }));
                continue;
            }
            match self.encode_planned(picture) {
                Ok(sample) => coded.push(Ok(sample)),
                Err(error) => {
                    self.times.next_dts();
                    self.dpb.clear();
                    self.restart = true;
                    failed = true;
                    coded.push(Err(FailedSample {
                        id,
                        error: Some(error),
                    }));
                }
            }
        }
        coded
    }

    /// Changes the bitrate or QPs from the next picture. The mode cannot
    /// change: the driver's configuration is made for one.
    pub fn set_rate_control(&mut self, rate_control: RateControl) -> Result<(), A::Error> {
        check(
            core::mem::discriminant(&rate_control)
                == core::mem::discriminant(&self.settings.rate_control),
            || "rate control cannot change mode mid-stream".into(),
        )?;
        let mut settings = self.settings.clone();
        settings.rate_control = rate_control;
        check_settings(&settings)?;
        if let (Some(rate), Some((average, peak))) = (&mut self.limit_rate, rates(rate_control)) {
            // The limit follows the peak, so that a new limit holds from
            // here.
            rate.set_rates(average, hard_limit(&settings, peak)?);
        }
        self.settings = settings;
        self.rate_control_changed = true;
        Ok(())
    }

    /// Changes the expected frame rate that rate control budgets by. The
    /// SPS keeps its timing, which says frames may come at any rate.
    pub fn set_frame_rate(&mut self, frame_rate: FrameRate) -> Result<(), A::Error> {
        check(
            frame_rate.numerator > 0 && frame_rate.denominator > 0,
            || "a frame rate of 0".into(),
        )?;
        self.settings.frame_rate = frame_rate;
        if let Some(rate) = &mut self.limit_rate {
            rate.set_frame_rate(frame_rate.as_f64());
        }
        self.rate_control_changed = true;
        Ok(())
    }

    /// Changes the key-frame interval from the next frame.
    pub fn set_key_frame_interval(&mut self, frames: Option<u32>, duration: Option<i64>) {
        self.settings.key_frame_interval = frames;
        self.settings.key_frame_interval_duration = duration;
        self.planner.set_key_frame_interval(frames, duration);
    }

    fn reference(&self, number: Option<u64>) -> Result<Option<ReferencePicture>, Unsupported> {
        match number {
            None => Ok(None),
            Some(number) => self
                .dpb
                .iter()
                .find(|r| r.number == number)
                .copied()
                .map(Some)
                .ok_or_else(|| Unsupported(format!("reference {number} is not held"))),
        }
    }

    fn qp(&self, frame_type: FrameType) -> u8 {
        match self.settings.rate_control {
            RateControl::ConstantQp {
                intra,
                predicted,
                bidirectional,
            } => match frame_type {
                FrameType::Idr => intra,
                FrameType::P => predicted,
                FrameType::B => bidirectional,
            },
            _ => PPS_QP,
        }
    }

    /// Pictures to the next planned key frame, counting the next: by the
    /// interval in frames, or in time at the expected frame rate, whichever
    /// comes first; none where neither is set.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ticks as a float, and a count of frames clamped to a u32's range"
    )]
    fn remaining(&self) -> Option<u32> {
        let s = &self.settings;
        let by_time = s.key_frame_interval_duration.map(|ticks| {
            let seconds = ticks as f64 / f64::from(s.timescale);
            (seconds * s.frame_rate.as_f64())
                .floor()
                .clamp(1.0, f64::from(u32::MAX)) as u32
        });
        let interval = match (s.key_frame_interval, by_time) {
            (Some(frames), Some(time)) => frames.min(time),
            (frames, time) => frames.or(time)?,
        };
        Some(interval.saturating_sub(self.since_key).max(1))
    }

    /// A picture's order count and slice header, at `at`'s QP.
    fn slice_header(
        &self,
        at: &SliceAt,
    ) -> Result<
        (
            i32,
            Option<h264::SliceHeader>,
            Option<hevc::SliceHeader>,
            PackedUnit,
        ),
        Unsupported,
    > {
        let SliceAt {
            frame_type,
            order,
            frame_num,
            idr_pic_id,
            list0,
            list1,
            qp,
        } = *at;
        // A QP less the PPS's: two u8s' difference fits an i32.
        let qp_delta = |init: u8| {
            i32::from(qp)
                .checked_sub(i32::from(init))
                .ok_or_else(|| Unsupported::overflow("slice_qp_delta"))
        };
        match &self.headers {
            Headers::H264 { pps, .. } => {
                // TopFieldOrderCnt, and BottomFieldOrderCnt with it: within
                // 32 bits at gop::MAX_ORDER.
                let poc = order
                    .checked_mul(2)
                    .ok_or_else(|| Unsupported::overflow("TopFieldOrderCnt"))?;
                let slice = h264::SliceHeader {
                    slice_type: match frame_type {
                        FrameType::Idr => h264::SliceType::I,
                        FrameType::P => h264::SliceType::P,
                        FrameType::B => h264::SliceType::B,
                    },
                    idr: frame_type == FrameType::Idr,
                    nal_ref_idc: match frame_type {
                        FrameType::Idr => 3,
                        FrameType::P => 2,
                        FrameType::B => 0,
                    },
                    frame_num,
                    idr_pic_id,
                    pic_order_cnt_lsb: poc.cast_unsigned()
                        & ((1 << (h264::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4 + 4)) - 1),
                    slice_qp_delta: qp_delta(pps.pic_init_qp)?,
                };
                let (raw, bits) = slice.to_bits(pps)?;
                let packed = PackedUnit::slice(&raw, 1, bits)?;
                Ok((poc, Some(slice), None, packed))
            }
            Headers::Hevc { sps, pps } => {
                let poc = order;
                let slice = hevc::SliceHeader {
                    nal_unit_type: if frame_type == FrameType::Idr {
                        hevc::nal_type::IDR_W_RADL
                    } else {
                        hevc::nal_type::TRAIL_R
                    },
                    slice_type: match frame_type {
                        FrameType::Idr => hevc::SliceType::I,
                        FrameType::P => hevc::SliceType::P,
                        FrameType::B => hevc::SliceType::B,
                    },
                    pic_order_cnt_lsb: poc.cast_unsigned()
                        & ((1 << (hevc::LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4 + 4)) - 1),
                    references: [list0, list1]
                        .iter()
                        .flatten()
                        .map(|r| r.poc.checked_sub(poc))
                        .collect::<Option<_>>()
                        .ok_or_else(|| Unsupported::overflow("a reference's POC distance"))?,
                    slice_qp_delta: qp_delta(pps.init_qp)?,
                    max_num_merge_cand: MAX_NUM_MERGE_CAND,
                };
                let raw = slice.to_bytes(sps)?;
                // byte_alignment() ends it on a byte.
                let bits = (raw.len().checked_mul(8))
                    .ok_or_else(|| Unsupported::overflow("the slice header's length"))?;
                let packed = PackedUnit::slice(&raw, 2, bits)?;
                Ok((poc, None, Some(slice), packed))
            }
        }
    }

    fn encode_planned(
        &mut self,
        planned: Planned<Pending<A::Input>>,
    ) -> Result<EncodedSample, A::Error> {
        let Planned {
            frame,
            frame_type,
            order,
            reference,
            forward,
            backward,
        } = planned;
        if frame_type == FrameType::Idr {
            self.dpb.clear();
            self.frame_num = 0;
            self.since_key = 0;
        }
        let list0 = self.reference(forward)?;
        let list1 = self.reference(backward)?;
        let dts = self.times.peek();
        let mut qp = match &self.limit_rate {
            Some(rate) => rate.plan(frame_type, self.remaining(), dts),
            None => self.qp(frame_type),
        };
        // At most gop::MAX_ORDER, so within an i32.
        let order =
            i32::try_from(order).map_err(|_| Unsupported::overflow("a picture's order count"))?;
        let frame_num = self.frame_num;
        let idr_pic_id = self.idr_pic_id;
        let slice = |qp| SliceAt {
            frame_type,
            order,
            frame_num,
            idr_pic_id,
            list0,
            list1,
            qp,
        };
        let poc = self.slice_header(&slice(qp))?.0;
        let dpb: Vec<ReferencePicture> = self.dpb.iter().copied().collect();
        let mut after = self.dpb.clone();
        let this = reference.map(|number| ReferencePicture {
            number,
            poc,
            frame_num,
        });
        if let Some(this) = this {
            after.push_back(this);
            while after.len() > self.max_references {
                after.pop_front();
            }
        }
        let retained: Vec<u64> = after.iter().map(|r| r.number).collect();
        let parameter_sets: &[PackedUnit] = if frame_type == FrameType::Idr {
            &self.parameter_sets
        } else {
            &[]
        };
        // The tries after the first.
        let mut retries = 1..LIMIT_TRIES;
        let coded = loop {
            let (_, h264_slice, hevc_slice, slice_header) = self.slice_header(&slice(qp))?;
            let picture = PictureParams {
                settings: &self.settings,
                headers: &self.headers,
                frame_type,
                poc,
                frame_num,
                reference,
                list0,
                list1,
                dpb: &dpb,
                retained: &retained,
                parameter_sets,
                slice_header: &slice_header,
                h264_slice: h264_slice.as_ref(),
                hevc_slice: hevc_slice.as_ref(),
                rate_control_changed: self.rate_control_changed,
            };
            let coded = self.accelerator.encode(&frame.input, &picture)?;
            if let Some(rate) = &mut self.limit_rate {
                // The access unit's bits, parameter sets and start codes
                // included, which a sample's length prefixes match.
                let bits = bits_of(&coded);
                if bits > rate.allowance(dts)
                    && retries.next().is_some()
                    && let Some(next) = rate.retry(frame_type, qp, bits, dts)
                {
                    qp = next;
                    continue;
                }
                rate.coded(frame_type, qp, bits, dts);
            }
            self.accelerator.keep(&picture);
            break coded;
        };
        self.rate_control_changed = false;
        self.since_key = self.since_key.saturating_add(1);
        self.dpb = after;
        // frame_num counts references modulo MaxFrameNum (7.4.3), and
        // idr_pic_id goes round its range, 0 to 65535: each is under its
        // modulus, so the add never wraps.
        if this.is_some() {
            self.frame_num =
                frame_num.wrapping_add(1) % (1 << (h264::LOG2_MAX_FRAME_NUM_MINUS4 + 4));
        }
        if frame_type == FrameType::Idr {
            self.idr_pic_id = idr_pic_id.wrapping_add(1) % 65_536;
        }
        let data = self.sample_data(&coded, frame_type == FrameType::Idr)?;
        let configuration = self.record_due.then(|| self.record.clone());
        self.record_due = false;
        Ok(EncodedSample {
            id: frame.id,
            data,
            pts: frame.pts,
            dts: self.times.next_dts(),
            duration: frame.duration,
            frame_type,
            is_sync: frame_type == FrameType::Idr,
            depends_on_others: frame_type != FrameType::Idr,
            is_depended_on: reference.is_some(),
            configuration,
        })
    }

    /// The sample's NAL units, length-prefixed, without parameter sets and
    /// delimiters. Parameter sets the hardware wrote that differ from the
    /// record's replace it, and the new record goes with this sample.
    fn sample_data(&mut self, coded: &[u8], idr: bool) -> Result<Vec<u8>, Unsupported> {
        let codec = self.settings.profile.codec();
        let mut data = Vec::with_capacity(coded.len());
        let mut seen: Vec<Vec<u8>> = Vec::new();
        let mut sliced = false;
        for unit in nal::split_annex_b(coded) {
            match unit_role(codec, unit) {
                Role::ParameterSet(index) => {
                    while seen.len() <= index {
                        seen.push(Vec::new());
                    }
                    if let Some(set) = seen.get_mut(index) {
                        *set = unit.to_vec();
                    }
                }
                Role::Delimiter => {}
                Role::Slice => {
                    sliced = true;
                    nal::push_length_prefixed(&mut data, unit);
                }
                Role::Other => nal::push_length_prefixed(&mut data, unit),
            }
        }
        check(sliced, || "the hardware wrote no slice".into())?;
        if idr
            && !seen.is_empty()
            && seen.iter().all(|u| !u.is_empty())
            && seen != self.record_units
        {
            let record = record_of(&self.headers, &seen)
                .ok_or_else(|| Unsupported("the hardware's parameter sets do not read".into()))?;
            self.record = record.into();
            self.record_units = seen;
            self.record_due = true;
        }
        Ok(data)
    }
}

/// The bits of `coded`, as a float, as the rate control counts them.
#[expect(clippy::as_conversions, reason = "a length as a float")]
fn bits_of(coded: &[u8]) -> f64 {
    coded.len() as f64 * 8.0
}

/// What a NAL unit is to a sample.
enum Role {
    /// A parameter set, by its index in the record's order (VPS, SPS, PPS
    /// for HEVC; SPS, PPS for H.264).
    ParameterSet(usize),
    Delimiter,
    Slice,
    Other,
}

fn unit_role(codec: Codec, unit: &[u8]) -> Role {
    let first = unit.first().copied().unwrap_or(0);
    match codec {
        Codec::H264 => match first & 0x1f {
            h264::nal_type::SPS => Role::ParameterSet(0),
            h264::nal_type::PPS => Role::ParameterSet(1),
            h264::nal_type::AUD => Role::Delimiter,
            1..=5 => Role::Slice,
            _ => Role::Other,
        },
        Codec::Hevc => match (first >> 1) & 0x3f {
            hevc::nal_type::VPS => Role::ParameterSet(0),
            hevc::nal_type::SPS => Role::ParameterSet(1),
            hevc::nal_type::PPS => Role::ParameterSet(2),
            hevc::nal_type::AUD => Role::Delimiter,
            0..=31 => Role::Slice,
            _ => Role::Other,
        },
    }
}

/// The packed parameter sets of a stream, and its units for the record.
fn parameter_sets(headers: &Headers) -> Result<(Vec<PackedUnit>, Vec<Vec<u8>>), Unsupported> {
    let units: Vec<(PackedKind, Vec<u8>)> = match headers {
        Headers::H264 { sps, pps } => vec![
            (PackedKind::Sequence, sps.to_nal()?),
            (PackedKind::Picture, pps.to_nal()?),
        ],
        Headers::Hevc { sps, pps } => vec![
            (PackedKind::Sequence, sps.vps_nal()),
            (PackedKind::Sequence, sps.sps_nal()?),
            (PackedKind::Picture, pps.to_nal()?),
        ],
    };
    let packed = units
        .iter()
        .map(|(kind, unit)| PackedUnit::whole(*kind, unit))
        .collect::<Result<_, _>>()?;
    Ok((packed, units.into_iter().map(|(_, unit)| unit).collect()))
}

/// The configuration record of parameter set units in the record's order.
fn record_of(headers: &Headers, units: &[Vec<u8>]) -> Option<Vec<u8>> {
    match (headers, units) {
        (Headers::H264 { .. }, [sps, pps]) => Some(h264::avcc(sps, pps)),
        (Headers::Hevc { .. }, [vps, sps, pps]) => hevc::hvcc(vps, sps, pps),
        _ => None,
    }
}

/// The encoder's tests, and their fake accelerator.
#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::encode::gop::MAX_ORDER;
    use crate::encode::{ColourDescription, Plane};

    /// Records what it is asked, and writes what a driver that honours the
    /// packed headers would: the parameter sets, then a slice of the
    /// packed header's bytes and some data.
    /// A picture as the fake saw it: type, POC, the POCs of its two
    /// references, the references held after it, and whether rate control
    /// went with it.
    type Seen = (FrameType, i32, Option<i32>, Option<i32>, Vec<u64>, bool);

    /// An accelerator that records each picture it is given (module doc).
    #[derive(Default)]
    pub(crate) struct Fake {
        uploads: u64,
        pictures: Vec<Seen>,
        /// Replaces the first parameter set the hardware writes.
        own_sps: Option<Vec<u8>>,
        /// The encode calls so far, and the one that fails, counted from 0.
        calls: usize,
        fail_call: Option<usize>,
        /// The bytes of slice data a picture takes, by its type and QP;
        /// none for five.
        sizes: Option<fn(FrameType, u8) -> usize>,
        /// Each kept picture's QP, as its slice header says.
        qps: Vec<u8>,
        /// Each kept picture's `pic_order_cnt_lsb`, as its slice header
        /// says.
        lsbs: Vec<u32>,
        /// Whether the stream was configured for QPs the encoder picks.
        picture_qp: bool,
    }

    impl Accelerator for Fake {
        type Input = u64;
        type Error = Unsupported;

        fn configure(&mut self, stream: &StreamParams<'_>) -> Result<(), Unsupported> {
            self.picture_qp = stream.picture_qp;
            Ok(())
        }

        fn upload(&mut self, _: &InputFrame<'_>) -> Result<u64, Unsupported> {
            self.uploads += 1;
            Ok(self.uploads)
        }

        fn encode(&mut self, _: &u64, p: &PictureParams<'_>) -> Result<Vec<u8>, Unsupported> {
            self.calls += 1;
            if self.fail_call == Some(self.calls - 1) {
                return Err(Unsupported("the picture fails".into()));
            }
            let mut out = Vec::new();
            for (n, unit) in p.parameter_sets.iter().enumerate() {
                match (&self.own_sps, n) {
                    (Some(sps), 0) => {
                        out.extend_from_slice(&nal::START_CODE);
                        out.extend_from_slice(sps);
                    }
                    _ => out.extend_from_slice(&unit.data),
                }
            }
            out.extend_from_slice(&p.slice_header.data[..p.slice_header.bits.div_ceil(8)]);
            match self.sizes {
                Some(size) => out.resize(out.len() + size(p.frame_type, qp_of(p)), 0x55),
                None => out.extend_from_slice(&[0x55, 0xaa, 0x80, 0, 0]),
            }
            Ok(out)
        }

        fn keep(&mut self, p: &PictureParams<'_>) {
            self.pictures.push((
                p.frame_type,
                p.poc,
                p.list0.map(|r| r.poc),
                p.list1.map(|r| r.poc),
                p.retained.to_vec(),
                p.rate_control_changed,
            ));
            self.qps.push(qp_of(p));
            let lsb = match (p.h264_slice, p.hevc_slice) {
                (Some(slice), _) => slice.pic_order_cnt_lsb,
                (_, Some(slice)) => slice.pic_order_cnt_lsb,
                _ => 0,
            };
            self.lsbs.push(lsb);
        }
    }

    /// A picture's QP, as its slice header says.
    fn qp_of(p: &PictureParams<'_>) -> u8 {
        let qp = match (p.headers, p.h264_slice, p.hevc_slice) {
            (Headers::H264 { pps, .. }, Some(slice), _) => {
                i32::from(pps.pic_init_qp) + slice.slice_qp_delta
            }
            (Headers::Hevc { pps, .. }, _, Some(slice)) => {
                i32::from(pps.init_qp) + slice.slice_qp_delta
            }
            _ => 0,
        };
        u8::try_from(qp).unwrap_or(0)
    }

    /// A driver like radeonsi: a constant QP, CBR and VBR but not QVBR, two
    /// references in list 0 and one in list 1, 128x128 to 4096x4096.
    pub(crate) fn limits() -> DriverLimits {
        DriverLimits {
            constant_qp: true,
            constant_bit_rate: true,
            variable_bit_rate: true,
            quality_variable_bit_rate: false,
            keeps_buffer: false,
            max_l0: 2,
            max_l1: 1,
            min_width: 128,
            min_height: 128,
            max_width: 4096,
            max_height: 4096,
            hevc_tools: Some(hevc::Tools {
                log2_min_cb_size: 3,
                log2_ctb_size: 6,
                log2_min_tb_size: 2,
                log2_max_tb_size: 5,
                max_transform_hierarchy_depth_inter: 0,
                max_transform_hierarchy_depth_intra: 0,
                amp: true,
                sao: true,
                strong_intra_smoothing: true,
                transform_skip: true,
                cu_qp_delta: true,
                cu_qp_delta_required: false,
                constrained_intra_pred: false,
            }),
        }
    }

    fn frame(n: u32, force: bool) -> InputFrame<'static> {
        InputFrame {
            planes: [
                Plane {
                    data: &[],
                    bytes_per_row: 0,
                },
                Plane {
                    data: &[],
                    bytes_per_row: 0,
                },
            ],
            pts: i64::from(n) * 3000,
            duration: 3000,
            force_key_frame: force,
            id: u64::from(n),
        }
    }

    /// The samples of frames that all encoded, or the first error.
    fn samples(coded: Vec<Coded<Unsupported>>) -> Result<Vec<EncodedSample>, Unsupported> {
        coded
            .into_iter()
            .map(|coded| {
                coded.map_err(|failed| {
                    failed
                        .error
                        .unwrap_or_else(|| Unsupported(format!("frame {} was lost", failed.id)))
                })
            })
            .collect()
    }

    fn run(
        settings: EncoderSettings,
        frames: u32,
    ) -> Result<(Vec<EncodedSample>, Fake), Unsupported> {
        let mut encoder = Encoder::new(Fake::default(), settings, &limits())?;
        let mut coded = Vec::new();
        for n in 0..frames {
            coded.extend(encoder.encode(&frame(n, n == 7))?);
        }
        coded.extend(encoder.flush());
        let Encoder { accelerator, .. } = encoder;
        Ok((samples(coded)?, accelerator))
    }

    /// Each frame that came out, by its tag: `Ok` with its sample, or its
    /// failure, `Some` with its own error and `None` for one lost with it.
    type Outcome = (u64, Result<(), Option<Unsupported>>);

    fn outcomes(coded: &[Coded<Unsupported>]) -> Vec<Outcome> {
        coded
            .iter()
            .map(|coded| match coded {
                Ok(sample) => (sample.id, Ok(())),
                Err(failed) => (failed.id, Err(failed.error.clone())),
            })
            .collect()
    }

    #[test]
    fn h264_with_b_pictures_comes_out_in_decode_order() -> Result<(), Unsupported> {
        let mut settings = EncoderSettings::new(Profile::H264High, 1920, 1080);
        settings.key_frame_interval = Some(30);
        settings.colour = Some(ColourDescription {
            primaries: 1,
            transfer: 1,
            matrix: 1,
        });
        let (samples, fake) = run(settings, 10)?;
        let order: Vec<u64> = samples.iter().map(|s| s.id).collect();
        // One B picture a run; frame 7 is forced, with nothing waiting.
        assert_eq!(order, [0, 2, 1, 4, 3, 6, 5, 7, 9, 8]);
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        use FrameType::{B, Idr, P};
        assert_eq!(types, [Idr, P, B, P, B, P, B, Idr, P, B]);
        // Each DTS is the PTS of the frame sent in its place.
        let dts: Vec<i64> = samples.iter().map(|s| s.dts).collect();
        assert_eq!(dts, (0..10).map(|n| n * 3000).collect::<Vec<_>>());
        assert!(samples[2].dts > samples[2].pts);
        for s in &samples {
            assert_eq!(s.is_sync, s.frame_type == Idr);
            assert_eq!(s.is_depended_on, s.frame_type != B);
            assert_eq!(s.depends_on_others, s.frame_type != Idr);
            // The sample holds the slice alone, length-prefixed.
            let units = nal::length_prefixed_units(&s.data).unwrap_or_default();
            assert_eq!(units.len(), 1);
        }
        assert!(samples[0].configuration.is_some());
        assert!(samples[1..].iter().all(|s| s.configuration.is_none()));
        // B picture 1 predicts from 0 and 2 (POCs 0 and 4).
        assert_eq!(fake.pictures[2].1, 2);
        assert_eq!(fake.pictures[2].2, Some(0));
        assert_eq!(fake.pictures[2].3, Some(4));
        // Rate control goes with the first picture only.
        assert!(fake.pictures[0].5);
        assert!(fake.pictures[1..].iter().all(|p| !p.5));
        // Two references at most are held.
        assert!(fake.pictures.iter().all(|p| p.4.len() <= 2));
        Ok(())
    }

    #[test]
    fn a_failed_picture_restarts_the_stream_at_the_next_frame() -> Result<(), Unsupported> {
        use FrameType::{B, Idr, P};
        let failed = || Err(Some(Unsupported("the picture fails".into())));
        // P pictures only: picture 3 fails, and comes out with its error.
        // The frame after it is coded as an IDR picture, where it would
        // have been a P picture predicting from the lost one.
        let mut settings = EncoderSettings::new(Profile::H264Main, 640, 360);
        settings.key_frame_interval = Some(30);
        settings.allow_frame_reordering = false;
        let fake = Fake {
            fail_call: Some(3),
            ..Fake::default()
        };
        let mut encoder = Encoder::new(fake, settings, &limits())?;
        let mut coded = Vec::new();
        for n in 0..7 {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        let ok = || Ok(());
        assert_eq!(
            outcomes(&coded),
            [
                (0, ok()),
                (1, ok()),
                (2, ok()),
                (3, failed()),
                (4, ok()),
                (5, ok()),
                (6, ok())
            ]
        );
        let samples: Vec<EncodedSample> = coded.into_iter().filter_map(Result::ok).collect();
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        assert_eq!(types, [Idr, P, P, Idr, P, P]);
        assert!(samples.iter().all(|s| s.dts == s.pts));
        // With B pictures: the run of P picture 2 and B picture 1 fails at
        // its P picture, which has its error, and B picture 1, which
        // predicts from it, is lost with it. Frame 3 restarts, and the
        // decode time stamps after it stay those of a stream without the
        // two.
        let mut settings = EncoderSettings::new(Profile::H264High, 640, 360);
        settings.key_frame_interval = Some(30);
        settings.allow_frame_reordering = true;
        let fake = Fake {
            fail_call: Some(1),
            ..Fake::default()
        };
        let mut encoder = Encoder::new(fake, settings.clone(), &limits())?;
        let mut coded = Vec::new();
        for n in 0..7 {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        coded.extend(encoder.flush());
        assert_eq!(
            outcomes(&coded),
            [
                (0, ok()),
                (2, failed()),
                (1, Err(None)),
                (3, ok()),
                (5, ok()),
                (4, ok()),
                (6, ok())
            ]
        );
        let samples: Vec<EncodedSample> = coded.into_iter().filter_map(Result::ok).collect();
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        assert_eq!(types, [Idr, Idr, P, B, P]);
        // Each lost picture's DTS is taken, so each sample's is still the
        // PTS of the frame sent in its place: frame 3's is its own, and B
        // picture 4's is frame 5's.
        let dts: Vec<i64> = samples.iter().map(|s| s.dts).collect();
        assert_eq!(dts, [0, 9000, 12000, 15000, 18000]);
        // The run's B picture fails after its P picture is coded: the P
        // picture keeps its sample, where the one result of the run lost it
        // too.
        let fake = Fake {
            fail_call: Some(2),
            ..Fake::default()
        };
        let mut encoder = Encoder::new(fake, settings, &limits())?;
        let mut coded = Vec::new();
        for n in 0..7 {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        coded.extend(encoder.flush());
        assert_eq!(
            outcomes(&coded),
            [
                (0, ok()),
                (2, ok()),
                (1, failed()),
                (3, ok()),
                (5, ok()),
                (4, ok()),
                (6, ok())
            ]
        );
        let samples: Vec<EncodedSample> = coded.into_iter().filter_map(Result::ok).collect();
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        assert_eq!(types, [Idr, P, Idr, P, B, P]);
        let dts: Vec<i64> = samples.iter().map(|s| s.dts).collect();
        assert_eq!(dts, [0, 3000, 9000, 12000, 15000, 18000]);
        Ok(())
    }

    #[test]
    fn the_record_follows_the_hardwares_parameter_sets() -> Result<(), Unsupported> {
        let settings = EncoderSettings::new(Profile::H264Main, 640, 360);
        let mut encoder = Encoder::new(Fake::default(), settings.clone(), &limits())?;
        let ours = encoder.configuration();
        let first = samples(encoder.encode(&frame(0, false))?)?;
        assert_eq!(first[0].configuration.as_deref(), Some(&ours[..]));
        // A driver that writes its own SPS: the record is made of it.
        let mut other = settings;
        other.level = Some(51);
        let Headers::H264 { sps, .. } = Encoder::new(Fake::default(), other, &limits())?.headers
        else {
            return Err(Unsupported("not H.264".into()));
        };
        let fake = Fake {
            own_sps: Some(sps.to_nal()?),
            ..Fake::default()
        };
        let mut encoder = Encoder::new(
            fake,
            EncoderSettings::new(Profile::H264Main, 640, 360),
            &limits(),
        )?;
        let first = samples(encoder.encode(&frame(0, false))?)?;
        let record = first[0].configuration.clone().unwrap_or_default();
        assert_eq!(record[3], 51);
        assert_eq!(&encoder.configuration()[..], &record[..]);
        Ok(())
    }

    #[test]
    fn hevc_predicts_from_the_last_picture_in_order() -> Result<(), Unsupported> {
        let mut settings = EncoderSettings::new(Profile::HevcMain10, 100, 50);
        settings.key_frame_interval = Some(4);
        let mut encoder = Encoder::new(Fake::default(), settings.clone(), &limits())?;
        // Below the driver's smallest surface: coded at 128x128, cropped.
        assert_eq!(encoder.coded_size(), (128, 128));
        assert_eq!(encoder.b_frames(), 0);
        drop(encoder.encode(&frame(0, false))?);
        let (samples, fake) = run(settings, 6)?;
        let order: Vec<u64> = samples.iter().map(|s| s.id).collect();
        assert_eq!(order, [0, 1, 2, 3, 4, 5]);
        assert!(samples.iter().all(|s| s.dts == s.pts));
        assert_eq!(fake.pictures[1].2, Some(0));
        assert_eq!(fake.pictures[3].2, Some(2));
        assert_eq!(fake.pictures[4].0, FrameType::Idr);
        assert_eq!(fake.pictures[5].2, Some(0));
        Ok(())
    }

    #[test]
    fn h264_order_counts_stay_within_32_bits() -> Result<(), Unsupported> {
        use FrameType::{B, Idr, P};
        let mut settings = EncoderSettings::new(Profile::H264High, 640, 360);
        settings.key_frame_interval = None;
        let mut encoder = Encoder::new(Fake::default(), settings, &limits())?;
        assert_eq!(encoder.b_frames(), 1);
        let mut coded = encoder.encode(&frame(0, false))?;
        encoder.planner.skip_to(MAX_ORDER - 2);
        for n in 1..6 {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        coded.extend(encoder.flush());
        let samples = samples(coded)?;
        let Encoder { accelerator, .. } = encoder;
        // Frame 3 has the highest order, and frame 4, which would pass it,
        // is an IDR picture: I0 B1 P2 P3 I4 P5 in display order.
        let order: Vec<u64> = samples.iter().map(|s| s.id).collect();
        assert_eq!(order, [0, 2, 1, 3, 4, 5]);
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        assert_eq!(types, [Idr, P, B, P, Idr, P]);
        let pocs: Vec<i32> = accelerator.pictures.iter().map(|p| p.1).collect();
        let top = i32::MAX - 1;
        assert_eq!(pocs, [0, top - 2, top - 4, top, 0, 2]);
        assert_eq!(accelerator.lsbs, [0, 252, 250, 254, 0, 2]);
        Ok(())
    }

    #[test]
    fn hevc_order_counts_end_where_h264s_do() -> Result<(), Unsupported> {
        use FrameType::{Idr, P};
        let mut settings = EncoderSettings::new(Profile::HevcMain, 640, 360);
        settings.key_frame_interval = None;
        let mut encoder = Encoder::new(Fake::default(), settings, &limits())?;
        let mut coded = encoder.encode(&frame(0, false))?;
        encoder.planner.skip_to(MAX_ORDER - 1);
        for n in 1..5 {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        let samples = samples(coded)?;
        let Encoder { accelerator, .. } = encoder;
        let types: Vec<FrameType> = samples.iter().map(|s| s.frame_type).collect();
        assert_eq!(types, [Idr, P, P, Idr, P]);
        let pocs: Vec<i32> = accelerator.pictures.iter().map(|p| p.1).collect();
        let top = i32::try_from(MAX_ORDER).unwrap_or(0);
        assert_eq!(pocs, [0, top - 1, top, 0, 1]);
        assert_eq!(accelerator.lsbs, [0, 254, 255, 0, 1]);
        // The picture after the IDR picture predicts from it.
        assert_eq!(accelerator.pictures[4].2, Some(0));
        Ok(())
    }

    #[test]
    fn what_the_driver_lacks_is_refused() {
        let mut settings = EncoderSettings::new(Profile::H264Main, 1920, 1080);
        settings.rate_control = RateControl::QualityVariableBitRate {
            average_bits_per_second: 1_000_000,
            peak_bits_per_second: 2_000_000,
            quality: 25,
        };
        assert!(Encoder::new(Fake::default(), settings, &limits()).is_err());
        let settings = EncoderSettings::new(Profile::H264Main, 1921, 1080);
        assert!(Encoder::new(Fake::default(), settings, &limits()).is_err());
        let mut settings = EncoderSettings::new(Profile::H264Main, 1920, 1080);
        settings.level = Some(30);
        assert!(Encoder::new(Fake::default(), settings, &limits()).is_err());
    }

    #[test]
    fn frame_delay_and_reordering_bound_b_pictures() -> Result<(), Unsupported> {
        let mut settings = EncoderSettings::new(Profile::H264Main, 640, 360);
        settings.b_frames = Some(3);
        assert_eq!(
            Encoder::new(Fake::default(), settings.clone(), &limits())?.b_frames(),
            3
        );
        settings.max_frame_delay = Some(2);
        assert_eq!(
            Encoder::new(Fake::default(), settings.clone(), &limits())?.b_frames(),
            2
        );
        settings.max_frame_delay = Some(0);
        assert_eq!(
            Encoder::new(Fake::default(), settings.clone(), &limits())?.b_frames(),
            0
        );
        settings.max_frame_delay = None;
        settings.allow_frame_reordering = false;
        assert_eq!(
            Encoder::new(Fake::default(), settings.clone(), &limits())?.b_frames(),
            0
        );
        settings.allow_frame_reordering = true;
        settings.profile = Profile::H264ConstrainedBaseline;
        assert_eq!(
            Encoder::new(Fake::default(), settings, &limits())?.b_frames(),
            0
        );
        Ok(())
    }

    #[test]
    fn rate_control_changes_go_with_the_next_picture() -> Result<(), Unsupported> {
        let settings = EncoderSettings::new(Profile::H264Main, 640, 360);
        let mut encoder = Encoder::new(Fake::default(), settings, &limits())?;
        drop(encoder.encode(&frame(0, false))?);
        assert!(
            encoder
                .set_rate_control(RateControl::ConstantQp {
                    intra: 20,
                    predicted: 22,
                    bidirectional: 24,
                })
                .is_err()
        );
        encoder.set_rate_control(RateControl::VariableBitRate {
            average_bits_per_second: 2_000_000,
            peak_bits_per_second: 3_000_000,
        })?;
        for n in 1..4 {
            drop(encoder.encode(&frame(n, false))?);
        }
        let changed: Vec<bool> = encoder.accelerator().pictures.iter().map(|p| p.5).collect();
        assert_eq!(changed, [true, true, false]);
        assert!(
            encoder.encode(&frame(2, false)).is_err(),
            "a PTS going back"
        );
        Ok(())
    }

    /// A picture's bytes in the hard limit's tests: key frames 400,000 at
    /// QP 26 and P pictures 1,000, each 6 QP halving them.
    fn costly_key_frames(frame_type: FrameType, qp: u8) -> usize {
        let at_26 = if frame_type == FrameType::Idr {
            400_000.0
        } else {
            1_000.0
        };
        (at_26 * ((26.0 - f64::from(qp)) / 6.0).exp2()) as usize
    }

    /// Settings held to a limit of 450 kbit/s: a peak of 300 kbit/s, and
    /// half a second's buffer at it.
    fn limited() -> EncoderSettings {
        let mut settings = EncoderSettings::new(Profile::H264Main, 640, 360);
        settings.rate_control = RateControl::VariableBitRate {
            average_bits_per_second: 300_000,
            peak_bits_per_second: 300_000,
        };
        settings.buffer_milliseconds = 500;
        settings.hard_limit = true;
        settings.key_frame_interval = Some(30);
        settings.allow_frame_reordering = false;
        settings
    }

    /// `frames` frames of 30 a second through a fake whose pictures cost
    /// [`costly_key_frames`].
    fn run_costly(
        settings: EncoderSettings,
        limits: &DriverLimits,
        frames: u32,
    ) -> Result<(Vec<EncodedSample>, Fake), Unsupported> {
        let fake = Fake {
            sizes: Some(costly_key_frames),
            ..Fake::default()
        };
        let mut encoder = Encoder::new(fake, settings, limits)?;
        let mut coded = Vec::new();
        for n in 0..frames {
            coded.extend(encoder.encode(&frame(n, false))?);
        }
        coded.extend(encoder.flush());
        let Encoder { accelerator, .. } = encoder;
        Ok((samples(coded)?, accelerator))
    }

    #[test]
    fn a_hard_limit_codes_a_picture_again_until_it_fits() -> Result<(), Unsupported> {
        let (samples, fake) = run_costly(limited(), &limits(), 90)?;
        assert!(fake.picture_qp);
        assert_eq!(samples.len(), 90);
        assert_eq!(fake.pictures.len(), 90);
        // The first key frame, 3.2 Mbit at QP 26, does not fit the
        // 450 kbit second at the QP planned for it before any was coded.
        assert!(fake.calls > 90, "{} calls", fake.calls);
        assert!(fake.qps[0] > 26);
        // No second holds more than the buffer and a second at the peak.
        let bits: Vec<usize> = samples.iter().map(|s| s.data.len() * 8).collect();
        let busiest = bits
            .windows(30)
            .map(|w| w.iter().sum::<usize>())
            .max()
            .unwrap_or(0);
        assert!(busiest <= 450_000, "{busiest}");
        // A picture coded again keeps its place: the next predicts from it.
        assert_eq!(fake.pictures[1].2, Some(0));
        assert!(samples.iter().enumerate().all(|(n, s)| s.id == n as u64));
        Ok(())
    }

    #[test]
    fn a_driver_that_keeps_its_buffer_keeps_its_rate_control() -> Result<(), Unsupported> {
        let mut keeps = limits();
        keeps.keeps_buffer = true;
        let mut unlimited = limited();
        unlimited.hard_limit = false;
        for (settings, limits) in [(limited(), keeps), (unlimited, limits())] {
            let (_, fake) = run_costly(settings, &limits, 30)?;
            assert!(!fake.picture_qp);
            assert_eq!(fake.calls, 30);
            assert!(fake.qps.iter().all(|&qp| qp == PPS_QP));
        }
        Ok(())
    }
}
