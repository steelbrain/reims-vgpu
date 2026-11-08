//! An NVENC encode session: raw 4:2:0 frames in, H.264 or HEVC access
//! units out, in decode order.
//!
//! NVENC decides each picture's type (`enablePTD`), writes its own
//! parameter sets and slice headers, and keeps its own references; this
//! module configures it from a preset, copies each frame into an input
//! buffer of NVENC's, and reads each access unit back. A key frame is
//! forced where the caller asks for one; NVENC places the rest by its GOP
//! length. B pictures are never references here, so a reference is encoded
//! at most one frame ahead of its display order.

use std::collections::VecDeque;
use std::sync::Arc;

use super::ffi::{self, GUID, Zeroed};
use super::{Buffer, Session, Submission};
use crate::cuda::Gpu;
use crate::error::Error;

/// A profile NVENC is asked to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncodeProfile {
    /// H.264 Baseline: CAVLC, no B slices.
    H264Baseline,
    /// H.264 Main.
    H264Main,
    /// H.264 High.
    H264High,
    /// HEVC Main: 8-bit 4:2:0.
    HevcMain,
    /// HEVC Main 10: 10-bit 4:2:0.
    HevcMain10,
}

impl EncodeProfile {
    fn codec_guid(self) -> GUID {
        match self {
            Self::H264Baseline | Self::H264Main | Self::H264High => ffi::NV_ENC_CODEC_H264_GUID,
            Self::HevcMain | Self::HevcMain10 => ffi::NV_ENC_CODEC_HEVC_GUID,
        }
    }

    fn profile_guid(self) -> GUID {
        match self {
            Self::H264Baseline => ffi::NV_ENC_H264_PROFILE_BASELINE_GUID,
            Self::H264Main => ffi::NV_ENC_H264_PROFILE_MAIN_GUID,
            Self::H264High => ffi::NV_ENC_H264_PROFILE_HIGH_GUID,
            Self::HevcMain => ffi::NV_ENC_HEVC_PROFILE_MAIN_GUID,
            Self::HevcMain10 => ffi::NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        }
    }

    /// Whether this is an HEVC profile.
    pub fn is_hevc(self) -> bool {
        matches!(self, Self::HevcMain | Self::HevcMain10)
    }

    /// Bits a sample, which the input frames have too.
    pub fn bit_depth(self) -> u32 {
        if self == Self::HevcMain10 { 10 } else { 8 }
    }

    /// Whether the profile has B slices.
    pub fn has_b_slices(self) -> bool {
        !matches!(self, Self::H264Baseline)
    }

    fn buffer_format(self) -> ffi::NV_ENC_BUFFER_FORMAT {
        if self.bit_depth() > 8 {
            ffi::NV_ENC_BUFFER_FORMAT_YUV420_10BIT
        } else {
            ffi::NV_ENC_BUFFER_FORMAT_NV12
        }
    }
}

/// How NVENC controls the bitrate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeRateControl {
    /// A fixed quantiser for each picture type.
    ConstantQp {
        /// I pictures.
        intra: u32,
        /// P pictures.
        predicted: u32,
        /// B pictures.
        bidirectional: u32,
    },
    /// A constant bitrate.
    ConstantBitRate {
        /// Bits a second.
        bits_per_second: u32,
    },
    /// A variable bitrate around an average, under a peak, aiming at a
    /// quality where one is given (NVENC's target quality, 1 to 51).
    VariableBitRate {
        /// Bits a second on average.
        average_bits_per_second: u32,
        /// Bits a second at most.
        peak_bits_per_second: u32,
        /// The target quality, if any.
        target_quality: Option<u8>,
    },
}

/// NVENC's preset tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tuning {
    /// For files and streams that may be buffered.
    HighQuality,
    /// For live streams: no B pictures by default.
    LowLatency,
}

/// What an encode session is made with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeParams {
    /// The profile, and so the codec and bit depth.
    pub profile: EncodeProfile,
    /// The picture's width.
    pub width: u32,
    /// The picture's height.
    pub height: u32,
    /// Frames a second: numerator and denominator.
    pub frame_rate: (u32, u32),
    /// The rate control.
    pub rate_control: EncodeRateControl,
    /// How long the VBV buffer holds at the peak rate, in milliseconds.
    pub buffer_milliseconds: u32,
    /// The frames from one IDR picture to the next; none for NVENC never
    /// placing one of its own.
    pub gop_length: Option<u32>,
    /// The B pictures between references; 0 for none.
    pub b_frames: u32,
    /// The level as the bitstream numbers it, or none for NVENC's choice.
    pub level: Option<u32>,
    /// H.264's entropy coder: true for CABAC, false for CAVLC, none for the
    /// profile's default.
    pub cabac: Option<bool>,
    /// The VUI's `video_full_range_flag`.
    pub full_range: bool,
    /// The VUI's colour code points (H.273): primaries, transfer, matrix.
    pub colour: Option<(u8, u8, u8)>,
    /// The display aspect ratio NVENC derives the sample aspect ratio from;
    /// none for square samples.
    pub display_aspect: Option<(u32, u32)>,
    /// The preset, 1 (fastest) to 7 (best).
    pub preset: u8,
    /// The preset's tuning.
    pub tuning: Tuning,
}

/// A picture's type, as NVENC reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PictureType {
    /// An IDR picture.
    Idr,
    /// An I picture that is not IDR.
    Intra,
    /// A P picture, a reference.
    Predicted,
    /// A P picture nothing predicts from.
    NonReferencePredicted,
    /// A B picture (never a reference here).
    Bidirectional,
}

impl PictureType {
    fn from_raw(raw: ffi::NV_ENC_PIC_TYPE) -> Result<Self, Error> {
        Ok(match raw {
            ffi::NV_ENC_PIC_TYPE_IDR => Self::Idr,
            ffi::NV_ENC_PIC_TYPE_I => Self::Intra,
            ffi::NV_ENC_PIC_TYPE_P => Self::Predicted,
            ffi::NV_ENC_PIC_TYPE_NONREF_P => Self::NonReferencePredicted,
            ffi::NV_ENC_PIC_TYPE_B => Self::Bidirectional,
            other => {
                return Err(Error::Unsupported(format!(
                    "NVENC wrote a picture of type {other}"
                )));
            }
        })
    }

    /// Whether later pictures may predict from it.
    pub fn is_reference(self) -> bool {
        matches!(self, Self::Idr | Self::Intra | Self::Predicted)
    }
}

/// One pending bitstream's own result: its packet, or why it does not
/// read. A bitstream that does not read names no frame.
pub type Bitstream = Result<Packet, Error>;

/// One encoded access unit, in decode order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packet {
    /// The access unit as NVENC writes it: Annex B, with parameter sets
    /// before each IDR picture.
    pub data: Vec<u8>,
    /// The index of the frame it encodes, as [`RawFrame::index`] gave it.
    pub index: u64,
    /// The picture's type.
    pub picture_type: PictureType,
    /// NVENC's average QP for the picture.
    pub average_qp: u32,
}

/// One raw frame: a luma plane and an interleaved chroma plane half as
/// tall, at the profile's bit depth (16-bit samples, high bits, for 10).
#[derive(Clone, Copy, Debug)]
pub struct RawFrame<'a> {
    /// The luma plane and its bytes per row.
    pub luma: (&'a [u8], usize),
    /// The chroma plane and its bytes per row.
    pub chroma: (&'a [u8], usize),
    /// The caller's index for the frame, given back with its packet; each
    /// later than the last.
    pub index: u64,
    /// Whether it must be an IDR picture.
    pub force_idr: bool,
}

/// An input buffer and a bitstream buffer of the session.
struct Slot {
    input: Buffer,
    output: Buffer,
}

/// An NVENC encoder.
pub struct NvencEncoder {
    params: EncodeParams,
    slots: Vec<Slot>,
    /// The slot the next frame goes in.
    next: usize,
    /// Slots submitted whose bitstreams are not read yet, oldest first.
    pending: VecDeque<usize>,
    /// The frames NVENC may hold before it writes one: the B pictures.
    held: u32,
    /// The frames submitted, which number them for NVENC (`frameIdx`).
    submitted: u32,
    /// The parameter sets, Annex B, as `nvEncGetSequenceParams` gives them.
    sequence_parameters: Vec<u8>,
    session: Session,
}

/// Ours: input and bitstream buffers beyond what NVENC holds, so that a
/// frame never waits for a buffer to come back.
const EXTRA_SLOTS: usize = 3;

impl NvencEncoder {
    /// Opens a session on `gpu` and configures it for `params`.
    pub fn new(gpu: Arc<Gpu>, params: &EncodeParams) -> Result<Self, Error> {
        let mut session = Session::open(gpu)?;
        let codec = params.profile.codec_guid();
        check_params(&session, params, codec)?;
        let preset = preset_guid(params.preset)?;
        let tuning = match params.tuning {
            Tuning::HighQuality => ffi::NV_ENC_TUNING_INFO_HIGH_QUALITY,
            Tuning::LowLatency => ffi::NV_ENC_TUNING_INFO_LOW_LATENCY,
        };
        let mut config = session.preset_config(codec, preset, tuning)?.presetCfg;
        configure(&mut config, params);
        let mut init = ffi::NV_ENC_INITIALIZE_PARAMS::zeroed();
        init.version = ffi::NV_ENC_INITIALIZE_PARAMS_VER;
        init.encodeGUID = codec;
        init.presetGUID = preset;
        init.encodeWidth = params.width;
        init.encodeHeight = params.height;
        let (dar_width, dar_height) = params
            .display_aspect
            .unwrap_or((params.width, params.height));
        init.darWidth = dar_width;
        init.darHeight = dar_height;
        init.frameRateNum = params.frame_rate.0.max(1);
        init.frameRateDen = params.frame_rate.1.max(1);
        init.enableEncodeAsync = 0;
        init.enablePTD = 1;
        init.maxEncodeWidth = params.width;
        init.maxEncodeHeight = params.height;
        init.tuningInfo = tuning;
        session.initialize(init, config)?;
        let mut encoder = Self {
            params: params.clone(),
            slots: Vec::new(),
            next: 0,
            pending: VecDeque::new(),
            held: params.b_frames,
            submitted: 0,
            sequence_parameters: Vec::new(),
            session,
        };
        let count = usize::try_from(encoder.held).unwrap_or(0) + 1 + EXTRA_SLOTS;
        for _ in 0..count {
            let slot = Slot {
                input: encoder.session.create_input(
                    params.width,
                    params.height,
                    params.profile.buffer_format(),
                )?,
                output: encoder.session.create_bitstream()?,
            };
            encoder.slots.push(slot);
        }
        encoder.sequence_parameters = encoder
            .session
            .sequence_parameters(SEQUENCE_PARAMETERS_BYTES)?;
        Ok(encoder)
    }

    /// What the session was made with.
    pub fn params(&self) -> &EncodeParams {
        &self.params
    }

    /// The parameter sets (VPS, SPS, PPS for HEVC; SPS, PPS for H.264),
    /// Annex B, as NVENC writes them before each IDR picture.
    pub fn sequence_parameters(&self) -> &[u8] {
        &self.sequence_parameters
    }

    /// The frames NVENC may hold before their packets come out: the B
    /// pictures before each reference.
    pub fn delay(&self) -> u32 {
        self.held
    }

    /// Encodes one frame, returning the bitstreams that come out: none
    /// while NVENC holds frames back for B pictures, then each held one in
    /// decode order, each with its own result. `Err` is a frame NVENC did
    /// not take.
    pub fn encode(&mut self, frame: &RawFrame<'_>) -> Result<Vec<Bitstream>, Error> {
        let slot_index = self.next;
        if self.pending.contains(&slot_index) {
            return Err(Error::Unsupported(
                "NVENC holds more frames than it has buffers".to_owned(),
            ));
        }
        let Slot { input, output } = self.slots[slot_index];
        let bytes_per_sample = if self.params.profile.bit_depth() > 8 {
            2
        } else {
            1
        };
        let row_bytes = usize::try_from(self.params.width)
            .unwrap_or(0)
            .saturating_mul(bytes_per_sample);
        let rows = usize::try_from(self.params.height).unwrap_or(0);
        let pitch = self
            .session
            .upload(input, frame.luma, frame.chroma, row_bytes, rows)?;
        let status = self.session.encode_picture(Submission::Frame {
            input,
            output,
            pitch,
            index: self.submitted,
            timestamp: frame.index,
            idr: frame.force_idr,
        })?;
        self.submitted = self.submitted.wrapping_add(1);
        self.next = (slot_index + 1) % self.slots.len();
        self.pending.push_back(slot_index);
        if status == ffi::NV_ENC_ERR_NEED_MORE_INPUT {
            return Ok(Vec::new());
        }
        Ok(self.drain())
    }

    /// Ends the stream: every frame NVENC still holds comes out, each with
    /// its own result. `Err` is an end NVENC did not take.
    pub fn flush(&mut self) -> Result<Vec<Bitstream>, Error> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        self.session.encode_picture(Submission::End)?;
        Ok(self.drain())
    }

    /// Reads every pending bitstream, oldest first: NVENC fills them in
    /// decode order. Each has its own result, so a lock that fails loses
    /// no packet read before it or after it.
    fn drain(&mut self) -> Vec<Bitstream> {
        let mut bitstreams = Vec::with_capacity(self.pending.len());
        while let Some(slot_index) = self.pending.pop_front() {
            let output = self.slots[slot_index].output;
            bitstreams.push(self.read_bitstream(output));
        }
        bitstreams
    }

    fn read_bitstream(&self, output: Buffer) -> Result<Packet, Error> {
        let (data, index, picture_type, average_qp) =
            self.session.lock_bitstream(output, |locked| {
                let size = usize::try_from(locked.bitstreamSizeInBytes).unwrap_or(0);
                let data = if locked.bitstreamBufferPtr.is_null() {
                    Vec::new()
                } else {
                    // SAFETY: while locked, NVENC's buffer holds the
                    // picture's `bitstreamSizeInBytes` bytes, which it
                    // wrote, from `bitstreamBufferPtr`.
                    unsafe {
                        std::slice::from_raw_parts(locked.bitstreamBufferPtr.cast::<u8>(), size)
                    }
                    .to_vec()
                };
                Ok((
                    data,
                    locked.outputTimeStamp,
                    locked.pictureType,
                    locked.frameAvgQP,
                ))
            })?;
        Ok(Packet {
            data,
            index,
            picture_type: PictureType::from_raw(picture_type)?,
            average_qp,
        })
    }

    /// Changes the bitrate, and the frame rate, of the running session
    /// (`nvEncReconfigureEncoder`, with no reset and no forced IDR).
    pub fn reconfigure(
        &mut self,
        rate_control: EncodeRateControl,
        frame_rate: (u32, u32),
    ) -> Result<(), Error> {
        let mut params = self.params.clone();
        params.rate_control = rate_control;
        params.frame_rate = frame_rate;
        self.session.reconfigure(|init, config| {
            set_rate_control(&mut config.rcParams, &params);
            init.frameRateNum = frame_rate.0.max(1);
            init.frameRateDen = frame_rate.1.max(1);
        })?;
        self.params = params;
        self.sequence_parameters = self
            .session
            .sequence_parameters(SEQUENCE_PARAMETERS_BYTES)?;
        Ok(())
    }
}

/// What NVENC on a GPU encodes of one profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeProfileSupport {
    /// The profile.
    pub profile: EncodeProfile,
    /// The narrowest picture (`NV_ENC_CAPS_WIDTH_MIN`).
    pub min_width: u32,
    /// The shortest picture (`NV_ENC_CAPS_HEIGHT_MIN`).
    pub min_height: u32,
    /// The widest picture (`NV_ENC_CAPS_WIDTH_MAX`).
    pub max_width: u32,
    /// The tallest picture (`NV_ENC_CAPS_HEIGHT_MAX`).
    pub max_height: u32,
    /// The most B pictures in a row (`NV_ENC_CAPS_NUM_MAX_BFRAMES`); 0 for
    /// none, and always 0 for H.264 Baseline.
    pub max_b_frames: u32,
}

/// The profiles NVENC on `gpu` encodes, as its session lists them: each
/// codec's profile GUIDs, with 10 bits only where the codec's caps say so.
pub fn encode_profiles(gpu: Arc<Gpu>) -> Result<Vec<EncodeProfileSupport>, Error> {
    const ALL: [EncodeProfile; 5] = [
        EncodeProfile::H264Baseline,
        EncodeProfile::H264Main,
        EncodeProfile::H264High,
        EncodeProfile::HevcMain,
        EncodeProfile::HevcMain10,
    ];
    let session = Session::open(gpu)?;
    let codecs = session.codec_guids()?;
    let mut supported = Vec::new();
    for profile in ALL {
        let codec = profile.codec_guid();
        if !codecs.contains(&codec)
            || !session
                .profile_guids(codec)?
                .contains(&profile.profile_guid())
        {
            continue;
        }
        let cap = |capability| -> Result<u32, Error> {
            Ok(u32::try_from(session.capability(codec, capability)?).unwrap_or(0))
        };
        if profile.bit_depth() > 8 && cap(ffi::caps::SUPPORT_10BIT_ENCODE)? == 0 {
            continue;
        }
        supported.push(EncodeProfileSupport {
            profile,
            min_width: cap(ffi::caps::WIDTH_MIN)?,
            min_height: cap(ffi::caps::HEIGHT_MIN)?,
            max_width: cap(ffi::caps::WIDTH_MAX)?,
            max_height: cap(ffi::caps::HEIGHT_MAX)?,
            max_b_frames: if profile.has_b_slices() {
                cap(ffi::caps::NUM_MAX_BFRAMES)?
            } else {
                0
            },
        });
    }
    Ok(supported)
}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        // Frames NVENC still holds are ended and read first: destroying
        // the session with them in flight crashed in the driver
        // (libnvcuvid, under nvEncDestroyEncoder, with lookahead on; RTX
        // 5080 Laptop GPU, driver 615.71.09, 2026-10-24).
        if !self.pending.is_empty() {
            match self.flush() {
                Ok(bitstreams) => {
                    for e in bitstreams.into_iter().filter_map(Result::err) {
                        log::warn!("{e}");
                    }
                }
                Err(e) => log::warn!("{e}"),
            }
        }
        // The session destroys its buffers, then itself, as its field
        // drops after this.
    }
}

/// The parameter set buffer's size: NVENC's limit for a sequence header
/// (`NV_MAX_SEQ_HDR_LEN`, 512) twice over, for VPS, SPS and PPS together.
const SEQUENCE_PARAMETERS_BYTES: usize = 1024;

/// NVENC's GUID for preset P`preset`, 1 to 7.
pub(crate) fn preset_guid(preset: u8) -> Result<GUID, Error> {
    Ok(match preset {
        1 => ffi::NV_ENC_PRESET_P1_GUID,
        2 => ffi::NV_ENC_PRESET_P2_GUID,
        3 => ffi::NV_ENC_PRESET_P3_GUID,
        4 => ffi::NV_ENC_PRESET_P4_GUID,
        5 => ffi::NV_ENC_PRESET_P5_GUID,
        6 => ffi::NV_ENC_PRESET_P6_GUID,
        7 => ffi::NV_ENC_PRESET_P7_GUID,
        other => return Err(Error::Unsupported(format!("NVENC preset P{other}"))),
    })
}

/// Checks `params` against what the session says it encodes.
fn check_params(session: &Session, params: &EncodeParams, codec: GUID) -> Result<(), Error> {
    if !session.codec_guids()?.contains(&codec) {
        return Err(Error::Unsupported(format!(
            "NVENC does not encode {:?}",
            params.profile
        )));
    }
    if !session
        .profile_guids(codec)?
        .contains(&params.profile.profile_guid())
    {
        return Err(Error::Unsupported(format!(
            "NVENC does not encode {:?}",
            params.profile
        )));
    }
    let cap = |capability| session.capability(codec, capability);
    let (min_width, min_height) = (cap(ffi::caps::WIDTH_MIN)?, cap(ffi::caps::HEIGHT_MIN)?);
    let (max_width, max_height) = (cap(ffi::caps::WIDTH_MAX)?, cap(ffi::caps::HEIGHT_MAX)?);
    let width = i32::try_from(params.width).unwrap_or(i32::MAX);
    let height = i32::try_from(params.height).unwrap_or(i32::MAX);
    if width < min_width || height < min_height || width > max_width || height > max_height {
        return Err(Error::Unsupported(format!(
            "NVENC encodes {:?} from {min_width}x{min_height} to {max_width}x{max_height}, \
             not {}x{}",
            params.profile, params.width, params.height
        )));
    }
    if params.profile.bit_depth() > 8 && cap(ffi::caps::SUPPORT_10BIT_ENCODE)? == 0 {
        return Err(Error::Unsupported(
            "NVENC does not encode 10 bits".to_owned(),
        ));
    }
    let max_b_frames = u32::try_from(cap(ffi::caps::NUM_MAX_BFRAMES)?).unwrap_or(0);
    if params.b_frames > max_b_frames || (params.b_frames > 0 && !params.profile.has_b_slices()) {
        return Err(Error::Unsupported(format!(
            "NVENC encodes at most {max_b_frames} B pictures in a row for {:?}",
            params.profile
        )));
    }
    if params.cabac == Some(true) && cap(ffi::caps::SUPPORT_CABAC)? == 0 {
        return Err(Error::Unsupported("NVENC has no CABAC".to_owned()));
    }
    Ok(())
}

/// Sets the rate control `params` ask for in `rc`, keeping the preset's
/// other choices.
fn set_rate_control(rc: &mut ffi::NV_ENC_RC_PARAMS, params: &EncodeParams) {
    rc.version = ffi::NV_ENC_RC_PARAMS_VER;
    match params.rate_control {
        EncodeRateControl::ConstantQp {
            intra,
            predicted,
            bidirectional,
        } => {
            rc.rateControlMode = ffi::NV_ENC_PARAMS_RC_CONSTQP;
            rc.constQP = ffi::NV_ENC_QP {
                qpInterP: predicted,
                qpInterB: bidirectional,
                qpIntra: intra,
            };
            rc.averageBitRate = 0;
            rc.maxBitRate = 0;
            rc.targetQuality = 0;
        }
        EncodeRateControl::ConstantBitRate { bits_per_second } => {
            rc.rateControlMode = ffi::NV_ENC_PARAMS_RC_CBR;
            rc.averageBitRate = bits_per_second;
            rc.maxBitRate = bits_per_second;
            rc.targetQuality = 0;
        }
        EncodeRateControl::VariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
            target_quality,
        } => {
            rc.rateControlMode = ffi::NV_ENC_PARAMS_RC_VBR;
            rc.averageBitRate = average_bits_per_second;
            rc.maxBitRate = peak_bits_per_second.max(average_bits_per_second);
            rc.targetQuality = target_quality.unwrap_or(0);
        }
    }
    // The VBV buffer holds as long at the peak as the session asks, a
    // second unless a hard limit needs less, and starts three quarters
    // full, as the VA-API crate sizes its HRD buffer, for the same
    // streams on either. NVENC's VBR runs its first second near its peak
    // whatever the buffer: peaking at 1.5 times the average, with NVENC's
    // default buffer, a second's or half a second's, it came to 1.45 to
    // 1.65 times the average (RTX 5080 Laptop GPU, 2026-10-25). So the
    // session peaks near the average (the codec crate's
    // `encode/session.rs`).
    let peak = u64::from(rc.maxBitRate);
    let buffer = u32::try_from(peak * u64::from(params.buffer_milliseconds) / 1000)
        .unwrap_or(u32::MAX)
        .max(1);
    rc.vbvBufferSize = buffer;
    rc.vbvInitialDelay = buffer / 4 * 3;
}

/// Makes `config`, a preset's, encode what `params` ask for.
pub(crate) fn configure(config: &mut ffi::NV_ENC_CONFIG, params: &EncodeParams) {
    config.version = ffi::NV_ENC_CONFIG_VER;
    config.profileGUID = params.profile.profile_guid();
    // B pictures need a finite GOP (the header: with an infinite GOP
    // "frameIntervalP should be set to 1"). Ours, with none asked for: a
    // GOP longer than any session runs, so that NVENC places no IDR
    // picture of its own.
    let gop = params.gop_length.unwrap_or(if params.b_frames > 0 {
        0x7fff_ffff
    } else {
        ffi::NVENC_INFINITE_GOPLENGTH
    });
    config.gopLength = gop;
    config.frameIntervalP = i32::try_from(params.b_frames + 1).unwrap_or(1);
    set_rate_control(&mut config.rcParams, params);
    // Ours: no lookahead. P6 and P7 at high quality turn it on, 28 frames
    // deep (RTX 5080 Laptop GPU, 2026-10-24), and NVENC then writes a frame only
    // that many frames after it, which a session's guest does not expect
    // of a hardware encoder. A picture locked before that failed with
    // NV_ENC_ERR_INVALID_PARAM.
    config.rcParams.flags = ffi::bits::rc::enableLookahead.put(config.rcParams.flags, 0);
    config.rcParams.lookaheadDepth = 0;
    let vui = |vui: &mut ffi::NV_ENC_CONFIG_H264_VUI_PARAMETERS| {
        vui.videoSignalTypePresentFlag = 1;
        vui.videoFormat = ffi::NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        vui.videoFullRangeFlag = u32::from(params.full_range);
        match params.colour {
            Some((primaries, transfer, matrix)) => {
                vui.colourDescriptionPresentFlag = 1;
                vui.colourPrimaries = primaries.into();
                vui.transferCharacteristics = transfer.into();
                vui.colourMatrix = matrix.into();
            }
            None => vui.colourDescriptionPresentFlag = 0,
        }
    };
    let level = params.level.unwrap_or(ffi::NV_ENC_LEVEL_AUTOSELECT);
    let depth = if params.profile.bit_depth() > 8 {
        ffi::NV_ENC_BIT_DEPTH_10
    } else {
        ffi::NV_ENC_BIT_DEPTH_8
    };
    if params.profile.is_hevc() {
        // SAFETY: the union's HEVC member, for an HEVC session; every bit
        // pattern of its integers is valid.
        let hevc = unsafe { &mut config.encodeCodecConfig.hevcConfig };
        hevc.level = level;
        hevc.tier = 0;
        hevc.idrPeriod = gop;
        hevc.flags = ffi::bits::hevc::repeatSPSPPS.put(hevc.flags, 1);
        hevc.flags = ffi::bits::hevc::chromaFormatIDC.put(hevc.flags, 1);
        hevc.useBFramesAsRef = ffi::NV_ENC_BFRAME_REF_MODE_DISABLED;
        hevc.outputBitDepth = depth;
        hevc.inputBitDepth = depth;
        vui(&mut hevc.hevcVUIParameters);
    } else {
        // SAFETY: the union's H.264 member, for an H.264 session; every
        // bit pattern of its integers is valid.
        let h264 = unsafe { &mut config.encodeCodecConfig.h264Config };
        h264.level = level;
        h264.idrPeriod = gop;
        h264.flags = ffi::bits::h264::repeatSPSPPS.put(h264.flags, 1);
        h264.chromaFormatIDC = 1;
        h264.useBFramesAsRef = ffi::NV_ENC_BFRAME_REF_MODE_DISABLED;
        h264.outputBitDepth = depth;
        h264.inputBitDepth = depth;
        let cabac = params
            .cabac
            .unwrap_or(params.profile != EncodeProfile::H264Baseline);
        h264.entropyCodingMode = if cabac && params.profile != EncodeProfile::H264Baseline {
            ffi::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC
        } else {
            ffi::NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC
        };
        vui(&mut h264.h264VUIParameters);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn params(profile: EncodeProfile) -> EncodeParams {
        EncodeParams {
            profile,
            width: 640,
            height: 360,
            frame_rate: (30, 1),
            rate_control: EncodeRateControl::ConstantBitRate {
                bits_per_second: 2_000_000,
            },
            buffer_milliseconds: 1000,
            gop_length: Some(30),
            b_frames: 2,
            level: None,
            cabac: None,
            full_range: true,
            colour: Some((1, 1, 1)),
            display_aspect: None,
            preset: 4,
            tuning: Tuning::HighQuality,
        }
    }

    #[test]
    fn a_config_asks_for_what_the_params_say() {
        let mut config = ffi::NV_ENC_CONFIG::zeroed();
        configure(&mut config, &params(EncodeProfile::H264High));
        assert_eq!(config.version, ffi::NV_ENC_CONFIG_VER);
        assert_eq!(config.profileGUID, ffi::NV_ENC_H264_PROFILE_HIGH_GUID);
        assert_eq!((config.gopLength, config.frameIntervalP), (30, 3));
        assert_eq!(config.rcParams.rateControlMode, ffi::NV_ENC_PARAMS_RC_CBR);
        assert_eq!(config.rcParams.averageBitRate, 2_000_000);
        // SAFETY: an H.264 config was made.
        let h264 = unsafe { config.encodeCodecConfig.h264Config };
        assert_eq!(h264.idrPeriod, 30);
        assert_eq!(
            h264.entropyCodingMode,
            ffi::NV_ENC_H264_ENTROPY_CODING_MODE_CABAC
        );
        assert_eq!(h264.h264VUIParameters.videoFullRangeFlag, 1);
        assert_eq!(h264.h264VUIParameters.colourPrimaries, 1);
        assert_eq!(h264.flags & ffi::bits::h264::repeatSPSPPS.mask(), 1 << 12);
    }

    #[test]
    fn baseline_is_cavlc_and_hevc_main_10_is_ten_bits() {
        let mut config = ffi::NV_ENC_CONFIG::zeroed();
        let mut baseline = params(EncodeProfile::H264Baseline);
        baseline.cabac = Some(true);
        configure(&mut config, &baseline);
        // SAFETY: an H.264 config was made.
        let h264 = unsafe { config.encodeCodecConfig.h264Config };
        assert_eq!(
            h264.entropyCodingMode,
            ffi::NV_ENC_H264_ENTROPY_CODING_MODE_CAVLC
        );
        let mut config = ffi::NV_ENC_CONFIG::zeroed();
        configure(&mut config, &params(EncodeProfile::HevcMain10));
        // SAFETY: an HEVC config was made.
        let hevc = unsafe { config.encodeCodecConfig.hevcConfig };
        assert_eq!(hevc.outputBitDepth, ffi::NV_ENC_BIT_DEPTH_10);
        assert_eq!(hevc.flags & ffi::bits::hevc::chromaFormatIDC.mask(), 1 << 9);
    }

    /// A frame of `params`' size and depth: noise that moves with `n`.
    pub(crate) fn noise(params: &EncodeParams, n: u64) -> (Vec<u8>, Vec<u8>, usize) {
        let sample = if params.profile.bit_depth() > 8 { 2 } else { 1 };
        let row = params.width as usize * sample;
        let rows = params.height as usize;
        let mut seed = n.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u8
        };
        let luma = (0..row * rows).map(|_| next()).collect();
        let chroma = (0..row * rows.div_ceil(2)).map(|_| next()).collect();
        (luma, chroma, row)
    }

    /// What a raw `nvEncLockBitstream` answers for a 4K picture, in turn:
    /// with `doNotWait` as soon as the picture is submitted, with
    /// `doNotWait` 40 ms later, and waiting 40 ms later; each with how long
    /// the call took, its status, size and first bytes (`nvenc::locks`).
    /// Run by hand on the NVIDIA host.
    #[test]
    #[ignore = "needs an NVIDIA GPU: run on the NVIDIA host, with --nocapture"]
    fn what_each_lock_returns() -> Result<(), Error> {
        use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};

        use crate::cuda::GpuSelector;
        use crate::nvenc::{BufferKind, Submission};

        let gpu = Arc::new(Gpu::open(&GpuSelector::First)?);
        let params = EncodeParams {
            width: 3840,
            height: 2160,
            b_frames: 0,
            ..params(EncodeProfile::HevcMain)
        };
        let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params)?;
        let session = &encoder.session;
        let api = &session.nvenc.api;
        let (lock, unlock) = (
            api.nvEncLockBitstream.unwrap(),
            api.nvEncUnlockBitstream.unwrap(),
        );
        let frames: Vec<_> = (0..4).map(|n| noise(&params, n)).collect();
        for n in 0..12u32 {
            let (luma, chroma, row) = &frames[usize::try_from(n % 4).unwrap_or(0)];
            let row = *row;
            let Slot { input, output } = encoder.slots[usize::try_from(n).unwrap_or(0) % 2];
            let pitch = session.upload(input, (luma, row), (chroma, row), row, 2160)?;
            let submitting = StdRuntime.now();
            session.encode_picture(Submission::Frame {
                input,
                output,
                pitch,
                index: n,
                timestamp: u64::from(n),
                idr: (n / 3) % 2 == 0,
            })?;
            let submit = StdRuntime.elapsed_since(submitting).as_micros();
            let handle = session.output(output, &[BufferKind::Bitstream])?;
            let _current = gpu.push()?;
            // 0: doNotWait at once; 1: doNotWait once the encode is surely
            // done; 2: a waiting lock then.
            let mode = n % 3;
            if mode > 0 {
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            let mut locked = ffi::NV_ENC_LOCK_BITSTREAM::zeroed();
            locked.version = ffi::NV_ENC_LOCK_BITSTREAM_VER;
            if mode < 2 {
                locked.flags = ffi::bits::lock::doNotWait.put(0, 1);
            }
            locked.outputBitstream = handle;
            let calling = StdRuntime.now();
            // SAFETY: a live session and its submitted output buffer.
            let status = unsafe { lock(session.encoder, &raw mut locked) };
            let call = StdRuntime.elapsed_since(calling).as_micros();
            let head = if status == ffi::NV_ENC_SUCCESS && !locked.bitstreamBufferPtr.is_null() {
                let size = usize::try_from(locked.bitstreamSizeInBytes)
                    .unwrap_or(0)
                    .min(8);
                // SAFETY: the locked buffer's first bytes, at most its size.
                unsafe { std::slice::from_raw_parts(locked.bitstreamBufferPtr.cast::<u8>(), size) }
                    .to_vec()
            } else {
                Vec::new()
            };
            eprintln!(
                "frame {n} idr {} submit {submit}us {}: lock call {call}us status {status} hw {} \
                 size {} (0x{:x}) type {} ts {} head {head:02x?}",
                (n / 3) % 2 == 0,
                [
                    "doNotWait at once",
                    "doNotWait after 40 ms",
                    "waiting after 40 ms"
                ][mode as usize],
                locked.hwEncodeStatus,
                locked.bitstreamSizeInBytes,
                locked.bitstreamSizeInBytes,
                locked.pictureType,
                locked.outputTimeStamp
            );
            if status == ffi::NV_ENC_SUCCESS {
                // SAFETY: the buffer just locked.
                unsafe { unlock(session.encoder, handle) };
            }
            // Locked, as the session's own locks record it, so that the
            // next picture may be submitted into it.
            session.submitted.borrow_mut().locked(handle);
        }
        encoder.pending.clear();
        Ok(())
    }

    /// A lock past its deadline, here one far shorter than a 4K encode,
    /// fails its picture and the pictures after it as stalled, and each
    /// later call at once; the session, once dropped, is kept until its lock
    /// thread has finished and destroyed it (`nvenc::locks`).
    #[test]
    #[ignore = "needs an NVIDIA GPU: run on the NVIDIA host"]
    fn a_lock_past_its_deadline_stalls_the_session_until_nvenc_is_done() -> Result<(), Error> {
        use std::sync::atomic::Ordering;
        use std::time::Duration;

        use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};

        use crate::cuda::GpuSelector;
        use crate::wait::tests::{KEPT_EVER, WAIT, within_5_s};

        let gpu = Arc::new(Gpu::open(&GpuSelector::First)?);
        let params = EncodeParams {
            width: 3840,
            height: 2160,
            b_frames: 2,
            ..params(EncodeProfile::HevcMain)
        };
        let mut encoder = NvencEncoder::new(Arc::clone(&gpu), &params)?;
        let frames: Vec<_> = (0..3).map(|n| noise(&params, n)).collect();
        let encode = |encoder: &mut NvencEncoder, n: u64| {
            let (luma, chroma, row) = &frames[usize::try_from(n % 3).unwrap_or(0)];
            encoder.encode(&RawFrame {
                luma: (luma, *row),
                chroma: (chroma, *row),
                index: n,
                force_idr: n == 0,
            })
        };
        let names = |bitstreams: &[Bitstream]| -> Vec<String> {
            bitstreams
                .iter()
                .map(|b| match b {
                    Ok(packet) => format!("packet of frame {}", packet.index),
                    Err(e) => e.to_string(),
                })
                .collect()
        };
        // The IDR picture comes out at once; NVENC holds the next two for B
        // pictures, then all three come out.
        let mut out = Vec::new();
        for n in 0..4 {
            out.extend(encode(&mut encoder, n)?);
        }
        assert!(
            out.len() == 4 && out.iter().all(Result::is_ok),
            "{:?}",
            names(&out)
        );
        WAIT.set(Some(Duration::from_micros(100)));
        let kept = KEPT_EVER.load(Ordering::SeqCst);
        let mut out = Vec::new();
        for n in 4..7 {
            out.extend(encode(&mut encoder, n)?);
        }
        let stalled = |b: &Bitstream| matches!(b, Err(Error::Stalled(_)));
        assert!(
            out.len() == 3 && out.iter().all(stalled),
            "{:?}",
            names(&out)
        );
        let start = StdRuntime.now();
        let later = encode(&mut encoder, 7);
        assert!(matches!(later, Err(Error::Stalled(_))), "{later:?}");
        assert!(StdRuntime.elapsed_since(start) < Duration::from_millis(5));
        WAIT.set(None);
        drop(encoder);
        assert_eq!(KEPT_EVER.load(Ordering::SeqCst), kept + 1);
        assert!(
            within_5_s(|| crate::wait::kept() == 0),
            "the session is still kept"
        );
        Ok(())
    }
}
