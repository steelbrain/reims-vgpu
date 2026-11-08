//! The reims-graphics device's hardware encode on VA-API: H.264 and HEVC on the VA crate's
//! `VideoEncoder` (`EncSlice`, else `EncSliceLP`), in a [`HostEncodeSession`] whose event the host
//! signals.
//!
//! - **What it answers** ([`encode_caps_of`]) is the driver's: an entry for each codec of which
//!   it encodes a profile (the VA crate's `encoder_support`), with those profiles, the sizes their
//!   configs and surfaces give, NV12 sources (and P010 for HEVC Main 10), and the references
//!   their configs take. Its rate controls ([`encode_caps`]) are those `VAConfigAttribRateControl`
//!   lists of CQP, CBR and VBR, and no B-frames: the encoder keeps frames in display order.
//! - **The settings** ([`settings_of`]) map onto the VA crate's: the profile, size and colour; a
//!   VBR with no peak peaks at 1.05 times its average; `quality` 0 to 100 onto `VAConfigAttribEncQualityRange`'s levels, the fastest to
//!   1, the best. A change of what makes a new sequence makes a new encoder, whose first access
//!   unit is a keyframe with its parameter sets; a change of the bitrates, QPs or frame rate goes
//!   to the encoder from the next frame. `min_qp`, `max_qp` and a change of `quality` alone are
//!   not taken: the VA crate's encoder has no such setting.
//! - **A frame** ([`VaFrame`]) is read on the session's encode thread from the source's host
//!   bytes, uploaded and encoded by the VA crate's encoder, and its sample, NAL units behind
//!   four-byte lengths, written as Annex B: a keyframe's parameter sets first, from the encoder's
//!   configuration record (`avcC`, `hvcC`). One that does not fit the capacity is not written,
//!   and the next frame is made a keyframe.

use std::sync::Arc;

use reims_graphics::ops::video::{
    RGFX_VIDEO_ENCODE_HARD_LIMIT, RGFX_VIDEO_ENTROPY_CABAC, RGFX_VIDEO_ENTROPY_CAVLC,
    RGFX_VIDEO_PICTURE_B, RGFX_VIDEO_PICTURE_I, RGFX_VIDEO_PICTURE_P, RGFX_VIDEO_RATE_CBR,
    RGFX_VIDEO_RATE_CQP, RGFX_VIDEO_RATE_VBR, RgfxVideoEncodeSettings,
};
use reims_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_ENCODE, RGFX_VIDEO_FORMAT_NV12,
    RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN,
    RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps, RgfxVideoEncodeCaps,
};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_vaapi::encode::{
    ColourDescription, EncoderSettings, Entropy, FrameRate, FrameType, InputFrame, Plane,
    Profile as EncodeProfile, RateControl,
};
use reims_vgpu_video_vaapi::{Capabilities, Display, EncoderSupport, VideoEncoder};

use super::super::video_host::{
    BackendError, EncodeFrame, EncodeSource, Encoded, HostEncodedFrame, HostEncoder, SourcePlane,
};
use super::{H264_TOP_LEVEL, HEVC_TOP_LEVEL, MIN_SIZE};

/// `VA_RC_CBR`, `VA_RC_VBR` and `VA_RC_CQP` (va.h): rate control bits of
/// `VAConfigAttribRateControl`.
const VA_RC_CBR: u32 = 0x0000_0002;
const VA_RC_VBR: u32 = 0x0000_0004;
const VA_RC_CQP: u32 = 0x0000_0010;

/// Ours: a VBR's peak where the guest sets none, as a share of its average: 1.05.
const PEAK_NUMERATOR: u64 = 21;
const PEAK_DENOMINATOR: u64 = 20;

/// Ours: the frame rate where the guest says none, as `EncoderSettings::new` assumes.
const DEFAULT_FRAME_RATE: FrameRate = FrameRate::new(30, 1);

/// The four-byte start code each NAL unit of an Annex-B access unit follows.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The wire's profile bit and codec of an encode profile.
fn wire_profile(profile: EncodeProfile) -> (u32, u32) {
    match profile {
        EncodeProfile::H264ConstrainedBaseline => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_BASELINE),
        EncodeProfile::H264Main => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_MAIN),
        EncodeProfile::H264High => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH),
        EncodeProfile::HevcMain => (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN),
        EncodeProfile::HevcMain10 => (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10),
    }
}

/// The encode profile of the wire's `profile` bit of `codec`.
fn encode_profile(codec: u32, profile: u32) -> Option<EncodeProfile> {
    Some(match (codec, profile) {
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_BASELINE) => EncodeProfile::H264ConstrainedBaseline,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_MAIN) => EncodeProfile::H264Main,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH) => EncodeProfile::H264High,
        (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN) => EncodeProfile::HevcMain,
        (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10) => EncodeProfile::HevcMain10,
        _ => return None,
    })
}

/// The caps of H.264 and HEVC encode `supports` gives, with the surface sizes `caps` lists for
/// each profile's config (module doc): an entry for each codec of which it encodes a profile.
pub(super) fn encode_caps_of(
    supports: &[EncoderSupport],
    caps: &Capabilities,
) -> Vec<RgfxVideoCodecCaps> {
    let mut entries: Vec<RgfxVideoCodecCaps> = Vec::new();
    for support in supports {
        let (codec, bit) = wire_profile(support.profile);
        let index = match entries.iter().position(|e| e.codec == codec) {
            Some(index) => index,
            None => {
                entries.push(RgfxVideoCodecCaps {
                    codec,
                    direction: RGFX_VIDEO_ENCODE,
                    max_level: if codec == RGFX_VIDEO_CODEC_H264 {
                        H264_TOP_LEVEL
                    } else {
                        HEVC_TOP_LEVEL
                    },
                    min_size: [MIN_SIZE; 2],
                    max_size: [u32::MAX; 2],
                    max_dpb_pictures: u32::MAX,
                    max_references: u32::MAX,
                    ..RgfxVideoCodecCaps::default()
                });
                entries.len() - 1
            }
        };
        let entry = &mut entries[index];
        entry.profiles |= bit;
        entry.formats |= if support.profile == EncodeProfile::HevcMain10 {
            RGFX_VIDEO_FORMAT_P010
        } else {
            RGFX_VIDEO_FORMAT_NV12
        };
        let va = &support.capabilities;
        let surfaces =
            (caps.find(va.profile, va.entrypoint)).and_then(|c| c.surfaces.as_ref().ok());
        let (max_width, max_height) = (
            va.max_width.or(surfaces.and_then(|s| s.max_width)),
            va.max_height.or(surfaces.and_then(|s| s.max_height)),
        );
        if let (Some(width), Some(height)) = (max_width, max_height) {
            entry.max_size = [entry.max_size[0].min(width), entry.max_size[1].min(height)];
        }
        let min = |of: Option<u32>| of.unwrap_or(MIN_SIZE).max(1);
        entry.min_size[0] = entry.min_size[0].max(min(surfaces.and_then(|s| s.min_width)));
        entry.min_size[1] = entry.min_size[1].max(min(surfaces.and_then(|s| s.min_height)));
        // List 0's references: no B-frames, so list 1 is never used.
        let (l0, _) = va.max_references();
        entry.max_references = entry.max_references.min(l0.max(1));
        entry.max_dpb_pictures = entry.max_references.saturating_add(1);
    }
    // A codec no config gives a largest picture for is not listed.
    entries.retain(|e| e.max_size != [u32::MAX; 2]);
    entries
}

/// What an encode of `codec`, an entry [`encode_caps_of`] made, takes beyond it: the rate
/// controls the configs of its profiles list, and no B-frames (module doc); all zero for a decode
/// entry.
pub(super) fn encode_caps(
    codec: &RgfxVideoCodecCaps,
    supports: &[EncoderSupport],
) -> RgfxVideoEncodeCaps {
    if codec.direction != RGFX_VIDEO_ENCODE {
        return RgfxVideoEncodeCaps::default();
    }
    let rate = (supports.iter())
        .filter(|s| wire_profile(s.profile).0 == codec.codec)
        .fold(0, |rate, s| rate | s.capabilities.rate_control.unwrap_or(0));
    let mut rate_controls = 0;
    for (va, wire) in [
        (VA_RC_CQP, RGFX_VIDEO_RATE_CQP),
        (VA_RC_CBR, RGFX_VIDEO_RATE_CBR),
        (VA_RC_VBR, RGFX_VIDEO_RATE_VBR),
    ] {
        if rate & va != 0 {
            rate_controls |= 1 << wire;
        }
    }
    RgfxVideoEncodeCaps {
        rate_controls,
        max_b_frames: 0,
        flags: 0,
        reserved_0c: 0,
    }
}

/// A rate in bits a second as the VA crate's settings take it: a u32's largest past one, and at
/// least one.
fn bits(rate: u64) -> u32 {
    u32::try_from(rate).unwrap_or(u32::MAX).max(1)
}

/// The rate control `settings` asks for (module doc).
fn rate_of(settings: &RgfxVideoEncodeSettings) -> RateControl {
    let qp = |qp: u32| u8::try_from(qp.min(51)).unwrap_or(51);
    match settings.rate_control {
        RGFX_VIDEO_RATE_CQP => RateControl::ConstantQp {
            intra: qp(settings.qp_i),
            predicted: qp(settings.qp_p),
            bidirectional: qp(settings.qp_b),
        },
        RGFX_VIDEO_RATE_CBR => RateControl::ConstantBitRate {
            bits_per_second: bits(settings.average_bitrate),
        },
        _ => {
            let average = bits(settings.average_bitrate);
            let peak = match settings.max_bitrate {
                0 => (u64::from(average).checked_mul(PEAK_NUMERATOR))
                    .map_or(u32::MAX, |b| bits(b / PEAK_DENOMINATOR)),
                peak => bits(peak),
            };
            RateControl::VariableBitRate {
                average_bits_per_second: average,
                peak_bits_per_second: peak.max(average),
            }
        }
    }
}

/// The frame rate `settings` asks for; [`DEFAULT_FRAME_RATE`] where it says none.
fn frame_rate_of(settings: &RgfxVideoEncodeSettings) -> FrameRate {
    match (settings.frame_rate_num, settings.frame_rate_den) {
        (0, _) | (_, 0) => DEFAULT_FRAME_RATE,
        (num, den) => FrameRate::new(num, den),
    }
}

/// VA's quality level of `quality`, 0 the fastest to 100 the best, onto `range` levels, 1 the
/// best (module doc); `None` where the driver lists no range.
fn quality_level(quality: u32, range: Option<u32>) -> Option<u32> {
    let levels = range.filter(|&r| r > 0)?;
    let quality = u64::from(quality.min(100));
    let steps = u64::from(levels - 1);
    let below_fastest = (quality * steps + 50) / 100;
    u32::try_from(u64::from(levels) - below_fastest).ok()
}

/// The VA crate's settings of `settings` of `codec`, the frame rate's numerator its clock
/// (module doc), at the quality `range` the driver lists for the profile.
pub(super) fn settings_of(
    codec: u32,
    settings: &RgfxVideoEncodeSettings,
    range: Option<u32>,
) -> Result<EncoderSettings, String> {
    let profile = encode_profile(codec, settings.profile)
        .ok_or_else(|| format!("profile {:#x} of codec {codec}", settings.profile))?;
    let mut out = EncoderSettings::new(profile, settings.width, settings.height);
    out.full_range = settings.full_range != 0;
    let (primaries, transfer, matrix) =
        (settings.color_primaries, settings.transfer, settings.matrix);
    out.colour = (primaries != 0 || transfer != 0 || matrix != 0).then_some(ColourDescription {
        primaries,
        transfer,
        matrix,
    });
    out.frame_rate = frame_rate_of(settings);
    out.timescale = out.frame_rate.numerator;
    out.rate_control = rate_of(settings);
    if settings.max_bitrate_window_ms > 0 {
        out.buffer_milliseconds = settings.max_bitrate_window_ms;
    }
    out.set_hard_limit(settings.flags & RGFX_VIDEO_ENCODE_HARD_LIMIT != 0);
    if settings.key_frame_interval > 0 {
        out.key_frame_interval = Some(settings.key_frame_interval);
    }
    out.allow_frame_reordering = false;
    out.max_frame_delay = Some(0);
    out.b_frames = Some(0);
    out.level = (settings.level > 0)
        .then(|| u8::try_from(settings.level).ok())
        .flatten();
    out.entropy = match settings.entropy {
        RGFX_VIDEO_ENTROPY_CAVLC => Some(Entropy::Cavlc),
        RGFX_VIDEO_ENTROPY_CABAC => Some(Entropy::Cabac),
        _ => None,
    };
    out.quality_level = quality_level(settings.quality, range);
    Ok(out)
}

/// Whether `new` starts a new sequence after `old` (the wire's settings doc): another size,
/// format, profile, level, entropy coder, B-frames, keyframe interval or colour.
fn new_sequence(old: &RgfxVideoEncodeSettings, new: &RgfxVideoEncodeSettings) -> bool {
    let sequence = |s: &RgfxVideoEncodeSettings| {
        (
            (s.width, s.height, s.format, s.profile, s.level),
            (s.entropy, s.b_frames, s.key_frame_interval),
            (s.color_primaries, s.transfer, s.matrix, s.full_range),
        )
    };
    sequence(old) != sequence(new)
}

/// A session's encoder, as its frames share it.
struct State {
    display: Display,
    codec: u32,
    /// The quality range of each profile the driver encodes, by the wire's profile bit.
    ranges: Vec<(u32, Option<u32>)>,
    encoder: Option<VideoEncoder>,
    settings: Option<RgfxVideoEncodeSettings>,
    /// Whether the next frame is made a keyframe: the last one did not fit.
    key_next: bool,
}

impl State {
    /// Takes `settings` from the next frame on (module doc).
    fn apply(&mut self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError> {
        let encoder = match (self.encoder.as_mut(), self.settings.as_ref()) {
            (Some(encoder), Some(old)) if !new_sequence(old, settings) => encoder,
            _ => {
                let range = (self.ranges.iter())
                    .find(|(bit, _)| *bit == settings.profile)
                    .and_then(|(_, range)| *range);
                let made = settings_of(self.codec, settings, range)?;
                self.encoder = None;
                self.encoder = Some(VideoEncoder::new(&self.display, made)?);
                self.settings = Some(*settings);
                return Ok(());
            }
        };
        let old = self.settings.unwrap_or_default();
        if rate_of(&old) != rate_of(settings) {
            encoder.set_rate_control(rate_of(settings))?;
        }
        if frame_rate_of(&old) != frame_rate_of(settings) {
            encoder.set_frame_rate(frame_rate_of(settings))?;
        }
        self.settings = Some(*settings);
        Ok(())
    }
}

/// A session's encoder on VA-API ([`HostEncoder`]).
pub(super) struct VaEncoder {
    state: Arc<Mutex<State>>,
    /// The profiles the driver encodes, which [`HostEncoder::check`] holds settings to.
    supports: Arc<[EncoderSupport]>,
    /// The sizes the codec's entry gives.
    min_size: [u32; 2],
    max_size: [u32; 2],
    /// The rate controls the entry lists, `1 << RGFX_VIDEO_RATE_*` bits.
    rate_controls: u32,
}

impl VaEncoder {
    /// An encoder of `codec` on `display`, which encodes `supports`, as its entry `entry` and
    /// `encode` list them.
    pub(super) fn new(
        display: &Display,
        codec: u32,
        supports: Arc<[EncoderSupport]>,
        entry: &RgfxVideoCodecCaps,
        encode: &RgfxVideoEncodeCaps,
    ) -> Self {
        let ranges = (supports.iter())
            .map(|s| (wire_profile(s.profile), s.capabilities.quality_range))
            .filter(|((c, _), _)| *c == codec)
            .map(|((_, bit), range)| (bit, range))
            .collect();
        Self {
            state: Arc::new(Mutex::new(State {
                display: display.clone_ref(),
                codec,
                ranges,
                encoder: None,
                settings: None,
                key_next: false,
            })),
            supports,
            min_size: entry.min_size,
            max_size: entry.max_size,
            rate_controls: encode.rate_controls,
        }
    }
}

impl HostEncoder for VaEncoder {
    fn check(&self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError> {
        let codec = self.state.lock().codec;
        let profile = encode_profile(codec, settings.profile)
            .filter(|p| self.supports.iter().any(|s| s.profile == *p))
            .ok_or_else(|| {
                format!(
                    "profile {:#x}, which the driver does not encode",
                    settings.profile
                )
            })?;
        let (width, height) = (settings.width, settings.height);
        let fits = (self.min_size[0]..=self.max_size[0]).contains(&width)
            && (self.min_size[1]..=self.max_size[1]).contains(&height);
        if !fits {
            return Err(format!(
                "{width}x{height} of {profile:?}, outside {}x{} to {}x{}",
                self.min_size[0], self.min_size[1], self.max_size[0], self.max_size[1]
            )
            .into());
        }
        if settings.b_frames > 0 {
            return Err(format!("{} B-frames: the encoder has none", settings.b_frames).into());
        }
        if self.rate_controls & (1 << settings.rate_control) == 0 {
            return Err(format!(
                "rate control {}, which the driver does not list",
                settings.rate_control
            )
            .into());
        }
        Ok(())
    }

    fn encode(&mut self, frame: EncodeFrame) -> Result<Box<dyn HostEncodedFrame>, BackendError> {
        Ok(Box::new(VaFrame {
            state: Arc::clone(&self.state),
            frame,
        }))
    }
}

/// One frame's encode on VA-API, which the session's encode thread finishes (module doc).
struct VaFrame {
    state: Arc<Mutex<State>>,
    frame: EncodeFrame,
}

impl HostEncodedFrame for VaFrame {
    fn finish(
        self: Box<Self>,
        source: &EncodeSource,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError> {
        let mut state = self.state.lock();
        if let Some(settings) = &self.frame.settings {
            state.apply(settings)?;
        }
        let codec = state.codec;
        let force_key_frame = self.frame.force_keyframe || state.key_next;
        let encoder = state.encoder.as_mut().ok_or("an encode with no settings")?;
        let duration = i64::from(encoder.settings().frame_rate.denominator);
        let index = i64::try_from(self.frame.frame_index).unwrap_or(i64::MAX);
        let input = InputFrame {
            planes: [plane(&source.luma)?, plane(&source.chroma)?],
            pts: index.saturating_mul(duration),
            duration,
            force_key_frame,
            id: self.frame.frame_index,
        };
        let mut coded = encoder.encode(&input)?;
        let sample = match (coded.pop(), coded.len()) {
            (Some(Ok(sample)), 0) => sample,
            (Some(Err(failed)), 0) => {
                let why = failed
                    .error
                    .map_or_else(|| "lost".to_owned(), |e| e.to_string());
                return Err(format!("the frame's picture: {why}").into());
            }
            (None, _) => return Err("the encoder gave no sample for the frame".into()),
            (Some(_), n) => return Err(format!("the encoder gave {} samples", n + 1).into()),
        };
        let configuration = sample.is_sync.then(|| encoder.configuration());
        let au = annex_b(codec, configuration.as_deref(), &sample.data)?;
        let mut encoded = Encoded {
            size: au.len() as u64,
            overflow: false,
            keyframe: sample.is_sync,
            parameter_sets: sample.is_sync,
            frame_index: sample.id,
            picture_type: match sample.frame_type {
                FrameType::Idr => RGFX_VIDEO_PICTURE_I,
                FrameType::P => RGFX_VIDEO_PICTURE_P,
                FrameType::B => RGFX_VIDEO_PICTURE_B,
            },
        };
        match out.get_mut(..au.len()) {
            Some(to) => {
                to.copy_from_slice(&au);
                state.key_next = false;
            }
            None => {
                // The frame is in the encoder's references: the next one starts afresh.
                encoded.overflow = true;
                state.key_next = true;
            }
        }
        Ok(encoded)
    }
}

/// The rows of `p` as the VA crate's encoder reads a plane: from its first row's first byte to
/// its last row's last.
fn plane(p: &SourcePlane) -> Result<Plane<'_>, BackendError> {
    let len = (p.rows.checked_sub(1))
        .and_then(|rows| rows.checked_mul(p.bytes_per_row))
        .and_then(|before| before.checked_add(p.row_bytes))
        .ok_or("an empty source plane")?;
    // SAFETY: the plane's rows lie in host bytes the session holds until the frame's signal,
    // which nothing writes before it (`video_host::encode`'s module doc), for as long as `p`,
    // the encode thread's, lives.
    let data = unsafe { std::slice::from_raw_parts(p.at, len) };
    Ok(Plane {
        data,
        bytes_per_row: p.bytes_per_row,
    })
}

/// The Annex-B access unit of `sample`, NAL units behind four-byte big-endian lengths, of
/// `codec`, the parameter sets of `configuration` (an `avcC` or `hvcC` record) before them where
/// it is given.
pub(super) fn annex_b(
    codec: u32,
    configuration: Option<&[u8]>,
    sample: &[u8],
) -> Result<Vec<u8>, BackendError> {
    let mut au = Vec::with_capacity(sample.len() + 256);
    if let Some(record) = configuration {
        let sets: Vec<&[u8]> = if codec == RGFX_VIDEO_CODEC_H264 {
            let config = reims_vgpu_video_vaapi::h264::avcc::DecoderConfiguration::parse(record)
                .map_err(|e| format!("the encoder's avcC: {e:?}"))?;
            (config.sequence_parameter_sets.into_iter())
                .chain(config.picture_parameter_sets)
                .collect()
        } else {
            let config = reims_vgpu_video_vaapi::hevc::hvcc::DecoderConfiguration::parse(record)
                .map_err(|e| format!("the encoder's hvcC: {e:?}"))?;
            config.nal_units
        };
        for set in sets {
            au.extend_from_slice(&START_CODE);
            au.extend_from_slice(set);
        }
    }
    let mut rest = sample;
    while !rest.is_empty() {
        let (Some(len), Some(after)) = (rest.get(..4), rest.get(4..)) else {
            return Err("a sample's NAL unit length cut short".into());
        };
        let len = u32::from_be_bytes([len[0], len[1], len[2], len[3]]) as usize;
        let unit = after.get(..len).ok_or("a sample's NAL unit past its end")?;
        au.extend_from_slice(&START_CODE);
        au.extend_from_slice(unit);
        rest = &after[len..];
    }
    Ok(au)
}

#[cfg(test)]
mod tests;
