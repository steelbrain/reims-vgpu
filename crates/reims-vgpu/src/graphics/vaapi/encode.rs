//! The reims-graphics device's hardware encode on VA-API: H.264 and HEVC on the VA crate's
//! `VideoEncoder` (`EncSlice`, else `EncSliceLP`), in a
//! [`HostEncodeSession`](crate::graphics::video_host::HostEncodeSession) whose event the host
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
//! - **A frame** ([`VaFrame`]) reaches the encoder on the session's encode thread one of two
//!   ways. By the runtime device's GPU, where the session has its copier (`super::gpu_copy`): a
//!   copy of the source's planes, where they lie, in the surface's pages or its device caches,
//!   into an input surface of the encoder's, exported as dma-bufs (the VA crate's
//!   `VideoEncoder::input_surface`), its margin to the coded size filled as an upload fills it, so
//!   the encoder codes the same bytes either way. Else, and where the copy is declined (said once
//!   a session, with why, at info), from the source's host bytes, which the VA crate's encoder
//!   uploads. Its sample, NAL units behind four-byte lengths, is written as Annex B: a keyframe's
//!   parameter sets first, from the encoder's configuration record (`avcC`, `hvcC`). One that
//!   does not fit the capacity is not written, and the next frame is made a keyframe.
//! - **VideoToolbox's paravirtual encoders** (`VaGuestEncoder`, `super::super::vt`): an arm64
//!   guest's encoders, one for each codec the driver encodes a profile of ([`guest_encoders`]),
//!   whose settings the guest's session properties make, `Quality` onto the quality levels as
//!   `quality` maps. Each source frame reaches the encoder as an x86 guest's frame does: copied by
//!   the runtime device's GPU from a Shared buffer over the IOSurface's pages, which the guest
//!   lent, into an input surface of the encoder's; else, and where the copy is declined (said
//!   once a session), from the host's view of the pages, which the VA crate's encoder uploads.
//!   A change of the bitrate, frame rate or key-frame interval goes to the encoder from the next
//!   frame; one that makes a new sequence makes a new encoder before the first frame, and after
//!   it is answered as taken and left out, the running sequence keeping its settings to its end,
//!   as Apple's hardware encoders keep theirs.

use std::sync::Arc;

use reims_vgpu_graphics::ops::video::{
    RGFX_VIDEO_ENCODE_HARD_LIMIT, RGFX_VIDEO_ENTROPY_CABAC, RGFX_VIDEO_ENTROPY_CAVLC,
    RGFX_VIDEO_PICTURE_B, RGFX_VIDEO_PICTURE_I, RGFX_VIDEO_PICTURE_P, RGFX_VIDEO_RATE_CBR,
    RGFX_VIDEO_RATE_CQP, RGFX_VIDEO_RATE_VBR, RgfxVideoEncodeSettings,
};
use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_ENCODE, RGFX_VIDEO_FORMAT_NV12,
    RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN,
    RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps, RgfxVideoEncodeCaps,
};
use reims_vgpu_video::codec::encode::{
    Coded, ColourDescription, EncoderSettings, Entropy, FailedSample, FrameRate, FrameType,
    InputFrame, Plane, Profile as EncodeProfile, RateControl,
};
use reims_vgpu_video::core::backend::BackendError as VideoError;
use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::core::surfaces::PlaneLayout;
use reims_vgpu_video::paravirt::{Encoder, EncoderConfig, SourceFrame};
use reims_vgpu_video::vaapi::{Capabilities, Display, EncoderSupport, InputSurface, VideoEncoder};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::super::decode::{GuestEncoder, GuestSource};
use super::super::video_host::{
    BackendError, EncodeFrame, EncodeSource, Encoded, FrameSource, HostEncodedFrame, HostEncoder,
    SourcePlane,
};
use super::super::vt_encode::{
    buffer_planes, encoders_of, host_planes, quality_percent, source_planes,
};
use super::{GpuFiller, H264_TOP_LEVEL, HEVC_TOP_LEVEL, MIN_SIZE, NAME};

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

/// A session's GPU filler, its frames' to share, or why it has none; and the reasons a frame was
/// uploaded instead, each said once.
struct Fills<P: Platform> {
    filler: Result<Box<dyn GpuFiller<P>>, String>,
    said: Vec<String>,
    /// The input surfaces of fills that failed on the GPU, which may write them still: kept out
    /// of the encoder's pool while the session lives.
    failed: Vec<InputSurface>,
}

impl<P: Platform> Fills<P> {
    /// Says, once a session, and at info once per process (`kind`), why a frame was uploaded by
    /// the CPU.
    fn uploaded_because(&mut self, why: String) {
        if !self.said.contains(&why) {
            log::info!(kind:% = why; "hardware video: {NAME}: a frame uploaded by the CPU: {why}");
            self.said.push(why);
        }
    }
}

/// A session's encoder on VA-API ([`HostEncoder`]).
pub(super) struct VaEncoder<P: Platform> {
    state: Arc<Mutex<State>>,
    /// What writes its frames into the encoder's input surfaces on the GPU (module doc).
    fills: Arc<Mutex<Fills<P>>>,
    /// The profiles the driver encodes, which [`HostEncoder::check`] holds settings to.
    supports: Arc<[EncoderSupport]>,
    /// The sizes the codec's entry gives.
    min_size: [u32; 2],
    max_size: [u32; 2],
    /// The rate controls the entry lists, `1 << RGFX_VIDEO_RATE_*` bits.
    rate_controls: u32,
}

impl<P: Platform> VaEncoder<P> {
    /// An encoder of `codec` on `display`, which encodes `supports`, as its entry `entry` and
    /// `encode` list them, its frames written into its input surfaces by `filler` where it is
    /// given (or why not).
    pub(super) fn new(
        display: &Display,
        codec: u32,
        supports: Arc<[EncoderSupport]>,
        entry: &RgfxVideoCodecCaps,
        encode: &RgfxVideoEncodeCaps,
        filler: Result<Box<dyn GpuFiller<P>>, String>,
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
            fills: Arc::new(Mutex::new(Fills {
                filler,
                said: Vec::new(),
                failed: Vec::new(),
            })),
            supports,
            min_size: entry.min_size,
            max_size: entry.max_size,
            rate_controls: encode.rate_controls,
        }
    }
}

impl<P: Platform> HostEncoder<P> for VaEncoder<P> {
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

    fn encode(&mut self, frame: EncodeFrame) -> Result<Box<dyn HostEncodedFrame<P>>, BackendError> {
        Ok(Box::new(self.frame(frame)))
    }
}

impl<P: Platform> VaEncoder<P> {
    /// The encode of `frame` ([`HostEncoder::encode`]).
    fn frame(&self, frame: EncodeFrame) -> VaFrame<P> {
        VaFrame {
            state: Arc::clone(&self.state),
            fills: Arc::clone(&self.fills),
            frame,
        }
    }
}

/// One frame's encode on VA-API, which the session's encode thread finishes (module doc).
struct VaFrame<P: Platform> {
    state: Arc<Mutex<State>>,
    fills: Arc<Mutex<Fills<P>>>,
    frame: EncodeFrame,
}

impl<P: Platform> HostEncodedFrame<P> for VaFrame<P> {
    fn finish(
        self: Box<Self>,
        source: &mut FrameSource<'_, P>,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError> {
        let mut state = self.state.lock();
        if let Some(settings) = &self.frame.settings {
            state.apply(settings)?;
        }
        let input = self.times(&state)?;
        if let Some(encoded) = self.filled(&mut state, &input, source, out)? {
            return Ok(encoded);
        }
        let host = source.host()?;
        encode_host(&mut state, &input, &host, out)
    }
}

impl<P: Platform> VaFrame<P> {
    /// The frame encoded from `source`'s planes where they lie, copied by the GPU into an input
    /// surface of the encoder's (module doc), into `out`; `None` where the copy was declined,
    /// nothing written, which is said once a session. `Err` where the encode failed, or the copy
    /// failed once on the GPU.
    fn filled(
        &self,
        state: &mut State,
        input: &InputFrame<'_>,
        source: &FrameSource<'_, P>,
        out: &mut [u8],
    ) -> Result<Option<Encoded>, BackendError> {
        let mut fills = self.fills.lock();
        let declined = match &mut fills.filler {
            Ok(filler) => {
                let encoder = state.encoder.as_mut().ok_or("an encode with no settings")?;
                match encoder.input_surface() {
                    Ok(surface) => {
                        let [luma, chroma] = source.planes();
                        let sample = encoder.settings().input_format().bytes_per_sample();
                        let sample = u32::try_from(sample).unwrap_or(0);
                        match filler.fill(&surface, luma, chroma, source.size(), sample) {
                            Ok(DeviceWrite::Written) => {
                                drop(fills);
                                let coded = encoder.encode_surface(surface, input)?;
                                return finish_coded(state, coded, out).map(Some);
                            }
                            Ok(DeviceWrite::Declined(why)) => format!("the GPU's copy: {why}"),
                            // The GPU may write the surface still: it stays out of the pool, and
                            // the session's frames are uploaded from here on.
                            Err(error) => {
                                fills.failed.push(surface);
                                fills.filler = Err(format!("the GPU's copy failed: {error}"));
                                return Err(error);
                            }
                        }
                    }
                    // The driver's surfaces export so for none of its frames: the session's
                    // frames are uploaded from here on.
                    Err(why) => {
                        let why = format!("no input surface for the GPU's copy: {why}");
                        fills.filler = Err(why.clone());
                        why
                    }
                }
            }
            Err(why) => why.clone(),
        };
        fills.uploaded_because(declined);
        Ok(None)
    }

    /// [`HostEncodedFrame::finish`] from `source`'s host bytes, which the encoder uploads.
    #[cfg(test)]
    #[cfg(unix)]
    fn finish_from_host(
        self,
        source: &EncodeSource,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError> {
        let mut state = self.state.lock();
        if let Some(settings) = &self.frame.settings {
            state.apply(settings)?;
        }
        let input = self.times(&state)?;
        encode_host(&mut state, &input, source, out)
    }

    /// The frame's times, key-frame demand and tag as the encoder takes them, with no planes.
    fn times(&self, state: &State) -> Result<InputFrame<'static>, BackendError> {
        let encoder = state.encoder.as_ref().ok_or("an encode with no settings")?;
        let duration = i64::from(encoder.settings().frame_rate.denominator);
        let index = i64::try_from(self.frame.frame_index).unwrap_or(i64::MAX);
        let none = Plane {
            data: &[],
            bytes_per_row: 0,
        };
        Ok(InputFrame {
            planes: [none, none],
            pts: index.saturating_mul(duration),
            duration,
            force_key_frame: self.frame.force_keyframe || state.key_next,
            id: self.frame.frame_index,
        })
    }
}

/// Encodes the frame of `input`'s times from `source`'s host bytes, which the encoder uploads,
/// into `out`.
fn encode_host(
    state: &mut State,
    input: &InputFrame<'_>,
    source: &EncodeSource,
    out: &mut [u8],
) -> Result<Encoded, BackendError> {
    let encoder = state.encoder.as_mut().ok_or("an encode with no settings")?;
    let input = InputFrame {
        planes: [plane(&source.luma)?, plane(&source.chroma)?],
        ..*input
    };
    let coded = encoder.encode(&input)?;
    finish_coded(state, coded, out)
}

/// Writes the one sample of `coded`, the frame's, into `out` as an Annex-B access unit where it
/// fits (module doc), and answers what came of it.
fn finish_coded(
    state: &mut State,
    mut coded: Vec<Coded<reims_vgpu_video::vaapi::Error>>,
    out: &mut [u8],
) -> Result<Encoded, BackendError> {
    let codec = state.codec;
    let encoder = state.encoder.as_mut().ok_or("an encode with no settings")?;
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
            let config = reims_vgpu_video::codec::h264::avcc::DecoderConfiguration::parse(record)
                .map_err(|e| format!("the encoder's avcC: {e:?}"))?;
            (config.sequence_parameter_sets.into_iter())
                .chain(config.picture_parameter_sets)
                .collect()
        } else {
            let config = reims_vgpu_video::codec::hevc::hvcc::DecoderConfiguration::parse(record)
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

/// VideoToolbox's paravirtual encoders the driver's profiles make (module doc): one for each
/// codec of which it encodes a profile, with those profiles and the largest picture any of them
/// takes.
pub(super) fn guest_encoders(supports: &[EncoderSupport]) -> Vec<Encoder> {
    encoders_of((supports.iter()).map(|s| (s.profile, s.max_width, s.max_height)))
}

/// A VideoToolbox paravirtual encoder on the VA crate's `VideoEncoder` (module doc), its source
/// frames read from the surfaces the guest lends.
pub(super) struct VaGuestEncoder<P: Platform> {
    display: Display,
    /// The quality range of each profile the driver encodes.
    ranges: Vec<(EncodeProfile, Option<u32>)>,
    encoder: VideoEncoder,
    settings: EncoderSettings,
    /// Whether a frame was encoded: until then, settings that make a new sequence make a new
    /// encoder.
    encoded: bool,
    /// What writes its source frames into the encoder's input surfaces on the GPU (module doc).
    fills: Fills<P>,
}

impl<P: Platform> VaGuestEncoder<P> {
    /// An encoder of `config` on `display`, whose driver encodes `supports`, its frames written
    /// into its input surfaces by `filler` where it is given (or why not).
    pub(super) fn new(
        display: &Display,
        supports: &[EncoderSupport],
        config: &EncoderConfig,
        filler: Result<Box<dyn GpuFiller<P>>, String>,
    ) -> Result<Self, VideoError> {
        let ranges: Vec<_> = (supports.iter())
            .map(|s| (s.profile, s.capabilities.quality_range))
            .collect();
        let settings = guest_settings(config, &ranges);
        let encoder = VideoEncoder::new(display, settings.clone()).map_err(super::video_error)?;
        Ok(Self {
            display: display.clone_ref(),
            ranges,
            encoder,
            settings,
            encoded: false,
            fills: Fills {
                filler,
                said: Vec::new(),
                failed: Vec::new(),
            },
        })
    }

    /// The frame of `times` encoded from `planes`, the source surface's luma and chroma in
    /// `buffer`, over its pages, copied by the GPU into an input surface of the encoder's
    /// (module doc); `None` where the copy was declined, nothing written, which is said once a
    /// session. `Err` where the encode failed, or the copy failed once on the GPU.
    fn filled(
        &mut self,
        buffer: &P::Buffer,
        planes: &[PlaneLayout; 2],
        times: &InputFrame<'_>,
    ) -> Result<Option<Vec<Coded<reims_vgpu_video::vaapi::Error>>>, VideoError> {
        let declined = match &mut self.fills.filler {
            Ok(filler) => match self.encoder.input_surface() {
                Ok(surface) => {
                    let (width, height) = (self.settings.width, self.settings.height);
                    let sample = self.settings.input_format().bytes_per_sample();
                    let sample = u32::try_from(sample).unwrap_or(0);
                    let [luma, chroma] = buffer_planes(buffer, planes, (width, height), sample);
                    match filler.fill(&surface, &luma, &chroma, (width, height), sample) {
                        Ok(DeviceWrite::Written) => {
                            let coded = (self.encoder.encode_surface(surface, times))
                                .map_err(super::video_error)?;
                            return Ok(Some(coded));
                        }
                        Ok(DeviceWrite::Declined(why)) => format!("the GPU's copy: {why}"),
                        // The GPU may write the surface still: it stays out of the pool, and the
                        // session's frames are uploaded from here on.
                        Err(error) => {
                            self.fills.failed.push(surface);
                            self.fills.filler = Err(format!("the GPU's copy failed: {error}"));
                            return Err(VideoError::Malfunction(error.to_string()));
                        }
                    }
                }
                // The driver's surfaces export so for none of its frames: the session's frames
                // are uploaded from here on.
                Err(why) => {
                    let why = format!("no input surface for the GPU's copy: {why}");
                    self.fills.filler = Err(why.clone());
                    why
                }
            },
            Err(why) => why.clone(),
        };
        self.fills.uploaded_because(declined);
        Ok(None)
    }
}

/// The VA crate's settings of `config`: its settings, `Quality` (0 to 1) onto the profile's
/// quality levels as the device's `quality` (0 to 100) maps onto them.
fn guest_settings(
    config: &EncoderConfig,
    ranges: &[(EncodeProfile, Option<u32>)],
) -> EncoderSettings {
    let mut settings = config.settings.clone();
    let range = (ranges.iter())
        .find(|(profile, _)| *profile == settings.profile)
        .and_then(|(_, range)| *range);
    settings.quality_level =
        (config.quality).and_then(|quality| quality_level(quality_percent(quality), range));
    settings
}

/// `coded`'s errors as a guest's session sees them.
fn guest_coded(coded: Vec<Coded<reims_vgpu_video::vaapi::Error>>) -> Vec<Coded<VideoError>> {
    (coded.into_iter())
        .map(|coded| {
            coded.map_err(|failed| FailedSample {
                id: failed.id,
                error: failed.error.map(super::video_error),
            })
        })
        .collect()
}

impl<P: Platform> GuestEncoder<P> for VaGuestEncoder<P> {
    fn encode(
        &mut self,
        source: &GuestSource<'_, P>,
        frame: &SourceFrame,
    ) -> Result<Vec<Coded<VideoError>>, VideoError> {
        let [luma, chroma] = source_planes(source, &self.settings)?;
        let none = Plane {
            data: &[],
            bytes_per_row: 0,
        };
        let times = InputFrame {
            planes: [none, none],
            pts: frame.pts,
            duration: frame.duration,
            force_key_frame: frame.force_key_frame,
            id: frame.id,
        };
        if let Some(buffer) = source.buffer
            && let Some(coded) = self.filled(buffer, &[luma, chroma], &times)?
        {
            self.encoded = true;
            return Ok(guest_coded(coded));
        }
        let view = (source.memory.host_view())
            .map_err(|error| VideoError::Malfunction(error.to_string()))?;
        let [y, uv] = host_planes(&view, &[luma, chroma])?;
        let input = InputFrame {
            planes: [
                Plane {
                    data: y,
                    bytes_per_row: luma.bytes_per_row,
                },
                Plane {
                    data: uv,
                    bytes_per_row: chroma.bytes_per_row,
                },
            ],
            ..times
        };
        let coded = self.encoder.encode(&input).map_err(super::video_error)?;
        drop(view);
        self.encoded = true;
        Ok(guest_coded(coded))
    }

    fn flush(&mut self) -> Result<Vec<Coded<VideoError>>, VideoError> {
        Ok(guest_coded(self.encoder.flush()))
    }

    /// The bitrate, the frame rate and the key-frame interval go to the encoder from the next
    /// frame; anything else makes a new encoder before the first frame, and after it is taken
    /// and left out: the running sequence keeps its own to its end.
    fn reconfigure(&mut self, config: &EncoderConfig) -> Result<(), VideoError> {
        let mut new = guest_settings(config, &self.ranges);
        let mut same_sequence = self.settings.clone();
        same_sequence.rate_control = new.rate_control;
        same_sequence.frame_rate = new.frame_rate;
        same_sequence.key_frame_interval = new.key_frame_interval;
        same_sequence.key_frame_interval_duration = new.key_frame_interval_duration;
        if same_sequence != new {
            if self.encoded {
                new = same_sequence;
            } else {
                self.encoder =
                    VideoEncoder::new(&self.display, new.clone()).map_err(super::video_error)?;
                self.settings = new;
                return Ok(());
            }
        }
        if new.rate_control != self.settings.rate_control {
            (self.encoder.set_rate_control(new.rate_control)).map_err(super::video_error)?;
        }
        if new.frame_rate != self.settings.frame_rate {
            (self.encoder.set_frame_rate(new.frame_rate)).map_err(super::video_error)?;
        }
        (self.encoder)
            .set_key_frame_interval(new.key_frame_interval, new.key_frame_interval_duration);
        self.settings = new;
        Ok(())
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests;
