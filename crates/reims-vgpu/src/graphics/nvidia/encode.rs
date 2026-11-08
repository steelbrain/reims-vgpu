//! The reims-graphics device's hardware encode on NVENC: H.264 and HEVC on the NVIDIA crate's
//! `NvencEncoder`, in a [`HostEncodeSession`] whose event the host signals, as the VA-API arm
//! encodes.
//!
//! - **What it answers** ([`encode_caps_of`]) is NVENC's: an entry for each codec of which it
//!   encodes a profile (the NVIDIA crate's `encode_profiles`), with those profiles, NV12 sources
//!   (and P010 for HEVC Main 10), the least of their sizes, the codec's top level, and the
//!   references NVENC takes. Its rate controls ([`encode_caps`]) are CQP, and CBR and VBR where
//!   NVENC lists them, and no B-frames: the encoder keeps frames in display order, a frame in and
//!   its access unit out.
//! - **The settings** ([`params_of`]) map onto the NVIDIA crate's: the profile, size, level,
//!   entropy coder and colour; a VBR with no peak peaks at 1.05 times its average, as the VA-API
//!   arm's does; `quality` 0 to 100 onto NVENC's presets P1 (the fastest) to P7 (the best), at
//!   its high-quality tuning, whose lookahead the crate turns off; the keyframe interval onto
//!   the GOP, and with none NVENC places no IDR picture of its own. NVENC holds every stream to
//!   its VBV buffer, the window's length at the peak, so the wire's `HARD_LIMIT` asks nothing
//!   more. A change of what makes a new sequence makes a new encoder, whose first access unit is
//!   an IDR picture with its parameter sets; a change of the bitrates, QPs or frame rate goes to
//!   the encoder from the next frame. `min_qp`, `max_qp` and a change of `quality` alone are not
//!   taken, as the VA-API arm takes none of them.
//! - **A frame** ([`NvencFrame`]) is read on the session's encode thread from the source's host
//!   bytes into NVENC's input buffer and encoded there. NVENC writes the access unit in Annex-B
//!   form, its parameter sets before each IDR picture (the crate's `repeatSPSPPS`), so it is
//!   written as it comes. One that does not fit the capacity is not written, and the next frame
//!   is made an IDR picture.

use std::sync::Arc;

use reims_vgpu_graphics::ops::video::{
    RGFX_VIDEO_ENTROPY_CABAC, RGFX_VIDEO_ENTROPY_CAVLC, RGFX_VIDEO_PICTURE_B, RGFX_VIDEO_PICTURE_I,
    RGFX_VIDEO_PICTURE_P, RGFX_VIDEO_RATE_CBR, RGFX_VIDEO_RATE_CQP, RGFX_VIDEO_RATE_VBR,
    RgfxVideoEncodeSettings,
};
use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_ENCODE, RGFX_VIDEO_FORMAT_NV12,
    RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN,
    RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps, RgfxVideoEncodeCaps,
};
use reims_vgpu_video::nvidia::{
    EncodeParams, EncodeProfile, EncodeProfileSupport, EncodeRateControl, Gpu, NvencEncoder,
    PictureType, RawFrame, Tuning,
};
use reims_vgpu_vulkan::core::sync::Mutex;

use super::super::video_host::{
    BackendError, EncodeFrame, EncodeSource, Encoded, HostEncodedFrame, HostEncoder, SourcePlane,
};
use super::{H264_TOP_LEVEL, HEVC_TOP_LEVEL};

/// Ours, as the VA-API arm's: a VBR's peak where the guest sets none, as a share of its
/// average: 1.05.
const PEAK_NUMERATOR: u64 = 21;
const PEAK_DENOMINATOR: u64 = 20;

/// Ours, as the VA-API arm's: the frame rate where the guest says none.
const DEFAULT_FRAME_RATE: (u32, u32) = (30, 1);

/// The VBV buffer's length, at the peak, where the guest gives no window: a second.
const DEFAULT_BUFFER_MILLISECONDS: u32 = 1000;

/// The wire's profile bit and codec of an encode profile.
fn wire_profile(profile: EncodeProfile) -> (u32, u32) {
    match profile {
        EncodeProfile::H264Baseline => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_BASELINE),
        EncodeProfile::H264Main => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_MAIN),
        EncodeProfile::H264High => (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH),
        EncodeProfile::HevcMain => (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN),
        EncodeProfile::HevcMain10 => (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10),
    }
}

/// The encode profile of the wire's `profile` bit of `codec`.
fn encode_profile(codec: u32, profile: u32) -> Option<EncodeProfile> {
    Some(match (codec, profile) {
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_BASELINE) => EncodeProfile::H264Baseline,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_MAIN) => EncodeProfile::H264Main,
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_H264_HIGH) => EncodeProfile::H264High,
        (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN) => EncodeProfile::HevcMain,
        (RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_HEVC_MAIN_10) => EncodeProfile::HevcMain10,
        _ => return None,
    })
}

/// The caps of H.264 and HEVC encode `supports` gives (module doc): an entry for each codec of
/// which NVENC encodes a profile, at the least of its profiles' sizes and references.
pub(super) fn encode_caps_of(supports: &[EncodeProfileSupport]) -> Vec<RgfxVideoCodecCaps> {
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
                    min_size: [1; 2],
                    max_size: [u32::MAX; 2],
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
        entry.min_size = [
            entry.min_size[0].max(support.min_width),
            entry.min_size[1].max(support.min_height),
        ];
        entry.max_size = [
            entry.max_size[0].min(support.max_width),
            entry.max_size[1].min(support.max_height),
        ];
        entry.max_references = entry.max_references.min(support.max_references.max(1));
        entry.max_dpb_pictures = entry.max_references.saturating_add(1);
    }
    entries
}

/// What an encode of `codec`, an entry [`encode_caps_of`] made, takes beyond it: CQP, and CBR
/// and VBR where NVENC lists them for each of its profiles, and no B-frames (module doc); all
/// zero for a decode entry.
pub(super) fn encode_caps(
    codec: &RgfxVideoCodecCaps,
    supports: &[EncodeProfileSupport],
) -> RgfxVideoEncodeCaps {
    if codec.direction != RGFX_VIDEO_ENCODE {
        return RgfxVideoEncodeCaps::default();
    }
    let profiles = || (supports.iter()).filter(|s| wire_profile(s.profile).0 == codec.codec);
    let mut rate_controls = 1 << RGFX_VIDEO_RATE_CQP;
    if profiles().all(|s| s.constant_bit_rate) {
        rate_controls |= 1 << RGFX_VIDEO_RATE_CBR;
    }
    if profiles().all(|s| s.variable_bit_rate) {
        rate_controls |= 1 << RGFX_VIDEO_RATE_VBR;
    }
    RgfxVideoEncodeCaps {
        rate_controls,
        max_b_frames: 0,
        flags: 0,
        reserved_0c: 0,
    }
}

/// A rate in bits a second as the NVIDIA crate's settings take it: a u32's largest past one, and
/// at least one.
fn bits(rate: u64) -> u32 {
    u32::try_from(rate).unwrap_or(u32::MAX).max(1)
}

/// The rate control `settings` asks for (module doc).
fn rate_of(settings: &RgfxVideoEncodeSettings) -> EncodeRateControl {
    let qp = |qp: u32| qp.min(51);
    match settings.rate_control {
        RGFX_VIDEO_RATE_CQP => EncodeRateControl::ConstantQp {
            intra: qp(settings.qp_i),
            predicted: qp(settings.qp_p),
            bidirectional: qp(settings.qp_b),
        },
        RGFX_VIDEO_RATE_CBR => EncodeRateControl::ConstantBitRate {
            bits_per_second: bits(settings.average_bitrate),
        },
        _ => {
            let average = bits(settings.average_bitrate);
            let peak = match settings.max_bitrate {
                0 => (u64::from(average).checked_mul(PEAK_NUMERATOR))
                    .map_or(u32::MAX, |b| bits(b / PEAK_DENOMINATOR)),
                peak => bits(peak),
            };
            EncodeRateControl::VariableBitRate {
                average_bits_per_second: average,
                peak_bits_per_second: peak.max(average),
                target_quality: None,
            }
        }
    }
}

/// The frame rate `settings` asks for; [`DEFAULT_FRAME_RATE`] where it says none.
fn frame_rate_of(settings: &RgfxVideoEncodeSettings) -> (u32, u32) {
    match (settings.frame_rate_num, settings.frame_rate_den) {
        (0, _) | (_, 0) => DEFAULT_FRAME_RATE,
        rate => rate,
    }
}

/// NVENC's preset of `quality`, 0 the fastest to 100 the best: P1 to P7, the nearest.
fn preset_of(quality: u32) -> u8 {
    let quality = quality.min(100);
    u8::try_from(1 + (quality * 6 + 50) / 100).unwrap_or(7)
}

/// The NVIDIA crate's settings of `settings` of `codec` (module doc).
pub(super) fn params_of(
    codec: u32,
    settings: &RgfxVideoEncodeSettings,
) -> Result<EncodeParams, String> {
    let profile = encode_profile(codec, settings.profile)
        .ok_or_else(|| format!("profile {:#x} of codec {codec}", settings.profile))?;
    let (primaries, transfer, matrix) =
        (settings.color_primaries, settings.transfer, settings.matrix);
    Ok(EncodeParams {
        profile,
        width: settings.width,
        height: settings.height,
        frame_rate: frame_rate_of(settings),
        rate_control: rate_of(settings),
        buffer_milliseconds: match settings.max_bitrate_window_ms {
            0 => DEFAULT_BUFFER_MILLISECONDS,
            window => window,
        },
        gop_length: (settings.key_frame_interval > 0).then_some(settings.key_frame_interval),
        b_frames: 0,
        level: (settings.level > 0).then_some(settings.level),
        cabac: match settings.entropy {
            RGFX_VIDEO_ENTROPY_CAVLC => Some(false),
            RGFX_VIDEO_ENTROPY_CABAC => Some(true),
            _ => None,
        },
        full_range: settings.full_range != 0,
        colour: (primaries != 0 || transfer != 0 || matrix != 0)
            .then_some((primaries, transfer, matrix)),
        display_aspect: None,
        preset: preset_of(settings.quality),
        tuning: Tuning::HighQuality,
    })
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
    gpu: Arc<Gpu>,
    codec: u32,
    encoder: Option<NvencEncoder>,
    settings: Option<RgfxVideoEncodeSettings>,
    /// Whether the next frame is made an IDR picture: the last one did not fit.
    key_next: bool,
}

impl State {
    /// Takes `settings` from the next frame on (module doc).
    fn apply(&mut self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError> {
        match (self.encoder.as_mut(), self.settings.as_ref()) {
            (Some(encoder), Some(old)) if !new_sequence(old, settings) => {
                let (rate, frame_rate) = (rate_of(settings), frame_rate_of(settings));
                if rate_of(old) != rate || frame_rate_of(old) != frame_rate {
                    encoder.reconfigure(rate, frame_rate)?;
                }
            }
            _ => {
                let params = params_of(self.codec, settings)?;
                self.encoder = None;
                self.encoder = Some(NvencEncoder::new(Arc::clone(&self.gpu), &params)?);
            }
        }
        self.settings = Some(*settings);
        Ok(())
    }
}

/// A session's encoder on NVENC ([`HostEncoder`]).
pub(super) struct NvencHostEncoder {
    state: Arc<Mutex<State>>,
    /// The profiles NVENC encodes of the codec, which [`HostEncoder::check`] holds settings to.
    profiles: Vec<EncodeProfile>,
    /// The sizes the codec's entry gives.
    min_size: [u32; 2],
    max_size: [u32; 2],
    /// The rate controls the entry lists, `1 << RGFX_VIDEO_RATE_*` bits.
    rate_controls: u32,
}

impl NvencHostEncoder {
    /// An encoder of `codec` on `gpu`, which encodes `supports`, as its entry `entry` and
    /// `encode` list them.
    pub(super) fn new(
        gpu: &Arc<Gpu>,
        codec: u32,
        supports: &[EncodeProfileSupport],
        entry: &RgfxVideoCodecCaps,
        encode: &RgfxVideoEncodeCaps,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                gpu: Arc::clone(gpu),
                codec,
                encoder: None,
                settings: None,
                key_next: false,
            })),
            profiles: (supports.iter())
                .map(|s| s.profile)
                .filter(|p| wire_profile(*p).0 == codec)
                .collect(),
            min_size: entry.min_size,
            max_size: entry.max_size,
            rate_controls: encode.rate_controls,
        }
    }
}

impl HostEncoder for NvencHostEncoder {
    fn check(&self, settings: &RgfxVideoEncodeSettings) -> Result<(), BackendError> {
        let codec = self.state.lock().codec;
        let profile = encode_profile(codec, settings.profile)
            .filter(|p| self.profiles.contains(p))
            .ok_or_else(|| {
                format!(
                    "profile {:#x}, which NVENC does not encode",
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
                "rate control {}, which NVENC does not list",
                settings.rate_control
            )
            .into());
        }
        Ok(())
    }

    fn encode(&mut self, frame: EncodeFrame) -> Result<Box<dyn HostEncodedFrame>, BackendError> {
        Ok(Box::new(NvencFrame {
            state: Arc::clone(&self.state),
            frame,
        }))
    }
}

/// One frame's encode on NVENC, which the session's encode thread finishes (module doc).
struct NvencFrame {
    state: Arc<Mutex<State>>,
    frame: EncodeFrame,
}

impl HostEncodedFrame for NvencFrame {
    fn finish(
        self: Box<Self>,
        source: &EncodeSource,
        out: &mut [u8],
    ) -> Result<Encoded, BackendError> {
        let mut state = self.state.lock();
        if let Some(settings) = &self.frame.settings {
            state.apply(settings)?;
        }
        let force_idr = self.frame.force_keyframe || state.key_next;
        let encoder = state.encoder.as_mut().ok_or("an encode with no settings")?;
        let raw = RawFrame {
            luma: (plane(&source.luma)?, source.luma.bytes_per_row),
            chroma: (plane(&source.chroma)?, source.chroma.bytes_per_row),
            index: self.frame.frame_index,
            force_idr,
        };
        let mut packets = encoder.encode(&raw)?;
        let packet = match (packets.pop(), packets.len()) {
            (Some(Ok(packet)), 0) => packet,
            (Some(Err(error)), 0) => return Err(format!("the frame's access unit: {error}").into()),
            (None, _) => return Err("NVENC gave no access unit for the frame".into()),
            (Some(_), n) => return Err(format!("NVENC gave {} access units", n + 1).into()),
        };
        let keyframe = packet.picture_type == PictureType::Idr;
        let mut encoded = Encoded {
            size: packet.data.len() as u64,
            overflow: false,
            keyframe,
            // NVENC writes the parameter sets before each IDR picture (module doc).
            parameter_sets: keyframe,
            frame_index: packet.index,
            picture_type: match packet.picture_type {
                PictureType::Idr | PictureType::Intra => RGFX_VIDEO_PICTURE_I,
                PictureType::Predicted | PictureType::NonReferencePredicted => RGFX_VIDEO_PICTURE_P,
                PictureType::Bidirectional => RGFX_VIDEO_PICTURE_B,
            },
        };
        match out.get_mut(..packet.data.len()) {
            Some(to) => {
                to.copy_from_slice(&packet.data);
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

/// The rows of `p` as NVENC's upload reads a plane: from its first row's first byte to its last
/// row's last.
fn plane(p: &SourcePlane) -> Result<&[u8], BackendError> {
    let len = (p.rows.checked_sub(1))
        .and_then(|rows| rows.checked_mul(p.bytes_per_row))
        .and_then(|before| before.checked_add(p.row_bytes))
        .ok_or("an empty source plane")?;
    // SAFETY: the plane's rows lie in host bytes the session holds until the frame's signal,
    // which nothing writes before it (`video_host::encode`'s module doc), for as long as `p`,
    // the encode thread's, lives.
    Ok(unsafe { std::slice::from_raw_parts(p.at, len) })
}

#[cfg(test)]
mod tests;
