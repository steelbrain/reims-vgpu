//! VideoToolbox's paravirtual encoders on NVENC ([`NvGuestEncoder`], `super::super::super::vt`):
//! an arm64 guest's encoders, one for each codec of which NVENC encodes a profile
//! ([`guest_encoders`]), whose settings the guest's session properties make.
//!
//! - **The settings** ([`params_of`]) map onto the NVIDIA crate's as an x86 guest's do (`super`'s
//!   module doc): the profile (Constrained Baseline onto NVENC's Baseline), size, level, entropy
//!   coder, colour, rate control and VBV buffer; `Quality` onto the preset and tuning the device's
//!   quality takes (`super::speed_of`), the middle where the guest sets none
//!   ([`DEFAULT_QUALITY`]). No B-frames: a frame in and its sample out, its decode time its
//!   presentation time, whether the guest allows frame reordering or not.
//! - **Key frames** are placed here ([`KeyFrames`]), and NVENC places none of its own: the first
//!   frame, a forced one, the first past the key-frame interval in frames or in time, and the
//!   next frame after one NVENC did not take or whose sample failed are IDR pictures.
//! - **A frame** reaches NVENC as an x86 guest's does: copied by the runtime device's GPU from a
//!   Shared buffer over the IOSurface's pages, which the guest lent, into memory NVENC reads
//!   (`super::fill`); else, and where the copy or NVENC's read of it is declined (said once a
//!   session), from the host's view of the pages, which NVENC's input buffer takes by the CPU's
//!   copy. NVENC encodes the same bytes either way.
//! - **Its samples** ([`Samples`]) are NVENC's access units, each with its parameter sets taken
//!   out into the configuration record ([`Record`]): the record of the sets NVENC gives as it is
//!   made goes with the first sample, and one of an IDR picture's sets that differ, with that
//!   picture's sample.
//! - **A change** of the bitrate or the frame rate goes to the encoder from the next frame
//!   (`NvencEncoder::reconfigure`), and of the key-frame interval to the placing of key frames;
//!   one of anything else NVENC is made with, or of the rate control's mode, makes a new encoder
//!   before the first frame, and after it is answered as taken and left out, the running
//!   sequence keeping its settings to its end, as the VA-API arm's. Past the first frame a
//!   constant QP, a `Quality`'s, stays as it is.

use std::collections::VecDeque;
use std::sync::Arc;

use reims_vgpu_video::codec::encode::gop::KeyFrames;
use reims_vgpu_video::codec::encode::record::Record;
use reims_vgpu_video::codec::encode::{
    Codec as EncodeCodec, Coded, EncodedSample, Entropy, FailedSample, FrameType, Profile,
    RateControl,
};
use reims_vgpu_video::core::backend::BackendError as VideoError;
use reims_vgpu_video::nvidia::{
    Bitstream, EncodeParams, EncodeProfile, EncodeProfileSupport, EncodeRateControl, Gpu,
    NvencEncoder, PictureType, RawFrame,
};
use reims_vgpu_video::paravirt::{Encoder, EncoderConfig, SourceFrame};
use reims_vgpu_vulkan::core::platform::Platform;

use super::super::super::decode::{
    GuestEncoder, GuestSource, rate_past_first_frame, same_rate_mode,
};
use super::super::super::vt_encode::{
    buffer_planes, encoders_of, host_planes, quality_percent, source_planes,
};
use super::super::video_error;
use super::{Filler, Fills, speed_of};

/// The device's quality where the guest sets no `Quality`: the middle, 50, which the x86 guest's
/// video driver asks of each of its encodes.
const DEFAULT_QUALITY: u32 = 50;

/// The encoder profile of a profile the guest names.
fn nvenc_profile(profile: Profile) -> EncodeProfile {
    match profile {
        Profile::H264ConstrainedBaseline => EncodeProfile::H264Baseline,
        Profile::H264Main => EncodeProfile::H264Main,
        Profile::H264High => EncodeProfile::H264High,
        Profile::HevcMain => EncodeProfile::HevcMain,
        Profile::HevcMain10 => EncodeProfile::HevcMain10,
    }
}

/// The profile the guest names of an encoder profile.
fn guest_profile(profile: EncodeProfile) -> Profile {
    match profile {
        EncodeProfile::H264Baseline => Profile::H264ConstrainedBaseline,
        EncodeProfile::H264Main => Profile::H264Main,
        EncodeProfile::H264High => Profile::H264High,
        EncodeProfile::HevcMain => Profile::HevcMain,
        EncodeProfile::HevcMain10 => Profile::HevcMain10,
    }
}

/// VideoToolbox's paravirtual encoders `supports`, NVENC's profiles, make (module doc): one for
/// each codec of which it encodes a profile, with those profiles and the largest picture any of
/// them takes.
pub(in super::super) fn guest_encoders(supports: &[EncodeProfileSupport]) -> Vec<Encoder> {
    encoders_of((supports.iter()).map(|s| (guest_profile(s.profile), s.max_width, s.max_height)))
}

/// NVENC's rate control of `rate`: a QVBR as a VBR at its target quality.
fn rate_of(rate: RateControl) -> EncodeRateControl {
    let qp = |qp: u8| u32::from(qp.min(51));
    match rate {
        RateControl::ConstantQp {
            intra,
            predicted,
            bidirectional,
        } => EncodeRateControl::ConstantQp {
            intra: qp(intra),
            predicted: qp(predicted),
            bidirectional: qp(bidirectional),
        },
        RateControl::ConstantBitRate { bits_per_second } => EncodeRateControl::ConstantBitRate {
            bits_per_second: bits_per_second.max(1),
        },
        RateControl::VariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
        } => EncodeRateControl::VariableBitRate {
            average_bits_per_second: average_bits_per_second.max(1),
            peak_bits_per_second: peak_bits_per_second.max(average_bits_per_second).max(1),
            target_quality: None,
        },
        RateControl::QualityVariableBitRate {
            average_bits_per_second,
            peak_bits_per_second,
            quality,
        } => EncodeRateControl::VariableBitRate {
            average_bits_per_second: average_bits_per_second.max(1),
            peak_bits_per_second: peak_bits_per_second.max(average_bits_per_second).max(1),
            target_quality: Some(quality),
        },
    }
}

/// The NVIDIA crate's settings of `config` (module doc).
fn params_of(config: &EncoderConfig) -> EncodeParams {
    let s = &config.settings;
    let quality = config.quality.map_or(DEFAULT_QUALITY, quality_percent);
    let (preset, tuning) = speed_of(quality);
    EncodeParams {
        profile: nvenc_profile(s.profile),
        width: s.width,
        height: s.height,
        frame_rate: (s.frame_rate.numerator, s.frame_rate.denominator),
        rate_control: rate_of(s.rate_control),
        buffer_milliseconds: s.buffer_milliseconds,
        gop_length: None,
        b_frames: 0,
        level: s.level.map(u32::from),
        cabac: (s.profile.codec() == EncodeCodec::H264).then(|| s.h264_entropy() == Entropy::Cabac),
        full_range: s.full_range,
        colour: (s.colour).map(|c| (c.primaries, c.transfer, c.matrix)),
        display_aspect: (s.sample_aspect_ratio).map(|(across, down)| {
            (
                s.width.saturating_mul(u32::from(across)),
                s.height.saturating_mul(u32::from(down)),
            )
        }),
        preset,
        tuning,
    }
}

/// An encoder's samples of the access units NVENC writes (module doc), and the frames it took
/// whose samples are not out yet.
struct Samples {
    record: Record,
    /// The frames NVENC took, oldest first: with no B-frames, its access units come out in the
    /// order they went in.
    pending: VecDeque<SourceFrame>,
    /// Whether the next frame is made an IDR picture: a frame NVENC did not take, or whose
    /// sample failed, left the guest's stream without a picture it may predict from.
    restart: bool,
}

impl Samples {
    /// The samples of NVENC's `bitstreams`, in decode order, each with its own result.
    fn of(&mut self, bitstreams: Vec<Bitstream>) -> Vec<Coded<VideoError>> {
        (bitstreams.into_iter())
            .filter_map(|bitstream| self.sample(bitstream))
            .collect()
    }

    /// The sample of the oldest pending frame in `bitstream`, or why there is none; `None` where
    /// no frame is pending, which NVENC never answers.
    fn sample(&mut self, bitstream: Bitstream) -> Option<Coded<VideoError>> {
        let frame = self.pending.pop_front()?;
        let failed = |error| {
            Err(FailedSample {
                id: frame.id,
                error: Some(error),
            })
        };
        let packet = match bitstream {
            Ok(packet) => packet,
            Err(error) => {
                self.restart = true;
                return Some(failed(video_error(error)));
            }
        };
        let idr = packet.picture_type == PictureType::Idr;
        let data = match self.record.sample_data(&packet.data, idr) {
            Ok(data) => data,
            Err(why) => {
                self.restart = true;
                return Some(failed(VideoError::Malfunction(why.0)));
            }
        };
        let intra = matches!(packet.picture_type, PictureType::Idr | PictureType::Intra);
        Some(Ok(EncodedSample {
            id: frame.id,
            data,
            pts: frame.pts,
            dts: frame.pts,
            duration: frame.duration,
            // A sample's type has no I picture but an IDR one; its flags say what it is.
            frame_type: match packet.picture_type {
                PictureType::Idr => FrameType::Idr,
                PictureType::Bidirectional => FrameType::B,
                _ => FrameType::P,
            },
            is_sync: idr,
            depends_on_others: !intra,
            is_depended_on: packet.picture_type.is_reference(),
            configuration: self.record.take_due(),
        }))
    }
}

/// A VideoToolbox paravirtual encoder on NVENC (module doc), its source frames read from the
/// surfaces the guest lends.
pub(in super::super) struct NvGuestEncoder<P: Platform> {
    gpu: Arc<Gpu>,
    encoder: NvencEncoder,
    config: EncoderConfig,
    samples: Samples,
    key_frames: KeyFrames,
    /// The index NVENC is given with the next frame.
    next_index: u64,
    /// Whether NVENC took a frame: until then, settings that make a new sequence make a new
    /// encoder.
    encoded: bool,
    /// What writes its source frames into memory NVENC reads, on the GPU (module doc).
    fills: Fills<P>,
}

impl<P: Platform> NvGuestEncoder<P> {
    /// An encoder of `config` on `gpu`, its frames written into memory NVENC reads by `filler`
    /// where it is given (or why not).
    pub(in super::super) fn new(
        gpu: &Arc<Gpu>,
        config: &EncoderConfig,
        filler: Result<Filler<P>, String>,
    ) -> Result<Self, VideoError> {
        let (encoder, record) = Self::made(gpu, config)?;
        let settings = &config.settings;
        Ok(Self {
            gpu: Arc::clone(gpu),
            encoder,
            config: config.clone(),
            samples: Samples {
                record,
                pending: VecDeque::new(),
                restart: false,
            },
            key_frames: KeyFrames::new(
                settings.key_frame_interval,
                settings.key_frame_interval_duration,
            ),
            next_index: 0,
            encoded: false,
            fills: Fills {
                filler,
                said: Vec::new(),
            },
        })
    }

    /// NVENC's encoder of `config` on `gpu`, and the record of the parameter sets it gives.
    fn made(gpu: &Arc<Gpu>, config: &EncoderConfig) -> Result<(NvencEncoder, Record), VideoError> {
        let encoder =
            NvencEncoder::new(Arc::clone(gpu), &params_of(config)).map_err(video_error)?;
        let codec = config.settings.profile.codec();
        let record = Record::of_annex_b(codec, encoder.sequence_parameters()).ok_or_else(|| {
            VideoError::Malfunction("NVENC's parameter sets make no configuration record".into())
        })?;
        Ok((encoder, record))
    }

    /// NVENC's packets of `frame` from `source`, as frame `index`: copied by the GPU where the
    /// source has a buffer and the copy is not declined, else read by the CPU (module doc).
    /// `Err` is a frame NVENC did not take.
    fn submit(
        &mut self,
        source: &GuestSource<'_, P>,
        index: u64,
        force_idr: bool,
    ) -> Result<Vec<Bitstream>, VideoError> {
        let settings = &self.config.settings;
        let planes = source_planes(source, settings)?;
        let size = (settings.width, settings.height);
        if let Some(buffer) = source.buffer {
            let sample = settings.input_format().bytes_per_sample();
            let sample = u32::try_from(sample).unwrap_or(0);
            let [luma, chroma] = buffer_planes(buffer, &planes, size, sample);
            let filled = (self.fills)
                .encode_filled(&mut self.encoder, [&luma, &chroma], size, index, force_idr)
                .map_err(|error| VideoError::Malfunction(error.to_string()))?;
            if let Some(bitstreams) = filled {
                return Ok(bitstreams);
            }
        }
        let view = (source.memory.host_view())
            .map_err(|error| VideoError::Malfunction(error.to_string()))?;
        let [y, uv] = host_planes(&view, &planes)?;
        let raw = RawFrame {
            luma: (y, planes[0].bytes_per_row),
            chroma: (uv, planes[1].bytes_per_row),
            index,
            force_idr,
        };
        let bitstreams = self.encoder.encode(&raw).map_err(video_error)?;
        drop(view);
        Ok(bitstreams)
    }
}

impl<P: Platform> GuestEncoder<P> for NvGuestEncoder<P> {
    fn encode(
        &mut self,
        source: &GuestSource<'_, P>,
        frame: &SourceFrame,
    ) -> Result<Vec<Coded<VideoError>>, VideoError> {
        let index = self.next_index;
        let force = frame.force_key_frame || self.samples.restart;
        let force_idr = self.key_frames.place(frame.pts, force) == 0;
        let bitstreams = match self.submit(source, index, force_idr) {
            Ok(bitstreams) => bitstreams,
            Err(error) => {
                self.samples.restart = true;
                return Err(error);
            }
        };
        self.next_index += 1;
        self.encoded = true;
        if force_idr {
            self.samples.restart = false;
        }
        self.samples.pending.push_back(*frame);
        Ok(self.samples.of(bitstreams))
    }

    fn flush(&mut self) -> Result<Vec<Coded<VideoError>>, VideoError> {
        let bitstreams = self.encoder.flush().map_err(video_error)?;
        Ok(self.samples.of(bitstreams))
    }

    /// The bitrate and the frame rate go to the encoder from the next frame, and the key-frame
    /// interval to the placing of key frames; anything else NVENC is made with, a change of the
    /// rate control's mode among it, makes a new encoder before the first frame, and after it is
    /// taken and left out: the running sequence keeps its own to its end, and its rate control
    /// is `rate_past_first_frame`'s.
    fn reconfigure(&mut self, config: &EncoderConfig) -> Result<(), VideoError> {
        let mut config = config.clone();
        let running = self.config.settings.rate_control;
        if self.encoded {
            config.settings.rate_control =
                rate_past_first_frame(running, config.settings.rate_control);
        }
        let mut new = params_of(&config);
        let old = self.encoder.params();
        let same_sequence = EncodeParams {
            rate_control: old.rate_control,
            frame_rate: old.frame_rate,
            ..new.clone()
        } == *old
            && same_rate_mode(running, config.settings.rate_control);
        if !same_sequence && self.encoded {
            let settings = config.settings;
            config = self.config.clone();
            config.settings.rate_control = settings.rate_control;
            config.settings.frame_rate = settings.frame_rate;
            config.settings.key_frame_interval = settings.key_frame_interval;
            config.settings.key_frame_interval_duration = settings.key_frame_interval_duration;
            new = params_of(&config);
        }
        if !same_sequence && !self.encoded {
            let (encoder, record) = Self::made(&self.gpu, &config)?;
            self.encoder = encoder;
            self.samples.record = record;
        } else if new.rate_control != old.rate_control || new.frame_rate != old.frame_rate {
            (self.encoder)
                .reconfigure(new.rate_control, new.frame_rate)
                .map_err(video_error)?;
        }
        let settings = &config.settings;
        (self.key_frames).set_interval(
            settings.key_frame_interval,
            settings.key_frame_interval_duration,
        );
        self.config = config;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
