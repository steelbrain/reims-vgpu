//! Video encode on Vulkan Video (`VK_KHR_video_encode_queue`,
//! `VK_KHR_video_encode_h264`, `VK_KHR_video_encode_h265`): H.264
//! Constrained Baseline, Main and High from NV12, and H.265 Main from NV12
//! and Main 10 from P010, on the encode queue attached to a
//! [`VideoDevice`] ([`VideoDevice::attach_encode`]).
//!
//! - **Sources** are read where they lie: an image the caller made as
//!   [`SourceSpec`] says, in place, or planes in a buffer, host pages
//!   imported by their pointer among them, through one GPU copy
//!   ([`SourcePicture`]). The CPU copies no frame.
//! - **Bitstreams** go straight into a buffer the caller's
//!   [`BitstreamSource`] gives for each picture, imported host pages among
//!   them ([`ImportedBitstream`](io::ImportedBitstream)); the parameter sets go there too, in front
//!   of each IDR picture, where they are asked for in-band
//!   ([`ParameterSetMode::InBand`]), and are always the encoder's to read
//!   ([`H264Encoder::parameter_sets`], [`H264Encoder::avcc`]).
//! - **Pictures** are planned with no GPU ([`plan`]): IDR, I, P and B
//!   pictures as [`GopConfig`] asks and the device takes, forced IDR
//!   pictures, references by H.264's sliding window. Rate control is a
//!   constant QP, CBR or VBR ([`RateControl`]), within an HRD buffer
//!   ([`Vbv`]), at a quality level of the device's.
//! - **Frames** come out in decode order ([`EncodedFrame`]): each with its
//!   type, order count and `frame_num`, the timeline value after which its
//!   bitstream is there, and what the device says it wrote ([`Feedback`](frame::Feedback)).

pub(crate) mod caps;
pub(crate) mod device;
pub(crate) mod frame;
pub(crate) mod h264;
pub(crate) mod h265;
pub(crate) mod io;
pub(crate) mod plan;
pub(crate) mod rate;
pub(crate) mod session;

use std::sync::Arc;

use ash::vk;
use reims_vgpu_video_codec::encode::gop::DecodeTimes;
use reims_vgpu_video_codec::encode::{ColourDescription, FrameRate};

use self::caps::{EncodeCapabilities, EncodeProfile};
use self::frame::{EncodedFrame, FrameInfo, Output, Queries};
use self::io::{BitstreamSource, BitstreamSpec, EncodeSource, SourcePicture, SourceSpec};
use self::plan::{GopConfig, Limits, PictureKind, Planned, Planner};
use self::rate::{RateControl, RatePlan, Vbv};
use self::session::{CodecRate, EncodeSession, Submitted};
use crate::device::VideoDevice;
use crate::error::Error;
use crate::timeline::Timeline;

/// Where the parameter sets go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParameterSetMode {
    /// In front of each IDR picture's NAL units, in its bitstream buffer,
    /// as an Annex B stream carries them.
    InBand,
    /// Out of the stream: the caller takes them from the encoder, as the
    /// `avcC` or `hvcC` record a container carries.
    #[default]
    OutOfBand,
}

/// What an encoder is made with. [`EncodeConfig::new`] gives the defaults;
/// the encoder checks the rest against the device when it is made.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodeConfig {
    /// The frame's width, even.
    pub width: u32,
    /// The frame's height, even.
    pub height: u32,
    /// The frame rate: the VUI's timing, and what rate control budgets by.
    pub frame_rate: FrameRate,
    /// Where IDR, I and B pictures fall, and how many references P
    /// pictures predict from.
    pub gop: GopConfig,
    /// How the bitrate is controlled.
    pub rate_control: RateControl,
    /// The HRD buffer a bitrate is kept within; none for a second at the
    /// peak, three quarters full at the start.
    pub vbv: Option<Vbv>,
    /// The device's quality level, from 0; none for 0.
    pub quality_level: Option<u32>,
    /// Slices (H.265: slice segments) a picture, each of whole rows of
    /// macroblocks or CTBs: 1 at least.
    pub slices: u32,
    /// Where the parameter sets go.
    pub parameter_sets: ParameterSetMode,
    /// The level as the bitstream numbers it (H.264's `level_idc`, H.265's
    /// `general_level_idc`); none for the lowest that fits.
    pub level: Option<u8>,
    /// Whether samples use the full range, as the VUI says.
    pub full_range: bool,
    /// The colour code points the VUI carries; none for unspecified.
    pub colour: Option<ColourDescription>,
    /// H.264 Main and High: CABAC (the default where the device writes
    /// it) or CAVLC.
    pub cabac: bool,
    /// Whether each picture's work is timed with timestamps
    /// ([`EncodedFrame::gpu_time`]), where the encode family writes them.
    pub timing: bool,
}

impl EncodeConfig {
    /// `width` by `height` at 30 frames a second, an IDR picture every 60
    /// frames, no B picture, one reference, a variable 5 Mbit/s under a
    /// 7.5 Mbit/s peak, quality level 0, one slice, the parameter sets out
    /// of band, CABAC where the profile has it.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frame_rate: FrameRate::new(30, 1),
            gop: GopConfig::default(),
            rate_control: RateControl::VariableBitrate {
                average: 5_000_000,
                peak: 7_500_000,
            },
            vbv: None,
            quality_level: None,
            slices: 1,
            parameter_sets: ParameterSetMode::OutOfBand,
            level: None,
            full_range: false,
            colour: None,
            cabac: true,
            timing: false,
        }
    }
}

/// One frame to encode, in display order.
#[derive(Clone, Debug, Default)]
pub struct FrameParams {
    /// The caller's tag, given back with the frame.
    pub id: u64,
    /// The presentation time stamp, in the caller's ticks, each frame's
    /// later than the last.
    pub pts: i64,
    /// Whether the frame must be an IDR picture.
    pub force_idr: bool,
    /// Semaphores, and their values, the frame's encode waits for: the
    /// work that wrote its source. Each lives until the encode is done.
    pub waits: Vec<(vk::Semaphore, u64)>,
}

/// How an encoder's pictures read their sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceCounts {
    /// Sources read in place: images, no copy.
    pub in_place: u64,
    /// Sources in buffers, each copied once on the GPU into a picture of
    /// the encoder's.
    pub gpu_copies: u64,
    /// Copies of a frame on the CPU: none, ever. Counted, so that a check
    /// says so.
    pub cpu_copies: u64,
}

/// A frame waiting to be encoded.
pub(crate) struct Pending<S> {
    source: S,
    id: u64,
    pts: i64,
    waits: Vec<(vk::Semaphore, u64)>,
}

/// What the codec's half of an encoder makes of each picture.
pub(crate) trait CodecEncoder: Send {
    /// The parameter sets, Annex B, as the device wrote them.
    fn parameter_sets(&self) -> &[u8];
    /// The picture's order count as the encoder names it: H.264's
    /// `PicOrderCnt`, H.265's `PicOrderCntVal`.
    fn order_count(&self, order: u32) -> i32;
    /// Encodes `planned` with its codec's std information.
    fn encode(
        &mut self,
        session: &mut EncodeSession,
        planned: &PictureRefs,
        op: CodecOp<'_>,
    ) -> Result<Submitted, Error>;
}

/// A planned picture's place, without its frame.
pub(crate) struct PictureRefs {
    pub(crate) kind: PictureKind,
    pub(crate) order: u32,
    pub(crate) frame_num: u32,
    pub(crate) idr_pic_id: u16,
    pub(crate) reference: bool,
    pub(crate) setup_slot: u32,
    pub(crate) list0: Vec<plan::Reference>,
    pub(crate) list1: Vec<plan::Reference>,
    pub(crate) dpb: Vec<plan::Reference>,
}

/// What the codec's encode is handed besides its picture.
pub(crate) struct CodecOp<'a> {
    pub(crate) source: SourcePicture,
    pub(crate) target: io::BitstreamBuffer,
    pub(crate) header: Option<&'a [u8]>,
    pub(crate) qp: i32,
    pub(crate) waits: &'a [(vk::Semaphore, u64)],
    pub(crate) query: &'a frame::QuerySlot,
}

/// The encoder both codecs' share: the planner, the session, and the
/// frames in flight.
pub(crate) struct Encoder<S: EncodeSource, B: BitstreamSource> {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    pub(crate) caps: EncodeCapabilities,
    pub(crate) config: EncodeConfig,
    pub(crate) limits: Limits,
    pub(crate) rate: RatePlan,
    pub(crate) codec: Box<dyn CodecEncoder>,
    pub(crate) session: EncodeSession,
    planner: Planner<Pending<S>>,
    times: DecodeTimes,
    target_source: B,
    queries: Arc<Queries>,
    counts: SourceCounts,
    last_pts: Option<i64>,
}

/// An encoder's checked settings, as its codec makes its session and
/// parameter sets from them.
pub(crate) struct Settings {
    pub(crate) caps: EncodeCapabilities,
    pub(crate) config: EncodeConfig,
    pub(crate) limits: Limits,
    pub(crate) rate: RatePlan,
    pub(crate) quality_level: u32,
    /// The coded extent: the frame rounded up to the codec's coding
    /// blocks.
    pub(crate) coded: vk::Extent2D,
}

impl Settings {
    /// `config` for `profile` on `device`, checked: the device encodes
    /// the profile, at the size, with the rate control and quality level
    /// asked; the GOP within its limits; `block` the codec's least coding
    /// block, which the coded size is a whole number of; `log2_max_frame_num`
    /// H.264's.
    pub(crate) fn new(
        device: &VideoDevice,
        profile: EncodeProfile,
        config: &EncodeConfig,
        block: u32,
        log2_max_frame_num: u8,
    ) -> Result<Self, Error> {
        if !device.encode_codecs().contains(profile.operation()) {
            return Err(Error::NoVideoEncode(format!(
                "{}: no encode queue of {:?}",
                device.name(),
                profile.codec
            )));
        }
        let caps = caps::encode_capabilities(device, profile)?.ok_or_else(|| {
            Error::NoVideoEncode(format!(
                "{} encodes no {profile:?} from sources of format {}",
                device.name(),
                profile.format().as_raw()
            ))
        })?;
        if config.width == 0
            || config.height == 0
            || !config.width.is_multiple_of(2)
            || !config.height.is_multiple_of(2)
        {
            return Err(Error::Unsupported(format!(
                "a frame of {}x{}: each side even, above 0",
                config.width, config.height
            )));
        }
        let coded = vk::Extent2D {
            width: config.width.next_multiple_of(block),
            height: config.height.next_multiple_of(block),
        };
        caps.check_extent(coded.width, coded.height)?;
        let quality_level = config.quality_level.unwrap_or(0);
        if quality_level >= caps.max_quality_levels.max(1) {
            return Err(Error::Unsupported(format!(
                "quality level {quality_level}: the device has {}",
                caps.max_quality_levels
            )));
        }
        let rows = coded.height / block;
        if config.slices == 0 || config.slices > caps.max_slices.max(1) || config.slices > rows {
            return Err(Error::Unsupported(format!(
                "{} slices: the device takes {} a picture, of {rows} rows",
                config.slices, caps.max_slices
            )));
        }
        let rate = rate::plan(config.rate_control, config.vbv, config.frame_rate, &caps)?;
        let limits = Limits::new(
            &config.gop,
            caps.encodes_b_frames(),
            caps.max_p_l0,
            caps.max_dpb_slots
                .min(caps.max_active_reference_pictures.saturating_add(1)),
            log2_max_frame_num,
        );
        Ok(Self {
            caps,
            config: config.clone(),
            limits,
            rate,
            quality_level,
            coded,
        })
    }

    /// The codec's part of the rate control state: the GOP, as a regular
    /// one of flat references where it has I pictures at a period, which
    /// forced IDR pictures break only where they fall. Both codecs' flags
    /// have the same bits.
    pub(crate) fn codec_rate(&self) -> CodecRate {
        let gop = &self.config.gop;
        let idr_period = gop.idr_period.unwrap_or(0);
        // Frames from one I or IDR picture to the next: no more than the
        // IDR period.
        let gop_frame_count = match (gop.intra_period, gop.idr_period) {
            (Some(intra), Some(idr)) => intra.min(idr),
            (intra, idr) => intra.or(idr).unwrap_or(0),
        };
        let (flags, b_frames) = if gop_frame_count > 0 {
            (
                vk::VideoEncodeH264RateControlFlagsKHR::REGULAR_GOP
                    | vk::VideoEncodeH264RateControlFlagsKHR::REFERENCE_PATTERN_FLAT,
                self.limits.b_frames.min(gop_frame_count - 1),
            )
        } else {
            // No GOP size: no B picture count either, which must be under
            // it.
            (vk::VideoEncodeH264RateControlFlagsKHR::empty(), 0)
        };
        CodecRate {
            flags: flags.as_raw(),
            gop_frame_count,
            idr_period,
            b_frames,
        }
    }
}

impl<S: EncodeSource, B: BitstreamSource> Encoder<S, B> {
    /// An encoder of `settings`, its session and codec half made by the
    /// codec.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        timeline: Arc<Timeline>,
        settings: Settings,
        session: EncodeSession,
        codec: Box<dyn CodecEncoder>,
        target_source: B,
    ) -> Result<Self, Error> {
        let encode = device.encoding()?;
        let timing = (settings.config.timing && encode.timestamp_bits > 0).then(|| {
            // SAFETY: the physical device is the instance's.
            let limits = unsafe {
                device
                    .instance()
                    .get_physical_device_properties(device.physical_device())
            }
            .limits;
            (limits.timestamp_period, encode.timestamp_bits)
        });
        let wanted = vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
            | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN;
        if !settings.caps.feedback_flags.contains(wanted) {
            return Err(Error::Unsupported(format!(
                "encode feedback {:#x}: the offset and bytes written are needed",
                settings.caps.feedback_flags.as_raw()
            )));
        }
        let queries = Queries::new(device, settings.caps.profile, wanted, timing);
        Ok(Self {
            device: Arc::clone(device),
            planner: Planner::new(&settings.config.gop, settings.limits),
            timeline,
            caps: settings.caps,
            config: settings.config,
            limits: settings.limits,
            rate: settings.rate,
            codec,
            session,
            times: DecodeTimes::new(),
            target_source,
            queries,
            counts: SourceCounts::default(),
            last_pts: None,
        })
    }

    /// How sources read in place are made.
    pub(crate) fn source_spec(&self) -> SourceSpec {
        SourceSpec {
            extent: self.session.image_extent,
            profile: self.caps.profile,
            profile_independent: self.device.encode_profile_independent(),
        }
    }

    /// How bitstream buffers are made.
    pub(crate) fn bitstream_spec(&self) -> BitstreamSpec {
        // A picture's worst: its samples, raw, half again for H.265, and
        // room for headers.
        let samples =
            u64::from(self.session.extent.width) * u64::from(self.session.extent.height) * 3 / 2
                * u64::from(self.caps.profile.sample_bytes());
        let worst = match self.caps.profile.codec {
            caps::EncodeCodec::H264 => samples,
            caps::EncodeCodec::H265 => samples + samples / 2,
        };
        BitstreamSpec {
            size: (worst + (2 << 20))
                .next_multiple_of(self.caps.min_bitstream_buffer_size_alignment.max(1)),
            offset_alignment: self.caps.min_bitstream_buffer_offset_alignment.max(1),
            profile: self.caps.profile,
            profile_independent: self.device.encode_profile_independent(),
        }
    }

    /// The semaphore every picture's work signals.
    pub(crate) fn timeline(&self) -> &Arc<Timeline> {
        &self.timeline
    }

    /// How the sources so far were read.
    pub(crate) fn counts(&self) -> SourceCounts {
        self.counts
    }

    /// Takes `source` as the next frame in display order and returns the
    /// frames encoded now, in decode order: none while it waits as a B
    /// picture. Their encodes may still run: each frame's value says when
    /// its bitstream is there. A picture that fails ends the call with its
    /// error, the frames of its run with it, and the next frame is an IDR
    /// picture.
    pub(crate) fn encode(
        &mut self,
        source: S,
        params: FrameParams,
    ) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        if self.last_pts.is_some_and(|last| params.pts <= last) {
            return Err(Error::Unsupported(format!(
                "a frame at {} after one at {:?}: time stamps increase",
                params.pts, self.last_pts
            )));
        }
        self.last_pts = Some(params.pts);
        self.timeline.collect();
        self.times.push(params.pts);
        let pending = Pending {
            source,
            id: params.id,
            pts: params.pts,
            waits: params.waits,
        };
        let planned = self.planner.push(pending, params.pts, params.force_idr);
        self.code_all(planned)
    }

    /// Encodes the frames still waiting as B pictures, the last as their
    /// anchor, and returns them in decode order.
    pub(crate) fn flush(&mut self) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        let planned = self.planner.flush();
        self.code_all(planned)
    }

    fn code_all(
        &mut self,
        planned: Vec<Planned<Pending<S>>>,
    ) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        let mut out = Vec::with_capacity(planned.len());
        let mut planned = planned.into_iter();
        for picture in planned.by_ref() {
            match self.code(picture) {
                Ok(frame) => out.push(frame),
                Err(error) => {
                    // The frames after it predict from it, or are B
                    // pictures of its run: lost with it.
                    for _ in planned.by_ref() {
                        self.times.next_dts();
                    }
                    self.planner.restart();
                    return Err(error);
                }
            }
        }
        Ok(out)
    }

    fn code(&mut self, planned: Planned<Pending<S>>) -> Result<EncodedFrame<B::Target>, Error> {
        let dts = self.times.next_dts();
        let Planned {
            frame,
            kind,
            order,
            frame_num,
            idr_pic_id,
            reference,
            setup_slot,
            list0,
            list1,
            dpb,
        } = planned;
        let refs = PictureRefs {
            kind,
            order,
            frame_num,
            idr_pic_id,
            reference: reference.is_some(),
            setup_slot,
            list0,
            list1,
            dpb,
        };
        let target = self.target_source.target(&self.bitstream_spec())?;
        let header = (kind == PictureKind::Idr
            && self.config.parameter_sets == ParameterSetMode::InBand)
            .then(|| self.codec.parameter_sets().to_vec());
        let query = self.queries.slot()?;
        let source = frame.source.picture();
        let op = CodecOp {
            source,
            target: io::BitstreamTarget::bitstream(&target),
            header: header.as_deref(),
            qp: self.rate.qp(kind),
            waits: &frame.waits,
            query: &query,
        };
        let submitted = self.codec.encode(&mut self.session, &refs, op)?;
        if submitted.copied {
            self.counts.gpu_copies = self.counts.gpu_copies.saturating_add(1);
        } else {
            self.counts.in_place = self.counts.in_place.saturating_add(1);
        }
        // The source is the caller's again once the encode that reads it
        // is done.
        self.timeline
            .retire(submitted.value, Box::new(RetiredSource(frame.source)));
        let output = Output::new(
            &self.timeline,
            target,
            query,
            submitted.value,
            submitted.encode_offset,
            submitted.header,
        );
        Ok(EncodedFrame::new(
            FrameInfo {
                id: frame.id,
                pts: frame.pts,
                dts,
                kind,
                order_count: self.codec.order_count(order),
                frame_num,
                reference: reference.is_some(),
            },
            output,
        ))
    }

    /// Asks for `rate_control` from the next picture on, within `vbv`.
    pub(crate) fn set_rate_control(
        &mut self,
        rate_control: RateControl,
        vbv: Option<Vbv>,
    ) -> Result<(), Error> {
        let rate = rate::plan(rate_control, vbv, self.config.frame_rate, &self.caps)?;
        if rate.constant_qp.is_some() != self.rate.constant_qp.is_some() {
            return Err(Error::Unsupported(
                "a switch between a constant QP and a bitrate mid-stream".into(),
            ));
        }
        self.rate = rate;
        self.config.rate_control = rate_control;
        self.config.vbv = vbv;
        let codec = self.session.rate().1;
        self.session.set_rate((rate, codec));
        Ok(())
    }
}

/// A source held until the encode that reads it is done.
struct RetiredSource<S>(S);

pub use self::h264::H264Encoder;
pub use self::h265::H265Encoder;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::rate::tests::caps;

    fn settings(gop: GopConfig) -> Settings {
        let caps = caps();
        let mut config = EncodeConfig::new(1280, 720);
        config.gop = gop;
        let rate = rate::plan(config.rate_control, None, config.frame_rate, &caps).unwrap();
        Settings {
            limits: Limits::new(&gop, true, caps.max_p_l0, 17, 8),
            caps,
            config,
            rate,
            quality_level: 0,
            coded: vk::Extent2D {
                width: 1280,
                height: 720,
            },
        }
    }

    const REGULAR: u32 = 0x2 | 0x4;

    #[test]
    fn a_gop_with_a_period_is_regular_and_flat() {
        let rate = settings(GopConfig {
            idr_period: Some(120),
            intra_period: Some(30),
            b_frames: 3,
            references: 2,
        })
        .codec_rate();
        assert_eq!(
            rate,
            CodecRate {
                flags: REGULAR,
                gop_frame_count: 30,
                idr_period: 120,
                b_frames: 3,
            }
        );
    }

    #[test]
    fn the_gop_is_no_longer_than_the_idr_period_and_holds_its_b_pictures() {
        // An intra period past the IDR period is the IDR period, and the B
        // pictures a GOP is told of are fewer than its frames.
        let rate = settings(GopConfig {
            idr_period: Some(3),
            intra_period: Some(30),
            b_frames: 5,
            references: 1,
        })
        .codec_rate();
        assert_eq!(
            (rate.gop_frame_count, rate.idr_period, rate.b_frames),
            (3, 3, 2)
        );
        assert_eq!(rate.flags, REGULAR);
    }

    #[test]
    fn a_gop_without_a_period_says_nothing_of_its_shape() {
        let rate = settings(GopConfig {
            idr_period: None,
            intra_period: None,
            b_frames: 2,
            references: 1,
        })
        .codec_rate();
        assert_eq!(
            rate,
            CodecRate {
                flags: 0,
                gop_frame_count: 0,
                idr_period: 0,
                b_frames: 0,
            }
        );
    }
}
