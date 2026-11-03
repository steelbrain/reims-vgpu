//! H.264 encoding on Vulkan Video: the parameter sets as std structures,
//! which the device writes as NAL units
//! (`vkGetEncodedVideoSessionParametersKHR`), and each planned picture's
//! std information: its type, numbers and order count, its reference lists
//! in DPB slots, its slices' headers, and its references'.
//!
//! The stream's shape: frames only, picture order count type 0 (twice the
//! place in display order, the LSBs wrapping at 256), `frame_num` wrapping
//! at 256, references by the sliding window, lists in their default order
//! (no modification), B pictures' motion predicted spatially in direct
//! mode, the 8x8 transform in High, CABAC in Main and High where the device
//! writes it. The VUI carries the frame rate, the colour, and how many
//! frames a decoder holds back and keeps.

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH264PictureInfo, StdVideoEncodeH264PictureInfoFlags,
    StdVideoEncodeH264ReferenceInfo, StdVideoEncodeH264ReferenceInfoFlags,
    StdVideoEncodeH264ReferenceListsInfo, StdVideoEncodeH264ReferenceListsInfoFlags,
    StdVideoEncodeH264SliceHeader, StdVideoEncodeH264SliceHeaderFlags,
    StdVideoH264PictureParameterSet, StdVideoH264PictureType, StdVideoH264PpsFlags,
    StdVideoH264SequenceParameterSet, StdVideoH264SequenceParameterSetVui, StdVideoH264SliceType,
    StdVideoH264SpsFlags, StdVideoH264SpsVuiFlags,
};
use reims_vgpu_video_codec::encode::h264::{LevelNeeds, avcc, choose_level, raise_mv_lengths};
use reims_vgpu_video_codec::encode::{Profile, nal};

use super::caps::{EncodeCapabilities, EncodeProfile};
use super::frame::EncodedFrame;
use super::io::{BitstreamSource, BitstreamSpec, EncodeSource, SourceSpec};
use super::plan::{PictureKind, Reference};
use super::rate::{RateControl, Vbv};
use super::session::{
    CodecInfo, CodecSession, EncodeOp, EncodeSession, Submitted, encoded_parameters,
};
use super::{
    CodecEncoder, CodecOp, EncodeConfig, Encoder, FrameParams, PictureRefs, Settings, SourceCounts,
};
use crate::device::VideoDevice;
use crate::error::Error;
use crate::params::Parameters;
use crate::profile::std_level;
use crate::std_types::zeroed;
use crate::timeline::Timeline;

/// `log2_max_frame_num_minus4`: `frame_num` wraps at 256.
const LOG2_MAX_FRAME_NUM_MINUS4: u8 = 4;
/// `log2_max_pic_order_cnt_lsb_minus4`: the order count's LSBs wrap at
/// 256, 128 frames, far past any distance between a picture and its
/// references.
const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u8 = 4;

/// `VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR`.
const STD_TRANSFORM_8X8: u32 = 0x200;
/// `VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR`.
const STD_CABAC: u32 = 0x1000;

/// An H.264 encoder on a [`VideoDevice`]'s encode queue, reading frames
/// from sources `S` and writing each picture into a buffer `B` gives.
///
/// Frames go in in display order ([`Self::encode`]); encoded frames come
/// out in decode order, as soon as each can be encoded: at once without B
/// pictures, else once the anchor after a run of them comes. Each encode is
/// submitted and not waited for: a frame names the timeline value after
/// which its bitstream is there.
pub struct H264Encoder<S: EncodeSource, B: BitstreamSource> {
    inner: Encoder<S, B>,
    level_idc: u8,
}

impl<S: EncodeSource, B: BitstreamSource> H264Encoder<S, B> {
    /// An encoder of `profile` (an H.264 one: [`EncodeProfile::H264_HIGH`]
    /// and the like, with a tuning) for `config`, its bitstreams in the
    /// buffers `targets` gives, checked against what the device takes: the
    /// profile, the size, the level, the rate control, the quality level and
    /// the slices. The GOP is kept within the device's limits: no B picture
    /// where it or the profile has none, and as many references as it takes.
    pub fn new(
        device: &Arc<VideoDevice>,
        profile: EncodeProfile,
        config: &EncodeConfig,
        targets: B,
    ) -> Result<Self, Error> {
        if profile.codec != super::caps::EncodeCodec::H264 {
            return Err(Error::Unsupported(format!(
                "{profile:?} in an H.264 encoder"
            )));
        }
        let settings = Settings::new(device, profile, config, 16, LOG2_MAX_FRAME_NUM_MINUS4 + 4)?;
        let level_idc = level(&settings)?;
        let timeline = Timeline::new(device)?;
        let mut session = EncodeSession::new(
            device,
            &timeline,
            &settings,
            CodecSession::H264 {
                max_level: std_level(level_idc),
            },
        )?;
        let codec = H264Codec::new(device, &settings, level_idc, &mut session)?;
        Ok(Self {
            inner: Encoder::new(
                device,
                timeline,
                settings,
                session,
                Box::new(codec),
                targets,
            )?,
            level_idc,
        })
    }

    /// How a source image it reads in place is made.
    pub fn source_spec(&self) -> SourceSpec {
        self.inner.source_spec()
    }

    /// How its bitstream buffers are made: what it asks its
    /// [`BitstreamSource`] for.
    pub fn bitstream_spec(&self) -> BitstreamSpec {
        self.inner.bitstream_spec()
    }

    /// Takes `source` as the next frame in display order, and returns the
    /// frames encoded now, in decode order. A picture that fails ends the
    /// call with its error, and the frames planned with it are lost; the
    /// next frame is an IDR picture.
    pub fn encode(
        &mut self,
        source: S,
        params: FrameParams,
    ) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        self.inner.encode(source, params)
    }

    /// Encodes the frames waiting as B pictures, the last as their
    /// anchor, and returns them in decode order.
    pub fn flush(&mut self) -> Result<Vec<EncodedFrame<B::Target>>, Error> {
        self.inner.flush()
    }

    /// Asks for `rate_control` within `vbv` from the next picture on: a
    /// bitrate for a bitrate, a QP for a QP.
    pub fn set_rate_control(
        &mut self,
        rate_control: RateControl,
        vbv: Option<Vbv>,
    ) -> Result<(), Error> {
        self.inner.set_rate_control(rate_control, vbv)
    }

    /// The SPS and PPS, Annex B, as the device wrote them: what goes in
    /// front of each IDR picture in band.
    pub fn parameter_sets(&self) -> &[u8] {
        self.inner.codec.parameter_sets()
    }

    /// The `avcC` record of the parameter sets, with four-byte NAL
    /// lengths.
    pub fn avcc(&self) -> Vec<u8> {
        let units = nal::split_annex_b(self.parameter_sets());
        let find = |kind: u8| {
            units
                .iter()
                .find(|u| u.first().map(|b| b & 0x1f) == Some(kind))
        };
        match (find(7), find(8)) {
            (Some(sps), Some(pps)) => avcc(sps, pps),
            _ => Vec::new(),
        }
    }

    /// What the device answers for its profile.
    pub fn capabilities(&self) -> &EncodeCapabilities {
        &self.inner.caps
    }

    /// The level, as the SPS numbers it.
    pub fn level_idc(&self) -> u8 {
        self.level_idc
    }

    /// The coded size: the frame rounded up to whole macroblocks.
    pub fn coded_extent(&self) -> vk::Extent2D {
        self.inner.session.extent
    }

    /// Consecutive B pictures the stream has.
    pub fn b_frames(&self) -> u32 {
        self.inner.limits.b_frames
    }

    /// References a P picture predicts from.
    pub fn references(&self) -> u32 {
        self.inner.limits.p_l0
    }

    /// The timeline semaphore each picture's work signals.
    pub fn semaphore(&self) -> vk::Semaphore {
        self.inner.timeline().semaphore()
    }

    /// The value [`Self::semaphore`] reaches once every encode submitted
    /// so far is done.
    pub fn submitted(&self) -> u64 {
        self.inner.timeline().last_submitted()
    }

    /// How the sources so far were read: in place, or copied on the GPU.
    pub fn source_counts(&self) -> SourceCounts {
        self.inner.counts()
    }
}

/// The level: the one asked for, else the lowest that fits the size, the
/// frame rate, the peak and the DPB; refused past the device's.
fn level(settings: &Settings) -> Result<u8, Error> {
    let profile = match settings.caps.profile.profile_idc {
        66 => Profile::H264ConstrainedBaseline,
        77 => Profile::H264Main,
        _ => Profile::H264High,
    };
    let needs = LevelNeeds {
        profile,
        width_in_mbs: settings.coded.width / 16,
        height_in_mbs: settings.coded.height / 16,
        frames_per_second: settings.rate.frame_rate.as_f64(),
        bits_per_second: settings
            .rate
            .peak_bits_per_second()
            .map(|peak| u32::try_from(peak).unwrap_or(u32::MAX)),
        dpb_frames: settings.limits.max_references,
    };
    let level = match settings.config.level {
        Some(level) => level,
        None => choose_level(&needs).map_err(|e| Error::Unsupported(e.0))?,
    };
    if level > settings.caps.max_level_idc {
        return Err(Error::Unsupported(format!(
            "H.264 level {level}: the device encodes up to {}",
            settings.caps.max_level_idc
        )));
    }
    Ok(level)
}

/// The SPS and its VUI, which it points at.
struct StdSps {
    sps: StdVideoH264SequenceParameterSet,
    /// What `sps` points at.
    _vui: Box<StdVideoH264SequenceParameterSetVui>,
}

/// The parameter sets of a stream of `settings` at `level_idc`, as std
/// structures.
fn parameter_sets(settings: &Settings, level_idc: u8) -> (StdSps, StdVideoH264PictureParameterSet) {
    let profile = settings.caps.profile;
    let config = &settings.config;
    let mut flags: StdVideoH264SpsFlags = zeroed();
    if profile.profile_idc == 66 {
        // Constrained Baseline: decodable as Baseline and as Main.
        flags.set_constraint_set0_flag(1);
        flags.set_constraint_set1_flag(1);
    }
    flags.set_direct_8x8_inference_flag(1);
    flags.set_frame_mbs_only_flag(1);
    let crop_right = (settings.coded.width - config.width) / 2;
    let crop_bottom = (settings.coded.height - config.height) / 2;
    flags.set_frame_cropping_flag(u32::from(crop_right > 0 || crop_bottom > 0));
    flags.set_vui_parameters_present_flag(1);

    let mut vui_flags: StdVideoH264SpsVuiFlags = zeroed();
    vui_flags.set_video_signal_type_present_flag(u32::from(
        config.full_range || config.colour.is_some(),
    ));
    vui_flags.set_video_full_range_flag(u32::from(config.full_range));
    vui_flags.set_color_description_present_flag(u32::from(config.colour.is_some()));
    vui_flags.set_timing_info_present_flag(1);
    vui_flags.set_fixed_frame_rate_flag(1);
    vui_flags.set_bitstream_restriction_flag(1);
    let mut vui: StdVideoH264SequenceParameterSetVui = zeroed();
    vui.flags = vui_flags;
    // Unspecified, where no colour is given.
    vui.video_format = 5;
    let colour = config
        .colour
        .map_or((2, 2, 2), |c| (c.primaries, c.transfer, c.matrix));
    (
        vui.colour_primaries,
        vui.transfer_characteristics,
        vui.matrix_coefficients,
    ) = colour;
    // Two ticks a frame (E.2.1).
    let rate = settings.rate.frame_rate;
    vui.num_units_in_tick = rate.denominator;
    vui.time_scale = rate.numerator.saturating_mul(2);
    let small = |n: u32| u8::try_from(n).unwrap_or(u8::MAX);
    // B pictures are never references, so only their anchor is decoded
    // before frames it follows in display order.
    vui.max_num_reorder_frames = u8::from(settings.limits.b_frames > 0);
    vui.max_dec_frame_buffering = small(settings.limits.max_references);
    let vui = Box::new(vui);

    let mut sps: StdVideoH264SequenceParameterSet = zeroed();
    sps.flags = flags;
    sps.profile_idc = u32::from(profile.profile_idc);
    sps.level_idc = std_level(level_idc);
    sps.chroma_format_idc = 1;
    sps.seq_parameter_set_id = 0;
    sps.log2_max_frame_num_minus4 = LOG2_MAX_FRAME_NUM_MINUS4;
    sps.pic_order_cnt_type = 0;
    sps.log2_max_pic_order_cnt_lsb_minus4 = LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4;
    sps.max_num_ref_frames = small(settings.limits.max_references);
    sps.pic_width_in_mbs_minus1 = settings.coded.width / 16 - 1;
    sps.pic_height_in_map_units_minus1 = settings.coded.height / 16 - 1;
    sps.frame_crop_right_offset = crop_right;
    sps.frame_crop_bottom_offset = crop_bottom;
    sps.pSequenceParameterSetVui = std::ptr::from_ref(vui.as_ref());

    let mut pps_flags: StdVideoH264PpsFlags = zeroed();
    let high = profile.profile_idc == 100;
    pps_flags
        .set_transform_8x8_mode_flag(u32::from(high && settings.caps.writes(STD_TRANSFORM_8X8)));
    pps_flags.set_deblocking_filter_control_present_flag(1);
    pps_flags.set_entropy_coding_mode_flag(u32::from(cabac(settings)));
    let mut pps: StdVideoH264PictureParameterSet = zeroed();
    pps.flags = pps_flags;
    pps.seq_parameter_set_id = 0;
    pps.pic_parameter_set_id = 0;
    pps.num_ref_idx_l0_default_active_minus1 = 0;
    pps.num_ref_idx_l1_default_active_minus1 = 0;
    (StdSps { sps, _vui: vui }, pps)
}

/// Whether the stream's entropy coder is CABAC: in Main and High, where
/// asked and the device writes it.
fn cabac(settings: &Settings) -> bool {
    settings.caps.profile.profile_idc != 66
        && settings.config.cabac
        && settings.caps.writes(STD_CABAC)
}

/// The H.264 half of an encoder: its parameter sets, as the device wrote
/// them, and what each picture's std information is made from.
struct H264Codec {
    parameter_sets: Vec<u8>,
    width_in_mbs: u32,
    height_in_mbs: u32,
    slices: u32,
}

impl H264Codec {
    /// The parameter sets of `settings` at `level_idc` in a parameters
    /// object of `session`'s, which it takes, and their NAL units.
    fn new(
        device: &Arc<VideoDevice>,
        settings: &Settings,
        level_idc: u8,
        session: &mut EncodeSession,
    ) -> Result<Self, Error> {
        let (sps, pps) = parameter_sets(settings, level_idc);
        let sps_array = [sps.sps];
        let pps_array = [pps];
        let add = vk::VideoEncodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let mut h264 = vk::VideoEncodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add);
        let mut quality =
            vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(session.quality_level());
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session.handle())
            .push_next(&mut h264)
            .push_next(&mut quality);
        let parameters = Parameters::new(device, &info)?;
        let parameter_sets = with_mv_lengths(encoded_parameter_sets(device, parameters.handle())?);
        drop(sps);
        session.set_parameters(parameters);
        Ok(Self {
            parameter_sets,
            width_in_mbs: settings.coded.width / 16,
            height_in_mbs: settings.coded.height / 16,
            slices: settings.config.slices,
        })
    }
}

/// The SPS and PPS of `parameters`, Annex B, as the device writes them.
fn encoded_parameter_sets(
    device: &VideoDevice,
    parameters: vk::VideoSessionParametersKHR,
) -> Result<Vec<u8>, Error> {
    let mut h264 = vk::VideoEncodeH264SessionParametersGetInfoKHR::default()
        .write_std_sps(true)
        .write_std_pps(true)
        .std_sps_id(0)
        .std_pps_id(0);
    let get = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(parameters)
        .push_next(&mut h264);
    let mut h264_feedback = vk::VideoEncodeH264SessionParametersFeedbackInfoKHR::default();
    let mut feedback =
        vk::VideoEncodeSessionParametersFeedbackInfoKHR::default().push_next(&mut h264_feedback);
    let data = encoded_parameters(device, &get, &mut feedback)?;
    if feedback.has_overrides == vk::TRUE {
        log::info!(
            "the device overrides the H.264 parameter sets (SPS {}, PPS {})",
            h264_feedback.has_std_sps_overrides,
            h264_feedback.has_std_pps_overrides
        );
    }
    Ok(data)
}

/// The bound on motion vectors the SPS's VUI gives
/// (`log2_max_mv_length_horizontal` and `_vertical`), at least: what x264
/// writes, and what no motion vector passes.
const LOG2_MAX_MV_LENGTH: u32 = 15;

/// `stream`'s parameter sets, Annex B, with the SPS's motion vector bounds
/// raised to [`LOG2_MAX_MV_LENGTH`] where the device wrote them lower: the
/// std SPS has no field for them, and NVIDIA's 615.71.09 writes 1, which
/// its own motion vectors pass ([`raise_mv_lengths`]). Every other bit is
/// the device's.
fn with_mv_lengths(stream: Vec<u8>) -> Vec<u8> {
    let units = nal::split_annex_b(&stream);
    let raised: Vec<Option<Vec<u8>>> = units
        .iter()
        .map(|unit| {
            (unit.first().map(|b| b & 0x1f) == Some(7))
                .then(|| raise_mv_lengths(unit, LOG2_MAX_MV_LENGTH))
                .flatten()
        })
        .collect();
    if raised.iter().all(Option::is_none) {
        return stream;
    }
    log::debug!("the device's SPS bounds motion vectors below 2^{LOG2_MAX_MV_LENGTH}: raised");
    let mut out = Vec::with_capacity(stream.len() + 4);
    for (unit, raised) in units.iter().zip(&raised) {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(raised.as_deref().unwrap_or(unit));
    }
    out
}

/// A picture's std information, with what it points at.
struct PictureStd {
    picture: StdVideoEncodeH264PictureInfo,
    /// What `picture` points at.
    _lists: Box<StdVideoEncodeH264ReferenceListsInfo>,
    slices: Vec<StdVideoEncodeH264SliceHeader>,
    setup: StdVideoEncodeH264ReferenceInfo,
    references: Vec<StdVideoEncodeH264ReferenceInfo>,
    reference_slots: Vec<i32>,
}

/// `STD_VIDEO_H264_NO_REFERENCE_PICTURE`.
const NO_REFERENCE: u8 = 0xff;

fn picture_type(kind: PictureKind) -> StdVideoH264PictureType {
    match kind {
        PictureKind::P => 0,
        PictureKind::B => 1,
        PictureKind::I => 2,
        PictureKind::Idr => 5,
    }
}

fn slice_type(kind: PictureKind) -> StdVideoH264SliceType {
    match kind {
        PictureKind::P => 0,
        PictureKind::B => 1,
        PictureKind::I | PictureKind::Idr => 2,
    }
}

/// A reference's std information.
fn reference_info(reference: &Reference) -> StdVideoEncodeH264ReferenceInfo {
    let mut info: StdVideoEncodeH264ReferenceInfo = zeroed();
    info.flags = zeroed::<StdVideoEncodeH264ReferenceInfoFlags>();
    info.primary_pic_type = picture_type(reference.kind);
    info.FrameNum = reference.frame_num;
    info.PicOrderCnt = order_count(reference.order);
    info
}

/// H.264's order count of a frame at `order`: two a frame.
fn order_count(order: u32) -> i32 {
    i32::try_from(order.saturating_mul(2)).unwrap_or(i32::MAX)
}

impl H264Codec {
    /// `picture`'s std information.
    fn picture(&self, picture: &PictureRefs) -> PictureStd {
        let slot = |r: &Reference| u8::try_from(r.slot).unwrap_or(NO_REFERENCE);
        let mut lists: StdVideoEncodeH264ReferenceListsInfo = zeroed();
        lists.flags = zeroed::<StdVideoEncodeH264ReferenceListsInfoFlags>();
        lists.RefPicList0 = [NO_REFERENCE; 32];
        lists.RefPicList1 = [NO_REFERENCE; 32];
        for (at, reference) in lists.RefPicList0.iter_mut().zip(&picture.list0) {
            *at = slot(reference);
        }
        for (at, reference) in lists.RefPicList1.iter_mut().zip(&picture.list1) {
            *at = slot(reference);
        }
        let active = |list: &[Reference]| u8::try_from(list.len().saturating_sub(1)).unwrap_or(0);
        lists.num_ref_idx_l0_active_minus1 = active(&picture.list0);
        lists.num_ref_idx_l1_active_minus1 = active(&picture.list1);
        let lists = Box::new(lists);

        // The references the picture names, each once, in list order.
        let mut named: Vec<&Reference> = Vec::new();
        for reference in picture.list0.iter().chain(&picture.list1) {
            if !named.iter().any(|r| r.slot == reference.slot) {
                named.push(reference);
            }
        }

        let mut flags: StdVideoEncodeH264PictureInfoFlags = zeroed();
        flags.set_IdrPicFlag(u32::from(picture.kind == PictureKind::Idr));
        flags.set_is_reference(u32::from(picture.reference));
        let mut info: StdVideoEncodeH264PictureInfo = zeroed();
        info.flags = flags;
        info.seq_parameter_set_id = 0;
        info.pic_parameter_set_id = 0;
        info.idr_pic_id = picture.idr_pic_id;
        info.primary_pic_type = picture_type(picture.kind);
        info.frame_num = picture.frame_num;
        info.PicOrderCnt = order_count(picture.order);
        info.pRefLists = std::ptr::from_ref(lists.as_ref());

        let mut setup: StdVideoEncodeH264ReferenceInfo = zeroed();
        setup.primary_pic_type = info.primary_pic_type;
        setup.FrameNum = picture.frame_num;
        setup.PicOrderCnt = info.PicOrderCnt;

        // The PPS's defaults are one reference a list.
        let overrides =
            picture.list0.len() > 1 || (picture.kind == PictureKind::B && picture.list1.len() > 1);
        let slices = self
            .slice_starts()
            .into_iter()
            .map(|first_mb| {
                let mut flags: StdVideoEncodeH264SliceHeaderFlags = zeroed();
                flags.set_direct_spatial_mv_pred_flag(1);
                flags.set_num_ref_idx_active_override_flag(u32::from(overrides));
                let mut header: StdVideoEncodeH264SliceHeader = zeroed();
                header.flags = flags;
                header.first_mb_in_slice = first_mb;
                header.slice_type = slice_type(picture.kind);
                header
            })
            .collect();
        PictureStd {
            picture: info,
            _lists: lists,
            slices,
            setup,
            references: named.iter().map(|r| reference_info(r)).collect(),
            reference_slots: named
                .iter()
                .map(|r| i32::try_from(r.slot).unwrap_or(-1))
                .collect(),
        }
    }

    /// Each slice's first macroblock: the picture's rows shared out evenly.
    fn slice_starts(&self) -> Vec<u32> {
        let slices = self.slices.clamp(1, self.height_in_mbs.max(1));
        (0..slices)
            .map(|i| i * self.height_in_mbs / slices * self.width_in_mbs)
            .collect()
    }
}

impl CodecEncoder for H264Codec {
    fn parameter_sets(&self) -> &[u8] {
        &self.parameter_sets
    }

    fn order_count(&self, order: u32) -> i32 {
        order_count(order)
    }

    fn encode(
        &mut self,
        session: &mut EncodeSession,
        picture: &PictureRefs,
        op: CodecOp<'_>,
    ) -> Result<Submitted, Error> {
        let std = self.picture(picture);
        session.encode(&EncodeOp {
            source: op.source,
            target: op.target,
            header: op.header,
            setup_slot: i32::try_from(picture.setup_slot).unwrap_or(-1),
            reference_slots: &std.reference_slots,
            codec: CodecInfo::H264 {
                picture: &std.picture,
                slices: &std.slices,
                setup: &std.setup,
                references: &std.references,
            },
            qp: op.qp,
            waits: op.waits,
            query: op.query,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::plan::{GopConfig, Limits};
    use crate::encode::rate::tests::caps;
    use reims_vgpu_video_codec::encode::FrameRate;

    fn settings(width: u32, height: u32, profile: EncodeProfile) -> Settings {
        let mut caps = caps();
        caps.profile = profile;
        let mut config = EncodeConfig::new(width, height);
        config.gop = GopConfig {
            idr_period: Some(60),
            intra_period: None,
            b_frames: 2,
            references: 3,
        };
        let rate =
            crate::encode::rate::plan(config.rate_control, None, FrameRate::new(30, 1), &caps)
                .unwrap();
        let limits = Limits::new(&config.gop, profile.has_b_slices(), 7, 17, 8);
        Settings {
            coded: vk::Extent2D {
                width: width.next_multiple_of(16),
                height: height.next_multiple_of(16),
            },
            caps,
            config,
            limits,
            rate,
            quality_level: 0,
        }
    }

    #[test]
    fn the_sps_crops_the_coded_size_and_carries_the_vui() {
        let settings = settings(1920, 1080, EncodeProfile::H264_HIGH);
        let (sps, pps) = parameter_sets(&settings, 41);
        // SAFETY: the VUI `sps` holds.
        let vui = unsafe { &*sps.sps.pSequenceParameterSetVui };
        assert_eq!(sps.sps.profile_idc, 100);
        assert_eq!(crate::profile::level_idc(sps.sps.level_idc), 41);
        assert_eq!(sps.sps.pic_width_in_mbs_minus1, 119);
        assert_eq!(sps.sps.pic_height_in_map_units_minus1, 67);
        assert_eq!(sps.sps.flags.frame_cropping_flag(), 1);
        assert_eq!(sps.sps.frame_crop_bottom_offset, 4);
        assert_eq!(sps.sps.frame_crop_right_offset, 0);
        assert_eq!(sps.sps.max_num_ref_frames, 3);
        assert_eq!(sps.sps.flags.constraint_set0_flag(), 0);
        assert_eq!(vui.time_scale, 60);
        assert_eq!(vui.num_units_in_tick, 1);
        assert_eq!(vui.max_num_reorder_frames, 1);
        assert_eq!(vui.max_dec_frame_buffering, 3);
        assert_eq!(vui.flags.video_signal_type_present_flag(), 0);
        assert!(std::ptr::eq(
            sps.sps.pSequenceParameterSetVui,
            sps._vui.as_ref()
        ));
        assert_eq!(pps.flags.transform_8x8_mode_flag(), 1);
        assert_eq!(pps.flags.entropy_coding_mode_flag(), 1);
    }

    #[test]
    fn constrained_baseline_is_cavlc_without_b_pictures() {
        let settings = settings(1280, 720, EncodeProfile::H264_BASELINE);
        let (sps, pps) = parameter_sets(&settings, 31);
        // SAFETY: the VUI `sps` holds.
        let vui = unsafe { &*sps.sps.pSequenceParameterSetVui };
        assert_eq!(sps.sps.profile_idc, 66);
        assert_eq!(sps.sps.flags.constraint_set0_flag(), 1);
        assert_eq!(sps.sps.flags.constraint_set1_flag(), 1);
        assert_eq!(sps.sps.flags.frame_cropping_flag(), 0);
        assert_eq!(vui.max_num_reorder_frames, 0);
        assert_eq!(pps.flags.entropy_coding_mode_flag(), 0);
        assert_eq!(pps.flags.transform_8x8_mode_flag(), 0);
    }

    fn reference(order: u32, slot: u32, frame_num: u32, kind: PictureKind) -> Reference {
        Reference {
            number: u64::from(order),
            slot,
            order,
            frame_num,
            kind,
        }
    }

    fn lists(std: &PictureStd) -> &StdVideoEncodeH264ReferenceListsInfo {
        assert!(std::ptr::eq(std.picture.pRefLists, std._lists.as_ref()));
        &std._lists
    }

    fn codec(slices: u32) -> H264Codec {
        H264Codec {
            parameter_sets: Vec::new(),
            width_in_mbs: 80,
            height_in_mbs: 45,
            slices,
        }
    }

    #[test]
    fn a_p_picture_names_its_references_by_slot_in_list_order() {
        let list0 = vec![
            reference(6, 2, 3, PictureKind::P),
            reference(3, 1, 2, PictureKind::P),
            reference(0, 0, 0, PictureKind::Idr),
        ];
        let picture = PictureRefs {
            kind: PictureKind::P,
            order: 9,
            frame_num: 4,
            idr_pic_id: 3,
            reference: true,
            setup_slot: 3,
            list0: list0.clone(),
            list1: Vec::new(),
            dpb: list0,
        };
        let std = codec(1).picture(&picture);
        assert_eq!(lists(&std).num_ref_idx_l0_active_minus1, 2);
        assert_eq!(&lists(&std).RefPicList0[..4], [2, 1, 0, NO_REFERENCE]);
        assert!(lists(&std).RefPicList1.iter().all(|&s| s == NO_REFERENCE));
        assert_eq!(std.reference_slots, [2, 1, 0]);
        assert_eq!(std.references[0].PicOrderCnt, 12);
        assert_eq!(std.references[2].primary_pic_type, 5);
        assert_eq!(std.picture.frame_num, 4);
        assert_eq!(std.picture.PicOrderCnt, 18);
        assert_eq!(std.picture.flags.IdrPicFlag(), 0);
        assert_eq!(std.picture.flags.is_reference(), 1);
        assert_eq!(std.picture.primary_pic_type, 0);
        assert_eq!(std.setup.FrameNum, 4);
        // Three references against the PPS's one.
        assert_eq!(std.slices[0].flags.num_ref_idx_active_override_flag(), 1);
        assert_eq!(std.slices[0].slice_type, 0);
    }

    #[test]
    fn a_b_picture_names_each_anchor_once() {
        let before = reference(3, 1, 1, PictureKind::P);
        let after = reference(6, 0, 2, PictureKind::P);
        let picture = PictureRefs {
            kind: PictureKind::B,
            order: 4,
            frame_num: 3,
            idr_pic_id: 0,
            reference: false,
            setup_slot: 2,
            list0: vec![before],
            list1: vec![after],
            dpb: vec![before, after],
        };
        let std = codec(1).picture(&picture);
        assert_eq!(lists(&std).RefPicList0[0], 1);
        assert_eq!(lists(&std).RefPicList1[0], 0);
        assert_eq!(
            (
                lists(&std).num_ref_idx_l0_active_minus1,
                lists(&std).num_ref_idx_l1_active_minus1
            ),
            (0, 0)
        );
        assert_eq!(std.reference_slots, [1, 0]);
        assert_eq!(std.picture.flags.is_reference(), 0);
        assert_eq!(std.picture.primary_pic_type, 1);
        assert_eq!(std.slices[0].slice_type, 1);
        assert_eq!(std.slices[0].flags.direct_spatial_mv_pred_flag(), 1);
        assert_eq!(std.slices[0].flags.num_ref_idx_active_override_flag(), 0);
    }

    #[test]
    fn an_idr_picture_names_no_reference() {
        let picture = PictureRefs {
            kind: PictureKind::Idr,
            order: 0,
            frame_num: 0,
            idr_pic_id: 7,
            reference: true,
            setup_slot: 0,
            list0: Vec::new(),
            list1: Vec::new(),
            dpb: Vec::new(),
        };
        let std = codec(1).picture(&picture);
        assert!(std.reference_slots.is_empty());
        assert_eq!(std.picture.flags.IdrPicFlag(), 1);
        assert_eq!(std.picture.idr_pic_id, 7);
        assert_eq!(std.picture.primary_pic_type, 5);
        assert_eq!(std.slices[0].slice_type, 2);
    }

    #[test]
    fn the_devices_mv_bounds_are_raised_in_its_parameter_sets() {
        // The SPS and PPS NVIDIA's 615.71.09 wrote for a 4K High stream
        // (11-03): its SPS bounds motion vectors at 2^1 quarter samples.
        let hex = |s: &str| -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        };
        let sps = hex("67640033ac2cac03c0043e84000003000400000300f23d24e0");
        let pps = hex("68ee3cb0");
        let device = [&[0, 0, 0, 1][..], &sps, &[0, 0, 0, 1], &pps].concat();
        let fixed = with_mv_lengths(device.clone());
        assert_ne!(fixed, device);
        let units = nal::split_annex_b(&fixed);
        assert_eq!(units.len(), 2);
        assert_eq!(
            units[0],
            raise_mv_lengths(&sps, LOG2_MAX_MV_LENGTH).unwrap()
        );
        assert_eq!(units[1], pps);
        // Raised, they stay.
        assert_eq!(with_mv_lengths(fixed.clone()), fixed);
    }

    #[test]
    fn slices_share_the_rows_out_evenly() {
        // 45 rows of 80 macroblocks in 4 slices: from rows 0, 11, 22 and 33.
        assert_eq!(codec(4).slice_starts(), [0, 880, 1760, 2640]);
        assert_eq!(codec(1).slice_starts(), [0]);
        // No more slices than rows.
        assert_eq!(codec(100).slice_starts().len(), 45);
    }
}
