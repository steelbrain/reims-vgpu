//! H.265 encoding on Vulkan Video: the VPS, SPS and PPS as std structures,
//! which the device writes as NAL units
//! (`vkGetEncodedVideoSessionParametersKHR`), and each planned picture's
//! std information: its type and order count, its short-term reference
//! picture set, its reference lists in DPB slots, its slice segments'
//! headers, and its references'.
//!
//! The stream's shape: Main (8-bit, from NV12) or Main 10 (from P010), Main
//! tier, one temporal layer, the largest CTB the device codes, coding
//! blocks from 16x16, `PicOrderCntVal` the place in display order from the
//! IDR picture (its LSBs wrapping at 256). IDR pictures are `IDR_W_RADL`;
//! other I pictures are trailing pictures, not IRAP, since the B pictures
//! before them in display order predict across them. Each picture carries
//! its own reference picture set, every reference the DPB keeps, those its
//! lists name marked used by it; lists in their default order (no
//! modification). Temporal MV prediction and SAO where the device writes
//! them; a QP that varies within a picture where the bitrate is controlled.
//!
//! **Coding blocks from 16x16**, as NVENC's own presets have them: with 8x8
//! ones allowed the RTX 5080 (615.71.09) takes 3.6 to 4.6 times the GPU
//! time a picture, 15.6 to 17.9 ms at 4K and 3.9 ms at 1080p, against 4.0
//! to 4.3 ms and 0.85 ms (quality levels 0 and 1, 11-03).

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoEncodeH265PictureInfo, StdVideoEncodeH265PictureInfoFlags,
    StdVideoEncodeH265ReferenceInfo, StdVideoEncodeH265ReferenceListsInfo,
    StdVideoEncodeH265SliceSegmentHeader, StdVideoEncodeH265SliceSegmentHeaderFlags,
    StdVideoH265DecPicBufMgr, StdVideoH265PictureParameterSet, StdVideoH265PictureType,
    StdVideoH265PpsFlags, StdVideoH265ProfileTierLevel, StdVideoH265ProfileTierLevelFlags,
    StdVideoH265SequenceParameterSet, StdVideoH265SequenceParameterSetVui,
    StdVideoH265ShortTermRefPicSet, StdVideoH265SliceType, StdVideoH265SpsFlags,
    StdVideoH265SpsVuiFlags, StdVideoH265VideoParameterSet, StdVideoH265VpsFlags,
};
use reims_vgpu_video_codec::encode::hevc::{LevelNeeds, choose_level, hvcc};
use reims_vgpu_video_codec::encode::nal;

use super::caps::{EncodeCapabilities, EncodeCodec, EncodeProfile};
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
use crate::profile::h265_std_level;
use crate::std_types::zeroed;
use crate::timeline::Timeline;

/// `log2_max_pic_order_cnt_lsb_minus4`: the order count's LSBs wrap at
/// 256, far past any distance between a picture and its references.
const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u8 = 4;
/// `log2(MinCbSizeY)`: 16x16 coding blocks, so the coded size is the
/// frame's rounded up to 16.
const LOG2_MIN_CB: u32 = 4;
/// The coded size's step: `MinCbSizeY`.
const MIN_CB: u32 = 1 << LOG2_MIN_CB;

/// `VK_VIDEO_ENCODE_H265_STD_SAMPLE_ADAPTIVE_OFFSET_ENABLED_FLAG_SET_BIT_KHR`.
const STD_SAO: u32 = 0x2;
/// `VK_VIDEO_ENCODE_H265_STD_SPS_TEMPORAL_MVP_ENABLED_FLAG_SET_BIT_KHR`.
const STD_TEMPORAL_MVP: u32 = 0x10;

/// An H.265 encoder on a [`VideoDevice`]'s encode queue, reading frames
/// from sources `S` and writing each picture into a buffer `B` gives:
/// [`super::H264Encoder`]'s shape, Main from NV12 and Main 10 from P010.
pub struct H265Encoder<S: EncodeSource, B: BitstreamSource> {
    inner: Encoder<S, B>,
    level_idc: u8,
}

impl<S: EncodeSource, B: BitstreamSource> H265Encoder<S, B> {
    /// An encoder of `profile` ([`EncodeProfile::H265_MAIN`] or
    /// [`EncodeProfile::H265_MAIN_10`], with a tuning) for `config`, its
    /// bitstreams in the buffers `targets` gives, checked against what the
    /// device takes, as [`super::H264Encoder::new`] checks.
    pub fn new(
        device: &Arc<VideoDevice>,
        profile: EncodeProfile,
        config: &EncodeConfig,
        targets: B,
    ) -> Result<Self, Error> {
        if profile.codec != EncodeCodec::H265 {
            return Err(Error::Unsupported(format!(
                "{profile:?} in an H.265 encoder"
            )));
        }
        let settings = Settings::new(device, profile, config, MIN_CB, 8)?;
        let blocks = Blocks::of(&settings.caps)?;
        let ctb_rows = settings.coded.height.div_ceil(blocks.ctb());
        if settings.config.slices > ctb_rows {
            return Err(Error::Unsupported(format!(
                "{} slice segments: the picture has {ctb_rows} rows of CTBs",
                settings.config.slices
            )));
        }
        let level_idc = level(&settings)?;
        let timeline = Timeline::new(device)?;
        let mut session = EncodeSession::new(
            device,
            &timeline,
            &settings,
            CodecSession::H265 {
                max_level: h265_std_level(level_idc),
            },
        )?;
        let codec = H265Codec::new(device, &settings, blocks, level_idc, &mut session)?;
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
    /// frames encoded now, in decode order, as
    /// [`super::H264Encoder::encode`] does.
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

    /// The VPS, SPS and PPS, Annex B, as the device wrote them: what goes
    /// in front of each IDR picture in band.
    pub fn parameter_sets(&self) -> &[u8] {
        self.inner.codec.parameter_sets()
    }

    /// The `hvcC` record of the parameter sets, with four-byte NAL
    /// lengths; empty where the device wrote no SPS it parses.
    pub fn hvcc(&self) -> Vec<u8> {
        let units = nal::split_annex_b(self.parameter_sets());
        let find = |kind: u8| {
            units
                .iter()
                .find(|u| u.first().map(|b| (b >> 1) & 0x3f) == Some(kind))
        };
        match (find(32), find(33), find(34)) {
            (Some(vps), Some(sps), Some(pps)) => hvcc(vps, sps, pps).unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// What the device answers for its profile.
    pub fn capabilities(&self) -> &EncodeCapabilities {
        &self.inner.caps
    }

    /// The level, as `general_level_idc` numbers it: 30 times the level.
    pub fn level_idc(&self) -> u8 {
        self.level_idc
    }

    /// The coded size: the frame rounded up to whole 16x16 coding blocks.
    /// The device may code more, to whole CTBs, and say so in the SPS it
    /// writes; the conformance window crops either to the frame.
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
/// frame rate, the peak and the DPB, Main tier; refused past the
/// device's.
fn level(settings: &Settings) -> Result<u8, Error> {
    let needs = LevelNeeds {
        width: settings.coded.width,
        height: settings.coded.height,
        frames_per_second: settings.rate.frame_rate.as_f64(),
        bits_per_second: settings
            .rate
            .peak_bits_per_second()
            .map(|peak| u32::try_from(peak).unwrap_or(u32::MAX)),
        dpb_pictures: settings.limits.max_references + 1,
    };
    let level = match settings.config.level {
        Some(level) => level,
        None => choose_level(&needs).map_err(|e| Error::Unsupported(e.0))?,
    };
    if level > settings.caps.max_level_idc {
        return Err(Error::Unsupported(format!(
            "H.265 level {}.{}: the device encodes up to {}.{}",
            level / 30,
            level % 30 / 3,
            settings.caps.max_level_idc / 30,
            settings.caps.max_level_idc % 30 / 3
        )));
    }
    Ok(level)
}

/// The block sizes of the stream: the largest CTB the device codes, and
/// the transform blocks it does within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Blocks {
    log2_ctb: u32,
    log2_min_tb: u32,
    log2_max_tb: u32,
}

impl Blocks {
    /// The sizes from the device's answers: CTBs of 16 (bit 0), 32 and 64,
    /// transform blocks of 4 (bit 0), 8, 16 and 32. The least transform
    /// block is under the least coding block (7.4.3.2.1): 4x4 or 8x8.
    fn of(caps: &EncodeCapabilities) -> Result<Self, Error> {
        let ctbs = caps.ctb_sizes & 0x7;
        let tbs = caps.transform_block_sizes & 0x3;
        if ctbs == 0 || tbs == 0 {
            return Err(Error::Unsupported(format!(
                "H.265 CTB sizes {:#x} and transform blocks {:#x}: a CTB, and 4x4 or 8x8 \
                 transform blocks, are needed",
                caps.ctb_sizes, caps.transform_block_sizes
            )));
        }
        let log2_ctb = 4 + (31 - ctbs.leading_zeros());
        let all = caps.transform_block_sizes & 0xf;
        let log2_min_tb = 2 + tbs.trailing_zeros();
        let log2_max_tb = (2 + (31 - all.leading_zeros())).min(log2_ctb).min(5);
        Ok(Self {
            log2_ctb,
            log2_min_tb,
            log2_max_tb,
        })
    }

    fn ctb(self) -> u32 {
        1 << self.log2_ctb
    }
}

/// The parameter sets, as std structures, and what they point at.
struct StdSets {
    vps: StdVideoH265VideoParameterSet,
    sps: StdVideoH265SequenceParameterSet,
    pps: StdVideoH265PictureParameterSet,
    _ptl: Box<StdVideoH265ProfileTierLevel>,
    _dpb: Box<StdVideoH265DecPicBufMgr>,
    _vui: Box<StdVideoH265SequenceParameterSetVui>,
}

/// The parameter sets of a stream of `settings` at `level_idc`, of
/// `blocks`.
fn parameter_sets(settings: &Settings, blocks: Blocks, level_idc: u8) -> StdSets {
    let caps = &settings.caps;
    let config = &settings.config;
    let small = |n: u32| u8::try_from(n).unwrap_or(u8::MAX);

    let mut ptl_flags: StdVideoH265ProfileTierLevelFlags = zeroed();
    ptl_flags.set_general_progressive_source_flag(1);
    ptl_flags.set_general_frame_only_constraint_flag(1);
    let mut ptl: StdVideoH265ProfileTierLevel = zeroed();
    ptl.flags = ptl_flags;
    ptl.general_profile_idc = u32::from(caps.profile.profile_idc);
    ptl.general_level_idc = h265_std_level(level_idc);
    let ptl = Box::new(ptl);

    // The DPB holds the references and the picture being decoded; B
    // pictures are never references, so only their anchor is decoded
    // before frames it follows in display order.
    let mut dpb: StdVideoH265DecPicBufMgr = zeroed();
    dpb.max_dec_pic_buffering_minus1[0] = small(settings.limits.max_references);
    dpb.max_num_reorder_pics[0] = u8::from(settings.limits.b_frames > 0);
    let dpb = Box::new(dpb);

    let mut vps_flags: StdVideoH265VpsFlags = zeroed();
    vps_flags.set_vps_temporal_id_nesting_flag(1);
    vps_flags.set_vps_sub_layer_ordering_info_present_flag(1);
    let mut vps: StdVideoH265VideoParameterSet = zeroed();
    vps.flags = vps_flags;
    vps.vps_video_parameter_set_id = 0;
    vps.pDecPicBufMgr = std::ptr::from_ref(dpb.as_ref());
    vps.pProfileTierLevel = std::ptr::from_ref(ptl.as_ref());

    let mut vui_flags: StdVideoH265SpsVuiFlags = zeroed();
    vui_flags.set_video_signal_type_present_flag(u32::from(
        config.full_range || config.colour.is_some(),
    ));
    vui_flags.set_video_full_range_flag(u32::from(config.full_range));
    vui_flags.set_colour_description_present_flag(u32::from(config.colour.is_some()));
    vui_flags.set_vui_timing_info_present_flag(1);
    let mut vui: StdVideoH265SequenceParameterSetVui = zeroed();
    vui.flags = vui_flags;
    // Unspecified, where no colour is given.
    vui.video_format = 5;
    let colour = config
        .colour
        .map_or((2, 2, 2), |c| (c.primaries, c.transfer, c.matrix));
    (
        vui.colour_primaries,
        vui.transfer_characteristics,
        vui.matrix_coeffs,
    ) = colour;
    // One tick a frame (E.3.1).
    let rate = settings.rate.frame_rate;
    vui.vui_num_units_in_tick = rate.denominator;
    vui.vui_time_scale = rate.numerator;
    let vui = Box::new(vui);

    let sao = caps.writes(STD_SAO);
    let mut sps_flags: StdVideoH265SpsFlags = zeroed();
    sps_flags.set_sps_temporal_id_nesting_flag(1);
    sps_flags.set_sps_sub_layer_ordering_info_present_flag(1);
    let crop_right = (settings.coded.width - config.width) / 2;
    let crop_bottom = (settings.coded.height - config.height) / 2;
    sps_flags.set_conformance_window_flag(u32::from(crop_right > 0 || crop_bottom > 0));
    sps_flags.set_sample_adaptive_offset_enabled_flag(u32::from(sao));
    sps_flags.set_sps_temporal_mvp_enabled_flag(u32::from(caps.writes(STD_TEMPORAL_MVP)));
    sps_flags.set_strong_intra_smoothing_enabled_flag(1);
    sps_flags.set_vui_parameters_present_flag(1);
    let mut sps: StdVideoH265SequenceParameterSet = zeroed();
    sps.flags = sps_flags;
    // 4:2:0.
    sps.chroma_format_idc = 1;
    sps.pic_width_in_luma_samples = settings.coded.width;
    sps.pic_height_in_luma_samples = settings.coded.height;
    sps.sps_video_parameter_set_id = 0;
    sps.sps_seq_parameter_set_id = 0;
    let depth = caps.profile.bit_depth() - 8;
    sps.bit_depth_luma_minus8 = depth;
    sps.bit_depth_chroma_minus8 = depth;
    sps.log2_max_pic_order_cnt_lsb_minus4 = LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4;
    sps.log2_min_luma_coding_block_size_minus3 = small(LOG2_MIN_CB - 3);
    sps.log2_diff_max_min_luma_coding_block_size = small(blocks.log2_ctb - LOG2_MIN_CB);
    sps.log2_min_luma_transform_block_size_minus2 = small(blocks.log2_min_tb - 2);
    sps.log2_diff_max_min_luma_transform_block_size =
        small(blocks.log2_max_tb - blocks.log2_min_tb);
    let depth = small(blocks.log2_ctb - blocks.log2_min_tb);
    sps.max_transform_hierarchy_depth_inter = depth;
    sps.max_transform_hierarchy_depth_intra = depth;
    sps.num_short_term_ref_pic_sets = 0;
    sps.num_long_term_ref_pics_sps = 0;
    // In chroma samples (7.4.3.2.1).
    sps.conf_win_right_offset = crop_right;
    sps.conf_win_bottom_offset = crop_bottom;
    sps.pProfileTierLevel = std::ptr::from_ref(ptl.as_ref());
    sps.pDecPicBufMgr = std::ptr::from_ref(dpb.as_ref());
    sps.pSequenceParameterSetVui = std::ptr::from_ref(vui.as_ref());

    let mut pps_flags: StdVideoH265PpsFlags = zeroed();
    pps_flags.set_cu_qp_delta_enabled_flag(u32::from(settings.rate.constant_qp.is_none()));
    pps_flags.set_pps_loop_filter_across_slices_enabled_flag(1);
    let mut pps: StdVideoH265PictureParameterSet = zeroed();
    pps.flags = pps_flags;
    pps.pps_pic_parameter_set_id = 0;
    pps.pps_seq_parameter_set_id = 0;
    pps.sps_video_parameter_set_id = 0;
    pps.num_ref_idx_l0_default_active_minus1 = 0;
    pps.num_ref_idx_l1_default_active_minus1 = 0;

    StdSets {
        vps,
        sps,
        pps,
        _ptl: ptl,
        _dpb: dpb,
        _vui: vui,
    }
}

/// The H.265 half of an encoder: its parameter sets, as the device wrote
/// them, and what each picture's std information is made from.
struct H265Codec {
    parameter_sets: Vec<u8>,
    width_in_ctbs: u32,
    height_in_ctbs: u32,
    slices: u32,
    sao: bool,
    temporal_mvp: bool,
}

impl H265Codec {
    /// The parameter sets of `settings` at `level_idc` in a parameters
    /// object of `session`'s, which it takes, and their NAL units.
    fn new(
        device: &Arc<VideoDevice>,
        settings: &Settings,
        blocks: Blocks,
        level_idc: u8,
        session: &mut EncodeSession,
    ) -> Result<Self, Error> {
        let sets = parameter_sets(settings, blocks, level_idc);
        let vps = [sets.vps];
        let sps = [sets.sps];
        let pps = [sets.pps];
        let add = vk::VideoEncodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps)
            .std_sp_ss(&sps)
            .std_pp_ss(&pps);
        let mut h265 = vk::VideoEncodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add);
        let mut quality =
            vk::VideoEncodeQualityLevelInfoKHR::default().quality_level(session.quality_level());
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session.handle())
            .push_next(&mut h265)
            .push_next(&mut quality);
        let parameters = Parameters::new(device, &info)?;
        let parameter_sets = encoded_parameter_sets(device, parameters.handle())?;
        drop(sets);
        session.set_parameters(parameters);
        let ctb = blocks.ctb();
        Ok(Self {
            parameter_sets,
            width_in_ctbs: settings.coded.width.div_ceil(ctb),
            height_in_ctbs: settings.coded.height.div_ceil(ctb),
            slices: settings.config.slices,
            sao: settings.caps.writes(STD_SAO),
            temporal_mvp: settings.caps.writes(STD_TEMPORAL_MVP),
        })
    }
}

/// The VPS, SPS and PPS of `parameters`, Annex B, as the device writes
/// them.
fn encoded_parameter_sets(
    device: &VideoDevice,
    parameters: vk::VideoSessionParametersKHR,
) -> Result<Vec<u8>, Error> {
    let mut h265 = vk::VideoEncodeH265SessionParametersGetInfoKHR::default()
        .write_std_vps(true)
        .write_std_sps(true)
        .write_std_pps(true)
        .std_vps_id(0)
        .std_sps_id(0)
        .std_pps_id(0);
    let get = vk::VideoEncodeSessionParametersGetInfoKHR::default()
        .video_session_parameters(parameters)
        .push_next(&mut h265);
    let mut h265_feedback = vk::VideoEncodeH265SessionParametersFeedbackInfoKHR::default();
    let mut feedback =
        vk::VideoEncodeSessionParametersFeedbackInfoKHR::default().push_next(&mut h265_feedback);
    let data = encoded_parameters(device, &get, &mut feedback)?;
    if feedback.has_overrides == vk::TRUE {
        log::info!(
            "the device overrides the H.265 parameter sets (VPS {}, SPS {}, PPS {})",
            h265_feedback.has_std_vps_overrides,
            h265_feedback.has_std_sps_overrides,
            h265_feedback.has_std_pps_overrides
        );
    }
    Ok(data)
}

/// A picture's std information, with what it points at.
struct PictureStd {
    picture: StdVideoEncodeH265PictureInfo,
    /// What `picture` points at.
    _lists: Box<StdVideoEncodeH265ReferenceListsInfo>,
    /// What `picture` points at.
    _rps: Box<StdVideoH265ShortTermRefPicSet>,
    slices: Vec<StdVideoEncodeH265SliceSegmentHeader>,
    setup: StdVideoEncodeH265ReferenceInfo,
    references: Vec<StdVideoEncodeH265ReferenceInfo>,
    reference_slots: Vec<i32>,
}

/// `STD_VIDEO_H265_NO_REFERENCE_PICTURE`.
const NO_REFERENCE: u8 = 0xff;

fn picture_type(kind: PictureKind) -> StdVideoH265PictureType {
    match kind {
        PictureKind::P => 0,
        PictureKind::B => 1,
        PictureKind::I => 2,
        PictureKind::Idr => 3,
    }
}

fn slice_type(kind: PictureKind) -> StdVideoH265SliceType {
    match kind {
        PictureKind::B => 0,
        PictureKind::P => 1,
        PictureKind::I | PictureKind::Idr => 2,
    }
}

/// H.265's `PicOrderCntVal` of a frame at `order`.
fn order_count(order: u32) -> i32 {
    i32::try_from(order).unwrap_or(i32::MAX)
}

/// The short-term reference picture set of a picture at `order` with
/// `dpb` kept (8.3.2, 7.4.8): those before it nearest first, then those
/// after it nearest first, each marked used where `used` names it.
fn reference_set(
    order: u32,
    dpb: &[Reference],
    used: &[Reference],
) -> StdVideoH265ShortTermRefPicSet {
    let mut before: Vec<&Reference> = dpb.iter().filter(|r| r.order < order).collect();
    let mut after: Vec<&Reference> = dpb.iter().filter(|r| r.order > order).collect();
    before.sort_by_key(|r| std::cmp::Reverse(r.order));
    after.sort_by_key(|r| r.order);
    let is_used = |r: &Reference| used.iter().any(|u| u.slot == r.slot);
    let mut rps: StdVideoH265ShortTermRefPicSet = zeroed();
    let mut last = order;
    for (i, r) in before.iter().take(16).enumerate() {
        rps.delta_poc_s0_minus1[i] = u16::try_from(last - r.order - 1).unwrap_or(u16::MAX);
        if is_used(r) {
            rps.used_by_curr_pic_s0_flag |= 1 << i;
        }
        last = r.order;
    }
    let mut last = order;
    for (i, r) in after.iter().take(16).enumerate() {
        rps.delta_poc_s1_minus1[i] = u16::try_from(r.order - last - 1).unwrap_or(u16::MAX);
        if is_used(r) {
            rps.used_by_curr_pic_s1_flag |= 1 << i;
        }
        last = r.order;
    }
    rps.num_negative_pics = u8::try_from(before.len().min(16)).unwrap_or(16);
    rps.num_positive_pics = u8::try_from(after.len().min(16)).unwrap_or(16);
    rps
}

impl H265Codec {
    /// `picture`'s std information.
    fn picture(&self, picture: &PictureRefs) -> PictureStd {
        let slot = |r: &Reference| u8::try_from(r.slot).unwrap_or(NO_REFERENCE);
        let mut lists: StdVideoEncodeH265ReferenceListsInfo = zeroed();
        lists.RefPicList0 = [NO_REFERENCE; 15];
        lists.RefPicList1 = [NO_REFERENCE; 15];
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
        let used: Vec<Reference> = named.iter().map(|r| **r).collect();
        let rps = Box::new(if picture.kind == PictureKind::Idr {
            zeroed()
        } else {
            reference_set(picture.order, &picture.dpb, &used)
        });

        let intra = matches!(picture.kind, PictureKind::Idr | PictureKind::I);
        let mut flags: StdVideoEncodeH265PictureInfoFlags = zeroed();
        flags.set_is_reference(u32::from(picture.reference));
        flags.set_IrapPicFlag(u32::from(picture.kind == PictureKind::Idr));
        flags.set_pic_output_flag(1);
        flags.set_slice_temporal_mvp_enabled_flag(u32::from(self.temporal_mvp && !intra));
        let mut info: StdVideoEncodeH265PictureInfo = zeroed();
        info.flags = flags;
        info.pic_type = picture_type(picture.kind);
        info.sps_video_parameter_set_id = 0;
        info.pps_seq_parameter_set_id = 0;
        info.pps_pic_parameter_set_id = 0;
        info.short_term_ref_pic_set_idx = 0;
        info.PicOrderCntVal = order_count(picture.order);
        info.TemporalId = 0;
        info.pRefLists = std::ptr::from_ref(lists.as_ref());
        info.pShortTermRefPicSet = std::ptr::from_ref(rps.as_ref());

        let mut setup: StdVideoEncodeH265ReferenceInfo = zeroed();
        setup.pic_type = info.pic_type;
        setup.PicOrderCntVal = info.PicOrderCntVal;

        // The PPS's defaults are one reference a list.
        let overrides = match picture.kind {
            PictureKind::P => picture.list0.len() != 1,
            PictureKind::B => picture.list0.len() != 1 || picture.list1.len() != 1,
            PictureKind::I | PictureKind::Idr => false,
        };
        let slices = self
            .slice_starts()
            .into_iter()
            .enumerate()
            .map(|(i, address)| {
                let mut flags: StdVideoEncodeH265SliceSegmentHeaderFlags = zeroed();
                flags.set_first_slice_segment_in_pic_flag(u32::from(i == 0));
                flags.set_slice_sao_luma_flag(u32::from(self.sao));
                flags.set_slice_sao_chroma_flag(u32::from(self.sao));
                flags.set_num_ref_idx_active_override_flag(u32::from(overrides));
                flags.set_collocated_from_l0_flag(1);
                flags.set_slice_loop_filter_across_slices_enabled_flag(1);
                let mut header: StdVideoEncodeH265SliceSegmentHeader = zeroed();
                header.flags = flags;
                header.slice_type = slice_type(picture.kind);
                header.slice_segment_address = address;
                header.collocated_ref_idx = 0;
                header.MaxNumMergeCand = 5;
                header
            })
            .collect();
        PictureStd {
            picture: info,
            _lists: lists,
            _rps: rps,
            slices,
            setup,
            references: named
                .iter()
                .map(|r| {
                    let mut info: StdVideoEncodeH265ReferenceInfo = zeroed();
                    info.pic_type = picture_type(r.kind);
                    info.PicOrderCntVal = order_count(r.order);
                    info
                })
                .collect(),
            reference_slots: named
                .iter()
                .map(|r| i32::try_from(r.slot).unwrap_or(-1))
                .collect(),
        }
    }

    /// Each slice segment's first CTB: the picture's CTB rows shared out
    /// evenly.
    fn slice_starts(&self) -> Vec<u32> {
        let slices = self.slices.clamp(1, self.height_in_ctbs.max(1));
        (0..slices)
            .map(|i| i * self.height_in_ctbs / slices * self.width_in_ctbs)
            .collect()
    }
}

impl CodecEncoder for H265Codec {
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
            codec: CodecInfo::H265 {
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

    /// The RTX 5080's H.265 answers that shape the stream.
    fn h265_caps(profile: EncodeProfile) -> EncodeCapabilities {
        let mut caps = caps();
        caps.profile = profile;
        caps.max_level_idc = 186;
        caps.ctb_sizes = 0x2;
        caps.transform_block_sizes = 0xf;
        caps.std_syntax_flags = 0x53d02;
        caps.max_p_l0 = 4;
        caps.max_b_l0 = 4;
        caps.max_l1 = 2;
        caps
    }

    fn settings(width: u32, height: u32, profile: EncodeProfile, b_frames: u32) -> Settings {
        let caps = h265_caps(profile);
        let mut config = EncodeConfig::new(width, height);
        config.gop = GopConfig {
            idr_period: Some(60),
            intra_period: None,
            b_frames,
            references: 2,
        };
        config.frame_rate = FrameRate::new(60, 1);
        let rate =
            crate::encode::rate::plan(config.rate_control, None, config.frame_rate, &caps).unwrap();
        let limits = Limits::new(&config.gop, true, 4, 16, 8);
        Settings {
            coded: vk::Extent2D {
                width: width.next_multiple_of(MIN_CB),
                height: height.next_multiple_of(MIN_CB),
            },
            caps,
            config,
            limits,
            rate,
            quality_level: 0,
        }
    }

    #[test]
    fn blocks_take_the_largest_ctb_and_transform_the_device_codes() {
        let caps = h265_caps(EncodeProfile::H265_MAIN);
        assert_eq!(
            Blocks::of(&caps).unwrap(),
            Blocks {
                log2_ctb: 5,
                log2_min_tb: 2,
                log2_max_tb: 5
            }
        );
        let mut small = caps.clone();
        small.ctb_sizes = 0x1;
        assert_eq!(Blocks::of(&small).unwrap().log2_max_tb, 4);
        let mut no_4x4 = caps.clone();
        no_4x4.transform_block_sizes = 0xe;
        assert_eq!(Blocks::of(&no_4x4).unwrap().log2_min_tb, 3);
        let mut large = caps;
        large.transform_block_sizes = 0xc;
        assert!(Blocks::of(&large).is_err());
    }

    #[test]
    fn main_10_at_1080p_codes_whole_min_cbs_and_its_depth() {
        let settings = settings(1920, 1080, EncodeProfile::H265_MAIN_10, 2);
        let blocks = Blocks::of(&settings.caps).unwrap();
        let sets = parameter_sets(&settings, blocks, 123);
        // 1080 rounded up to 16x16 coding blocks, cropped back.
        assert_eq!(sets.sps.pic_height_in_luma_samples, 1088);
        assert_eq!(sets.sps.flags.conformance_window_flag(), 1);
        assert_eq!(sets.sps.conf_win_bottom_offset, 4);
        assert_eq!(sets.sps.bit_depth_luma_minus8, 2);
        assert_eq!(sets.sps.bit_depth_chroma_minus8, 2);
        assert_eq!(sets.sps.log2_min_luma_coding_block_size_minus3, 1);
        assert_eq!(sets.sps.log2_diff_max_min_luma_coding_block_size, 1);
        assert_eq!(sets.sps.log2_diff_max_min_luma_transform_block_size, 3);
        assert_eq!(sets.sps.max_transform_hierarchy_depth_inter, 3);
        assert_eq!(sets.sps.flags.sample_adaptive_offset_enabled_flag(), 1);
        // The 5080 does not set temporal MV prediction.
        assert_eq!(sets.sps.flags.sps_temporal_mvp_enabled_flag(), 0);
        // SAFETY: the pointer is to a box `sets` holds.
        let ptl = unsafe { &*sets.sps.pProfileTierLevel };
        // SAFETY: as above.
        let dpb = unsafe { &*sets.sps.pDecPicBufMgr };
        assert_eq!(ptl.general_profile_idc, 2);
        assert_eq!(crate::profile::h265_level_idc(ptl.general_level_idc), 123);
        assert_eq!(dpb.max_dec_pic_buffering_minus1[0], 2);
        assert_eq!(dpb.max_num_reorder_pics[0], 1);
        assert!(std::ptr::eq(sets.vps.pDecPicBufMgr, sets.sps.pDecPicBufMgr));
        assert_eq!(sets._vui.vui_time_scale, 60);
        assert_eq!(sets._vui.vui_num_units_in_tick, 1);
        assert_eq!(sets.pps.flags.cu_qp_delta_enabled_flag(), 1);
    }

    #[test]
    fn an_odd_size_is_cropped_in_chroma_samples() {
        let settings = settings(1282, 722, EncodeProfile::H265_MAIN, 0);
        let blocks = Blocks::of(&settings.caps).unwrap();
        let sets = parameter_sets(&settings, blocks, 93);
        assert_eq!(sets.sps.pic_width_in_luma_samples, 1296);
        assert_eq!(sets.sps.pic_height_in_luma_samples, 736);
        assert_eq!(sets.sps.flags.conformance_window_flag(), 1);
        assert_eq!(sets.sps.conf_win_right_offset, 7);
        assert_eq!(sets.sps.conf_win_bottom_offset, 7);
        // SAFETY: as above.
        let dpb = unsafe { &*sets.sps.pDecPicBufMgr };
        assert_eq!(dpb.max_num_reorder_pics[0], 0);
    }

    fn reference(order: u32, slot: u32, kind: PictureKind) -> Reference {
        Reference {
            number: u64::from(order),
            slot,
            order,
            frame_num: 0,
            kind,
        }
    }

    #[test]
    fn a_reference_set_lists_the_dpb_nearest_first_marking_the_lists() {
        // A B picture at 5 between anchors at 4 and 8, with 0 kept.
        let dpb = [
            reference(0, 0, PictureKind::Idr),
            reference(4, 1, PictureKind::P),
            reference(8, 2, PictureKind::P),
        ];
        let rps = reference_set(5, &dpb, &[dpb[1], dpb[2]]);
        assert_eq!((rps.num_negative_pics, rps.num_positive_pics), (2, 1));
        // 5 - 4 - 1, then 4 - 0 - 1.
        assert_eq!(&rps.delta_poc_s0_minus1[..2], [0, 3]);
        assert_eq!(rps.used_by_curr_pic_s0_flag, 0b01);
        // 8 - 5 - 1.
        assert_eq!(rps.delta_poc_s1_minus1[0], 2);
        assert_eq!(rps.used_by_curr_pic_s1_flag, 0b1);
    }

    fn codec(slices: u32) -> H265Codec {
        H265Codec {
            parameter_sets: Vec::new(),
            width_in_ctbs: 60,
            height_in_ctbs: 34,
            slices,
            sao: true,
            temporal_mvp: false,
        }
    }

    #[test]
    fn a_p_picture_names_its_references_and_keeps_the_rest() {
        let dpb = vec![
            reference(0, 0, PictureKind::Idr),
            reference(1, 1, PictureKind::P),
            reference(2, 2, PictureKind::P),
        ];
        let picture = PictureRefs {
            kind: PictureKind::P,
            order: 3,
            frame_num: 3,
            idr_pic_id: 0,
            reference: true,
            setup_slot: 3,
            list0: vec![dpb[2], dpb[1]],
            list1: Vec::new(),
            dpb,
        };
        let std = codec(1).picture(&picture);
        assert_eq!(&std._lists.RefPicList0[..3], [2, 1, NO_REFERENCE]);
        assert_eq!(std._lists.num_ref_idx_l0_active_minus1, 1);
        assert_eq!(std.reference_slots, [2, 1]);
        assert_eq!(std._rps.num_negative_pics, 3);
        assert_eq!(std._rps.used_by_curr_pic_s0_flag, 0b011);
        assert_eq!(std.picture.pic_type, 0);
        assert_eq!(std.picture.PicOrderCntVal, 3);
        assert_eq!(std.picture.flags.IrapPicFlag(), 0);
        assert!(std::ptr::eq(
            std.picture.pShortTermRefPicSet,
            std._rps.as_ref()
        ));
        assert_eq!(std.slices[0].slice_type, 1);
        assert_eq!(std.slices[0].flags.num_ref_idx_active_override_flag(), 1);
        assert_eq!(std.slices[0].flags.slice_sao_luma_flag(), 1);
        assert_eq!(std.references[1].PicOrderCntVal, 1);
    }

    #[test]
    fn an_idr_picture_is_irap_and_an_i_picture_trails() {
        let mut picture = PictureRefs {
            kind: PictureKind::Idr,
            order: 0,
            frame_num: 0,
            idr_pic_id: 1,
            reference: true,
            setup_slot: 0,
            list0: Vec::new(),
            list1: Vec::new(),
            dpb: Vec::new(),
        };
        let std = codec(1).picture(&picture);
        assert_eq!(std.picture.flags.IrapPicFlag(), 1);
        assert_eq!(std.picture.pic_type, 3);
        assert_eq!(std.slices[0].slice_type, 2);
        picture.kind = PictureKind::I;
        picture.order = 30;
        picture.dpb = vec![reference(29, 1, PictureKind::P)];
        let std = codec(1).picture(&picture);
        assert_eq!(std.picture.flags.IrapPicFlag(), 0);
        assert_eq!(std.picture.pic_type, 2);
        // Kept for the B pictures before it, used by none of its slices.
        assert_eq!(std._rps.num_negative_pics, 1);
        assert_eq!(std._rps.used_by_curr_pic_s0_flag, 0);
    }

    #[test]
    fn slice_segments_share_the_ctb_rows_out_evenly() {
        assert_eq!(codec(1).slice_starts(), [0]);
        // 34 rows of 60 CTBs in 3: from rows 0, 11 and 22.
        assert_eq!(codec(3).slice_starts(), [0, 660, 1320]);
    }
}
