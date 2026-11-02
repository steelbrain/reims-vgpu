//! The parameter sets as Vulkan Video's std structures
//! (vulkan_video_codec_h265std.h), and the session parameters object that
//! holds them.
//!
//! - **The VPS** is made from the SPS. The codec crate's decoding process
//!   reads none (a base-layer decode needs none) and drops VPS units, but a
//!   parameters object holds one for each SPS's
//!   `sps_video_parameter_set_id`: it has the SPS's sub-layers, temporal
//!   nesting, profile, tier, level and DPB limits, as the one-layer VPS of
//!   a conforming stream has them, and no timing or HRD.
//! - **Short-term sets** of the SPS go each as an `st_ref_pic_set()` coded
//!   outright, of the pictures the stream's syntax derives (7.4.8), whether
//!   it predicts the set from another or not: the same set, whatever reads
//!   it. A slice header's own set is the hardware's to skip, as the
//!   picture's information gives its size in bits and, where it is
//!   predicted from an SPS set, that set's `NumDeltaPocs`.
//! - **Scaling lists** go as the codec crate resolves them for each set
//!   (the defaults where the SPS enables lists and codes none), marked
//!   present, in coding order (the up-right diagonal scan): the std lists
//!   are the syntax's (7.3.4), which codes them in that order. The two
//!   32x32 lists are `matrixId` 0 and 3, as 4:2:0 codes them. The H.265
//!   check's asymmetric lists (`main-720p-cqm-file`) catch a list in the
//!   wrong order or slot.
//! - **Tiles** go with every column's and row's size the std arrays hold,
//!   uniform spacing too, as 6.5.1 derives them.
//! - The VUI is left out: decoding needs none of it.

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoH265DecPicBufMgr, StdVideoH265LongTermRefPicsSps, StdVideoH265PictureParameterSet,
    StdVideoH265PpsFlags, StdVideoH265ProfileTierLevel, StdVideoH265ProfileTierLevelFlags,
    StdVideoH265ScalingLists, StdVideoH265SequenceParameterSet, StdVideoH265ShortTermRefPicSet,
    StdVideoH265SpsFlags, StdVideoH265VideoParameterSet, StdVideoH265VpsFlags,
};
use reims_vgpu_video_codec::hevc::pps::Pps;
use reims_vgpu_video_codec::hevc::scaling::ScalingLists;
use reims_vgpu_video_codec::hevc::sps::Sps;

use crate::device::VideoDevice;
use crate::error::Error;
use crate::params::Parameters;
use crate::profile::{H265Profile, h265_std_level};
use crate::session::SessionParameters;
use crate::std_types::zeroed;

/// The SPS's general profile, tier and level as a std structure: its
/// `general_profile_idc` where it is Main, Main 10 or Main Still Picture,
/// else `decoded`'s, the profile the session decodes it with (a stream the
/// codec crate takes as compatible with one of those). The source and
/// constraint flags, which the decoding process reads none of, are 0.
fn profile_tier_level(sps: &Sps, decoded: H265Profile) -> StdVideoH265ProfileTierLevel {
    let ptl = &sps.profile_tier_level;
    let mut flags: StdVideoH265ProfileTierLevelFlags = zeroed();
    flags.set_general_tier_flag(u32::from(ptl.general_tier_flag));
    let mut out: StdVideoH265ProfileTierLevel = zeroed();
    out.flags = flags;
    out.general_profile_idc = u32::from(if (1..=3).contains(&ptl.general_profile_idc) {
        ptl.general_profile_idc
    } else {
        decoded.profile_idc
    });
    out.general_level_idc = h265_std_level(ptl.general_level_idc);
    out
}

/// Each sub-layer's DPB limits, as the SPS has them (the ones it leaves
/// out inferred from the highest's).
fn dec_pic_buf_mgr(sps: &Sps) -> StdVideoH265DecPicBufMgr {
    let mut out: StdVideoH265DecPicBufMgr = zeroed();
    for (i, ordering) in sps.sub_layer_ordering.iter().enumerate().take(7) {
        if let (Some(latency), Some(buffering), Some(reorder)) = (
            out.max_latency_increase_plus1.get_mut(i),
            out.max_dec_pic_buffering_minus1.get_mut(i),
            out.max_num_reorder_pics.get_mut(i),
        ) {
            *latency = ordering.max_latency_increase_plus1;
            *buffering = ordering.max_dec_pic_buffering_minus1;
            *reorder = ordering.max_num_reorder_pics;
        }
    }
    out
}

/// A VPS as a std structure, with what it points at.
pub(crate) struct StdVps {
    pub(crate) vps: StdVideoH265VideoParameterSet,
    /// What `vps.pProfileTierLevel` points at: boxed, so that it stays put
    /// as this moves.
    _ptl: Box<StdVideoH265ProfileTierLevel>,
    /// What `vps.pDecPicBufMgr` points at.
    _dpb: Box<StdVideoH265DecPicBufMgr>,
}

impl StdVps {
    /// The VPS `sps` names, made from it, for a session of `decoded`.
    pub(crate) fn new(sps: &Sps, decoded: H265Profile) -> Self {
        let ptl = Box::new(profile_tier_level(sps, decoded));
        let dpb = Box::new(dec_pic_buf_mgr(sps));
        let mut flags: StdVideoH265VpsFlags = zeroed();
        flags.set_vps_temporal_id_nesting_flag(u32::from(sps.sps_temporal_id_nesting_flag));
        flags.set_vps_sub_layer_ordering_info_present_flag(1);
        flags.set_vps_timing_info_present_flag(0);
        let mut out: StdVideoH265VideoParameterSet = zeroed();
        out.flags = flags;
        out.vps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        out.vps_max_sub_layers_minus1 = sps.sps_max_sub_layers_minus1;
        out.pProfileTierLevel = std::ptr::from_ref(ptl.as_ref());
        out.pDecPicBufMgr = std::ptr::from_ref(dpb.as_ref());
        Self {
            vps: out,
            _ptl: ptl,
            _dpb: dpb,
        }
    }
}

/// A short-term set as `st_ref_pic_set()` codes one outright (7.3.7): its
/// pictures before the current one (`DeltaPocS0` with `UsedByCurrPicS0`,
/// nearest first) and after it, each a distance less one from the one
/// before, and the used flags as bit masks. Refused where the pictures are
/// not in that order, or more than 16 each way, which no set the codec
/// crate reads is.
pub(crate) fn explicit_set(
    negative: &[(i32, bool)],
    positive: &[(i32, bool)],
) -> Result<StdVideoH265ShortTermRefPicSet, Error> {
    let mut out: StdVideoH265ShortTermRefPicSet = zeroed();
    let bad = || Error::Unsupported("a short-term set whose pictures are out of order".into());
    let count = |side: &[(i32, bool)]| u8::try_from(side.len()).ok().filter(|&n| n <= 16);
    out.num_negative_pics = count(negative).ok_or_else(bad)?;
    out.num_positive_pics = count(positive).ok_or_else(bad)?;
    // 7-61, 7-62: each picture further than the one before it, by at
    // least 1, from 0.
    let distances = |side: &[(i32, bool)], sign: i32, minus1: &mut [u16], used: &mut u16| {
        let mut previous = 0i32;
        for (i, &(delta, flag)) in side.iter().enumerate() {
            let distance = delta
                .checked_sub(previous)
                .and_then(|d| d.checked_mul(sign))
                .and_then(|d| d.checked_sub(1))
                .and_then(|d| u16::try_from(d).ok())
                .ok_or_else(bad)?;
            *minus1.get_mut(i).ok_or_else(bad)? = distance;
            *used |= u16::from(flag) << i;
            previous = delta;
        }
        Ok::<_, Error>(())
    };
    let mut used_s0 = 0;
    let mut used_s1 = 0;
    distances(negative, -1, &mut out.delta_poc_s0_minus1, &mut used_s0)?;
    distances(positive, 1, &mut out.delta_poc_s1_minus1, &mut used_s1)?;
    out.used_by_curr_pic_s0_flag = used_s0;
    out.used_by_curr_pic_s1_flag = used_s1;
    Ok(out)
}

/// `lists` as std lists, in coding order, the two 32x32 lists those of
/// `matrixId` 0 and 3.
pub(crate) fn std_scaling_lists(lists: &ScalingLists) -> StdVideoH265ScalingLists {
    let [intra32, _, _, inter32, _, _] = lists.list_32x32;
    let [intra32_dc, _, _, inter32_dc, _, _] = lists.dc_32x32;
    let mut out: StdVideoH265ScalingLists = zeroed();
    out.ScalingList4x4 = lists.list_4x4;
    out.ScalingList8x8 = lists.list_8x8;
    out.ScalingList16x16 = lists.list_16x16;
    out.ScalingList32x32 = [intra32, inter32];
    out.ScalingListDCCoef16x16 = lists.dc_16x16;
    out.ScalingListDCCoef32x32 = [intra32_dc, inter32_dc];
    out
}

/// An SPS as a std structure, with what it points at.
pub(crate) struct StdSps {
    pub(crate) sps: StdVideoH265SequenceParameterSet,
    /// What `sps.pProfileTierLevel` points at, boxed as [`StdVps`]'s.
    _ptl: Box<StdVideoH265ProfileTierLevel>,
    /// What `sps.pDecPicBufMgr` points at.
    _dpb: Box<StdVideoH265DecPicBufMgr>,
    /// What `sps.pScalingLists` points at.
    _scaling: Option<Box<StdVideoH265ScalingLists>>,
    /// What `sps.pShortTermRefPicSet` points at.
    _short_term: Vec<StdVideoH265ShortTermRefPicSet>,
    /// What `sps.pLongTermRefPicsSps` points at.
    _long_term: Option<Box<StdVideoH265LongTermRefPicsSps>>,
}

impl StdSps {
    /// `sps`, for a session of `decoded`.
    pub(crate) fn new(sps: &Sps, decoded: H265Profile) -> Result<Self, Error> {
        let bit = u32::from;
        let pcm = sps.pcm.unwrap_or_default();
        let mut flags: StdVideoH265SpsFlags = zeroed();
        flags.set_sps_temporal_id_nesting_flag(bit(sps.sps_temporal_id_nesting_flag));
        flags.set_separate_colour_plane_flag(bit(sps.separate_colour_plane_flag));
        flags.set_conformance_window_flag(bit(sps.conformance_window.iter().any(|&c| c != 0)));
        // Every sub-layer's limits are given.
        flags.set_sps_sub_layer_ordering_info_present_flag(1);
        flags.set_scaling_list_enabled_flag(bit(sps.scaling_list_enabled_flag));
        flags.set_sps_scaling_list_data_present_flag(bit(sps.scaling_list_enabled_flag));
        flags.set_amp_enabled_flag(bit(sps.amp_enabled_flag));
        flags.set_sample_adaptive_offset_enabled_flag(bit(sps.sample_adaptive_offset_enabled_flag));
        flags.set_pcm_enabled_flag(bit(sps.pcm.is_some()));
        flags.set_pcm_loop_filter_disabled_flag(bit(pcm.loop_filter_disabled_flag));
        flags.set_long_term_ref_pics_present_flag(bit(sps.long_term_ref_pics_present_flag));
        flags.set_sps_temporal_mvp_enabled_flag(bit(sps.sps_temporal_mvp_enabled_flag));
        flags.set_strong_intra_smoothing_enabled_flag(bit(sps.strong_intra_smoothing_enabled_flag));
        flags.set_vui_parameters_present_flag(0);
        flags.set_sps_extension_present_flag(0);

        let ptl = Box::new(profile_tier_level(sps, decoded));
        let dpb = Box::new(dec_pic_buf_mgr(sps));
        let scaling = sps.scaling_list_enabled_flag.then(|| {
            Box::new(std_scaling_lists(
                &sps.scaling_lists.clone().unwrap_or_default(),
            ))
        });
        let short_term = sps
            .short_term_ref_pic_sets
            .iter()
            .map(|set| explicit_set(set.negative(), set.positive()))
            .collect::<Result<Vec<_>, Error>>()?;
        let long_term = sps.long_term_ref_pics_present_flag.then(|| {
            let mut lists: StdVideoH265LongTermRefPicsSps = zeroed();
            for (i, &(lsb, used)) in sps.long_term_ref_pics.iter().enumerate().take(32) {
                if let Some(slot) = lists.lt_ref_pic_poc_lsb_sps.get_mut(i) {
                    *slot = lsb;
                }
                lists.used_by_curr_pic_lt_sps_flag |= u32::from(used) << i;
            }
            Box::new(lists)
        });
        let [left, right, top, bottom] = sps.conformance_window;
        // num_short_term_ref_pic_sets is 64 at most and
        // num_long_term_ref_pics_sps 32 (7.4.3.2.1), as the codec crate reads
        // them.
        let count = |n: usize| u8::try_from(n).unwrap_or(u8::MAX);
        let mut out: StdVideoH265SequenceParameterSet = zeroed();
        out.flags = flags;
        out.chroma_format_idc = u32::from(sps.chroma_format_idc);
        out.pic_width_in_luma_samples = sps.pic_width_in_luma_samples;
        out.pic_height_in_luma_samples = sps.pic_height_in_luma_samples;
        out.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        out.sps_max_sub_layers_minus1 = sps.sps_max_sub_layers_minus1;
        out.sps_seq_parameter_set_id = sps.sps_seq_parameter_set_id;
        out.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
        out.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
        out.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
        out.log2_min_luma_coding_block_size_minus3 = sps.log2_min_luma_coding_block_size_minus3;
        out.log2_diff_max_min_luma_coding_block_size = sps.log2_diff_max_min_luma_coding_block_size;
        out.log2_min_luma_transform_block_size_minus2 =
            sps.log2_min_luma_transform_block_size_minus2;
        out.log2_diff_max_min_luma_transform_block_size =
            sps.log2_diff_max_min_luma_transform_block_size;
        out.max_transform_hierarchy_depth_inter = sps.max_transform_hierarchy_depth_inter;
        out.max_transform_hierarchy_depth_intra = sps.max_transform_hierarchy_depth_intra;
        out.num_short_term_ref_pic_sets = count(short_term.len());
        out.num_long_term_ref_pics_sps = count(sps.long_term_ref_pics.len());
        out.pcm_sample_bit_depth_luma_minus1 = pcm.sample_bit_depth_luma_minus1;
        out.pcm_sample_bit_depth_chroma_minus1 = pcm.sample_bit_depth_chroma_minus1;
        out.log2_min_pcm_luma_coding_block_size_minus3 = pcm.log2_min_luma_coding_block_size_minus3;
        out.log2_diff_max_min_pcm_luma_coding_block_size =
            pcm.log2_diff_max_min_luma_coding_block_size;
        out.conf_win_left_offset = left;
        out.conf_win_right_offset = right;
        out.conf_win_top_offset = top;
        out.conf_win_bottom_offset = bottom;
        out.pProfileTierLevel = std::ptr::from_ref(ptl.as_ref());
        out.pDecPicBufMgr = std::ptr::from_ref(dpb.as_ref());
        if let Some(lists) = &scaling {
            out.pScalingLists = std::ptr::from_ref(lists.as_ref());
        }
        if !short_term.is_empty() {
            out.pShortTermRefPicSet = short_term.as_ptr();
        }
        if let Some(lists) = &long_term {
            out.pLongTermRefPicsSps = std::ptr::from_ref(lists.as_ref());
        }
        Ok(Self {
            sps: out,
            _ptl: ptl,
            _dpb: dpb,
            _scaling: scaling,
            _short_term: short_term,
            _long_term: long_term,
        })
    }
}

/// A PPS as a std structure, with what it points at.
pub(crate) struct StdPps {
    pub(crate) pps: StdVideoH265PictureParameterSet,
    /// What `pps.pScalingLists` points at, boxed as [`StdSps`]'s.
    _scaling: Option<Box<StdVideoH265ScalingLists>>,
}

impl StdPps {
    /// `pps`, read against `sps`, whose lists it overrides where it codes
    /// its own.
    pub(crate) fn new(pps: &Pps, sps: &Sps) -> Result<Self, Error> {
        let bit = u32::from;
        let lists = pps
            .scaling_lists
            .as_ref()
            .filter(|_| sps.scaling_list_enabled_flag);
        let mut flags: StdVideoH265PpsFlags = zeroed();
        flags.set_dependent_slice_segments_enabled_flag(bit(
            pps.dependent_slice_segments_enabled_flag
        ));
        flags.set_output_flag_present_flag(bit(pps.output_flag_present_flag));
        flags.set_sign_data_hiding_enabled_flag(bit(pps.sign_data_hiding_enabled_flag));
        flags.set_cabac_init_present_flag(bit(pps.cabac_init_present_flag));
        flags.set_constrained_intra_pred_flag(bit(pps.constrained_intra_pred_flag));
        flags.set_transform_skip_enabled_flag(bit(pps.transform_skip_enabled_flag));
        flags.set_cu_qp_delta_enabled_flag(bit(pps.cu_qp_delta_enabled_flag));
        flags.set_pps_slice_chroma_qp_offsets_present_flag(bit(
            pps.pps_slice_chroma_qp_offsets_present_flag
        ));
        flags.set_weighted_pred_flag(bit(pps.weighted_pred_flag));
        flags.set_weighted_bipred_flag(bit(pps.weighted_bipred_flag));
        flags.set_transquant_bypass_enabled_flag(bit(pps.transquant_bypass_enabled_flag));
        flags.set_tiles_enabled_flag(bit(pps.tiles_enabled_flag));
        flags.set_entropy_coding_sync_enabled_flag(bit(pps.entropy_coding_sync_enabled_flag));
        flags.set_uniform_spacing_flag(bit(pps.uniform_spacing_flag));
        flags.set_loop_filter_across_tiles_enabled_flag(bit(
            pps.loop_filter_across_tiles_enabled_flag
        ));
        flags.set_pps_loop_filter_across_slices_enabled_flag(bit(
            pps.pps_loop_filter_across_slices_enabled_flag
        ));
        flags.set_deblocking_filter_control_present_flag(bit(
            pps.deblocking_filter_control_present_flag
        ));
        flags.set_deblocking_filter_override_enabled_flag(bit(
            pps.deblocking_filter_override_enabled_flag
        ));
        flags.set_pps_deblocking_filter_disabled_flag(bit(pps.pps_deblocking_filter_disabled_flag));
        flags.set_pps_scaling_list_data_present_flag(bit(lists.is_some()));
        flags.set_lists_modification_present_flag(bit(pps.lists_modification_present_flag));
        flags.set_slice_segment_header_extension_present_flag(bit(
            pps.slice_segment_header_extension_present_flag
        ));
        flags.set_pps_extension_present_flag(0);

        let mut out: StdVideoH265PictureParameterSet = zeroed();
        out.flags = flags;
        out.pps_pic_parameter_set_id = pps.pps_pic_parameter_set_id;
        out.pps_seq_parameter_set_id = pps.pps_seq_parameter_set_id;
        out.sps_video_parameter_set_id = sps.sps_video_parameter_set_id;
        out.num_extra_slice_header_bits = pps.num_extra_slice_header_bits;
        out.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        out.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        out.init_qp_minus26 = pps.init_qp_minus26;
        out.diff_cu_qp_delta_depth = pps.diff_cu_qp_delta_depth;
        out.pps_cb_qp_offset = pps.pps_cb_qp_offset;
        out.pps_cr_qp_offset = pps.pps_cr_qp_offset;
        out.pps_beta_offset_div2 = pps.pps_beta_offset_div2;
        out.pps_tc_offset_div2 = pps.pps_tc_offset_div2;
        out.log2_parallel_merge_level_minus2 = pps.log2_parallel_merge_level_minus2;
        out.num_tile_columns_minus1 = pps.num_tile_columns_minus1;
        out.num_tile_rows_minus1 = pps.num_tile_rows_minus1;
        if pps.tiles_enabled_flag {
            // Each column's and row's size in CTBs, less one, as many as
            // the arrays hold: 20 columns and 22 rows at most (A.4.2), the
            // last of each derived from the others.
            let minus1 = |size: u32| u16::try_from(size.saturating_sub(1)).unwrap_or(u16::MAX);
            for (slot, &width) in out
                .column_width_minus1
                .iter_mut()
                .zip(&pps.column_widths(sps)?)
            {
                *slot = minus1(width);
            }
            for (slot, &height) in out.row_height_minus1.iter_mut().zip(&pps.row_heights(sps)?) {
                *slot = minus1(height);
            }
        }
        let scaling = lists.map(|lists| Box::new(std_scaling_lists(lists)));
        if let Some(lists) = &scaling {
            out.pScalingLists = std::ptr::from_ref(lists.as_ref());
        }
        Ok(Self {
            pps: out,
            _scaling: scaling,
        })
    }
}

/// The sets `old_sps` and `old_pps` with `sps` and `pps`, each in place of a
/// set of its id. A PPS of an SPS whose id is replaced goes, but for `pps`:
/// it was read against the old SPS.
pub(crate) fn merged(
    old_sps: &[Sps],
    old_pps: &[Pps],
    sps: &Sps,
    pps: &Pps,
) -> (Vec<Sps>, Vec<Pps>) {
    let mut sets_sps = old_sps.to_vec();
    let mut sets_pps = old_pps.to_vec();
    if !sets_sps.iter().any(|s| s == sps) {
        let id = sps.sps_seq_parameter_set_id;
        sets_sps.retain(|s| s.sps_seq_parameter_set_id != id);
        sets_pps.retain(|p| p.pps_seq_parameter_set_id != id);
        sets_sps.push(sps.clone());
    }
    sets_pps.retain(|p| p.pps_pic_parameter_set_id != pps.pps_pic_parameter_set_id);
    sets_pps.push(pps.clone());
    (sets_sps, sets_pps)
}

/// The SPS each VPS is made from: for each `sps_video_parameter_set_id`,
/// the last SPS that names it.
pub(crate) fn vps_sources(sets: &[Sps]) -> Vec<&Sps> {
    let mut sources: Vec<&Sps> = Vec::new();
    for sps in sets.iter().rev() {
        let id = sps.sps_video_parameter_set_id;
        if !sources.iter().any(|s| s.sps_video_parameter_set_id == id) {
            sources.push(sps);
        }
    }
    sources
}

/// The parameter sets a session parameters object holds, and the object.
pub(crate) struct ParameterSets {
    sps: Vec<Sps>,
    pps: Vec<Pps>,
    handle: Parameters,
}

impl SessionParameters for ParameterSets {
    fn handle(&self) -> vk::VideoSessionParametersKHR {
        self.handle.handle()
    }
}

impl ParameterSets {
    /// Whether it holds `sps` and `pps` as they are.
    pub(crate) fn holds(&self, sps: &Sps, pps: &Pps) -> bool {
        self.sps.iter().any(|s| s == sps) && self.pps.iter().any(|p| p == pps)
    }

    /// An object for `session`, of `decoded`, with what `old` holds and
    /// `sps` and `pps` ([`merged`]), and a VPS for each SPS's
    /// `sps_video_parameter_set_id` ([`vps_sources`]).
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        session: vk::VideoSessionKHR,
        decoded: H265Profile,
        old: Option<&Self>,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<Self, Error> {
        let (sets_sps, sets_pps) = match old {
            Some(old) => merged(&old.sps, &old.pps, sps, pps),
            None => merged(&[], &[], sps, pps),
        };
        let std_vps: Vec<StdVps> = vps_sources(&sets_sps)
            .iter()
            .map(|s| StdVps::new(s, decoded))
            .collect();
        let std_sps = sets_sps
            .iter()
            .map(|s| StdSps::new(s, decoded))
            .collect::<Result<Vec<_>, Error>>()?;
        let mut std_pps = Vec::with_capacity(sets_pps.len());
        for p in &sets_pps {
            let id = p.pps_seq_parameter_set_id;
            if let Some(sps) = sets_sps.iter().find(|s| s.sps_seq_parameter_set_id == id) {
                std_pps.push(StdPps::new(p, sps)?);
            }
        }
        let vps_array: Vec<StdVideoH265VideoParameterSet> = std_vps.iter().map(|v| v.vps).collect();
        let sps_array: Vec<StdVideoH265SequenceParameterSet> =
            std_sps.iter().map(|s| s.sps).collect();
        let pps_array: Vec<StdVideoH265PictureParameterSet> =
            std_pps.iter().map(|p| p.pps).collect();
        let add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&vps_array)
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let mut h265 = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(count(vps_array.len()))
            .max_std_sps_count(count(sps_array.len()))
            .max_std_pps_count(count(pps_array.len()))
            .parameters_add_info(&add);
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h265);
        Ok(Self {
            sps: sets_sps,
            pps: sets_pps,
            handle: Parameters::new(device, &info)?,
        })
    }
}
