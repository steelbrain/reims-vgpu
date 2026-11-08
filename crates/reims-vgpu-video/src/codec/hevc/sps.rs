//! The sequence parameter set (H.265 7.3.2.2), its `profile_tier_level()`
//! (7.3.3) and its VUI (E.2.1), as far as decoding and output order need
//! them.

use super::Error;
use super::bits::BitReader;
use super::nal::{NalUnit, payload_reader};
use super::rps::ShortTermRps;
use super::scaling::ScalingLists;
use crate::codec::num::ToUsize;
use std::num::NonZeroU32;

/// The general profile, tier and level (7.3.3), as the SPS codes them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfileTierLevel {
    /// `general_profile_space`.
    pub(crate) general_profile_space: u8,
    /// `general_tier_flag`.
    pub general_tier_flag: bool,
    /// `general_profile_idc`: 1 Main, 2 Main 10, 3 Main Still Picture, 4
    /// the format range extensions, 9 screen content coding (A.3).
    pub general_profile_idc: u8,
    /// `general_profile_compatibility_flag[j]`, flag `j` in bit `31 - j`.
    pub(crate) general_profile_compatibility_flags: u32,
    /// `general_level_idc`: 30 times the level (153 is level 5.1).
    pub general_level_idc: u8,
}

impl ProfileTierLevel {
    /// `general_profile_compatibility_flag[profile]`.
    pub(crate) fn compatible_with(&self, profile: u8) -> bool {
        // Reversed, flag j is bit j; there is no flag past 31.
        self.general_profile_compatibility_flags
            .reverse_bits()
            .checked_shr(u32::from(profile))
            .is_some_and(|flags| flags & 1 != 0)
    }

    /// `profile_tier_level(1, maxNumSubLayersMinus1)`.
    fn parse(r: &mut BitReader<'_>, max_sub_layers_minus1: u8) -> Result<Self, Error> {
        let general_profile_space = r.u8(2)?;
        let general_tier_flag = r.bit()?;
        let general_profile_idc = r.u8(5)?;
        let general_profile_compatibility_flags = r.bits(32)?;
        // The four source and constraint flags, then 43 bits of further
        // constraints and one of general_inbld_flag or reserved.
        r.skip(48)?;
        let general_level_idc = r.u8(8)?;
        // sub_layer_profile_present_flag[i] and sub_layer_level_present_flag[i].
        let mut present = Vec::with_capacity(usize::from(max_sub_layers_minus1));
        for _ in 0..max_sub_layers_minus1 {
            present.push((r.bit()?, r.bit()?));
        }
        if max_sub_layers_minus1 > 0 {
            for _ in max_sub_layers_minus1..8 {
                r.skip(2)?; // reserved_zero_2bits
            }
        }
        for (profile_present, level_present) in present {
            if profile_present {
                r.skip(88)?;
            }
            if level_present {
                r.skip(8)?;
            }
        }
        Ok(Self {
            general_profile_space,
            general_tier_flag,
            general_profile_idc,
            general_profile_compatibility_flags,
            general_level_idc,
        })
    }
}

/// `sps_max_dec_pic_buffering_minus1`, `sps_max_num_reorder_pics` and
/// `sps_max_latency_increase_plus1` of one sub-layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubLayerOrdering {
    /// `sps_max_dec_pic_buffering_minus1`.
    pub max_dec_pic_buffering_minus1: u8,
    /// `sps_max_num_reorder_pics`.
    pub max_num_reorder_pics: u8,
    /// `sps_max_latency_increase_plus1`.
    pub max_latency_increase_plus1: u32,
}

/// The PCM sample parameters, when `pcm_enabled_flag`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pcm {
    /// `pcm_sample_bit_depth_luma_minus1`.
    pub sample_bit_depth_luma_minus1: u8,
    /// `pcm_sample_bit_depth_chroma_minus1`.
    pub sample_bit_depth_chroma_minus1: u8,
    /// `log2_min_pcm_luma_coding_block_size_minus3`.
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_pcm_luma_coding_block_size`.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// `pcm_loop_filter_disabled_flag`.
    pub loop_filter_disabled_flag: bool,
}

/// What this decoder keeps of the VUI (E.2.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vui {
    /// `video_full_range_flag`.
    pub video_full_range_flag: bool,
    /// `colour_primaries`, `transfer_characteristics`, `matrix_coeffs`.
    pub colour_description: Option<(u8, u8, u8)>,
    /// `chroma_sample_loc_type_top_field` and `_bottom_field`, as coded:
    /// 0 to 5 in a conformant stream (E.3.1). Kept for what a stream says,
    /// not used in decoding.
    pub chroma_sample_loc: Option<(u32, u32)>,
    /// `field_seq_flag`: the pictures are fields.
    pub field_seq_flag: bool,
    /// `vui_num_units_in_tick` and `vui_time_scale`.
    pub(crate) timing: Option<(u32, u32)>,
}

/// A sequence parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// `sps_video_parameter_set_id`.
    pub sps_video_parameter_set_id: u8,
    /// `sps_max_sub_layers_minus1`.
    pub sps_max_sub_layers_minus1: u8,
    /// `sps_temporal_id_nesting_flag`.
    pub sps_temporal_id_nesting_flag: bool,
    /// The general profile, tier and level.
    pub profile_tier_level: ProfileTierLevel,
    /// `sps_seq_parameter_set_id`.
    pub sps_seq_parameter_set_id: u8,
    /// `chroma_format_idc`: 0 monochrome, 1 4:2:0, 2 4:2:2, 3 4:4:4.
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`.
    pub separate_colour_plane_flag: bool,
    /// `pic_width_in_luma_samples`.
    pub pic_width_in_luma_samples: u32,
    /// `pic_height_in_luma_samples`.
    pub pic_height_in_luma_samples: u32,
    /// `conf_win_{left,right,top,bottom}_offset`, in chroma samples.
    pub conformance_window: [u32; 4],
    /// `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8`.
    pub bit_depth_chroma_minus8: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4`.
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// Each sub-layer's ordering limits, those `sps_sub_layer_ordering_info_present_flag`
    /// leaves out inferred from the highest's.
    pub sub_layer_ordering: Vec<SubLayerOrdering>,
    /// `log2_min_luma_coding_block_size_minus3`.
    pub log2_min_luma_coding_block_size_minus3: u8,
    /// `log2_diff_max_min_luma_coding_block_size`.
    pub log2_diff_max_min_luma_coding_block_size: u8,
    /// `log2_min_luma_transform_block_size_minus2`.
    pub log2_min_luma_transform_block_size_minus2: u8,
    /// `log2_diff_max_min_luma_transform_block_size`.
    pub log2_diff_max_min_luma_transform_block_size: u8,
    /// `max_transform_hierarchy_depth_inter`.
    pub max_transform_hierarchy_depth_inter: u8,
    /// `max_transform_hierarchy_depth_intra`.
    pub max_transform_hierarchy_depth_intra: u8,
    /// `scaling_list_enabled_flag`.
    pub scaling_list_enabled_flag: bool,
    /// The SPS's own lists, when `sps_scaling_list_data_present_flag`.
    pub scaling_lists: Option<ScalingLists>,
    /// `amp_enabled_flag`.
    pub amp_enabled_flag: bool,
    /// `sample_adaptive_offset_enabled_flag`.
    pub sample_adaptive_offset_enabled_flag: bool,
    /// The PCM parameters, when `pcm_enabled_flag`.
    pub pcm: Option<Pcm>,
    /// The `num_short_term_ref_pic_sets` sets.
    pub short_term_ref_pic_sets: Vec<ShortTermRps>,
    /// `long_term_ref_pics_present_flag`.
    pub long_term_ref_pics_present_flag: bool,
    /// `lt_ref_pic_poc_lsb_sps[]` with `used_by_curr_pic_lt_sps_flag[]`.
    pub long_term_ref_pics: Vec<(u32, bool)>,
    /// `sps_temporal_mvp_enabled_flag`.
    pub sps_temporal_mvp_enabled_flag: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing_enabled_flag: bool,
    /// The VUI, when present and readable.
    pub vui: Option<Vui>,
    /// `sps_range_extension_flag`.
    pub(crate) sps_range_extension_flag: bool,
    /// `sps_multilayer_extension_flag`.
    pub(crate) sps_multilayer_extension_flag: bool,
    /// `sps_3d_extension_flag`.
    pub(crate) sps_3d_extension_flag: bool,
    /// `sps_scc_extension_flag`.
    pub(crate) sps_scc_extension_flag: bool,
    /// What decoding derives from the fields above.
    pub(crate) derived: Derived,
}

/// The variables 7.4.3.2.1 derives from an SPS's syntax elements that the
/// decoder reads, computed once, when the SPS is parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Derived {
    /// `BitDepthY` (7-4).
    bit_depth_luma: u8,
    /// `QpBdOffsetY` (7-5).
    qp_bd_offset_y: i32,
    /// `CtbLog2SizeY` (7-11).
    ctb_log2_size: u32,
    /// `PicWidthInCtbsY` and `PicHeightInCtbsY` (7-15, 7-17).
    pic_size_in_ctbs: (u32, u32),
    /// `PicSizeInCtbsY` (7-19).
    pic_ctbs: u32,
    /// `MaxPicOrderCntLsb` (7-8).
    max_pic_order_cnt_lsb: NonZeroU32,
    /// The conformance window: x, y, width and height in luma samples.
    crop_rect: (u32, u32, u32, u32),
}

impl Derived {
    /// The variables of an SPS of this `bit_depth_luma_minus8`, coding
    /// block sizes (`MinCbLog2SizeY` and `CtbLog2SizeY`),
    /// picture size in luma samples, `log2_max_pic_order_cnt_lsb_minus4`,
    /// chroma subsampling (`SubWidthC`, `SubHeightC`) and conformance
    /// window (left, right, top and bottom, in chroma samples). It refuses
    /// a picture that is not whole coding blocks, and a window that leaves
    /// no samples (7.4.3.2.1).
    fn of(
        bit_depth_luma_minus8: u8,
        (min_cb_log2, ctb_log2): (u8, u8),
        (width, height): (u32, u32),
        log2_max_pic_order_cnt_lsb_minus4: u8,
        (sub_width, sub_height): (u32, u32),
        [left, right, top, bottom]: [u32; 4],
    ) -> Result<Self, Error> {
        let overflow = Error::Overflow;
        let bit_depth_luma = bit_depth_luma_minus8
            .checked_add(8)
            .ok_or(overflow("BitDepthY"))?;
        let qp_bd_offset_y = i32::from(bit_depth_luma_minus8)
            .checked_mul(6)
            .ok_or(overflow("QpBdOffsetY"))?;
        let min_cb_size = 1u32
            .checked_shl(u32::from(min_cb_log2))
            .ok_or(overflow("MinCbSizeY"))?;
        if width == 0
            || height == 0
            || !width.is_multiple_of(min_cb_size)
            || !height.is_multiple_of(min_cb_size)
        {
            return Err(Error::OutOfRange("pic_width_in_luma_samples"));
        }
        let ctb_size = 1u32
            .checked_shl(u32::from(ctb_log2))
            .ok_or(overflow("CtbSizeY"))?;
        let width_in_ctbs = width.div_ceil(ctb_size);
        let height_in_ctbs = height.div_ceil(ctb_size);
        let pic_ctbs = width_in_ctbs
            .checked_mul(height_in_ctbs)
            .ok_or(overflow("PicSizeInCtbsY"))?;
        let max_pic_order_cnt_lsb = 1u32
            .checked_shl(u32::from(log2_max_pic_order_cnt_lsb_minus4) + 4)
            .and_then(NonZeroU32::new)
            .ok_or(overflow("MaxPicOrderCntLsb"))?;
        // The window's offsets, in chroma samples, leave samples each way.
        let cropped = |size: u32, unit: u32, before: u32, after: u32| {
            let offset = unit.checked_mul(before)?;
            let both = offset.checked_add(unit.checked_mul(after)?)?;
            Some((offset, size.checked_sub(both).filter(|&left| left > 0)?))
        };
        let ((x, crop_width), (y, crop_height)) = cropped(width, sub_width, left, right)
            .zip(cropped(height, sub_height, top, bottom))
            .ok_or(Error::OutOfRange("conformance window"))?;
        Ok(Self {
            bit_depth_luma,
            qp_bd_offset_y,
            ctb_log2_size: u32::from(ctb_log2),
            pic_size_in_ctbs: (width_in_ctbs, height_in_ctbs),
            pic_ctbs,
            max_pic_order_cnt_lsb,
            crop_rect: (x, y, crop_width, crop_height),
        })
    }
}

impl Sps {
    /// Parses an SPS NAL unit, header included.
    pub fn parse(unit: &[u8]) -> Result<Self, Error> {
        let nal = NalUnit::new(unit)?;
        let mut r = payload_reader(&nal)?;
        let sps_video_parameter_set_id = r.u8(4)?;
        let sps_max_sub_layers_minus1 = r.u8(3)?;
        if sps_max_sub_layers_minus1 > 6 {
            return Err(Error::OutOfRange("sps_max_sub_layers_minus1"));
        }
        let sps_temporal_id_nesting_flag = r.bit()?;
        let profile_tier_level = ProfileTierLevel::parse(&mut r, sps_max_sub_layers_minus1)?;
        let sps_seq_parameter_set_id = r.ue_u8(15, "sps_seq_parameter_set_id")?;
        let chroma_format_idc = r.ue_u8(3, "chroma_format_idc")?;
        let separate_colour_plane_flag = chroma_format_idc == 3 && r.bit()?;
        let pic_width_in_luma_samples = r.ue_max(16888, "pic_width_in_luma_samples")?;
        let pic_height_in_luma_samples = r.ue_max(16888, "pic_height_in_luma_samples")?;
        let mut conformance_window = [0; 4];
        if r.bit()? {
            for offset in &mut conformance_window {
                *offset = r.ue_max(16888, "conf_win_offset")?;
            }
        }
        let bit_depth_luma_minus8 = r.ue_u8(8, "bit_depth_luma_minus8")?;
        let bit_depth_chroma_minus8 = r.ue_u8(8, "bit_depth_chroma_minus8")?;
        let log2_max_pic_order_cnt_lsb_minus4 = r.ue_u8(12, "log2_max_pic_order_cnt_lsb_minus4")?;
        let ordering_info_present = r.bit()?;
        // Each sub-layer's limits, or the highest's alone, which then hold
        // for every sub-layer (7.4.3.2.1).
        let layers = usize::from(sps_max_sub_layers_minus1) + 1;
        let coded = if ordering_info_present { layers } else { 1 };
        let mut sub_layer_ordering = Vec::with_capacity(layers);
        let mut highest = SubLayerOrdering::default();
        for _ in 0..coded {
            let max_dec_pic_buffering_minus1 = r.ue_u8(15, "sps_max_dec_pic_buffering_minus1")?;
            highest = SubLayerOrdering {
                max_dec_pic_buffering_minus1,
                max_num_reorder_pics: r
                    .ue_u8(max_dec_pic_buffering_minus1, "sps_max_num_reorder_pics")?,
                max_latency_increase_plus1: r.ue()?,
            };
            sub_layer_ordering.push(highest);
        }
        sub_layer_ordering.resize(layers, highest);
        let log2_min_luma_coding_block_size_minus3 =
            r.ue_u8(3, "log2_min_luma_coding_block_size_minus3")?;
        let log2_diff_max_min_luma_coding_block_size =
            r.ue_u8(3, "log2_diff_max_min_luma_coding_block_size")?;
        // MinCbLog2SizeY and CtbLog2SizeY (7-10, 7-11), the CTB 64 at most
        // (7.4.3.2.1).
        let min_cb_log2 = log2_min_luma_coding_block_size_minus3
            .checked_add(3)
            .ok_or(Error::Overflow("MinCbLog2SizeY"))?;
        let ctb_log2 = min_cb_log2
            .checked_add(log2_diff_max_min_luma_coding_block_size)
            .filter(|&log2| log2 <= 6)
            .ok_or(Error::OutOfRange("CtbLog2SizeY"))?;
        let log2_min_luma_transform_block_size_minus2 =
            r.ue_u8(3, "log2_min_luma_transform_block_size_minus2")?;
        let log2_diff_max_min_luma_transform_block_size =
            r.ue_u8(3, "log2_diff_max_min_luma_transform_block_size")?;
        // 7.4.3.2.1: transform blocks smaller than the smallest coding
        // block and no larger than the CTB or 32, split no deeper than the
        // CTB takes to reach the smallest.
        let min_tb_log2 = log2_min_luma_transform_block_size_minus2
            .checked_add(2)
            .ok_or(Error::Overflow("MinTbLog2SizeY"))?;
        let max_tb_log2 = min_tb_log2
            .checked_add(log2_diff_max_min_luma_transform_block_size)
            .ok_or(Error::Overflow("MaxTbLog2SizeY"))?;
        if min_tb_log2 >= min_cb_log2 {
            return Err(Error::OutOfRange("MinTbLog2SizeY"));
        }
        if max_tb_log2 > ctb_log2.min(5) {
            return Err(Error::OutOfRange("MaxTbLog2SizeY"));
        }
        let max_depth = ctb_log2
            .checked_sub(min_tb_log2)
            .ok_or(Error::OutOfRange("MinTbLog2SizeY"))?;
        let max_transform_hierarchy_depth_inter =
            r.ue_u8(max_depth, "max_transform_hierarchy_depth_inter")?;
        let max_transform_hierarchy_depth_intra =
            r.ue_u8(max_depth, "max_transform_hierarchy_depth_intra")?;
        let scaling_list_enabled_flag = r.bit()?;
        let scaling_lists = if scaling_list_enabled_flag && r.bit()? {
            Some(ScalingLists::parse(&mut r)?)
        } else {
            None
        };
        let amp_enabled_flag = r.bit()?;
        let sample_adaptive_offset_enabled_flag = r.bit()?;
        let pcm = if r.bit()? {
            let pcm = Pcm {
                sample_bit_depth_luma_minus1: r.u8(4)?,
                sample_bit_depth_chroma_minus1: r.u8(4)?,
                log2_min_luma_coding_block_size_minus3: r
                    .ue_u8(2, "log2_min_pcm_luma_coding_block_size_minus3")?,
                log2_diff_max_min_luma_coding_block_size: r
                    .ue_u8(2, "log2_diff_max_min_pcm_luma_coding_block_size")?,
                loop_filter_disabled_flag: r.bit()?,
            };
            check_pcm(
                &pcm,
                bit_depth_luma_minus8,
                bit_depth_chroma_minus8,
                min_cb_log2,
                ctb_log2,
            )?;
            Some(pcm)
        } else {
            None
        };
        let set_count = r.ue_max(64, "num_short_term_ref_pic_sets")?.to_usize();
        let mut short_term_ref_pic_sets = Vec::with_capacity(set_count);
        let max_pictures = u32::from(highest.max_dec_pic_buffering_minus1);
        for _ in 0..set_count {
            let set = ShortTermRps::parse(&mut r, &short_term_ref_pic_sets, false, max_pictures)?;
            short_term_ref_pic_sets.push(set);
        }
        let long_term_ref_pics_present_flag = r.bit()?;
        let mut long_term_ref_pics = Vec::new();
        if long_term_ref_pics_present_flag {
            let count = r.ue_max(32, "num_long_term_ref_pics_sps")?;
            let lsb_bits = u32::from(log2_max_pic_order_cnt_lsb_minus4) + 4;
            for _ in 0..count {
                long_term_ref_pics.push((r.bits(lsb_bits)?, r.bit()?));
            }
        }
        let sps_temporal_mvp_enabled_flag = r.bit()?;
        let strong_intra_smoothing_enabled_flag = r.bit()?;
        let derived = Derived::of(
            bit_depth_luma_minus8,
            (min_cb_log2, ctb_log2),
            (pic_width_in_luma_samples, pic_height_in_luma_samples),
            log2_max_pic_order_cnt_lsb_minus4,
            subsampling(chroma_array_type(
                chroma_format_idc,
                separate_colour_plane_flag,
            )),
            conformance_window,
        )?;
        let mut sps = Self {
            sps_video_parameter_set_id,
            sps_max_sub_layers_minus1,
            sps_temporal_id_nesting_flag,
            profile_tier_level,
            sps_seq_parameter_set_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples,
            pic_height_in_luma_samples,
            conformance_window,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            log2_max_pic_order_cnt_lsb_minus4,
            sub_layer_ordering,
            log2_min_luma_coding_block_size_minus3,
            log2_diff_max_min_luma_coding_block_size,
            log2_min_luma_transform_block_size_minus2,
            log2_diff_max_min_luma_transform_block_size,
            max_transform_hierarchy_depth_inter,
            max_transform_hierarchy_depth_intra,
            scaling_list_enabled_flag,
            scaling_lists,
            amp_enabled_flag,
            sample_adaptive_offset_enabled_flag,
            pcm,
            short_term_ref_pic_sets,
            long_term_ref_pics_present_flag,
            long_term_ref_pics,
            sps_temporal_mvp_enabled_flag,
            strong_intra_smoothing_enabled_flag,
            vui: None,
            sps_range_extension_flag: false,
            sps_multilayer_extension_flag: false,
            sps_3d_extension_flag: false,
            sps_scc_extension_flag: false,
            derived,
        };
        if r.bit()? {
            // A VUI that does not read (one cut short, say) is left out,
            // as are the extension flags after it: nothing decoding needs
            // is in either but the range, which then defaults to video.
            match parse_vui(&mut r, sps_max_sub_layers_minus1) {
                Ok(vui) => sps.vui = Some(vui),
                Err(_) => return Ok(sps),
            }
        }
        if r.bit()? {
            sps.sps_range_extension_flag = r.bit()?;
            sps.sps_multilayer_extension_flag = r.bit()?;
            sps.sps_3d_extension_flag = r.bit()?;
            sps.sps_scc_extension_flag = r.bit()?;
        }
        Ok(sps)
    }

    /// `BitDepthY`.
    pub(crate) fn bit_depth_luma(&self) -> u8 {
        self.derived.bit_depth_luma
    }

    /// `QpBdOffsetY`.
    pub(crate) fn qp_bd_offset_y(&self) -> i32 {
        self.derived.qp_bd_offset_y
    }

    /// `CtbLog2SizeY`.
    pub fn ctb_log2_size(&self) -> u32 {
        self.derived.ctb_log2_size
    }

    /// `PicWidthInCtbsY`.
    pub(crate) fn pic_width_in_ctbs(&self) -> u32 {
        self.derived.pic_size_in_ctbs.0
    }

    /// `PicHeightInCtbsY`.
    pub(crate) fn pic_height_in_ctbs(&self) -> u32 {
        self.derived.pic_size_in_ctbs.1
    }

    /// `PicSizeInCtbsY`.
    pub(crate) fn pic_size_in_ctbs(&self) -> u32 {
        self.derived.pic_ctbs
    }

    /// `MaxPicOrderCntLsb`.
    pub(crate) fn max_pic_order_cnt_lsb(&self) -> NonZeroU32 {
        self.derived.max_pic_order_cnt_lsb
    }

    /// `ChromaArrayType`: `chroma_format_idc`, or 0 for separate planes.
    pub(crate) fn chroma_array_type(&self) -> u8 {
        chroma_array_type(self.chroma_format_idc, self.separate_colour_plane_flag)
    }

    /// The highest sub-layer's ordering limits, which the decoder keeps
    /// (`HighestTid` is the highest sub-layer: all of them are decoded).
    pub fn highest_ordering(&self) -> SubLayerOrdering {
        self.sub_layer_ordering.last().copied().unwrap_or_default()
    }

    /// The DPB's size in pictures: `sps_max_dec_pic_buffering_minus1 + 1`.
    pub fn dpb_size(&self) -> usize {
        usize::from(self.highest_ordering().max_dec_pic_buffering_minus1) + 1
    }

    /// `SpsMaxLatencyPictures`, when `sps_max_latency_increase_plus1` sets
    /// one (7-9).
    pub(crate) fn max_latency_pictures(&self) -> Option<u32> {
        let ordering = self.highest_ordering();
        // The increase may be up to 2^32 - 2 (7.4.3.2.1), so the sum may
        // pass 32 bits. It saturates, as the pictures' PicLatencyCount
        // does (`Decoder`): the two compare as the standard's do until a
        // picture has waited 2^32 - 1 pictures.
        ordering
            .max_latency_increase_plus1
            .checked_sub(1)
            .map(|increase| u32::from(ordering.max_num_reorder_pics).saturating_add(increase))
    }

    /// The conformance window: x, y, width and height in luma samples.
    pub fn crop_rect(&self) -> (u32, u32, u32, u32) {
        self.derived.crop_rect
    }

    /// Whether the samples use the full range: the VUI's
    /// `video_full_range_flag`, false without a VUI.
    pub fn is_full_range(&self) -> bool {
        self.vui.as_ref().is_some_and(|v| v.video_full_range_flag)
    }
}

/// `ChromaArrayType`: `chroma_format_idc`, or 0 for separate planes.
fn chroma_array_type(chroma_format_idc: u8, separate_colour_plane_flag: bool) -> u8 {
    if separate_colour_plane_flag {
        0
    } else {
        chroma_format_idc
    }
}

/// `SubWidthC` and `SubHeightC` (Table 6-1) of a `ChromaArrayType`.
fn subsampling(chroma_array_type: u8) -> (u32, u32) {
    match chroma_array_type {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    }
}

/// 7.4.3.2.1's PCM bounds: samples of no more bits than the picture's,
/// in blocks from the smallest coding block, or 32, to the CTB, or 32.
fn check_pcm(
    pcm: &Pcm,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    min_cb_log2: u8,
    ctb_log2: u8,
) -> Result<(), Error> {
    if u32::from(pcm.sample_bit_depth_luma_minus1) + 1 > u32::from(bit_depth_luma_minus8) + 8 {
        return Err(Error::OutOfRange("PcmBitDepthY"));
    }
    if u32::from(pcm.sample_bit_depth_chroma_minus1) + 1 > u32::from(bit_depth_chroma_minus8) + 8 {
        return Err(Error::OutOfRange("PcmBitDepthC"));
    }
    let min_log2 = pcm
        .log2_min_luma_coding_block_size_minus3
        .checked_add(3)
        .ok_or(Error::Overflow("Log2MinIpcmCbSizeY"))?;
    let max_log2 = min_log2
        .checked_add(pcm.log2_diff_max_min_luma_coding_block_size)
        .ok_or(Error::Overflow("Log2MaxIpcmCbSizeY"))?;
    if !(min_cb_log2.min(5)..=ctb_log2.min(5)).contains(&min_log2) {
        return Err(Error::OutOfRange("Log2MinIpcmCbSizeY"));
    }
    if max_log2 > ctb_log2.min(5) {
        return Err(Error::OutOfRange("Log2MaxIpcmCbSizeY"));
    }
    Ok(())
}

/// `vui_parameters()` (E.2.1), read through to its end so that the SPS's
/// extension flags after it can be read.
fn parse_vui(r: &mut BitReader<'_>, max_sub_layers_minus1: u8) -> Result<Vui, Error> {
    let mut vui = Vui::default();
    if r.bit()? {
        // aspect_ratio_idc; EXTENDED_SAR codes the ratio.
        if r.u8(8)? == 255 {
            r.skip(32)?;
        }
    }
    if r.bit()? {
        r.skip(1)?; // overscan_appropriate_flag
    }
    if r.bit()? {
        r.skip(3)?; // video_format
        vui.video_full_range_flag = r.bit()?;
        if r.bit()? {
            vui.colour_description = Some((r.u8(8)?, r.u8(8)?, r.u8(8)?));
        }
    }
    if r.bit()? {
        vui.chroma_sample_loc = Some((r.ue()?, r.ue()?));
    }
    r.skip(1)?; // neutral_chroma_indication_flag
    vui.field_seq_flag = r.bit()?;
    r.skip(1)?; // frame_field_info_present_flag
    if r.bit()? {
        // def_disp_win_{left,right,top,bottom}_offset
        for _ in 0..4 {
            r.ue()?;
        }
    }
    if r.bit()? {
        let units = r.bits(32)?;
        let scale = r.bits(32)?;
        vui.timing = Some((units, scale));
        if r.bit()? {
            r.ue()?; // vui_num_ticks_poc_diff_one_minus1
        }
        if r.bit()? {
            skip_hrd_parameters(r, true, max_sub_layers_minus1)?;
        }
    }
    if r.bit()? {
        // bitstream_restriction_flag: three flags, then five ue(v).
        r.skip(3)?;
        for _ in 0..5 {
            r.ue()?;
        }
    }
    Ok(vui)
}

/// `hrd_parameters(commonInfPresentFlag, maxNumSubLayersMinus1)` (E.2.2),
/// read and dropped.
fn skip_hrd_parameters(
    r: &mut BitReader<'_>,
    common_info_present: bool,
    max_sub_layers_minus1: u8,
) -> Result<(), Error> {
    let mut nal_hrd = false;
    let mut vcl_hrd = false;
    let mut sub_pic_hrd_params = false;
    if common_info_present {
        nal_hrd = r.bit()?;
        vcl_hrd = r.bit()?;
        if nal_hrd || vcl_hrd {
            sub_pic_hrd_params = r.bit()?;
            if sub_pic_hrd_params {
                // tick_divisor_minus2, du_cpb_removal_delay_increment_length_minus1,
                // sub_pic_cpb_params_in_pic_timing_sei_flag,
                // dpb_output_delay_du_length_minus1
                r.skip(8 + 5 + 1 + 5)?;
            }
            r.skip(4 + 4)?; // bit_rate_scale, cpb_size_scale
            if sub_pic_hrd_params {
                r.skip(4)?; // cpb_size_du_scale
            }
            // initial_cpb_removal_delay_length_minus1,
            // au_cpb_removal_delay_length_minus1,
            // dpb_output_delay_length_minus1
            r.skip(5 + 5 + 5)?;
        }
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_pic_rate_general = r.bit()?;
        let fixed_pic_rate_within_cvs = fixed_pic_rate_general || r.bit()?;
        let mut low_delay = false;
        if fixed_pic_rate_within_cvs {
            r.ue()?; // elemental_duration_in_tc_minus1
        } else {
            low_delay = r.bit()?;
        }
        let cpb_cnt_minus1 = if low_delay {
            0
        } else {
            r.ue_max(31, "cpb_cnt_minus1")?
        };
        for present in [nal_hrd, vcl_hrd] {
            if present {
                // sub_layer_hrd_parameters(): per CPB, bit_rate_value_minus1
                // and cpb_size_value_minus1, the two du values with
                // sub-picture parameters, then cbr_flag.
                for _ in 0..=cpb_cnt_minus1 {
                    r.ue()?;
                    r.ue()?;
                    if sub_pic_hrd_params {
                        r.ue()?;
                        r.ue()?;
                    }
                    r.skip(1)?;
                }
            }
        }
    }
    Ok(())
}
