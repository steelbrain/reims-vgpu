//! The parameter sets as Vulkan Video's std structures
//! (vulkan_video_codec_h264std.h), and the session parameters object that
//! holds them.
//!
//! Scaling lists go as the codec crate resolves them for each set, every
//! list the set's syntax can carry marked present, in zig-zag order: the
//! std lists "correspond to the identically named syntax elements" of
//! 7.3.2.1 and 7.3.2.2 (the Vulkan specification, H.264 decode), which
//! are coded in that order. A resolved list marked present is what the
//! stream's list, or the default or fall-back it stands for, means.

use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoH264PictureParameterSet, StdVideoH264PpsFlags, StdVideoH264ScalingLists,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags,
};
use reims_vgpu_video_codec::h264::pps::Pps;
use reims_vgpu_video_codec::h264::scaling::ScalingLists;
use reims_vgpu_video_codec::h264::sps::Sps;

use crate::device::VideoDevice;
use crate::error::Error;
use crate::params::Parameters;
use crate::profile::std_level;
use crate::session::SessionParameters;
use crate::std_types::zeroed;

/// An SPS as a std structure, with what it points at.
pub(crate) struct StdSps {
    pub(crate) sps: StdVideoH264SequenceParameterSet,
    /// What `sps.pScalingLists` points at: boxed, so that it stays put as
    /// this moves.
    _scaling: Option<Box<StdVideoH264ScalingLists>>,
    /// What `sps.pOffsetForRefFrame` points at.
    _offsets: Vec<i32>,
}

impl StdSps {
    pub(crate) fn new(sps: &Sps) -> Self {
        let bit = u32::from;
        let constraint = |n: u8| u32::from((sps.constraint_flags >> (7 - n)) & 1);
        let mut flags: StdVideoH264SpsFlags = zeroed();
        flags.set_constraint_set0_flag(constraint(0));
        flags.set_constraint_set1_flag(constraint(1));
        flags.set_constraint_set2_flag(constraint(2));
        flags.set_constraint_set3_flag(constraint(3));
        flags.set_constraint_set4_flag(constraint(4));
        flags.set_constraint_set5_flag(constraint(5));
        flags.set_direct_8x8_inference_flag(bit(sps.direct_8x8_inference_flag));
        flags.set_mb_adaptive_frame_field_flag(bit(sps.mb_adaptive_frame_field_flag));
        flags.set_frame_mbs_only_flag(bit(sps.frame_mbs_only_flag));
        flags.set_delta_pic_order_always_zero_flag(bit(sps.delta_pic_order_always_zero_flag));
        flags.set_separate_colour_plane_flag(bit(sps.separate_colour_plane_flag));
        flags.set_gaps_in_frame_num_value_allowed_flag(bit(
            sps.gaps_in_frame_num_value_allowed_flag
        ));
        flags.set_qpprime_y_zero_transform_bypass_flag(bit(
            sps.qpprime_y_zero_transform_bypass_flag
        ));
        flags.set_frame_cropping_flag(bit(sps.frame_crop.iter().any(|&c| c != 0)));
        flags.set_seq_scaling_matrix_present_flag(bit(sps.seq_scaling_matrix_present_flag));
        // The VUI is for display and output order, which the codec crate
        // keeps: decoding needs none of it.
        flags.set_vui_parameters_present_flag(0);

        let scaling = sps.seq_scaling_matrix_present_flag.then(|| {
            Box::new(std_scaling_lists(
                &sps.scaling_lists,
                6 + lists_8x8(sps.chroma_format_idc),
            ))
        });
        let offsets = sps.offset_for_ref_frame.clone();
        let [left, right, top, bottom] = sps.frame_crop;
        let mut out: StdVideoH264SequenceParameterSet = zeroed();
        out.flags = flags;
        out.profile_idc = u32::from(sps.profile_idc);
        out.level_idc = std_level(sps.level_idc);
        out.chroma_format_idc = u32::from(sps.chroma_format_idc);
        out.seq_parameter_set_id = sps.seq_parameter_set_id;
        out.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
        out.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
        out.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4;
        out.pic_order_cnt_type = u32::from(sps.pic_order_cnt_type);
        out.offset_for_non_ref_pic = sps.offset_for_non_ref_pic;
        out.offset_for_top_to_bottom_field = sps.offset_for_top_to_bottom_field;
        out.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
        // At most 255 (7.4.2.1.1), as the codec crate reads it.
        out.num_ref_frames_in_pic_order_cnt_cycle = u8::try_from(offsets.len()).unwrap_or(u8::MAX);
        out.max_num_ref_frames = sps.max_num_ref_frames;
        out.pic_width_in_mbs_minus1 = u32::from(sps.pic_width_in_mbs_minus1);
        out.pic_height_in_map_units_minus1 = u32::from(sps.pic_height_in_map_units_minus1);
        out.frame_crop_left_offset = left;
        out.frame_crop_right_offset = right;
        out.frame_crop_top_offset = top;
        out.frame_crop_bottom_offset = bottom;
        if !offsets.is_empty() {
            out.pOffsetForRefFrame = offsets.as_ptr();
        }
        if let Some(lists) = &scaling {
            out.pScalingLists = std::ptr::from_ref(lists.as_ref());
        }
        Self {
            sps: out,
            _scaling: scaling,
            _offsets: offsets,
        }
    }
}

/// A PPS as a std structure, with what it points at.
pub(crate) struct StdPps {
    pub(crate) pps: StdVideoH264PictureParameterSet,
    /// What `pps.pScalingLists` points at, boxed as [`StdSps`]'s.
    _scaling: Option<Box<StdVideoH264ScalingLists>>,
}

impl StdPps {
    /// `pps`, whose lists fall back to `sps`'s where it leaves one out.
    pub(crate) fn new(pps: &Pps, sps: &Sps) -> Self {
        let bit = u32::from;
        let mut flags: StdVideoH264PpsFlags = zeroed();
        flags.set_transform_8x8_mode_flag(bit(pps.transform_8x8_mode_flag));
        flags.set_redundant_pic_cnt_present_flag(bit(pps.redundant_pic_cnt_present_flag));
        flags.set_constrained_intra_pred_flag(bit(pps.constrained_intra_pred_flag));
        flags.set_deblocking_filter_control_present_flag(bit(
            pps.deblocking_filter_control_present_flag
        ));
        flags.set_weighted_pred_flag(bit(pps.weighted_pred_flag));
        flags.set_bottom_field_pic_order_in_frame_present_flag(bit(
            pps.bottom_field_pic_order_in_frame_present_flag
        ));
        flags.set_entropy_coding_mode_flag(bit(pps.entropy_coding_mode_flag));
        flags.set_pic_scaling_matrix_present_flag(bit(pps.pic_scaling_matrix_present_flag()));

        let scaling = pps.pic_scaling_matrix_present_flag().then(|| {
            let lists_8x8 = if pps.transform_8x8_mode_flag {
                lists_8x8(sps.chroma_format_idc)
            } else {
                0
            };
            Box::new(std_scaling_lists(&pps.scaling_lists(sps), 6 + lists_8x8))
        });
        let mut out: StdVideoH264PictureParameterSet = zeroed();
        out.flags = flags;
        out.seq_parameter_set_id = pps.seq_parameter_set_id;
        out.pic_parameter_set_id = pps.pic_parameter_set_id;
        out.num_ref_idx_l0_default_active_minus1 = pps.num_ref_idx_l0_default_active_minus1;
        out.num_ref_idx_l1_default_active_minus1 = pps.num_ref_idx_l1_default_active_minus1;
        out.weighted_bipred_idc = u32::from(pps.weighted_bipred_idc);
        out.pic_init_qp_minus26 = pps.pic_init_qp_minus26;
        out.pic_init_qs_minus26 = pps.pic_init_qs_minus26;
        out.chroma_qp_index_offset = pps.chroma_qp_index_offset;
        out.second_chroma_qp_index_offset = pps.second_chroma_qp_index_offset;
        if let Some(lists) = &scaling {
            out.pScalingLists = std::ptr::from_ref(lists.as_ref());
        }
        Self {
            pps: out,
            _scaling: scaling,
        }
    }
}

/// The 8x8 lists a set codes: 2, or 6 in 4:4:4 (7.3.2.1.1, 7.3.2.2).
fn lists_8x8(chroma_format_idc: u8) -> usize {
    if chroma_format_idc == 3 { 6 } else { 2 }
}

/// `lists` as std lists, the first `count` (6 4x4, then 8x8) marked
/// present.
fn std_scaling_lists(lists: &ScalingLists, count: usize) -> StdVideoH264ScalingLists {
    let mut out: StdVideoH264ScalingLists = zeroed();
    out.scaling_list_present_mask = (1u16 << count.min(12)).wrapping_sub(1);
    out.use_default_scaling_matrix_mask = 0;
    out.ScalingList4x4 = lists.lists_4x4;
    out.ScalingList8x8 = lists.lists_8x8;
    out
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

    /// An object for `session` with what `old` holds and `sps` and `pps`,
    /// each in place of a set of its id. A PPS of an SPS whose id is
    /// replaced goes, but for `pps`: it was read against the old SPS.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        session: vk::VideoSessionKHR,
        old: Option<&Self>,
        sps: &Sps,
        pps: &Pps,
    ) -> Result<Self, Error> {
        let mut sets_sps: Vec<Sps> = old.map(|o| o.sps.clone()).unwrap_or_default();
        let mut sets_pps: Vec<Pps> = old.map(|o| o.pps.clone()).unwrap_or_default();
        if !sets_sps.iter().any(|s| s == sps) {
            sets_sps.retain(|s| s.seq_parameter_set_id != sps.seq_parameter_set_id);
            sets_pps.retain(|p| p.seq_parameter_set_id != sps.seq_parameter_set_id);
            sets_sps.push(sps.clone());
        }
        sets_pps.retain(|p| p.pic_parameter_set_id != pps.pic_parameter_set_id);
        sets_pps.push(pps.clone());

        let std_sps: Vec<StdSps> = sets_sps.iter().map(StdSps::new).collect();
        let std_pps: Vec<StdPps> = sets_pps
            .iter()
            .filter_map(|p| {
                let sps = sets_sps
                    .iter()
                    .find(|s| s.seq_parameter_set_id == p.seq_parameter_set_id)?;
                Some(StdPps::new(p, sps))
            })
            .collect();
        let sps_array: Vec<StdVideoH264SequenceParameterSet> =
            std_sps.iter().map(|s| s.sps).collect();
        let pps_array: Vec<StdVideoH264PictureParameterSet> =
            std_pps.iter().map(|p| p.pps).collect();
        let add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&sps_array)
            .std_pp_ss(&pps_array);
        let count = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let mut h264 = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(count(sps_array.len()))
            .max_std_pps_count(count(pps_array.len()))
            .parameters_add_info(&add);
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h264);
        Ok(Self {
            sps: sets_sps,
            pps: sets_pps,
            handle: Parameters::new(device, &info)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_video_codec::encode::h264 as written;
    use reims_vgpu_video_codec::encode::{Entropy, Profile};

    /// An SPS and a PPS as the codec crate's encoder writes them (High,
    /// 1280x720, level 3.1, 4 references, CABAC, the 8x8 transform), as
    /// the decoder reads them.
    fn sets() -> (Sps, Pps) {
        let sps = written::Sps {
            profile: Profile::H264High,
            level_idc: 31,
            max_num_ref_frames: 4,
            width_in_mbs: 80,
            height_in_mbs: 45,
            crop_right: 0,
            crop_bottom: 0,
            vui: written::Vui::default(),
        };
        let pps = written::Pps {
            profile: Profile::H264High,
            entropy: Entropy::Cabac,
            pic_init_qp: 26,
            transform_8x8_mode_flag: true,
        };
        let sps = Sps::parse(&sps.to_nal().unwrap()).unwrap();
        let sps_arc = Arc::new(sps.clone());
        let pps = Pps::parse(&pps.to_nal().unwrap(), |_| Some(sps_arc.clone())).unwrap();
        (sps, pps)
    }

    #[test]
    fn an_sps_maps_field_by_field() {
        let (sps, _) = sets();
        let std = StdSps::new(&sps);
        assert_eq!(std.sps.profile_idc, 100);
        assert_eq!(crate::profile::level_idc(std.sps.level_idc), 31);
        assert_eq!(std.sps.pic_width_in_mbs_minus1, 79);
        assert_eq!(std.sps.pic_height_in_map_units_minus1, 44);
        assert_eq!(std.sps.flags.frame_mbs_only_flag(), 1);
        assert_eq!(std.sps.flags.vui_parameters_present_flag(), 0);
        assert_eq!(std.sps.chroma_format_idc, 1);
        assert!(std.sps.pScalingLists.is_null());
        assert_eq!(std.sps.max_num_ref_frames, sps.max_num_ref_frames);
    }

    #[test]
    fn scaling_lists_stay_in_zigzag_order_and_are_all_present() {
        let mut lists = ScalingLists {
            lists_4x4: [[16; 16]; 6],
            lists_8x8: [[16; 64]; 6],
        };
        lists.lists_4x4[3] = core::array::from_fn(|k| u8::try_from(k + 1).unwrap());
        let std = std_scaling_lists(&lists, 8);
        assert_eq!(std.scaling_list_present_mask, 0xff);
        assert_eq!(std.use_default_scaling_matrix_mask, 0);
        // As coded: zig-zag index k holds k + 1, unmoved.
        assert_eq!(std.ScalingList4x4[3][2], 3);
        assert_eq!(std_scaling_lists(&lists, 6).scaling_list_present_mask, 0x3f);
        assert_eq!(
            std_scaling_lists(&lists, 12).scaling_list_present_mask,
            0xfff
        );
    }

    #[test]
    fn a_pps_maps_its_flags_and_ids() {
        let (sps, pps) = sets();
        let std = StdPps::new(&pps, &sps);
        assert_eq!(std.pps.pic_parameter_set_id, pps.pic_parameter_set_id);
        assert_eq!(std.pps.seq_parameter_set_id, sps.seq_parameter_set_id);
        assert_eq!(
            std.pps.flags.entropy_coding_mode_flag(),
            u32::from(pps.entropy_coding_mode_flag)
        );
        assert_eq!(
            std.pps.flags.transform_8x8_mode_flag(),
            u32::from(pps.transform_8x8_mode_flag)
        );
        assert!(std.pps.pScalingLists.is_null());
    }
}
