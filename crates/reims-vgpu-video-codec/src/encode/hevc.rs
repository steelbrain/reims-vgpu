//! HEVC's headers as the encoder writes them: the VPS (7.3.2.1), the SPS
//! with its VUI (7.3.2.2, E.2.1), the PPS (7.3.2.3), the slice segment
//! header (7.3.6.1), `profile_tier_level()` (7.3.3), the `hvcC` record
//! (ISO/IEC 14496-15, 8.3.3.1), and the level from Tables A.8 and A.9 (Main
//! tier).
//!
//! What the encoder writes is a fixed shape: one temporal layer, one slice
//! segment a picture, the reference picture set in each slice header, no
//! long-term references, tiles, wavefronts, scaling lists or weighted
//! prediction. The coding tools the driver may use are its to say
//! ([`Tools`]), so the SPS and PPS follow what it reports.

use super::bits::BitWriter;
use super::nal;
use super::{ColourDescription, Profile, Unsupported};
use crate::hevc::bits::BitReader;

/// `nal_unit_type`s the encoder writes or strips (Table 7-1).
pub mod nal_type {
    /// A trailing picture others may predict from.
    pub(crate) const TRAIL_R: u8 = 1;
    /// An IDR picture that may have leading pictures.
    pub(crate) const IDR_W_RADL: u8 = 19;
    /// A video parameter set.
    pub(crate) const VPS: u8 = 32;
    /// A sequence parameter set.
    pub(crate) const SPS: u8 = 33;
    /// A picture parameter set.
    pub(crate) const PPS: u8 = 34;
    /// An access unit delimiter.
    pub(crate) const AUD: u8 = 35;
}

/// `log2_max_pic_order_cnt_lsb_minus4`: the order count's LSBs wrap at 256.
pub(crate) const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u8 = 4;

/// The coding tools and block sizes the SPS and PPS enable: what the
/// driver allows (`VAConfigAttribEncHEVCFeatures` and `BlockSizes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tools {
    /// `log2_min_luma_coding_block_size_minus3 + 3`.
    pub log2_min_cb_size: u8,
    /// The CTB size's log2: the minimum coding block's plus `log2_diff_max_min_luma_coding_block_size`.
    pub log2_ctb_size: u8,
    /// `log2_min_luma_transform_block_size_minus2 + 2`.
    pub log2_min_tb_size: u8,
    /// The largest transform block's log2.
    pub log2_max_tb_size: u8,
    /// `max_transform_hierarchy_depth_inter`.
    pub max_transform_hierarchy_depth_inter: u8,
    /// `max_transform_hierarchy_depth_intra`.
    pub max_transform_hierarchy_depth_intra: u8,
    /// `amp_enabled_flag`: asymmetric motion partitions.
    pub amp: bool,
    /// `sample_adaptive_offset_enabled_flag`.
    pub sao: bool,
    /// `strong_intra_smoothing_enabled_flag`.
    pub strong_intra_smoothing: bool,
    /// `transform_skip_enabled_flag`.
    pub transform_skip: bool,
    /// `cu_qp_delta_enabled_flag`.
    pub cu_qp_delta: bool,
    /// Whether the driver requires `cu_qp_delta_enabled_flag`, so that a
    /// constant QP keeps it too.
    pub cu_qp_delta_required: bool,
    /// `constrained_intra_pred_flag`.
    pub constrained_intra_pred: bool,
}

/// The VUI the encoder writes (E.2.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vui {
    /// `aspect_ratio_idc` 255's `sar_width` and `sar_height`, or none written.
    pub sample_aspect_ratio: Option<(u16, u16)>,
    /// `video_full_range_flag`.
    pub(crate) full_range: bool,
    /// `colour_primaries`, `transfer_characteristics` and `matrix_coefficients`, or none written.
    pub(crate) colour: Option<ColourDescription>,
    /// `vui_num_units_in_tick` and `vui_time_scale`: one tick a frame.
    pub timing: Option<(u32, u32)>,
}

/// What the VPS and SPS say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// The profile, and with it `general_profile_idc` and the compatibility flags.
    pub profile: Profile,
    /// `general_level_idc`: 30 times the level, 93 for 3.1.
    pub level_idc: u8,
    /// The coded size: multiples of the smallest coding block.
    pub width: u32,
    /// The coded height, a multiple of the smallest coding block.
    pub height: u32,
    /// Luma samples the conformance window crops off the right and the
    /// bottom: even.
    pub(crate) crop_right: u32,
    /// Luma samples the conformance window crops off the bottom: even.
    pub(crate) crop_bottom: u32,
    /// `sps_max_dec_pic_buffering_minus1`: the pictures the DPB holds,
    /// less one.
    pub(crate) max_dec_pic_buffering_minus1: u32,
    /// `sps_max_num_reorder_pics`: the pictures one may wait behind.
    pub(crate) max_num_reorder_pics: u32,
    /// The coding tools and block sizes.
    pub tools: Tools,
    /// The VUI.
    pub vui: Vui,
}

/// The two-byte NAL unit header (7.3.1.2): layer 0, temporal ID 0.
fn header(unit_type: u8) -> [u8; 2] {
    [unit_type << 1, 1]
}

impl Sps {
    /// `general_profile_compatibility_flag[j]` as hvcC's 32 bits, flag 0
    /// first: Main streams are Main 10 streams too (A.3.2, A.3.3).
    pub(crate) fn compatibility_flags(&self) -> u32 {
        let flag = |j: u32| 0x8000_0000u32 >> j;
        match self.profile {
            Profile::HevcMain => flag(1) | flag(2),
            _ => flag(2),
        }
    }

    fn write_profile_tier_level(&self, w: &mut BitWriter) {
        w.u(2, 0); // general_profile_space
        w.flag(false); // general_tier_flag: Main tier
        w.u(5, u32::from(self.profile.profile_idc()));
        w.u(32, self.compatibility_flags());
        w.flag(true); // general_progressive_source_flag
        w.flag(false); // general_interlaced_source_flag
        w.flag(false); // general_non_packed_constraint_flag
        w.flag(true); // general_frame_only_constraint_flag
        // general_reserved_zero_43bits (or Main 10's reserved and
        // one_picture_only bits, all 0), then general_inbld_flag.
        w.bits(44, 0);
        w.u(8, u32::from(self.level_idc));
    }

    fn timing(&self) -> Option<(u32, u32)> {
        self.vui.timing
    }

    /// The VPS NAL unit, escaped, header first.
    pub(crate) fn vps_nal(&self) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.u(4, 0); // vps_video_parameter_set_id
        w.flag(true); // vps_base_layer_internal_flag
        w.flag(true); // vps_base_layer_available_flag
        w.u(6, 0); // vps_max_layers_minus1
        w.u(3, 0); // vps_max_sub_layers_minus1
        w.flag(true); // vps_temporal_id_nesting_flag
        w.u(16, 0xffff); // vps_reserved_0xffff_16bits
        self.write_profile_tier_level(&mut w);
        w.flag(true); // vps_sub_layer_ordering_info_present_flag
        w.ue(self.max_dec_pic_buffering_minus1);
        w.ue(self.max_num_reorder_pics);
        w.ue(0); // vps_max_latency_increase_plus1
        w.u(6, 0); // vps_max_layer_id
        w.ue(0); // vps_num_layer_sets_minus1
        match self.timing() {
            Some((num_units_in_tick, time_scale)) => {
                w.flag(true);
                w.u(32, num_units_in_tick);
                w.u(32, time_scale);
                w.flag(false); // vps_poc_proportional_to_timing_flag
                w.ue(0); // vps_num_hrd_parameters
            }
            None => w.flag(false),
        }
        w.flag(false); // vps_extension_flag
        w.trailing_bits();
        nal::escape(&header(nal_type::VPS), &w.into_bytes())
    }

    /// The SPS NAL unit, escaped, header first; refused where the tools'
    /// sizes do not nest.
    pub(crate) fn sps_nal(&self) -> Result<Vec<u8>, Unsupported> {
        let mut w = BitWriter::new();
        let t = &self.tools;
        w.u(4, 0); // sps_video_parameter_set_id
        w.u(3, 0); // sps_max_sub_layers_minus1
        w.flag(true); // sps_temporal_id_nesting_flag
        self.write_profile_tier_level(&mut w);
        w.ue(0); // sps_seq_parameter_set_id
        w.ue(1); // chroma_format_idc: 4:2:0
        w.ue(self.width);
        w.ue(self.height);
        let cropped = self.crop_right > 0 || self.crop_bottom > 0;
        w.flag(cropped); // conformance_window_flag
        if cropped {
            // In chroma samples: SubWidthC and SubHeightC are 2 (6-1).
            w.ue(0);
            w.ue(self.crop_right / 2);
            w.ue(0);
            w.ue(self.crop_bottom / 2);
        }
        let depth = u32::from(self.profile.bit_depth())
            .checked_sub(8)
            .ok_or_else(|| Unsupported::overflow("bit_depth_luma_minus8"))?;
        w.ue(depth); // bit_depth_luma_minus8
        w.ue(depth); // bit_depth_chroma_minus8
        w.ue(u32::from(LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4));
        w.flag(true); // sps_sub_layer_ordering_info_present_flag
        w.ue(self.max_dec_pic_buffering_minus1);
        w.ue(self.max_num_reorder_pics);
        w.ue(0); // sps_max_latency_increase_plus1
        let less = |a: u8, b: u8, name| {
            a.checked_sub(b)
                .map(u32::from)
                .ok_or_else(|| Unsupported::overflow(name))
        };
        w.ue(less(
            t.log2_min_cb_size,
            3,
            "log2_min_luma_coding_block_size_minus3",
        )?);
        w.ue(less(
            t.log2_ctb_size,
            t.log2_min_cb_size,
            "log2_diff_max_min_luma_coding_block_size",
        )?);
        w.ue(less(
            t.log2_min_tb_size,
            2,
            "log2_min_luma_transform_block_size_minus2",
        )?);
        w.ue(less(
            t.log2_max_tb_size,
            t.log2_min_tb_size,
            "log2_diff_max_min_luma_transform_block_size",
        )?);
        w.ue(u32::from(t.max_transform_hierarchy_depth_inter));
        w.ue(u32::from(t.max_transform_hierarchy_depth_intra));
        w.flag(false); // scaling_list_enabled_flag
        w.flag(t.amp);
        w.flag(t.sao);
        w.flag(false); // pcm_enabled_flag
        w.ue(0); // num_short_term_ref_pic_sets: each slice header has its own
        w.flag(false); // long_term_ref_pics_present_flag
        w.flag(false); // sps_temporal_mvp_enabled_flag
        w.flag(t.strong_intra_smoothing);
        w.flag(true); // vui_parameters_present_flag
        self.write_vui(&mut w);
        w.flag(false); // sps_extension_present_flag
        w.trailing_bits();
        Ok(nal::escape(&header(nal_type::SPS), &w.into_bytes()))
    }

    fn write_vui(&self, w: &mut BitWriter) {
        let vui = &self.vui;
        match vui.sample_aspect_ratio {
            Some((h, v)) => {
                w.flag(true);
                if h == v {
                    w.u(8, 1); // square
                } else {
                    w.u(8, 255); // EXTENDED_SAR
                    w.u(16, u32::from(h));
                    w.u(16, u32::from(v));
                }
            }
            None => w.flag(false),
        }
        w.flag(false); // overscan_info_present_flag
        let signal = vui.full_range || vui.colour.is_some();
        w.flag(signal); // video_signal_type_present_flag
        if signal {
            w.u(3, 5); // video_format: unspecified
            w.flag(vui.full_range);
            w.flag(vui.colour.is_some());
            if let Some(colour) = vui.colour {
                w.u(8, u32::from(colour.primaries));
                w.u(8, u32::from(colour.transfer));
                w.u(8, u32::from(colour.matrix));
            }
        }
        w.flag(false); // chroma_loc_info_present_flag
        w.flag(false); // neutral_chroma_indication_flag
        w.flag(false); // field_seq_flag
        w.flag(false); // frame_field_info_present_flag
        w.flag(false); // default_display_window_flag
        match vui.timing {
            Some((num_units_in_tick, time_scale)) => {
                w.flag(true);
                w.u(32, num_units_in_tick);
                w.u(32, time_scale);
                w.flag(false); // vui_poc_proportional_to_timing_flag
                w.flag(false); // vui_hrd_parameters_present_flag
            }
            None => w.flag(false),
        }
        w.flag(true); // bitstream_restriction_flag
        w.flag(false); // tiles_fixed_structure_flag
        w.flag(true); // motion_vectors_over_pic_boundaries_flag
        w.flag(true); // restricted_ref_pic_lists_flag
        w.ue(0); // min_spatial_segmentation_idc
        w.ue(2); // max_bytes_per_pic_denom: the default
        w.ue(1); // max_bits_per_min_cu_denom: the default
        w.ue(15); // log2_max_mv_length_horizontal
        w.ue(15); // log2_max_mv_length_vertical
    }
}

/// What the PPS says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    /// `init_qp_minus26 + 26`.
    pub init_qp: u8,
    /// The coding tools, as the SPS's.
    pub tools: Tools,
}

impl Pps {
    /// The PPS NAL unit, escaped, header first.
    pub(crate) fn to_nal(&self) -> Result<Vec<u8>, Unsupported> {
        let mut w = BitWriter::new();
        let t = &self.tools;
        w.ue(0); // pps_pic_parameter_set_id
        w.ue(0); // pps_seq_parameter_set_id
        w.flag(false); // dependent_slice_segments_enabled_flag
        w.flag(false); // output_flag_present_flag
        w.u(3, 0); // num_extra_slice_header_bits
        w.flag(false); // sign_data_hiding_enabled_flag
        w.flag(false); // cabac_init_present_flag
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        let init_qp_minus26 = i32::from(self.init_qp)
            .checked_sub(26)
            .ok_or_else(|| Unsupported::overflow("init_qp_minus26"))?;
        w.se(init_qp_minus26);
        w.flag(t.constrained_intra_pred);
        w.flag(t.transform_skip);
        w.flag(t.cu_qp_delta);
        if t.cu_qp_delta {
            w.ue(0); // diff_cu_qp_delta_depth
        }
        w.se(0); // pps_cb_qp_offset
        w.se(0); // pps_cr_qp_offset
        w.flag(false); // pps_slice_chroma_qp_offsets_present_flag
        w.flag(false); // weighted_pred_flag
        w.flag(false); // weighted_bipred_flag
        w.flag(false); // transquant_bypass_enabled_flag
        w.flag(false); // tiles_enabled_flag
        w.flag(false); // entropy_coding_sync_enabled_flag
        w.flag(true); // pps_loop_filter_across_slices_enabled_flag
        w.flag(false); // deblocking_filter_control_present_flag
        w.flag(false); // pps_scaling_list_data_present_flag
        w.flag(false); // lists_modification_present_flag
        w.ue(0); // log2_parallel_merge_level_minus2
        w.flag(false); // slice_segment_header_extension_present_flag
        w.flag(false); // pps_extension_present_flag
        w.trailing_bits();
        Ok(nal::escape(&header(nal_type::PPS), &w.into_bytes()))
    }
}

/// `slice_type` (Table 7-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    /// 0: predicted from lists 0 and 1.
    B,
    /// 1: predicted from list 0.
    P,
    /// 2: intra only.
    I,
}

impl SliceType {
    /// The value 0 to 2 VA-API's slice parameters and the header take.
    pub fn raw(self) -> u8 {
        match self {
            Self::B => 0,
            Self::P => 1,
            Self::I => 2,
        }
    }
}

/// What one slice segment header says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    /// `nal_unit_type`: IDR_W_RADL (19) or TRAIL_R (1).
    pub nal_unit_type: u8,
    /// `slice_type`.
    pub slice_type: SliceType,
    /// `slice_pic_order_cnt_lsb`, on all but IDR pictures.
    pub(crate) pic_order_cnt_lsb: u32,
    /// The reference picture set: each reference's order count less the
    /// picture's, negative before it and positive after it, all used.
    pub(crate) references: Vec<i32>,
    /// `slice_qp_delta`: the slice's QP less the PPS's.
    pub slice_qp_delta: i32,
    /// `MaxNumMergeCand`, 1 to 5.
    pub max_num_merge_cand: u8,
}

impl SliceHeader {
    fn is_idr(&self) -> bool {
        matches!(self.nal_unit_type, nal_type::IDR_W_RADL | 20)
    }

    fn is_irap(&self) -> bool {
        (16..=23).contains(&self.nal_unit_type)
    }

    /// The NAL unit header and the slice segment header, through its
    /// `byte_alignment()`, unescaped: what a packed slice header holds.
    pub(crate) fn to_bytes(&self, sps: &Sps) -> Result<Vec<u8>, Unsupported> {
        let mut w = BitWriter::new();
        for byte in header(self.nal_unit_type) {
            w.u(8, u32::from(byte));
        }
        w.flag(true); // first_slice_segment_in_pic_flag
        if self.is_irap() {
            w.flag(false); // no_output_of_prior_pics_flag
        }
        w.ue(0); // slice_pic_parameter_set_id
        w.ue(u32::from(self.slice_type.raw()));
        if !self.is_idr() {
            w.u(
                u32::from(LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4) + 4,
                self.pic_order_cnt_lsb,
            );
            w.flag(false); // short_term_ref_pic_set_sps_flag
            self.write_reference_set(&mut w)?;
        }
        if sps.tools.sao {
            w.flag(true); // slice_sao_luma_flag
            w.flag(true); // slice_sao_chroma_flag
        }
        if self.slice_type != SliceType::I {
            w.flag(false); // num_ref_idx_active_override_flag
            if self.slice_type == SliceType::B {
                w.flag(false); // mvd_l1_zero_flag
            }
            let five_minus_max_num_merge_cand = 5u8
                .checked_sub(self.max_num_merge_cand.clamp(1, 5))
                .ok_or_else(|| Unsupported::overflow("five_minus_max_num_merge_cand"))?;
            w.ue(u32::from(five_minus_max_num_merge_cand));
        }
        w.se(self.slice_qp_delta);
        // The deblocking filter is on (the PPS has no control), so the
        // loop filter flag follows.
        w.flag(true); // slice_loop_filter_across_slices_enabled_flag
        w.trailing_bits(); // byte_alignment()
        Ok(w.into_bytes())
    }

    /// `st_ref_pic_set(num_short_term_ref_pic_sets)` (7.3.7): the first
    /// set coded, so no prediction from another.
    fn write_reference_set(&self, w: &mut BitWriter) -> Result<(), Unsupported> {
        let mut before: Vec<i32> = self.references.iter().copied().filter(|&d| d < 0).collect();
        let mut after: Vec<i32> = self.references.iter().copied().filter(|&d| d > 0).collect();
        before.sort_unstable_by(|a, b| b.cmp(a)); // closest first
        after.sort_unstable();
        w.ue(u32::try_from(before.len()).unwrap_or(0));
        w.ue(u32::try_from(after.len()).unwrap_or(0));
        // Each delta's distance from the one before it, closer, or from 0
        // for the first, less one (7.4.8).
        let minus1 = |far: i32, near: i32, name| {
            far.checked_sub(near)
                .and_then(|d| d.checked_sub(1))
                .map(i32::unsigned_abs)
                .ok_or_else(|| Unsupported::overflow(name))
        };
        let mut previous = 0;
        for delta in before {
            w.ue(minus1(previous, delta, "delta_poc_s0_minus1")?);
            w.flag(true); // used_by_curr_pic_s0_flag
            previous = delta;
        }
        previous = 0;
        for delta in after {
            w.ue(minus1(delta, previous, "delta_poc_s1_minus1")?);
            w.flag(true); // used_by_curr_pic_s1_flag
            previous = delta;
        }
        Ok(())
    }
}

/// The `hvcC` record (HEVCDecoderConfigurationRecord) for one VPS, SPS and
/// PPS, escaped NAL units with their headers, with four-byte NAL lengths.
/// The profile, tier, level, chroma format and bit depths are read from
/// the SPS, so that a record of the driver's units describes them; `None`
/// when the SPS does not read.
pub fn hvcc(vps: &[u8], sps: &[u8], pps: &[u8]) -> Option<Vec<u8>> {
    let (ptl, chroma_format_idc, luma_depth, chroma_depth) = record_fields(sps).ok()?;
    let mut record = Vec::new();
    record.push(1); // configurationVersion
    record.extend_from_slice(&ptl); // profile space to level, as the SPS has them
    record.extend_from_slice(&0xf000u16.to_be_bytes()); // min_spatial_segmentation_idc 0
    record.push(0xfc); // parallelismType 0: unknown
    record.push(0xfc | u8::try_from(chroma_format_idc).ok()?);
    record.push(0xf8 | u8::try_from(luma_depth).ok()?);
    record.push(0xf8 | u8::try_from(chroma_depth).ok()?);
    record.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate: unspecified
    // constantFrameRate 0, numTemporalLayers 1, temporalIdNested 1,
    // lengthSizeMinusOne 3.
    record.push((1 << 3) | (1 << 2) | 3);
    record.push(3); // numOfArrays
    for (unit_type, unit) in [
        (nal_type::VPS, vps),
        (nal_type::SPS, sps),
        (nal_type::PPS, pps),
    ] {
        record.push(0x80 | unit_type); // array_completeness, NAL_unit_type
        record.extend_from_slice(&1u16.to_be_bytes()); // numNalus
        record.extend_from_slice(&u16::try_from(unit.len()).ok()?.to_be_bytes());
        record.extend_from_slice(unit);
    }
    Some(record)
}

/// What the `hvcC` record takes from an SPS, escaped, header first: its
/// `general_profile_tier_level` (12 bytes), `chroma_format_idc`, and
/// `bit_depth_luma_minus8` and `bit_depth_chroma_minus8`.
fn record_fields(sps: &[u8]) -> Result<([u8; 12], u32, u32, u32), crate::hevc::Error> {
    let mut r = BitReader::new(sps);
    r.skip(16)?; // nal_unit_header()
    r.skip(4)?; // sps_video_parameter_set_id
    if r.bits(3)? != 0 {
        // Sub-layers: not what the encoder writes.
        return Err(crate::hevc::Error::Unsupported("sub-layers"));
    }
    r.skip(1)?; // sps_temporal_id_nesting_flag
    let mut ptl = [0u8; 12];
    for word in ptl.as_chunks_mut::<4>().0 {
        *word = r.bits(32)?.to_be_bytes();
    }
    r.ue()?; // sps_seq_parameter_set_id
    let chroma_format_idc = r.ue_max(3, "chroma_format_idc")?;
    if chroma_format_idc == 3 {
        r.skip(1)?; // separate_colour_plane_flag
    }
    r.ue()?; // pic_width_in_luma_samples
    r.ue()?; // pic_height_in_luma_samples
    if r.bit()? {
        for _ in 0..4 {
            r.ue()?; // conf_win_*_offset
        }
    }
    let luma_depth = r.ue_max(7, "bit_depth_luma_minus8")?;
    let chroma_depth = r.ue_max(7, "bit_depth_chroma_minus8")?;
    Ok((ptl, chroma_format_idc, luma_depth, chroma_depth))
}

/// One row of Tables A.8 and A.9, Main tier.
struct Level {
    level_idc: u8,
    /// MaxLumaPs: luma samples a picture.
    max_luma_ps: u64,
    /// MaxLumaSr: luma samples a second.
    max_luma_sr: u64,
    /// MaxBR, in units of 1000 bit/s (CpbVclFactor, Table A.11).
    max_br: u64,
}

const LEVELS: [Level; 13] = [
    level(30, 36_864, 552_960, 128),
    level(60, 122_880, 3_686_400, 1_500),
    level(63, 245_760, 7_372_800, 3_000),
    level(90, 552_960, 16_588_800, 6_000),
    level(93, 983_040, 33_177_600, 10_000),
    level(120, 2_228_224, 66_846_720, 12_000),
    level(123, 2_228_224, 133_693_440, 20_000),
    level(150, 8_912_896, 267_386_880, 25_000),
    level(153, 8_912_896, 534_773_760, 40_000),
    level(156, 8_912_896, 1_069_547_520, 60_000),
    level(180, 35_651_584, 1_069_547_520, 60_000),
    level(183, 35_651_584, 2_139_095_040, 120_000),
    level(186, 35_651_584, 4_278_190_080, 240_000),
];

const fn level(level_idc: u8, max_luma_ps: u64, max_luma_sr: u64, max_br: u64) -> Level {
    Level {
        level_idc,
        max_luma_ps,
        max_luma_sr,
        max_br,
    }
}

/// What a stream needs of its level.
#[derive(Clone, Copy, Debug)]
pub struct LevelNeeds {
    /// The width in luma samples.
    pub width: u32,
    /// The height in luma samples.
    pub height: u32,
    /// Pictures a second.
    pub frames_per_second: f64,
    /// The peak bitrate, if bounded.
    pub bits_per_second: Option<u32>,
    /// `sps_max_dec_pic_buffering_minus1 + 1`.
    pub dpb_pictures: u32,
}

impl LevelNeeds {
    #[expect(
        clippy::as_conversions,
        reason = "a luma sample count and rate compared as floats, the frame rate being one"
    )]
    fn fits(&self, level: &Level) -> bool {
        let (w, h) = (u64::from(self.width), u64::from(self.height));
        // A product past a u64 is past every limit.
        let Some(size) = w.checked_mul(h) else {
            return false;
        };
        // Each dimension at most sqrt(8 * MaxLumaPs) (A.4.1), so its
        // square over 8, rounded up, at most MaxLumaPs.
        let side = |d: u64| {
            d.checked_mul(d)
                .is_some_and(|sq| sq.div_ceil(8) <= level.max_luma_ps)
        };
        // maxDpbSize (A.4.2): 6 pictures at the level's largest size, up
        // to 16 for smaller ones. Three quarters of MaxLumaPs, rounded
        // down, is it less a quarter rounded up.
        let three_quarters = level.max_luma_ps.abs_diff(level.max_luma_ps.div_ceil(4));
        let max_dpb = if size <= level.max_luma_ps / 4 {
            16
        } else if size <= level.max_luma_ps / 2 {
            12
        } else if size <= three_quarters {
            8
        } else {
            6
        };
        size <= level.max_luma_ps
            && side(w)
            && side(h)
            && (size as f64) * self.frames_per_second <= level.max_luma_sr as f64
            && u64::from(self.dpb_pictures) <= max_dpb
            // At most MaxBR thousand, so in thousands, rounded up, at most
            // MaxBR.
            && self
                .bits_per_second
                .is_none_or(|bps| u64::from(bps).div_ceil(1000) <= level.max_br)
    }
}

/// The lowest Main tier level that fits, or why none does.
pub fn choose_level(needs: &LevelNeeds) -> Result<u8, Unsupported> {
    LEVELS
        .iter()
        .find(|level| needs.fits(level))
        .map(|level| level.level_idc)
        .ok_or_else(|| Unsupported(format!("no HEVC level fits {needs:?}")))
}

/// Whether `level_idc` is a Main tier level that fits.
pub(crate) fn level_fits(level_idc: u8, needs: &LevelNeeds) -> bool {
    LEVELS
        .iter()
        .any(|level| level.level_idc == level_idc && needs.fits(level))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tools radeonsi reports on an RX 7900 XTX, with the optional
    /// ones on.
    pub(crate) const TOOLS: Tools = Tools {
        log2_min_cb_size: 3,
        log2_ctb_size: 6,
        log2_min_tb_size: 2,
        log2_max_tb_size: 5,
        max_transform_hierarchy_depth_inter: 0,
        max_transform_hierarchy_depth_intra: 0,
        amp: true,
        sao: true,
        strong_intra_smoothing: true,
        transform_skip: true,
        cu_qp_delta: true,
        cu_qp_delta_required: false,
        constrained_intra_pred: false,
    };

    fn sps() -> Sps {
        Sps {
            profile: Profile::HevcMain10,
            level_idc: 123,
            width: 1920,
            height: 1080,
            crop_right: 0,
            crop_bottom: 0,
            max_dec_pic_buffering_minus1: 1,
            max_num_reorder_pics: 0,
            tools: TOOLS,
            vui: Vui {
                sample_aspect_ratio: None,
                full_range: false,
                colour: Some(ColourDescription {
                    primaries: 9,
                    transfer: 16,
                    matrix: 9,
                }),
                timing: Some((1001, 30_000)),
            },
        }
    }

    #[test]
    fn the_sps_reads_back_field_by_field() -> Result<(), crate::hevc::Error> {
        let sps = sps();
        let unit = sps.sps_nal().unwrap();
        assert_eq!(&unit[..2], [0x42, 0x01]);
        let mut r = BitReader::new(&unit);
        r.skip(16)?;
        assert_eq!(r.bits(4)?, 0);
        assert_eq!(r.bits(3)?, 0);
        assert_eq!(r.bits(1)?, 1);
        assert_eq!(r.bits(8)?, 2); // space, tier, profile 2
        assert_eq!(r.bits(32)?, 0x2000_0000);
        assert_eq!(r.bits(4)?, 0b1001);
        assert_eq!(r.bits(32)?, 0);
        assert_eq!(r.bits(12)?, 0);
        assert_eq!(r.bits(8)?, 123);
        assert_eq!(r.ue()?, 0);
        assert_eq!(r.ue()?, 1);
        assert_eq!((r.ue()?, r.ue()?), (1920, 1080));
        assert!(!r.bit()?); // no conformance window
        assert_eq!((r.ue()?, r.ue()?), (2, 2));
        assert_eq!(r.ue()?, 4);
        assert!(r.bit()?);
        assert_eq!((r.ue()?, r.ue()?, r.ue()?), (1, 0, 0));
        assert_eq!((r.ue()?, r.ue()?, r.ue()?, r.ue()?), (0, 3, 0, 3));
        assert_eq!((r.ue()?, r.ue()?), (0, 0));
        assert_eq!(r.bits(4)?, 0b0110); // scaling, amp, sao, pcm
        assert_eq!(r.ue()?, 0);
        assert_eq!(r.bits(3)?, 0b001); // long-term, temporal MVP, strong intra
        assert!(r.bit()?); // VUI
        assert_eq!(r.bits(3)?, 0b001); // aspect, overscan, signal
        assert_eq!(r.bits(3)?, 5);
        assert_eq!(r.bits(2)?, 0b01);
        assert_eq!((r.bits(8)?, r.bits(8)?, r.bits(8)?), (9, 16, 9));
        assert_eq!(r.bits(5)?, 0);
        assert!(r.bit()?); // timing
        assert_eq!((r.bits(32)?, r.bits(32)?), (1001, 30_000));
        assert_eq!(r.bits(2)?, 0);
        assert!(r.bit()?); // bitstream restriction
        assert_eq!(r.bits(3)?, 0b011);
        assert_eq!(
            (r.ue()?, r.ue()?, r.ue()?, r.ue()?, r.ue()?),
            (0, 2, 1, 15, 15)
        );
        assert!(!r.bit()?); // extensions
        assert!(r.bit()?); // stop bit
        assert_eq!(
            r.position().div_ceil(8) + r.emulation_prevention_bytes(),
            unit.len()
        );
        Ok(())
    }

    #[test]
    fn slice_headers_carry_their_reference_sets() -> Result<(), crate::hevc::Error> {
        let sps = sps();
        let header = SliceHeader {
            nal_unit_type: nal_type::TRAIL_R,
            slice_type: SliceType::P,
            pic_order_cnt_lsb: 5,
            references: vec![-1, -3],
            slice_qp_delta: -2,
            max_num_merge_cand: 5,
        };
        let bytes = header.to_bytes(&sps).unwrap();
        assert_eq!(&bytes[..2], [0x02, 0x01]);
        let mut r = BitReader::new(&bytes[2..]);
        assert!(r.bit()?); // first slice segment
        assert_eq!(r.ue()?, 0);
        assert_eq!(r.ue()?, 1); // P
        assert_eq!(r.bits(8)?, 5);
        assert!(!r.bit()?); // the set is here
        assert_eq!((r.ue()?, r.ue()?), (2, 0));
        assert_eq!((r.ue()?, r.bit()?), (0, true)); // -1
        assert_eq!((r.ue()?, r.bit()?), (1, true)); // -3
        assert_eq!(r.bits(2)?, 0b11); // SAO
        assert!(!r.bit()?);
        assert_eq!(r.ue()?, 0); // five_minus_max_num_merge_cand
        assert_eq!(r.se()?, -2);
        assert!(r.bit()?);
        assert!(r.bit()?); // byte_alignment
        assert_eq!(r.position().div_ceil(8), bytes.len() - 2);
        let idr = SliceHeader {
            nal_unit_type: nal_type::IDR_W_RADL,
            slice_type: SliceType::I,
            pic_order_cnt_lsb: 0,
            references: Vec::new(),
            slice_qp_delta: 0,
            max_num_merge_cand: 5,
        };
        let bytes = idr.to_bytes(&sps).unwrap();
        assert_eq!(&bytes[..2], [0x26, 0x01]);
        let mut r = BitReader::new(&bytes[2..]);
        assert_eq!(r.bits(2)?, 0b10); // first segment, no_output_of_prior_pics
        assert_eq!((r.ue()?, r.ue()?), (0, 2));
        assert_eq!(r.bits(2)?, 0b11);
        assert_eq!(r.se()?, 0);
        Ok(())
    }

    #[test]
    fn hvcc_carries_the_parameter_sets() {
        let sps = sps();
        let (v, s) = (sps.vps_nal(), sps.sps_nal().unwrap());
        let p = Pps {
            init_qp: 26,
            tools: TOOLS,
        }
        .to_nal()
        .unwrap();
        let record = hvcc(&v, &s, &p).unwrap_or_default();
        assert_eq!(record[1], 2);
        assert_eq!(&record[2..6], [0x20, 0, 0, 0]);
        assert_eq!(record[12], 123);
        assert_eq!(record[16], 0xfd); // 4:2:0
        assert_eq!(record[17], 0xfa); // 10-bit
        assert_eq!(record[21] & 3, 3); // four-byte lengths
        assert_eq!(record[22], 3);
        let mut at = 23;
        for (unit_type, unit) in [(32, &v), (33, &s), (34, &p)] {
            assert_eq!(record[at], 0x80 | unit_type);
            let length = usize::from(u16::from_be_bytes([record[at + 3], record[at + 4]]));
            assert_eq!(&record[at + 5..at + 5 + length], &unit[..]);
            at += 5 + length;
        }
        assert_eq!(at, record.len());
    }

    #[test]
    fn levels_come_from_tables_a_8_and_a_9() -> Result<(), Unsupported> {
        let needs = |width, height, fps, bps| LevelNeeds {
            width,
            height,
            frames_per_second: fps,
            bits_per_second: bps,
            dpb_pictures: 2,
        };
        assert_eq!(
            choose_level(&needs(1920, 1080, 30.0, Some(8_000_000)))?,
            120
        );
        assert_eq!(
            choose_level(&needs(1920, 1080, 60.0, Some(8_000_000)))?,
            123
        );
        assert_eq!(
            choose_level(&needs(1920, 1080, 30.0, Some(15_000_000)))?,
            123
        );
        assert_eq!(choose_level(&needs(1280, 720, 30.0, None))?, 93);
        assert_eq!(choose_level(&needs(3840, 2160, 60.0, None))?, 153);
        assert_eq!(choose_level(&needs(136, 128, 30.0, None))?, 30);
        assert!(choose_level(&needs(8192, 8192, 60.0, None)).is_err());
        Ok(())
    }
}
