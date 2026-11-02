//! The sequence parameter set (H.264 7.3.2.1.1) and its VUI (E.1.1), as
//! far as decoding and output order need them.

use std::num::NonZeroU32;

use super::Error;
use super::bits::BitReader;
use super::scaling::{ScalingLists, ScalingSyntax};
use crate::num::ToUsize;

/// A sequence parameter set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// `profile_idc`.
    pub profile_idc: u8,
    /// `constraint_set0_flag` to `constraint_set5_flag`, in bits 7 to 2.
    pub constraint_flags: u8,
    /// `level_idc`.
    pub level_idc: u8,
    /// `seq_parameter_set_id`.
    pub seq_parameter_set_id: u8,
    /// `chroma_format_idc` (1, 4:2:0, when the profile does not code it).
    pub chroma_format_idc: u8,
    /// `separate_colour_plane_flag`.
    pub separate_colour_plane_flag: bool,
    /// `bit_depth_luma_minus8`.
    pub bit_depth_luma_minus8: u8,
    /// `bit_depth_chroma_minus8`.
    pub bit_depth_chroma_minus8: u8,
    /// `qpprime_y_zero_transform_bypass_flag`.
    pub qpprime_y_zero_transform_bypass_flag: bool,
    /// The sequence-level scaling lists: flat unless
    /// `seq_scaling_matrix_present_flag`, else resolved by fall-back rule A.
    pub scaling_lists: ScalingLists,
    /// `seq_scaling_matrix_present_flag`.
    pub seq_scaling_matrix_present_flag: bool,
    /// `log2_max_frame_num_minus4`.
    pub log2_max_frame_num_minus4: u8,
    /// `pic_order_cnt_type`.
    pub pic_order_cnt_type: u8,
    /// `log2_max_pic_order_cnt_lsb_minus4` (type 0).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// `delta_pic_order_always_zero_flag` (type 1).
    pub delta_pic_order_always_zero_flag: bool,
    /// `offset_for_non_ref_pic` (type 1).
    pub offset_for_non_ref_pic: i32,
    /// `offset_for_top_to_bottom_field` (type 1).
    pub offset_for_top_to_bottom_field: i32,
    /// `offset_for_ref_frame[]` (type 1).
    pub offset_for_ref_frame: Vec<i32>,
    /// `max_num_ref_frames`.
    pub max_num_ref_frames: u8,
    /// `gaps_in_frame_num_value_allowed_flag`.
    pub gaps_in_frame_num_value_allowed_flag: bool,
    /// `pic_width_in_mbs_minus1`.
    pub pic_width_in_mbs_minus1: u16,
    /// `pic_height_in_map_units_minus1`.
    pub pic_height_in_map_units_minus1: u16,
    /// `frame_mbs_only_flag`.
    pub frame_mbs_only_flag: bool,
    /// `mb_adaptive_frame_field_flag`.
    pub mb_adaptive_frame_field_flag: bool,
    /// `direct_8x8_inference_flag`.
    pub direct_8x8_inference_flag: bool,
    /// `frame_crop_{left,right,top,bottom}_offset`, in crop units.
    pub frame_crop: [u32; 4],
    /// The VUI, when present and readable.
    pub vui: Option<Vui>,
    /// What the standard derives from the fields above, worked out when
    /// the SPS is read, where a value that overflows refuses it.
    pub(crate) derived: Derived,
}

/// The variables an SPS's fields give (7.4.2.1.1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Derived {
    /// `MaxFrameNum` (7-10): a power of two, 16 to 65536.
    max_frame_num: NonZeroU32,
    /// `MaxPicOrderCntLsb` (7-11).
    max_pic_order_cnt_lsb: u32,
    /// `PicWidthInMbs` (7-13).
    width_in_mbs: u32,
    /// `FrameHeightInMbs` (7-18).
    frame_height_in_mbs: u32,
    /// `PicWidthInMbs * FrameHeightInMbs`: the macroblocks of a frame.
    frame_size_in_mbs: NonZeroU32,
    /// The frame's width and height in samples, before cropping.
    coded_size: (u32, u32),
    /// The cropping rectangle in luma samples: x, y, width, height.
    crop_rect: (u32, u32, u32, u32),
}

impl Derived {
    /// The variables of an SPS with these fields, refusing one whose
    /// cropping leaves nothing (7.4.2.1.1) and one that overflows.
    fn of(
        log2_max_frame_num_minus4: u8,
        log2_max_pic_order_cnt_lsb_minus4: u8,
        pic_width_in_mbs_minus1: u16,
        pic_height_in_map_units_minus1: u16,
        frame_mbs_only_flag: bool,
        crop_units: (u32, u32),
        frame_crop: [u32; 4],
    ) -> Result<Self, Error> {
        let overflow = Error::Overflow;
        let max_frame_num = 1u32
            .checked_shl(u32::from(log2_max_frame_num_minus4) + 4)
            .and_then(NonZeroU32::new)
            .ok_or(overflow("MaxFrameNum"))?;
        let max_pic_order_cnt_lsb = 1u32
            .checked_shl(u32::from(log2_max_pic_order_cnt_lsb_minus4) + 4)
            .ok_or(overflow("MaxPicOrderCntLsb"))?;
        let width_in_mbs = u32::from(pic_width_in_mbs_minus1)
            .checked_add(1)
            .ok_or(overflow("PicWidthInMbs"))?;
        // (7-16) and (7-18): FrameHeightInMbs is
        // (2 − frame_mbs_only_flag) * PicHeightInMapUnits.
        let fields = if frame_mbs_only_flag { 1 } else { 2 };
        let frame_height_in_mbs = u32::from(pic_height_in_map_units_minus1)
            .checked_add(1)
            .and_then(|units| units.checked_mul(fields))
            .ok_or(overflow("FrameHeightInMbs"))?;
        let frame_size_in_mbs = width_in_mbs
            .checked_mul(frame_height_in_mbs)
            .and_then(NonZeroU32::new)
            .ok_or(overflow("FrameSizeInMbs"))?;
        let coded_width = width_in_mbs
            .checked_mul(16)
            .ok_or(overflow("PicWidthInSamplesL"))?;
        let coded_height = frame_height_in_mbs
            .checked_mul(16)
            .ok_or(overflow("PicHeightInSamplesL"))?;
        // 7.4.2.1.1: the crop offsets, in crop units, leave samples on
        // each side; offsets whose product passes 32 bits are past any
        // frame's.
        let (unit_x, unit_y) = crop_units;
        let [left, right, top, bottom] = frame_crop;
        let cropped = |size: u32, unit: u32, before: u32, after: u32| {
            let offset = unit.checked_mul(before)?;
            let both = offset.checked_add(unit.checked_mul(after)?)?;
            Some((offset, size.checked_sub(both).filter(|&left| left > 0)?))
        };
        let ((x, width), (y, height)) = cropped(coded_width, unit_x, left, right)
            .zip(cropped(coded_height, unit_y, top, bottom))
            .ok_or(Error::OutOfRange("frame_crop offsets"))?;
        Ok(Self {
            max_frame_num,
            max_pic_order_cnt_lsb,
            width_in_mbs,
            frame_height_in_mbs,
            frame_size_in_mbs,
            coded_size: (coded_width, coded_height),
            crop_rect: (x, y, width, height),
        })
    }
}

/// What this decoder keeps of the VUI (E.1.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vui {
    /// `sar_width` and `sar_height`, from `aspect_ratio_idc` (Table E-1).
    pub(crate) sample_aspect_ratio: Option<(u16, u16)>,
    /// `video_full_range_flag`.
    pub video_full_range_flag: bool,
    /// `colour_primaries`, `transfer_characteristics`, `matrix_coefficients`.
    pub colour_description: Option<(u8, u8, u8)>,
    /// `chroma_sample_loc_type_top_field` and `_bottom_field`, as coded:
    /// 0 to 5 in a conformant stream (E.2.1). Kept for what a stream says,
    /// not used in decoding.
    pub chroma_sample_loc: Option<(u32, u32)>,
    /// `num_units_in_tick` and `time_scale`.
    pub(crate) timing: Option<(u32, u32)>,
    /// `max_num_reorder_frames`, from `bitstream_restriction_flag`.
    pub max_num_reorder_frames: Option<u32>,
    /// `max_dec_frame_buffering`, from `bitstream_restriction_flag`.
    pub(crate) max_dec_frame_buffering: Option<u32>,
}

/// The profiles whose SPS codes chroma format, bit depth and scaling lists
/// (7.3.2.1.1).
fn has_chroma_info(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

impl Sps {
    /// Parses an SPS NAL unit, header byte included.
    pub fn parse(nal: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(nal);
        r.skip(8)?;
        let profile_idc = r.u8(8)?;
        let constraint_flags = r.u8(8)?;
        let level_idc = r.u8(8)?;
        let seq_parameter_set_id = r.ue_u8(31, "seq_parameter_set_id")?;
        let mut chroma_format_idc = 1;
        let mut separate_colour_plane_flag = false;
        let mut bit_depth_luma_minus8 = 0;
        let mut bit_depth_chroma_minus8 = 0;
        let mut qpprime_y_zero_transform_bypass_flag = false;
        let mut seq_scaling_matrix_present_flag = false;
        let mut scaling_lists = ScalingLists::flat();
        if has_chroma_info(profile_idc) {
            chroma_format_idc = r.ue_u8(3, "chroma_format_idc")?;
            if chroma_format_idc == 3 {
                separate_colour_plane_flag = r.bit()?;
            }
            bit_depth_luma_minus8 = r.ue_u8(6, "bit_depth_luma_minus8")?;
            bit_depth_chroma_minus8 = r.ue_u8(6, "bit_depth_chroma_minus8")?;
            qpprime_y_zero_transform_bypass_flag = r.bit()?;
            seq_scaling_matrix_present_flag = r.bit()?;
            if seq_scaling_matrix_present_flag {
                let lists_8x8 = if chroma_format_idc == 3 { 6 } else { 2 };
                scaling_lists = ScalingSyntax::parse(&mut r, lists_8x8)?.resolve(None);
            }
        }
        let log2_max_frame_num_minus4 = r.ue_u8(12, "log2_max_frame_num_minus4")?;
        let pic_order_cnt_type = r.ue_u8(2, "pic_order_cnt_type")?;
        let mut log2_max_pic_order_cnt_lsb_minus4 = 0;
        let mut delta_pic_order_always_zero_flag = false;
        let mut offset_for_non_ref_pic = 0;
        let mut offset_for_top_to_bottom_field = 0;
        let mut offset_for_ref_frame = Vec::new();
        match pic_order_cnt_type {
            0 => {
                log2_max_pic_order_cnt_lsb_minus4 =
                    r.ue_u8(12, "log2_max_pic_order_cnt_lsb_minus4")?;
            }
            1 => {
                delta_pic_order_always_zero_flag = r.bit()?;
                offset_for_non_ref_pic =
                    r.se_range(i32::MIN + 1, i32::MAX, "offset_for_non_ref_pic")?;
                offset_for_top_to_bottom_field =
                    r.se_range(i32::MIN + 1, i32::MAX, "offset_for_top_to_bottom_field")?;
                let cycle = r.ue_max(255, "num_ref_frames_in_pic_order_cnt_cycle")?;
                for _ in 0..cycle {
                    offset_for_ref_frame.push(r.se_range(
                        i32::MIN + 1,
                        i32::MAX,
                        "offset_for_ref_frame",
                    )?);
                }
            }
            _ => {}
        }
        let max_num_ref_frames = r.ue_u8(16, "max_num_ref_frames")?;
        let gaps_in_frame_num_value_allowed_flag = r.bit()?;
        // 8192 macroblocks a side is past every level's MaxFS (Table A-1).
        let pic_width_in_mbs_minus1 = r.ue_u16(8191, "pic_width_in_mbs_minus1")?;
        let pic_height_in_map_units_minus1 = r.ue_u16(8191, "pic_height_in_map_units_minus1")?;
        let frame_mbs_only_flag = r.bit()?;
        let mb_adaptive_frame_field_flag = if frame_mbs_only_flag { false } else { r.bit()? };
        let direct_8x8_inference_flag = r.bit()?;
        let mut frame_crop = [0u32; 4];
        if r.bit()? {
            for offset in &mut frame_crop {
                *offset = r.ue()?;
            }
        }
        let vui_parameters_present_flag = r.bit()?;
        // A VUI that does not parse costs nothing the decode needs: it
        // is dropped, as if absent, rather than refusing the stream.
        let vui = if vui_parameters_present_flag {
            Vui::parse(&mut r).ok()
        } else {
            None
        };
        let crop_units = crop_units(
            if separate_colour_plane_flag {
                0
            } else {
                chroma_format_idc
            },
            frame_mbs_only_flag,
        );
        let derived = Derived::of(
            log2_max_frame_num_minus4,
            log2_max_pic_order_cnt_lsb_minus4,
            pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1,
            frame_mbs_only_flag,
            crop_units,
            frame_crop,
        )?;
        Ok(Self {
            profile_idc,
            constraint_flags,
            level_idc,
            seq_parameter_set_id,
            chroma_format_idc,
            separate_colour_plane_flag,
            bit_depth_luma_minus8,
            bit_depth_chroma_minus8,
            qpprime_y_zero_transform_bypass_flag,
            scaling_lists,
            seq_scaling_matrix_present_flag,
            log2_max_frame_num_minus4,
            pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb_minus4,
            delta_pic_order_always_zero_flag,
            offset_for_non_ref_pic,
            offset_for_top_to_bottom_field,
            offset_for_ref_frame,
            max_num_ref_frames,
            gaps_in_frame_num_value_allowed_flag,
            pic_width_in_mbs_minus1,
            pic_height_in_map_units_minus1,
            frame_mbs_only_flag,
            mb_adaptive_frame_field_flag,
            direct_8x8_inference_flag,
            frame_crop,
            vui,
            derived,
        })
    }

    /// `MaxFrameNum` (7-10).
    pub fn max_frame_num(&self) -> u32 {
        self.derived.max_frame_num.get()
    }

    /// `(frame_num + n) % MaxFrameNum`: frame numbers wrap at MaxFrameNum
    /// (7.4.3), a power of two no larger than 2^32, so the wrap of the
    /// sum at 2^32 leaves the remainder as it is.
    pub(crate) fn frame_num_plus(&self, frame_num: u32, n: u32) -> u32 {
        frame_num.wrapping_add(n) % self.derived.max_frame_num
    }

    /// `(a − b) % MaxFrameNum`, as [`Self::frame_num_plus`] wraps.
    pub(crate) fn frame_num_minus(&self, a: u32, b: u32) -> u32 {
        a.wrapping_sub(b) % self.derived.max_frame_num
    }

    /// `MaxPicOrderCntLsb` (7-11).
    pub fn max_pic_order_cnt_lsb(&self) -> u32 {
        self.derived.max_pic_order_cnt_lsb
    }

    /// `PicWidthInMbs` (7-13).
    pub fn width_in_mbs(&self) -> u32 {
        self.derived.width_in_mbs
    }

    /// `FrameHeightInMbs` (7-18).
    pub fn frame_height_in_mbs(&self) -> u32 {
        self.derived.frame_height_in_mbs
    }

    /// `PicWidthInMbs * FrameHeightInMbs`: the macroblocks of a frame,
    /// `PicSizeInMbs` of every picture the decoder takes.
    pub(crate) fn frame_size_in_mbs(&self) -> u32 {
        self.derived.frame_size_in_mbs.get()
    }

    /// The decoded frame's width in samples, before cropping.
    pub fn coded_width(&self) -> u32 {
        self.derived.coded_size.0
    }

    /// The decoded frame's height in samples, before cropping.
    pub fn coded_height(&self) -> u32 {
        self.derived.coded_size.1
    }

    /// `ChromaArrayType` (7.4.2.1.1).
    pub fn chroma_array_type(&self) -> u8 {
        if self.separate_colour_plane_flag {
            0
        } else {
            self.chroma_format_idc
        }
    }

    /// Whether the samples use the full range, `video_full_range_flag`
    /// (E.2.1): false when the VUI or its video signal type is absent.
    pub fn is_full_range(&self) -> bool {
        self.vui
            .as_ref()
            .is_some_and(|vui| vui.video_full_range_flag)
    }

    /// The cropping rectangle in luma samples: x, y, width, height.
    pub fn crop_rect(&self) -> (u32, u32, u32, u32) {
        self.derived.crop_rect
    }

    /// `MaxDpbFrames` (A.3.1, item h): from the level's `MaxDpbMbs`
    /// (Table A-1), at most 16. The VUI's `max_dec_frame_buffering`
    /// replaces it where given. Never fewer than `max_num_ref_frames`.
    pub fn dpb_frames(&self) -> usize {
        let from_level = (self.max_dpb_mbs() / self.derived.frame_size_in_mbs).min(16);
        let frames = self
            .vui
            .as_ref()
            .and_then(|vui| vui.max_dec_frame_buffering)
            .unwrap_or(from_level)
            .min(16);
        frames
            .max(u32::from(self.max_num_ref_frames))
            .max(1)
            .to_usize()
    }

    /// `max_num_reorder_frames`, or the DPB's size where the VUI does not
    /// bound it.
    pub fn max_num_reorder_frames(&self) -> usize {
        let dpb = self.dpb_frames();
        self.vui
            .as_ref()
            .and_then(|vui| vui.max_num_reorder_frames)
            .map_or(dpb, |n| n.to_usize().min(dpb))
    }

    /// `MaxDpbMbs` of Table A-1, by `level_idc`. Level 1b is `level_idc`
    /// 11 with `constraint_set3_flag` in Baseline, Main and Extended, or 9.
    fn max_dpb_mbs(&self) -> u32 {
        match self.level_idc {
            9 | 10 => 396,
            11 if self.constraint_flags & 0x10 != 0 && matches!(self.profile_idc, 66 | 77 | 88) => {
                396
            }
            11 => 900,
            12 | 13 | 20 => 2376,
            21 => 4752,
            22 | 30 => 8100,
            31 => 18000,
            32 => 20480,
            40 | 41 => 32768,
            42 => 34816,
            50 => 110400,
            51 | 52 => 184320,
            _ => 696320,
        }
    }
}

impl Vui {
    fn parse(r: &mut BitReader<'_>) -> Result<Self, Error> {
        let mut vui = Self::default();
        if r.bit()? {
            let idc = r.u8(8)?;
            vui.sample_aspect_ratio = match idc {
                255 => Some((r.u16()?, r.u16()?)),
                _ => idc
                    .checked_sub(1)
                    .and_then(|i| SAR_TABLE.get(usize::from(i)))
                    .copied(),
            };
        }
        if r.bit()? {
            // overscan_appropriate_flag
            r.skip(1)?;
        }
        if r.bit()? {
            // video_format
            r.skip(3)?;
            vui.video_full_range_flag = r.bit()?;
            if r.bit()? {
                vui.colour_description = Some((r.u8(8)?, r.u8(8)?, r.u8(8)?));
            }
        }
        if r.bit()? {
            vui.chroma_sample_loc = Some((r.ue()?, r.ue()?));
        }
        if r.bit()? {
            let num_units_in_tick = r.bits(32)?;
            let time_scale = r.bits(32)?;
            // fixed_frame_rate_flag
            r.skip(1)?;
            vui.timing = Some((num_units_in_tick, time_scale));
        }
        let nal_hrd = r.bit()?;
        if nal_hrd {
            skip_hrd_parameters(r)?;
        }
        let vcl_hrd = r.bit()?;
        if vcl_hrd {
            skip_hrd_parameters(r)?;
        }
        if nal_hrd || vcl_hrd {
            // low_delay_hrd_flag
            r.skip(1)?;
        }
        // pic_struct_present_flag
        r.skip(1)?;
        if r.bit()? {
            // motion_vectors_over_pic_boundaries_flag
            r.skip(1)?;
            r.ue()?; // max_bytes_per_pic_denom
            r.ue()?; // max_bits_per_mb_denom
            r.ue()?; // log2_max_mv_length_horizontal
            r.ue()?; // log2_max_mv_length_vertical
            vui.max_num_reorder_frames = Some(r.ue_max(16, "max_num_reorder_frames")?);
            vui.max_dec_frame_buffering = Some(r.ue_max(16, "max_dec_frame_buffering")?);
        }
        Ok(vui)
    }
}

/// `CropUnitX` and `CropUnitY` (7-19 to 7-22): `SubWidthC` and
/// `SubHeightC` (Table 6-1) where `ChromaArrayType` is 1 or 2, else 1,
/// the height's doubled where the frames hold fields
/// (`2 − frame_mbs_only_flag`).
fn crop_units(chroma_array_type: u8, frame_mbs_only_flag: bool) -> (u32, u32) {
    match (chroma_array_type, frame_mbs_only_flag) {
        (1, true) => (2, 2),
        (1, false) => (2, 4),
        (2, true) => (2, 1),
        (2, false) => (2, 2),
        (_, true) => (1, 1),
        (_, false) => (1, 2),
    }
}

/// `hrd_parameters()` (E.1.2): read past.
fn skip_hrd_parameters(r: &mut BitReader<'_>) -> Result<(), Error> {
    let cpb_cnt_minus1 = r.ue_max(31, "cpb_cnt_minus1")?;
    r.skip(8)?; // bit_rate_scale, cpb_size_scale
    for _ in 0..=cpb_cnt_minus1 {
        r.ue()?; // bit_rate_value_minus1
        r.ue()?; // cpb_size_value_minus1
        r.skip(1)?; // cbr_flag
    }
    // initial_cpb_removal_delay_length_minus1, cpb_removal_delay_length_minus1,
    // dpb_output_delay_length_minus1, time_offset_length
    r.skip(20)
}

/// Table E-1's sample aspect ratios, for `aspect_ratio_idc` 1 to 16.
const SAR_TABLE: [(u16, u16); 16] = [
    (1, 1),
    (12, 11),
    (10, 11),
    (16, 11),
    (40, 33),
    (24, 11),
    (20, 11),
    (32, 11),
    (80, 33),
    (18, 11),
    (15, 11),
    (64, 33),
    (160, 99),
    (4, 3),
    (3, 2),
    (2, 1),
];
