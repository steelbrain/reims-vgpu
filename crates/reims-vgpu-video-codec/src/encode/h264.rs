//! H.264's headers as the encoder writes them: the SPS with its VUI
//! (7.3.2.1, E.1.1), the PPS (7.3.2.2), the slice header (7.3.3), the
//! `avcC` record (ISO/IEC 14496-15, 5.3.3.1), and the level from Table A-1.
//!
//! What the encoder writes is a fixed shape: frames only, picture order
//! count type 0, one slice a picture, references by sliding window, no
//! weighted prediction, list modification or scaling matrices.

use super::bits::BitWriter;
use super::nal;
use super::{ColourDescription, Entropy, Profile, Unsupported};

/// `nal_unit_type`s the encoder writes or strips (Table 7-1).
pub mod nal_type {
    /// A non-IDR slice.
    pub(crate) const SLICE: u8 = 1;
    /// An IDR slice.
    pub(crate) const IDR: u8 = 5;
    /// A sequence parameter set.
    pub(crate) const SPS: u8 = 7;
    /// A picture parameter set.
    pub(crate) const PPS: u8 = 8;
    /// An access unit delimiter.
    pub(crate) const AUD: u8 = 9;
}

/// `log2_max_frame_num_minus4` the encoder writes: `frame_num` wraps at 256.
pub const LOG2_MAX_FRAME_NUM_MINUS4: u8 = 4;
/// `log2_max_pic_order_cnt_lsb_minus4`: the order count's LSBs wrap at 256,
/// 128 frames, far beyond any distance between a picture and its
/// references.
pub const LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4: u8 = 4;

/// The VUI the encoder writes (E.1.1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Vui {
    /// `aspect_ratio_idc` 255's `sar_width` and `sar_height`, or none written.
    pub sample_aspect_ratio: Option<(u16, u16)>,
    /// `video_full_range_flag`.
    pub(crate) full_range: bool,
    /// `colour_primaries`, `transfer_characteristics` and `matrix_coefficients`, or none written.
    pub(crate) colour: Option<ColourDescription>,
    /// `num_units_in_tick` and `time_scale`: two ticks a frame.
    pub timing: Option<(u32, u32)>,
    /// `max_num_reorder_frames`: the B pictures a frame may wait behind.
    pub(crate) max_num_reorder_frames: u32,
    /// `max_dec_frame_buffering`: the frames the DPB holds.
    pub(crate) max_dec_frame_buffering: u32,
}

/// What the SPS says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    /// The profile, and with it `profile_idc` and the constraint flags.
    pub profile: Profile,
    /// `level_idc`: 31 is level 3.1.
    pub level_idc: u8,
    /// `max_num_ref_frames`.
    pub max_num_ref_frames: u8,
    /// `pic_width_in_mbs_minus1 + 1`.
    pub width_in_mbs: u32,
    /// `pic_height_in_map_units_minus1 + 1`: frames only.
    pub height_in_mbs: u32,
    /// Luma samples cropped off the right and the bottom: even.
    pub crop_right: u32,
    /// Luma samples cropped off the bottom: even.
    pub crop_bottom: u32,
    /// The VUI.
    pub vui: Vui,
}

impl Sps {
    /// The constraint flags byte: `constraint_set0_flag` is its top bit.
    /// Constrained Baseline sets 0 and 1, Main sets 1 (as x264 does), High
    /// none.
    pub(crate) fn constraint_flags(&self) -> u8 {
        match self.profile {
            Profile::H264ConstrainedBaseline => 0b1100_0000,
            Profile::H264Main => 0b0100_0000,
            _ => 0,
        }
    }

    fn has_chroma_info(&self) -> bool {
        self.profile.profile_idc() >= 100
    }

    /// The SPS NAL unit, escaped, header byte first; refused for a
    /// picture of no macroblocks across or down, which no SPS codes.
    pub fn to_nal(&self) -> Result<Vec<u8>, Unsupported> {
        let mut w = BitWriter::new();
        w.u(8, u32::from(self.profile.profile_idc()));
        w.u(8, u32::from(self.constraint_flags()));
        w.u(8, u32::from(self.level_idc));
        w.ue(0); // seq_parameter_set_id
        if self.has_chroma_info() {
            w.ue(1); // chroma_format_idc: 4:2:0
            w.ue(0); // bit_depth_luma_minus8
            w.ue(0); // bit_depth_chroma_minus8
            w.flag(false); // qpprime_y_zero_transform_bypass_flag
            w.flag(false); // seq_scaling_matrix_present_flag
        }
        w.ue(u32::from(LOG2_MAX_FRAME_NUM_MINUS4));
        w.ue(0); // pic_order_cnt_type
        w.ue(u32::from(LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4));
        w.ue(u32::from(self.max_num_ref_frames));
        w.flag(false); // gaps_in_frame_num_value_allowed_flag
        // PicWidthInMbs and PicHeightInMapUnits are their `_minus1`
        // elements plus one (7-13, 7-16): 1 at least, as the encoder's
        // pictures are, an empty one being refused. No element codes a 0,
        // so the check refuses it.
        let minus1 = |value: u32, name| {
            value
                .checked_sub(1)
                .ok_or_else(|| Unsupported::overflow(name))
        };
        w.ue(minus1(self.width_in_mbs, "pic_width_in_mbs_minus1")?);
        w.ue(minus1(
            self.height_in_mbs,
            "pic_height_in_map_units_minus1",
        )?);
        w.flag(true); // frame_mbs_only_flag
        w.flag(true); // direct_8x8_inference_flag
        let cropped = self.crop_right > 0 || self.crop_bottom > 0;
        w.flag(cropped);
        if cropped {
            // CropUnitX and CropUnitY are 2 for 4:2:0 frames (7-19, 7-20).
            w.ue(0);
            w.ue(self.crop_right / 2);
            w.ue(0);
            w.ue(self.crop_bottom / 2);
        }
        w.flag(true); // vui_parameters_present_flag
        write_vui(&mut w, &self.vui);
        w.trailing_bits();
        Ok(nal::escape(&[0x60 | nal_type::SPS], &w.into_bytes()))
    }
}

fn write_vui(w: &mut BitWriter, vui: &Vui) {
    match vui.sample_aspect_ratio {
        Some((h, v)) => {
            w.flag(true);
            if h == v {
                w.u(8, 1); // aspect_ratio_idc 1: square
            } else {
                w.u(8, 255); // Extended_SAR
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
    match vui.timing {
        Some((num_units_in_tick, time_scale)) => {
            w.flag(true);
            w.u(32, num_units_in_tick);
            w.u(32, time_scale);
            w.flag(false); // fixed_frame_rate_flag: frames may come late
        }
        None => w.flag(false),
    }
    w.flag(false); // nal_hrd_parameters_present_flag
    w.flag(false); // vcl_hrd_parameters_present_flag
    w.flag(false); // pic_struct_present_flag
    w.flag(true); // bitstream_restriction_flag
    w.flag(true); // motion_vectors_over_pic_boundaries_flag
    w.ue(2); // max_bytes_per_pic_denom: the default
    w.ue(1); // max_bits_per_mb_denom: the default
    w.ue(15); // log2_max_mv_length_horizontal
    w.ue(15); // log2_max_mv_length_vertical
    w.ue(vui.max_num_reorder_frames);
    w.ue(vui.max_dec_frame_buffering);
}

/// What the PPS says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    /// The profile: High's PPS carries `transform_8x8_mode_flag` and its tail.
    pub profile: Profile,
    /// `entropy_coding_mode_flag`: CAVLC or CABAC.
    pub entropy: Entropy,
    /// `pic_init_qp_minus26 + 26`.
    pub pic_init_qp: u8,
    /// High's 8x8 transform.
    pub transform_8x8_mode_flag: bool,
}

impl Pps {
    /// The PPS NAL unit, escaped, header byte first.
    pub fn to_nal(&self) -> Result<Vec<u8>, Unsupported> {
        let mut w = BitWriter::new();
        w.ue(0); // pic_parameter_set_id
        w.ue(0); // seq_parameter_set_id
        w.flag(self.entropy == Entropy::Cabac);
        w.flag(false); // bottom_field_pic_order_in_frame_present_flag
        w.ue(0); // num_slice_groups_minus1
        w.ue(0); // num_ref_idx_l0_default_active_minus1
        w.ue(0); // num_ref_idx_l1_default_active_minus1
        w.flag(false); // weighted_pred_flag
        w.u(2, 0); // weighted_bipred_idc
        // A u8 less 26 fits an i32: the check never refuses.
        let pic_init_qp_minus26 = i32::from(self.pic_init_qp)
            .checked_sub(26)
            .ok_or_else(|| Unsupported::overflow("pic_init_qp_minus26"))?;
        w.se(pic_init_qp_minus26);
        w.se(0); // pic_init_qs_minus26
        w.se(0); // chroma_qp_index_offset
        w.flag(true); // deblocking_filter_control_present_flag
        w.flag(false); // constrained_intra_pred_flag
        w.flag(false); // redundant_pic_cnt_present_flag
        if self.profile == Profile::H264High {
            w.flag(self.transform_8x8_mode_flag);
            w.flag(false); // pic_scaling_matrix_present_flag
            w.se(0); // second_chroma_qp_index_offset
        }
        w.trailing_bits();
        Ok(nal::escape(&[0x60 | nal_type::PPS], &w.into_bytes()))
    }
}

/// `slice_type` (Table 7-6), all slices of the picture alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceType {
    /// 0: predicted from list 0.
    P,
    /// 1: predicted from lists 0 and 1.
    B,
    /// 2: intra only.
    I,
}

impl SliceType {
    /// The value 0 to 2 VA-API's slice parameters and the header take.
    pub fn raw(self) -> u8 {
        match self {
            Self::P => 0,
            Self::B => 1,
            Self::I => 2,
        }
    }
}

/// What one slice header says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    /// `slice_type`, as the picture's.
    pub slice_type: SliceType,
    /// Whether the slice is an IDR picture's (`nal_unit_type` 5).
    pub(crate) idr: bool,
    /// `nal_ref_idc`: 0 for a picture nothing predicts from.
    pub(crate) nal_ref_idc: u8,
    /// `frame_num`: references since the IDR picture, modulo 256.
    pub(crate) frame_num: u32,
    /// `idr_pic_id`, on IDR pictures.
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb`: the order count, modulo 256.
    pub pic_order_cnt_lsb: u32,
    /// `slice_qp_delta`: the slice's QP less the PPS's.
    pub slice_qp_delta: i32,
}

impl SliceHeader {
    /// The NAL unit's header byte and the slice header's bits, unescaped,
    /// and how many bits that is: what a packed slice header holds.
    pub(crate) fn to_bits(&self, pps: &Pps) -> Result<(Vec<u8>, usize), Unsupported> {
        let mut w = BitWriter::new();
        let unit_type = if self.idr {
            nal_type::IDR
        } else {
            nal_type::SLICE
        };
        w.u(1, 0); // forbidden_zero_bit
        w.u(2, u32::from(self.nal_ref_idc));
        w.u(5, u32::from(unit_type));
        w.ue(0); // first_mb_in_slice
        w.ue(u32::from(self.slice_type.raw()));
        w.ue(0); // pic_parameter_set_id
        w.u(
            u32::from(LOG2_MAX_FRAME_NUM_MINUS4) + 4,
            self.frame_num & ((1 << (LOG2_MAX_FRAME_NUM_MINUS4 + 4)) - 1),
        );
        if self.idr {
            w.ue(self.idr_pic_id);
        }
        w.u(
            u32::from(LOG2_MAX_PIC_ORDER_CNT_LSB_MINUS4) + 4,
            self.pic_order_cnt_lsb,
        );
        if self.slice_type == SliceType::B {
            w.flag(true); // direct_spatial_mv_pred_flag
        }
        if self.slice_type != SliceType::I {
            w.flag(false); // num_ref_idx_active_override_flag
            w.flag(false); // ref_pic_list_modification_flag_l0
            if self.slice_type == SliceType::B {
                w.flag(false); // ref_pic_list_modification_flag_l1
            }
        }
        if self.nal_ref_idc != 0 {
            if self.idr {
                w.flag(false); // no_output_of_prior_pics_flag
                w.flag(false); // long_term_reference_flag
            } else {
                w.flag(false); // adaptive_ref_pic_marking_mode_flag
            }
        }
        if pps.entropy == Entropy::Cabac && self.slice_type != SliceType::I {
            w.ue(0); // cabac_init_idc
        }
        w.se(self.slice_qp_delta);
        w.ue(0); // disable_deblocking_filter_idc: on
        w.se(0); // slice_alpha_c0_offset_div2
        w.se(0); // slice_beta_offset_div2
        let bits = w
            .bit_len()
            .ok_or_else(|| Unsupported::overflow("the slice header's length"))?;
        Ok((w.into_bytes(), bits))
    }
}

/// The `avcC` record (AVCDecoderConfigurationRecord) for one SPS and one
/// PPS, both escaped NAL units, with four-byte NAL lengths. High profiles
/// carry the chroma format and bit depth fields after the PPS (5.3.3.1.2).
pub fn avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut record = Vec::new();
    let profile = sps.get(1).copied().unwrap_or(0);
    record.push(1); // configurationVersion
    record.push(profile);
    record.push(sps.get(2).copied().unwrap_or(0)); // profile_compatibility
    record.push(sps.get(3).copied().unwrap_or(0)); // AVCLevelIndication
    record.push(0xfc | 3); // lengthSizeMinusOne
    record.push(0xe0 | 1); // numOfSequenceParameterSets
    record.extend_from_slice(&u16::try_from(sps.len()).unwrap_or(0).to_be_bytes());
    record.extend_from_slice(sps);
    record.push(1); // numOfPictureParameterSets
    record.extend_from_slice(&u16::try_from(pps.len()).unwrap_or(0).to_be_bytes());
    record.extend_from_slice(pps);
    if matches!(profile, 100 | 110 | 122 | 144) {
        record.push(0xfc | 1); // chroma_format: 4:2:0
        record.push(0xf8); // bit_depth_luma_minus8
        record.push(0xf8); // bit_depth_chroma_minus8
        record.push(0); // numOfSequenceParameterSetExt
    }
    record
}

/// One row of Table A-1.
struct Level {
    level_idc: u8,
    /// MaxMBPS: macroblocks a second.
    max_mbps: u64,
    /// MaxFS: macroblocks a frame.
    max_fs: u64,
    /// MaxDpbMbs.
    max_dpb_mbs: u64,
    /// MaxBR, in units of 1000 bit/s (cpbBrVclFactor for Baseline and
    /// Main; High's is 1.25 times, Table A-2).
    max_br: u64,
}

/// Table A-1, but level 1b, which needs `constraint_set3_flag`.
const LEVELS: [Level; 19] = [
    level(10, 1_485, 99, 396, 64),
    level(11, 3_000, 396, 900, 192),
    level(12, 6_000, 396, 2_376, 384),
    level(13, 11_880, 396, 2_376, 768),
    level(20, 11_880, 396, 2_376, 2_000),
    level(21, 19_800, 792, 4_752, 4_000),
    level(22, 20_250, 1_620, 8_100, 4_000),
    level(30, 40_500, 1_620, 8_100, 10_000),
    level(31, 108_000, 3_600, 18_000, 14_000),
    level(32, 216_000, 5_120, 20_480, 20_000),
    level(40, 245_760, 8_192, 32_768, 20_000),
    level(41, 245_760, 8_192, 32_768, 50_000),
    level(42, 522_240, 8_704, 34_816, 50_000),
    level(50, 589_824, 22_080, 110_400, 135_000),
    level(51, 983_040, 36_864, 184_320, 240_000),
    level(52, 2_073_600, 36_864, 184_320, 240_000),
    level(60, 4_177_920, 139_264, 696_320, 240_000),
    level(61, 8_355_840, 139_264, 696_320, 480_000),
    level(62, 16_711_680, 139_264, 696_320, 800_000),
];

const fn level(level_idc: u8, max_mbps: u64, max_fs: u64, max_dpb_mbs: u64, max_br: u64) -> Level {
    Level {
        level_idc,
        max_mbps,
        max_fs,
        max_dpb_mbs,
        max_br,
    }
}

/// What a stream needs of its level.
#[derive(Clone, Copy, Debug)]
pub struct LevelNeeds {
    /// The profile: High's bitrate limits are 1.25 times the others' (Table
    /// A-2's `cpbBrVclFactor`).
    pub profile: Profile,
    /// The width in macroblocks.
    pub width_in_mbs: u32,
    /// The height in macroblocks.
    pub height_in_mbs: u32,
    /// Frames a second.
    pub frames_per_second: f64,
    /// The peak bitrate, if bounded.
    pub bits_per_second: Option<u32>,
    /// Frames the DPB holds.
    pub dpb_frames: u32,
}

impl LevelNeeds {
    #[expect(
        clippy::as_conversions,
        reason = "a macroblock count and rate compared as floats, the frame rate being one"
    )]
    fn fits(&self, level: &Level) -> bool {
        let w = u64::from(self.width_in_mbs);
        let h = u64::from(self.height_in_mbs);
        // A product past a u64 is past every limit.
        let Some(fs) = w.checked_mul(h) else {
            return false;
        };
        // A.3.1 (f): each dimension at most sqrt(8 * MaxFS), so its square
        // over 8, rounded up, at most MaxFS.
        let side = |d: u64| {
            d.checked_mul(d)
                .is_some_and(|sq| sq.div_ceil(8) <= level.max_fs)
        };
        // MaxMBPS bounds the macroblock rate (A.3.1 a, b).
        let mbps = (fs as f64) * self.frames_per_second;
        let factor = if self.profile == Profile::H264High {
            1250
        } else {
            1000
        };
        fs <= level.max_fs
            && side(w)
            && side(h)
            && mbps <= level.max_mbps as f64
            && (u64::from(self.dpb_frames).checked_mul(fs))
                .is_some_and(|mbs| mbs <= level.max_dpb_mbs)
            // At most MaxBR times the factor, so over the factor, rounded
            // up, at most MaxBR.
            && self
                .bits_per_second
                .is_none_or(|bps| u64::from(bps).div_ceil(factor) <= level.max_br)
    }
}

/// The lowest level that fits, or why none does.
pub fn choose_level(needs: &LevelNeeds) -> Result<u8, Unsupported> {
    LEVELS
        .iter()
        .find(|level| needs.fits(level))
        .map(|level| level.level_idc)
        .ok_or_else(|| Unsupported(format!("no H.264 level fits {needs:?}")))
}

/// Whether `level_idc` is a level of Table A-1 that fits.
pub(crate) fn level_fits(level_idc: u8, needs: &LevelNeeds) -> bool {
    LEVELS
        .iter()
        .any(|level| level.level_idc == level_idc && needs.fits(level))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::h264::{
        avcc::DecoderConfiguration,
        nal::NalUnit,
        pps::Pps as ParsedPps,
        slice::{SliceHeader as ParsedSlice, SliceType as ParsedType},
        sps::Sps as ParsedSps,
    };

    fn sps(profile: Profile) -> Sps {
        Sps {
            profile,
            level_idc: 40,
            max_num_ref_frames: 2,
            width_in_mbs: 120,
            height_in_mbs: 68,
            crop_right: 0,
            crop_bottom: 8,
            vui: Vui {
                sample_aspect_ratio: Some((4, 3)),
                full_range: true,
                colour: Some(ColourDescription {
                    primaries: 9,
                    transfer: 16,
                    matrix: 9,
                }),
                timing: Some((1001, 60_000)),
                max_num_reorder_frames: 1,
                max_dec_frame_buffering: 2,
            },
        }
    }

    #[test]
    fn our_sps_parses_back_as_written() -> Result<(), crate::h264::Error> {
        for profile in [
            Profile::H264ConstrainedBaseline,
            Profile::H264Main,
            Profile::H264High,
        ] {
            let written = sps(profile);
            let parsed = ParsedSps::parse(&written.to_nal().unwrap())?;
            assert_eq!(parsed.profile_idc, profile.profile_idc());
            assert_eq!(parsed.constraint_flags, written.constraint_flags());
            assert_eq!(parsed.level_idc, 40);
            assert_eq!(parsed.max_num_ref_frames, 2);
            assert_eq!((parsed.coded_width(), parsed.coded_height()), (1920, 1088));
            assert_eq!(parsed.crop_rect(), (0, 0, 1920, 1080));
            assert_eq!(parsed.pic_order_cnt_type, 0);
            assert_eq!(parsed.max_frame_num(), 256);
            assert_eq!(parsed.max_pic_order_cnt_lsb(), 256);
            let vui = parsed.vui.ok_or(crate::h264::Error::Truncated)?;
            assert_eq!(vui.sample_aspect_ratio, Some((4, 3)));
            assert!(vui.video_full_range_flag);
            assert_eq!(vui.colour_description, Some((9, 16, 9)));
            assert_eq!(vui.timing, Some((1001, 60_000)));
            assert_eq!(vui.max_num_reorder_frames, Some(1));
            assert_eq!(vui.max_dec_frame_buffering, Some(2));
        }
        Ok(())
    }

    #[test]
    fn an_sps_of_no_macroblocks_across_or_down_is_refused() {
        let across = Sps {
            width_in_mbs: 0,
            ..sps(Profile::H264Main)
        };
        let down = Sps {
            height_in_mbs: 0,
            ..sps(Profile::H264Main)
        };
        assert_eq!(
            across.to_nal(),
            Err(Unsupported::overflow("pic_width_in_mbs_minus1"))
        );
        assert_eq!(
            down.to_nal(),
            Err(Unsupported::overflow("pic_height_in_map_units_minus1"))
        );
    }

    #[test]
    fn our_pps_and_slice_headers_parse_back() -> Result<(), crate::h264::Error> {
        let written_sps = sps(Profile::H264High);
        let parsed_sps = Arc::new(ParsedSps::parse(&written_sps.to_nal().unwrap())?);
        let pps = Pps {
            profile: Profile::H264High,
            entropy: Entropy::Cabac,
            pic_init_qp: 30,
            transform_8x8_mode_flag: true,
        };
        let sps_lookup = |_| Some(parsed_sps.clone());
        let parsed_pps = Arc::new(ParsedPps::parse(&pps.to_nal().unwrap(), sps_lookup)?);
        assert!(parsed_pps.entropy_coding_mode_flag);
        assert!(parsed_pps.transform_8x8_mode_flag);
        assert_eq!(parsed_pps.pic_init_qp_minus26, 4);
        assert!(parsed_pps.deblocking_filter_control_present_flag);
        let cases = [
            (SliceType::I, true, 3, ParsedType::I),
            (SliceType::P, false, 2, ParsedType::P),
            (SliceType::B, false, 0, ParsedType::B),
        ];
        for (slice_type, idr, nal_ref_idc, parsed_type) in cases {
            let header = SliceHeader {
                slice_type,
                idr,
                nal_ref_idc,
                frame_num: 7,
                idr_pic_id: 3,
                pic_order_cnt_lsb: 14,
                slice_qp_delta: -4,
            };
            let (mut bytes, bits) = header.to_bits(&pps).unwrap();
            // Slice data after the header: some 1 bits, so the parser has
            // its stop bit beyond the header.
            assert!(bits <= bytes.len() * 8);
            bytes.extend_from_slice(&[0xff, 0x80]);
            let escaped = nal::escape(&bytes[..1], &bytes[1..]);
            let unit = NalUnit::new(&escaped)?;
            let parsed = ParsedSlice::parse(
                unit,
                |_| Some(parsed_pps.clone()),
                |_| Some(parsed_sps.clone()),
            )?;
            assert_eq!(parsed.slice_type, parsed_type);
            assert_eq!(parsed.idr, idr);
            assert_eq!(parsed.nal_ref_idc, nal_ref_idc);
            assert_eq!(parsed.frame_num, 7);
            assert_eq!(parsed.pic_order_cnt_lsb, 14);
            assert_eq!(parsed.slice_qp_delta, -4);
            assert_eq!(parsed.header_bits, bits);
            if idr {
                assert_eq!(parsed.idr_pic_id, 3);
            }
        }
        Ok(())
    }

    #[test]
    fn avcc_carries_the_parameter_sets() -> Result<(), crate::h264::Error> {
        let sps = sps(Profile::H264High).to_nal().unwrap();
        let pps = Pps {
            profile: Profile::H264High,
            entropy: Entropy::Cabac,
            pic_init_qp: 26,
            transform_8x8_mode_flag: true,
        }
        .to_nal()
        .unwrap();
        let record = avcc(&sps, &pps);
        let parsed = DecoderConfiguration::parse(&record)?;
        assert_eq!(parsed.profile_indication, 100);
        assert_eq!(record[3], 40, "AVCLevelIndication");
        assert_eq!(parsed.length_size, 4);
        assert_eq!(parsed.sequence_parameter_sets, [&sps[..]]);
        assert_eq!(parsed.picture_parameter_sets, [&pps[..]]);
        assert_eq!(&record[record.len() - 4..], [0xfd, 0xf8, 0xf8, 0]);
        Ok(())
    }

    #[test]
    fn levels_come_from_table_a_1() -> Result<(), Unsupported> {
        let needs = |w, h, fps, bps, dpb| LevelNeeds {
            profile: Profile::H264Main,
            width_in_mbs: w,
            height_in_mbs: h,
            frames_per_second: fps,
            bits_per_second: bps,
            dpb_frames: dpb,
        };
        assert_eq!(choose_level(&needs(120, 68, 30.0, Some(8_000_000), 2))?, 40);
        assert_eq!(choose_level(&needs(120, 68, 60.0, Some(8_000_000), 2))?, 42);
        assert_eq!(
            choose_level(&needs(120, 68, 30.0, Some(30_000_000), 2))?,
            41
        );
        assert_eq!(choose_level(&needs(80, 45, 30.0, None, 1))?, 31);
        assert_eq!(choose_level(&needs(240, 135, 30.0, None, 2))?, 51);
        assert_eq!(choose_level(&needs(8, 8, 30.0, None, 1))?, 11);
        assert_eq!(choose_level(&needs(8, 8, 15.0, None, 1))?, 10);
        assert!(choose_level(&needs(512, 512, 60.0, None, 2)).is_err());
        assert!(level_fits(40, &needs(120, 68, 30.0, None, 2)));
        assert!(!level_fits(31, &needs(120, 68, 30.0, None, 2)));
        Ok(())
    }
}
