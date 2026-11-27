//! The sequence header OBU (AV1 5.5 and 6.4): the stream's profile, its largest frame, the
//! tools its frames may use, its colour configuration and operating points.

use super::Error;
use super::bits::BitReader;

/// `SELECT_SCREEN_CONTENT_TOOLS` and `SELECT_INTEGER_MV`: the frame header says.
pub const SELECT: u8 = 2;

/// An operating point (5.5.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperatingPoint {
    /// `operating_point_idc`: the temporal layers in its low 8 bits, the spatial layers in
    /// the next 4.
    pub idc: u16,
    /// `seq_level_idx`.
    pub seq_level_idx: u8,
    /// `seq_tier`.
    pub seq_tier: bool,
    /// `decoder_model_present_for_this_op`.
    pub decoder_model_present: bool,
}

/// `decoder_model_info()` (5.5.4): the field lengths frame headers read by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecoderModelInfo {
    /// `buffer_delay_length_minus_1`.
    pub buffer_delay_length_minus_1: u8,
    /// `buffer_removal_time_length_minus_1`.
    pub buffer_removal_time_length_minus_1: u8,
    /// `frame_presentation_time_length_minus_1`.
    pub frame_presentation_time_length_minus_1: u8,
}

/// `color_config()` (5.5.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorConfig {
    /// `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// `mono_chrome`.
    pub mono_chrome: bool,
    /// `color_primaries`.
    pub color_primaries: u8,
    /// `transfer_characteristics`.
    pub transfer_characteristics: u8,
    /// `matrix_coefficients`.
    pub matrix_coefficients: u8,
    /// `color_range`: full range where set.
    pub color_range: bool,
    /// `subsampling_x`.
    pub subsampling_x: bool,
    /// `subsampling_y`.
    pub subsampling_y: bool,
    /// `chroma_sample_position`.
    pub chroma_sample_position: u8,
    /// `separate_uv_delta_q`.
    pub separate_uv_delta_q: bool,
}

/// `frame_id_numbers_present_flag`'s lengths.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameIdNumbers {
    /// `delta_frame_id_length_minus_2`.
    pub delta_frame_id_length_minus_2: u8,
    /// `additional_frame_id_length_minus_1`.
    pub additional_frame_id_length_minus_1: u8,
}

impl FrameIdNumbers {
    /// `idLen`, which a sequence header holds to 16 at most.
    pub fn id_len(&self) -> u32 {
        // At most 7 + 15 + 3.
        u32::from(self.additional_frame_id_length_minus_1)
            .saturating_add(u32::from(self.delta_frame_id_length_minus_2))
            .saturating_add(3)
    }

    /// The bits of `delta_frame_id_minus_1`.
    pub fn delta_len(&self) -> u32 {
        u32::from(self.delta_frame_id_length_minus_2).saturating_add(2)
    }
}

/// A sequence header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceHeader {
    /// `seq_profile`: 0 (Main), 1 (High) or 2 (Professional).
    pub seq_profile: u8,
    /// `still_picture`.
    pub still_picture: bool,
    /// `reduced_still_picture_header`.
    pub reduced_still_picture_header: bool,
    /// `decoder_model_info()`, where `decoder_model_info_present_flag` is set.
    pub decoder_model_info: Option<DecoderModelInfo>,
    /// `equal_picture_interval` of `timing_info()`; false where there is none.
    pub equal_picture_interval: bool,
    /// The operating points, one at least.
    pub operating_points: Vec<OperatingPoint>,
    /// `frame_width_bits_minus_1` + 1.
    pub frame_width_bits: u32,
    /// `frame_height_bits_minus_1` + 1.
    pub frame_height_bits: u32,
    /// `max_frame_width_minus_1` + 1.
    pub max_frame_width: u32,
    /// `max_frame_height_minus_1` + 1.
    pub max_frame_height: u32,
    /// The frame id lengths, where `frame_id_numbers_present_flag` is set.
    pub frame_id_numbers: Option<FrameIdNumbers>,
    /// `use_128x128_superblock`.
    pub use_128x128_superblock: bool,
    /// `enable_filter_intra`.
    pub enable_filter_intra: bool,
    /// `enable_intra_edge_filter`.
    pub enable_intra_edge_filter: bool,
    /// `enable_interintra_compound`.
    pub enable_interintra_compound: bool,
    /// `enable_masked_compound`.
    pub enable_masked_compound: bool,
    /// `enable_warped_motion`.
    pub enable_warped_motion: bool,
    /// `enable_dual_filter`.
    pub enable_dual_filter: bool,
    /// `enable_order_hint`.
    pub enable_order_hint: bool,
    /// `enable_jnt_comp`.
    pub enable_jnt_comp: bool,
    /// `enable_ref_frame_mvs`.
    pub enable_ref_frame_mvs: bool,
    /// `seq_force_screen_content_tools`: 0, 1 or [`SELECT`].
    pub seq_force_screen_content_tools: u8,
    /// `seq_force_integer_mv`: 0, 1 or [`SELECT`].
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits`: 0 where `enable_order_hint` is clear, else 1 to 8.
    pub order_hint_bits: u32,
    /// `enable_superres`.
    pub enable_superres: bool,
    /// `enable_cdef`.
    pub enable_cdef: bool,
    /// `enable_restoration`.
    pub enable_restoration: bool,
    /// `color_config()`.
    pub color: ColorConfig,
    /// `film_grain_params_present`.
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Reads a sequence header OBU's payload.
    pub fn parse(payload: &[u8]) -> Result<Self, Error> {
        let mut r = BitReader::new(payload);
        let mut seq = Self {
            seq_profile: r.u8(3)?,
            still_picture: r.bit()?,
            reduced_still_picture_header: r.bit()?,
            ..Self::default()
        };
        if seq.seq_profile > 2 {
            return Err(Error::OutOfRange("seq_profile"));
        }
        if seq.reduced_still_picture_header {
            seq.operating_points.push(OperatingPoint {
                seq_level_idx: r.u8(5)?,
                ..OperatingPoint::default()
            });
        } else {
            if r.bit()? {
                // timing_info()
                r.f(32)?;
                r.f(32)?;
                seq.equal_picture_interval = r.bit()?;
                if seq.equal_picture_interval {
                    r.uvlc()?;
                }
                if r.bit()? {
                    let buffer_delay_length_minus_1 = r.u8(5)?;
                    r.f(32)?;
                    seq.decoder_model_info = Some(DecoderModelInfo {
                        buffer_delay_length_minus_1,
                        buffer_removal_time_length_minus_1: r.u8(5)?,
                        frame_presentation_time_length_minus_1: r.u8(5)?,
                    });
                }
            }
            let initial_display_delay_present = r.bit()?;
            let count = r.f(5)?.saturating_add(1);
            for _ in 0..count {
                let idc = r.u16(12)?;
                let seq_level_idx = r.u8(5)?;
                let seq_tier = seq_level_idx > 7 && r.bit()?;
                let mut decoder_model_present = false;
                if let Some(model) = seq.decoder_model_info {
                    decoder_model_present = r.bit()?;
                    if decoder_model_present {
                        // operating_parameters_info(): two delays and low_delay_mode_flag.
                        let n = u32::from(model.buffer_delay_length_minus_1).saturating_add(1);
                        r.f(n)?;
                        r.f(n)?;
                        r.bit()?;
                    }
                }
                if initial_display_delay_present && r.bit()? {
                    r.f(4)?;
                }
                seq.operating_points.push(OperatingPoint {
                    idc,
                    seq_level_idx,
                    seq_tier,
                    decoder_model_present,
                });
            }
        }
        // Each at most 16 bits, its value at most 2^16 - 1: none saturates.
        seq.frame_width_bits = r.f(4)?.saturating_add(1);
        seq.frame_height_bits = r.f(4)?.saturating_add(1);
        seq.max_frame_width = r.f(seq.frame_width_bits)?.saturating_add(1);
        seq.max_frame_height = r.f(seq.frame_height_bits)?.saturating_add(1);
        if !seq.reduced_still_picture_header && r.bit()? {
            let ids = FrameIdNumbers {
                delta_frame_id_length_minus_2: r.u8(4)?,
                additional_frame_id_length_minus_1: r.u8(3)?,
            };
            // 6.8.2: idLen is at most 16.
            if ids.id_len() > 16 {
                return Err(Error::OutOfRange("additional_frame_id_length_minus_1"));
            }
            seq.frame_id_numbers = Some(ids);
        }
        seq.use_128x128_superblock = r.bit()?;
        seq.enable_filter_intra = r.bit()?;
        seq.enable_intra_edge_filter = r.bit()?;
        if seq.reduced_still_picture_header {
            seq.seq_force_screen_content_tools = SELECT;
            seq.seq_force_integer_mv = SELECT;
        } else {
            seq.enable_interintra_compound = r.bit()?;
            seq.enable_masked_compound = r.bit()?;
            seq.enable_warped_motion = r.bit()?;
            seq.enable_dual_filter = r.bit()?;
            seq.enable_order_hint = r.bit()?;
            if seq.enable_order_hint {
                seq.enable_jnt_comp = r.bit()?;
                seq.enable_ref_frame_mvs = r.bit()?;
            }
            seq.seq_force_screen_content_tools = if r.bit()? { SELECT } else { r.u8(1)? };
            seq.seq_force_integer_mv = if seq.seq_force_screen_content_tools > 0 {
                if r.bit()? { SELECT } else { r.u8(1)? }
            } else {
                SELECT
            };
            if seq.enable_order_hint {
                seq.order_hint_bits = r.f(3)?.saturating_add(1);
            }
        }
        seq.enable_superres = r.bit()?;
        seq.enable_cdef = r.bit()?;
        seq.enable_restoration = r.bit()?;
        seq.color = color_config(&mut r, seq.seq_profile)?;
        seq.film_grain_params_present = r.bit()?;
        Ok(seq)
    }

    /// `NumPlanes`.
    pub fn num_planes(&self) -> usize {
        if self.color.mono_chrome { 1 } else { 3 }
    }

    /// `OperatingPointIdc` of operating point 0, the one this decoder decodes
    /// (`choose_operating_point()` returning 0, 6.4.1).
    pub fn operating_point_idc(&self) -> u16 {
        self.operating_points.first().map_or(0, |op| op.idc)
    }
}

fn color_config(r: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig, Error> {
    let high_bitdepth = r.bit()?;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        if r.bit()? { 12 } else { 10 }
    } else if high_bitdepth {
        10
    } else {
        8
    };
    let mono_chrome = seq_profile != 1 && r.bit()?;
    let mut color = ColorConfig {
        bit_depth,
        mono_chrome,
        // CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED.
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
        ..ColorConfig::default()
    };
    if r.bit()? {
        color.color_primaries = r.u8(8)?;
        color.transfer_characteristics = r.u8(8)?;
        color.matrix_coefficients = r.u8(8)?;
    }
    if mono_chrome {
        color.color_range = r.bit()?;
        color.subsampling_x = true;
        color.subsampling_y = true;
        return Ok(color);
    }
    // CP_BT_709, TC_SRGB, MC_IDENTITY: 4:4:4 at full range.
    if color.color_primaries == 1
        && color.transfer_characteristics == 13
        && color.matrix_coefficients == 0
    {
        color.color_range = true;
        if seq_profile != 1 && !(seq_profile == 2 && bit_depth == 12) {
            // 6.4.2: sRGB 4:4:4 is profile 1's, or profile 2's at 12 bits.
            return Err(Error::OutOfRange("seq_profile"));
        }
    } else {
        color.color_range = r.bit()?;
        match seq_profile {
            0 => {
                color.subsampling_x = true;
                color.subsampling_y = true;
            }
            1 => {}
            _ if bit_depth == 12 => {
                color.subsampling_x = r.bit()?;
                color.subsampling_y = color.subsampling_x && r.bit()?;
            }
            _ => color.subsampling_x = true,
        }
        if color.subsampling_x && color.subsampling_y {
            color.chroma_sample_position = r.u8(2)?;
        }
    }
    color.separate_uv_delta_q = r.bit()?;
    Ok(color)
}
