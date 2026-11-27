//! The frame header (AV1 5.9 and 6.8): `uncompressed_header()`, with the processes it runs
//! that set what a stateless decoder is given: `set_frame_refs()` (7.8), superres and the
//! frame's size, `tile_info()`, the segmentation, loop filter and global motion parameters
//! loaded from the primary reference frame (`load_previous()`, 7.20 and 7.21), skip mode,
//! the global motion parameters' shear check (7.11.3.6) and film grain.
//!
//! What a frame header reads of the frames before it is each reference slot's
//! [`ReferenceState`], which the decoder keeps (the reference frame update process, 7.20).
//! Counts and sizes are held to the bounds the specification sets as they are read: a frame
//! is no larger than its sequence header's largest, has at most 64 tile columns and rows,
//! and its film grain points fit the arrays the hardware takes.

use super::Error;
use super::bits::BitReader;
use super::obu::Extension;
use super::sequence::{SELECT, SequenceHeader};

/// `NUM_REF_FRAMES`: the reference slots.
pub const NUM_REF_FRAMES: usize = 8;
/// `REFS_PER_FRAME`: the references a frame names, `LAST_FRAME` to `ALTREF_FRAME`.
pub const REFS_PER_FRAME: usize = 7;
/// `PRIMARY_REF_NONE`.
pub const PRIMARY_REF_NONE: u8 = 7;
/// `MAX_TILE_COLS` and `MAX_TILE_ROWS`.
pub const MAX_TILE_COLS: u32 = 64;
const MAX_TILE_WIDTH: u32 = 4096;
const MAX_TILE_AREA: u32 = 4096 * 2304;
const SUPERRES_NUM: u32 = 8;
const SUPERRES_DENOM_MIN: u32 = 9;
const WARPEDMODEL_PREC_BITS: u32 = 16;
/// `SEG_LVL_ALT_Q`.
const SEG_LVL_ALT_Q: usize = 0;
/// `Segmentation_Feature_Bits`, `_Signed` and `_Max` (5.9.14).
const SEGMENTATION_FEATURES: [(u32, bool, i16); 8] = [
    (8, true, 255),
    (6, true, 63),
    (6, true, 63),
    (6, true, 63),
    (6, true, 63),
    (3, false, 7),
    (0, false, 0),
    (0, false, 0),
];
/// The loop filter's reference deltas after `setup_past_independence()` (7.20), for
/// `INTRA_FRAME` to `ALTREF_FRAME`.
const DEFAULT_REF_DELTAS: [i8; 8] = [1, 0, 0, 0, -1, 0, -1, -1];
/// `Remap_Lr_Type` (6.10.15): `lr_type` to `FrameRestorationType`.
const REMAP_LR_TYPE: [u8; 4] = [0, 3, 1, 2];

/// `frame_type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrameType {
    /// `KEY_FRAME`.
    #[default]
    Key,
    /// `INTER_FRAME`.
    Inter,
    /// `INTRA_ONLY_FRAME`.
    IntraOnly,
    /// `SWITCH_FRAME`.
    Switch,
}

impl FrameType {
    fn from_code(code: u8) -> Self {
        match code & 3 {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }

    /// Its code.
    pub fn code(self) -> u8 {
        match self {
            Self::Key => 0,
            Self::Inter => 1,
            Self::IntraOnly => 2,
            Self::Switch => 3,
        }
    }

    /// `FrameIsIntra`.
    pub fn is_intra(self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

/// A frame's size (5.9.5 to 5.9.8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSize {
    /// `UpscaledWidth`: the frame's width once superres has upscaled it, what is output.
    pub upscaled_width: u32,
    /// `FrameWidth`: the width it is coded at.
    pub frame_width: u32,
    /// `FrameHeight`.
    pub frame_height: u32,
    /// `RenderWidth`.
    pub render_width: u32,
    /// `RenderHeight`.
    pub render_height: u32,
    /// `use_superres`.
    pub use_superres: bool,
    /// `coded_denom`, where `use_superres` is set.
    pub coded_denom: u8,
    /// `SuperresDenom`: 8 where superres is off.
    pub superres_denom: u32,
}

impl FrameSize {
    /// `MiCols`.
    pub fn mi_cols(&self) -> u32 {
        // FrameWidth is at most 2^16.
        (self.frame_width.saturating_add(7) >> 3).saturating_mul(2)
    }

    /// `MiRows`.
    pub fn mi_rows(&self) -> u32 {
        (self.frame_height.saturating_add(7) >> 3).saturating_mul(2)
    }
}

/// `tile_info()` (5.9.15).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    /// `uniform_tile_spacing_flag`.
    pub uniform: bool,
    /// `TileColsLog2`.
    pub cols_log2: u32,
    /// `TileRowsLog2`.
    pub rows_log2: u32,
    /// Each tile column's width in superblocks: `TileCols` of them, 64 at most.
    pub col_widths_sb: Vec<u32>,
    /// Each tile row's height in superblocks: `TileRows` of them, 64 at most.
    pub row_heights_sb: Vec<u32>,
    /// `context_update_tile_id`, below the tile count.
    pub context_update_tile_id: u32,
    /// `TileSizeBytes`: 1 to 4, where there is more than one tile.
    pub tile_size_bytes: u32,
}

impl TileInfo {
    /// `TileCols`.
    pub fn cols(&self) -> u32 {
        u32::try_from(self.col_widths_sb.len()).unwrap_or(u32::MAX)
    }

    /// `TileRows`.
    pub fn rows(&self) -> u32 {
        u32::try_from(self.row_heights_sb.len()).unwrap_or(u32::MAX)
    }

    /// `NumTiles`, at most 4096.
    pub fn num_tiles(&self) -> u32 {
        self.cols().saturating_mul(self.rows())
    }
}

/// `quantization_params()` (5.9.12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    /// `base_q_idx`.
    pub base_q_idx: u8,
    /// `DeltaQYDc`.
    pub delta_q_y_dc: i8,
    /// `DeltaQUDc`.
    pub delta_q_u_dc: i8,
    /// `DeltaQUAc`.
    pub delta_q_u_ac: i8,
    /// `DeltaQVDc`.
    pub delta_q_v_dc: i8,
    /// `DeltaQVAc`.
    pub delta_q_v_ac: i8,
    /// `using_qmatrix`.
    pub using_qmatrix: bool,
    /// `qm_y`.
    pub qm_y: u8,
    /// `qm_u`.
    pub qm_u: u8,
    /// `qm_v`.
    pub qm_v: u8,
}

/// `FeatureEnabled` and `FeatureData` (5.9.14), which a frame loads from its primary
/// reference frame where it does not code them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentationFeatures {
    /// `FeatureEnabled[segment][feature]`.
    pub enabled: [[bool; 8]; 8],
    /// `FeatureData[segment][feature]`, clipped as 5.9.14 clips it.
    pub data: [[i16; 8]; 8],
}

/// `segmentation_params()` (5.9.14).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    /// `segmentation_enabled`.
    pub enabled: bool,
    /// `segmentation_update_map`.
    pub update_map: bool,
    /// `segmentation_temporal_update`.
    pub temporal_update: bool,
    /// `segmentation_update_data`.
    pub update_data: bool,
    /// The features in force.
    pub features: SegmentationFeatures,
}

/// `loop_filter_params()` (5.9.11).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopFilter {
    /// `loop_filter_level[0..3]`.
    pub level: [u8; 4],
    /// `loop_filter_sharpness`.
    pub sharpness: u8,
    /// `loop_filter_delta_enabled`.
    pub delta_enabled: bool,
    /// `loop_filter_delta_update`.
    pub delta_update: bool,
    /// `loop_filter_ref_deltas`, as in force: coded, or loaded from the primary reference.
    pub ref_deltas: [i8; 8],
    /// `loop_filter_mode_deltas`, likewise.
    pub mode_deltas: [i8; 2],
}

/// `cdef_params()` (5.9.19), each strength as coded: a secondary strength of 3 is 4 to the
/// filter, and the hardware's structures take the coded value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cdef {
    /// `cdef_damping_minus_3`.
    pub damping_minus_3: u8,
    /// `cdef_bits`.
    pub bits: u8,
    /// `cdef_y_pri_strength`.
    pub y_pri_strength: [u8; 8],
    /// `cdef_y_sec_strength`, as coded.
    pub y_sec_strength: [u8; 8],
    /// `cdef_uv_pri_strength`.
    pub uv_pri_strength: [u8; 8],
    /// `cdef_uv_sec_strength`, as coded.
    pub uv_sec_strength: [u8; 8],
}

/// `lr_params()` (5.9.20).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopRestoration {
    /// `FrameRestorationType` of each plane: `RESTORE_NONE` 0, `RESTORE_WIENER` 1,
    /// `RESTORE_SGRPROJ` 2, `RESTORE_SWITCHABLE` 3.
    pub frame_restoration_type: [u8; 3],
    /// `lr_unit_shift`, once `lr_unit_extra_shift` or the 128x128 superblock's increment is
    /// added: 0 to 2.
    pub unit_shift: u8,
    /// `lr_uv_shift`.
    pub uv_shift: u8,
}

/// A reference frame's global motion parameters (5.9.24).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalMotion {
    /// `is_global`.
    pub is_global: bool,
    /// `GmType`: `IDENTITY` 0, `TRANSLATION` 1, `ROTZOOM` 2, `AFFINE` 3.
    pub gm_type: u8,
    /// `gm_params`.
    pub params: [i32; 6],
    /// Whether the setup shear process finds them valid (`warpValid`, 7.11.3.6).
    pub warp_valid: bool,
}

/// `gm_params` of the identity transform.
const IDENTITY_PARAMS: [i32; 6] = [
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
    0,
    0,
    1 << WARPEDMODEL_PREC_BITS,
];

impl Default for GlobalMotion {
    fn default() -> Self {
        Self {
            is_global: false,
            gm_type: 0,
            params: IDENTITY_PARAMS,
            warp_valid: true,
        }
    }
}

/// `film_grain_params()` (5.9.30), as in force: coded, or loaded from a reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilmGrainParams {
    /// `apply_grain`.
    pub apply_grain: bool,
    /// `grain_seed`.
    pub grain_seed: u16,
    /// `update_grain`.
    pub update_grain: bool,
    /// `num_y_points`, at most 14.
    pub num_y_points: u8,
    /// `point_y_value`.
    pub point_y_value: [u8; 14],
    /// `point_y_scaling`.
    pub point_y_scaling: [u8; 14],
    /// `chroma_scaling_from_luma`.
    pub chroma_scaling_from_luma: bool,
    /// `num_cb_points`, at most 10.
    pub num_cb_points: u8,
    /// `point_cb_value`.
    pub point_cb_value: [u8; 10],
    /// `point_cb_scaling`.
    pub point_cb_scaling: [u8; 10],
    /// `num_cr_points`, at most 10.
    pub num_cr_points: u8,
    /// `point_cr_value`.
    pub point_cr_value: [u8; 10],
    /// `point_cr_scaling`.
    pub point_cr_scaling: [u8; 10],
    /// `grain_scaling_minus_8`.
    pub grain_scaling_minus_8: u8,
    /// `ar_coeff_lag`.
    pub ar_coeff_lag: u8,
    /// `ar_coeffs_y_plus_128`, less 128.
    pub ar_coeffs_y: [i8; 24],
    /// `ar_coeffs_cb_plus_128`, less 128.
    pub ar_coeffs_cb: [i8; 25],
    /// `ar_coeffs_cr_plus_128`, less 128.
    pub ar_coeffs_cr: [i8; 25],
    /// `ar_coeff_shift_minus_6`.
    pub ar_coeff_shift_minus_6: u8,
    /// `grain_scale_shift`.
    pub grain_scale_shift: u8,
    /// `cb_mult`.
    pub cb_mult: u8,
    /// `cb_luma_mult`.
    pub cb_luma_mult: u8,
    /// `cb_offset`.
    pub cb_offset: u16,
    /// `cr_mult`.
    pub cr_mult: u8,
    /// `cr_luma_mult`.
    pub cr_luma_mult: u8,
    /// `cr_offset`.
    pub cr_offset: u16,
    /// `overlap_flag`.
    pub overlap_flag: bool,
    /// `clip_to_restricted_range`.
    pub clip_to_restricted_range: bool,
}

/// What a reference slot holds that later frame headers read (7.20): `RefValid`,
/// `RefFrameId`, `RefFrameType`, `RefOrderHint`, its sizes, and the parameters
/// `load_previous()` and `load_grain_params()` load.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceState {
    /// `RefValid`.
    pub valid: bool,
    /// `RefFrameId`.
    pub frame_id: u32,
    /// `RefFrameType`.
    pub frame_type: FrameType,
    /// Whether the frame may be shown by `show_existing_frame`: its `showable_frame`, cleared
    /// once a key frame has been shown so.
    pub showable: bool,
    /// `RefOrderHint`.
    pub order_hint: u8,
    /// `RefUpscaledWidth`, `RefFrameWidth`, `RefFrameHeight`, `RefRenderWidth` and
    /// `RefRenderHeight`.
    pub size: FrameSize,
    /// `RefBitDepth`.
    pub bit_depth: u8,
    /// The loop filter's deltas (`save_loop_filter_params()`).
    pub loop_filter_ref_deltas: [i8; 8],
    /// Likewise.
    pub loop_filter_mode_deltas: [i8; 2],
    /// `save_segmentation_params()`.
    pub segmentation: SegmentationFeatures,
    /// `SavedGmParams`, for `LAST_FRAME` to `ALTREF_FRAME` at 1 to 7.
    pub gm_params: [[i32; 6]; 8],
    /// `save_grain_params()`.
    pub film_grain: FilmGrainParams,
}

/// A frame header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHeader {
    /// The OBU extension's `temporal_id`, 0 without one.
    pub temporal_id: u8,
    /// The OBU extension's `spatial_id`, 0 without one.
    pub spatial_id: u8,
    /// `show_existing_frame`.
    pub show_existing_frame: bool,
    /// `frame_to_show_map_idx`.
    pub frame_to_show_map_idx: u8,
    /// `frame_type`; a shown frame's, where `show_existing_frame` is set.
    pub frame_type: FrameType,
    /// `show_frame`.
    pub show_frame: bool,
    /// `showable_frame`.
    pub showable_frame: bool,
    /// `error_resilient_mode`.
    pub error_resilient_mode: bool,
    /// `disable_cdf_update`.
    pub disable_cdf_update: bool,
    /// `allow_screen_content_tools`.
    pub allow_screen_content_tools: bool,
    /// `force_integer_mv`.
    pub force_integer_mv: bool,
    /// `current_frame_id`.
    pub current_frame_id: u32,
    /// `frame_size_override_flag`.
    pub frame_size_override_flag: bool,
    /// `OrderHint`.
    pub order_hint: u8,
    /// `primary_ref_frame`.
    pub primary_ref_frame: u8,
    /// `refresh_frame_flags`.
    pub refresh_frame_flags: u8,
    /// `ref_frame_idx`, for `LAST_FRAME` to `ALTREF_FRAME`.
    pub ref_frame_idx: [u8; 7],
    /// The frame's size.
    pub size: FrameSize,
    /// `allow_intrabc`.
    pub allow_intrabc: bool,
    /// `allow_high_precision_mv`.
    pub allow_high_precision_mv: bool,
    /// `interpolation_filter`: 4 is `SWITCHABLE`.
    pub interpolation_filter: u8,
    /// `is_motion_mode_switchable`.
    pub is_motion_mode_switchable: bool,
    /// `use_ref_frame_mvs`.
    pub use_ref_frame_mvs: bool,
    /// `OrderHints`, for `LAST_FRAME` to `ALTREF_FRAME` at 1 to 7.
    pub order_hints: [u8; 8],
    /// `disable_frame_end_update_cdf`.
    pub disable_frame_end_update_cdf: bool,
    /// `tile_info()`.
    pub tile_info: TileInfo,
    /// `quantization_params()`.
    pub quantization: Quantization,
    /// `segmentation_params()`.
    pub segmentation: Segmentation,
    /// `delta_q_present`.
    pub delta_q_present: bool,
    /// `delta_q_res`.
    pub delta_q_res: u8,
    /// `delta_lf_present`.
    pub delta_lf_present: bool,
    /// `delta_lf_res`.
    pub delta_lf_res: u8,
    /// `delta_lf_multi`.
    pub delta_lf_multi: bool,
    /// `CodedLossless`.
    pub coded_lossless: bool,
    /// `AllLossless`.
    pub all_lossless: bool,
    /// `loop_filter_params()`.
    pub loop_filter: LoopFilter,
    /// `cdef_params()`.
    pub cdef: Cdef,
    /// `lr_params()`.
    pub loop_restoration: LoopRestoration,
    /// `TxMode`: `ONLY_4X4` 0, `TX_MODE_LARGEST` 1, `TX_MODE_SELECT` 2.
    pub tx_mode: u8,
    /// `reference_select`.
    pub reference_select: bool,
    /// `skip_mode_present`.
    pub skip_mode_present: bool,
    /// `SkipModeFrame`, where skip mode is allowed: two of `LAST_FRAME` to `ALTREF_FRAME`.
    pub skip_mode_frame: [u8; 2],
    /// `allow_warped_motion`.
    pub allow_warped_motion: bool,
    /// `reduced_tx_set`.
    pub reduced_tx_set: bool,
    /// The global motion parameters of `LAST_FRAME` to `ALTREF_FRAME`, at 1 to 7.
    pub global_motion: [GlobalMotion; 8],
    /// The film grain parameters in force.
    pub film_grain: FilmGrainParams,
    /// `RefValid` of each slot once the header is read: a shown key frame, frame ids and an
    /// error resilient frame's `ref_order_hint` may clear it.
    pub ref_valid: [bool; 8],
    /// `RefOrderHint` of each slot once the header is read.
    pub ref_order_hint: [u8; 8],
}

/// Reads `uncompressed_header()` from `r`, for the OBU whose extension is `extension`, over
/// the reference slots `refs`.
pub(crate) fn parse(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    refs: &[ReferenceState; NUM_REF_FRAMES],
    extension: Option<Extension>,
) -> Result<FrameHeader, Error> {
    let mut h = FrameHeader {
        temporal_id: extension.map_or(0, |e| e.temporal_id),
        spatial_id: extension.map_or(0, |e| e.spatial_id),
        ref_valid: refs.each_ref().map(|slot| slot.valid),
        ref_order_hint: refs.each_ref().map(|slot| slot.order_hint),
        ..FrameHeader::default()
    };
    let id_len = seq.frame_id_numbers.map(|ids| ids.id_len());
    if seq.reduced_still_picture_header {
        h.frame_type = FrameType::Key;
        h.show_frame = true;
    } else {
        h.show_existing_frame = r.bit()?;
        if h.show_existing_frame {
            h.frame_to_show_map_idx = r.u8(3)?;
            temporal_point_info(r, seq)?;
            let shown = slot(refs, h.frame_to_show_map_idx)?;
            if let Some(len) = id_len {
                let display_frame_id = r.f(len)?;
                // 6.8.2: the frame shown is the one whose id is coded.
                if shown.valid && display_frame_id != shown.frame_id {
                    return Err(Error::OutOfRange("display_frame_id"));
                }
            }
            if !shown.valid {
                return Err(Error::MissingReference);
            }
            // 6.8.2: only a showable frame is shown again, and a key frame once at most.
            if !shown.showable {
                return Err(Error::OutOfRange("frame_to_show_map_idx"));
            }
            h.frame_type = shown.frame_type;
            if h.frame_type == FrameType::Key {
                h.refresh_frame_flags = 0xff;
            }
            if seq.film_grain_params_present {
                h.film_grain = shown.film_grain;
            }
            h.size = shown.size;
            h.order_hint = shown.order_hint;
            h.current_frame_id = shown.frame_id;
            h.show_frame = true;
            return Ok(h);
        }
        h.frame_type = FrameType::from_code(r.u8(2)?);
        h.show_frame = r.bit()?;
        if h.show_frame {
            temporal_point_info(r, seq)?;
        }
        h.showable_frame = if h.show_frame {
            h.frame_type != FrameType::Key
        } else {
            r.bit()?
        };
        h.error_resilient_mode = h.frame_type == FrameType::Switch
            || (h.frame_type == FrameType::Key && h.show_frame)
            || r.bit()?;
    }
    let intra = h.frame_type.is_intra();
    if h.frame_type == FrameType::Key && h.show_frame {
        h.ref_valid = [false; 8];
        h.ref_order_hint = [0; 8];
    }
    h.disable_cdf_update = r.bit()?;
    h.allow_screen_content_tools = if seq.seq_force_screen_content_tools == SELECT {
        r.bit()?
    } else {
        seq.seq_force_screen_content_tools == 1
    };
    // Read where screen content tools are on, and 1 for an intra frame whatever was read.
    let force_integer_mv = h.allow_screen_content_tools
        && if seq.seq_force_integer_mv == SELECT {
            r.bit()?
        } else {
            seq.seq_force_integer_mv == 1
        };
    h.force_integer_mv = intra || force_integer_mv;
    if let (Some(len), Some(ids)) = (id_len, seq.frame_id_numbers) {
        h.current_frame_id = r.f(len)?;
        mark_ref_frames(
            &mut h.ref_valid,
            refs,
            h.current_frame_id,
            len,
            ids.delta_len(),
        );
    }
    h.frame_size_override_flag =
        h.frame_type == FrameType::Switch || (!seq.reduced_still_picture_header && r.bit()?);
    h.order_hint = r.u8(seq.order_hint_bits)?;
    h.primary_ref_frame = if intra || h.error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        r.u8(3)?
    };
    if let Some(model) = seq.decoder_model_info
        && r.bit()?
    {
        for op in &seq.operating_points {
            if !op.decoder_model_present {
                continue;
            }
            let in_temporal_layer = (op.idc >> h.temporal_id) & 1 == 1;
            let in_spatial_layer = (op.idc >> h.spatial_id.saturating_add(8)) & 1 == 1;
            if op.idc == 0 || (in_temporal_layer && in_spatial_layer) {
                // buffer_removal_time
                r.f(u32::from(model.buffer_removal_time_length_minus_1).saturating_add(1))?;
            }
        }
    }
    h.refresh_frame_flags =
        if h.frame_type == FrameType::Switch || (h.frame_type == FrameType::Key && h.show_frame) {
            0xff
        } else {
            r.u8(8)?
        };
    // 6.8.2: an intra-only frame does not refresh every slot.
    if h.frame_type == FrameType::IntraOnly && h.refresh_frame_flags == 0xff {
        return Err(Error::OutOfRange("refresh_frame_flags"));
    }
    if (!intra || h.refresh_frame_flags != 0xff) && h.error_resilient_mode && seq.enable_order_hint
    {
        for (valid, hint) in h.ref_valid.iter_mut().zip(h.ref_order_hint.iter_mut()) {
            let ref_order_hint = r.u8(seq.order_hint_bits)?;
            if ref_order_hint != *hint {
                *valid = false;
                *hint = ref_order_hint;
            }
        }
    }
    if intra {
        frame_size(r, seq, &mut h)?;
        render_size(r, &mut h)?;
        h.allow_intrabc = h.allow_screen_content_tools
            && h.size.upscaled_width == h.size.frame_width
            && r.bit()?;
    } else {
        inter_references(r, seq, refs, id_len, &mut h)?;
    }
    h.disable_frame_end_update_cdf =
        seq.reduced_still_picture_header || h.disable_cdf_update || r.bit()?;
    let previous = if h.primary_ref_frame == PRIMARY_REF_NONE {
        None
    } else {
        let index = *h
            .ref_frame_idx
            .get(usize::from(h.primary_ref_frame))
            .ok_or(Error::OutOfRange("primary_ref_frame"))?;
        let previous = slot(refs, index)?;
        if !valid(&h.ref_valid, index) {
            return Err(Error::MissingReference);
        }
        Some(previous)
    };
    tile_info(r, seq, &mut h)?;
    quantization_params(r, seq, &mut h)?;
    segmentation_params(r, &mut h, previous)?;
    h.delta_q_present = h.quantization.base_q_idx > 0 && r.bit()?;
    if h.delta_q_present {
        h.delta_q_res = r.u8(2)?;
        h.delta_lf_present = !h.allow_intrabc && r.bit()?;
        if h.delta_lf_present {
            h.delta_lf_res = r.u8(2)?;
            h.delta_lf_multi = r.bit()?;
        }
    }
    h.coded_lossless = (0..8).all(|segment| lossless(&h, segment));
    h.all_lossless = h.coded_lossless && h.size.frame_width == h.size.upscaled_width;
    loop_filter_params(r, seq, &mut h, previous)?;
    cdef_params(r, seq, &mut h)?;
    lr_params(r, seq, &mut h)?;
    h.tx_mode = if h.coded_lossless {
        0
    } else if r.bit()? {
        2
    } else {
        1
    };
    h.reference_select = !intra && r.bit()?;
    skip_mode_params(r, seq, &mut h)?;
    h.allow_warped_motion =
        !intra && !h.error_resilient_mode && seq.enable_warped_motion && r.bit()?;
    h.reduced_tx_set = r.bit()?;
    global_motion_params(r, &mut h, previous)?;
    film_grain_params(r, seq, refs, &mut h)?;
    Ok(h)
}

/// The slot `index`, from a 3-bit field.
fn slot(refs: &[ReferenceState; NUM_REF_FRAMES], index: u8) -> Result<&ReferenceState, Error> {
    refs.get(usize::from(index))
        .ok_or(Error::OutOfRange("ref_frame_idx"))
}

fn valid(ref_valid: &[bool; 8], index: u8) -> bool {
    ref_valid.get(usize::from(index)).copied().unwrap_or(false)
}

/// `temporal_point_info()` where the header has one (5.9.31).
fn temporal_point_info(r: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<(), Error> {
    if let Some(model) = seq.decoder_model_info
        && !seq.equal_picture_interval
    {
        r.f(u32::from(model.frame_presentation_time_length_minus_1).saturating_add(1))?;
    }
    Ok(())
}

/// `mark_ref_frames()` (5.9.4): the slots whose frame ids are too far from the current one
/// are no longer valid.
fn mark_ref_frames(
    ref_valid: &mut [bool; 8],
    refs: &[ReferenceState; NUM_REF_FRAMES],
    current: u32,
    id_len: u32,
    diff_len: u32,
) {
    // id_len is at most 16 and diff_len at most 17: no shift overflows a u64.
    let current = u64::from(current);
    let diff = 1u64 << diff_len;
    let span = 1u64 << id_len;
    for (valid, slot) in ref_valid.iter_mut().zip(refs) {
        let id = u64::from(slot.frame_id);
        let stale = if current > diff {
            id > current || id < current.saturating_sub(diff)
        } else {
            id > current && id < span.saturating_add(current).saturating_sub(diff)
        };
        if stale {
            *valid = false;
        }
    }
}

/// The references of an inter or switch frame: `frame_refs_short_signaling` and
/// `set_frame_refs()`, `ref_frame_idx`, the frame's size, and what follows them to
/// `RefFrameSignBias` (5.9.2).
fn inter_references(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    refs: &[ReferenceState; NUM_REF_FRAMES],
    id_len: Option<u32>,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let short_signaling = seq.enable_order_hint && r.bit()?;
    if short_signaling {
        let last = r.u8(3)?;
        let gold = r.u8(3)?;
        h.ref_frame_idx = set_frame_refs(seq, &h.ref_order_hint, h.order_hint, last, gold)?;
    }
    for index in &mut h.ref_frame_idx {
        if !short_signaling {
            *index = r.u8(3)?;
        }
        if let (Some(len), Some(ids)) = (id_len, seq.frame_id_numbers) {
            // DeltaFrameId is 1 to 2^17, and the ids below 2^16.
            let delta = u64::from(r.f(ids.delta_len())?).saturating_add(1);
            let span = 1u64 << len;
            let expected = u64::from(h.current_frame_id)
                .saturating_add(span)
                .saturating_sub(delta)
                .checked_rem(span)
                .unwrap_or(0);
            // 6.8.2: the slot holds the frame whose id the frame expects.
            if u64::from(slot(refs, *index)?.frame_id) != expected {
                return Err(Error::MissingReference);
            }
        }
    }
    for &index in &h.ref_frame_idx {
        if !valid(&h.ref_valid, index) {
            return Err(Error::MissingReference);
        }
    }
    if h.frame_size_override_flag && !h.error_resilient_mode {
        frame_size_with_refs(r, seq, refs, h)?;
    } else {
        frame_size(r, seq, h)?;
        render_size(r, h)?;
    }
    for &index in &h.ref_frame_idx {
        // 7.9 and 6.8.6: a reference is at most twice the frame's size and at least a
        // sixteenth of it.
        let reference = slot(refs, index)?.size;
        let (width, height) = (h.size.frame_width, h.size.frame_height);
        if width.saturating_mul(2) < reference.upscaled_width
            || height.saturating_mul(2) < reference.frame_height
            || width > reference.upscaled_width.saturating_mul(16)
            || height > reference.frame_height.saturating_mul(16)
        {
            return Err(Error::OutOfRange("a reference's size"));
        }
    }
    h.allow_high_precision_mv = !h.force_integer_mv && r.bit()?;
    h.interpolation_filter = if r.bit()? { 4 } else { r.u8(2)? };
    h.is_motion_mode_switchable = r.bit()?;
    h.use_ref_frame_mvs = !h.error_resilient_mode && seq.enable_ref_frame_mvs && r.bit()?;
    for (hint, &index) in h.order_hints.iter_mut().skip(1).zip(&h.ref_frame_idx) {
        *hint = h
            .ref_order_hint
            .get(usize::from(index))
            .copied()
            .unwrap_or(0);
    }
    Ok(())
}

/// `get_relative_dist()` (7.12.1).
pub(crate) fn relative_dist(seq: &SequenceHeader, a: u8, b: u8) -> i32 {
    if !seq.enable_order_hint {
        return 0;
    }
    // Order hints are below 2^8: nothing here overflows.
    let diff = i32::from(a).wrapping_sub(i32::from(b));
    let m = 1i32 << seq.order_hint_bits.saturating_sub(1);
    (diff & m.wrapping_sub(1)).wrapping_sub(diff & m)
}

/// `set_frame_refs()` (7.8): the references `frame_refs_short_signaling` leaves the decoder
/// to choose.
pub(crate) fn set_frame_refs(
    seq: &SequenceHeader,
    ref_order_hint: &[u8; 8],
    order_hint: u8,
    last: u8,
    gold: u8,
) -> Result<[u8; 7], Error> {
    // LAST_FRAME - LAST_FRAME and GOLDEN_FRAME - LAST_FRAME.
    let mut chosen: [Option<u8>; 7] = [Some(last), None, None, Some(gold), None, None, None];
    let mut used = [false; 8];
    for index in [last, gold] {
        if let Some(used) = used.get_mut(usize::from(index)) {
            *used = true;
        }
    }
    let current = 1i32 << seq.order_hint_bits.saturating_sub(1);
    // Each within current ± 2^7.
    let shifted =
        ref_order_hint.map(|hint| current.saturating_add(relative_dist(seq, hint, order_hint)));
    let hint_of = |index: u8| shifted.get(usize::from(index)).copied().unwrap_or(current);
    // 7.8: LAST_FRAME and GOLDEN_FRAME are before the current frame.
    if hint_of(last) >= current {
        return Err(Error::OutOfRange("last_frame_idx"));
    }
    if hint_of(gold) >= current {
        return Err(Error::OutOfRange("gold_frame_idx"));
    }
    // find_latest_backward(), find_earliest_backward() and find_latest_forward(): the slot
    // not yet used whose shifted hint is on the side `backward` says, latest or earliest,
    // the last of equals for the latest and the first for the earliest.
    let find = |used: &[bool; 8], backward: bool, latest: bool| -> Option<u8> {
        let mut best: Option<(u8, i32)> = None;
        for (index, (&hint, &used)) in (0u8..).zip(shifted.iter().zip(used)) {
            if used || (hint >= current) != backward {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, best)) if latest => hint >= best,
                Some((_, best)) => hint < best,
            };
            if better {
                best = Some((index, hint));
            }
        }
        best.map(|(index, _)| index)
    };
    // ALTREF_FRAME, then BWDREF_FRAME, then ALTREF2_FRAME.
    let found = find(&used, true, true);
    take(&mut chosen, &mut used, 6, found);
    let found = find(&used, true, false);
    take(&mut chosen, &mut used, 4, found);
    let found = find(&used, true, false);
    take(&mut chosen, &mut used, 5, found);
    // Ref_Frame_List: LAST2_FRAME, LAST3_FRAME, BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME.
    for reference in [1, 2, 4, 5, 6] {
        if chosen.get(reference).copied().flatten().is_none() {
            let found = find(&used, false, true);
            take(&mut chosen, &mut used, reference, found);
        }
    }
    // The rest take the slot of the earliest hint, the first of equals.
    let mut earliest: Option<(u8, i32)> = None;
    for (index, &hint) in (0u8..).zip(&shifted) {
        if earliest.is_none_or(|(_, best)| hint < best) {
            earliest = Some((index, hint));
        }
    }
    let earliest = earliest.map_or(0, |(index, _)| index);
    Ok(chosen.map(|index| index.unwrap_or(earliest)))
}

/// Gives `reference` the slot `found`, which is used from then on.
fn take(chosen: &mut [Option<u8>; 7], used: &mut [bool; 8], reference: usize, found: Option<u8>) {
    if let (Some(index), Some(slot)) = (found, chosen.get_mut(reference)) {
        *slot = Some(index);
        if let Some(used) = used.get_mut(usize::from(index)) {
            *used = true;
        }
    }
}

/// `frame_size()` (5.9.5) with `superres_params()`.
fn frame_size(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let (width, height) = if h.frame_size_override_flag {
        // Each at most 2^16.
        let width = r.f(seq.frame_width_bits)?.saturating_add(1);
        let height = r.f(seq.frame_height_bits)?.saturating_add(1);
        if width > seq.max_frame_width {
            return Err(Error::OutOfRange("frame_width_minus_1"));
        }
        if height > seq.max_frame_height {
            return Err(Error::OutOfRange("frame_height_minus_1"));
        }
        (width, height)
    } else {
        (seq.max_frame_width, seq.max_frame_height)
    };
    h.size.frame_height = height;
    superres_params(r, seq, &mut h.size, width)
}

/// `superres_params()` (5.9.8) and the frame's coded width, held to 16 at least where the
/// upscaled width is (Annex A, as libaom and dav1d hold it).
fn superres_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    size: &mut FrameSize,
    upscaled_width: u32,
) -> Result<(), Error> {
    size.use_superres = seq.enable_superres && r.bit()?;
    size.coded_denom = 0;
    size.superres_denom = SUPERRES_NUM;
    if size.use_superres {
        size.coded_denom = r.u8(3)?;
        size.superres_denom = u32::from(size.coded_denom).saturating_add(SUPERRES_DENOM_MIN);
    }
    size.upscaled_width = upscaled_width;
    // Widths are at most 2^16 and the denominator 8 to 16.
    let scaled = upscaled_width
        .saturating_mul(SUPERRES_NUM)
        .saturating_add(size.superres_denom / 2)
        .checked_div(size.superres_denom)
        .ok_or(Error::OutOfRange("SuperresDenom"))?;
    size.frame_width = if size.use_superres {
        scaled.max(upscaled_width.min(16))
    } else {
        upscaled_width
    };
    Ok(())
}

/// `render_size()` (5.9.6).
fn render_size(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<(), Error> {
    if r.bit()? {
        h.size.render_width = r.f(16)?.saturating_add(1);
        h.size.render_height = r.f(16)?.saturating_add(1);
    } else {
        h.size.render_width = h.size.upscaled_width;
        h.size.render_height = h.size.frame_height;
    }
    Ok(())
}

/// `frame_size_with_refs()` (5.9.7).
fn frame_size_with_refs(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    refs: &[ReferenceState; NUM_REF_FRAMES],
    h: &mut FrameHeader,
) -> Result<(), Error> {
    for &index in &h.ref_frame_idx {
        if r.bit()? {
            let reference = slot(refs, index)?.size;
            // A slot of an earlier sequence may be larger than this one's largest frame.
            if reference.upscaled_width > seq.max_frame_width
                || reference.frame_height > seq.max_frame_height
            {
                return Err(Error::OutOfRange("found_ref"));
            }
            h.size.frame_height = reference.frame_height;
            h.size.render_width = reference.render_width;
            h.size.render_height = reference.render_height;
            return superres_params(r, seq, &mut h.size, reference.upscaled_width);
        }
    }
    frame_size(r, seq, h)?;
    render_size(r, h)
}

/// `tile_log2()` (5.9.16).
fn tile_log2(block: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while k < 32 && (u64::from(block) << k) < u64::from(target) {
        k = k.saturating_add(1);
    }
    k
}

/// `tile_info()` (5.9.15).
fn tile_info(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    // sbShift, (1 << sbShift) - 1, and sbSize = sbShift + 2.
    let (sb_shift, round, sb_size) = if seq.use_128x128_superblock {
        (5, 31, 7u32)
    } else {
        (4, 15, 6)
    };
    // MiCols and MiRows are at most 2^14 + 2.
    let sb_cols = h.size.mi_cols().saturating_add(round) >> sb_shift;
    let sb_rows = h.size.mi_rows().saturating_add(round) >> sb_shift;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let mut max_tile_area_sb = MAX_TILE_AREA >> sb_size.saturating_mul(2);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_COLS));
    let sb_area = sb_rows.saturating_mul(sb_cols);
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_area));
    let info = &mut h.tile_info;
    info.uniform = r.bit()?;
    if info.uniform {
        info.cols_log2 = min_log2_tile_cols;
        while info.cols_log2 < max_log2_tile_cols && r.bit()? {
            info.cols_log2 = info.cols_log2.saturating_add(1);
        }
        info.col_widths_sb = uniform_sizes(sb_cols, info.cols_log2);
        let min_log2_tile_rows = min_log2_tiles.saturating_sub(info.cols_log2);
        info.rows_log2 = min_log2_tile_rows;
        while info.rows_log2 < max_log2_tile_rows && r.bit()? {
            info.rows_log2 = info.rows_log2.saturating_add(1);
        }
        info.row_heights_sb = uniform_sizes(sb_rows, info.rows_log2);
    } else {
        let mut widest_tile_sb = 0;
        let mut start = 0u32;
        while start < sb_cols {
            if info.col_widths_sb.len() >= 64 {
                return Err(Error::OutOfRange("TileCols"));
            }
            let max_width = sb_cols.saturating_sub(start).min(max_tile_width_sb);
            let size = r.ns(max_width)?.saturating_add(1);
            widest_tile_sb = widest_tile_sb.max(size);
            info.col_widths_sb.push(size);
            start = start.saturating_add(size);
        }
        info.cols_log2 = tile_log2(1, info.cols());
        if min_log2_tiles > 0 {
            max_tile_area_sb = sb_area >> min_log2_tiles.saturating_add(1);
        } else {
            max_tile_area_sb = sb_area;
        }
        let max_tile_height_sb = max_tile_area_sb
            .checked_div(widest_tile_sb)
            .unwrap_or(0)
            .max(1);
        start = 0;
        while start < sb_rows {
            if info.row_heights_sb.len() >= 64 {
                return Err(Error::OutOfRange("TileRows"));
            }
            let max_height = sb_rows.saturating_sub(start).min(max_tile_height_sb);
            let size = r.ns(max_height)?.saturating_add(1);
            info.row_heights_sb.push(size);
            start = start.saturating_add(size);
        }
        info.rows_log2 = tile_log2(1, info.rows());
    }
    if info.cols() > MAX_TILE_COLS || info.rows() > MAX_TILE_COLS {
        return Err(Error::OutOfRange("TileCols"));
    }
    if info.cols_log2 > 0 || info.rows_log2 > 0 {
        info.context_update_tile_id = r.f(info.rows_log2.saturating_add(info.cols_log2))?;
        if info.context_update_tile_id >= info.num_tiles() {
            return Err(Error::OutOfRange("context_update_tile_id"));
        }
        info.tile_size_bytes = r.f(2)?.saturating_add(1);
    } else {
        info.context_update_tile_id = 0;
        info.tile_size_bytes = 4;
    }
    Ok(())
}

/// The sizes, in superblocks, of the tiles `uniform_tile_spacing_flag` cuts `count`
/// superblocks into: `2^log2` tiles' worth, each but the last `tileWidthSb`.
fn uniform_sizes(count: u32, log2: u32) -> Vec<u32> {
    // log2 is at most 6 and count at most 2^10 or so.
    let size = count.saturating_add((1u32 << log2).saturating_sub(1)) >> log2;
    let size = size.max(1);
    let mut sizes = Vec::new();
    let mut start = 0u32;
    while start < count {
        sizes.push(size.min(count.saturating_sub(start)));
        start = start.saturating_add(size);
    }
    sizes
}

/// `read_delta_q()` (5.9.13).
fn read_delta_q(r: &mut BitReader<'_>) -> Result<i8, Error> {
    if r.bit()? {
        i8::try_from(r.su(7)?).map_err(|_| Error::OutOfRange("delta_q"))
    } else {
        Ok(0)
    }
}

/// `quantization_params()` (5.9.12).
fn quantization_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let q = &mut h.quantization;
    q.base_q_idx = r.u8(8)?;
    q.delta_q_y_dc = read_delta_q(r)?;
    if seq.num_planes() > 1 {
        let diff_uv_delta = seq.color.separate_uv_delta_q && r.bit()?;
        q.delta_q_u_dc = read_delta_q(r)?;
        q.delta_q_u_ac = read_delta_q(r)?;
        if diff_uv_delta {
            q.delta_q_v_dc = read_delta_q(r)?;
            q.delta_q_v_ac = read_delta_q(r)?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = r.bit()?;
    if q.using_qmatrix {
        q.qm_y = r.u8(4)?;
        q.qm_u = r.u8(4)?;
        q.qm_v = if seq.color.separate_uv_delta_q {
            r.u8(4)?
        } else {
            q.qm_u
        };
    }
    Ok(())
}

/// `segmentation_params()` (5.9.14), the features loaded from the primary reference frame
/// where the frame codes none (`load_previous()`).
fn segmentation_params(
    r: &mut BitReader<'_>,
    h: &mut FrameHeader,
    previous: Option<&ReferenceState>,
) -> Result<(), Error> {
    let s = &mut h.segmentation;
    s.enabled = r.bit()?;
    s.features = previous.map(|p| p.segmentation).unwrap_or_default();
    if !s.enabled {
        s.features = SegmentationFeatures::default();
        return Ok(());
    }
    if h.primary_ref_frame == PRIMARY_REF_NONE {
        s.update_map = true;
        s.update_data = true;
    } else {
        s.update_map = r.bit()?;
        s.temporal_update = s.update_map && r.bit()?;
        s.update_data = r.bit()?;
    }
    if s.update_data {
        for (enabled, data) in s.features.enabled.iter_mut().zip(&mut s.features.data) {
            for ((enabled, data), &(bits, signed, max)) in enabled
                .iter_mut()
                .zip(data.iter_mut())
                .zip(&SEGMENTATION_FEATURES)
            {
                *enabled = r.bit()?;
                *data = 0;
                if *enabled {
                    *data = if signed {
                        r.su(bits.saturating_add(1))?
                            .clamp(max.saturating_neg(), max)
                    } else {
                        i16::try_from(r.f(bits)?)
                            .map_err(|_| Error::OutOfRange("feature_value"))?
                            .clamp(0, max)
                    };
                }
            }
        }
    }
    Ok(())
}

/// Whether `segment` is lossless (5.9.2): its `get_qindex(1, segment)` and every DC and AC
/// delta 0.
fn lossless(h: &FrameHeader, segment: usize) -> bool {
    let q = &h.quantization;
    let features = &h.segmentation.features;
    let alt_q = features
        .enabled
        .get(segment)
        .and_then(|enabled| enabled.get(SEG_LVL_ALT_Q))
        .copied()
        .unwrap_or(false);
    let qindex = if h.segmentation.enabled && alt_q {
        let data = features
            .data
            .get(segment)
            .and_then(|data| data.get(SEG_LVL_ALT_Q))
            .copied()
            .unwrap_or(0);
        // A byte plus at most ±255.
        i32::from(q.base_q_idx)
            .saturating_add(i32::from(data))
            .clamp(0, 255)
    } else {
        i32::from(q.base_q_idx)
    };
    qindex == 0
        && q.delta_q_y_dc == 0
        && q.delta_q_u_ac == 0
        && q.delta_q_u_dc == 0
        && q.delta_q_v_ac == 0
        && q.delta_q_v_dc == 0
}

/// `loop_filter_params()` (5.9.11), the deltas loaded from the primary reference frame or
/// set by `setup_past_independence()`.
fn loop_filter_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
    previous: Option<&ReferenceState>,
) -> Result<(), Error> {
    let lf = &mut h.loop_filter;
    (lf.ref_deltas, lf.mode_deltas) = previous.map_or((DEFAULT_REF_DELTAS, [0; 2]), |p| {
        (p.loop_filter_ref_deltas, p.loop_filter_mode_deltas)
    });
    if h.coded_lossless || h.allow_intrabc {
        lf.ref_deltas = DEFAULT_REF_DELTAS;
        lf.mode_deltas = [0; 2];
        return Ok(());
    }
    lf.level[0] = r.u8(6)?;
    lf.level[1] = r.u8(6)?;
    if seq.num_planes() > 1 && (lf.level[0] > 0 || lf.level[1] > 0) {
        lf.level[2] = r.u8(6)?;
        lf.level[3] = r.u8(6)?;
    }
    lf.sharpness = r.u8(3)?;
    lf.delta_enabled = r.bit()?;
    if lf.delta_enabled {
        lf.delta_update = r.bit()?;
        if lf.delta_update {
            for delta in lf.ref_deltas.iter_mut().chain(lf.mode_deltas.iter_mut()) {
                if r.bit()? {
                    *delta = i8::try_from(r.su(7)?)
                        .map_err(|_| Error::OutOfRange("loop_filter_ref_deltas"))?;
                }
            }
        }
    }
    Ok(())
}

/// `cdef_params()` (5.9.19).
fn cdef_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let cdef = &mut h.cdef;
    if h.coded_lossless || h.allow_intrabc || !seq.enable_cdef {
        *cdef = Cdef::default();
        return Ok(());
    }
    cdef.damping_minus_3 = r.u8(2)?;
    cdef.bits = r.u8(2)?;
    let strengths = 1usize << cdef.bits;
    let chroma = seq.num_planes() > 1;
    for (((y_pri, y_sec), uv_pri), uv_sec) in cdef
        .y_pri_strength
        .iter_mut()
        .zip(cdef.y_sec_strength.iter_mut())
        .zip(cdef.uv_pri_strength.iter_mut())
        .zip(cdef.uv_sec_strength.iter_mut())
        .take(strengths)
    {
        *y_pri = r.u8(4)?;
        *y_sec = r.u8(2)?;
        if chroma {
            *uv_pri = r.u8(4)?;
            *uv_sec = r.u8(2)?;
        }
    }
    Ok(())
}

/// `lr_params()` (5.9.20).
fn lr_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    let lr = &mut h.loop_restoration;
    *lr = LoopRestoration::default();
    if h.all_lossless || h.allow_intrabc || !seq.enable_restoration {
        return Ok(());
    }
    let mut uses_lr = false;
    let mut uses_chroma_lr = false;
    for (plane, restoration) in lr
        .frame_restoration_type
        .iter_mut()
        .take(seq.num_planes())
        .enumerate()
    {
        let lr_type = r.u8(2)?;
        *restoration = REMAP_LR_TYPE
            .get(usize::from(lr_type))
            .copied()
            .unwrap_or(0);
        if *restoration != 0 {
            uses_lr = true;
            uses_chroma_lr |= plane > 0;
        }
    }
    if uses_lr {
        lr.unit_shift = if seq.use_128x128_superblock {
            r.u8(1)?.saturating_add(1)
        } else {
            let shift = r.u8(1)?;
            if shift == 1 {
                shift.saturating_add(r.u8(1)?)
            } else {
                shift
            }
        };
        lr.uv_shift = if seq.color.subsampling_x && seq.color.subsampling_y && uses_chroma_lr {
            r.u8(1)?
        } else {
            0
        };
    }
    Ok(())
}

/// `skip_mode_params()` (5.9.22).
fn skip_mode_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    h: &mut FrameHeader,
) -> Result<(), Error> {
    h.skip_mode_present = false;
    h.skip_mode_frame = [0; 2];
    if h.frame_type.is_intra() || !h.reference_select || !seq.enable_order_hint {
        return Ok(());
    }
    let hints: [u8; 7] = h.ref_frame_idx.map(|index| {
        h.ref_order_hint
            .get(usize::from(index))
            .copied()
            .unwrap_or(0)
    });
    let mut forward: Option<(u8, u8)> = None;
    let mut backward: Option<(u8, u8)> = None;
    for (i, &hint) in (0u8..).zip(&hints) {
        let dist = relative_dist(seq, hint, h.order_hint);
        if dist < 0 {
            if forward.is_none_or(|(_, best)| relative_dist(seq, hint, best) > 0) {
                forward = Some((i, hint));
            }
        } else if dist > 0 && backward.is_none_or(|(_, best)| relative_dist(seq, hint, best) < 0) {
            backward = Some((i, hint));
        }
    }
    let Some((forward_index, forward_hint)) = forward else {
        return Ok(());
    };
    let second = match backward {
        Some((index, _)) => index,
        None => {
            let mut second: Option<(u8, u8)> = None;
            for (i, &hint) in (0u8..).zip(&hints) {
                if relative_dist(seq, hint, forward_hint) < 0
                    && second.is_none_or(|(_, best)| relative_dist(seq, hint, best) > 0)
                {
                    second = Some((i, hint));
                }
            }
            let Some((index, _)) = second else {
                return Ok(());
            };
            index
        }
    };
    // LAST_FRAME + the indices, each below 7.
    h.skip_mode_frame = [
        forward_index.min(second).saturating_add(1),
        forward_index.max(second).saturating_add(1),
    ];
    h.skip_mode_present = r.bit()?;
    Ok(())
}

/// `global_motion_params()` (5.9.24), from the primary reference frame's parameters, or the
/// identity's (`setup_past_independence()`).
fn global_motion_params(
    r: &mut BitReader<'_>,
    h: &mut FrameHeader,
    previous: Option<&ReferenceState>,
) -> Result<(), Error> {
    h.global_motion = [GlobalMotion::default(); 8];
    if h.frame_type.is_intra() {
        return Ok(());
    }
    let previous = previous.map_or([IDENTITY_PARAMS; 8], |p| p.gm_params);
    let high_precision = h.allow_high_precision_mv;
    for (motion, prev) in h.global_motion.iter_mut().zip(&previous).skip(1) {
        motion.is_global = r.bit()?;
        let gm_type = if !motion.is_global {
            0
        } else if r.bit()? {
            2
        } else if r.bit()? {
            1
        } else {
            3
        };
        let mut params = IDENTITY_PARAMS;
        let mut read = |params: &mut [i32; 6], index: usize| -> Result<(), Error> {
            let prev = prev.get(index).copied().unwrap_or(0);
            let value = read_global_param(r, gm_type, index, prev, high_precision)?;
            if let Some(param) = params.get_mut(index) {
                *param = value;
            }
            Ok(())
        };
        if gm_type >= 2 {
            read(&mut params, 2)?;
            read(&mut params, 3)?;
            if gm_type == 3 {
                read(&mut params, 4)?;
                read(&mut params, 5)?;
            } else {
                params[4] = params[3].saturating_neg();
                params[5] = params[2];
            }
        }
        if gm_type >= 1 {
            read(&mut params, 0)?;
            read(&mut params, 1)?;
        }
        motion.gm_type = gm_type;
        motion.params = params;
        motion.warp_valid = warp_valid(&params);
    }
    Ok(())
}

/// `read_global_param()` (5.9.25).
fn read_global_param(
    r: &mut BitReader<'_>,
    gm_type: u8,
    index: usize,
    prev: i32,
    high_precision: bool,
) -> Result<i32, Error> {
    let low_precision = u32::from(!high_precision);
    let (abs_bits, prec_bits) = if index >= 2 {
        // GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS.
        (12, 15)
    } else if gm_type == 1 {
        // GM_ABS_TRANS_ONLY_BITS, GM_TRANS_ONLY_PREC_BITS.
        (
            9u32.saturating_sub(low_precision),
            3u32.saturating_sub(low_precision),
        )
    } else {
        // GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS.
        (12, 6)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS.saturating_sub(prec_bits);
    let (round, sub) = if index % 3 == 2 {
        (1i64 << WARPEDMODEL_PREC_BITS, 1i64 << prec_bits)
    } else {
        (0, 0)
    };
    let mx = 1i64 << abs_bits;
    let reference = (i64::from(prev) >> prec_diff).saturating_sub(sub);
    let value =
        decode_signed_subexp_with_ref(r, mx.saturating_neg(), mx.saturating_add(1), reference)?;
    // At most 2^12 + 1 shifted by at most 13, plus 2^16.
    i32::try_from(
        value
            .saturating_mul(1i64 << prec_diff)
            .saturating_add(round),
    )
    .map_err(|_| Error::OutOfRange("gm_params"))
}

/// `decode_signed_subexp_with_ref()` (5.9.26).
fn decode_signed_subexp_with_ref(
    r: &mut BitReader<'_>,
    low: i64,
    high: i64,
    reference: i64,
) -> Result<i64, Error> {
    let x = decode_unsigned_subexp_with_ref(
        r,
        high.saturating_sub(low),
        reference.saturating_sub(low),
    )?;
    Ok(x.saturating_add(low))
}

/// `decode_unsigned_subexp_with_ref()` (5.9.27).
fn decode_unsigned_subexp_with_ref(
    r: &mut BitReader<'_>,
    mx: i64,
    reference: i64,
) -> Result<i64, Error> {
    let v = decode_subexp(r, mx)?;
    if reference.saturating_mul(2) <= mx {
        Ok(inverse_recenter(reference, v))
    } else {
        let top = mx.saturating_sub(1);
        Ok(top.saturating_sub(inverse_recenter(top.saturating_sub(reference), v)))
    }
}

/// `decode_subexp()` (5.9.28).
fn decode_subexp(r: &mut BitReader<'_>, num_syms: i64) -> Result<i64, Error> {
    let mut i = 0u32;
    let mut mk = 0i64;
    let k = 3u32;
    loop {
        let b2 = if i > 0 {
            k.saturating_add(i).saturating_sub(1)
        } else {
            k
        };
        if b2 > 30 {
            return Err(Error::OutOfRange("subexp_more_bits"));
        }
        let a = 1i64 << b2;
        if num_syms <= mk.saturating_add(a.saturating_mul(3)) {
            let n = u32::try_from(num_syms.saturating_sub(mk))
                .map_err(|_| Error::OutOfRange("subexp_final_bits"))?;
            return Ok(i64::from(r.ns(n)?).saturating_add(mk));
        }
        if r.bit()? {
            i = i.saturating_add(1);
            mk = mk.saturating_add(a);
        } else {
            return Ok(i64::from(r.f(b2)?).saturating_add(mk));
        }
    }
}

/// `inverse_recenter()` (5.9.29).
fn inverse_recenter(r: i64, v: i64) -> i64 {
    if v > r.saturating_mul(2) {
        v
    } else if v & 1 == 1 {
        r.saturating_sub(v.saturating_add(1) >> 1)
    } else {
        r.saturating_add(v >> 1)
    }
}

/// `Round2()` of a signed value (4.7), for `n` from 0 to 62.
fn round2_signed(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    let half = 1i64 << n.saturating_sub(1);
    if x >= 0 {
        x.saturating_add(half) >> n
    } else {
        (x.saturating_neg().saturating_add(half) >> n).saturating_neg()
    }
}

/// The setup shear process's `warpValid` (7.11.3.6) for global motion parameters, with
/// `resolve_divisor` (7.11.3.7) and `Div_Lut`, whose entries are 2^14 × 256 / (256 + i),
/// rounded. Parameters whose `gm_params[2]` is not positive are not valid, as ffmpeg finds.
pub(crate) fn warp_valid(params: &[i32; 6]) -> bool {
    let [_, _, p2, p3, p4, p5] = params.map(i64::from);
    if p2 <= 0 {
        return false;
    }
    let clip = |x: i64| x.clamp(-32768, 32767);
    // resolve_divisor(p2): n = FloorLog2(p2), at most 30.
    let n = 63u32.saturating_sub(p2.leading_zeros());
    let e = p2.saturating_sub(1i64 << n);
    let f = if n > 8 {
        round2_signed(e, n.saturating_sub(8))
    } else {
        e << (8u32.saturating_sub(n))
    };
    let d = f.saturating_add(256);
    let div_factor = (1i64 << 22)
        .saturating_add(d / 2)
        .checked_div(d)
        .unwrap_or(0);
    let div_shift = n.saturating_add(14);
    let alpha = clip(p2.saturating_sub(1 << WARPEDMODEL_PREC_BITS));
    let beta = clip(p3);
    // The parameters are below 2^17 in magnitude and div_factor at most 2^14: the products
    // stay well within an i64.
    let v = p4.saturating_mul(1 << WARPEDMODEL_PREC_BITS);
    let gamma = clip(round2_signed(v.saturating_mul(div_factor), div_shift));
    let w = p3.saturating_mul(p4);
    let delta = clip(
        p5.saturating_sub(round2_signed(w.saturating_mul(div_factor), div_shift))
            .saturating_sub(1 << WARPEDMODEL_PREC_BITS),
    );
    // WARP_PARAM_REDUCE_BITS is 6.
    let reduce = |x: i64| round2_signed(x, 6).saturating_mul(64);
    let (alpha, beta, gamma, delta) = (reduce(alpha), reduce(beta), reduce(gamma), reduce(delta));
    let limit = 1i64 << WARPEDMODEL_PREC_BITS;
    alpha
        .abs()
        .saturating_mul(4)
        .saturating_add(beta.abs().saturating_mul(7))
        < limit
        && gamma
            .abs()
            .saturating_mul(4)
            .saturating_add(delta.abs().saturating_mul(4))
            < limit
}

/// `film_grain_params()` (5.9.30).
fn film_grain_params(
    r: &mut BitReader<'_>,
    seq: &SequenceHeader,
    refs: &[ReferenceState; NUM_REF_FRAMES],
    h: &mut FrameHeader,
) -> Result<(), Error> {
    h.film_grain = FilmGrainParams::default();
    if !seq.film_grain_params_present || (!h.show_frame && !h.showable_frame) {
        return Ok(());
    }
    if !r.bit()? {
        return Ok(());
    }
    let grain_seed = r.u16(16)?;
    let update_grain = h.frame_type != FrameType::Inter || r.bit()?;
    if !update_grain {
        let index = r.u8(3)?;
        // 6.8.20: the parameters are a reference's of this frame's.
        if !h.ref_frame_idx.contains(&index) {
            return Err(Error::OutOfRange("film_grain_params_ref_idx"));
        }
        if !valid(&h.ref_valid, index) {
            return Err(Error::MissingReference);
        }
        h.film_grain = slot(refs, index)?.film_grain;
        h.film_grain.grain_seed = grain_seed;
        return Ok(());
    }
    let g = &mut h.film_grain;
    g.apply_grain = true;
    g.grain_seed = grain_seed;
    g.update_grain = true;
    g.num_y_points = read_points(
        r,
        14,
        &mut g.point_y_value,
        &mut g.point_y_scaling,
        "num_y_points",
    )?;
    g.chroma_scaling_from_luma = !seq.color.mono_chrome && r.bit()?;
    let color = &seq.color;
    if !(color.mono_chrome
        || g.chroma_scaling_from_luma
        || (color.subsampling_x && color.subsampling_y && g.num_y_points == 0))
    {
        g.num_cb_points = read_points(
            r,
            10,
            &mut g.point_cb_value,
            &mut g.point_cb_scaling,
            "num_cb_points",
        )?;
        g.num_cr_points = read_points(
            r,
            10,
            &mut g.point_cr_value,
            &mut g.point_cr_scaling,
            "num_cr_points",
        )?;
        // 6.8.20: 4:2:0 codes both chroma planes' points or neither's.
        if color.subsampling_x
            && color.subsampling_y
            && (g.num_cb_points == 0) != (g.num_cr_points == 0)
        {
            return Err(Error::OutOfRange("num_cr_points"));
        }
    }
    g.grain_scaling_minus_8 = r.u8(2)?;
    g.ar_coeff_lag = r.u8(2)?;
    // 2 × lag × (lag + 1): at most 24.
    let num_pos_luma = usize::from(g.ar_coeff_lag)
        .saturating_mul(usize::from(g.ar_coeff_lag).saturating_add(1))
        .saturating_mul(2);
    let num_pos_chroma = if g.num_y_points > 0 {
        read_coefficients(r, &mut g.ar_coeffs_y, num_pos_luma)?;
        num_pos_luma.saturating_add(1)
    } else {
        num_pos_luma
    };
    if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
        read_coefficients(r, &mut g.ar_coeffs_cb, num_pos_chroma)?;
    }
    if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
        read_coefficients(r, &mut g.ar_coeffs_cr, num_pos_chroma)?;
    }
    g.ar_coeff_shift_minus_6 = r.u8(2)?;
    g.grain_scale_shift = r.u8(2)?;
    if g.num_cb_points > 0 {
        g.cb_mult = r.u8(8)?;
        g.cb_luma_mult = r.u8(8)?;
        g.cb_offset = r.u16(9)?;
    }
    if g.num_cr_points > 0 {
        g.cr_mult = r.u8(8)?;
        g.cr_luma_mult = r.u8(8)?;
        g.cr_offset = r.u16(9)?;
    }
    g.overlap_flag = r.bit()?;
    g.clip_to_restricted_range = r.bit()?;
    Ok(())
}

/// A plane's scaling points: their count, at most `max`, and each value and scaling, the
/// values increasing (6.8.20).
fn read_points(
    r: &mut BitReader<'_>,
    max: u8,
    values: &mut [u8],
    scalings: &mut [u8],
    name: &'static str,
) -> Result<u8, Error> {
    let count = r.u8(4)?;
    if count > max {
        return Err(Error::OutOfRange(name));
    }
    let mut last: Option<u8> = None;
    for (value, scaling) in values
        .iter_mut()
        .zip(scalings.iter_mut())
        .take(usize::from(count))
    {
        *value = r.u8(8)?;
        *scaling = r.u8(8)?;
        if last.is_some_and(|last| *value <= last) {
            return Err(Error::OutOfRange(name));
        }
        last = Some(*value);
    }
    Ok(count)
}

/// `count` auto-regressive coefficients, each coded plus 128.
fn read_coefficients(
    r: &mut BitReader<'_>,
    coefficients: &mut [i8],
    count: usize,
) -> Result<(), Error> {
    if count > coefficients.len() {
        return Err(Error::OutOfRange("ar_coeff_lag"));
    }
    for coefficient in coefficients.iter_mut().take(count) {
        // A byte less 128 is an i8: flipping the top bit of its two's complement.
        *coefficient = i8::from_ne_bytes([r.u8(8)? ^ 0x80]);
    }
    Ok(())
}
