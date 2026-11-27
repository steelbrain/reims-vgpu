//! The uncompressed header (6.2, 7.2), and what it keeps from one frame to
//! the next ([`State`]): the loop filter's deltas, the segmentation's
//! probabilities and features, and the colour configuration the last intra
//! frame gave. Each element is read as the specification reads it; the
//! values the accelerators take are kept in the specification's terms,
//! and each accelerator maps them to its own.

use super::Error;
use super::bits::BitReader;

/// `frame_sync_code()` (6.2.1).
const SYNC_CODE: [u8; 3] = [0x49, 0x83, 0x42];
/// `color_space`'s `CS_BT_601` (7.2.2).
pub const CS_BT_601: u8 = 1;
/// `color_space`'s `CS_RGB` (7.2.2).
pub const CS_RGB: u8 = 7;
/// The interpolation filter that each block chooses (`SWITCHABLE`), in the
/// numbering VA-API's `mcomp_filter_type` and NVDEC's take: libvpx's, where
/// `EIGHTTAP` is 0 and `EIGHTTAP_SMOOTH` 1.
pub const SWITCHABLE: u8 = 4;
/// The segments a frame has (`MAX_SEGMENTS`).
pub const MAX_SEGMENTS: usize = 8;
/// The features a segment has (`SEG_LVL_MAX`): the quantizer, the loop
/// filter level, the reference frame and skip, in that order.
pub const SEG_LVL_MAX: usize = 4;
/// `segmentation_feature_bits` (7.2.10).
const FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
/// `segmentation_feature_signed` (7.2.10).
const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];
/// `loop_filter_ref_deltas` as `setup_past_independence()` leaves them:
/// intra 1, last 0, golden −1, altref −1.
const DEFAULT_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];

/// A frame's colour configuration (7.2.2): what a key frame and a
/// profile's intra-only frame code, and what inter frames take from them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorConfig {
    /// `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// `color_space`.
    pub color_space: u8,
    /// `color_range`.
    pub full_range: bool,
    /// `subsampling_x`.
    pub subsampling_x: bool,
    /// `subsampling_y`.
    pub subsampling_y: bool,
}

/// `loop_filter_params()` (7.2.8), with the deltas in force.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopFilter {
    /// `loop_filter_level`: 0 to 63.
    pub level: u8,
    /// `loop_filter_sharpness`: 0 to 7.
    pub sharpness: u8,
    /// `loop_filter_delta_enabled`.
    pub delta_enabled: bool,
    /// `loop_filter_ref_deltas`, intra, last, golden and altref: those this
    /// frame coded, else those kept.
    pub ref_deltas: [i8; 4],
    /// `loop_filter_mode_deltas`, likewise.
    pub mode_deltas: [i8; 2],
}

/// `quantization_params()` (7.2.9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    /// `base_q_idx`.
    pub base_q_idx: u8,
    /// `delta_q_y_dc`: −15 to 15.
    pub delta_q_y_dc: i8,
    /// `delta_q_uv_dc`.
    pub delta_q_uv_dc: i8,
    /// `delta_q_uv_ac`.
    pub delta_q_uv_ac: i8,
}

impl Quantization {
    /// `Lossless` (7.2.9).
    pub fn lossless(&self) -> bool {
        self.base_q_idx == 0
            && self.delta_q_y_dc == 0
            && self.delta_q_uv_dc == 0
            && self.delta_q_uv_ac == 0
    }
}

/// `segmentation_params()` (7.2.10), with the probabilities and features in
/// force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segmentation {
    /// `segmentation_enabled`.
    pub enabled: bool,
    /// `segmentation_update_map`: false where segmentation is off.
    pub update_map: bool,
    /// `segmentation_temporal_update`: false where the map is not updated.
    pub temporal_update: bool,
    /// `segmentation_abs_or_delta_update`, as last coded since the last
    /// intra or error-resilient frame.
    pub abs_or_delta_update: bool,
    /// `segmentation_tree_probs`, as last coded.
    pub tree_probs: [u8; 7],
    /// `segmentation_pred_prob`, as last coded.
    pub pred_probs: [u8; 3],
    /// `FeatureEnabled`, each segment's four.
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `FeatureData`, likewise.
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            abs_or_delta_update: false,
            tree_probs: [255; 7],
            pred_probs: [255; 3],
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }
}

/// What the uncompressed header keeps between frames: each frame's header
/// is read against it and leaves it as the frame changed it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct State {
    /// The colour configuration in force: the last key frame's or intra-only
    /// frame's; `None` before the first.
    pub color: Option<ColorConfig>,
    /// The loop filter's deltas.
    pub ref_deltas: [i8; 4],
    /// See `ref_deltas`.
    pub mode_deltas: [i8; 2],
    /// The segmentation's probabilities and features.
    pub segmentation: Segmentation,
}

impl State {
    /// `setup_past_independence()` (7.2), what this module keeps of it:
    /// the segmentation's features cleared and the loop filter's deltas as
    /// a key frame starts them. The probability tables it resets are the
    /// accelerator's, which `reset_frame_context` and the frame's type tell.
    fn setup_past_independence(&mut self) {
        let segmentation = &mut self.segmentation;
        segmentation.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
        segmentation.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
        segmentation.abs_or_delta_update = false;
        self.ref_deltas = DEFAULT_REF_DELTAS;
        self.mode_deltas = [0; 2];
    }
}

/// A frame's uncompressed header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// `Profile`: 0 to 3.
    pub profile: u8,
    /// `frame_to_show_map_idx` of a frame with `show_existing_frame`, which
    /// shows that slot's frame and has no other field.
    pub show_existing_frame: Option<u8>,
    /// `frame_type` is `KEY_FRAME`.
    pub key_frame: bool,
    /// `show_frame`.
    pub show_frame: bool,
    /// `error_resilient_mode`.
    pub error_resilient_mode: bool,
    /// `intra_only`.
    pub intra_only: bool,
    /// `reset_frame_context`: 0 where not coded.
    pub reset_frame_context: u8,
    /// The colour configuration the frame decodes with.
    pub color: ColorConfig,
    /// `refresh_frame_flags`: bit `i` stores the frame in slot `i`.
    pub refresh_frame_flags: u8,
    /// `ref_frame_idx`: the slots of the last, golden and altref frames;
    /// zeros in an intra frame.
    pub ref_frame_idx: [u8; 3],
    /// `ref_frame_sign_bias` of the last, golden and altref frames.
    pub ref_frame_sign_bias: [bool; 3],
    /// `FrameWidth`: 1 to 65536.
    pub width: u32,
    /// `FrameHeight`.
    pub height: u32,
    /// `RenderWidth`.
    pub render_width: u32,
    /// `RenderHeight`.
    pub render_height: u32,
    /// `allow_high_precision_mv`: false in an intra frame.
    pub allow_high_precision_mv: bool,
    /// `interp_filter`, in libvpx's numbering ([`SWITCHABLE`]); 0 in an
    /// intra frame.
    pub interp_filter: u8,
    /// `refresh_frame_context`.
    pub refresh_frame_context: bool,
    /// `frame_parallel_decoding_mode`.
    pub frame_parallel_decoding_mode: bool,
    /// `frame_context_idx` as coded, but 0 in a key frame and an intra-only
    /// frame, which libvpx and ffmpeg hand their hardware so.
    pub frame_context_idx: u8,
    /// The loop filter.
    pub loop_filter: LoopFilter,
    /// The quantizer.
    pub quantization: Quantization,
    /// The segmentation.
    pub segmentation: Segmentation,
    /// `tile_cols_log2`.
    pub tile_cols_log2: u8,
    /// `tile_rows_log2`.
    pub tile_rows_log2: u8,
    /// `header_size_in_bytes`: the compressed header's.
    pub header_size_in_bytes: u16,
    /// The uncompressed header's bytes, `trailing_bits()` included.
    pub uncompressed_header_size: u8,
}

impl FrameHeader {
    /// `FrameIsIntra`.
    pub fn is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }

    /// A frame that shows slot `index`'s frame.
    fn show_existing(profile: u8, index: u8, state: &State, header_bytes: usize) -> Self {
        let color = state.color.unwrap_or(ColorConfig {
            bit_depth: 8,
            color_space: 0,
            full_range: false,
            subsampling_x: true,
            subsampling_y: true,
        });
        Self {
            profile,
            show_existing_frame: Some(index),
            key_frame: false,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            reset_frame_context: 0,
            color,
            refresh_frame_flags: 0,
            ref_frame_idx: [0; 3],
            ref_frame_sign_bias: [false; 3],
            width: 0,
            height: 0,
            render_width: 0,
            render_height: 0,
            allow_high_precision_mv: false,
            interp_filter: 0,
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
            loop_filter: LoopFilter::default(),
            quantization: Quantization::default(),
            segmentation: state.segmentation,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            header_size_in_bytes: 0,
            uncompressed_header_size: u8::try_from(header_bytes).unwrap_or(u8::MAX),
        }
    }
}

/// Reads `frame`'s uncompressed header against `state` and the sizes of the
/// frames in the eight reference slots, which an inter frame's size may be
/// taken from, and answers it with the state it leaves. `state` itself is
/// left as it was, for a caller that drops a frame that does not decode.
///
/// A compressed header past the frame, or empty, is
/// [`Error::BadHeaderSize`]; an inter frame before any frame gave a colour
/// configuration, or taking its size from an empty slot, is
/// [`Error::MissingReference`].
pub fn parse(
    frame: &[u8],
    state: &State,
    ref_sizes: &[Option<(u32, u32)>; 8],
) -> Result<(FrameHeader, State), Error> {
    let mut r = BitReader::new(frame);
    let mut next = state.clone();
    if r.u8(2)? != 2 {
        return Err(Error::OutOfRange("frame_marker"));
    }
    let low = u8::from(r.bit()?);
    let high = u8::from(r.bit()?);
    let profile = (high << 1) | low;
    if profile == 3 && r.bit()? {
        return Err(Error::OutOfRange("reserved_zero"));
    }
    if r.bit()? {
        let index = r.u8(3)?;
        let header = FrameHeader::show_existing(profile, index, state, r.bytes_read());
        return Ok((header, next));
    }
    let key_frame = !r.bit()?;
    let show_frame = r.bit()?;
    let error_resilient_mode = r.bit()?;
    let (intra_only, reset_frame_context) = if key_frame {
        (false, 0)
    } else {
        let intra_only = !show_frame && r.bit()?;
        let reset = if error_resilient_mode { 0 } else { r.u8(2)? };
        (intra_only, reset)
    };
    let mut ref_frame_idx = [0u8; 3];
    let mut ref_frame_sign_bias = [false; 3];
    let mut allow_high_precision_mv = false;
    let mut interp_filter = 0;
    // Each branch reads `render_size()` right after the frame's size, as
    // `frame_size_with_refs()` does (6.2.5).
    let (color, refresh_frame_flags, (width, height), (render_width, render_height)) = if key_frame
    {
        sync_code(&mut r)?;
        let color = color_config(&mut r, profile)?;
        let size = frame_size(&mut r)?;
        (color, 0xff, size, render_size(&mut r, size)?)
    } else if intra_only {
        sync_code(&mut r)?;
        let color = if profile > 0 {
            color_config(&mut r, profile)?
        } else {
            // Profile 0's intra-only frame codes no colour configuration:
            // 4:2:0 at 8 bits, BT.601, and libvpx's studio range.
            ColorConfig {
                bit_depth: 8,
                color_space: CS_BT_601,
                full_range: false,
                subsampling_x: true,
                subsampling_y: true,
            }
        };
        let refresh = r.u8(8)?;
        let size = frame_size(&mut r)?;
        (color, refresh, size, render_size(&mut r, size)?)
    } else {
        let color = state.color.ok_or(Error::MissingReference)?;
        let refresh = r.u8(8)?;
        for (index, bias) in ref_frame_idx.iter_mut().zip(&mut ref_frame_sign_bias) {
            *index = r.u8(3)?;
            *bias = r.bit()?;
        }
        let size = frame_size_with_refs(&mut r, &ref_frame_idx, ref_sizes)?;
        let render = render_size(&mut r, size)?;
        allow_high_precision_mv = r.bit()?;
        interp_filter = read_interpolation_filter(&mut r)?;
        (color, refresh, size, render)
    };
    let (refresh_frame_context, frame_parallel_decoding_mode) = if error_resilient_mode {
        (false, true)
    } else {
        (r.bit()?, r.bit()?)
    };
    let mut frame_context_idx = r.u8(2)?;
    if key_frame || intra_only {
        frame_context_idx = 0;
    }
    next.color = Some(color);
    if key_frame || intra_only || error_resilient_mode {
        next.setup_past_independence();
    }
    let loop_filter = loop_filter_params(&mut r, &mut next)?;
    let quantization = quantization_params(&mut r)?;
    let segmentation = segmentation_params(&mut r, &mut next)?;
    let (tile_cols_log2, tile_rows_log2) = tile_info(&mut r, width)?;
    let header_size_in_bytes = r.bits(16)?;
    let uncompressed = r.bytes_read();
    let past = uncompressed.checked_add(usize::from(header_size_in_bytes));
    if header_size_in_bytes == 0 || past.is_none_or(|end| end > frame.len()) {
        return Err(Error::BadHeaderSize);
    }
    let header = FrameHeader {
        profile,
        show_existing_frame: None,
        key_frame,
        show_frame,
        error_resilient_mode,
        intra_only,
        reset_frame_context,
        color,
        refresh_frame_flags,
        ref_frame_idx,
        ref_frame_sign_bias,
        width,
        height,
        render_width,
        render_height,
        allow_high_precision_mv,
        interp_filter,
        refresh_frame_context,
        frame_parallel_decoding_mode,
        frame_context_idx,
        loop_filter,
        quantization,
        segmentation,
        tile_cols_log2,
        tile_rows_log2,
        header_size_in_bytes,
        uncompressed_header_size: u8::try_from(uncompressed)
            .map_err(|_| Error::OutOfRange("the uncompressed header's size"))?,
    };
    Ok((header, next))
}

/// `frame_sync_code()` (6.2.1).
fn sync_code(r: &mut BitReader<'_>) -> Result<(), Error> {
    for expected in SYNC_CODE {
        if r.u8(8)? != expected {
            return Err(Error::OutOfRange("frame_sync_code"));
        }
    }
    Ok(())
}

/// `color_config()` (6.2.2).
fn color_config(r: &mut BitReader<'_>, profile: u8) -> Result<ColorConfig, Error> {
    let bit_depth = if profile >= 2 {
        if r.bit()? { 12 } else { 10 }
    } else {
        8
    };
    let color_space = r.u8(3)?;
    let odd_profile = profile == 1 || profile == 3;
    let (full_range, subsampling_x, subsampling_y) = if color_space != CS_RGB {
        let full_range = r.bit()?;
        if odd_profile {
            let x = r.bit()?;
            let y = r.bit()?;
            if r.bit()? {
                return Err(Error::OutOfRange("reserved_zero"));
            }
            (full_range, x, y)
        } else {
            (full_range, true, true)
        }
    } else if odd_profile {
        if r.bit()? {
            return Err(Error::OutOfRange("reserved_zero"));
        }
        (true, false, false)
    } else {
        // 4:4:4 RGB is profile 1's and 3's alone (7.2.2).
        return Err(Error::Unsupported("RGB in profile 0 or 2"));
    };
    Ok(ColorConfig {
        bit_depth,
        color_space,
        full_range,
        subsampling_x,
        subsampling_y,
    })
}

/// `frame_size()` (6.2.3): each side's `f(16)` plus one.
fn frame_size(r: &mut BitReader<'_>) -> Result<(u32, u32), Error> {
    let width = u32::from(r.bits(16)?).saturating_add(1);
    let height = u32::from(r.bits(16)?).saturating_add(1);
    Ok((width, height))
}

/// `render_size()` (6.2.4).
fn render_size(r: &mut BitReader<'_>, frame: (u32, u32)) -> Result<(u32, u32), Error> {
    if r.bit()? { frame_size(r) } else { Ok(frame) }
}

/// `frame_size_with_refs()` (6.2.5) up to its `render_size()`: the first
/// reference whose `found_ref` is set gives the size, else it is coded.
fn frame_size_with_refs(
    r: &mut BitReader<'_>,
    ref_frame_idx: &[u8; 3],
    ref_sizes: &[Option<(u32, u32)>; 8],
) -> Result<(u32, u32), Error> {
    for &index in ref_frame_idx {
        if r.bit()? {
            return ref_sizes
                .get(usize::from(index))
                .copied()
                .flatten()
                .ok_or(Error::MissingReference);
        }
    }
    frame_size(r)
}

/// `read_interpolation_filter()` (6.2.7), in libvpx's numbering: its
/// `literal_to_filter` maps the coded 0 to `EIGHTTAP_SMOOTH` (1) and 1 to
/// `EIGHTTAP` (0), and keeps 2 (sharp) and 3 (bilinear).
fn read_interpolation_filter(r: &mut BitReader<'_>) -> Result<u8, Error> {
    if r.bit()? {
        return Ok(SWITCHABLE);
    }
    let literal = r.u8(2)?;
    Ok(if literal <= 1 { literal ^ 1 } else { literal })
}

/// `loop_filter_params()` (6.2.8): the deltas coded update those `state`
/// keeps.
fn loop_filter_params(r: &mut BitReader<'_>, state: &mut State) -> Result<LoopFilter, Error> {
    let level = r.u8(6)?;
    let sharpness = r.u8(3)?;
    let delta_enabled = r.bit()?;
    if delta_enabled && r.bit()? {
        for delta in &mut state.ref_deltas {
            if r.bit()? {
                *delta = narrow_i8(r.su(6)?);
            }
        }
        for delta in &mut state.mode_deltas {
            if r.bit()? {
                *delta = narrow_i8(r.su(6)?);
            }
        }
    }
    Ok(LoopFilter {
        level,
        sharpness,
        delta_enabled,
        ref_deltas: state.ref_deltas,
        mode_deltas: state.mode_deltas,
    })
}

/// A value `su(6)` or `su(4)` read, at most 63 in magnitude.
fn narrow_i8(value: i16) -> i8 {
    i8::try_from(value).unwrap_or(0)
}

/// `quantization_params()` (6.2.9).
fn quantization_params(r: &mut BitReader<'_>) -> Result<Quantization, Error> {
    let base_q_idx = r.u8(8)?;
    let mut delta = || -> Result<i8, Error> { Ok(if r.bit()? { narrow_i8(r.su(4)?) } else { 0 }) };
    Ok(Quantization {
        base_q_idx,
        delta_q_y_dc: delta()?,
        delta_q_uv_dc: delta()?,
        delta_q_uv_ac: delta()?,
    })
}

/// `read_prob()` (6.2.11).
fn read_prob(r: &mut BitReader<'_>) -> Result<u8, Error> {
    if r.bit()? { r.u8(8) } else { Ok(255) }
}

/// `segmentation_params()` (6.2.11): the probabilities and features coded
/// update those `state` keeps.
fn segmentation_params(r: &mut BitReader<'_>, state: &mut State) -> Result<Segmentation, Error> {
    let kept = &mut state.segmentation;
    kept.enabled = r.bit()?;
    kept.update_map = false;
    kept.temporal_update = false;
    if !kept.enabled {
        return Ok(*kept);
    }
    kept.update_map = r.bit()?;
    if kept.update_map {
        for prob in &mut kept.tree_probs {
            *prob = read_prob(r)?;
        }
        kept.temporal_update = r.bit()?;
        for prob in &mut kept.pred_probs {
            *prob = if kept.temporal_update {
                read_prob(r)?
            } else {
                255
            };
        }
    }
    if r.bit()? {
        kept.abs_or_delta_update = r.bit()?;
        for (enabled, data) in kept
            .feature_enabled
            .iter_mut()
            .zip(kept.feature_data.iter_mut())
        {
            for (((enabled, data), &bits), &signed) in enabled
                .iter_mut()
                .zip(data.iter_mut())
                .zip(&FEATURE_BITS)
                .zip(&FEATURE_SIGNED)
            {
                *enabled = r.bit()?;
                let mut value = 0;
                if *enabled {
                    value = i16::try_from(r.bits(bits)?)
                        .map_err(|_| Error::OutOfRange("feature_value"))?;
                    if signed && r.bit()? {
                        value = value.wrapping_neg();
                    }
                }
                *data = value;
            }
        }
    }
    Ok(*kept)
}

/// `tile_info()` (6.2.12), for a frame `width` wide.
fn tile_info(r: &mut BitReader<'_>, width: u32) -> Result<(u8, u8), Error> {
    // MiCols and Sb64Cols (7.2.6): at most 8192 and 1024 for a width of
    // 65536.
    let mi_cols = width.saturating_add(7) >> 3;
    let sb64_cols = mi_cols.saturating_add(7) >> 3;
    let mut min_log2 = 0u8;
    while (64u32 << min_log2) < sb64_cols {
        min_log2 = min_log2.saturating_add(1);
    }
    let mut max_log2 = 1u8;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 = max_log2.saturating_add(1);
    }
    let max_log2 = max_log2.saturating_sub(1);
    let mut cols = min_log2;
    while cols < max_log2 {
        if r.bit()? {
            cols = cols.saturating_add(1);
        } else {
            break;
        }
    }
    let mut rows = u8::from(r.bit()?);
    if rows == 1 {
        rows = rows.saturating_add(u8::from(r.bit()?));
    }
    Ok((cols, rows))
}
