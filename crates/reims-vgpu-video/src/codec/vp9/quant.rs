//! Each segment's quantizer scales and loop filter levels, as VA-API's
//! `VASegmentParameterVP9` takes them already derived: the quantizer index
//! of the segment (`get_qindex()`, 8.6.1) looked up in the specification's
//! tables, and the filter level of each reference frame and mode (8.8.1).
//! NVDEC takes the header's own fields and derives these itself.

mod tables;

use super::header::{FrameHeader, MAX_SEGMENTS};

/// The segment feature that sets the quantizer (`SEG_LVL_ALT_Q`).
const SEG_LVL_ALT_Q: usize = 0;
/// The segment feature that sets the loop filter level (`SEG_LVL_ALT_L`).
const SEG_LVL_ALT_L: usize = 1;
/// The highest quantizer index.
const MAX_Q_INDEX: i32 = 255;
/// The highest loop filter level.
const MAX_LOOP_FILTER: i32 = 63;

/// One segment's derived values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentParams {
    /// The luma DC coefficients' quantizer scale.
    pub luma_dc: i16,
    /// The luma AC coefficients' scale.
    pub luma_ac: i16,
    /// The chroma DC coefficients' scale.
    pub chroma_dc: i16,
    /// The chroma AC coefficients' scale.
    pub chroma_ac: i16,
    /// The loop filter level of each reference frame (intra, last, golden,
    /// altref) and mode delta (0: a zero motion vector, 1: another),
    /// `lvlSeg` with the deltas (8.8.1).
    pub filter_level: [[u8; 2]; 4],
}

/// The DC and AC tables of `header`'s bit depth: 8, 10 or 12 bits.
fn tables_of(header: &FrameHeader) -> (&'static [i16; 256], &'static [i16; 256]) {
    let [dc8, dc10, dc12] = &tables::DC_QLOOKUP;
    let [ac8, ac10, ac12] = &tables::AC_QLOOKUP;
    match header.color.bit_depth {
        8 => (dc8, ac8),
        10 => (dc10, ac10),
        _ => (dc12, ac12),
    }
}

/// `value` within `0..=max`.
fn clip(value: i32, max: i32) -> i32 {
    value.clamp(0, max)
}

/// The tables' entry at a clipped index.
fn lookup(table: &[i16; 256], index: i32) -> i16 {
    usize::try_from(clip(index, MAX_Q_INDEX))
        .ok()
        .and_then(|index| table.get(index))
        .copied()
        .unwrap_or(0)
}

/// Every segment's values for `header`: those of each segment's features
/// where segmentation is on, else the frame's own for all eight.
pub fn segment_params(header: &FrameHeader) -> [SegmentParams; MAX_SEGMENTS] {
    let (dc_table, ac_table) = tables_of(header);
    let q = &header.quantization;
    let segmentation = &header.segmentation;
    let filter = &header.loop_filter;
    let mut out = [SegmentParams::default(); MAX_SEGMENTS];
    for (segment, params) in out.iter_mut().enumerate() {
        let feature = |level: usize| -> Option<i32> {
            let enabled = segmentation.feature_enabled.get(segment)?.get(level)?;
            let data = segmentation.feature_data.get(segment)?.get(level)?;
            (segmentation.enabled && *enabled).then_some(i32::from(*data))
        };
        // get_qindex() (8.6.1): the feature's value, absolute or a delta.
        let base = i32::from(q.base_q_idx);
        let q_index = match feature(SEG_LVL_ALT_Q) {
            Some(value) if segmentation.abs_or_delta_update => clip(value, MAX_Q_INDEX),
            Some(value) => clip(base.saturating_add(value), MAX_Q_INDEX),
            None => base,
        };
        params.luma_dc = lookup(dc_table, q_index.saturating_add(i32::from(q.delta_q_y_dc)));
        params.luma_ac = lookup(ac_table, q_index);
        params.chroma_dc = lookup(dc_table, q_index.saturating_add(i32::from(q.delta_q_uv_dc)));
        params.chroma_ac = lookup(ac_table, q_index.saturating_add(i32::from(q.delta_q_uv_ac)));
        // 8.8.1: the segment's level, then each reference's and mode's.
        let level = i32::from(filter.level);
        let segment_level = match feature(SEG_LVL_ALT_L) {
            Some(value) if segmentation.abs_or_delta_update => clip(value, MAX_LOOP_FILTER),
            Some(value) => clip(level.saturating_add(value), MAX_LOOP_FILTER),
            None => level,
        };
        let shift = if segment_level >= 32 { 1 } else { 0 };
        let narrow = |value: i32| u8::try_from(clip(value, MAX_LOOP_FILTER)).unwrap_or(0);
        for (reference, levels) in params.filter_level.iter_mut().enumerate() {
            let ref_delta = i32::from(filter.ref_deltas.get(reference).copied().unwrap_or(0));
            for (mode, level) in levels.iter_mut().enumerate() {
                *level = if !filter.delta_enabled {
                    narrow(segment_level)
                } else if reference == 0 {
                    // An intra block has no mode delta.
                    narrow(segment_level.saturating_add(ref_delta << shift))
                } else {
                    let mode_delta = i32::from(filter.mode_deltas.get(mode).copied().unwrap_or(0));
                    narrow(
                        segment_level.saturating_add(ref_delta.saturating_add(mode_delta) << shift),
                    )
                };
            }
        }
    }
    out
}
