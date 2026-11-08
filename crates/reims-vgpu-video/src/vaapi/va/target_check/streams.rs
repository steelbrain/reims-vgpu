//! The check's streams (`super`): short H.264 and HEVC streams of our own, written here, whose
//! every decoded sample is known. Each starts with an IDR picture coded in PCM, every sample in
//! the stream as it is decoded ([`sample`]); each later picture predicts all of itself from the
//! one before, every block skipped, so it decodes to the same samples, read from the reference.
//! Nothing filters a picture: H.264's deblocking is off in each slice, and HEVC's in the picture
//! parameter set, with no SAO.
//!
//! - **H.264** (ITU-T H.264 08/2021): Constrained Baseline, CAVLC. The IDR picture's macroblocks
//!   are `I_PCM`; each P picture is one `mb_skip_run` of every macroblock. One reference frame,
//!   so each P picture's reference is the picture before it. A gap in `frame_num` where the
//!   sequence allows none, before the last picture, has the decoder stand a copy of the picture
//!   before in for the frame lost (8.2.5.2), which the last picture is then predicted from.
//! - **HEVC** (ITU-T H.265 09/2023): Main, or Main 10 at 10 bits, CABAC, 16x16 coding tree
//!   blocks with 8x8 coding blocks the least, as many rows as leave the last row of blocks half
//!   in the picture. The IDR picture's coding units are PCM (`pcm_flag`); each P picture's are
//!   skipped (`cu_skip_flag`) with one merge candidate, which is the zero vector to the picture
//!   before (no temporal candidate).

use crate::codec::encode::bits::BitWriter;
use crate::codec::encode::nal::{START_CODE, escape};

/// A stream of the check's (module docs): its access units, Annex B, in decode order, every one
/// a picture, and the pictures' size and bit depth.
pub(crate) struct Stream {
    pub(crate) units: Vec<Vec<u8>>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bit_depth: u32,
}

/// The sample of `plane` (0 luma, 1 Cb, 2 Cr) at column `x` and row `y` of that plane, at
/// `bit_depth`, which every picture of the check's streams decodes to: each a value well inside
/// the range, so that no sample of a picture is the one of a neighbour by chance.
pub(crate) fn sample(plane: usize, x: u32, y: u32, bit_depth: u32) -> u16 {
    let (base, dx, dy, span) = match plane {
        0 => (16u32, 3u32, 7u32, 220u32),
        1 => (24, 5, 2, 200),
        _ => (36, 2, 9, 190),
    };
    let value = base + (x.wrapping_mul(dx).wrapping_add(y.wrapping_mul(dy))) % span;
    // At 10 bits the same pattern, four times as fine.
    let value = if bit_depth > 8 {
        (value << 2) + (x.wrapping_add(y) & 3)
    } else {
        value
    };
    u16::try_from(value).unwrap_or(u16::MAX)
}

/// An H.264 NAL unit of `nal_ref_idc` and `nal_unit_type` holding `rbsp`, after a start code.
fn h264_unit(out: &mut Vec<u8>, nal_ref_idc: u8, nal_unit_type: u8, rbsp: BitWriter) {
    out.extend_from_slice(&START_CODE);
    out.extend(escape(
        &[(nal_ref_idc << 5) | nal_unit_type],
        &rbsp.into_bytes(),
    ));
}

/// The check's H.264 stream (module docs) of `width` by `height` samples, each a multiple of 16:
/// an IDR picture, two P pictures, then one after a gap in `frame_num`.
pub(crate) fn h264(width: u32, height: u32) -> Stream {
    let (mbs_wide, mbs_high) = (width / 16, height / 16);
    let mut units = Vec::new();

    let mut idr = Vec::new();
    // Sequence parameter set (7.3.2.1.1): Constrained Baseline, level 3, frame_num of 4 bits,
    // picture order of type 2 (the decode order), one reference frame, no gaps in frame_num.
    let mut sps = BitWriter::new();
    sps.u(8, 66);
    sps.u(8, 0b0100_0000); // constraint_set1_flag: Constrained Baseline
    sps.u(8, 30);
    sps.ue(0); // seq_parameter_set_id
    sps.ue(0); // log2_max_frame_num_minus4
    sps.ue(2); // pic_order_cnt_type
    sps.ue(1); // max_num_ref_frames
    sps.flag(false); // gaps_in_frame_num_value_allowed_flag
    sps.ue(mbs_wide - 1);
    sps.ue(mbs_high - 1);
    sps.flag(true); // frame_mbs_only_flag
    sps.flag(true); // direct_8x8_inference_flag
    sps.flag(false); // frame_cropping_flag
    sps.flag(false); // vui_parameters_present_flag
    sps.trailing_bits();
    h264_unit(&mut idr, 3, 7, sps);
    // Picture parameter set (7.3.2.2): CAVLC, deblocking controlled per slice.
    let mut pps = BitWriter::new();
    pps.ue(0); // pic_parameter_set_id
    pps.ue(0); // seq_parameter_set_id
    pps.flag(false); // entropy_coding_mode_flag
    pps.flag(false); // bottom_field_pic_order_in_frame_present_flag
    pps.ue(0); // num_slice_groups_minus1
    pps.ue(0); // num_ref_idx_l0_default_active_minus1
    pps.ue(0); // num_ref_idx_l1_default_active_minus1
    pps.flag(false); // weighted_pred_flag
    pps.u(2, 0); // weighted_bipred_idc
    pps.se(0); // pic_init_qp_minus26
    pps.se(0); // pic_init_qs_minus26
    pps.se(0); // chroma_qp_index_offset
    pps.flag(true); // deblocking_filter_control_present_flag
    pps.flag(false); // constrained_intra_pred_flag
    pps.flag(false); // redundant_pic_cnt_present_flag
    pps.trailing_bits();
    h264_unit(&mut idr, 3, 8, pps);
    // The IDR slice (7.3.3, 7.3.4): every macroblock I_PCM (7.3.5).
    let mut slice = BitWriter::new();
    slice.ue(0); // first_mb_in_slice
    slice.ue(7); // slice_type: I, every slice of the picture
    slice.ue(0); // pic_parameter_set_id
    slice.u(4, 0); // frame_num
    slice.ue(0); // idr_pic_id
    slice.flag(false); // no_output_of_prior_pics_flag
    slice.flag(false); // long_term_reference_flag
    slice.se(0); // slice_qp_delta
    slice.ue(1); // disable_deblocking_filter_idc: off
    for mb_y in 0..mbs_high {
        for mb_x in 0..mbs_wide {
            slice.ue(25); // mb_type: I_PCM
            while !slice.byte_aligned() {
                slice.bit(false); // pcm_alignment_zero_bit
            }
            for y in 0..16 {
                for x in 0..16 {
                    slice.u(8, u32::from(sample(0, mb_x * 16 + x, mb_y * 16 + y, 8)));
                }
            }
            for plane in 1..3 {
                for y in 0..8 {
                    for x in 0..8 {
                        slice.u(8, u32::from(sample(plane, mb_x * 8 + x, mb_y * 8 + y, 8)));
                    }
                }
            }
        }
    }
    slice.trailing_bits();
    h264_unit(&mut idr, 3, 5, slice);
    units.push(idr);

    // P slices, every macroblock skipped: frame_num 1 and 2, then 4, past a gap.
    for frame_num in [1, 2, 4] {
        let mut slice = BitWriter::new();
        slice.ue(0); // first_mb_in_slice
        slice.ue(5); // slice_type: P, every slice of the picture
        slice.ue(0); // pic_parameter_set_id
        slice.u(4, frame_num);
        slice.flag(false); // num_ref_idx_active_override_flag
        slice.flag(false); // ref_pic_list_modification_flag_l0
        slice.flag(false); // adaptive_ref_pic_marking_mode_flag
        slice.se(0); // slice_qp_delta
        slice.ue(1); // disable_deblocking_filter_idc: off
        slice.ue(mbs_wide * mbs_high); // mb_skip_run: every macroblock
        slice.trailing_bits();
        let mut unit = Vec::new();
        h264_unit(&mut unit, 2, 1, slice);
        units.push(unit);
    }
    Stream {
        units,
        width,
        height,
        bit_depth: 8,
    }
}

/// An HEVC NAL unit of `nal_unit_type` holding `rbsp`, after a start code: layer 0, temporal
/// sub-layer 0.
fn hevc_unit(out: &mut Vec<u8>, nal_unit_type: u8, rbsp: BitWriter) {
    out.extend_from_slice(&START_CODE);
    out.extend(escape(&[nal_unit_type << 1, 1], &rbsp.into_bytes()));
}

/// `profile_tier_level(1, 0)` (7.3.3): Main tier, Main or Main 10, level 3.1.
fn profile_tier_level(w: &mut BitWriter, main10: bool) {
    let profile = if main10 { 2 } else { 1 };
    w.u(2, 0); // general_profile_space
    w.flag(false); // general_tier_flag
    w.u(5, profile);
    // general_profile_compatibility_flag[j]: Main is Main 10's too.
    for j in 0..32 {
        w.flag(j == profile || (!main10 && j == 2));
    }
    w.flag(true); // general_progressive_source_flag
    w.flag(false); // general_interlaced_source_flag
    w.flag(false); // general_non_packed_constraint_flag
    w.flag(true); // general_frame_only_constraint_flag
    w.bits(43, 0); // general_reserved_zero_43bits
    w.flag(false); // general_reserved_zero_bit
    w.u(8, 93); // general_level_idc
}

/// The check's HEVC stream (module docs) of `width` by `height` samples, each a multiple of 8, at
/// `bit_depth` (8 or 10): an IDR picture, then two P pictures.
pub(crate) fn hevc(width: u32, height: u32, bit_depth: u32) -> Stream {
    let main10 = bit_depth > 8;
    let mut units = Vec::new();
    let mut idr = Vec::new();

    // Video parameter set (7.3.2.1).
    let mut vps = BitWriter::new();
    vps.u(4, 0); // vps_video_parameter_set_id
    vps.flag(true); // vps_base_layer_internal_flag
    vps.flag(true); // vps_base_layer_available_flag
    vps.u(6, 0); // vps_max_layers_minus1
    vps.u(3, 0); // vps_max_sub_layers_minus1
    vps.flag(true); // vps_temporal_id_nesting_flag
    vps.u(16, 0xffff); // vps_reserved_0xffff_16bits
    profile_tier_level(&mut vps, main10);
    vps.flag(true); // vps_sub_layer_ordering_info_present_flag
    vps.ue(1); // vps_max_dec_pic_buffering_minus1
    vps.ue(0); // vps_max_num_reorder_pics
    vps.ue(0); // vps_max_latency_increase_plus1
    vps.u(6, 0); // vps_max_layer_id
    vps.ue(0); // vps_num_layer_sets_minus1
    vps.flag(false); // vps_timing_info_present_flag
    vps.flag(false); // vps_extension_flag
    vps.trailing_bits();
    hevc_unit(&mut idr, 32, vps);

    // Sequence parameter set (7.3.2.2): 4:2:0, CTBs of 16, coding blocks of 8 the least, PCM of
    // 8x8 and 16x16 blocks at the bit depth, unfiltered.
    let mut sps = BitWriter::new();
    sps.u(4, 0); // sps_video_parameter_set_id
    sps.u(3, 0); // sps_max_sub_layers_minus1
    sps.flag(true); // sps_temporal_id_nesting_flag
    profile_tier_level(&mut sps, main10);
    sps.ue(0); // sps_seq_parameter_set_id
    sps.ue(1); // chroma_format_idc: 4:2:0
    sps.ue(width);
    sps.ue(height);
    sps.flag(false); // conformance_window_flag
    sps.ue(bit_depth - 8); // bit_depth_luma_minus8
    sps.ue(bit_depth - 8); // bit_depth_chroma_minus8
    sps.ue(0); // log2_max_pic_order_cnt_lsb_minus4
    sps.flag(true); // sps_sub_layer_ordering_info_present_flag
    sps.ue(1); // sps_max_dec_pic_buffering_minus1
    sps.ue(0); // sps_max_num_reorder_pics
    sps.ue(0); // sps_max_latency_increase_plus1
    sps.ue(0); // log2_min_luma_coding_block_size_minus3: 8
    sps.ue(1); // log2_diff_max_min_luma_coding_block_size: CTBs of 16
    sps.ue(0); // log2_min_luma_transform_block_size_minus2: 4
    sps.ue(2); // log2_diff_max_min_luma_transform_block_size: 16
    sps.ue(0); // max_transform_hierarchy_depth_inter
    sps.ue(0); // max_transform_hierarchy_depth_intra
    sps.flag(false); // scaling_list_enabled_flag
    sps.flag(false); // amp_enabled_flag
    sps.flag(false); // sample_adaptive_offset_enabled_flag
    sps.flag(true); // pcm_enabled_flag
    sps.u(4, bit_depth - 1); // pcm_sample_bit_depth_luma_minus1
    sps.u(4, bit_depth - 1); // pcm_sample_bit_depth_chroma_minus1
    sps.ue(0); // log2_min_pcm_luma_coding_block_size_minus3: 8
    sps.ue(1); // log2_diff_max_min_pcm_luma_coding_block_size: 16
    sps.flag(true); // pcm_loop_filter_disabled_flag
    sps.ue(0); // num_short_term_ref_pic_sets
    sps.flag(false); // long_term_ref_pics_present_flag
    sps.flag(false); // sps_temporal_mvp_enabled_flag
    sps.flag(false); // strong_intra_smoothing_enabled_flag
    sps.flag(false); // vui_parameters_present_flag
    sps.flag(false); // sps_extension_present_flag
    sps.trailing_bits();
    hevc_unit(&mut idr, 33, sps);

    // Picture parameter set (7.3.2.3): no deblocking, one reference.
    let mut pps = BitWriter::new();
    pps.ue(0); // pps_pic_parameter_set_id
    pps.ue(0); // pps_seq_parameter_set_id
    pps.flag(false); // dependent_slice_segments_enabled_flag
    pps.flag(false); // output_flag_present_flag
    pps.u(3, 0); // num_extra_slice_header_bits
    pps.flag(false); // sign_data_hiding_enabled_flag
    pps.flag(false); // cabac_init_present_flag
    pps.ue(0); // num_ref_idx_l0_default_active_minus1
    pps.ue(0); // num_ref_idx_l1_default_active_minus1
    pps.se(0); // init_qp_minus26
    pps.flag(false); // constrained_intra_pred_flag
    pps.flag(false); // transform_skip_enabled_flag
    pps.flag(false); // cu_qp_delta_enabled_flag
    pps.se(0); // pps_cb_qp_offset
    pps.se(0); // pps_cr_qp_offset
    pps.flag(false); // pps_slice_chroma_qp_offsets_present_flag
    pps.flag(false); // weighted_pred_flag
    pps.flag(false); // weighted_bipred_flag
    pps.flag(false); // transquant_bypass_enabled_flag
    pps.flag(false); // tiles_enabled_flag
    pps.flag(false); // entropy_coding_sync_enabled_flag
    pps.flag(false); // pps_loop_filter_across_slices_enabled_flag
    pps.flag(true); // deblocking_filter_control_present_flag
    pps.flag(false); // deblocking_filter_override_enabled_flag
    pps.flag(true); // pps_deblocking_filter_disabled_flag
    pps.flag(false); // pps_scaling_list_data_present_flag
    pps.flag(false); // lists_modification_present_flag
    pps.ue(0); // log2_parallel_merge_level_minus2
    pps.flag(false); // slice_segment_header_extension_present_flag
    pps.flag(false); // pps_extension_present_flag
    pps.trailing_bits();
    hevc_unit(&mut idr, 34, pps);

    // The IDR slice segment (7.3.6.1, IDR_W_RADL), every coding unit PCM.
    let mut slice = BitWriter::new();
    slice.flag(true); // first_slice_segment_in_pic_flag
    slice.flag(false); // no_output_of_prior_pics_flag
    slice.ue(0); // slice_pic_parameter_set_id
    slice.ue(2); // slice_type: I
    slice.se(0); // slice_qp_delta
    slice.trailing_bits(); // byte_alignment()
    let mut data = Cabac::new(slice, Contexts::new(SliceKind::Intra));
    coding_tree(&mut data, width, height, &mut |cabac, x0, y0, log2| {
        if log2 == MIN_CB_LOG2 {
            cabac.decision(Ctx::PartMode, true); // part_mode: PART_2Nx2N
        }
        cabac.terminate(true); // pcm_flag
        let size = 1u32 << log2;
        for y in 0..size {
            for x in 0..size {
                cabac.raw(bit_depth, sample(0, x0 + x, y0 + y, bit_depth));
            }
        }
        for plane in 1..3 {
            for y in 0..size / 2 {
                for x in 0..size / 2 {
                    cabac.raw(bit_depth, sample(plane, x0 / 2 + x, y0 / 2 + y, bit_depth));
                }
            }
        }
        cabac.restart();
    });
    hevc_unit(&mut idr, 19, data.finish());
    units.push(idr);

    // TRAIL_R slice segments, every coding unit skipped, each referring to the picture before.
    for poc in 1..3u32 {
        let mut slice = BitWriter::new();
        slice.flag(true); // first_slice_segment_in_pic_flag
        slice.ue(0); // slice_pic_parameter_set_id
        slice.ue(1); // slice_type: P
        slice.u(4, poc); // slice_pic_order_cnt_lsb
        slice.flag(false); // short_term_ref_pic_set_sps_flag
        // st_ref_pic_set(0) (7.3.7): the picture before, used.
        slice.ue(1); // num_negative_pics
        slice.ue(0); // num_positive_pics
        slice.ue(0); // delta_poc_s0_minus1
        slice.flag(true); // used_by_curr_pic_s0_flag
        slice.flag(false); // num_ref_idx_active_override_flag
        slice.ue(4); // five_minus_max_num_merge_cand: one candidate
        slice.se(0); // slice_qp_delta
        slice.trailing_bits(); // byte_alignment()
        let mut data = Cabac::new(slice, Contexts::new(SliceKind::Predicted));
        coding_tree(&mut data, width, height, &mut |cabac, x0, y0, _| {
            // cu_skip_flag, its context from whether the blocks left and above are in the
            // picture, every one being skipped (9.3.4.2.2).
            let inc = usize::from(x0 > 0) + usize::from(y0 > 0);
            cabac.decision(Ctx::CuSkip(inc), true);
        });
        let mut unit = Vec::new();
        hevc_unit(&mut unit, 1, data.finish());
        units.push(unit);
    }
    Stream {
        units,
        width,
        height,
        bit_depth,
    }
}

/// The least coding block's log2 size, and the coding tree block's.
const MIN_CB_LOG2: u32 = 3;
const CTB_LOG2: u32 = 4;

/// `slice_segment_data()` (7.3.8.1) of a picture of `width` by `height`: each coding tree unit's
/// coding quadtree (7.3.8.4), each coding unit written by `unit` with its corner and log2 size,
/// then `end_of_slice_segment_flag`.
fn coding_tree(
    cabac: &mut Cabac,
    width: u32,
    height: u32,
    unit: &mut dyn FnMut(&mut Cabac, u32, u32, u32),
) {
    let ctb = 1u32 << CTB_LOG2;
    let (ctbs_wide, ctbs_high) = (width.div_ceil(ctb), height.div_ceil(ctb));
    for row in 0..ctbs_high {
        for column in 0..ctbs_wide {
            quadtree(
                cabac,
                width,
                height,
                column * ctb,
                row * ctb,
                CTB_LOG2,
                unit,
            );
            cabac.terminate(row + 1 == ctbs_high && column + 1 == ctbs_wide);
        }
    }
}

/// `coding_quadtree(x0, y0, log2, 0)` (7.3.8.4) of a picture whose coding units are all one
/// size: a block wholly in the picture is one coding unit (`split_cu_flag` 0, written where the
/// block may split); one past its edge splits, as it must, into those of its quarters in it.
fn quadtree(
    cabac: &mut Cabac,
    width: u32,
    height: u32,
    x0: u32,
    y0: u32,
    log2: u32,
    unit: &mut dyn FnMut(&mut Cabac, u32, u32, u32),
) {
    let size = 1u32 << log2;
    let inside = x0 + size <= width && y0 + size <= height;
    if log2 > MIN_CB_LOG2 && !inside {
        let half = size / 2;
        for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
            if x0 + dx < width && y0 + dy < height {
                quadtree(cabac, width, height, x0 + dx, y0 + dy, log2 - 1, unit);
            }
        }
        return;
    }
    if log2 > MIN_CB_LOG2 {
        // Every block it could neighbour is unsplit, so its context is the first (9.3.4.2.2).
        cabac.decision(Ctx::SplitCu, false);
    }
    unit(cabac, x0, y0, log2);
}

/// A slice's kind, which its contexts are initialized for (9.3.2.2): initType 0 for I, 1 for P.
#[derive(Clone, Copy)]
enum SliceKind {
    Intra,
    Predicted,
}

/// The context variables the check's slices code bins with.
#[derive(Clone, Copy)]
enum Ctx {
    SplitCu,
    PartMode,
    CuSkip(usize),
}

/// A context variable: its probability state and most probable value (9.3.2.2).
#[derive(Clone, Copy)]
struct State {
    p_state_idx: u8,
    val_mps: bool,
}

/// The slice's context variables, initialized at slice QP 26.
struct Contexts {
    split_cu: State,
    part_mode: State,
    cu_skip: [State; 3],
}

impl Contexts {
    fn new(kind: SliceKind) -> Contexts {
        // initValue (Tables 9-11, 9-13, 9-14 of H.265) for ctxInc 0 (and 0 to 2 for
        // cu_skip_flag, which I slices do not code).
        let (split, part, skip) = match kind {
            SliceKind::Intra => (139, 184, [154, 154, 154]),
            SliceKind::Predicted => (107, 154, [197, 185, 201]),
        };
        Contexts {
            split_cu: init(split),
            part_mode: init(part),
            cu_skip: skip.map(init),
        }
    }

    fn get(&mut self, ctx: Ctx) -> &mut State {
        match ctx {
            Ctx::SplitCu => &mut self.split_cu,
            Ctx::PartMode => &mut self.part_mode,
            Ctx::CuSkip(inc) => &mut self.cu_skip[inc.min(2)],
        }
    }
}

/// A context variable of `init_value` at SliceQpY 26 (9.3.2.2, 9-6).
fn init(init_value: i32) -> State {
    const SLICE_QP: i32 = 26;
    let slope_idx = init_value >> 4;
    let offset_idx = init_value & 15;
    let m = slope_idx * 5 - 45;
    let n = (offset_idx << 3) - 16;
    let pre = (((m * SLICE_QP) >> 4) + n).clamp(1, 126);
    let val_mps = pre > 63;
    let p_state_idx = if val_mps { pre - 64 } else { 63 - pre };
    State {
        p_state_idx: u8::try_from(p_state_idx).unwrap_or(0),
        val_mps,
    }
}

/// `rangeTabLps[pStateIdx][qRangeIdx]` (H.265 Table 9-52, H.264 Table 9-44).
const RANGE_TAB_LPS: [[u8; 4]; 64] = [
    [128, 176, 208, 240],
    [128, 167, 197, 227],
    [128, 158, 187, 216],
    [123, 150, 178, 205],
    [116, 142, 169, 195],
    [111, 135, 160, 185],
    [105, 128, 152, 175],
    [100, 122, 144, 166],
    [95, 116, 137, 158],
    [90, 110, 130, 150],
    [85, 104, 123, 142],
    [81, 99, 117, 135],
    [77, 94, 111, 128],
    [73, 89, 105, 122],
    [69, 85, 100, 116],
    [66, 80, 95, 110],
    [62, 76, 90, 104],
    [59, 72, 86, 99],
    [56, 69, 81, 94],
    [53, 65, 77, 89],
    [51, 62, 73, 85],
    [48, 59, 69, 80],
    [46, 56, 66, 76],
    [43, 53, 63, 72],
    [41, 50, 59, 69],
    [39, 48, 56, 65],
    [37, 45, 54, 62],
    [35, 43, 51, 59],
    [33, 41, 48, 56],
    [32, 39, 46, 53],
    [30, 37, 43, 50],
    [29, 35, 41, 48],
    [27, 33, 39, 45],
    [26, 31, 37, 43],
    [24, 30, 35, 41],
    [23, 28, 33, 39],
    [22, 27, 32, 37],
    [21, 26, 30, 35],
    [20, 24, 29, 33],
    [19, 23, 27, 31],
    [18, 22, 26, 30],
    [17, 21, 25, 28],
    [16, 20, 23, 27],
    [15, 19, 22, 25],
    [14, 18, 21, 24],
    [14, 17, 20, 23],
    [13, 16, 19, 22],
    [12, 15, 18, 21],
    [12, 14, 17, 20],
    [11, 14, 16, 19],
    [11, 13, 15, 18],
    [10, 12, 15, 17],
    [10, 12, 14, 16],
    [9, 11, 13, 15],
    [9, 11, 12, 14],
    [8, 10, 12, 14],
    [8, 9, 11, 13],
    [7, 9, 11, 12],
    [7, 9, 10, 12],
    [7, 8, 10, 11],
    [6, 8, 9, 11],
    [6, 7, 9, 10],
    [6, 7, 8, 9],
    [2, 2, 2, 2],
];

/// `transIdxLps[pStateIdx]` (H.265 Table 9-53); `transIdxMps` is one more, at most 62.
const TRANS_IDX_LPS: [u8; 64] = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12, 13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21,
    21, 22, 22, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33, 33, 33, 34,
    34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

/// CABAC's arithmetic encoder (H.264 9.3.4.2, which H.265's decoding process inverts), writing
/// after a slice segment header.
struct Cabac {
    out: BitWriter,
    contexts: Contexts,
    low: u32,
    range: u32,
    first_bit: bool,
    outstanding: u32,
}

impl Cabac {
    fn new(out: BitWriter, contexts: Contexts) -> Cabac {
        Cabac {
            out,
            contexts,
            low: 0,
            range: 510,
            first_bit: true,
            outstanding: 0,
        }
    }

    /// The engine initialized again (9.3.2.5), after PCM samples; the contexts are kept.
    fn restart(&mut self) {
        self.low = 0;
        self.range = 510;
        self.first_bit = true;
        self.outstanding = 0;
    }

    /// `EncodeDecision` (9.3.4.2) of `bin` in `ctx`.
    fn decision(&mut self, ctx: Ctx, bin: bool) {
        let state = self.contexts.get(ctx);
        let p = usize::from(state.p_state_idx);
        let q = usize::try_from((self.range >> 6) & 3).unwrap_or(0);
        let lps = u32::from(RANGE_TAB_LPS[p][q]);
        self.range -= lps;
        if bin == state.val_mps {
            state.p_state_idx = (state.p_state_idx + 1).min(62);
        } else {
            self.low += self.range;
            self.range = lps;
            if state.p_state_idx == 0 {
                state.val_mps = !state.val_mps;
            }
            state.p_state_idx = TRANS_IDX_LPS[p];
        }
        self.renormalize();
    }

    /// `EncodeTerminate` (9.3.4.5) of `bin`: with 1, the flush (`EncodeFlush`), whose last bit is
    /// 1, then zeros to the byte, as `pcm_alignment_zero_bit`s or `rbsp_slice_segment_trailing_bits`
    /// have it.
    fn terminate(&mut self, bin: bool) {
        self.range -= 2;
        if !bin {
            self.renormalize();
            return;
        }
        self.low += self.range;
        self.range = 2;
        self.renormalize();
        self.put_bit((self.low >> 9) & 1 == 1);
        let last = ((self.low >> 7) & 3) | 1;
        self.out.u(2, last);
        while !self.out.byte_aligned() {
            self.out.bit(false);
        }
    }

    /// A PCM sample of `bits` bits, written as it is.
    fn raw(&mut self, bits: u32, value: u16) {
        self.out.u(bits, u32::from(value));
    }

    /// `RenormE` (9.3.4.3).
    fn renormalize(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(false);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(true);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// `PutBit` (9.3.4.4).
    fn put_bit(&mut self, bit: bool) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.out.bit(bit);
        }
        while self.outstanding > 0 {
            self.out.bit(!bit);
            self.outstanding -= 1;
        }
    }

    /// The slice segment's RBSP: its last bin, `end_of_slice_segment_flag` 1, has written its
    /// trailing bits.
    fn finish(self) -> BitWriter {
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_context_states_are_the_standards() {
        // 9-6 for two initValues at QP 26, worked by hand: 154 is the even state (m 0, n 64),
        // 197 (m 15, n 24) preCtxState 48.
        let even = init(154);
        assert_eq!((even.p_state_idx, even.val_mps), (0, true));
        let skip = init(197);
        assert_eq!((skip.p_state_idx, skip.val_mps), (15, false));
    }

    #[test]
    fn the_samples_differ_from_their_neighbours() {
        for depth in [8, 10] {
            for plane in 0..3 {
                for y in 0..32 {
                    for x in 0..32 {
                        let s = sample(plane, x, y, depth);
                        assert_ne!(s, sample(plane, x + 1, y, depth));
                        assert!(s < (1 << depth));
                    }
                }
            }
        }
    }
}
