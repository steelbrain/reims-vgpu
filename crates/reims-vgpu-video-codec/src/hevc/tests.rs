//! The decoding process end to end, on streams written here from the
//! standard's syntax and an accelerator that records what it is asked to
//! decode. The slice data is a placeholder: nothing here decodes samples.
//! What x265 does not write, or writes rarely (inter RPS prediction in a
//! slice header, long-term pictures with MSB cycles, list modification,
//! pic_output_flag, dependent slice segments, refusals), is covered here;
//! the VA-API crate's tools/hevc-check.sh covers the rest on hardware.

use super::bits::BitReader;
use super::nal::nal_type;
use super::slice::SliceHeader;
use super::*;

/// Writes syntax elements, then raw bytes or an escaped NAL unit.
#[derive(Default)]
pub(crate) struct BitWriter {
    bits: Vec<bool>,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn u(&mut self, n: u32, value: u32) -> &mut Self {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
        self
    }

    pub(crate) fn bit(&mut self, value: bool) -> &mut Self {
        self.bits.push(value);
        self
    }

    pub(crate) fn ue(&mut self, value: u32) -> &mut Self {
        let code = u64::from(value) + 1;
        let len = 64 - code.leading_zeros();
        for _ in 1..len {
            self.bits.push(false);
        }
        for i in (0..len).rev() {
            self.bits.push((code >> i) & 1 == 1);
        }
        self
    }

    pub(crate) fn se(&mut self, value: i32) -> &mut Self {
        let code = if value > 0 {
            2 * value.unsigned_abs() - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(code)
    }

    /// `byte_alignment()`: a one, then zeros to the byte's end.
    fn align(&mut self) -> &mut Self {
        self.bits.push(true);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        self
    }

    fn bytes(&self) -> Vec<u8> {
        self.bits
            .chunks(8)
            .map(|b| {
                b.iter()
                    .chain(core::iter::repeat(&false))
                    .take(8)
                    .fold(0u8, |acc, &bit| (acc << 1) | u8::from(bit))
            })
            .collect()
    }

    /// The bits with `rbsp_trailing_bits()`, unescaped.
    pub(crate) fn finish(&mut self) -> Vec<u8> {
        self.align();
        self.bytes()
    }

    /// The NAL unit: the two header bytes, the bits and
    /// `rbsp_trailing_bits()`, with emulation prevention bytes inserted.
    fn nal(&mut self, unit_type: u8, temporal_id: u8) -> Vec<u8> {
        let rbsp = self.finish();
        let mut out = vec![unit_type << 1, temporal_id + 1];
        let mut zeros = 0;
        for byte in rbsp {
            if zeros >= 2 && byte <= 3 {
                out.push(3);
                zeros = 0;
            }
            zeros = if byte == 0 { zeros + 1 } else { 0 };
            out.push(byte);
        }
        out
    }
}

/// A short-term set as a test writes it.
#[derive(Clone)]
enum Set {
    /// `delta_poc_s0_minus1` and `delta_poc_s1_minus1` with their used
    /// flags.
    Explicit(Vec<(u32, bool)>, Vec<(u32, bool)>),
    /// Predicted: `delta_idx_minus1` (slice header only), deltaRps, and
    /// each `(used_by_curr_pic_flag, use_delta_flag)`.
    Predicted(u32, i32, Vec<(bool, bool)>),
}

fn write_set(w: &mut BitWriter, index: usize, count: usize, set: &Set) {
    match set {
        Set::Explicit(negative, positive) => {
            if index != 0 {
                w.bit(false);
            }
            w.ue(negative.len() as u32).ue(positive.len() as u32);
            for &(minus1, used) in negative.iter().chain(positive) {
                w.ue(minus1).bit(used);
            }
        }
        Set::Predicted(delta_idx_minus1, delta_rps, flags) => {
            w.bit(true);
            if index == count {
                w.ue(*delta_idx_minus1);
            }
            w.bit(*delta_rps < 0).ue(delta_rps.unsigned_abs() - 1);
            for &(used, use_delta) in flags {
                w.bit(used);
                if !used {
                    w.bit(use_delta);
                }
            }
        }
    }
}

/// What a test stream's SPS says.
#[derive(Clone)]
struct SpsSpec {
    profile_idc: u8,
    chroma_format_idc: u32,
    bit_depth_minus8: u32,
    /// `log2_max_pic_order_cnt_lsb_minus4`.
    lsb_minus4: u32,
    dpb: u32,
    reorder: u32,
    sets: Vec<Set>,
    /// `lt_ref_pic_poc_lsb_sps` and `used_by_curr_pic_lt_sps_flag`; the
    /// SPS has long-term pictures when this is `Some`.
    long_term: Option<Vec<(u32, bool)>>,
    scaling_list_enabled: bool,
    range_extension: bool,
    /// `sps_max_latency_increase_plus1`.
    latency_plus1: u32,
    /// The luma sizes in luma samples.
    size: (u32, u32),
    /// `log2_min_luma_coding_block_size_minus3`,
    /// `log2_diff_max_min_luma_coding_block_size`,
    /// `log2_min_luma_transform_block_size_minus2`,
    /// `log2_diff_max_min_luma_transform_block_size`, and the inter and
    /// intra `max_transform_hierarchy_depth`s.
    blocks: [u32; 6],
    /// `pcm_sample_bit_depth_luma_minus1` and `_chroma_minus1`,
    /// `log2_min_pcm_luma_coding_block_size_minus3` and
    /// `log2_diff_max_min_pcm_luma_coding_block_size`, when PCM is on.
    pcm: Option<[u32; 4]>,
    /// A VUI of `video_full_range_flag` alone, when `Some`.
    vui_full_range: Option<bool>,
    /// A VUI of timing information alone, 60000/1001, with
    /// `hrd_parameters()` of a NAL HRD, sub-picture parameters and two
    /// CPBs: everything the parser reads past to reach the SPS's
    /// extension flags. Takes the place of `vui_full_range`'s.
    vui_hrd: bool,
}

impl Default for SpsSpec {
    fn default() -> Self {
        Self {
            profile_idc: 1,
            chroma_format_idc: 1,
            bit_depth_minus8: 0,
            lsb_minus4: 4,
            dpb: 5,
            reorder: 2,
            sets: Vec::new(),
            long_term: None,
            scaling_list_enabled: false,
            range_extension: false,
            latency_plus1: 0,
            size: (64, 48),
            blocks: [0, 1, 0, 2, 1, 1],
            pcm: None,
            vui_full_range: None,
            vui_hrd: false,
        }
    }
}

/// 64x48 luma samples by default, cropped by two rows: CTBs of 16, four by
/// three.
fn sps(spec: &SpsSpec) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(4, 0).u(3, 0).bit(true);
    // profile_tier_level(1, 0)
    w.u(2, 0).bit(false).u(5, u32::from(spec.profile_idc));
    w.u(32, 1 << (31 - u32::from(spec.profile_idc)));
    w.bit(true)
        .bit(false)
        .bit(false)
        .bit(true)
        .u(32, 0)
        .u(12, 0);
    w.u(8, 93);
    w.ue(0).ue(spec.chroma_format_idc);
    if spec.chroma_format_idc == 3 {
        w.bit(false);
    }
    w.ue(spec.size.0).ue(spec.size.1);
    w.bit(true).ue(0).ue(0).ue(0).ue(1);
    w.ue(spec.bit_depth_minus8).ue(spec.bit_depth_minus8);
    w.ue(spec.lsb_minus4);
    w.bit(true)
        .ue(spec.dpb - 1)
        .ue(spec.reorder)
        .ue(spec.latency_plus1);
    for value in spec.blocks {
        w.ue(value);
    }
    w.bit(spec.scaling_list_enabled);
    if spec.scaling_list_enabled {
        w.bit(false);
    }
    w.bit(true).bit(true).bit(spec.pcm.is_some());
    if let Some([luma, chroma, min_minus3, diff]) = spec.pcm {
        w.u(4, luma).u(4, chroma).ue(min_minus3).ue(diff).bit(false);
    }
    w.ue(spec.sets.len() as u32);
    for (i, set) in spec.sets.iter().enumerate() {
        write_set(&mut w, i, spec.sets.len(), set);
    }
    w.bit(spec.long_term.is_some());
    if let Some(pictures) = &spec.long_term {
        w.ue(pictures.len() as u32);
        for &(lsb, used) in pictures {
            w.u(spec.lsb_minus4 + 4, lsb).bit(used);
        }
    }
    w.bit(false).bit(true); // no temporal MVP; strong intra smoothing
    w.bit(spec.vui_full_range.is_some() || spec.vui_hrd);
    if spec.vui_hrd {
        // Aspect ratio, overscan, video signal type, chroma location,
        // neutral chroma, field_seq_flag, frame field info and the
        // default display window: all absent.
        for _ in 0..8 {
            w.bit(false);
        }
        // vui_timing_info_present_flag, the timing, no POC proportion,
        // vui_hrd_parameters_present_flag.
        w.bit(true).u(32, 1001).u(32, 60000).bit(false).bit(true);
        // hrd_parameters(1, 0): NAL HRD alone, with sub-picture
        // parameters (8 + 5 + 1 + 5 bits), the scales (4 + 4 + 4), the
        // lengths (5 + 5 + 5).
        w.bit(true).bit(false).bit(true);
        w.u(8, 23).u(5, 7).bit(true).u(5, 9);
        w.u(4, 1).u(4, 2).u(4, 3);
        w.u(5, 23).u(5, 23).u(5, 23);
        // Sub-layer 0: rates not fixed, not low delay, two CPBs, each its
        // four values and cbr_flag.
        w.bit(false).bit(false).bit(false).ue(1);
        for cpb in 0..2 {
            w.ue(1000 + cpb).ue(2000).ue(300).ue(400).bit(cpb == 1);
        }
        w.bit(false); // bitstream_restriction_flag
    } else if let Some(full_range) = spec.vui_full_range {
        // video_signal_type_present_flag alone: video_format 5, the range,
        // no colour description; every other flag clear.
        w.bit(false)
            .bit(false)
            .bit(true)
            .u(3, 5)
            .bit(full_range)
            .bit(false);
        w.bit(false).bit(false).bit(false).bit(false).bit(false);
        w.bit(false).bit(false);
    }
    w.bit(spec.range_extension);
    if spec.range_extension {
        w.bit(true).bit(false).bit(false).bit(false).u(4, 0);
        // An empty sps_range_extension(): nine flags, all clear.
        w.u(9, 0);
    }
    w.nal(nal_type::SPS, 0)
}

#[derive(Clone, Default)]
struct PpsSpec {
    dependent_slices: bool,
    output_flag_present: bool,
    weighted: bool,
    lists_modification: bool,
    tiles: Option<(u32, u32)>,
    wavefronts: bool,
    l0_default_minus1: u32,
    l1_default_minus1: u32,
    init_qp_minus26: i32,
    /// `diff_cu_qp_delta_depth`, when `cu_qp_delta_enabled_flag`.
    cu_qp_delta_depth: Option<u32>,
    /// `pps_cb_qp_offset` and `pps_cr_qp_offset`.
    chroma_qp_offsets: (i32, i32),
    /// `pps_slice_chroma_qp_offsets_present_flag`.
    slice_chroma_qp_offsets: bool,
    log2_parallel_merge_level_minus2: u32,
}

fn pps(spec: &PpsSpec) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.ue(0)
        .ue(0)
        .bit(spec.dependent_slices)
        .bit(spec.output_flag_present);
    w.u(3, 0).bit(false).bit(true);
    w.ue(spec.l0_default_minus1).ue(spec.l1_default_minus1);
    w.se(spec.init_qp_minus26).bit(false).bit(false);
    w.bit(spec.cu_qp_delta_depth.is_some());
    if let Some(depth) = spec.cu_qp_delta_depth {
        w.ue(depth);
    }
    w.se(spec.chroma_qp_offsets.0)
        .se(spec.chroma_qp_offsets.1)
        .bit(spec.slice_chroma_qp_offsets);
    w.bit(spec.weighted).bit(spec.weighted).bit(false);
    w.bit(spec.tiles.is_some()).bit(spec.wavefronts);
    if let Some((columns, rows)) = spec.tiles {
        w.ue(columns - 1).ue(rows - 1).bit(true).bit(true);
    }
    w.bit(true)
        .bit(false)
        .bit(false)
        .bit(spec.lists_modification);
    w.ue(spec.log2_parallel_merge_level_minus2)
        .bit(false)
        .bit(false);
    w.nal(nal_type::PPS, 0)
}

/// A long-term picture from the SPS's list: `lt_idx_sps` and the MSB
/// cycle.
type FromSps = (u32, Option<u32>);
/// A slice header's own long-term picture: `poc_lsb_lt`,
/// `used_by_curr_pic_lt_flag` and the MSB cycle.
type OwnLongTerm = (u32, bool, Option<u32>);

/// What a test slice segment says.
#[derive(Clone)]
struct SliceSpec {
    unit_type: u8,
    temporal_id: u8,
    first: bool,
    dependent: bool,
    address: u32,
    slice_type: u32,
    pic_output: bool,
    poc_lsb: u32,
    /// The SPS's set by index, or one of the header's own.
    set: Result<u32, Set>,
    /// `num_long_term_sps` entries by `lt_idx_sps`, then the header's own
    /// `(poc_lsb_lt, used)`, each with `delta_poc_msb_cycle_lt`.
    long_term: (Vec<FromSps>, Vec<OwnLongTerm>),
    /// `num_ref_idx_lX_active_minus1`, overriding the PPS's defaults.
    active: Option<(u32, u32)>,
    list_entries: [Option<Vec<u32>>; 2],
    entry_points: u32,
    /// No placeholder for `slice_segment_data()`.
    no_data: bool,
    slice_qp_delta: i32,
    /// `slice_cb_qp_offset` and `slice_cr_qp_offset`, when the PPS has
    /// them.
    chroma_qp_offsets: (i32, i32),
    /// List 0's `luma_weight_l0_flag` and `chroma_weight_l0_flag` for each
    /// entry, with weights and offsets of 0, in place of the default
    /// table, when the PPS weights.
    weights: Option<Vec<(bool, bool)>>,
}

impl Default for SliceSpec {
    fn default() -> Self {
        Self {
            unit_type: nal_type::TRAIL_R,
            temporal_id: 0,
            first: true,
            dependent: false,
            address: 0,
            slice_type: 2,
            pic_output: true,
            poc_lsb: 0,
            set: Ok(0),
            long_term: (Vec::new(), Vec::new()),
            active: None,
            list_entries: [None, None],
            entry_points: 0,
            no_data: false,
            slice_qp_delta: 0,
            chroma_qp_offsets: (0, 0),
            weights: None,
        }
    }
}

fn slice(sps: &SpsSpec, pps: &PpsSpec, spec: &SliceSpec) -> Vec<u8> {
    let mut w = BitWriter::new();
    let irap = nal::is_irap(spec.unit_type);
    w.bit(spec.first);
    if irap {
        w.bit(false);
    }
    w.ue(0);
    if !spec.first {
        if pps.dependent_slices {
            w.bit(spec.dependent);
        }
        w.u(address_bits(sps), spec.address);
    }
    if !spec.dependent {
        w.ue(spec.slice_type);
        if pps.output_flag_present {
            w.bit(spec.pic_output);
        }
        let mut total_curr = 0;
        if !nal::is_idr(spec.unit_type) {
            w.u(sps.lsb_minus4 + 4, spec.poc_lsb);
            match &spec.set {
                Ok(index) => {
                    w.bit(true);
                    if sps.sets.len() > 1 {
                        let bits = 32 - (sps.sets.len() as u32 - 1).leading_zeros();
                        w.u(bits, *index);
                    }
                    total_curr = used(&sps.sets, *index as usize);
                }
                Err(set) => {
                    w.bit(false);
                    write_set(&mut w, sps.sets.len(), sps.sets.len(), set);
                    total_curr = used_by_own(&sps.sets, set);
                }
            }
            if let Some(candidates) = &sps.long_term {
                let (from_sps, own) = &spec.long_term;
                if !candidates.is_empty() {
                    w.ue(from_sps.len() as u32);
                }
                w.ue(own.len() as u32);
                for &(index, msb) in from_sps {
                    if candidates.len() > 1 {
                        let bits = 32 - (candidates.len() as u32 - 1).leading_zeros();
                        w.u(bits, index);
                    }
                    total_curr += u32::from(candidates[index as usize].1);
                    w.bit(msb.is_some());
                    if let Some(cycle) = msb {
                        w.ue(cycle);
                    }
                }
                for &(lsb, used, msb) in own {
                    w.u(sps.lsb_minus4 + 4, lsb).bit(used);
                    total_curr += u32::from(used);
                    w.bit(msb.is_some());
                    if let Some(cycle) = msb {
                        w.ue(cycle);
                    }
                }
            }
        }
        w.bit(true).bit(true); // SAO, luma and chroma
        if spec.slice_type != 2 {
            let b = spec.slice_type == 0;
            w.bit(spec.active.is_some());
            if let Some((l0, l1)) = spec.active {
                w.ue(l0);
                if b {
                    w.ue(l1);
                }
            }
            let counts = spec
                .active
                .unwrap_or((pps.l0_default_minus1, pps.l1_default_minus1));
            if pps.lists_modification && total_curr > 1 {
                let bits = 32 - (total_curr - 1).leading_zeros();
                for entries in &spec.list_entries[..if b { 2 } else { 1 }] {
                    w.bit(entries.is_some());
                    for &entry in entries.iter().flatten() {
                        w.u(bits, entry);
                    }
                }
            }
            if b {
                w.bit(false);
            }
            w.bit(false); // cabac_init_flag
            if let (true, Some(weights)) = (pps.weighted, &spec.weights) {
                w.ue(6).se(0);
                for list in 0..if b { 2 } else { 1 } {
                    let n = if list == 0 { counts.0 } else { counts.1 } + 1;
                    let flags = |i: u32| {
                        let entry = weights.get(i as usize).filter(|_| list == 0);
                        entry.copied().unwrap_or((false, false))
                    };
                    for i in 0..n {
                        w.bit(flags(i).0);
                    }
                    for i in 0..n {
                        w.bit(flags(i).1);
                    }
                    for i in 0..n {
                        let (luma, chroma) = flags(i);
                        if luma {
                            w.se(0).se(0);
                        }
                        if chroma {
                            w.se(0).se(0).se(0).se(0);
                        }
                    }
                }
            } else if pps.weighted {
                // luma_log2_weight_denom 6, chroma's the same; list 0's
                // first entry has luma weights, the rest none.
                w.ue(6).se(0);
                for list in 0..if b { 2 } else { 1 } {
                    let n = if list == 0 { counts.0 } else { counts.1 } + 1;
                    for i in 0..n {
                        w.bit(list == 0 && i == 0);
                    }
                    for _ in 0..n {
                        w.bit(false);
                    }
                    if list == 0 {
                        w.se(-3).se(5);
                    }
                }
            }
            w.ue(0);
        }
        w.se(spec.slice_qp_delta);
        if pps.slice_chroma_qp_offsets {
            w.se(spec.chroma_qp_offsets.0).se(spec.chroma_qp_offsets.1);
        }
        w.bit(true); // slice_loop_filter_across_slices_enabled_flag
    }
    if pps.tiles.is_some() || pps.wavefronts {
        w.ue(spec.entry_points);
        if spec.entry_points > 0 {
            w.ue(7);
            for _ in 0..spec.entry_points {
                w.u(8, 9);
            }
        }
    }
    w.align();
    if !spec.no_data {
        // A placeholder for slice_data(), with a run of zeros that needs an
        // emulation prevention byte.
        w.u(8, 0xa5).u(8, 0).u(8, 0).u(8, 1);
    }
    w.nal(spec.unit_type, spec.temporal_id)
}

/// The CTBs of `sps`'s pictures: `PicSizeInCtbsY` (7-19).
fn ctbs(sps: &SpsSpec) -> (u32, u32) {
    let ctb = 1 << (3 + sps.blocks[0] + sps.blocks[1]);
    (sps.size.0.div_ceil(ctb), sps.size.1.div_ceil(ctb))
}

/// `slice_segment_address`'s bits: Ceil(Log2(PicSizeInCtbsY)), 4 for the
/// default 64x48.
fn address_bits(sps: &SpsSpec) -> u32 {
    let (columns, rows) = ctbs(sps);
    let total = (columns * rows).max(1);
    32 - (total - 1).leading_zeros()
}

/// The SPS's sets as the decoder parses them.
fn parsed_sets(sets: &[Set]) -> Vec<rps::ShortTermRps> {
    let mut parsed: Vec<rps::ShortTermRps> = Vec::new();
    for (i, set) in sets.iter().enumerate() {
        let mut w = BitWriter::new();
        write_set(&mut w, i, sets.len(), set);
        let bytes = w.finish();
        let rps = rps::ShortTermRps::parse(&mut BitReader::new(&bytes), &parsed, false, 15)
            .unwrap_or_default();
        parsed.push(rps);
    }
    parsed
}

/// `NumPicTotalCurr`'s short-term part for the SPS's set `index`.
fn used(sets: &[Set], index: usize) -> u32 {
    parsed_sets(sets)
        .get(index)
        .map_or(0, |set| set.used_by_current() as u32)
}

/// The same for a slice header's own set.
fn used_by_own(sets: &[Set], own: &Set) -> u32 {
    let parsed = parsed_sets(sets);
    let mut w = BitWriter::new();
    write_set(&mut w, sets.len(), sets.len(), own);
    let bytes = w.finish();
    rps::ShortTermRps::parse(&mut BitReader::new(&bytes), &parsed, true, 15)
        .map_or(0, |set| set.used_by_current() as u32)
}

/// An Annex B sample of the given units.
fn sample(units: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(unit);
    }
    out
}

/// What the accelerator was asked to decode, for one picture.
#[derive(Debug)]
struct Decoded {
    /// `PicOrderCntVal`, which `run` reports too.
    #[expect(dead_code, reason = "kept for a failing test's Debug output")]
    poc: i32,
    /// Each reference's order count, set, and long-term flag.
    references: Vec<(i32, RpsMembership, bool)>,
    /// Each slice's two lists, as order counts.
    lists: Vec<[Vec<i32>; 2]>,
    intra_only: bool,
    has_b_slices: bool,
    scaling: bool,
    headers: Vec<SliceHeader>,
    last: Vec<bool>,
}

#[derive(Default)]
struct Recorder {
    configured: Vec<(u32, u32, u8, usize)>,
    decoded: Vec<Decoded>,
    next: u32,
}

impl Accelerator for Recorder {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, sps: &sps::Sps, dpb_size: usize) -> Result<(), Error> {
        self.configured.push((
            sps.pic_width_in_luma_samples,
            sps.pic_height_in_luma_samples,
            sps.bit_depth_luma_minus8 + 8,
            dpb_size,
        ));
        Ok(())
    }

    fn new_picture(&mut self) -> Result<u32, Error> {
        self.next += 1;
        Ok(self.next)
    }

    fn decode_picture(
        &mut self,
        _picture: &u32,
        params: &PictureParams<'_, u32>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), Error> {
        let pocs = |list: &[usize]| list.iter().map(|&i| params.references[i].poc).collect();
        self.decoded.push(Decoded {
            poc: params.poc,
            references: params
                .references
                .iter()
                .map(|r| (r.poc, r.membership, r.long_term))
                .collect(),
            lists: slices
                .iter()
                .map(|s| [pocs(&s.ref_lists[0]), pocs(&s.ref_lists[1])])
                .collect(),
            intra_only: params.intra_only,
            has_b_slices: params.has_b_slices,
            scaling: params.scaling_lists.is_some(),
            headers: slices.iter().map(|s| s.header.clone()).collect(),
            last: slices.iter().map(|s| s.last).collect(),
        });
        Ok(())
    }
}

fn decoder() -> Decoder<Recorder> {
    Decoder::new(Recorder::default(), NalFormat::AnnexB)
}

/// Decodes each sample in turn, returning every output's order count. A
/// RASL picture skipped gives none.
fn run(decoder: &mut Decoder<Recorder>, samples: &[Vec<u8>]) -> Result<Vec<i32>, Error> {
    let mut out = Vec::new();
    for (id, sample) in samples.iter().enumerate() {
        match decoder.decode(sample, id as u64) {
            Ok(outputs) => out.extend(outputs.iter().map(|o| o.poc)),
            Err(Error::SkippedLeadingPicture) => {}
            Err(e) => return Err(e),
        }
    }
    out.extend(decoder.flush().iter().map(|o| o.poc));
    Ok(out)
}

/// The SPS sets of a small hierarchical-B GOP of four: P4 from I0; B2
/// from 0 and 4; B1 from 0, 2 and 4; B3 from 2 and 4 (0 kept).
fn gop_sets() -> Vec<Set> {
    vec![
        // 0: P4 (-4 used)
        Set::Explicit(vec![(3, true)], vec![]),
        // 1: B2 (-2 used, +2 used)
        Set::Explicit(vec![(1, true)], vec![(1, true)]),
        // 2: B1 (-1 used, +1 used, +3 used)
        Set::Explicit(vec![(0, true)], vec![(0, true), (1, true)]),
        // 3: B3 (-1 used, -3 kept, +1 used)
        Set::Explicit(vec![(0, true), (1, false)], vec![(0, true)]),
    ]
}

fn gop_stream(order: OutputOrder) -> Result<(Decoder<Recorder>, Vec<i32>), Error> {
    let spec = SpsSpec {
        sets: gop_sets(),
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let pictures = [
        (nal_type::IDR_W_RADL, 2, 0, 0),
        (nal_type::TRAIL_R, 1, 4, 0),
        (nal_type::TRAIL_R, 0, 2, 1),
        (nal_type::TRAIL_N, 0, 1, 2),
        (nal_type::TRAIL_N, 0, 3, 3),
    ];
    let mut samples = Vec::new();
    for (i, &(unit_type, slice_type, poc_lsb, set)) in pictures.iter().enumerate() {
        let mut units = Vec::new();
        if i == 0 {
            units.push(sps(&spec));
            units.push(pps(&pps_spec));
        }
        units.push(slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type,
                slice_type,
                poc_lsb,
                set: Ok(set),
                ..SliceSpec::default()
            },
        ));
        samples.push(sample(&units));
    }
    let mut decoder = decoder();
    decoder.set_output_order(order);
    let out = run(&mut decoder, &samples)?;
    Ok((decoder, out))
}

#[test]
fn a_hierarchical_gop_builds_its_sets_and_lists() -> Result<(), Error> {
    let (decoder, out) = gop_stream(OutputOrder::Decode)?;
    assert_eq!(out, [0, 4, 2, 1, 3]);
    let decoded = &decoder.accelerator().decoded;
    assert_eq!(decoder.accelerator().configured, [(64, 48, 8, 5)]);
    assert!(decoded[0].intra_only && decoded[0].references.is_empty());
    use RpsMembership::*;
    assert_eq!(decoded[1].references, [(0, StCurrBefore, false)]);
    assert_eq!(decoded[1].lists[0], [[0].to_vec(), Vec::new()]);
    // B2: list 0 is before then after, list 1 after then before.
    assert_eq!(
        decoded[2].references,
        [(0, StCurrBefore, false), (4, StCurrAfter, false)]
    );
    assert_eq!(decoded[2].lists[0], [vec![0], vec![4]]);
    assert!(decoded[2].has_b_slices && !decoded[2].intra_only);
    // B1: before 0; after 2, 4. One active reference each by default.
    assert_eq!(decoded[3].lists[0], [vec![0], vec![2]]);
    // B3: before 2, kept 0, after 4.
    let mut references = decoded[4].references.clone();
    references.sort_by_key(|r| r.0);
    assert_eq!(
        references,
        [
            (0, Foll, false),
            (2, StCurrBefore, false),
            (4, StCurrAfter, false)
        ]
    );
    assert_eq!(decoded[4].lists[0], [vec![2], vec![4]]);
    assert!(decoded.iter().all(|d| d.last == [true] && !d.scaling));
    Ok(())
}

#[test]
fn display_order_bumps_by_order_count() -> Result<(), Error> {
    let (_, out) = gop_stream(OutputOrder::Display)?;
    assert_eq!(out, [0, 1, 2, 3, 4]);
    Ok(())
}

fn picture(spec: &SpsSpec, unit_type: u8, slice_type: u32, poc_lsb: u32, set: u32) -> Vec<u8> {
    slice(
        spec,
        &PpsSpec::default(),
        &SliceSpec {
            unit_type,
            slice_type,
            poc_lsb,
            set: Ok(set),
            ..SliceSpec::default()
        },
    )
}

#[test]
fn a_leading_cra_skips_its_rasl_pictures_and_decodes_radl_ones() -> Result<(), Error> {
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![], vec![(1, true)]),          // 0: +2
            Set::Explicit(vec![(0, true)], vec![(0, true)]), // 1: -1, +1
            Set::Explicit(vec![(1, true)], vec![]),          // 2: -2
            Set::Explicit(vec![(3, false)], vec![]),         // 3: -4, kept
            Set::Explicit(vec![(1, true)], vec![(1, true)]), // 4: -2, +2
        ],
        ..SpsSpec::default()
    };
    let parameter_sets = [sps(&spec), pps(&PpsSpec::default())];
    // A CRA at POC 10, as if cut from a longer stream; a RASL picture at 9
    // that predicts from 8, which this decoder never had, and 10; a RADL
    // picture at 8 from 10; a trailing picture at 12.
    // The CRA's set keeps a picture and predicts from none, as 7.4.7.1
    // has a CRA's.
    let mut cra = parameter_sets.to_vec();
    cra.push(picture(&spec, nal_type::CRA_NUT, 2, 10, 3));
    let samples = [
        sample(&cra),
        sample(&[picture(&spec, nal_type::RASL_N, 0, 9, 1)]),
        sample(&[picture(&spec, nal_type::RADL_N, 1, 8, 0)]),
        sample(&[picture(&spec, nal_type::TRAIL_R, 1, 12, 2)]),
    ];
    let mut d = decoder();
    assert_eq!(run(&mut d, &samples)?, [10, 8, 12]);
    let decoded = &d.accelerator().decoded;
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[1].lists[0][0], [10]);
    // The RASL picture is refused as Apple's decoder drops it, and the
    // pictures after it decode.
    let mut d = decoder();
    d.decode(&samples[0], 0)?;
    assert_eq!(
        d.decode(&samples[1], 1).err(),
        Some(Error::SkippedLeadingPicture)
    );
    assert_eq!(d.decode(&samples[2], 2)?.len(), 1);

    // A CRA in mid-stream has NoRaslOutputFlag 0: its RASL picture at 14,
    // from 12 before the CRA and 16, the CRA, is decoded.
    let samples = [
        sample(&cra),
        sample(&[picture(&spec, nal_type::TRAIL_R, 1, 12, 2)]),
        sample(&[picture(&spec, nal_type::CRA_NUT, 2, 16, 3)]),
        sample(&[picture(&spec, nal_type::RASL_N, 0, 14, 4)]),
        sample(&[picture(&spec, nal_type::TRAIL_R, 1, 18, 2)]),
    ];
    let mut d = decoder();
    assert_eq!(run(&mut d, &samples)?, [10, 12, 16, 14, 18]);
    let decoded = &d.accelerator().decoded;
    assert_eq!(decoded[3].lists[0], [vec![12], vec![16]]);
    // After a reset the next CRA starts afresh, and its RASL picture is
    // skipped again.
    d.reset();
    assert_eq!(run(&mut d, &samples[2..4])?, [16]);
    Ok(())
}

/// A sample whose one NAL unit has `forbidden_zero_bit` set: one that
/// does not decode.
fn forbidden() -> Vec<u8> {
    sample(&[vec![0x80 | (nal_type::TRAIL_R << 1), 1, 0x55, 0x55]])
}

#[test]
fn after_a_sample_that_fails_pictures_are_refused_until_an_irap_picture_decodes()
-> Result<(), Error> {
    // Apple's HEVC decoder answers each picture after one that failed with
    // kVTVideoDecoderBadDataErr until an IDR, CRA or BLA picture decodes;
    // a picture of I slices alone that is not one does not end it.
    let spec = SpsSpec {
        sets: gop_sets(),
        ..SpsSpec::default()
    };
    let idr = || {
        let mut units = vec![sps(&spec), pps(&PpsSpec::default())];
        units.push(picture(&spec, nal_type::IDR_W_RADL, 2, 0, 0));
        sample(&units)
    };
    let p = |lsb| sample(&[picture(&spec, nal_type::TRAIL_R, 1, lsb, 0)]);
    let mut d = decoder();
    d.decode(&idr(), 0)?;
    d.decode(&p(4), 1)?;
    assert_eq!(
        d.decode(&forbidden(), 2).err(),
        Some(Error::OutOfRange("forbidden_zero_bit"))
    );
    assert_eq!(d.decode(&p(8), 3).err(), Some(Error::Unrecovered));
    let intra = sample(&[picture(&spec, nal_type::TRAIL_R, 2, 12, 0)]);
    assert_eq!(d.decode(&intra, 4).err(), Some(Error::Unrecovered));
    assert_eq!(d.decode(&p(16), 5).err(), Some(Error::Unrecovered));
    // None reached the accelerator: nothing after the IRAP picture
    // predicts from them.
    assert_eq!(d.accelerator().decoded.len(), 2);
    assert_eq!(d.decode(&idr(), 6)?.len(), 1);
    assert_eq!(d.decode(&p(4), 7)?.len(), 1);
    assert_eq!(d.accelerator().decoded.len(), 4);

    // An IRAP picture that fails itself ends nothing.
    let mut cut = picture(&spec, nal_type::IDR_W_RADL, 2, 0, 0);
    cut.truncate(2);
    assert_eq!(d.decode(&sample(&[cut]), 8).err(), Some(Error::Truncated));
    assert_eq!(d.decode(&p(4), 9).err(), Some(Error::Unrecovered));
    assert_eq!(d.decode(&idr(), 10)?.len(), 1);

    // A reset forgets the failure; the next picture still needs an IRAP
    // picture before it.
    assert!(d.decode(&forbidden(), 11).is_err());
    d.reset();
    assert_eq!(d.decode(&p(4), 12).err(), Some(Error::MissingReference));
    assert_eq!(d.decode(&p(8), 13).err(), Some(Error::Unrecovered));
    Ok(())
}

#[test]
fn a_picture_the_hardware_failed_fails_each_after_it_until_an_irap_picture() -> Result<(), Error> {
    // A truncated slice decodes here, and only the hardware's status, read
    // once the picture is done, says it failed: Apple's HEVC decoder answers
    // it and each picture after it with kVTVideoDecoderBadDataErr until an
    // IRAP picture decodes.
    let spec = SpsSpec {
        sets: gop_sets(),
        ..SpsSpec::default()
    };
    let idr = || {
        let mut units = vec![sps(&spec), pps(&PpsSpec::default())];
        units.push(picture(&spec, nal_type::IDR_W_RADL, 2, 0, 0));
        sample(&units)
    };
    let p = |lsb| sample(&[picture(&spec, nal_type::TRAIL_R, 1, lsb, 0)]);
    let mut d = decoder();
    d.decode(&idr(), 0)?;
    assert_eq!(d.decode(&p(4), 1)?.len(), 1);
    d.mark_failed();
    assert_eq!(d.decode(&p(8), 2).err(), Some(Error::Unrecovered));
    let intra = sample(&[picture(&spec, nal_type::TRAIL_R, 2, 12, 0)]);
    assert_eq!(d.decode(&intra, 3).err(), Some(Error::Unrecovered));
    // Neither reached the accelerator.
    assert_eq!(d.accelerator().decoded.len(), 2);
    assert_eq!(d.decode(&idr(), 4)?.len(), 1);
    assert_eq!(d.decode(&p(4), 5)?.len(), 1);

    // An IRAP picture the hardware failed ends nothing.
    assert_eq!(d.decode(&idr(), 6)?.len(), 1);
    d.mark_failed();
    assert_eq!(d.decode(&p(4), 7).err(), Some(Error::Unrecovered));
    assert_eq!(d.decode(&idr(), 8)?.len(), 1);
    assert_eq!(d.decode(&p(4), 9)?.len(), 1);
    Ok(())
}

#[test]
fn a_cra_picture_that_ends_a_failure_the_hardware_reported_skips_its_rasl_pictures()
-> Result<(), Error> {
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![], vec![(1, true)]),          // 0: +2
            Set::Explicit(vec![(0, true)], vec![(0, true)]), // 1: -1, +1
            Set::Explicit(vec![(1, true)], vec![]),          // 2: -2
            Set::Explicit(vec![(3, false)], vec![]),         // 3: -4, kept
            Set::Explicit(vec![(1, true)], vec![(1, true)]), // 4: -2, +2
        ],
        ..SpsSpec::default()
    };
    let mut cra = vec![sps(&spec), pps(&PpsSpec::default())];
    cra.push(picture(&spec, nal_type::CRA_NUT, 2, 10, 3));
    let mut d = decoder();
    d.decode(&sample(&cra), 0)?;
    d.decode(&sample(&[picture(&spec, nal_type::TRAIL_R, 1, 12, 2)]), 1)?;
    d.mark_failed();
    let out = d.decode(&sample(&[picture(&spec, nal_type::CRA_NUT, 2, 16, 3)]), 2)?;
    assert_eq!(out.iter().map(|o| o.poc).collect::<Vec<_>>(), [16]);
    let rasl = sample(&[picture(&spec, nal_type::RASL_N, 0, 14, 4)]);
    assert_eq!(d.decode(&rasl, 3).err(), Some(Error::SkippedLeadingPicture));
    let out = d.decode(&sample(&[picture(&spec, nal_type::TRAIL_R, 1, 18, 2)]), 4)?;
    assert_eq!(out.iter().map(|o| o.poc).collect::<Vec<_>>(), [18]);
    Ok(())
}

#[test]
fn a_cra_picture_that_ends_a_failure_skips_its_rasl_pictures() -> Result<(), Error> {
    // Apple's HEVC decoder answers the RASL pictures of the CRA picture
    // that ends a failure with kVTVideoDecoderReferenceMissingErr, as a
    // BLA picture's. Without the failure they decode
    // (a_leading_cra_skips_its_rasl_pictures_and_decodes_radl_ones).
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![], vec![(1, true)]),          // 0: +2
            Set::Explicit(vec![(0, true)], vec![(0, true)]), // 1: -1, +1
            Set::Explicit(vec![(1, true)], vec![]),          // 2: -2
            Set::Explicit(vec![(3, false)], vec![]),         // 3: -4, kept
            Set::Explicit(vec![(1, true)], vec![(1, true)]), // 4: -2, +2
        ],
        ..SpsSpec::default()
    };
    let mut cra = vec![sps(&spec), pps(&PpsSpec::default())];
    cra.push(picture(&spec, nal_type::CRA_NUT, 2, 10, 3));
    let mut d = decoder();
    d.decode(&sample(&cra), 0)?;
    d.decode(&sample(&[picture(&spec, nal_type::TRAIL_R, 1, 12, 2)]), 1)?;
    assert!(d.decode(&forbidden(), 2).is_err());
    let trail = sample(&[picture(&spec, nal_type::TRAIL_R, 1, 14, 2)]);
    assert_eq!(d.decode(&trail, 3).err(), Some(Error::Unrecovered));
    let out = d.decode(&sample(&[picture(&spec, nal_type::CRA_NUT, 2, 16, 3)]), 4)?;
    assert_eq!(out.iter().map(|o| o.poc).collect::<Vec<_>>(), [16]);
    // At 14, from 12 before the CRA picture and 16.
    let rasl = sample(&[picture(&spec, nal_type::RASL_N, 0, 14, 4)]);
    assert_eq!(d.decode(&rasl, 5).err(), Some(Error::SkippedLeadingPicture));
    let out = d.decode(&sample(&[picture(&spec, nal_type::TRAIL_R, 1, 18, 2)]), 6)?;
    assert_eq!(out.iter().map(|o| o.poc).collect::<Vec<_>>(), [18]);
    // The CRA picture and the trailing one reached the accelerator, the
    // RASL picture did not; the trailing one predicts from the CRA.
    let decoded = &d.accelerator().decoded;
    assert_eq!(decoded.len(), 4);
    assert_eq!(decoded[3].lists[0][0], [16]);
    // A RASL picture skipped is no failure: the next is decoded.
    let rasl_after_trail = sample(&[picture(&spec, nal_type::TRAIL_R, 1, 20, 2)]);
    assert_eq!(d.decode(&rasl_after_trail, 7)?.len(), 1);
    Ok(())
}

#[test]
fn a_slice_headers_own_set_may_be_predicted() -> Result<(), Error> {
    let spec = SpsSpec {
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let idr = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            ..SliceSpec::default()
        },
    );
    let p1 = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            slice_type: 1,
            poc_lsb: 1,
            set: Ok(0),
            ..SliceSpec::default()
        },
    );
    // The header's own set predicts from SPS set 0 (-1) with deltaRps -1:
    // the shifted -2 and the reference picture itself, -1, both used.
    let p2 = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            slice_type: 1,
            poc_lsb: 2,
            set: Err(Set::Predicted(0, -1, vec![(true, true), (true, true)])),
            active: Some((1, 0)),
            ..SliceSpec::default()
        },
    );
    let samples = [
        sample(&[sps(&spec), pps(&pps_spec), idr]),
        sample(&[p1]),
        sample(&[p2]),
    ];
    let mut d = decoder();
    assert_eq!(run(&mut d, &samples)?, [0, 1, 2]);
    let last = &d.accelerator().decoded[2];
    assert_eq!(last.lists[0][0], [1, 0]);
    let header = &last.headers[0];
    assert!(!header.short_term_ref_pic_set_sps_flag);
    // inter_ref_pic_set_prediction_flag, delta_idx_minus1 "1", the sign,
    // abs_delta_rps_minus1 "1", and two used flags: 6 bits.
    assert_eq!(header.st_rps_bits, 6);
    Ok(())
}

#[test]
fn long_term_pictures_are_found_by_lsb_or_by_full_order_count() -> Result<(), Error> {
    // MaxPicOrderCntLsb 16. Pictures 0 (IDR), then 1 to 20 each from the
    // one before, the IDR kept as long-term from 2 on.
    let spec = SpsSpec {
        lsb_minus4: 0,
        dpb: 4,
        reorder: 0,
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        long_term: Some(vec![(0, true)]),
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let mut samples = vec![sample(&[
        sps(&spec),
        pps(&pps_spec),
        slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                ..SliceSpec::default()
            },
        ),
    ])];
    for poc in 1..=20u32 {
        let long_term = match poc {
            1 => (Vec::new(), Vec::new()),
            // By the SPS's candidate, LSB 0, while POC 16 is not yet
            // decoded, so the LSB alone is unambiguous.
            2..=15 => (vec![(0, None)], Vec::new()),
            // From 16 on, POC 16 has LSB 0 too: the header names 0 with
            // its MSB cycle, one cycle back.
            _ => (Vec::new(), vec![(0, true, Some(1))]),
        };
        samples.push(sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: poc % 16,
                set: Ok(0),
                long_term,
                active: Some((1, 0)),
                ..SliceSpec::default()
            },
        )]));
    }
    let mut d = decoder();
    assert_eq!(run(&mut d, &samples)?, (0..=20).collect::<Vec<i32>>());
    let decoded = &d.accelerator().decoded;
    use RpsMembership::*;
    let mut refs = decoded[5].references.clone();
    refs.sort_by_key(|r| r.0);
    assert_eq!(refs, [(0, LtCurr, true), (4, StCurrBefore, false)]);
    assert_eq!(decoded[5].lists[0][0], [4, 0]);
    let mut refs = decoded[18].references.clone();
    refs.sort_by_key(|r| r.0);
    assert_eq!(refs, [(0, LtCurr, true), (17, StCurrBefore, false)]);
    Ok(())
}

#[test]
fn list_modification_picks_entries_of_the_temporary_list() -> Result<(), Error> {
    let spec = SpsSpec {
        sets: vec![
            Set::Explicit(vec![(0, true)], vec![]),
            Set::Explicit(vec![(0, true), (0, true)], vec![]),
        ],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec {
        lists_modification: true,
        ..PpsSpec::default()
    };
    let samples = [
        sample(&[
            sps(&spec),
            pps(&pps_spec),
            slice(
                &spec,
                &pps_spec,
                &SliceSpec {
                    unit_type: nal_type::IDR_W_RADL,
                    ..SliceSpec::default()
                },
            ),
        ]),
        sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: 1,
                set: Ok(0),
                ..SliceSpec::default()
            },
        )]),
        // StCurrBefore is 1, 0; the list asks for 0, 1, 0.
        sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: 2,
                set: Ok(1),
                active: Some((2, 0)),
                list_entries: [Some(vec![1, 0, 1]), None],
                ..SliceSpec::default()
            },
        )]),
    ];
    let mut d = decoder();
    run(&mut d, &samples)?;
    assert_eq!(d.accelerator().decoded[2].lists[0][0], [0, 1, 0]);
    Ok(())
}

#[test]
fn order_counts_carry_their_msb_across_the_lsb_wrap() -> Result<(), Error> {
    let spec = SpsSpec {
        lsb_minus4: 0,
        reorder: 0,
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let mut samples = vec![sample(&[
        sps(&spec),
        pps(&pps_spec),
        slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                ..SliceSpec::default()
            },
        ),
    ])];
    for poc in 1..40u32 {
        samples.push(sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: poc % 16,
                set: Ok(0),
                ..SliceSpec::default()
            },
        )]));
    }
    let mut d = decoder();
    assert_eq!(run(&mut d, &samples)?, (0..40).collect::<Vec<i32>>());
    Ok(())
}

#[test]
fn pictures_with_pic_output_flag_0_are_decoded_but_not_output() -> Result<(), Error> {
    let spec = SpsSpec {
        sets: vec![Set::Explicit(vec![(0, true)], vec![])],
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec {
        output_flag_present: true,
        ..PpsSpec::default()
    };
    let samples = [
        sample(&[
            sps(&spec),
            pps(&pps_spec),
            slice(
                &spec,
                &pps_spec,
                &SliceSpec {
                    unit_type: nal_type::IDR_W_RADL,
                    ..SliceSpec::default()
                },
            ),
        ]),
        sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: 1,
                pic_output: false,
                ..SliceSpec::default()
            },
        )]),
        sample(&[slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                slice_type: 1,
                poc_lsb: 2,
                ..SliceSpec::default()
            },
        )]),
    ];
    for order in [OutputOrder::Decode, OutputOrder::Display] {
        let mut d = decoder();
        d.set_output_order(order);
        assert_eq!(run(&mut d, &samples)?, [0, 2]);
        assert_eq!(d.accelerator().decoded.len(), 3);
    }
    Ok(())
}

#[test]
fn dependent_slice_segments_take_their_independent_segments_fields() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec {
        dependent_slices: true,
        wavefronts: true,
        ..PpsSpec::default()
    };
    let first = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            entry_points: 2,
            ..SliceSpec::default()
        },
    );
    let dependent = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            first: false,
            dependent: true,
            address: 8,
            entry_points: 1,
            ..SliceSpec::default()
        },
    );
    let mut d = decoder();
    d.decode(&sample(&[sps(&spec), pps(&pps_spec), first, dependent]), 0)?;
    let decoded = &d.accelerator().decoded[0];
    let [a, b] = &decoded.headers[..] else {
        return Err(Error::OutOfRange("two slice segments"));
    };
    assert!(b.dependent_slice_segment_flag && !a.dependent_slice_segment_flag);
    assert_eq!((a.slice_segment_address, b.slice_segment_address), (0, 8));
    assert_eq!(
        (a.num_entry_point_offsets, b.num_entry_point_offsets),
        (2, 1)
    );
    assert_eq!(b.slice_type, a.slice_type);
    assert_eq!(b.slice_sao_luma_flag, a.slice_sao_luma_flag);
    assert_eq!(decoded.last, [false, true]);
    // The placeholder's 00 00 01 needs one emulation prevention byte, in
    // the slice data rather than the header.
    assert_eq!(a.emulation_prevention_bytes, 0);
    assert!(a.header_bytes > 2 && b.header_bytes > 2);
    Ok(())
}

#[test]
fn a_picture_has_no_more_slice_segments_than_ctbs() -> Result<(), Error> {
    // 64x48 in CTBs of 16: twelve.
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec {
        dependent_slices: true,
        ..PpsSpec::default()
    };
    let segment = |address: u32| {
        slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                first: address == 0,
                dependent: address > 0,
                address,
                ..SliceSpec::default()
            },
        )
    };
    let mut units = vec![sps(&spec), pps(&pps_spec)];
    units.extend((0..12).map(segment));
    let mut d = decoder();
    d.decode(&sample(&units), 0)?;
    assert_eq!(d.accelerator().decoded[0].headers.len(), 12);
    units.push(segment(11));
    let mut d = decoder();
    assert_eq!(
        d.decode(&sample(&units), 0).err(),
        Some(Error::MoreSliceSegmentsThanCtbs)
    );
    assert!(d.accelerator().decoded.is_empty());
    Ok(())
}

#[test]
fn the_largest_latency_increase_does_not_overflow() -> Result<(), Error> {
    // 2^32 - 2, the most 7.4.3.2.1 allows, past two reorder pictures.
    let spec = SpsSpec {
        latency_plus1: u32::MAX - 1,
        sets: gop_sets(),
        ..SpsSpec::default()
    };
    let mut d = decoder();
    d.set_output_order(OutputOrder::Display);
    let idr = picture(&spec, nal_type::IDR_W_RADL, 2, 0, 0);
    let p = picture(&spec, nal_type::TRAIL_R, 1, 4, 0);
    let first = sample(&[sps(&spec), pps(&PpsSpec::default()), idr]);
    let out = run(&mut d, &[first, sample(&[p])])?;
    assert_eq!(out, [0, 4]);
    let latency = sps::Sps::parse(&sps(&spec))?.max_latency_pictures();
    assert_eq!(latency, Some(u32::MAX));
    Ok(())
}

/// Decodes `samples` into a fresh decoder: the error, and how many
/// pictures reached the accelerator.
fn refused(samples: &[Vec<u8>]) -> (Option<Error>, usize) {
    let mut d = decoder();
    let error = samples
        .iter()
        .enumerate()
        .find_map(|(id, sample)| d.decode(sample, id as u64).err());
    (error, d.accelerator().decoded.len())
}

#[test]
fn slice_segments_of_one_picture_keep_its_sps_and_agree() {
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec::default();
    let idr = |unit_type: u8, address: u32| {
        slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type,
                first: address == 0,
                address,
                ..SliceSpec::default()
            },
        )
    };
    let other_sps = SpsSpec {
        reorder: 1,
        ..SpsSpec::default()
    };
    let changed = sample(&[
        sps(&spec),
        pps(&pps_spec),
        idr(nal_type::IDR_W_RADL, 0),
        sps(&other_sps),
        idr(nal_type::IDR_W_RADL, 3),
    ]);
    assert_eq!(refused(&[changed]), (Some(Error::SpsChangedInPicture), 0));
    let disagree = sample(&[
        sps(&spec),
        pps(&pps_spec),
        idr(nal_type::IDR_W_RADL, 0),
        idr(nal_type::IDR_N_LP, 3),
    ]);
    assert_eq!(
        refused(&[disagree]),
        (Some(Error::SliceSegmentsDisagree), 0)
    );
    let whole = sample(&[
        sps(&spec),
        pps(&pps_spec),
        idr(nal_type::IDR_W_RADL, 0),
        sps(&spec),
        idr(nal_type::IDR_W_RADL, 3),
    ]);
    assert_eq!(refused(&[whole]), (None, 1));
}

#[test]
fn slice_segments_at_one_address_never_reach_the_accelerator() {
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec::default();
    let at = |address: u32| {
        slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                first: address == 0,
                address,
                ..SliceSpec::default()
            },
        )
    };
    let twice = sample(&[sps(&spec), pps(&pps_spec), at(0), at(3), at(3)]);
    assert_eq!(
        refused(&[twice]),
        (Some(Error::OverlappingSliceSegments), 0)
    );
    let apart = sample(&[sps(&spec), pps(&pps_spec), at(0), at(3), at(7)]);
    assert_eq!(refused(&[apart]), (None, 1));
}

#[test]
fn a_slice_segment_with_no_data_is_refused() {
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec::default();
    let empty = SliceSpec {
        unit_type: nal_type::IDR_W_RADL,
        no_data: true,
        ..SliceSpec::default()
    };
    let units = [sps(&spec), pps(&pps_spec), slice(&spec, &pps_spec, &empty)];
    assert_eq!(
        refused(&[sample(&units)]),
        (Some(Error::EmptySliceSegment), 0)
    );
}

#[test]
fn references_past_the_dpb_or_at_the_current_order_count_are_refused() {
    // A DPB of two pictures, the current one among them: a set of one,
    // which the parser takes, and a long-term picture beside it are one
    // too many. P1 predicts from I0; P2 from P1, and from I0 as a
    // long-term picture.
    let spec = SpsSpec {
        dpb: 2,
        reorder: 0,
        sets: vec![Set::Explicit(vec![(0, true)], Vec::new())],
        long_term: Some(Vec::new()),
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let first = sample(&[
        sps(&spec),
        pps(&pps_spec),
        picture(&spec, nal_type::IDR_W_RADL, 2, 0, 0),
    ]);
    let p1 = sample(&[picture(&spec, nal_type::TRAIL_R, 1, 1, 0)]);
    let p2 = SliceSpec {
        slice_type: 1,
        poc_lsb: 2,
        long_term: (Vec::new(), vec![(0, true, None)]),
        ..SliceSpec::default()
    };
    let p2 = sample(&[slice(&spec, &pps_spec, &p2)]);
    assert_eq!(
        refused(&[first, p1, p2]),
        (Some(Error::TooManyReferences), 2)
    );
    // A long-term picture named by LSB 0 is the IDR, POC 0, and so is the
    // P picture of LSB 0 after it.
    let spec = SpsSpec {
        lsb_minus4: 0,
        dpb: 4,
        reorder: 0,
        sets: vec![Set::Explicit(Vec::new(), Vec::new())],
        long_term: Some(Vec::new()),
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let idr = SliceSpec {
        unit_type: nal_type::IDR_W_RADL,
        ..SliceSpec::default()
    };
    let same = SliceSpec {
        slice_type: 1,
        poc_lsb: 0,
        long_term: (Vec::new(), vec![(0, true, None)]),
        ..SliceSpec::default()
    };
    let first = sample(&[sps(&spec), pps(&pps_spec), slice(&spec, &pps_spec, &idr)]);
    let p = sample(&[slice(&spec, &pps_spec, &same)]);
    assert_eq!(
        refused(&[first, p]),
        (Some(Error::ReferenceAtCurrentPoc), 1)
    );
}

#[test]
fn what_radeonsi_cannot_decode_is_refused() {
    let pps_spec = PpsSpec::default();
    for (spec, why) in [
        (
            SpsSpec {
                profile_idc: 4,
                chroma_format_idc: 2,
                ..SpsSpec::default()
            },
            "unsupported: profiles outside Main, Main 10 and Main Still Picture",
        ),
        (
            SpsSpec {
                chroma_format_idc: 2,
                ..SpsSpec::default()
            },
            "unsupported: chroma formats other than 4:2:0",
        ),
        (
            SpsSpec {
                profile_idc: 2,
                bit_depth_minus8: 4,
                ..SpsSpec::default()
            },
            "unsupported: bit depths above 10",
        ),
        (
            SpsSpec {
                range_extension: true,
                ..SpsSpec::default()
            },
            "unsupported: the range extensions",
        ),
    ] {
        let idr = slice(
            &spec,
            &pps_spec,
            &SliceSpec {
                unit_type: nal_type::IDR_W_RADL,
                ..SliceSpec::default()
            },
        );
        let mut d = decoder();
        let result = d.decode(&sample(&[sps(&spec), pps(&pps_spec), idr]), 0);
        assert_eq!(result.err().map(|e| e.to_string()).as_deref(), Some(why));
        assert!(d.accelerator().decoded.is_empty());
    }
}

#[test]
fn main_10_configures_ten_bits_and_scaling_lists_default() -> Result<(), Error> {
    let spec = SpsSpec {
        profile_idc: 2,
        bit_depth_minus8: 2,
        scaling_list_enabled: true,
        ..SpsSpec::default()
    };
    let pps_spec = PpsSpec::default();
    let idr = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_N_LP,
            ..SliceSpec::default()
        },
    );
    let mut d = decoder();
    let out = d.decode(&sample(&[sps(&spec), pps(&pps_spec), idr]), 7)?;
    assert_eq!(d.accelerator().configured, [(64, 48, 10, 5)]);
    assert_eq!(out.len(), 1);
    assert_eq!(
        (out[0].bit_depth, out[0].crop, out[0].id),
        (10, (0, 0, 64, 46), 7)
    );
    assert!(d.accelerator().decoded[0].scaling);
    Ok(())
}

#[test]
fn a_reset_refuses_pictures_until_a_random_access_point() -> Result<(), Error> {
    let (mut d, _) = gop_stream(OutputOrder::Decode)?;
    d.reset();
    let spec = SpsSpec {
        sets: gop_sets(),
        ..SpsSpec::default()
    };
    let p = slice(
        &spec,
        &PpsSpec::default(),
        &SliceSpec {
            slice_type: 1,
            poc_lsb: 8,
            set: Ok(0),
            ..SliceSpec::default()
        },
    );
    assert_eq!(
        d.decode(&sample(&[p]), 9).err(),
        Some(Error::MissingReference)
    );
    Ok(())
}

#[test]
fn the_vuis_hrd_parameters_are_read_past_to_the_extension_flags() -> Result<(), Error> {
    let spec = SpsSpec {
        vui_hrd: true,
        range_extension: true,
        ..SpsSpec::default()
    };
    let parsed = sps::Sps::parse(&sps(&spec))?;
    assert_eq!(parsed.vui.and_then(|vui| vui.timing), Some((1001, 60000)));
    assert!(parsed.sps_range_extension_flag);
    Ok(())
}

#[test]
fn an_hvcc_record_carries_the_parameter_sets() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let pps_spec = PpsSpec::default();
    let mut record = vec![1, 0x01, 0x40, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 93];
    record.extend_from_slice(&[0xf0, 0, 0xfc, 0xfd, 0xf8, 0xf8, 0, 0, 0x0f, 2]);
    for (kind, unit) in [(33u8, sps(&spec)), (34, pps(&pps_spec))] {
        record.extend_from_slice(&[0x80 | kind, 0, 1]);
        record.extend_from_slice(&(unit.len() as u16).to_be_bytes());
        record.extend_from_slice(&unit);
    }
    let idr = slice(
        &spec,
        &pps_spec,
        &SliceSpec {
            unit_type: nal_type::IDR_W_RADL,
            ..SliceSpec::default()
        },
    );
    let mut sample = (idr.len() as u32).to_be_bytes().to_vec();
    sample.extend_from_slice(&idr);
    let mut d = Decoder::from_hvcc(Recorder::default(), &record)?;
    assert_eq!(d.decode(&sample, 3)?.len(), 1);
    Ok(())
}

mod fuzz;
mod ranges;
