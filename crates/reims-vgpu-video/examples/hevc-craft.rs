//! Rewrites an HEVC stream into the cases no encoder at hand writes, every
//! picture decoding as the source's does, for tests of them: reference
//! picture list modification, short-term sets predicted from others (inter
//! RPS prediction) in the SPS and in slice segment headers, and long-term
//! reference pictures. x265 4.3 and kvazaar 2.3.2 write none of them: their
//! SPSs carry no short-term sets, and each slice segment header codes its
//! own set outright (`hevc-info`, 2026-10-24). Ours.
//!
//! ```sh
//! cargo run --example hevc-craft -- in.265 out.265 [EDIT...] [--check]
//! ```
//!
//! A slice segment header ends byte-aligned before the slice data
//! (7.3.6.1), so a header can be written again at another length with its
//! slice data kept byte for byte, entry points and all. The edits:
//!
//! - `--lists`: every PPS gets `lists_modification_present_flag`, and each P
//!   or B slice segment header whose picture predicts from more than one
//!   picture (`NumPicTotalCurr` > 1) a `ref_pic_lists_modification()` that
//!   leaves its lists as they were: `list_entry_lX[i]` is `i %
//!   NumPicTotalCurr`, since `RefPicListTempX` repeats its `NumPicTotalCurr`
//!   pictures to its length (8-8, 8-10). The flags for list 0 and list 1
//!   turn from header to header: both, list 0 alone, list 1 alone.
//! - `--sets sps`: every SPS carries the short-term sets the slice segment
//!   headers use, up to 64 in order of first use, each after the first
//!   predicted from the one before it where some `deltaRps` gives it
//!   exactly (7.4.8), and coded outright where none does; each header
//!   names its set (`short_term_ref_pic_set_idx`).
//! - `--sets slice`: the same SPS sets, and each header codes its own set,
//!   predicted from the nearest SPS set that gives it exactly
//!   (`delta_idx_minus1`), or outright where none does.
//! - `--long-term`: the SPS gets `long_term_ref_pics_present_flag`, and a
//!   reference picture becomes long-term from the first picture after
//!   which marking it so changes no picture's decode, to the end of its
//!   life. Long-term marking changes motion vector prediction (8.5.3.2.7
//!   to 8.5.3.2.9: no candidate between a long-term and a short-term
//!   reference, no scaling between long-term ones), and once long-term a
//!   picture never comes back to short-term (8.3.2). So a picture `X`
//!   becomes long-term in picture `Q` and every one after it whose set
//!   holds it only where, in each, `X` is in none of the slices' lists
//!   (as a `Foll` picture, or one past `num_ref_idx_active`), or every
//!   entry of every list is `X` and the SPS has temporal motion vector
//!   prediction off, which a later picture's collocated motion would
//!   otherwise see: there every candidate names `X`, which is never
//!   scaled, long-term or not. A picture made long-term goes last in
//!   `RefPicListTempX`, so where that changes a list the PPSs get
//!   `lists_modification_present_flag` and the header a modification that
//!   puts the list back. x265's and kvazaar's streams with B-frames drop a
//!   picture from their sets soon after the last list that holds it, so
//!   they give a long-term picture or two; one reference, no B-frames and no
//!   temporal prediction (x265's `--ref 1 --bframes 0 --no-temporal-mvp`,
//!   kvazaar's `--gop 0 --ref 1 --no-tmvp`) make every reference
//!   long-term. The SPS lists the first 16 distinct
//!   `lt_ref_pic_poc_lsb_sps` and `used_by_curr_pic_lt_sps_flag` pairs, and
//!   every other entry that matches one names it (`lt_idx_sps`), the rest
//!   being coded in the header; every other entry carries its MSB cycle
//!   (`delta_poc_msb_present_flag`), and each carries it where another
//!   picture of the coded video sequence before it has its order count's
//!   LSBs, as 7.4.7.1 requires. An entry whose MSB cycle would be negative
//!   (a picture after the current one in display order, across an LSB
//!   wrap) and whose LSBs are not unique cannot be coded: such a picture
//!   is made long-term after it, if at all. Within a group and an MSB
//!   cycle, every other picture codes its entries nearest first and the
//!   rest farthest first, so that a long-term set in the lists' order and
//!   one against it, which a modification puts back, are both met.
//!
//! The source's own reference picture sets and lists stay, with each slice
//! data unchanged, so a decoder that reads the crafted syntax right
//! decodes every picture as the source's.
//!
//! `--check` decodes the source and the output through this crate's
//! decoder over a stand-in accelerator that records, for each picture,
//! its order count and its reference picture set (each picture's order
//! count, set and marking), and for each slice segment its lists as order
//! counts; then the output order. It fails unless each is the source's, a
//! reference made long-term apart, whose order count and whether the
//! picture predicts from it must stay. It prints what the output's headers
//! hold, as the decoder parsed them.
//!
//! Takes an Annex B stream whose SPSs are copies of one, as x265 and
//! kvazaar write. Writes Annex B; the checks mux it into an MP4. Runs on
//! any host: it needs no GPU.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use reims_vgpu_video::codec::h264::nal::AnnexB;
use reims_vgpu_video::codec::hevc::nal::{self, NalUnit, is_idr, nal_type};
use reims_vgpu_video::codec::hevc::pps::Pps;
use reims_vgpu_video::codec::hevc::sps::Sps;
use reims_vgpu_video::codec::hevc::{
    self, Accelerator, Decoder, NalFormat, OutputOrder, PictureParams, RpsMembership, SliceParams,
};

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hevc-craft: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

const USAGE: &str =
    "usage: hevc-craft <in.265> <out.265> [--lists] [--sets sps|slice] [--long-term] [--check]";

/// `IDR_W_RADL`, the first IDR type (Table 7-1).
const IDR_W_RADL: u8 = 19;
/// `RSV_IRAP_VCL23`, the last IRAP type.
const RSV_IRAP_VCL23: u8 = 23;
/// `BLA_W_LP`, the first IRAP type.
const BLA_W_LP: u8 = 16;
/// How many `lt_ref_pic_poc_lsb_sps` the SPS lists, of the 32 7.4.3.2.1
/// allows.
const SPS_LONG_TERM_CANDIDATES: usize = 16;

// ---------------------------------------------------------------------------
// The edits.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetsIn {
    /// The headers name the SPS's sets.
    Sps,
    /// The headers code their own, predicted from the SPS's.
    Slice,
}

#[derive(Default)]
struct Edits {
    lists: bool,
    sets: Option<SetsIn>,
    long_term: bool,
    check: bool,
}

fn parse_edits(args: &[String]) -> Result<Edits, String> {
    let mut edits = Edits::default();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--lists" => edits.lists = true,
            "--sets" => {
                edits.sets = Some(match args.next().map(String::as_str) {
                    Some("sps") => SetsIn::Sps,
                    Some("slice") => SetsIn::Slice,
                    _ => return Err("--sets is sps or slice".into()),
                });
            }
            "--long-term" => edits.long_term = true,
            "--check" => edits.check = true,
            other => return Err(format!("{other}: unknown option")),
        }
    }
    Ok(edits)
}

// ---------------------------------------------------------------------------
// RBSP bits.
// ---------------------------------------------------------------------------

fn unescape(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut zeros = 0;
    for &b in data {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// An RBSP escaped (7.4.2): one that ends in a nonzero byte, as every unit
/// written here does, a stop bit or an alignment bit in its last byte.
fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 8);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// The byte of an escaped unit at which its RBSP's byte `offset` is.
fn escaped_offset(escaped: &[u8], offset: usize) -> usize {
    let mut zeros = 0;
    let mut count = 0;
    for (i, &b) in escaped.iter().enumerate() {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        if count == offset {
            return i;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        count += 1;
    }
    escaped.len()
}

/// A reader of an RBSP, positions in bits from its first byte (the NAL
/// unit header's).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn bit(&mut self) -> Result<bool, String> {
        let byte = self.data.get(self.pos / 8).ok_or("the unit ends early")?;
        let bit = (byte >> (7 - self.pos % 8)) & 1 == 1;
        self.pos += 1;
        Ok(bit)
    }

    fn bits(&mut self, n: u32) -> Result<u32, String> {
        let mut value = 0;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.bit()?);
        }
        Ok(value)
    }

    fn skip(&mut self, n: usize) -> Result<(), String> {
        for _ in 0..n {
            self.bit()?;
        }
        Ok(())
    }

    fn ue(&mut self) -> Result<u32, String> {
        let mut zeros = 0;
        while !self.bit()? {
            zeros += 1;
            if zeros > 31 {
                return Err("bad Exp-Golomb code".into());
            }
        }
        let mut value = 1u64;
        for _ in 0..zeros {
            value = (value << 1) | u64::from(self.bit()?);
        }
        u32::try_from(value - 1).map_err(|e| e.to_string())
    }

    fn se(&mut self) -> Result<i32, String> {
        let code = self.ue()?;
        let magnitude = i32::try_from(code.div_ceil(2)).map_err(|e| e.to_string())?;
        Ok(if code % 2 == 1 { magnitude } else { -magnitude })
    }
}

fn get_bit(data: &[u8], pos: usize) -> bool {
    (data[pos / 8] >> (7 - pos % 8)) & 1 == 1
}

/// Bits being written.
#[derive(Default)]
struct Writer {
    bits: Vec<bool>,
}

impl Writer {
    fn bit(&mut self, value: bool) {
        self.bits.push(value);
    }

    fn bits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
    }

    fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let len = 64 - code.leading_zeros();
        self.bits
            .extend(std::iter::repeat_n(false, len as usize - 1));
        for i in (0..len).rev() {
            self.bits.push((code >> i) & 1 == 1);
        }
    }

    fn copy(&mut self, data: &[u8], from: usize, to: usize) {
        self.bits.extend((from..to).map(|pos| get_bit(data, pos)));
    }

    /// A one, then zeros to the byte's end: `rbsp_trailing_bits()` and
    /// `byte_alignment()` alike.
    fn finish(mut self) -> Vec<u8> {
        self.bits.push(true);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        let mut out = vec![0u8; self.bits.len() / 8];
        for (i, &bit) in self.bits.iter().enumerate() {
            if bit {
                out[i / 8] |= 0x80 >> (i % 8);
            }
        }
        out
    }
}

/// The position of the RBSP's stop bit: its last 1.
fn stop_bit(rbsp: &[u8]) -> Result<usize, String> {
    let last = rbsp.iter().rposition(|&b| b != 0).ok_or("an empty RBSP")?;
    Ok(last * 8 + 7 - rbsp[last].trailing_zeros() as usize)
}

/// `Ceil(Log2(n))`.
fn ceil_log2(n: usize) -> u32 {
    usize::BITS - n.saturating_sub(1).leading_zeros()
}

// ---------------------------------------------------------------------------
// Short-term reference picture sets (7.3.7, 7.4.8).
// ---------------------------------------------------------------------------

/// A short-term set: each picture's order count less the current one's,
/// with whether the current picture predicts from it; those before it
/// nearest first, then those after it nearest first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Set {
    negative: Vec<(i32, bool)>,
    positive: Vec<(i32, bool)>,
}

impl Set {
    fn pictures(&self) -> impl Iterator<Item = &(i32, bool)> {
        self.negative.iter().chain(&self.positive)
    }

    fn used(&self) -> usize {
        self.pictures().filter(|p| p.1).count()
    }

    /// The set without the pictures at `deltas`.
    fn without(&self, deltas: &[i32]) -> Self {
        let keep = |list: &[(i32, bool)]| {
            list.iter()
                .copied()
                .filter(|(d, _)| !deltas.contains(d))
                .collect()
        };
        Self {
            negative: keep(&self.negative),
            positive: keep(&self.positive),
        }
    }
}

/// `st_ref_pic_set(stRpsIdx)`, `earlier` the sets before it: the SPS's so
/// far, or all of them for a slice segment header's own.
fn read_set(r: &mut Reader<'_>, earlier: &[Set], in_header: bool) -> Result<Set, String> {
    let index = earlier.len();
    if index != 0 && r.bit()? {
        let delta_idx = if in_header { r.ue()? as usize + 1 } else { 1 };
        let reference = index
            .checked_sub(delta_idx)
            .and_then(|i| earlier.get(i))
            .ok_or("delta_idx_minus1 names no set")?;
        let sign = r.bit()?;
        let magnitude = i32::try_from(r.ue()? + 1).map_err(|e| e.to_string())?;
        let delta_rps = if sign { -magnitude } else { magnitude };
        let mut flags = Vec::new();
        for _ in 0..=reference.pictures().count() {
            let used = r.bit()?;
            let kept = used || r.bit()?;
            flags.push((used, kept));
        }
        return Ok(predicted(reference, delta_rps, &flags));
    }
    let negatives = r.ue()?;
    let positives = r.ue()?;
    if negatives + positives > 16 {
        return Err("a short-term set of more than 16 pictures".into());
    }
    let mut set = Set::default();
    let mut poc = 0i32;
    for _ in 0..negatives {
        poc -= i32::try_from(r.ue()? + 1).map_err(|e| e.to_string())?;
        set.negative.push((poc, r.bit()?));
    }
    poc = 0;
    for _ in 0..positives {
        poc += i32::try_from(r.ue()? + 1).map_err(|e| e.to_string())?;
        set.positive.push((poc, r.bit()?));
    }
    Ok(set)
}

/// 7-61 and 7-62: the set predicted from `reference` by `delta_rps`, with
/// each of the reference's pictures and then the reference picture itself
/// flagged (`used_by_curr_pic_flag`, kept by `use_delta_flag`).
fn predicted(reference: &Set, delta_rps: i32, flags: &[(bool, bool)]) -> Set {
    let negatives = reference.negative.len();
    let own = flags[reference.pictures().count()];
    let mut set = Set::default();
    for (j, &(d, _)) in reference.positive.iter().enumerate().rev() {
        let (used, kept) = flags[negatives + j];
        if d + delta_rps < 0 && kept {
            set.negative.push((d + delta_rps, used));
        }
    }
    if delta_rps < 0 && own.1 {
        set.negative.push((delta_rps, own.0));
    }
    for (j, &(d, _)) in reference.negative.iter().enumerate() {
        let (used, kept) = flags[j];
        if d + delta_rps < 0 && kept {
            set.negative.push((d + delta_rps, used));
        }
    }
    for (j, &(d, _)) in reference.negative.iter().enumerate().rev() {
        let (used, kept) = flags[j];
        if d + delta_rps > 0 && kept {
            set.positive.push((d + delta_rps, used));
        }
    }
    if delta_rps > 0 && own.1 {
        set.positive.push((delta_rps, own.0));
    }
    for (j, &(d, _)) in reference.positive.iter().enumerate() {
        let (used, kept) = flags[negatives + j];
        if d + delta_rps > 0 && kept {
            set.positive.push((d + delta_rps, used));
        }
    }
    set
}

/// The `deltaRps` that predicts `target` from `reference` exactly, the
/// smallest: one under which each of `target`'s pictures is one of
/// `reference`'s moved by it, or the reference picture itself, at
/// `deltaRps`. The derivation keeps the order it needs, nearest first.
fn prediction(target: &Set, reference: &Set) -> Option<i32> {
    let wanted: Vec<i32> = target.pictures().map(|p| p.0).collect();
    // The reference's pictures, and 0 for the reference picture itself.
    let from: Vec<i32> = reference.pictures().map(|p| p.0).chain([0]).collect();
    let mut candidates: Vec<i32> = if wanted.is_empty() {
        vec![-1]
    } else {
        wanted
            .iter()
            .flat_map(|&t| from.iter().map(move |&f| t - f))
            .filter(|&d| d != 0 && d.abs() <= 1 << 15)
            .collect()
    };
    candidates.sort_by_key(|&d| (d.abs(), d));
    candidates.dedup();
    let found = candidates
        .into_iter()
        .find(|&d| wanted.iter().all(|&t| from.contains(&(t - d))))?;
    let flags = prediction_flags(target, reference, found);
    (predicted(reference, found, &flags) == *target).then_some(found)
}

/// The flags that keep `target`'s pictures of `reference` moved by
/// `delta_rps`, each `used_by_curr_pic_flag` as `target` has it.
fn prediction_flags(target: &Set, reference: &Set, delta_rps: i32) -> Vec<(bool, bool)> {
    let flag = |d: i32| match target.pictures().find(|p| p.0 == d) {
        Some(&(_, used)) => (used, true),
        None => (false, false),
    };
    reference
        .pictures()
        .map(|&(d, _)| flag(d + delta_rps))
        .chain([flag(delta_rps)])
        .collect()
}

/// `st_ref_pic_set(earlier.len())`: predicted from the set
/// `reference` names in `earlier` by `delta_rps`, or outright. In the SPS
/// the reference is always the set before.
fn write_set(
    w: &mut Writer,
    set: &Set,
    earlier: &[Set],
    in_header: bool,
    reference: Option<(usize, i32)>,
) {
    let index = earlier.len();
    if index != 0 {
        w.bit(reference.is_some());
    }
    if let Some((at, delta_rps)) = reference {
        if in_header {
            w.ue((index - 1 - at) as u32);
        }
        w.bit(delta_rps < 0);
        w.ue(delta_rps.unsigned_abs() - 1);
        for (used, kept) in prediction_flags(set, &earlier[at], delta_rps) {
            w.bit(used);
            if !used {
                w.bit(kept);
            }
        }
        return;
    }
    w.ue(set.negative.len() as u32);
    w.ue(set.positive.len() as u32);
    let mut last = 0;
    for &(d, used) in &set.negative {
        w.ue((last - d - 1) as u32);
        w.bit(used);
        last = d;
    }
    last = 0;
    for &(d, used) in &set.positive {
        w.ue((d - last - 1) as u32);
        w.bit(used);
        last = d;
    }
}

// ---------------------------------------------------------------------------
// Where the fields are.
// ---------------------------------------------------------------------------

/// `profile_tier_level(1, max_sub_layers_minus1)` (7.3.3), skipped.
fn skip_profile_tier_level(r: &mut Reader<'_>, max_sub_layers_minus1: u32) -> Result<(), String> {
    r.skip(96)?;
    let mut present = Vec::new();
    for _ in 0..max_sub_layers_minus1 {
        present.push((r.bit()?, r.bit()?));
    }
    if max_sub_layers_minus1 > 0 {
        r.skip(2 * (8 - max_sub_layers_minus1 as usize))?;
    }
    for (profile, level) in present {
        if profile {
            r.skip(88)?;
        }
        if level {
            r.skip(8)?;
        }
    }
    Ok(())
}

/// `scaling_list_data()` (7.3.4), skipped.
fn skip_scaling_list_data(r: &mut Reader<'_>) -> Result<(), String> {
    for size_id in 0..4 {
        let step = if size_id == 3 { 3 } else { 1 };
        for _ in (0..6).step_by(step) {
            if !r.bit()? {
                r.ue()?;
            } else {
                if size_id > 1 {
                    r.se()?;
                }
                for _ in 0..(1usize << (4 + 2 * size_id)).min(64) {
                    r.se()?;
                }
            }
        }
    }
    Ok(())
}

/// What the rewrites need of an SPS, and where its sets are.
#[derive(Clone)]
struct SpsFields {
    id: u32,
    /// `num_short_term_ref_pic_sets`' first bit, and
    /// `sps_temporal_mvp_enabled_flag`'s: the sets and the long-term
    /// pictures between them are written again.
    sets_at: usize,
    /// `long_term_ref_pics_present_flag`'s bit.
    long_term_at: usize,
    temporal_mvp_at: usize,
    sets: Vec<Set>,
    long_term_present: bool,
    /// `lt_ref_pic_poc_lsb_sps[]` with `used_by_curr_pic_lt_sps_flag[]`.
    long_term: Vec<(u32, bool)>,
    temporal_mvp: bool,
    log2_max_poc_lsb: u32,
    /// `PicSizeInCtbsY`.
    ctbs: u32,
    separate_colour_plane: bool,
    chroma_array_type: u32,
    sao: bool,
}

fn sps_fields(rbsp: &[u8]) -> Result<SpsFields, String> {
    let mut r = Reader {
        data: rbsp,
        pos: 16,
    };
    r.skip(4)?; // sps_video_parameter_set_id
    let max_sub_layers_minus1 = r.bits(3)?;
    r.skip(1)?;
    skip_profile_tier_level(&mut r, max_sub_layers_minus1)?;
    let id = r.ue()?;
    let chroma_format_idc = r.ue()?;
    let separate_colour_plane = chroma_format_idc == 3 && r.bit()?;
    let width = r.ue()?;
    let height = r.ue()?;
    if r.bit()? {
        for _ in 0..4 {
            r.ue()?;
        }
    }
    r.ue()?;
    r.ue()?;
    let log2_max_poc_lsb = r.ue()? + 4;
    let all_layers = r.bit()?;
    for _ in (if all_layers { 0 } else { max_sub_layers_minus1 })..=max_sub_layers_minus1 {
        r.ue()?;
        r.ue()?;
        r.ue()?;
    }
    let min_cb_log2 = r.ue()? + 3;
    let ctb_log2 = min_cb_log2 + r.ue()?;
    for _ in 0..4 {
        r.ue()?;
    }
    if r.bit()? && r.bit()? {
        skip_scaling_list_data(&mut r)?;
    }
    r.skip(1)?; // amp_enabled_flag
    let sao = r.bit()?;
    if r.bit()? {
        r.skip(8)?;
        r.ue()?;
        r.ue()?;
        r.skip(1)?;
    }
    let sets_at = r.pos;
    let mut sets = Vec::new();
    for _ in 0..r.ue()? {
        let set = read_set(&mut r, &sets, false)?;
        sets.push(set);
    }
    let long_term_at = r.pos;
    let long_term_present = r.bit()?;
    let mut long_term = Vec::new();
    if long_term_present {
        for _ in 0..r.ue()? {
            long_term.push((r.bits(log2_max_poc_lsb)?, r.bit()?));
        }
    }
    let temporal_mvp_at = r.pos;
    let temporal_mvp = r.bit()?;
    let ctb = 1u32 << ctb_log2;
    Ok(SpsFields {
        id,
        sets_at,
        long_term_at,
        temporal_mvp_at,
        sets,
        long_term_present,
        long_term,
        temporal_mvp,
        log2_max_poc_lsb,
        ctbs: width.div_ceil(ctb) * height.div_ceil(ctb),
        separate_colour_plane,
        chroma_array_type: if separate_colour_plane {
            0
        } else {
            chroma_format_idc
        },
        sao,
    })
}

/// What the slice headers need of a PPS, and where its
/// `lists_modification_present_flag` is.
#[derive(Clone)]
struct PpsFields {
    sps_id: u32,
    dependent_slice_segments: bool,
    output_flag_present: bool,
    extra_slice_header_bits: u32,
    num_ref_idx_default_minus1: [u32; 2],
    lists_modification_at: usize,
    lists_modification: bool,
}

fn pps_fields(rbsp: &[u8]) -> Result<(u32, PpsFields), String> {
    let mut r = Reader {
        data: rbsp,
        pos: 16,
    };
    let id = r.ue()?;
    let sps_id = r.ue()?;
    let dependent_slice_segments = r.bit()?;
    let output_flag_present = r.bit()?;
    let extra_slice_header_bits = r.bits(3)?;
    r.skip(2)?; // sign_data_hiding_enabled_flag, cabac_init_present_flag
    let num_ref_idx_default_minus1 = [r.ue()?, r.ue()?];
    r.se()?; // init_qp_minus26
    r.skip(2)?; // constrained_intra_pred_flag, transform_skip_enabled_flag
    if r.bit()? {
        r.ue()?; // diff_cu_qp_delta_depth
    }
    r.se()?;
    r.se()?;
    r.skip(4)?; // chroma offsets present, weighted P and B, transquant bypass
    let tiles = r.bit()?;
    r.skip(1)?; // entropy_coding_sync_enabled_flag
    if tiles {
        let columns = r.ue()?;
        let rows = r.ue()?;
        if !r.bit()? {
            for _ in 0..columns + rows {
                r.ue()?;
            }
        }
        r.skip(1)?;
    }
    r.skip(1)?; // pps_loop_filter_across_slices_enabled_flag
    if r.bit()? {
        // deblocking_filter_control_present_flag: the override flag, the
        // disabled flag and, when not disabled, the offsets.
        r.skip(1)?;
        if !r.bit()? {
            r.se()?;
            r.se()?;
        }
    }
    if r.bit()? {
        skip_scaling_list_data(&mut r)?;
    }
    let lists_modification_at = r.pos;
    let lists_modification = r.bit()?;
    Ok((
        id,
        PpsFields {
            sps_id,
            dependent_slice_segments,
            output_flag_present,
            extra_slice_header_bits,
            num_ref_idx_default_minus1,
            lists_modification_at,
            lists_modification,
        },
    ))
}

/// Where an independent slice segment header's rewritten fields are, in
/// RBSP bits, and what they hold.
struct SliceFields {
    /// From `short_term_ref_pic_set_sps_flag` to the long-term pictures'
    /// end, and where the long-term pictures start; none for an IDR
    /// picture.
    rps: Option<(usize, usize, usize)>,
    /// The short-term set in force.
    set: Set,
    /// The long-term pictures the picture predicts from.
    long_term_used: usize,
    /// `slice_type`: 0 B, 1 P, 2 I.
    slice_type: u32,
    num_ref_idx_active_minus1: [u32; 2],
    /// Where `ref_pic_lists_modification()` is, or would be, and its end.
    lists: (usize, usize),
    /// `byte_alignment()`'s one bit, and the slice data's first byte.
    alignment: usize,
    data: usize,
}

impl SliceFields {
    fn num_pic_total_curr(&self) -> usize {
        self.set.used() + self.long_term_used
    }
}

/// The fields of an independent slice segment header up to its list
/// modification. `header_bytes` is the header's length as the codec
/// crate's parser read it, which places `byte_alignment()`: its one bit is
/// the last one of the header's last byte.
fn slice_fields(
    rbsp: &[u8],
    unit_type: u8,
    pps: &PpsFields,
    sps: &SpsFields,
    header_bytes: usize,
) -> Result<Option<SliceFields>, String> {
    let mut r = Reader {
        data: rbsp,
        pos: 16,
    };
    let first = r.bit()?;
    if (BLA_W_LP..=RSV_IRAP_VCL23).contains(&unit_type) {
        r.skip(1)?;
    }
    r.ue()?; // slice_pic_parameter_set_id
    if !first {
        if pps.dependent_slice_segments && r.bit()? {
            return Ok(None);
        }
        r.skip(ceil_log2(sps.ctbs as usize) as usize)?;
    }
    r.skip(pps.extra_slice_header_bits as usize)?;
    let slice_type = r.ue()?;
    if pps.output_flag_present {
        r.skip(1)?;
    }
    if sps.separate_colour_plane {
        r.skip(2)?;
    }
    let mut rps = None;
    let mut set = Set::default();
    let mut long_term_used = 0;
    if !is_idr(unit_type) {
        r.skip(sps.log2_max_poc_lsb as usize)?;
        let start = r.pos;
        if r.bit()? {
            let index = if sps.sets.len() > 1 {
                r.bits(ceil_log2(sps.sets.len()))? as usize
            } else {
                0
            };
            set = sps
                .sets
                .get(index)
                .cloned()
                .ok_or("short_term_ref_pic_set_idx names no set")?;
        } else {
            set = read_set(&mut r, &sps.sets, true)?;
        }
        let long_term_at = r.pos;
        if sps.long_term_present {
            let from_sps = if sps.long_term.is_empty() { 0 } else { r.ue()? };
            let own = r.ue()?;
            for i in 0..from_sps + own {
                let (_, used) = if i < from_sps {
                    let index = if sps.long_term.len() > 1 {
                        r.bits(ceil_log2(sps.long_term.len()))?
                    } else {
                        0
                    };
                    *sps.long_term
                        .get(index as usize)
                        .ok_or("lt_idx_sps names no picture")?
                } else {
                    (r.bits(sps.log2_max_poc_lsb)?, r.bit()?)
                };
                if r.bit()? {
                    r.ue()?; // delta_poc_msb_cycle_lt
                }
                long_term_used += usize::from(used);
            }
        }
        rps = Some((start, long_term_at, r.pos));
        if sps.temporal_mvp {
            r.skip(1)?; // slice_temporal_mvp_enabled_flag
        }
    }
    if sps.sao {
        r.skip(1)?;
        if sps.chroma_array_type != 0 {
            r.skip(1)?;
        }
    }
    let mut num_ref_idx_active_minus1 = pps.num_ref_idx_default_minus1;
    let lists_at;
    let mut fields = SliceFields {
        rps,
        set,
        long_term_used,
        slice_type,
        num_ref_idx_active_minus1,
        lists: (0, 0),
        alignment: 0,
        data: header_bytes,
    };
    if slice_type != 2 {
        if r.bit()? {
            num_ref_idx_active_minus1[0] = r.ue()?;
            if slice_type == 0 {
                num_ref_idx_active_minus1[1] = r.ue()?;
            }
        }
        fields.num_ref_idx_active_minus1 = num_ref_idx_active_minus1;
        lists_at = r.pos;
        let total = fields.num_pic_total_curr();
        if pps.lists_modification && total > 1 {
            let bits = ceil_log2(total);
            for &count in num_ref_idx_active_minus1
                .iter()
                .take(if slice_type == 0 { 2 } else { 1 })
            {
                if r.bit()? {
                    for _ in 0..=count {
                        r.bits(bits)?;
                    }
                }
            }
        }
    } else {
        lists_at = r.pos;
    }
    fields.lists = (lists_at, r.pos);
    let last = header_bytes
        .checked_sub(1)
        .and_then(|i| rbsp.get(i))
        .filter(|&&b| b != 0)
        .ok_or("a slice segment header with no byte_alignment()")?;
    fields.alignment = (header_bytes - 1) * 8 + 7 - last.trailing_zeros() as usize;
    if fields.alignment < fields.lists.1 {
        return Err("the header's fields run past its byte_alignment()".into());
    }
    Ok(Some(fields))
}

// ---------------------------------------------------------------------------
// The source, read.
// ---------------------------------------------------------------------------

/// One NAL unit of the source, escaped, header first.
struct Unit {
    bytes: Vec<u8>,
    unit_type: u8,
    /// An independent slice segment's fields.
    slice: Option<SliceFields>,
}

/// One picture: its access unit's units, and what the decoder made of it.
struct Picture {
    units: Vec<Unit>,
    unit_type: u8,
    /// `PicOrderCntVal`, when the picture decoded.
    poc: Option<i32>,
    /// Each slice segment's lists, as order counts, as the source decodes.
    lists: Vec<[Vec<i32>; 2]>,
    /// Every slice is an I slice.
    intra: bool,
}

/// What the stand-in accelerator saw of one picture.
#[derive(Clone, Debug, Default)]
struct Seen {
    access_unit: usize,
    poc: i32,
    /// Each picture of its reference picture set: order count, set,
    /// long-term marking.
    references: Vec<(i32, RpsMembership, bool)>,
    /// Each slice segment's lists, as order counts.
    lists: Vec<[Vec<i32>; 2]>,
    /// Each slice segment header's facts, as the crate parsed them:
    /// whether it names an SPS set, its own set's bits and the size of the
    /// set it predicts from, its long-term pictures, and whether it
    /// modifies list 0 and list 1.
    headers: Vec<(bool, u32, u32, usize, [bool; 2])>,
}

/// A stand-in accelerator that records what each picture would give the
/// GPU.
#[derive(Default)]
struct Recorder {
    next: usize,
    /// The access unit being decoded.
    access_unit: Cell<usize>,
    seen: RefCell<Vec<Seen>>,
}

impl Accelerator for Recorder {
    type Picture = usize;
    type Error = hevc::Error;

    fn configure(&mut self, _sps: &Sps, _dpb_size: usize) -> Result<(), hevc::Error> {
        Ok(())
    }

    fn new_picture(&mut self) -> Result<usize, hevc::Error> {
        self.next += 1;
        Ok(self.next)
    }

    fn decode_picture(
        &mut self,
        _picture: &usize,
        params: &PictureParams<'_, usize>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), hevc::Error> {
        let pocs: Vec<i32> = params.references.iter().map(|r| r.poc).collect();
        let mut references: Vec<_> = params
            .references
            .iter()
            .map(|r| (r.poc, r.membership, r.long_term))
            .collect();
        references.sort_by_key(|r| r.0);
        self.seen.borrow_mut().push(Seen {
            access_unit: self.access_unit.get(),
            poc: params.poc,
            references,
            lists: slices
                .iter()
                .map(|s| {
                    s.ref_lists
                        .clone()
                        .map(|l| l.iter().map(|&i| pocs[i]).collect())
                })
                .collect(),
            headers: slices
                .iter()
                .filter(|s| !s.header.dependent_slice_segment_flag)
                .map(|s| {
                    let h = s.header;
                    (
                        h.short_term_ref_pic_set_sps_flag,
                        h.st_rps_bits,
                        h.st_rps_reference_delta_pocs,
                        h.long_term.len(),
                        [h.list_entries[0].is_some(), h.list_entries[1].is_some()],
                    )
                })
                .collect(),
        });
        Ok(())
    }
}

/// Decodes a stream's access units through the crate's decoder, and
/// returns what each picture gave the stand-in and the output order, as
/// order counts.
fn decode(units: &[Vec<u8>]) -> Result<(Vec<Seen>, Vec<i32>), String> {
    let mut decoder = Decoder::new(Recorder::default(), NalFormat::AnnexB);
    decoder.set_output_order(OutputOrder::Display);
    let mut order = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        decoder.accelerator().access_unit.set(index);
        let frames = decoder
            .decode(unit, index as u64)
            .map_err(|e| format!("access unit {index} does not decode: {e}"))?;
        order.extend(frames.iter().map(|f| f.poc));
    }
    order.extend(decoder.flush().iter().map(|f| f.poc));
    let seen = decoder.accelerator().seen.take();
    Ok((seen, order))
}

/// The source: its pictures, one an access unit, its one SPS and its PPSs.
struct Source {
    pictures: Vec<Picture>,
    sps: SpsFields,
    pps: BTreeMap<u32, PpsFields>,
}

fn read_source(data: &[u8]) -> Result<Source, String> {
    let mut sps: Option<(Vec<u8>, SpsFields)> = None;
    let mut pps = BTreeMap::new();
    let mut codec_sps = BTreeMap::new();
    let mut codec_pps = BTreeMap::new();
    let mut pictures = Vec::new();
    for (index, au) in nal::access_units(data).into_iter().enumerate() {
        let at = |e: String| format!("access unit {index}: {e}");
        let mut picture = Picture {
            units: Vec::new(),
            unit_type: 0,
            poc: None,
            lists: Vec::new(),
            intra: true,
        };
        let mut independent = None;
        for bytes in AnnexB::new(au) {
            let unit = NalUnit::new(bytes).map_err(|e| at(e.to_string()))?;
            let unit_type = unit.unit_type();
            let mut slice = None;
            match unit_type {
                nal_type::SPS => {
                    let parsed = Sps::parse(bytes).map_err(|e| at(e.to_string()))?;
                    match &sps {
                        Some((first, _)) if first != bytes => {
                            return Err(at("an SPS other than the first: one is crafted".into()));
                        }
                        Some(_) => {}
                        None => sps = Some((bytes.to_vec(), sps_fields(&unescape(bytes))?)),
                    }
                    codec_sps.insert(parsed.sps_seq_parameter_set_id, std::sync::Arc::new(parsed));
                }
                nal_type::PPS => {
                    let parsed = Pps::parse(bytes).map_err(|e| at(e.to_string()))?;
                    let (id, fields) = pps_fields(&unescape(bytes))?;
                    pps.insert(id, fields);
                    codec_pps.insert(parsed.pps_pic_parameter_set_id, std::sync::Arc::new(parsed));
                }
                0..=9 | BLA_W_LP..=21 => {
                    let header = hevc::slice::SliceHeader::parse(
                        unit,
                        |id| codec_pps.get(&id).cloned(),
                        |id| codec_sps.get(&id).cloned(),
                        independent.as_ref(),
                    )
                    .map_err(|e| at(e.to_string()))?;
                    picture.unit_type = unit_type;
                    if !header.dependent_slice_segment_flag {
                        let rbsp = unescape(bytes);
                        let mut r = Reader {
                            data: &rbsp,
                            pos: 17,
                        };
                        if (BLA_W_LP..=RSV_IRAP_VCL23).contains(&unit_type) {
                            r.skip(1)?;
                        }
                        let pps_id = r.ue()?;
                        let p = pps.get(&pps_id).ok_or(at(format!("no PPS {pps_id}")))?;
                        let s = &sps.as_ref().ok_or(at("no SPS".into()))?.1;
                        if p.sps_id != s.id {
                            return Err(at("a PPS of another SPS".into()));
                        }
                        let fields =
                            slice_fields(&rbsp, unit_type, p, s, header.header_bytes as usize)
                                .map_err(at)?
                                .ok_or(at(
                                    "a dependent segment the crate reads as independent".into()
                                ))?;
                        picture.intra &= fields.slice_type == 2;
                        slice = Some(fields);
                        independent = Some(header);
                    }
                }
                _ => {}
            }
            picture.units.push(Unit {
                bytes: bytes.to_vec(),
                unit_type,
                slice,
            });
        }
        pictures.push(picture);
    }
    let sps = sps.ok_or("no SPS")?.1;
    Ok(Source { pictures, sps, pps })
}

// ---------------------------------------------------------------------------
// The plan.
// ---------------------------------------------------------------------------

/// `PicOrderCntVal & (MaxPicOrderCntLsb - 1)`.
fn lsb(poc: i32, log2_max: u32) -> u32 {
    poc.rem_euclid(1 << log2_max) as u32
}

/// `DeltaPocMsbCycleLt` that finds the picture of order count `poc` from
/// the current one's, `current` (8-5); negative for a picture whose MSBs
/// are above the current one's.
fn msb_cycle(current: i32, poc: i32, log2_max: u32) -> i32 {
    let msb = |p: i32| p - lsb(p, log2_max) as i32;
    (msb(current) - msb(poc)) >> log2_max
}

/// One long-term picture of a picture's set, as it is to be coded.
#[derive(Clone, Copy, Debug)]
struct Planned {
    poc: i32,
    used: bool,
    /// `DeltaPocMsbCycleLt`, when the entry carries it.
    cycle: Option<u32>,
    /// The SPS's candidate it names, when it does.
    candidate: Option<usize>,
}

/// For each access unit, the references made long-term in it, as order
/// counts: each picture from the first after which, to the end of its
/// life, being long-term changes no decode and can be coded (the module's
/// doc, `--long-term`).
fn long_term_plan(source: &Source) -> Vec<Vec<(i32, bool)>> {
    let log2 = source.sps.log2_max_poc_lsb;
    // For each reference, each picture whose set holds it: the access
    // unit, whether the picture predicts from it, and whether it may be
    // long-term there.
    let mut lives: BTreeMap<i32, Vec<(usize, bool, bool)>> = BTreeMap::new();
    // The order counts decoded before the current picture in its coded
    // video sequence: a superset of 7.4.7.1's setOfPrevPocVals.
    let mut earlier: Vec<i32> = Vec::new();
    for (index, picture) in source.pictures.iter().enumerate() {
        let Some(poc) = picture.poc else { continue };
        if is_idr(picture.unit_type) || (BLA_W_LP..IDR_W_RADL).contains(&picture.unit_type) {
            earlier.clear();
        }
        let Some(fields) = picture.units.iter().find_map(|u| u.slice.as_ref()) else {
            continue;
        };
        if fields.rps.is_some() {
            let listed: Vec<i32> = picture.lists.iter().flatten().flatten().copied().collect();
            for &(delta, used) in fields.set.pictures() {
                let x = poc + delta;
                let unambiguous = earlier
                    .iter()
                    .filter(|&&p| lsb(p, log2) == lsb(x, log2))
                    .count()
                    <= 1;
                let codable = unambiguous || msb_cycle(poc, x, log2) >= 0;
                // In no list, or the one picture of every list without
                // temporal motion vector prediction.
                let unchanged = !listed.contains(&x)
                    || (listed.iter().all(|&p| p == x) && !source.sps.temporal_mvp);
                lives
                    .entry(x)
                    .or_default()
                    .push((index, used, codable && unchanged));
            }
        }
        earlier.push(poc);
    }
    let mut plan = vec![Vec::new(); source.pictures.len()];
    for (x, life) in lives {
        let from = life
            .iter()
            .rposition(|&(_, _, ok)| !ok)
            .map_or(0, |last| last + 1);
        for &(index, used, _) in &life[from..] {
            plan[index].push((x, used));
        }
    }
    plan
}

/// What the rewrite writes, decided over the whole stream.
struct Plan {
    /// The SPS's short-term sets, and for each after the first the
    /// `deltaRps` that predicts it from the one before, when one does.
    sets: Vec<(Set, Option<i32>)>,
    /// The SPS's long-term candidates: `lt_ref_pic_poc_lsb_sps`,
    /// `used_by_curr_pic_lt_sps_flag`.
    candidates: Vec<(u32, bool)>,
    long_term_present: bool,
    /// Whether the PPSs get `lists_modification_present_flag`: with
    /// `--lists`, and where a picture's long-term pictures, last in
    /// `RefPicListTempX`, would change its lists unless they are modified
    /// back.
    modify_lists: bool,
    /// For each access unit: its short-term set and long-term pictures.
    pictures: Vec<Option<(Set, Vec<Planned>)>>,
}

/// `RefPicListTemp0` and `RefPicListTemp1`'s first `NumPicTotalCurr`
/// pictures (8-8, 8-10), as order counts: those the picture predicts from,
/// short-term before it, short-term after it, then long-term in the order
/// they are coded.
fn temp_lists(poc: i32, set: &Set, long_term: &[Planned]) -> [Vec<i32>; 2] {
    let used = |list: &[(i32, bool)]| -> Vec<i32> {
        list.iter().filter(|p| p.1).map(|p| poc + p.0).collect()
    };
    let (before, after) = (used(&set.negative), used(&set.positive));
    let lt: Vec<i32> = long_term.iter().filter(|e| e.used).map(|e| e.poc).collect();
    [
        [&before[..], &after, &lt].concat(),
        [&after[..], &before, &lt].concat(),
    ]
}

/// The `list_entry_lX` that give a slice's `lists` from `temp`: each
/// picture's place among the pictures the picture predicts from. Each with
/// whether it is the identity, which no modification gives too.
fn list_entries(
    lists: &[Vec<i32>; 2],
    temp: &[Vec<i32>; 2],
) -> Result<[(Vec<u32>, bool); 2], String> {
    let mut out: [(Vec<u32>, bool); 2] = Default::default();
    for ((list, temp), (entries, identity)) in lists.iter().zip(temp).zip(&mut out) {
        for poc in list {
            let at = temp
                .iter()
                .position(|p| p == poc)
                .ok_or(format!("poc {poc} of a list is not predicted from"))?;
            entries.push(at as u32);
        }
        *identity = entries
            .iter()
            .enumerate()
            .all(|(i, &at)| at as usize == i % temp.len());
    }
    Ok(out)
}

fn plan(source: &Source, edits: &Edits) -> Result<Plan, String> {
    let sps = &source.sps;
    let log2 = sps.log2_max_poc_lsb;
    let long_term = if edits.long_term {
        long_term_plan(source)
    } else {
        vec![Vec::new(); source.pictures.len()]
    };
    let mut pictures = Vec::new();
    for (picture, converted) in source.pictures.iter().zip(&long_term) {
        let fields = picture.units.iter().find_map(|u| u.slice.as_ref());
        let (Some(fields), Some(poc)) = (fields, picture.poc) else {
            pictures.push(None);
            continue;
        };
        let deltas: Vec<i32> = converted.iter().map(|&(x, _)| x - poc).collect();
        let entries: Vec<Planned> = converted
            .iter()
            .map(|&(x, used)| Planned {
                poc: x,
                used,
                cycle: None,
                candidate: None,
            })
            .collect();
        pictures.push(Some((fields.set.without(&deltas), entries)));
    }

    // The long-term entries' MSB cycles, every other one's where it may go
    // without, and the SPS's candidates.
    let mut earlier: Vec<i32> = Vec::new();
    let mut count = 0usize;
    let mut candidates: Vec<(u32, bool)> = Vec::new();
    for (picture, planned) in source.pictures.iter().zip(&mut pictures) {
        let Some(poc) = picture.poc else { continue };
        if is_idr(picture.unit_type) || (BLA_W_LP..IDR_W_RADL).contains(&picture.unit_type) {
            earlier.clear();
        }
        if let Some((_, entries)) = planned {
            for entry in entries.iter_mut() {
                let x = entry.poc;
                let ambiguous = earlier
                    .iter()
                    .filter(|&&p| lsb(p, log2) == lsb(x, log2))
                    .count()
                    > 1;
                let cycle = msb_cycle(poc, x, log2);
                if cycle >= 0 && (ambiguous || count % 2 == 1) {
                    entry.cycle = Some(cycle as u32);
                }
                count += 1;
                let pair = (lsb(x, log2), entry.used);
                if !candidates.contains(&pair) && candidates.len() < SPS_LONG_TERM_CANDIDATES {
                    candidates.push(pair);
                }
            }
        }
        earlier.push(poc);
    }
    let mut count = 0usize;
    for (picture, (_, entries)) in pictures.iter_mut().flatten().enumerate() {
        for entry in entries.iter_mut() {
            let pair = (lsb(entry.poc, log2), entry.used);
            if let Some(at) = candidates.iter().position(|&c| c == pair) {
                if count.is_multiple_of(2) {
                    entry.candidate = Some(at);
                }
                count += 1;
            }
        }
        // The SPS's candidates first (7.3.6.1), each group with the cycles
        // it carries in rising order, so that each delta (7-52) is
        // positive; an entry without one may go anywhere. Within a cycle,
        // nearest first in every other picture and farthest first in the
        // rest, so that `RefPicSetLtCurr` is met both in the lists' order
        // and against it, which a modification then puts back.
        let reverse = picture % 2 == 1;
        entries.sort_by_key(|e| {
            let distance = if reverse { -e.poc } else { e.poc };
            (e.candidate.is_none(), e.cycle.unwrap_or(0), -distance)
        });
    }

    // Whether any picture's lists need modifying back.
    let mut modify_lists = edits.lists;
    for (picture, planned) in source.pictures.iter().zip(&pictures) {
        if let (Some((set, entries)), Some(poc)) = (planned, picture.poc) {
            let temp = temp_lists(poc, set, entries);
            for lists in &picture.lists {
                let entries = list_entries(lists, &temp).map_err(|e| format!("poc {poc}: {e}"))?;
                modify_lists |= entries.iter().any(|(_, identity)| !identity);
            }
        }
    }

    // The SPS's short-term sets: with --sets those the pictures use, in
    // order of first use; else the source's, which are copied.
    let mut sets: Vec<(Set, Option<i32>)> = Vec::new();
    if edits.sets.is_some() {
        for (set, _) in pictures.iter().flatten() {
            if sets.len() < 64 && !sets.iter().any(|(s, _)| s == set) {
                let delta = sets.last().and_then(|(before, _)| prediction(set, before));
                sets.push((set.clone(), delta));
            }
        }
    } else {
        sets = sps.sets.iter().map(|s| (s.clone(), None)).collect();
    }
    Ok(Plan {
        sets,
        candidates: if edits.long_term {
            candidates
        } else {
            sps.long_term.clone()
        },
        long_term_present: edits.long_term || sps.long_term_present,
        modify_lists,
        pictures,
    })
}

// ---------------------------------------------------------------------------
// The rewrite.
// ---------------------------------------------------------------------------

/// What was written, for the summary.
#[derive(Default)]
struct Counts {
    headers: usize,
    by_index: usize,
    own_predicted: usize,
    own_outright: usize,
    modified: [usize; 2],
    /// Lists whose modification is not the identity.
    reordered: usize,
    long_term: usize,
    long_term_used: usize,
    long_term_cycles: usize,
    long_term_from_sps: usize,
    long_term_pictures: usize,
}

fn rewrite_sps(
    bytes: &[u8],
    fields: &SpsFields,
    plan: &Plan,
    edits: &Edits,
) -> Result<Vec<u8>, String> {
    let rbsp = unescape(bytes);
    let mut w = Writer::default();
    w.copy(&rbsp, 0, fields.sets_at);
    let sets: Vec<Set> = plan.sets.iter().map(|(s, _)| s.clone()).collect();
    if edits.sets.is_some() {
        w.ue(plan.sets.len() as u32);
        for (i, (set, delta)) in plan.sets.iter().enumerate() {
            let reference = delta.map(|d| (i - 1, d));
            write_set(&mut w, set, &sets[..i], false, reference);
        }
    } else {
        w.copy(&rbsp, fields.sets_at, fields.long_term_at);
    }
    if edits.long_term {
        w.bit(true);
        w.ue(plan.candidates.len() as u32);
        for &(lsb, used) in &plan.candidates {
            w.bits(fields.log2_max_poc_lsb, lsb);
            w.bit(used);
        }
    } else {
        w.copy(&rbsp, fields.long_term_at, fields.temporal_mvp_at);
    }
    w.copy(&rbsp, fields.temporal_mvp_at, stop_bit(&rbsp)?);
    let out = escape(&w.finish());
    Sps::parse(&out).map_err(|e| format!("the rewritten SPS: {e}"))?;
    let reread = sps_fields(&unescape(&out))?;
    if reread.sets != sets || reread.long_term != plan.candidates {
        return Err("the rewritten SPS reads back other sets".into());
    }
    Ok(out)
}

fn rewrite_pps(bytes: &[u8], fields: &PpsFields) -> Result<Vec<u8>, String> {
    let rbsp = unescape(bytes);
    let mut w = Writer::default();
    w.copy(&rbsp, 0, fields.lists_modification_at);
    w.bit(true);
    w.copy(&rbsp, fields.lists_modification_at + 1, stop_bit(&rbsp)?);
    let out = escape(&w.finish());
    Pps::parse(&out).map_err(|e| format!("the rewritten PPS: {e}"))?;
    Ok(out)
}

/// The long-term pictures of a slice segment header (7.3.6.1), the SPS's
/// candidates first.
fn write_long_term(w: &mut Writer, entries: &[Planned], plan: &Plan, log2: u32) {
    let from_sps = entries.iter().filter(|e| e.candidate.is_some()).count();
    if !plan.candidates.is_empty() {
        w.ue(from_sps as u32);
    }
    w.ue((entries.len() - from_sps) as u32);
    // DeltaPocMsbCycleLt of the entry before, within each group (7-52).
    let mut running = 0u32;
    for (i, entry) in entries.iter().enumerate() {
        if i == 0 || i == from_sps {
            running = 0;
        }
        match entry.candidate {
            Some(at) => {
                if plan.candidates.len() > 1 {
                    w.bits(ceil_log2(plan.candidates.len()), at as u32);
                }
            }
            None => {
                w.bits(log2, lsb(entry.poc, log2));
                w.bit(entry.used);
            }
        }
        w.bit(entry.cycle.is_some());
        if let Some(cycle) = entry.cycle {
            w.ue(cycle - running);
            running = cycle;
        }
    }
}

/// A slice segment header written again from `planned`, its set and its
/// long-term pictures, with `lists`, its lists as the source decodes them,
/// kept; then its slice data as it was.
#[allow(clippy::too_many_arguments)]
fn rewrite_slice(
    unit: &Unit,
    fields: &SliceFields,
    poc: i32,
    lists: &[Vec<i32>; 2],
    planned: &(Set, Vec<Planned>),
    plan: &Plan,
    edits: &Edits,
    log2: u32,
    counts: &mut Counts,
) -> Result<Vec<u8>, String> {
    let rbsp = unescape(&unit.bytes);
    let mut w = Writer::default();
    let Some((start, long_term_at, end)) = fields.rps else {
        return Ok(unit.bytes.clone());
    };
    counts.headers += 1;
    let (set, entries) = planned;
    let sets: Vec<Set> = plan.sets.iter().map(|(s, _)| s.clone()).collect();
    w.copy(&rbsp, 0, start);
    let index = sets.iter().position(|s| s == set);
    match (edits.sets, index) {
        (Some(SetsIn::Sps), Some(at)) => {
            w.bit(true);
            if sets.len() > 1 {
                w.bits(ceil_log2(sets.len()), at as u32);
            }
            counts.by_index += 1;
        }
        (Some(SetsIn::Slice), _) | (Some(SetsIn::Sps), None) => {
            w.bit(false);
            let reference = (0..sets.len())
                .rev()
                .find_map(|at| prediction(set, &sets[at]).map(|d| (at, d)))
                .filter(|_| edits.sets == Some(SetsIn::Slice));
            if reference.is_some() {
                counts.own_predicted += 1;
            } else {
                counts.own_outright += 1;
            }
            write_set(&mut w, set, &sets, true, reference);
        }
        (None, _) if *set == fields.set => w.copy(&rbsp, start, long_term_at),
        (None, _) => {
            w.bit(false);
            write_set(&mut w, set, &sets, true, None);
            counts.own_outright += 1;
        }
    }
    if edits.long_term {
        if plan.long_term_present {
            write_long_term(&mut w, entries, plan, log2);
        }
        if !entries.is_empty() {
            counts.long_term_pictures += 1;
        }
        counts.long_term += entries.len();
        counts.long_term_used += entries.iter().filter(|e| e.used).count();
        counts.long_term_cycles += entries.iter().filter(|e| e.cycle.is_some()).count();
        counts.long_term_from_sps += entries.iter().filter(|e| e.candidate.is_some()).count();
    } else {
        w.copy(&rbsp, long_term_at, end);
    }
    if plan.modify_lists && fields.slice_type != 2 {
        w.copy(&rbsp, end, fields.lists.0);
        let total = fields.num_pic_total_curr();
        if total > 1 {
            let entries = list_entries(lists, &temp_lists(poc, set, entries))?;
            // With --lists, both lists, list 0 alone, list 1 alone, in
            // turn; a list whose long-term pictures moved, always.
            let turn = counts.headers;
            let rotation = [turn % 3 != 2, turn % 3 != 1];
            for (list, (entries, identity)) in entries
                .iter()
                .enumerate()
                .take(if fields.slice_type == 0 { 2 } else { 1 })
            {
                let flag = !identity || (edits.lists && rotation[list]);
                w.bit(flag);
                if flag {
                    for &at in entries {
                        w.bits(ceil_log2(total), at);
                    }
                    counts.modified[list] += 1;
                    counts.reordered += usize::from(!identity);
                }
            }
        }
        w.copy(&rbsp, fields.lists.1, fields.alignment);
    } else {
        w.copy(&rbsp, end, fields.alignment);
    }
    let mut out = escape(&w.finish());
    out.extend_from_slice(&unit.bytes[escaped_offset(&unit.bytes, fields.data)..]);
    Ok(out)
}

fn craft(source: &Source, plan: &Plan, edits: &Edits) -> Result<(Vec<Vec<u8>>, Counts), String> {
    let mut counts = Counts::default();
    let mut out = Vec::new();
    let log2 = source.sps.log2_max_poc_lsb;
    for (index, (picture, planned)) in source.pictures.iter().zip(&plan.pictures).enumerate() {
        let at = |e: String| format!("access unit {index}: {e}");
        let mut au = Vec::new();
        // The slice segments before the unit, which index the picture's
        // lists.
        let mut segment = 0;
        for unit in &picture.units {
            let bytes = match (unit.unit_type, &unit.slice, planned, picture.poc) {
                (nal_type::SPS, ..) => {
                    rewrite_sps(&unit.bytes, &source.sps, plan, edits).map_err(at)?
                }
                (nal_type::PPS, ..) if plan.modify_lists => {
                    let (_, fields) = pps_fields(&unescape(&unit.bytes))?;
                    rewrite_pps(&unit.bytes, &fields).map_err(at)?
                }
                (_, Some(fields), Some(planned), Some(poc)) => {
                    let lists = picture
                        .lists
                        .get(segment)
                        .ok_or(at("more slice segments than the decoder saw".into()))?;
                    rewrite_slice(
                        unit,
                        fields,
                        poc,
                        lists,
                        planned,
                        plan,
                        edits,
                        log2,
                        &mut counts,
                    )
                    .map_err(at)?
                }
                _ => unit.bytes.clone(),
            };
            segment += usize::from(matches!(unit.unit_type, 0..=9 | BLA_W_LP..=21));
            au.extend_from_slice(&[0, 0, 0, 1]);
            au.extend_from_slice(&bytes);
        }
        out.push(au);
    }
    Ok((out, counts))
}

// ---------------------------------------------------------------------------
// The check.
// ---------------------------------------------------------------------------

/// Fails unless every picture of `crafted` has `source`'s order count,
/// reference picture set and lists, and they come out in the same order;
/// a reference made long-term may change its set from `StCurrBefore` or
/// `StCurrAfter` to `LtCurr`, and its marking.
fn compare(source: &(Vec<Seen>, Vec<i32>), crafted: &(Vec<Seen>, Vec<i32>)) -> Result<(), String> {
    if source.0.len() != crafted.0.len() {
        return Err(format!(
            "{} pictures decoded, the source {}",
            crafted.0.len(),
            source.0.len()
        ));
    }
    for (s, c) in source.0.iter().zip(&crafted.0) {
        let at = format!("access unit {} (poc {})", s.access_unit, s.poc);
        if s.access_unit != c.access_unit || s.poc != c.poc {
            return Err(format!(
                "{at}: decoded as access unit {} poc {}",
                c.access_unit, c.poc
            ));
        }
        if s.references.len() != c.references.len() {
            return Err(format!(
                "{at}: {} references, the source {}",
                c.references.len(),
                s.references.len()
            ));
        }
        for (&(poc, set, long_term), &(cpoc, cset, clong_term)) in
            s.references.iter().zip(&c.references)
        {
            let same = poc == cpoc
                && if clong_term && !long_term {
                    cset == set
                        || (cset == RpsMembership::LtCurr
                            && matches!(
                                set,
                                RpsMembership::StCurrBefore | RpsMembership::StCurrAfter
                            ))
                } else {
                    cset == set && clong_term == long_term
                };
            if !same {
                return Err(format!(
                    "{at}: reference poc {cpoc} {cset:?} long-term {clong_term}, the source poc {poc} {set:?} long-term {long_term}"
                ));
            }
        }
        if s.lists != c.lists {
            return Err(format!(
                "{at}: lists {:?}, the source {:?}",
                c.lists, s.lists
            ));
        }
    }
    if source.1 != crafted.1 {
        return Err(format!(
            "output order {:?}, the source {:?}",
            crafted.1, source.1
        ));
    }
    Ok(())
}

/// What the crafted stream's slice segment headers hold, as the crate's
/// parser read them.
fn parsed_summary(seen: &[Seen]) -> String {
    let headers: Vec<_> = seen.iter().flat_map(|s| &s.headers).collect();
    let by_index = headers.iter().filter(|h| h.0).count();
    let own = headers.iter().filter(|h| !h.0 && h.1 > 0).count();
    let predicted_from = headers.iter().filter(|h| h.2 > 0).count();
    let long_term: usize = headers.iter().map(|h| h.3).sum();
    let modified = [
        headers.iter().filter(|h| h.4[0]).count(),
        headers.iter().filter(|h| h.4[1]).count(),
    ];
    let long_term_refs = seen
        .iter()
        .flat_map(|s| &s.references)
        .filter(|r| r.2)
        .count();
    let long_term_curr = seen
        .iter()
        .flat_map(|s| &s.references)
        .filter(|r| r.1 == RpsMembership::LtCurr)
        .count();
    format!(
        "{} independent slice segment headers: {by_index} name an SPS set, {own} code their own \
         ({predicted_from} predicted from an SPS set of pictures), {long_term} long-term entries, \
         list 0 modified in {}, list 1 in {}; {long_term_refs} references long-term \
         ({long_term_curr} predicted from)",
        headers.len(),
        modified[0],
        modified[1],
    )
}

fn run(args: &[String]) -> Result<(), String> {
    let [input, output, rest @ ..] = args else {
        return Err(USAGE.into());
    };
    let edits = parse_edits(rest)?;
    let data = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let mut source = read_source(&data)?;
    let units: Vec<Vec<u8>> = nal::access_units(&data)
        .into_iter()
        .map(<[u8]>::to_vec)
        .collect();
    let decoded = decode(&units)?;
    for seen in &decoded.0 {
        if let Some(picture) = source.pictures.get_mut(seen.access_unit) {
            picture.poc = Some(seen.poc);
            picture.lists.clone_from(&seen.lists);
        }
    }
    if edits.long_term && source.sps.long_term_present {
        return Err("the source has long-term pictures already".into());
    }
    if (edits.lists || edits.long_term) && source.pps.values().any(|p| p.lists_modification) {
        return Err("the source modifies its lists already".into());
    }
    let plan = plan(&source, &edits)?;
    let (crafted, counts) = craft(&source, &plan, &edits)?;
    std::fs::write(output, crafted.concat()).map_err(|e| format!("{output}: {e}"))?;
    let predicted = plan.sets.iter().filter(|(_, d)| d.is_some()).count();
    eprintln!(
        "wrote {output}: {} access units; the SPS: {} short-term sets ({predicted} predicted from \
         the one before), long-term {} with {} candidates; {} headers rewritten: {} name an SPS \
         set, {} code their own predicted and {} outright; list 0 modified in {}, list 1 in {}, \
         {} of them reordered; {} long-term entries in {} pictures ({} predicted from, {} with MSB \
         cycles, {} from the SPS)",
        crafted.len(),
        plan.sets.len(),
        plan.long_term_present,
        plan.candidates.len(),
        counts.headers,
        counts.by_index,
        counts.own_predicted,
        counts.own_outright,
        counts.modified[0],
        counts.modified[1],
        counts.reordered,
        counts.long_term,
        counts.long_term_pictures,
        counts.long_term_used,
        counts.long_term_cycles,
        counts.long_term_from_sps,
    );
    if edits.check {
        let checked = decode(&crafted)?;
        compare(&decoded, &checked)?;
        println!(
            "check: {} pictures, each with the source's order count, reference picture set and \
             lists, output in the source's order; {}",
            checked.0.len(),
            parsed_summary(&checked.0)
        );
    }
    Ok(())
}
