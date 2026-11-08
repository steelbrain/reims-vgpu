//! Rewrites an H.264 stream into the cases no encoder at hand writes, for
//! tests of them: long-term references and memory management operations 2
//! to 6, frame number gaps, NAL unit lengths of 1 to 3 bytes, and damage.
//! x264 writes none of the first two: only operation 1 (x264 0.165,
//! 2026-10-24). Ours.
//!
//! ```sh
//! cargo run --example h264-craft -- in.264 out [EDIT...] [--length-size N] [--check]
//! ```
//!
//! Writes `out.264` (Annex B) and `out.mp4` (an `avcC` record from the
//! stream's first SPS and PPS, each sample its access unit's NAL units
//! after `N`-byte lengths, 4 by default, times from the order counts).
//! The edits, each on access unit `F`, counted from 0 in decode order:
//!
//! - `--max-refs N`: every SPS's `max_num_ref_frames` becomes `N`;
//! - `--gaps-allowed`: every SPS's `gaps_in_frame_num_value_allowed_flag`
//!   becomes 1;
//! - `--mark F:lt`: the IDR picture at `F` becomes a long-term reference
//!   (`long_term_reference_flag`);
//! - `--mark F:OPS`: the reference picture at `F` gets the memory
//!   management operations `OPS`, a comma-separated list of
//!   `op[:arg[:arg]]` as 7.3.3.3 has them: `1:diff`, `2:long_term_pic_num`,
//!   `3:diff:idx`, `4:max_idx_plus1`, `5`, `6:idx`. After operation 5 the
//!   pictures that follow, to the next IDR, have their `frame_num` and
//!   `pic_order_cnt_lsb` renumbered from it, as 8.2.1 counts them;
//! - `--gap F:G`: `frame_num` grows by `G` from `F` on, to the next IDR, so
//!   that the decoder infers `G` frames before `F` (8.2.5.2). `F` should be
//!   an I picture, since a P picture there would predict from a frame that
//!   was never sent;
//! - `--no-sei`: SEI units are left out, x264's options among them, so
//!   that a small stream's units fit 1-byte lengths;
//! - `--drop F`: access unit `F` is left out: a lost picture;
//! - `--truncate F:PCT`: each slice of `F` is cut to `PCT` per cent of its
//!   bytes, its header kept;
//! - `--garbage F:KIND`: a NAL unit of 48 pseudo-random bytes goes after
//!   `F`'s slices: `reserved` (type 23), `slice` (a non-reference slice,
//!   type 1, whose header is noise) or `forbidden` (type 1 with
//!   `forbidden_zero_bit` set).
//!
//! The times come from the edited stream before any damage: each sample's
//! presentation time is its rank in display order. `--check` then decodes
//! `out.mp4` as the host session would, through this crate's decoder over a
//! stand-in accelerator that only records, and copies a picture for each
//! lost frame as VA-API's does, and prints for each sample what would
//! reach the GPU, and the error the sample is refused with, if any: a
//! damaged stream is checked so before any GPU sees it.
//!
//! Takes an Annex B stream of frames with one slice group, as x264 writes.
//! Runs on any host: it needs no GPU.

use std::sync::Arc;

use reims_vgpu_video::codec::h264::nal::{AnnexB, NalUnit, access_units, nal_type};
use reims_vgpu_video::codec::h264::pps::Pps;
use reims_vgpu_video::codec::h264::sps::Sps;
use reims_vgpu_video::codec::h264::{
    self, Accelerator, Decoder, NalFormat, OutputOrder, PictureParams, SliceParams,
};
use reims_vgpu_video_conformance::mp4_write;

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("h264-craft: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// The edits.
// ---------------------------------------------------------------------------

/// One memory management operation as written (7.3.3.3).
#[derive(Clone, Copy, Debug)]
enum Mmco {
    UnmarkShortTerm(u32),
    UnmarkLongTerm(u32),
    ShortTermToLongTerm(u32, u32),
    MaxLongTermIdxPlus1(u32),
    UnmarkAll,
    CurrentToLongTerm(u32),
}

#[derive(Clone, Debug)]
enum Marking {
    LongTermIdr,
    Operations(Vec<Mmco>),
}

#[derive(Clone, Copy, Debug)]
enum Garbage {
    Reserved,
    Slice,
    Forbidden,
}

#[derive(Default)]
struct Edits {
    max_refs: Option<u32>,
    gaps_allowed: bool,
    marks: Vec<(usize, Marking)>,
    gaps: Vec<(usize, u32)>,
    drops: Vec<usize>,
    truncations: Vec<(usize, u32)>,
    garbage: Vec<(usize, Garbage)>,
    length_size: u8,
    check: bool,
    no_sei: bool,
}

fn frame_and(spec: &str) -> Result<(usize, &str), String> {
    let (frame, rest) = spec.split_once(':').ok_or(format!("{spec}: not F:..."))?;
    Ok((
        frame.parse().map_err(|_| format!("{spec}: bad frame"))?,
        rest,
    ))
}

fn number(text: &str) -> Result<u32, String> {
    text.parse().map_err(|_| format!("{text}: not a number"))
}

fn parse_marking(spec: &str) -> Result<Marking, String> {
    if spec == "lt" {
        return Ok(Marking::LongTermIdr);
    }
    let mut ops = Vec::new();
    for op in spec.split(',') {
        let parts: Vec<&str> = op.split(':').collect();
        let arg = |i: usize| {
            parts
                .get(i)
                .ok_or(format!("{op}: missing argument"))
                .and_then(|a| number(a))
        };
        ops.push(match parts[0] {
            "1" => Mmco::UnmarkShortTerm(arg(1)?),
            "2" => Mmco::UnmarkLongTerm(arg(1)?),
            "3" => Mmco::ShortTermToLongTerm(arg(1)?, arg(2)?),
            "4" => Mmco::MaxLongTermIdxPlus1(arg(1)?),
            "5" => Mmco::UnmarkAll,
            "6" => Mmco::CurrentToLongTerm(arg(1)?),
            other => return Err(format!("{other}: no such operation")),
        });
    }
    Ok(Marking::Operations(ops))
}

fn parse_edits(args: &[String]) -> Result<Edits, String> {
    let mut edits = Edits {
        length_size: 4,
        ..Edits::default()
    };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--max-refs" => edits.max_refs = Some(number(value()?)?),
            "--gaps-allowed" => edits.gaps_allowed = true,
            "--mark" => {
                let (frame, spec) = frame_and(value()?)?;
                edits.marks.push((frame, parse_marking(spec)?));
            }
            "--gap" => {
                let (frame, gap) = frame_and(value()?)?;
                edits.gaps.push((frame, number(gap)?));
            }
            "--drop" => edits.drops.push(value()?.parse().map_err(|_| "--drop F")?),
            "--truncate" => {
                let (frame, pct) = frame_and(value()?)?;
                edits.truncations.push((frame, number(pct)?.min(100)));
            }
            "--garbage" => {
                let (frame, kind) = frame_and(value()?)?;
                let kind = match kind {
                    "reserved" => Garbage::Reserved,
                    "slice" => Garbage::Slice,
                    "forbidden" => Garbage::Forbidden,
                    other => return Err(format!("{other}: not reserved, slice or forbidden")),
                };
                edits.garbage.push((frame, kind));
            }
            "--length-size" => {
                edits.length_size = u8::try_from(number(value()?)?)
                    .ok()
                    .filter(|n| (1..=4).contains(n))
                    .ok_or("--length-size is 1 to 4")?;
            }
            "--check" => edits.check = true,
            "--no-sei" => edits.no_sei = true,
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

/// A reader of an RBSP, positions in bits from its first byte (the NAL
/// header).
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

    fn into_bytes(self) -> Vec<u8> {
        let mut out = vec![0u8; self.bits.len().div_ceil(8)];
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

// ---------------------------------------------------------------------------
// Where the fields are.
// ---------------------------------------------------------------------------

const HIGH_PROFILES: [u8; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

/// An SPS's `max_num_ref_frames` (its start and end) and the gaps flag.
fn sps_fields(rbsp: &[u8]) -> Result<(usize, usize, usize), String> {
    let mut r = Reader { data: rbsp, pos: 8 };
    let profile_idc = r.bits(8)? as u8;
    r.bits(16)?;
    r.ue()?;
    if HIGH_PROFILES.contains(&profile_idc) {
        let chroma = r.ue()?;
        if chroma == 3 {
            r.bit()?;
        }
        r.ue()?;
        r.ue()?;
        r.bit()?;
        if r.bit()? {
            for i in 0..if chroma == 3 { 12 } else { 8 } {
                if r.bit()? {
                    let size = if i < 6 { 16 } else { 64 };
                    let (mut last, mut next) = (8i32, 8i32);
                    for _ in 0..size {
                        if next != 0 {
                            next = (last + r.se()? + 256) % 256;
                        }
                        last = if next == 0 { last } else { next };
                    }
                }
            }
        }
    }
    r.ue()?; // log2_max_frame_num_minus4
    match r.ue()? {
        0 => {
            r.ue()?;
        }
        1 => {
            r.bit()?;
            r.se()?;
            r.se()?;
            for _ in 0..r.ue()? {
                r.se()?;
            }
        }
        _ => {}
    }
    let refs_start = r.pos;
    r.ue()?;
    let refs_end = r.pos;
    Ok((refs_start, refs_end, r.pos))
}

/// Where a slice header's rewritten fields are, in RBSP bits.
struct SliceFields {
    frame_num: (usize, u32),
    poc_lsb: Option<(usize, u32)>,
    /// `dec_ref_pic_marking()`, start and end.
    marking: Option<(usize, usize)>,
    idr: bool,
    header_end: usize,
    cabac: bool,
}

fn slice_fields(
    rbsp: &[u8],
    unit_type: u8,
    ref_idc: u8,
    pps: &Pps,
    sps: &Sps,
) -> Result<SliceFields, String> {
    let mut r = Reader { data: rbsp, pos: 8 };
    let idr = unit_type == nal_type::IDR;
    r.ue()?; // first_mb_in_slice
    let slice_type = r.ue()? % 5;
    r.ue()?; // pic_parameter_set_id
    if sps.separate_colour_plane_flag {
        r.bits(2)?;
    }
    let frame_num_bits = u32::from(sps.log2_max_frame_num_minus4) + 4;
    let frame_num = (r.pos, frame_num_bits);
    r.bits(frame_num_bits)?;
    let mut field = false;
    if !sps.frame_mbs_only_flag {
        field = r.bit()?;
        if field {
            r.bit()?;
        }
    }
    if idr {
        r.ue()?;
    }
    let mut poc_lsb = None;
    if sps.pic_order_cnt_type == 0 {
        let bits = u32::from(sps.log2_max_pic_order_cnt_lsb_minus4) + 4;
        poc_lsb = Some((r.pos, bits));
        r.bits(bits)?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field {
            r.se()?;
        }
    }
    if sps.pic_order_cnt_type == 1 && !sps.delta_pic_order_always_zero_flag {
        r.se()?;
        if pps.bottom_field_pic_order_in_frame_present_flag && !field {
            r.se()?;
        }
    }
    if pps.redundant_pic_cnt_present_flag {
        r.ue()?;
    }
    let (p, b, i) = (
        slice_type == 0 || slice_type == 3,
        slice_type == 1,
        slice_type == 2 || slice_type == 4,
    );
    if b {
        r.bit()?;
    }
    let mut active = [
        u32::from(pps.num_ref_idx_l0_default_active_minus1),
        u32::from(pps.num_ref_idx_l1_default_active_minus1),
    ];
    if (p || b) && r.bit()? {
        active[0] = r.ue()?;
        if b {
            active[1] = r.ue()?;
        }
    }

    if !i {
        for _ in 0..if b { 2 } else { 1 } {
            if r.bit()? {
                loop {
                    let idc = r.ue()?;
                    if idc == 3 {
                        break;
                    }
                    if idc > 5 {
                        return Err("bad modification_of_pic_nums_idc".into());
                    }
                    r.ue()?;
                }
            }
        }
    }
    if (pps.weighted_pred_flag && p) || (pps.weighted_bipred_idc == 1 && b) {
        r.ue()?;
        let chroma = sps.chroma_array_type() != 0;
        if chroma {
            r.ue()?;
        }
        for &count in active.iter().take(if b { 2 } else { 1 }) {
            for _ in 0..=count {
                if r.bit()? {
                    r.se()?;
                    r.se()?;
                }
                if chroma && r.bit()? {
                    for _ in 0..4 {
                        r.se()?;
                    }
                }
            }
        }
    }
    let mut marking = None;
    if ref_idc != 0 {
        let start = r.pos;
        if idr {
            r.bits(2)?;
        } else if r.bit()? {
            loop {
                match r.ue()? {
                    0 => break,
                    1 | 2 | 4 | 6 => {
                        r.ue()?;
                    }
                    3 => {
                        r.ue()?;
                        r.ue()?;
                    }
                    5 => {}
                    _ => return Err("bad memory_management_control_operation".into()),
                }
            }
        }
        marking = Some((start, r.pos));
    }
    if pps.entropy_coding_mode_flag && !i {
        r.ue()?;
    }
    r.se()?; // slice_qp_delta
    if slice_type == 3 || slice_type == 4 {
        if slice_type == 3 {
            r.bit()?;
        }
        r.se()?;
    }
    if pps.deblocking_filter_control_present_flag && r.ue()? != 1 {
        r.se()?;
        r.se()?;
    }
    if pps.num_slice_groups_minus1 > 0 {
        return Err("slice groups are not crafted".into());
    }
    Ok(SliceFields {
        frame_num,
        poc_lsb,
        marking,
        idr,
        header_end: r.pos,
        cabac: pps.entropy_coding_mode_flag,
    })
}

fn write_marking(w: &mut Writer, marking: &Marking) {
    match marking {
        Marking::LongTermIdr => w.bits(2, 0b01),
        Marking::Operations(ops) => {
            w.bits(1, 1);
            for op in ops {
                match *op {
                    Mmco::UnmarkShortTerm(d) => {
                        w.ue(1);
                        w.ue(d);
                    }
                    Mmco::UnmarkLongTerm(n) => {
                        w.ue(2);
                        w.ue(n);
                    }
                    Mmco::ShortTermToLongTerm(d, idx) => {
                        w.ue(3);
                        w.ue(d);
                        w.ue(idx);
                    }
                    Mmco::MaxLongTermIdxPlus1(n) => {
                        w.ue(4);
                        w.ue(n);
                    }
                    Mmco::UnmarkAll => w.ue(5),
                    Mmco::CurrentToLongTerm(idx) => {
                        w.ue(6);
                        w.ue(idx);
                    }
                }
            }
            w.ue(0);
        }
    }
}

/// A slice's RBSP with its `frame_num` and order count lsb replaced and,
/// with `marking`, its marking rewritten. The slice data is kept: after
/// CABAC's alignment its bytes, after CAVLC's header its bits.
fn rewrite_slice(
    rbsp: &[u8],
    fields: &SliceFields,
    frame_num: u32,
    poc_lsb: Option<u32>,
    marking: Option<&Marking>,
) -> Result<Vec<u8>, String> {
    let mut w = Writer::default();
    let (fn_pos, fn_bits) = fields.frame_num;
    w.copy(rbsp, 0, fn_pos);
    w.bits(fn_bits, frame_num);
    let mut at = fn_pos + fn_bits as usize;
    if let (Some((pos, bits)), Some(lsb)) = (fields.poc_lsb, poc_lsb) {
        w.copy(rbsp, at, pos);
        w.bits(bits, lsb);
        at = pos + bits as usize;
    }
    match (marking, fields.marking) {
        (Some(marking), Some((start, end))) => {
            if matches!(marking, Marking::LongTermIdr) != fields.idr {
                return Err("lt marks an IDR picture, operations a non-IDR one".into());
            }
            w.copy(rbsp, at, start);
            if fields.idr {
                // no_output_of_prior_pics_flag as it was.
                w.bits(1, u32::from(get_bit(rbsp, start)));
                w.bits(1, 1);
            } else {
                write_marking(&mut w, marking);
            }
            at = end;
        }
        (Some(_), None) => return Err("a mark on a non-reference picture".into()),
        _ => {}
    }
    w.copy(rbsp, at, fields.header_end);
    if fields.cabac {
        while w.bits.len() % 8 != 0 {
            w.bits.push(true);
        }
        let mut out = w.into_bytes();
        out.extend_from_slice(&rbsp[fields.header_end.div_ceil(8)..]);
        Ok(out)
    } else {
        w.copy(rbsp, fields.header_end, stop_bit(rbsp)?);
        w.bits.push(true);
        while w.bits.len() % 8 != 0 {
            w.bits.push(false);
        }
        Ok(w.into_bytes())
    }
}

// ---------------------------------------------------------------------------
// The stream.
// ---------------------------------------------------------------------------

/// One access unit: its NAL units, escaped, header byte first.
type Unit = Vec<Vec<u8>>;

fn craft(data: &[u8], edits: &Edits) -> Result<Vec<Unit>, String> {
    let mut sps: Vec<Option<Arc<Sps>>> = vec![None; 32];
    let mut pps: Vec<Option<Arc<Pps>>> = vec![None; 256];
    let mut out = Vec::new();
    // What is added to frame_num and pic_order_cnt_lsb, wrapping, to the
    // next IDR picture: a gap's frames, or after operation 5 the minus of
    // that picture's values, so that the next picture counts from 0
    // (8.2.1). MaxFrameNum and MaxPicOrderCntLsb divide 2^32.
    let (mut frame_num_add, mut lsb_add) = (0u32, 0u32);
    for (index, au) in access_units(data).into_iter().enumerate() {
        let is_idr =
            AnnexB::new(au).any(|unit| unit.first().is_some_and(|b| b & 0x1f == nal_type::IDR));
        if is_idr {
            (frame_num_add, lsb_add) = (0, 0);
        }
        if let Some(&(_, gap)) = edits.gaps.iter().find(|(f, _)| *f == index) {
            frame_num_add = frame_num_add.wrapping_add(gap);
        }
        let marking = edits
            .marks
            .iter()
            .find(|(f, _)| *f == index)
            .map(|(_, m)| m);
        let unmarks_all = matches!(marking, Some(Marking::Operations(ops))
            if ops.iter().any(|op| matches!(op, Mmco::UnmarkAll)));
        let mut picture = None;
        let mut units = Vec::new();
        for unit in AnnexB::new(au) {
            let nal = NalUnit::new(unit).map_err(|e| format!("access unit {index}: {e}"))?;
            match nal.unit_type() {
                nal_type::SPS => {
                    let parsed = Sps::parse(unit).map_err(|e| e.to_string())?;
                    let rbsp = unescape(unit);
                    let (start, end, gaps) = sps_fields(&rbsp)?;
                    let mut w = Writer::default();
                    w.copy(&rbsp, 0, start);
                    match edits.max_refs {
                        Some(n) => w.ue(n),
                        None => w.copy(&rbsp, start, end),
                    }
                    w.bits.push(edits.gaps_allowed || get_bit(&rbsp, gaps));
                    w.copy(&rbsp, gaps + 1, stop_bit(&rbsp)?);
                    w.bits.push(true);
                    while w.bits.len() % 8 != 0 {
                        w.bits.push(false);
                    }
                    let rewritten = escape(&w.into_bytes());
                    let reparsed =
                        Sps::parse(&rewritten).map_err(|e| format!("rewritten SPS: {e}"))?;
                    let id = usize::from(parsed.seq_parameter_set_id);
                    sps[id] = Some(Arc::new(reparsed));
                    units.push(rewritten);
                }
                nal_type::PPS => {
                    let parsed = Pps::parse(unit, |id| sps[usize::from(id)].clone())
                        .map_err(|e| e.to_string())?;
                    let id = usize::from(parsed.pic_parameter_set_id);
                    pps[id] = Some(Arc::new(parsed));
                    units.push(unit.to_vec());
                }
                nal_type::SLICE | nal_type::IDR => {
                    let rbsp = unescape(unit);
                    let mut r = Reader {
                        data: &rbsp,
                        pos: 8,
                    };
                    r.ue()?;
                    r.ue()?;
                    let pps_id = r.ue()?;
                    let p = pps
                        .get(pps_id as usize)
                        .cloned()
                        .flatten()
                        .ok_or(format!("access unit {index}: PPS {pps_id} missing"))?;
                    let s = sps[usize::from(p.seq_parameter_set_id)]
                        .clone()
                        .ok_or("SPS missing")?;
                    let fields = slice_fields(&rbsp, nal.unit_type(), nal.ref_idc(), &p, &s)?;
                    let (fn_pos, fn_bits) = fields.frame_num;
                    let old_frame_num = Reader {
                        data: &rbsp,
                        pos: fn_pos,
                    }
                    .bits(fn_bits)?;
                    let old_lsb = match fields.poc_lsb {
                        Some((pos, bits)) => Some(Reader { data: &rbsp, pos }.bits(bits)?),
                        None => None,
                    };
                    picture = Some((old_frame_num, old_lsb));
                    let frame_num = old_frame_num.wrapping_add(frame_num_add) % s.max_frame_num();
                    let lsb =
                        old_lsb.map(|lsb| lsb.wrapping_add(lsb_add) % s.max_pic_order_cnt_lsb());
                    let rewritten = rewrite_slice(&rbsp, &fields, frame_num, lsb, marking)?;
                    units.push(escape(&rewritten));
                }
                nal_type::SEI if edits.no_sei => {}
                _ => units.push(unit.to_vec()),
            }
        }
        if unmarks_all && let Some((old_frame_num, old_lsb)) = picture {
            frame_num_add = old_frame_num.wrapping_neg();
            lsb_add = old_lsb.unwrap_or(0).wrapping_neg();
        }
        out.push(units);
    }
    Ok(out)
}

/// A stand-in accelerator that records what each picture would give the
/// GPU.
#[derive(Default)]
struct Recorder {
    next: usize,
    /// The last picture's, taken by the check after each sample.
    last: std::cell::RefCell<Option<String>>,
    /// The copies made for lost frames since the check last looked, as
    /// VA-API's accelerator makes them: each copy's source.
    concealed: std::cell::RefCell<Vec<usize>>,
    /// The pictures decoded since the check last looked.
    pictures: std::cell::Cell<usize>,
}

impl Accelerator for Recorder {
    type Picture = usize;
    type Error = h264::Error;

    fn configure(&mut self, _sps: &Sps, _dpb_frames: usize) -> Result<(), h264::Error> {
        Ok(())
    }

    fn new_picture(&mut self) -> Result<usize, h264::Error> {
        self.next += 1;
        Ok(self.next)
    }

    fn decode_picture(
        &mut self,
        _picture: &usize,
        params: &PictureParams<'_, usize>,
        slices: &[SliceParams<'_, usize>],
    ) -> Result<(), h264::Error> {
        self.pictures.set(self.pictures.get() + 1);
        let long_term = params.references.iter().filter(|r| r.long_term).count();
        let bytes: usize = slices.iter().map(|s| s.nal.len()).sum();
        let header_bytes: usize = slices
            .iter()
            .map(|s| s.header.header_bits.div_ceil(8))
            .sum();
        *self.last.borrow_mut() = Some(format!(
            "{:?}{} {} slices, {bytes} bytes ({header_bytes} of headers), frame_num {}, poc {}/{}, {} references ({long_term} long-term), lists {:?}",
            params.header.slice_type,
            if params.header.nal_ref_idc == 0 {
                ""
            } else {
                " ref"
            },
            slices.len(),
            params.header.frame_num,
            params.top_poc,
            params.bottom_poc,
            params.references.len(),
            slices
                .iter()
                .map(|s| [s.ref_lists[0].len(), s.ref_lists[1].len()])
                .collect::<Vec<_>>(),
        ));
        Ok(())
    }

    fn conceal(&mut self, from: &usize) -> Result<Option<usize>, h264::Error> {
        self.concealed.borrow_mut().push(*from);
        self.new_picture().map(Some)
    }
}

impl Recorder {
    /// What reached the accelerator for the last sample, and the copies
    /// made before it.
    fn take(&self) -> String {
        let mut what = self.last.take().unwrap_or_else(|| "nothing".into());
        let pictures = self.pictures.replace(0);
        if pictures > 1 {
            what = format!("{pictures} pictures, the last {what}");
        }
        let copies = std::mem::take(&mut *self.concealed.borrow_mut());
        if copies.is_empty() {
            what
        } else {
            format!("copies of pictures {copies:?} for lost frames, then {what}")
        }
    }
}

fn annex_b(units: &[Unit]) -> Vec<u8> {
    let mut out = Vec::new();
    for unit in units.iter().flatten() {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(unit);
    }
    out
}

/// Each access unit's rank in display order, through the decoder's
/// bumping process.
fn display_ranks(units: &[Unit]) -> Result<Vec<usize>, String> {
    let mut decoder = Decoder::new(Recorder::default(), NalFormat::AnnexB);
    decoder.set_output_order(OutputOrder::Display);
    let mut order = Vec::new();
    for (index, unit) in units.iter().enumerate() {
        let id = u64::try_from(index).map_err(|e| e.to_string())?;
        let frames = decoder
            .decode(&annex_b(std::slice::from_ref(unit)), id)
            .map_err(|e| format!("access unit {index} does not decode: {e}"))?;
        order.extend(frames.iter().map(|f| f.id));
    }
    order.extend(decoder.flush().iter().map(|f| f.id));
    let mut ranks = vec![0; units.len()];
    for (rank, id) in order.into_iter().enumerate() {
        if let Some(slot) = usize::try_from(id).ok().and_then(|i| ranks.get_mut(i)) {
            *slot = rank;
        }
    }
    Ok(ranks)
}

/// 48 bytes from a fixed seed: the same noise every run.
fn noise(seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..48)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn damage(units: Vec<Unit>, edits: &Edits) -> Vec<Unit> {
    let mut out = Vec::new();
    for (index, mut unit) in units.into_iter().enumerate() {
        if edits.drops.contains(&index) {
            continue;
        }
        if let Some(&(_, pct)) = edits.truncations.iter().find(|(f, _)| *f == index) {
            for nal in &mut unit {
                if matches!(nal[0] & 0x1f, nal_type::SLICE | nal_type::IDR) {
                    let keep = (nal.len() * pct as usize / 100).max(48).min(nal.len());
                    nal.truncate(keep);
                }
            }
        }
        if let Some(&(_, kind)) = edits.garbage.iter().find(|(f, _)| *f == index) {
            let mut nal = noise(index as u64 + 0x9e37_79b9);
            nal[0] = match kind {
                Garbage::Reserved => 23,
                Garbage::Slice => 1,
                Garbage::Forbidden => 0x80 | 1,
            };
            // No start code inside: the noise has no two zeros in a row.
            for byte in nal.iter_mut().skip(1) {
                if *byte == 0 {
                    *byte = 0x55;
                }
            }
            unit.push(nal);
        }
        out.push(unit);
    }
    out
}

fn avcc(sps: &[u8], pps: &[u8], length_size: u8) -> Result<Vec<u8>, String> {
    let parsed = Sps::parse(sps).map_err(|e| e.to_string())?;
    let mut out = vec![1, sps[1], sps[2], sps[3], 0xfc | (length_size - 1), 0xe1];
    out.extend_from_slice(
        &u16::try_from(sps.len())
            .map_err(|e| e.to_string())?
            .to_be_bytes(),
    );
    out.extend_from_slice(sps);
    out.push(1);
    out.extend_from_slice(
        &u16::try_from(pps.len())
            .map_err(|e| e.to_string())?
            .to_be_bytes(),
    );
    out.extend_from_slice(pps);
    if matches!(parsed.profile_idc, 100 | 110 | 122 | 144) {
        out.push(0xfc | parsed.chroma_format_idc);
        out.push(0xf8 | parsed.bit_depth_luma_minus8);
        out.push(0xf8 | parsed.bit_depth_chroma_minus8);
        out.push(0);
    }
    Ok(out)
}

fn run(args: &[String]) -> Result<(), String> {
    let [input, output, rest @ ..] = args else {
        return Err(
            "usage: h264-craft <in.264> <out> [EDIT...] [--length-size N] [--check]".into(),
        );
    };
    let edits = parse_edits(rest)?;
    let data = std::fs::read(input).map_err(|e| format!("{input}: {e}"))?;
    let crafted = craft(&data, &edits)?;
    let ranks = display_ranks(&crafted)?;
    let delay = ranks
        .iter()
        .enumerate()
        .map(|(i, &rank)| i.saturating_sub(rank))
        .max()
        .unwrap_or(0);
    let idr: Vec<bool> = crafted
        .iter()
        .map(|unit| unit.iter().any(|nal| nal[0] & 0x1f == nal_type::IDR))
        .collect();
    let timed: Vec<(Unit, usize, usize, bool)> = crafted
        .into_iter()
        .enumerate()
        .map(|(i, unit)| (unit, i, ranks[i] + delay, idr[i]))
        .collect();
    let kept: Vec<usize> = (0..timed.len())
        .filter(|i| !edits.drops.contains(i))
        .collect();
    let units: Vec<Unit> = timed.iter().map(|t| t.0.clone()).collect();
    let damaged = damage(units, &edits);

    let first = |kind: u8| {
        damaged
            .iter()
            .flatten()
            .find(|nal| nal[0] & 0x1f == kind)
            .cloned()
            .ok_or(format!("no NAL unit of type {kind}"))
    };
    let (sps, pps) = (first(nal_type::SPS)?, first(nal_type::PPS)?);
    let record = avcc(&sps, &pps, edits.length_size)?;
    let parsed = Sps::parse(&sps).map_err(|e| e.to_string())?;
    let (_, _, width, height) = parsed.crop_rect();
    let max_length = (1u64 << (8 * u32::from(edits.length_size))) - 1;
    let mut samples = Vec::new();
    for (unit, &original) in damaged.iter().zip(&kept) {
        let mut sample = Vec::new();
        for nal in unit {
            let length = u64::try_from(nal.len()).map_err(|e| e.to_string())?;
            if length > max_length {
                return Err(format!(
                    "a {length}-byte NAL unit does not fit {} bytes of length",
                    edits.length_size
                ));
            }
            sample.extend_from_slice(&length.to_be_bytes()[8 - usize::from(edits.length_size)..]);
            sample.extend_from_slice(nal);
        }
        let (_, dts, pts, sync) = &timed[original];
        samples.push(mp4_write::Sample {
            data: sample,
            pts: i64::try_from(*pts).map_err(|e| e.to_string())?,
            dts: i64::try_from(*dts).map_err(|e| e.to_string())?,
            duration: 1,
            sync: *sync,
        });
    }
    let mp4 = mp4_write::write(b"avc1", b"avcC", &record, width, height, 30, &samples);
    std::fs::write(format!("{output}.264"), annex_b(&damaged))
        .map_err(|e| format!("{output}.264: {e}"))?;
    std::fs::write(format!("{output}.mp4"), &mp4).map_err(|e| format!("{output}.mp4: {e}"))?;
    eprintln!(
        "wrote {output}.264 and {output}.mp4: {} samples of {}x{}, {}-byte lengths",
        samples.len(),
        width,
        height,
        edits.length_size
    );

    if edits.check {
        // As the host session decodes: the avcC's sets, then each sample
        // in decode order.
        let mut decoder =
            Decoder::from_avcc(Recorder::default(), &record).map_err(|e| e.to_string())?;
        for (i, sample) in samples.iter().enumerate() {
            let id = u64::try_from(i).map_err(|e| e.to_string())?;
            match decoder.decode(&sample.data, id) {
                Ok(frames) => {
                    let what = decoder.accelerator().take();
                    println!(
                        "sample {i} (access unit {}): {} frame(s) out; to the accelerator: {what}",
                        kept[i],
                        frames.len()
                    );
                }
                Err(e) => {
                    let what = decoder.accelerator().take();
                    println!(
                        "sample {i} (access unit {}): refused: {e}; to the accelerator: {what}",
                        kept[i]
                    );
                }
            }
        }
    }
    Ok(())
}
