//! The decoding process end to end, on streams written here from the
//! standard's syntax and an accelerator that records what it is asked to
//! decode. The slice data is a placeholder: nothing here decodes samples.
//! What x264 does not write (order count type 1 with non-reference
//! pictures, long-term references, operation 5, frame number gaps) is
//! covered here; the VA-API crate's tools/h264-check.sh covers the rest on
//! hardware.

use std::sync::Arc;

use super::*;

/// Writes syntax elements, then an escaped NAL unit.
#[derive(Default)]
struct Writer {
    bits: Vec<bool>,
}

impl Writer {
    fn u(&mut self, n: u32, value: u32) -> &mut Self {
        for i in (0..n).rev() {
            self.bits.push((value >> i) & 1 == 1);
        }
        self
    }

    fn flag(&mut self, value: bool) -> &mut Self {
        self.bits.push(value);
        self
    }

    fn ue(&mut self, value: u32) -> &mut Self {
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

    fn se(&mut self, value: i32) -> &mut Self {
        let code = if value > 0 {
            2 * value.unsigned_abs() - 1
        } else {
            2 * value.unsigned_abs()
        };
        self.ue(code)
    }

    /// The NAL unit: header byte, the bits, `rbsp_trailing_bits()`, with
    /// emulation prevention bytes inserted.
    fn nal(&mut self, header: u8) -> Vec<u8> {
        self.bits.push(true);
        while !self.bits.len().is_multiple_of(8) {
            self.bits.push(false);
        }
        let rbsp: Vec<u8> = self
            .bits
            .chunks(8)
            .map(|b| b.iter().fold(0u8, |acc, &bit| (acc << 1) | u8::from(bit)))
            .collect();
        let mut out = vec![header];
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

#[derive(Clone)]
struct SpsSpec {
    profile_idc: u8,
    poc_type: u8,
    log2_max_frame_num_minus4: u32,
    /// Type 1: `offset_for_non_ref_pic` and `offset_for_ref_frame[]`, with
    /// `delta_pic_order_always_zero_flag` set.
    type1: (i32, Vec<i32>),
    max_num_ref_frames: u32,
    gaps: bool,
    /// `frame_mbs_only_flag` 0, with `mb_adaptive_frame_field_flag` as
    /// given; `None` is progressive.
    fields: Option<bool>,
    /// The VUI's `max_num_reorder_frames` and `max_dec_frame_buffering`.
    restriction: Option<(u32, u32)>,
    /// The VUI's `video_full_range_flag`, in a video signal type with
    /// `video_format` 5 (unspecified) and no colour description; written
    /// only with a restriction, which makes the VUI.
    full_range: bool,
}

impl Default for SpsSpec {
    fn default() -> Self {
        Self {
            profile_idc: 77,
            poc_type: 0,
            log2_max_frame_num_minus4: 0,
            type1: (0, vec![2]),
            max_num_ref_frames: 4,
            gaps: false,
            fields: None,
            restriction: Some((2, 4)),
            full_range: false,
        }
    }
}

fn sps(spec: &SpsSpec) -> Vec<u8> {
    let mut w = Writer::default();
    w.u(8, u32::from(spec.profile_idc)).u(8, 0).u(8, 40).ue(0);
    if spec.profile_idc == 100 {
        w.ue(1).ue(0).ue(0).flag(false).flag(false);
    }
    w.ue(spec.log2_max_frame_num_minus4)
        .ue(u32::from(spec.poc_type));
    match spec.poc_type {
        0 => {
            w.ue(4); // log2_max_pic_order_cnt_lsb_minus4: MaxPicOrderCntLsb 256
        }
        1 => {
            w.flag(true).se(spec.type1.0).se(0);
            w.ue(u32::try_from(spec.type1.1.len()).unwrap_or(0));
            for &offset in &spec.type1.1 {
                w.se(offset);
            }
        }
        _ => {}
    }
    // 1920x1080: 120 by 68 macroblocks, cropped by 8 rows (4 crop units).
    w.ue(spec.max_num_ref_frames).flag(spec.gaps).ue(119).ue(67);
    w.flag(spec.fields.is_none()); // frame_mbs_only_flag
    if let Some(mbaff) = spec.fields {
        w.flag(mbaff);
    }
    w.flag(true); // direct_8x8_inference_flag
    w.flag(true).ue(0).ue(0).ue(0).ue(4); // frame_cropping_flag and offsets
    match spec.restriction {
        Some((reorder, buffering)) => {
            w.flag(true); // vui_parameters_present_flag
            w.flag(false).flag(false); // aspect ratio, overscan: absent
            w.flag(spec.full_range);
            if spec.full_range {
                // video_format, video_full_range_flag, no colour description
                w.u(3, 5).flag(true).flag(false);
            }
            for _ in 0..5 {
                // chroma location, timing, NAL HRD, VCL HRD, pic_struct:
                // all absent.
                w.flag(false);
            }
            w.flag(true).flag(true).ue(0).ue(0).ue(16).ue(16);
            w.ue(reorder).ue(buffering);
        }
        None => {
            w.flag(false);
        }
    }
    w.nal(0x67)
}

fn pps(transform_8x8_with_lists: bool) -> Vec<u8> {
    pps_with(transform_8x8_with_lists, false)
}

/// A PPS, with `bottom_field_pic_order_in_frame_present_flag` as given.
fn pps_with(transform_8x8_with_lists: bool, bottom_field_pic_order: bool) -> Vec<u8> {
    let mut w = Writer::default();
    w.ue(0)
        .ue(0)
        .flag(false)
        .flag(bottom_field_pic_order)
        .ue(0)
        .ue(0)
        .ue(0);
    w.flag(false).u(2, 0).se(0).se(0).se(0);
    w.flag(true).flag(false).flag(false);
    if transform_8x8_with_lists {
        // transform_8x8_mode_flag, pic_scaling_matrix_present_flag, and
        // eight lists left out.
        w.flag(true).flag(true);
        for _ in 0..8 {
            w.flag(false);
        }
        w.se(0);
    }
    w.nal(0x68)
}

#[derive(Clone, Default)]
struct SliceSpec {
    idr: bool,
    reference: bool,
    /// 0 P, 1 B, 2 I.
    slice_type: u32,
    frame_num: u32,
    poc_lsb: u32,
    /// `num_ref_idx_l0_active_minus1` and `_l1_`, overriding the PPS's 0.
    active: Option<(u32, u32)>,
    /// `(modification_of_pic_nums_idc, value)` for list 0.
    modifications: Vec<(u32, u32)>,
    /// `(operation, first, second)`, for `adaptive_ref_pic_marking`.
    mmco: Vec<(u32, u32, u32)>,
    long_term_idr: bool,
    first_mb: u32,
    /// `delta_pic_order_cnt_bottom`, for a PPS that codes it.
    delta_bottom: Option<i32>,
    /// A B slice's `direct_spatial_mv_pred_flag` cleared.
    temporal_direct: bool,
    slice_qp_delta: i32,
    /// No placeholder for `slice_data()`.
    no_data: bool,
}

fn slice(spec: &SliceSpec, sps: &SpsSpec) -> Vec<u8> {
    let mut w = Writer::default();
    w.ue(spec.first_mb).ue(spec.slice_type).ue(0);
    w.u(sps.log2_max_frame_num_minus4 + 4, spec.frame_num);
    if sps.fields.is_some() {
        w.flag(false); // field_pic_flag: a frame
    }
    if spec.idr {
        w.ue(0);
    }
    if sps.poc_type == 0 {
        w.u(8, spec.poc_lsb);
        if let Some(delta) = spec.delta_bottom {
            w.se(delta);
        }
    }
    if spec.slice_type == 1 {
        w.flag(!spec.temporal_direct); // direct_spatial_mv_pred_flag
    }
    if spec.slice_type != 2 {
        match spec.active {
            Some((l0, l1)) => {
                w.flag(true).ue(l0);
                if spec.slice_type == 1 {
                    w.ue(l1);
                }
            }
            None => {
                w.flag(false);
            }
        }
        w.flag(!spec.modifications.is_empty());
        if !spec.modifications.is_empty() {
            for &(idc, value) in &spec.modifications {
                w.ue(idc).ue(value);
            }
            w.ue(3);
        }
        if spec.slice_type == 1 {
            w.flag(false);
        }
    }
    if spec.reference {
        if spec.idr {
            w.flag(false).flag(spec.long_term_idr);
        } else if spec.mmco.is_empty() {
            w.flag(false);
        } else {
            w.flag(true);
            for &(op, a, b) in &spec.mmco {
                w.ue(op);
                match op {
                    1 | 2 | 4 | 6 => {
                        w.ue(a);
                    }
                    3 => {
                        w.ue(a).ue(b);
                    }
                    _ => {}
                }
            }
            w.ue(0);
        }
    }
    w.se(spec.slice_qp_delta);
    w.ue(1); // disable_deblocking_filter_idc
    if !spec.no_data {
        // A placeholder for slice_data().
        w.u(8, 0xa5);
    }
    let ref_idc = if spec.reference { 3 << 5 } else { 0 };
    w.nal(ref_idc | if spec.idr { 5 } else { 1 })
}

/// What one decode_picture call was given.
#[derive(Debug)]
struct Submitted {
    picture: u32,
    pocs: (i32, i32),
    /// The DPB's references: picture, frame_idx, long-term.
    references: Vec<(u32, u32, bool)>,
    /// List 0 and list 1 of the first slice, by picture.
    lists: [Vec<Option<u32>>; 2],
}

#[derive(Default)]
struct Recorder {
    next: u32,
    configured: usize,
    submitted: Vec<Submitted>,
    /// Whether it copies pictures to stand in for lost frames.
    conceals: bool,
    /// Each copy: the picture copied, and the copy.
    concealed: Vec<(u32, u32)>,
}

impl Accelerator for Recorder {
    type Picture = u32;
    type Error = Error;

    fn configure(&mut self, _sps: &sps::Sps, _dpb_frames: usize) -> Result<(), Error> {
        self.configured += 1;
        Ok(())
    }

    fn new_picture(&mut self) -> Result<u32, Error> {
        self.next += 1;
        Ok(self.next - 1)
    }

    fn decode_picture(
        &mut self,
        picture: &u32,
        params: &PictureParams<'_, u32>,
        slices: &[SliceParams<'_, u32>],
    ) -> Result<(), Error> {
        let list = |l: &Vec<Option<ReferenceFrame<'_, u32>>>| -> Vec<Option<u32>> {
            l.iter().map(|r| r.as_ref().map(|r| *r.picture)).collect()
        };
        self.submitted.push(Submitted {
            picture: *picture,
            pocs: (params.top_poc, params.bottom_poc),
            references: params
                .references
                .iter()
                .map(|r| (*r.picture, r.frame_idx, r.long_term))
                .collect(),
            lists: [list(&slices[0].ref_lists[0]), list(&slices[0].ref_lists[1])],
        });
        Ok(())
    }

    fn conceal(&mut self, from: &u32) -> Result<Option<u32>, Error> {
        if !self.conceals {
            return Ok(None);
        }
        let copy = self.new_picture()?;
        self.concealed.push((*from, copy));
        Ok(Some(copy))
    }
}

/// An Annex B sample of the given NAL units.
fn sample(units: &[Vec<u8>]) -> Vec<u8> {
    units
        .iter()
        .flat_map(|u| [0, 0, 0, 1].into_iter().chain(u.iter().copied()))
        .collect()
}

fn decoder(order: OutputOrder) -> Decoder<Recorder> {
    let mut d = Decoder::new(Recorder::default(), NalFormat::AnnexB);
    d.set_output_order(order);
    d
}

/// Decodes each slice as one sample, the first with the parameter sets,
/// and returns the order counts of what came out, flush included.
fn run(
    d: &mut Decoder<Recorder>,
    spec: &SpsSpec,
    slices: &[SliceSpec],
) -> Result<Vec<(u64, i32)>, Error> {
    let mut out = Vec::new();
    for (i, s) in slices.iter().enumerate() {
        let mut units = Vec::new();
        if i == 0 {
            units.push(sps(spec));
            units.push(pps(false));
        }
        units.push(slice(s, spec));
        let id = u64::try_from(i).unwrap_or(0);
        out.extend(d.decode(&sample(&units), id)?.iter().map(|o| (o.id, o.poc)));
    }
    out.extend(d.flush().iter().map(|o| (o.id, o.poc)));
    Ok(out)
}

fn idr() -> SliceSpec {
    SliceSpec {
        idr: true,
        reference: true,
        slice_type: 2,
        ..SliceSpec::default()
    }
}

fn p(frame_num: u32, poc_lsb: u32) -> SliceSpec {
    SliceSpec {
        reference: true,
        slice_type: 0,
        frame_num,
        poc_lsb,
        ..SliceSpec::default()
    }
}

fn b(frame_num: u32, poc_lsb: u32, reference: bool) -> SliceSpec {
    SliceSpec {
        reference,
        slice_type: 1,
        frame_num,
        poc_lsb,
        ..SliceSpec::default()
    }
}

#[test]
fn b_frames_come_out_in_display_order_and_predict_from_both_sides() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Display);
    // I0 P6 B2 B4, as an encoder with two B-frames sends them.
    let out = run(
        &mut d,
        &spec,
        &[idr(), p(1, 6), b(2, 2, false), b(2, 4, false)],
    )?;
    assert_eq!(out, [(0, 0), (2, 2), (3, 4), (1, 6)]);
    let rec = &d.accelerator().submitted;
    // B2, the third picture: list 0 is I0, list 1 is P6.
    assert_eq!(rec[2].picture, 2);
    assert_eq!(rec[2].lists, [vec![Some(0)], vec![Some(1)]]);
    Ok(())
}

#[test]
fn decode_order_emits_each_picture_as_it_is_decoded() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let out = run(
        &mut d,
        &spec,
        &[idr(), p(1, 6), b(2, 2, false), b(2, 4, false)],
    )?;
    assert_eq!(out, [(0, 0), (1, 6), (2, 2), (3, 4)]);
    Ok(())
}

#[test]
fn display_order_waits_only_for_max_num_reorder_frames() -> Result<(), Error> {
    let spec = SpsSpec {
        restriction: Some((0, 4)),
        ..SpsSpec::default()
    };
    let mut d = decoder(OutputOrder::Display);
    let mut emitted = Vec::new();
    let s = sps(&spec);
    let pp = pps(false);
    let first = d.decode(&sample(&[s, pp, slice(&idr(), &spec)]), 0)?;
    emitted.push(first.len());
    let second = d.decode(&sample(&[slice(&p(1, 2), &spec)]), 1)?;
    emitted.push(second.len());
    assert_eq!(emitted, [1, 1]);
    Ok(())
}

#[test]
fn order_count_type_1_follows_its_cycle_and_non_reference_offset() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 1,
        type1: (-1, vec![2, 4]),
        ..SpsSpec::default()
    };
    let mut d = decoder(OutputOrder::Decode);
    let nonref = SliceSpec {
        reference: false,
        ..p(3, 0)
    };
    let out = run(&mut d, &spec, &[idr(), p(1, 0), p(2, 0), nonref, p(3, 0)])?;
    // absFrameNum 1, 2: 2, 2+4. The non-reference picture's is 3 - 1 = 2,
    // so 6 - 1. Then absFrameNum 3: one cycle (6) and 2.
    let pocs: Vec<i32> = out.iter().map(|&(_, poc)| poc).collect();
    assert_eq!(pocs, [0, 2, 6, 5, 8]);
    Ok(())
}

#[test]
fn order_count_type_2_continues_across_frame_num_wrap() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        max_num_ref_frames: 2,
        ..SpsSpec::default()
    };
    let mut slices = vec![idr()];
    // MaxFrameNum is 16: frame_num runs 1..15, 0..3.
    for n in 1..20u32 {
        slices.push(p(n % 16, 0));
    }
    let mut d = decoder(OutputOrder::Decode);
    let out = run(&mut d, &spec, &slices)?;
    let pocs: Vec<i32> = out.iter().map(|&(_, poc)| poc).collect();
    let expected: Vec<i32> = (0..20).map(|n| 2 * n).collect();
    assert_eq!(pocs, expected);
    // Just after the wrap, frame_num 1 predicts from frame_num 0 (picture
    // 16, FrameNumWrap 0) before frame_num 15 (picture 15, FrameNumWrap -1).
    assert_eq!(d.accelerator().submitted[17].lists[0], [Some(16)]);
    Ok(())
}

#[test]
fn operation_5_restarts_order_counts_and_bumps_what_came_before() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        ..SpsSpec::default()
    };
    let reset = SliceSpec {
        mmco: vec![(5, 0, 0)],
        ..p(2, 0)
    };
    let mut d = decoder(OutputOrder::Display);
    let out = run(&mut d, &spec, &[idr(), p(1, 0), reset, p(1, 0)])?;
    // The reset picture is decoded with count 4, then becomes 0; the next
    // picture's frame_num 1 follows it as if it were 0.
    assert_eq!(d.accelerator().submitted[2].pocs, (4, 4));
    assert_eq!(out, [(0, 0), (1, 2), (2, 0), (3, 2)]);
    // Only the reset picture is left to predict from.
    assert_eq!(d.accelerator().submitted[3].references, [(2, 0, false)]);
    Ok(())
}

#[test]
fn long_term_references_follow_short_terms_and_carry_their_index() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        max_num_ref_frames: 3,
        ..SpsSpec::default()
    };
    let to_long_term = SliceSpec {
        // max_long_term_frame_idx_plus1 1, then this picture long-term 0.
        mmco: vec![(4, 1, 0), (6, 0, 0)],
        ..p(1, 0)
    };
    let mut d = decoder(OutputOrder::Decode);
    let three = SliceSpec {
        active: Some((2, 0)),
        ..p(3, 0)
    };
    run(&mut d, &spec, &[idr(), to_long_term, p(2, 0), three])?;
    let last = &d.accelerator().submitted[3];
    // Short-term by descending PicNum (2, then the IDR's 0), long-term last.
    assert_eq!(last.lists[0], [Some(2), Some(0), Some(1)]);
    assert!(last.references.contains(&(1, 0, true)));
    Ok(())
}

#[test]
fn operations_1_to_3_unmark_and_convert_the_frames_they_name() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        max_num_ref_frames: 4,
        ..SpsSpec::default()
    };
    // Long-term indices up to 0, then the IDR long-term 0: CurrPicNum 2,
    // so the IDR's PicNum 0 is difference_of_pic_nums_minus1 1.
    let convert = SliceSpec {
        mmco: vec![(4, 1, 0), (3, 1, 0)],
        ..p(2, 0)
    };
    // CurrPicNum 3: frame 1's PicNum 1 is difference_of_pic_nums_minus1 1.
    let unmark_short_term = SliceSpec {
        mmco: vec![(1, 1, 0)],
        ..p(3, 0)
    };
    // LongTermPicNum 0: the IDR.
    let unmark_long_term = SliceSpec {
        mmco: vec![(2, 0, 0)],
        ..p(4, 0)
    };
    let mut d = decoder(OutputOrder::Decode);
    let slices = [
        idr(),
        p(1, 0),
        convert,
        unmark_short_term,
        unmark_long_term,
        p(5, 0),
    ];
    run(&mut d, &spec, &slices)?;
    // Each picture's references: picture, frame_idx, long-term.
    let references = |i: usize| {
        let mut references = d.accelerator().submitted[i].references.clone();
        references.sort_unstable();
        references
    };
    assert_eq!(references(2), [(0, 0, false), (1, 1, false)]);
    assert_eq!(references(3), [(0, 0, true), (1, 1, false), (2, 2, false)]);
    assert_eq!(references(4), [(0, 0, true), (2, 2, false), (3, 3, false)]);
    assert_eq!(references(5), [(2, 2, false), (3, 3, false), (4, 4, false)]);
    Ok(())
}

#[test]
fn list_modification_reaches_the_accelerator() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        ..SpsSpec::default()
    };
    let modified = SliceSpec {
        active: Some((1, 0)),
        // CurrPicNum 3: subtract 3 -> PicNum 0, the IDR.
        modifications: vec![(0, 2)],
        ..p(3, 0)
    };
    let mut d = decoder(OutputOrder::Decode);
    run(&mut d, &spec, &[idr(), p(1, 0), p(2, 0), modified])?;
    assert_eq!(d.accelerator().submitted[3].lists[0], [Some(0), Some(2)]);
    Ok(())
}

#[test]
fn a_p_picture_with_nothing_to_predict_from_is_refused() {
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&spec), pps(false), slice(&p(1, 2), &spec)];
    assert_eq!(
        d.decode(&sample(&units), 0).err(),
        Some(Error::MissingReference)
    );
    assert!(d.accelerator().submitted.is_empty());
}

#[test]
fn a_frame_num_gap_leaves_frames_without_samples_that_are_refused() -> Result<(), Error> {
    let spec = SpsSpec {
        poc_type: 2,
        max_num_ref_frames: 2,
        gaps: true,
        ..SpsSpec::default()
    };
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    // frame_num 1 and 2 are skipped: the sliding window keeps them, and
    // drops the IDR, so a P picture can only predict from frames with no
    // samples.
    let after_gap = d.decode(&sample(&[slice(&p(3, 0), &spec)]), 1);
    assert_eq!(after_gap.err(), Some(Error::MissingReference));
    // An I picture decodes, and the next P predicts from it.
    let intra = SliceSpec {
        slice_type: 2,
        ..p(3, 0)
    };
    d.decode(&sample(&[slice(&intra, &spec)]), 2)?;
    d.decode(&sample(&[slice(&p(4, 0), &spec)]), 3)?;
    let last = d.accelerator().submitted.last().map(|s| s.lists[0].clone());
    assert_eq!(last, Some(vec![Some(1)]));
    Ok(())
}

#[test]
fn an_avcc_record_carries_the_parameter_sets() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let (s, pp) = (sps(&spec), pps(false));
    let mut record = vec![1, 77, 0, 40, 0xff, 0xe1];
    record.extend_from_slice(&u16::try_from(s.len()).unwrap_or(0).to_be_bytes());
    record.extend_from_slice(&s);
    record.push(1);
    record.extend_from_slice(&u16::try_from(pp.len()).unwrap_or(0).to_be_bytes());
    record.extend_from_slice(&pp);
    let mut d = Decoder::from_avcc(Recorder::default(), &record)?;
    let unit = slice(&idr(), &spec);
    let mut sample = u32::try_from(unit.len())
        .unwrap_or(0)
        .to_be_bytes()
        .to_vec();
    sample.extend_from_slice(&unit);
    let out = d.decode(&sample, 7)?;
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].id, 7);
    assert_eq!(out[0].crop, (0, 0, 1920, 1080));
    assert_eq!(out[0].coded_size, (1920, 1088));
    assert!(!out[0].full_range);
    Ok(())
}

#[test]
fn a_full_range_sps_marks_its_frames_full_range() -> Result<(), Error> {
    let spec = SpsSpec {
        full_range: true,
        ..SpsSpec::default()
    };
    let parsed = sps::Sps::parse(&sps(&spec))?;
    assert!(parsed.is_full_range());
    assert_eq!(parsed.max_num_reorder_frames(), 2);
    assert!(!sps::Sps::parse(&sps(&SpsSpec::default()))?.is_full_range());
    let mut d = decoder(OutputOrder::Decode);
    let out = d.decode(&sample(&[sps(&spec), pps(false), slice(&idr(), &spec)]), 1)?;
    assert_eq!(out.len(), 1);
    assert!(out[0].full_range);
    Ok(())
}

#[test]
fn the_sps_reads_cropping_and_the_vui_restriction() -> Result<(), Error> {
    let parsed = sps::Sps::parse(&sps(&SpsSpec::default()))?;
    assert_eq!(parsed.crop_rect(), (0, 0, 1920, 1080));
    assert_eq!(parsed.max_num_reorder_frames(), 2);
    assert_eq!(parsed.dpb_frames(), 4);
    let without = sps::Sps::parse(&sps(&SpsSpec {
        restriction: None,
        ..SpsSpec::default()
    }))?;
    // Level 4.0's MaxDpbMbs, 32768, over 8160 macroblocks a frame.
    assert_eq!(without.dpb_frames(), 4);
    Ok(())
}

#[test]
fn pps_lists_fall_back_to_the_defaults_without_a_sequence_matrix() -> Result<(), Error> {
    let s = Arc::new(sps::Sps::parse(&sps(&SpsSpec {
        profile_idc: 100,
        ..SpsSpec::default()
    }))?);
    let parsed = pps::Pps::parse(&pps(true), |_| Some(s.clone()))?;
    let lists = parsed.scaling_lists(&s);
    // Table 7-3 and 7-4's first and last entries.
    assert_eq!((lists.lists_4x4[0][0], lists.lists_4x4[2][15]), (6, 42));
    assert_eq!((lists.lists_4x4[3][0], lists.lists_4x4[5][15]), (10, 34));
    assert_eq!((lists.lists_8x8[0][0], lists.lists_8x8[0][63]), (6, 42));
    assert_eq!((lists.lists_8x8[1][0], lists.lists_8x8[1][63]), (9, 35));
    Ok(())
}

/// A Main SPS of one macroblock, 16x16, with frame cropping `crop` (left,
/// right, top, bottom, in crop units of two samples) and, where given, a
/// VUI that `vui` writes.
fn one_macroblock_sps(crop: [u32; 4], vui: Option<&dyn Fn(&mut Writer)>) -> Vec<u8> {
    let mut w = Writer::default();
    w.u(8, 77).u(8, 0).u(8, 10).ue(0);
    // frame_num and pic_order_cnt_lsb of 4 bits, a reference, no gaps,
    // one macroblock, progressive.
    w.ue(0).ue(0).ue(0).ue(1).flag(false).ue(0).ue(0);
    w.flag(true).flag(true);
    w.flag(true);
    for offset in crop {
        w.ue(offset);
    }
    w.flag(vui.is_some());
    if let Some(vui) = vui {
        vui(&mut w);
    }
    w.nal(0x67)
}

#[test]
fn cropping_leaves_some_of_the_picture() -> Result<(), Error> {
    let parsed = sps::Sps::parse(&one_macroblock_sps([3, 4, 0, 1], None))?;
    assert_eq!(parsed.crop_rect(), (6, 0, 2, 14));
    for crop in [[4, 4, 0, 0], [0, 0, 8, 0], [u32::MAX - 1, 0, 0, 0]] {
        assert_eq!(
            sps::Sps::parse(&one_macroblock_sps(crop, None)).err(),
            Some(Error::OutOfRange("frame_crop offsets")),
            "{crop:?}"
        );
    }
    Ok(())
}

#[test]
fn sample_aspect_ratios_follow_table_e_1() -> Result<(), Error> {
    for (idc, extended, ratio) in [
        (1, None, Some((1, 1))),
        (14, None, Some((4, 3))),
        (16, None, Some((2, 1))),
        (255, Some((64, 45)), Some((64, 45))),
        (0, None, None),
        (17, None, None),
    ] {
        let vui = |w: &mut Writer| {
            w.flag(true).u(8, idc);
            if let Some((width, height)) = extended {
                w.u(16, width).u(16, height);
            }
            // Overscan, video signal type, chroma location, timing, the
            // HRDs, pic_struct and the restriction: all absent.
            for _ in 0..8 {
                w.flag(false);
            }
        };
        let parsed = sps::Sps::parse(&one_macroblock_sps([0; 4], Some(&vui)))?;
        let vui = parsed.vui.ok_or(Error::OutOfRange("vui"))?;
        assert_eq!(vui.sample_aspect_ratio, ratio, "aspect_ratio_idc {idc}");
    }
    Ok(())
}

#[test]
fn the_vuis_hrd_parameters_are_read_past_to_the_restriction() -> Result<(), Error> {
    let vui = |w: &mut Writer| {
        // No aspect ratio, overscan, video signal type or chroma location;
        // timing of 1001/60000, not fixed.
        w.flag(false).flag(false).flag(false).flag(false);
        w.flag(true).u(32, 1001).u(32, 60000).flag(false);
        // A NAL HRD of two CPBs (E.1.2), no VCL HRD, low_delay_hrd_flag,
        // no pic_struct.
        w.flag(true).ue(1).u(4, 2).u(4, 3);
        for cpb in 0..2 {
            w.ue(5000 + cpb).ue(9000).flag(cpb == 0);
        }
        w.u(5, 23).u(5, 23).u(5, 23).u(5, 24);
        w.flag(false).flag(true).flag(false);
        // The restriction: three frames reordered of four.
        w.flag(true)
            .flag(true)
            .ue(0)
            .ue(0)
            .ue(16)
            .ue(16)
            .ue(3)
            .ue(4);
    };
    let parsed = sps::Sps::parse(&one_macroblock_sps([0; 4], Some(&vui)))?;
    let vui = parsed.vui.ok_or(Error::OutOfRange("vui"))?;
    assert_eq!(vui.timing, Some((1001, 60000)));
    assert_eq!(vui.max_num_reorder_frames, Some(3));
    assert_eq!(vui.max_dec_frame_buffering, Some(4));
    Ok(())
}

#[test]
fn an_idr_slice_that_predicts_is_refused() {
    let spec = SpsSpec::default();
    let predicting = SliceSpec {
        slice_type: 0,
        ..idr()
    };
    let units = [sps(&spec), pps(false), slice(&predicting, &spec)];
    assert_eq!(
        refused(&units),
        (Some(Error::OutOfRange("slice_type of an IDR picture")), 0)
    );
}

/// Decodes `units` as one sample into a fresh decoder, for what it
/// refuses.
fn refused(units: &[Vec<u8>]) -> (Option<Error>, usize) {
    let mut d = decoder(OutputOrder::Decode);
    let error = d.decode(&sample(units), 0).err();
    (error, d.accelerator().submitted.len())
}

#[test]
fn a_slice_qp_past_51_never_reaches_the_accelerator() {
    let spec = SpsSpec::default();
    // 26 + 0 + 26: slice_qp_delta parses, as it would at 14 bits.
    let over = SliceSpec {
        slice_qp_delta: 26,
        ..idr()
    };
    let units = [sps(&spec), pps(false), slice(&over, &spec)];
    assert_eq!(refused(&units), (Some(Error::OutOfRange("SliceQPY")), 0));
    let under = SliceSpec {
        slice_qp_delta: -27,
        ..idr()
    };
    let units = [sps(&spec), pps(false), slice(&under, &spec)];
    assert_eq!(refused(&units), (Some(Error::OutOfRange("SliceQPY")), 0));
}

#[test]
fn a_slice_with_no_slice_data_never_reaches_the_accelerator() {
    let spec = SpsSpec::default();
    let empty = SliceSpec {
        no_data: true,
        ..idr()
    };
    let units = [sps(&spec), pps(false), slice(&empty, &spec)];
    assert_eq!(refused(&units), (Some(Error::EmptySlice), 0));
}

#[test]
fn slices_at_one_macroblock_or_past_the_macroblocks_never_reach_the_accelerator() {
    // 1920x1080: 8,160 macroblocks, each a slice of its own at most.
    let spec = SpsSpec::default();
    let at = |first_mb| slice(&SliceSpec { first_mb, ..idr() }, &spec);
    let mut units = vec![sps(&spec), pps(false)];
    units.extend((0..8160).map(at));
    assert_eq!(refused(&units), (None, 1));
    let units = [sps(&spec), pps(false), at(0), at(5), at(0)];
    assert_eq!(refused(&units), (Some(Error::OverlappingSlices), 0));
    // Gathered no further than that, whatever their starts.
    let mut units = vec![sps(&spec), pps(false)];
    units.extend(std::iter::repeat_n(at(0), 8161));
    assert_eq!(refused(&units), (Some(Error::MoreSlicesThanMacroblocks), 0));
}

#[test]
fn a_pps_changed_between_slices_of_a_picture_never_reaches_the_accelerator() {
    let spec = SpsSpec::default();
    let second = SliceSpec {
        first_mb: 1,
        ..idr()
    };
    let units = [
        sps(&spec),
        pps(false),
        slice(&idr(), &spec),
        pps(true),
        slice(&second, &spec),
    ];
    assert_eq!(
        refused(&units),
        (Some(Error::ParameterSetsChangedInPicture), 0)
    );
    // The same PPS again between them changes nothing.
    let units = [
        sps(&spec),
        pps(false),
        slice(&idr(), &spec),
        pps(false),
        slice(&second, &spec),
    ];
    assert_eq!(refused(&units), (None, 1));
}

#[test]
fn references_past_max_num_ref_frames_never_reach_the_accelerator() {
    // A long-term IDR fills the one reference max_num_ref_frames allows,
    // and the sliding window has no short-term frame to drop (8.2.5.3), so
    // P1 makes two.
    let spec = SpsSpec {
        max_num_ref_frames: 1,
        ..SpsSpec::default()
    };
    let long_term = SliceSpec {
        long_term_idr: true,
        ..idr()
    };
    let mut d = decoder(OutputOrder::Decode);
    let result = run(&mut d, &spec, &[long_term, p(1, 2), p(2, 4)]);
    assert_eq!(result.err(), Some(Error::TooManyReferences));
    assert_eq!(d.accelerator().submitted.len(), 2);
}

#[test]
fn order_count_differences_past_16_bits_are_refused_where_b_slices_use_them() -> Result<(), Error> {
    // Type 1 counts: I0 at 0, P1 at 40000, the non-reference B2 at 20000.
    // B2's lists are I0 and P1, 40000 apart, past 8.2.1's 2^15 - 1.
    let spec = SpsSpec {
        poc_type: 1,
        type1: (-20000, vec![40000]),
        ..SpsSpec::default()
    };
    let temporal = SliceSpec {
        temporal_direct: true,
        ..b(2, 0, false)
    };
    let mut d = decoder(OutputOrder::Decode);
    let result = run(&mut d, &spec, &[idr(), p(1, 0), temporal]);
    assert_eq!(result.err(), Some(Error::OutOfRange("DiffPicOrderCnt")));
    assert_eq!(d.accelerator().submitted.len(), 2);
    // Spatial direct prediction without implicit weights uses no
    // difference, and decodes.
    let mut d = decoder(OutputOrder::Decode);
    let out = run(&mut d, &spec, &[idr(), p(1, 0), b(2, 0, false)])?;
    assert_eq!(out, [(0, 0), (1, 40000), (2, 20000)]);
    Ok(())
}

#[test]
fn an_sps_that_changes_frame_numbering_at_a_non_idr_picture_is_refused() -> Result<(), Error> {
    // MaxFrameNum 256, up to frame 20; then an SPS of MaxFrameNum 16
    // that would take over without a reconfiguration, as only frame
    // numbering differs.
    let wide = SpsSpec {
        log2_max_frame_num_minus4: 4,
        ..SpsSpec::default()
    };
    let narrow = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&wide), pps(false), slice(&idr(), &wide)];
    d.decode(&sample(&units), 0)?;
    for n in 1..=20 {
        d.decode(&sample(&[slice(&p(n, 2 * n), &wide)]), u64::from(n))?;
    }
    // A change to the VUI alone takes over at a non-IDR picture.
    let full_range = SpsSpec {
        full_range: true,
        ..wide.clone()
    };
    let units = [sps(&full_range), slice(&p(21, 42), &full_range)];
    let out = d.decode(&sample(&units), 21)?;
    assert!(out.iter().all(|o| o.full_range));
    assert_eq!(out.len(), 1);
    let units = [sps(&narrow), slice(&p(0, 44), &narrow)];
    assert_eq!(
        d.decode(&sample(&units), 22).err(),
        Some(Error::SpsChangedAtNonIdr)
    );
    Ok(())
}

#[test]
fn operation_5_on_order_counts_at_the_ends_of_32_bits_does_not_overflow() -> Result<(), Error> {
    // TopFieldOrderCnt 10 and a bottom 2^31 - 1 past it, which wraps:
    // operation 5 then subtracts the wrapped count.
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let first = SliceSpec {
        delta_bottom: Some(0),
        ..idr()
    };
    let units = [sps(&spec), pps_with(false, true), slice(&first, &spec)];
    d.decode(&sample(&units), 0)?;
    let far = SliceSpec {
        delta_bottom: Some(i32::MAX),
        mmco: vec![(5, 0, 0)],
        ..p(1, 10)
    };
    let out = d.decode(&sample(&[slice(&far, &spec)]), 1)?;
    assert_eq!(out.len(), 1);
    Ok(())
}

mod fuzz;

#[test]
fn an_sps_that_allows_fields_is_refused_before_any_slice_reaches_the_accelerator() {
    // Apple's AVC decoder refuses fields, MBAFF frames and frames alike
    // once frame_mbs_only_flag is 0.
    for mbaff in [false, true] {
        let spec = SpsSpec {
            fields: Some(mbaff),
            ..SpsSpec::default()
        };
        let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
        assert_eq!(
            refused(&units),
            (Some(Error::Interlaced), 0),
            "MBAFF {mbaff}"
        );
        let parsed = sps::Sps::parse(&sps(&spec)).map(|s| check_supported(&s));
        assert_eq!(parsed, Ok(Err(Error::Interlaced)));
    }
}

#[test]
fn an_interlaced_sps_in_band_is_refused_until_a_progressive_idr_picture() -> Result<(), Error> {
    let progressive = SpsSpec::default();
    let interlaced = SpsSpec {
        fields: Some(true),
        ..SpsSpec::default()
    };
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&progressive), pps(false), slice(&idr(), &progressive)];
    d.decode(&sample(&units), 0)?;
    d.decode(&sample(&[slice(&p(1, 2), &progressive)]), 1)?;
    let units = [sps(&interlaced), pps(false), slice(&idr(), &interlaced)];
    assert_eq!(d.decode(&sample(&units), 2).err(), Some(Error::Interlaced));
    assert!(
        d.decode(&sample(&[slice(&p(1, 2), &interlaced)]), 3)
            .is_err()
    );
    assert_eq!(d.accelerator().submitted.len(), 2);
    let units = [sps(&progressive), pps(false), slice(&idr(), &progressive)];
    assert_eq!(d.decode(&sample(&units), 4)?.len(), 1);
    assert_eq!(
        d.decode(&sample(&[slice(&p(1, 2), &progressive)]), 5)?
            .len(),
        1
    );
    Ok(())
}

/// A sample whose one NAL unit has `forbidden_zero_bit` set.
fn forbidden() -> Vec<u8> {
    sample(&[vec![0x81, 0x55, 0x55, 0x55]])
}

#[test]
fn after_a_sample_that_fails_pictures_are_refused_until_an_i_picture_decodes() -> Result<(), Error>
{
    // Apple's AVC decoder answers each picture after a damaged one with
    // kVTVideoDecoderBadDataErr until an I picture, IDR or not, decodes.
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    d.decode(&sample(&[slice(&p(1, 2), &spec)]), 1)?;
    assert_eq!(
        d.decode(&forbidden(), 2).err(),
        Some(Error::OutOfRange("forbidden_zero_bit"))
    );
    for (id, n) in [(3, 2), (4, 3)] {
        let refused = d.decode(&sample(&[slice(&p(n, 2 * n), &spec)]), id);
        assert_eq!(refused.err(), Some(Error::Unrecovered));
    }
    // Both went to the accelerator, so that they stay references.
    assert_eq!(d.accelerator().submitted.len(), 4);
    let intra = SliceSpec {
        slice_type: 2,
        ..p(4, 8)
    };
    assert_eq!(d.decode(&sample(&[slice(&intra, &spec)]), 5)?.len(), 1);
    assert_eq!(d.decode(&sample(&[slice(&p(5, 10), &spec)]), 6)?.len(), 1);
    // The last P predicts from the I picture, then P4 and P3, which did
    // not come out but were decoded.
    let last = d.accelerator().submitted.last().map(|s| s.lists[0].clone());
    assert_eq!(last, Some(vec![Some(4)]));
    let references = d.accelerator().submitted.last().map(|s| s.references.len());
    assert_eq!(references, Some(4));
    Ok(())
}

#[test]
fn a_picture_the_hardware_failed_fails_each_after_it_until_an_i_picture() -> Result<(), Error> {
    // A truncated slice decodes here, and only the hardware's status, read
    // once the picture is done, says it failed: Apple's AVC decoder answers
    // it and each picture after it with kVTVideoDecoderBadDataErr until an
    // I picture decodes.
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    assert_eq!(d.decode(&sample(&[slice(&p(1, 2), &spec)]), 1)?.len(), 1);
    d.mark_failed();
    for (id, n) in [(2, 2), (3, 3)] {
        let refused = d.decode(&sample(&[slice(&p(n, 2 * n), &spec)]), id);
        assert_eq!(refused.err(), Some(Error::Unrecovered));
    }
    // Each was decoded, so that it stays a reference.
    assert_eq!(d.accelerator().submitted.len(), 4);
    let intra = |n| SliceSpec {
        slice_type: 2,
        ..p(n, 2 * n)
    };
    assert_eq!(d.decode(&sample(&[slice(&intra(4), &spec)]), 4)?.len(), 1);
    assert_eq!(d.decode(&sample(&[slice(&p(5, 10), &spec)]), 5)?.len(), 1);

    // An I picture the hardware failed ends nothing.
    assert_eq!(d.decode(&sample(&[slice(&intra(6), &spec)]), 6)?.len(), 1);
    d.mark_failed();
    let refused = d.decode(&sample(&[slice(&p(7, 14), &spec)]), 7);
    assert_eq!(refused.err(), Some(Error::Unrecovered));
    assert_eq!(d.decode(&sample(&[slice(&intra(8), &spec)]), 8)?.len(), 1);

    // A reset forgets it.
    d.mark_failed();
    d.reset();
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    assert_eq!(d.decode(&sample(&units), 9)?.len(), 1);
    assert_eq!(d.decode(&sample(&[slice(&p(1, 2), &spec)]), 10)?.len(), 1);
    Ok(())
}

#[test]
fn in_display_order_a_picture_refused_after_a_failure_never_comes_out() -> Result<(), Error> {
    let spec = SpsSpec::default();
    let mut d = decoder(OutputOrder::Display);
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    let mut out: Vec<u64> = Vec::new();
    out.extend(d.decode(&sample(&units), 0)?.iter().map(|o| o.id));
    out.extend(
        d.decode(&sample(&[slice(&p(1, 4), &spec)]), 1)?
            .iter()
            .map(|o| o.id),
    );
    assert!(d.decode(&forbidden(), 2).is_err());
    for (id, slice_spec) in [(3, b(2, 2, false)), (4, p(2, 8))] {
        let refused = d.decode(&sample(&[slice(&slice_spec, &spec)]), id);
        assert_eq!(refused.err(), Some(Error::Unrecovered));
    }
    let intra = SliceSpec {
        slice_type: 2,
        ..p(3, 12)
    };
    out.extend(
        d.decode(&sample(&[slice(&intra, &spec)]), 5)?
            .iter()
            .map(|o| o.id),
    );
    out.extend(d.flush().iter().map(|o| o.id));
    assert_eq!(out, [0, 1, 5]);
    Ok(())
}

#[test]
fn a_frame_lost_where_gaps_are_not_allowed_is_stood_in_for_by_a_copy() -> Result<(), Error> {
    // Apple's AVC decoder emits every frame after a lost reference with an
    // image.
    let spec = SpsSpec {
        max_num_ref_frames: 2,
        ..SpsSpec::default()
    };
    let mut d = Decoder::new(
        Recorder {
            conceals: true,
            ..Recorder::default()
        },
        NalFormat::AnnexB,
    );
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    d.decode(&sample(&[slice(&p(1, 2), &spec)]), 1)?;
    // P2 is lost: P3 predicts from a copy of P1 in its place.
    assert_eq!(d.decode(&sample(&[slice(&p(3, 6), &spec)]), 2)?.len(), 1);
    assert_eq!(d.accelerator().concealed, [(1, 2)]);
    let last = d.accelerator().submitted.last().map(|s| s.lists[0].clone());
    assert_eq!(last, Some(vec![Some(2)]));
    assert_eq!(d.decode(&sample(&[slice(&p(4, 8), &spec)]), 3)?.len(), 1);

    // Over an accelerator that copies nothing, P3 is refused, and what
    // follows until an I picture: P4 too predicts from a frame with no
    // picture, as P3 never decoded.
    let mut d = decoder(OutputOrder::Decode);
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    d.decode(&sample(&[slice(&p(1, 2), &spec)]), 1)?;
    let after = d.decode(&sample(&[slice(&p(3, 6), &spec)]), 2);
    assert_eq!(after.err(), Some(Error::MissingReference));
    let next = d.decode(&sample(&[slice(&p(4, 8), &spec)]), 3);
    assert_eq!(next.err(), Some(Error::MissingReference));
    let intra = SliceSpec {
        slice_type: 2,
        ..p(5, 10)
    };
    assert_eq!(d.decode(&sample(&[slice(&intra, &spec)]), 4)?.len(), 1);
    Ok(())
}

#[test]
fn a_gap_the_sps_allows_is_not_concealed() -> Result<(), Error> {
    let spec = SpsSpec {
        max_num_ref_frames: 2,
        gaps: true,
        ..SpsSpec::default()
    };
    let mut d = Decoder::new(
        Recorder {
            conceals: true,
            ..Recorder::default()
        },
        NalFormat::AnnexB,
    );
    let units = [sps(&spec), pps(false), slice(&idr(), &spec)];
    d.decode(&sample(&units), 0)?;
    let after = d.decode(&sample(&[slice(&p(3, 6), &spec)]), 1);
    assert_eq!(after.err(), Some(Error::MissingReference));
    assert!(d.accelerator().concealed.is_empty());
    Ok(())
}
