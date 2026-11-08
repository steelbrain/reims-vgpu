//! The decoding process above the slice data: parameter sets, picture
//! boundaries, order counts, frame number gaps, reference lists, marking
//! and output order. What decodes the slice data is an [`Accelerator`],
//! such as VA-API's.

use std::sync::Arc;

use super::Error;
use super::avcc::DecoderConfiguration;
use super::bounds;
use super::dpb::{Dpb, Entry, Marking};
use super::nal::{NalFormat, NalUnit, nal_type, sample_units};
use super::poc::PocState;
use super::pps::Pps;
use super::scaling::ScalingLists;
use super::slice::{SliceHeader, SliceType};
use super::sps::Sps;

/// The order frames come out in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputOrder {
    /// Each frame as soon as it is decoded, as VideoToolbox emits them.
    Decode,
    /// By order count, through the DPB's bumping process (C.4.5.3), with
    /// the VUI's `max_num_reorder_frames` where it is given.
    Display,
}

/// A decoded frame, as the decoder emits it.
pub struct Output<P> {
    /// The accelerator's picture holding the samples.
    pub picture: Arc<P>,
    /// The id the sample that carried the frame was decoded with.
    pub id: u64,
    /// `PicOrderCnt()`, after the subtraction memory management operation
    /// 5 makes.
    pub poc: i32,
    /// Whether the frame is an IDR picture.
    pub idr: bool,
    /// Whether the frame is a reference (`nal_ref_idc` above 0).
    pub reference: bool,
    /// The coded width and height in samples.
    pub coded_size: (u32, u32),
    /// The SPS's cropping: x, y, width and height in luma samples.
    pub crop: (u32, u32, u32, u32),
    /// The SPS's `video_full_range_flag` ([`crate::codec::h264::sps::Sps::is_full_range`]).
    pub full_range: bool,
}

/// A reference frame as the accelerator sees it.
pub struct ReferenceFrame<'a, P> {
    /// Its picture.
    pub picture: &'a P,
    /// `FrameNum` of a short-term frame, `LongTermFrameIdx` of a long-term
    /// one: VA-API's `frame_idx`.
    pub frame_idx: u32,
    /// Whether it is a long-term reference.
    pub long_term: bool,
    /// `TopFieldOrderCnt`.
    pub top_poc: i32,
    /// `BottomFieldOrderCnt`.
    pub bottom_poc: i32,
}

/// What is the same for every slice of a picture.
pub struct PictureParams<'a, P> {
    /// The active SPS.
    pub sps: &'a Sps,
    /// The picture's PPS.
    pub pps: &'a Pps,
    /// The picture's first slice header.
    pub header: &'a SliceHeader,
    /// `TopFieldOrderCnt` of the picture, as decoded.
    pub top_poc: i32,
    /// `BottomFieldOrderCnt` of the picture, as decoded.
    pub bottom_poc: i32,
    /// The DPB's reference frames that have samples.
    pub references: Vec<ReferenceFrame<'a, P>>,
    /// The scaling lists in force.
    pub scaling_lists: ScalingLists,
}

/// One slice of a picture.
pub struct SliceParams<'a, P> {
    /// Its header.
    pub header: &'a SliceHeader,
    /// Its NAL unit, escaped, from the header byte on.
    pub nal: &'a [u8],
    /// `RefPicList0` and `RefPicList1`, `num_ref_idx_lX_active_minus1 + 1`
    /// long each (empty where the slice type has no such list). `None` is
    /// "no reference picture".
    pub ref_lists: [Vec<Option<ReferenceFrame<'a, P>>>; 2],
}

/// What decodes the slice data: the hardware's half of a stateless
/// decoder.
pub trait Accelerator {
    /// A decoded picture's storage, such as a VA surface.
    type Picture;
    /// The accelerator's error, which carries this module's errors too.
    type Error: From<Error>;

    /// Prepares for frames of `sps`, with a DPB of `dpb_frames` frames.
    /// Called before the first picture, and again when a new SPS changes
    /// what the frames need.
    fn configure(&mut self, sps: &Sps, dpb_frames: usize) -> Result<(), Self::Error>;

    /// Storage for a new picture.
    fn new_picture(&mut self) -> Result<Self::Picture, Self::Error>;

    /// Decodes one picture from all its slices into `picture`.
    fn decode_picture(
        &mut self,
        picture: &Self::Picture,
        params: &PictureParams<'_, Self::Picture>,
        slices: &[SliceParams<'_, Self::Picture>],
    ) -> Result<(), Self::Error>;

    /// A new picture holding what `from` holds, to stand in for a lost
    /// frame (8.2.5.2, a gap in `frame_num` where the SPS allows none), or
    /// `None` where the accelerator cannot copy one. With a stand-in, what
    /// predicts from the lost frame decodes, as Apple's AVC decoder decodes
    /// it; without one it is refused with [`Error::MissingReference`].
    fn conceal(&mut self, from: &Self::Picture) -> Result<Option<Self::Picture>, Self::Error> {
        let _ = from;
        Ok(None)
    }
}

/// An H.264 decoder over an [`Accelerator`], one access unit per sample.
///
/// Progressive Baseline (without FMO, ASO or redundant pictures), Main and
/// High at 8 bits and 4:2:0 are decoded. An SPS that allows fields is
/// refused with [`Error::Interlaced`]; other chroma formats and bit
/// depths, data partitioning and SP/SI slices with [`Error::Unsupported`].
///
/// Errors follow Apple's AVC decoder: after a sample that does not decode,
/// each picture is refused with [`Error::Unrecovered`] until an I picture
/// decodes, and a frame lost from a stream that allows no gap in
/// `frame_num` is stood in for by a copy of the last reference, where the
/// accelerator makes one ([`Accelerator::conceal`]).
pub struct Decoder<A: Accelerator> {
    accelerator: A,
    format: NalFormat,
    order: OutputOrder,
    sps: ParameterSets<Sps>,
    pps: ParameterSets<Pps>,
    /// The SPS the accelerator is configured for.
    active: Option<Arc<Sps>>,
    poc: PocState,
    dpb: Dpb<A::Picture, Output<A::Picture>>,
    /// `PrevRefFrameNum` (7.4.3), and whether a reference has been
    /// decoded since the start or a reset.
    prev_ref_frame_num: Option<u32>,
    /// Whether a sample has failed since the last I picture decoded.
    damaged: bool,
}

impl<A: Accelerator> Decoder<A> {
    /// A decoder for samples in `format`, with no parameter sets yet.
    pub fn new(accelerator: A, format: NalFormat) -> Self {
        Self {
            accelerator,
            format,
            order: OutputOrder::Decode,
            // seq_parameter_set_id is 0 to 31, pic_parameter_set_id 0 to
            // 255 (7.4.2.1.1, 7.4.2.2).
            sps: ParameterSets::new(32),
            pps: ParameterSets::new(256),
            active: None,
            poc: PocState::default(),
            dpb: Dpb::new(),
            prev_ref_frame_num: None,
            damaged: false,
        }
    }

    /// A decoder for length-prefixed samples described by an `avcC`
    /// record, whose parameter sets it takes.
    pub fn from_avcc(accelerator: A, record: &[u8]) -> Result<Self, A::Error> {
        let config = DecoderConfiguration::parse(record)?;
        let mut decoder = Self::new(accelerator, NalFormat::LengthPrefixed(config.length_size));
        for unit in config
            .sequence_parameter_sets
            .iter()
            .chain(&config.picture_parameter_sets)
        {
            decoder.parameter_set(NalUnit::new(unit)?)?;
        }
        Ok(decoder)
    }

    /// Chooses the output order. Decode order is the default.
    pub fn set_output_order(&mut self, order: OutputOrder) {
        self.order = order;
    }

    /// The accelerator.
    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    /// The accelerator, to change what it holds between samples.
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }

    /// The SPS the decoder is configured for, once a picture has started:
    /// what the tests check an SPS took over by.
    #[cfg(test)]
    pub(crate) fn active_sps(&self) -> Option<&Sps> {
        self.active.as_deref()
    }

    /// Decodes one sample, an access unit, tagged `id`. Returns the frames
    /// that come out: in decode order, the sample's own picture; in
    /// display order, those the DPB bumps.
    ///
    /// A sample with an error decodes nothing of the picture it carries;
    /// the parameter sets in it before the error are kept. Each picture
    /// after it is decoded but refused with [`Error::Unrecovered`], until
    /// one of I slices alone decodes.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Output<A::Picture>>, A::Error> {
        let result = self.decode_sample(sample, id);
        if result.is_err() {
            self.damaged = true;
        }
        result
    }

    fn decode_sample(
        &mut self,
        sample: &[u8],
        id: u64,
    ) -> Result<Vec<Output<A::Picture>>, A::Error> {
        let units = sample_units(sample, self.format).map_err(Error::from)?;
        let mut outputs = Vec::new();
        let mut slices: Vec<(SliceHeader, &[u8])> = Vec::new();
        for unit in units {
            let nal = NalUnit::new(unit)?;
            match nal.unit_type() {
                nal_type::SLICE | nal_type::IDR => {
                    let header = SliceHeader::parse(
                        nal,
                        |id| self.pps.get(id).cloned(),
                        |id| self.sps.get(id).cloned(),
                    )?;
                    if header.redundant_pic_cnt > 0 {
                        // Redundant pictures are for decoders that lost
                        // the primary one (7.4.3).
                        continue;
                    }
                    if let Some((first, _)) = slices.first()
                        && header.starts_new_picture(first)
                    {
                        self.picture(&slices, id, &mut outputs)?;
                        slices.clear();
                    }
                    // 7.3.4: each slice holds a macroblock at least, and no
                    // two share one (7.4.3), so a picture has no more slices
                    // than macroblocks.
                    let macroblocks = header.sps.frame_size_in_mbs();
                    if u32::try_from(slices.len()).map_or(true, |count| count >= macroblocks) {
                        return Err(Error::MoreSlicesThanMacroblocks.into());
                    }
                    slices.push((header, unit));
                }
                nal_type::SLICE_PARTITION_A..=nal_type::SLICE_PARTITION_C => {
                    return Err(Error::Unsupported("data partitioning").into());
                }
                nal_type::SPS | nal_type::PPS => self.parameter_set(nal)?,
                nal_type::AUD | nal_type::END_OF_SEQUENCE | nal_type::END_OF_STREAM
                    if !slices.is_empty() =>
                {
                    self.picture(&slices, id, &mut outputs)?;
                    slices.clear();
                }
                _ => {}
            }
        }
        if !slices.is_empty() {
            self.picture(&slices, id, &mut outputs)?;
        }
        Ok(outputs)
    }

    /// Takes a picture already decoded as one that failed: the hardware
    /// reports so once its decode is done, after the sample that carried it
    /// came out, as NVDEC's status does. Each picture after it is decoded
    /// but refused with [`Error::Unrecovered`] until an I picture decodes,
    /// as after a sample that fails here; an I picture that failed so ends
    /// nothing.
    pub fn mark_failed(&mut self) {
        self.damaged = true;
    }

    /// Emits every frame still waiting for output, in display order.
    /// References are kept, so decoding can go on. In decode order
    /// nothing waits, and nothing comes out.
    pub fn flush(&mut self) -> Vec<Output<A::Picture>> {
        let out = self.dpb.bump_all();
        self.dpb.remove_unused();
        out
    }

    /// Forgets every reference and waiting frame, as for a seek: the next
    /// picture must be an IDR or an I picture. Parameter sets are kept.
    pub fn reset(&mut self) {
        self.dpb.clear();
        self.poc = PocState::default();
        self.prev_ref_frame_num = None;
        self.damaged = false;
    }

    fn parameter_set(&mut self, nal: NalUnit<'_>) -> Result<(), Error> {
        match nal.unit_type() {
            nal_type::SPS => {
                let sps = Sps::parse(nal.bytes())?;
                self.sps.insert(sps.seq_parameter_set_id, sps);
            }
            nal_type::PPS => {
                let pps = Pps::parse(nal.bytes(), |id| self.sps.get(id).cloned())?;
                self.pps.insert(pps.pic_parameter_set_id, pps);
            }
            _ => {}
        }
        Ok(())
    }

    /// Activates the picture's SPS, configuring the accelerator when what
    /// the frames need has changed.
    fn activate(
        &mut self,
        sps: &Arc<Sps>,
        idr: bool,
        outputs: &mut Vec<Output<A::Picture>>,
    ) -> Result<(), A::Error> {
        if let Some(active) = &self.active {
            if Arc::ptr_eq(active, sps) || **active == **sps {
                return Ok(());
            }
            if same_frames(active, sps) {
                // A new SPS that changes nothing the frames depend on (the
                // VUI, say) takes over without a reconfiguration. Within a
                // sequence only the VUI may change: frame numbers and order
                // counts go on from the active SPS's (7.4.1.2.1).
                let renumbered = !same_but_vui(active, sps);
                if !idr && renumbered {
                    return Err(Error::SpsChangedAtNonIdr.into());
                }
                check_supported(sps)?;
                if renumbered {
                    // A new sequence's IDR picture: frame numbers and order
                    // counts start again, as the IDR's own decode starts
                    // them, now, so that they do not go on in the new SPS's
                    // numbering where the picture fails after this.
                    self.poc = PocState::default();
                    self.prev_ref_frame_num = None;
                }
                self.active = Some(sps.clone());
                return Ok(());
            }
            if !idr {
                return Err(Error::SpsChangedAtNonIdr.into());
            }
        }
        check_supported(sps)?;
        if self.order == OutputOrder::Display {
            outputs.extend(self.dpb.bump_all());
        }
        self.dpb.clear();
        // Frame numbers and order counts start again with the new SPS, as
        // at the stream's start: a picture that activates it need not be
        // an IDR (after a refused configuration, say), and the last SPS's
        // frame number may be past the new MaxFrameNum.
        self.poc = PocState::default();
        self.prev_ref_frame_num = None;
        // An accelerator that refuses a configuration may have dropped the
        // last one, as VA-API's does: until one succeeds, none is active,
        // so that the next IDR picture configures again, whatever its SPS.
        self.active = None;
        self.accelerator.configure(sps, sps.dpb_frames())?;
        self.active = Some(sps.clone());
        Ok(())
    }

    /// Decodes one picture from its slices.
    fn picture(
        &mut self,
        slices: &[(SliceHeader, &[u8])],
        id: u64,
        outputs: &mut Vec<Output<A::Picture>>,
    ) -> Result<(), A::Error> {
        let Some((first, _)) = slices.first() else {
            return Ok(());
        };
        let sps = first.sps.clone();
        let pps = first.pps.clone();
        // An active SPS has frame_mbs_only_flag 1 (check_supported), so no
        // slice of it is a field and no frame MBAFF.
        self.activate(&sps, first.idr, outputs)?;
        if slices
            .iter()
            .any(|(h, _)| matches!(h.slice_type, SliceType::Sp | SliceType::Si))
        {
            return Err(Error::Unsupported("SP and SI slices").into());
        }
        let max_frame_num = sps.max_frame_num();
        let frame_num = first.frame_num;
        if first.idr {
            self.prev_ref_frame_num = None;
        }
        if let Some(prev) = self.prev_ref_frame_num
            && frame_num != prev
            && frame_num != sps.frame_num_plus(prev, 1)
        {
            self.fill_frame_num_gap(&sps, prev, frame_num)?;
        }
        let order = self.poc.compute(&sps, first)?;
        self.dpb.update_frame_num_wrap(frame_num, max_frame_num)?;

        // Every slice's lists, before anything reaches the accelerator.
        let mut lists = Vec::with_capacity(slices.len());
        for (header, _) in slices {
            let mut pair = [Vec::new(), Vec::new()];
            let initial = match header.slice_type {
                SliceType::P => Some([self.dpb.initial_list_p(), Vec::new()]),
                SliceType::B => Some(self.dpb.initial_lists_b(order.frame())),
                _ => None,
            };
            if let Some(initial) = initial {
                let counts = [
                    usize::from(header.num_ref_idx_l0_active_minus1) + 1,
                    usize::from(header.num_ref_idx_l1_active_minus1) + 1,
                ];
                let used = if header.slice_type == SliceType::B {
                    2
                } else {
                    1
                };
                let lists = pair
                    .iter_mut()
                    .zip(&initial)
                    .zip(&header.list_modifications)
                    .zip(counts)
                    .take(used);
                for (((out, initial), modifications), count) in lists {
                    let list = self.dpb.modify_list(
                        initial,
                        modifications,
                        count,
                        frame_num,
                        max_frame_num,
                    )?;
                    // A slice with no frame to predict from, or one that
                    // names an inferred frame, would have the hardware
                    // read a surface that holds nothing.
                    let has_samples =
                        |&i: &usize| self.dpb.entries.get(i).is_some_and(|e| e.picture.is_some());
                    let missing = list.first().is_none_or(Option::is_none)
                        || !list.iter().flatten().all(has_samples);
                    if missing {
                        return Err(Error::MissingReference.into());
                    }
                    *out = list;
                }
            }
            lists.push(pair);
        }

        let picture = self.accelerator.new_picture()?;
        {
            let entries = &self.dpb.entries;
            let reference = reference_frame::<A::Picture, Output<A::Picture>>;
            let params = PictureParams {
                sps: &sps,
                pps: &pps,
                header: first,
                top_poc: order.top,
                bottom_poc: order.bottom,
                references: entries
                    .iter()
                    .filter(|e| e.marking != Marking::Unused)
                    .filter_map(reference)
                    .collect(),
                scaling_lists: pps.scaling_lists(&sps),
            };
            let slice_params: Vec<SliceParams<'_, A::Picture>> = slices
                .iter()
                .zip(&lists)
                .map(|((header, nal), pair)| SliceParams {
                    header,
                    nal,
                    ref_lists: pair.each_ref().map(|list| {
                        list.iter()
                            .map(|i| i.and_then(|i| entries.get(i)).and_then(reference))
                            .collect()
                    }),
                })
                .collect();
            bounds::check_picture(&params, &slice_params)?;
            self.accelerator
                .decode_picture(&picture, &params, &slice_params)?;
        }
        // Apple's decoder gives no frame after one that failed until an I
        // picture decodes; the picture still goes on as a reference.
        let intra = slices.iter().all(|(h, _)| h.slice_type == SliceType::I);
        let unrecovered = self.damaged && !intra;
        if intra {
            self.damaged = false;
        }

        // Marking (8.2.5), then the order count's adjustment for
        // operation 5 (8.2.1).
        let reference = first.nal_ref_idc != 0;
        let mut marking = Marking::Unused;
        let mut unmark_all = false;
        if reference {
            let current = self.dpb.mark(
                first.idr,
                &first.ref_pic_marking,
                frame_num,
                max_frame_num,
                sps.max_num_ref_frames,
            )?;
            marking = current.marking;
            unmark_all = current.unmark_all;
        }
        let (mut top, mut bottom) = (order.top, order.bottom);
        if unmark_all {
            // Wrapping, as the counts are computed: a stream whose counts
            // pass 32 bits breaks 8.2.1, and only output order follows.
            let temp = top.min(bottom);
            top = top.wrapping_sub(temp);
            bottom = bottom.wrapping_sub(temp);
        }
        self.poc.finish(first, &order, unmark_all, top);
        if reference {
            self.prev_ref_frame_num = Some(if unmark_all { 0 } else { frame_num });
        }

        let picture = Arc::new(picture);
        let output = Output {
            picture: picture.clone(),
            id,
            poc: top.min(bottom),
            idr: first.idr,
            reference,
            coded_size: (sps.coded_width(), sps.coded_height()),
            crop: sps.crop_rect(),
            full_range: sps.is_full_range(),
        };
        let entry = |pending| Entry {
            picture: Some(picture.clone()),
            frame_num: if unmark_all { 0 } else { frame_num },
            frame_num_wrap: 0,
            top,
            bottom,
            marking,
            pending,
        };
        if unrecovered {
            // Kept as a reference, never output.
            if self.order == OutputOrder::Display && (first.idr || unmark_all) {
                outputs.extend(self.dpb.bump_all());
            }
            self.dpb.remove_unused();
            if reference {
                self.dpb.entries.push(entry(None));
            }
            return Err(Error::Unrecovered.into());
        }
        match self.order {
            OutputOrder::Decode => {
                outputs.push(output);
                self.dpb.remove_unused();
                if reference {
                    self.dpb.entries.push(entry(None));
                }
            }
            OutputOrder::Display => {
                // C.4.4: an IDR picture or operation 5 empties the DPB,
                // with output unless no_output_of_prior_pics_flag.
                if first.idr || unmark_all {
                    if first.idr && first.ref_pic_marking.no_output_of_prior_pics_flag {
                        self.dpb.discard_pending();
                    } else {
                        outputs.extend(self.dpb.bump_all());
                    }
                }
                self.dpb.remove_unused();
                // C.4.5.1 and C.4.5.2: with no empty frame buffer, a
                // non-reference picture that would come out before every
                // waiting one is output at once; otherwise the bumping
                // process makes room for it.
                let size = sps.dpb_frames();
                let output_at_once = |dpb: &Dpb<A::Picture, Output<A::Picture>>| {
                    !reference && dpb.min_pending_poc().is_none_or(|min| output.poc < min)
                };
                self.dpb
                    .bump_while(outputs, |dpb| dpb.len() >= size && !output_at_once(dpb));
                if self.dpb.len() >= size && output_at_once(&self.dpb) {
                    outputs.push(output);
                } else {
                    self.dpb.entries.push(entry(Some(output)));
                    // The VUI's max_num_reorder_frames (E.2.1), or the
                    // DPB's size: no more frames wait than that.
                    let reorder = sps.max_num_reorder_frames();
                    self.dpb
                        .bump_while(outputs, |dpb| dpb.pending_count() > reorder);
                }
            }
        }
        Ok(())
    }

    /// 8.2.5.2: frames for the `frame_num` values a gap skipped, each
    /// marked short-term through the sliding window, with no samples.
    /// Where the SPS allows no gap, the frames were lost, and each is a
    /// copy of the last reference where the accelerator makes one
    /// ([`Accelerator::conceal`]); otherwise it has no picture, and a
    /// slice that predicts from it is refused.
    fn fill_frame_num_gap(&mut self, sps: &Sps, prev: u32, frame_num: u32) -> Result<(), A::Error> {
        let source = if sps.gaps_in_frame_num_value_allowed_flag {
            None
        } else {
            self.dpb
                .entries
                .iter()
                .rev()
                .filter(|e| e.marking != Marking::Unused)
                .find_map(|e| e.picture.clone())
        };
        let max = sps.max_frame_num();
        // The frame numbers skipped, from PrevRefFrameNum + 1 on, modulo
        // MaxFrameNum (7.4.3).
        let gap = sps.frame_num_minus(frame_num, sps.frame_num_plus(prev, 1));
        // Frames past the last max_num_ref_frames slide straight out; skip
        // them rather than loop over a MaxFrameNum-long gap.
        let keep = u32::from(sps.max_num_ref_frames.max(1));
        // UnusedShortTermFrameNum (7-24), from the first frame kept.
        let mut unused = sps.frame_num_plus(sps.frame_num_plus(prev, 1), gap.saturating_sub(keep));
        while unused != frame_num {
            self.dpb.update_frame_num_wrap(unused, max)?;
            self.dpb.sliding_window(sps.max_num_ref_frames);
            self.dpb.remove_unused();
            let picture = match &source {
                Some(from) => self.accelerator.conceal(from)?.map(Arc::new),
                None => None,
            };
            self.dpb.entries.push(Entry {
                picture,
                frame_num: unused,
                frame_num_wrap: 0,
                top: 0,
                bottom: 0,
                marking: Marking::ShortTerm,
                pending: None,
            });
            self.poc.skip_frame(unused, max);
            self.prev_ref_frame_num = Some(unused);
            unused = sps.frame_num_plus(unused, 1);
        }
        Ok(())
    }
}

/// A DPB frame as the accelerator sees it: `None` for an inferred frame,
/// which has no samples.
fn reference_frame<P, O>(e: &Entry<P, O>) -> Option<ReferenceFrame<'_, P>> {
    let picture = e.picture.as_deref()?;
    let (frame_idx, long_term) = match e.marking {
        Marking::LongTerm(idx) => (idx, true),
        _ => (e.frame_num, false),
    };
    Some(ReferenceFrame {
        picture,
        frame_idx,
        long_term,
        top_poc: e.top,
        bottom_poc: e.bottom,
    })
}

/// A decoder's parameter sets of one kind, by id: a slot for each id the
/// standard allows, holding the last set received with it.
pub(crate) struct ParameterSets<T> {
    slots: Vec<Option<Arc<T>>>,
}

impl<T> ParameterSets<T> {
    /// Slots for ids 0 to `ids - 1`.
    pub(crate) fn new(ids: usize) -> Self {
        Self {
            slots: (0..ids).map(|_| None).collect(),
        }
    }

    /// The set with `id`, once received.
    pub(crate) fn get(&self, id: u8) -> Option<&Arc<T>> {
        self.slots.get(usize::from(id))?.as_ref()
    }

    /// Keeps `set` as the one with `id`, in place of any before it. The
    /// parsers refuse an id past the slots.
    pub(crate) fn insert(&mut self, id: u8, set: T) {
        if let Some(slot) = self.slots.get_mut(usize::from(id)) {
            *slot = Some(Arc::new(set));
        }
    }
}

/// Whether two SPSs make frames alike: the same size, format and DPB.
fn same_frames(a: &Sps, b: &Sps) -> bool {
    a.coded_width() == b.coded_width()
        && a.coded_height() == b.coded_height()
        && a.chroma_format_idc == b.chroma_format_idc
        && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8
        && a.bit_depth_chroma_minus8 == b.bit_depth_chroma_minus8
        && a.profile_idc == b.profile_idc
        && a.dpb_frames() == b.dpb_frames()
        && a.frame_mbs_only_flag == b.frame_mbs_only_flag
}

/// Whether two SPSs are the same but for their VUI.
fn same_but_vui(a: &Sps, b: &Sps) -> bool {
    let without_vui = |sps: &Sps| Sps {
        vui: None,
        ..sps.clone()
    };
    without_vui(a) == without_vui(b)
}

/// Whether [`Decoder`] takes `sps`'s stream: progressive
/// (`frame_mbs_only_flag` 1, else [`Error::Interlaced`]), 4:2:0 at 8 bits,
/// without lossless macroblocks or the Extended profile.
pub fn check_supported(sps: &Sps) -> Result<(), Error> {
    if !sps.frame_mbs_only_flag {
        return Err(Error::Interlaced);
    }
    if sps.chroma_format_idc != 1 {
        return Err(Error::Unsupported("chroma formats other than 4:2:0"));
    }
    if sps.bit_depth_luma_minus8 != 0 || sps.bit_depth_chroma_minus8 != 0 {
        return Err(Error::Unsupported("bit depths other than 8"));
    }
    if sps.qpprime_y_zero_transform_bypass_flag {
        return Err(Error::Unsupported("lossless macroblocks"));
    }
    if sps.profile_idc == 88 {
        return Err(Error::Unsupported("the Extended profile"));
    }
    Ok(())
}
