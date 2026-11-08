//! The decoding process above the slice data: parameter sets, picture
//! boundaries, random access (CRA, BLA, IDR and the RASL pictures they
//! skip), picture order counts, the reference picture set, the reference
//! lists and output order. What decodes the slice data is an
//! [`Accelerator`], such as VA-API's.

use std::num::NonZeroU32;
use std::sync::Arc;

use super::bounds;
use super::dpb::{Dpb, Entry, Marking, RpsPictures, poc_lsb};
use super::hvcc::DecoderConfiguration;
use super::nal::{
    NalUnit, is_bla, is_idr, is_irap, is_radl, is_rasl, is_sub_layer_non_reference, nal_type,
};
use super::pps::Pps;
use super::scaling::ScalingLists;
use super::slice::{SliceHeader, SliceType};
use super::sps::Sps;
use super::{Error, NalFormat, OutputOrder};
use crate::codec::h264::ParameterSets;
use crate::codec::h264::nal::sample_units;

pub use super::dpb::RpsMembership;

/// A decoded picture, as the decoder emits it.
pub struct Output<P> {
    /// The accelerator's picture holding the samples.
    pub picture: Arc<P>,
    /// The id the sample that carried the picture was decoded with.
    pub id: u64,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Whether the picture is an IDR picture.
    pub idr: bool,
    /// Whether pictures of its own sub-layer may predict from it: all but
    /// the sub-layer non-reference pictures (`TRAIL_N` and the like).
    pub reference: bool,
    /// The coded width and height in luma samples.
    pub coded_size: (u32, u32),
    /// The conformance window: x, y, width and height in luma samples.
    pub crop: (u32, u32, u32, u32),
    /// The VUI's `video_full_range_flag` ([`Sps::is_full_range`]).
    pub full_range: bool,
    /// The bit depth, `BitDepthY` (the same as `BitDepthC`).
    pub bit_depth: u8,
}

/// A picture of the DPB the current picture's reference picture set
/// names, as the accelerator sees it.
pub struct ReferencePicture<'a, P> {
    /// Its picture.
    pub picture: &'a P,
    /// `PicOrderCntVal`.
    pub poc: i32,
    /// Whether it is marked as a long-term reference.
    pub long_term: bool,
    /// The set it is in.
    pub membership: RpsMembership,
}

/// What is the same for every slice segment of a picture.
pub struct PictureParams<'a, P> {
    /// The active SPS.
    pub sps: &'a Sps,
    /// The picture's PPS.
    pub pps: &'a Pps,
    /// The picture's first slice segment header.
    pub header: &'a SliceHeader,
    /// `PicOrderCntVal` of the picture.
    pub poc: i32,
    /// Every picture of the reference picture set that is in the DPB,
    /// fifteen at most.
    pub references: Vec<ReferencePicture<'a, P>>,
    /// `RefPicSetStCurrBefore`, `RefPicSetStCurrAfter` and
    /// `RefPicSetLtCurr` (8.3.2), each in the order 8-5 derives it, as
    /// indices into [`Self::references`]: what the reference lists are
    /// built from (8.3.4), for an accelerator that builds them itself.
    pub current_sets: [Vec<usize>; 3],
    /// Whether every slice is an I slice.
    pub intra_only: bool,
    /// Whether any slice is a B slice.
    pub has_b_slices: bool,
    /// The scaling lists in force, when `scaling_list_enabled_flag`: the
    /// PPS's, else the SPS's, else the defaults (7.4.5).
    pub scaling_lists: Option<ScalingLists>,
}

/// One slice segment of a picture.
pub struct SliceParams<'a> {
    /// Its header.
    pub header: &'a SliceHeader,
    /// Its NAL unit, escaped, from the header on.
    pub nal: &'a [u8],
    /// `RefPicList0` and `RefPicList1` (8.3.4), as indices into
    /// [`PictureParams::references`]; empty where the slice type has no
    /// such list.
    pub ref_lists: [Vec<usize>; 2],
    /// Whether it is the picture's last slice segment.
    pub last: bool,
}

/// What decodes the slice data: the hardware's half of a stateless
/// decoder.
pub trait Accelerator {
    /// A decoded picture's storage, such as a VA surface.
    type Picture;
    /// The accelerator's error, which carries this module's errors too.
    type Error: From<Error>;

    /// Prepares for pictures of `sps`, with a DPB of `dpb_size` pictures.
    /// Called before the first picture, and again when a coded video
    /// sequence's SPS changes what the pictures need.
    fn configure(&mut self, sps: &Sps, dpb_size: usize) -> Result<(), Self::Error>;

    /// Storage for a new picture.
    fn new_picture(&mut self) -> Result<Self::Picture, Self::Error>;

    /// Decodes one picture from all its slice segments into `picture`.
    fn decode_picture(
        &mut self,
        picture: &Self::Picture,
        params: &PictureParams<'_, Self::Picture>,
        slices: &[SliceParams<'_>],
    ) -> Result<(), Self::Error>;
}

/// An HEVC decoder over an [`Accelerator`], one access unit per sample.
///
/// Main, Main 10 and Main Still Picture streams of the base layer are
/// decoded: 4:2:0 at 8 or 10 bits, with tiles, wavefront parallel
/// processing, dependent slice segments, scaling lists, weighted
/// prediction and long-term references. Other chroma formats and bit
/// depths, and the range and screen content extensions, are refused with
/// [`Error::Unsupported`].
///
/// Errors follow Apple's HEVC decoder: after a sample that does not
/// decode, each picture is refused with [`Error::Unrecovered`] until an
/// IRAP picture decodes, and that IRAP picture's RASL pictures are
/// skipped, as at the start, with [`Error::SkippedLeadingPicture`].
pub struct Decoder<A: Accelerator> {
    accelerator: A,
    format: NalFormat,
    order: OutputOrder,
    sps: ParameterSets<Sps>,
    pps: ParameterSets<Pps>,
    /// The SPS the accelerator is configured for.
    active: Option<Arc<Sps>>,
    dpb: Dpb<A::Picture, Output<A::Picture>>,
    /// `PicOrderCntVal` of `prevTid0Pic` (8.3.1).
    prev_tid0_poc: Option<i32>,
    /// Whether the next picture starts the bitstream for this decoder: at
    /// the start, after a reset and after an end of sequence, where a CRA
    /// picture has `NoRaslOutputFlag` 1.
    new_sequence: bool,
    /// `NoRaslOutputFlag` of the last IRAP picture: its RASL pictures are
    /// skipped.
    skip_rasl: bool,
    /// The key the next picture gets in the DPB.
    next_key: u64,
    /// Whether a sample has failed since the last IRAP picture decoded.
    damaged: bool,
    /// Whether the sample being decoded held a RASL picture that was
    /// skipped.
    skipped_leading: bool,
}

impl<A: Accelerator> Decoder<A> {
    /// A decoder for samples in `format`, with no parameter sets yet.
    pub fn new(accelerator: A, format: NalFormat) -> Self {
        Self {
            accelerator,
            format,
            order: OutputOrder::Decode,
            // sps_seq_parameter_set_id is 0 to 15, pps_pic_parameter_set_id
            // 0 to 63 (7.4.3.2.1, 7.4.3.3.1).
            sps: ParameterSets::new(16),
            pps: ParameterSets::new(64),
            active: None,
            dpb: Dpb::new(),
            prev_tid0_poc: None,
            new_sequence: true,
            skip_rasl: false,
            next_key: 0,
            damaged: false,
            skipped_leading: false,
        }
    }

    /// A decoder for length-prefixed samples described by an `hvcC`
    /// record, whose parameter sets it takes.
    pub fn from_hvcc(accelerator: A, record: &[u8]) -> Result<Self, A::Error> {
        let config = DecoderConfiguration::parse(record)?;
        let mut decoder = Self::new(accelerator, NalFormat::LengthPrefixed(config.length_size));
        for unit in &config.nal_units {
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

    /// Decodes one sample, an access unit, tagged `id`. Returns the
    /// pictures that come out: in decode order, the sample's own picture
    /// unless it is not output (a skipped RASL picture, or one with
    /// `pic_output_flag` 0); in display order, those the DPB bumps.
    ///
    /// A sample with an error decodes nothing of the picture it carries;
    /// the parameter sets in it before the error are kept. Each picture
    /// after it is refused with [`Error::Unrecovered`] until an IRAP
    /// picture decodes, whose RASL pictures are then skipped. A sample
    /// whose one picture is a RASL picture skipped is refused with
    /// [`Error::SkippedLeadingPicture`], which is no failure.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Output<A::Picture>>, A::Error> {
        self.skipped_leading = false;
        match self.decode_sample(sample, id) {
            Err(error) => {
                self.damaged = true;
                Err(error)
            }
            Ok(outputs) if outputs.is_empty() && self.skipped_leading => {
                Err(Error::SkippedLeadingPicture.into())
            }
            ok => ok,
        }
    }

    fn decode_sample(
        &mut self,
        sample: &[u8],
        id: u64,
    ) -> Result<Vec<Output<A::Picture>>, A::Error> {
        let units = sample_units(sample, self.format).map_err(Error::from)?;
        let mut outputs = Vec::new();
        let mut slices: Vec<(SliceHeader, &[u8])> = Vec::new();
        // The last independent slice segment in `slices`, whose fields a
        // dependent one takes.
        let mut independent_at: Option<usize> = None;
        for unit in units {
            let nal = NalUnit::new(unit)?;
            if nal.layer_id() != 0 {
                // Only the base layer is decoded (F.8: other layers' units
                // are for multi-layer decoders).
                continue;
            }
            match nal.unit_type() {
                nal_type::TRAIL_N..=nal_type::RASL_R | nal_type::BLA_W_LP..=nal_type::CRA_NUT => {
                    let independent = independent_at.and_then(|at| slices.get(at)).map(|(h, _)| h);
                    let header = SliceHeader::parse(
                        nal,
                        |id| self.pps.get(id).cloned(),
                        |id| self.sps.get(id).cloned(),
                        independent,
                    )?;
                    if header.first_slice_segment_in_pic_flag && !slices.is_empty() {
                        self.picture(&slices, id, &mut outputs)?;
                        slices.clear();
                        independent_at = None;
                    }
                    // 7.3.8.1: each segment holds a CTU at least, at an
                    // address of its own (7.4.7.1), so a picture has no more
                    // segments than CTBs.
                    let ctbs = header.sps.pic_size_in_ctbs();
                    if u32::try_from(slices.len()).map_or(true, |count| count >= ctbs) {
                        return Err(Error::MoreSliceSegmentsThanCtbs.into());
                    }
                    if !header.dependent_slice_segment_flag {
                        independent_at = Some(slices.len());
                    }
                    slices.push((header, unit));
                }
                nal_type::SPS | nal_type::PPS => self.parameter_set(nal)?,
                nal_type::AUD if !slices.is_empty() => {
                    self.picture(&slices, id, &mut outputs)?;
                    slices.clear();
                    independent_at = None;
                }
                nal_type::EOS | nal_type::EOB => {
                    if !slices.is_empty() {
                        self.picture(&slices, id, &mut outputs)?;
                        slices.clear();
                        independent_at = None;
                    }
                    // The next picture starts a coded video sequence, with
                    // NoRaslOutputFlag 1 (8.1.3).
                    self.new_sequence = true;
                }
                // VPS, SEI, the reserved VCL types, and the rest: nothing
                // the decoding process reads.
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
    /// came out, as NVDEC's status does. Each picture after it is refused
    /// with [`Error::Unrecovered`] until an IRAP picture decodes, whose RASL
    /// pictures are then skipped, as after a sample that fails here; an
    /// IRAP picture that failed so ends nothing.
    pub fn mark_failed(&mut self) {
        self.damaged = true;
    }

    /// Emits every picture still waiting for output, in display order.
    /// References are kept, so decoding can go on. In decode order nothing
    /// waits, and nothing comes out.
    pub fn flush(&mut self) -> Vec<Output<A::Picture>> {
        let out = self.dpb.bump_all();
        self.dpb.remove_unused();
        out
    }

    /// Forgets every reference and waiting picture, as for a seek: the
    /// next picture must be an IRAP picture, and a CRA's RASL pictures are
    /// skipped. Parameter sets are kept.
    pub fn reset(&mut self) {
        self.dpb.clear();
        self.prev_tid0_poc = None;
        self.new_sequence = true;
        self.skip_rasl = false;
        self.damaged = false;
    }

    fn parameter_set(&mut self, nal: NalUnit<'_>) -> Result<(), Error> {
        match nal.unit_type() {
            nal_type::SPS => {
                let sps = Sps::parse(nal.bytes())?;
                self.sps.insert(sps.sps_seq_parameter_set_id, sps);
            }
            nal_type::PPS => {
                let pps = Pps::parse(nal.bytes())?;
                self.pps.insert(pps.pps_pic_parameter_set_id, pps);
            }
            _ => {}
        }
        Ok(())
    }

    /// Makes `sps` the active SPS, configuring the accelerator when what
    /// the pictures need has changed, which only a picture starting a
    /// coded video sequence may do.
    fn activate(
        &mut self,
        sps: &Arc<Sps>,
        irap: bool,
        starts_sequence: bool,
    ) -> Result<(), A::Error> {
        if let Some(active) = &self.active {
            if Arc::ptr_eq(active, sps) || **active == **sps {
                return Ok(());
            }
            if same_pictures(active, sps) {
                // A new SPS that changes nothing the pictures depend on
                // (the VUI, say) takes over without a reconfiguration. At
                // a picture that is not IRAP only the VUI may change: order
                // counts and sets go on from the active SPS's (7.4.3.2.1).
                if !irap && !same_but_vui(active, sps) {
                    return Err(Error::SpsChangedAtNonIrap.into());
                }
                self.active = Some(sps.clone());
                return Ok(());
            }
            if !starts_sequence {
                return Err(Error::SpsChangedWithinSequence.into());
            }
        }
        self.dpb.clear();
        // An accelerator that refuses a configuration may have dropped the
        // last one, as VA-API's does: until one succeeds, none is active,
        // so that the next IRAP picture configures again, whatever its
        // SPS.
        self.active = None;
        self.accelerator.configure(sps, sps.dpb_size())?;
        self.active = Some(sps.clone());
        Ok(())
    }

    /// Decodes one picture from its slice segments.
    fn picture(
        &mut self,
        slices: &[(SliceHeader, &[u8])],
        id: u64,
        outputs: &mut Vec<Output<A::Picture>>,
    ) -> Result<(), A::Error> {
        let Some((first, _)) = slices.first() else {
            return Ok(());
        };
        if !first.first_slice_segment_in_pic_flag {
            return Err(Error::NoFirstSliceSegment.into());
        }
        let unit_type = first.nal_unit_type;
        let irap = is_irap(unit_type);
        if self.damaged && !irap {
            // Apple's HEVC decoder gives no picture after one that failed
            // until an IRAP picture decodes. That IRAP picture starts
            // afresh, so no picture after it predicts from this one, which
            // is not decoded.
            return Err(Error::Unrecovered.into());
        }
        // NoRaslOutputFlag (8.1.3). HandleCraAsBlaFlag is 1 for a CRA
        // picture after a failure: Apple's decoder skips its RASL
        // pictures, as a BLA picture's.
        let starts_sequence =
            irap && (is_idr(unit_type) || is_bla(unit_type) || self.new_sequence || self.damaged);
        if irap {
            self.skip_rasl = starts_sequence;
        }
        if is_rasl(unit_type) && self.skip_rasl {
            // A RASL picture of an IRAP picture with NoRaslOutputFlag
            // predicts from pictures before it that this decoder never had:
            // it is not output (8.1.3), and not decoded.
            self.skipped_leading = true;
            return Ok(());
        }
        let sps = first.sps.clone();
        let pps = first.pps.clone();
        check_supported(&sps)?;
        check_pps_supported(&pps, &sps)?;
        if slices
            .iter()
            .any(|(h, _)| !Arc::ptr_eq(&h.pps, &pps) && *h.pps != *pps)
        {
            return Err(Error::PpsChangedInPicture.into());
        }
        let previous = if starts_sequence {
            None
        } else {
            self.prev_tid0_poc
        };
        let poc = order_count(
            previous,
            first.slice_pic_order_cnt_lsb,
            sps.max_pic_order_cnt_lsb(),
        )?;

        // 8.3.2: the reference picture set, found before anything changes.
        let found = if starts_sequence || is_idr(unit_type) {
            RpsPictures::default()
        } else {
            self.dpb.find(
                poc,
                &first.short_term_rps,
                &first.long_term,
                sps.max_pic_order_cnt_lsb(),
            )
        };
        if !found.current_complete() {
            return Err(Error::MissingReference.into());
        }

        // C.5.2.2: the RPS's marking, then output and removal before the
        // picture is decoded.
        if starts_sequence {
            self.dpb.unmark_all();
            if !self.dpb.is_empty() {
                // NoOutputOfPriorPicsFlag: 1 for a CRA picture, else the
                // slice header's flag.
                let no_output =
                    unit_type == nal_type::CRA_NUT || first.no_output_of_prior_pics_flag;
                if no_output {
                    self.dpb.discard_pending();
                } else {
                    outputs.extend(self.dpb.bump_all());
                }
                self.dpb.clear();
            }
        } else {
            self.dpb.mark(&found);
        }
        self.activate(&sps, irap, starts_sequence)?;
        self.dpb.remove_unused();
        if self.order == OutputOrder::Display && !starts_sequence {
            // C.5.2.2: bumping until no more pictures wait than
            // sps_max_num_reorder_pics, none has reached
            // SpsMaxLatencyPictures, and the DPB has room.
            let reorder = usize::from(sps.highest_ordering().max_num_reorder_pics);
            let latency = sps.max_latency_pictures();
            let size = sps.dpb_size();
            self.dpb.bump_while(outputs, |dpb| {
                dpb.pending_count() > reorder || dpb.latency_reached(latency) || dpb.len() >= size
            });
        }

        // The pictures the accelerator sees, and each slice's lists
        // (8.3.4) as indices into them.
        let picture = self.accelerator.new_picture()?;
        {
            let references: Vec<&Entry<_, _>> = self
                .dpb
                .entries
                .iter()
                .filter(|e| e.marking != Marking::Unused && e.membership.is_some())
                .take(15)
                .collect();
            let index_of = |key: Option<u64>| -> Result<usize, Error> {
                references
                    .iter()
                    .position(|e| Some(e.key) == key)
                    .ok_or(Error::MissingReference)
            };
            let curr = |set: &[Option<u64>]| -> Result<Vec<usize>, Error> {
                set.iter().map(|&key| index_of(key)).collect()
            };
            let before = curr(&found.st_curr_before)?;
            let after = curr(&found.st_curr_after)?;
            let long_term = curr(&found.lt_curr)?;
            let mut slice_params = Vec::with_capacity(slices.len());
            let last = slices.len().checked_sub(1);
            for (i, (header, nal)) in slices.iter().enumerate() {
                let mut ref_lists = [Vec::new(), Vec::new()];
                let groups: [[&[usize]; 3]; 2] =
                    [[&before, &after, &long_term], [&after, &before, &long_term]];
                for (list, ((out, groups), entries)) in ref_lists
                    .iter_mut()
                    .zip(&groups)
                    .zip(&header.list_entries)
                    .enumerate()
                {
                    let count = header.active_references(list);
                    if count != 0 {
                        *out = reference_list(groups, count, entries.as_deref())?;
                    }
                }
                slice_params.push(SliceParams {
                    header,
                    nal,
                    ref_lists,
                    last: Some(i) == last,
                });
            }
            let scaling_lists = sps.scaling_list_enabled_flag.then(|| {
                pps.scaling_lists
                    .clone()
                    .or_else(|| sps.scaling_lists.clone())
                    .unwrap_or_default()
            });
            let params = PictureParams {
                sps: &sps,
                pps: &pps,
                header: first,
                poc,
                references: references
                    .iter()
                    .map(|e| ReferencePicture {
                        picture: &*e.picture,
                        poc: e.poc,
                        long_term: e.marking == Marking::LongTerm,
                        membership: e.membership.unwrap_or(RpsMembership::Foll),
                    })
                    .collect(),
                current_sets: [before, after, long_term],
                intra_only: slices.iter().all(|(h, _)| h.slice_type == SliceType::I),
                has_b_slices: slices.iter().any(|(h, _)| h.slice_type == SliceType::B),
                scaling_lists,
            };
            bounds::check_picture(&params, &slice_params)?;
            self.accelerator
                .decode_picture(&picture, &params, &slice_params)?;
        }

        // C.5.2.3: the current picture is a short-term reference, waiting
        // for output when PicOutputFlag says so.
        let picture = Arc::new(picture);
        let (x, y, width, height) = sps.crop_rect();
        let output = Output {
            picture: picture.clone(),
            id,
            poc,
            idr: is_idr(unit_type),
            reference: !is_sub_layer_non_reference(unit_type),
            coded_size: (
                sps.pic_width_in_luma_samples,
                sps.pic_height_in_luma_samples,
            ),
            crop: (x, y, width, height),
            full_range: sps.is_full_range(),
            bit_depth: sps.bit_depth_luma(),
        };
        let output_flag = first.pic_output_flag;
        let key = self.next_key;
        self.next_key = key
            .checked_add(1)
            .ok_or(Error::Overflow("the DPB's picture keys"))?;
        let mut entry = Entry {
            key,
            picture,
            poc,
            marking: Marking::ShortTerm,
            membership: None,
            pending: None,
            latency: 0,
        };
        match self.order {
            OutputOrder::Decode => {
                if output_flag {
                    outputs.push(output);
                }
                self.dpb.entries.push(entry);
            }
            OutputOrder::Display => {
                if output_flag {
                    for waiting in &mut self.dpb.entries {
                        if waiting.pending.is_some() && waiting.poc > poc {
                            // PicLatencyCount, saturating as
                            // `Sps::max_latency_pictures` does (C.5.2.3).
                            waiting.latency = waiting.latency.saturating_add(1);
                        }
                    }
                    entry.pending = Some(output);
                }
                self.dpb.entries.push(entry);
                // C.5.2.3's "additional bumping": by the same order and
                // latency, the DPB's fullness aside.
                let reorder = usize::from(sps.highest_ordering().max_num_reorder_pics);
                let latency = sps.max_latency_pictures();
                self.dpb.bump_while(outputs, |dpb| {
                    dpb.pending_count() > reorder || dpb.latency_reached(latency)
                });
            }
        }
        if first.temporal_id == 0
            && !is_rasl(unit_type)
            && !is_radl(unit_type)
            && !is_sub_layer_non_reference(unit_type)
        {
            self.prev_tid0_poc = Some(poc);
        }
        self.new_sequence = false;
        if irap {
            self.damaged = false;
        }
        Ok(())
    }
}

/// `PicOrderCntVal` (8.3.1) of a picture with `slice_pic_order_cnt_lsb`
/// `lsb`, after `prevTid0Pic`'s `previous`; `None` where the picture
/// starts a coded video sequence. In 64 bits nothing overflows: the
/// previous count is in 32 bits and `MaxPicOrderCntLsb` 2^16 at most.
/// Conforming streams keep the count within 32 bits; others saturate, as
/// H.264's order counts do, since only output order follows from it.
fn order_count(previous: Option<i32>, lsb: u32, max_lsb: NonZeroU32) -> Result<i32, Error> {
    let overflow = || Error::Overflow("PicOrderCntMsb");
    let lsb = i64::from(lsb);
    let msb = match previous {
        Some(previous) => {
            let max = i64::from(max_lsb.get());
            let half = max / 2;
            let previous_lsb = i64::from(poc_lsb(previous, max_lsb));
            let previous_msb = i64::from(previous)
                .checked_sub(previous_lsb)
                .ok_or_else(overflow)?;
            let difference = |a: i64, b: i64| a.checked_sub(b).ok_or_else(overflow);
            let msb = if lsb < previous_lsb && difference(previous_lsb, lsb)? >= half {
                previous_msb.checked_add(max)
            } else if lsb > previous_lsb && difference(lsb, previous_lsb)? > half {
                previous_msb.checked_sub(max)
            } else {
                Some(previous_msb)
            };
            msb.ok_or_else(overflow)?
        }
        None => 0,
    };
    let value = msb
        .checked_add(lsb)
        .ok_or(Error::Overflow("PicOrderCntVal"))?;
    Ok(i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX }))
}

/// 8.3.4: `RefPicListTempX` cycles through `groups` (StCurrBefore,
/// StCurrAfter and LtCurr for list 0; StCurrAfter first for list 1) until
/// it has `max(count, NumPicTotalCurr)` entries; the list is its first
/// `count`, or the entries `list_entry_lX` picks.
fn reference_list(
    groups: &[&[usize]; 3],
    count: usize,
    entries: Option<&[u8]>,
) -> Result<Vec<usize>, Error> {
    let total: usize = groups.iter().map(|g| g.len()).sum();
    if total == 0 {
        return Err(Error::MissingReference);
    }
    let mut temp: Vec<usize> = groups
        .iter()
        .flat_map(|g| g.iter().copied())
        .cycle()
        .take(count.max(total))
        .collect();
    match entries {
        Some(entries) => entries
            .iter()
            .take(count)
            .map(|&e| {
                temp.get(usize::from(e))
                    .copied()
                    .ok_or(Error::OutOfRange("list_entry"))
            })
            .collect(),
        None => {
            temp.truncate(count);
            Ok(temp)
        }
    }
}

/// Whether two SPSs make pictures alike: the same size, format and DPB,
/// so that the accelerator's session serves both.
fn same_pictures(a: &Sps, b: &Sps) -> bool {
    a.pic_width_in_luma_samples == b.pic_width_in_luma_samples
        && a.pic_height_in_luma_samples == b.pic_height_in_luma_samples
        && a.chroma_format_idc == b.chroma_format_idc
        && a.bit_depth_luma_minus8 == b.bit_depth_luma_minus8
        && a.bit_depth_chroma_minus8 == b.bit_depth_chroma_minus8
        && a.dpb_size() == b.dpb_size()
}

/// Whether two SPSs differ in their VUI alone, which the decoding process
/// reads only for output.
fn same_but_vui(a: &Sps, b: &Sps) -> bool {
    let without_vui = |sps: &Sps| Sps {
        vui: None,
        ..sps.clone()
    };
    without_vui(a) == without_vui(b)
}

/// Whether [`Decoder`] takes `sps`'s stream: a profile of the Main family
/// (Main, Main 10, Main Still Picture, or a stream compatible with one),
/// 4:2:0 at 8 or 10 bits alike for luma and chroma, without the range or
/// screen content extensions.
pub fn check_supported(sps: &Sps) -> Result<(), Error> {
    let ptl = &sps.profile_tier_level;
    if !matches!(ptl.general_profile_idc, 1..=3) && !(1..=3).any(|p| ptl.compatible_with(p)) {
        return Err(Error::Unsupported(
            "profiles outside Main, Main 10 and Main Still Picture",
        ));
    }
    // A.3.2 to A.3.4: CTBs of 16 to 64 in the Main profiles; the syntax
    // also has 8.
    if !(4..=6).contains(&sps.ctb_log2_size()) {
        return Err(Error::OutOfRange("CtbLog2SizeY"));
    }
    if sps.chroma_format_idc != 1 {
        return Err(Error::Unsupported("chroma formats other than 4:2:0"));
    }
    if sps.bit_depth_luma_minus8 > 2 || sps.bit_depth_chroma_minus8 > 2 {
        return Err(Error::Unsupported("bit depths above 10"));
    }
    if sps.bit_depth_luma_minus8 != sps.bit_depth_chroma_minus8 {
        return Err(Error::Unsupported(
            "luma and chroma of different bit depths",
        ));
    }
    if sps.sps_range_extension_flag {
        return Err(Error::Unsupported("the range extensions"));
    }
    if sps.sps_scc_extension_flag {
        return Err(Error::Unsupported("screen content coding"));
    }
    Ok(())
}

/// Whether [`Decoder`] takes `pps` for pictures of `sps`: without the
/// range or screen content extensions, and with what the PPS's ranges take
/// from the SPS within them.
fn check_pps_supported(pps: &Pps, sps: &Sps) -> Result<(), Error> {
    if pps.pps_range_extension_flag {
        return Err(Error::Unsupported("the range extensions"));
    }
    if pps.pps_scc_extension_flag {
        return Err(Error::Unsupported("screen content coding"));
    }
    // 7.4.3.3: -(26 + QpBdOffsetY) to 25, the parser having taken the
    // widest; QP deltas no finer than the smallest coding block; a merge
    // level within the CTB.
    let lowest = sps
        .qp_bd_offset_y()
        .checked_add(26)
        .and_then(i32::checked_neg)
        .ok_or(Error::Overflow("QpBdOffsetY"))?;
    if i32::from(pps.init_qp_minus26) < lowest {
        return Err(Error::OutOfRange("init_qp_minus26"));
    }
    if pps.diff_cu_qp_delta_depth > sps.log2_diff_max_min_luma_coding_block_size {
        return Err(Error::OutOfRange("diff_cu_qp_delta_depth"));
    }
    if u32::from(pps.log2_parallel_merge_level_minus2) + 2 > sps.ctb_log2_size() {
        return Err(Error::OutOfRange("log2_parallel_merge_level_minus2"));
    }
    // Tile columns of [`MIN_TILE_COLUMN_WIDTH`] luma samples across at
    // least; rows of any height.
    if pps.tiles_enabled_flag {
        let ctb = 1u32 << sps.ctb_log2_size();
        // A width whose samples pass 32 bits is past the bound.
        if pps.column_widths(sps)?.iter().any(|&width| {
            width
                .checked_mul(ctb)
                .is_some_and(|size| size < MIN_TILE_COLUMN_WIDTH)
        }) {
            return Err(Error::OutOfRange("ColumnWidthInLumaSamples"));
        }
    }
    Ok(())
}

/// The narrowest tile column [`Decoder`] takes, in luma samples
/// (`ColumnWidthInLumaSamples`, a column's CTBs times their size), below
/// the 256 the Main profiles allow (A.3.2), and the tile rows of any height
/// below their 64 (A.3.3), as the hardware decodes them. kvazaar 2.3.2's
/// streams of CTBs of 64 with columns of 128 and 192 samples and rows of 16
/// and 64 decoded to every frame of ffmpeg's decode, which kvazaar's MD5
/// SEI and its reconstruction confirmed, on radeonsi (a Radeon 780M, Mesa
/// 26.2.6), iHD (an Iris Xe, intel-media-driver 26.3.5) and NVDEC (an RTX
/// 5080 Laptop GPU, driver 615.71.09), and on Apple's decoders of an x86-64
/// Mac (macOS 26.7.1: an Intel UHD 630, and AMD's driver on a Radeon Pro
/// 5300M); columns of 64, one CTB, came out wrong on every frame, with no
/// error, on radeonsi and on AMD's driver, and right on iHD, NVDEC and the
/// Intel GPU (**measured**, 2026-11-08). No API reports the
/// bound, so it is the same on every driver. The reims-graphics device
/// gives it to the guest, whose video driver refuses a narrower column
/// before it schedules the picture.
pub const MIN_TILE_COLUMN_WIDTH: u32 = 128;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_counts_past_32_bits_saturate_toward_their_side() -> Result<(), Error> {
        // MaxPicOrderCntLsb 16: the lsb 14 after 5 is the cycle below.
        let max = NonZeroU32::new(16).ok_or(Error::Overflow("MaxPicOrderCntLsb"))?;
        assert_eq!(order_count(Some(i32::MIN + 5), 14, max)?, i32::MIN);
        assert_eq!(order_count(Some(i32::MAX - 2), 2, max)?, i32::MAX);
        assert_eq!(order_count(Some(21), 3, max)?, 19);
        assert_eq!(order_count(Some(-3), 15, max)?, -1);
        assert_eq!(order_count(None, 7, max)?, 7);
        Ok(())
    }

    #[test]
    fn lists_cycle_through_the_sets_and_take_their_entries() -> Result<(), Error> {
        let (before, after, long_term) = ([0usize, 1], [2usize], [3usize]);
        let list0 = [&before[..], &after[..], &long_term[..]];
        let list1 = [&after[..], &before[..], &long_term[..]];
        assert_eq!(reference_list(&list0, 2, None)?, [0, 1]);
        assert_eq!(reference_list(&list1, 3, None)?, [2, 0, 1]);
        // More active references than pictures: the sets repeat.
        assert_eq!(reference_list(&list0, 6, None)?, [0, 1, 2, 3, 0, 1]);
        assert_eq!(reference_list(&list0, 2, Some(&[3, 0]))?, [3, 0]);
        assert_eq!(
            reference_list(&list0, 1, Some(&[4])),
            Err(Error::OutOfRange("list_entry"))
        );
        assert_eq!(
            reference_list(&[&[], &[], &[]], 1, None),
            Err(Error::MissingReference)
        );
        Ok(())
    }
}
