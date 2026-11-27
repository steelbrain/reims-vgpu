//! The decoding process above the tiles: the sequence header in force, each temporal unit's
//! frames, the eight reference slots and what each frame shows. What decodes the tiles is an
//! [`Accelerator`], such as VA-API's.
//!
//! AV1 outputs a frame as it is shown: a frame whose `show_frame` is set once decoded, and a
//! slot's frame again where a header's `show_existing_frame` names it. There is no
//! reordering, so a sample's output is the frames its temporal unit shows, in order: one in a
//! conforming stream of one spatial layer.

use std::sync::Arc;

use super::bits::BitReader;
use super::frame::{self, FrameHeader, FrameType, NUM_REF_FRAMES, ReferenceState};
use super::obu::{self, Obu};
use super::sequence::SequenceHeader;
use super::tile_group::TileGroup;
use super::{DecoderConfiguration, Error};

/// What the hardware decodes of AV1: its largest and smallest frames, as the arm's capability
/// query lists them. A sequence header whose largest frame is past them is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The widest frame, in luma samples.
    pub max_width: u32,
    /// The tallest frame, in luma samples.
    pub max_height: u32,
    /// The narrowest frame, in luma samples: 0 where the query lists none.
    pub min_width: u32,
    /// The shortest frame, in luma samples.
    pub min_height: u32,
}

impl Limits {
    /// Frames up to `max_width` by `max_height`, of any smaller size.
    pub const fn up_to(max_width: u32, max_height: u32) -> Self {
        Self {
            max_width,
            max_height,
            min_width: 0,
            min_height: 0,
        }
    }
}

/// A decoded frame, as the decoder emits it.
pub struct Output<P> {
    /// The accelerator's picture holding the samples.
    pub picture: Arc<P>,
    /// The id the sample that showed the frame was decoded with.
    pub id: u64,
    /// Its `frame_type`.
    pub frame_type: FrameType,
    /// Whether a `show_existing_frame` header showed it, decoded by an earlier sample.
    pub shown_existing: bool,
    /// Its size: `UpscaledWidth` by `FrameHeight`, in luma samples.
    pub size: (u32, u32),
    /// `RenderWidth` and `RenderHeight`.
    pub render_size: (u32, u32),
    /// The sequence's `color_range`: full range where set.
    pub full_range: bool,
    /// `BitDepth`.
    pub bit_depth: u8,
    /// Whether the frame's film grain parameters apply grain: where they do, the picture an
    /// accelerator outputs is the frame with grain applied (7.18.3).
    pub film_grain: bool,
}

/// A reference slot's frame, as the accelerator sees it.
pub struct ReferenceFrame<'a, P> {
    /// Its picture.
    pub picture: &'a P,
    /// What the slot holds of it.
    pub state: &'a ReferenceState,
}

/// What a frame is decoded with.
pub struct FrameParams<'a, P> {
    /// The sequence header in force.
    pub sequence: &'a SequenceHeader,
    /// The frame's header.
    pub header: &'a FrameHeader,
    /// The eight reference slots as the frame finds them (`ref_frame_map`): `None` where a slot
    /// is not valid (`RefValid`).
    pub slots: [Option<ReferenceFrame<'a, P>>; NUM_REF_FRAMES],
}

/// What decodes the tiles: the hardware's half of a stateless decoder.
pub trait Accelerator {
    /// A decoded frame's storage, such as a VA surface, and, where the frame applies film
    /// grain, the storage of the frame with grain applied, which is what is output.
    type Picture;
    /// The accelerator's error, which carries this module's errors too.
    type Error: From<Error>;

    /// Prepares for frames of `sequence`. Called before the first frame, and again before the
    /// first frame of a new sequence header.
    fn configure(&mut self, sequence: &SequenceHeader) -> Result<(), Self::Error>;

    /// Storage for the frame `header` starts.
    fn new_picture(&mut self, header: &FrameHeader) -> Result<Self::Picture, Self::Error>;

    /// Decodes one frame from all its tile groups into `picture`.
    fn decode_frame(
        &mut self,
        picture: &Self::Picture,
        params: &FrameParams<'_, Self::Picture>,
        tile_groups: &[TileGroup<'_>],
    ) -> Result<(), Self::Error>;
}

/// Whether this decoder takes `sequence` under `limits`: Main profile (`seq_profile` 0), 4:2:0
/// at 8 or 10 bits and not monochrome, a largest frame within the limits, and operating point
/// 0 of one spatial layer. Others are refused with [`Error::Unsupported`].
pub fn check_supported(sequence: &SequenceHeader, limits: &Limits) -> Result<(), Error> {
    if sequence.seq_profile != 0 {
        return Err(Error::Unsupported(
            "a profile other than Main (4:4:4 or 12 bits)",
        ));
    }
    if sequence.color.mono_chrome {
        return Err(Error::Unsupported("monochrome"));
    }
    if sequence.max_frame_width > limits.max_width || sequence.max_frame_height > limits.max_height
    {
        return Err(Error::Unsupported(
            "frames larger than the hardware decodes",
        ));
    }
    if sequence.max_frame_width < limits.min_width || sequence.max_frame_height < limits.min_height
    {
        return Err(Error::Unsupported(
            "frames smaller than the hardware decodes",
        ));
    }
    // More than one of the spatial layer bits.
    if (sequence.operating_point_idc() >> 8).count_ones() > 1 {
        return Err(Error::Unsupported("spatial layers"));
    }
    Ok(())
}

/// What [`check_supported`] refuses of a stream that an `av1C` record alone tells, for a record
/// whose `configOBUs` hold no sequence header: a profile other than Main, or monochrome.
pub fn check_record(record: &DecoderConfiguration<'_>) -> Result<(), Error> {
    if record.seq_profile != 0 {
        return Err(Error::Unsupported(
            "a profile other than Main (4:4:4 or 12 bits)",
        ));
    }
    if record.monochrome {
        return Err(Error::Unsupported("monochrome"));
    }
    Ok(())
}

/// The sequence header in force, and the bytes it was read from.
struct Active {
    header: Arc<SequenceHeader>,
    bytes: Vec<u8>,
    configured: bool,
}

/// A frame whose header has been read and whose tile groups are arriving (`SeenFrameHeader`).
struct Pending<'a> {
    header: FrameHeader,
    tile_groups: Vec<TileGroup<'a>>,
    /// The tile the next group starts at.
    next_tile: u32,
}

/// An AV1 decoder over an [`Accelerator`], one temporal unit per sample.
///
/// Main profile streams are decoded: 4:2:0 at 8 or 10 bits, with every tool of the profile,
/// superres, film grain where the accelerator applies it, and temporal layers. Other
/// profiles, monochrome and spatial layers are refused ([`check_supported`]), as are large
/// scale tile streams (tile list OBUs).
pub struct Decoder<A: Accelerator> {
    accelerator: A,
    limits: Limits,
    active: Option<Active>,
    slots: [ReferenceState; NUM_REF_FRAMES],
    pictures: [Option<Arc<A::Picture>>; NUM_REF_FRAMES],
    /// The frames decoded, shown or not, for the logs.
    frames: u64,
    /// What the accelerator is handed of a frame's film grain in place of what the stream
    /// coded: nothing changed (`None`), always, but for the card tests' decodes of a stream
    /// without its grain or with grain of other parameters (`alter_film_grain`).
    film_grain: Option<fn(&mut frame::FilmGrainParams)>,
}

impl<A: Accelerator> Decoder<A> {
    /// A decoder of frames up to `limits`.
    pub fn new(accelerator: A, limits: Limits) -> Self {
        Self {
            accelerator,
            limits,
            active: None,
            slots: Default::default(),
            pictures: Default::default(),
            frames: 0,
            film_grain: None,
        }
    }

    /// Has `alter` change the film grain parameters of each frame that applies grain before
    /// the accelerator is handed them, the slots keeping what the stream coded: the card tests'
    /// decodes of a stream without its grain, so that what is output is the decoded frame
    /// alone, and with grain of other parameters, which their grain check must refuse.
    #[cfg(test)]
    pub(crate) fn alter_film_grain(&mut self, alter: fn(&mut frame::FilmGrainParams)) {
        self.film_grain = Some(alter);
    }

    /// The accelerator.
    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    /// The accelerator, mutably.
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }

    /// The sequence header in force, once one has been read.
    pub fn sequence(&self) -> Option<&SequenceHeader> {
        self.active.as_ref().map(|active| &*active.header)
    }

    /// The frames decoded so far, shown or not.
    pub fn frames_decoded(&self) -> u64 {
        self.frames
    }

    /// Reads the sequence header of an `av1C` record's `configOBUs`, where it has one, so that
    /// the first sample need not carry it.
    pub fn configure(&mut self, record: &DecoderConfiguration<'_>) -> Result<(), A::Error> {
        for obu in obu::obus(record.config_obus) {
            let obu = obu?;
            if obu.obu_type == obu::SEQUENCE_HEADER {
                self.sequence_header(obu.payload)?;
            }
        }
        Ok(())
    }

    /// Forgets every reference slot, as a seek does: decoding starts again at the next key
    /// frame shown at once. The sequence header in force is kept.
    pub fn reset(&mut self) {
        self.slots = Default::default();
        self.pictures = Default::default();
    }

    /// Decodes one sample, a temporal unit, as `id`: the frames it shows.
    ///
    /// Where the sample fails, the slots of the frame that failed are emptied, every slot
    /// where its header could not be read, so that the frames after it that predict from
    /// them fail with [`Error::MissingReference`] until a key frame refreshes them.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Output<A::Picture>>, A::Error> {
        let mut outputs = Vec::new();
        // The slots the frame in progress would refresh: all of them until its header reads.
        let mut refreshing = 0xffu8;
        let result = self.decode_obus(sample, id, &mut outputs, &mut refreshing);
        if result.is_err() {
            for (i, (slot, picture)) in self.slots.iter_mut().zip(&mut self.pictures).enumerate() {
                if (refreshing >> i) & 1 == 1 {
                    slot.valid = false;
                    *picture = None;
                }
            }
        }
        result.map(|()| outputs)
    }

    fn decode_obus<'s>(
        &mut self,
        sample: &'s [u8],
        id: u64,
        outputs: &mut Vec<Output<A::Picture>>,
        refreshing: &mut u8,
    ) -> Result<(), A::Error> {
        let mut pending: Option<Pending<'s>> = None;
        for obu in obu::obus(sample) {
            let obu = obu?;
            let idc = self
                .sequence()
                .map_or(0, SequenceHeader::operating_point_idc);
            if obu.dropped_by(idc) {
                continue;
            }
            match obu.obu_type {
                obu::SEQUENCE_HEADER => {
                    if pending.is_some() {
                        return Err(Error::IncompleteFrame.into());
                    }
                    self.sequence_header(obu.payload)?;
                }
                obu::TEMPORAL_DELIMITER => {
                    if pending.is_some() {
                        return Err(Error::IncompleteFrame.into());
                    }
                }
                obu::FRAME_HEADER | obu::REDUNDANT_FRAME_HEADER => {
                    // frame_header_copy(): a frame's header again, or a redundant one of a
                    // frame already decoded, which this decoder needs neither of.
                    if pending.is_some() || obu.obu_type == obu::REDUNDANT_FRAME_HEADER {
                        continue;
                    }
                    *refreshing = 0xff;
                    let header = self.frame_header(&mut BitReader::new(obu.payload), &obu)?;
                    *refreshing = header.refresh_frame_flags;
                    if header.show_existing_frame {
                        self.show_existing(&header, id, outputs)?;
                        *refreshing = 0;
                    } else {
                        pending = Some(Pending {
                            header,
                            tile_groups: Vec::new(),
                            next_tile: 0,
                        });
                    }
                }
                obu::FRAME => {
                    if pending.is_some() {
                        return Err(Error::IncompleteFrame.into());
                    }
                    *refreshing = 0xff;
                    let mut r = BitReader::new(obu.payload);
                    let header = self.frame_header(&mut r, &obu)?;
                    *refreshing = header.refresh_frame_flags;
                    // 6.8.2: a frame OBU's header does not show an existing frame.
                    if header.show_existing_frame {
                        return Err(Error::OutOfRange("show_existing_frame").into());
                    }
                    r.byte_alignment()?;
                    let rest = obu
                        .payload
                        .get(r.byte_position()..)
                        .ok_or(Error::Truncated)?;
                    let mut frame = Pending {
                        header,
                        tile_groups: Vec::new(),
                        next_tile: 0,
                    };
                    if self.tile_group(&mut frame, rest)? {
                        self.decode_frame(frame, id, outputs)?;
                        *refreshing = 0;
                    } else {
                        pending = Some(frame);
                    }
                }
                obu::TILE_GROUP => {
                    let mut frame = pending.take().ok_or(Error::NoFrameHeader)?;
                    if self.tile_group(&mut frame, obu.payload)? {
                        self.decode_frame(frame, id, outputs)?;
                        *refreshing = 0;
                    } else {
                        pending = Some(frame);
                    }
                }
                obu::TILE_LIST => {
                    return Err(Error::Unsupported("large scale tile (tile list OBUs)").into());
                }
                // Metadata, padding and the reserved types.
                _ => {}
            }
        }
        if pending.is_some() {
            return Err(Error::IncompleteFrame.into());
        }
        Ok(())
    }

    /// A sequence header OBU's payload: the same bytes again change nothing; a new header is
    /// checked against what this decoder takes, and the reference slots are emptied, as a
    /// new coded video sequence starts at a key frame shown at once (7.5).
    fn sequence_header(&mut self, payload: &[u8]) -> Result<(), A::Error> {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.bytes == payload)
        {
            return Ok(());
        }
        let header = SequenceHeader::parse(payload)?;
        check_supported(&header, &self.limits)?;
        if self
            .active
            .as_ref()
            .is_some_and(|active| *active.header == header)
        {
            return Ok(());
        }
        self.reset();
        self.active = Some(Active {
            header: Arc::new(header),
            bytes: payload.to_vec(),
            configured: false,
        });
        Ok(())
    }

    fn frame_header(&self, r: &mut BitReader<'_>, obu: &Obu<'_>) -> Result<FrameHeader, Error> {
        let active = self.active.as_ref().ok_or(Error::NoSequenceHeader)?;
        frame::parse(r, &active.header, &self.slots, obu.extension)
    }

    /// Adds the tile group `data` holds to `frame`; whether it holds the frame's last tile.
    fn tile_group<'s>(&self, frame: &mut Pending<'s>, data: &'s [u8]) -> Result<bool, Error> {
        let group = TileGroup::parse(data, &frame.header.tile_info)?;
        if group.start != frame.next_tile {
            return Err(Error::TileGroupOutOfOrder);
        }
        frame.next_tile = group.end.saturating_add(1);
        frame.tile_groups.push(group);
        Ok(frame.next_tile == frame.header.tile_info.num_tiles())
    }

    /// `show_existing_frame`: outputs the slot's frame; a key frame shown so refreshes every
    /// slot with itself (7.21).
    fn show_existing(
        &mut self,
        header: &FrameHeader,
        id: u64,
        outputs: &mut Vec<Output<A::Picture>>,
    ) -> Result<(), A::Error> {
        let index = usize::from(header.frame_to_show_map_idx);
        let picture = self
            .pictures
            .get(index)
            .cloned()
            .flatten()
            .ok_or(Error::MissingReference)?;
        let state = self.slots.get(index).cloned().unwrap_or_default();
        outputs.push(self.output(&picture, id, header, &state.size, true));
        if header.frame_type == FrameType::Key {
            let state = ReferenceState {
                showable: false,
                ..state
            };
            for (slot, stored) in self.slots.iter_mut().zip(&mut self.pictures) {
                *slot = state.clone();
                *stored = Some(Arc::clone(&picture));
            }
        }
        Ok(())
    }

    fn output(
        &self,
        picture: &Arc<A::Picture>,
        id: u64,
        header: &FrameHeader,
        size: &frame::FrameSize,
        shown_existing: bool,
    ) -> Output<A::Picture> {
        let color = self.sequence().map(|s| s.color).unwrap_or_default();
        Output {
            picture: Arc::clone(picture),
            id,
            frame_type: header.frame_type,
            shown_existing,
            size: (size.upscaled_width, size.frame_height),
            render_size: (size.render_width, size.render_height),
            full_range: color.color_range,
            bit_depth: color.bit_depth,
            film_grain: header.film_grain.apply_grain,
        }
    }

    /// Decodes a frame whose tiles have all arrived, then runs the reference frame update
    /// process (7.20) and outputs it where it is shown.
    fn decode_frame(
        &mut self,
        frame: Pending<'_>,
        id: u64,
        outputs: &mut Vec<Output<A::Picture>>,
    ) -> Result<(), A::Error> {
        let header = frame.header;
        let active = self.active.as_mut().ok_or(Error::NoSequenceHeader)?;
        let sequence = Arc::clone(&active.header);
        if !active.configured {
            self.accelerator.configure(&sequence)?;
            active.configured = true;
        }
        // The slots a shown key frame, frame ids or ref_order_hint left invalid.
        for ((slot, picture), (&valid, &hint)) in self
            .slots
            .iter_mut()
            .zip(&mut self.pictures)
            .zip(header.ref_valid.iter().zip(&header.ref_order_hint))
        {
            if !valid {
                slot.valid = false;
                slot.order_hint = hint;
                *picture = None;
            }
        }
        // The header the accelerator is handed; the slots keep the one the stream coded.
        let altered;
        let handed = match self.film_grain {
            Some(alter) if header.film_grain.apply_grain => {
                let mut grain = header.film_grain;
                alter(&mut grain);
                altered = FrameHeader {
                    film_grain: grain,
                    ..header.clone()
                };
                &altered
            }
            _ => &header,
        };
        let picture = self.accelerator.new_picture(handed)?;
        let mut slots: [Option<ReferenceFrame<'_, A::Picture>>; NUM_REF_FRAMES] =
            Default::default();
        for ((reference, state), picture) in slots.iter_mut().zip(&self.slots).zip(&self.pictures) {
            if let (true, Some(picture)) = (state.valid, picture) {
                *reference = Some(ReferenceFrame {
                    picture: &**picture,
                    state,
                });
            }
        }
        let params = FrameParams {
            sequence: &sequence,
            header: handed,
            slots,
        };
        self.accelerator
            .decode_frame(&picture, &params, &frame.tile_groups)?;
        self.frames = self.frames.saturating_add(1);
        let picture = Arc::new(picture);
        let state = ReferenceState {
            valid: true,
            frame_id: header.current_frame_id,
            frame_type: header.frame_type,
            showable: header.showable_frame,
            order_hint: header.order_hint,
            size: header.size,
            bit_depth: sequence.color.bit_depth,
            loop_filter_ref_deltas: header.loop_filter.ref_deltas,
            loop_filter_mode_deltas: header.loop_filter.mode_deltas,
            segmentation: header.segmentation.features,
            gm_params: header.global_motion.map(|motion| motion.params),
            film_grain: header.film_grain,
        };
        for (i, (slot, stored)) in self.slots.iter_mut().zip(&mut self.pictures).enumerate() {
            if (header.refresh_frame_flags >> i) & 1 == 1 {
                *slot = state.clone();
                *stored = Some(Arc::clone(&picture));
            }
        }
        if header.show_frame {
            outputs.push(self.output(&picture, id, handed, &header.size, false));
        }
        Ok(())
    }
}
