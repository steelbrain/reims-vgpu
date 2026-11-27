//! The decoding process above the compressed header: each sample's frames,
//! their headers against the state the frames before them left, the eight
//! reference slots, and which frames come out. What decodes a frame's
//! compressed header and tiles is an [`Accelerator`].

use std::sync::Arc;

use super::header::{self, FrameHeader, State};
use super::quant::{self, SegmentParams};
use super::superframe;
use super::{Error, config::VpCodecConfiguration};

/// The largest side this decoder takes, past what any lab driver decodes
/// (8192 on the AMD cards and the 3090, 16384 on the Iris Xe) and far
/// below VP9's 65536: a frame past it is [`Error::TooLarge`] before an
/// accelerator sees it. Each accelerator holds frames to its driver's own
/// limits when it is configured.
pub const MAX_SIDE: u32 = 16384;

/// What an accelerator decodes a run of frames as: their profile, sample
/// format and size. A frame of another is decoded after a new
/// [`Accelerator::configure`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    /// `Profile`: 0 or 2 here ([`check_supported`]).
    pub profile: u8,
    /// `BitDepth`: 8 or 10 here.
    pub bit_depth: u8,
    /// `FrameWidth`.
    pub width: u32,
    /// `FrameHeight`.
    pub height: u32,
}

/// A frame in a reference slot, as an accelerator sees it.
pub struct Reference<'a, P> {
    /// Its picture.
    pub picture: &'a P,
    /// Its `FrameWidth`.
    pub width: u32,
    /// Its `FrameHeight`.
    pub height: u32,
}

/// What one frame's decode is given.
pub struct FrameParams<'a, P> {
    /// The frame's uncompressed header.
    pub header: &'a FrameHeader,
    /// Each segment's derived quantizer scales and filter levels.
    pub segments: &'a [SegmentParams; header::MAX_SEGMENTS],
    /// The eight reference slots, `None` where a slot holds no frame.
    pub references: [Option<Reference<'a, P>>; 8],
    /// The frame's bytes, the uncompressed header first.
    pub frame: &'a [u8],
}

/// What decodes a frame's compressed header and tiles: the hardware's
/// half of a stateless decoder.
pub trait Accelerator {
    /// A decoded frame's storage, such as a VA surface.
    type Picture;
    /// The accelerator's error, which carries this module's errors too.
    type Error: From<Error>;

    /// Prepares for frames of `format`: called before the first frame, and
    /// again before a frame of another format or size. Frames decoded
    /// before stay referenced by the slots that hold them.
    fn configure(&mut self, format: &Format) -> Result<(), Self::Error>;

    /// Storage for the frame about to decode, of the configured format.
    fn new_picture(&mut self) -> Result<Self::Picture, Self::Error>;

    /// Decodes one frame into `picture`.
    fn decode_frame(
        &mut self,
        picture: &Self::Picture,
        params: &FrameParams<'_, Self::Picture>,
    ) -> Result<(), Self::Error>;
}

/// A frame as the decoder emits it: each frame shown, in decode order,
/// which is VP9's display order.
pub struct Output<P> {
    /// The accelerator's picture holding the samples.
    pub picture: Arc<P>,
    /// The id the sample that carried the frame was decoded with.
    pub id: u64,
    /// Whether the frame is a key frame.
    pub key_frame: bool,
    /// Whether a slot holds it, for later frames to predict from: a frame
    /// that refreshes a slot, or one a slot shows.
    pub reference: bool,
    /// `FrameWidth` and `FrameHeight`.
    pub size: (u32, u32),
    /// `BitDepth`.
    pub bit_depth: u8,
    /// `color_range`.
    pub full_range: bool,
}

/// A frame a slot holds.
struct Slot<P> {
    picture: Arc<P>,
    key_frame: bool,
    width: u32,
    height: u32,
    color: header::ColorConfig,
}

/// A VP9 decoder over an [`Accelerator`], one sample (a frame or a
/// superframe) per call.
///
/// A sample that does not decode leaves the slots and the header's state
/// as the frames before it left them, and the decoder takes the next
/// sample as it would any: one that predicts from a slot no frame filled
/// is [`Error::MissingReference`], until a key frame fills them.
pub struct Decoder<A: Accelerator> {
    accelerator: A,
    slots: [Option<Slot<A::Picture>>; 8],
    state: State,
    /// The format the accelerator is configured for.
    active: Option<Format>,
}

impl<A: Accelerator> Decoder<A> {
    /// A decoder with no frame yet.
    pub fn new(accelerator: A) -> Self {
        Self {
            accelerator,
            slots: Default::default(),
            state: State::default(),
            active: None,
        }
    }

    /// The accelerator.
    pub fn accelerator(&self) -> &A {
        &self.accelerator
    }

    /// The accelerator, to change its settings.
    pub fn accelerator_mut(&mut self) -> &mut A {
        &mut self.accelerator
    }

    /// Forgets every slot and the header's state, as for a seek: the next
    /// frame decodes only if it is a key frame.
    pub fn reset(&mut self) {
        self.slots = Default::default();
        self.state = State::default();
    }

    /// Decodes `sample`, a frame or a superframe, tagged `id`, and answers
    /// the frames it shows: one for a sample a container holds, none for a
    /// sample of hidden frames alone.
    pub fn decode(&mut self, sample: &[u8], id: u64) -> Result<Vec<Output<A::Picture>>, A::Error> {
        let frames = superframe::frames(sample)?;
        let mut outputs = Vec::new();
        for frame in frames {
            if let Some(output) = self.decode_frame(frame, id)? {
                outputs.push(output);
            }
        }
        Ok(outputs)
    }

    /// Decodes one frame, and answers it where it is shown.
    fn decode_frame(
        &mut self,
        frame: &[u8],
        id: u64,
    ) -> Result<Option<Output<A::Picture>>, A::Error> {
        let ref_sizes = self
            .slots
            .each_ref()
            .map(|slot| slot.as_ref().map(|s| (s.width, s.height)));
        let (header, state) = header::parse(frame, &self.state, &ref_sizes)?;
        if let Some(index) = header.show_existing_frame {
            let slot = self
                .slots
                .get(usize::from(index))
                .and_then(Option::as_ref)
                .ok_or(Error::MissingReference)?;
            return Ok(Some(Output {
                picture: Arc::clone(&slot.picture),
                id,
                key_frame: slot.key_frame,
                reference: true,
                size: (slot.width, slot.height),
                bit_depth: slot.color.bit_depth,
                full_range: slot.color.full_range,
            }));
        }
        let format = check_frame(&header)?;
        if !header.is_intra() {
            self.check_references(&header)?;
        }
        if self.active != Some(format) {
            self.active = None;
            self.accelerator.configure(&format)?;
            self.active = Some(format);
        }
        let picture = self.accelerator.new_picture()?;
        let segments = quant::segment_params(&header);
        let references = self.slots.each_ref().map(|slot| {
            slot.as_ref().map(|slot| Reference {
                picture: &*slot.picture,
                width: slot.width,
                height: slot.height,
            })
        });
        let params = FrameParams {
            header: &header,
            segments: &segments,
            references,
            frame,
        };
        self.accelerator.decode_frame(&picture, &params)?;
        self.state = state;
        let picture = Arc::new(picture);
        for (bit, slot) in self.slots.iter_mut().enumerate() {
            if (header.refresh_frame_flags >> bit) & 1 == 1 {
                *slot = Some(Slot {
                    picture: Arc::clone(&picture),
                    key_frame: header.key_frame,
                    width: header.width,
                    height: header.height,
                    color: header.color,
                });
            }
        }
        Ok(header.show_frame.then(|| Output {
            picture,
            id,
            key_frame: header.key_frame,
            reference: header.refresh_frame_flags != 0,
            size: (header.width, header.height),
            bit_depth: header.color.bit_depth,
            full_range: header.color.full_range,
        }))
    }

    /// An inter frame's three references: each slot holds a frame of the
    /// frame's bit depth and subsampling, of a size it scales from
    /// (8.5.2.3: twice the frame's at most, a sixteenth at least).
    fn check_references(&self, header: &FrameHeader) -> Result<(), Error> {
        for &index in &header.ref_frame_idx {
            let slot = self
                .slots
                .get(usize::from(index))
                .and_then(Option::as_ref)
                .ok_or(Error::MissingReference)?;
            let same_format = slot.color.bit_depth == header.color.bit_depth
                && slot.color.subsampling_x == header.color.subsampling_x
                && slot.color.subsampling_y == header.color.subsampling_y;
            let scales = |frame: u32, reference: u32| {
                u64::from(frame).saturating_mul(2) >= u64::from(reference)
                    && u64::from(frame) <= u64::from(reference).saturating_mul(16)
            };
            if !same_format
                || !scales(header.width, slot.width)
                || !scales(header.height, slot.height)
            {
                return Err(Error::BadReference);
            }
        }
        Ok(())
    }
}

/// A frame's format, where this decoder takes it: profile 0 or 2, 4:2:0 at
/// 8 or 10 bits, no side past [`MAX_SIDE`].
fn check_frame(header: &FrameHeader) -> Result<Format, Error> {
    let color = &header.color;
    if !(color.subsampling_x && color.subsampling_y) {
        return Err(Error::Unsupported("VP9 other than 4:2:0"));
    }
    match (header.profile, color.bit_depth) {
        (0, 8) | (2, 10) => {}
        (2, _) => return Err(Error::Unsupported("VP9 at 12 bits")),
        _ => return Err(Error::Unsupported("VP9 profiles 1 and 3")),
    }
    if header.width > MAX_SIDE || header.height > MAX_SIDE {
        return Err(Error::TooLarge);
    }
    Ok(Format {
        profile: header.profile,
        bit_depth: color.bit_depth,
        width: header.width,
        height: header.height,
    })
}

/// Whether this decoder takes a stream of `config`'s record: profile 0 at
/// 8 bits or profile 2 at 10, 4:2:0. What a frame's own header says is
/// checked again at each frame.
pub fn check_supported(config: &VpCodecConfiguration) -> Result<(), Error> {
    if !config.is_4_2_0() {
        return Err(Error::Unsupported("VP9 other than 4:2:0"));
    }
    match (config.profile, config.bit_depth) {
        (0, 8) | (2, 10) => Ok(()),
        (2, _) => Err(Error::Unsupported("VP9 at 12 bits")),
        _ => Err(Error::Unsupported("VP9 profiles 1 and 3")),
    }
}
