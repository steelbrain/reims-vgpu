//! HEVC's syntax and the decoding process above the slice data, for a
//! stateless accelerator: NAL units in `hvcC` and Annex B form, parameter
//! sets with their scaling lists and reference picture sets, slice segment
//! headers, picture order counts, the reference picture set's marking, the
//! reference lists and the DPB's bumping (ITU-T H.265, by clause).
//!
//! This module is plain Rust and runs on any host, so that its tests run
//! on a Mac too. What decodes the slice data is an [`Accelerator`], which
//! the VA-API and NVIDIA crates implement over their hardware. It shares
//! H.264's bit reader, NAL unit framing and parameter set storage, which
//! are the same in both standards, and refuses with errors of its own;
//! [`crate::codec::h264::sei`] reads both standards' SEI messages.

pub(crate) mod bits;
mod bounds;
mod decoder;
mod dpb;
pub mod hvcc;
pub mod nal;
pub mod pps;
pub mod rps;
pub mod scaling;
pub mod slice;
pub mod sps;
#[cfg(test)]
mod tests;

pub use crate::codec::h264::{NalFormat, OutputOrder};
pub use decoder::{
    Accelerator, Decoder, Output, PictureParams, ReferencePicture, RpsMembership, SliceParams,
    check_supported,
};

/// Why a stream could not be read or decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The data ended inside a syntax element.
    Truncated,
    /// A syntax element, or a variable the standard derives from them,
    /// named, is outside what the standard allows.
    OutOfRange(&'static str),
    /// A variable the standard derives from the stream's values, named,
    /// past what its type holds: no conforming stream makes it.
    Overflow(&'static str),
    /// A NAL unit's length prefix runs past the end of its sample.
    BadLengthPrefix,
    /// An `hvcC` record shorter than its fixed fields, before its arrays.
    HvccTruncated,
    /// An `hvcC` record whose `configurationVersion` is not 1.
    HvccVersion,
    /// An `hvcC` record's array whose fields or NAL units run past the
    /// record.
    HvccArrayPastEnd,
    /// A slice segment names a PPS, by its id, that has not arrived.
    MissingPps(u8),
    /// A PPS names an SPS, by its id, that has not arrived.
    MissingSps(u8),
    /// A slice predicts from a picture the decoder does not have, as after
    /// starting or resetting at a picture that is not a random access
    /// point.
    MissingReference,
    /// A feature this decoder does not implement, named.
    Unsupported(&'static str),
    /// A picture after one that did not decode, before an IRAP picture
    /// has: Apple's HEVC decoder decodes none until then, a picture of I
    /// slices alone included. This decoder does not decode it either,
    /// since nothing after the IRAP picture predicts from it (8.3.2).
    Unrecovered,
    /// A RASL picture of an IRAP picture with `NoRaslOutputFlag` 1
    /// (8.1.3): one the decoder started at, a BLA picture, or a CRA
    /// picture that ended [`Self::Unrecovered`]. It predicts from pictures
    /// the decoder does not have, so it is neither decoded nor output, and
    /// Apple's HEVC decoder answers it with
    /// `kVTVideoDecoderReferenceMissingErr` and
    /// `kVTDecodeInfo_SkippedLeadingFrameDropped`.
    SkippedLeadingPicture,
    /// A picture of more slice segments than CTBs: each holds one at least
    /// (7.3.8.1).
    MoreSliceSegmentsThanCtbs,
    /// A picture's first slice segment is not one with
    /// `first_slice_segment_in_pic_flag` (7.4.7.1).
    NoFirstSliceSegment,
    /// A dependent slice segment with no independent one before it in its
    /// picture, whose fields it would take (7.4.7.1).
    NoIndependentSliceSegment,
    /// A picture other than an IRAP picture activates an SPS that changes
    /// more than the VUI (7.4.3.2.1).
    SpsChangedAtNonIrap,
    /// A picture that does not start a coded video sequence activates an
    /// SPS that changes what the pictures need (7.4.3.2.1).
    SpsChangedWithinSequence,
    /// A slice segment whose SPS differs from its picture's first one's
    /// (7.4.2.4.2).
    SpsChangedInPicture,
    /// A slice segment whose PPS differs from its picture's first one's
    /// (7.4.7.1).
    PpsChangedInPicture,
    /// Slice segments of a picture that disagree on what is the same in
    /// all of them: the NAL unit type, the order count's LSBs, the
    /// reference picture set and the like (7.4.2.2, 7.4.7.1).
    SliceSegmentsDisagree,
    /// More reference pictures than the DPB holds beside the current one
    /// (C.5.2.2).
    TooManyReferences,
    /// A reference picture with the current picture's `PicOrderCntVal`
    /// (8.3.1).
    ReferenceAtCurrentPoc,
    /// A slice segment whose header nothing follows:
    /// `slice_segment_data()` holds a CTU at least (7.3.8.1).
    EmptySliceSegment,
    /// A reference list of more than 15 entries, or one naming a picture
    /// the accelerator is not given (7.4.7.1).
    BadReferenceList,
    /// Two slice segments of a picture at one address (7.4.7.1).
    OverlappingSliceSegments,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "the bitstream ends inside a syntax element"),
            Self::OutOfRange(name) => write!(f, "{name} is out of range"),
            Self::Overflow(name) => write!(f, "{name} overflows"),
            Self::BadLengthPrefix => write!(f, "a NAL unit's length runs past its sample"),
            Self::HvccTruncated => write!(f, "bad hvcC record: truncated"),
            Self::HvccVersion => write!(f, "bad hvcC record: configurationVersion is not 1"),
            Self::HvccArrayPastEnd => write!(f, "bad hvcC record: an array runs past the end"),
            Self::MissingPps(id) => write!(f, "PPS {id} has not been received"),
            Self::MissingSps(id) => write!(f, "SPS {id} has not been received"),
            Self::MissingReference => write!(f, "a reference picture is missing"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Unrecovered => write!(
                f,
                "no IRAP picture has decoded since a picture that did not decode"
            ),
            Self::SkippedLeadingPicture => write!(
                f,
                "a RASL picture whose IRAP picture starts the decoding afresh is skipped"
            ),
            Self::MoreSliceSegmentsThanCtbs => write!(f, "more slice segments than CTBs"),
            Self::NoFirstSliceSegment => write!(f, "a picture without its first slice segment"),
            Self::NoIndependentSliceSegment => write!(
                f,
                "a dependent slice segment with no independent one before it"
            ),
            Self::SpsChangedAtNonIrap => write!(f, "the SPS changed at a non-IRAP picture"),
            Self::SpsChangedWithinSequence => {
                write!(f, "the SPS changed within a coded video sequence")
            }
            Self::SpsChangedInPicture => write!(f, "the SPS changed within a picture"),
            Self::PpsChangedInPicture => write!(f, "the PPS changed within a picture"),
            Self::SliceSegmentsDisagree => {
                write!(f, "slice segments of a picture that disagree")
            }
            Self::TooManyReferences => {
                write!(f, "references past sps_max_dec_pic_buffering_minus1")
            }
            Self::ReferenceAtCurrentPoc => {
                write!(f, "a reference with the current PicOrderCntVal")
            }
            Self::EmptySliceSegment => write!(f, "a slice segment with no data"),
            Self::BadReferenceList => write!(f, "a reference list past its bounds"),
            Self::OverlappingSliceSegments => write!(f, "two slice segments at one address"),
        }
    }
}

impl std::error::Error for Error {}

/// The NAL framing H.264 and HEVC share refuses a sample so.
impl From<crate::codec::h264::nal::BadLengthPrefix> for Error {
    fn from(_: crate::codec::h264::nal::BadLengthPrefix) -> Self {
        Self::BadLengthPrefix
    }
}
