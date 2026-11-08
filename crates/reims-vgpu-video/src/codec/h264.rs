//! H.264's syntax and the decoding process above the slice data, for a
//! stateless accelerator: NAL units in `avcC` and Annex B form, parameter
//! sets with their scaling lists, slice headers, order counts, the DPB's
//! marking, lists and bumping (ITU-T H.264, 08/2021, by clause).
//!
//! This module is plain Rust and runs on any host, so that its tests run
//! on a Mac too. What decodes the slice data is an [`Accelerator`], which
//! the VA-API and NVIDIA crates implement over their hardware.

pub mod avcc;
pub mod bits;
mod bounds;
mod decoder;
mod dpb;
pub mod nal;
mod poc;
pub mod pps;
pub mod scaling;
pub mod sei;
pub mod slice;
pub mod sps;
#[cfg(test)]
mod tests;

pub(crate) use decoder::ParameterSets;
pub use decoder::{
    Accelerator, Decoder, Output, OutputOrder, PictureParams, ReferenceFrame, SliceParams,
    check_supported,
};
pub use nal::NalFormat;

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
    /// An `avcC` record that ends inside a field.
    AvccTruncated,
    /// An `avcC` record whose `configurationVersion` is not 1.
    AvccVersion,
    /// An `avcC` record's parameter set whose length runs past the record.
    AvccParameterSetPastEnd,
    /// A slice names a PPS, by its id, that has not arrived.
    MissingPps(u8),
    /// A PPS names an SPS, by its id, that has not arrived.
    MissingSps(u8),
    /// A slice predicts from a frame the decoder does not have, as after
    /// starting or resetting at a non-I picture.
    MissingReference,
    /// A feature this decoder does not implement, named.
    Unsupported(&'static str),
    /// The SPS lets pictures be fields (`frame_mbs_only_flag` 0). Apple's
    /// AVC decoder decodes none of its pictures, field, MBAFF or frame
    /// alike, and neither does this one.
    Interlaced,
    /// A picture that is not all I slices, after one that did not decode
    /// and before an I picture has: Apple's AVC decoder decodes none until
    /// then. This decoder decodes it, so that its references go on, but
    /// gives no frame for it.
    Unrecovered,
    /// A picture of more slices than macroblocks: each slice holds one at
    /// least (7.3.4).
    MoreSlicesThanMacroblocks,
    /// A picture other than an IDR picture activates an SPS that changes
    /// more than the VUI: the active SPS changes only at an IDR picture
    /// (7.4.1.2.1).
    SpsChangedAtNonIdr,
    /// A slice whose SPS or PPS differs from its picture's first slice's
    /// (7.4.1.2.1).
    ParameterSetsChangedInPicture,
    /// More frames marked as references than `max_num_ref_frames`
    /// (7.4.3.3).
    TooManyReferences,
    /// A slice whose header nothing follows: `slice_data()` holds a
    /// macroblock at least (7.3.4).
    EmptySlice,
    /// Two slices of a picture that start at one macroblock (7.4.3).
    OverlappingSlices,
    /// A reference list of another length than its slice's
    /// `num_ref_idx_lX_active_minus1 + 1`, or of more than 32 entries
    /// (7.4.3).
    ReferenceListLength,
    /// A reference list names a frame the DPB does not hold as a
    /// reference.
    ReferenceOutsideDpb,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "the bitstream ends inside a syntax element"),
            Self::OutOfRange(name) => write!(f, "{name} is out of range"),
            Self::Overflow(name) => write!(f, "{name} overflows"),
            Self::BadLengthPrefix => write!(f, "a NAL unit's length runs past its sample"),
            Self::AvccTruncated => write!(f, "bad avcC record: truncated"),
            Self::AvccVersion => write!(f, "bad avcC record: configurationVersion is not 1"),
            Self::AvccParameterSetPastEnd => {
                write!(f, "bad avcC record: parameter set runs past the end")
            }
            Self::MissingPps(id) => write!(f, "PPS {id} has not been received"),
            Self::MissingSps(id) => write!(f, "SPS {id} has not been received"),
            Self::MissingReference => write!(f, "a reference frame is missing"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Interlaced => write!(f, "interlaced: the SPS's frame_mbs_only_flag is 0"),
            Self::Unrecovered => write!(
                f,
                "no I picture has decoded since a picture that did not decode"
            ),
            Self::MoreSlicesThanMacroblocks => write!(f, "more slices than macroblocks"),
            Self::SpsChangedAtNonIdr => write!(f, "the SPS changed at a non-IDR picture"),
            Self::ParameterSetsChangedInPicture => {
                write!(f, "the parameter sets changed within a picture")
            }
            Self::TooManyReferences => write!(f, "reference frames past max_num_ref_frames"),
            Self::EmptySlice => write!(f, "a slice with no slice data"),
            Self::OverlappingSlices => write!(f, "two slices at one macroblock"),
            Self::ReferenceListLength => write!(f, "a reference list of the wrong length"),
            Self::ReferenceOutsideDpb => write!(f, "a reference outside the DPB"),
        }
    }
}

impl std::error::Error for Error {}
