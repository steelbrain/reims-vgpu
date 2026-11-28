//! VP9's syntax above the compressed header, for a stateless accelerator:
//! superframes, the uncompressed header and what it keeps from frame to
//! frame (the loop filter's deltas and the segmentation's features), the
//! eight reference slots, and which frames are shown (the VP9 Bitstream and
//! Decoding Process Specification, version 0.6, by section).
//!
//! This module is plain Rust and runs on any host, so that its tests run on
//! a Mac too. What decodes the compressed header and the tiles is an
//! [`Accelerator`], which the VA-API and NVIDIA crates implement over their
//! hardware: both take a frame's bytes whole, uncompressed header first,
//! with the header's fields beside them.
//!
//! A sample is a frame or a superframe (Annex B), as a container's sample
//! and VideoToolbox's sample buffer hold it: the host splits a superframe
//! itself, from bytes the guest wrote ([`superframe`]). Every profile is
//! decoded at its depths: 4:2:0 in profiles 0 and 2, 4:2:2 and 4:4:4 in
//! profiles 1 and 3; 4:4:0 and the RGB colour space are refused with
//! [`Error::Unsupported`] ([`check_supported`]), as no format the guest takes
//! holds them.

mod bits;
pub mod config;
mod decoder;
pub mod header;
pub mod quant;
pub mod superframe;
#[cfg(test)]
pub(crate) mod tests;

pub use decoder::{
    Accelerator, Decoder, Format, FrameParams, MAX_SIDE, Output, Reference, check_supported,
};

/// VP9's highest level, 6.2, as `vpcC` and VideoToolbox number it: ten times the level. Apple's
/// own decoders' capabilities number AV1's levels the same way (63 for 6.3; the M3 Max's
/// AppleAVD, macOS 27.0.1, which lists no VP9 capabilities of its own).
pub const TOP_LEVEL: u32 = 62;

/// Why a sample could not be read or decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The data ended inside a syntax element.
    Truncated,
    /// A syntax element, or a variable the specification derives from
    /// them, named, is outside what the specification allows.
    OutOfRange(&'static str),
    /// A superframe index (Annex B) whose frames' sizes do not add up to
    /// the bytes before it, or that holds a frame of no bytes.
    BadSuperframe,
    /// A sample of no bytes, or more frames than a superframe holds.
    EmptySample,
    /// A frame whose compressed header, `header_size_in_bytes` long, runs
    /// past the frame, or is empty (7.2).
    BadHeaderSize,
    /// An inter frame names a reference slot that holds no frame, or a
    /// frame shows a slot that holds none: as after starting at a frame
    /// that is not a key frame.
    MissingReference,
    /// An inter frame whose reference has another bit depth or chroma
    /// subsampling, or a size it does not scale from (2 to 1/16 of its
    /// own; 8.5.2.3).
    BadReference,
    /// A frame larger than [`MAX_SIDE`] on a side.
    TooLarge,
    /// A feature this decoder does not implement, named.
    Unsupported(&'static str),
    /// A `vpcC` record shorter than its fixed fields, or of another version.
    BadVpcc,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "the frame ends inside a syntax element"),
            Self::OutOfRange(name) => write!(f, "{name} is out of range"),
            Self::BadSuperframe => write!(f, "a superframe index that does not fit its sample"),
            Self::EmptySample => write!(f, "a sample of no frame, or of more than a superframe"),
            Self::BadHeaderSize => write!(f, "a compressed header past its frame"),
            Self::MissingReference => write!(f, "a reference frame is missing"),
            Self::BadReference => write!(f, "a reference frame of another format or scale"),
            Self::TooLarge => write!(f, "a frame past {MAX_SIDE} on a side"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::BadVpcc => write!(f, "bad vpcC record"),
        }
    }
}

impl Error {
    /// Whether the error is one sample's of the stream, after which the
    /// decoder goes on with the next: what this decoder does not take
    /// ([`Self::Unsupported`]) and a `vpcC` record's errors are not.
    pub fn is_picture_error(&self) -> bool {
        !matches!(self, Self::Unsupported(_) | Self::BadVpcc)
    }
}

impl std::error::Error for Error {}
