//! AV1's syntax and the decoding process above the tile data, for a stateless accelerator
//! (AV1 Bitstream and Decoding Process Specification, by section): OBUs in the low overhead
//! format MP4 samples carry, `av1C`, the sequence header, frame headers with the processes
//! that set what the hardware is given (`set_frame_refs()`, superres, tile info, the state
//! loaded from the primary reference frame, skip mode, global motion and film grain), tile
//! groups, the eight reference slots and the frames each temporal unit shows.
//!
//! This module is plain Rust and runs on any host. What decodes the tiles is an
//! [`Accelerator`], which the VA-API and NVIDIA modules implement over their hardware.
//!
//! The host parses bytes the guest wrote. Every size and count here is read from a field
//! checked against the bounds the specification sets, or against the decoder's own limit
//! (the largest frame its hardware lists, [`Limits`]): a sequence header whose largest frame
//! is past it is refused before any frame is read, and each frame's size is held to its
//! sequence header's. A tile's bytes are taken only within its OBU.

mod av1c;
pub(crate) mod bits;
mod decoder;
pub mod frame;
pub mod obu;
pub mod sequence;
#[cfg(test)]
pub(crate) mod tests;
pub mod tile_group;

pub use av1c::DecoderConfiguration;
pub use decoder::{
    Accelerator, Decoder, FrameParams, Limits, Output, ReferenceFrame, check_record,
    check_supported,
};

/// Why a stream could not be read or decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The data ended inside a syntax element.
    Truncated,
    /// A syntax element, or a variable the specification derives from them, named, is outside
    /// what the specification allows.
    OutOfRange(&'static str),
    /// An OBU's size runs past the end of its sample.
    ObuPastEnd,
    /// An `av1C` record shorter than its four fixed bytes.
    Av1cTruncated,
    /// An `av1C` record whose `marker` is not 1 or whose `version` is not 1.
    Av1cVersion,
    /// A frame header before any sequence header.
    NoSequenceHeader,
    /// A tile group with no frame header before it in its temporal unit.
    NoFrameHeader,
    /// A sample that ends before the last tile of a frame it started.
    IncompleteFrame,
    /// A tile group that does not start at the tile after the last one's end (5.11.1).
    TileGroupOutOfOrder,
    /// A tile of no bytes.
    EmptyTile,
    /// A frame that predicts from, or shows, a reference slot the decoder does not hold, as
    /// after starting or resetting at a frame that is not a key frame shown at once.
    MissingReference,
    /// A feature this decoder does not implement, named.
    Unsupported(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "the bitstream ends inside a syntax element"),
            Self::OutOfRange(name) => write!(f, "{name} is out of range"),
            Self::ObuPastEnd => write!(f, "an OBU's size runs past its sample"),
            Self::Av1cTruncated => write!(f, "bad av1C record: truncated"),
            Self::Av1cVersion => write!(f, "bad av1C record: marker or version is not 1"),
            Self::NoSequenceHeader => write!(f, "a frame header before any sequence header"),
            Self::NoFrameHeader => write!(f, "a tile group with no frame header"),
            Self::IncompleteFrame => write!(f, "a sample ends before its frame's last tile"),
            Self::TileGroupOutOfOrder => write!(f, "a tile group out of order"),
            Self::EmptyTile => write!(f, "a tile of no bytes"),
            Self::MissingReference => write!(f, "a reference frame is missing"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
        }
    }
}

impl Error {
    /// Whether the error is one frame's of the stream, after which the decoder goes on: the
    /// frame's bytes do not decode as the specification reads them, or it predicts from a
    /// slot the decoder does not hold ([`Self::MissingReference`]). The decoder takes the next
    /// sample as it would any, and the slots the frame would have refreshed are empty until a
    /// later frame refreshes them. Not one frame's: what this decoder does not take
    /// ([`Self::Unsupported`]), and an `av1C` record's errors, which are no frame's.
    pub fn is_picture_error(&self) -> bool {
        !matches!(
            self,
            Self::Unsupported(_) | Self::Av1cTruncated | Self::Av1cVersion
        )
    }
}

impl std::error::Error for Error {}
