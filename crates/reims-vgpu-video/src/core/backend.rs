//! What a backend answers: the pixel formats it writes ([`PixelFormat`]),
//! whether its device wrote a frame into a guest surface itself
//! ([`DeviceWrite`]), and why it could not do what it was asked
//! ([`BackendError`]).

use crate::core::fourcc::FourCc;

/// A pixel format the host decodes into: the CoreVideo formats a guest
/// pixel buffer can have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 8-bit 4:2:0, Y then interleaved CbCr, video range: `420v`.
    Nv12VideoRange,
    /// The same, full range: `420f`.
    Nv12FullRange,
    /// 10-bit 4:2:0 in 16-bit samples, video range: `x420`.
    P010VideoRange,
    /// The same, full range: `xf20`.
    P010FullRange,
}

impl PixelFormat {
    /// The CoreVideo pixel format type.
    pub const fn fourcc(self) -> FourCc {
        FourCc::new(match self {
            Self::Nv12VideoRange => *b"420v",
            Self::Nv12FullRange => *b"420f",
            Self::P010VideoRange => *b"x420",
            Self::P010FullRange => *b"xf20",
        })
    }

    /// The format a CoreVideo pixel format type names, if it is one of
    /// these.
    pub fn from_fourcc(format: FourCc) -> Option<Self> {
        [
            Self::Nv12VideoRange,
            Self::Nv12FullRange,
            Self::P010VideoRange,
            Self::P010FullRange,
        ]
        .into_iter()
        .find(|known| known.fourcc() == format)
    }

    /// The bytes one sample takes: 1 for 8-bit formats, 2 for 10-bit ones.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::Nv12VideoRange | Self::Nv12FullRange => 1,
            Self::P010VideoRange | Self::P010FullRange => 2,
        }
    }
}

/// What a backend's write of a frame into a guest surface's pages did
/// ([`crate::core::device_memory::GuestTarget`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceWrite {
    /// The frame is in the guest's buffer.
    Written,
    /// Nothing was written, for the reason given: the frame is copied by
    /// the CPU instead.
    Declined(String),
}

/// Why a backend could not do what it was asked.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BackendError {
    /// The codec, profile, level or size is not one the backend decodes.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The data does not decode.
    #[error("bad data: {0}")]
    BadData(String),
    /// The backend or its driver failed.
    #[error("malfunction: {0}")]
    Malfunction(String),
    /// A leading picture skipped, since it predicts from pictures before a
    /// random access point the decoding started at: an HEVC RASL picture
    /// of an IRAP picture with `NoRaslOutputFlag` 1. No failure: the frames
    /// after it decode.
    #[error("skipped: {0}")]
    SkippedLeadingPicture(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_their_fourccs() {
        assert_eq!(PixelFormat::Nv12VideoRange.fourcc().0, 875_704_438);
        assert_eq!(
            PixelFormat::from_fourcc(FourCc::new(*b"xf20")),
            Some(PixelFormat::P010FullRange)
        );
        assert_eq!(PixelFormat::from_fourcc(FourCc::new(*b"BGRA")), None);
    }
}
