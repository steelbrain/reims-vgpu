//! The crate's errors.

use std::fmt;

use crate::nvidia::dl::LoadError;

/// Why an NVIDIA operation failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A driver library or one of its functions could not be loaded: the
    /// host has no NVIDIA driver, or an older one.
    Load(LoadError),
    /// A CUDA driver API or NVDEC call failed (both return `CUresult`).
    Cuda {
        /// The function.
        call: &'static str,
        /// Its `CUresult`.
        code: i32,
        /// `cuGetErrorName` and `cuGetErrorString` of the code.
        message: String,
    },
    /// An NVENC call failed.
    Nvenc {
        /// The function.
        call: &'static str,
        /// Its `NVENCSTATUS`.
        status: i32,
        /// The status's name, and `nvEncGetLastErrorString` where the
        /// session gave one.
        message: String,
    },
    /// No GPU answers to the name given.
    NoDevice(String),
    /// The GPU, its driver or this crate cannot do it: what, in words.
    Unsupported(String),
    /// A picture smaller than NVDEC decodes: what, in words. The guest is
    /// told so with kVTVideoDecoderMalfunctionErr, as Apple's decoders
    /// refuse a size they do not decode.
    TooSmall(String),
    /// The stream is malformed, or uses what our parsers refuse.
    Bitstream(String),
    /// One picture of the stream does not decode, and the decoder goes on:
    /// the codec crate's errors that are one picture's
    /// (`crate::codec::h264::Error::is_picture_error`), such as a picture
    /// that predicts from one the decoder does not hold.
    Picture(String),
    /// NVDEC reports a picture's decode failed (`cuvidGetDecodeStatus`):
    /// its slice data is damaged, past what our parsers read. The guest is
    /// told so as of a malformed stream, and an H.264 or HEVC session fails
    /// the pictures after it until a random access point
    /// (`crate::nvidia::decode`'s `Failures`).
    DecodeFailed(String),
    /// A leading picture skipped, as the codec crate's HEVC decoder skips
    /// a RASL picture it cannot decode
    /// (`crate::codec::hevc::Error::SkippedLeadingPicture`).
    SkippedLeadingPicture(String),
    /// Every decode surface is held by frames not yet released.
    OutOfSurfaces,
    /// Work on the GPU was not done by its deadline, or a decoder's or
    /// encoder's earlier work was not (`crate::nvidia::wait`): what, in words.
    Stalled(String),
    /// A destination or source plane is too small for the frame.
    PlaneTooSmall(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(e) => write!(f, "{e}"),
            Self::Cuda {
                call,
                code,
                message,
            } => write!(f, "{call} failed: {message} ({code})"),
            Self::Nvenc {
                call,
                status,
                message,
            } => write!(f, "{call} failed: {message} ({status})"),
            Self::NoDevice(what) => write!(f, "no NVIDIA GPU {what}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::TooSmall(what) => write!(f, "too small: {what}"),
            Self::Bitstream(what) | Self::Picture(what) => write!(f, "bad bitstream: {what}"),
            Self::DecodeFailed(what) => write!(f, "{what}"),
            Self::SkippedLeadingPicture(what) => write!(f, "{what}"),
            Self::OutOfSurfaces => write!(f, "every decode surface is held"),
            Self::Stalled(what) => write!(f, "the GPU did not finish {what}"),
            Self::PlaneTooSmall(plane) => write!(f, "the {plane} plane is too small"),
        }
    }
}

impl Error {
    /// Whether the error is one picture's of the stream, after which the
    /// decoder goes on ([`Self::Picture`], and a leading picture skipped,
    /// [`Self::SkippedLeadingPicture`]).
    pub fn is_picture_error(&self) -> bool {
        matches!(self, Self::Picture(_) | Self::SkippedLeadingPicture(_))
    }
}

impl std::error::Error for Error {}

impl From<LoadError> for Error {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}

/// A codec crate error: what our parsers refuse is unsupported, one
/// picture's error a picture's, the rest a malformed stream.
macro_rules! from_codec {
    ($($codec:ident),*) => {$(
        impl From<crate::codec::$codec::Error> for Error {
            fn from(e: crate::codec::$codec::Error) -> Self {
                match e {
                    crate::codec::$codec::Error::Unsupported(_) => {
                        Self::Unsupported(e.to_string())
                    }
                    _ if e.is_picture_error() => Self::Picture(e.to_string()),
                    _ => Self::Bitstream(e.to_string()),
                }
            }
        }
    )*};
}

from_codec!(h264);

/// An HEVC error: as `from_codec`'s, and a RASL picture skipped is no
/// malformed stream.
impl From<crate::codec::hevc::Error> for Error {
    fn from(e: crate::codec::hevc::Error) -> Self {
        use crate::codec::hevc::Error as Hevc;
        match e {
            Hevc::Unsupported(_) => Self::Unsupported(e.to_string()),
            Hevc::SkippedLeadingPicture => Self::SkippedLeadingPicture(e.to_string()),
            _ if e.is_picture_error() => Self::Picture(e.to_string()),
            _ => Self::Bitstream(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{h264, hevc};

    /// A codec error that is one picture's is a picture's, what the parsers refuse unsupported,
    /// a RASL picture skipped one apart, and the rest a malformed stream's.
    #[test]
    fn codec_errors_are_a_picture_s_unsupported_or_the_stream_s() {
        let picture = |e: &Error| matches!(e, Error::Picture(_));
        assert!(picture(&h264::Error::MissingReference.into()));
        assert!(picture(&h264::Error::Unrecovered.into()));
        assert!(picture(&hevc::Error::MissingReference.into()));
        assert!(picture(&hevc::Error::Unrecovered.into()));
        let unsupported = Error::from(h264::Error::Unsupported("data partitioning"));
        assert!(matches!(unsupported, Error::Unsupported(_)), "{unsupported}");
        let skipped = Error::from(hevc::Error::SkippedLeadingPicture);
        assert!(matches!(skipped, Error::SkippedLeadingPicture(_)), "{skipped}");
        for error in [
            Error::from(h264::Error::AvccVersion),
            Error::from(hevc::Error::HvccVersion),
        ] {
            assert!(matches!(error, Error::Bitstream(_)), "{error}");
        }
        for error in [
            Error::from(h264::Error::Truncated),
            Error::from(hevc::Error::Unrecovered),
            skipped,
        ] {
            assert!(error.is_picture_error(), "{error}");
        }
        for error in [
            Error::from(h264::Error::AvccVersion),
            unsupported,
            Error::DecodeFailed("status 8".to_owned()),
            Error::OutOfSurfaces,
        ] {
            assert!(!error.is_picture_error(), "{error}");
        }
    }
}
