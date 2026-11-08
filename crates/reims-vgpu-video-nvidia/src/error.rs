//! The crate's errors.

use std::fmt;

use crate::dl::LoadError;

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
    /// NVDEC reports a picture's decode failed (`cuvidGetDecodeStatus`):
    /// its slice data is damaged, past what our parsers read. The guest is
    /// told so as of a malformed stream, and an H.264 or HEVC session fails
    /// the pictures after it until a random access point
    /// (`crate::decode`'s `Failures`).
    DecodeFailed(String),
    /// A leading picture skipped, as the codec crate's HEVC decoder skips
    /// a RASL picture it cannot decode
    /// (`reims_vgpu_video_codec::hevc::Error::SkippedLeadingPicture`).
    SkippedLeadingPicture(String),
    /// Every decode surface is held by frames not yet released.
    OutOfSurfaces,
    /// Work on the GPU was not done by its deadline, or a decoder's,
    /// encoder's or estimator's earlier work was not (`crate::wait`): what,
    /// in words. The guest is told so as the backend's malfunction.
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
            Self::Bitstream(what) => write!(f, "bad bitstream: {what}"),
            Self::DecodeFailed(what) => write!(f, "{what}"),
            Self::SkippedLeadingPicture(what) => write!(f, "{what}"),
            Self::OutOfSurfaces => write!(f, "every decode surface is held"),
            Self::Stalled(what) => write!(f, "the GPU did not finish {what}"),
            Self::PlaneTooSmall(plane) => write!(f, "the {plane} plane is too small"),
        }
    }
}

impl std::error::Error for Error {}

impl From<LoadError> for Error {
    fn from(e: LoadError) -> Self {
        Self::Load(e)
    }
}

/// A codec crate error: what our parsers refuse is unsupported, the rest a
/// malformed stream.
macro_rules! from_codec {
    ($($codec:ident),*) => {$(
        impl From<reims_vgpu_video_codec::$codec::Error> for Error {
            fn from(e: reims_vgpu_video_codec::$codec::Error) -> Self {
                match e {
                    reims_vgpu_video_codec::$codec::Error::Unsupported(_) => {
                        Self::Unsupported(e.to_string())
                    }
                    _ => Self::Bitstream(e.to_string()),
                }
            }
        }
    )*};
}

from_codec!(h264);

/// An HEVC error: as `from_codec`'s, and a RASL picture skipped is no
/// malformed stream.
impl From<reims_vgpu_video_codec::hevc::Error> for Error {
    fn from(e: reims_vgpu_video_codec::hevc::Error) -> Self {
        use reims_vgpu_video_codec::hevc::Error as Hevc;
        match e {
            Hevc::Unsupported(_) => Self::Unsupported(e.to_string()),
            Hevc::SkippedLeadingPicture => Self::SkippedLeadingPicture(e.to_string()),
            _ => Self::Bitstream(e.to_string()),
        }
    }
}
