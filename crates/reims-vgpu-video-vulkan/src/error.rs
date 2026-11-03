//! Why a decoder, or the device under it, failed.

use ash::vk;
use reims_vgpu_video_codec::{h264, hevc};

/// Why a decoder, or the device under it, failed.
#[derive(Debug)]
pub enum Error {
    /// The Vulkan loader did not open: why.
    Loader(String),
    /// A Vulkan call, named, that returned an error.
    Vulkan(&'static str, vk::Result),
    /// No device decodes the codec through Vulkan Video, or the device
    /// given does not: why, device by device.
    NoVideoDecode(String),
    /// No encode queue encodes the codec through Vulkan Video on the
    /// device given, or the device has no encode queue attached: why.
    NoVideoEncode(String),
    /// What the stream needs and the device does not do.
    Unsupported(String),
    /// An H.264 stream, refused by the decoding process above the
    /// hardware.
    Stream(h264::Error),
    /// An H.265 stream, refused by the decoding process above the
    /// hardware, or a RASL picture it skips
    /// ([`hevc::Error::SkippedLeadingPicture`], which is no failure).
    H265Stream(hevc::Error),
    /// A wait for the GPU that did not end in its time.
    Timeout,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Loader(why) => write!(f, "no Vulkan loader: {why}"),
            Self::Vulkan(call, result) => write!(f, "{call}: {result}"),
            Self::NoVideoDecode(why) => write!(f, "no Vulkan Video decode: {why}"),
            Self::NoVideoEncode(why) => write!(f, "no Vulkan Video encode: {why}"),
            Self::Unsupported(what) => write!(f, "unsupported: {what}"),
            Self::Stream(error) => write!(f, "{error}"),
            Self::H265Stream(error) => write!(f, "{error}"),
            Self::Timeout => write!(f, "the GPU did not finish in time"),
        }
    }
}

impl std::error::Error for Error {}

impl From<h264::Error> for Error {
    fn from(error: h264::Error) -> Self {
        Self::Stream(error)
    }
}

impl From<hevc::Error> for Error {
    fn from(error: hevc::Error) -> Self {
        Self::H265Stream(error)
    }
}

/// A Vulkan call's result, named for its error.
pub(crate) trait Named<T> {
    /// `Ok` as it is; an error as [`Error::Vulkan`] of `call`.
    fn named(self, call: &'static str) -> Result<T, Error>;
}

impl<T> Named<T> for Result<T, vk::Result> {
    fn named(self, call: &'static str) -> Result<T, Error> {
        self.map_err(|result| Error::Vulkan(call, result))
    }
}

impl Named<()> for vk::Result {
    fn named(self, call: &'static str) -> Result<(), Error> {
        self.result().named(call)
    }
}
