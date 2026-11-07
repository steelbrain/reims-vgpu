//! What a JPEG backend gives the host session's JPEG engine: still-image
//! decodes for `jdec`, and what `jcap` says of them.
//!
//! The session reads the guest's options, works out what part of the image
//! goes where and at what scale, and checks the destination; a backend only
//! decodes one JPEG and writes a rectangle of it into the destination's
//! planes in one of the layouts below. A backend with no JPEG decoder
//! offers none ([`crate::backend::Backend::jpeg`]), and the guest is
//! told so as Apple's host without JPEG hardware tells it.

use crate::fourcc::FourCc;
use crate::surfaces::PlaneDestination;

mod samples;

pub use samples::{DecodedSamples, SampleView, write_bgra, write_samples};

/// How a JPEG's chroma is sampled, numbered as CMPhoto numbers it in the
/// options' `Subsampling`. The variant names are ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JpegSubsampling {
    /// Three components, the first sampled 2x2 and the others 1x1.
    Yuv420,
    /// Three components, the first sampled 2x1 and the others 1x1.
    Yuv422,
    /// Three components, each sampled 1x1.
    Yuv444,
    /// Three components, the first sampled 4x1 and the others 1x1.
    Yuv411,
    /// One component.
    Grey,
}

impl JpegSubsampling {
    /// CMPhoto's number for it.
    pub const fn code(self) -> i64 {
        match self {
            Self::Yuv420 => 0,
            Self::Yuv422 => 1,
            Self::Yuv444 => 2,
            Self::Yuv411 => 3,
            Self::Grey => 4,
        }
    }

    /// The subsampling CMPhoto's number names, if it names one.
    pub fn from_code(code: i64) -> Option<Self> {
        [
            Self::Yuv420,
            Self::Yuv422,
            Self::Yuv444,
            Self::Yuv411,
            Self::Grey,
        ]
        .into_iter()
        .find(|known| known.code() == code)
    }

    /// How many luma samples across and down share one chroma sample,
    /// or none for grey.
    pub const fn chroma_divisors(self) -> Option<(u32, u32)> {
        match self {
            Self::Yuv420 => Some((2, 2)),
            Self::Yuv422 => Some((2, 1)),
            Self::Yuv444 => Some((1, 1)),
            Self::Yuv411 => Some((4, 1)),
            Self::Grey => None,
        }
    }

    /// The MCU's width and height in pixels: 8 times the first
    /// component's sampling factors, and 8x8 for one component (ITU-T
    /// T.81, A.2).
    pub const fn mcu_size(self) -> (u32, u32) {
        match self {
            Self::Yuv420 => (16, 16),
            Self::Yuv422 => (16, 8),
            Self::Yuv444 | Self::Grey => (8, 8),
            Self::Yuv411 => (32, 8),
        }
    }
}

/// The pixel layouts the engine writes. Each is one or more CoreVideo
/// formats that lay their samples out alike: Apple's host writes the
/// JPEG's own samples, full range, whatever range the format names
/// (measured: `420v` held the same bytes as `420f`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum JpegOutputFormat {
    /// `420f`, `420v`: Y, then interleaved CbCr at half width and height.
    Biplanar420,
    /// `422f`, `422v`: Y, then interleaved CbCr at half width.
    Biplanar422,
    /// `444f`, `444v`: Y, then interleaved CbCr at full size.
    Biplanar444,
    /// `yuvs`: one plane, Y0 Cb Y1 Cr.
    Yuyv,
    /// `2vuy`: one plane, Cb Y0 Cr Y1.
    Uyvy,
    /// `BGRA`: one plane, B G R A.
    Bgra,
}

impl JpegOutputFormat {
    /// The layout of the CoreVideo pixel format `format`, if the engine
    /// has one for it.
    pub fn from_fourcc(format: FourCc) -> Option<Self> {
        Some(match &format.0.to_be_bytes() {
            b"420f" | b"420v" => Self::Biplanar420,
            b"422f" | b"422v" => Self::Biplanar422,
            b"444f" | b"444v" => Self::Biplanar444,
            b"yuvs" => Self::Yuyv,
            b"2vuy" => Self::Uyvy,
            b"BGRA" => Self::Bgra,
            _ => return None,
        })
    }

    /// How many luma samples across and down share one chroma sample in
    /// this layout, or none for one that is not YCbCr.
    pub const fn chroma_divisors(self) -> Option<(u32, u32)> {
        match self {
            Self::Biplanar420 => Some((2, 2)),
            Self::Biplanar422 | Self::Yuyv | Self::Uyvy => Some((2, 1)),
            Self::Biplanar444 => Some((1, 1)),
            Self::Bgra => None,
        }
    }

    /// How many planes a surface of this layout has.
    pub const fn plane_count(self) -> usize {
        match self {
            Self::Biplanar420 | Self::Biplanar422 | Self::Biplanar444 => 2,
            Self::Yuyv | Self::Uyvy | Self::Bgra => 1,
        }
    }
}

/// What the engine says of itself in `jcap`'s capabilities, whose keys
/// CMPhoto reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegCapabilities {
    /// `NumberOfCores`: how many decodes the engine runs at once.
    pub cores: u32,
    /// `Supports16BitQuantizationTables`: whether every stream with
    /// 16-bit quantization tables decodes.
    pub sixteen_bit_quantization_tables: bool,
    /// `SupportsMissingEOI`: whether a stream without its EOI decodes.
    pub missing_eoi: bool,
    /// `SupportsRSTLogging`: whether the engine reports where its restart
    /// markers are.
    pub restart_logging: bool,
    /// `SupportsInterchangeFormats`: whether it writes Apple's compressed
    /// interchange pixel formats.
    pub interchange_formats: bool,
}

/// What a backend decodes at all, as its hardware answers: the samplings
/// and the largest picture. A stream outside them is refused as
/// [`JpegError::UnsupportedStream`] before any work. The guest hears of it
/// only through those refusals (`jcap` has no key for it); the conformance
/// suite's `jpeg-check` reads it, to tell such a refusal from one the
/// backend should not make.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JpegDecodes {
    /// The samplings it decodes.
    pub subsamplings: Vec<JpegSubsampling>,
    /// The largest width and height it decodes.
    pub max_size: (u32, u32),
}

impl JpegDecodes {
    /// Whether a stream of `subsampling`, `width` by `height`, is within
    /// what it decodes.
    pub fn takes(&self, subsampling: JpegSubsampling, (width, height): (u32, u32)) -> bool {
        self.subsamplings.contains(&subsampling)
            && width <= self.max_size.0
            && height <= self.max_size.1
    }
}

/// A rectangle of an image, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelRect {
    /// The left edge.
    pub x: u32,
    /// The top edge.
    pub y: u32,
    /// The width.
    pub width: u32,
    /// The height.
    pub height: u32,
}

/// One decode, as the session has checked it against the guest's options.
#[derive(Clone, Copy, Debug)]
pub struct JpegDecode<'a> {
    /// The JPEG, from its SOI.
    pub data: &'a [u8],
    /// The image's size, as the guest's options give it. A stream of
    /// another size is [`JpegError::SizeMismatch`].
    pub width: u32,
    /// The image's height, as the guest's options give it.
    pub height: u32,
    /// The image's subsampling, as the guest's options give it. A stream
    /// sampled otherwise is [`JpegError::SubsamplingMismatch`].
    pub subsampling: JpegSubsampling,
    /// The part of the image to write: the whole image, or a rectangle of
    /// whole MCUs within it (cut at the image's right and bottom edges).
    pub region: PixelRect,
    /// 1, 2, 4 or 8: the region is written at 1/scale its size, rounded
    /// up.
    pub scale: u32,
    /// How the destination lays its samples out.
    pub format: JpegOutputFormat,
}

impl JpegDecode<'_> {
    /// The size written: the region at 1/scale, rounded up.
    pub fn output_size(&self) -> (u32, u32) {
        (
            self.region.width.div_ceil(self.scale),
            self.region.height.div_ceil(self.scale),
        )
    }
}

/// A backend's JPEG decoder.
pub trait JpegDecoder: Send + Sync {
    /// What `jcap` says of it.
    fn capabilities(&self) -> JpegCapabilities;

    /// What it decodes at all ([`JpegDecodes`]).
    fn decodes(&self) -> JpegDecodes;

    /// Decodes `request.data` and writes its region, at its scale, into
    /// the top-left corner of `planes`, in `request.format`: one plane for
    /// a packed layout, Y then CbCr for a biplanar one. What lies outside
    /// that corner is left as it was. The planes are at least the output's
    /// size. It may be called from several threads at once.
    ///
    /// The host session's planes are a buffer of its own, which it copies
    /// into the guest's surface after: it calls this on a thread of its
    /// own with no guest surface lent, and does not clear the buffer
    /// first, so every sample of the output is to be written.
    fn decode(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<(), JpegError>;
}

/// Why a decode failed. The session turns each into the status Apple's
/// host gives for the like.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JpegError {
    /// The stream does not parse, or does not decode.
    BadData(String),
    /// The stream's scan ends, at a marker, before its last MCU: a stream
    /// cut short and closed with its EOI.
    ScanEndsEarly,
    /// The stream is valid but uses what the backend does not decode:
    /// progressive or arithmetic coding, 12-bit samples, several scans.
    UnsupportedStream(String),
    /// The stream's size is not the options'.
    SizeMismatch {
        /// The stream's width.
        width: u32,
        /// The stream's height.
        height: u32,
    },
    /// The stream is not sampled as the options say.
    SubsamplingMismatch,
    /// The backend does not write this layout from this stream.
    UnsupportedFormat(String),
    /// The backend does not decode at this scale.
    UnsupportedScale(u32),
    /// The backend or its driver failed.
    Malfunction(String),
}

impl std::fmt::Display for JpegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadData(why) => write!(f, "bad JPEG: {why}"),
            Self::ScanEndsEarly => write!(f, "bad JPEG: its scan ends before its last MCU"),
            Self::UnsupportedStream(what) => write!(f, "unsupported JPEG: {what}"),
            Self::SizeMismatch { width, height } => {
                write!(f, "the stream is {width}x{height}, not the options' size")
            }
            Self::SubsamplingMismatch => write!(f, "the stream is not sampled as the options say"),
            Self::UnsupportedFormat(what) => write!(f, "unsupported output: {what}"),
            Self::UnsupportedScale(scale) => write!(f, "unsupported scale 1/{scale}"),
            Self::Malfunction(why) => write!(f, "malfunction: {why}"),
        }
    }
}

impl std::error::Error for JpegError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsamplings_are_cmphotos_numbers() {
        assert_eq!(JpegSubsampling::from_code(0), Some(JpegSubsampling::Yuv420));
        assert_eq!(JpegSubsampling::from_code(4), Some(JpegSubsampling::Grey));
        assert_eq!(JpegSubsampling::from_code(5), None);
        assert_eq!(JpegSubsampling::Yuv422.mcu_size(), (16, 8));
    }

    #[test]
    fn formats_share_a_layout_across_ranges() {
        let layout = |s: &[u8; 4]| JpegOutputFormat::from_fourcc(FourCc::new(*s));
        assert_eq!(layout(b"420v"), Some(JpegOutputFormat::Biplanar420));
        assert_eq!(layout(b"420f"), Some(JpegOutputFormat::Biplanar420));
        assert_eq!(layout(b"2vuy"), Some(JpegOutputFormat::Uyvy));
        assert_eq!(layout(b"L008"), None);
        assert_eq!(JpegOutputFormat::Yuyv.plane_count(), 1);
    }

    #[test]
    fn output_rounds_up() {
        let request = JpegDecode {
            data: &[],
            width: 333,
            height: 211,
            subsampling: JpegSubsampling::Yuv420,
            region: PixelRect {
                x: 0,
                y: 0,
                width: 333,
                height: 211,
            },
            scale: 2,
            format: JpegOutputFormat::Biplanar420,
        };
        assert_eq!(request.output_size(), (167, 106));
    }
}
