//! What a backend answers: the pixel formats it writes ([`PixelFormat`]),
//! whether its device wrote a frame into a guest surface itself
//! ([`DeviceWrite`]), and why it could not do what it was asked
//! ([`BackendError`]).

use crate::core::fourcc::FourCc;

/// A pixel format the host decodes into: the CoreVideo formats a guest
/// pixel buffer can have, each of two planes, luma then interleaved Cb and
/// Cr, deeper samples in the top bits of 16. Past HEVC Main 10's they are
/// the formats an arm64 guest's own software decoder lands each chroma
/// format and depth in (CoreVideo's descriptions name `tv20`, `tv22` and
/// `tv44` 12 bits a component, video range, black at 0x1000; **measured**
/// on an M3 Max, macOS 27.0.1, potluck jobs m3-max-77228, 77231 and 77237).
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
    /// 8-bit 4:4:4, Y then interleaved CbCr as wide and tall, video range: `444v`.
    Yuv444VideoRange,
    /// The same, full range: `444f`.
    Yuv444FullRange,
    /// 8-bit 4:2:2, Y then interleaved CbCr half as wide and as tall, video range: `422v`.
    Nv16VideoRange,
    /// The same, full range: `422f`.
    Nv16FullRange,
    /// 10-bit 4:2:2 in 16-bit samples, video range: `x422`.
    P210VideoRange,
    /// The same, full range: `xf22`.
    P210FullRange,
    /// 10-bit 4:4:4 in 16-bit samples, video range: `x444`.
    P410VideoRange,
    /// The same, full range: `xf44`.
    P410FullRange,
    /// 12-bit 4:2:0 in 16-bit samples, video range: `tv20`.
    P012VideoRange,
    /// The same, full range: `tf20`.
    P012FullRange,
    /// 12-bit 4:2:2 in 16-bit samples, video range: `tv22`.
    P212VideoRange,
    /// The same, full range: `tf22`.
    P212FullRange,
    /// 12-bit 4:4:4 in 16-bit samples, video range: `tv44`.
    P412VideoRange,
    /// The same, full range: `tf44`.
    P412FullRange,
}

/// How a [`PixelFormat`]'s chroma plane is sampled against its luma.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Chroma {
    /// A Cb and Cr pair for every two columns and two rows.
    Yuv420,
    /// A pair for every two columns of each row.
    Yuv422,
    /// A pair for every pixel.
    Yuv444,
}

impl Chroma {
    /// The `chroma_format_idc` of its pictures.
    pub const fn chroma_format_idc(self) -> u8 {
        match self {
            Self::Yuv420 => 1,
            Self::Yuv422 => 2,
            Self::Yuv444 => 3,
        }
    }

    /// The chroma format of `chroma_format_idc`, of 4:2:0, 4:2:2 and 4:4:4.
    pub const fn of_idc(chroma_format_idc: u8) -> Option<Self> {
        match chroma_format_idc {
            1 => Some(Self::Yuv420),
            2 => Some(Self::Yuv422),
            3 => Some(Self::Yuv444),
            _ => None,
        }
    }

    /// The pairs a chroma row holds and the chroma rows of a picture `width` by `height`: an
    /// odd last column or row takes the pair it shares, as CoreVideo's planes have it.
    pub fn plane_size(self, width: usize, height: u32) -> (usize, u32) {
        match self {
            Self::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
            Self::Yuv422 => (width.div_ceil(2), height),
            Self::Yuv444 => (width, height),
        }
    }

    /// Its name, as a refusal spells it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Yuv420 => "4:2:0",
            Self::Yuv422 => "4:2:2",
            Self::Yuv444 => "4:4:4",
        }
    }
}

impl PixelFormat {
    /// Every format.
    const ALL: [Self; 18] = [
        Self::Nv12VideoRange,
        Self::Nv12FullRange,
        Self::P010VideoRange,
        Self::P010FullRange,
        Self::Yuv444VideoRange,
        Self::Yuv444FullRange,
        Self::Nv16VideoRange,
        Self::Nv16FullRange,
        Self::P210VideoRange,
        Self::P210FullRange,
        Self::P410VideoRange,
        Self::P410FullRange,
        Self::P012VideoRange,
        Self::P012FullRange,
        Self::P212VideoRange,
        Self::P212FullRange,
        Self::P412VideoRange,
        Self::P412FullRange,
    ];

    /// The CoreVideo pixel format type.
    pub const fn fourcc(self) -> FourCc {
        FourCc::new(match self {
            Self::Nv12VideoRange => *b"420v",
            Self::Nv12FullRange => *b"420f",
            Self::P010VideoRange => *b"x420",
            Self::P010FullRange => *b"xf20",
            Self::Yuv444VideoRange => *b"444v",
            Self::Yuv444FullRange => *b"444f",
            Self::Nv16VideoRange => *b"422v",
            Self::Nv16FullRange => *b"422f",
            Self::P210VideoRange => *b"x422",
            Self::P210FullRange => *b"xf22",
            Self::P410VideoRange => *b"x444",
            Self::P410FullRange => *b"xf44",
            Self::P012VideoRange => *b"tv20",
            Self::P012FullRange => *b"tf20",
            Self::P212VideoRange => *b"tv22",
            Self::P212FullRange => *b"tf22",
            Self::P412VideoRange => *b"tv44",
            Self::P412FullRange => *b"tf44",
        })
    }

    /// The format a CoreVideo pixel format type names, if it is one of
    /// these.
    pub fn from_fourcc(format: FourCc) -> Option<Self> {
        Self::ALL.into_iter().find(|known| known.fourcc() == format)
    }

    /// The format pictures of `chroma` at `bits` bits land in, by their range: `None` for a
    /// depth other than 8, 10 and 12.
    pub fn of(chroma: Chroma, bits: u8, full_range: bool) -> Option<Self> {
        let (video, full) = match (chroma, bits) {
            (Chroma::Yuv420, 8) => (Self::Nv12VideoRange, Self::Nv12FullRange),
            (Chroma::Yuv420, 10) => (Self::P010VideoRange, Self::P010FullRange),
            (Chroma::Yuv420, 12) => (Self::P012VideoRange, Self::P012FullRange),
            (Chroma::Yuv422, 8) => (Self::Nv16VideoRange, Self::Nv16FullRange),
            (Chroma::Yuv422, 10) => (Self::P210VideoRange, Self::P210FullRange),
            (Chroma::Yuv422, 12) => (Self::P212VideoRange, Self::P212FullRange),
            (Chroma::Yuv444, 8) => (Self::Yuv444VideoRange, Self::Yuv444FullRange),
            (Chroma::Yuv444, 10) => (Self::P410VideoRange, Self::P410FullRange),
            (Chroma::Yuv444, 12) => (Self::P412VideoRange, Self::P412FullRange),
            _ => return None,
        };
        Some(if full_range { full } else { video })
    }

    /// The bits a sample holds: 8, 10 or 12.
    pub const fn bit_depth(self) -> u8 {
        match self {
            Self::Nv12VideoRange
            | Self::Nv12FullRange
            | Self::Yuv444VideoRange
            | Self::Yuv444FullRange
            | Self::Nv16VideoRange
            | Self::Nv16FullRange => 8,
            Self::P010VideoRange
            | Self::P010FullRange
            | Self::P210VideoRange
            | Self::P210FullRange
            | Self::P410VideoRange
            | Self::P410FullRange => 10,
            Self::P012VideoRange
            | Self::P012FullRange
            | Self::P212VideoRange
            | Self::P212FullRange
            | Self::P412VideoRange
            | Self::P412FullRange => 12,
        }
    }

    /// The bytes one sample takes: 1 for 8-bit formats, 2 for deeper ones.
    pub const fn bytes_per_sample(self) -> usize {
        if self.bit_depth() > 8 { 2 } else { 1 }
    }

    /// How its chroma is sampled.
    pub const fn chroma(self) -> Chroma {
        match self {
            Self::Nv12VideoRange
            | Self::Nv12FullRange
            | Self::P010VideoRange
            | Self::P010FullRange
            | Self::P012VideoRange
            | Self::P012FullRange => Chroma::Yuv420,
            Self::Nv16VideoRange
            | Self::Nv16FullRange
            | Self::P210VideoRange
            | Self::P210FullRange
            | Self::P212VideoRange
            | Self::P212FullRange => Chroma::Yuv422,
            Self::Yuv444VideoRange
            | Self::Yuv444FullRange
            | Self::P410VideoRange
            | Self::P410FullRange
            | Self::P412VideoRange
            | Self::P412FullRange => Chroma::Yuv444,
        }
    }

    /// Whether its chroma is as wide and tall as its luma (4:4:4).
    pub const fn is_4_4_4(self) -> bool {
        matches!(self.chroma(), Chroma::Yuv444)
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
        assert_eq!(
            PixelFormat::from_fourcc(FourCc::new(*b"444f")),
            Some(PixelFormat::Yuv444FullRange)
        );
        assert_eq!(PixelFormat::from_fourcc(FourCc::new(*b"BGRA")), None);
        for format in PixelFormat::ALL {
            assert_eq!(PixelFormat::from_fourcc(format.fourcc()), Some(format));
        }
    }

    /// Each chroma format and depth lands in the format an arm64 guest's own software decoder
    /// lands it in, by its range (m3-max-77228, 77237): `tv20` for 4:2:0 at 12 bits, `x422` for
    /// 4:2:2 at 10, and so on; a depth past 12, or 4:2:0 at 9, in none.
    #[test]
    fn each_chroma_format_and_depth_lands_in_the_guest_decoders_own_format() {
        let fourcc = |chroma, bits, full| PixelFormat::of(chroma, bits, full).map(|f| f.fourcc());
        let cases = [
            (Chroma::Yuv420, 8, *b"420v", *b"420f"),
            (Chroma::Yuv420, 10, *b"x420", *b"xf20"),
            (Chroma::Yuv420, 12, *b"tv20", *b"tf20"),
            (Chroma::Yuv422, 8, *b"422v", *b"422f"),
            (Chroma::Yuv422, 10, *b"x422", *b"xf22"),
            (Chroma::Yuv422, 12, *b"tv22", *b"tf22"),
            (Chroma::Yuv444, 8, *b"444v", *b"444f"),
            (Chroma::Yuv444, 10, *b"x444", *b"xf44"),
            (Chroma::Yuv444, 12, *b"tv44", *b"tf44"),
        ];
        for (chroma, bits, video, full) in cases {
            assert_eq!(fourcc(chroma, bits, false), Some(FourCc::new(video)));
            assert_eq!(fourcc(chroma, bits, true), Some(FourCc::new(full)));
            let format = PixelFormat::of(chroma, bits, false).expect("a format");
            assert_eq!((format.chroma(), format.bit_depth()), (chroma, bits));
            assert_eq!(format.bytes_per_sample(), if bits > 8 { 2 } else { 1 });
        }
        assert_eq!(PixelFormat::of(Chroma::Yuv420, 9, false), None);
        assert_eq!(PixelFormat::of(Chroma::Yuv444, 16, false), None);
        assert_eq!(Chroma::Yuv422.plane_size(1919, 1081), (960, 1081));
        assert_eq!(Chroma::Yuv420.plane_size(1919, 1081), (960, 541));
        assert_eq!(Chroma::Yuv444.plane_size(1919, 1081), (1919, 1081));
    }
}
