//! `CUVIDVP9PICPARAMS` (ffnvcodec's `dynlink_cuviddec.h`, NVDECAPI 13.1): one VP9 frame's
//! parameters, `CUVIDPICPARAMS::CodecSpecific.vp9`. The header's bit-fields are held in
//! words named for what they hold, each marked as ours, with their fields' positions in
//! [`bits`]; `crate::nvidia::layout`'s test checks them against the header.

use std::ffi::{c_int, c_short, c_uint, c_ushort};

use super::Zeroed;

/// `CUVIDVP9PICPARAMS`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CUVIDVP9PICPARAMS {
    pub(crate) width: c_uint,
    pub(crate) height: c_uint,
    pub(crate) LastRefIdx: u8,
    pub(crate) GoldenRefIdx: u8,
    pub(crate) AltRefIdx: u8,
    pub(crate) colorSpace: u8,
    /// The frame's bit-fields, `profile` to `reserved2Bits`: [`bits::frame`]. Name is ours.
    pub(crate) frame_word: c_ushort,
    pub(crate) reserved16Bits: c_ushort,
    pub(crate) refFrameSignBias: [u8; 4],
    pub(crate) bitDepthMinus8Luma: u8,
    pub(crate) bitDepthMinus8Chroma: u8,
    pub(crate) loopFilterLevel: u8,
    pub(crate) loopFilterSharpness: u8,
    pub(crate) modeRefLfEnabled: u8,
    pub(crate) log2_tile_columns: u8,
    pub(crate) log2_tile_rows: u8,
    /// The segmentation's bit-fields, `segmentEnabled` to `reserved4Bits`:
    /// [`bits::segmentation`]. Name is ours.
    pub(crate) segmentation_word: u8,
    pub(crate) segmentFeatureEnable: [[u8; 4]; 8],
    pub(crate) segmentFeatureData: [[c_short; 4]; 8],
    pub(crate) mb_segment_tree_probs: [u8; 7],
    pub(crate) segment_pred_probs: [u8; 3],
    pub(crate) reservedSegment16Bits: [u8; 2],
    pub(crate) qpYAc: c_int,
    pub(crate) qpYDc: c_int,
    pub(crate) qpChDc: c_int,
    pub(crate) qpChAc: c_int,
    pub(crate) activeRefIdx: [c_uint; 3],
    pub(crate) resetFrameContext: c_uint,
    pub(crate) mcomp_filter_type: c_uint,
    pub(crate) mbRefLfDelta: [c_uint; 4],
    pub(crate) mbModeLfDelta: [c_uint; 2],
    pub(crate) frameTagSize: c_uint,
    pub(crate) offsetToDctParts: c_uint,
    pub(crate) reserved128Bits: [c_uint; 4],
}

// SAFETY: repr(C), and holds only integers and arrays of them (`Zeroed`'s contract).
unsafe impl Zeroed for CUVIDVP9PICPARAMS {}

// The size ffnvcodec 13.1.15.0's header gives on x86-64 Linux; `crate::nvidia::layout`'s test
// checks every offset and bit-field too.
const _: () = assert!(size_of::<CUVIDVP9PICPARAMS>() == 220);

/// The positions of the header's bit-fields in the words above, from the least significant bit
/// up in declaration order (GCC's x86-64 layout).
pub(crate) mod bits {
    use super::super::bits::Bits;

    const fn b(shift: u32, width: u32) -> Bits {
        Bits { shift, width }
    }

    /// `CUVIDVP9PICPARAMS::frame_word`.
    pub(crate) mod frame {
        use super::{Bits, b};
        pub(crate) const profile: Bits = b(0, 3);
        pub(crate) const frameContextIdx: Bits = b(3, 2);
        pub(crate) const frameType: Bits = b(5, 1);
        pub(crate) const showFrame: Bits = b(6, 1);
        pub(crate) const errorResilient: Bits = b(7, 1);
        pub(crate) const frameParallelDecoding: Bits = b(8, 1);
        pub(crate) const subSamplingX: Bits = b(9, 1);
        pub(crate) const subSamplingY: Bits = b(10, 1);
        pub(crate) const intraOnly: Bits = b(11, 1);
        pub(crate) const allow_high_precision_mv: Bits = b(12, 1);
        pub(crate) const refreshEntropyProbs: Bits = b(13, 1);
    }

    /// `CUVIDVP9PICPARAMS::segmentation_word`.
    pub(crate) mod segmentation {
        use super::{Bits, b};
        pub(crate) const segmentEnabled: Bits = b(0, 1);
        pub(crate) const segmentMapUpdate: Bits = b(1, 1);
        pub(crate) const segmentMapTemporalUpdate: Bits = b(2, 1);
        pub(crate) const segmentFeatureMode: Bits = b(3, 1);
    }
}
