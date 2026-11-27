//! The VP codec configuration record, `vpcC` (VP Codec ISO Media File Format
//! Binding, 2.2): the box VideoToolbox's format description of a VP9 stream
//! carries as a sample description extension atom, as an app made it from
//! its container. It is a full box: version 1 and three flag bytes, then
//! the stream's profile, level, bit depth, chroma subsampling, range and
//! colour, and codec initialization data, which VP9 has none of.

use super::Error;

/// What a `vpcC` record says of its stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VpCodecConfiguration {
    /// `profile`: 0 to 3.
    pub profile: u8,
    /// `level`: the level times ten (10 is level 1), or 0 where the record
    /// names none.
    pub level: u8,
    /// `bitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// `chromaSubsampling`: 0 is 4:2:0 with chroma beside its luma
    /// vertically, 1 4:2:0 colocated with luma, 2 4:2:2, 3 4:4:4.
    pub chroma_subsampling: u8,
    /// `videoFullRangeFlag`.
    pub full_range: bool,
    /// `colourPrimaries` (ISO/IEC 23091-2).
    pub colour_primaries: u8,
    /// `transferCharacteristics`.
    pub transfer_characteristics: u8,
    /// `matrixCoefficients`.
    pub matrix_coefficients: u8,
}

impl VpCodecConfiguration {
    /// The fixed part of a version 1 record: the full box's version and
    /// flags, then eight bytes, the initialization data's length last.
    const FIXED_BYTES: usize = 12;

    /// Reads a `vpcC` record's payload, the box's version and flags first.
    /// A record shorter than its fixed fields, or of another version, is
    /// [`Error::BadVpcc`]; its initialization data, VP9's empty, is not
    /// read.
    pub fn parse(record: &[u8]) -> Result<Self, Error> {
        let fixed: &[u8; Self::FIXED_BYTES] = record
            .get(..Self::FIXED_BYTES)
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or(Error::BadVpcc)?;
        let [
            version,
            _,
            _,
            _,
            profile,
            level,
            packed,
            primaries,
            transfer,
            matrix,
            _,
            _,
        ] = *fixed;
        if version != 1 {
            return Err(Error::BadVpcc);
        }
        Ok(Self {
            profile,
            level,
            bit_depth: packed >> 4,
            chroma_subsampling: (packed >> 1) & 7,
            full_range: packed & 1 == 1,
            colour_primaries: primaries,
            transfer_characteristics: transfer,
            matrix_coefficients: matrix,
        })
    }

    /// Whether the record names 4:2:0 samples (either siting).
    pub fn is_4_2_0(&self) -> bool {
        self.chroma_subsampling <= 1
    }
}
