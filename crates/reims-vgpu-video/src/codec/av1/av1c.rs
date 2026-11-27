//! `av1C`, the AV1 codec configuration record (AV1 Codec ISO Media File Format Binding 2.3):
//! what an MP4 sample entry and a VideoToolbox format description carry, its sequence header
//! among its `configOBUs`.

use super::Error;

/// An `AV1CodecConfigurationRecord`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecoderConfiguration<'a> {
    /// `seq_profile`.
    pub seq_profile: u8,
    /// `seq_level_idx_0`.
    pub seq_level_idx_0: u8,
    /// `seq_tier_0`.
    pub seq_tier_0: bool,
    /// `high_bitdepth`.
    pub high_bitdepth: bool,
    /// `twelve_bit`.
    pub twelve_bit: bool,
    /// `monochrome`.
    pub monochrome: bool,
    /// `chroma_subsampling_x`.
    pub chroma_subsampling_x: bool,
    /// `chroma_subsampling_y`.
    pub chroma_subsampling_y: bool,
    /// `chroma_sample_position`.
    pub chroma_sample_position: u8,
    /// `configOBUs`: the OBUs after the four fixed bytes, a sequence header among them where
    /// the record has one.
    pub config_obus: &'a [u8],
}

impl<'a> DecoderConfiguration<'a> {
    /// Reads a record.
    pub fn parse(record: &'a [u8]) -> Result<Self, Error> {
        let (fixed, config_obus) = record.split_at_checked(4).ok_or(Error::Av1cTruncated)?;
        let &[first, second, third, _fourth] = fixed else {
            return Err(Error::Av1cTruncated);
        };
        // marker (1) and version (7).
        if first != 0x81 {
            return Err(Error::Av1cVersion);
        }
        Ok(Self {
            seq_profile: second >> 5,
            seq_level_idx_0: second & 0x1f,
            seq_tier_0: third & 0x80 != 0,
            high_bitdepth: third & 0x40 != 0,
            twelve_bit: third & 0x20 != 0,
            monochrome: third & 0x10 != 0,
            chroma_subsampling_x: third & 0x08 != 0,
            chroma_subsampling_y: third & 0x04 != 0,
            chroma_sample_position: third & 0x03,
            config_obus,
        })
    }

    /// The bit depth the record says: 8, 10 or 12.
    pub fn bit_depth(&self) -> u8 {
        match (self.high_bitdepth, self.twelve_bit) {
            (false, _) => 8,
            (true, false) => 10,
            (true, true) => 12,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_reads_its_fields_and_keeps_its_obus() {
        // Main profile, level 8 (4.0), main tier, 10 bits, 4:2:0, colocated.
        let record = [0x81, 0x08, 0x4d, 0x00, 0x0a, 0x00];
        let config = DecoderConfiguration::parse(&record);
        assert_eq!(
            config,
            Ok(DecoderConfiguration {
                seq_profile: 0,
                seq_level_idx_0: 8,
                seq_tier_0: false,
                high_bitdepth: true,
                twelve_bit: false,
                monochrome: false,
                chroma_subsampling_x: true,
                chroma_subsampling_y: true,
                chroma_sample_position: 1,
                config_obus: &[0x0a, 0x00],
            })
        );
        assert_eq!(config.map(|c| c.bit_depth()), Ok(10));
    }

    #[test]
    fn a_short_record_or_another_version_is_refused() {
        assert_eq!(
            DecoderConfiguration::parse(&[0x81, 0x00, 0x0c]),
            Err(Error::Av1cTruncated)
        );
        assert_eq!(
            DecoderConfiguration::parse(&[0x82, 0x00, 0x0c, 0x00]),
            Err(Error::Av1cVersion)
        );
        assert_eq!(
            DecoderConfiguration::parse(&[0x01, 0x00, 0x0c, 0x00]),
            Err(Error::Av1cVersion)
        );
    }
}
