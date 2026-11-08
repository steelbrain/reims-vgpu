//! The `avcC` record: AVCDecoderConfigurationRecord (ISO/IEC 14496-15,
//! 5.3.3.1), which carries a stream's parameter sets and its NAL length
//! size. VideoToolbox's format description carries it as the `avcC`
//! sample description extension atom.

use super::Error;

/// A parsed `avcC` record, borrowing its parameter sets from the record.
#[derive(Debug)]
pub struct DecoderConfiguration<'a> {
    /// `AVCProfileIndication`.
    pub profile_indication: u8,
    /// `lengthSizeMinusOne + 1`: the bytes before each NAL unit of a sample.
    pub length_size: u8,
    /// The SPS NAL units, header bytes included.
    pub sequence_parameter_sets: Vec<&'a [u8]>,
    /// The PPS NAL units, header bytes included.
    pub picture_parameter_sets: Vec<&'a [u8]>,
}

impl<'a> DecoderConfiguration<'a> {
    /// Parses the record. Anything after the PPS list (the High profiles'
    /// chroma and bit depth fields, and their SPS extensions) is ignored:
    /// the SPS says the same.
    pub fn parse(record: &'a [u8]) -> Result<Self, Error> {
        let mut r = Bytes(record);
        if r.u8()? != 1 {
            return Err(Error::AvccVersion);
        }
        let profile_indication = r.u8()?;
        // profile_compatibility and AVCLevelIndication: the SPS says the
        // same.
        r.u8()?;
        r.u8()?;
        // ISO/IEC 14496-15 allows 1, 2 and 4. A three-byte length is just
        // as unambiguous, so it is taken too.
        let length_size = (r.u8()? & 3)
            .checked_add(1)
            .ok_or(Error::Overflow("lengthSizeMinusOne + 1"))?;
        let sps_count = r.u8()? & 0x1f;
        let mut sequence_parameter_sets = Vec::new();
        for _ in 0..sps_count {
            sequence_parameter_sets.push(r.unit()?);
        }
        let pps_count = r.u8()?;
        let mut picture_parameter_sets = Vec::new();
        for _ in 0..pps_count {
            picture_parameter_sets.push(r.unit()?);
        }
        Ok(Self {
            profile_indication,
            length_size,
            sequence_parameter_sets,
            picture_parameter_sets,
        })
    }
}

struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn u8(&mut self) -> Result<u8, Error> {
        let (&first, rest) = self.0.split_first().ok_or(Error::AvccTruncated)?;
        self.0 = rest;
        Ok(first)
    }

    fn unit(&mut self) -> Result<&'a [u8], Error> {
        let length = u16::from_be_bytes([self.u8()?, self.u8()?]);
        let (unit, rest) = self
            .0
            .split_at_checked(usize::from(length))
            .ok_or(Error::AvccParameterSetPastEnd)?;
        self.0 = rest;
        Ok(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_yields_its_length_size_and_parameter_sets() -> Result<(), Error> {
        let record = [
            1, 0x64, 0, 0x28, 0xff, 0xe1, 0, 3, 0x67, 0x64, 0x28, 1, 0, 2, 0x68, 0xee,
        ];
        let config = DecoderConfiguration::parse(&record)?;
        assert_eq!(config.profile_indication, 0x64);
        assert_eq!(config.length_size, 4);
        assert_eq!(config.sequence_parameter_sets, [&[0x67, 0x64, 0x28][..]]);
        assert_eq!(config.picture_parameter_sets, [&[0x68, 0xee][..]]);
        Ok(())
    }

    #[test]
    fn a_truncated_record_is_refused() {
        let record = [1, 0x64, 0, 0x28, 0xff, 0xe1, 0, 9, 0x67];
        assert!(DecoderConfiguration::parse(&record).is_err());
    }
}
