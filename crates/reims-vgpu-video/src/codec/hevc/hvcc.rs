//! The `hvcC` record: HEVCDecoderConfigurationRecord (ISO/IEC 14496-15,
//! 8.3.3.1), which carries a stream's parameter sets and its NAL length
//! size. VideoToolbox's format description carries it as the `hvcC`
//! sample description extension atom.

use super::Error;

/// The record's fixed part, before its arrays: 23 bytes.
const HEADER_BYTES: usize = 23;

/// A parsed `hvcC` record, borrowing its NAL units from the record.
#[derive(Debug)]
pub struct DecoderConfiguration<'a> {
    /// `general_profile_idc`.
    pub general_profile_idc: u8,
    /// `bitDepthLumaMinus8`.
    pub bit_depth_luma_minus8: u8,
    /// `lengthSizeMinusOne + 1`: the bytes before each NAL unit of a
    /// sample.
    pub length_size: u8,
    /// Every array's NAL units, header bytes included, in the record's
    /// order: VPS, SPS and PPS by convention, then SEI.
    pub nal_units: Vec<&'a [u8]>,
}

impl<'a> DecoderConfiguration<'a> {
    /// Parses the record. Its `configurationVersion` must be 1.
    pub fn parse(record: &'a [u8]) -> Result<Self, Error> {
        let (header, mut rest) = record
            .split_first_chunk::<HEADER_BYTES>()
            .ok_or(Error::HvccTruncated)?;
        if header[0] != 1 {
            return Err(Error::HvccVersion);
        }
        let mut nal_units = Vec::new();
        for _ in 0..header[22] {
            // array_completeness, reserved and NAL_unit_type; then numNalus.
            let &[_, count @ ..] = take::<3>(&mut rest)?;
            for _ in 0..u16::from_be_bytes(count) {
                let length = u16::from_be_bytes(*take(&mut rest)?);
                nal_units.push(take_slice(&mut rest, usize::from(length))?);
            }
        }
        Ok(Self {
            general_profile_idc: header[1] & 0x1f,
            bit_depth_luma_minus8: header[17] & 7,
            // ISO/IEC 14496-15 allows 1, 2 and 4. A three-byte length is
            // just as unambiguous, so it is taken too, as for avcC.
            length_size: (header[21] & 3)
                .checked_add(1)
                .ok_or(Error::Overflow("lengthSizeMinusOne + 1"))?,
            nal_units,
        })
    }
}

/// The first `N` bytes of `rest`, taken off it.
fn take<'a, const N: usize>(rest: &mut &'a [u8]) -> Result<&'a [u8; N], Error> {
    let (taken, after) = rest.split_first_chunk().ok_or(Error::HvccArrayPastEnd)?;
    *rest = after;
    Ok(taken)
}

/// The first `n` bytes of `rest`, taken off it.
fn take_slice<'a>(rest: &mut &'a [u8], n: usize) -> Result<&'a [u8], Error> {
    let (taken, after) = rest.split_at_checked(n).ok_or(Error::HvccArrayPastEnd)?;
    *rest = after;
    Ok(taken)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record's fixed part: Main 10 (2), level 120 (4.0), 4:2:0, 10-bit,
    /// four-byte lengths, then `arrays` arrays.
    fn header(arrays: u8) -> Vec<u8> {
        let mut record = vec![1, 0x02, 0x20, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 120];
        record.extend_from_slice(&[0xf0, 0x00, 0xfc, 0xfd, 0xfa, 0xfa, 0, 0, 0x0f, arrays]);
        record
    }

    #[test]
    fn a_record_yields_its_length_size_and_nal_units() -> Result<(), Error> {
        let mut record = header(2);
        record.extend_from_slice(&[0xa0, 0, 1, 0, 2, 0x40, 0x01]);
        record.extend_from_slice(&[0xa1, 0, 2, 0, 3, 0x42, 0x01, 0x01, 0, 2, 0x42, 0x01]);
        let config = DecoderConfiguration::parse(&record)?;
        assert_eq!(config.general_profile_idc, 2);
        assert_eq!(config.bit_depth_luma_minus8, 2);
        assert_eq!(config.length_size, 4);
        assert_eq!(
            config.nal_units,
            [&[0x40, 0x01][..], &[0x42, 0x01, 0x01], &[0x42, 0x01]]
        );
        Ok(())
    }

    #[test]
    fn a_truncated_record_is_refused() {
        let mut record = header(1);
        record.extend_from_slice(&[0xa0, 0, 1, 0, 9, 0x40]);
        assert!(DecoderConfiguration::parse(&record).is_err());
        assert!(DecoderConfiguration::parse(&header(0)[..20]).is_err());
    }
}
