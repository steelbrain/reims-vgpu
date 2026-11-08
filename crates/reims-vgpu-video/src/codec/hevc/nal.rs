//! HEVC's NAL units (H.265 7.3.1): the two-byte header, the unit types
//! the decoder acts on, and the splitting of an Annex B byte stream into
//! access units. The framing itself, length prefixes and start codes, is
//! H.264's ([`crate::codec::h264::nal::sample_units`]).

use super::Error;
use super::bits::BitReader;
use crate::codec::h264::nal::{UnitKind, split_access_units};

/// `nal_unit_type` values (Table 7-1). The field is six bits of the
/// stream, so any value can arrive: the decoder skips those it does not
/// act on.
pub mod nal_type {
    /// Trailing picture, sub-layer non-reference.
    pub(crate) const TRAIL_N: u8 = 0;
    /// Trailing picture.
    pub const TRAIL_R: u8 = 1;
    /// Random access decodable leading picture, sub-layer non-reference.
    pub(crate) const RADL_N: u8 = 6;
    /// Random access decodable leading picture.
    pub(crate) const RADL_R: u8 = 7;
    /// Random access skipped leading picture, sub-layer non-reference.
    pub(crate) const RASL_N: u8 = 8;
    /// Random access skipped leading picture.
    pub(crate) const RASL_R: u8 = 9;
    /// Broken link access, with leading pictures.
    pub const BLA_W_LP: u8 = 16;
    /// Broken link access, with no leading pictures.
    pub(crate) const BLA_N_LP: u8 = 18;
    /// Instantaneous decoding refresh, with decodable leading pictures.
    pub(crate) const IDR_W_RADL: u8 = 19;
    /// Instantaneous decoding refresh, with no leading pictures.
    pub(crate) const IDR_N_LP: u8 = 20;
    /// Clean random access.
    pub const CRA_NUT: u8 = 21;
    /// The last IRAP type, reserved.
    pub(crate) const RSV_IRAP_VCL23: u8 = 23;
    /// The last VCL type, reserved.
    pub const RSV_VCL31: u8 = 31;
    /// Video parameter set.
    pub const VPS: u8 = 32;
    /// Sequence parameter set.
    pub const SPS: u8 = 33;
    /// Picture parameter set.
    pub const PPS: u8 = 34;
    /// Access unit delimiter.
    pub(crate) const AUD: u8 = 35;
    /// End of sequence.
    pub(crate) const EOS: u8 = 36;
    /// End of bitstream.
    pub(crate) const EOB: u8 = 37;
    /// Supplemental enhancement information, before the picture's slices.
    pub const PREFIX_SEI: u8 = 39;
}

/// One NAL unit, still escaped, from its two header bytes on.
#[derive(Clone, Copy, Debug)]
pub struct NalUnit<'a> {
    bytes: &'a [u8],
    /// The header, `bytes`' first two.
    header: [u8; 2],
    /// `TemporalId`.
    temporal_id: u8,
}

impl<'a> NalUnit<'a> {
    /// Checks the header (7.3.1.2): two bytes, `forbidden_zero_bit` clear
    /// and `nuh_temporal_id_plus1` not zero.
    pub fn new(bytes: &'a [u8]) -> Result<Self, Error> {
        let Some(&header @ [first, second]) = bytes.first_chunk() else {
            return Err(Error::Truncated);
        };
        if first & 0x80 != 0 {
            return Err(Error::OutOfRange("forbidden_zero_bit"));
        }
        let temporal_id = (second & 7)
            .checked_sub(1)
            .ok_or(Error::OutOfRange("nuh_temporal_id_plus1"))?;
        Ok(Self {
            bytes,
            header,
            temporal_id,
        })
    }

    /// The unit, header included, emulation prevention bytes kept.
    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// `nal_unit_type`.
    pub fn unit_type(&self) -> u8 {
        let [first, _] = self.header;
        (first >> 1) & 0x3f
    }

    /// `nuh_layer_id`.
    pub fn layer_id(&self) -> u8 {
        let [first, second] = self.header;
        ((first & 1) << 5) | (second >> 3)
    }

    /// `TemporalId`: `nuh_temporal_id_plus1 - 1`.
    pub fn temporal_id(&self) -> u8 {
        self.temporal_id
    }

    /// Whether the unit is a slice segment (a VCL NAL unit).
    pub(crate) fn is_vcl(&self) -> bool {
        self.unit_type() <= nal_type::RSV_VCL31
    }
}

/// Whether a picture of `unit_type` is an IRAP picture: BLA, IDR or CRA
/// (or a reserved IRAP type).
pub fn is_irap(unit_type: u8) -> bool {
    (nal_type::BLA_W_LP..=nal_type::RSV_IRAP_VCL23).contains(&unit_type)
}

/// Whether a picture of `unit_type` is an IDR picture.
pub fn is_idr(unit_type: u8) -> bool {
    matches!(unit_type, nal_type::IDR_W_RADL | nal_type::IDR_N_LP)
}

/// Whether a picture of `unit_type` is a BLA picture.
pub fn is_bla(unit_type: u8) -> bool {
    (nal_type::BLA_W_LP..=nal_type::BLA_N_LP).contains(&unit_type)
}

/// Whether a picture of `unit_type` is a RASL picture.
pub fn is_rasl(unit_type: u8) -> bool {
    matches!(unit_type, nal_type::RASL_N | nal_type::RASL_R)
}

/// Whether a picture of `unit_type` is a RADL picture.
pub fn is_radl(unit_type: u8) -> bool {
    matches!(unit_type, nal_type::RADL_N | nal_type::RADL_R)
}

/// Whether a picture of `unit_type` is a sub-layer non-reference picture
/// (3.149): `TRAIL_N`, `TSA_N`, `STSA_N`, `RADL_N`, `RASL_N` and the
/// reserved `RSV_VCL_N10`, `_N12` and `_N14`.
pub(crate) fn is_sub_layer_non_reference(unit_type: u8) -> bool {
    unit_type <= 14 && unit_type.is_multiple_of(2)
}

/// Splits an Annex B byte stream into access units (7.4.2.4.4), for
/// feeding a stream to a decoder one sample at a time.
///
/// After a slice segment, a new access unit starts at an access unit
/// delimiter, a VPS, SPS or PPS, a prefix SEI, a unit of types 41 to 44 or
/// 48 to 55, or the first slice segment of a picture
/// (`first_slice_segment_in_pic_flag`), each of layer 0.
pub fn access_units(stream: &[u8]) -> Vec<&[u8]> {
    split_access_units(stream, |unit| {
        let nal = NalUnit::new(unit).ok()?;
        let starts_access_unit = nal.layer_id() == 0
            && match nal.unit_type() {
                nal_type::VPS..=nal_type::AUD | nal_type::PREFIX_SEI | 41..=44 | 48..=55 => true,
                t if t <= nal_type::RSV_VCL31 => unit.get(2).is_some_and(|b| b & 0x80 != 0),
                _ => false,
            };
        Some(UnitKind {
            starts_access_unit,
            is_slice: nal.is_vcl(),
        })
    })
}

/// A reader over a unit's payload, past its two header bytes.
pub(crate) fn payload_reader<'a>(nal: &NalUnit<'a>) -> Result<BitReader<'a>, Error> {
    let mut reader = BitReader::new(nal.bytes());
    reader.skip(16)?;
    Ok(reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_header_fields_read_from_their_bits() -> Result<(), Error> {
        // nal_unit_type 19 (IDR_W_RADL), layer 0, TemporalId 0.
        let idr = NalUnit::new(&[0x26, 0x01, 0xaf])?;
        assert_eq!(idr.unit_type(), nal_type::IDR_W_RADL);
        assert_eq!(idr.layer_id(), 0);
        assert_eq!(idr.temporal_id(), 0);
        assert!(idr.is_vcl());
        // nal_unit_type 33 (SPS), layer 33, TemporalId 2.
        let sps = NalUnit::new(&[0x43, 0x0b])?;
        assert_eq!(sps.unit_type(), nal_type::SPS);
        assert_eq!(sps.layer_id(), 33);
        assert_eq!(sps.temporal_id(), 2);
        assert_eq!(
            NalUnit::new(&[0x26, 0x00]).err(),
            Some(Error::OutOfRange("nuh_temporal_id_plus1"))
        );
        assert_eq!(NalUnit::new(&[0x26]).err(), Some(Error::Truncated));
        Ok(())
    }

    #[test]
    fn picture_kinds_follow_table_7_1() {
        assert!(is_irap(nal_type::CRA_NUT) && is_irap(nal_type::BLA_W_LP));
        assert!(!is_irap(nal_type::RASL_R) && !is_irap(24));
        assert!(is_idr(nal_type::IDR_N_LP) && !is_idr(nal_type::CRA_NUT));
        assert!(is_sub_layer_non_reference(nal_type::TRAIL_N));
        assert!(is_sub_layer_non_reference(nal_type::RASL_N));
        assert!(!is_sub_layer_non_reference(nal_type::TRAIL_R));
        assert!(!is_sub_layer_non_reference(16));
    }

    #[test]
    fn access_units_start_at_parameter_sets_and_first_slice_segments() {
        // VPS, SPS, PPS, an IDR's two slice segments (the second's first
        // byte has first_slice_segment_in_pic_flag 0), then a TRAIL_R that
        // starts a picture, then a suffix SEI that stays with it.
        let stream = [
            0, 0, 0, 1, 0x40, 0x01, 0x0c, //
            0, 0, 1, 0x42, 0x01, 0x01, //
            0, 0, 1, 0x44, 0x01, 0xc1, //
            0, 0, 1, 0x26, 0x01, 0xaf, //
            0, 0, 1, 0x26, 0x01, 0x40, //
            0, 0, 1, 0x02, 0x01, 0xd0, //
            0, 0, 1, 0x50, 0x01, 0x05,
        ];
        let units = access_units(&stream);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], &stream[..31]);
        assert_eq!(units[1], &stream[31..]);
    }
}
