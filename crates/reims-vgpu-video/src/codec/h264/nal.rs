//! NAL units (H.264 7.3.1) and the two ways a stream carries them: with a
//! length before each, as in an `avcC` sample (ISO/IEC 14496-15, 5.3.3),
//! and after start codes, as in an Annex B byte stream. HEVC's units are
//! framed the same way (H.265 B.2, ISO/IEC 14496-15 8.3.3).

use super::Error;
use super::bits::BitReader;

/// `nal_unit_type` values this decoder acts on (Table 7-1). The field is
/// five bits of the stream, so any value can arrive: the rest are skipped.
pub mod nal_type {
    /// Coded slice of a non-IDR picture.
    pub const SLICE: u8 = 1;
    /// Coded slice data partition A.
    pub(crate) const SLICE_PARTITION_A: u8 = 2;
    /// Coded slice data partition C.
    pub(crate) const SLICE_PARTITION_C: u8 = 4;
    /// Coded slice of an IDR picture.
    pub const IDR: u8 = 5;
    /// Supplemental enhancement information.
    pub const SEI: u8 = 6;
    /// Sequence parameter set.
    pub const SPS: u8 = 7;
    /// Picture parameter set.
    pub const PPS: u8 = 8;
    /// Access unit delimiter.
    pub(crate) const AUD: u8 = 9;
    /// End of sequence.
    pub(crate) const END_OF_SEQUENCE: u8 = 10;
    /// End of stream.
    pub(crate) const END_OF_STREAM: u8 = 11;
}

/// One NAL unit, still escaped, from its header byte on.
#[derive(Clone, Copy, Debug)]
pub struct NalUnit<'a> {
    bytes: &'a [u8],
    /// The header byte, `bytes`' first.
    header: u8,
}

impl<'a> NalUnit<'a> {
    /// Checks the header byte: present, with `forbidden_zero_bit` clear.
    pub fn new(bytes: &'a [u8]) -> Result<Self, Error> {
        match bytes.first() {
            None => Err(Error::Truncated),
            Some(header) if header & 0x80 != 0 => Err(Error::OutOfRange("forbidden_zero_bit")),
            Some(&header) => Ok(Self { bytes, header }),
        }
    }

    /// The unit, header byte included, emulation prevention bytes kept.
    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// `nal_ref_idc`.
    pub fn ref_idc(&self) -> u8 {
        (self.header >> 5) & 3
    }

    /// `nal_unit_type`.
    pub fn unit_type(&self) -> u8 {
        self.header & 0x1f
    }

    /// Whether this is a slice of a primary coded picture this decoder
    /// reads: `nal_unit_type` 1 or 5.
    pub(crate) fn is_slice(&self) -> bool {
        matches!(self.unit_type(), nal_type::SLICE | nal_type::IDR)
    }
}

/// A NAL unit's length prefix that runs past the end of its sample: what
/// the framing H.264 and HEVC share refuses, which each codec's error
/// carries as its own `BadLengthPrefix`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BadLengthPrefix;

impl std::fmt::Display for BadLengthPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "a NAL unit's length runs past its sample")
    }
}

impl std::error::Error for BadLengthPrefix {}

/// H.264's own refusal of the sample.
impl From<BadLengthPrefix> for Error {
    fn from(BadLengthPrefix: BadLengthPrefix) -> Self {
        Self::BadLengthPrefix
    }
}

/// The NAL units of an `avcC` sample: each after a big-endian length of
/// `length_size` bytes.
pub struct LengthPrefixed<'a> {
    data: &'a [u8],
    length_size: usize,
}

impl<'a> LengthPrefixed<'a> {
    /// `length_size` is `lengthSizeMinusOne + 1` from the `avcC` record.
    pub fn new(data: &'a [u8], length_size: u8) -> Self {
        Self {
            data,
            length_size: usize::from(length_size),
        }
    }
}

impl<'a> Iterator for LengthPrefixed<'a> {
    type Item = Result<&'a [u8], BadLengthPrefix>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }
        let Some((prefix, rest)) = self.data.split_at_checked(self.length_size) else {
            self.data = &[];
            return Some(Err(BadLengthPrefix));
        };
        let length = prefix.iter().try_fold(0usize, |length, &byte| {
            length.checked_mul(256).map(|high| high | usize::from(byte))
        });
        let Some((unit, rest)) = length.and_then(|length| rest.split_at_checked(length)) else {
            self.data = &[];
            return Some(Err(BadLengthPrefix));
        };
        self.data = rest;
        Some(Ok(unit))
    }
}

/// How samples carry their NAL units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NalFormat {
    /// Each unit after a big-endian length of this many bytes (`avcC`).
    LengthPrefixed(u8),
    /// Each unit after a start code (Annex B).
    AnnexB,
}

/// A sample's NAL units, one at a time, as a decoder reads them. Every
/// length prefix is checked first, so a sample with a bad one yields
/// nothing; and nothing is kept per unit, which a sample of many tiny
/// ones would make many times its size.
pub(crate) fn sample_units(
    sample: &[u8],
    format: NalFormat,
) -> Result<impl Iterator<Item = &[u8]>, BadLengthPrefix> {
    let length_size = match format {
        NalFormat::LengthPrefixed(size) => Some(size),
        NalFormat::AnnexB => None,
    };
    if let Some(size) = length_size {
        LengthPrefixed::new(sample, size).try_for_each(|unit| unit.map(drop))?;
    }
    let prefixed = length_size.map(|size| LengthPrefixed::new(sample, size).flatten());
    let annex_b = length_size.is_none().then(|| AnnexB::new(sample));
    Ok(prefixed
        .into_iter()
        .flatten()
        .chain(annex_b.into_iter().flatten()))
}

/// The NAL units of an Annex B byte stream (B.1): each after a `00 00 01`
/// start code, with the zero bytes before the next start code dropped.
pub struct AnnexB<'a> {
    data: &'a [u8],
}

impl<'a> AnnexB<'a> {
    /// Skips anything before the first start code.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data: split_at_start_code(data).map_or(&[][..], |(_, after)| after),
        }
    }
}

impl<'a> Iterator for AnnexB<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        while !self.data.is_empty() {
            let (mut unit, rest) = split_at_start_code(self.data).unwrap_or((self.data, &[]));
            self.data = rest;
            // trailing_zero_8bits, and a leading_zero_8bits of the next
            // start code, belong to no unit.
            while let [before @ .., 0] = unit {
                unit = before;
            }
            if !unit.is_empty() {
                return Some(unit);
            }
        }
        None
    }
}

/// `data` around its first `00 00 01` start code: what comes before it,
/// and what comes after. The search is memchr's vectorized one, as it reads
/// each picture's every byte: measured on the M3 Max over the lane gate's
/// streams, 32 to 37 GB/s against 2.6 for a byte-wise window, 1 us of a
/// 1080p H.264 picture's 34 KB rather than 13.
fn split_at_start_code(data: &[u8]) -> Option<(&[u8], &[u8])> {
    let at = memchr::memmem::find(data, &[0, 0, 1])?;
    let (before, code_and_after) = data.split_at_checked(at)?;
    Some((before, code_and_after.get(3..)?))
}

/// What [`access_units`] needs of a NAL unit: whether it starts a new
/// access unit when it follows a slice, and whether it is a slice.
pub(crate) struct UnitKind {
    pub(crate) starts_access_unit: bool,
    pub(crate) is_slice: bool,
}

/// Splits an Annex B byte stream into access units (7.4.1.2.3), for
/// feeding a stream to a decoder one sample at a time.
///
/// A new access unit starts at an access unit delimiter, SPS, PPS or SEI,
/// or at the first slice of a primary picture (`first_mb_in_slice` 0),
/// whenever a slice has come since the last start. Arbitrary slice order
/// is not recognised, which is no loss: the decoder does not take it.
pub fn access_units(stream: &[u8]) -> Vec<&[u8]> {
    split_access_units(stream, |unit| {
        let nal = NalUnit::new(unit).ok()?;
        let starts_access_unit = match nal.unit_type() {
            nal_type::AUD | nal_type::SPS | nal_type::PPS | nal_type::SEI | 14..=18 => true,
            nal_type::SLICE | nal_type::IDR => first_mb_in_slice(unit).is_ok_and(|mb| mb == 0),
            _ => false,
        };
        Some(UnitKind {
            starts_access_unit,
            is_slice: nal.is_slice(),
        })
    })
}

/// Splits an Annex B byte stream into access units, each itself an Annex
/// B stream: a new one starts at a unit `kind` says starts one, when a
/// slice has come since the last start. A unit `kind` cannot read
/// (`None`) starts nothing, and is left out at the stream's end.
pub(crate) fn split_access_units(
    stream: &[u8],
    kind: impl Fn(&[u8]) -> Option<UnitKind>,
) -> Vec<&[u8]> {
    let mut units = Vec::new();
    let mut start: Option<usize> = None;
    let mut seen_slice = false;
    let mut end = 0usize;
    for unit in AnnexB::new(stream) {
        // A unit of the stream's own, so its offset is exact. Each sum and
        // difference below stays within the stream, so none saturates.
        let offset = unit.as_ptr().addr().saturating_sub(stream.as_ptr().addr());
        // Back up over the start code before the unit, and one leading
        // zero, so that each access unit is itself an Annex B stream.
        let leading_zero = offset
            .checked_sub(4)
            .and_then(|at| stream.get(at))
            .is_some_and(|&byte| byte == 0);
        let unit_start = offset.saturating_sub(if leading_zero { 4 } else { 3 });
        let Some(kind) = kind(unit) else {
            continue;
        };
        if kind.starts_access_unit && seen_slice {
            if let Some(s) = start {
                units.extend(stream.get(s..unit_start));
            }
            start = Some(unit_start);
            seen_slice = false;
        }
        if start.is_none() {
            start = Some(unit_start);
        }
        if kind.is_slice {
            seen_slice = true;
        }
        end = offset.saturating_add(unit.len());
    }
    if let Some(s) = start {
        units.extend(stream.get(s..end));
    }
    units
}

fn first_mb_in_slice(unit: &[u8]) -> Result<u32, Error> {
    let mut r = BitReader::new(unit);
    r.skip(8)?;
    r.ue()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_samples_units_come_one_at_a_time_once_every_prefix_reads() -> Result<(), BadLengthPrefix> {
        // A mebibyte of zero lengths: a million empty units, none kept.
        let zeros = vec![0u8; 1 << 20];
        assert_eq!(
            sample_units(&zeros, NalFormat::LengthPrefixed(1))?.count(),
            1 << 20
        );
        // A unit, then a prefix past the sample's end: nothing at all.
        let bad = [1, 0x65, 9, 0x41];
        assert_eq!(
            sample_units(&bad, NalFormat::LengthPrefixed(1)).err(),
            Some(BadLengthPrefix)
        );
        let annex_b = [0, 0, 1, 0x65, 7, 0, 0, 1, 0x41];
        let units: Vec<&[u8]> = sample_units(&annex_b, NalFormat::AnnexB)?.collect();
        assert_eq!(units, [&[0x65, 7][..], &[0x41][..]]);
        Ok(())
    }

    #[test]
    fn length_prefixed_units_split_for_each_length_size() -> Result<(), BadLengthPrefix> {
        for size in 1..=4u8 {
            let mut sample = Vec::new();
            for unit in [&[0x65u8, 1, 2][..], &[0x41, 9]] {
                let len = unit.len().to_be_bytes();
                sample.extend_from_slice(&len[len.len() - usize::from(size)..]);
                sample.extend_from_slice(unit);
            }
            let units: Vec<&[u8]> = LengthPrefixed::new(&sample, size).collect::<Result<_, _>>()?;
            assert_eq!(units, [&[0x65, 1, 2][..], &[0x41, 9]]);
        }
        Ok(())
    }

    #[test]
    fn a_length_too_wide_for_an_address_is_refused() {
        // Nine bytes of length: 2^64 + 1.
        let sample = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0x65];
        let units: Vec<_> = LengthPrefixed::new(&sample, 9).collect();
        assert_eq!(units, [Err(BadLengthPrefix)]);
    }

    #[test]
    fn a_length_past_the_sample_end_is_refused() {
        let sample = [0, 0, 0, 5, 0x65, 1];
        let units: Vec<_> = LengthPrefixed::new(&sample, 4).collect();
        assert_eq!(units, [Err(BadLengthPrefix)]);
    }

    #[test]
    fn annex_b_units_split_at_three_and_four_byte_start_codes() {
        let stream = [
            0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 0, 0, 3, 1,
        ];
        let units: Vec<&[u8]> = AnnexB::new(&stream).collect();
        assert_eq!(units, [&[0x67, 1, 2][..], &[0x68, 3], &[0x65, 0, 0, 3, 1]]);
    }

    #[test]
    fn access_units_start_at_parameter_sets_and_first_slices() {
        // SPS, PPS, IDR slice (first_mb 0: ue "1" -> 0x80), second slice
        // (first_mb 1: "010" -> 0x40), then a non-IDR slice with first_mb 0.
        let stream = [
            0, 0, 0, 1, 0x67, 0x42, //
            0, 0, 1, 0x68, 0xce, //
            0, 0, 1, 0x65, 0x88, //
            0, 0, 1, 0x65, 0x48, //
            0, 0, 1, 0x41, 0x9a,
        ];
        let units = access_units(&stream);
        assert_eq!(units.len(), 2);
        assert_eq!(units[0], &stream[..21]);
        assert_eq!(units[1], &stream[21..]);
    }
}
