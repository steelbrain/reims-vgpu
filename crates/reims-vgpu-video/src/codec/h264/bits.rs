//! Reading a NAL unit's syntax elements (H.264 7.2): fixed-length fields
//! and Exp-Golomb codes (9.1), with the emulation prevention bytes (7.4.1)
//! removed as they are met, so that positions are in the RBSP. HEVC's
//! units are read with it too: H.265 7.2, 7.4.2 and 9.2 are the same. The
//! reader refuses in the error of the codec it reads for
//! ([`ReadError`]): [`BitReader`] is H.264's, `hevc::bits::BitReader`
//! HEVC's.

use std::marker::PhantomData;

use super::Error;
use crate::codec::bits::ReadError;
use crate::codec::num::ToUsize;

impl ReadError for Error {
    const TRUNCATED: Self = Self::Truncated;

    fn out_of_range(name: &'static str) -> Self {
        Self::OutOfRange(name)
    }
}

/// H.264's reader, which refuses with H.264's errors.
pub(crate) type BitReader<'a> = RbspReader<'a, Error>;

/// The RBSP bytes of an escaped NAL unit: each emulation prevention byte,
/// the 03 after two zero bytes (7.4.1), is taken out as it is met.
#[derive(Clone, Debug)]
pub(crate) struct Unescape<'a> {
    rest: &'a [u8],
    /// The zero bytes just read in a row, up to 2.
    zeros: u8,
    /// The emulation prevention bytes taken out so far.
    removed: usize,
}

impl<'a> Unescape<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            rest: data,
            zeros: 0,
            removed: 0,
        }
    }
}

impl Iterator for Unescape<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        let (&byte, rest) = self.rest.split_first()?;
        self.rest = rest;
        let byte = if self.zeros == 2 && byte == 0x03 {
            // emulation_prevention_three_byte: the byte after it is data,
            // whatever it is.
            let (&byte, rest) = self.rest.split_first()?;
            self.rest = rest;
            // At most one in two of the unit's bytes.
            self.removed = self.removed.saturating_add(1);
            self.zeros = 0;
            byte
        } else {
            byte
        };
        self.zeros = if byte == 0 {
            self.zeros.saturating_add(1).min(2)
        } else {
            0
        };
        Some(byte)
    }
}

/// A reader over one escaped NAL unit, header byte included, that refuses
/// with `E`.
///
/// Positions count the bits of the NAL unit with its emulation prevention
/// bytes removed. VA-API wants the slice header's length in exactly those
/// terms (`VASliceParameterBufferH264::slice_data_bit_offset`, va.h).
pub(crate) struct RbspReader<'a, E> {
    /// The whole unit, for `more_rbsp_data`.
    data: &'a [u8],
    /// The RBSP bytes not yet loaded.
    bytes: Unescape<'a>,
    /// The byte being read, and how many of its low bits are still unread.
    byte: u8,
    bits_left: u8,
    /// The RBSP bytes loaded so far.
    loaded: usize,
    error: PhantomData<E>,
}

impl<'a, E: ReadError> RbspReader<'a, E> {
    /// A reader at the first bit of `data`.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bytes: Unescape::new(data),
            byte: 0,
            bits_left: 0,
            loaded: 0,
            error: PhantomData,
        }
    }

    /// The number of RBSP bits read so far.
    pub(crate) fn position(&self) -> usize {
        // Neither saturates: `loaded` is at most the unit's length.
        self.loaded
            .saturating_mul(8)
            .saturating_sub(usize::from(self.bits_left))
    }

    /// The emulation prevention bytes skipped so far.
    pub(crate) fn emulation_prevention_bytes(&self) -> usize {
        self.bytes.removed
    }

    /// Whether the next bit starts a byte (`byte_aligned()`, 7.2).
    pub(crate) fn byte_aligned(&self) -> bool {
        self.bits_left == 0
    }

    /// `u(1)`.
    pub(crate) fn bit(&mut self) -> Result<bool, E> {
        self.bits_left = match self.bits_left.checked_sub(1) {
            Some(left) => left,
            None => {
                self.byte = self.bytes.next().ok_or(E::TRUNCATED)?;
                // A byte past a usize's count is past the data.
                self.loaded = self.loaded.checked_add(1).ok_or(E::TRUNCATED)?;
                7
            }
        };
        Ok((self.byte >> self.bits_left) & 1 == 1)
    }

    /// `u(n)`, for `n` up to 32.
    pub(crate) fn bits(&mut self, n: u32) -> Result<u32, E> {
        debug_assert!(n <= 32);
        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 1) | u64::from(self.bit()?);
        }
        u32::try_from(value).map_err(|_| E::out_of_range("u(n)"))
    }

    /// `u(n)`, for `n` up to 8.
    pub(crate) fn u8(&mut self, n: u32) -> Result<u8, E> {
        debug_assert!(n <= 8);
        u8::try_from(self.bits(n)?).map_err(|_| E::out_of_range("u(n)"))
    }

    /// `u(16)`.
    pub(crate) fn u16(&mut self) -> Result<u16, E> {
        u16::try_from(self.bits(16)?).map_err(|_| E::out_of_range("u(n)"))
    }

    /// Skips `n` bits.
    pub(crate) fn skip(&mut self, n: u32) -> Result<(), E> {
        for _ in 0..n {
            self.bit()?;
        }
        Ok(())
    }

    /// `ue(v)` (9.1). A code of more than 31 leading zero bits, past 32
    /// bits of value, is refused.
    pub(crate) fn ue(&mut self) -> Result<u32, E> {
        for leading_zeros in 0..32 {
            if self.bit()? {
                // 9-2: codeNum is 2^leadingZeroBits − 1 plus the suffix,
                // which is the code from its leading one on, less one: a
                // code of 1 at least.
                let code = (1u64 << leading_zeros) | u64::from(self.bits(leading_zeros)?);
                return code
                    .checked_sub(1)
                    .and_then(|code_num| u32::try_from(code_num).ok())
                    .ok_or_else(|| E::out_of_range("Exp-Golomb code"));
            }
        }
        Err(E::out_of_range("Exp-Golomb code"))
    }

    /// `ue(v)` checked against the inclusive range the semantics give it.
    pub(crate) fn ue_max(&mut self, max: u32, name: &'static str) -> Result<u32, E> {
        let value = self.ue()?;
        if value > max {
            return Err(E::out_of_range(name));
        }
        Ok(value)
    }

    /// `ue(v)` of a syntax element whose range, 0 to `max`, fits a byte.
    pub(crate) fn ue_u8(&mut self, max: u8, name: &'static str) -> Result<u8, E> {
        u8::try_from(self.ue()?)
            .ok()
            .filter(|&value| value <= max)
            .ok_or_else(|| E::out_of_range(name))
    }

    /// `ue(v)` of a syntax element whose range, 0 to `max`, fits 16 bits.
    pub(crate) fn ue_u16(&mut self, max: u16, name: &'static str) -> Result<u16, E> {
        u16::try_from(self.ue()?)
            .ok()
            .filter(|&value| value <= max)
            .ok_or_else(|| E::out_of_range(name))
    }

    /// `se(v)` (9.1.1).
    pub(crate) fn se(&mut self) -> Result<i32, E> {
        let code = self.ue()?;
        // Table 9-3: codes 1, 2, 3, 4 ... are 1, −1, 2, −2 ...
        let magnitude = i64::from(code.div_ceil(2));
        let value = if code % 2 == 1 {
            Some(magnitude)
        } else {
            magnitude.checked_neg()
        };
        value
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| E::out_of_range("Exp-Golomb code"))
    }

    /// `se(v)` checked against the inclusive range the semantics give it.
    pub(crate) fn se_range(&mut self, min: i32, max: i32, name: &'static str) -> Result<i32, E> {
        let value = self.se()?;
        if !(min..=max).contains(&value) {
            return Err(E::out_of_range(name));
        }
        Ok(value)
    }

    /// `se(v)` of a syntax element whose range, `min` to `max`, fits a
    /// byte.
    pub(crate) fn se_i8(&mut self, min: i8, max: i8, name: &'static str) -> Result<i8, E> {
        i8::try_from(self.se()?)
            .ok()
            .filter(|value| (min..=max).contains(value))
            .ok_or_else(|| E::out_of_range(name))
    }

    /// `se(v)` of a syntax element whose range, `min` to `max`, fits 16
    /// bits.
    pub(crate) fn se_i16(&mut self, min: i16, max: i16, name: &'static str) -> Result<i16, E> {
        i16::try_from(self.se()?)
            .ok()
            .filter(|value| (min..=max).contains(value))
            .ok_or_else(|| E::out_of_range(name))
    }

    /// `more_rbsp_data()` (7.2): whether anything but `rbsp_trailing_bits()`
    /// is left. The stop bit is the last set bit of the RBSP.
    pub(crate) fn more_rbsp_data(&self) -> bool {
        rbsp_stop_bit(self.data).is_some_and(|stop| self.position() < stop)
    }
}

/// `Ceil(Log2(n))` (5.7): the bits of a `u(v)` index below `n`.
pub(crate) fn ceil_log2(n: usize) -> u32 {
    n.checked_next_power_of_two()
        .map_or(usize::BITS, usize::trailing_zeros)
}

/// The RBSP position of `rbsp_stop_one_bit`: the last set bit once the
/// emulation prevention bytes are gone.
fn rbsp_stop_bit(data: &[u8]) -> Option<usize> {
    let (index, byte) = Unescape::new(data)
        .enumerate()
        .filter(|&(_, byte)| byte != 0)
        .last()?;
    // The byte's lowest set bit, counted from its most significant: the
    // byte is not 0, so it has one.
    let bit = 7u32.checked_sub(byte.trailing_zeros())?;
    index.checked_mul(8)?.checked_add(bit.to_usize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_codes_decode_to_their_values() -> Result<(), Error> {
        // 1 | 010 | 011 | 00100 | 00101 | 0001000 -> 0, 1, 2, 3, 4, 7
        let data = [0b1010_0110, 0b0100_0010, 0b1000_1000];
        let mut r = BitReader::new(&data);
        let values: Vec<u32> = (0..6).map(|_| r.ue()).collect::<Result<_, _>>()?;
        assert_eq!(values, [0, 1, 2, 3, 4, 7]);
        let mut r = BitReader::new(&data);
        let values: Vec<i32> = (0..6).map(|_| r.se()).collect::<Result<_, _>>()?;
        assert_eq!(values, [0, 1, -1, 2, -2, 4]);
        Ok(())
    }

    #[test]
    fn emulation_prevention_bytes_are_skipped_and_not_counted() -> Result<(), Error> {
        let data = [0x00, 0x00, 0x03, 0x01, 0xff];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(16)?, 0);
        assert_eq!(r.bits(8)?, 0x01);
        assert_eq!(r.position(), 24);
        assert_eq!(r.bits(8)?, 0xff);
        assert!(r.bit().is_err());
        Ok(())
    }

    #[test]
    fn emulation_prevention_bytes_are_counted() -> Result<(), Error> {
        let data = [0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x03, 0x00, 0xff];
        let mut r = BitReader::new(&data);
        r.bits(16)?;
        assert_eq!(r.emulation_prevention_bytes(), 0);
        r.bits(8)?;
        assert_eq!(r.emulation_prevention_bytes(), 1);
        assert!(r.byte_aligned());
        r.bits(24)?;
        assert_eq!(r.emulation_prevention_bytes(), 2);
        r.bit()?;
        assert!(!r.byte_aligned());
        Ok(())
    }

    #[test]
    fn a_three_after_one_zero_is_data() -> Result<(), Error> {
        let data = [0x00, 0x03, 0x00];
        let mut r = BitReader::new(&data);
        assert_eq!(r.bits(24)?, 0x0000_0300);
        Ok(())
    }

    #[test]
    fn more_rbsp_data_stops_at_the_trailing_bits() -> Result<(), Error> {
        // One data bit (1), then the stop bit and alignment: 1 1 000000.
        let data = [0b1100_0000];
        let mut r = BitReader::new(&data);
        assert!(r.more_rbsp_data());
        assert!(r.bit()?);
        assert!(!r.more_rbsp_data());
        Ok(())
    }

    #[test]
    fn ceil_log2_counts_index_bits() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(510), 9);
        assert_eq!(ceil_log2(8160), 13);
        assert_eq!(ceil_log2(usize::MAX), 64);
    }

    #[test]
    fn typed_reads_refuse_values_past_their_range() -> Result<(), Error> {
        // ue 4 (00101), then ue 3 (00100).
        let mut r = BitReader::new(&[0x29, 0x00]);
        assert_eq!(r.ue_u8(3, "four"), Err(Error::OutOfRange("four")));
        assert_eq!(r.ue_u8(3, "three")?, 3);
        // ue 256: past a byte, whatever the range.
        let mut r = BitReader::new(&[0x00, 0x80, 0x80]);
        assert_eq!(r.ue_u8(255, "wide"), Err(Error::OutOfRange("wide")));
        // se -2 (00101), then se 1 (010).
        let mut r = BitReader::new(&[0x2a]);
        assert_eq!(r.se_i8(-1, 1, "low"), Err(Error::OutOfRange("low")));
        assert_eq!(r.se_i8(-1, 1, "one")?, 1);
        Ok(())
    }

    #[test]
    fn unescaping_drops_each_emulation_prevention_byte() {
        // The 03 after two zeros goes and the 03 after it stays; a unit's
        // last byte, a 03 after two zeros, goes too (7.4.1).
        let data = [0x00, 0x00, 0x03, 0x03, 0x00, 0x00, 0x03];
        let mut bytes = Unescape::new(&data);
        assert_eq!(bytes.by_ref().collect::<Vec<_>>(), [0, 0, 3, 0, 0]);
        assert_eq!(bytes.removed, 1);
    }

    #[test]
    fn overlong_exp_golomb_codes_are_refused() {
        let data = [0u8; 8];
        let mut r = BitReader::new(&data);
        assert_eq!(r.ue(), Err(Error::OutOfRange("Exp-Golomb code")));
    }
}
