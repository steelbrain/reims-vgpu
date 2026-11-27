//! Reading AV1's syntax elements (AV1 4.10 and 8.1): bits from each byte's most significant on,
//! with no emulation prevention, so that a position is a plain bit count of the OBU.

use super::Error;

/// A reader at a bit of an OBU's payload.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// The bits read so far.
    position: usize,
}

impl<'a> BitReader<'a> {
    /// A reader at the first bit of `data`.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// `f(1)`.
    pub(crate) fn bit(&mut self) -> Result<bool, Error> {
        let byte = self.data.get(self.position / 8).ok_or(Error::Truncated)?;
        // 7 - position % 8: the bit's shift within its byte.
        let shift = !self.position & 7;
        // At most the data's bit count, which a usize holds.
        self.position = self.position.saturating_add(1);
        Ok((byte >> shift) & 1 == 1)
    }

    /// `f(n)`, for `n` up to 32 (4.10.2).
    pub(crate) fn f(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n <= 32);
        let mut value = 0u64;
        for _ in 0..n {
            value = (value << 1) | u64::from(self.bit()?);
        }
        u32::try_from(value).map_err(|_| Error::OutOfRange("f(n)"))
    }

    /// `f(n)`, for `n` up to 8.
    pub(crate) fn u8(&mut self, n: u32) -> Result<u8, Error> {
        debug_assert!(n <= 8);
        u8::try_from(self.f(n)?).map_err(|_| Error::OutOfRange("f(n)"))
    }

    /// `f(n)`, for `n` up to 16.
    pub(crate) fn u16(&mut self, n: u32) -> Result<u16, Error> {
        debug_assert!(n <= 16);
        u16::try_from(self.f(n)?).map_err(|_| Error::OutOfRange("f(n)"))
    }

    /// `su(n)` (4.10.6): `n` bits, the first of them the sign, for `n` from 1 to 16.
    pub(crate) fn su(&mut self, n: u32) -> Result<i16, Error> {
        debug_assert!((1..=16).contains(&n));
        let value = i32::try_from(self.f(n)?).map_err(|_| Error::OutOfRange("su(n)"))?;
        // n is at least 1.
        let sign_mask = 1i32 << n.saturating_sub(1);
        let value = if value & sign_mask == 0 {
            value
        } else {
            // 2 × sign_mask is at most 2^16, value below it.
            value.saturating_sub(sign_mask.saturating_mul(2))
        };
        i16::try_from(value).map_err(|_| Error::OutOfRange("su(n)"))
    }

    /// `ns(n)` (4.10.7): a value below `n` in as few bits as the spec's non-symmetric code
    /// takes. `n` is at least 1.
    pub(crate) fn ns(&mut self, n: u32) -> Result<u32, Error> {
        if n <= 1 {
            return Ok(0);
        }
        // FloorLog2(n) + 1: n's bit count.
        let w = u32::BITS.saturating_sub(n.leading_zeros());
        let m = (1u64 << w).saturating_sub(u64::from(n));
        // n >= 2, so w >= 2.
        let v = u64::from(self.f(w.saturating_sub(1))?);
        if v < m {
            return u32::try_from(v).map_err(|_| Error::OutOfRange("ns(n)"));
        }
        let extra_bit = u64::from(self.bit()?);
        // (v << 1) - m + extra_bit: below n, as v < 2^(w-1) and m <= 2^(w-1).
        u32::try_from((v << 1).saturating_sub(m).saturating_add(extra_bit))
            .map_err(|_| Error::OutOfRange("ns(n)"))
    }

    /// `uvlc()` (4.10.3). A code of 32 leading zeros or more is `2^32 - 1`, as the spec reads
    /// it.
    pub(crate) fn uvlc(&mut self) -> Result<u32, Error> {
        let mut leading_zeros = 0u32;
        while !self.bit()? {
            leading_zeros = leading_zeros.saturating_add(1);
            // A run past the data ends at its end; one past 4096 bits is no conforming code.
            if leading_zeros > 4096 {
                return Err(Error::OutOfRange("uvlc()"));
            }
        }
        if leading_zeros >= 32 {
            return Ok(u32::MAX);
        }
        let value = u64::from(self.f(leading_zeros)?);
        // value + 2^leading_zeros - 1, below 2^32.
        u32::try_from(value.saturating_add((1u64 << leading_zeros).saturating_sub(1)))
            .map_err(|_| Error::OutOfRange("uvlc()"))
    }

    /// `byte_alignment()` (5.3.5): the zero bits to the next byte boundary.
    pub(crate) fn byte_alignment(&mut self) -> Result<(), Error> {
        while !self.position.is_multiple_of(8) {
            if self.bit()? {
                return Err(Error::OutOfRange("zero_bit"));
            }
        }
        Ok(())
    }

    /// The bytes read so far, once aligned.
    pub(crate) fn byte_position(&self) -> usize {
        self.position.div_ceil(8)
    }
}

/// `leb128()` (4.10.5) at the start of `data`: the value and the bytes it took. A value past
/// `2^32 - 1`, which no conforming stream codes, is refused.
pub(crate) fn leb128(data: &[u8]) -> Result<(u32, usize), Error> {
    const SHIFTS: [u32; 8] = [0, 7, 14, 21, 28, 35, 42, 49];
    let mut value = 0u64;
    for (i, shift) in SHIFTS.into_iter().enumerate() {
        let byte = *data.get(i).ok_or(Error::Truncated)?;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            let value = u32::try_from(value).map_err(|_| Error::OutOfRange("leb128()"))?;
            return Ok((value, i.saturating_add(1)));
        }
    }
    Err(Error::OutOfRange("leb128()"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn su_reads_the_sign_from_the_first_bit() -> Result<(), Error> {
        // 7 bits: 0b1111111 is -1, 0b0000011 is 3, 0b1000000 is -64.
        let mut r = BitReader::new(&[0b1111_1110, 0b0000_1110, 0b0000_0000]);
        assert_eq!(r.su(7)?, -1);
        assert_eq!(r.su(7)?, 3);
        assert_eq!(r.su(7)?, -64);
        Ok(())
    }

    #[test]
    fn ns_reads_each_value_below_its_bound() -> Result<(), Error> {
        // ns(5): w = 3, m = 3; values 0..2 in 2 bits, 3 and 4 in 3 bits.
        let mut bits = Vec::new();
        for value in 0u32..5 {
            if value < 3 {
                bits.extend([(value >> 1) & 1, value & 1]);
            } else {
                let coded = value + 3;
                bits.extend([(coded >> 2) & 1, (coded >> 1) & 1, coded & 1]);
            }
        }
        let mut bytes = vec![0u8; bits.len().div_ceil(8)];
        for (i, bit) in bits.iter().enumerate() {
            bytes[i / 8] |= (*bit as u8) << (7 - i % 8);
        }
        let mut r = BitReader::new(&bytes);
        for value in 0..5 {
            assert_eq!(r.ns(5)?, value);
        }
        assert_eq!(r.ns(1)?, 0, "ns(1) reads nothing");
        Ok(())
    }

    #[test]
    fn uvlc_reads_its_codes_and_saturates_past_32_zeros() -> Result<(), Error> {
        // 1 = 0; 010 = 1; 011 = 2; 00100 = 3.
        let mut r = BitReader::new(&[0b1010_0110, 0b0100_0000]);
        assert_eq!(r.uvlc()?, 0);
        assert_eq!(r.uvlc()?, 1);
        assert_eq!(r.uvlc()?, 2);
        assert_eq!(r.uvlc()?, 3);
        let mut long = vec![0u8; 5];
        long.push(0x80);
        assert_eq!(BitReader::new(&long).uvlc()?, u32::MAX);
        assert_eq!(BitReader::new(&[0u8; 10]).uvlc(), Err(Error::Truncated));
        assert_eq!(
            BitReader::new(&[0u8; 600]).uvlc(),
            Err(Error::OutOfRange("uvlc()")),
            "a run of zeros past any code"
        );
        Ok(())
    }

    #[test]
    fn leb128_reads_up_to_eight_bytes_and_refuses_past_32_bits() {
        assert_eq!(leb128(&[0x05]), Ok((5, 1)));
        assert_eq!(leb128(&[0x80 | 0x2c, 0x02]), Ok((300, 2)));
        assert_eq!(
            leb128(&[0x80, 0x80, 0x80, 0x80, 0x0f]),
            Ok((0xf000_0000, 5))
        );
        assert_eq!(
            leb128(&[0x80, 0x80, 0x80, 0x80, 0x10]),
            Err(Error::OutOfRange("leb128()"))
        );
        assert_eq!(leb128(&[0x80; 8]), Err(Error::OutOfRange("leb128()")));
        assert_eq!(leb128(&[0x80, 0x80]), Err(Error::Truncated));
        // Zero padding bytes are allowed: 0x80 0x00 is 0 in two bytes.
        assert_eq!(leb128(&[0x80, 0x00]), Ok((0, 2)));
    }

    #[test]
    fn alignment_refuses_a_set_bit() {
        let mut r = BitReader::new(&[0b1010_0000]);
        assert_eq!(r.bit(), Ok(true));
        assert_eq!(r.byte_alignment(), Err(Error::OutOfRange("zero_bit")));
        let mut r = BitReader::new(&[0b1000_0000, 0xff]);
        assert_eq!(r.bit(), Ok(true));
        assert_eq!(r.byte_alignment(), Ok(()));
        assert_eq!(r.byte_position(), 1);
        assert_eq!(r.f(8), Ok(0xff));
    }
}
