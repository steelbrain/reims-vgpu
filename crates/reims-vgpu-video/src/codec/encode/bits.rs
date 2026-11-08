//! Writing syntax elements: fixed-length fields and Exp-Golomb codes
//! (H.264 9.1, H.265 9.2), most significant bit first, into an RBSP that
//! [`super::nal`] escapes into a NAL unit.

use crate::codec::num::ToUsize;

/// The bit of a byte written first.
const FIRST_BIT: u8 = 0x80;

/// An RBSP being written, bit by bit.
#[derive(Clone, Debug)]
pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    /// The byte being filled, from its high bit.
    current: u8,
    /// The bit of `current` the next bit goes to.
    next: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            next: FIRST_BIT,
        }
    }

    /// The number of bits written; none past a `usize`.
    pub(crate) fn bit_len(&self) -> Option<usize> {
        // The bits of `current` written are those above `next`.
        let used = self.next.leading_zeros().to_usize();
        self.bytes.len().checked_mul(8)?.checked_add(used)
    }

    pub(crate) fn bit(&mut self, value: bool) {
        if value {
            self.current |= self.next;
        }
        self.next >>= 1;
        if self.next == 0 {
            self.bytes.push(self.current);
            self.current = 0;
            self.next = FIRST_BIT;
        }
    }

    /// `u(n)`: the low `n` bits of `value`, `n` up to 64.
    pub(crate) fn bits(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64);
        for i in (0..n).rev() {
            self.bit((value >> i) & 1 == 1);
        }
    }

    /// `u(n)` of a 32-bit value.
    pub(crate) fn u(&mut self, n: u32, value: u32) {
        self.bits(n, u64::from(value));
    }

    /// `ue(v)` (9.1): `value` as an unsigned Exp-Golomb code.
    pub(crate) fn ue(&mut self, value: u32) {
        // codeNum + 1; u32::MAX's is 2^32.
        self.exp_golomb(value.checked_add(1).map_or(1 << 32, u64::from));
    }

    /// `se(v)` (9.1.1): positive values map to odd codes.
    pub(crate) fn se(&mut self, value: i32) {
        // codeNum is 2|v| - 1 for a positive v, else 2|v| (Table 9-3), so
        // codeNum + 1 is 2|v|, and one more for v of 0 or less. The
        // largest, for i32::MIN, is 2^32 + 1: one bit more than ue's u32
        // takes, so write it here.
        let code = (u64::from(value.unsigned_abs()) << 1) | u64::from(value <= 0);
        self.exp_golomb(code);
    }

    /// The Exp-Golomb code of `code`, codeNum + 1, at least 1 (9-2): as
    /// many zeros as it has bits after its leading one, then its bits.
    fn exp_golomb(&mut self, code: u64) {
        let zeros = code.ilog2();
        self.bits(zeros, 0);
        self.bit(true);
        self.bits(zeros, code);
    }

    /// `flag`, for readability where the syntax names a flag.
    pub(crate) fn flag(&mut self, value: bool) {
        self.bit(value);
    }

    /// Whether the next bit starts a byte.
    pub(crate) fn byte_aligned(&self) -> bool {
        self.next == FIRST_BIT
    }

    /// `rbsp_trailing_bits()` (7.3.2.11): the stop bit, then zeros to the
    /// byte. H.265's `byte_alignment()` writes the same bits.
    pub(crate) fn trailing_bits(&mut self) {
        self.bit(true);
        while !self.byte_aligned() {
            self.bit(false);
        }
    }

    /// The bytes written, a partial last byte padded with zeros.
    pub(crate) fn into_bytes(mut self) -> Vec<u8> {
        if !self.byte_aligned() {
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_golomb_codes_are_written_as_9_1_spells_them() {
        let mut w = BitWriter::new();
        for v in [0, 1, 2, 3, 4, 7] {
            w.ue(v);
        }
        // 1 | 010 | 011 | 00100 | 00101 | 0001000
        assert_eq!(w.bit_len(), Some(24));
        assert_eq!(w.into_bytes(), [0b1010_0110, 0b0100_0010, 0b1000_1000]);
        let mut w = BitWriter::new();
        for v in [0, 1, -1, 2, -2, 4] {
            w.se(v);
        }
        assert_eq!(w.into_bytes(), [0b1010_0110, 0b0100_0010, 0b1000_1000]);
    }

    #[test]
    fn extreme_codes_fit() {
        let mut w = BitWriter::new();
        w.ue(u32::MAX);
        assert_eq!(w.bit_len(), Some(65));
        let mut w = BitWriter::new();
        w.se(i32::MIN);
        assert_eq!(w.bit_len(), Some(65));
    }

    #[test]
    fn what_is_written_reads_back() -> Result<(), crate::codec::h264::Error> {
        let mut w = BitWriter::new();
        w.u(5, 0b10110);
        w.ue(1234);
        w.se(-77);
        w.bits(40, 0x12_3456_789a);
        let bytes = w.into_bytes();
        let mut r = crate::codec::h264::bits::BitReader::new(&bytes);
        assert_eq!(r.bits(5)?, 0b10110);
        assert_eq!(r.ue()?, 1234);
        assert_eq!(r.se()?, -77);
        assert_eq!((r.bits(8)?, r.bits(32)?), (0x12, 0x3456_789a));
        assert_eq!(r.position().div_ceil(8), bytes.len());
        Ok(())
    }

    #[test]
    fn trailing_bits_stop_and_align() {
        let mut w = BitWriter::new();
        w.u(3, 0b101);
        w.trailing_bits();
        assert!(w.byte_aligned());
        assert_eq!(w.into_bytes(), [0b1011_0000]);
        let mut w = BitWriter::new();
        w.u(8, 0xab);
        w.trailing_bits();
        assert_eq!(w.into_bytes(), [0xab, 0x80]);
    }
}
