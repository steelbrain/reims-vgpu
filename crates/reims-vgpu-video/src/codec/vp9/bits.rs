//! Reading the uncompressed header's syntax elements (4.9, 9.1): `f(n)`,
//! most significant bit first, and `su(n)`. VP9 has no emulation
//! prevention, so positions are the frame's own.

use super::Error;

/// A reader over a frame's bytes that refuses with [`Error::Truncated`]
/// past its end.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// The next bit's position, from the frame's first bit.
    position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// `f(1)`.
    pub(crate) fn bit(&mut self) -> Result<bool, Error> {
        let byte = self
            .data
            .get(self.position / 8)
            .copied()
            .ok_or(Error::Truncated)?;
        // `position % 8` is below 8, so the shift does not overflow.
        let shift = 7u32.saturating_sub(u32::try_from(self.position % 8).unwrap_or(7));
        self.position = self.position.checked_add(1).ok_or(Error::Truncated)?;
        Ok((byte >> shift) & 1 == 1)
    }

    /// `f(n)`, for `n` up to 16.
    pub(crate) fn bits(&mut self, n: u32) -> Result<u16, Error> {
        debug_assert!(n <= 16);
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | u32::from(self.bit()?);
        }
        u16::try_from(value).map_err(|_| Error::OutOfRange("f(n)"))
    }

    /// `f(n)`, for `n` up to 8.
    pub(crate) fn u8(&mut self, n: u32) -> Result<u8, Error> {
        debug_assert!(n <= 8);
        u8::try_from(self.bits(n)?).map_err(|_| Error::OutOfRange("f(n)"))
    }

    /// `su(n)` (4.9): `n` bits of magnitude, then a sign bit, for `n` up to 8.
    pub(crate) fn su(&mut self, n: u32) -> Result<i16, Error> {
        let value = i16::from(self.u8(n)?);
        Ok(if self.bit()? {
            value.wrapping_neg()
        } else {
            value
        })
    }

    /// The bytes read so far, a partial byte counting whole: where
    /// `trailing_bits()` ends the uncompressed header (6.2).
    pub(crate) fn bytes_read(&self) -> usize {
        self.position.div_ceil(8)
    }
}
