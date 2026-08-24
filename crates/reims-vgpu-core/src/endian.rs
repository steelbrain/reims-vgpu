//! Little-endian byte readers matching the C `ld16`/`ld32`/`ld64` helpers.

#[inline]
pub fn ld16(p: &[u8]) -> u16 {
    debug_assert!(p.len() >= 2);
    u16::from_le_bytes([p[0], p[1]])
}

#[inline]
pub fn ld32(p: &[u8]) -> u32 {
    debug_assert!(p.len() >= 4);
    u32::from_le_bytes([p[0], p[1], p[2], p[3]])
}

#[inline]
pub fn ld64(p: &[u8]) -> u64 {
    debug_assert!(p.len() >= 8);
    u64::from_le_bytes([p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]])
}

#[inline]
pub fn st16(p: &mut [u8], v: u16) {
    let b = v.to_le_bytes();
    p[0] = b[0];
    p[1] = b[1];
}

#[inline]
pub fn st32(p: &mut [u8], v: u32) {
    let b = v.to_le_bytes();
    p[..4].copy_from_slice(&b);
}

#[inline]
pub fn st64(p: &mut [u8], v: u64) {
    let b = v.to_le_bytes();
    p[..8].copy_from_slice(&b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_loads_are_little_endian() {
        let mut bytes = [0u8; 14];
        st16(&mut bytes[0..2], 0x1234);
        st32(&mut bytes[2..6], 0x89ab_cdef);
        st64(&mut bytes[6..14], 0x0123_4567_89ab_cdef);

        assert_eq!(&bytes[0..6], &[0x34, 0x12, 0xef, 0xcd, 0xab, 0x89]);
        assert_eq!(ld16(&bytes[0..2]), 0x1234);
        assert_eq!(ld32(&bytes[2..6]), 0x89ab_cdef);
        assert_eq!(ld64(&bytes[6..14]), 0x0123_4567_89ab_cdef);
    }
}
