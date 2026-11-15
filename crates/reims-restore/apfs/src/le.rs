//! Little-endian field access over byte slices.

#[inline]
pub(crate) fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

#[inline]
pub(crate) fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

#[inline]
pub(crate) fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

#[inline]
pub(crate) fn uuid_at(b: &[u8], off: usize) -> [u8; 16] {
    b[off..off + 16].try_into().unwrap()
}
