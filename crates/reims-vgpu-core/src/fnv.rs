//! Stable FNV-1a folding for semantic content identities.

pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn fold_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

pub fn fold_u64(hash: u64, value: u64) -> u64 {
    fold_bytes(hash, &value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_single_zero_vector_pins_the_algorithm() {
        assert_eq!(fold_bytes(FNV_OFFSET_BASIS, &[0]), 0xaf63_bd4c_8601_b7df);
    }

    #[test]
    fn integer_folding_is_little_endian() {
        assert_eq!(
            fold_u64(FNV_OFFSET_BASIS, 0x0123_4567_89ab_cdef),
            fold_bytes(
                FNV_OFFSET_BASIS,
                &[0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]
            )
        );
    }
}
