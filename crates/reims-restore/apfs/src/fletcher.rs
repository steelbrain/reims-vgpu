//! The Fletcher-64 checksum every APFS object header carries in its first eight bytes.
//!
//! The sum runs over the object's 32-bit little-endian words after the checksum field, modulo
//! 2^32 - 1, and the stored value is the pair of check words that make the whole object's sum
//! zero (Apple File System Reference, "Objects").

const MOD: u64 = 0xffff_ffff;

/// The checksum of `block`, whose first eight bytes are the checksum field and are skipped.
#[must_use]
pub fn checksum(block: &[u8]) -> u64 {
    debug_assert!(block.len() % 4 == 0 && block.len() >= 8);
    let (mut a, mut b) = (0u64, 0u64);
    // 2^14 words (a 64 KiB block) keep both sums under 2^60 before the reduction.
    for chunk in block[8..].chunks(1 << 16) {
        for w in chunk.as_chunks::<4>().0 {
            a += u64::from(u32::from_le_bytes(*w));
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    let c1 = MOD - ((a + b) % MOD);
    let c2 = MOD - ((a + c1) % MOD);
    (c2 << 32) | c1
}

/// The checksum stored in `block`'s header.
#[must_use]
pub fn stored(block: &[u8]) -> u64 {
    u64::from_le_bytes(block[..8].try_into().unwrap())
}

/// Whether `block`'s stored checksum is right.
#[must_use]
pub fn verify(block: &[u8]) -> bool {
    stored(block) == checksum(block)
}

/// Write `block`'s checksum into its header.
pub fn seal(block: &mut [u8]) {
    let c = checksum(block);
    block[..8].copy_from_slice(&c.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sealed_block_verifies_and_a_flipped_bit_does_not() {
        let mut block = vec![0u8; 4096];
        for (i, b) in block.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        seal(&mut block);
        assert!(verify(&block));
        block[100] ^= 1;
        assert!(!verify(&block));
    }

    #[test]
    fn all_ones_words_reduce_to_zero_sums() {
        // 0xffffffff is 0 modulo 2^32 - 1, so the sums are zero and both check words are the
        // modulus itself.
        let mut block = vec![0xffu8; 64];
        seal(&mut block);
        assert_eq!(stored(&block), 0xffff_ffff_ffff_ffff);
        assert!(verify(&block));
    }
}
