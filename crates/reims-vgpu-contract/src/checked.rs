//! Checked integer helpers used across geometry and planners.

#[inline]
pub fn checked_add_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

#[inline]
pub fn checked_mul_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_mul(b)
}

/// Align `value` up to a power-of-two `align`. Returns None if align is 0/non-pow2 or overflow.
#[inline]
pub fn align_up_u64(value: u64, align: u64) -> Option<u64> {
    if align == 0 || (align & (align - 1)) != 0 {
        return None;
    }
    let add = align - 1;
    let rounded = value.checked_add(add)?;
    Some(rounded & !add)
}

#[inline]
pub fn size_fits_u32(value: usize) -> bool {
    value <= u32::MAX as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_arithmetic_rejects_overflow() {
        assert_eq!(checked_add_u64(20, 22), Some(42));
        assert_eq!(checked_add_u64(u64::MAX, 1), None);
        assert_eq!(checked_mul_u64(6, 7), Some(42));
        assert_eq!(checked_mul_u64(u64::MAX, 2), None);
    }

    #[test]
    fn align_up_requires_a_power_of_two_and_detects_overflow() {
        assert_eq!(align_up_u64(0, 8), Some(0));
        assert_eq!(align_up_u64(9, 8), Some(16));
        assert_eq!(align_up_u64(9, 0), None);
        assert_eq!(align_up_u64(9, 3), None);
        assert_eq!(align_up_u64(u64::MAX, 8), None);
    }

    #[test]
    fn u32_fit_includes_the_boundary() {
        assert!(size_fits_u32(u32::MAX as usize));
        if usize::BITS > u32::BITS {
            assert!(!size_fits_u32(u32::MAX as usize + 1));
        }
    }
}
