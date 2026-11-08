//! A vCPU's 16-byte store and load on arm64: `stp` and `ldp` of two `x` registers, single-copy
//! atomic at a 16-byte-aligned address under FEAT_LSE2.

/// One `stp lo, hi, [a]`.
///
/// # Safety
/// `a..a+16` is writable memory, 16-byte aligned.
pub(super) unsafe fn store_16(a: *mut u8, lo: u64, hi: u64) {
    // SAFETY: the caller's.
    unsafe {
        core::arch::asm!("stp {lo}, {hi}, [{a}]", a = in(reg) a, lo = in(reg) lo,
            hi = in(reg) hi, options(nostack, preserves_flags));
    }
}

/// One `ldp lo, hi, [a]`: `(bytes 0..8, bytes 8..16)`.
///
/// # Safety
/// `a..a+16` is readable memory, 16-byte aligned.
pub(super) unsafe fn load_16(a: *mut u8) -> (u64, u64) {
    let (lo, hi): (u64, u64);
    // SAFETY: the caller's.
    unsafe {
        core::arch::asm!("ldp {lo}, {hi}, [{a}]", a = in(reg) a, lo = out(reg) lo,
            hi = out(reg) hi, options(nostack, preserves_flags, readonly));
    }
    (lo, hi)
}
