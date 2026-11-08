//! A vCPU's 16-byte store and load on x86-64: one `movdqa` each, which Intel performs atomically
//! at a 16-byte-aligned address on a part with AVX.

/// One `movdqa [a], xmm` of `lo` then `hi`.
///
/// # Safety
/// `a..a+16` is writable memory, 16-byte aligned.
pub(super) unsafe fn store_16(a: *mut u8, lo: u64, hi: u64) {
    // SAFETY: the caller's.
    unsafe {
        core::arch::asm!("movq {x}, {lo}", "movq {t}, {hi}", "punpcklqdq {x}, {t}",
            "movdqa xmmword ptr [{a}], {x}", a = in(reg) a, lo = in(reg) lo, hi = in(reg) hi,
            x = out(xmm_reg) _, t = out(xmm_reg) _, options(nostack, preserves_flags));
    }
}

/// One `movdqa xmm, [a]`: `(bytes 0..8, bytes 8..16)`.
///
/// # Safety
/// `a..a+16` is readable memory, 16-byte aligned.
pub(super) unsafe fn load_16(a: *mut u8) -> (u64, u64) {
    let (lo, hi): (u64, u64);
    // SAFETY: the caller's.
    unsafe {
        core::arch::asm!("movdqa {t}, xmmword ptr [{a}]", "movq {lo}, {t}", "psrldq {t}, 8",
            "movq {hi}, {t}", a = in(reg) a, lo = out(reg) lo, hi = out(reg) hi,
            t = out(xmm_reg) _, options(nostack, preserves_flags, readonly));
    }
    (lo, hi)
}
