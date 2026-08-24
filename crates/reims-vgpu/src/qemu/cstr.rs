//! The one rule for writing a string into a caller-supplied C buffer.
//!
//! Not a shim symbol, unlike everything else under [`super`]: it is the rule two
//! shim-facing writers share. `reims_vgpu_qemu_abi.h` hands both of them a
//! `(char *, size_t)` pair, and both used to spell the truncation out.
//!
//! They agreed, which is the hazard rather than the reassurance. The rule is
//! four lines and one of them is a raw `copy_nonoverlapping`, so a third writer
//! that reasons "capacity is `cap`, so copy `min(len, cap)`" writes the
//! terminator one byte past the end of the caller's buffer. Nothing in the
//! toolchain compares one copy of that arithmetic to another, and the header
//! says only that the buffer is valid for `cap` bytes.
//!
//! It lives under `qemu` because the contract is the header's.

use std::os::raw::c_char;

/// Write `s` into `buf`, truncated to fit, always NUL-terminated.
///
/// `cap` is the buffer's whole size **including** the terminator, which is the
/// header's meaning of the second parameter. So at most `cap - 1` bytes of `s`
/// are copied and the terminator lands at or before `buf[cap - 1]`. A null
/// pointer or a zero capacity writes nothing — there is no room for even a
/// terminator, and the callers treat "no buffer" as a legitimate ask.
///
/// Truncation is on bytes, not on `char` boundaries, so a multi-byte character
/// straddling the limit is cut. Every caller here writes an ASCII refusal slug
/// or backend name; the note is here so a caller with UTF-8 to write knows it
/// must not use this.
///
/// # Safety
///
/// `buf` must be null or valid for writes of `cap` bytes.
pub(crate) unsafe fn write_c_str(buf: *mut c_char, cap: usize, s: &str) {
    if buf.is_null() || cap == 0 {
        return;
    }
    // SAFETY: the caller promises `cap` writable bytes at `buf`. `n` is capped
    // at `cap - 1`, so the copy and the terminator at `n` both land inside.
    unsafe {
        let bytes = s.as_bytes();
        let n = bytes.len().min(cap - 1);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), n);
        *buf.add(n) = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes into a buffer one byte larger than `cap`, and asserts the guard
    /// byte is untouched. Without the extra byte the test could not tell a
    /// correct write from a one-past-the-end one, which is the failure this
    /// module exists to prevent.
    fn write_into(cap: usize, s: &str) -> (Vec<u8>, u8) {
        let mut buf = vec![0xAAu8; cap + 1];
        unsafe { write_c_str(buf.as_mut_ptr().cast::<c_char>(), cap, s) };
        let guard = buf[cap];
        buf.truncate(cap);
        (buf, guard)
    }

    #[test]
    fn a_string_that_fits_is_written_whole_and_terminated() {
        let (buf, guard) = write_into(8, "vulkan");
        assert_eq!(&buf[..6], b"vulkan");
        assert_eq!(buf[6], 0);
        assert_eq!(guard, 0xAA, "wrote past the caller's capacity");
    }

    #[test]
    fn an_exact_fit_still_leaves_room_for_the_terminator() {
        // "abcd" is four bytes and the capacity is five: the last byte is the
        // terminator, not a fifth character.
        let (buf, guard) = write_into(5, "abcd");
        assert_eq!(buf, b"abcd\0");
        assert_eq!(guard, 0xAA);
    }

    #[test]
    fn a_long_string_is_truncated_and_the_terminator_stays_inside() {
        let (buf, guard) = write_into(4, "abcdefgh");
        assert_eq!(buf, b"abc\0", "capacity includes the terminator");
        assert_eq!(guard, 0xAA, "wrote past the caller's capacity");
    }

    #[test]
    fn a_one_byte_buffer_holds_only_the_terminator() {
        let (buf, guard) = write_into(1, "abc");
        assert_eq!(buf, b"\0");
        assert_eq!(guard, 0xAA);
    }

    #[test]
    fn no_buffer_and_no_capacity_write_nothing() {
        // Zero capacity: not even a terminator fits, so the guard byte that
        // would be `buf[0]` must survive.
        let mut buf = [0xAAu8; 1];
        unsafe { write_c_str(buf.as_mut_ptr().cast::<c_char>(), 0, "abc") };
        assert_eq!(buf[0], 0xAA);

        // Null pointer: no write, no panic.
        unsafe { write_c_str(std::ptr::null_mut(), 16, "abc") };
    }
}
