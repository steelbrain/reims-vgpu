//! Integer operations std has no single spelling for.
//!
//! Wrapping and overflowing arithmetic, widening, sign reinterpretation and `bool` conversions
//! use std (`wrapping_add`, `overflowing_mul`, `u64::from`, `cast_signed`). This module holds the
//! rest: the `usize` conversions that are lossless on the 64-bit target, truncation to the low
//! bits, the one rounding rule the display fixes, and the integer rules the guest sees.
//! It is the only module of core's production code that may use `as`.
//!
//! **The guest's integer rules.** Guests were written against a device whose arithmetic is
//! arm64's, and what that arithmetic gives for an odd operand reaches the guest's memory and
//! registers. These results are guest ABI, kept as they are, each under its own name:
//! - division by a divisor the guest can make zero ([`div0`], [`rem0`]);
//! - shift widths: a mask bit is a shift taken mod its register's width ([`bit32`], [`bit64`],
//!   [`bit64_set`]);
//! - sign versus zero extension of a narrow field ([`zero_extend_16`], [`sign_extend_16`],
//!   [`sign_extend_32`]);
//! - the forward copy through aliased mappings ([`forward_copy`]).

#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the conversions every other module reaches by name"
)]

// The device model is for arm64: `usize` and `u64` are one width, so the conversions between
// them below are lossless.
const _: () = assert!(usize::BITS == u64::BITS);

/// Division whose divisor may be zero, with arm64's results: `udiv` by zero gives 0, and the
/// remainder (`udiv` then `msub`) is then the dividend.
///
/// Kept as guest ABI: a guest-supplied divisor can be zero (a ring or table of size 0, a stride
/// alignment of 0), and the device keeps going with these values rather than faulting; the
/// offsets and indices they give reach the guest's memory.
pub trait DivZero: Copy {
    /// `self / d`, or 0 when `d` is 0.
    fn div0(self, d: Self) -> Self;
    /// `self % d`, or `self` when `d` is 0.
    fn rem0(self, d: Self) -> Self;
}

macro_rules! div_zero {
    ($($t:ty),*) => {$(
        impl DivZero for $t {
            #[inline]
            fn div0(self, d: Self) -> Self {
                match self.checked_div(d) {
                    Some(q) => q,
                    None => 0,
                }
            }

            #[inline]
            fn rem0(self, d: Self) -> Self {
                match self.checked_rem(d) {
                    Some(r) => r,
                    None => self,
                }
            }
        }
    )*};
}

div_zero!(u32, u64);

/// `a / b`, or 0 when `b` is 0 (arm64's `udiv`).
#[inline]
pub fn div0<T: DivZero>(a: T, b: T) -> T {
    a.div0(b)
}

/// `a % b`, or `a` when `b` is 0 (arm64's `udiv` then `msub`).
#[inline]
pub fn rem0<T: DivZero>(a: T, b: T) -> T {
    a.rem0(b)
}

/// Bit `n` of a 32-bit mask, as arm64's 32-bit `lsl` makes it: the count is taken mod 32, so an
/// `n` of 32 or more names bit `n mod 32`.
///
/// Kept as guest ABI: the guest reads the masks built this way (the stamp interrupt bits, the
/// waiter mask, a display port's bits, an IOSurface element dimension), and a guest index of 32
/// or more lands on the aliased bit.
#[inline]
pub const fn bit32(n: u32) -> u32 {
    1u32.wrapping_shl(n)
}

/// Bit `n` of a 64-bit mask, as arm64's 64-bit `lsl` makes it: the count is taken mod 64.
///
/// Kept as guest ABI: the fault mask and the stamp-index masks are built this way, so an id of
/// 64 or more sets or clears the bit of the id mod 64 (0x1028 = 64 clears the root's fault
/// bit).
#[inline]
pub const fn bit64(n: u32) -> u64 {
    1u64.wrapping_shl(n)
}

/// Whether bit `n` of `word` is set, read as a 64-bit `lsr` by `n`: the count is taken mod 64.
///
/// Kept as guest ABI: the stamp-wait walk tests the waiting word this way with a guest-chosen
/// stamp index, so an index of 64 or more tests the bit of the index mod 64.
#[inline]
pub const fn bit64_set(word: u64, n: u32) -> bool {
    word.wrapping_shr(n) & 1 != 0
}

/// A 16-bit field widened by zero extension (arm64's `ldrh`): 0xffff is 65535.
///
/// Kept as guest ABI: render command 0x0b's base vertex is unsigned, so its 0xffff draws from
/// vertex 65535, where 0x1d's reads −1 ([`sign_extend_16`]).
#[inline]
pub const fn zero_extend_16(x: u16) -> i64 {
    x as i64
}

/// A 16-bit field widened by sign extension (arm64's `ldrsh`): 0xffff is −1.
///
/// Kept as guest ABI: render command 0x1d's base vertex is signed, where 0x0b's is not
/// ([`zero_extend_16`]).
#[inline]
pub const fn sign_extend_16(x: i16) -> i64 {
    x as i64
}

/// The 32 bits of an `int` widened to 64 by sign extension (arm64's `sxtw`): bit 31 fills bits
/// 32-63.
///
/// Kept as guest ABI: the masks built as 32-bit `int`s and stored into 64-bit words (a display
/// port's busy bit, a finished wait's `!(1 << waiter)`) carry bit 31 into the high half, and the
/// guest sees the bits that sets or clears.
#[inline]
pub const fn sign_extend_32(x: u32) -> u64 {
    (x.cast_signed() as i64).cast_unsigned()
}

/// Copies `n` bytes as libc's forward `memcpy` does when the source and the destination are two
/// mappings of the same pages: `chunk.len()` bytes at a time, each chunk read in full through
/// `read` before it is written through `write`, in ascending order.
///
/// With both mappings starting on an alias granule and `chunk` that granule's size, this gives
/// exactly the forward copy's bytes: an alias maps byte `i` of one side onto byte `j` of the other
/// only with `i ≡ j` modulo the granule, so inside one chunk the only alias of a written byte is
/// the byte itself, which the forward copy also reads before it writes, and every byte of an
/// earlier chunk is written before any later chunk is read. Where the two overlap, the data
/// repeats with their distance as its period: it is not a snapshot copy.
///
/// Kept as guest ABI: a guest that moves a list onto memory overlapping the old one sees these
/// bytes. An empty `chunk` copies nothing.
pub fn forward_copy(
    n: u64,
    chunk: &mut [u8],
    mut read: impl FnMut(u64, &mut [u8]),
    mut write: impl FnMut(u64, &[u8]),
) {
    let size = chunk.len() as u64;
    if size == 0 {
        return;
    }
    let mut done: u64 = 0;
    while done < n {
        let len = n.wrapping_sub(done).min(size) as usize;
        let Some(bytes) = chunk.get_mut(..len) else {
            return;
        };
        read(done, bytes);
        write(done, bytes);
        done = done.wrapping_add(len as u64);
    }
}

/// `x` rounded up to a multiple of the power of two `align`, as `(x + align - 1) & !(align - 1)`
/// with every step wrapping: a sum past `u64::MAX` wraps to a small result, and an `align` of 0
/// gives 0.
#[inline]
pub const fn align_up_wrapping(x: u64, align: u64) -> u64 {
    let mask = align.wrapping_sub(1);
    x.wrapping_add(mask) & !mask
}

/// `a + b` stored into 32 bits: the low 32 bits of the sum, and whether the exact sum does not
/// fit in 32 bits (a carry out of 64 bits counts too).
#[inline]
pub const fn add_u64_into_u32(a: u64, b: u64) -> (u32, bool) {
    let (sum, carry) = a.overflowing_add(b);
    (sum as u32, carry || sum > 0xffff_ffff)
}

/// A refresh rate in hertz as the 16.16 fixed-point value the display reports:
/// `floor(hz * 65536 + 0.5)`, saturating to `0..=u32::MAX`, with NaN giving 0 (arm64's
/// `fcvtmu`).
#[inline]
pub fn refresh_16_16(hz: f64) -> u32 {
    (hz * 65536.0 + 0.5).floor() as u32
}

/// Truncation to the low bits, as a store from a wider register does.
pub trait Truncate: Copy {
    /// The low 32 bits.
    fn low_u32(self) -> u32;
    /// The low 16 bits.
    fn low_u16(self) -> u16;
    /// The low 8 bits.
    fn low_u8(self) -> u8;
}

macro_rules! truncate {
    ($($t:ty),*) => {$(
        impl Truncate for $t {
            #[inline]
            fn low_u32(self) -> u32 {
                self as u32
            }

            #[inline]
            fn low_u16(self) -> u16 {
                self as u16
            }

            #[inline]
            fn low_u8(self) -> u8 {
                self as u8
            }
        }
    )*};
}

truncate!(u64, usize);

impl Truncate for u32 {
    #[inline]
    fn low_u32(self) -> u32 {
        self
    }

    #[inline]
    fn low_u16(self) -> u16 {
        self as u16
    }

    #[inline]
    fn low_u8(self) -> u8 {
        self as u8
    }
}

/// Lossless conversion to `usize` on the 64-bit target.
pub trait ToUsize: Copy {
    /// `self` as `usize`.
    fn to_usize(self) -> usize;
}

impl ToUsize for u64 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl ToUsize for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

/// Lossless conversion of a `usize` to `u64` on the 64-bit target.
pub trait ToU64: Copy {
    /// `self` as `u64`.
    fn to_u64(self) -> u64;
}

impl ToU64 for usize {
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
}

/// A `usize` known at compile time (a `size_of` or an `offset_of!`) as `u64`, for the const
/// items where [`ToU64::to_u64`] cannot be called.
#[inline]
pub const fn const_u64(x: usize) -> u64 {
    x as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn division_by_zero_gives_zero_and_the_remainder_the_dividend() {
        assert_eq!(div0(7u64, 0), 0);
        assert_eq!(rem0(7u64, 0), 7);
        assert_eq!(div0(7u64, 2), 3);
        assert_eq!(rem0(7u64, 2), 1);
        assert_eq!(div0(9u32, 0), 0);
        assert_eq!(rem0(9u32, 0), 9);
        assert_eq!(div0(9u32, 4), 2);
        assert_eq!(rem0(9u32, 4), 1);
    }

    #[test]
    fn mask_bits_take_the_shift_mod_the_width() {
        assert_eq!(bit32(0), 1);
        assert_eq!(bit32(31), 0x8000_0000);
        assert_eq!(bit32(32), 1);
        assert_eq!(bit32(33), 2);
        assert_eq!(bit32(0xa0), 1);
        assert_eq!(bit64(63), 1 << 63);
        assert_eq!(bit64(64), 1);
        assert_eq!(bit64(65), 2);
        assert_eq!(bit64(u32::MAX), 1 << 63);
        assert!(bit64_set(0b10, 1));
        assert!(bit64_set(0b10, 65));
        assert!(!bit64_set(0b10, 64));
    }

    #[test]
    fn narrow_fields_widen_by_their_sign_rule() {
        assert_eq!(zero_extend_16(0xffff), 65535);
        assert_eq!(sign_extend_16(-1), -1);
        assert_eq!(sign_extend_16(i16::MIN), -32768);
        assert_eq!(sign_extend_32(0x8000_0000), 0xffff_ffff_8000_0000);
        assert_eq!(sign_extend_32(!0x8000_0000), 0x7fff_ffff);
        assert_eq!(sign_extend_32(!1), 0xffff_ffff_ffff_fffe);
    }

    /// Through two views of one buffer a granule apart, a forward copy in granule-sized chunks
    /// gives a byte-by-byte forward copy's bytes: the source repeats with the distance as its
    /// period
    #[test]
    fn forward_copy_reads_back_what_it_wrote() {
        let (src, dst, n) = (0usize, 4usize, 10u64);
        let mut want: Vec<u8> = (0..16).collect();
        for i in 0..n as usize {
            want[dst + i] = want[src + i];
        }
        let mem = std::cell::RefCell::new((0..16).collect::<Vec<u8>>());
        let mut chunk = [0u8; 4];
        forward_copy(
            n,
            &mut chunk,
            |at, b| b.copy_from_slice(&mem.borrow()[src + at as usize..][..b.len()]),
            |at, b| mem.borrow_mut()[dst + at as usize..][..b.len()].copy_from_slice(b),
        );
        assert_eq!(*mem.borrow(), want);
        assert_eq!(&want[4..14], &[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]);
        let mut none: [u8; 0] = [];
        forward_copy(8, &mut none, |_, _| panic!("read"), |_, _| panic!("write"));
    }

    #[test]
    fn align_up_wraps_at_the_top_and_align_zero_gives_zero() {
        assert_eq!(align_up_wrapping(0x4001, 0x4000), 0x8000);
        assert_eq!(align_up_wrapping(0x4000, 0x4000), 0x4000);
        assert_eq!(align_up_wrapping(0, 0x4000), 0);
        assert_eq!(align_up_wrapping(u64::MAX, 0x4000), 0);
        assert_eq!(align_up_wrapping(5, 0), 0);
        assert_eq!(align_up_wrapping(0, 0), 0);
    }

    #[test]
    fn sum_into_u32_reports_a_sum_past_32_bits() {
        assert_eq!(add_u64_into_u32(0xffff_fffe, 1), (0xffff_ffff, false));
        assert_eq!(add_u64_into_u32(0xffff_ffff, 1), (0, true));
        assert_eq!(add_u64_into_u32(u64::MAX, 3), (2, true));
        assert_eq!(add_u64_into_u32(1 << 40, 0), (0, true));
    }

    #[test]
    fn refresh_rounds_half_up_and_saturates() {
        assert_eq!(refresh_16_16(60.0), 3_932_160);
        assert_eq!(refresh_16_16(59.94), 3_928_228);
        assert_eq!(refresh_16_16(f64::NAN), 0);
        assert_eq!(refresh_16_16(-1.0), 0);
        assert_eq!(refresh_16_16(1e12), u32::MAX);
    }

    #[test]
    fn truncation_keeps_the_low_bits() {
        assert_eq!(0x1_2345_6789u64.low_u32(), 0x2345_6789);
        assert_eq!(0x1_0002_ffffu64.low_u16(), 0xffff);
        assert_eq!(0x1ffu64.low_u8(), 0xff);
        assert_eq!(0x0001_0005u32.low_u16(), 5);
        assert_eq!(0x100u32.low_u8(), 0);
        assert_eq!(0x1_0000_0003usize.low_u32(), 3);
    }

    #[test]
    fn usize_conversions_are_lossless() {
        assert_eq!(u64::MAX.to_usize(), usize::MAX);
        assert_eq!(u32::MAX.to_usize(), 0xffff_ffff);
        assert_eq!(usize::MAX.to_u64(), u64::MAX);
    }
}
