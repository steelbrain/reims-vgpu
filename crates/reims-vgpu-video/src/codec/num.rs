//! The integer conversions std has no lossless spelling for on the
//! targets the library builds for, whose `usize` is 64 bits wide: `u32`
//! and `u64` to `usize`. A stream's sizes and counts are `u32` or `u64`
//! where the standards code them, and index the parsers' slices as
//! `usize`; the conversion is reached by name rather than as an `as` at
//! each site.

#![expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "the conversions every other module reaches by name"
)]

const _: () = assert!(usize::BITS == u64::BITS);

/// Lossless conversion to `usize`.
pub(crate) trait ToUsize: Copy {
    /// `self` as `usize`.
    fn to_usize(self) -> usize;
}

impl ToUsize for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl ToUsize for u64 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
}
