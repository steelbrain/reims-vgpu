//! The integer conversion std has no lossless spelling for on the targets
//! the library builds for, whose `usize` is 64 bits wide: `usize` to `u64`,
//! reached by name rather than as an `as` at each site.

#![expect(
    clippy::as_conversions,
    reason = "the conversion every other module reaches by name"
)]

const _: () = assert!(usize::BITS == u64::BITS);

/// Lossless conversion of a `usize` to `u64`.
pub(crate) trait ToU64: Copy {
    /// `self` as `u64`.
    fn to_u64(self) -> u64;
}

impl ToU64 for usize {
    #[inline]
    fn to_u64(self) -> u64 {
        self as u64
    }
}
