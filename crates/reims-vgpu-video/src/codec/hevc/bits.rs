//! Reading HEVC's syntax elements (H.265 7.2 and 9.2): H.264's reader
//! (`crate::codec::h264::bits`), whose descriptors and emulation prevention are
//! the same in both standards, refusing with HEVC's errors.

use super::Error;
use crate::codec::bits::ReadError;

impl ReadError for Error {
    const TRUNCATED: Self = Self::Truncated;

    fn out_of_range(name: &'static str) -> Self {
        Self::OutOfRange(name)
    }
}

/// HEVC's reader, which refuses with HEVC's errors.
pub(crate) type BitReader<'a> = crate::codec::h264::bits::RbspReader<'a, Error>;
