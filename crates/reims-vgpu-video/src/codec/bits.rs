//! What a codec's bit readers refuse with. H.264's and HEVC's reader is
//! `h264::bits`, which removes the emulation prevention bytes as it goes,
//! and refuses through [`ReadError`].

/// A codec's error, as the readers refuse with it (`h264::bits`).
pub(crate) trait ReadError {
    /// The data ended inside a syntax element.
    const TRUNCATED: Self;

    /// A read the descriptor does not allow, named.
    fn out_of_range(name: &'static str) -> Self;
}
