//! Superframes (Annex B): one sample holding several frames, each but the
//! last usually not shown, with an index of their sizes at its end. A
//! container's sample, and VideoToolbox's sample buffer, hold the
//! superframe whole; the host splits it here, from bytes the guest wrote,
//! so every size is checked against the sample before a frame is cut.

use super::Error;

/// The most frames a superframe holds: `frames_in_superframe_minus_1` is
/// 3 bits.
pub const MAX_FRAMES: usize = 8;

/// The frames of `sample`: those its superframe index names, in order,
/// or the sample whole where it ends in no index.
///
/// An index is its marker byte (`0b110` in its top bits, then
/// `bytes_per_framesize_minus_1` and `frames_in_superframe_minus_1`), each
/// frame's size in that many bytes, little-endian, then the marker byte
/// again (B.2). A last byte that looks like a marker without its twin where
/// the index would begin is no index, as libvpx reads it, and the sample is
/// one frame. An index whose sizes run past the bytes before it, or that
/// names a frame of no bytes, is [`Error::BadSuperframe`]; bytes after the
/// last frame and before the index are left out. An empty sample is
/// [`Error::EmptySample`].
pub fn frames(sample: &[u8]) -> Result<Vec<&[u8]>, Error> {
    let Some(&marker) = sample.last() else {
        return Err(Error::EmptySample);
    };
    if marker & 0xe0 != 0xc0 {
        return Ok(vec![sample]);
    }
    let count = usize::from(marker & 7).saturating_add(1);
    let size_bytes = usize::from((marker >> 3) & 3).saturating_add(1);
    // At most 2 + 4 * 8 bytes: no overflow.
    let index_bytes = size_bytes.saturating_mul(count).saturating_add(2);
    let Some(frames_end) = sample.len().checked_sub(index_bytes) else {
        return Ok(vec![sample]);
    };
    if sample.get(frames_end) != Some(&marker) {
        return Ok(vec![sample]);
    }
    let sizes = sample
        .get(frames_end.saturating_add(1)..sample.len().saturating_sub(1))
        .ok_or(Error::BadSuperframe)?;
    let mut frames = Vec::with_capacity(count);
    let mut start = 0usize;
    for size in sizes.chunks_exact(size_bytes) {
        let size = size
            .iter()
            .rev()
            .fold(0usize, |value, &byte| (value << 8) | usize::from(byte));
        if size == 0 {
            return Err(Error::BadSuperframe);
        }
        let end = start.checked_add(size).ok_or(Error::BadSuperframe)?;
        if end > frames_end {
            return Err(Error::BadSuperframe);
        }
        frames.push(sample.get(start..end).ok_or(Error::BadSuperframe)?);
        start = end;
    }
    if frames.len() != count || count > MAX_FRAMES {
        return Err(Error::BadSuperframe);
    }
    Ok(frames)
}
