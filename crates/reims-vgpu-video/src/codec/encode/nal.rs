//! NAL units on the way out: an RBSP escaped with emulation prevention
//! bytes (H.264 7.4.1, H.265 7.4.2), Annex B byte streams split into
//! their units (B.2), and samples in the length-prefixed form MP4 and
//! VideoToolbox carry (ISO/IEC 14496-15).

use crate::codec::h264::nal::{AnnexB, LengthPrefixed};

/// The bytes of a NAL unit: `header` (one byte for H.264, two for H.265),
/// then `rbsp` with an emulation prevention byte inserted wherever two zero
/// bytes would be followed by a byte of 3 or less.
pub(crate) fn escape(header: &[u8], rbsp: &[u8]) -> Vec<u8> {
    let mut out = header.to_vec();
    out.reserve(rbsp.len());
    let mut zeros = 0u32;
    for &byte in rbsp {
        if zeros >= 2 && byte <= 3 {
            out.push(3);
            zeros = 0;
        }
        out.push(byte);
        // Only whether two have run matters, which a count held at its
        // largest still says.
        zeros = if byte == 0 {
            zeros.saturating_add(1)
        } else {
            0
        };
    }
    out
}

/// The four-byte start code the encoders put before each unit they hand
/// the driver.
pub const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// The NAL units of an Annex B byte stream, each without its start code
/// or the zero bytes that trail it, as the decoders split one
/// ([`crate::codec::h264::nal::AnnexB`]). Bytes before the first start code are
/// skipped.
pub fn split_annex_b(stream: &[u8]) -> Vec<&[u8]> {
    AnnexB::new(stream).collect()
}

/// Appends `unit` to `sample` behind a four-byte big-endian length.
pub fn push_length_prefixed(sample: &mut Vec<u8>, unit: &[u8]) {
    let length = u32::try_from(unit.len()).unwrap_or(u32::MAX);
    sample.extend_from_slice(&length.to_be_bytes());
    sample.extend_from_slice(unit);
}

/// The units of a sample of four-byte length-prefixed NAL units, or
/// `None` when a length runs past the end, as the decoders read them
/// ([`crate::codec::h264::nal::LengthPrefixed`]).
pub fn length_prefixed_units(sample: &[u8]) -> Option<Vec<&[u8]>> {
    LengthPrefixed::new(sample, 4)
        .collect::<Result<_, _>>()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emulation_prevention_bytes_go_where_9_1_needs_them() {
        assert_eq!(
            escape(&[0x67], &[0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0]),
            [0x67, 0, 0, 3, 0, 0, 3, 0, 1, 0, 0, 4, 0, 0]
        );
        assert_eq!(escape(&[], &[0, 0, 3]), [0, 0, 3, 3]);
    }

    #[test]
    fn annex_b_streams_split_into_units() {
        let stream = [
            0xff, 0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 0, 1, 0x65, 4, 0, 0,
        ];
        assert_eq!(
            split_annex_b(&stream),
            [&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 4][..]]
        );
    }

    #[test]
    fn length_prefixed_samples_round_trip() {
        let mut sample = Vec::new();
        push_length_prefixed(&mut sample, &[0x65, 1, 2]);
        push_length_prefixed(&mut sample, &[0x06]);
        assert_eq!(sample, [0, 0, 0, 3, 0x65, 1, 2, 0, 0, 0, 1, 6]);
        assert_eq!(
            length_prefixed_units(&sample),
            Some(vec![&[0x65, 1, 2][..], &[6][..]])
        );
        assert_eq!(length_prefixed_units(&sample[..5]), None);
    }
}
