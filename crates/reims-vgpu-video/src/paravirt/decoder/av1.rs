//! An AV1 session's configuration record, `av1C` (AV1 Codec ISOBMFF Binding, 2.3): what the
//! host reads of it, as [`super::read_record`] reads H.264's and HEVC's.
//!
//! An AV1 sample is a temporal unit of OBUs in the low overhead format, which the decoder takes
//! whole: no NAL unit lengths are rewritten, and the record's `configOBUs` (its sequence header,
//! where it carries one) are the parameter sets the first decode is handed. The stream is told
//! from the sequence header among them where there is one, else from the record's own fields,
//! which give its profile, bit depth and subsampling but no size or range.

use super::{DecoderConfig, Record, Stream, Undecoded, pixel_format};
use crate::codec::av1::sequence::SequenceHeader;
use crate::codec::av1::{DecoderConfiguration, Limits, check_record, check_supported, obu};
use crate::paravirt::Codec;
use crate::paravirt::status::{self, OsStatus};

/// No limit on the frame's size: the arm holds the stream to its own largest frame
/// (`Backend::declines`), so the codec's check here is of the rest.
const ANY_SIZE: Limits = Limits {
    max_width: u32::MAX,
    max_height: u32::MAX,
    min_width: 0,
    min_height: 0,
};

/// Reads `config`'s `av1C` record, or `kVTVideoDecoderUnsupportedDataFormatErr` for one that
/// does not read, or whose `configOBUs` do not.
pub(super) fn read_record(config: &DecoderConfig) -> Result<Record, OsStatus> {
    let unsupported = status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT;
    let record = DecoderConfiguration::parse(&config.record).map_err(|_| unsupported)?;
    let mut sequence = None;
    for unit in obu::obus(record.config_obus) {
        let unit = unit.map_err(|_| unsupported)?;
        if unit.obu_type == obu::SEQUENCE_HEADER {
            sequence = Some(SequenceHeader::parse(unit.payload).map_err(|_| unsupported)?);
        }
    }
    let bits = match &sequence {
        Some(sequence) => sequence.color.bit_depth,
        None => record.bit_depth(),
    };
    let (width, height) = match &sequence {
        Some(sequence) => (sequence.max_frame_width, sequence.max_frame_height),
        None => (config.width, config.height),
    };
    let stream = Stream {
        codec: Codec::Av1,
        profile: record.seq_profile,
        chroma_format_idc: chroma_format(&record),
        luma_bits: bits,
        chroma_bits: bits,
        width,
        height,
        range_extensions: false,
    };
    let refused = match &sequence {
        Some(sequence) => check_supported(sequence, &ANY_SIZE),
        None => check_record(&record),
    };
    let undecoded = refused.err().map(|error| Undecoded {
        why: error.to_string(),
        stream,
    });
    Ok(Record {
        length_size: 0,
        parameter_sets: record.config_obus.to_vec(),
        format: pixel_format(
            &stream,
            sequence.as_ref().is_some_and(|s| s.color.color_range),
        ),
        stream,
        undecoded,
    })
}

/// The record's subsampling as HEVC's `chroma_format_idc` numbers it: 0 monochrome, 1 4:2:0,
/// 2 4:2:2, 3 4:4:4.
fn chroma_format(record: &DecoderConfiguration<'_>) -> u8 {
    match (
        record.monochrome,
        record.chroma_subsampling_x,
        record.chroma_subsampling_y,
    ) {
        (true, _, _) => 0,
        (false, true, true) => 1,
        (false, true, false) => 2,
        (false, false, _) => 3,
    }
}
