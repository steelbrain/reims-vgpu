//! AV1 sessions, on our own streams (`codec::av1::tests`): each record's `av1C`, written here from
//! the stream's sequence header, starts a session whose buffers are in the format of its depth
//! and range, or is refused at its start; the record's `configOBUs` go to the backend before the
//! first sample, and each sample goes whole.

use super::*;
use crate::codec::av1::obu;
use crate::codec::av1::sequence::SequenceHeader;
use crate::codec::av1::tests::{samples, stream};

/// The stream `name`'s first sample's sequence header OBU, whole, and the header it reads as.
fn sequence_obu(name: &str) -> (Vec<u8>, SequenceHeader) {
    let first = samples(stream(name).ivf)[0];
    let unit = obu::obus(first)
        .map(|unit| unit.expect("an OBU"))
        .find(|unit| unit.obu_type == obu::SEQUENCE_HEADER)
        .expect("a sequence header in the first sample");
    let header = SequenceHeader::parse(unit.payload).expect("a sequence header");
    // obu_type 1 with obu_has_size_field, then the payload's size in one leb128 byte.
    let size = u8::try_from(unit.payload.len()).expect("a short sequence header");
    assert!(size < 0x80);
    let mut whole = vec![0x0a, size];
    whole.extend_from_slice(unit.payload);
    (whole, header)
}

/// An `av1C` of the fields `header` gives, with `config_obus` after its four fixed bytes.
fn av1c(header: &SequenceHeader, config_obus: &[u8]) -> Vec<u8> {
    let color = &header.color;
    let flags = (u8::from(color.bit_depth > 8) << 6)
        | (u8::from(color.bit_depth == 12) << 5)
        | (u8::from(color.mono_chrome) << 4)
        | (u8::from(color.subsampling_x) << 3)
        | (u8::from(color.subsampling_y) << 2);
    let mut record = vec![0x81, header.seq_profile << 5 | 8, flags, 0];
    record.extend_from_slice(config_obus);
    record
}

/// The stream `name`'s `av1C`, its sequence header in `configOBUs`.
fn record_of(name: &str) -> Vec<u8> {
    let (unit, header) = sequence_obu(name);
    av1c(&header, &unit)
}

#[test]
fn each_av1_stream_lands_in_its_format_or_is_refused_at_its_start() {
    // Main at 8 and 10 bits lands in the two-plane formats of its depth and range. High (4:4:4)
    // and Professional (12 bits) are refused at their start, as no output format holds them.
    for (name, format) in [
        ("svt-main8", Some(b"420v")),
        ("svt-main10", Some(b"x420")),
        ("aom-odd", Some(b"420v")),
        ("aom-444", None),
        ("aom-main12", None),
    ] {
        let (_, header) = sequence_obu(name);
        assert!(!header.color.color_range, "{name}: video range");
        starts_in(Codec::Av1, &record_of(name), false, format, name);
    }
}

#[test]
fn a_record_without_a_sequence_header_is_told_by_its_own_fields() {
    // A record whose configOBUs are empty: its fields alone give the profile and depth, and the
    // first sample brings the sequence header.
    for (name, format) in [
        ("svt-main8", Some(b"420v")),
        ("svt-main10", Some(b"x420")),
        ("aom-444", None),
    ] {
        let (_, header) = sequence_obu(name);
        starts_in(Codec::Av1, &av1c(&header, &[]), false, format, name);
    }
}

#[test]
fn an_av1_session_hands_the_backend_its_config_obus_then_whole_samples() {
    let guest = Guest::of(Codec::Av1, Ok(Decoded::Landed), None);
    let (unit, _) = sequence_obu("svt-main8");
    guest.start(&record_of("svt-main8"));
    guest.next_of(b"Dpba");
    let (_, reply) = guest.next_of(b"dsta");
    assert_eq!(status_of(&reply), Some(0));
    let samples = samples(stream("svt-main8").ivf);
    let buffer = GuestUuid([0xa1; 16]);
    for (frame, sample) in samples.iter().take(2).enumerate() {
        let frame = i64::try_from(frame).unwrap();
        guest.frame(frame, 0, &one_sample(sample, None, false));
        let (dbuf, _) = guest.next_of(b"Dbuf");
        guest.answer(&dbuf, 0, Some((buffer, 9)));
        let (demt, emitted) = guest.next_of(b"Demt");
        assert_eq!(status_of(&emitted), Some(0));
        guest.answer(&demt, 0, None);
        assert_eq!(guest.next(), surface_release(&[9]));
        guest.next_of(b"dfrm");
    }
    let calls = guest.calls.lock();
    assert_eq!(
        calls[0],
        (unit, samples[0].to_vec(), 9),
        "the record's OBUs first"
    );
    assert_eq!(calls[1], (Vec::new(), samples[1].to_vec(), 9), "once");
}

#[test]
fn a_refused_av1_stream_is_named_by_its_profile_and_depths() {
    let refused = undecoded_on(&FakeBackend::landing(), Codec::Av1, record_of("aom-444"))
        .expect("High is refused");
    let line = refused.to_string();
    assert!(line.contains("AV1 seq_profile 1"), "{line}");
    assert!(line.contains("chroma_format_idc 3, luma 8 bits"), "{line}");
    assert!(line.contains("64x64"), "{line}");
    assert_eq!(
        refused.kind(),
        "AV1 profile one chroma format three, luma eight bits, chroma eight bits"
    );
    let refused = undecoded_on(&FakeBackend::landing(), Codec::Av1, record_of("aom-main12"))
        .expect("Professional is refused");
    assert!(refused.to_string().contains("luma 12 bits"), "{refused}");
    assert_eq!(
        undecoded_on(&FakeBackend::landing(), Codec::Av1, record_of("svt-main10")),
        None
    );
}

#[test]
fn an_av1_record_that_does_not_read_is_refused() {
    for record in [Vec::new(), vec![0x81, 0, 0], vec![0x01, 0, 0, 0], {
        let mut cut = record_of("svt-main8");
        cut.truncate(cut.len() - 1);
        cut
    }] {
        let config = DecoderConfig {
            codec: Codec::Av1,
            width: 64,
            height: 48,
            record,
        };
        assert!(
            matches!(
                read_record(&config),
                Err(status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT)
            ),
            "{:02x?}",
            config.record
        );
    }
}
