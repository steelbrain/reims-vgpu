//! SEI messages, which H.264 (7.3.2.3.1, D.1) and HEVC (7.3.5, D.2) code
//! alike: each a payload type and a payload size, both as runs of 0xFF
//! bytes and a last byte, then that many bytes of payload, until the
//! RBSP's trailing bits.
//!
//! Only the colour volume messages are read, numbered alike in both
//! standards: the mastering display's colour volume, the content light
//! level and the alternative transfer characteristics. Nothing here is
//! used to decode: they say what a stream carries (the info examples print
//! them).
//!
//! Bounded by the unit: every byte read is the unit's, so a message's
//! payload, and the count of messages, cannot pass the unit's length, and
//! the unit is read once, whatever it holds.

use super::Error;
use super::bits::Unescape;
use crate::codec::num::ToUsize;

/// `mastering_display_colour_volume()`'s payload type (H.264 D.1.29, HEVC
/// D.2.28).
pub(crate) const MASTERING_DISPLAY_COLOUR_VOLUME: u32 = 137;
/// `content_light_level_info()`'s (H.264 D.1.31, HEVC D.2.35).
pub(crate) const CONTENT_LIGHT_LEVEL_INFO: u32 = 144;
/// `alternative_transfer_characteristics()`'s (H.264 D.1.32, HEVC
/// D.2.38).
pub(crate) const ALTERNATIVE_TRANSFER_CHARACTERISTICS: u32 = 147;

/// One SEI message: its payload type and its payload, emulation
/// prevention bytes removed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Message {
    /// `payloadType`.
    pub(crate) payload_type: u32,
    /// The payload's `payloadSize` bytes.
    pub(crate) payload: Vec<u8>,
}

/// The messages of one SEI NAL unit, as escaped in the stream, with a NAL
/// unit header of `header_bytes` before the RBSP: 1 for H.264, 2 for HEVC.
///
/// Every message ends on a byte (7.3.2.3.1), so the unit is read as bytes,
/// once: the bit reader's `more_rbsp_data` looks for the stop bit from the
/// start each time it is asked, which a unit of many short messages would
/// make quadratic.
pub(crate) fn messages(nal: &[u8], header_bytes: usize) -> Result<Vec<Message>, Error> {
    let rbsp: Vec<u8> = Unescape::new(nal.get(header_bytes..).ok_or(Error::Truncated)?).collect();
    // The stop bit is in the last byte that is not zero; with none, there
    // is nothing but trailing bits, as more_rbsp_data says (7.2).
    // `from_stop` is that byte and the zero bytes after it.
    let Some((_, from_stop @ [stop_byte, ..])) = rbsp
        .iter()
        .rposition(|&byte| byte != 0)
        .and_then(|stop| rbsp.split_at_checked(stop))
    else {
        return Ok(Vec::new());
    };
    let mut messages = Vec::new();
    let mut rest = rbsp.as_slice();
    // more_rbsp_data at a byte: data is left before the stop byte, or in
    // the stop byte's bits above its stop bit.
    while rest.len() > from_stop.len() || (rest.len() == from_stop.len() && *stop_byte != 0x80) {
        let payload_type = byte_run(&mut rest, "payloadType")?;
        let size = byte_run(&mut rest, "payloadSize")?.to_usize();
        let (payload, after) = rest.split_at_checked(size).ok_or(Error::Truncated)?;
        rest = after;
        messages.push(Message {
            payload_type,
            payload: payload.to_vec(),
        });
    }
    Ok(messages)
}

/// A payload type or size at the start of `rest`: 0xFF bytes, each adding
/// 255, then the last byte (7.3.2.3.1).
fn byte_run(rest: &mut &[u8], name: &'static str) -> Result<u32, Error> {
    let mut value = 0u32;
    loop {
        let (&byte, after) = rest.split_first().ok_or(Error::Truncated)?;
        *rest = after;
        value = value
            .checked_add(u32::from(byte))
            .ok_or(Error::OutOfRange(name))?;
        if byte != 0xff {
            return Ok(value);
        }
    }
}

/// `mastering_display_colour_volume()`: SMPTE ST 2086's colour volume of
/// the display the content was mastered on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasteringDisplay {
    /// `display_primaries_x[c]` and `display_primaries_y[c]` in units of
    /// 0.00002, in coded order; the standards advise green, blue then red.
    pub display_primaries: [(u16, u16); 3],
    /// `white_point_x` and `white_point_y`, in units of 0.00002.
    pub white_point: (u16, u16),
    /// `max_display_mastering_luminance`, in units of 0.0001 cd/m2.
    pub max_luminance: u32,
    /// `min_display_mastering_luminance`, in units of 0.0001 cd/m2.
    pub min_luminance: u32,
}

/// `content_light_level_info()`: CTA-861.3's light levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentLightLevel {
    /// `max_content_light_level`, in cd/m2.
    pub max_content_light_level: u16,
    /// `max_pic_average_light_level`, in cd/m2.
    pub max_pic_average_light_level: u16,
}

/// The colour volume messages of an SEI unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColourMessage {
    /// A mastering display colour volume.
    MasteringDisplay(MasteringDisplay),
    /// A content light level.
    ContentLightLevel(ContentLightLevel),
    /// `preferred_transfer_characteristics`, a transfer characteristics
    /// code to use in place of the VUI's (18 for HLG).
    AlternativeTransfer(u8),
}

/// The big-endian `N` bytes of `payload` from `at`.
fn be<const N: usize>(payload: &[u8], at: usize) -> Result<[u8; N], Error> {
    payload
        .get(at..)
        .and_then(<[u8]>::first_chunk)
        .copied()
        .ok_or(Error::Truncated)
}

fn be16(payload: &[u8], at: usize) -> Result<u16, Error> {
    be(payload, at).map(u16::from_be_bytes)
}

fn be32(payload: &[u8], at: usize) -> Result<u32, Error> {
    be(payload, at).map(u32::from_be_bytes)
}

impl MasteringDisplay {
    /// Reads the message's 24 bytes of payload.
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            display_primaries: [
                (be16(payload, 0)?, be16(payload, 2)?),
                (be16(payload, 4)?, be16(payload, 6)?),
                (be16(payload, 8)?, be16(payload, 10)?),
            ],
            white_point: (be16(payload, 12)?, be16(payload, 14)?),
            max_luminance: be32(payload, 16)?,
            min_luminance: be32(payload, 20)?,
        })
    }
}

impl ContentLightLevel {
    /// Reads the message's 4 bytes of payload.
    pub(crate) fn parse(payload: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            max_content_light_level: be16(payload, 0)?,
            max_pic_average_light_level: be16(payload, 2)?,
        })
    }
}

/// The colour volume messages of one SEI NAL unit, in order; other
/// messages are left out.
pub fn colour_messages(nal: &[u8], header_bytes: usize) -> Result<Vec<ColourMessage>, Error> {
    let mut found = Vec::new();
    for message in messages(nal, header_bytes)? {
        found.push(match message.payload_type {
            MASTERING_DISPLAY_COLOUR_VOLUME => {
                ColourMessage::MasteringDisplay(MasteringDisplay::parse(&message.payload)?)
            }
            CONTENT_LIGHT_LEVEL_INFO => {
                ColourMessage::ContentLightLevel(ContentLightLevel::parse(&message.payload)?)
            }
            ALTERNATIVE_TRANSFER_CHARACTERISTICS => ColourMessage::AlternativeTransfer(
                *message.payload.first().ok_or(Error::Truncated)?,
            ),
            _ => continue,
        });
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An H.264 SEI unit (type 6) of the given messages, written by hand
    /// to 7.3.2.3.1: type and size as 0xFF runs, the payload, the trailing
    /// bits. No emulation prevention is needed for these bytes unless a
    /// test adds it.
    fn unit(messages: &[(u32, &[u8])]) -> Vec<u8> {
        let mut out = vec![0x06];
        for &(kind, payload) in messages {
            for mut value in [kind, payload.len() as u32] {
                while value >= 255 {
                    out.push(0xff);
                    value -= 255;
                }
                out.push(value as u8);
            }
            out.extend_from_slice(payload);
        }
        out.push(0x80);
        out
    }

    const P3_D65_1000: [u8; 24] = [
        0x33, 0xc2, 0x86, 0xc4, // G 13250, 34500
        0x1d, 0x4c, 0x0b, 0xb8, // B 7500, 3000
        0x84, 0xd0, 0x3e, 0x80, // R 34000, 16000
        0x3d, 0x13, 0x40, 0x42, // white 15635, 16450
        0x00, 0x98, 0x96, 0x80, // 10,000,000: 1000 cd/m2
        0x00, 0x00, 0x00, 0x01, // 1: 0.0001 cd/m2
    ];

    #[test]
    fn hdr10s_messages_read_as_coded() -> Result<(), Error> {
        let nal = unit(&[(137, &P3_D65_1000), (144, &[0x03, 0xe8, 0x01, 0x90])]);
        assert_eq!(
            colour_messages(&nal, 1)?,
            [
                ColourMessage::MasteringDisplay(MasteringDisplay {
                    display_primaries: [(13250, 34500), (7500, 3000), (34000, 16000)],
                    white_point: (15635, 16450),
                    max_luminance: 10_000_000,
                    min_luminance: 1,
                }),
                ColourMessage::ContentLightLevel(ContentLightLevel {
                    max_content_light_level: 1000,
                    max_pic_average_light_level: 400,
                }),
            ]
        );
        Ok(())
    }

    #[test]
    fn an_alternative_transfer_is_its_code_and_other_messages_are_left_out() -> Result<(), Error> {
        // A user data unregistered message (5) first, of 20 bytes.
        let nal = unit(&[(5, &[0xaa; 20]), (147, &[18])]);
        assert_eq!(messages(&nal, 1)?.len(), 2);
        assert_eq!(
            colour_messages(&nal, 1)?,
            [ColourMessage::AlternativeTransfer(18)]
        );
        Ok(())
    }

    #[test]
    fn hevcs_two_byte_header_is_skipped() -> Result<(), Error> {
        // A prefix SEI (39): nal_unit_type 39 << 1, then layer 0, tid 1.
        let mut nal = vec![0x4e, 0x01];
        nal.extend_from_slice(&unit(&[(144, &[0x00, 0x64, 0x00, 0x32])])[1..]);
        assert_eq!(
            colour_messages(&nal, 2)?,
            [ColourMessage::ContentLightLevel(ContentLightLevel {
                max_content_light_level: 100,
                max_pic_average_light_level: 50,
            })]
        );
        Ok(())
    }

    #[test]
    fn types_and_sizes_past_255_take_ff_runs() -> Result<(), Error> {
        // Type 600 is 0xff 0xff 90; a 300-byte payload is 0xff 45.
        let payload = vec![0x11; 300];
        let nal = unit(&[(600, &payload)]);
        assert_eq!(&nal[1..6], &[0xff, 0xff, 90, 0xff, 45]);
        let read = messages(&nal, 1)?;
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].payload_type, 600);
        assert_eq!(read[0].payload, payload);
        Ok(())
    }

    #[test]
    fn emulation_prevention_is_removed_from_payloads() -> Result<(), Error> {
        // A light level of 0 and 3: 00 00 00 03, which the stream escapes
        // as 00 00 03 00 03.
        let nal = [0x06, 144, 4, 0x00, 0x00, 0x03, 0x00, 0x03, 0x80];
        assert_eq!(
            colour_messages(&nal, 1)?,
            [ColourMessage::ContentLightLevel(ContentLightLevel {
                max_content_light_level: 0,
                max_pic_average_light_level: 3,
            })]
        );
        Ok(())
    }

    #[test]
    fn a_size_past_the_unit_or_a_short_payload_is_refused() {
        // 137 says 200 bytes, and 3 follow.
        let nal = [0x06, 137, 200, 1, 2, 3, 0x80];
        assert_eq!(messages(&nal, 1), Err(Error::Truncated));
        // A mastering display of 23 bytes.
        let nal = unit(&[(137, &P3_D65_1000[..23])]);
        assert!(messages(&nal, 1).is_ok());
        assert_eq!(colour_messages(&nal, 1), Err(Error::Truncated));
        // An alternative transfer of no bytes.
        let nal = unit(&[(147, &[])]);
        assert_eq!(colour_messages(&nal, 1), Err(Error::Truncated));
    }

    #[test]
    fn a_run_of_ff_bytes_ends_with_the_unit() {
        // 16,843,010 0xFF bytes would pass 2^32; a run of 4096 ends with
        // the unit long before, as Truncated.
        let mut nal = vec![0x06];
        nal.extend(std::iter::repeat_n(0xff, 4096));
        assert_eq!(messages(&nal, 1), Err(Error::Truncated));
    }

    #[test]
    fn many_short_messages_are_read_in_one_pass() -> Result<(), Error> {
        // 500,000 empty messages of type 1, two bytes each: read once, this
        // takes milliseconds, where a stop-bit search per message would take
        // minutes.
        let mut nal = vec![0x06];
        for _ in 0..500_000 {
            nal.extend_from_slice(&[0x01, 0x00]);
        }
        nal.push(0x80);
        assert_eq!(messages(&nal, 1)?.len(), 500_000);
        Ok(())
    }

    #[test]
    fn a_stop_byte_with_data_above_its_stop_bit_is_a_message_start() {
        // After one message, 0x81: more_rbsp_data is true at it, so it is
        // read as a payload type of 129 whose size is missing.
        let nal = [0x06, 0x05, 0x00, 0x81];
        assert_eq!(messages(&nal, 1), Err(Error::Truncated));
        // Zero bytes after the trailing bits change nothing.
        let nal = [0x06, 0x05, 0x00, 0x80, 0x00, 0x00];
        assert_eq!(messages(&nal, 1).map(|m| m.len()), Ok(1));
    }

    #[test]
    fn a_unit_of_trailing_bits_alone_has_no_messages() -> Result<(), Error> {
        assert!(messages(&[0x06, 0x80], 1)?.is_empty());
        Ok(())
    }
}
