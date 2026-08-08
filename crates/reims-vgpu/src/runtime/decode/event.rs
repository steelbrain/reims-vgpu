//! Event/sync command decoder (port of `host/utils/reims-vgpu-event-decode`).

use crate::contract::endian::{ld32, ld64};
use reims_vgpu_wire::ops::blit as wire_blit;

pub const U32_SIZE: usize = 4;
pub const U64_SIZE: usize = 8;
/// The serializer op header's own two fields. This decoder frames records no
/// capture has driven — the event encoder is the one segment family
/// `reims-vgpu-wire` deliberately does not name — but the *header* around them
/// is the one every other record carries, so it comes from the same place.
pub const OPCODE_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, opcode);
pub const LENGTH_OFFSET: usize = core::mem::offset_of!(reims_vgpu_wire::OpHeader, length);
/// Shared serializer op-header length from `reims-vgpu-wire`.
use reims_vgpu_wire::OP_HEADER_LEN;

pub const VALUE_REF: usize = 0;
pub const VALUE_VALUE: usize = 4;
pub const TIMEOUT: usize = 12;
pub const SIGNAL_WAIT_PAYLOAD_LEN: usize = VALUE_VALUE + U64_SIZE;
pub const WAIT_TIMEOUT_PAYLOAD_LEN: usize = TIMEOUT + U32_SIZE;
pub const SIGNAL_WAIT_LEN: usize = OP_HEADER_LEN + SIGNAL_WAIT_PAYLOAD_LEN;
pub const WAIT_TIMEOUT_LEN: usize = OP_HEADER_LEN + WAIT_TIMEOUT_PAYLOAD_LEN;

pub const OP_WAIT_EVENT: u32 = 0x190;
pub const OP_SIGNAL_EVENT: u32 = 0x191;
pub const OP_WAIT_EVENT_TIMEOUT: u32 = 0x192;

/// Opcodes the event deserializer refuses, and why each is here.
///
/// The first two are the **blit encoder's** fence records, and naming them is
/// this list's whole point: `0x13c`/`0x13d` are real opcodes in another space,
/// so an event decoder that accepted them would be reading a blit fence as an
/// event. They are taken from the crate that derived them rather than written
/// again — a number whose only job is to be another encoder's opcode should be
/// that opcode, or the two can part company and this list starts refusing
/// something Apple never writes while letting a renumbered fence through.
///
/// The last two are the boundary probes, one below and one above the event
/// window `OP_WAIT_EVENT`..=`OP_WAIT_EVENT_TIMEOUT`. They are derived from the
/// window rather than transcribed beside it, so extending the window moves the
/// probes with it instead of leaving one of them *inside* the accepted range —
/// which would make this predicate refuse a command the decoder implements.
pub const REJECTED_BLIT_UPDATE_FENCE: u32 = wire_blit::OPCODE_UPDATE_FENCE;
pub const REJECTED_BLIT_WAIT_FENCE: u32 = wire_blit::OPCODE_WAIT_FOR_FENCE;
pub const REJECTED_BEFORE_WINDOW: u32 = OP_WAIT_EVENT - 1;
pub const REJECTED_AFTER_WINDOW: u32 = OP_WAIT_EVENT_TIMEOUT + 1;

/// Why the event decoder refused a command.
///
/// No `Ok` and no `ErrArgs`, for the reason recorded on `blit::DecodeStatus`:
/// success is the result's own `Ok`, and a bad argument here is a payload
/// shorter than the field, which `ErrShort` already names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrBadLength,
    ErrUnknownOpcode,
    ErrRejectedOpcode,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `event_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the event decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "event_decode_short",
            Self::ErrBadLength => "event_decode_bad_length",
            Self::ErrUnknownOpcode => "event_decode_unknown_opcode",
            Self::ErrRejectedOpcode => "event_decode_rejected_opcode",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Unknown = 0,
    SignalEvent,
    WaitEvent,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub event_ref: u32,
    pub value: u64,
    pub has_timeout: bool,
    pub timeout: u32,
    pub raw_payload_offset: usize,
    pub raw_payload_length: usize,
}

pub fn opcode_rejected_by_deserializer(opcode: u32) -> bool {
    matches!(
        opcode,
        REJECTED_BLIT_UPDATE_FENCE
            | REJECTED_BLIT_WAIT_FENCE
            | REJECTED_BEFORE_WINDOW
            | REJECTED_AFTER_WINDOW
    )
}

/// Decode one event command. Transactional: returns Ok only with a full snapshot.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    if command.len() < OP_HEADER_LEN {
        return Err(DecodeStatus::ErrShort);
    }
    let opcode = ld32(&command[OPCODE_OFFSET..]);
    let command_length = ld32(&command[LENGTH_OFFSET..]) as usize;
    if command_length < OP_HEADER_LEN || command_length > command.len() {
        return Err(DecodeStatus::ErrShort);
    }
    let payload = &command[OP_HEADER_LEN..command_length];
    let mut decoded = Command {
        opcode,
        command_length: command_length as u32,
        kind: Kind::Unknown,
        event_ref: 0,
        value: 0,
        has_timeout: false,
        timeout: 0,
        raw_payload_offset: OP_HEADER_LEN,
        raw_payload_length: command_length - OP_HEADER_LEN,
    };

    match opcode {
        OP_WAIT_EVENT => {
            if command_length < SIGNAL_WAIT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::WaitEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            Ok(decoded)
        }
        OP_SIGNAL_EVENT => {
            if command_length < SIGNAL_WAIT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::SignalEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            Ok(decoded)
        }
        OP_WAIT_EVENT_TIMEOUT => {
            if command_length < WAIT_TIMEOUT_LEN {
                return Err(DecodeStatus::ErrBadLength);
            }
            decoded.kind = Kind::WaitEvent;
            decoded.event_ref = ld32(&payload[VALUE_REF..]);
            decoded.value = ld64(&payload[VALUE_VALUE..]);
            decoded.has_timeout = true;
            decoded.timeout = ld32(&payload[TIMEOUT..]);
            Ok(decoded)
        }
        _ => {
            if opcode_rejected_by_deserializer(opcode) {
                Err(DecodeStatus::ErrRejectedOpcode)
            } else {
                Err(DecodeStatus::ErrUnknownOpcode)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    /// A malformed event command used to be dropped at the dispatch site with no
    /// log line at all — indistinguishable from a segment carrying no event
    /// work. Each check names itself now, `Ok` still produces nothing, and the
    /// prefix keeps them apart from the six sibling `DecodeStatus` enums.
    #[test]
    fn every_event_decode_failure_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrShort,
            DecodeStatus::ErrBadLength,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrRejectedOpcode,
        ];
        let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
        assert!(slugs.iter().all(|s| s.starts_with("event_decode_")));
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two event decode checks share a slug");
    }
    use super::*;
    use crate::contract::endian::{st32, st64};

    fn build(opcode: u32, payload: &[u8]) -> Vec<u8> {
        let len = (OP_HEADER_LEN + payload.len()) as u32;
        let mut v = vec![0u8; OP_HEADER_LEN + payload.len()];
        st32(&mut v[0..4], opcode);
        st32(&mut v[4..8], len);
        v[OP_HEADER_LEN..].copy_from_slice(payload);
        v
    }

    #[test]
    fn signal_and_wait() {
        let mut payload = [0u8; SIGNAL_WAIT_PAYLOAD_LEN];
        st32(&mut payload[0..4], 7);
        st64(&mut payload[4..12], 0x100);
        let cmd = decode(&build(OP_SIGNAL_EVENT, &payload)).unwrap();
        assert_eq!(cmd.kind, Kind::SignalEvent);
        assert_eq!(cmd.event_ref, 7);
        assert_eq!(cmd.value, 0x100);

        let cmd = decode(&build(OP_WAIT_EVENT, &payload)).unwrap();
        assert_eq!(cmd.kind, Kind::WaitEvent);

        let mut p2 = [0u8; WAIT_TIMEOUT_PAYLOAD_LEN];
        p2[..SIGNAL_WAIT_PAYLOAD_LEN].copy_from_slice(&payload);
        st32(&mut p2[TIMEOUT..TIMEOUT + 4], 42);
        let cmd = decode(&build(OP_WAIT_EVENT_TIMEOUT, &p2)).unwrap();
        assert!(cmd.has_timeout);
        assert_eq!(cmd.timeout, 42);
    }

    #[test]
    fn rejected_and_unknown() {
        assert_eq!(
            decode(&build(REJECTED_BLIT_UPDATE_FENCE, &[])).unwrap_err(),
            DecodeStatus::ErrRejectedOpcode
        );
        assert_eq!(
            decode(&build(0x999, &[])).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
    }

    #[test]
    fn short_header() {
        assert_eq!(decode(&[0; 4]).unwrap_err(), DecodeStatus::ErrShort);
    }

    /// No refused opcode is one this decoder implements.
    ///
    /// The two boundary probes are now `OP_WAIT_EVENT - 1` and
    /// `OP_WAIT_EVENT_TIMEOUT + 1` rather than `0x18f`/`0x193` written beside
    /// them, and this is the property that derivation buys: a window that grows
    /// downward or upward carries its probes with it. Transcribed, one of them
    /// would end up *inside* the accepted range and
    /// [`opcode_rejected_by_deserializer`] would refuse a command the match
    /// below decodes — a guest's event silently declined by a constant that was
    /// only ever meant to sit outside.
    ///
    /// The blit pair is checked the same way and for the sharper reason: those
    /// two are real opcodes in another encoder's space, so this list is only
    /// correct while it names *that* space's fences and not this one's events.
    #[test]
    fn no_refused_opcode_is_one_this_decoder_implements() {
        for (op, name) in [
            (OP_WAIT_EVENT, "OP_WAIT_EVENT"),
            (OP_SIGNAL_EVENT, "OP_SIGNAL_EVENT"),
            (OP_WAIT_EVENT_TIMEOUT, "OP_WAIT_EVENT_TIMEOUT"),
        ] {
            assert!(
                !opcode_rejected_by_deserializer(op),
                "{name} = {op:#x} is both implemented and refused"
            );
        }
        // And every refused opcode really is refused, so the list is not
        // quietly empty of the case it exists for.
        for (op, name) in [
            (REJECTED_BLIT_UPDATE_FENCE, "REJECTED_BLIT_UPDATE_FENCE"),
            (REJECTED_BLIT_WAIT_FENCE, "REJECTED_BLIT_WAIT_FENCE"),
            (REJECTED_BEFORE_WINDOW, "REJECTED_BEFORE_WINDOW"),
            (REJECTED_AFTER_WINDOW, "REJECTED_AFTER_WINDOW"),
        ] {
            assert!(opcode_rejected_by_deserializer(op), "{name} is not refused");
            assert_eq!(
                decode(&build(op, &[0u8; 32])).unwrap_err(),
                DecodeStatus::ErrRejectedOpcode,
                "{name} = {op:#x} must refuse by name rather than as an unknown"
            );
        }
    }

    #[test]
    fn property_fuzz_opcodes() {
        for op in 0u32..0x200 {
            let mut v = build(op, &[0u8; 32]);
            // Force length large enough
            let len = v.len() as u32;
            st32(&mut v[4..8], len);
            let _ = decode(&v);
        }
    }
}
