//! `vtpv` messages: the 48-byte header atom and the `para` parameter atoms
//! that follow it as its siblings.
//!
//! A message may start with other atoms, which the scanners skip. A
//! fragmented message is several messages one after another, each but the
//! last flagged [`MessageFlags::MORE_FRAGMENTS`]; the fragment-aware
//! readers ([`Message::data`], [`Message::sample_buffer`]) walk them.

use crate::core::fourcc::FourCc;
use crate::paravirt::WireError;
use crate::paravirt::atom::{AtomWriter, read_u32, read_u64};
use crate::paravirt::plist;

/// The header atom's type.
pub const MESSAGE_ATOM: FourCc = FourCc::new(*b"vtpv");
/// A parameter atom's type.
pub const PARAMETER_ATOM: FourCc = FourCc::new(*b"para");
/// The header atom's size: its own header, type, flags, UUID, reply
/// identifier and 8 reserved bytes.
pub const HEADER_SIZE: usize = 48;
/// A parameter atom's header: atom header, key and value type.
pub const PARAMETER_HEADER_SIZE: usize = 16;

/// A message's flags (the constant names here are ours, by direction).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MessageFlags(pub u32);

impl MessageFlags {
    /// No flags: a notification either way.
    pub const NONE: Self = Self(0);
    /// A guest request that waits for the host's reply.
    pub const GUEST_EXPECTS_REPLY: Self = Self(0x1);
    /// A host request that waits for the guest's reply.
    pub const HOST_EXPECTS_REPLY: Self = Self(0x2);
    /// The host's reply to a guest request.
    pub const REPLY_TO_GUEST: Self = Self(0x4);
    /// The guest's reply to a host request.
    pub const REPLY_TO_HOST: Self = Self(0x8);
    /// Another fragment of the same message follows.
    pub const MORE_FRAGMENTS: Self = Self(0x10);

    /// Whether every bit of `other` is set.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// These flags with the bits of `mask` replaced by those of `flags`, as
    /// VTParavirtualizationMessageChangeMessageFlags computes them.
    pub const fn changed(self, flags: Self, mask: Self) -> Self {
        Self((self.0 & !mask.0) | (flags.0 & mask.0))
    }
}

/// The UUID that names a guest session, as its 16 bytes. The host's own
/// list requests (`decs`, `dcap`, ...) carry the zero UUID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct GuestUuid(pub [u8; 16]);

impl GuestUuid {
    /// The zero UUID, which a NULL UUID is written as.
    pub const ZERO: Self = Self([0; 16]);
}

/// A `CMTime` as the wire carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MediaTime {
    /// The numerator.
    pub value: i64,
    /// The denominator; a valid time has a positive one.
    pub timescale: i32,
    /// `kCMTimeFlags_*`: bit 0 is valid.
    pub flags: u32,
    /// The epoch.
    pub epoch: i64,
}

impl MediaTime {
    /// `kCMTimeInvalid`: all zero.
    pub const INVALID: Self = Self {
        value: 0,
        timescale: 0,
        flags: 0,
        epoch: 0,
    };
    /// `kCMTimeFlags_Valid`.
    pub const FLAG_VALID: u32 = 1;

    /// Whether the valid flag is set.
    pub const fn is_valid(self) -> bool {
        self.flags & Self::FLAG_VALID != 0
    }

    /// The time from its native (little-endian, guest x86_64) 24-byte form,
    /// as sample buffer serializations write it.
    pub fn from_native(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            value: i64::from_le_bytes(bytes.get(0..8)?.try_into().ok()?),
            timescale: i32::from_le_bytes(bytes.get(8..12)?.try_into().ok()?),
            flags: u32::from_le_bytes(bytes.get(12..16)?.try_into().ok()?),
            epoch: i64::from_le_bytes(bytes.get(16..24)?.try_into().ok()?),
        })
    }

    /// The native 24-byte form.
    pub fn to_native(self) -> [u8; 24] {
        let mut bytes = [0; 24];
        bytes[0..8].copy_from_slice(&self.value.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.timescale.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.flags.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.epoch.to_le_bytes());
        bytes
    }
}

/// A parameter's value type, the FourCC after its key
/// (each appender).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// `=u32`: a big-endian u32.
    U32,
    /// `=s32`: a big-endian i32.
    S32,
    /// `=s64`: a big-endian i64.
    S64,
    /// `=tim`: a big-endian CMTime.
    Time,
    /// `=wxh`: CMVideoDimensions, two big-endian i32.
    Dimensions,
    /// `=32p`: VTInt32Point, two big-endian i32.
    Point,
    /// `=32s`: VTInt32Size, two big-endian i32.
    Size,
    /// `=dat`: raw bytes, fragmented across messages when large.
    Data,
    /// `=pls`: a property list; empty means NULL.
    PropertyList,
    /// `=uui`: a UUID's 16 bytes.
    Uuid,
    /// `=ui+`: UUIDs, 16 bytes each.
    UuidArray,
    /// `=tc+`: FigTagCollections.
    TagCollections,
    /// `=fds`: a big-endian QuickTime ImageDescription.
    FormatDescription,
    /// `=sbf`: a sample buffer serialization, fragmented across messages
    /// when large.
    SampleBuffer,
}

impl ValueType {
    /// The FourCC the wire spells it as.
    pub const fn fourcc(self) -> FourCc {
        FourCc::new(match self {
            Self::U32 => *b"=u32",
            Self::S32 => *b"=s32",
            Self::S64 => *b"=s64",
            Self::Time => *b"=tim",
            Self::Dimensions => *b"=wxh",
            Self::Point => *b"=32p",
            Self::Size => *b"=32s",
            Self::Data => *b"=dat",
            Self::PropertyList => *b"=pls",
            Self::Uuid => *b"=uui",
            Self::UuidArray => *b"=ui+",
            Self::TagCollections => *b"=tc+",
            Self::FormatDescription => *b"=fds",
            Self::SampleBuffer => *b"=sbf",
        })
    }
}

/// The parsed header of a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageHeader {
    /// The message type.
    pub message_type: FourCc,
    /// The flags.
    pub flags: MessageFlags,
    /// The session the message belongs to.
    pub guest_uuid: GuestUuid,
    /// The reply identifier: the requester's for a request that expects a
    /// reply and for its reply, 0 otherwise.
    pub reply_id: u64,
}

/// Finds the first `vtpv` atom at or after `start`, as
/// `vtpvScanForTopLevelBoxWithType` does: a size under 8 or past the end is
/// malformed, a `vtpv` atom under 48 bytes is malformed, and no `vtpv` atom
/// at all is malformed.
///
/// Returns the atom's offset in `bytes`.
pub fn find_header(bytes: &[u8], start: usize) -> Result<usize, WireError> {
    let region = bytes
        .get(start..)
        .ok_or(WireError::Malformed("no message header"))?;
    let mut offset = 0usize;
    while offset + 8 <= region.len() {
        let size = atom_size(region, offset)?;
        let kind = read_u32(region, offset + 4).ok_or(WireError::Malformed("atom type"))?;
        if FourCc(kind) == MESSAGE_ATOM {
            if size < HEADER_SIZE {
                return Err(WireError::Malformed("short message header"));
            }
            return Ok(start + offset);
        }
        offset += size;
    }
    Err(WireError::Malformed("no message header"))
}

/// The size of the atom at `offset` in `region`, checked as both scanners
/// check it.
fn atom_size(region: &[u8], offset: usize) -> Result<usize, WireError> {
    let size = read_u32(region, offset).ok_or(WireError::Malformed("atom size"))?;
    let size = usize::try_from(size).map_err(|_| WireError::Malformed("atom size"))?;
    if size < 8 {
        return Err(WireError::Malformed("atom size"));
    }
    if region.len() - offset < size {
        return Err(WireError::Malformed("atom past the end"));
    }
    Ok(size)
}

/// Reads the header of the first `vtpv` atom at or after `start`.
pub fn read_header(bytes: &[u8], start: usize) -> Result<MessageHeader, WireError> {
    let at = find_header(bytes, start)?;
    let field = |offset: usize| read_u32(bytes, at + offset);
    let message_type = field(8).ok_or(WireError::Malformed("message type"))?;
    let flags = field(12).ok_or(WireError::Malformed("message flags"))?;
    let uuid = bytes
        .get(at + 16..at + 32)
        .and_then(|uuid| <[u8; 16]>::try_from(uuid).ok())
        .ok_or(WireError::Malformed("message UUID"))?;
    let reply_id = read_u64(bytes, at + 32).ok_or(WireError::Malformed("reply identifier"))?;
    Ok(MessageHeader {
        message_type: FourCc(message_type),
        flags: MessageFlags(flags),
        guest_uuid: GuestUuid(uuid),
        reply_id,
    })
}

/// A message read from guest bytes: its header and its parameters.
///
/// The getters follow Apple's: a parameter that is absent reads as `None`
/// (Apple's default: 0, NULL or `kCMTimeInvalid`), one whose key matches
/// but whose value type differs is malformed, and one whose length does
/// not fit its type is malformed.
#[derive(Clone, Copy, Debug)]
pub struct Message<'a> {
    bytes: &'a [u8],
    header: MessageHeader,
}

impl<'a> Message<'a> {
    /// Reads the header; the parameters are read on demand.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, WireError> {
        let header = read_header(bytes, 0)?;
        Ok(Self { bytes, header })
    }

    /// The header.
    pub fn header(&self) -> MessageHeader {
        self.header
    }

    /// The whole message, every fragment included.
    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// The first parameter atom with `key`, from `start` on, as
    /// `vtpvScanForParameterBoxWithType` finds it. Returns the atom's
    /// offset in the message and its bytes.
    fn find(
        &self,
        start: usize,
        key: FourCc,
        value_type: ValueType,
    ) -> Result<Option<(usize, &'a [u8])>, WireError> {
        let Some(region) = self.bytes.get(start..) else {
            return Ok(None);
        };
        let mut offset = 0usize;
        while offset + PARAMETER_HEADER_SIZE <= region.len() {
            let size = atom_size(region, offset)?;
            let kind = read_u32(region, offset + 4).ok_or(WireError::Malformed("atom type"))?;
            if FourCc(kind) == PARAMETER_ATOM
                && read_u32(region, offset + 8).map(FourCc) == Some(key)
            {
                // The key and type are read from the next 8 bytes whatever
                // the atom's size, so a short atom is caught by the
                // getter's length check, as in Apple's.
                if read_u32(region, offset + 12).map(FourCc) != Some(value_type.fourcc()) {
                    return Err(WireError::Malformed("parameter value type"));
                }
                let atom = region
                    .get(offset..offset + size)
                    .ok_or(WireError::Malformed("atom past the end"))?;
                return Ok(Some((start + offset, atom)));
            }
            offset += size;
        }
        Ok(None)
    }

    /// The payload of the parameter `key` of type `value_type`, which must
    /// be `length` bytes when present.
    fn fixed(
        &self,
        key: FourCc,
        value_type: ValueType,
        length: usize,
    ) -> Result<Option<&'a [u8]>, WireError> {
        match self.find(0, key, value_type)? {
            None => Ok(None),
            Some((_, atom)) if atom.len() == PARAMETER_HEADER_SIZE + length => {
                Ok(atom.get(PARAMETER_HEADER_SIZE..))
            }
            Some(_) => Err(WireError::Malformed("parameter length")),
        }
    }

    /// Whether the message has the parameter, with a payload
    /// (`VTParavirtualizationMessageContainsParameter`).
    pub fn contains(&self, key: FourCc, value_type: ValueType) -> bool {
        matches!(self.find(0, key, value_type), Ok(Some(_)))
    }

    /// A `=u32` parameter.
    pub fn u32(&self, key: FourCc) -> Result<Option<u32>, WireError> {
        Ok(self
            .fixed(key, ValueType::U32, 4)?
            .and_then(|payload| read_u32(payload, 0)))
    }

    /// A `=s32` parameter.
    pub fn s32(&self, key: FourCc) -> Result<Option<i32>, WireError> {
        Ok(self
            .fixed(key, ValueType::S32, 4)?
            .and_then(|payload| read_u32(payload, 0))
            .map(u32::cast_signed))
    }

    /// A `=s64` parameter.
    pub fn s64(&self, key: FourCc) -> Result<Option<i64>, WireError> {
        Ok(self
            .fixed(key, ValueType::S64, 8)?
            .and_then(|payload| read_u64(payload, 0))
            .map(u64::cast_signed))
    }

    /// A `=tim` parameter.
    pub fn time(&self, key: FourCc) -> Result<Option<MediaTime>, WireError> {
        Ok(self.fixed(key, ValueType::Time, 24)?.and_then(|payload| {
            Some(MediaTime {
                value: read_u64(payload, 0)?.cast_signed(),
                timescale: read_u32(payload, 8)?.cast_signed(),
                flags: read_u32(payload, 12)?,
                epoch: read_u64(payload, 16)?.cast_signed(),
            })
        }))
    }

    /// A pair of big-endian i32: `=wxh`, `=32p` or `=32s`.
    pub fn i32_pair(
        &self,
        key: FourCc,
        value_type: ValueType,
    ) -> Result<Option<(i32, i32)>, WireError> {
        Ok(self.fixed(key, value_type, 8)?.and_then(|payload| {
            Some((
                read_u32(payload, 0)?.cast_signed(),
                read_u32(payload, 4)?.cast_signed(),
            ))
        }))
    }

    /// A `=uui` parameter.
    pub fn uuid(&self, key: FourCc) -> Result<Option<GuestUuid>, WireError> {
        Ok(self
            .fixed(key, ValueType::Uuid, 16)?
            .and_then(|payload| <[u8; 16]>::try_from(payload).ok())
            .map(GuestUuid))
    }

    /// A `=ui+` parameter: whole 16-byte UUIDs, none left over.
    pub fn uuid_array(&self, key: FourCc) -> Result<Option<Vec<GuestUuid>>, WireError> {
        let Some((_, atom)) = self.find(0, key, ValueType::UuidArray)? else {
            return Ok(None);
        };
        let payload = atom
            .get(PARAMETER_HEADER_SIZE..)
            .ok_or(WireError::Malformed("UUID array length"))?;
        if payload.len() % 16 != 0 {
            return Err(WireError::Malformed("UUID array length"));
        }
        Ok(Some(
            payload
                .as_chunks::<16>()
                .0
                .iter()
                .map(|uuid| GuestUuid(*uuid))
                .collect(),
        ))
    }

    /// A `=pls` parameter's bytes as written, a binary property list or
    /// none for NULL.
    pub fn property_list_bytes(&self, key: FourCc) -> Result<Option<&'a [u8]>, WireError> {
        Ok(self
            .find(0, key, ValueType::PropertyList)?
            .and_then(|(_, atom)| atom.get(PARAMETER_HEADER_SIZE..)))
    }

    /// A `=fds` parameter's ImageDescription bytes, at least the 86-byte
    /// header.
    pub fn image_description(&self, key: FourCc) -> Result<Option<&'a [u8]>, WireError> {
        let Some((_, atom)) = self.find(0, key, ValueType::FormatDescription)? else {
            return Ok(None);
        };
        atom.get(PARAMETER_HEADER_SIZE..)
            .filter(|payload| payload.len() >= 86)
            .map(Some)
            .ok_or(WireError::Malformed("short image description"))
    }

    /// A `=dat` parameter, gathered from every fragment.
    pub fn data(&self, key: FourCc) -> Result<Option<Vec<u8>>, WireError> {
        self.fragmented(key, ValueType::Data)
    }

    /// A `=sbf` parameter's serialization, gathered from every fragment.
    pub fn sample_buffer(&self, key: FourCc) -> Result<Option<Vec<u8>>, WireError> {
        self.fragmented(key, ValueType::SampleBuffer)
    }

    /// Walks the fragments: each fragment's flags come from the first
    /// header at or after where the walk stands, its piece is the first
    /// parameter with `key` from there on, and the walk goes on past that
    /// parameter while the flags say more fragments follow. A fragment
    /// without the parameter makes the whole value NULL; an empty piece is
    /// malformed.
    ///
    /// The header found for one fragment is still the first at or after
    /// the walk's place until the walk passes it, so it is looked for again
    /// only then: each header search starts past the last header found,
    /// and the walk stays linear in the message. Looking from each place
    /// again, as Apple's reader does, is quadratic in the fragments a
    /// message holds before its next header.
    fn fragmented(&self, key: FourCc, value_type: ValueType) -> Result<Option<Vec<u8>>, WireError> {
        let mut value = Vec::new();
        let mut start = 0usize;
        let mut header: Option<(usize, MessageFlags)> = None;
        loop {
            let flags = match header {
                Some((at, flags)) if at >= start => flags,
                _ => {
                    let at = find_header(self.bytes, start)?;
                    let flags = read_header(self.bytes, at)?.flags;
                    header = Some((at, flags));
                    flags
                }
            };
            let Some((offset, atom)) = self.find(start, key, value_type)? else {
                return Ok(None);
            };
            let piece = atom
                .get(PARAMETER_HEADER_SIZE..)
                .filter(|piece| !piece.is_empty())
                .ok_or(WireError::Malformed("empty fragment"))?;
            value.extend_from_slice(piece);
            start = offset + atom.len();
            if !flags.contains(MessageFlags::MORE_FRAGMENTS) {
                return Ok(Some(value));
            }
        }
    }
}

/// Builds a message: the header atom, then parameters in the order they
/// are appended.
#[derive(Debug)]
pub struct MessageWriter {
    writer: AtomWriter,
}

impl MessageWriter {
    /// A message with this header; `None` writes the zero UUID.
    pub fn new(
        message_type: FourCc,
        flags: MessageFlags,
        guest_uuid: Option<GuestUuid>,
        reply_id: u64,
    ) -> Self {
        let mut writer = AtomWriter::new();
        writer
            .begin(MESSAGE_ATOM)
            .append(&message_type.to_be_bytes())
            .append(&flags.0.to_be_bytes())
            .append(&guest_uuid.unwrap_or(GuestUuid::ZERO).0)
            .append(&reply_id.to_be_bytes())
            .append(&[0; 8])
            .end();
        Self { writer }
    }

    /// The reply to `request`: its type, UUID and reply identifier with
    /// `flags`.
    pub fn reply_to(request: &MessageHeader, flags: MessageFlags) -> Self {
        Self::new(
            request.message_type,
            flags,
            Some(request.guest_uuid),
            request.reply_id,
        )
    }

    /// The bytes written so far.
    pub fn len(&self) -> usize {
        self.writer.len()
    }

    /// Never true: the header is always there.
    pub fn is_empty(&self) -> bool {
        self.writer.is_empty()
    }

    fn parameter(&mut self, key: FourCc, value_type: ValueType, payload: &[u8]) -> &mut Self {
        self.writer
            .begin(PARAMETER_ATOM)
            .append(&key.to_be_bytes())
            .append(&value_type.fourcc().to_be_bytes())
            .append(payload)
            .end();
        self
    }

    /// Appends a `=u32` parameter.
    pub fn u32(&mut self, key: FourCc, value: u32) -> &mut Self {
        self.parameter(key, ValueType::U32, &value.to_be_bytes())
    }

    /// Appends a `=s32` parameter.
    pub fn s32(&mut self, key: FourCc, value: i32) -> &mut Self {
        self.parameter(key, ValueType::S32, &value.to_be_bytes())
    }

    /// Appends a `=s64` parameter.
    pub fn s64(&mut self, key: FourCc, value: i64) -> &mut Self {
        self.parameter(key, ValueType::S64, &value.to_be_bytes())
    }

    /// Appends a `=tim` parameter.
    pub fn time(&mut self, key: FourCc, time: MediaTime) -> &mut Self {
        let mut payload = [0; 24];
        payload[0..8].copy_from_slice(&time.value.to_be_bytes());
        payload[8..12].copy_from_slice(&time.timescale.to_be_bytes());
        payload[12..16].copy_from_slice(&time.flags.to_be_bytes());
        payload[16..24].copy_from_slice(&time.epoch.to_be_bytes());
        self.parameter(key, ValueType::Time, &payload)
    }

    /// Appends a pair of big-endian i32: `=wxh`, `=32p` or `=32s`.
    pub fn i32_pair(&mut self, key: FourCc, value_type: ValueType, pair: (i32, i32)) -> &mut Self {
        let mut payload = [0; 8];
        payload[0..4].copy_from_slice(&pair.0.to_be_bytes());
        payload[4..8].copy_from_slice(&pair.1.to_be_bytes());
        self.parameter(key, value_type, &payload)
    }

    /// Appends a `=uui` parameter.
    pub fn uuid(&mut self, key: FourCc, uuid: GuestUuid) -> &mut Self {
        self.parameter(key, ValueType::Uuid, &uuid.0)
    }

    /// Appends a `=ui+` parameter.
    pub fn uuid_array(&mut self, key: FourCc, uuids: &[GuestUuid]) -> &mut Self {
        let payload: Vec<u8> = uuids.iter().flat_map(|uuid| uuid.0).collect();
        self.parameter(key, ValueType::UuidArray, &payload)
    }

    /// Appends a `=pls` parameter, a binary property list ([`plist`]): `None` writes NULL, an
    /// empty payload.
    pub fn property_list(&mut self, key: FourCc, value: Option<&plist::Value>) -> &mut Self {
        let payload = value.map(plist::encode).unwrap_or_default();
        self.parameter(key, ValueType::PropertyList, &payload)
    }

    /// Appends a `=dat` parameter whole.
    pub fn data(&mut self, key: FourCc, data: &[u8]) -> &mut Self {
        self.parameter(key, ValueType::Data, data)
    }

    /// Appends a `=fds` parameter from ImageDescription bytes.
    pub fn image_description(&mut self, key: FourCc, description: &[u8]) -> &mut Self {
        self.parameter(key, ValueType::FormatDescription, description)
    }

    /// Appends as much of `serialization` as fits a message of `limit`
    /// bytes as a `=sbf` parameter, and returns the rest, as
    /// `…AppendCMSampleBufferRemainderDataWithLimitAndCopyRemainingData`
    /// does: nothing is appended when fewer than 17 bytes are left.
    pub fn sample_buffer_with_limit<'s>(
        &mut self,
        key: FourCc,
        limit: usize,
        serialization: &'s [u8],
    ) -> &'s [u8] {
        let current = self.len();
        if current + 17 > limit {
            return serialization;
        }
        let room = limit - current - PARAMETER_HEADER_SIZE;
        let chunk = room.min(serialization.len());
        let (piece, rest) = serialization.split_at(chunk);
        self.parameter(key, ValueType::SampleBuffer, piece);
        rest
    }

    /// Replaces the header's flags under `mask`.
    pub fn change_flags(&mut self, flags: MessageFlags, mask: MessageFlags) -> &mut Self {
        if let Some(field) = self.writer.bytes_mut().get_mut(12..16)
            && let Ok(current) = <[u8; 4]>::try_from(&*field)
        {
            let changed = MessageFlags(u32::from_be_bytes(current)).changed(flags, mask);
            field.copy_from_slice(&changed.0.to_be_bytes());
        }
        self
    }

    /// Replaces the header's reply identifier.
    pub fn set_reply_id(&mut self, reply_id: u64) -> &mut Self {
        if let Some(field) = self.writer.bytes_mut().get_mut(32..40) {
            field.copy_from_slice(&reply_id.to_be_bytes());
        }
        self
    }

    /// The message's bytes.
    pub fn finish(self) -> Vec<u8> {
        self.writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: FourCc = FourCc::new(*b",key");

    /// A message written by hand: `+dec`, flags 1, UUID 0x11.., reply 7,
    /// then `,cod` as `=s32` 'avc1' and `,ver` as `=u32` 6.
    fn hand_written_create() -> Vec<u8> {
        let mut bytes = vec![0, 0, 0, 48];
        bytes.extend_from_slice(b"vtpv+dec");
        bytes.extend_from_slice(&[0, 0, 0, 1]);
        bytes.extend_from_slice(&[0x11; 16]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 7]);
        bytes.extend_from_slice(&[0; 8]);
        bytes.extend_from_slice(&[0, 0, 0, 20]);
        bytes.extend_from_slice(b"para,cod=s32avc1");
        bytes.extend_from_slice(&[0, 0, 0, 20]);
        bytes.extend_from_slice(b"para,ver=u32");
        bytes.extend_from_slice(&[0, 0, 0, 6]);
        bytes
    }

    #[test]
    fn a_hand_written_message_reads_back() -> Result<(), WireError> {
        let bytes = hand_written_create();
        let message = Message::parse(&bytes)?;
        assert_eq!(
            message.header(),
            MessageHeader {
                message_type: FourCc::new(*b"+dec"),
                flags: MessageFlags::GUEST_EXPECTS_REPLY,
                guest_uuid: GuestUuid([0x11; 16]),
                reply_id: 7,
            }
        );
        assert_eq!(
            message.s32(FourCc::new(*b",cod"))?,
            Some(i32::from_be_bytes(*b"avc1"))
        );
        assert_eq!(message.u32(FourCc::new(*b",ver"))?, Some(6));
        assert_eq!(message.u32(FourCc::new(*b",gin"))?, None);
        Ok(())
    }

    #[test]
    fn the_writer_writes_the_hand_written_bytes() {
        let mut writer = MessageWriter::new(
            FourCc::new(*b"+dec"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            Some(GuestUuid([0x11; 16])),
            7,
        );
        writer
            .s32(FourCc::new(*b",cod"), i32::from_be_bytes(*b"avc1"))
            .u32(FourCc::new(*b",ver"), 6);
        assert_eq!(writer.finish(), hand_written_create());
    }

    #[test]
    fn a_key_with_another_value_type_is_malformed() -> Result<(), WireError> {
        // Measured on the M3 Max (macOS 27, 26A434): `dcap` with `,cod` as `=u32`
        // answers -19093 and no reply; the host reads it as `=s32`.
        let bytes = hand_written_create();
        let message = Message::parse(&bytes)?;
        assert_eq!(
            message.u32(FourCc::new(*b",cod")),
            Err(WireError::Malformed("parameter value type"))
        );
        Ok(())
    }

    #[test]
    fn a_parameter_of_the_wrong_length_is_malformed() -> Result<(), WireError> {
        let mut writer = MessageWriter::new(FourCc::new(*b"test"), MessageFlags::NONE, None, 0);
        writer.parameter(KEY, ValueType::U32, &[0, 0, 0, 0, 0]);
        let bytes = writer.finish();
        let message = Message::parse(&bytes)?;
        assert_eq!(
            message.u32(KEY),
            Err(WireError::Malformed("parameter length"))
        );
        Ok(())
    }

    #[test]
    fn leading_atoms_are_skipped() -> Result<(), WireError> {
        let mut bytes = vec![0, 0, 0, 0x30];
        bytes.extend_from_slice(b"subs");
        bytes.extend_from_slice(&[0; 40]);
        bytes.extend_from_slice(&hand_written_create());
        let message = Message::parse(&bytes)?;
        assert_eq!(message.header().message_type, FourCc::new(*b"+dec"));
        assert_eq!(message.u32(FourCc::new(*b",ver"))?, Some(6));
        Ok(())
    }

    #[test]
    fn a_short_header_atom_is_malformed() {
        let mut bytes = vec![0, 0, 0, 40];
        bytes.extend_from_slice(b"vtpv");
        bytes.extend_from_slice(&[0; 32]);
        assert_eq!(
            Message::parse(&bytes).err(),
            Some(WireError::Malformed("short message header"))
        );
    }

    #[test]
    fn bytes_without_a_header_are_malformed() {
        assert_eq!(
            Message::parse(&[0, 0, 0, 8, b'a', b'b', b'c', b'd']).err(),
            Some(WireError::Malformed("no message header"))
        );
        assert_eq!(
            Message::parse(&[]).err(),
            Some(WireError::Malformed("no message header"))
        );
    }

    #[test]
    fn every_scalar_type_round_trips() -> Result<(), WireError> {
        let time = MediaTime {
            value: -3,
            timescale: 600,
            flags: MediaTime::FLAG_VALID,
            epoch: 9,
        };
        let uuid = GuestUuid([7; 16]);
        let mut writer = MessageWriter::new(FourCc::new(*b"test"), MessageFlags::NONE, None, 0);
        writer
            .s64(FourCc::new(*b",s64"), -5)
            .time(FourCc::new(*b",tim"), time)
            .i32_pair(FourCc::new(*b",wxh"), ValueType::Dimensions, (1920, 1080))
            .uuid(FourCc::new(*b",uui"), uuid)
            .uuid_array(FourCc::new(*b",ui+"), &[uuid, GuestUuid::ZERO]);
        let bytes = writer.finish();
        let message = Message::parse(&bytes)?;
        assert_eq!(message.s64(FourCc::new(*b",s64"))?, Some(-5));
        assert_eq!(message.time(FourCc::new(*b",tim"))?, Some(time));
        assert_eq!(
            message.i32_pair(FourCc::new(*b",wxh"), ValueType::Dimensions)?,
            Some((1920, 1080))
        );
        assert_eq!(message.uuid(FourCc::new(*b",uui"))?, Some(uuid));
        assert_eq!(
            message.uuid_array(FourCc::new(*b",ui+"))?,
            Some(vec![uuid, GuestUuid::ZERO])
        );
        Ok(())
    }

    #[test]
    fn a_time_is_big_endian_on_the_wire() {
        let mut writer = MessageWriter::new(FourCc::new(*b"test"), MessageFlags::NONE, None, 0);
        writer.time(
            KEY,
            MediaTime {
                value: 1,
                timescale: 2,
                flags: 3,
                epoch: 4,
            },
        );
        let bytes = writer.finish();
        assert_eq!(
            &bytes[HEADER_SIZE + 16..],
            &[
                0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4
            ]
        );
    }

    #[test]
    fn a_uuid_array_with_a_partial_uuid_is_malformed() -> Result<(), WireError> {
        let mut writer = MessageWriter::new(FourCc::new(*b"test"), MessageFlags::NONE, None, 0);
        writer.parameter(KEY, ValueType::UuidArray, &[0; 17]);
        let bytes = writer.finish();
        let message = Message::parse(&bytes)?;
        assert_eq!(
            message.uuid_array(KEY),
            Err(WireError::Malformed("UUID array length"))
        );
        Ok(())
    }

    #[test]
    fn a_reply_carries_its_requests_identity() {
        let request = MessageHeader {
            message_type: FourCc::new(*b"dfrm"),
            flags: MessageFlags::GUEST_EXPECTS_REPLY,
            guest_uuid: GuestUuid([3; 16]),
            reply_id: 0x0102_0304_0506_0708,
        };
        let bytes = MessageWriter::reply_to(&request, MessageFlags::REPLY_TO_GUEST).finish();
        assert_eq!(
            read_header(&bytes, 0),
            Ok(MessageHeader {
                flags: MessageFlags::REPLY_TO_GUEST,
                ..request
            })
        );
    }

    /// Splits `serialization` into fragments as the guest decoder does: the
    /// first message carries `,frm` and then as much as fits, each full one
    /// is re-flagged "more fragments" with "expects reply" cleared, and the
    /// last keeps "expects reply".
    fn fragment(serialization: &[u8], limit: usize) -> Vec<u8> {
        let header = || {
            MessageWriter::new(
                FourCc::new(*b"dfrm"),
                MessageFlags::GUEST_EXPECTS_REPLY,
                Some(GuestUuid([1; 16])),
                0,
            )
        };
        let mut message = header();
        message.s64(FourCc::new(*b",frm"), 9);
        let mut rest = message.sample_buffer_with_limit(KEY, limit, serialization);
        let mut bytes = Vec::new();
        while !rest.is_empty() {
            message.change_flags(
                MessageFlags::MORE_FRAGMENTS,
                MessageFlags(MessageFlags::MORE_FRAGMENTS.0 | MessageFlags::GUEST_EXPECTS_REPLY.0),
            );
            bytes.extend_from_slice(&std::mem::replace(&mut message, header()).finish());
            rest = message.sample_buffer_with_limit(KEY, limit, rest);
        }
        bytes.extend_from_slice(&message.finish());
        bytes
    }

    #[test]
    fn fragments_reassemble_in_order() -> Result<(), WireError> {
        let serialization: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let bytes = fragment(&serialization, 200);
        let headers = [0, 200, 400, 600, 800, 1000]
            .iter()
            .filter_map(|start| read_header(&bytes, *start).ok())
            .count();
        assert!(headers > 4, "split into several fragments");
        let message = Message::parse(&bytes)?;
        assert_eq!(message.sample_buffer(KEY)?, Some(serialization));
        assert_eq!(message.s64(FourCc::new(*b",frm"))?, Some(9));
        Ok(())
    }

    #[test]
    fn nothing_is_appended_when_17_bytes_do_not_fit() {
        let mut message = MessageWriter::new(FourCc::new(*b"dfrm"), MessageFlags::NONE, None, 0);
        let rest = message.sample_buffer_with_limit(KEY, HEADER_SIZE + 16, &[1, 2, 3]);
        assert_eq!(rest, &[1, 2, 3]);
        assert_eq!(message.len(), HEADER_SIZE);
    }

    #[test]
    fn an_empty_fragment_is_malformed() -> Result<(), WireError> {
        let mut writer = MessageWriter::new(FourCc::new(*b"test"), MessageFlags::NONE, None, 0);
        writer.data(KEY, &[]);
        let bytes = writer.finish();
        let message = Message::parse(&bytes)?;
        assert_eq!(
            message.data(KEY),
            Err(WireError::Malformed("empty fragment"))
        );
        Ok(())
    }

    #[test]
    fn changing_flags_keeps_the_bits_outside_the_mask() {
        let flags = MessageFlags(0x1 | 0x8);
        assert_eq!(
            flags.changed(MessageFlags(0x10), MessageFlags(0x11)),
            MessageFlags(0x18)
        );
    }

    /// A reply identifier set after the parameters is the header's, and
    /// leaves the rest as it was.
    #[test]
    fn a_reply_identifier_set_later_is_the_headers() -> Result<(), WireError> {
        let uuid = GuestUuid([0x11; 16]);
        let mut message = MessageWriter::new(
            FourCc::new(*b"Dbuf"),
            MessageFlags::HOST_EXPECTS_REPLY,
            Some(uuid),
            0,
        );
        message.s64(KEY, 7);
        let mut expected = MessageWriter::new(
            FourCc::new(*b"Dbuf"),
            MessageFlags::HOST_EXPECTS_REPLY,
            Some(uuid),
            0x0102_0304_0506_0708,
        );
        expected.s64(KEY, 7);
        message.set_reply_id(0x0102_0304_0506_0708);
        let bytes = message.finish();
        assert_eq!(bytes, expected.finish());
        assert_eq!(read_header(&bytes, 0)?.reply_id, 0x0102_0304_0506_0708);
        Ok(())
    }
}
