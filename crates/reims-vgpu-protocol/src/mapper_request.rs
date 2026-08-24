//! Semantic mapper request records.

pub const MAPPER_REQUEST_ENTRY_LEN: usize = 16;
pub const MAPPER_REQUEST_TYPE: usize = 0x00;
pub const MAPPER_REQUEST_MAPPING_ID: usize = 0x04;
pub const MAPPER_REQUEST_RESERVED: usize = 0x08;
pub const MAPPER_REQUEST_MAP: u32 = 1;
pub const MAPPER_REQUEST_UNMAP: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapperRequestKind {
    Map,
    Unmap,
    Other(u32),
}

impl Default for MapperRequestKind {
    fn default() -> Self {
        Self::Other(0)
    }
}

impl MapperRequestKind {
    pub const fn from_raw(raw: u32) -> Self {
        match raw {
            MAPPER_REQUEST_MAP => Self::Map,
            MAPPER_REQUEST_UNMAP => Self::Unmap,
            other => Self::Other(other),
        }
    }

    pub const fn raw(self) -> u32 {
        match self {
            Self::Map => MAPPER_REQUEST_MAP,
            Self::Unmap => MAPPER_REQUEST_UNMAP,
            Self::Other(raw) => raw,
        }
    }

    pub const fn is_known(self) -> bool {
        matches!(self, Self::Map | Self::Unmap)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapperRequestEntry {
    pub kind: MapperRequestKind,
    pub mapping_id: u32,
    pub reserved: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapperRequestTooShort {
    pub actual: usize,
    pub required: usize,
}

pub fn mapper_request_entry_offset(index: u32) -> u64 {
    u64::from(index) * MAPPER_REQUEST_ENTRY_LEN as u64
}

pub fn mapper_request_published_entry_offset(producer: u32) -> Option<u64> {
    producer.checked_sub(1).map(mapper_request_entry_offset)
}

pub fn decode_mapper_request_entry(
    bytes: &[u8],
) -> Result<MapperRequestEntry, MapperRequestTooShort> {
    if bytes.len() < MAPPER_REQUEST_ENTRY_LEN {
        return Err(MapperRequestTooShort {
            actual: bytes.len(),
            required: MAPPER_REQUEST_ENTRY_LEN,
        });
    }
    Ok(MapperRequestEntry {
        kind: MapperRequestKind::from_raw(u32::from_le_bytes(
            bytes[MAPPER_REQUEST_TYPE..][..4].try_into().unwrap(),
        )),
        mapping_id: u32::from_le_bytes(bytes[MAPPER_REQUEST_MAPPING_ID..][..4].try_into().unwrap()),
        reserved: u64::from_le_bytes(bytes[MAPPER_REQUEST_RESERVED..][..8].try_into().unwrap()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mapper_request_record_decodes_as_one_owned_shape() {
        let mut bytes = [0u8; MAPPER_REQUEST_ENTRY_LEN];
        bytes[MAPPER_REQUEST_TYPE..][..4].copy_from_slice(&MAPPER_REQUEST_MAP.to_le_bytes());
        bytes[MAPPER_REQUEST_MAPPING_ID..][..4].copy_from_slice(&0x8765_4321u32.to_le_bytes());
        bytes[MAPPER_REQUEST_RESERVED..][..8]
            .copy_from_slice(&0xfedc_ba98_7654_3210u64.to_le_bytes());
        assert_eq!(
            decode_mapper_request_entry(&bytes),
            Ok(MapperRequestEntry {
                kind: MapperRequestKind::Map,
                mapping_id: 0x8765_4321,
                reserved: 0xfedc_ba98_7654_3210,
            })
        );
        assert_eq!(mapper_request_published_entry_offset(0), None);
        assert_eq!(mapper_request_published_entry_offset(3), Some(32));
    }

    #[test]
    fn a_short_mapper_request_is_a_typed_boundary_refusal() {
        assert_eq!(
            decode_mapper_request_entry(&[0; MAPPER_REQUEST_ENTRY_LEN - 1]),
            Err(MapperRequestTooShort {
                actual: MAPPER_REQUEST_ENTRY_LEN - 1,
                required: MAPPER_REQUEST_ENTRY_LEN,
            })
        );
    }
}
