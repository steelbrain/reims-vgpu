//! A disk's GUID partition table, read through gptman, which checks its headers' and entry
//! array's CRC-32s.
//!
//! GUIDs here are in their usual string order; on disk, a GPT stores the first three groups
//! little-endian, which [`guid_from_disk`] converts.

use crate::dev::{Cursor, Source};
use crate::{Error, Result};

/// A partition, as the table records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// The partition's slot, from 1.
    pub index: u32,
    pub type_guid: String,
    pub unique_guid: String,
    pub name: String,
    pub first_lba: u64,
    pub last_lba: u64,
}

impl Partition {
    /// The partition's first byte and its length in bytes, for `sector_size`-byte sectors.
    #[must_use]
    pub const fn byte_range(&self, sector_size: u64) -> (u64, u64) {
        (
            self.first_lba * sector_size,
            (self.last_lba + 1 - self.first_lba) * sector_size,
        )
    }
}

/// A read partition table.
#[derive(Clone, Debug)]
pub struct Table {
    pub sector_size: u64,
    pub disk_guid: String,
    pub partitions: Vec<Partition>,
}

/// A GUID's usual string form, from its on-disk byte order.
#[must_use]
pub fn guid_from_disk(d: &[u8; 16]) -> String {
    let mut b = *d;
    b[0..4].reverse();
    b[4..6].reverse();
    b[6..8].reverse();
    let h: Vec<String> = b.iter().map(|x| format!("{x:02X}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

/// Read the partition table of `src`, trying 512- and then 4096-byte sectors.
///
/// # Errors
///
/// Neither sector size finds a valid table.
pub fn read(src: &dyn Source) -> Result<Table> {
    let mut last = String::new();
    for ss in [512u64, 4096] {
        match gptman::GPT::read_from(&mut Cursor::new(src), ss) {
            Ok(g) => {
                let partitions = g
                    .iter()
                    .filter(|(_, p)| p.is_used())
                    .map(|(i, p)| Partition {
                        index: i,
                        type_guid: guid_from_disk(&p.partition_type_guid),
                        unique_guid: guid_from_disk(&p.unique_partition_guid),
                        name: p.partition_name.as_str().to_string(),
                        first_lba: p.starting_lba,
                        last_lba: p.ending_lba,
                    })
                    .collect();
                return Ok(Table {
                    sector_size: ss,
                    disk_guid: guid_from_disk(&g.header.disk_guid),
                    partitions,
                });
            }
            Err(e) => last = e.to_string(),
        }
    }
    Err(Error::Gpt(last))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guids_read_in_their_string_order() {
        let d = [
            0xef, 0x57, 0x34, 0x7c, 0, 0, 0xaa, 0x11, 0xaa, 0x11, 0x00, 0x30, 0x65, 0x43, 0xec,
            0xac,
        ];
        assert_eq!(guid_from_disk(&d), crate::ondisk::part::APFS);
    }
}
