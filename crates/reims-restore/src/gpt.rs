//! The disk's GUID partition table, laid out as Virtualization.framework's restore lays it out
//! (docs/design.md, "The disk"): three partitions, the iBoot system container, the main APFS
//! container and the recoveryOS container, in 512-byte sectors.

use std::io::{Seek, SeekFrom, Write};

use anyhow::{Result, ensure};
use uuid::Uuid;

/// The sector size of the table.
pub const SECTOR: u64 = 512;
/// The partition entries: 128 of 128 bytes, from LBA 2, and again before the backup header.
const ENTRIES: usize = 128;
const ENTRY_LEN: usize = 128;
const ENTRY_SECTORS: u64 = (ENTRIES * ENTRY_LEN) as u64 / SECTOR;
/// The first LBA a partition may use.
const FIRST_USABLE: u64 = 2 + ENTRY_SECTORS;

/// The iBoot system container's type, `Apple_APFS_ISC`.
pub const ISC_TYPE: Uuid = Uuid::from_u128(0x69646961_6700_11aa_aa11_00306543ecac);
/// The main container's type, `Apple_APFS`.
pub const APFS_TYPE: Uuid = Uuid::from_u128(0x7c3457ef_0000_11aa_aa11_00306543ecac);
/// The recoveryOS container's type, `Apple_APFS_Recovery`.
pub const RECOVERY_TYPE: Uuid = Uuid::from_u128(0x52637672_7900_11aa_aa11_00306543ecac);

/// The iBoot system container: 500 MiB from LBA 40.
const ISC_FIRST: u64 = 40;
const ISC_SECTORS: u64 = (500 << 20) / SECTOR;
/// The recoveryOS container: 5 GiB less 40 KiB, ending 7 sectors before the last usable one.
const RECOVERY_SECTORS: u64 = ((5 << 30) - (40 << 10)) / SECTOR;
const RECOVERY_GAP: u64 = 7;

/// One partition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// Its type.
    pub kind: Uuid,
    /// Its own GUID.
    pub guid: Uuid,
    /// Its name.
    pub name: &'static str,
    /// Its first sector.
    pub first: u64,
    /// Its last sector, inclusive.
    pub last: u64,
}

impl Partition {
    /// Its first byte.
    pub fn offset(&self) -> u64 {
        self.first * SECTOR
    }

    /// Its length in bytes.
    pub fn len(&self) -> u64 {
        (self.last + 1 - self.first) * SECTOR
    }

    /// Whether it holds no sector (never, for a table [`Table::layout`] made).
    pub fn is_empty(&self) -> bool {
        self.last < self.first
    }
}

/// The table of a disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    /// The disk's GUID.
    pub disk: Uuid,
    /// The disk's size in sectors.
    pub sectors: u64,
    /// The iBoot system container, the main container and the recoveryOS container.
    pub parts: [Partition; 3],
}

impl Table {
    /// The table of a disk of `size` bytes, with random GUIDs. The layout is that of a 64 GiB
    /// disk Virtualization.framework restored (14.6.1 to 27.2.1, all alike) **[vz]**; on other
    /// sizes the first and last partitions keep their places from each end.
    pub fn layout(size: u64) -> Result<Self> {
        let g = || Uuid::new_v4();
        Self::layout_with(size, g(), [g(), g(), g()])
    }

    /// [`Table::layout`] with the GUIDs given.
    pub fn layout_with(size: u64, disk: Uuid, guids: [Uuid; 3]) -> Result<Self> {
        ensure!(
            size.is_multiple_of(4096),
            "the disk's size, {size} bytes, is not a multiple of 4 KiB"
        );
        let sectors = size / SECTOR;
        let last_usable = sectors - 1 - ENTRY_SECTORS - 1;
        let recovery_last = last_usable - RECOVERY_GAP;
        let recovery_first = recovery_last + 1 - RECOVERY_SECTORS;
        let main_first = ISC_FIRST + ISC_SECTORS;
        ensure!(
            recovery_first > main_first + (1 << 30) / SECTOR,
            "a disk of {size} bytes is too small for the three containers"
        );
        Ok(Self {
            disk,
            sectors,
            parts: [
                Partition {
                    kind: ISC_TYPE,
                    guid: guids[0],
                    name: "iBootSystemContainer",
                    first: ISC_FIRST,
                    last: main_first - 1,
                },
                Partition {
                    kind: APFS_TYPE,
                    guid: guids[1],
                    name: "Container",
                    first: main_first,
                    last: recovery_first - 1,
                },
                Partition {
                    kind: RECOVERY_TYPE,
                    guid: guids[2],
                    name: "RecoveryOSContainer",
                    first: recovery_first,
                    last: recovery_last,
                },
            ],
        })
    }

    /// The iBoot system container.
    pub fn isc(&self) -> &Partition {
        &self.parts[0]
    }

    /// The main container.
    pub fn main(&self) -> &Partition {
        &self.parts[1]
    }

    /// The recoveryOS container.
    pub fn recovery(&self) -> &Partition {
        &self.parts[2]
    }

    /// The partition entries, as they are written twice.
    fn entries(&self) -> Vec<u8> {
        let mut e = vec![0u8; ENTRIES * ENTRY_LEN];
        for (i, p) in self.parts.iter().enumerate() {
            let at = &mut e[i * ENTRY_LEN..(i + 1) * ENTRY_LEN];
            at[..16].copy_from_slice(&guid_bytes(p.kind));
            at[16..32].copy_from_slice(&guid_bytes(p.guid));
            at[32..40].copy_from_slice(&p.first.to_le_bytes());
            at[40..48].copy_from_slice(&p.last.to_le_bytes());
            for (j, c) in p.name.encode_utf16().enumerate() {
                at[56 + 2 * j..58 + 2 * j].copy_from_slice(&c.to_le_bytes());
            }
        }
        e
    }

    /// A header: the primary one at LBA 1 when `primary`, else the backup at the last LBA.
    fn header(&self, primary: bool, entries_crc: u32) -> [u8; SECTOR as usize] {
        let last = self.sectors - 1;
        let (me, other, entries_at) = if primary {
            (1, last, 2)
        } else {
            (last, 1, last - ENTRY_SECTORS)
        };
        let mut h = [0u8; SECTOR as usize];
        h[..8].copy_from_slice(b"EFI PART");
        h[8..12].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        h[12..16].copy_from_slice(&92u32.to_le_bytes());
        h[24..32].copy_from_slice(&me.to_le_bytes());
        h[32..40].copy_from_slice(&other.to_le_bytes());
        h[40..48].copy_from_slice(&FIRST_USABLE.to_le_bytes());
        h[48..56].copy_from_slice(&(last - ENTRY_SECTORS - 1).to_le_bytes());
        h[56..72].copy_from_slice(&guid_bytes(self.disk));
        h[72..80].copy_from_slice(&entries_at.to_le_bytes());
        h[80..84].copy_from_slice(&(ENTRIES as u32).to_le_bytes());
        h[84..88].copy_from_slice(&(ENTRY_LEN as u32).to_le_bytes());
        h[88..92].copy_from_slice(&entries_crc.to_le_bytes());
        let crc = crc32fast::hash(&h[..92]);
        h[16..20].copy_from_slice(&crc.to_le_bytes());
        h
    }

    /// The protective MBR: one partition of type 0xEE over the whole disk, up to 2^32 - 1
    /// sectors.
    fn mbr(&self) -> [u8; SECTOR as usize] {
        let mut m = [0u8; SECTOR as usize];
        let e = &mut m[446..462];
        e.copy_from_slice(&[
            0x00, 0xfe, 0xff, 0xff, 0xee, 0xfe, 0xff, 0xff, 1, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let n = u32::try_from(self.sectors - 1).unwrap_or(u32::MAX);
        e[12..].copy_from_slice(&n.to_le_bytes());
        m[510] = 0x55;
        m[511] = 0xaa;
        m
    }

    /// Writes the protective MBR, both headers and both copies of the entries to `disk`, a
    /// file already the table's size.
    pub fn write<W: Write + Seek>(&self, disk: &mut W) -> Result<()> {
        let entries = self.entries();
        let crc = crc32fast::hash(&entries);
        let last = self.sectors - 1;
        disk.seek(SeekFrom::Start(0))?;
        disk.write_all(&self.mbr())?;
        disk.write_all(&self.header(true, crc))?;
        disk.write_all(&entries)?;
        disk.seek(SeekFrom::Start((last - ENTRY_SECTORS) * SECTOR))?;
        disk.write_all(&entries)?;
        disk.write_all(&self.header(false, crc))?;
        Ok(())
    }
}

/// A GUID as GPT stores it: the first three fields little-endian, the rest as they are.
pub fn guid_bytes(u: Uuid) -> [u8; 16] {
    u.to_bytes_le()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 64 GiB layout every reference guest has **[vz]**, with GUIDs of the tests' own.
    #[test]
    fn a_64_gib_disk_is_laid_out_as_vzs() {
        let t = Table::layout(64 << 30).unwrap();
        assert_eq!((t.isc().first, t.isc().last), (40, 1_024_039));
        assert_eq!((t.main().first, t.main().last), (1_024_040, 123_732_007));
        assert_eq!(
            (t.recovery().first, t.recovery().last),
            (123_732_008, 134_217_687)
        );
        assert_eq!(t.recovery().len(), 5_368_668_160);
        assert_eq!(t.main().len(), 62_826_479_616);
        assert_eq!(t.isc().len(), 500 << 20);
        for p in &t.parts {
            assert!(p.offset().is_multiple_of(4096) && p.len().is_multiple_of(4096));
        }
    }

    #[test]
    fn the_table_reads_back() {
        let size = 40u64 << 30;
        let ids = [Uuid::from_u128(1), Uuid::from_u128(2), Uuid::from_u128(3)];
        let t = Table::layout_with(size, Uuid::from_u128(0xd15c), ids).unwrap();
        let mut disk = Sparse {
            len: size,
            ..Sparse::default()
        };
        t.write(&mut disk).unwrap();
        let read = reims_apfs::gpt::read(&disk).unwrap();
        assert_eq!(read.sector_size, SECTOR);
        assert_eq!(read.disk_guid, "00000000-0000-0000-0000-00000000D15C");
        assert_eq!(read.partitions.len(), 3);
        for (r, p) in read.partitions.iter().zip(&t.parts) {
            assert_eq!(r.type_guid, p.kind.to_string().to_uppercase());
            assert_eq!(r.unique_guid, p.guid.to_string().to_uppercase());
            assert_eq!(r.name, p.name);
            assert_eq!((r.first_lba, r.last_lba), (p.first, p.last));
        }
        // The backup header names the primary, carries the same entries' CRC, and checks.
        let last = t.sectors - 1;
        let backup = disk.bytes_at(last * SECTOR, SECTOR as usize);
        let primary = disk.bytes_at(SECTOR, SECTOR as usize);
        assert_eq!(&backup[..8], b"EFI PART");
        assert_eq!(u64::from_le_bytes(backup[32..40].try_into().unwrap()), 1);
        assert_eq!(
            u64::from_le_bytes(backup[72..80].try_into().unwrap()),
            last - ENTRY_SECTORS
        );
        assert_eq!(backup[88..92], primary[88..92]);
        let mut h = backup[..92].to_vec();
        h[16..20].fill(0);
        assert_eq!(crc32fast::hash(&h).to_le_bytes(), backup[16..20]);
        let mbr = disk.bytes_at(0, 512);
        assert_eq!(&mbr[510..], &[0x55, 0xaa]);
        assert_eq!(
            u64::from(u32::from_le_bytes(mbr[458..462].try_into().unwrap())),
            t.sectors - 1
        );
    }

    #[test]
    fn a_disk_too_small_is_refused() {
        assert!(Table::layout(6 << 30).is_err());
        assert!(Table::layout((64 << 30) + 512).is_err());
    }

    /// A disk image of `len` bytes held as the writes made to it, the rest zero.
    #[derive(Default)]
    struct Sparse {
        len: u64,
        at: u64,
        writes: Vec<(u64, Vec<u8>)>,
    }

    impl Sparse {
        fn bytes_at(&self, off: u64, len: usize) -> Vec<u8> {
            let mut out = vec![0u8; len];
            for (o, d) in &self.writes {
                for (i, b) in d.iter().enumerate() {
                    let p = o + i as u64;
                    if p >= off && p < off + len as u64 {
                        out[(p - off) as usize] = *b;
                    }
                }
            }
            out
        }
    }

    impl Write for Sparse {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.writes.push((self.at, b.to_vec()));
            self.at += b.len() as u64;
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Seek for Sparse {
        fn seek(&mut self, p: SeekFrom) -> std::io::Result<u64> {
            match p {
                SeekFrom::Start(o) => self.at = o,
                _ => unimplemented!("the table seeks from the start"),
            }
            Ok(self.at)
        }
    }

    impl reims_apfs::dev::Source for Sparse {
        fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
            buf.copy_from_slice(&self.bytes_at(off, buf.len()));
            Ok(())
        }
        fn len(&self) -> u64 {
            self.len
        }
    }
}
