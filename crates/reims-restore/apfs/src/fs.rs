//! File-system records: keys and values of a volume's file-system tree and of a sealed volume's
//! file extent tree (Apple File System Reference, "File-System Objects", "Sealed Volumes").

use std::cmp::Ordering;

use crate::le::{u16_at, u32_at, u64_at};
use crate::ondisk::rec;

// A key's fields, read as zero where the key is too short to hold them, so comparing a
// malformed key orders it rather than panics.
fn key_u16(k: &[u8], off: usize) -> u16 {
    k.get(off..off + 2)
        .map_or(0, |b| u16::from_le_bytes(b.try_into().unwrap()))
}

fn key_u32(k: &[u8], off: usize) -> u32 {
    k.get(off..off + 4)
        .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
}

fn key_u64(k: &[u8], off: usize) -> u64 {
    k.get(off..off + 8)
        .map_or(0, |b| u64::from_le_bytes(b.try_into().unwrap()))
}

/// A record key's object identifier and type.
#[must_use]
pub fn key_header(key: &[u8]) -> (u64, u8) {
    let w = key_u64(key, 0);
    (w & rec::OBJ_ID_MASK, (w >> rec::TYPE_SHIFT) as u8)
}

/// A key's first word for `oid` and record type `t`.
#[must_use]
pub const fn key_word(oid: u64, t: u8) -> u64 {
    (oid & rec::OBJ_ID_MASK) | ((t as u64) << rec::TYPE_SHIFT)
}

/// The order of two file-system tree keys: by identifier, then type, then what the type keys
/// by (a directory entry by hash then name, an attribute by name, a file extent by offset, a
/// sibling link by sibling, a file info by its address word, a snapshot name by name).
#[must_use]
pub fn compare_keys(a: &[u8], b: &[u8]) -> Ordering {
    let (ia, ta) = key_header(a);
    let (ib, tb) = key_header(b);
    (ia, ta).cmp(&(ib, tb)).then_with(|| match ta {
        rec::FILE_EXTENT | rec::SIBLING_LINK | rec::FILE_INFO => key_u64(a, 8).cmp(&key_u64(b, 8)),
        rec::DIR_REC => {
            let (ha, hb) = (drec_hash(a), drec_hash(b));
            ha.cmp(&hb).then_with(|| drec_name(a).cmp(drec_name(b)))
        }
        rec::XATTR | rec::SNAP_NAME => name16(a).cmp(name16(b)),
        _ => Ordering::Equal,
    })
}

/// A name after a 16-bit length at byte 8 (an attribute's or a snapshot name's key), without its
/// terminating NUL.
fn name16(key: &[u8]) -> &[u8] {
    let n = usize::from(key_u16(key, 8));
    let name = key.get(10..(10 + n).min(key.len())).unwrap_or_default();
    name.strip_suffix(&[0]).unwrap_or(name)
}

/// A hashed directory entry key's name, without its NUL.
#[must_use]
pub fn drec_name(key: &[u8]) -> &[u8] {
    let n = (key_u32(key, 8) & 0x3ff) as usize;
    let name = key.get(12..(12 + n).min(key.len())).unwrap_or_default();
    name.strip_suffix(&[0]).unwrap_or(name)
}

/// A hashed directory entry key's 22-bit name hash.
#[must_use]
pub fn drec_hash(key: &[u8]) -> u32 {
    key_u32(key, 8) >> 10
}

/// The order of two keys of the file extent tree, or of the object map: two 64-bit words.
#[must_use]
pub fn compare_u64_pairs(a: &[u8], b: &[u8]) -> Ordering {
    (key_u64(a, 0), key_u64(a, 8)).cmp(&(key_u64(b, 0), key_u64(b, 8)))
}

/// An extended field of an inode or directory entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XField {
    pub kind: u8,
    pub flags: u8,
    pub data: Vec<u8>,
}

/// Parse an `xf_blob_t`.
#[must_use]
pub fn parse_xfields(b: &[u8]) -> Vec<XField> {
    if b.len() < 4 {
        return Vec::new();
    }
    let n = usize::from(u16_at(b, 0));
    let mut out = Vec::with_capacity(n);
    let mut data = 4 + n * 4;
    for i in 0..n {
        let h = 4 + i * 4;
        if h + 4 > b.len() {
            break;
        }
        let size = usize::from(u16_at(b, h + 2));
        if data + size > b.len() {
            break;
        }
        out.push(XField {
            kind: b[h],
            flags: b[h + 1],
            data: b[data..data + size].to_vec(),
        });
        data += size.div_ceil(8) * 8;
    }
    out
}

pub mod xf {
    pub const SNAP_XID: u8 = 1;
    pub const DELTA_TREE_OID: u8 = 2;
    pub const DOCUMENT_ID: u8 = 3;
    pub const NAME: u8 = 4;
    pub const PREV_FSIZE: u8 = 5;
    pub const FINDER_INFO: u8 = 7;
    pub const DSTREAM: u8 = 8;
    pub const DIR_STATS_KEY: u8 = 10;
    pub const FS_UUID: u8 = 11;
    pub const SPARSE_BYTES: u8 = 13;
    pub const RDEV: u8 = 14;
    pub const PURGEABLE_FLAGS: u8 = 15;
    pub const ORIG_SYNC_ROOT_ID: u8 = 16;
    /// A directory entry's sibling identifier.
    pub const DREC_SIBLING_ID: u8 = 1;

    pub const FLAG_DATA_DEPENDENT: u8 = 0x1;
    pub const FLAG_DO_NOT_COPY: u8 = 0x2;
    pub const FLAG_CHILDREN_INHERIT: u8 = 0x8;
    pub const FLAG_USER_FIELD: u8 = 0x10;
    pub const FLAG_SYSTEM_FIELD: u8 = 0x20;
}

/// An inode record's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inode {
    pub parent_id: u64,
    pub private_id: u64,
    pub create_time: u64,
    pub mod_time: u64,
    pub change_time: u64,
    pub access_time: u64,
    pub internal_flags: u64,
    /// Children of a directory, links of anything else.
    pub nchildren_or_nlink: i32,
    pub default_protection_class: u32,
    pub write_generation_counter: u32,
    pub bsd_flags: u32,
    pub owner: u32,
    pub group: u32,
    pub mode: u16,
    pub pad1: u16,
    pub uncompressed_size: u64,
    pub xfields: Vec<XField>,
}

/// The fixed part of `j_inode_val_t`.
pub const INODE_FIXED: usize = 92;

impl Inode {
    #[must_use]
    pub fn parse(v: &[u8]) -> Option<Self> {
        if v.len() < INODE_FIXED {
            return None;
        }
        Some(Self {
            parent_id: u64_at(v, 0),
            private_id: u64_at(v, 8),
            create_time: u64_at(v, 16),
            mod_time: u64_at(v, 24),
            change_time: u64_at(v, 32),
            access_time: u64_at(v, 40),
            internal_flags: u64_at(v, 48),
            nchildren_or_nlink: u32_at(v, 56) as i32,
            default_protection_class: u32_at(v, 60),
            write_generation_counter: u32_at(v, 64),
            bsd_flags: u32_at(v, 68),
            owner: u32_at(v, 72),
            group: u32_at(v, 76),
            mode: u16_at(v, 80),
            pad1: u16_at(v, 82),
            uncompressed_size: u64_at(v, 84),
            xfields: parse_xfields(&v[INODE_FIXED..]),
        })
    }

    /// The data stream extended field: size, allocated size, crypto, bytes written and read.
    #[must_use]
    pub fn dstream(&self) -> Option<Dstream> {
        self.xfields
            .iter()
            .find(|x| x.kind == xf::DSTREAM)
            .and_then(|x| Dstream::parse(&x.data))
    }

    /// The name extended field (a hard-linked inode's name), without its NUL.
    #[must_use]
    pub fn name(&self) -> Option<&[u8]> {
        self.xfields
            .iter()
            .find(|x| x.kind == xf::NAME)
            .map(|x| x.data.strip_suffix(&[0]).unwrap_or(&x.data))
    }

    #[must_use]
    pub const fn is_dir(&self) -> bool {
        self.mode & 0o170000 == 0o040000
    }
}

/// `j_dstream_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Dstream {
    pub size: u64,
    pub alloced_size: u64,
    pub default_crypto_id: u64,
    pub total_bytes_written: u64,
    pub total_bytes_read: u64,
}

impl Dstream {
    #[must_use]
    pub fn parse(b: &[u8]) -> Option<Self> {
        (b.len() >= 40).then(|| Self {
            size: u64_at(b, 0),
            alloced_size: u64_at(b, 8),
            default_crypto_id: u64_at(b, 16),
            total_bytes_written: u64_at(b, 24),
            total_bytes_read: u64_at(b, 32),
        })
    }
}

/// A directory entry record's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub file_id: u64,
    pub date_added: u64,
    /// The low four bits are the entry's `DT_*` type.
    pub flags: u16,
    pub xfields: Vec<XField>,
}

impl DirEntry {
    #[must_use]
    pub fn parse(v: &[u8]) -> Option<Self> {
        (v.len() >= 18).then(|| Self {
            file_id: u64_at(v, 0),
            date_added: u64_at(v, 8),
            flags: u16_at(v, 16),
            xfields: parse_xfields(&v[18..]),
        })
    }
}

/// An extended attribute record's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XattrData {
    Embedded(Vec<u8>),
    /// A data stream's identifier and its `j_dstream_t`.
    Stream(u64, Dstream),
}

pub const XATTR_DATA_STREAM: u16 = 0x1;
pub const XATTR_DATA_EMBEDDED: u16 = 0x2;
pub const XATTR_FILE_SYSTEM_OWNED: u16 = 0x4;

/// Parse an extended attribute value: its flags and data.
#[must_use]
pub fn parse_xattr(v: &[u8]) -> Option<(u16, XattrData)> {
    if v.len() < 4 {
        return None;
    }
    let flags = u16_at(v, 0);
    let len = usize::from(u16_at(v, 2));
    let data = v.get(4..4 + len)?;
    if flags & XATTR_DATA_STREAM != 0 {
        if data.len() < 48 {
            return None;
        }
        Some((
            flags,
            XattrData::Stream(u64_at(data, 0), Dstream::parse(&data[8..])?),
        ))
    } else {
        Some((flags, XattrData::Embedded(data.to_vec())))
    }
}

/// An attribute key's name, without its NUL.
#[must_use]
pub fn xattr_name(key: &[u8]) -> &[u8] {
    name16(key)
}

/// A file extent record: its length in bytes, flags, first physical block and crypto id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileExtent {
    pub logical: u64,
    pub len: u64,
    pub flags: u8,
    pub phys: u64,
    pub crypto_id: u64,
}

impl FileExtent {
    #[must_use]
    pub fn parse(key: &[u8], v: &[u8]) -> Self {
        let lf = u64_at(v, 0);
        Self {
            logical: u64_at(key, 8),
            len: lf & 0x00ff_ffff_ffff_ffff,
            flags: (lf >> 56) as u8,
            phys: u64_at(v, 8),
            crypto_id: u64_at(v, 16),
        }
    }
}

/// A sealed volume's file extent tree record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FextRecord {
    pub private_id: u64,
    pub logical: u64,
    pub len: u64,
    pub flags: u8,
    pub phys: u64,
}

impl FextRecord {
    #[must_use]
    pub fn parse(key: &[u8], v: &[u8]) -> Self {
        let lf = u64_at(v, 0);
        Self {
            private_id: u64_at(key, 0),
            logical: u64_at(key, 8),
            len: lf & 0x00ff_ffff_ffff_ffff,
            flags: (lf >> 56) as u8,
            phys: u64_at(v, 8),
        }
    }
}
