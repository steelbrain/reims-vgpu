//! Building file-system records — inodes, directory entries, their extended fields, data streams
//! and file extents — as `fs.rs` reads them back (Apple File System Reference, "File-System
//! Objects").
//!
//! A record is a (key, value) pair. Keys order a volume's tree; `fs::compare_keys` gives that
//! order, and a tree is built from records sorted by it. The builders here write the little-endian
//! fields each record type holds; the caller supplies the identifiers, times and modes.

use crate::fs::{key_word, xf};
use crate::names;
use crate::ondisk::rec;

/// A directory entry's `DT_*` type: a directory.
pub const DT_DIR: u16 = 4;
/// A directory entry's `DT_*` type: a regular file.
pub const DT_REG: u16 = 8;
/// A directory entry's `DT_*` type: a symbolic link.
pub const DT_LNK: u16 = 10;

/// An inode's internal flags, as a freshly made volume's root and private directories carry them.
pub const INODE_FLAGS_BASE: u64 = 0x8000;

/// One extended field of an inode or directory entry: its kind, flags and data.
pub struct XField {
    pub kind: u8,
    pub flags: u8,
    pub data: Vec<u8>,
}

/// Encode an `xf_blob_t`: a count and used-data size, then each field's header, then the field
/// data each padded to eight bytes.
#[must_use]
pub fn xfields(fields: &[XField]) -> Vec<u8> {
    if fields.is_empty() {
        return Vec::new();
    }
    let used: usize = fields.iter().map(|f| f.data.len().div_ceil(8) * 8).sum();
    let mut b = Vec::new();
    b.extend_from_slice(&(fields.len() as u16).to_le_bytes());
    b.extend_from_slice(&(used as u16).to_le_bytes());
    for f in fields {
        b.push(f.kind);
        b.push(f.flags);
        b.extend_from_slice(&(f.data.len() as u16).to_le_bytes());
    }
    for f in fields {
        b.extend_from_slice(&f.data);
        let pad = f.data.len().div_ceil(8) * 8 - f.data.len();
        b.extend(std::iter::repeat_n(0u8, pad));
    }
    b
}

/// A NAME extended field (an inode's own name, or a hard link's): the name and a NUL, kind 4,
/// not copied with the inode.
#[must_use]
pub fn name_xfield(name: &[u8]) -> XField {
    let mut data = name.to_vec();
    data.push(0);
    XField {
        kind: xf::NAME,
        flags: xf::FLAG_DO_NOT_COPY,
        data,
    }
}

/// A `j_dstream_t`: logical size, allocated size, crypto id, and bytes written and read.
#[must_use]
pub fn dstream(size: u64, alloced: u64, crypto: u64, written: u64, read: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(40);
    for v in [size, alloced, crypto, written, read] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// A DSTREAM extended field (a file's data stream), kind 8, a system field.
#[must_use]
pub fn dstream_xfield(stream: Vec<u8>) -> XField {
    XField {
        kind: xf::DSTREAM,
        flags: xf::FLAG_SYSTEM_FIELD,
        data: stream,
    }
}

/// What an inode record holds.
pub struct InodeBuilder {
    pub oid: u64,
    pub parent: u64,
    pub private_id: u64,
    pub create_time: u64,
    pub mod_time: u64,
    pub change_time: u64,
    pub access_time: u64,
    pub internal_flags: u64,
    pub nchildren_or_nlink: i32,
    pub default_protection_class: u32,
    pub write_generation_counter: u32,
    pub bsd_flags: u32,
    pub owner: u32,
    pub group: u32,
    pub mode: u16,
    pub uncompressed_size: u64,
    pub xfields: Vec<XField>,
}

impl InodeBuilder {
    /// The (key, value) of this inode.
    #[must_use]
    pub fn build(&self) -> (Vec<u8>, Vec<u8>) {
        let key = key_word(self.oid, rec::INODE).to_le_bytes().to_vec();
        let mut v = Vec::with_capacity(92);
        for x in [
            self.parent,
            self.private_id,
            self.create_time,
            self.mod_time,
            self.change_time,
            self.access_time,
            self.internal_flags,
        ] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v.extend_from_slice(&self.nchildren_or_nlink.to_le_bytes());
        v.extend_from_slice(&self.default_protection_class.to_le_bytes());
        v.extend_from_slice(&self.write_generation_counter.to_le_bytes());
        v.extend_from_slice(&self.bsd_flags.to_le_bytes());
        v.extend_from_slice(&self.owner.to_le_bytes());
        v.extend_from_slice(&self.group.to_le_bytes());
        v.extend_from_slice(&self.mode.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes()); // pad1
        v.extend_from_slice(&self.uncompressed_size.to_le_bytes());
        v.extend_from_slice(&xfields(&self.xfields));
        (key, v)
    }
}

/// A directory entry record (a hashed `j_drec_hashed_key_t`): the parent's identifier keys it,
/// then a word holding the name length (with its NUL) and the name's 22-bit hash, then the name
/// and a NUL. The value names the child, when it was added, and its `DT_*` type.
#[must_use]
pub fn dir_rec(
    parent: u64,
    name: &[u8],
    fold: bool,
    file_id: u64,
    date_added: u64,
    dtype: u16,
) -> (Vec<u8>, Vec<u8>) {
    let name_len = (name.len() + 1) as u32; // includes the NUL
    let nlh = (name_len & 0x3ff) | (names::hash(name, fold) << 10);
    let mut key = key_word(parent, rec::DIR_REC).to_le_bytes().to_vec();
    key.extend_from_slice(&nlh.to_le_bytes());
    key.extend_from_slice(name);
    key.push(0);
    let mut val = Vec::with_capacity(18);
    val.extend_from_slice(&file_id.to_le_bytes());
    val.extend_from_slice(&date_added.to_le_bytes());
    val.extend_from_slice(&dtype.to_le_bytes());
    (key, val)
}

/// The two records a freshly made volume's root directory (inode 2) and private directory (inode
/// 3) and their two directory entries amount to, in key order, every time set to `t` nanoseconds.
///
/// A new APFS volume holds exactly these four records: the root directory's entry for
/// `private-dir`, its entry for `root`, the root inode, and the private-directory inode. Both
/// directories are `drwxr-xr-x`, owned by root, parented to inode 1, and carry their own name as
/// a NAME extended field.
#[must_use]
pub fn empty_volume_records(fold: bool, t: u64) -> Vec<(Vec<u8>, Vec<u8>)> {
    let dir = |oid: u64, name: &[u8], mode: u16| {
        InodeBuilder {
            oid,
            parent: 1,
            private_id: oid,
            create_time: t,
            mod_time: t,
            change_time: t,
            access_time: t,
            internal_flags: INODE_FLAGS_BASE,
            nchildren_or_nlink: 0,
            default_protection_class: 0,
            write_generation_counter: 1,
            bsd_flags: 0,
            owner: 0,
            group: 0,
            mode,
            uncompressed_size: 0,
            xfields: vec![name_xfield(name)],
        }
        .build()
    };
    // The root directory is `drwxr-xr-x`, the private directory `drw-r--r--`.
    let mut recs = vec![
        dir_rec(1, b"private-dir", fold, 3, t, DT_DIR),
        dir_rec(1, b"root", fold, 2, t, DT_DIR),
        dir(2, b"root", 0o040755),
        dir(3, b"private-dir", 0o040644),
    ];
    recs.sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));
    recs
}
