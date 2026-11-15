//! Turning a tree of files, directories, symbolic links and clones into a volume's file-system
//! records and data streams, as a volume holding them records them (Apple File System
//! Reference, "File-System Objects"; each record's form checked against volumes `fsck_apfs`
//! accepts, and a clone's against the one Virtualization.framework's restore makes in Preboot).
//!
//! Every entry is numbered in the order given, from the first user identifier up, so a parent,
//! which comes before its children, is numbered first. Each gets an inode, carrying its own name,
//! and a directory entry in its parent. A file with bytes gets a data stream, its identifier the
//! inode's; a clone shares the data stream of the file it clones, which then has two references,
//! as `clonefile` leaves a clone until either is written. A symbolic link keeps its target in the
//! `com.apple.fs.symlink` attribute.

use std::collections::HashMap;

use crate::fs::key_word;
use crate::ondisk::{ino, rec};

use super::records::{
    self, DT_DIR, DT_LNK, DT_REG, INODE_FLAGS_BASE, InodeBuilder, XField, dir_rec, dstream,
    dstream_xfield, name_xfield,
};
use super::volume::{Counts, Stream, blocks_of};

/// `INODE_WAS_CLONED`: the inode was made by cloning another, or another by cloning it.
const INODE_WAS_CLONED: u64 = 0x10;
/// `INODE_WAS_EVER_CLONED`.
const INODE_WAS_EVER_CLONED: u64 = 0x400;
/// An attribute's value is stored in the record (`XATTR_DATA_EMBEDDED`).
const XATTR_EMBEDDED: u16 = crate::fs::XATTR_DATA_EMBEDDED;
/// An attribute the file system owns (`XATTR_FILE_SYSTEM_OWNED`).
const XATTR_FS_OWNED: u16 = crate::fs::XATTR_FILE_SYSTEM_OWNED;
/// The attribute a symbolic link keeps its target in.
pub const SYMLINK_XATTR: &[u8] = b"com.apple.fs.symlink";
/// The most bytes an attribute's value may have stored in its record.
pub const MAX_EMBEDDED_XATTR: usize = 3804;

/// What an entry is.
#[derive(Clone, Debug)]
pub enum Kind {
    Dir,
    /// A regular file of `len` bytes, its bytes written by the caller where the volume places
    /// its stream.
    File {
        len: u64,
    },
    /// A symbolic link to the target given.
    Symlink(Vec<u8>),
    /// A clone of the file at the path given, which comes before it.
    Clone(String),
}

/// One entry of the tree.
#[derive(Clone, Debug)]
pub struct Item {
    /// Its path from the volume's root, without a leading `/`.
    pub path: String,
    pub kind: Kind,
    /// Its permission bits; the file type comes from `kind`.
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub bsd_flags: u32,
    /// Its extended attributes, each stored in its record.
    pub xattrs: Vec<(Vec<u8>, Vec<u8>)>,
}

/// A tree's records, its data streams with the entry each is for, and what the volume's
/// superblock counts.
pub struct Built {
    pub records: Vec<(Vec<u8>, Vec<u8>)>,
    pub streams: Vec<Stream>,
    /// The index into the items of the file each stream holds the bytes of.
    pub stream_items: Vec<usize>,
    pub counts: Counts,
    /// The first identifier the volume has not handed out.
    pub next_obj_id: u64,
}

/// A tree that cannot be a volume.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("{0}: its parent is not a directory before it")]
    NoParent(String),
    #[error("{0}: given twice")]
    Duplicate(String),
    #[error("{0}: a clone of {1}, which is not a file with bytes before it")]
    BadClone(String, String),
    #[error("{0}: attribute {1} is {2} bytes, more than a record holds")]
    XattrTooBig(String, String, usize),
}

/// An attribute record of inode `oid`: its name (and a NUL) keys it, and its value is stored.
fn xattr(oid: u64, name: &[u8], value: &[u8], flags: u16) -> (Vec<u8>, Vec<u8>) {
    let mut k = key_word(oid, rec::XATTR).to_le_bytes().to_vec();
    k.extend_from_slice(&((name.len() + 1) as u16).to_le_bytes());
    k.extend_from_slice(name);
    k.push(0);
    let mut v = Vec::with_capacity(4 + value.len());
    v.extend_from_slice(&flags.to_le_bytes());
    v.extend_from_slice(&(value.len() as u16).to_le_bytes());
    v.extend_from_slice(value);
    (k, v)
}

/// A data stream's reference count record.
fn dstream_id(id: u64, refcnt: u32) -> (Vec<u8>, Vec<u8>) {
    (
        key_word(id, rec::DSTREAM_ID).to_le_bytes().to_vec(),
        refcnt.to_le_bytes().to_vec(),
    )
}

/// The last component of a path.
fn base_name(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |p| p.1)
}

/// The directory a path is in, `""` for the root.
fn parent_of(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |p| p.0)
}

/// Build the records of `items`, every time `t`, names hashed folding case where `fold`.
///
/// # Errors
///
/// [`BuildError`] for an entry whose parent is missing or not a directory, a path given twice, a
/// clone of anything but an earlier file with bytes, or an attribute too big to store in its
/// record.
pub fn build(items: &[Item], fold: bool, t: u64) -> Result<Built, BuildError> {
    // Number every entry, and find each one's parent.
    let mut oid_of: HashMap<&str, u64> = HashMap::from([("", ino::ROOT_DIR)]);
    let mut is_dir: HashMap<u64, bool> = HashMap::from([(ino::ROOT_DIR, true)]);
    let mut children: HashMap<u64, u32> = HashMap::new();
    let mut next = ino::MIN_USER;
    let mut oids = Vec::with_capacity(items.len());
    let mut parents = Vec::with_capacity(items.len());
    for it in items {
        let parent = *oid_of
            .get(parent_of(&it.path))
            .filter(|p| is_dir.get(p).copied().unwrap_or(false))
            .ok_or_else(|| BuildError::NoParent(it.path.clone()))?;
        if oid_of.contains_key(it.path.as_str()) {
            return Err(BuildError::Duplicate(it.path.clone()));
        }
        let oid = next;
        next += 1;
        oid_of.insert(&it.path, oid);
        is_dir.insert(oid, matches!(it.kind, Kind::Dir));
        *children.entry(parent).or_default() += 1;
        oids.push(oid);
        parents.push(parent);
    }

    // Each file's data stream: its own, or, for a clone, the one it shares.
    let mut stream_of: Vec<Option<u64>> = vec![None; items.len()];
    let mut refs: HashMap<u64, u32> = HashMap::new();
    let mut len_of: HashMap<u64, u64> = HashMap::new();
    let mut cloned: Vec<bool> = vec![false; items.len()];
    let mut index_of: HashMap<u64, usize> = HashMap::new();
    let mut built = Built {
        records: Vec::new(),
        streams: Vec::new(),
        stream_items: Vec::new(),
        counts: Counts::default(),
        next_obj_id: next,
    };
    for (i, it) in items.iter().enumerate() {
        index_of.insert(oids[i], i);
        match &it.kind {
            Kind::File { len } if *len > 0 => {
                stream_of[i] = Some(oids[i]);
                refs.insert(oids[i], 1);
                len_of.insert(oids[i], *len);
                built.streams.push(Stream {
                    id: oids[i],
                    owner: oids[i],
                    len: *len,
                });
                built.stream_items.push(i);
            }
            Kind::Clone(of) => {
                let orig = oid_of
                    .get(of.as_str())
                    .and_then(|o| index_of.get(o).map(|&j| (*o, j)))
                    .filter(|&(o, j)| j < i && stream_of[j] == Some(o))
                    .ok_or_else(|| BuildError::BadClone(it.path.clone(), of.clone()))?;
                stream_of[i] = Some(orig.0);
                *refs.get_mut(&orig.0).expect("a stream's references") += 1;
                cloned[i] = true;
                cloned[orig.1] = true;
            }
            _ => {}
        }
    }

    // The root directory, the private directory and their entries, then each item's.
    let root_children = children.get(&ino::ROOT_DIR).copied().unwrap_or(0);
    for (k, v) in records::empty_volume_records(fold, t) {
        let (oid, kind) = crate::fs::key_header(&k);
        if kind == rec::INODE && oid == ino::ROOT_DIR {
            let mut v = v;
            v[56..60].copy_from_slice(&(root_children as i32).to_le_bytes());
            built.records.push((k, v));
        } else {
            built.records.push((k, v));
        }
    }
    for (i, it) in items.iter().enumerate() {
        let oid = oids[i];
        let name = base_name(&it.path).as_bytes();
        let (dtype, type_bits) = match it.kind {
            Kind::Dir => (DT_DIR, 0o040000),
            Kind::File { .. } | Kind::Clone(_) => (DT_REG, 0o100000),
            Kind::Symlink(_) => (DT_LNK, 0o120000),
        };
        built
            .records
            .push(dir_rec(parents[i], name, fold, oid, t, dtype));

        let mut xfields: Vec<XField> = vec![name_xfield(name)];
        let mut private_id = oid;
        if let Some(s) = stream_of[i] {
            private_id = s;
            let len = len_of[&s];
            xfields.push(dstream_xfield(dstream(
                len,
                blocks_of(len) * super::geometry::BLOCK_SIZE,
                0,
                len,
                0,
            )));
        }
        let mut internal_flags = INODE_FLAGS_BASE;
        if cloned[i] {
            internal_flags |= INODE_WAS_CLONED | INODE_WAS_EVER_CLONED;
        }
        let nchildren = match it.kind {
            Kind::Dir => children.get(&oid).copied().unwrap_or(0) as i32,
            _ => 1,
        };
        built.records.push(
            InodeBuilder {
                oid,
                parent: parents[i],
                private_id,
                create_time: t,
                mod_time: t,
                change_time: t,
                access_time: t,
                internal_flags,
                nchildren_or_nlink: nchildren,
                default_protection_class: 0,
                write_generation_counter: 1,
                bsd_flags: it.bsd_flags,
                owner: it.uid,
                group: it.gid,
                mode: type_bits | (it.mode & 0o7777),
                uncompressed_size: 0,
                xfields,
            }
            .build(),
        );
        if stream_of[i] == Some(oid) {
            built.records.push(dstream_id(oid, refs[&oid]));
        }
        if let Kind::Symlink(target) = &it.kind {
            let mut v = target.clone();
            v.push(0);
            built.records.push(xattr(
                oid,
                SYMLINK_XATTR,
                &v,
                XATTR_EMBEDDED | XATTR_FS_OWNED,
            ));
        }
        for (n, v) in &it.xattrs {
            if v.len() > MAX_EMBEDDED_XATTR {
                return Err(BuildError::XattrTooBig(
                    it.path.clone(),
                    String::from_utf8_lossy(n).into_owned(),
                    v.len(),
                ));
            }
            built.records.push(xattr(oid, n, v, XATTR_EMBEDDED));
        }
        match it.kind {
            Kind::Dir => built.counts.directories += 1,
            Kind::File { .. } | Kind::Clone(_) => built.counts.files += 1,
            Kind::Symlink(_) => built.counts.symlinks += 1,
        }
    }
    built
        .records
        .sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));
    Ok(built)
}
